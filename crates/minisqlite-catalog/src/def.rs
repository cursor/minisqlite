//! The schema definitions the catalog stores: a column, a table, an index, a view,
//! a trigger. These are plain data — the shapes `CREATE TABLE` / `CREATE INDEX`
//! produce (built from the AST by `builder::table_def_from_ast` /
//! `builder::index_def_from_ast`) and every reader borrows, plus the two text-only
//! shapes (`ViewDef` / `TriggerDef`) a view/trigger persists. Kept apart
//! from the `Catalog` trait and its stores so each lands in its own file and a
//! change to one is not a change to the seam.

use minisqlite_pager::PageId;
use minisqlite_sql::Expr;
use minisqlite_types::{affinity_of_declared_type, Affinity, Value};

// Re-exported so `ForeignKeyDef` (below) is usable by downstream crates naming the
// `on_delete` / `on_update` action without a direct `minisqlite-sql` dependency. The
// action vocabulary itself is the parser's, so it is re-exported rather than duplicated.
pub use minisqlite_sql::ReferentialAction;

/// A column's definition: its name, declared type text, and the constraint facts a
/// reader needs without re-parsing the `CREATE TABLE`. Affinity is derived from the
/// declared type during evaluation, not stored here.
///
/// The boolean flags mirror the column's own (column-level) constraints; a
/// table-level `PRIMARY KEY`/`UNIQUE` that happens to name this column is recorded
/// on the table, not folded into these per-column flags.
///
/// `default` holds the raw SQL text of a column `DEFAULT` (or `None` when the column
/// has none) — the form the INSERT planner re-binds and `PRAGMA table_info` prints.
/// `default_value` is that same `DEFAULT` pre-evaluated to its constant [`Value`],
/// materialized once by the builder so the read path never re-parses per row. It is
/// `Some` only for a constant literal default (the only kind `ADD COLUMN` allows);
/// it is `None` both when there is no default AND when the default is non-constant
/// (a `DEFAULT (expr)`, or a time-dependent `CURRENT_*`) that cannot be folded to a
/// fixed value at build time. A row stored before this column existed (a "short"
/// record from `ADD COLUMN`) decodes the missing column as this value — SQLite's
/// rule that a short row reads a newly-added column as its DEFAULT, not NULL — so
/// `None` correctly falls back to NULL.
///
/// `generated` is `Some` iff the column is a GENERATED column (`... AS (expr)
/// [STORED|VIRTUAL]`, `lang_createtable.html` § generated columns). It records the
/// generation expression and whether it is `STORED` (materialized on disk) or
/// `VIRTUAL` (computed on read). Like `checks`/`default`, this is a constraint FACT
/// recorded for the planner/executor; nothing here EVALUATES the expression — that
/// (computing the value, excluding a generated column from `INSERT`, the STORED
/// row-layout change) is the executor's, a deliberately separate follow-up.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub declared_type: Option<String>,
    pub not_null: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub collation: Option<String>,
    pub default: Option<String>,
    pub default_value: Option<Value>,
    pub generated: Option<GeneratedColumn>,
}

impl ColumnDef {
    /// Whether this column is a VIRTUAL generated column (`... AS (expr) VIRTUAL`, or a
    /// bare `AS (expr)` since VIRTUAL is the default — `gencol.html`). A VIRTUAL column is
    /// the ONLY column kind that occupies no physical record slot: it is recomputed on
    /// every read and never written. A STORED generated column (`g.stored == true`) and an
    /// ordinary column are both physically stored, so both report `false`.
    ///
    /// This is the single predicate every storage-layout decision keys on — "does this
    /// column take a slot in the stored record?" — so a schema column ordinal maps to a
    /// physical slot by counting the columns before it for which this is `false`. The
    /// executor's decode/record paths and the `ALTER TABLE DROP COLUMN` rewrite both rely
    /// on it; keeping it on the type is the one source of truth so those paths cannot drift.
    pub fn is_virtual_generated(&self) -> bool {
        matches!(&self.generated, Some(g) if !g.stored)
    }
}

/// The metadata of a GENERATED column (`lang_createtable.html` § generated columns,
/// `gencol.html`): the generation expression and its storage kind.
///
/// `expr` is the parsed `AS (expr)` predicate, recorded verbatim for a later executor
/// to evaluate (it is NOT evaluated here). `stored` is `true` for `STORED` (the value
/// is materialized in the row) and `false` for `VIRTUAL` (computed on read, SQLite's
/// default when neither keyword is written). Mirrors how [`TableDef::checks`] carries a
/// not-yet-evaluated `Expr` for the planner.
#[derive(Debug, Clone)]
pub struct GeneratedColumn {
    pub expr: Expr,
    pub stored: bool,
}

/// A table's schema and the page where its b-tree root lives.
///
/// `rowid_alias` is the index into `columns` of the `INTEGER PRIMARY KEY` column
/// that aliases the rowid (`None` when the table has no such alias — a `WITHOUT
/// ROWID` table, no single-column integer primary key, or the early-SQLite
/// `PRIMARY KEY DESC` quirk). See `builder` for the exact rule.
///
/// `auto_indexes` are the implicit indexes each `UNIQUE` / `PRIMARY KEY`
/// constraint implies (`sqlite_autoindex_<name>_<N>`), derived once by
/// `builder::auto_indexes_for` and stored here so the create path (which writes
/// their `sqlite_schema` rows) and the load path (which reconstructs their
/// columns from the NULL-`sql` rows) share one source of truth and cannot drift.
#[derive(Debug, Clone)]
pub struct TableDef {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub root_page: PageId,
    pub without_rowid: bool,
    pub rowid_alias: Option<usize>,
    pub auto_indexes: Vec<AutoIndexSpec>,
    /// The table's CHECK constraint predicates (column-level and table-level unified,
    /// in declaration order), as parsed `Expr`s. The planner binds each against the
    /// table's columns and emits it onto the INSERT/UPDATE plan; the executor evaluates
    /// it per new row (cast-to-NUMERIC; 0/0.0 => violation, NULL/nonzero => ok). Mirrors
    /// how `ColumnDef.default` carries a not-yet-evaluated constraint fact for the planner.
    pub checks: Vec<Expr>,
    /// The table's FOREIGN KEY constraints (column-level `REFERENCES` and table-level
    /// `FOREIGN KEY(...) REFERENCES ...` unified, in DECLARATION order: each column's
    /// column-level FK as the columns are walked, then the table-level FKs). Recorded
    /// for the planner/executor exactly like [`checks`](TableDef::checks); ENFORCEMENT
    /// (parent-key existence, ON DELETE/UPDATE actions) is the executor's, gated on
    /// `PRAGMA foreign_keys`, and is a deliberately separate follow-up. `PRAGMA
    /// foreign_key_list` reads this vec; note it numbers the LAST-declared FK as id 0,
    /// i.e. it iterates this vec in reverse (see the pragma handler).
    pub foreign_keys: Vec<ForeignKeyDef>,
    /// True when the table has an `INTEGER PRIMARY KEY AUTOINCREMENT` column
    /// (`spec/sqlite-doc/autoinc.html`). The INSERT operator uses this to seed the next
    /// rowid from the `sqlite_sequence` high-water (monotonic, never-reused) rather than
    /// from `max(rowid)+1`. Only a column-level `INTEGER PRIMARY KEY AUTOINCREMENT` sets
    /// it — SQLite does not allow AUTOINCREMENT on a table-level PRIMARY KEY.
    pub autoincrement: bool,
    /// The table's PRIMARY KEY columns as 0-based indices into `columns`, in PRIMARY KEY
    /// DECLARATION order (empty when the table has no declared PRIMARY KEY). Covers all forms
    /// uniformly: an `INTEGER PRIMARY KEY` rowid alias, a column-level `PRIMARY KEY`, and a
    /// table-level `PRIMARY KEY(c1, c2, ...)` (composite). This is the single explicit source
    /// of PK-column order, so `PRAGMA table_info.pk` positions and `index_list.origin='pk'` do
    /// not have to reverse-engineer the key from the per-column flag + auto-index specs.
    pub primary_key: Vec<usize>,
    /// True when the table was declared with the trailing `STRICT` option
    /// (`stricttables.html`). The builder has already checked that every column declares
    /// one of the six STRICT datatypes; the executor uses this flag to reject a value of
    /// the wrong type on INSERT/UPDATE, and [`TableDef::column_affinity`] uses it to give
    /// an `ANY` column no affinity.
    pub strict: bool,
}

impl TableDef {
    /// The affinity of `columns[i]` as a stored value or comparison operand: the
    /// declared-type affinity (`datatype3.html` §3.1), except that an `ANY` column of a
    /// STRICT table has none (BLOB), so its value is kept exactly as given
    /// (`stricttables.html` §3). In an ordinary table `ANY` is NUMERIC by the usual rule.
    pub fn column_affinity(&self, i: usize) -> Affinity {
        let declared = self.columns[i].declared_type.as_deref();
        if self.strict && declared.is_some_and(|d| d.trim().eq_ignore_ascii_case("ANY")) {
            Affinity::Blob
        } else {
            affinity_of_declared_type(declared)
        }
    }
}

/// One resolved FOREIGN KEY constraint of a table (`lang_createtable.html` § foreign
/// key clause, `foreignkeys.html`), captured from a column-level `REFERENCES` or a
/// table-level `FOREIGN KEY(...) REFERENCES ...`.
///
/// `child_columns` are the columns in THIS table that reference out — the single owning
/// column for a column-level FK, or the listed columns for a table-level FK, in order.
/// `parent_table` is the referenced table. `parent_columns` are the referenced columns,
/// in order — and an EMPTY vec means "the parent table's PRIMARY KEY" (SQLite's rule for
/// a `REFERENCES` with no column list); it is deliberately NOT resolved to the PK here,
/// so the empty vec is preserved and interpreted downstream (e.g. `PRAGMA
/// foreign_key_list` prints `to` = NULL for it).
///
/// `on_delete` / `on_update` are the referential actions, each defaulting to
/// [`ReferentialAction::NoAction`] when the clause omits it (SQLite's default).
/// `deferred` is true ONLY for `DEFERRABLE INITIALLY DEFERRED` — the one timing that a
/// later deferred-enforcement pass treats differently; `NOT DEFERRABLE` and `DEFERRABLE
/// INITIALLY IMMEDIATE` are both non-deferred. The `MATCH` clause is intentionally NOT
/// recorded: SQLite parses but ignores it, and `foreign_key_list` always prints
/// `match` = `NONE`.
///
/// This is metadata only — RECORDING it is what makes FK constraints round-trip through
/// the schema and drive `PRAGMA foreign_key_list`; ENFORCING them is the executor's, a
/// separate follow-up gated on `PRAGMA foreign_keys`.
#[derive(Debug, Clone)]
pub struct ForeignKeyDef {
    pub child_columns: Vec<String>,
    pub parent_table: String,
    pub parent_columns: Vec<String>,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
    pub deferred: bool,
}

/// The per-key-column metadata an index carries ALONGSIDE its `columns` names: the
/// collation override and sort direction of one key column, in the same position as
/// the matching `columns[i]`.
///
/// `collation` is `None` when the key column inherits its collation — the target
/// column's declared `COLLATE` (see [`ColumnDef::collation`]) or, failing that,
/// `BINARY`. It is `Some(name)` only for an explicit per-column override written in the
/// key itself, e.g. `CREATE INDEX i ON t(x COLLATE NOCASE)`. Because SQLite folds a
/// key-column `COLLATE` through the expression grammar, the builder unwraps that
/// `COLLATE`-over-a-bare-column back to a plain key column with this override set (see
/// [`crate::builder`]). `descending` is true for a `DESC` key column (unspecified /
/// `ASC` is ascending).
///
/// This is metadata only — collation-aware key comparison and `DESC`-aware ordering are
/// the executor's/planner's to consume. Recording it here is what lets a `COLLATE` /
/// `DESC` index ACCEPT and round-trip through the on-disk schema instead of being
/// dropped, matching the `.db` files real `sqlite3` writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyColumn {
    pub collation: Option<String>,
    pub descending: bool,
}

/// One auto-created index implied by a `UNIQUE` / `PRIMARY KEY` constraint, in the
/// order SQLite numbers them (fileformat: `sqlite_autoindex_TABLE_N`). Derived from
/// the `CREATE TABLE` by [`crate::builder::auto_indexes_for`], the single derivation
/// both the write and read paths use.
///
/// `n` is the 1-based ordinal SQLite assigns (increasing by one with each such
/// constraint seen in declaration order); it is baked into `name`. `columns` are the
/// constrained column names in listed order, and `key_columns` is the parallel
/// per-column collation/sort metadata (same length as `columns`; a table-level
/// `UNIQUE(x COLLATE NOCASE)` records the override here). `emit_row` is true for every
/// real auto-index that owns a `sqlite_schema` row and b-tree root; it is false ONLY
/// for a `WITHOUT ROWID` table's `PRIMARY KEY`, which reserves its `n` (so later
/// `UNIQUE` constraints number past it) but owns no separate index — the table's own
/// b-tree IS that key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoIndexSpec {
    pub n: usize,
    pub name: String,
    pub columns: Vec<String>,
    pub key_columns: Vec<KeyColumn>,
    pub emit_row: bool,
}

/// An index's schema: the table and columns it covers, and its b-tree root.
///
/// `columns` are the key column names in key order (the stable read seam the executor
/// and planner consume). `key_columns` is the parallel per-column collation/sort
/// metadata — same length as `columns`, one [`KeyColumn`] per key column — carrying an
/// explicit `COLLATE` override (`None` = inherit the column's/default collation) and
/// the `DESC` flag. It is kept separate from `columns` so the read seam is unchanged
/// while a `COLLATE` / `DESC` index still round-trips faithfully.
///
/// `key_exprs` is the parallel per-key-column EXPRESSION metadata for an INDEX ON AN
/// EXPRESSION (`lang_createindex.html` §1.2, e.g. `CREATE INDEX i ON t(a+b)` /
/// `t(lower(a))`) — same length as `columns` / `key_columns`, one entry per key column.
/// `key_exprs[i] == Some(expr)` means key column `i` is a GENUINE EXPRESSION whose key
/// value is COMPUTED by evaluating `expr` against the row; `key_exprs[i] == None` means
/// an ordinary (optionally `COLLATE` / `DESC`) column whose NAME lives in `columns[i]`.
/// The parsed `Expr` is stored verbatim for a later planner to BIND (see
/// [`crate::builder`] and the planner's index-expression binder) and the executor to
/// evaluate at index-maintenance time — nothing here evaluates it. This mirrors
/// [`TableDef::checks`] exactly: the catalog already stores parsed CHECK AST that the
/// planner later binds and the executor evaluates; an expression index's key exprs are
/// stored the same way.
///
/// `partial` is true for a partial index (`CREATE INDEX ... WHERE <expr>`). It is the
/// load-bearing flag the planner reads: a planner must NOT treat a partial index as if
/// it covered every row, so it declines to use one for a general scan rather than
/// returning a complete-but-wrong result. A full (non-partial) index has
/// `partial == false`, and the invariant `partial == partial_predicate.is_some()` holds
/// for every index the builder produces.
///
/// `partial_predicate` is that partial index's parsed WHERE predicate (`Some(expr)` iff
/// `partial`), stored verbatim for a later planner to BIND and the executor to evaluate
/// per row at index-maintenance time — mirroring [`key_exprs`](IndexDef::key_exprs) and
/// [`TableDef::checks`] exactly. It is what makes DML honor the partial index's row
/// membership: only rows for which the predicate evaluates to TRUE are in the index
/// (`partialindex.html` §2 — a FALSE or NULL result omits the row), so UNIQUE is
/// enforced only across the entries that are actually IN the index. Without it, DML
/// index maintenance would index every row and a UNIQUE partial index would
/// over-enforce. Because schema RELOAD re-parses the stored `CREATE INDEX` SQL through
/// the SAME `index_def_from_ast`, the predicate is captured on create AND reopen.
#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub key_columns: Vec<KeyColumn>,
    pub key_exprs: Vec<Option<Expr>>,
    pub root_page: PageId,
    pub unique: bool,
    pub partial: bool,
    pub partial_predicate: Option<Expr>,
}

impl IndexDef {
    /// Build the [`IndexDef`] for an auto-created index from its [`AutoIndexSpec`]. An
    /// auto-index is ALWAYS `unique` and never `partial`, and its `columns` +
    /// `key_columns` come straight from the spec (the single derivation
    /// [`crate::builder::auto_indexes_for`] produces, shared by the create and load
    /// paths). Only `name`, `table`, and `root_page` vary by call site — create, the two
    /// ALTER cascades, and load — so they are passed in. This is the ONE place the
    /// auto-index field set is spelled: a new [`IndexDef`] field is then wired for every
    /// auto-index here instead of at four sites that must be kept in lockstep by hand (a
    /// missed one would ship an auto-index with, e.g., empty `key_columns`).
    pub(crate) fn from_auto_spec(
        name: String,
        table: String,
        spec: &AutoIndexSpec,
        root_page: PageId,
    ) -> Self {
        // Preconditions every caller already upholds, asserted at the ONE construction
        // site so a future fifth caller that forgets them fails loud (in debug) instead
        // of shipping a malformed index: an auto-index turned into an `IndexDef` must own
        // a b-tree row (`emit_row`; a WITHOUT ROWID PK spec reserves its ordinal but owns
        // no separate index, so it must never reach here), and `columns` / `key_columns`
        // are parallel — the same standing guard `builder` holds at the explicit
        // `CREATE INDEX` construction sites.
        debug_assert!(
            spec.emit_row,
            "from_auto_spec on a non-row-owning auto-index spec (WITHOUT ROWID PK)"
        );
        // An auto-index implied by UNIQUE / PRIMARY KEY is never an expression index — it
        // always keys ordinary named columns — so every key-expr slot is `None`, parallel
        // to `columns` / `key_columns`.
        let key_exprs = vec![None; spec.columns.len()];
        debug_assert_eq!(spec.columns.len(), spec.key_columns.len());
        debug_assert_eq!(spec.columns.len(), key_exprs.len());
        IndexDef {
            name,
            table,
            columns: spec.columns.clone(),
            key_columns: spec.key_columns.clone(),
            key_exprs,
            root_page,
            unique: true,
            partial: false,
            // An auto-index (UNIQUE / PRIMARY KEY) is never partial, so it carries no
            // predicate — the invariant `partial == partial_predicate.is_some()`.
            partial_predicate: None,
        }
    }
}

/// A view's schema entry: its name and the verbatim `CREATE VIEW` text.
///
/// The stored `sql` is the source of truth — the planner re-parses it to expand the
/// view when a query references it. The view's output column list is deliberately
/// NOT computed here: resolving it means binding the underlying `SELECT`, which needs
/// the planner and so is out of the catalog's scope. A view owns no b-tree, so unlike
/// [`TableDef`] there is no root page to record.
#[derive(Debug, Clone)]
pub struct ViewDef {
    pub name: String,
    pub sql: String,
}

/// WHERE a trigger's `ON`-target lives, RELATIVE to the store that holds the trigger —
/// the fire-time binding recorded on every [`TriggerDef`].
///
/// A cross-namespace TEMP trigger (`lang_createtrigger.html` §7 "TEMP Triggers on Non-TEMP
/// Tables") binds to its target by a STABLE identifier — the target's schema NAME, or
/// "unqualified, resolve by search order" — deliberately NOT a cached numeric [`DbIndex`].
/// A `DbIndex` is only stable WITHIN one statement: `DETACH` removes an attached store and
/// shifts every higher index down, so a `DbIndex` cached across statements in the temp
/// store's trigger cache would go stale (the trigger would silently stop firing on its real
/// target, or fire on whichever db later reused the slot). Fire-time discovery
/// ([`triggers_on_in`](crate::Catalog::triggers_on_in)) instead RE-RESOLVES this binding
/// against the connection's live namespace registry every statement, so a trigger bound to
/// `aux.u` keeps firing on `aux.u` across an unrelated `DETACH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerTargetDb {
    /// The target is in the SAME store as the trigger. Every persisted trigger (a non-TEMP
    /// trigger always shares its target's database, `lang_createtrigger.html` §2) and a
    /// TEMP trigger on a temp object, so `load` and the common create path record this and
    /// the on-disk format is unchanged.
    SameStore,
    /// A cross-namespace TEMP trigger whose target is in the namespace named `schema`
    /// (`"main"` or an ATTACH alias — never `"temp"`, which is same-store). Resolved to a
    /// live [`DbIndex`] through the connection registry at fire time, so it survives
    /// `DETACH` remapping and reconstructs exactly on a rollback-resync reload of the
    /// qualified stored SQL.
    ForeignSchema(String),
    /// A cross-namespace TEMP trigger whose `ON`-target was UNQUALIFIED: its namespace is
    /// re-resolved by name-resolution search order (temp, main, attached) at fire time —
    /// the doc's "an unqualified TEMP trigger reattaches to a same-named table in another
    /// database whenever a schema change occurs" behavior. Recorded for an unqualified
    /// foreign target at create time and on a reload whose stored SQL carries no qualifier.
    ForeignUnqualified,
}

impl TriggerTargetDb {
    /// Is the target in the SAME store as the trigger (the common, persisted shape)?
    pub fn is_same_store(&self) -> bool {
        matches!(self, TriggerTargetDb::SameStore)
    }

    /// Is this a CROSS-NAMESPACE (foreign-bound) TEMP trigger — the §7 case that only the
    /// temp store ever holds?
    pub fn is_foreign(&self) -> bool {
        !self.is_same_store()
    }
}

/// A trigger's schema entry: its name, the target table it fires on, and the verbatim
/// `CREATE TRIGGER` text.
///
/// `table` is the trigger's TARGET table's BARE name (the `ON <table>` of the `CREATE
/// TRIGGER`, without any schema qualifier), recorded so a `DROP TABLE` can cascade the
/// triggers that fire on it (fileformat2 stores it as the row's `tbl_name`). As with a
/// view, `sql` is the source of truth: the trigger's actions are re-parsed from it when
/// it fires. A trigger owns no b-tree, so there is no root page.
///
/// `target` records WHICH namespace the target lives in relative to this trigger's store
/// (see [`TriggerTargetDb`]). It is [`SameStore`](TriggerTargetDb::SameStore) for every
/// persisted trigger and a temp-on-temp trigger (the on-disk format is unchanged); the
/// foreign variants are held only in the in-memory temp store and are the fire-time key
/// that matches a write to `(db, table)` and distinguishes a temp trigger bound to `main.t`
/// from one bound to a shadowing `temp.t`.
#[derive(Debug, Clone)]
pub struct TriggerDef {
    pub name: String,
    pub table: String,
    pub sql: String,
    pub target: TriggerTargetDb,
}

impl TriggerDef {
    /// A trigger whose target is in the SAME store as the trigger itself — every
    /// persisted trigger and a temp trigger on a temp object
    /// ([`TriggerTargetDb::SameStore`]). The overwhelmingly common shape, so it earns a
    /// named constructor; a cross-namespace temp trigger sets `target` explicitly instead.
    pub fn same_store(name: String, table: String, sql: String) -> TriggerDef {
        TriggerDef { name, table, sql, target: TriggerTargetDb::SameStore }
    }
}
