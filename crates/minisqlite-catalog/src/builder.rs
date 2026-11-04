//! The `CREATE TABLE` / `CREATE INDEX` AST -> def builders. Both the write side
//! (persisting a fresh `CREATE ...`) and the load side (rebuilding the cache from
//! stored `sql`) go through the same builder for each object, so a schema object's
//! in-memory shape is derived one way and the two can never disagree.
//!
//! The subtle part of the table builder is `rowid_alias` — which column, if any, is
//! the `INTEGER PRIMARY KEY` that aliases the rowid. The rule is transcribed from
//! `spec/sqlite-doc/lang_createtable.html` §5 and pinned by the unit tests below.

use minisqlite_sql::{
    ColumnConstraintKind, ColumnDef as SqlColumnDef, CreateIndex, CreateTable, CreateTableBody,
    Deferrable, DefaultValue, Expr, ForeignKeyAction, ForeignKeyClause, FrameBound, FunctionArgs,
    IndexedColumn, IndexedColumnTarget, InRhs, InitiallyTiming, Literal, OverClause,
    ReferentialAction, SortOrder, TableConstraint, TableConstraintKind, WindowFrame, WindowSpec,
};
use minisqlite_pager::PageId;
use minisqlite_types::{Error, Result, Value};

use crate::def::{
    AutoIndexSpec, ColumnDef, ForeignKeyDef, GeneratedColumn, IndexDef, KeyColumn, TableDef,
};

/// Build a [`TableDef`] from a parsed `CREATE TABLE` and the b-tree root page the
/// table's rows live in.
///
/// `CREATE TABLE ... AS SELECT` is a deferred gap: its column shape comes from
/// running the query, which the catalog cannot do, so this returns an error rather
/// than fabricating columns.
pub(crate) fn table_def_from_ast(stmt: &CreateTable, root_page: PageId) -> Result<TableDef> {
    let (columns, constraints, options) = match &stmt.body {
        CreateTableBody::Columns { columns, constraints, options } => (columns, constraints, options),
        CreateTableBody::AsSelect(_) => {
            return Err(Error::Sql(
                "CREATE TABLE ... AS SELECT is not yet supported in the catalog".into(),
            ));
        }
    };

    // CHECK predicates, column-level and table-level unified in declaration order: each
    // column's checks as its constraints are walked (below), then the table-level checks
    // (after the column loop). SQLite does not make the order observable for correctness,
    // but keeping it stable makes the def deterministic. The planner binds each against
    // the table's columns; the executor evaluates it per new row.
    let mut checks: Vec<Expr> = Vec::new();

    // FOREIGN KEY constraints, column-level and table-level unified in DECLARATION order:
    // each column's column-level FK as its constraints are walked (below), then the
    // table-level FKs (after the column loop). Recorded like `checks`; enforcement is a
    // separate follow-up. `PRAGMA foreign_key_list` numbers the LAST-declared FK as id 0,
    // so it iterates this declaration-ordered vec in reverse (see the pragma handler).
    let mut foreign_keys: Vec<ForeignKeyDef> = Vec::new();

    let mut cols = Vec::with_capacity(columns.len());
    for c in columns {
        let mut cd = ColumnDef {
            name: c.name.clone(),
            declared_type: c.type_name.clone(),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: None,
            default: None,
            default_value: None,
            generated: None,
        };
        for cons in &c.constraints {
            match &cons.kind {
                ColumnConstraintKind::NotNull { .. } => cd.not_null = true,
                ColumnConstraintKind::Unique { .. } => cd.unique = true,
                ColumnConstraintKind::PrimaryKey { .. } => cd.primary_key = true,
                ColumnConstraintKind::Collate(name) => cd.collation = Some(name.clone()),
                // Store both the raw text (for the INSERT planner / PRAGMA) and, when the
                // default folds to a constant, its evaluated Value — so a short row's
                // missing column can decode to the default without re-parsing per row.
                ColumnConstraintKind::Default(d) => {
                    cd.default = render_default_text(d);
                    cd.default_value = eval_constant_default(d);
                }
                // A column-level `CHECK(expr)` is just a check that references this column
                // (lang_createtable.html §3.7); collect its raw predicate for the planner,
                // identical to a table-level check.
                ColumnConstraintKind::Check(e) => checks.push(e.clone()),
                // A column-level `REFERENCES ...` is a single-column FK whose child column
                // is this column (lang_createtable.html § foreign key clause). Record it in
                // declaration order alongside the table-level FKs below.
                ColumnConstraintKind::ForeignKey(clause) => {
                    foreign_keys.push(foreign_key_def(vec![c.name.clone()], clause));
                }
                // A GENERATED column carries its generation expression + storage kind
                // (lang_createtable.html § generated columns). Record the fact for a later
                // executor; nothing evaluates the expression here.
                ColumnConstraintKind::Generated { expr, stored } => {
                    cd.generated = Some(GeneratedColumn { expr: expr.clone(), stored: *stored });
                }
                // A bare `NULL` constraint imposes nothing and models nothing. Listed
                // explicitly (no wildcard) so a newly-added constraint kind is a compile
                // error here, forcing a decision on whether it maps to a column fact.
                ColumnConstraintKind::Null { .. } => {}
            }
        }
        cols.push(cd);
    }

    // Table-level `CHECK(expr)` constraints, in declaration order, appended after the
    // column-level checks. The other table constraints (PRIMARY KEY / UNIQUE / FOREIGN
    // KEY) are consumed elsewhere (`compute_rowid_alias` / `auto_indexes_for`), so this
    // loop only harvests checks. Listed explicitly (no wildcard), like the column-level
    // loop above, so a newly-added `TableConstraintKind` is a compile error here that
    // forces a decision on whether it also carries a check to harvest.
    for cons in constraints {
        match &cons.kind {
            TableConstraintKind::Check(e) => checks.push(e.clone()),
            // A table-level `FOREIGN KEY(cols) REFERENCES ...` names its own child columns.
            // Appended after the column-level FKs, preserving overall declaration order.
            TableConstraintKind::ForeignKey { columns, clause } => {
                foreign_keys.push(foreign_key_def(columns.clone(), clause));
            }
            // PRIMARY KEY / UNIQUE are consumed elsewhere (`compute_rowid_alias` /
            // `auto_indexes_for`); nothing to harvest here. Listed explicitly (no
            // wildcard) so a newly-added `TableConstraintKind` is a compile error.
            TableConstraintKind::PrimaryKey { .. } | TableConstraintKind::Unique { .. } => {}
        }
    }

    let without_rowid = options.without_rowid;
    let table_name = stmt.name.name.as_str();

    // Create-time structural validation the purely-syntactic parser cannot do. Every rule
    // below rejects a `CREATE TABLE` that real sqlite errors on at create time, failing closed
    // with `Err(Error::Sql(..))` BEFORE the def is built — the same fail-closed pattern as
    // `validate_autoincrement`, and enforced in the SINGLE builder both the create path and
    // the load path funnel through: an illegal `CREATE TABLE` (or an `ADD COLUMN` that
    // rewrites a STRICT table's sql) is rejected before anything is persisted, and a corrupt
    // foreign schema fails closed on load. The order follows SQLite's rough create-time
    // precedence — duplicate column (as each column is added), then a second PRIMARY KEY and
    // the AUTOINCREMENT placement (as the offending clause is added), then the end-of-table
    // checks (a table constraint over an unknown column, the generated-column restrictions, a
    // WITHOUT ROWID table missing its PRIMARY KEY, then the STRICT datatypes). Only
    // single-violation precedence is pinned; exotic multi-violation ordering is best-effort.
    validate_no_duplicate_columns(columns)?;
    validate_single_primary_key(table_name, columns, constraints)?;
    // AUTOINCREMENT is a schema-validation rule, not a syntax one: the parser accepts
    // `PRIMARY KEY AUTOINCREMENT` on any column, so the catalog builder is where the
    // spec's placement restriction is enforced (autoinc.html §3).
    validate_autoincrement(columns, constraints, without_rowid)?;
    validate_table_constraint_columns(table_name, columns, constraints)?;
    // A table-level `FOREIGN KEY(<child_cols>) REFERENCES ...` must name child columns the
    // table actually declares. Unlike the parent-side FK errors — which need the parent's
    // definition too and so are DML errors deferred to statement-prepare (foreignkeys.html
    // §3) — child-column existence is resolvable from this one table, so it is a create-time
    // reject, the same single-table-definition reasoning as `validate_table_constraint_columns`.
    validate_foreign_key_child_columns(columns, constraints)?;
    // Generated-column restrictions (gencol.html §2.3): a generated column may not carry a
    // DEFAULT nor be part of the PRIMARY KEY, and a table needs at least one non-generated
    // column. The expression-level restrictions (no subquery / aggregate / self-reference /
    // direct ROWID) need binding/dependency analysis and are a separate binder/planner slice.
    validate_generated_columns(columns, constraints)?;
    // A column `DEFAULT (<expr>)` must be a CONSTANT expression (lang_createtable.html §3.2):
    // no column/table reference, bound parameter, or sub-query. The purely-syntactic parser
    // sends every parenthesized `DEFAULT (...)` through the full expression grammar, so a
    // non-constant one slips through as a `DefaultValue::Expr`; this is the create-time
    // fail-closed check matching real sqlite's `default value of column [<col>] is not
    // constant`. Ordered AFTER the generated-column rules so a generated column that ALSO
    // carries a DEFAULT keeps the more specific "cannot use DEFAULT on a generated column".
    validate_constant_defaults(columns)?;
    // A CHECK constraint's expression may not contain a sub-query (lang_createtable.html).
    // The parser accepts one, so this is the create-time fail-closed reject. Runs over the
    // already-unified `checks` vec (column- and table-level in one pass).
    validate_check_constraints(&checks)?;
    // A WITHOUT ROWID table keys its rows by the PRIMARY KEY it MUST declare
    // (withoutrowid.html); sqlite raises this at end-of-table.
    validate_without_rowid_has_primary_key(table_name, columns, constraints, without_rowid)?;
    validate_strict_datatypes(table_name, columns, options.strict)?;
    let rowid_alias = compute_rowid_alias(columns, constraints, without_rowid);
    // The explicit, ordered PK-column list (see `TableDef::primary_key`). Derived from the
    // SAME constraints `compute_rowid_alias`/`auto_indexes_for` reason about, via the shared
    // `bare_column_name` unwrap, so the three cannot drift on which columns the key names.
    let primary_key = primary_key_column_indices(columns, constraints);
    let auto_indexes = auto_indexes_for(stmt);
    // `validate_autoincrement` above has already rejected every illegal placement, so a
    // present autoincrement column here is a *valid* `INTEGER PRIMARY KEY AUTOINCREMENT`
    // (autoinc.html §3). Reuse the single `autoincrement_column` predicate so this flag
    // cannot drift from the placement check or the `sqlite_sequence` auto-create trigger.
    let autoincrement_col = autoincrement_column(columns);
    let autoincrement = autoincrement_col.is_some();
    // A valid AUTOINCREMENT column IS the rowid alias: `validate_autoincrement` fails closed
    // unless the autoincrement column is exactly the INTEGER-PK alias of a rowid table, so by
    // here `autoincrement` implies `rowid_alias` names that very column. Pin the coupling at
    // the construction site (mirroring the standing guards in `IndexDef::from_auto_spec`) so a
    // future reordering — the flag computed before validation, or a changed alias rule — fails
    // loud here instead of shipping a def whose flag and alias disagree and misdirects the
    // INSERT rowid-seeding path.
    debug_assert!(
        !autoincrement || rowid_alias == autoincrement_col,
        "autoincrement set but rowid_alias ({rowid_alias:?}) is not the autoincrement column ({autoincrement_col:?})"
    );

    Ok(TableDef {
        name: stmt.name.name.clone(),
        columns: cols,
        root_page,
        without_rowid,
        rowid_alias,
        auto_indexes,
        checks,
        foreign_keys,
        autoincrement,
        primary_key,
        strict: options.strict,
    })
}

/// The table's PRIMARY KEY columns as 0-based indices into `columns`, in PRIMARY KEY
/// DECLARATION order (empty when the table declares no PRIMARY KEY) — the value stored in
/// [`TableDef::primary_key`].
///
/// This is the ONE place the ordered PK-column list is derived, and it routes the table-level
/// case through the SAME [`named_columns`] classifier [`auto_indexes_for`] uses, so it cannot
/// describe a different PK than [`auto_indexes_for`] (or, via that shared unwrap,
/// [`compute_rowid_alias`]):
///
/// - A TABLE-level `PRIMARY KEY(c1, c2, ...)`: [`named_columns`] resolves each member to the
///   column it names — unwrapping a `COLLATE`/`DESC` wrapper — in declaration order, and it is
///   ALL-OR-NOTHING. A GENUINE expression term (`PRIMARY KEY(a, b + c)`), which real sqlite
///   rejects in a PK and this engine leaves as a deferred gap, makes [`named_columns`] yield
///   `None` and [`auto_indexes_for`] emit NO auto-index for the key — so this returns EMPTY
///   too, the two surfaces agreeing on "no modelled PK" rather than this reporting a PARTIAL
///   key `index_list` never produces. On the bare-column case
///   [`validate_table_constraint_columns`] has already proven every name exists, so each index
///   lookup resolves.
/// - Else a COLUMN-level `PRIMARY KEY` (at most one, guaranteed by
///   [`validate_single_primary_key`]): its single column index. This covers `INTEGER PRIMARY
///   KEY` (the rowid alias) and `INTEGER PRIMARY KEY DESC` (still the PK — position 1 —
///   though not the rowid alias) uniformly.
/// - Else: empty.
///
/// The two forms are mutually exclusive after [`validate_single_primary_key`] (at most one
/// PRIMARY KEY clause total), so the table-level branch taking precedence is unambiguous.
fn primary_key_column_indices(
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
) -> Vec<usize> {
    for cons in constraints {
        if let TableConstraintKind::PrimaryKey { columns: pk_cols, .. } = &cons.kind {
            // All-or-nothing, exactly like `auto_indexes_for`: `named_columns` yields `None`
            // if ANY term is a genuine expression, in which case that path emits no
            // auto-index — so model no PK here either, never a partial key the index side
            // never produces.
            let Some((names, _)) = named_columns(pk_cols) else {
                return Vec::new();
            };
            let indices: Vec<usize> = names
                .iter()
                .filter_map(|name| columns.iter().position(|c| c.name.eq_ignore_ascii_case(name)))
                .collect();
            // `named_columns` returned `Some`, so every name is a bare column
            // `validate_table_constraint_columns` already proved exists: the filter_map drops
            // nothing. Assert that here (like the rowid_alias/autoincrement coupling above) so
            // a future regression in the validation order fails LOUD instead of silently
            // yielding a PK shorter than the auto-index's columns — which would quietly flip
            // `index_is_primary_key` from 'pk' to 'u'.
            debug_assert_eq!(
                indices.len(),
                names.len(),
                "every table-level PK column name resolves to a column index"
            );
            return indices;
        }
    }
    columns
        .iter()
        .position(|c| {
            c.constraints
                .iter()
                .any(|k| matches!(k.kind, ColumnConstraintKind::PrimaryKey { .. }))
        })
        .map(|idx| vec![idx])
        .unwrap_or_default()
}

/// Build a [`ForeignKeyDef`] from a resolved child-column list and its parsed
/// [`ForeignKeyClause`]. The single home for the AST -> def projection, shared by the
/// column-level (`REFERENCES`) and table-level (`FOREIGN KEY(...)`) capture sites so the
/// two cannot drift. `on_delete` / `on_update` default to [`ReferentialAction::NoAction`]
/// (SQLite's default) when the clause omits them; a later duplicate wins (matching the
/// last `ON DELETE` / `ON UPDATE` written). `MATCH` is dropped (SQLite ignores it).
fn foreign_key_def(child_columns: Vec<String>, clause: &ForeignKeyClause) -> ForeignKeyDef {
    let mut on_delete = ReferentialAction::NoAction;
    let mut on_update = ReferentialAction::NoAction;
    for action in &clause.actions {
        match action {
            ForeignKeyAction::OnDelete(a) => on_delete = *a,
            ForeignKeyAction::OnUpdate(a) => on_update = *a,
        }
    }
    ForeignKeyDef {
        child_columns,
        parent_table: clause.table.clone(),
        parent_columns: clause.columns.clone(),
        on_delete,
        on_update,
        deferred: is_deferred(clause.deferrable),
    }
}

/// Whether a foreign key is `DEFERRABLE INITIALLY DEFERRED` — the only timing a later
/// deferred-enforcement pass treats differently. `NOT DEFERRABLE` (`not == true`) and
/// `DEFERRABLE INITIALLY IMMEDIATE` (and the unspecified default) are all NOT deferred.
fn is_deferred(deferrable: Option<Deferrable>) -> bool {
    matches!(
        deferrable,
        Some(Deferrable { not: false, initially: Some(InitiallyTiming::Deferred) })
    )
}

/// Derive, in SQLite's numbering order, the auto-created indexes a `CREATE TABLE`'s
/// `UNIQUE` / `PRIMARY KEY` constraints imply (`schematab.html`, the
/// `sqlite_autoindex_TABLE_N` paragraph). This is the ONE place the rule lives: the
/// create path persists a `sqlite_schema` row per [`AutoIndexSpec`] and the load path
/// reconstructs each auto-index's columns from the matching spec, so the two cannot
/// disagree.
///
/// The rule, verbatim from the spec:
/// - A `UNIQUE` or `PRIMARY KEY` constraint (column-level OR table-level) causes an
///   index named `sqlite_autoindex_<TABLE>_<N>`, TABLE being the table's original
///   spelling, N an integer starting at 1 and increasing by one with EACH such
///   constraint seen in the table definition, in declaration (textual) order.
/// - Declaration order = walk the column definitions in order and, within each, its
///   constraints in order (a column-level PK/UNIQUE is "seen" at that column's
///   position), THEN walk the table-level constraints in order.
/// - The `INTEGER PRIMARY KEY` is NEVER given a name and NEVER consumes an N — in a
///   rowid table (it IS the rowid; there is no separate index) AND in a `WITHOUT ROWID`
///   table (schematab.html: the name "is never allocated for an INTEGER PRIMARY KEY,
///   either in rowid tables or WITHOUT ROWID tables"). Identified via the
///   `compute_rowid_alias` integer-PK rule, applied regardless of the table's kind.
/// - A NON-integer `WITHOUT ROWID` `PRIMARY KEY` gets NO `sqlite_schema` row (the
///   table's own b-tree IS the PK index), BUT its N is still reserved "as if the entry
///   existed", so later `UNIQUE` constraints number past it — such a PK yields a spec
///   with `emit_row == false`. `UNIQUE` constraints in a WITHOUT ROWID table DO emit
///   real auto-index rows.
///
/// A table-level PK/UNIQUE over a GENUINE expression (`UNIQUE(a+b)`) is a deferred gap
/// — its N is still consumed (matching real SQLite's numbering of the constraints we
/// CAN model) but no spec is emitted, the same way `index_def_from_ast` declines an
/// expression index. A per-column `COLLATE` over a bare column (`UNIQUE(x COLLATE
/// NOCASE)`), which the parser lands as a `COLLATE`-over-column expression target, is
/// NOT such a gap: [`normalize_key_column`] unwraps it to a plain key column with a
/// collation override, so its auto-index row IS emitted (the override carried on the
/// spec's `key_columns`).
pub(crate) fn auto_indexes_for(stmt: &CreateTable) -> Vec<AutoIndexSpec> {
    let CreateTableBody::Columns { columns, constraints, options } = &stmt.body else {
        // `AS SELECT` carries no column-list constraints (and `table_def_from_ast`
        // rejects it before this is reached); nothing to derive.
        return Vec::new();
    };
    let table = &stmt.name.name;
    let without_rowid = options.without_rowid;

    // The INTEGER PRIMARY KEY is never given a `sqlite_autoindex` name and never
    // consumes an N — both when it is the rowid alias of a rowid table AND in a WITHOUT
    // ROWID table (schematab.html). This is the ONE case that differs from a WITHOUT
    // ROWID *non-integer* PK, which DOES reserve an N (emit_row=false). Since
    // `compute_rowid_alias` short-circuits to None for a WITHOUT ROWID table (no rowid
    // to alias), ask it which column is the integer PK *as if* this were a rowid table
    // (`without_rowid = false`) to get the column to exclude regardless of table kind. A
    // column-level `INTEGER PRIMARY KEY DESC` is not an integer alias by that rule, so
    // it still reserves its N — matching the rowid-alias DESC quirk.
    let integer_pk = compute_rowid_alias(columns, constraints, false);

    // `compute_rowid_alias` accepts a column-level integer PK only when exactly one
    // column carries a PRIMARY KEY (and no table-level PK exists), so the presence of
    // ANY column-level PK tells us the integer PK, if there is one, is column-level;
    // otherwise a lone table-level PRIMARY KEY is the candidate.
    let alias_is_column_level = columns
        .iter()
        .any(|c| c.constraints.iter().any(|k| matches!(k.kind, ColumnConstraintKind::PrimaryKey { .. })));

    let mut specs = Vec::new();
    let mut n: usize = 0;

    // Pass A: each column, its constraints in order. A column-level PK/UNIQUE
    // auto-index inherits its key's collation from the column's own declared `COLLATE`
    // (`ColumnDef.collation`), so its `KeyColumn.collation` stays `None` (= inherit) —
    // it is NOT duplicated here. A column-level `PRIMARY KEY DESC` on a non-alias column
    // does carry its sort direction into the auto-index key.
    for (i, col) in columns.iter().enumerate() {
        for cons in &col.constraints {
            match &cons.kind {
                ColumnConstraintKind::PrimaryKey { order, .. } => {
                    if integer_pk == Some(i) {
                        // INTEGER PRIMARY KEY: never named, never consumes an N — the
                        // rowid alias in a rowid table, and excluded outright in a
                        // WITHOUT ROWID table too (schematab.html).
                        continue;
                    }
                    n += 1;
                    specs.push(AutoIndexSpec {
                        n,
                        name: autoindex_name(table, n),
                        columns: vec![col.name.clone()],
                        key_columns: vec![KeyColumn {
                            collation: None,
                            descending: matches!(order, Some(SortOrder::Desc)),
                        }],
                        // A WITHOUT ROWID PK reserves N but owns no separate index.
                        emit_row: !without_rowid,
                    });
                }
                ColumnConstraintKind::Unique { .. } => {
                    n += 1;
                    specs.push(AutoIndexSpec {
                        n,
                        name: autoindex_name(table, n),
                        columns: vec![col.name.clone()],
                        // A column-level UNIQUE has no sort order in the grammar; ASC.
                        key_columns: vec![KeyColumn { collation: None, descending: false }],
                        // A UNIQUE always owns a real index, even in a WITHOUT ROWID table.
                        emit_row: true,
                    });
                }
                _ => {}
            }
        }
    }

    // Pass B: table-level constraints, in order (they textually follow all columns).
    for cons in constraints {
        match &cons.kind {
            TableConstraintKind::PrimaryKey { columns: pk_cols, .. } => {
                if integer_pk.is_some() && !alias_is_column_level {
                    // The lone table-level PRIMARY KEY over a single INTEGER column: the
                    // rowid alias in a rowid table, and excluded outright in a WITHOUT
                    // ROWID table too — no name, no N either way.
                    continue;
                }
                // Each constraint seen advances N even when its columns are an
                // (unmodelled) expression, so a following named constraint keeps the
                // N real SQLite would give it.
                n += 1;
                if let Some((cols, key_cols)) = named_columns(pk_cols) {
                    specs.push(AutoIndexSpec {
                        n,
                        name: autoindex_name(table, n),
                        columns: cols,
                        key_columns: key_cols,
                        emit_row: !without_rowid,
                    });
                }
            }
            TableConstraintKind::Unique { columns: uq_cols, .. } => {
                n += 1;
                if let Some((cols, key_cols)) = named_columns(uq_cols) {
                    specs.push(AutoIndexSpec {
                        n,
                        name: autoindex_name(table, n),
                        columns: cols,
                        key_columns: key_cols,
                        emit_row: true,
                    });
                }
            }
            _ => {}
        }
    }

    specs
}

/// The `sqlite_autoindex_<table>_<n>` name, TABLE the table's original spelling.
fn autoindex_name(table: &str, n: usize) -> String {
    format!("sqlite_autoindex_{table}_{n}")
}

/// The plain column NAME an [`IndexedColumn`] target denotes, or `None` if the target is
/// a GENUINE expression. Both a `Name(x)` target and the `x COLLATE <c>` shape the parser
/// folds to `Expr(Collate { Column(x), .. })` name the SAME bare column `x` (the parser
/// unwraps a bare `Expr::Column { schema: None, table: None, .. }` to a `Name` in
/// `ddl.rs`); a real expression — `a + 1`, `lower(a)`, or a `COLLATE` over a non-column —
/// names none. The collation override and sort order do NOT change WHICH column is named,
/// so they are ignored here.
///
/// This is the ONE definition of "is this target a plain column, and which one", shared by
/// [`normalize_key_column`] (which builds the key-column metadata for create / table-level
/// constraints) and [`compute_rowid_alias`] (table-level INTEGER-PK detection). Both MUST
/// agree — `auto_indexes_for` decides whether to emit a table-level PK's auto-index from
/// `compute_rowid_alias`, so if the two disagreed on `PRIMARY KEY(x COLLATE NOCASE)` the
/// schema model would be self-contradictory (x both "not the rowid alias" and "owns an
/// auto-index over itself"). Keeping the unwrap in one place makes that drift impossible.
fn bare_column_name(ic: &IndexedColumn) -> Option<&str> {
    match &ic.target {
        IndexedColumnTarget::Name(name) => Some(name),
        IndexedColumnTarget::Expr(Expr::Collate { expr, .. }) => match expr.as_ref() {
            Expr::Column { schema: None, table: None, name, .. } => Some(name),
            _ => None,
        },
        IndexedColumnTarget::Expr(_) => None,
    }
}

/// Classify each `CREATE INDEX` key column as an ordinary named column or a GENUINE
/// EXPRESSION (`lang_createindex.html` §1.2), returning a slot parallel to
/// `stmt.columns`: `None` for a plain column, `Some(expr)` for an expression key whose
/// value is COMPUTED from the row.
///
/// This is the SHARED definition of "which indexed columns are genuine expressions",
/// used by both the catalog builder ([`index_def_from_ast`]) and the engine's
/// `CREATE INDEX` backfill path (which compiles these exprs to key the new index's
/// entries). It classifies through [`bare_column_name`] so it agrees exactly with the
/// key-column builder: a plain `Name`, and the `x COLLATE <c>` shape the parser folds to
/// a `COLLATE`-over-a-bare-column, both return `Some(name)` there and so are `None`
/// (ordinary) here; only a real expression — `a + 1`, `lower(a)`, or a `COLLATE` wrapping
/// a non-column expression — has no bare-column name and yields `Some(<the target expr>)`.
pub fn index_ast_key_exprs(stmt: &CreateIndex) -> Vec<Option<Expr>> {
    stmt.columns
        .iter()
        .map(|ic| match &ic.target {
            // A bare column (a `Name`, or the `COLLATE`-over-column shape `bare_column_name`
            // unwraps) is an ordinary key column, never an expression.
            _ if bare_column_name(ic).is_some() => None,
            // A genuine expression: its `Expr` payload is the key expression to compute.
            IndexedColumnTarget::Expr(e) => Some(e.clone()),
            // Unreachable: a `Name` target always classifies as a bare column above.
            IndexedColumnTarget::Name(_) => None,
        })
        .collect()
}

/// Normalize one parsed [`IndexedColumn`] into a plain key column: the column NAME it
/// denotes plus its [`KeyColumn`] metadata (collation override + sort direction).
///
/// A per-column `COLLATE` in an indexed-column list is a plain key column with an
/// explicit collation, NOT an expression index — but because `COLLATE` binds as an
/// expression operator, the parser lands `x COLLATE NOCASE` as an
/// `Expr(Collate { Column(x), "NOCASE" })` target (with `IndexedColumn.collation` left
/// `None`) rather than a `Name` target, exactly the shape `minisqlite-sql`'s
/// `parse_indexed_column` builds. This unwraps that one shape — a `COLLATE` over a BARE
/// COLUMN reference, the same `Expr::Column { schema: None, table: None, .. }` the parser
/// folds to `Name` — back to the plain key column it denotes, recording the collation as
/// an override. A `Name` target that already carries `collation` / `order` (should the
/// parser start modelling it directly) is handled the same way.
///
/// A GENUINE expression target — `a + 1`, `lower(a)`, or a `COLLATE` wrapping a
/// non-column expression — has no plain column NAME, so it is an honest [`Error::Sql`]
/// here rather than a fabricated key column. Only the `COLLATE`-over-bare-column case is
/// unwrapped. This helper serves the table-level-constraint path ([`named_columns`], used
/// by `auto_indexes_for`), where a genuine expression (`UNIQUE(a+b)`) is still a deferred
/// gap; the caller `.ok()?`s the error to skip it. A genuine expression INDEX
/// (`CREATE INDEX i ON t(a+b)`) is instead accepted by [`index_def_from_ast`], which
/// captures its key expression rather than calling this helper for it.
fn normalize_key_column(ic: &IndexedColumn) -> Result<(String, KeyColumn)> {
    // `bare_column_name` is the shared classifier: it accepts a `Name` target and the
    // `x COLLATE <c>` shape (a `COLLATE` over a bare column) and rejects a genuine
    // expression. A rejection here is a genuine-expression table-level constraint gap.
    let name = bare_column_name(ic)
        .ok_or_else(|| Error::Sql("indexed constraint on an expression is not yet supported".into()))?;
    let descending = matches!(ic.order, Some(SortOrder::Desc));
    // The collation override rides on the `COLLATE` wrapper for the `x COLLATE <c>` shape;
    // a `Name` target carries it in `ic.collation` (populated only if the parser starts
    // modelling `Name COLLATE x` directly). By construction the target is one of those two
    // here — a genuine expression was already rejected above.
    let collation = match &ic.target {
        IndexedColumnTarget::Expr(Expr::Collate { collation, .. }) => Some(collation.clone()),
        _ => ic.collation.clone(),
    };
    Ok((name.to_string(), KeyColumn { collation, descending }))
}

/// The indexed columns of a table-level constraint as plain key columns — their names
/// and the parallel per-column [`KeyColumn`] metadata — or `None` if any target is a
/// GENUINE expression (an expression index is not modelled; the caller treats it as a
/// deferred gap whose N is still consumed). A `COLLATE`-over-column target IS a plain
/// key column and does NOT force `None`; its collation override is carried through.
fn named_columns(cols: &[IndexedColumn]) -> Option<(Vec<String>, Vec<KeyColumn>)> {
    let mut names = Vec::with_capacity(cols.len());
    let mut keys = Vec::with_capacity(cols.len());
    for c in cols {
        let (name, key) = normalize_key_column(c).ok()?;
        names.push(name);
        keys.push(key);
    }
    debug_assert_eq!(names.len(), keys.len());
    Some((names, keys))
}

/// Build an [`IndexDef`] from a parsed `CREATE INDEX` and the b-tree root page its
/// entries live in. Shared by the write side (persisting a fresh `CREATE INDEX`)
/// and the load side (rebuilding the cache from stored `sql`), the same rationale as
/// [`table_def_from_ast`] — one derivation, so create and reload cannot disagree.
///
/// `table_name` / `table_columns` come from the already-resolved target table; each
/// key column is checked to exist on it (ASCII case-insensitively, since
/// `CREATE INDEX i ON t(A)` indexes a column declared `a`).
///
/// A per-column `COLLATE` / `ASC` / `DESC` is captured: `columns` keeps the key names
/// (the stable read seam) and `key_columns` carries the parallel collation-override +
/// sort-direction metadata. A `COLLATE`-over-a-bare-column, which the parser lands as an
/// expression target, is unwrapped by [`normalize_key_column`] to the plain key column
/// it denotes — so it is ACCEPTED, not refused.
///
/// A GENUINE expression index — `CREATE INDEX ... ON t(a+1)` or `t(lower(a))`
/// (`lang_createindex.html` §1.2) — is captured too: an expression key column has no
/// name, so it stores an EMPTY-STRING sentinel in `columns`, its `COLLATE`/`DESC` on the
/// parallel `key_columns` entry, and the parsed key expression in `key_exprs[i]` for a
/// later planner to bind and the executor to evaluate at index-maintenance time (mirroring
/// how a table's CHECK predicates are stored parsed and bound later). Its column
/// references are NOT existence-checked here — the planner validates them when it binds
/// the expression — so only an ordinary named key column is checked against the table.
pub(crate) fn index_def_from_ast(
    stmt: &CreateIndex,
    table_name: &str,
    table_columns: &[ColumnDef],
    root_page: PageId,
) -> Result<IndexDef> {
    let mut columns = Vec::with_capacity(stmt.columns.len());
    let mut key_columns = Vec::with_capacity(stmt.columns.len());
    // Parallel per-key-column expression slot (see [`IndexDef::key_exprs`]): `None` for an
    // ordinary named column, `Some(expr)` for a genuine expression key. Built in lockstep
    // with `columns` / `key_columns`.
    let mut key_exprs: Vec<Option<Expr>> = Vec::with_capacity(stmt.columns.len());
    for ic in &stmt.columns {
        match bare_column_name(ic) {
            // Ordinary (optionally `COLLATE` / `DESC`) NAMED key column: existence-check
            // the name and record it; no key expression.
            Some(_) => {
                let (name, key) = normalize_key_column(ic)?;
                if !table_columns.iter().any(|c| c.name.eq_ignore_ascii_case(&name)) {
                    return Err(Error::Sql(format!("no such column: {name}")));
                }
                columns.push(name);
                key_columns.push(key);
                key_exprs.push(None);
            }
            // A GENUINE expression key (`lang_createindex.html` §1.2): it has no column
            // name, so store an empty-string sentinel in `columns`, capture the target
            // expression for the planner to bind, and mirror `normalize_key_column`'s
            // collation extraction (a trailing `COLLATE` overrides the key's collation).
            // Its column refs are validated when the planner binds them, so skip the
            // existence check here.
            None => {
                let IndexedColumnTarget::Expr(e) = &ic.target else {
                    unreachable!("bare_column_name returns Some for a Name target");
                };
                // An index key expression may not use a sub-query (expridx.html /
                // lang_createindex.html). The parser accepts one; reject it here at create
                // time, matching real sqlite (which errors at create time). Column refs in
                // the expression are legitimate and bound later, so only a sub-query fails.
                if expr_contains_subquery(e) {
                    return Err(Error::Sql("subqueries prohibited in index expressions".into()));
                }
                let collation = match e {
                    Expr::Collate { collation, .. } => Some(collation.clone()),
                    _ => ic.collation.clone(),
                };
                columns.push(String::new());
                key_columns
                    .push(KeyColumn { collation, descending: matches!(ic.order, Some(SortOrder::Desc)) });
                key_exprs.push(Some(e.clone()));
            }
        }
    }
    debug_assert_eq!(columns.len(), key_columns.len());
    debug_assert_eq!(columns.len(), key_exprs.len());

    // A partial-index WHERE clause may not contain a sub-query (partialindex.html: "The
    // WHERE clause may not contain subqueries, references to other tables, non-deterministic
    // functions, or bound parameters."). Only the sub-query prohibition is enforced here;
    // the other three need name binding / a determinism registry and are a separate binder
    // slice. The parser accepts a sub-query in the predicate; reject it here at create time.
    // Checked AFTER the key columns so an index-expression sub-query surfaces first, matching
    // sqlite binding the ON-list first.
    if let Some(w) = &stmt.where_clause {
        if expr_contains_subquery(w) {
            return Err(Error::Sql(
                "subqueries prohibited in partial index WHERE clauses".into(),
            ));
        }
    }

    Ok(IndexDef {
        name: stmt.name.name.clone(),
        table: table_name.to_string(),
        columns,
        key_columns,
        key_exprs,
        root_page,
        unique: stmt.unique,
        // A partial index carries a WHERE predicate; `partial` is the flag the planner
        // reads (it must not use a partial index as if it covered every row), and
        // `partial_predicate` stores the parsed predicate verbatim so DML index maintenance
        // can gate each row on it (only rows for which it is TRUE are in the index —
        // partialindex.html §2). Storing the same clause twice this way keeps the invariant
        // `partial == partial_predicate.is_some()`. Captured on schema RELOAD too, since
        // that re-parses the stored `CREATE INDEX` SQL through this same function.
        partial: stmt.where_clause.is_some(),
        partial_predicate: stmt.where_clause.clone(),
    })
}

/// The index of the column that is an alias for the rowid, per
/// `lang_createtable.html` §5: a rowid table (NOT `WITHOUT ROWID`) whose primary
/// key is a single column whose declared type is exactly "INTEGER" (ASCII
/// case-insensitive; "INT"/"BIGINT"/etc. do NOT qualify).
///
/// The one quirk: a *column-level* `PRIMARY KEY DESC` is NOT an alias (a retained
/// early-SQLite bug), but a *table-level* `PRIMARY KEY(x DESC)` IS. A table-level
/// PK naming more than one column, or a single expression column, is never an
/// alias.
fn compute_rowid_alias(
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
    without_rowid: bool,
) -> Option<usize> {
    if without_rowid {
        return None;
    }

    // Column-level primary keys: (column index, sort order on the PK clause).
    let mut col_pks: Vec<(usize, Option<SortOrder>)> = Vec::new();
    for (i, c) in columns.iter().enumerate() {
        for cons in &c.constraints {
            if let ColumnConstraintKind::PrimaryKey { order, .. } = &cons.kind {
                col_pks.push((i, *order));
            }
        }
    }

    // Table-level primary keys (each names one or more indexed columns).
    let tbl_pks: Vec<&[IndexedColumn]> = constraints
        .iter()
        .filter_map(|c| match &c.kind {
            TableConstraintKind::PrimaryKey { columns, .. } => Some(columns.as_slice()),
            _ => None,
        })
        .collect();

    // A rowid alias needs exactly one primary key declaration, of a single column.
    match (col_pks.len(), tbl_pks.len()) {
        (1, 0) => {
            let (idx, order) = col_pks[0];
            // The retained quirk: column-level PRIMARY KEY DESC is not an alias.
            if order == Some(SortOrder::Desc) {
                return None;
            }
            integer_alias(columns, idx)
        }
        (0, 1) => {
            let cols = tbl_pks[0];
            if cols.len() != 1 {
                return None; // composite primary key: no alias
            }
            // Only a plain named column can be an integer primary key; a genuine
            // expression PK never aliases the rowid. A per-column `COLLATE` does not change
            // WHICH column the PK names, so resolve it through the SAME `bare_column_name`
            // unwrap the key-column builder uses — real sqlite skips the COLLATE when
            // resolving the PK column, so `PRIMARY KEY(x COLLATE NOCASE)` on an INTEGER
            // column IS the rowid alias. This MUST agree with `auto_indexes_for`, which
            // excludes the same integer PK from auto-index emission via this very function;
            // a `Name`-only match here would leave x a non-alias while an auto-index row was
            // still emitted for it — a self-contradiction. Sort order does NOT disqualify a
            // table-level PK (unlike the column-level DESC quirk above).
            let pk_name = bare_column_name(&cols[0])?;
            let idx = columns.iter().position(|c| c.name.eq_ignore_ascii_case(pk_name))?;
            integer_alias(columns, idx)
        }
        // Zero primary keys, or more than one PK declaration (a malformed schema
        // real SQLite would reject): no alias.
        _ => None,
    }
}

/// `Some(idx)` iff `columns[idx]`'s declared type is exactly "INTEGER".
fn integer_alias(columns: &[SqlColumnDef], idx: usize) -> Option<usize> {
    let is_integer = columns[idx]
        .type_name
        .as_deref()
        .is_some_and(|t| t.eq_ignore_ascii_case("INTEGER"));
    if is_integer { Some(idx) } else { None }
}

/// The index of the single column carrying `PRIMARY KEY AUTOINCREMENT`, or `None`
/// when the table declares no autoincrement column. `AUTOINCREMENT` is representable
/// only on a column-level `PRIMARY KEY` (the grammar has no table-level form and it
/// cannot appear without `PRIMARY KEY`), so scanning the column constraints is
/// exhaustive. This is the ONE spelling of "which column is autoincrement", shared by
/// [`validate_autoincrement`] (the placement rule), [`has_autoincrement`] (the
/// `sqlite_sequence` auto-create trigger), and the `TableDef.autoincrement` flag
/// [`table_def_from_ast`] stores for the INSERT rowid-seeding path, so a future change
/// to how AUTOINCREMENT is modelled in the AST updates all three at a single site and
/// they cannot drift apart.
fn autoincrement_column(columns: &[SqlColumnDef]) -> Option<usize> {
    columns.iter().position(|c| {
        c.constraints
            .iter()
            .any(|k| matches!(k.kind, ColumnConstraintKind::PrimaryKey { autoincrement: true, .. }))
    })
}

/// Enforce SQLite's AUTOINCREMENT placement restrictions (`autoinc.html` §3):
/// AUTOINCREMENT is allowed ONLY on the single-column `INTEGER PRIMARY KEY` that is
/// the rowid alias of a rowid table.
///
/// Two errors, matching real SQLite's wording AND its precedence:
/// - The autoincrement column is not the `INTEGER PRIMARY KEY` alias — a non-`INTEGER`
///   declared type or the retained column-level `PRIMARY KEY DESC` quirk (which
///   `compute_rowid_alias` excludes) — yields `AUTOINCREMENT is only allowed on an INTEGER
///   PRIMARY KEY`. SQLite raises this while adding the primary key, i.e. BEFORE it processes
///   `WITHOUT ROWID`. (A *second* PK that would also defeat the single-PK alias is now
///   rejected earlier by `validate_single_primary_key`, so the multi-PK case never reaches
///   here — this remains correct as defense in depth: `compute_rowid_alias` returns `None`
///   for `> 1` PK regardless.)
/// - Otherwise, a valid alias on a `WITHOUT ROWID` table yields `AUTOINCREMENT not
///   allowed on WITHOUT ROWID tables`. SQLite raises this later, at end-of-table, so
///   it fires only once the column is already a valid alias — hence a `WITHOUT ROWID`
///   table whose autoincrement column is NON-integer reports the first error, not this.
///
/// The rowid-alias rule asked as if this were a rowid table (`without_rowid = false`)
/// is exactly SQLite's "is this the INTEGER PRIMARY KEY" test, so it is reused here
/// rather than re-deriving the INTEGER/DESC/single-column conditions.
fn validate_autoincrement(
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
    without_rowid: bool,
) -> Result<()> {
    let Some(ai_idx) = autoincrement_column(columns) else {
        return Ok(());
    };

    if compute_rowid_alias(columns, constraints, false) != Some(ai_idx) {
        return Err(Error::Sql(
            "AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY".into(),
        ));
    }
    if without_rowid {
        return Err(Error::Sql("AUTOINCREMENT not allowed on WITHOUT ROWID tables".into()));
    }
    Ok(())
}

/// The six datatype names a STRICT table permits (`stricttables.html` §2 rule 2): INT,
/// INTEGER, REAL, TEXT, BLOB, ANY. A STRICT column's declared type must equal one of these
/// EXACTLY (ASCII case-insensitive), matched against the WHOLE declared type text — so
/// `INTEGER(10)`, `VARCHAR(10)`, `DOUBLE`, and `UNSIGNED BIG INT` are all rejected; only
/// these six bare names pass.
const STRICT_DATATYPES: [&str; 6] = ["INT", "INTEGER", "REAL", "TEXT", "BLOB", "ANY"];

/// Reject two columns sharing a name (ASCII case-insensitive), a create-time rule the
/// purely-syntactic parser does not enforce (`lang_createtable.html`). SQLite reports the
/// SECOND occurrence's spelling, so comparing each column against the ones BEFORE it names
/// the later duplicate naturally — `CREATE TABLE t(a, A)` -> `duplicate column name: A`. The
/// wording matches the ADD COLUMN / RENAME COLUMN paths (`schemacatalog.rs`) and the INSERT
/// planner, so the whole engine speaks one duplicate-column message.
fn validate_no_duplicate_columns(columns: &[SqlColumnDef]) -> Result<()> {
    for (i, c) in columns.iter().enumerate() {
        if columns[..i].iter().any(|prev| prev.name.eq_ignore_ascii_case(&c.name)) {
            return Err(Error::Sql(format!("duplicate column name: {}", c.name)));
        }
    }
    Ok(())
}

/// Count the PRIMARY KEY *clauses* a `CREATE TABLE` declares: every column-level `PRIMARY
/// KEY` constraint occurrence PLUS every table-level `PRIMARY KEY(...)` constraint. A single
/// COMPOSITE table PK `PRIMARY KEY(a, b)` is ONE clause (one `TableConstraintKind`). Counting
/// OCCURRENCES — not distinct columns — makes the two PK rules exact and keeps them from
/// drifting: a valid table has exactly one clause, so `validate_single_primary_key` forbids
/// `> 1` and `validate_without_rowid_has_primary_key` forbids `== 0` on a WITHOUT ROWID table,
/// both off this one count.
fn count_primary_keys(columns: &[SqlColumnDef], constraints: &[TableConstraint]) -> usize {
    let column_pks = columns
        .iter()
        .flat_map(|c| &c.constraints)
        .filter(|k| matches!(k.kind, ColumnConstraintKind::PrimaryKey { .. }))
        .count();
    let table_pks = constraints
        .iter()
        .filter(|c| matches!(c.kind, TableConstraintKind::PrimaryKey { .. }))
        .count();
    column_pks + table_pks
}

/// Reject a table declaring more than one PRIMARY KEY (`lang_createtable.html` §3.5: "Each
/// table ... may have at most one PRIMARY KEY. ... An error is raised if more than one
/// PRIMARY KEY clause appears in a CREATE TABLE statement."). A single COMPOSITE table PK
/// `PRIMARY KEY(a, b)` is ONE clause and is accepted; the exotic `a PRIMARY KEY PRIMARY KEY`
/// the parser lands as two column-level clauses is caught. Counts via [`count_primary_keys`].
fn validate_single_primary_key(
    table_name: &str,
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
) -> Result<()> {
    if count_primary_keys(columns, constraints) > 1 {
        return Err(Error::Sql(format!("table \"{table_name}\" has more than one primary key")));
    }
    Ok(())
}

/// Reject a `WITHOUT ROWID` table that declares NO PRIMARY KEY (`withoutrowid.html`: "Every
/// WITHOUT ROWID table must have a PRIMARY KEY. An error is raised if a CREATE TABLE statement
/// with the WITHOUT ROWID clause lacks a PRIMARY KEY."). A WITHOUT ROWID table stores its rows
/// in a b-tree KEYED BY its PRIMARY KEY, so a missing PK leaves no key at all — `create_table`
/// even allocates an INDEX b-tree root for it (schemacatalog.rs). The sibling of
/// [`validate_single_primary_key`]: it forbids `> 1`, this forbids `== 0` on a WITHOUT ROWID
/// table, both off the shared [`count_primary_keys`]. Real sqlite: `PRIMARY KEY missing on
/// table <t>`. Running on the LOAD path too, this fail-closes a stored PK-less WITHOUT ROWID
/// row (mapped to `Error::Format` by `load_table_row`) — correct, since real sqlite never
/// persists one.
fn validate_without_rowid_has_primary_key(
    table_name: &str,
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
    without_rowid: bool,
) -> Result<()> {
    if without_rowid && count_primary_keys(columns, constraints) == 0 {
        return Err(Error::Sql(format!("PRIMARY KEY missing on table {table_name}")));
    }
    Ok(())
}

/// True iff `col` is a GENERATED column — its constraint list carries a
/// [`ColumnConstraintKind::Generated`]. The one spelling of "is this a generated column",
/// so the three [`validate_generated_columns`] rules classify identically.
fn is_generated_column(col: &SqlColumnDef) -> bool {
    col.constraints.iter().any(|k| matches!(k.kind, ColumnConstraintKind::Generated { .. }))
}

/// Enforce the create-time GENERATED-column restrictions (`gencol.html` §2.3), which the
/// purely-syntactic parser accepts but real sqlite rejects at create time:
///
/// - a generated column may not carry a `DEFAULT` — the value is always the `AS (expr)`
///   result (§2.3.1): `cannot use DEFAULT on a generated column`;
/// - a generated column may not be part of the PRIMARY KEY (§2.3.2), whether by a
///   column-level `PRIMARY KEY` on the generated column itself OR by its name appearing in a
///   table-level `PRIMARY KEY(...)` list (resolved through the shared [`bare_column_name`]
///   unwrap, ASCII case-insensitively): `generated columns cannot be part of the primary key`;
/// - every table must have at least one NON-generated column (§2.3.6):
///   `must have at least one non-generated column`.
///
/// The EXPRESSION-level restrictions (no subquery / aggregate / window, no self-reference or
/// circular dependency, no direct ROWID reference — §2.3.3–2.3.5) need binding and a
/// dependency graph and belong to the binder/planner, so they are deliberately NOT checked
/// here. Only the structural rules a `ColumnDef`/`TableConstraint` walk can decide are.
fn validate_generated_columns(
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
) -> Result<()> {
    for col in columns.iter().filter(|c| is_generated_column(c)) {
        // §2.3.1 — a generated column may not also declare a DEFAULT (either order, so match
        // on the whole constraint list).
        if col.constraints.iter().any(|k| matches!(k.kind, ColumnConstraintKind::Default(_))) {
            return Err(Error::Sql("cannot use DEFAULT on a generated column".into()));
        }
        // §2.3.2 — nor be part of the PRIMARY KEY: a column-level `PRIMARY KEY` on it, or its
        // name in any table-level `PRIMARY KEY(...)` clause. Table-level targets resolve
        // through `bare_column_name` (so a `COLLATE`-over-column term still names the column);
        // a genuine expression term names none and cannot reference this column.
        let column_level_pk = col
            .constraints
            .iter()
            .any(|k| matches!(k.kind, ColumnConstraintKind::PrimaryKey { .. }));
        let table_level_pk = constraints.iter().any(|cons| match &cons.kind {
            TableConstraintKind::PrimaryKey { columns: pk_cols, .. } => pk_cols
                .iter()
                .filter_map(bare_column_name)
                .any(|name| name.eq_ignore_ascii_case(&col.name)),
            // Only a PRIMARY KEY clause can place a column in the primary key; UNIQUE / CHECK
            // / FOREIGN KEY cannot. Listed explicitly (no wildcard), like the sibling
            // `validate_table_constraint_columns`, so a newly-added `TableConstraintKind` is a
            // compile error here and forces a decision rather than defaulting to "not a PK".
            TableConstraintKind::Unique { .. }
            | TableConstraintKind::Check(_)
            | TableConstraintKind::ForeignKey { .. } => false,
        });
        if column_level_pk || table_level_pk {
            return Err(Error::Sql("generated columns cannot be part of the primary key".into()));
        }
    }

    // §2.3.6 — at least one non-generated column. Guarded on non-empty because an empty
    // column list (a parse error real sqlite never reaches here) would satisfy `all`
    // vacuously and emit this misleadingly; a table with columns must have a plain one.
    if !columns.is_empty() && columns.iter().all(is_generated_column) {
        return Err(Error::Sql("must have at least one non-generated column".into()));
    }
    Ok(())
}

/// Reject a table-level `PRIMARY KEY(...)` / `UNIQUE(...)` that names a column the table
/// does not declare (`lang_createtable.html`), matching real sqlite's `table <t> has no
/// column named <c>`. This also HARDENS `auto_indexes_for`, which builds auto-indexes from
/// these same constraints and would otherwise reference a phantom column. Only a BARE COLUMN
/// target is existence-checked — through the SHARED [`bare_column_name`] unwrap, so a
/// `COLLATE`-over-column term resolves to the column it names — because a GENUINE expression
/// target is a separate deferred gap (`auto_indexes_for` already treats it as one), not a
/// missing column, so it is skipped here. The existence check is ASCII case-insensitive:
/// `PRIMARY KEY(A)` names a column declared `a`.
fn validate_table_constraint_columns(
    table_name: &str,
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
) -> Result<()> {
    let has_column = |name: &str| columns.iter().any(|c| c.name.eq_ignore_ascii_case(name));
    for cons in constraints {
        let indexed = match &cons.kind {
            TableConstraintKind::PrimaryKey { columns: cols, .. } => cols,
            TableConstraintKind::Unique { columns: cols, .. } => cols,
            // A CHECK names columns inside an expression bound later; a FOREIGN KEY's child
            // columns are existence-checked by `validate_foreign_key_child_columns` (which
            // carries the FK-specific message), not here.
            TableConstraintKind::Check(_) | TableConstraintKind::ForeignKey { .. } => continue,
        };
        for ic in indexed {
            if let Some(name) = bare_column_name(ic) {
                if !has_column(name) {
                    return Err(Error::Sql(format!(
                        "table {table_name} has no column named {name}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Reject a table-level `FOREIGN KEY(<child_cols>) REFERENCES ...` whose child-column list
/// names a column the table itself does not declare, matching real sqlite's
/// `unknown column "<c>" in foreign key definition`. Whether a child column exists is
/// resolvable from the SINGLE table being created, so — exactly like the sibling table-level
/// PK/UNIQUE unknown-column check in [`validate_table_constraint_columns`] — this is a
/// create-time (DDL) error. The FK errors sqlite instead defers to statement-prepare time are
/// the ones that need MORE than this one table's definition (the parent table absent, the
/// parent key columns absent, the parent key not unique, a parent-PK arity mismatch); those
/// are "DML errors" per `spec/sqlite-doc/foreignkeys.html` §3 (~lines 446-471). Child-column
/// existence is NOT in that carve-out, so this check looks ONLY at the child names against the
/// table's own `columns` and NEVER at the parent — `FOREIGN KEY(a) REFERENCES missing(x)` with
/// a real child column `a` still builds. A column-level `REFERENCES` needs no check either: its
/// child column IS the column being defined, so it always exists. Existence is ASCII
/// case-insensitive (`FOREIGN KEY(A)` names a column declared `a`), and the FIRST missing child
/// column in declaration order is reported, quoting the name as written.
fn validate_foreign_key_child_columns(
    columns: &[SqlColumnDef],
    constraints: &[TableConstraint],
) -> Result<()> {
    let has_column = |name: &str| columns.iter().any(|c| c.name.eq_ignore_ascii_case(name));
    for cons in constraints {
        let child_cols = match &cons.kind {
            TableConstraintKind::ForeignKey { columns: child_cols, .. } => child_cols,
            // PRIMARY KEY / UNIQUE bare columns are existence-checked by the sibling
            // `validate_table_constraint_columns`; a CHECK names columns inside an expression
            // bound later. Listed explicitly (no wildcard), like that sibling, so a
            // newly-added `TableConstraintKind` is a compile error here and forces a decision
            // on whether it too carries child columns to existence-check, rather than being
            // silently skipped.
            TableConstraintKind::PrimaryKey { .. }
            | TableConstraintKind::Unique { .. }
            | TableConstraintKind::Check(_) => continue,
        };
        for child in child_cols {
            if !has_column(child) {
                return Err(Error::Sql(format!(
                    "unknown column \"{child}\" in foreign key definition"
                )));
            }
        }
    }
    Ok(())
}

/// Enforce the STRICT-table datatype rules (`stricttables.html` §2 rules 1 & 2). When the
/// table carries the trailing `STRICT` option, EVERY column — including a generated column —
/// must declare a datatype (`missing datatype for <t>.<c>`), and that datatype must be
/// exactly one of the six [`STRICT_DATATYPES`], compared ASCII case-insensitively against
/// the WHOLE declared type text (`unknown datatype for <t>.<c>: "<TYPE>"`). This is the
/// create-time half only; the flag is recorded as [`TableDef::strict`] and the runtime half
/// (rejecting a value of the wrong type on INSERT/UPDATE) is the executor's
/// `enforce_strict_types`. A non-STRICT table is unrestricted —
/// any type, or none, is legal — so this is a no-op unless `strict`.
fn validate_strict_datatypes(
    table_name: &str,
    columns: &[SqlColumnDef],
    strict: bool,
) -> Result<()> {
    if !strict {
        return Ok(());
    }
    for c in columns {
        let Some(declared) = c.type_name.as_deref() else {
            return Err(Error::Sql(format!("missing datatype for {table_name}.{}", c.name)));
        };
        let declared = declared.trim();
        if !STRICT_DATATYPES.iter().any(|allowed| allowed.eq_ignore_ascii_case(declared)) {
            return Err(Error::Sql(format!(
                "unknown datatype for {table_name}.{}: \"{declared}\"",
                c.name
            )));
        }
    }
    Ok(())
}

/// Whether a `DEFAULT (<expr>)` expression is CONSTANT in the sense
/// `lang_createtable.html` §3.2 requires: it contains no sub-query, no column or table
/// reference, and no bound parameter. (The spec text also lists a double-quoted string
/// literal as making an expression non-constant, but this engine mirrors real sqlite's
/// DQS-enabled default build, where a bare double-quoted
/// token is a TEXT literal, i.e. `Expr::Column { from_dqs: true }`, and stays constant;
/// only a GENUINE reference, `from_dqs: false`, is rejected. See `render_default_text`.)
///
/// The walk is ITERATIVE over an explicit worklist, never recursive. The parser folds
/// left-associative operators in a loop, so a `DEFAULT (1+1+1+…)` chain can be taller than
/// the recursive-descent depth guard, and a recursive checker would overflow the native
/// stack (a SIGSEGV/SIGABRT — the worst outcome, not a recoverable `Error`) — the same
/// hazard that makes `Expr`'s own `Drop` iterative (`ast_expr.rs`). A subquery / EXISTS /
/// `IN (SELECT …)` / `IN table` node is non-constant on sight, so its `Select` body is
/// never descended into (its mere presence disqualifies the default).
///
/// The `match` is exhaustive with no wildcard so that adding an `Expr` variant is a compile
/// error here, forcing a decision on whether it is a constant leaf, a recursive node, or a
/// new non-constant form — the child enumeration then mirrors `take_expr_children`.
fn default_expr_is_constant(root: &Expr) -> bool {
    let mut worklist: Vec<&Expr> = vec![root];
    while let Some(e) = worklist.pop() {
        match e {
            // --- non-constant on sight: a §3.2 violation, reject immediately ---
            // A genuine (non-DQS) column or table reference.
            Expr::Column { from_dqs: false, .. } => return false,
            // A bound parameter (`?`, `?NNN`, `:name`, `@name`, `$name`).
            Expr::BindParam(_) => return false,
            // Any sub-query: a scalar `(SELECT …)` or `[NOT] EXISTS (…)`.
            Expr::Subquery(_) | Expr::Exists { .. } => return false,
            // `IN`/`NOT IN` whose right side is a SELECT or a table / table-valued function.
            // (An `IN (value-list)` is handled with the recursive nodes below.)
            Expr::In { rhs: InRhs::Select(_) | InRhs::Table { .. }, .. } => return false,

            // --- constant leaves: nothing to reject, no child to scan ---
            // A literal; a DQS token (a TEXT literal, not a reference); or `RAISE(…)` (carries
            // none of the forbidden nodes — nonsensical in a default, but constant).
            Expr::Literal(_) | Expr::Column { from_dqs: true, .. } | Expr::Raise(_) => {}

            // --- recursive nodes: constant iff every child Expr is; push children ---
            // (`&Box<Expr>` coerces to `&Expr` on push.)
            Expr::Unary { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::Collate { expr, .. }
            | Expr::IsNull(expr)
            | Expr::NotNull(expr) => worklist.push(expr),
            Expr::Binary { left, right, .. } => {
                worklist.push(left);
                worklist.push(right);
            }
            Expr::Like { lhs, rhs, escape, .. } => {
                worklist.push(lhs);
                worklist.push(rhs);
                if let Some(esc) = escape {
                    worklist.push(esc);
                }
            }
            Expr::Between { expr, low, high, .. } => {
                worklist.push(expr);
                worklist.push(low);
                worklist.push(high);
            }
            // The SELECT / table forms are rejected above; an `IN (value-list)` is constant
            // iff its scrutinee and every list element are.
            Expr::In { expr, rhs: InRhs::List(list), .. } => {
                worklist.push(expr);
                worklist.extend(list.iter());
            }
            // A function call over constants is constant (`abs(-1)`, `1 || 2`). Its child
            // Exprs are the argument list, the FILTER predicate, the aggregate ORDER BY keys,
            // AND the window `OVER` spec (scanned via `push_over_children`). The OVER spec is
            // where this DIVERGES from `take_expr_children`, which omits it: a teardown only
            // needs to reach every unbounded-depth child, but the constant check must reach
            // every child that could hide a forbidden node — e.g. the column ref in
            // `f(1) OVER (ORDER BY a)`. Destructured with NO `..` so a future Expr-bearing
            // field on `Function` (as `over` already was) is a compile error here, not a
            // silently-unscanned blind spot.
            Expr::Function { name: _, distinct: _, args, filter, over, order_by } => {
                if let FunctionArgs::List(list) = args {
                    worklist.extend(list.iter());
                }
                if let Some(f) = filter {
                    worklist.push(f);
                }
                worklist.extend(order_by.iter().map(|term| &term.expr));
                push_over_children(over, &mut worklist);
            }
            Expr::Case { operand, whens, else_expr } => {
                if let Some(op) = operand {
                    worklist.push(op);
                }
                for (when, then) in whens {
                    worklist.push(when);
                    worklist.push(then);
                }
                if let Some(els) = else_expr {
                    worklist.push(els);
                }
            }
            Expr::Parenthesized(list) => worklist.extend(list.iter()),
        }
    }
    true
}

/// Push every child `Expr` reachable through a window `OVER` clause onto `worklist` — a
/// window spec's `PARTITION BY` and `ORDER BY` keys and its frame-bound offsets. A
/// `DEFAULT (<expr>)` is non-constant if a forbidden node (a column/table reference, a bound
/// parameter, or a sub-query) hides inside one of these — e.g. `f(1) OVER (ORDER BY a)` — so
/// [`default_expr_is_constant`] scans them like any other child. (`take_expr_children` omits
/// the OVER spec because a teardown only needs to reach every unbounded-depth child; the
/// constant check needs every child that could hold a forbidden node, so it cannot share that
/// blind spot.) NOTE the boundary: a window function whose spec is itself all-constant —
/// `f(1) OVER (ORDER BY 1)` — has no forbidden node and is "constant" by this §3.2 rule, so it
/// is accepted here; real sqlite rejects any window function in a default through a SEPARATE
/// window-misuse rule, which is a follow-up outside this constant check. Destructured with no
/// `..` so a new window-spec / frame field is a compile error rather than an unscanned gap.
fn push_over_children<'a>(over: &'a Option<OverClause>, worklist: &mut Vec<&'a Expr>) {
    let spec = match over {
        Some(OverClause::Spec(spec)) => spec,
        // `OVER window-name` refers to a window defined elsewhere and carries no expr here.
        Some(OverClause::WindowName(_)) | None => return,
    };
    let WindowSpec { base: _, partition_by, order_by, frame } = spec;
    worklist.extend(partition_by.iter());
    worklist.extend(order_by.iter().map(|term| &term.expr));
    if let Some(WindowFrame { units: _, start, end, exclude: _ }) = frame {
        push_frame_bound(start, worklist);
        if let Some(end) = end {
            push_frame_bound(end, worklist);
        }
    }
}

/// Push a window frame bound's offset `Expr` (`<expr> PRECEDING` / `<expr> FOLLOWING`) onto
/// `worklist`; the `UNBOUNDED …` / `CURRENT ROW` bounds carry none. Exhaustive with no
/// wildcard so a new [`FrameBound`] shape is a compile error here.
fn push_frame_bound<'a>(bound: &'a FrameBound, worklist: &mut Vec<&'a Expr>) {
    match bound {
        FrameBound::Preceding(e) | FrameBound::Following(e) => worklist.push(e),
        FrameBound::UnboundedPreceding
        | FrameBound::CurrentRow
        | FrameBound::UnboundedFollowing => {}
    }
}

/// True iff `root` contains a sub-query node ANYWHERE within it: a scalar
/// `(SELECT …)`, `[NOT] EXISTS (…)`, or `IN (SELECT …)` / `IN table` (a table or
/// table-valued-function reference — a sub-query in disguise, `SELECT … FROM table`).
///
/// This is the create-time test for the three schema-expression positions SQLite
/// forbids a sub-query in: a CHECK constraint (`lang_createtable.html`: "The
/// expression of a CHECK constraint may not contain a subquery."), an index key
/// expression (`expridx.html` / `lang_createindex.html`: "Expressions in CREATE INDEX
/// statements may not use subqueries."), and a partial-index WHERE clause
/// (`partialindex.html`: "The WHERE clause may not contain subqueries …"). Column
/// references (`from_dqs` either way) and bound parameters are NOT sub-queries and are
/// walked THROUGH, not rejected — these positions legitimately reference the table's
/// columns, so only a genuine sub-query node disqualifies. (The partial-WHERE spec
/// ALSO forbids other-table references, non-deterministic functions, and bound
/// parameters, but detecting those needs name binding / a determinism registry and is
/// a separate binder slice — see the callers; this focused check is sub-queries only.)
///
/// The walk is ITERATIVE over an explicit worklist, never recursive: the parser folds
/// left-associative operators in a loop, so a `CHECK(a + a + … + (SELECT 1))` chain can
/// be taller than the recursive-descent depth guard and a recursive checker would
/// overflow the native stack (a SIGSEGV/SIGABRT — the worst outcome, not a recoverable
/// `Error`) — the same hazard that makes `Expr`'s own `Drop` iterative (`ast_expr.rs`).
/// A sub-query / EXISTS / `IN (SELECT …)` / `IN table` node is a hit ON SIGHT, so its
/// `Select` body is never descended into (its mere presence is the answer).
///
/// The `match` is exhaustive with no wildcard so adding an `Expr` variant is a compile
/// error here, forcing a decision on whether it is a leaf, a recursive node, or a new
/// sub-query form; the child enumeration mirrors [`default_expr_is_constant`], INCLUDING
/// the window `OVER` spec (via the shared [`push_over_children`]) so a sub-query hiding
/// in `f(x) OVER (ORDER BY (SELECT 1))` is still found.
fn expr_contains_subquery(root: &Expr) -> bool {
    let mut worklist: Vec<&Expr> = vec![root];
    while let Some(e) = worklist.pop() {
        match e {
            // --- a sub-query ON SIGHT: the answer; do not descend into its Select body ---
            // A scalar `(SELECT …)` or `[NOT] EXISTS (…)`.
            Expr::Subquery(_) | Expr::Exists { .. } => return true,
            // `IN`/`NOT IN` whose right side is a SELECT or a table / table-valued function.
            // (An `IN (value-list)` is handled with the recursive nodes below.)
            Expr::In { rhs: InRhs::Select(_) | InRhs::Table { .. }, .. } => return true,

            // --- leaves: no child Expr, nothing that can hide a sub-query ---
            // A literal; a column reference (either a DQS text token or a genuine ref — both
            // allowed in these positions); a bound parameter; or `RAISE(…)`.
            Expr::Literal(_)
            | Expr::Column { .. }
            | Expr::BindParam(_)
            | Expr::Raise(_) => {}

            // --- recursive nodes: a sub-query iff some child holds one; push children ---
            // (`&Box<Expr>` coerces to `&Expr` on push.)
            Expr::Unary { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::Collate { expr, .. }
            | Expr::IsNull(expr)
            | Expr::NotNull(expr) => worklist.push(expr),
            Expr::Binary { left, right, .. } => {
                worklist.push(left);
                worklist.push(right);
            }
            Expr::Like { lhs, rhs, escape, .. } => {
                worklist.push(lhs);
                worklist.push(rhs);
                if let Some(esc) = escape {
                    worklist.push(esc);
                }
            }
            Expr::Between { expr, low, high, .. } => {
                worklist.push(expr);
                worklist.push(low);
                worklist.push(high);
            }
            // The SELECT / table forms are caught above; an `IN (value-list)` holds a
            // sub-query iff its scrutinee or some list element does.
            Expr::In { expr, rhs: InRhs::List(list), .. } => {
                worklist.push(expr);
                worklist.extend(list.iter());
            }
            // A function call's child Exprs are its argument list, FILTER predicate,
            // aggregate ORDER BY keys, AND the window `OVER` spec (scanned via the shared
            // [`push_over_children`], as `default_expr_is_constant` does — a teardown-style
            // enumeration omits the OVER spec, but a sub-query can hide there too).
            // Destructured with NO `..` so a future Expr-bearing field on `Function` is a
            // compile error here, not a silently-unscanned blind spot.
            Expr::Function { name: _, distinct: _, args, filter, over, order_by } => {
                if let FunctionArgs::List(list) = args {
                    worklist.extend(list.iter());
                }
                if let Some(f) = filter {
                    worklist.push(f);
                }
                worklist.extend(order_by.iter().map(|term| &term.expr));
                push_over_children(over, &mut worklist);
            }
            Expr::Case { operand, whens, else_expr } => {
                if let Some(op) = operand {
                    worklist.push(op);
                }
                for (when, then) in whens {
                    worklist.push(when);
                    worklist.push(then);
                }
                if let Some(els) = else_expr {
                    worklist.push(els);
                }
            }
            Expr::Parenthesized(list) => worklist.extend(list.iter()),
        }
    }
    false
}

/// Reject a column `DEFAULT (<expr>)` whose expression is NOT constant
/// (`lang_createtable.html` §3.2), matching real sqlite's create-time
/// `default value of column [<col>] is not constant`. The purely-syntactic parser sends
/// every parenthesized `DEFAULT (...)` through the full expression grammar, so a
/// column/table reference, a bound parameter, or a sub-query reaches the builder as a
/// [`DefaultValue::Expr`]; this is where that is caught and failed closed. A
/// [`DefaultValue::Literal`] default (an unparenthesized literal / signed number /
/// `CURRENT_*`) is constant by construction and never checked. Constant-ness is decided by
/// [`default_expr_is_constant`]; the square-bracketed column name matches sqlite's wording.
fn validate_constant_defaults(columns: &[SqlColumnDef]) -> Result<()> {
    for c in columns {
        for cons in &c.constraints {
            if let ColumnConstraintKind::Default(DefaultValue::Expr(e, _)) = &cons.kind {
                if !default_expr_is_constant(e) {
                    return Err(Error::Sql(format!(
                        "default value of column [{}] is not constant",
                        c.name
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Reject a `CHECK(<expr>)` constraint whose expression contains a sub-query
/// (`lang_createtable.html`: "The expression of a CHECK constraint may not contain a
/// subquery."). The purely-syntactic parser routes a CHECK predicate through the general
/// expression grammar with no post-parse rejection, so a sub-query reaches the builder in
/// `checks`; this is the create-time fail-closed check matching real sqlite (which errors
/// at create time — this engine wrongly accepted it before). Operates on the
/// already-unified `checks` vec, so ONE pass covers both column-level and table-level
/// CHECK constraints. Only a genuine sub-query is rejected — a CHECK legitimately
/// references the table's columns, so [`expr_contains_subquery`] walks those through.
fn validate_check_constraints(checks: &[Expr]) -> Result<()> {
    for e in checks {
        if expr_contains_subquery(e) {
            return Err(Error::Sql("subqueries prohibited in CHECK constraints".into()));
        }
    }
    Ok(())
}

/// True iff the `CREATE TABLE` declares a column-level `PRIMARY KEY AUTOINCREMENT`.
/// This is the trigger for auto-creating `sqlite_sequence` (`autoinc.html` §3): once
/// [`table_def_from_ast`] has ACCEPTED the statement, a `true` result implies a
/// *valid* `INTEGER PRIMARY KEY AUTOINCREMENT`, because [`validate_autoincrement`] has
/// already rejected every other placement.
pub(crate) fn has_autoincrement(stmt: &CreateTable) -> bool {
    matches!(&stmt.body, CreateTableBody::Columns { columns, .. } if autoincrement_column(columns).is_some())
}

/// Raw-SQL text for a column `DEFAULT`. A literal is rendered exactly (this is what a
/// reload needs, and what the INSERT planner re-binds when a row omits the column). A
/// parenthesized bare double-quoted `DEFAULT ("lit")` is a DQS text literal (see the
/// `DefaultValue::Expr` arm below); any *other* `DEFAULT (expr)` uses the expression's
/// source text as the parser kept it, like sqlite3's `dflt_value`. The un-parenthesized `DEFAULT "lit"` never
/// reaches here as an expression: `parse_default_value` already folds it to a
/// `DefaultValue::Literal(Text)` in the parser.
fn render_default_text(d: &DefaultValue) -> Option<String> {
    match d {
        DefaultValue::Literal(lit) => Some(render_literal(lit)),
        // DQS legacy (quirks.html §8): a parenthesized bare double-quoted default
        // `DEFAULT ("lit")` names no column (a column default has no row to read), so it
        // is the text literal 'lit'. Render it single-quoted so a schema reload / the
        // INSERT planner re-parses it as exactly that string literal. Any other
        // expression default keeps its source text, which is what sqlite3 stores and
        // reports, and which the INSERT planner re-parses as the expression.
        DefaultValue::Expr(e, text) => match e.as_ref() {
            // Reuse `render_literal` so the single-quote text-escaping rule lives in ONE
            // place (a bare DQS default is exactly the text literal of its spelling).
            Expr::Column { schema: None, table: None, name, from_dqs: true } => {
                Some(render_literal(&Literal::Text(name.clone())))
            }
            _ => Some(text.clone()),
        },
    }
}

/// Fold a column `DEFAULT` to the constant [`Value`] a read decodes when a stored row
/// predates the column (a "short" record from `ADD COLUMN`). `Some` only for a literal
/// constant — the sign is already folded into the literal by the parser, and `TRUE` /
/// `FALSE` are SQLite's `1` / `0`. Returns `None` for a `DEFAULT (expr)` and for the
/// time-dependent `CURRENT_*` forms, which have no build-time constant value (a read
/// then correctly falls back to NULL); such non-constant forms are also rejected by
/// `ADD COLUMN`, so no short row ever carries one.
fn eval_constant_default(d: &DefaultValue) -> Option<Value> {
    let lit = match d {
        DefaultValue::Literal(lit) => lit,
        // DQS legacy: a parenthesized bare double-quoted default `DEFAULT ("lit")` folds
        // to the constant text 'lit' (mirrors render_default_text). Every other expression
        // default has no build-time constant here.
        DefaultValue::Expr(e, _) => {
            return match e.as_ref() {
                Expr::Column { schema: None, table: None, name, from_dqs: true } => {
                    Some(Value::Text(name.clone()))
                }
                _ => None,
            };
        }
    };
    match lit {
        Literal::Null => Some(Value::Null),
        Literal::Integer(i) => Some(Value::Integer(*i)),
        Literal::Real(f) => Some(Value::Real(*f)),
        Literal::Text(s) => Some(Value::Text(s.clone())),
        Literal::Blob(b) => Some(Value::Blob(b.clone())),
        Literal::True => Some(Value::Integer(1)),
        Literal::False => Some(Value::Integer(0)),
        Literal::CurrentTime | Literal::CurrentDate | Literal::CurrentTimestamp => None,
    }
}

/// Render a literal to its SQL text form.
fn render_literal(lit: &Literal) -> String {
    match lit {
        Literal::Null => "NULL".to_string(),
        Literal::Integer(i) => i.to_string(),
        Literal::Real(f) => f.to_string(),
        // Single-quote string literal with SQL quote-doubling.
        Literal::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Literal::Blob(b) => {
            let mut out = String::with_capacity(3 + b.len() * 2);
            out.push_str("X'");
            for byte in b {
                out.push_str(&format!("{byte:02X}"));
            }
            out.push('\'');
            out
        }
        Literal::CurrentTime => "CURRENT_TIME".to_string(),
        Literal::CurrentDate => "CURRENT_DATE".to_string(),
        Literal::CurrentTimestamp => "CURRENT_TIMESTAMP".to_string(),
        Literal::True => "TRUE".to_string(),
        Literal::False => "FALSE".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_sql::{parse, BinaryOp, Statement};
    use minisqlite_types::Affinity;

    /// Parse a single `CREATE TABLE` and build its def with a fixed root page.
    fn tdef(sql: &str) -> TableDef {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        let stmt = match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => ct,
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        };
        table_def_from_ast(stmt, 2).unwrap()
    }

    #[test]
    fn rowid_alias_exact_cases() {
        // The eight cases, straight from spec §5.
        assert_eq!(tdef("CREATE TABLE t(x INTEGER PRIMARY KEY)").rowid_alias, Some(0));
        assert_eq!(tdef("CREATE TABLE t(x INTEGER PRIMARY KEY ASC)").rowid_alias, Some(0));
        assert_eq!(tdef("CREATE TABLE t(x INTEGER, y, PRIMARY KEY(x))").rowid_alias, Some(0));
        assert_eq!(tdef("CREATE TABLE t(x INTEGER, y, PRIMARY KEY(x DESC))").rowid_alias, Some(0));
        assert_eq!(tdef("CREATE TABLE t(x INTEGER PRIMARY KEY DESC)").rowid_alias, None);
        assert_eq!(tdef("CREATE TABLE t(x INT PRIMARY KEY)").rowid_alias, None);
        assert_eq!(tdef("CREATE TABLE t(x INTEGER PRIMARY KEY) WITHOUT ROWID").rowid_alias, None);
        assert_eq!(
            tdef("CREATE TABLE t(a INTEGER, b INTEGER, PRIMARY KEY(a,b))").rowid_alias,
            None
        );
    }

    #[test]
    fn rowid_alias_picks_the_right_column_index() {
        // The alias index tracks the PK column's position, not always 0.
        assert_eq!(tdef("CREATE TABLE t(a, b INTEGER PRIMARY KEY)").rowid_alias, Some(1));
        // No primary key at all: no alias.
        assert_eq!(tdef("CREATE TABLE t(a INTEGER, b INTEGER)").rowid_alias, None);
    }

    #[test]
    fn integer_type_match_is_case_insensitive_but_exact() {
        assert_eq!(tdef("CREATE TABLE t(x integer PRIMARY KEY)").rowid_alias, Some(0));
        assert_eq!(tdef("CREATE TABLE t(x InTeGeR PRIMARY KEY)").rowid_alias, Some(0));
        // Exactly INTEGER only: other integer-ish names are ordinary columns.
        assert_eq!(tdef("CREATE TABLE t(x BIGINT PRIMARY KEY)").rowid_alias, None);
    }

    #[test]
    fn column_flags_and_defaults_are_captured() {
        let t = tdef(
            "CREATE TABLE t(a INTEGER NOT NULL, b TEXT UNIQUE DEFAULT 'x', c COLLATE NOCASE, d)",
        );
        assert!(t.columns[0].not_null);
        assert_eq!(t.columns[0].declared_type.as_deref(), Some("INTEGER"));
        assert!(t.columns[1].unique);
        assert_eq!(t.columns[1].default.as_deref(), Some("'x'"));
        assert_eq!(t.columns[2].collation.as_deref(), Some("NOCASE"));
        assert_eq!(t.columns[2].declared_type, None);
        assert_eq!(t.columns[3].declared_type, None);
        assert!(!t.columns[3].not_null);
    }

    #[test]
    fn column_level_primary_key_sets_the_flag() {
        let t = tdef("CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        assert!(t.columns[0].primary_key);
        assert!(!t.columns[1].primary_key);
    }

    #[test]
    fn primary_key_column_indices_cover_all_forms() {
        // No PRIMARY KEY at all (a plain rowid table): empty.
        assert!(tdef("CREATE TABLE t(a, b)").primary_key.is_empty());
        // Column-level PRIMARY KEY: the single column's index, not always 0.
        assert_eq!(tdef("CREATE TABLE t(a, b PRIMARY KEY)").primary_key, vec![1]);
        // INTEGER PRIMARY KEY (the rowid alias) is recorded as the PK too.
        let alias = tdef("CREATE TABLE t(id INTEGER PRIMARY KEY, x)");
        assert_eq!(alias.primary_key, vec![0]);
        assert_eq!(alias.rowid_alias, Some(0));
        // INTEGER PRIMARY KEY DESC is the PK (position 1) even though it is NOT the rowid
        // alias (the retained early-SQLite quirk).
        let desc = tdef("CREATE TABLE t(id INTEGER PRIMARY KEY DESC, x)");
        assert_eq!(desc.primary_key, vec![0]);
        assert_eq!(desc.rowid_alias, None);
        // Table-level composite PK, in declaration order.
        assert_eq!(tdef("CREATE TABLE t(a, b, PRIMARY KEY(a, b))").primary_key, vec![0, 1]);
        // DECLARATION order is the PK's, not the columns': PRIMARY KEY(b, a) -> [1, 0].
        assert_eq!(tdef("CREATE TABLE t(a, b, PRIMARY KEY(b, a))").primary_key, vec![1, 0]);
        // A single-column table-level PK.
        assert_eq!(tdef("CREATE TABLE t(x INTEGER, y, PRIMARY KEY(x))").primary_key, vec![0]);
        // Composite WITHOUT ROWID (its PK is the table b-tree, still recorded here).
        assert_eq!(
            tdef("CREATE TABLE t(a, b, PRIMARY KEY(a, b)) WITHOUT ROWID").primary_key,
            vec![0, 1]
        );
        // A COLLATE-wrapped PK term still resolves to the column it names.
        assert_eq!(
            tdef("CREATE TABLE t(a, b, PRIMARY KEY(b COLLATE NOCASE, a))").primary_key,
            vec![1, 0]
        );
        // A GENUINE expression term (`b + c`) in a table-level PK is a deferred gap real
        // sqlite rejects. `named_columns` (shared with `auto_indexes_for`) is all-or-nothing,
        // so the WHOLE PK is unmodelled: `primary_key` is EMPTY *and* no PK auto-index is
        // emitted — the two surfaces agree rather than drifting to a partial [0] key that
        // `index_list` would never back. (Pins the "cannot describe a different PK than
        // auto_indexes_for" invariant against a filter_map-style partial skip.)
        let expr_pk = tdef("CREATE TABLE t(a, b, c, PRIMARY KEY(a, b + c))");
        assert!(expr_pk.primary_key.is_empty(), "expression PK term -> no modelled PK");
        assert!(
            expr_pk.auto_indexes.is_empty(),
            "expression PK emits no auto-index, so primary_key must be empty too (congruent)"
        );
    }

    #[test]
    fn dqs_default_is_a_text_literal() {
        // SQLite DQS legacy (quirks.html §8): a bare double-quoted DEFAULT token can never
        // reference a column, so it is the TEXT literal of its spelling — for both the
        // unparenthesized form (parsed as DefaultValue::Literal) and the parenthesized form
        // `DEFAULT ("lit")` (parsed as DefaultValue::Expr(Column{from_dqs:true})). Both must
        // store the same single-quoted raw text (so a reload / the INSERT planner re-parses
        // it as that string) and fold to the same constant Value.
        for sql in [
            "CREATE TABLE t(id INT, x TEXT DEFAULT \"lit\")",
            "CREATE TABLE t(id INT, x TEXT DEFAULT (\"lit\"))",
        ] {
            let t = tdef(sql);
            assert_eq!(t.columns[1].default.as_deref(), Some("'lit'"), "raw text for {sql:?}");
            match &t.columns[1].default_value {
                Some(Value::Text(s)) => assert_eq!(s, "lit", "constant fold for {sql:?}"),
                other => panic!("expected Text(\"lit\") for {sql:?}, got {other:?}"),
            }
        }
        // A single quote inside the double-quoted token is SQL-doubled when rendered back
        // to the single-quoted raw form, and the folded Value keeps the raw character.
        let t = tdef("CREATE TABLE t(x TEXT DEFAULT \"a'b\")");
        assert_eq!(t.columns[0].default.as_deref(), Some("'a''b'"));
        match &t.columns[0].default_value {
            Some(Value::Text(s)) => assert_eq!(s, "a'b"),
            other => panic!("expected Text(\"a'b\"), got {other:?}"),
        }
    }

    #[test]
    fn check_constraints_collect_column_and_table_level_in_order() {
        // A column-level CHECK and a table-level CHECK are both collected onto `checks`
        // (lang_createtable.html §3.7 — a column check is just a check that names that
        // column), so this two-check table yields exactly two predicates.
        assert_eq!(tdef("CREATE TABLE t(x INT CHECK(x>0), CHECK(x<100))").checks.len(), 2);

        // A table with no CHECK constraint carries an empty `checks` vec.
        assert!(tdef("CREATE TABLE t(x INT, y TEXT)").checks.is_empty());

        // Multiple column checks plus multiple table checks all accumulate (order is
        // deterministic: column checks as walked, then the table checks).
        let t = tdef(
            "CREATE TABLE t(x INT CHECK(x>0), y INT CHECK(y<>0), CHECK(x<>y), CHECK(x+y>0))",
        );
        assert_eq!(t.checks.len(), 4, "2 column-level + 2 table-level checks");
    }

    // --- foreign key + generated column capture (metadata only) ----------------

    #[test]
    fn column_level_foreign_key_is_captured() {
        // A column-level `REFERENCES p(x)` is captured as a single-column FK whose child
        // column is the owning column — no longer silently dropped.
        let t = tdef("CREATE TABLE t(a INTEGER REFERENCES p(x), b)");
        assert_eq!(t.foreign_keys.len(), 1);
        let fk = &t.foreign_keys[0];
        assert_eq!(fk.child_columns, ["a"]);
        assert_eq!(fk.parent_table, "p");
        assert_eq!(fk.parent_columns, ["x"]);
        assert_eq!(fk.on_delete, ReferentialAction::NoAction);
        assert_eq!(fk.on_update, ReferentialAction::NoAction);
        assert!(!fk.deferred);
    }

    #[test]
    fn column_level_foreign_key_without_columns_means_parent_pk() {
        // `REFERENCES p` with no column list records an EMPTY `parent_columns`, the marker
        // for "the parent's PRIMARY KEY" — deliberately NOT resolved to a column here.
        let t = tdef("CREATE TABLE t(a REFERENCES p)");
        assert_eq!(t.foreign_keys.len(), 1);
        assert_eq!(t.foreign_keys[0].parent_table, "p");
        assert!(t.foreign_keys[0].parent_columns.is_empty(), "empty = references parent PK");
    }

    #[test]
    fn table_level_foreign_key_is_captured_with_all_child_columns() {
        // A table-level `FOREIGN KEY(a,b) REFERENCES p(x,y)` records both child columns and
        // the parallel parent columns, in order.
        let t = tdef("CREATE TABLE t(a, b, FOREIGN KEY(a,b) REFERENCES p(x,y))");
        assert_eq!(t.foreign_keys.len(), 1);
        let fk = &t.foreign_keys[0];
        assert_eq!(fk.child_columns, ["a", "b"]);
        assert_eq!(fk.parent_table, "p");
        assert_eq!(fk.parent_columns, ["x", "y"]);
    }

    #[test]
    fn foreign_key_actions_are_captured_and_absent_ones_default_to_no_action() {
        // ON DELETE / ON UPDATE actions are captured; an absent action defaults to NO
        // ACTION (SQLite's default), independently per direction.
        let both = tdef("CREATE TABLE t(a REFERENCES p ON DELETE CASCADE ON UPDATE SET NULL)");
        assert_eq!(both.foreign_keys[0].on_delete, ReferentialAction::Cascade);
        assert_eq!(both.foreign_keys[0].on_update, ReferentialAction::SetNull);

        // Only ON DELETE given: ON UPDATE stays the NO ACTION default.
        let del = tdef("CREATE TABLE t(a REFERENCES p ON DELETE SET DEFAULT)");
        assert_eq!(del.foreign_keys[0].on_delete, ReferentialAction::SetDefault);
        assert_eq!(del.foreign_keys[0].on_update, ReferentialAction::NoAction);

        // RESTRICT is captured too (on the other direction).
        let upd = tdef("CREATE TABLE t(a REFERENCES p ON UPDATE RESTRICT)");
        assert_eq!(upd.foreign_keys[0].on_update, ReferentialAction::Restrict);
        assert_eq!(upd.foreign_keys[0].on_delete, ReferentialAction::NoAction);
    }

    #[test]
    fn foreign_keys_preserve_declaration_order_column_then_table() {
        // Column-level FKs are recorded as columns are walked, then table-level FKs — the
        // overall declaration order the pragma later numbers (last-declared = id 0).
        let t = tdef(
            "CREATE TABLE t(a REFERENCES p1, b REFERENCES p2, FOREIGN KEY(a) REFERENCES p3)",
        );
        let parents: Vec<&str> = t.foreign_keys.iter().map(|fk| fk.parent_table.as_str()).collect();
        assert_eq!(parents, ["p1", "p2", "p3"], "column FKs in order, then table FKs");
    }

    #[test]
    fn deferred_foreign_key_flag_is_only_set_for_initially_deferred() {
        // DEFERRABLE INITIALLY DEFERRED is the one timing the flag records.
        let d = tdef("CREATE TABLE t(a REFERENCES p DEFERRABLE INITIALLY DEFERRED)");
        assert!(d.foreign_keys[0].deferred);
        // INITIALLY IMMEDIATE, bare DEFERRABLE, and NOT DEFERRABLE are all non-deferred.
        let imm = tdef("CREATE TABLE t(a REFERENCES p DEFERRABLE INITIALLY IMMEDIATE)");
        assert!(!imm.foreign_keys[0].deferred);
        let bare = tdef("CREATE TABLE t(a REFERENCES p DEFERRABLE)");
        assert!(!bare.foreign_keys[0].deferred);
        let not = tdef("CREATE TABLE t(a REFERENCES p NOT DEFERRABLE)");
        assert!(!not.foreign_keys[0].deferred);
    }

    #[test]
    fn a_table_without_foreign_keys_has_an_empty_vec() {
        assert!(tdef("CREATE TABLE t(a INT, b TEXT)").foreign_keys.is_empty());
    }

    #[test]
    fn generated_column_stored_and_virtual_are_captured() {
        // STORED and VIRTUAL both capture the generation expression with the right flag; a
        // bare `AS (expr)` (no keyword) is VIRTUAL (SQLite's default). A non-generated
        // column stays `generated: None`.
        let t =
            tdef("CREATE TABLE t(a INT, s AS (a+1) STORED, v AS (a*2) VIRTUAL, d AS (a-1))");
        assert!(t.columns[0].generated.is_none(), "a plain column is not generated");
        assert!(t.columns[1].generated.as_ref().expect("s is generated").stored, "STORED");
        assert!(!t.columns[2].generated.as_ref().expect("v is generated").stored, "VIRTUAL");
        assert!(
            !t.columns[3].generated.as_ref().expect("d is generated").stored,
            "bare AS (expr) defaults to VIRTUAL"
        );
    }

    #[test]
    fn generated_always_keyword_form_is_captured() {
        // The full `GENERATED ALWAYS AS (expr) STORED` spelling captures identically to the
        // short `AS (expr) STORED` form.
        let t = tdef("CREATE TABLE t(a INT, g INT GENERATED ALWAYS AS (a+1) STORED)");
        assert!(t.columns[1].generated.as_ref().expect("g is generated").stored);
    }

    #[test]
    fn generated_column_captures_the_exact_generation_expression() {
        // The STORED/VIRTUAL flag is not the only thing that must be captured: the generation
        // EXPRESSION itself must be recorded faithfully — a placeholder or wrong-source clone
        // would be a silent metadata loss that no `.stored` assertion notices. Cross-check the
        // captured expr against the SAME `a + 1` parsed through an INDEPENDENT builder arm — a
        // table-level CHECK, whose parser consumes `( expr )` identically (see parser::ddl) — so
        // a mutation of the generated-capture arm alone is caught by the `Expr` equality, with
        // no coupling to the Debug shape. A second, structurally different generated column
        // pins that each column captures its OWN expression, not one shared clone.
        let t = tdef("CREATE TABLE t(a INT, g AS (a + 1) STORED, h AS (a * 7) VIRTUAL, CHECK(a + 1))");
        assert_eq!(t.checks.len(), 1, "the table-level CHECK is the independent `a + 1` oracle");
        let g = t.columns[1].generated.as_ref().expect("g is generated");
        let h = t.columns[2].generated.as_ref().expect("h is generated");
        assert_eq!(&g.expr, &t.checks[0], "g's captured generation expr is exactly `a + 1`");
        assert_ne!(g.expr, h.expr, "each generated column captures its own expr, not a shared clone");
    }

    #[test]
    fn default_literal_forms_render_to_sql_text() {
        assert_eq!(tdef("CREATE TABLE t(a INT DEFAULT 5)").columns[0].default.as_deref(), Some("5"));
        assert_eq!(
            tdef("CREATE TABLE t(a INT DEFAULT -5)").columns[0].default.as_deref(),
            Some("-5")
        );
        assert_eq!(
            tdef("CREATE TABLE t(a TEXT DEFAULT 'it''s')").columns[0].default.as_deref(),
            Some("'it''s'")
        );
        assert_eq!(
            tdef("CREATE TABLE t(a TEXT DEFAULT NULL)").columns[0].default.as_deref(),
            Some("NULL")
        );
        assert_eq!(
            tdef("CREATE TABLE t(a TEXT DEFAULT CURRENT_TIMESTAMP)").columns[0].default.as_deref(),
            Some("CURRENT_TIMESTAMP")
        );
        // A parenthesized expression default keeps the source text between the
        // parentheses, outer whitespace trimmed, as sqlite3's `dflt_value` shows it.
        assert_eq!(
            tdef("CREATE TABLE t(a INT DEFAULT (1+2))").columns[0].default.as_deref(),
            Some("1+2")
        );
        assert_eq!(
            tdef("CREATE TABLE t(a TEXT DEFAULT (  'a' ||  'b' ))").columns[0].default.as_deref(),
            Some("'a' ||  'b'")
        );
    }

    #[test]
    fn default_values_are_materialized_for_each_literal_storage_class() {
        // `decode_table_row_enc` fills a short row's missing column from `default_value`, so a
        // wrong `eval_constant_default` arm silently returns a wrong value on an existing
        // row after ADD COLUMN. Pin every literal arm here (end-to-end coverage only
        // reaches TEXT/INTEGER). `Value` is not `PartialEq`, so match on the variant.
        fn dv(sql: &str) -> Option<Value> {
            tdef(sql).columns[0].default_value.clone()
        }
        assert!(matches!(dv("CREATE TABLE t(a INT DEFAULT 5)"), Some(Value::Integer(5))));
        assert!(matches!(dv("CREATE TABLE t(a INT DEFAULT -5)"), Some(Value::Integer(-5))));
        assert!(matches!(dv("CREATE TABLE t(a REAL DEFAULT 3.14)"), Some(Value::Real(f)) if f == 3.14));
        assert!(matches!(dv("CREATE TABLE t(a TEXT DEFAULT 'x')"), Some(Value::Text(s)) if s == "x"));
        assert!(matches!(dv("CREATE TABLE t(a BLOB DEFAULT x'01FF')"), Some(Value::Blob(b)) if b == [0x01, 0xFF]));
        // SQLite's TRUE/FALSE keyword literals are the integers 1/0.
        assert!(matches!(dv("CREATE TABLE t(a INT DEFAULT TRUE)"), Some(Value::Integer(1))));
        assert!(matches!(dv("CREATE TABLE t(a INT DEFAULT FALSE)"), Some(Value::Integer(0))));
        // `DEFAULT NULL` materializes to Some(Null); a short row then reads NULL, the same
        // as the no-default fallback.
        assert!(matches!(dv("CREATE TABLE t(a DEFAULT NULL)"), Some(Value::Null)));
        // No constant value: no default, a `DEFAULT (expr)`, and a time-dependent default.
        assert!(dv("CREATE TABLE t(a INT)").is_none());
        assert!(dv("CREATE TABLE t(a INT DEFAULT (1+2))").is_none());
        assert!(dv("CREATE TABLE t(a TEXT DEFAULT CURRENT_TIMESTAMP)").is_none());
    }

    #[test]
    fn as_select_body_is_a_reported_gap_not_a_fabrication() {
        let ast = parse("CREATE TABLE t AS SELECT 1").unwrap();
        let Statement::CreateTable(ct) = &ast.statements[0] else { panic!() };
        let err = table_def_from_ast(ct, 2).unwrap_err();
        assert!(matches!(err, Error::Sql(_)));
    }

    #[test]
    fn without_rowid_records_the_flag() {
        let t = tdef("CREATE TABLE t(x INTEGER PRIMARY KEY) WITHOUT ROWID");
        assert!(t.without_rowid);
        assert_eq!(t.rowid_alias, None);
    }

    #[test]
    fn strict_records_the_flag_and_any_has_no_affinity() {
        // The flag reaches the def (with and without WITHOUT ROWID), and an `ANY` column
        // of a STRICT table has no affinity while an ordinary table's `ANY` is NUMERIC.
        let t = tdef("CREATE TABLE t(a ANY, b INT) STRICT");
        assert!(t.strict);
        assert_eq!(t.column_affinity(0), Affinity::Blob);
        assert_eq!(t.column_affinity(1), Affinity::Integer);
        assert!(tdef("CREATE TABLE w(k INT PRIMARY KEY) STRICT, WITHOUT ROWID").strict);
        let n = tdef("CREATE TABLE n(a ANY)");
        assert!(!n.strict);
        assert_eq!(n.column_affinity(0), Affinity::Numeric);
    }

    // --- AUTOINCREMENT restrictions (autoinc.html §3) --------------------------

    /// Run the full parse + build pipeline, returning the `Result`. Unlike `tdef`,
    /// this does NOT unwrap the parse, so a case the *parser* rejects (a bare
    /// `AUTOINCREMENT` with no `PRIMARY KEY`) still surfaces as an `Err` to assert on.
    fn try_build(sql: &str) -> Result<TableDef> {
        let ast = parse(sql)?;
        match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => table_def_from_ast(ct, 2),
            _ => Err(Error::Sql(format!("not a single CREATE TABLE: {sql:?}"))),
        }
    }

    #[test]
    fn valid_integer_pk_autoincrement_builds() {
        // The one legal placement: a single-column INTEGER PRIMARY KEY (rowid alias)
        // in a rowid table. It must build, and the column is still the rowid alias.
        let t = tdef("CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");
        assert_eq!(t.rowid_alias, Some(0));
        assert!(!t.without_rowid);
        // ASC is explicitly allowed too (it does not trip the DESC quirk).
        assert_eq!(tdef("CREATE TABLE t(x INTEGER PRIMARY KEY ASC AUTOINCREMENT)").rowid_alias, Some(0));
    }

    #[test]
    fn autoincrement_on_non_integer_pk_is_rejected() {
        // A PRIMARY KEY whose declared type is not exactly INTEGER cannot carry
        // AUTOINCREMENT: TEXT, the integer-ish INT/BIGINT, and an untyped column.
        for sql in [
            "CREATE TABLE t(x TEXT PRIMARY KEY AUTOINCREMENT)",
            "CREATE TABLE t(x INT PRIMARY KEY AUTOINCREMENT)",
            "CREATE TABLE t(x BIGINT PRIMARY KEY AUTOINCREMENT)",
            "CREATE TABLE t(x PRIMARY KEY AUTOINCREMENT)",
        ] {
            let err = try_build(sql).unwrap_err();
            assert!(
                matches!(&err, Error::Sql(m) if m == "AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY"),
                "{sql:?} -> {err:?}"
            );
        }
    }

    #[test]
    fn autoincrement_on_column_level_desc_pk_is_rejected() {
        // The retained early-SQLite quirk: a column-level `INTEGER PRIMARY KEY DESC`
        // is not a rowid alias, so AUTOINCREMENT on it is the "only allowed on an
        // INTEGER PRIMARY KEY" error — the same as a wrong-typed PK.
        let err = try_build("CREATE TABLE t(x INTEGER PRIMARY KEY DESC AUTOINCREMENT)").unwrap_err();
        assert!(
            matches!(&err, Error::Sql(m) if m == "AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY"),
            "{err:?}"
        );
    }

    #[test]
    fn autoincrement_on_without_rowid_is_rejected() {
        // A valid INTEGER PK alias, but WITHOUT ROWID: the second, distinct error.
        let err =
            try_build("CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT) WITHOUT ROWID").unwrap_err();
        assert!(
            matches!(&err, Error::Sql(m) if m == "AUTOINCREMENT not allowed on WITHOUT ROWID tables"),
            "{err:?}"
        );
    }

    #[test]
    fn autoincrement_error_precedence_type_before_without_rowid() {
        // Both restrictions apply (non-integer PK AND WITHOUT ROWID). SQLite raises
        // the type error first (at PK-add time), so pin that ordering.
        let err =
            try_build("CREATE TABLE t(x TEXT PRIMARY KEY AUTOINCREMENT) WITHOUT ROWID").unwrap_err();
        assert!(
            matches!(&err, Error::Sql(m) if m == "AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY"),
            "{err:?}"
        );
    }

    #[test]
    fn bare_autoincrement_without_primary_key_is_rejected_by_the_parser() {
        // AUTOINCREMENT with no PRIMARY KEY is not even representable in the AST (the
        // flag lives only on a column PRIMARY KEY constraint); the parser rejects it.
        // Asserting the whole pipeline errors keeps this a behavioral guard, not a
        // claim about which layer catches it.
        assert!(try_build("CREATE TABLE t(x INTEGER AUTOINCREMENT)").is_err());
        assert!(try_build("CREATE TABLE t(x TEXT AUTOINCREMENT)").is_err());
    }

    #[test]
    fn has_autoincrement_detects_only_a_column_pk_autoincrement() {
        fn detect(sql: &str) -> bool {
            let ast = parse(sql).unwrap();
            let Statement::CreateTable(ct) = &ast.statements[0] else { panic!() };
            has_autoincrement(ct)
        }
        assert!(detect("CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)"));
        // A plain INTEGER PRIMARY KEY (no AUTOINCREMENT keyword) is NOT autoincrement.
        assert!(!detect("CREATE TABLE t(x INTEGER PRIMARY KEY)"));
        assert!(!detect("CREATE TABLE t(x INTEGER PRIMARY KEY, y UNIQUE)"));
        assert!(!detect("CREATE TABLE t(a, b)"));
    }

    #[test]
    fn table_def_records_the_autoincrement_flag() {
        // The built def carries the schema fact the INSERT operator gates rowid-seeding
        // on: an `INTEGER PRIMARY KEY AUTOINCREMENT` sets it, a plain `INTEGER PRIMARY
        // KEY` does not. Only a valid placement reaches the def (validate_autoincrement
        // rejects the rest before construction), so a `true` here is always a legal alias.
        assert!(tdef("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)").autoincrement);
        assert!(!tdef("CREATE TABLE t(id INTEGER PRIMARY KEY, v)").autoincrement);
        // ASC AUTOINCREMENT is still autoincrement (it does not trip the DESC quirk).
        assert!(tdef("CREATE TABLE t(id INTEGER PRIMARY KEY ASC AUTOINCREMENT, v)").autoincrement);
        // A table with no primary key at all is not autoincrement.
        assert!(!tdef("CREATE TABLE t(a, b)").autoincrement);
        // A non-INTEGER-PK table (a UNIQUE column) is not autoincrement.
        assert!(!tdef("CREATE TABLE t(a INTEGER, b UNIQUE)").autoincrement);
    }

    // --- index builder ---------------------------------------------------------

    /// Parse a single `CREATE INDEX` and build its def against `table_columns` at a
    /// fixed root page (3).
    fn idef(sql: &str, table_name: &str, table_columns: &[ColumnDef]) -> Result<IndexDef> {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        let stmt = match ast.statements.as_slice() {
            [Statement::CreateIndex(ci)] => ci,
            other => panic!("expected one CREATE INDEX, got {other:?}"),
        };
        index_def_from_ast(stmt, table_name, table_columns, 3)
    }

    /// Minimal catalog `ColumnDef`s named `names` (types/flags irrelevant to the
    /// index builder, which only checks column existence by name).
    fn cols(names: &[&str]) -> Vec<ColumnDef> {
        names
            .iter()
            .map(|n| ColumnDef {
                name: n.to_string(),
                declared_type: None,
                not_null: false,
                primary_key: false,
                unique: false,
                collation: None,
                default: None,
                default_value: None,
                generated: None,
            })
            .collect()
    }

    #[test]
    fn index_named_columns_build_in_declared_order() {
        let t = cols(&["a", "b", "c"]);
        let def = idef("CREATE INDEX i ON t(b, a)", "t", &t).unwrap();
        assert_eq!(def.name, "i");
        assert_eq!(def.table, "t");
        assert_eq!(def.columns, ["b", "a"], "columns keep the CREATE INDEX order");
        assert!(!def.unique);
        assert!(!def.partial);
        assert_eq!(def.root_page, 3);
    }

    #[test]
    fn index_key_exprs_are_all_none_and_parallel_to_columns() {
        // `key_exprs` is built in lockstep with `columns`, and a plain NAMED-column index
        // (a `Name` target or a `COLLATE`-over-bare-column) fabricates no key expression, so
        // every slot is `None` — even with COLLATE / DESC / multiple key columns or a partial
        // WHERE. (A GENUINE expression key instead captures its `Some(expr)`; see
        // `expression_index_is_accepted_and_captures_its_key_expr`.) Pin it here so a builder
        // change that desyncs the vectors or fabricates a `Some` for a plain column is caught
        // by a real assertion (the parallel-length `debug_assert` is trivially true by
        // construction, so it guards nothing on its own).
        let t = cols(&["a", "b", "c"]);
        for sql in [
            "CREATE INDEX i ON t(b, a)",
            "CREATE INDEX i ON t(a COLLATE NOCASE, b DESC)",
            "CREATE UNIQUE INDEX i ON t(a) WHERE a > 0",
        ] {
            let def = idef(sql, "t", &t).unwrap();
            assert_eq!(
                def.key_exprs.len(),
                def.columns.len(),
                "{sql:?}: key_exprs must stay parallel to columns"
            );
            assert!(
                def.key_exprs.iter().all(|e| e.is_none()),
                "{sql:?}: a plain named-column index fabricates no key expression"
            );
        }
    }

    #[test]
    fn index_column_names_match_case_insensitively() {
        // The column is declared `Abc`; indexing `ABC` must resolve to it, and the
        // def records the name as written in the CREATE INDEX.
        let def = idef("CREATE INDEX i ON t(ABC)", "t", &cols(&["Abc"])).unwrap();
        assert_eq!(def.columns, ["ABC"]);
    }

    #[test]
    fn index_unknown_column_is_reported() {
        let err = idef("CREATE INDEX i ON t(zzz)", "t", &cols(&["a"])).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "unknown column -> Sql, got {err:?}");
    }

    #[test]
    fn index_unique_and_partial_flags_are_captured() {
        let t = cols(&["a"]);
        assert!(idef("CREATE UNIQUE INDEX i ON t(a)", "t", &t).unwrap().unique);
        assert!(idef("CREATE INDEX i ON t(a) WHERE a > 0", "t", &t).unwrap().partial);
        assert!(!idef("CREATE INDEX i ON t(a)", "t", &t).unwrap().partial);
    }

    #[test]
    fn expression_index_is_accepted_and_captures_its_key_expr() {
        // A GENUINE expression index (`lang_createindex.html` §1.2) is now ACCEPTED, not
        // refused. An expression key column has no name, so `columns` carries the
        // empty-string sentinel and `key_exprs` carries the parsed expression; the parallel
        // vectors stay lockstep. `lower(a)`, and a `COLLATE` wrapping a non-column
        // expression (`lower(a) COLLATE NOCASE`), are both genuine expressions — only a
        // `COLLATE`-over-a-bare-column stays an ordinary key column.
        for sql in ["CREATE INDEX i ON t(a + 1)", "CREATE INDEX i ON t(lower(a))"] {
            let def = idef(sql, "t", &cols(&["a"])).unwrap();
            assert_eq!(def.columns, [""], "{sql:?}: an expression key has the empty-name sentinel");
            assert_eq!(def.key_exprs.len(), 1, "{sql:?}: one key_expr slot");
            assert!(def.key_exprs[0].is_some(), "{sql:?}: the key expression is captured");
            assert_eq!(def.key_columns.len(), 1, "{sql:?}: key_columns stays parallel");
            assert_eq!(def.key_columns[0].collation, None, "{sql:?}: no COLLATE override");
        }

        // A `COLLATE` over a genuine expression is captured too, with the collation lifted
        // onto the key column (mirroring the ordinary `COLLATE`-over-column extraction).
        let coll = idef("CREATE INDEX i ON t(lower(a) COLLATE NOCASE)", "t", &cols(&["a"])).unwrap();
        assert_eq!(coll.columns, [""]);
        assert!(coll.key_exprs[0].is_some());
        assert_eq!(coll.key_columns[0].collation.as_deref(), Some("NOCASE"));
    }

    #[test]
    fn expression_index_column_refs_are_not_existence_checked_here() {
        // An expression key's column references are validated by the planner when it binds
        // the expression, NOT by the catalog builder — so `t(zzz + 1)` on a table without
        // `zzz` still BUILDS here (it would fail loudly at plan time). This is unlike an
        // ordinary named key column, which IS existence-checked (see `index_unknown_column`).
        let def = idef("CREATE INDEX i ON t(zzz + 1)", "t", &cols(&["a"])).unwrap();
        assert_eq!(def.columns, [""]);
        assert!(def.key_exprs[0].is_some());
    }

    #[test]
    fn index_ast_key_exprs_classifies_columns_and_expressions() {
        // The shared classifier: a plain column and a `COLLATE`-over-bare-column are `None`
        // (ordinary), a genuine expression is `Some`. A mixed key preserves position.
        let plain = index_ast_key_exprs(&parse_index("CREATE INDEX i ON t(a)"));
        assert_eq!(plain.len(), 1);
        assert!(plain[0].is_none(), "a plain column is not an expression");

        let coll = index_ast_key_exprs(&parse_index("CREATE INDEX i ON t(a COLLATE NOCASE)"));
        assert!(coll[0].is_none(), "COLLATE-over-column is an ordinary key column");

        let expr = index_ast_key_exprs(&parse_index("CREATE INDEX i ON t(a + 1)"));
        assert!(expr[0].is_some(), "a + 1 is a genuine expression key");

        let mixed = index_ast_key_exprs(&parse_index("CREATE INDEX i ON t(a, b + c)"));
        assert_eq!(mixed.len(), 2);
        assert!(mixed[0].is_none(), "slot 0 (a) is ordinary");
        assert!(mixed[1].is_some(), "slot 1 (b + c) is an expression");
    }

    /// Parse one CREATE INDEX to its AST (for `index_ast_key_exprs`, which takes the stmt).
    fn parse_index(sql: &str) -> CreateIndex {
        let ast = parse(sql).unwrap();
        match ast.statements.as_slice() {
            [Statement::CreateIndex(ci)] => (**ci).clone(),
            other => panic!("expected one CREATE INDEX, got {other:?}"),
        }
    }

    #[test]
    fn index_collate_over_column_is_a_plain_key_column() {
        // `x COLLATE NOCASE` parses as an Expr(Collate) target, but it is a PLAIN key
        // column with a collation override — not an expression index. It must build.
        let def = idef("CREATE INDEX i ON t(x COLLATE NOCASE)", "t", &cols(&["x"])).unwrap();
        assert_eq!(def.columns, ["x"], "columns is the stable read seam: just the name");
        assert_eq!(def.key_columns.len(), 1);
        assert_eq!(def.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(!def.key_columns[0].descending);
    }

    #[test]
    fn index_collate_over_column_resolves_case_insensitively() {
        // Declared column `x`, indexed as `X COLLATE NOCASE`: it resolves (case-insensitive
        // existence check) and records the collation, with the name as written.
        let def = idef("CREATE INDEX i ON t(X COLLATE NOCASE)", "t", &cols(&["x"])).unwrap();
        assert_eq!(def.columns, ["X"]);
        assert_eq!(def.key_columns[0].collation.as_deref(), Some("NOCASE"));
    }

    #[test]
    fn index_asc_desc_orders_are_recorded() {
        // DESC records descending; ASC and unspecified both record ascending.
        let desc = idef("CREATE INDEX i ON t(x DESC)", "t", &cols(&["x"])).unwrap();
        assert!(desc.key_columns[0].descending, "DESC -> descending");
        assert_eq!(desc.key_columns[0].collation, None);

        let mixed = idef("CREATE INDEX i ON t(x ASC, y)", "t", &cols(&["x", "y"])).unwrap();
        assert_eq!(mixed.columns, ["x", "y"]);
        assert!(!mixed.key_columns[0].descending, "explicit ASC -> ascending");
        assert!(!mixed.key_columns[1].descending, "unspecified -> ascending");
    }

    #[test]
    fn index_collate_and_desc_combine_per_column() {
        // Each key column carries its own override; `a COLLATE NOCASE` (asc) then `b DESC`.
        let def =
            idef("CREATE INDEX i ON t(a COLLATE NOCASE, b DESC)", "t", &cols(&["a", "b"])).unwrap();
        assert_eq!(def.columns, ["a", "b"]);
        assert_eq!(def.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(!def.key_columns[0].descending);
        assert_eq!(def.key_columns[1].collation, None);
        assert!(def.key_columns[1].descending);

        // A single key column may carry BOTH a collation and DESC: `x COLLATE NOCASE DESC`.
        let both = idef("CREATE INDEX i ON t(x COLLATE NOCASE DESC)", "t", &cols(&["x"])).unwrap();
        assert_eq!(both.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(both.key_columns[0].descending);
    }

    #[test]
    fn index_collate_over_unknown_column_still_fails_closed() {
        // A `COLLATE`-over-column is not skipped as an expression, so a missing column
        // is still reported (not silently accepted).
        let err = idef("CREATE INDEX i ON t(zzz COLLATE NOCASE)", "t", &cols(&["a"])).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "unknown column -> Sql, got {err:?}");
    }

    // --- auto-index derivation (auto_indexes_for) ------------------------------

    /// Parse a single `CREATE TABLE` and derive its auto-index specs.
    fn auto(sql: &str) -> Vec<AutoIndexSpec> {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        let stmt = match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => ct,
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        };
        auto_indexes_for(stmt)
    }

    /// Build an expected [`AutoIndexSpec`] concisely. Every key column defaults to the
    /// inherit-collation (`None`), ascending [`KeyColumn`] — the shape for a plain
    /// UNIQUE/PK with no per-column `COLLATE`/`DESC`. Tests that assert an explicit
    /// override inspect `key_columns` directly instead.
    fn spec(n: usize, name: &str, columns: &[&str], emit_row: bool) -> AutoIndexSpec {
        AutoIndexSpec {
            n,
            name: name.to_string(),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            key_columns: columns
                .iter()
                .map(|_| KeyColumn { collation: None, descending: false })
                .collect(),
            emit_row,
        }
    }

    #[test]
    fn auto_index_numbering_and_columns() {
        // A column-level UNIQUE -> one auto-index on that column.
        assert_eq!(
            auto("CREATE TABLE t(a UNIQUE, b)"),
            vec![spec(1, "sqlite_autoindex_t_1", &["a"], true)]
        );

        // Two table-level UNIQUEs -> N in declaration (textual) order.
        assert_eq!(
            auto("CREATE TABLE t(a, b, UNIQUE(a), UNIQUE(b))"),
            vec![
                spec(1, "sqlite_autoindex_t_1", &["a"], true),
                spec(2, "sqlite_autoindex_t_2", &["b"], true),
            ]
        );

        // INTEGER PRIMARY KEY is the rowid alias: it consumes no N, so the UNIQUE is _1.
        assert_eq!(
            auto("CREATE TABLE t(a INTEGER PRIMARY KEY, b UNIQUE)"),
            vec![spec(1, "sqlite_autoindex_t_1", &["b"], true)]
        );

        // A table-level composite PRIMARY KEY (rowid table) -> one index on (a,b).
        assert_eq!(
            auto("CREATE TABLE t(a, b, PRIMARY KEY(a,b))"),
            vec![spec(1, "sqlite_autoindex_t_1", &["a", "b"], true)]
        );

        // No key constraints at all -> no auto-indexes.
        assert_eq!(auto("CREATE TABLE t(a,b)"), Vec::<AutoIndexSpec>::new());
    }

    #[test]
    fn without_rowid_pk_reserves_n_but_emits_no_row() {
        // The WITHOUT ROWID PRIMARY KEY reserves its N in declaration order but owns no
        // separate index (the table b-tree is the PK index): the UNIQUE before it is
        // _1 (emit), the PK is _2 (no row). This pins the exact N assignment.
        assert_eq!(
            auto("CREATE TABLE t(a UNIQUE, b, PRIMARY KEY(b)) WITHOUT ROWID"),
            vec![
                spec(1, "sqlite_autoindex_t_1", &["a"], true),
                spec(2, "sqlite_autoindex_t_2", &["b"], false),
            ]
        );
    }

    #[test]
    fn table_level_integer_pk_alias_consumes_no_n() {
        // A table-level PRIMARY KEY naming a single INTEGER column is the rowid alias:
        // no auto-index, no N — so the UNIQUE (seen earlier in declaration order) is _1.
        assert_eq!(
            auto("CREATE TABLE t(x INTEGER, y UNIQUE, PRIMARY KEY(x))"),
            vec![spec(1, "sqlite_autoindex_t_1", &["y"], true)]
        );
    }

    #[test]
    fn without_rowid_integer_pk_does_not_reserve_n() {
        // schematab.html: the sqlite_autoindex name is never allocated for an INTEGER
        // PRIMARY KEY, EITHER in rowid OR WITHOUT ROWID tables. So a column-level integer
        // PK consumes no N even WITHOUT ROWID, and the following UNIQUE(b) is _1 (not _2).
        // (Regression guard: the old code reserved N for it via the WITHOUT ROWID branch.)
        assert_eq!(
            auto("CREATE TABLE t(a INTEGER PRIMARY KEY, b UNIQUE) WITHOUT ROWID"),
            vec![spec(1, "sqlite_autoindex_t_1", &["b"], true)]
        );
    }

    #[test]
    fn without_rowid_table_level_integer_pk_does_not_reserve_n() {
        // The same exclusion for a table-level PRIMARY KEY over a single INTEGER column:
        // UNIQUE(b), seen earlier in declaration order, is _1.
        assert_eq!(
            auto("CREATE TABLE t(a INTEGER, b UNIQUE, PRIMARY KEY(a)) WITHOUT ROWID"),
            vec![spec(1, "sqlite_autoindex_t_1", &["b"], true)]
        );
    }

    #[test]
    fn table_level_integer_pk_with_collate_is_the_rowid_alias_and_owns_no_auto_index() {
        // A per-column COLLATE on a table-level PK term must NOT flip rowid-alias detection.
        // `compute_rowid_alias` (which sets `rowid_alias`) and `auto_indexes_for` (which
        // decides whether to emit the PK's auto-index) both resolve the PK column through
        // the SAME `bare_column_name` unwrap, so they cannot disagree. A `Name`-only match
        // here used to make `PRIMARY KEY(x COLLATE NOCASE)` on an INTEGER column report
        // `rowid_alias = None` WHILE `auto_indexes_for` still emitted `sqlite_autoindex_t_1`
        // for it — a self-contradictory model (x both "not the alias" and "owns an index
        // over itself"), diverging from the page-1 image real sqlite writes (only the table
        // row, no auto-index). Real sqlite skips the COLLATE resolving the PK column.
        let plain = tdef("CREATE TABLE t(x INTEGER, PRIMARY KEY(x))");
        assert_eq!(plain.rowid_alias, Some(0));
        assert!(plain.auto_indexes.is_empty());

        // COLLATE on the PK term is the ONLY difference — the model must be IDENTICAL.
        let coll = tdef("CREATE TABLE t(x INTEGER, PRIMARY KEY(x COLLATE NOCASE))");
        assert_eq!(coll.rowid_alias, Some(0), "COLLATE must not disqualify the integer alias");
        assert!(coll.auto_indexes.is_empty(), "the aliased INTEGER PK owns no auto-index");
        assert_eq!(coll.auto_indexes, plain.auto_indexes);

        // COLLATE + DESC together: table-level sort order does not disqualify the alias
        // (unlike the column-level DESC quirk), and COLLATE still resolves to column x.
        let coll_desc = tdef("CREATE TABLE t(x INTEGER, PRIMARY KEY(x COLLATE NOCASE DESC))");
        assert_eq!(coll_desc.rowid_alias, Some(0));
        assert!(coll_desc.auto_indexes.is_empty());
    }

    #[test]
    fn without_rowid_table_level_integer_pk_with_collate_reserves_no_n() {
        // schematab.html: the `sqlite_autoindex` name is never allocated for an INTEGER
        // PRIMARY KEY in a WITHOUT ROWID table either, so it reserves no N. A COLLATE on the
        // PK term names the same INTEGER column, so it stays excluded — the model must match
        // the non-COLLATE case (a WITHOUT ROWID table has no rowid, so `rowid_alias` is None
        // in both; the exclusion is driven by `compute_rowid_alias(.., false)` all the same).
        let plain = tdef("CREATE TABLE t(x INTEGER, y UNIQUE, PRIMARY KEY(x)) WITHOUT ROWID");
        let coll =
            tdef("CREATE TABLE t(x INTEGER, y UNIQUE, PRIMARY KEY(x COLLATE NOCASE)) WITHOUT ROWID");
        assert_eq!(plain.rowid_alias, None);
        assert_eq!(coll.rowid_alias, None);
        // The COLLATE is invisible to the schema model: same specs, INTEGER PK excluded so
        // the UNIQUE(y) is `_1` (the PK consumed no N — no spurious `_2` for x).
        assert_eq!(coll.auto_indexes, plain.auto_indexes);
        assert_eq!(coll.auto_indexes.len(), 1);
        assert_eq!(coll.auto_indexes[0].columns, ["y"]);
        assert_eq!(coll.auto_indexes[0].n, 1, "UNIQUE(y) is _1; the INTEGER PK consumed no N");
    }

    #[test]
    fn without_rowid_non_integer_pk_reserves_n() {
        // Contrast with the integer case: a NON-integer WITHOUT ROWID PK still reserves
        // its N (emit_row=false, the table b-tree is the key), so a following UNIQUE is
        // numbered past it. Column-level TEXT PK here -> PK is _1 (no row), UNIQUE(b) _2.
        assert_eq!(
            auto("CREATE TABLE t(a TEXT PRIMARY KEY, b UNIQUE) WITHOUT ROWID"),
            vec![
                spec(1, "sqlite_autoindex_t_1", &["a"], false),
                spec(2, "sqlite_autoindex_t_2", &["b"], true),
            ]
        );
    }

    #[test]
    fn without_rowid_column_pk_desc_still_reserves_n() {
        // A column-level `INTEGER PRIMARY KEY DESC` is NOT an integer alias (the retained
        // early-SQLite DESC quirk compute_rowid_alias encodes), so it is treated like a
        // non-integer WITHOUT ROWID PK: it reserves its N (emit_row=false) and UNIQUE(b)
        // is _2. Pins the quirk boundary so the integer-PK exclusion does not over-reach.
        // The reserved PK's key column also records its DESC direction (no separate index
        // owns it, but the sort metadata is captured uniformly with the explicit path).
        let mut pk = spec(1, "sqlite_autoindex_t_1", &["a"], false);
        pk.key_columns[0].descending = true;
        assert_eq!(
            auto("CREATE TABLE t(a INTEGER PRIMARY KEY DESC, b UNIQUE) WITHOUT ROWID"),
            vec![pk, spec(2, "sqlite_autoindex_t_2", &["b"], true)]
        );
    }

    #[test]
    fn rowid_table_non_integer_column_pk_emits_auto_index() {
        // A column-level PRIMARY KEY that is NOT the INTEGER rowid alias (here TEXT) is
        // a real auto-index in a rowid table.
        assert_eq!(
            auto("CREATE TABLE t(a TEXT PRIMARY KEY, b)"),
            vec![spec(1, "sqlite_autoindex_t_1", &["a"], true)]
        );
    }

    #[test]
    fn auto_index_name_uses_table_original_spelling() {
        // TABLE in sqlite_autoindex_TABLE_N is the table's original spelling.
        assert_eq!(auto("CREATE TABLE MyT(a UNIQUE)")[0].name, "sqlite_autoindex_MyT_1");
    }

    // --- auto-index COLLATE / DESC metadata (the on-disk-format gap being fixed) ------

    #[test]
    fn auto_index_table_level_collate_is_modelled_with_collation() {
        // A table-level UNIQUE over `x COLLATE NOCASE` is NOT dropped (as it was before):
        // its auto-index is emitted, carrying the collation override on its key column.
        let uq = auto("CREATE TABLE t(x TEXT, UNIQUE(x COLLATE NOCASE))");
        assert_eq!(uq.len(), 1, "the COLLATE UNIQUE emits an auto-index");
        assert_eq!(uq[0].name, "sqlite_autoindex_t_1");
        assert_eq!(uq[0].columns, ["x"]);
        assert_eq!(uq[0].key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(uq[0].emit_row);

        // Same for a table-level PRIMARY KEY over a non-integer COLLATE column.
        let pk = auto("CREATE TABLE t(x TEXT, PRIMARY KEY(x COLLATE NOCASE))");
        assert_eq!(pk.len(), 1);
        assert_eq!(pk[0].columns, ["x"]);
        assert_eq!(pk[0].key_columns[0].collation.as_deref(), Some("NOCASE"));
    }

    #[test]
    fn auto_index_column_level_unique_inherits_column_collation() {
        // `x TEXT COLLATE NOCASE UNIQUE`: the collation lives on the COLUMN
        // (ColumnDef.collation) and the auto-index key inherits it — its KeyColumn
        // collation stays None (= inherit), NOT a duplicated "NOCASE". Do not regress.
        let uq = auto("CREATE TABLE t(x TEXT COLLATE NOCASE UNIQUE)");
        assert_eq!(uq.len(), 1);
        assert_eq!(uq[0].columns, ["x"]);
        assert_eq!(uq[0].key_columns[0].collation, None, "inherit column collation, not duplicate");
    }

    #[test]
    fn auto_index_column_level_pk_desc_records_descending() {
        // A column-level `PRIMARY KEY DESC` on a non-alias (TEXT) column, rowid table:
        // its emitted auto-index key records descending order.
        let specs = auto("CREATE TABLE t(a TEXT PRIMARY KEY DESC, b)");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].columns, ["a"]);
        assert!(specs[0].emit_row);
        assert!(specs[0].key_columns[0].descending, "column-level PK DESC -> descending key");
    }

    #[test]
    fn auto_index_genuine_expression_constraint_consumes_n_but_emits_no_row() {
        // A GENUINE table-level expression constraint is still a deferred gap: it
        // consumes its N (so the following UNIQUE numbers past it) but emits no spec.
        // Only the COLLATE-over-column case above is now modelled.
        assert_eq!(
            auto("CREATE TABLE t(a, b, UNIQUE(a+b), UNIQUE(b))"),
            vec![spec(2, "sqlite_autoindex_t_2", &["b"], true)]
        );
    }

    // --- create-time structural validation --------
    //
    // Each rule below rejects a `CREATE TABLE` that real sqlite errors on at create time,
    // one the purely-syntactic parser accepts — before this landed, `table_def_from_ast` built
    // a def for it (the accept-what-sqlite-rejects gap). The builder is now
    // the single fail-closed site, mirroring `validate_autoincrement`. `try_build` runs the
    // full parse + build pipeline and returns the `Result`; the tests pin the ERROR (the
    // load-bearing correctness win) and its message, and pin that every legal shape still
    // BUILDS. The `assert_ok` / `assert_sql_err` helpers keep each case a single line.

    /// Assert `sql` builds (a legal shape the rules must NOT reject).
    fn assert_ok(sql: &str) {
        assert!(try_build(sql).is_ok(), "expected {sql:?} to build, got {:?}", try_build(sql));
    }

    /// Assert `sql` is rejected with exactly `want` as the `Error::Sql` message.
    fn assert_sql_err(sql: &str, want: &str) {
        match try_build(sql).unwrap_err() {
            Error::Sql(m) => assert_eq!(m, want, "{sql:?}"),
            other => panic!("{sql:?} -> expected Sql({want:?}), got {other:?}"),
        }
    }

    // Rule 1 — STRICT tables (stricttables.html §2 rules 1 & 2).

    #[test]
    fn strict_table_accepts_the_six_datatypes_case_insensitively() {
        // The exact six, and case variants of each, must build under STRICT.
        for ty in [
            "INT", "INTEGER", "REAL", "TEXT", "BLOB", "ANY", // canonical
            "int", "integer", "real", "text", "blob", "any", // lower
            "InT", "InTeGeR", "ReAl", "TeXt", "BlOb", "AnY", // mixed
        ] {
            assert_ok(&format!("CREATE TABLE t(a {ty})  STRICT"));
        }
    }

    #[test]
    fn strict_table_with_several_typed_columns_builds() {
        // A fully-typed STRICT table with one column of each allowed type.
        let t = tdef("CREATE TABLE t(a INT, b INTEGER, c REAL, d TEXT, e BLOB, f ANY) STRICT");
        assert_eq!(t.columns.len(), 6);
    }

    #[test]
    fn strict_with_without_rowid_together_builds() {
        // STRICT and WITHOUT ROWID compose (comma-separated trailing options); both honored.
        let t = tdef("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) STRICT, WITHOUT ROWID");
        assert!(t.without_rowid);
    }

    #[test]
    fn strict_generated_column_with_a_valid_type_builds() {
        // STRICT applies to generated columns too — a valid-typed one passes, VIRTUAL or
        // STORED. The short `AS (expr)` spelling is used because the current parser's
        // `parse_type_name` greedily consumes the identifier-capable keywords `GENERATED
        // ALWAYS` into the declared type (so `b INT GENERATED ALWAYS AS (...)` mis-parses
        // its type as "INT GENERATED ALWAYS"); the short form stops the type at `AS` and
        // parses `INT`/`TEXT` correctly. That parser quirk is a separate, out-of-crate bug;
        // this test pins the create-time STRICT rule, not the parser.
        assert_ok("CREATE TABLE t(a INT, b TEXT AS (a) VIRTUAL) STRICT");
        assert_ok("CREATE TABLE t(a INT, b INT AS (a + 1) STORED) STRICT");
    }

    #[test]
    fn strict_table_missing_datatype_is_rejected() {
        // Rule 1a: every STRICT column must specify a datatype — a plain untyped column and
        // an untyped GENERATED column both fail (STRICT covers generated columns).
        assert_sql_err("CREATE TABLE t(a) STRICT", "missing datatype for t.a");
        assert_sql_err("CREATE TABLE t(a INT, b) STRICT", "missing datatype for t.b");
        assert_sql_err("CREATE TABLE t(a INT, g AS (a + 1)) STRICT", "missing datatype for t.g");
    }

    #[test]
    fn strict_table_unknown_datatype_is_rejected_against_the_whole_declared_type() {
        // Rule 1b: the datatype must be EXACTLY one of the six; the check is against the
        // WHOLE declared type text, so a parenthesized type — even one whose leading word is
        // allowed, like INTEGER(10) — and a multi-word type are rejected, with the message
        // quoting the type verbatim.
        assert_sql_err("CREATE TABLE t(a FLOAT) STRICT", "unknown datatype for t.a: \"FLOAT\"");
        assert_sql_err("CREATE TABLE t(a DOUBLE) STRICT", "unknown datatype for t.a: \"DOUBLE\"");
        assert_sql_err(
            "CREATE TABLE t(a VARCHAR(10)) STRICT",
            "unknown datatype for t.a: \"VARCHAR(10)\"",
        );
        assert_sql_err(
            "CREATE TABLE t(a INTEGER(10)) STRICT",
            "unknown datatype for t.a: \"INTEGER(10)\"",
        );
        assert_sql_err(
            "CREATE TABLE t(a UNSIGNED BIG INT) STRICT",
            "unknown datatype for t.a: \"UNSIGNED BIG INT\"",
        );
        // A GENERATED column is checked uniformly: a valid-typed one builds (above), an
        // INVALID-typed one is the same unknown-datatype error as a plain column. (Short
        // `AS (expr)` spelling so the parser types it as `FLOAT`, not the long-form quirk.)
        assert_sql_err(
            "CREATE TABLE t(a INT, g FLOAT AS (a + 1)) STRICT",
            "unknown datatype for t.g: \"FLOAT\"",
        );
    }

    #[test]
    fn non_strict_table_allows_any_or_no_column_type() {
        // The six-type restriction applies ONLY under STRICT: a non-STRICT table with a
        // missing type or an arbitrary type still builds (no regression). `sqlite_sequence`
        // / `sqlite_stat1` are exactly this shape (untyped columns), so this pins that the
        // internal-table build path is untouched.
        assert_ok("CREATE TABLE t(a)");
        assert_ok("CREATE TABLE t(a FLOAT, b VARCHAR(10), c)");
        assert_ok("CREATE TABLE t(a UNSIGNED BIG INT)");
        assert_ok("CREATE TABLE sqlite_sequence(name, seq)");
    }

    // Rule 2 — duplicate column name (lang_createtable.html).

    #[test]
    fn duplicate_column_name_is_rejected_reporting_the_second_spelling() {
        // Two columns sharing a name (ASCII case-insensitive) is an error, and sqlite
        // reports the SECOND occurrence's exact spelling.
        assert_sql_err("CREATE TABLE t(a, a)", "duplicate column name: a");
        assert_sql_err("CREATE TABLE t(a, A)", "duplicate column name: A");
        assert_sql_err("CREATE TABLE t(a INT, b TEXT, A REAL)", "duplicate column name: A");
    }

    #[test]
    fn all_distinct_column_names_build() {
        assert_ok("CREATE TABLE t(a, b, c)");
        assert_ok("CREATE TABLE t(ab, ba, a_b)");
    }

    // Rule 3 — more than one PRIMARY KEY (lang_createtable.html §3.5).

    #[test]
    fn more_than_one_primary_key_is_rejected() {
        // column+column, column+table, and table+table each declare two PK clauses.
        let want = "table \"t\" has more than one primary key";
        assert_sql_err("CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER PRIMARY KEY)", want);
        assert_sql_err("CREATE TABLE t(a INTEGER PRIMARY KEY, b, PRIMARY KEY(b))", want);
        assert_sql_err("CREATE TABLE t(a, b, PRIMARY KEY(a), PRIMARY KEY(b))", want);
    }

    #[test]
    fn a_single_primary_key_declaration_builds_including_composite() {
        // One column PK, one table PK, and a single COMPOSITE table PK are each ONE clause.
        assert_ok("CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        assert_ok("CREATE TABLE t(a, b, PRIMARY KEY(a))");
        assert_ok("CREATE TABLE t(a, b, PRIMARY KEY(a, b))");
    }

    // Rule 4 — table-level PRIMARY KEY / UNIQUE naming an unknown column.

    #[test]
    fn table_constraint_naming_an_unknown_column_is_rejected() {
        assert_sql_err("CREATE TABLE t(a, PRIMARY KEY(b))", "table t has no column named b");
        assert_sql_err("CREATE TABLE t(a, UNIQUE(b))", "table t has no column named b");
        // A composite PK is one clause (Rule 3 passes); the missing member is reported here.
        assert_sql_err("CREATE TABLE t(a, b, PRIMARY KEY(a, c))", "table t has no column named c");
    }

    #[test]
    fn table_constraint_naming_existing_columns_builds() {
        // Existing columns pass, case-insensitively; a COLLATE-over-column term resolves to
        // the column it names (the shared bare_column_name unwrap), so it is not a phantom.
        assert_ok("CREATE TABLE t(a, b, PRIMARY KEY(a))");
        assert_ok("CREATE TABLE t(a, b, UNIQUE(A))");
        assert_ok("CREATE TABLE t(a TEXT, UNIQUE(a COLLATE NOCASE))");
        // A genuine expression table constraint is a deferred gap, not a missing column, so
        // it is skipped by Rule 4 and still builds (matching auto_indexes_for's treatment).
        assert_ok("CREATE TABLE t(a, b, UNIQUE(a + b))");
    }

    // Rule 4b — table-level FOREIGN KEY naming an unknown CHILD column. A child-column error
    // needs only this one table's definition, so it is a create-time DDL error — unlike the
    // parent-side FK errors (parent/parent-key), which are deferred DML errors
    // (foreignkeys.html §3, ~lines 446-471). The parent is therefore never validated here.

    #[test]
    fn foreign_key_naming_an_unknown_child_column_is_rejected() {
        // A bare table-level FK child column the table does not declare is a create-time error,
        // reported with the child name quoted exactly as written.
        assert_sql_err(
            "CREATE TABLE t(a, FOREIGN KEY(b) REFERENCES p(x))",
            "unknown column \"b\" in foreign key definition",
        );
        // Composite child list: the FIRST missing member in declaration order is reported
        // (here `a` exists, `c` does not).
        assert_sql_err(
            "CREATE TABLE t(a, b, FOREIGN KEY(a, c) REFERENCES p(x, y))",
            "unknown column \"c\" in foreign key definition",
        );
        // Existence is ASCII case-insensitive, so a child `B` with no column `b`/`B` at all is
        // missing — and the message quotes it exactly as written (`B`, not `b`).
        assert_sql_err(
            "CREATE TABLE t(a, FOREIGN KEY(B) REFERENCES p(x))",
            "unknown column \"B\" in foreign key definition",
        );
        // TWO missing members: the FIRST in declaration order (`x`) is reported, not the last —
        // pins the tie-break the spec specifies (an impl reporting `y` would be caught here).
        assert_sql_err(
            "CREATE TABLE t(a, FOREIGN KEY(x, y) REFERENCES p(m, n))",
            "unknown column \"x\" in foreign key definition",
        );
        // A miss in a LATER FK constraint (the first is valid) is caught by the outer
        // per-constraint loop, reporting the offending child (`z`) of the second FK.
        assert_sql_err(
            "CREATE TABLE t(a, b, FOREIGN KEY(a) REFERENCES p(x), FOREIGN KEY(z) REFERENCES q(y))",
            "unknown column \"z\" in foreign key definition",
        );
    }

    #[test]
    fn foreign_key_with_valid_child_columns_builds() {
        // Child column exists case-insensitively: `FOREIGN KEY(A)` naming column `a` must NOT
        // be rejected on case.
        assert_ok("CREATE TABLE t(a, FOREIGN KEY(A) REFERENCES p(x))");
        // The parent is NEVER validated at create time — a nonexistent parent (and parent
        // column) with a REAL child column still builds; that mismatch is a deferred DML error.
        assert_ok("CREATE TABLE t(a, FOREIGN KEY(a) REFERENCES nonexistent(zzz))");
        // A column-level `REFERENCES` has no explicit child list — its child column is the
        // column being defined, which always exists — so it is not subject to this check.
        assert_ok("CREATE TABLE t(a REFERENCES p(x))");
        // A composite child list with every member present builds.
        assert_ok("CREATE TABLE t(a, b, FOREIGN KEY(a, b) REFERENCES p(x, y))");
        // MULTIPLE table-level FK constraints, each naming a declared child column, all build
        // (the outer per-constraint loop accepts every valid one).
        assert_ok("CREATE TABLE t(a, b, FOREIGN KEY(a) REFERENCES p(x), FOREIGN KEY(b) REFERENCES q(y))");
    }

    // Rule 5 — a WITHOUT ROWID table must declare a PRIMARY KEY (withoutrowid.html).

    #[test]
    fn without_rowid_requires_a_primary_key() {
        // "Every WITHOUT ROWID table must have a PRIMARY KEY." A WITHOUT ROWID table keys its
        // rows by the PK, so with none there is no key at all — real sqlite errors
        // `PRIMARY KEY missing on table <t>`.
        assert_sql_err("CREATE TABLE t(a, b) WITHOUT ROWID", "PRIMARY KEY missing on table t");
        // A PK in any accepted form keeps the table legal: a column PK, a table PK, and a
        // single COMPOSITE table PK must all still build.
        assert_ok("CREATE TABLE t(a INTEGER PRIMARY KEY, b) WITHOUT ROWID");
        assert_ok("CREATE TABLE t(a, b, PRIMARY KEY(a)) WITHOUT ROWID");
        assert_ok("CREATE TABLE t(a, b, PRIMARY KEY(a, b)) WITHOUT ROWID");
        // The rule is WITHOUT ROWID only: an ordinary rowid table with no PK is fine (its
        // rows are keyed by the implicit rowid).
        assert_ok("CREATE TABLE t(a, b)");
    }

    // Rule 6 — generated-column restrictions (gencol.html §2.3). The short `AS (expr)`
    // spelling is used throughout because the parser's `parse_type_name` greedily folds
    // `GENERATED ALWAYS` into the declared type (see the STRICT tests' note); the short
    // form is a genuine generated column and sidesteps that separate parser quirk.

    #[test]
    fn generated_column_with_default_is_rejected() {
        // §2.3.1: a generated column may not carry a DEFAULT (its value is always the AS
        // expression). Rejected regardless of the order the two clauses are written in.
        let want = "cannot use DEFAULT on a generated column";
        assert_sql_err("CREATE TABLE t(a, b AS (a) DEFAULT 5)", want);
        assert_sql_err("CREATE TABLE t(a, b DEFAULT 5 AS (a))", want);
    }

    #[test]
    fn generated_column_without_default_builds() {
        // A generated column with no DEFAULT is legal; a DEFAULT on a NON-generated column
        // is equally legal — the rule is specific to generated columns.
        assert_ok("CREATE TABLE t(a, b AS (a))");
        assert_ok("CREATE TABLE t(a, b AS (a) VIRTUAL)");
        assert_ok("CREATE TABLE t(a, b AS (a + 1) STORED)");
        assert_ok("CREATE TABLE t(a, b DEFAULT 5)");
    }

    #[test]
    fn generated_column_in_primary_key_is_rejected() {
        // §2.3.2: a generated column may not be part of the PRIMARY KEY, whether declared
        // column-level on the generated column or named in a table-level PRIMARY KEY(...)
        // list (the latter matched case-insensitively, like the other constraint checks).
        let want = "generated columns cannot be part of the primary key";
        assert_sql_err("CREATE TABLE t(a, b AS (a) PRIMARY KEY)", want);
        assert_sql_err("CREATE TABLE t(a, b AS (a), PRIMARY KEY(b))", want);
        assert_sql_err("CREATE TABLE t(a, b AS (a), PRIMARY KEY(B))", want);
        // Also caught when the generated column is one member of a COMPOSITE table PK.
        assert_sql_err("CREATE TABLE t(a, b AS (a), PRIMARY KEY(a, b))", want);
        // A `COLLATE`-over-column PK term still names the generated column (the parser lands
        // `b COLLATE NOCASE` as a COLLATE-over-bare-column, which `bare_column_name` unwraps),
        // so it is caught too — the COLLATE does not smuggle a generated column into the PK.
        assert_sql_err("CREATE TABLE t(a, b AS (a), PRIMARY KEY(b COLLATE NOCASE))", want);
    }

    #[test]
    fn generated_column_outside_primary_key_builds() {
        // A generated column that is NOT part of the PRIMARY KEY builds, whether the PK is
        // a column-level PK on another column or a table-level PK naming only real columns.
        assert_ok("CREATE TABLE t(a PRIMARY KEY, b AS (a))");
        assert_ok("CREATE TABLE t(a, b AS (a), PRIMARY KEY(a))");
    }

    #[test]
    fn table_of_only_generated_columns_is_rejected() {
        // §2.3.6: every table must have at least one non-generated column.
        let want = "must have at least one non-generated column";
        assert_sql_err("CREATE TABLE t(a AS (1))", want);
        assert_sql_err("CREATE TABLE t(a AS (1), b AS (2))", want);
    }

    #[test]
    fn table_with_at_least_one_ordinary_column_builds() {
        // One ordinary column among any number of generated ones satisfies §2.3.6.
        assert_ok("CREATE TABLE t(a, b AS (a))");
        assert_ok("CREATE TABLE t(a INT, b AS (a) VIRTUAL, c AS (a + 1) STORED)");
    }

    // Rule 7 — a column `DEFAULT (<expr>)` must be a CONSTANT expression
    // (lang_createtable.html §3.2). The parser accepts any parenthesized expression as a
    // `DefaultValue::Expr`, so a non-constant one — a column/table reference, a bound
    // parameter, or a sub-query — is caught here at create time (matching real sqlite's
    // `default value of column [<col>] is not constant`), while a constant one (including a
    // DQS text literal and an `IN (value-list)`) still builds unchanged.

    /// The exact create-time error real sqlite reports for a non-constant default of `col`.
    fn not_constant(col: &str) -> String {
        format!("default value of column [{col}] is not constant")
    }

    #[test]
    fn column_reference_default_is_rejected() {
        // A bare column reference, and one nested under a binary op / inside a function
        // argument — the nested cases prove the constant-walk recurses into children, not
        // just the top node.
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (a))", &not_constant("b"));
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (a + 1))", &not_constant("b"));
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (abs(a)))", &not_constant("b"));
        // A qualified double-quoted reference `t."a"` is a GENUINE reference
        // (from_dqs:false), unlike a bare `"a"` (a DQS text literal, tested below), so it is
        // rejected. (Verified the parser yields Column{from_dqs:false} for the dotted form.)
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (t.\"a\"))", &not_constant("b"));
    }

    #[test]
    fn subquery_default_is_rejected() {
        // A scalar sub-query, an EXISTS, and an `IN (SELECT …)` are each non-constant on
        // sight (the walk never descends into the Select body). The leading `1` in the IN
        // case keeps the scrutinee constant, so the sub-query is unambiguously the trigger.
        assert_sql_err("CREATE TABLE t(a, b DEFAULT ((SELECT 1)))", &not_constant("b"));
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (EXISTS (SELECT 1)))", &not_constant("b"));
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (1 IN (SELECT 1)))", &not_constant("b"));
    }

    #[test]
    fn bound_parameter_default_is_rejected() {
        // A bound parameter is non-constant (the spec lists it explicitly). Both the
        // anonymous `?` and a named `:x` tokenize to a bind parameter the expression grammar
        // accepts, so both reach the builder as `DefaultValue::Expr(BindParam)`.
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (?))", &not_constant("b"));
        assert_sql_err("CREATE TABLE t(a, b DEFAULT (:x))", &not_constant("b"));
    }

    #[test]
    fn constant_default_expression_still_builds() {
        // Arithmetic over literals, a function over literals, string concat, unary minus,
        // an `IN (value-list)`, and a CASE over literals are all constant — the validator
        // must NOT reject them (erroring where sqlite succeeds is the worse divergence).
        assert_ok("CREATE TABLE t(a, b DEFAULT (1+2))");
        assert_ok("CREATE TABLE t(a, b DEFAULT (abs(-1)))");
        assert_ok("CREATE TABLE t(a, b DEFAULT (1 || 2))");
        assert_ok("CREATE TABLE t(a, b DEFAULT (- (3)))");
        assert_ok("CREATE TABLE t(a, b DEFAULT (1 IN (2,3)))");
        assert_ok("CREATE TABLE t(a, b DEFAULT (CASE WHEN 1 THEN 2 ELSE 3 END))");
    }

    #[test]
    fn literal_defaults_are_never_checked_for_constancy() {
        // A `DefaultValue::Literal` default (unparenthesized literal / signed number / NULL /
        // CURRENT_* / TRUE) is constant by construction — the validator ignores it entirely.
        assert_ok("CREATE TABLE t(a, b INT DEFAULT 5)");
        assert_ok("CREATE TABLE t(a, b INT DEFAULT -5)");
        assert_ok("CREATE TABLE t(a, b DEFAULT NULL)");
        assert_ok("CREATE TABLE t(a, b DEFAULT CURRENT_TIMESTAMP)");
        assert_ok("CREATE TABLE t(a, b INT DEFAULT TRUE)");
    }

    #[test]
    fn accepted_expression_default_materialization_is_unchanged() {
        // The accept path must not be disturbed by the constant-validator: a constant
        // expression default keeps its source text (re-parsed by the INSERT planner) and
        // folds to no build-time constant Value.
        let t = tdef("CREATE TABLE t(a, b DEFAULT (1+2))");
        assert_eq!(t.columns[1].default.as_deref(), Some("1+2"));
        assert!(t.columns[1].default_value.is_none());
    }

    #[test]
    fn dqs_expression_default_is_constant_and_keeps_its_text_literal() {
        // A bare double-quoted `DEFAULT ("lit")` is the DQS TEXT literal 'lit' (constant),
        // NOT a rejected reference — it builds AND keeps its rendered raw text and folded
        // Value, exactly as before this validator landed (see `dqs_default_is_a_text_literal`).
        let t = tdef("CREATE TABLE t(a, b TEXT DEFAULT (\"lit\"))");
        assert_eq!(t.columns[1].default.as_deref(), Some("'lit'"));
        match &t.columns[1].default_value {
            Some(Value::Text(s)) => assert_eq!(s, "lit"),
            other => panic!("expected Text(\"lit\"), got {other:?}"),
        }
    }

    #[test]
    fn deep_default_expression_is_classified_iteratively_without_overflow() {
        // Liveness: the constant-walk must be ITERATIVE. Build a `1 + 1 + … + 1` chain far
        // deeper than any recursive walk could survive on the default (2 MiB) test-thread
        // stack — a recursive checker would abort with a stack overflow here (the abort IS
        // the regression signal). Built directly rather than via `parse`, since the parser
        // caps a flat fold at MAX_EXPR_DEPTH=1000 (a longer chain is a parse error and never
        // reaches the builder); the check itself must still be safe at any tree height.
        const DEPTH: usize = 100_000;
        let lit = || Expr::Literal(Literal::Integer(1));
        let bin = |acc| Expr::Binary { op: BinaryOp::Add, left: Box::new(acc), right: Box::new(lit()) };

        // All-literal chain: constant, so it must return true (and terminate, not overflow).
        {
            let mut e = lit();
            for _ in 0..DEPTH {
                e = bin(e);
            }
            assert!(default_expr_is_constant(&e), "a deep all-literal chain is constant");
        }
        // Same depth with a genuine column reference at the very bottom: the walk must visit
        // every node, still find the reference, and return false — without overflowing.
        {
            let mut e = Expr::Column { schema: None, table: None, name: "a".into(), from_dqs: false };
            for _ in 0..DEPTH {
                e = bin(e);
            }
            assert!(!default_expr_is_constant(&e), "a buried column reference is non-constant");
        }
    }

    #[test]
    fn buried_reference_is_detected_through_every_recursive_arm() {
        // Regression guard for the walk's recursion: a column reference buried inside each
        // recursive Expr arm must still be found. A future edit that stopped descending one
        // arm would otherwise silently ACCEPT a non-constant default. One case per arm shape
        // beyond the Binary / Function(arg) ones already covered above.
        for sql in [
            "CREATE TABLE t(a, b DEFAULT (CASE WHEN a THEN 1 ELSE 2 END))", // Case: a `when`
            "CREATE TABLE t(a, b DEFAULT (CASE a WHEN 1 THEN 2 END))",      // Case: the operand
            "CREATE TABLE t(a, b DEFAULT (CASE WHEN 1 THEN 2 ELSE a END))", // Case: the `else`
            "CREATE TABLE t(a, b DEFAULT (1 IN (a, 2)))",                   // In::List element
            "CREATE TABLE t(a, b DEFAULT (CAST(a AS INT)))",               // Cast child
            "CREATE TABLE t(a, b DEFAULT (a COLLATE NOCASE))",             // Collate child
            "CREATE TABLE t(a, b DEFAULT (a BETWEEN 1 AND 2))",            // Between: scrutinee
            "CREATE TABLE t(a, b DEFAULT (1 BETWEEN a AND 2))",            // Between: low bound
            "CREATE TABLE t(a, b DEFAULT (1 LIKE 2 ESCAPE a))",           // Like: the escape
            "CREATE TABLE t(a, b DEFAULT (NOT a))",                        // Unary child
            "CREATE TABLE t(a, b DEFAULT (a ISNULL))",                     // IsNull child
        ] {
            assert_sql_err(sql, &not_constant("b"));
        }
    }

    #[test]
    fn window_over_clause_hiding_a_reference_is_rejected() {
        // A forbidden node can also hide inside a function's window OVER spec. The constant
        // walk must descend into PARTITION BY / ORDER BY / frame-bound offsets (unlike the
        // teardown enumeration, which omits them). Function args are all-constant here, so the
        // OVER content is the sole trigger — proving `push_over_children` is reached.
        assert_sql_err(
            "CREATE TABLE t(a, b DEFAULT (abs(1) OVER (ORDER BY a)))",
            &not_constant("b"),
        );
        assert_sql_err(
            "CREATE TABLE t(a, b DEFAULT (abs(1) OVER (PARTITION BY a)))",
            &not_constant("b"),
        );
        // A bound parameter buried in a frame-bound offset (`ROWS ? PRECEDING`).
        assert_sql_err(
            "CREATE TABLE t(a, b DEFAULT (max(1) OVER (ORDER BY 1 ROWS ? PRECEDING)))",
            &not_constant("b"),
        );
    }

    #[test]
    fn non_constant_default_names_the_offending_column_not_a_fixed_index() {
        // The error must name the ACTUAL offending column, not a hardcoded position — every
        // other reject test happens to put the bad default on the second column.
        assert_sql_err("CREATE TABLE t(a, b, c DEFAULT (a))", &not_constant("c"));
        assert_sql_err("CREATE TABLE t(first_col DEFAULT (x))", &not_constant("first_col"));
    }

    // Sub-queries prohibited in schema expressions — CHECK constraints
    // (lang_createtable.html), index key expressions (expridx.html /
    // lang_createindex.html), and partial-index WHERE clauses (partialindex.html). Real
    // sqlite errors at create time; this engine's parser accepts the sub-query (no
    // post-parse reject), so `expr_contains_subquery` + its callers fail closed at build
    // time. Only a sub-query is rejected: column references, `IN (value-list)`, and bound
    // parameters are legitimate here and must still build (over-rejecting is the worse
    // divergence).

    /// Extract the first TABLE-level CHECK predicate from a parsed `CREATE TABLE`, so a
    /// sub-query expression can be handed straight to [`expr_contains_subquery`] without
    /// reconstructing the AST by hand. Panics if there is no table-level CHECK.
    fn table_check_expr(sql: &str) -> Expr {
        let ast = parse(sql).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        let stmt = match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => ct,
            other => panic!("expected one CREATE TABLE, got {other:?}"),
        };
        let CreateTableBody::Columns { constraints, .. } = &stmt.body else {
            panic!("expected a columns body for {sql:?}");
        };
        for c in constraints {
            if let TableConstraintKind::Check(e) = &c.kind {
                return e.clone();
            }
        }
        panic!("no table-level CHECK in {sql:?}");
    }

    /// Assert a `CREATE INDEX` is rejected with exactly `want` as the `Error::Sql` message,
    /// built against a two-column table `t(a, b)`.
    fn idef_sql_err(sql: &str, want: &str) {
        match idef(sql, "t", &cols(&["a", "b"])).unwrap_err() {
            Error::Sql(m) => assert_eq!(m, want, "{sql:?}"),
            other => panic!("{sql:?} -> expected Sql({want:?}), got {other:?}"),
        }
    }

    /// Assert a `CREATE INDEX` builds against `t(a, b)`.
    fn idef_ok(sql: &str) {
        let r = idef(sql, "t", &cols(&["a", "b"]));
        assert!(r.is_ok(), "expected {sql:?} to build, got {r:?}");
    }

    #[test]
    fn expr_contains_subquery_flags_each_subquery_form() {
        // Each of the three sub-query node shapes is a hit, whether it is the whole
        // predicate or buried inside a binary op (proving the walk descends children before
        // it can reach the buried node).
        for sql in [
            "CREATE TABLE t(a, CHECK((SELECT 1)))",        // scalar sub-query
            "CREATE TABLE t(a, CHECK(EXISTS (SELECT 1)))", // EXISTS
            "CREATE TABLE t(a, CHECK(a IN (SELECT 1)))",   // IN (SELECT …)
            "CREATE TABLE t(a, CHECK(a IN foo))",          // IN table (a sub-query in disguise)
            "CREATE TABLE t(a, CHECK(a + (SELECT 1) > 0))", // buried under Binary
            "CREATE TABLE t(a, CHECK(CASE WHEN (SELECT 1) THEN 1 END))", // buried in a CASE when
        ] {
            assert!(expr_contains_subquery(&table_check_expr(sql)), "{sql:?}");
        }
    }

    #[test]
    fn expr_contains_subquery_walks_columns_and_params_through() {
        // The whole point of a FOCUSED sub-query check (vs the constant check): a column
        // reference, an `IN (value-list)`, and a bound parameter are NOT sub-queries and must
        // return false, so a valid CHECK / index / WHERE over ordinary columns is accepted.
        for sql in [
            "CREATE TABLE t(a, b, CHECK(a > 0 AND b < a))", // column refs + AND
            "CREATE TABLE t(a, CHECK(a IN (1, 2, 3)))",     // IN (value-list), not a SELECT
            "CREATE TABLE t(a, CHECK(a > ?))",              // bound parameter (allowed here)
            "CREATE TABLE t(a, CHECK(lower(a) = 'x'))",     // function over a column
        ] {
            assert!(!expr_contains_subquery(&table_check_expr(sql)), "{sql:?}");
        }
    }

    #[test]
    fn expr_contains_subquery_descends_through_every_recursive_arm() {
        // Regression guard for the walk's child enumeration: a sub-query buried inside EACH
        // recursive `Expr` arm must still be found. A future edit that stopped pushing a child
        // in one arm would otherwise silently ACCEPT a buried sub-query and no other test would
        // fail — the exact hazard the sibling `buried_reference_is_detected_through_every_
        // recursive_arm` guards for `default_expr_is_constant`. One case per arm shape, beyond
        // the Binary / CASE-when already covered in `expr_contains_subquery_flags_each_subquery_
        // form`. (The window `OVER` arm has its own test below.)
        for sql in [
            "CREATE TABLE t(a, CHECK(NOT (SELECT 1)))",                     // Unary
            "CREATE TABLE t(a, CHECK((SELECT 1) + 0 = 1))",                 // Binary: left
            "CREATE TABLE t(a, CHECK(0 = (SELECT 1)))",                     // Binary: right
            "CREATE TABLE t(a, CHECK(CAST((SELECT 1) AS INT)))",           // Cast child
            "CREATE TABLE t(a, CHECK((SELECT 1) COLLATE NOCASE))",         // Collate child
            "CREATE TABLE t(a, CHECK((SELECT 1) ISNULL))",                 // IsNull child
            "CREATE TABLE t(a, CHECK((SELECT 1) NOTNULL))",                // NotNull child
            "CREATE TABLE t(a, CHECK((SELECT 1) LIKE 'x'))",               // Like: lhs
            "CREATE TABLE t(a, CHECK('x' LIKE (SELECT 1)))",               // Like: rhs
            "CREATE TABLE t(a, CHECK('x' LIKE 'y' ESCAPE (SELECT 1)))",    // Like: escape
            "CREATE TABLE t(a, CHECK((SELECT 1) BETWEEN 1 AND 2))",        // Between: scrutinee
            "CREATE TABLE t(a, CHECK(1 BETWEEN (SELECT 1) AND 2))",        // Between: low
            "CREATE TABLE t(a, CHECK(1 BETWEEN 0 AND (SELECT 1)))",        // Between: high
            "CREATE TABLE t(a, CHECK((SELECT 1) IN (1, 2)))",             // In::List: scrutinee
            "CREATE TABLE t(a, CHECK(1 IN ((SELECT 1), 2)))",             // In::List: element
            "CREATE TABLE t(a, CHECK(abs((SELECT 1)) = 1))",             // Function: arg
            "CREATE TABLE t(a, CHECK(count(*) FILTER (WHERE (SELECT 1)) > 0))", // Function: FILTER
            "CREATE TABLE t(a, CHECK(group_concat(a ORDER BY (SELECT 1)) = 'x'))", // Function: agg ORDER BY
            "CREATE TABLE t(a, CHECK(CASE (SELECT 1) WHEN 1 THEN 2 END))", // Case: operand
            "CREATE TABLE t(a, CHECK(CASE WHEN 1 THEN (SELECT 1) END))",   // Case: a `then`
            "CREATE TABLE t(a, CHECK(CASE WHEN 1 THEN 2 ELSE (SELECT 1) END))", // Case: the `else`
            "CREATE TABLE t(a, CHECK((1, (SELECT 1)) = (2, 3)))",        // Parenthesized: list element
        ] {
            assert!(
                expr_contains_subquery(&table_check_expr(sql)),
                "a sub-query buried in this arm must be found: {sql:?}"
            );
        }
    }

    #[test]
    fn expr_contains_subquery_finds_a_subquery_in_a_window_over_spec() {
        // A sub-query can hide inside a function's window OVER spec — PARTITION BY, ORDER BY, or
        // a frame-bound offset. The walk must descend into it via `push_over_children` (unlike
        // the teardown enumeration `take_expr_children`, which omits the OVER spec). Removing
        // the `push_over_children` call would go unnoticed without this — mirrors the sibling
        // `window_over_clause_hiding_a_reference_is_rejected`. The aggregate args are all-literal
        // here, so the OVER content is the sole trigger.
        for sql in [
            "CREATE TABLE t(a, CHECK(abs(1) OVER (ORDER BY (SELECT 1)) > 0))", // OVER: ORDER BY
            "CREATE TABLE t(a, CHECK(abs(1) OVER (PARTITION BY (SELECT 1)) > 0))", // OVER: PARTITION BY
            "CREATE TABLE t(a, CHECK(max(1) OVER (ORDER BY 1 ROWS (SELECT 1) PRECEDING) > 0))", // OVER: frame offset
        ] {
            assert!(
                expr_contains_subquery(&table_check_expr(sql)),
                "a sub-query hiding in the window OVER spec must be found: {sql:?}"
            );
        }
    }

    #[test]
    fn check_constraint_with_subquery_is_rejected() {
        // All three sub-query shapes, at both column level and table level, fail closed with
        // the CHECK message (the `checks` vec unifies column- and table-level in one pass).
        let want = "subqueries prohibited in CHECK constraints";
        assert_sql_err("CREATE TABLE t(a, b, CHECK(a > (SELECT 1)))", want); // table-level scalar
        assert_sql_err("CREATE TABLE t(a CHECK(a IN (SELECT 1)))", want);    // column-level IN-select
        assert_sql_err("CREATE TABLE t(a, CHECK(EXISTS (SELECT 1)))", want); // table-level EXISTS
    }

    #[test]
    fn check_constraint_without_subquery_still_builds() {
        // Ordinary CHECKs reference columns and value-lists; they must not be rejected.
        assert_ok("CREATE TABLE t(a, b, CHECK(a > 0))");
        assert_ok("CREATE TABLE t(a, b, CHECK(a > 0 AND b < a))");
        assert_ok("CREATE TABLE t(a CHECK(a IN (1, 2, 3)))");
        // The accepted table still records its predicate(s) unchanged.
        assert_eq!(tdef("CREATE TABLE t(a, b, CHECK(a > 0 AND b < a))").checks.len(), 1);
    }

    #[test]
    fn index_expression_with_subquery_is_rejected() {
        // A genuine expression key containing a sub-query fails closed at create time.
        let want = "subqueries prohibited in index expressions";
        idef_sql_err("CREATE INDEX i ON t(a + (SELECT 1))", want);
        idef_sql_err("CREATE INDEX i ON t(a IN (SELECT 1))", want);
        idef_sql_err("CREATE INDEX i ON t(EXISTS (SELECT 1))", want);
    }

    #[test]
    fn partial_index_where_with_subquery_is_rejected() {
        // A partial-index WHERE predicate containing a sub-query fails closed at create time.
        let want = "subqueries prohibited in partial index WHERE clauses";
        idef_sql_err("CREATE INDEX i ON t(a) WHERE a IN (SELECT 1)", want);
        idef_sql_err("CREATE INDEX i ON t(a) WHERE EXISTS (SELECT 1)", want);
        idef_sql_err("CREATE INDEX i ON t(a) WHERE a > (SELECT 1)", want);
    }

    #[test]
    fn index_expression_precedes_partial_where_when_both_have_a_subquery() {
        // When both the ON-list expression and the WHERE predicate hold a sub-query, the
        // index-expression error surfaces first (sqlite binds the ON-list before the WHERE).
        idef_sql_err(
            "CREATE INDEX i ON t(a + (SELECT 1)) WHERE b IN (SELECT 2)",
            "subqueries prohibited in index expressions",
        );
    }

    #[test]
    fn expression_and_partial_index_without_subquery_still_build() {
        // Expression indexes and partial indexes over ordinary expressions / predicates are
        // valid and must not regress (conformance_expression_index / conformance_index_queries).
        idef_ok("CREATE INDEX i ON t(a + b)");
        idef_ok("CREATE INDEX i ON t(lower(a))");
        idef_ok("CREATE INDEX i ON t(a) WHERE a > 0");
        idef_ok("CREATE INDEX i ON t(a) WHERE a IN (1, 2)");
        idef_ok("CREATE INDEX i ON t(a + b) WHERE b > 0");
    }

    #[test]
    fn deep_expression_subquery_detection_is_iterative_without_overflow() {
        // Liveness: `expr_contains_subquery` must be ITERATIVE. Build a `1 + 1 + … + 1` chain
        // far deeper than any recursive walk could survive on the default (2 MiB) test-thread
        // stack — a recursive checker would abort with a stack overflow (the abort IS the
        // regression signal). Built directly rather than via `parse`, since the parser caps a
        // flat fold at MAX_EXPR_DEPTH (a longer chain is a parse error and never reaches the
        // builder); the check itself must still be safe at any tree height. Mirrors
        // `deep_default_expression_is_classified_iteratively_without_overflow`.
        const DEPTH: usize = 100_000;
        let lit = || Expr::Literal(Literal::Integer(1));
        let col = || Expr::Column { schema: None, table: None, name: "a".into(), from_dqs: false };
        let bin =
            |acc, leaf| Expr::Binary { op: BinaryOp::Add, left: Box::new(acc), right: Box::new(leaf) };

        // A deep all-literal chain contains no sub-query: false, and it must terminate.
        {
            let mut e = lit();
            for _ in 0..DEPTH {
                e = bin(e, lit());
            }
            assert!(!expr_contains_subquery(&e), "a deep all-literal chain has no sub-query");
        }
        // A deep all-COLUMN chain likewise has none — column refs are walked through, not a
        // reject (this is where a naive reuse of the constant check would wrongly fire).
        {
            let mut e = col();
            for _ in 0..DEPTH {
                e = bin(e, col());
            }
            assert!(!expr_contains_subquery(&e), "a deep all-column chain has no sub-query");
        }
        // Same depth with a scalar sub-query buried at the very bottom: the walk must visit
        // every node, still find it, and return true — without overflowing.
        {
            let mut e = table_check_expr("CREATE TABLE t(a, CHECK((SELECT 1)))"); // Expr::Subquery
            for _ in 0..DEPTH {
                e = bin(e, lit());
            }
            assert!(expr_contains_subquery(&e), "a buried sub-query is detected at any depth");
        }
    }
}
