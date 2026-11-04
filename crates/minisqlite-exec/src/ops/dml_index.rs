//! Shared index maintenance for the DML write operators (`INSERT`, `UPDATE`, and
//! `DELETE`).
//!
//! The index KEY encoding and the UNIQUE-conflict probe must be IDENTICAL across all
//! three operators: an index entry and the table row it points at must never disagree,
//! or a later read seeks a key that was written under different rules. Keeping the one
//! copy here — rather than a per-operator copy that can drift — makes that a structural
//! guarantee instead of a convention three files have to remember.
//!
//! TWO-PHASE maintenance (EVAL then WRITE). An index key column can be an ORDINARY
//! column (`IndexKeyPart::Col`) or a compiled EXPRESSION (`IndexKeyPart::Expr`, e.g.
//! `CREATE INDEX i ON t(a+b)`). Evaluating an expression key needs an [`EvalCtx`] whose
//! `env` holds a SHARED `&dyn Pager`; writing an index entry needs `&mut dyn Pager`. The
//! two borrows cannot co-exist, so key computation is split from the b-tree write exactly
//! as the DML ops already evaluate CHECK constraints before their mutating write:
//! [`index_key_values`] / [`index_keys_for_plans`] EVAL the keys (shared pager, a ctx),
//! then [`insert_index_entries`] / [`delete_index_entries`] WRITE the precomputed keys
//! (mut pager, no ctx). For an all-ordinary-column index (every key part `Col`) the eval
//! phase never touches the ctx and produces the same `[cols.., rowid]` key a single-phase
//! path would; an index with an expression key column evaluates that column through the ctx.
//! Both shapes share this one path, so the ordinary case is behavior-neutral and the two
//! can never encode a key differently.
//!
//! PARTIAL indexes (`CREATE INDEX … WHERE <pred>`, `partialindex.html` §2) gate on the SAME
//! eval phase: [`IndexPlan::partial_predicate`] carries the compiled WHERE predicate, and the
//! EVAL functions return `None` for a row the predicate does not admit ([`index_admits_row`]),
//! so that row gets no entry, no UNIQUE probe, and no delete in ANY operator. The keys a
//! frame produces are therefore `Option`-typed and stay 1:1 with the plans; the WRITE phase
//! and the operators' probe loops simply skip a `None`. A full index has `partial_predicate ==
//! None` and every key is `Some`, so its behavior is unchanged.
//!
//! Effect surface: [`index_key_values`] / [`index_keys_for_plans`] and [`build_index_plan`]
//! read only values/defs (and, for an expression key, the shared pager via the ctx);
//! [`build_index_plans`] READS the schema (catalog) then delegates per index to
//! [`build_index_plan`]; [`unique_conflict_rowid`] (with its boolean façade
//! [`unique_conflict`]) READS the index b-tree (a pager cursor); only
//! [`insert_index_entries`] / [`delete_index_entries`] WRITE. Every SECONDARY-INDEX mutation
//! flows through those two, so the delete path cannot encode a key differently from the
//! insert path. (The WITHOUT ROWID primary-key b-tree is the table's own storage, not a
//! secondary index, and is written directly by the INSERT operator — not through here.)

use std::cmp::Ordering;

use minisqlite_btree::{
    index_delete_with_order, index_insert_with_order, IndexCursor, KeyField, KeyOrder,
};
use minisqlite_catalog::{Catalog, IndexDef, TableDef};
use minisqlite_expr::{eval, truth, EvalExpr};
use minisqlite_fileformat::{decode_record_enc, encode_record_enc, TextEncoding};
use minisqlite_pager::{PageId, Pager};
use minisqlite_plan::{GeneratedProgram, Plan};
use minisqlite_types::{compare_values, Collation, DbIndex, Error, Result, Value};

use crate::context::EvalCtx;
use crate::env::{Env, Pagers};
use crate::ops::foreign_key;
use crate::ops::generated::compute_generated;
use crate::ops::without_rowid::{wr_layout, WrLayout};
use crate::runtime::Runtime;

/// One key column of an index: either an ordinary table column read by position, or a
/// compiled key EXPRESSION evaluated against the `[c0..c_{N-1}, rowid]` frame. An
/// all-named-column index resolves every part to `Col`; an expression index
/// (`CREATE INDEX i ON t(a+b)`) carries an `Expr` part for each computed key column, fed the
/// compiled expression by the maintenance paths (DML nodes, `CREATE INDEX` backfill, and the
/// FK cascade).
pub(crate) enum IndexKeyPart {
    /// Ordinary named column at this table-column position (`< N`, the table's column count).
    Col(usize),
    /// A compiled key expression, bound against the base-table `[c0..c_{N-1}, rowid]` scope.
    Expr(EvalExpr),
}

/// How a WITHOUT ROWID secondary index appends the trailing PRIMARY KEY to each entry (in
/// place of the rowid a rowid-table index appends) and recovers that PK back out of a
/// decoded entry. Present on an [`IndexPlan`] only for a WR table.
///
/// fileformat2 §2.5.1: a WR index entry key is `[indexed columns.., trailing PK columns..]`,
/// where a PRIMARY KEY column is appended to the suffix ONLY IF it is not already one of the
/// indexed columns WITH A MATCHING collating sequence (a redundant column is suppressed —
/// e.g. `CREATE INDEX ex25ce ON ex25(c,e)` over `PRIMARY KEY(d,c,a)` yields `c,e,d,a`, with
/// `c` suppressed from the suffix). The trailing PK plays the rowid's dual role: it keeps
/// each entry unique (the index b-tree requires unique keys) and it identifies the table row
/// to fetch by PRIMARY KEY. UNIQUE is still enforced over the indexed-column PREFIX alone.
pub(crate) struct WrIndexKey {
    /// Schema-column positions of the PK columns appended after the indexed columns, in PK
    /// order, EXCLUDING every suppressed (already-indexed-with-matching-collation) one.
    /// Empty when the indexed columns already cover the whole PK (§2.5.1's "extreme case").
    trailing: Vec<usize>,
    /// The sort order and collation of each `trailing` column (1:1): the PRIMARY KEY's own,
    /// which is how sqlite3 orders the suffix of a WITHOUT ROWID index entry.
    trailing_fields: Vec<KeyField>,
    /// For each PRIMARY KEY column (in PK order), the position IN A DECODED ENTRY where its
    /// value lives: the indexed-column slot `i` when the column is suppressed (it is present
    /// as indexed column `i`), else `k + t` (k = indexed-column count, t = its slot within
    /// `trailing`). This ONE map recovers the full PK from an entry for the read fetch-by-PK,
    /// the REPLACE victim delete, and the UNIQUE self-exclusion — so all three agree.
    pk_entry_positions: Vec<usize>,
}

/// A precomputed per-index write plan: where its b-tree root is, how each key column is
/// produced (a table-column position or a compiled expression, resolved once — not per
/// row), the per-key-column collating sequence, whether it is UNIQUE, and the
/// violation-message detail (`table.col, ...` for an ordinary index, `index '<name>'` for
/// an index with any expression key column).
pub(crate) struct IndexPlan {
    pub(crate) root: PageId,
    pub(crate) key: Vec<IndexKeyPart>,
    /// The effective collating sequence of each key column (1:1 with `key`, in key order):
    /// the index's explicit per-column `COLLATE` override, else the target column's declared
    /// collation, else `BINARY` (lang_createindex.html §1.5, datatype3.html §7). The UNIQUE
    /// probe consults this so a `NOCASE`/`RTRIM` UNIQUE index treats `'abc'` and `'ABC'` as
    /// the same key — a duplicate a BINARY compare would miss.
    pub(crate) collations: Vec<Collation>,
    /// The index b-tree's key order (sqlite3's `KeyInfo`): one [`KeyField`] per indexed
    /// column (its `collations` entry and the index's `DESC` flag), then, for a WITHOUT
    /// ROWID table, one per trailing PRIMARY KEY column carrying the PK's own sort order and
    /// collation. A rowid table's trailing rowid is past the fields, so it compares
    /// ascending BINARY. Every write, delete, probe, and read of this index uses it.
    pub(crate) order: KeyOrder,
    pub(crate) unique: bool,
    pub(crate) detail: String,
    /// `Some` for a WITHOUT ROWID table's index — the trailing-PK suffix layout (§2.5.1) that
    /// replaces the rowid the EVAL/WRITE/probe paths append for a rowid table; `None` for a
    /// rowid table (the common case), which keeps the byte-identical `[indexed cols.., rowid]`
    /// shape. Populated by [`build_index_plan`] from `def.without_rowid`.
    pub(crate) wr: Option<WrIndexKey>,
    /// `Some` for a PARTIAL index (`CREATE INDEX … WHERE <pred>`) — the compiled WHERE
    /// predicate, bound against the `[c0..c_{N-1}, rowid]` row frame (the SAME frame the key
    /// parts read). A row is IN this index — and so gets an entry, a UNIQUE probe, and a delete
    /// — only when the predicate evaluates to TRUE for it ([`index_admits_row`],
    /// `partialindex.html` §2); a FALSE-or-NULL row is omitted. `None` for a full index (the
    /// common case), which admits every row unconditionally. Populated by
    /// [`build_index_plan`] from the caller's per-index compiled predicate.
    pub(crate) partial_predicate: Option<EvalExpr>,
}

impl IndexPlan {
    /// The table-column positions of this index's key when EVERY key column is an ordinary
    /// named column (`IndexKeyPart::Col`); `None` if any key column is an expression. Used
    /// by the UPSERT conflict-target resolution, which matches an `ON CONFLICT (cols)`
    /// target against an index's column set — a target only names plain columns, so an
    /// expression index can never be the arbiter and `None` is the correct "not a
    /// column-only index" answer.
    pub(crate) fn col_positions(&self) -> Option<Vec<usize>> {
        self.key
            .iter()
            .map(|part| match part {
                IndexKeyPart::Col(p) => Some(*p),
                IndexKeyPart::Expr(_) => None,
            })
            .collect()
    }
}

/// Resolve ONE index's key columns to key parts and build its write plan. The single
/// source of truth for turning an [`IndexDef`] into an [`IndexPlan`] — both the
/// whole-table [`build_index_plans`] (DML maintenance) and the single-index backfill
/// ([`crate::index_build::build_index`]) go through it, so a `CREATE INDEX` backfill
/// encodes keys and names violations exactly as the INSERT/UPDATE path does and the two
/// can never drift.
///
/// `compiled` carries the index's compiled key expressions (aligned with `idx.columns` /
/// `idx.key_exprs`); an entry is `Some(e)` for an expression key column and `None` (or
/// absent) for an ordinary named column. For each key column i:
/// - `compiled[i] == Some(e)` → an expression key column → [`IndexKeyPart::Expr`];
/// - else the index declares an EXPRESSION key at i (`idx.key_exprs[i] == Some(_)`) but no
///   compiled expression was supplied → a LOUD error (this path — e.g. an FK cascade into
///   an expression-indexed child, or a caller that has not wired compiled exprs — cannot
///   honestly build the key, so it fails rather than silently writing a wrong/empty one);
/// - else an ordinary named column → resolve `idx.columns[i]` to a table-column position →
///   [`IndexKeyPart::Col`] (errors if the table lacks that column — a corrupt schema).
///
/// `partial_predicate` is the index's compiled WHERE predicate for a PARTIAL index (`None`
/// for a full index), stored on the plan so [`index_admits_row`] can gate this index's
/// per-row maintenance on it. The caller supplies it already bound (a DML node's
/// `index_partial_predicates`, the `CREATE INDEX` backfill's, or the FK cascade's); this path
/// only carries it, so a full index is byte-for-byte unchanged.
pub(crate) fn build_index_plan(
    def: &TableDef,
    idx: &IndexDef,
    compiled: &[Option<EvalExpr>],
    partial_predicate: Option<EvalExpr>,
) -> Result<IndexPlan> {
    let mut key = Vec::with_capacity(idx.columns.len());
    // The effective collating sequence and sort order of each key column: an explicit
    // per-column `COLLATE` written into the index key takes priority; otherwise it inherits
    // the target column's declared collation, defaulting to BINARY (an expression key with
    // no explicit override is BINARY — it has no source column to inherit from).
    let mut fields = index_key_fields(def, idx);
    let collations: Vec<Collation> = fields.iter().map(|f| f.collation).collect();
    for i in 0..idx.columns.len() {
        let part = if let Some(Some(e)) = compiled.get(i) {
            IndexKeyPart::Expr(e.clone())
        } else if matches!(idx.key_exprs.get(i), Some(Some(_))) {
            return Err(Error::format(format!(
                "expression index {} on {}: key expressions were not compiled for this path",
                idx.name, def.name
            )));
        } else {
            let col = &idx.columns[i];
            let pos = def
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(col))
                .ok_or_else(|| {
                    Error::format(format!(
                        "index {} references unknown column {} on table {}",
                        idx.name, col, def.name
                    ))
                })?;
            IndexKeyPart::Col(pos)
        };
        key.push(part);
    }
    // The `UNIQUE constraint failed: <detail>` message. Real SQLite names the key COLUMNS
    // (`table.col, ...`) for an ordinary index, but an index with ANY expression key column
    // has no column name to print, so it reports `index '<name>'` instead — match that so
    // the error text agrees with real sqlite (an expression key carries an empty-string sentinel
    // in `idx.columns`, which would otherwise render as a bare `table.`).
    let detail = if idx.key_exprs.iter().any(|e| e.is_some()) {
        format!("index '{}'", idx.name)
    } else {
        idx.columns.iter().map(|c| format!("{}.{}", def.name, c)).collect::<Vec<_>>().join(", ")
    };
    // A WITHOUT ROWID table's secondary index appends the trailing PRIMARY KEY (fileformat2
    // §2.5.1) instead of the rowid. Compute that suffix layout once here so the EVAL, WRITE,
    // read, and UNIQUE-probe paths share the ONE encoder; a rowid table gets `None` and keeps
    // its byte-identical `[indexed cols.., rowid]` shape.
    let wr = if def.without_rowid { Some(build_wr_index_key(def, idx, &key, &collations)?) } else { None };
    if let Some(wr) = &wr {
        fields.extend(wr.trailing_fields.iter().copied());
    }
    let order = KeyOrder::new(fields);
    Ok(IndexPlan {
        root: idx.root_page,
        key,
        collations,
        order,
        unique: idx.unique,
        detail,
        wr,
        partial_predicate,
    })
}

/// The sort order and effective collation of each INDEXED column of `idx` (key order), as
/// the index b-tree orders it: the index's explicit per-column `COLLATE`, else (for an
/// ordinary column) the column's declared collation, else BINARY; `DESC` from the index
/// definition. This is the leading part of every [`IndexPlan::order`], usable on its own by
/// a reader that has no compiled key expressions (an expression key column's collation
/// never depends on the expression). A key column naming no column of `def` falls back to
/// BINARY here; [`build_index_plan`] is the path that rejects such a schema.
pub(crate) fn index_key_fields(def: &TableDef, idx: &IndexDef) -> Vec<KeyField> {
    (0..idx.columns.len())
        .map(|i| {
            let kc = idx.key_columns.get(i);
            let explicit = kc.and_then(|kc| kc.collation.as_deref());
            let inherited = if matches!(idx.key_exprs.get(i), Some(Some(_))) {
                None
            } else {
                def.columns
                    .iter()
                    .find(|c| c.name.eq_ignore_ascii_case(&idx.columns[i]))
                    .and_then(|c| c.collation.as_deref())
            };
            KeyField {
                collation: collation_of(explicit.or(inherited)),
                descending: kc.is_some_and(|kc| kc.descending),
            }
        })
        .collect()
}

/// Derive a WITHOUT ROWID index's trailing-PK layout ([`WrIndexKey`]) from its resolved key
/// parts + collations (fileformat2 §2.5.1). Called only when `def.without_rowid`.
///
/// EXPRESSION indexes on a WR table are FAIL-CLOSED here: the planner binds an index key
/// expression against a `[c0.., rowid]` frame a WR row has no rowid for, so an expression key
/// cannot be encoded correctly without a cross-crate (planner) change. Only ordinary
/// named-column secondary indexes are supported on WR tables; an expression key errors loud
/// rather than silently mis-encoding.
fn build_wr_index_key(
    def: &TableDef,
    idx: &IndexDef,
    key: &[IndexKeyPart],
    collations: &[Collation],
) -> Result<WrIndexKey> {
    // The indexed columns as (schema position, effective collation). An expression key part
    // fails closed (see the fn doc): there is no source column to append/suppress against.
    let mut indexed: Vec<(usize, Collation)> = Vec::with_capacity(key.len());
    for (part, coll) in key.iter().zip(collations) {
        match part {
            IndexKeyPart::Col(p) => indexed.push((*p, *coll)),
            IndexKeyPart::Expr(_) => {
                return Err(Error::sql(format!(
                    "CREATE INDEX {} on WITHOUT ROWID table {}: an expression index on a \
                     WITHOUT ROWID table is not yet supported",
                    idx.name, def.name
                )))
            }
        }
    }
    // The PRIMARY KEY columns in PK order (deduped exactly as the WR PK b-tree stores them).
    let layout = wr_layout(def)?;
    let pk = layout.pk_columns().to_vec();
    let mut trailing: Vec<usize> = Vec::new();
    let mut trailing_fields: Vec<KeyField> = Vec::new();
    let mut pk_entry_positions: Vec<usize> = Vec::with_capacity(pk.len());
    for (j, &p) in pk.iter().enumerate() {
        // A PK column is SUPPRESSED from the suffix when it already appears among the indexed
        // columns with a MATCHING collating sequence (§2.5.1). The PK column's collation is
        // the PRIMARY KEY clause's explicit `COLLATE`, else its declared column collation
        // (BINARY default) — the same one the PK b-tree is keyed under.
        let pk_field = layout.pk_field(j);
        match indexed.iter().position(|(ip, ic)| *ip == p && *ic == pk_field.collation) {
            // Suppressed: the value is already carried at indexed slot `i`.
            Some(i) => pk_entry_positions.push(i),
            // Kept: appended after the indexed columns at `k + trailing.len()`.
            None => {
                pk_entry_positions.push(indexed.len() + trailing.len());
                trailing.push(p);
                trailing_fields.push(pk_field);
            }
        }
    }
    Ok(WrIndexKey { trailing, trailing_fields, pk_entry_positions })
}

/// Map a key column's declared/overridden `COLLATE` name to a built-in collating sequence,
/// defaulting to BINARY (datatype3.html §7.1). Only text comparison consults it. Unknown
/// names default to BINARY here — an unknown collation is rejected at `CREATE INDEX` time,
/// so a name that survives into an [`IndexDef`] is one of the three built-ins.
pub(crate) fn collation_of(name: Option<&str>) -> Collation {
    match name {
        Some(n) if n.eq_ignore_ascii_case("NOCASE") => Collation::NoCase,
        Some(n) if n.eq_ignore_ascii_case("RTRIM") => Collation::Rtrim,
        _ => Collation::Binary,
    }
}

/// Resolve every index of `def` to its write plan once, so the per-row loop does no
/// schema string matching. `index_key_exprs` is the per-index compiled key expressions
/// (`(index name, exprs aligned with the index's columns)`) the caller supplies — a DML
/// node's compiled exprs, a `CREATE INDEX` backfill's, or the FK cascade's; each index looks
/// up its own entry by name, defaulting to EMPTY when absent — the case for an ordinary
/// all-named-column index, whose every key column is then an [`IndexKeyPart::Col`]. Errors if
/// an index names a column the table does not have (a corrupt schema) or declares an
/// expression key with no supplied compiled expression (see [`build_index_plan`]).
///
/// `index_partial_predicates` is the parallel per-PARTIAL-index compiled WHERE predicate
/// (`(index name, bound predicate)`), looked up by the SAME index name — a DML node's
/// `index_partial_predicates` or the FK cascade's. An index with no entry is a full index
/// (`None`), maintained for every row; a partial index's per-row maintenance is gated on its
/// predicate ([`index_admits_row`]). The two lists are independent (an index may have a key
/// expression, a partial predicate, both, or neither), so each is matched by name separately.
pub(crate) fn build_index_plans(
    catalog: &dyn Catalog,
    db: DbIndex,
    def: &TableDef,
    index_key_exprs: &[(String, Vec<Option<EvalExpr>>)],
    index_partial_predicates: &[(String, EvalExpr)],
) -> Result<Vec<IndexPlan>> {
    // `db` is the namespace the write targets (a `db`-stamped DML node, or the FK cascade's
    // single namespace). An index lives in the same store as its table, so read this table's
    // indexes with `_in(db)`: the bare `indexes_on` follows the temp->main->attached search
    // order and, under a same-named shadow, would maintain/probe ANOTHER namespace's index for
    // this write. With no shadow `db == MAIN` and this equals the bare form.
    let indexes = catalog.indexes_on_in(db, &def.name)?;
    let mut plans = Vec::with_capacity(indexes.len());
    for idx in indexes {
        let compiled = index_key_exprs
            .iter()
            .find(|(n, _)| n == &idx.name)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[]);
        // A partial index's predicate is matched by name too; a full index finds none. Cloned
        // so each plan owns its predicate independent of the caller's list (the list is a
        // borrowed plan-node field / cascade-local vec, reused across rows).
        let partial_predicate =
            index_partial_predicates.iter().find(|(n, _)| n == &idx.name).map(|(_, e)| e.clone());
        plans.push(build_index_plan(def, idx, compiled, partial_predicate)?);
    }
    Ok(plans)
}

/// Whether a row is a MEMBER of `plan`'s index — always `true` for a full index, and for a
/// PARTIAL index exactly when its WHERE predicate evaluates to TRUE over `frame`
/// (`partialindex.html` §2: a row whose predicate is FALSE *or NULL* is omitted). Uses the
/// SAME `truth` helper as CHECK / WHERE / HAVING so partial-index membership and every other
/// SQL truthiness test agree (numeric != 0 is true; 0 / NULL / '' etc. are not). `frame` is
/// the `[c0..c_{N-1}, rowid]` row the predicate was bound against (the SAME frame the key
/// parts read).
///
/// This is the ONE membership gate: [`index_key_values`] (and its WR twin
/// [`wr_index_key_values`]) return `None` when it is false, so a non-member row gets no
/// entry, no UNIQUE probe, and no delete across every DML/backfill/cascade path — a UNIQUE
/// partial index is therefore enforced only over the rows actually in it.
fn index_admits_row(plan: &IndexPlan, frame: &[Value], ctx: &mut EvalCtx<'_>) -> Result<bool> {
    match &plan.partial_predicate {
        None => Ok(true),
        Some(pred) => Ok(truth(&eval(pred, frame, ctx)?) == Some(true)),
    }
}

/// Build ONE index key record (the EVAL phase): for each key column, the ordinary
/// column's value from `frame` or the compiled expression evaluated against `frame`,
/// followed by the trailing rowid — matching the table row's stored values so the index
/// and table never disagree.
///
/// Returns `None` when `plan` is a PARTIAL index whose predicate does not admit this row
/// ([`index_admits_row`]): the row is not in the index, so there is no key to write, probe,
/// or delete. `Some(key)` for a full index (always) or a partial index that admits the row.
///
/// `frame` MUST be the `[c0..c_{N-1}, rowid]` row (width `N + 1`) the key expressions and the
/// partial predicate were bound against: an `IndexKeyPart::Col(p)` reads `frame[p]` (`p < N`,
/// so it selects a column) and an `IndexKeyPart::Expr` may read `Column(N)` = the rowid. For a
/// full all-`Col` plan `ctx` is never touched, so this clones exactly the same columns + rowid
/// a single-phase path would; an `IndexKeyPart::Expr` part (or a partial predicate) is
/// evaluated through `ctx`.
pub(crate) fn index_key_values(
    plan: &IndexPlan,
    frame: &[Value],
    rowid: i64,
    ctx: &mut EvalCtx<'_>,
) -> Result<Option<Vec<Value>>> {
    if !index_admits_row(plan, frame, ctx)? {
        return Ok(None);
    }
    let mut key = Vec::with_capacity(plan.key.len() + 1);
    for part in &plan.key {
        let v = match part {
            // `p < N` by construction (a Col part is a resolved table-column position, and the
            // frame is `[c0..c_{N-1}, rowid]`), so this never fails in correct usage; guard it
            // anyway so a future wrong-width caller gets a clean engine-bug error, not a raw
            // index panic mid-transaction.
            IndexKeyPart::Col(p) => frame
                .get(*p)
                .ok_or_else(|| {
                    Error::format(format!(
                        "index maintenance: key column position {p} out of range for a \
                         width-{} frame (engine bug)",
                        frame.len()
                    ))
                })?
                .clone(),
            IndexKeyPart::Expr(e) => eval(e, frame, ctx)?,
        };
        key.push(v);
    }
    key.push(Value::Integer(rowid));
    Ok(Some(key))
}

/// Compute the key record for EVERY plan over `frame` (the EVAL phase for a whole row).
/// The result is 1:1 with `plans` BY CONSTRUCTION (one push per plan), which is what lets
/// the WRITE phase ([`insert_index_entries`] / [`delete_index_entries`]) pair a plan with
/// its key positionally without a length-mismatch corrupting the index set.
///
/// Each slot is `None` when its (PARTIAL) index does not admit the row over `frame`
/// ([`index_key_values`]) — the writers skip a `None` and the DML operators skip its UNIQUE
/// probe, so the row is simply not in that index. A full index is always `Some`.
pub(crate) fn index_keys_for_plans(
    plans: &[IndexPlan],
    frame: &[Value],
    rowid: i64,
    ctx: &mut EvalCtx<'_>,
) -> Result<Vec<Option<Vec<Value>>>> {
    let mut keys = Vec::with_capacity(plans.len());
    for ip in plans {
        keys.push(index_key_values(ip, frame, rowid, ctx)?);
    }
    Ok(keys)
}

/// Fail LOUD when the precomputed `keys` are not 1:1 with `plans` — the WRITE-phase
/// contract. A mismatch is an engine bug (`index_keys_for_plans` builds one key per plan):
/// silently `zip`-truncating would leave some index unwritten or delete the wrong entry, so
/// both writers reject it here instead. One spelling of the check so the two can't drift.
fn check_keys_match_plans(plans: &[IndexPlan], keys: &[Option<Vec<Value>>]) -> Result<()> {
    if plans.len() != keys.len() {
        return Err(Error::format(format!(
            "index maintenance: {} index plans but {} precomputed keys (engine bug)",
            plans.len(),
            keys.len()
        )));
    }
    Ok(())
}

/// Write every index entry for a freshly stored row (the WRITE phase): one precomputed
/// `[indexed cols.., rowid]` key per index plan, encoded exactly as the UNIQUE probe and
/// the delete path so the three never diverge. `keys` MUST be 1:1 with `plans` (produced
/// by [`index_keys_for_plans`] over the same plans); a length mismatch is an engine bug
/// that would leave some index unwritten, so it fails LOUD rather than silently skipping.
///
/// `enc` is the database's text encoding (fileformat2 §1.3.13), passed by the caller which
/// reads it ONCE per statement — never re-derived per row here, so the UTF-8 hot path pays
/// no per-row header parse. It must be the same `enc` the delete and probe paths use, since
/// the index b-tree's raw-byte key order depends on it.
///
/// A `None` key is a row this (PARTIAL) index does not hold — it is SKIPPED, adding no entry.
pub(crate) fn insert_index_entries(
    pager: &mut dyn Pager,
    plans: &[IndexPlan],
    keys: &[Option<Vec<Value>>],
    enc: TextEncoding,
) -> Result<()> {
    check_keys_match_plans(plans, keys)?;
    for (ip, key) in plans.iter().zip(keys) {
        if let Some(key) = key {
            index_insert_with_order(pager, ip.root, &encode_record_enc(key, enc), &ip.order)?;
        }
    }
    Ok(())
}

/// Remove every index entry for a row — the inverse of [`insert_index_entries`], for
/// `UPDATE`/`DELETE`. `keys` MUST be 1:1 with `plans` (same fail-loud contract as
/// [`insert_index_entries`]). `index_delete` returns `false` when the entry was already
/// absent; that is IGNORED so a repeated (or partially-applied) delete stays idempotent
/// rather than erroring on an entry that is simply not there.
///
/// A `None` key is a row this (PARTIAL) index did not hold — it is SKIPPED, removing nothing
/// (there was no entry). For UPDATE, the OLD frame decides this key's `None`/`Some`, so a row
/// that was outside the predicate has no stale entry to drop.
pub(crate) fn delete_index_entries(
    pager: &mut dyn Pager,
    plans: &[IndexPlan],
    keys: &[Option<Vec<Value>>],
    enc: TextEncoding,
) -> Result<()> {
    check_keys_match_plans(plans, keys)?;
    for (ip, key) in plans.iter().zip(keys) {
        if let Some(key) = key {
            index_delete_with_order(pager, ip.root, &encode_record_enc(key, enc), &ip.order)?;
        }
    }
    Ok(())
}

/// The rowid of an existing UNIQUE-index entry whose indexed columns equal this row's (at
/// a rowid other than `rowid`), or `None` if there is no conflict. NULLs are distinct in a
/// UNIQUE index (sqlite allows any number of NULLs — lang_createindex.html §1.1), so a key
/// with any NULL indexed column never conflicts. The probe seeks the shared column prefix
/// under the index's key order and compares the landed entry's columns value-wise under
/// each column's collation (consistent with the order the b-tree is built on, so keys equal
/// under a `NOCASE`/`RTRIM` column are adjacent and the seek finds them).
///
/// `key` is the PRECOMPUTED `[indexed cols.., rowid]` record for the plan (from
/// [`index_key_values`], width `ip.key.len() + 1`); the indexed columns probed are
/// `key[..ip.key.len()]`. The trailing rowid IN the key is IGNORED — the separate `rowid`
/// argument is the self-identity to exclude, which the UPDATE path deliberately sets to
/// the OLD rowid while `key` carries the NEW row's values, so the row's own moved/rewritten
/// entry is not mistaken for a conflict.
///
/// This is the ONE index-probe implementation; [`unique_conflict`] is its boolean façade,
/// so "is there a conflict?" and "which row conflicts?" can never diverge. `INSERT OR
/// REPLACE` needs the rowid (to delete the conflicting row); the other paths only need the
/// bool.
pub(crate) fn unique_conflict_rowid(
    pager: &dyn Pager,
    ip: &IndexPlan,
    key: &[Value],
    rowid: i64,
    enc: TextEncoding,
) -> Result<Option<i64>> {
    let k = ip.key.len();
    // The key must be the precomputed `[indexed cols.., rowid]` (width k+1). Enforce it with
    // a release-safe error (consistent with the writers' length guard) so a wrong-width key
    // is a clean engine-bug error, not a `key[..k]` slice panic in release.
    if key.len() != k + 1 {
        return Err(Error::format(format!(
            "index maintenance: unique probe key width {} != {} (indexed cols + rowid) \
             on {} (engine bug)",
            key.len(),
            k + 1,
            ip.detail
        )));
    }
    let cols = &key[..k];
    for v in cols {
        if v.is_null() {
            return Ok(None); // A NULL makes the key distinct from every other.
        }
    }

    // The index b-tree is ordered under the index's own key order (each column's collation
    // and sort order, `ip.order`), so every entry equal to this key's indexed prefix under
    // those collations is adjacent and `seek_ge(prefix)` lands on the first of them: an
    // O(log n) seek + single-entry check for every collation. A prefix key (indexed columns
    // without the rowid) sorts before every full entry sharing that prefix. The probe key is
    // encoded in the database's text encoding so its raw-byte compare in `seek_ge` lines up
    // with the stored (same-encoding) index keys.
    let prefix = encode_record_enc(cols, enc);
    let mut cur = IndexCursor::open_with_order(pager, ip.root, ip.order.clone())?;
    if !cur.seek_ge(&prefix)? {
        return Ok(None);
    }
    let found = cur.key()?;
    let decoded = decode_record_enc(&found, enc);
    if decoded.len() < k + 1 {
        // A malformed/short index entry: don't manufacture a false conflict.
        return Ok(None);
    }
    for (i, want) in cols.iter().enumerate() {
        if compare_values(&decoded[i], want, ip.collations[i]) != Ordering::Equal {
            return Ok(None); // The landed entry has a different key prefix.
        }
    }
    // Same indexed values. `decoded[k]` is that entry's rowid; report a conflict only when
    // it is a DIFFERENT row (a fresh insert's own entry is not written yet, so an equal
    // prefix is always another row; an UPDATE probes with its own rowid to exclude its
    // unchanged entry). Every entry this engine writes has an Integer rowid there (see
    // `index_key_values`); a non-integer means a corrupt index, so fail loud rather than
    // silently claim or drop a conflict against an unusable rowid.
    match &decoded[k] {
        Value::Integer(existing) if *existing != rowid => Ok(Some(*existing)),
        Value::Integer(_) => Ok(None),
        other => Err(Error::format(format!(
            "corrupt index ({}): entry key has a non-integer rowid slot {other:?}",
            ip.detail
        ))),
    }
}

/// Boolean façade over [`unique_conflict_rowid`]: is there an existing UNIQUE-index entry
/// with this row's indexed values at a different rowid? Delegates so there is a SINGLE
/// probe implementation the two questions can never answer differently. Takes the same
/// PRECOMPUTED `[indexed cols.., rowid]` key and separate self-identity `rowid`.
pub(crate) fn unique_conflict(
    pager: &dyn Pager,
    ip: &IndexPlan,
    key: &[Value],
    rowid: i64,
    enc: TextEncoding,
) -> Result<bool> {
    Ok(unique_conflict_rowid(pager, ip, key, rowid, enc)?.is_some())
}

// ===== WITHOUT ROWID secondary-index maintenance ======================================
//
// The WR analogues of the rowid EVAL/WRITE/probe paths above. They differ in ONE respect:
// where a rowid index appends the integer rowid to every entry, a WR index appends the
// trailing PRIMARY KEY columns (fileformat2 §2.5.1, laid out once by [`build_wr_index_key`]
// into [`WrIndexKey`]). The b-tree WRITE ([`insert_index_entries`] / [`delete_index_entries`])
// is shared unchanged — a WR key is just a precomputed `Vec<Value>` like any other — so only
// the EVAL (key building), the UNIQUE probe (self-exclusion by PK not rowid), and the read
// fetch-by-PK are WR-specific, and all of them go through the ONE [`WrIndexKey`] layout so
// write, read, and probe cannot disagree.

/// Build ONE WITHOUT ROWID index entry key (the EVAL phase): the indexed columns' values,
/// then the trailing PRIMARY KEY columns' values (fileformat2 §2.5.1). NO rowid is
/// appended — a WR table has none; the trailing PK keeps entries distinct AND identifies
/// the row to fetch. `row` is the width-`N` schema-order logical row `[c0..c_{N-1}]` with
/// every generated column already computed. All key columns are ordinary columns (an
/// expression index on a WR table is rejected at plan build).
///
/// Returns `None` when `plan` is a PARTIAL index whose predicate does not admit `row`
/// ([`index_admits_row`]) — the WR analogue of [`index_key_values`]'s gate. `ctx` is needed
/// only to evaluate that predicate (a WR partial predicate references the table's own columns,
/// which live in `row`; a WR table has no rowid register, so the predicate never reads past
/// `row`). A full WR index never touches `ctx`.
pub(crate) fn wr_index_key_values(
    plan: &IndexPlan,
    row: &[Value],
    ctx: &mut EvalCtx<'_>,
) -> Result<Option<Vec<Value>>> {
    if !index_admits_row(plan, row, ctx)? {
        return Ok(None);
    }
    let wr = plan.wr.as_ref().ok_or_else(|| {
        Error::format("wr_index_key_values called on a non-WITHOUT ROWID index plan (engine bug)")
    })?;
    let mut key = Vec::with_capacity(plan.key.len() + wr.trailing.len());
    for part in &plan.key {
        match part {
            IndexKeyPart::Col(p) => key.push(wr_col(row, *p)?),
            IndexKeyPart::Expr(_) => {
                return Err(Error::format(
                    "WR index key: unexpected expression key part (expression indexes on \
                     WITHOUT ROWID tables are rejected at plan build) (engine bug)",
                ))
            }
        }
    }
    for &p in &wr.trailing {
        key.push(wr_col(row, p)?);
    }
    Ok(Some(key))
}

/// Clone one column value from a width-`N` schema row, bounds-guarded so a wrong-width row
/// yields a clean engine-bug error rather than an index panic mid-statement.
fn wr_col(row: &[Value], p: usize) -> Result<Value> {
    row.get(p).cloned().ok_or_else(|| {
        Error::format(format!(
            "WR index key: column position {p} out of range for a width-{} row (engine bug)",
            row.len()
        ))
    })
}

/// Compute the WR key record for EVERY plan over `row` (the EVAL phase for a whole row),
/// 1:1 with `plans` BY CONSTRUCTION so the shared WRITE phase can pair a plan with its key
/// positionally. The WR counterpart of [`index_keys_for_plans`], including its `None`-per-
/// partial-index gating (`ctx` evaluates a partial predicate; a full index never uses it).
pub(crate) fn wr_index_keys_for_plans(
    plans: &[IndexPlan],
    row: &[Value],
    ctx: &mut EvalCtx<'_>,
) -> Result<Vec<Option<Vec<Value>>>> {
    let mut keys = Vec::with_capacity(plans.len());
    for ip in plans {
        keys.push(wr_index_key_values(ip, row, ctx)?);
    }
    Ok(keys)
}

/// Recover the full PRIMARY KEY (in PK order) from a DECODED WITHOUT ROWID index entry via
/// the plan's `pk_entry_positions`. `None` when the entry is too short to carry every mapped
/// position (a malformed/foreign entry) — the caller then treats it as no match rather than
/// fabricating a PK.
fn wr_reconstruct_pk(wr: &WrIndexKey, entry: &[Value]) -> Option<Vec<Value>> {
    let mut pk = Vec::with_capacity(wr.pk_entry_positions.len());
    for &pos in &wr.pk_entry_positions {
        pk.push(entry.get(pos)?.clone());
    }
    Some(pk)
}

/// Recover the trailing PRIMARY KEY from a decoded WR index entry for the READ path
/// (fetch-by-PK) — the public wrapper over [`wr_reconstruct_pk`] that errors (rather than
/// returning `None`) on a too-short entry, since the read path has a real entry in hand and
/// a short one is corruption to surface, not a silent skip.
pub(crate) fn wr_entry_pk(plan: &IndexPlan, entry: &[Value]) -> Result<Vec<Value>> {
    let wr = plan.wr.as_ref().ok_or_else(|| {
        Error::format("wr_entry_pk called on a non-WITHOUT ROWID index plan (engine bug)")
    })?;
    wr_reconstruct_pk(wr, entry).ok_or_else(|| {
        Error::format(format!(
            "WITHOUT ROWID index entry too short to recover its {}-column PRIMARY KEY (corrupt index)",
            wr.pk_entry_positions.len()
        ))
    })
}

/// Whether `found` is the row identified by `self_pk` (a per-column BINARY compare). Used
/// to exclude a row's OWN entry from a UNIQUE probe. `self_pk == None` excludes nothing (a
/// fresh INSERT has no entry yet, so any equal-prefix entry is genuinely another row).
fn pk_matches(self_pk: Option<&[Value]>, found: &[Value]) -> bool {
    match self_pk {
        None => false,
        Some(s) => {
            s.len() == found.len()
                && s.iter().zip(found).all(|(a, b)| {
                    compare_values(a, b, Collation::Binary) == Ordering::Equal
                })
        }
    }
}

/// The full PRIMARY KEY of an existing UNIQUE WITHOUT ROWID index entry whose indexed
/// columns equal this row's (at a PK other than `self_pk`), or `None` if there is no
/// conflict. The WR analogue of [`unique_conflict_rowid`]: UNIQUE is enforced over the
/// INDEXED-COLUMN PREFIX (`key[..k]`, `k = ip.key.len()`); a duplicate of those columns is
/// a violation, and the trailing PK only distinguishes the physical entries. NULLs are
/// distinct (lang_createindex.html §1.1), so a key with any NULL indexed column never
/// conflicts.
///
/// `key` is the PRECOMPUTED WR entry `[indexed cols.., trailing PK..]` (from
/// [`wr_index_key_values`]); only its `[..k]` prefix is probed. `self_pk` is the row's own
/// PRIMARY KEY to exclude: INSERT passes `None` (its entry is not yet written), UPDATE
/// passes the OLD PK so the row's still-present entry is not mistaken for a conflict. The
/// returned PK is reconstructed from the conflicting entry via the shared
/// `pk_entry_positions`, so `INSERT/UPDATE OR REPLACE` can delete that victim.
pub(crate) fn wr_unique_conflict(
    pager: &dyn Pager,
    ip: &IndexPlan,
    key: &[Value],
    self_pk: Option<&[Value]>,
    enc: TextEncoding,
) -> Result<Option<Vec<Value>>> {
    let wr = ip.wr.as_ref().ok_or_else(|| {
        Error::format("wr_unique_conflict called on a non-WITHOUT ROWID index plan (engine bug)")
    })?;
    let k = ip.key.len();
    if key.len() < k {
        return Err(Error::format(format!(
            "WR unique probe: key width {} < indexed-column count {k} on {} (engine bug)",
            key.len(),
            ip.detail
        )));
    }
    let cols = &key[..k];
    for v in cols {
        if v.is_null() {
            return Ok(None); // A NULL makes the key distinct from every other.
        }
    }
    // Mirrors `unique_conflict_rowid`: entries equal to the indexed prefix under the index's
    // key order are adjacent, so `seek_ge(prefix)` lands on the first; a UNIQUE index holds
    // at most one entry per indexed prefix, so that first landing is the only candidate.
    let prefix = encode_record_enc(cols, enc);
    let mut cur = IndexCursor::open_with_order(pager, ip.root, ip.order.clone())?;
    if !cur.seek_ge(&prefix)? {
        return Ok(None);
    }
    let decoded = decode_record_enc(&cur.key()?, enc);
    if decoded.len() < k {
        return Ok(None); // A malformed/short entry: don't manufacture a false conflict.
    }
    for (i, want) in cols.iter().enumerate() {
        if compare_values(&decoded[i], want, ip.collations[i]) != Ordering::Equal {
            return Ok(None); // The landed entry has a different key prefix.
        }
    }
    Ok(match wr_reconstruct_pk(wr, &decoded) {
        Some(found_pk) if !pk_matches(self_pk, &found_pk) => Some(found_pk),
        _ => None,
    })
}

/// Delete a WITHOUT ROWID row identified by its full PRIMARY KEY and EVERY one of its
/// secondary-index entries — the `OR REPLACE` victim cleanup shared by the WR INSERT and
/// UPDATE paths, so the two cannot diverge in what they remove. Idempotent: a PK already
/// gone is a no-op `Ok` (matching `index_delete`'s tolerance of an absent key).
///
/// The victim's index keys are recomputed from its DECODED stored row (with any VIRTUAL
/// generated column filled), so an index keyed on a generated column deletes the RIGHT
/// entry. `plans` are the table's WR index plans; `gprograms` its generated-column programs
/// (empty for a plain table — then no compute runs). Deletes the secondary entries FIRST,
/// then the PK-b-tree row, so no intermediate state leaves an index entry pointing at a
/// removed row (the rowid delete uses the same order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn wr_delete_victim_by_pk(
    pager: &mut dyn Pager,
    layout: &WrLayout,
    def: &TableDef,
    root: PageId,
    plans: &[IndexPlan],
    gprograms: &[GeneratedProgram],
    catalog: &dyn Catalog,
    plan: &Plan,
    db: DbIndex,
    victim_pk: &[Value],
    enc: TextEncoding,
    rt: &mut Runtime,
) -> Result<()> {
    let key_bytes = match layout.pk_conflict_key(&*pager, root, victim_pk, enc)? {
        None => return Ok(()), // Already gone — nothing to remove.
        Some(k) => k,
    };
    let mut scratch = Vec::new();
    let mut row = layout.decode_row_enc(&key_bytes, def, &mut scratch, enc);
    if !gprograms.is_empty() {
        // Fill VIRTUAL generated columns so an index keyed on one computes the stored value.
        let env = Env { catalog, pagers: Pagers::One { db, pager: &*pager }, plan };
        compute_generated(gprograms, true, &mut row, env, rt)?;
    }
    // Recompute this victim's index keys, gating each PARTIAL index on its predicate over the
    // victim row (a partial index holds the victim only when its predicate is true — a `None`
    // key is skipped by `delete_index_entries`). The eval ctx borrows pager/catalog/plan SHARED
    // and `rt` MUT via a reborrow, scoped so those borrows end before the mutable writes below.
    let keys = {
        let env = Env { catalog, pagers: Pagers::One { db, pager: &*pager }, plan };
        let mut ctx = EvalCtx { rt: &mut *rt, env, outer: &row };
        wr_index_keys_for_plans(plans, &row, &mut ctx)?
    };
    // FOREIGN KEY (parent side, ON DELETE): this victim is being removed to satisfy an
    // `OR REPLACE` (INSERT or UPDATE), which — like a standalone DELETE (`ops/delete.rs`) —
    // must fire every incoming FK's ON DELETE action on the row BEFORE removing it: with
    // `PRAGMA foreign_keys` ON a surviving NO ACTION/RESTRICT child aborts, CASCADE deletes the
    // referencing children, SET NULL/SET DEFAULT rewrite them. Run BEFORE the entry/record
    // removal below so children are still located by this victim's OLD key (`row`, its width-N
    // logical values). The parent side only reads `row[i]` for FK-referenced columns, so it is
    // WR-safe (a WR parent's b-tree is untouched here); a WR *child* fails closed inside the
    // helper rather than orphaning it. A no-op when the pragma is OFF (the default), so a
    // REPLACE with FK off is byte-for-byte the removal below.
    foreign_key::enforce_parent_delete(def, &row, catalog, db, &mut *pager, rt)?;
    delete_index_entries(&mut *pager, plans, &keys, enc)?;
    index_delete_with_order(&mut *pager, root, &key_bytes, layout.order())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use minisqlite_btree::{create_index_btree, init_database};
    use minisqlite_catalog::{Catalog, ColumnDef, KeyColumn, SchemaCatalog};
    use minisqlite_expr::ArithOp;
    use minisqlite_pager::MemPager;
    use minisqlite_plan::{Plan, PlanNode};
    use minisqlite_sql::{parse, Expr, Literal, Statement};

    use crate::env::Env;
    use crate::runtime::Runtime;

    // `Value` derives only Debug + Clone (no PartialEq), so assertions match variants
    // by hand rather than `assert_eq!` on the values.

    /// Compute the key records for `plans` over `frame` through a minimal in-memory
    /// [`EvalCtx`], returning the RAW `Option` per plan (so a partial-index test can see the
    /// `None` a non-admitted row produces). The catalog/pager/plan/runtime exist only to
    /// satisfy the ctx shape the eval phase requires; an expression key or a partial predicate
    /// really evaluates against `frame` through this ctx.
    fn eval_keys_opt(plans: &[IndexPlan], frame: &[Value], rowid: i64) -> Vec<Option<Vec<Value>>> {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let cat = SchemaCatalog::new();
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let mut rt = Runtime::new();
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: minisqlite_types::DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let mut ctx = EvalCtx { rt: &mut rt, env, outer: frame };
        index_keys_for_plans(plans, frame, rowid, &mut ctx).unwrap()
    }

    /// [`eval_keys_opt`] for FULL (non-partial) Col-only plans, unwrapping each `Some` so the
    /// key-content assertions read `keys[i]` directly (a `None` would be an engine bug here and
    /// panics).
    fn eval_keys(plans: &[IndexPlan], frame: &[Value], rowid: i64) -> Vec<Vec<Value>> {
        eval_keys_opt(plans, frame, rowid)
            .into_iter()
            .map(|k| k.expect("a full (non-partial) plan always produces a key"))
            .collect()
    }

    /// A minimal Col-only plan over the given key positions (root/detail are irrelevant to
    /// key building, which never writes or names a violation).
    fn col_plan(positions: &[usize]) -> IndexPlan {
        IndexPlan {
            root: 0,
            key: positions.iter().map(|&p| IndexKeyPart::Col(p)).collect(),
            collations: positions.iter().map(|_| Collation::Binary).collect(),
            order: KeyOrder::default(),
            unique: false,
            detail: String::new(),
            wr: None,
            partial_predicate: None,
        }
    }

    #[test]
    fn index_key_values_selects_columns_then_appends_rowid() {
        // frame = [a, b, c]; an index on (c, a) is Col(2), Col(0), so the key record is
        // [c, a, rowid] = [Integer(30), Integer(10), Integer(42)].
        let frame = vec![Value::Integer(10), Value::Text("b".into()), Value::Integer(30)];
        let plans = vec![col_plan(&[2, 0])];
        let keys = eval_keys(&plans, &frame, 42);
        assert_eq!(keys.len(), 1, "one key per plan");
        let key = &keys[0];
        assert_eq!(key.len(), 3);
        assert!(matches!(key[0], Value::Integer(30)), "first key col is position 2");
        assert!(matches!(key[1], Value::Integer(10)), "second key col is position 0");
        assert!(matches!(key[2], Value::Integer(42)), "trailing value is the rowid");
    }

    #[test]
    fn index_key_values_with_no_columns_is_just_the_rowid() {
        // A degenerate (empty key) plan still appends the rowid, so the key is never
        // empty — the trailing rowid keeps entries distinct per row.
        let plans = vec![col_plan(&[])];
        let keys = eval_keys(&plans, &[Value::Integer(1)], 7);
        let key = &keys[0];
        assert_eq!(key.len(), 1);
        assert!(matches!(key[0], Value::Integer(7)));
    }

    #[test]
    fn index_key_values_passes_non_integer_values_through_unchanged() {
        // The builder is variant-agnostic for a Col part (it just clones), so Text/Null/Real
        // selected columns land in the key verbatim, in key-part order, with rowid appended.
        let frame = vec![Value::Null, Value::Text("k".into()), Value::Real(2.5)];
        let plans = vec![col_plan(&[1, 0, 2])];
        let keys = eval_keys(&plans, &frame, 9);
        let key = &keys[0];
        assert_eq!(key.len(), 4);
        assert!(matches!(&key[0], Value::Text(s) if s == "k"), "position 1 (Text) first");
        assert!(matches!(key[1], Value::Null), "position 0 (Null) second");
        assert!(matches!(key[2], Value::Real(r) if r == 2.5), "position 2 (Real) third");
        assert!(matches!(key[3], Value::Integer(9)), "rowid appended last");
    }

    #[test]
    fn index_key_values_evaluates_expr_parts_against_the_frame() {
        // The HEADLINE capability: an `IndexKeyPart::Expr` is evaluated (not cloned) against
        // the `[c0..c_{N-1}, rowid]` frame. Pins every Expr shape the machinery must support:
        //  - an arithmetic key `a + b` (the `CREATE INDEX i ON t(a+b)` case),
        //  - a bare column via `Column(i)`,
        //  - the ROWID register read as `Column(N)` (an expression key referencing the rowid),
        //  - a constant `Literal`,
        // interleaved with an ordinary `Col` part. This is exactly the arm an earlier
        // bug broke (`Expr(_) => Value::Null`): with that break, every Expr slot below
        // would be NULL and these assertions fail — so this test is the red the Col-only
        // suite could not produce.
        //
        // N = 2, frame = [a=10, b=20, rowid=42] (width N+1 = 3). Key parts:
        //   Col(0)                         -> 10
        //   Expr(a + b) = Column0 + Column1 -> 30
        //   Expr(Column(2)) = rowid reg    -> 42
        //   Expr(Literal(7))               -> 7
        // then the trailing rowid append   -> 42.
        let frame = vec![Value::Integer(10), Value::Integer(20), Value::Integer(42)];
        let plan = IndexPlan {
            root: 0,
            key: vec![
                IndexKeyPart::Col(0),
                IndexKeyPart::Expr(EvalExpr::Arith {
                    op: ArithOp::Add,
                    left: Box::new(EvalExpr::Column(0)),
                    right: Box::new(EvalExpr::Column(1)),
                }),
                IndexKeyPart::Expr(EvalExpr::Column(2)),
                IndexKeyPart::Expr(EvalExpr::Literal(Value::Integer(7))),
            ],
            collations: vec![Collation::Binary; 4],
            order: KeyOrder::default(),
            unique: false,
            detail: String::new(),
            wr: None,
            partial_predicate: None,
        };
        let keys = eval_keys(&[plan], &frame, 42);
        assert_eq!(keys.len(), 1, "one key per plan");
        let key = &keys[0];
        assert_eq!(key.len(), 5, "4 key parts + trailing rowid");
        assert!(matches!(key[0], Value::Integer(10)), "Col(0) selects frame[0]");
        assert!(matches!(key[1], Value::Integer(30)), "Expr(a+b) evaluates to 30, not cloned/NULL");
        assert!(matches!(key[2], Value::Integer(42)), "Expr(Column(N)) reads the rowid register");
        assert!(matches!(key[3], Value::Integer(7)), "Expr(Literal(7)) is the constant");
        assert!(matches!(key[4], Value::Integer(42)), "trailing rowid appended last");
    }

    #[test]
    fn partial_predicate_gates_index_membership() {
        // A partial index's membership is decided by the truthiness of its WHERE predicate over
        // the row frame (`partialindex.html` §2). Here the predicate is column 1's own value
        // (`... WHERE b`): a row is IN the index iff `frame[1]` is truthy — Some(key) when true,
        // None (skipped: no entry, no probe, no delete) when 0 OR NULL. This is the ONE gate the
        // whole partial-index fix rests on, tested at the eval seam every operator shares.
        let mk = || IndexPlan {
            root: 0,
            key: vec![IndexKeyPart::Col(0)],
            collations: vec![Collation::Binary],
            order: KeyOrder::default(),
            unique: true,
            detail: "t.a".into(),
            wr: None,
            partial_predicate: Some(EvalExpr::Column(1)),
        };
        // frame = [a, b, rowid]; b = 5 is truthy -> the row is in the index.
        let keys = eval_keys_opt(&[mk()], &[Value::Integer(1), Value::Integer(5), Value::Integer(9)], 9);
        assert_eq!(keys.len(), 1, "one Option slot per plan (1:1 preserved)");
        match &keys[0] {
            Some(k) => {
                assert!(matches!(k[0], Value::Integer(1)), "key col a");
                assert!(matches!(k[1], Value::Integer(9)), "trailing rowid");
            }
            None => panic!("b = 5 is truthy: the row must be IN the partial index"),
        }
        // b = 0 is FALSE -> omitted.
        let keys = eval_keys_opt(&[mk()], &[Value::Integer(1), Value::Integer(0), Value::Integer(9)], 9);
        assert!(keys[0].is_none(), "b = 0 is false: the row must be OMITTED from the partial index");
        // b = NULL -> omitted (a NULL predicate omits, not just a false one).
        let keys = eval_keys_opt(&[mk()], &[Value::Integer(1), Value::Null, Value::Integer(9)], 9);
        assert!(keys[0].is_none(), "b NULL: the row must be OMITTED (partialindex.html §2)");
    }

    /// Count the entries in an index b-tree by walking the cursor from its first key.
    fn count_index_entries(pager: &dyn Pager, root: PageId) -> usize {
        let mut cur = IndexCursor::open(pager, root).unwrap();
        if !cur.first().unwrap() {
            return 0;
        }
        let mut n = 1;
        while cur.next().unwrap() {
            n += 1;
        }
        n
    }

    #[test]
    fn insert_then_delete_index_entries_round_trips_and_reinsert_is_idempotent() {
        // A pager-backed round-trip pinning the two behaviors this module adds on the
        // delete side: (1) `delete_index_entries` removes exactly what
        // `insert_index_entries` wrote (they share the precomputed-key encoding), and
        // (2) deleting an already-absent entry is idempotent — `index_delete`'s `false`
        // is swallowed, so it returns `Ok`, not `Err`. This guards against a future edit
        // turning the ignored bool into an error, which would break UPDATE/DELETE
        // re-application over an entry that legitimately isn't there.
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        pager.begin().unwrap();
        let root = create_index_btree(&mut pager).unwrap();

        // One non-unique index over column 0; two rows with distinct keys and rowids. The
        // keys are computed in the EVAL phase, then handed to the WRITE phase precomputed.
        let plans = vec![IndexPlan {
            root,
            key: vec![IndexKeyPart::Col(0)],
            collations: vec![Collation::Binary],
            order: KeyOrder::default(),
            unique: false,
            detail: "t.a".into(),
            wr: None,
            partial_predicate: None,
        }];
        // The writers take `Option`-typed keys (a `None` = a row not in a partial index).
        // These plans are full, so wrap each computed key as `Some`.
        let keys1: Vec<Option<Vec<Value>>> =
            eval_keys(&plans, &[Value::Integer(10)], 1).into_iter().map(Some).collect();
        let keys2: Vec<Option<Vec<Value>>> =
            eval_keys(&plans, &[Value::Integer(20)], 2).into_iter().map(Some).collect();

        insert_index_entries(&mut pager, &plans, &keys1, TextEncoding::Utf8).unwrap();
        insert_index_entries(&mut pager, &plans, &keys2, TextEncoding::Utf8).unwrap();
        assert_eq!(count_index_entries(&pager, root), 2, "both entries present after insert");

        // Deleting both removes exactly them — the index is empty again.
        delete_index_entries(&mut pager, &plans, &keys1, TextEncoding::Utf8).unwrap();
        delete_index_entries(&mut pager, &plans, &keys2, TextEncoding::Utf8).unwrap();
        assert_eq!(count_index_entries(&pager, root), 0, "index empty after deleting both");

        // Re-deleting an already-absent entry must stay Ok (idempotent), never Err.
        delete_index_entries(&mut pager, &plans, &keys1, TextEncoding::Utf8)
            .expect("deleting an absent index entry is idempotent, not an error");

        pager.commit().unwrap();
    }

    #[test]
    fn entries_writers_reject_a_key_count_mismatch() {
        // The fail-loud guard: a keys slice that is not 1:1 with plans is an engine bug
        // that would leave some index unwritten (or delete the wrong entry). Both writers
        // must return Err rather than silently `zip`-truncating.
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        pager.begin().unwrap();
        let root = create_index_btree(&mut pager).unwrap();
        let plans = vec![IndexPlan {
            root,
            key: vec![IndexKeyPart::Col(0)],
            collations: vec![Collation::Binary],
            order: KeyOrder::default(),
            unique: false,
            detail: "t.a".into(),
            wr: None,
            partial_predicate: None,
        }];

        // Zero keys for one plan: a mismatch (too FEW keys).
        let none: &[Option<Vec<Value>>] = &[];
        assert!(insert_index_entries(&mut pager, &plans, none, TextEncoding::Utf8).is_err());
        assert!(delete_index_entries(&mut pager, &plans, none, TextEncoding::Utf8).is_err());
        // Two keys for one plan: a mismatch the other way (too MANY keys) — also rejected, so
        // a stray extra key can't slip a phantom entry past the strict 1:1 contract.
        let extra = vec![Some(vec![Value::Integer(1)]), Some(vec![Value::Integer(2)])];
        assert!(insert_index_entries(&mut pager, &plans, &extra, TextEncoding::Utf8).is_err());
        assert!(delete_index_entries(&mut pager, &plans, &extra, TextEncoding::Utf8).is_err());
        pager.commit().unwrap();
    }

    #[test]
    fn col_positions_is_some_for_col_only_and_none_with_an_expr() {
        // A Col-only plan reports its positions (used by UPSERT conflict-target matching);
        // a plan with any Expr part is not a column set, so it reports None.
        let col_only = col_plan(&[3, 1]);
        assert_eq!(col_only.col_positions(), Some(vec![3, 1]));

        let with_expr = IndexPlan {
            root: 0,
            key: vec![IndexKeyPart::Col(0), IndexKeyPart::Expr(EvalExpr::Literal(Value::Integer(1)))],
            collations: vec![Collation::Binary; 2],
            order: KeyOrder::default(),
            unique: false,
            detail: String::new(),
            wr: None,
            partial_predicate: None,
        };
        assert_eq!(with_expr.col_positions(), None);
    }

    // ---- build_index_plan: key-part resolution + the honest 2b loud guard ----------------

    /// A minimal rowid `TableDef` over the given ordinary column names (other fields
    /// defaulted); `build_index_plan` only reads `name` + `columns` for Col resolution.
    fn tdef(cols: &[&str]) -> TableDef {
        TableDef {
            name: "t".into(),
            columns: cols
                .iter()
                .map(|c| ColumnDef {
                    name: (*c).into(),
                    declared_type: None,
                    not_null: false,
                    primary_key: false,
                    unique: false,
                    collation: None,
                    default: None,
                    default_value: None,
                    generated: None,
                })
                .collect(),
            root_page: 2,
            without_rowid: false,
            rowid_alias: None,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
        }
    }

    /// A minimal `IndexDef` over `cols` with the given per-column `key_exprs` (`Some(_)`
    /// marks an expression key column; its parallel `columns` entry is then the empty-string
    /// sentinel an expression key carries). `key_columns` is filled parallel but is not read
    /// by `build_index_plan`.
    fn idef(cols: &[&str], key_exprs: Vec<Option<Expr>>) -> IndexDef {
        IndexDef {
            name: "idx".into(),
            table: "t".into(),
            columns: cols.iter().map(|c| (*c).to_string()).collect(),
            key_columns: cols.iter().map(|_| KeyColumn { collation: None, descending: false }).collect(),
            key_exprs,
            root_page: 3,
            unique: false,
            partial: false,
            partial_predicate: None,
        }
    }

    #[test]
    fn build_index_plan_col_only_resolves_positions_and_detail() {
        // The ordinary-index path: named columns, empty `compiled` and all-`None` `key_exprs`
        // -> every part is `Col`, resolved case-insensitively to a table-column position, and
        // the detail is `table.col, ...`.
        let def = tdef(&["a", "b", "c"]);
        let idx = idef(&["C", "a"], vec![None, None]); // note the case-insensitive `C`
        let plan = build_index_plan(&def, &idx, &[], None).unwrap();
        assert_eq!(plan.col_positions(), Some(vec![2, 0]), "C->2 (case-insensitive), a->0");
        assert_eq!(plan.detail, "t.C, t.a");
    }

    #[test]
    fn build_index_plan_uses_a_supplied_compiled_expr() {
        // A supplied `compiled[i] == Some(e)` -> `IndexKeyPart::Expr(e)`, regardless of the
        // `columns[i]` sentinel; an unmarked column stays an ordinary `Col`. The Expr then
        // really evaluates when the key is built.
        let def = tdef(&["a", "b"]);
        let idx = idef(&["", "b"], vec![Some(Expr::Literal(Literal::Null)), None]);
        let compiled = vec![Some(EvalExpr::Literal(Value::Integer(7))), None];
        let plan = build_index_plan(&def, &idx, &compiled, None).unwrap();
        assert!(matches!(plan.key[0], IndexKeyPart::Expr(_)), "compiled expr -> Expr part");
        assert!(matches!(plan.key[1], IndexKeyPart::Col(1)), "ordinary column -> Col(1)");
        assert_eq!(plan.col_positions(), None, "an Expr part means not a column-only index");
        // frame = [a=1, b=2, rowid-slot=0] (width N+1); key = [Expr->7, Col(1)->2, rowid=9].
        let keys = eval_keys(&[plan], &[Value::Integer(1), Value::Integer(2), Value::Integer(0)], 9);
        assert!(matches!(keys[0][0], Value::Integer(7)), "Expr evaluates to the supplied literal");
        assert!(matches!(keys[0][1], Value::Integer(2)), "Col(1) selects frame[1]");
        assert!(matches!(keys[0][2], Value::Integer(9)), "rowid appended last");
    }

    #[test]
    fn build_index_plan_detail_names_the_index_for_an_expression_key() {
        // SQLite's UNIQUE-violation detail names the key COLUMNS for an ordinary index but
        // the INDEX itself for an index with any expression key column (there is no column
        // name to print, and the empty-string sentinel would otherwise render as a bare
        // `t.`). Pin both forms.
        let def = tdef(&["a", "b"]);
        let expr_idx = idef(&["", "b"], vec![Some(Expr::Literal(Literal::Null)), None]);
        let compiled = vec![Some(EvalExpr::Literal(Value::Integer(7))), None];
        let expr_plan = build_index_plan(&def, &expr_idx, &compiled, None).unwrap();
        assert_eq!(expr_plan.detail, "index 'idx'", "an expression index names the index");

        let ord_idx = idef(&["a", "b"], vec![None, None]);
        let ord_plan = build_index_plan(&def, &ord_idx, &[], None).unwrap();
        assert_eq!(ord_plan.detail, "t.a, t.b", "an ordinary index still names its columns");
    }

    #[test]
    fn build_index_plan_expression_key_without_compiled_expr_is_a_loud_error() {
        // The honest guard: an EXPRESSION key column (`key_exprs[i] == Some`) with NO
        // compiled expr supplied (empty `compiled`) cannot build a key, so it fails LOUD
        // rather than silently writing a wrong/empty one. This defends any caller that forgets
        // to compile an expression key; the FK cascade used to hit it before that path was
        // wired, and now supplies compiled exprs (see `foreign_key.rs`).
        let def = tdef(&["a", "b"]);
        let idx = idef(&["", "b"], vec![Some(Expr::Literal(Literal::Null)), None]);
        // `IndexPlan` intentionally has no `Debug`, so match rather than `unwrap_err()`.
        let err = match build_index_plan(&def, &idx, &[], None) {
            Ok(_) => panic!("expected the loud 2b error for an uncompiled expression key"),
            Err(e) => e,
        };
        assert!(
            matches!(&err, Error::Format(m) if m.contains("key expressions were not compiled")),
            "expected the loud 2b format error, got {err:?}"
        );
    }

    // ---- WITHOUT ROWID secondary-index key format (fileformat2 §2.5.1) --------------------

    /// Create a WR table + secondary index through the real catalog and return the built
    /// [`IndexPlan`] for the index (so its `wr` layout is derived exactly as production does).
    fn wr_plan(create_table: &str, create_index_sql: &str, table: &str, index: &str) -> IndexPlan {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let mut cat = SchemaCatalog::new();
        for sql in [create_table, create_index_sql] {
            let ast = parse(sql).unwrap();
            match &ast.statements[0] {
                Statement::CreateTable(s) => cat.create_table(&mut pager, s, sql).unwrap(),
                Statement::CreateIndex(s) => cat.create_index(&mut pager, s, sql).unwrap(),
                _ => panic!("expected CREATE TABLE/INDEX: {sql}"),
            }
        }
        let def = cat.table(table).unwrap().unwrap();
        let idx = cat.index(index).unwrap().unwrap();
        build_index_plan(def, idx, &[], None).unwrap()
    }

    /// A width-5 schema-order row `[a,b,c,d,e]` of distinguishable text values, so a produced
    /// key can be read back column-by-column.
    fn abcde() -> Vec<Value> {
        ["a", "b", "c", "d", "e"].iter().map(|s| Value::Text((*s).into())).collect()
    }

    fn txt(v: &Value) -> &str {
        match v {
            Value::Text(s) => s.as_str(),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    /// Build ONE WR index key through a minimal [`EvalCtx`], unwrapping the `Option` these
    /// full (non-partial) WR plans always produce. The ctx is never actually read (a full
    /// index has no predicate to evaluate); it only satisfies [`wr_index_key_values`]'s shape.
    fn wr_eval_key(plan: &IndexPlan, row: &[Value]) -> Vec<Value> {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let cat = SchemaCatalog::new();
        let eplan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let mut rt = Runtime::new();
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: minisqlite_types::DbIndex::MAIN, pager: &pager },
            plan: &eplan,
        };
        let mut ctx = EvalCtx { rt: &mut rt, env, outer: row };
        wr_index_key_values(plan, row, &mut ctx)
            .unwrap()
            .expect("a full (non-partial) WR plan always produces a key")
    }

    #[test]
    fn wr_secondary_key_suppresses_a_pk_column_already_indexed() {
        // fileformat2 §2.5.1, example ex25ce: PRIMARY KEY(d,c,a), INDEX(c,e). Each entry is
        // `c, e, d, a` — the trailing PK is d,c,a but `c` is suppressed because it already
        // appears among the indexed columns with a matching (BINARY) collation.
        let plan = wr_plan(
            "CREATE TABLE ex25(a,b,c,d,e,PRIMARY KEY(d,c,a)) WITHOUT ROWID",
            "CREATE INDEX ex25ce ON ex25(c,e)",
            "ex25",
            "ex25ce",
        );
        let key = wr_eval_key(&plan, &abcde());
        let got: Vec<&str> = key.iter().map(txt).collect();
        assert_eq!(got, ["c", "e", "d", "a"], "indexed (c,e) then trailing PK (d,a); c suppressed");
        // The PK-recovery map pulls the full PRIMARY KEY (d,c,a) back out of that entry.
        let wr = plan.wr.as_ref().unwrap();
        let pk = wr_reconstruct_pk(wr, &key).unwrap();
        assert_eq!(pk.iter().map(txt).collect::<Vec<_>>(), ["d", "c", "a"], "PK recovered in PK order");
    }

    #[test]
    fn wr_secondary_key_indexed_covers_whole_pk_appends_nothing() {
        // §2.5.1 "extreme case", ex25acde: INDEX(a,c,d,e) covers all of PRIMARY KEY(d,c,a),
        // so the entry is exactly the indexed columns with an EMPTY suffix.
        let plan = wr_plan(
            "CREATE TABLE ex25(a,b,c,d,e,PRIMARY KEY(d,c,a)) WITHOUT ROWID",
            "CREATE INDEX ex25acde ON ex25(a,c,d,e)",
            "ex25",
            "ex25acde",
        );
        let key = wr_eval_key(&plan, &abcde());
        assert_eq!(key.iter().map(txt).collect::<Vec<_>>(), ["a", "c", "d", "e"], "only indexed cols");
        assert!(plan.wr.as_ref().unwrap().trailing.is_empty(), "no trailing PK column");
        let pk = wr_reconstruct_pk(plan.wr.as_ref().unwrap(), &key).unwrap();
        assert_eq!(pk.iter().map(txt).collect::<Vec<_>>(), ["d", "c", "a"], "PK from the indexed cols");
    }

    #[test]
    fn wr_secondary_key_repeats_a_pk_column_indexed_under_a_different_collation() {
        // §2.5.1, ex25ae: INDEX(a COLLATE nocase, e) over PRIMARY KEY(d,c,a). The entry is
        // `a, e, d, c, a` — `a` is REPEATED because its indexed occurrence is NOCASE while the
        // PK's `a` is BINARY (a collation MISMATCH does not suppress).
        let plan = wr_plan(
            "CREATE TABLE ex25(a,b,c,d,e,PRIMARY KEY(d,c,a)) WITHOUT ROWID",
            "CREATE INDEX ex25ae ON ex25(a COLLATE nocase, e)",
            "ex25",
            "ex25ae",
        );
        let key = wr_eval_key(&plan, &abcde());
        assert_eq!(
            key.iter().map(txt).collect::<Vec<_>>(),
            ["a", "e", "d", "c", "a"],
            "a repeated (NOCASE index vs BINARY PK); trailing PK d,c,a all kept"
        );
    }
}
