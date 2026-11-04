//! Opportunistic `ORDER BY` -> scan-order optimization: when a single-base-table
//! `SELECT ... ORDER BY <cols> [LIMIT k]` can be served by walking the rowid or an index
//! b-tree in one direction, [`satisfy_order_by`] rewrites the access leaf and the caller
//! omits the `PlanNode::Sort` — streaming in `O(1)` sort-memory (and early-stopping under
//! a `LIMIT`) instead of materializing + sorting the whole input.
//!
//! It is PURE, OPPORTUNISTIC and FALLBACK-SAFE: it returns `Some(rewritten)` ONLY when
//! the scan order is provably BYTE-IDENTICAL to today's stable full sort, and `None`
//! (keep the Sort, exactly today's behaviour) for everything else. A missed skip is
//! fine; a wrong skip is a correctness bug, so every uncertain case keeps the Sort.
//!
//! # Correctness proof (why a skipped Sort is byte-identical to the full stable Sort)
//!
//! Today's plan is `Sort { keys }` over the access subtree, and that Sort is STABLE
//! ([`minisqlite_exec`]'s `ops::sort`): rows equal on all keys keep their INPUT order,
//! and the Sort's input is the access leaf's natural emission order (ascending rowid for
//! `SeqScan`/`RowidScan`; index-key order for an `IndexScan` `Seek`) threaded through a
//! row-order-preserving `Filter` (the WHERE residual). So the total order the current
//! plan emits is exactly: the `keys`, then the leaf's natural order for ties. We skip
//! the Sort only when the leaf, walked in a single direction, ALREADY emits that exact
//! total order.
//!
//! Load-bearing b-tree facts:
//! * **F1 (Binary).** An index b-tree is ordered under [`Collation::Binary`] only, so an
//!   index serves an `ORDER BY` term only when that term's collation is `Binary`.
//! * **F2 (NULLs first).** NULL keys sort before every value; a forward walk yields
//!   NULLs-then-ascending (= `ASC` default, `NULLS FIRST`), a reverse walk yields
//!   descending-then-NULLs (= `DESC` default, `NULLS LAST`).
//! * **F3 (rowid).** The rowid is a UNIQUE, never-NULL integer key: forward = ascending
//!   rowid, reverse = descending rowid.
//! * **F4 (reverse reverses ties).** A reverse walk steps the WHOLE key backwards
//!   (trailing index columns and the trailing rowid included), so within a group of rows
//!   equal on a PREFIX of the key it emits them in the reverse of the forward tie order.
//!
//! ## Mechanism A — rowid order (ASC and DESC)
//! The sole `ORDER BY` key is the rowid (or its `INTEGER PRIMARY KEY` alias). By F3 the
//! rowid is unique, so no two rows are ever equal on the key: the total order is decided
//! by the key alone, with NO tie-break to preserve. A forward rowid walk emits ascending
//! rowid (= `ORDER BY rowid ASC`); a reverse rowid walk emits descending rowid
//! (= `ORDER BY rowid DESC`). With no ties, F4 is vacuous, so BOTH directions are
//! byte-identical to the stable sort. A WHERE rowid range is kept on the rewritten
//! `RowidScan` (its bounds are direction-agnostic) and the range still emits surviving
//! rowids monotonically.
//!
//! ## Mechanism B — index order (ASC only)
//! The `ORDER BY` keys equal, in order, a contiguous run of an index's key columns —
//! every relied-on column ASCENDING and Binary, the term collation Binary (F1), and the
//! null placement the forward scan's natural one (F2). We serve ONLY ascending (Forward):
//!
//! * **ASC / Forward.** Two sub-cases:
//!   * (i) WHERE already chose `IndexScan Seek` on index `I` (equality-prefix length `p`)
//!     and the `ORDER BY` keys are `I.columns[p .. p+k]`. The Sort's INPUT is already this
//!     index order — `(I.col_p, .., I.col_{m-1}, rowid)`, a subsequence after the Filter —
//!     and the keys are a PREFIX of it. A stable sort by a prefix of an already-sorted
//!     sequence is the IDENTITY, so dropping the Sort changes nothing — for any trailing
//!     columns, any `p`, and any WHERE range on `I.col_p`.
//!   * (ii) WHERE chose a full `SeqScan`, and some non-partial index `I` has EXACTLY the
//!     `ORDER BY` columns as its key columns (`I.columns.len() == k`). The forward index
//!     order is `(keys, rowid)` — the rowid is the only tie-break after the keys — which
//!     equals the current plan's `(keys, SeqScan input = ascending rowid)`. NULL-keyed
//!     rows are present in a full (non-partial) index and sort first, matching `ASC`
//!     default `NULLS FIRST` (F2), rowid-tie-broken on both sides. The `== k` requirement
//!     is load-bearing: an index with EXTRA trailing key columns would tie-break by those
//!     before the rowid, differing from the stable sort — so such an index is declined.
//!
//! * **DESC / Reverse — declined (keep the Sort).** A reverse index walk reverses ties
//!   (F4): within a group of rows equal on the `ORDER BY` keys it emits them in
//!   DESCENDING rowid order, whereas the stable DESC sort keeps them in the input's
//!   ASCENDING rowid order. Whenever the keys have duplicates (the ordinary case for a
//!   secondary index) these differ, so a reverse index scan is NOT byte-identical to the
//!   current plan. (Mechanism A escapes this only because its rowid key is unique.) A
//!   descending index order therefore always keeps the Sort here.
//!
//! Any condition not proven above keeps the Sort.

use minisqlite_catalog::{Catalog, IndexDef, TableDef};
use minisqlite_expr::{EvalExpr, SortKey};
use minisqlite_types::{Collation, DbIndex};

use crate::access::{IndexOp, IndexScan, RowidOp, RowidScan, ScanDirection};
use crate::access_path::{col_reg, key_column_is_binary};
use crate::plan::PlanNode;

/// Try to rewrite `node` (the `Project` over the access subtree, BEFORE the `Sort`) so
/// the access path itself yields the `ORDER BY` order, letting the caller omit the
/// `Sort`. Returns `Some(rewritten_project)` when that is provably byte-identical to the
/// stable full sort, else `None` (the caller keeps the `Sort`).
///
/// The caller MUST have already established the outer eligibility guards: `base_offset ==
/// 0` (so the leaf's columns sit at registers `[0, N]` — see the module doc on why a
/// nonzero base makes the register trace unsound), the query is NOT `DISTINCT`, and it is
/// NOT an aggregate (both interpose / re-order rows between the scan and the Sort). `keys`
/// is the non-empty `ORDER BY` key list.
pub(crate) fn satisfy_order_by(
    catalog: &dyn Catalog,
    node: &PlanNode,
    keys: &[SortKey],
) -> Option<PlanNode> {
    // The pre-Sort node is always a bare `Project` here (DISTINCT is excluded by the
    // caller, and a window query keeps its `Window` under this `Project`, which the leaf
    // descent below then rejects). Anything else: keep the Sort.
    let PlanNode::Project { input, exprs } = node else {
        return None;
    };
    if keys.is_empty() {
        return None;
    }

    // A single, consistent scan direction: all keys ASC -> Forward, all DESC -> Reverse,
    // mixed -> keep the Sort (one walk can't serve both).
    let direction = uniform_direction(keys)?;

    // Every key must sort under Binary (F1) — a NOCASE / RTRIM / explicit-COLLATE term
    // cannot be served by a BINARY-ordered key column — and its NULL placement must match
    // the scan's natural one (F2).
    if !keys.iter().all(|k| k.collation == Collation::Binary) {
        return None;
    }
    if !keys.iter().all(|k| null_order_ok(k, &direction)) {
        return None;
    }

    // Find the single base-table access leaf, descending only row-order-preserving 1:1
    // `Filter` wrappers. A `Join` / `Window` / derived / VALUES leaf is not a single base
    // table and returns `None` here (keep the Sort).
    let leaf = base_table_leaf(input)?;
    let info = leaf_info(leaf)?;
    // A catalog `Err` (or a missing table) DECLINES here (`.ok()...?` -> keep the Sort),
    // rather than propagating like the surrounding planner `?`-sites: this is an
    // opportunistic `Option`-returning optimizer, and this exact table/index was already
    // resolved (any real error already surfaced) when `build_from` built the leaf, so the
    // error arm is effectively unreachable and declining is always a correct fallback. The
    // same reasoning covers the `.ok()` on `index_in`/`indexes_on_in` in the mechanisms.
    let td = catalog.table_in(info.db, info.table).ok().flatten()?;
    // A WITHOUT ROWID table has no integer rowid and stores rows in a PK index b-tree the
    // rowid/index leaves here cannot serve — skip entirely.
    if td.without_rowid {
        return None;
    }
    let n = td.columns.len();
    // Defensive: the leaf's declared width must agree with the catalog. A mismatch means
    // the plan and schema disagree, so decline rather than trust the trace.
    if info.column_count != n {
        return None;
    }

    // Trace each key through the projection to a base-table register (a column `i` -> `i`,
    // the rowid / INTEGER PRIMARY KEY alias -> `n`). Any computed / non-column key -> None.
    let regs = trace_keys(keys, exprs, n)?;

    // Mechanism A (rowid order) first, then Mechanism B (index order). Each returns the
    // rewritten leaf or `None`.
    let new_leaf = mechanism_a_rowid(&regs, n, &direction, leaf)
        .or_else(|| mechanism_b_index(catalog, td, n, &regs, &direction, leaf))?;

    // Rebuild the `Project` over the leaf-replaced subtree; the caller omits the `Sort`.
    Some(PlanNode::Project {
        input: Box::new(replace_leaf(input, new_leaf)),
        exprs: exprs.clone(),
    })
}

/// A single consistent scan direction for the whole key list, or `None` for a MIXED
/// `ASC`/`DESC` order (which one b-tree walk cannot serve).
fn uniform_direction(keys: &[SortKey]) -> Option<ScanDirection> {
    if keys.iter().all(|k| !k.desc) {
        Some(ScanDirection::Forward)
    } else if keys.iter().all(|k| k.desc) {
        Some(ScanDirection::Reverse)
    } else {
        None
    }
}

/// Whether a key's NULL placement matches the scan's natural one (F2). The b-tree puts
/// NULLs first, so a Forward scan is `NULLS FIRST` and a Reverse scan is `NULLS LAST`;
/// an ABSENT `NULLS` clause takes the direction's default (which is exactly that natural
/// placement), and an EXPLICIT clause is servable only when it agrees with the scan.
///
/// COUPLING: the "absent clause -> direction default" rule here MUST stay in agreement
/// with the Sort comparator this rewrite replaces — `minisqlite-exec`'s
/// `ops::sortkey::compare_one`, which resolves an absent clause as `nulls_first = !desc`
/// (Forward/`ASC` -> first, Reverse/`DESC` -> last) — and with the b-tree's physical
/// NULLs-first order. If that default ever changes, this function must change with it; the
/// NULL-bearing tests in `conformance_order_by_index` are the standing guard
/// that the skipped-scan and kept-Sort paths keep placing NULLs identically.
fn null_order_ok(key: &SortKey, dir: &ScanDirection) -> bool {
    match (dir, key.nulls_first) {
        (_, None) => true,
        (ScanDirection::Forward, Some(nulls_first)) => nulls_first,
        (ScanDirection::Reverse, Some(nulls_first)) => !nulls_first,
    }
}

/// The base-table leaf reachable from `node` through only row-order-preserving `Filter`
/// wrappers, or `None` if the path holds any other node (a `Join`, `Window`, derived
/// plan, VALUES, …) — i.e. not a single base-table scan.
fn base_table_leaf(node: &PlanNode) -> Option<&PlanNode> {
    match node {
        PlanNode::Filter { input, .. } => base_table_leaf(input),
        leaf @ (PlanNode::SeqScan(_) | PlanNode::RowidScan(_) | PlanNode::IndexScan(_)) => {
            Some(leaf)
        }
        _ => None,
    }
}

/// The `(table, db, column_count)` of a base-table access leaf.
struct LeafInfo<'a> {
    table: &'a str,
    db: DbIndex,
    column_count: usize,
}

fn leaf_info(leaf: &PlanNode) -> Option<LeafInfo<'_>> {
    match leaf {
        PlanNode::SeqScan(s) => {
            Some(LeafInfo { table: &s.table, db: s.db, column_count: s.column_count })
        }
        PlanNode::RowidScan(s) => {
            Some(LeafInfo { table: &s.table, db: s.db, column_count: s.column_count })
        }
        PlanNode::IndexScan(s) => {
            Some(LeafInfo { table: &s.table, db: s.db, column_count: s.column_count })
        }
        _ => None,
    }
}

/// Trace each `ORDER BY` key to the base-table register it orders by, or `None` if any
/// key is not a bare base-table column / rowid reference. A key is `Column(out)` over the
/// projected row; it qualifies only when `exprs[out]` is itself `Column(reg)` with `reg`
/// in `[0, n]` (a base column `0..n-1`, or the rowid `n`). An `ORDER BY a+b` / function /
/// literal projection is `None` (keep the Sort).
fn trace_keys(keys: &[SortKey], exprs: &[EvalExpr], n: usize) -> Option<Vec<usize>> {
    let mut regs = Vec::with_capacity(keys.len());
    for k in keys {
        let EvalExpr::Column(out_idx) = &k.expr else {
            return None;
        };
        let EvalExpr::Column(reg) = exprs.get(*out_idx)? else {
            return None;
        };
        // `n` is the rowid register; `> n` is a synthetic/hidden column we do not model.
        if *reg > n {
            return None;
        }
        regs.push(*reg);
    }
    Some(regs)
}

/// Mechanism A — rowid order. The sole key is the rowid (register `n`); serve ASC
/// (Forward) and DESC (Reverse). See the module doc: the rowid is unique, so neither
/// direction has ties to preserve.
fn mechanism_a_rowid(
    regs: &[usize],
    n: usize,
    dir: &ScanDirection,
    leaf: &PlanNode,
) -> Option<PlanNode> {
    if regs.len() != 1 || regs[0] != n {
        return None;
    }
    match leaf {
        // A `SeqScan` already yields ascending rowid: keep it for ASC; for DESC swap in a
        // full REVERSE rowid scan (a `SeqScan` has no direction, so descending rowid order
        // is a `RowidScan Range{None, None}` walked in reverse).
        PlanNode::SeqScan(s) => Some(match dir {
            ScanDirection::Forward => PlanNode::SeqScan(s.clone()),
            ScanDirection::Reverse => PlanNode::RowidScan(RowidScan {
                table: s.table.clone(),
                db: s.db,
                column_count: s.column_count,
                op: RowidOp::Range { lo: None, hi: None },
                direction: ScanDirection::Reverse,
            }),
        }),
        // A rowid path (WHERE point-eq or range) already walks the rowid key: set the
        // direction and keep its `op` (any WHERE bounds are direction-agnostic; an `Eq`
        // reaches one row, so the direction is immaterial there).
        PlanNode::RowidScan(s) => Some(PlanNode::RowidScan(RowidScan {
            table: s.table.clone(),
            db: s.db,
            column_count: s.column_count,
            op: s.op.clone(),
            direction: dir.clone(),
        })),
        // An `IndexScan` walks index order, not rowid order.
        _ => None,
    }
}

/// Mechanism B — index order (ASC / Forward only; see the module doc for why DESC keeps
/// the Sort). The keys equal a contiguous run of an index's ascending Binary key columns.
fn mechanism_b_index(
    catalog: &dyn Catalog,
    td: &TableDef,
    n: usize,
    regs: &[usize],
    dir: &ScanDirection,
    leaf: &PlanNode,
) -> Option<PlanNode> {
    // Descending index order is not byte-identical to the stable sort (F4 tie reversal).
    if !matches!(dir, ScanDirection::Forward) {
        return None;
    }
    // Every relied-on register must be a genuine column, not the rowid — a rowid key is
    // Mechanism A's job, and a mix like `[col, rowid]` is served by neither.
    if regs.iter().any(|r| *r >= n) {
        return None;
    }

    match leaf {
        // Sub-case (i): extend the WHERE-chosen index seek. The keys must continue THAT
        // index's columns immediately after its equality prefix; then the Sort's input is
        // already this index order and dropping it is a no-op (stable sort by a prefix).
        PlanNode::IndexScan(s) => {
            let IndexOp::Seek { eq_prefix, .. } = &s.op else {
                return None;
            };
            let idx = catalog.index_in(s.db, &s.index).ok().flatten()?;
            if idx.partial {
                return None;
            }
            if !index_run_serves_order(idx, td, n, regs, eq_prefix.len()) {
                return None;
            }
            Some(PlanNode::IndexScan(IndexScan {
                table: s.table.clone(),
                db: s.db,
                column_count: s.column_count,
                index: s.index.clone(),
                op: s.op.clone(),
                direction: ScanDirection::Forward,
                covering: s.covering,
            }))
        }
        // Sub-case (ii): no WHERE index (full `SeqScan`). Use a non-partial index whose
        // key columns are EXACTLY the `ORDER BY` columns (so its only post-key tie-break
        // is the rowid, matching the SeqScan+stable-sort order). Keep any WHERE residual
        // Filter above the leaf (preserved by `replace_leaf`).
        PlanNode::SeqScan(s) => {
            let indexes = catalog.indexes_on_in(s.db, &s.table).ok()?;
            let idx = indexes.iter().copied().find(|idx| {
                !idx.partial
                    && idx.columns.len() == regs.len()
                    && index_run_serves_order(idx, td, n, regs, 0)
            })?;
            Some(PlanNode::IndexScan(IndexScan {
                table: s.table.clone(),
                db: s.db,
                column_count: s.column_count,
                index: idx.name.clone(),
                op: IndexOp::FullScan,
                direction: ScanDirection::Forward,
                covering: false,
            }))
        }
        // A `RowidScan` (WHERE rowid range) can't be replaced by an index scan without
        // losing its rowid bounds — keep the Sort.
        _ => None,
    }
}

/// Whether index `idx`'s key columns at positions `p .. p + regs.len()` are exactly the
/// `ORDER BY` keys, in order: each an ordinary (non-expression) ASCENDING Binary key
/// column whose register equals the traced key register. `p` is the equality-prefix
/// length the WHERE seek pinned (0 for a full scan).
fn index_run_serves_order(
    idx: &IndexDef,
    td: &TableDef,
    n: usize,
    regs: &[usize],
    p: usize,
) -> bool {
    for (j, &want_reg) in regs.iter().enumerate() {
        let pos = p + j;
        // `columns` / `key_columns` / `key_exprs` are parallel (the builder keeps them in
        // lockstep). Read all three via `.get` so a run that overruns the index's key
        // columns — or a malformed index with mismatched lengths — DECLINES (keep the
        // Sort) rather than panicking; correctness never depends on an unchecked index.
        let (Some(key_expr), Some(kc), Some(name)) =
            (idx.key_exprs.get(pos), idx.key_columns.get(pos), idx.columns.get(pos))
        else {
            return false;
        };
        // An INDEX ON AN EXPRESSION key column has no plain column name to order by.
        if key_expr.is_some() {
            return false;
        }
        // Ascending + Binary only (F1 / F4): a DESC or non-Binary key column would not
        // walk in the ORDER BY's ascending Binary order.
        if kc.descending || !key_column_is_binary(kc, td, name) {
            return false;
        }
        // The index key column must be the SAME base column the ORDER BY key traced to.
        match col_reg(td, name, n) {
            Some(reg) if reg == want_reg => {}
            _ => return false,
        }
    }
    true
}

/// Rebuild `node` — a `Filter*` chain over a base-table leaf — with the leaf replaced by
/// `new_leaf`, preserving the (row-order-preserving) `Filter` wrappers. Mirrors
/// [`base_table_leaf`]'s descent, so it always reaches the leaf that descent found.
fn replace_leaf(node: &PlanNode, new_leaf: PlanNode) -> PlanNode {
    match node {
        PlanNode::Filter { input, predicate } => PlanNode::Filter {
            input: Box::new(replace_leaf(input, new_leaf)),
            predicate: predicate.clone(),
        },
        // The leaf `base_table_leaf` already validated (any non-`Filter` node here). This
        // and `base_table_leaf` MUST descend the SAME node set (today: `Filter` only); the
        // assert makes a future drift loud — if `base_table_leaf` learns a new row-preserving
        // wrapper without this catch-all learning it too, the catch-all would REPLACE that
        // wrapper with the leaf and silently drop its subtree. Debug-only, so release plans
        // still fall back safely (the caller keeps the Sort only via `satisfy_order_by`'s
        // `None`, never through here).
        other => {
            debug_assert!(
                matches!(
                    other,
                    PlanNode::SeqScan(_) | PlanNode::RowidScan(_) | PlanNode::IndexScan(_)
                ),
                "replace_leaf reached non-leaf {other:?}; it drifted from base_table_leaf"
            );
            new_leaf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `Catalog`, `IndexDef`, `TableDef`, the access/plan/`EvalExpr` types, `col_reg`,
    // `Collation`, and `DbIndex` all arrive via `use super::*`; only these are new.
    use minisqlite_catalog::{ColumnDef, KeyColumn};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Result;

    use crate::plan::Plan;
    use crate::{Planner, QueryPlanner};

    // A static catalog answering table + index lookups from hand-built defs — the same
    // shape `access_path.rs`'s tests use, exercising the real `QueryPlanner` path without
    // a pager/b-tree.
    struct IdxCatalog {
        tables: Vec<TableDef>,
        indexes: Vec<IndexDef>,
    }

    impl Catalog for IdxCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
            Ok(self.indexes.iter().find(|i| i.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(self.indexes.iter().filter(|i| i.table.eq_ignore_ascii_case(table)).collect())
        }
        fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn create_table(&mut self, _p: &mut dyn Pager, _s: &CreateTable, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn create_index(&mut self, _p: &mut dyn Pager, _s: &CreateIndex, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn drop_object(&mut self, _p: &mut dyn Pager, _s: &Drop) -> Result<()> {
            unimplemented!("static test catalog")
        }
    }

    fn col(name: &str, decl: Option<&str>, collation: Option<&str>) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: decl.map(str::to_string),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: collation.map(str::to_string),
            default: None,
            default_value: None,
            generated: None,
        }
    }

    fn tdef(name: &str, columns: Vec<ColumnDef>, rowid_alias: Option<usize>) -> TableDef {
        TableDef {
            name: name.to_string(),
            columns,
            root_page: 2,
            without_rowid: false,
            rowid_alias,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
        }
    }

    /// An index with per-column collation / direction control.
    fn idef_kc(
        name: &str,
        table: &str,
        cols: &[(&str, Option<&str>, bool)],
        unique: bool,
        partial: bool,
    ) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: cols.iter().map(|(c, _, _)| c.to_string()).collect(),
            key_columns: cols
                .iter()
                .map(|(_, coll, desc)| KeyColumn {
                    collation: coll.map(str::to_string),
                    descending: *desc,
                })
                .collect(),
            key_exprs: vec![None; cols.len()],
            root_page: 3,
            unique,
            partial,
            // Preserve `partial == partial_predicate.is_some()`: an always-true placeholder for
            // a partial fixture (the order-scan tests only check that a partial index is
            // DECLINED via `.partial`, never evaluate the predicate).
            partial_predicate: partial
                .then(|| minisqlite_sql::Expr::Literal(minisqlite_sql::Literal::Integer(1))),
        }
    }

    /// A plain ascending, inherited-collation, non-unique, non-partial index.
    fn idef(name: &str, table: &str, cols: &[&str]) -> IndexDef {
        let cols: Vec<(&str, Option<&str>, bool)> = cols.iter().map(|c| (*c, None, false)).collect();
        idef_kc(name, table, &cols, false, false)
    }

    fn plan_it(sql: &str, cat: &dyn Catalog) -> Plan {
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, cat).expect("plan ok")
    }

    /// Whether a `Sort` node appears anywhere on the path from the root to the leaf.
    fn has_sort(node: &PlanNode) -> bool {
        match node {
            PlanNode::Sort { .. } => true,
            PlanNode::Project { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Distinct { input, .. }
            | PlanNode::Limit { input, .. } => has_sort(input),
            _ => false,
        }
    }

    /// Descend the row-preserving wrappers to the access-path leaf.
    fn leaf(node: &PlanNode) -> &PlanNode {
        match node {
            PlanNode::Project { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Distinct { input, .. }
            | PlanNode::Limit { input, .. } => leaf(input),
            other => other,
        }
    }

    fn one_col(t: &str) -> Vec<ColumnDef> {
        vec![col(t, Some("INTEGER"), None)]
    }

    // ------------------------------------------------------------------
    // MUST-SKIP (no Sort) — the scan already yields the order.
    // ------------------------------------------------------------------

    #[test]
    fn order_by_rowid_asc_skips_sort_via_seqscan() {
        let cat = IdxCatalog { tables: vec![tdef("t", one_col("a"), None)], indexes: vec![] };
        let plan = plan_it("SELECT * FROM t ORDER BY rowid", &cat);
        assert!(!has_sort(&plan.root), "ORDER BY rowid ASC must skip the Sort");
        assert!(matches!(leaf(&plan.root), PlanNode::SeqScan(_)), "ascending rowid = SeqScan");
    }

    #[test]
    fn order_by_rowid_desc_skips_sort_via_reverse_rowid_scan() {
        let cat = IdxCatalog { tables: vec![tdef("t", one_col("a"), None)], indexes: vec![] };
        let plan = plan_it("SELECT * FROM t ORDER BY rowid DESC", &cat);
        assert!(!has_sort(&plan.root), "ORDER BY rowid DESC must skip the Sort");
        match leaf(&plan.root) {
            PlanNode::RowidScan(RowidScan {
                op: RowidOp::Range { lo: None, hi: None },
                direction: ScanDirection::Reverse,
                ..
            }) => {}
            other => panic!("expected a full reverse RowidScan, got {other:?}"),
        }
    }

    #[test]
    fn order_by_intpk_alias_takes_the_rowid_path() {
        // `id INTEGER PRIMARY KEY` aliases the rowid, so `ORDER BY id` is a rowid order.
        let cols = vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)];
        let cat = IdxCatalog { tables: vec![tdef("u", cols, Some(0))], indexes: vec![] };

        let asc = plan_it("SELECT * FROM u ORDER BY id", &cat);
        assert!(!has_sort(&asc.root), "ORDER BY intpk ASC must skip the Sort");
        assert!(matches!(leaf(&asc.root), PlanNode::SeqScan(_)));

        let desc = plan_it("SELECT * FROM u ORDER BY id DESC", &cat);
        assert!(!has_sort(&desc.root), "ORDER BY intpk DESC must skip the Sort");
        assert!(matches!(
            leaf(&desc.root),
            PlanNode::RowidScan(RowidScan { direction: ScanDirection::Reverse, .. })
        ));
    }

    #[test]
    fn order_by_indexed_col_asc_replaces_seqscan_with_full_index_scan() {
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        let plan = plan_it("SELECT * FROM t ORDER BY a", &cat);
        assert!(!has_sort(&plan.root), "ORDER BY indexed col ASC must skip the Sort");
        match leaf(&plan.root) {
            PlanNode::IndexScan(IndexScan {
                index,
                op: IndexOp::FullScan,
                direction: ScanDirection::Forward,
                ..
            }) => assert_eq!(index, "ia"),
            other => panic!("expected a forward FullScan on ia, got {other:?}"),
        }
    }

    #[test]
    fn range_on_indexed_col_then_order_on_same_col_skips_sort() {
        // `WHERE a >= 5 ORDER BY a`: the WHERE index range already scans `a` ascending.
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        let plan = plan_it("SELECT a FROM t WHERE a >= 5 ORDER BY a", &cat);
        assert!(!has_sort(&plan.root), "range + same-column ORDER BY must skip the Sort");
        assert!(matches!(leaf(&plan.root), PlanNode::IndexScan(_)));
    }

    #[test]
    fn eq_prefix_then_order_on_next_column_skips_sort() {
        // Composite (a,b): `WHERE a = 1 ORDER BY b` continues the index after the a= prefix.
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 ORDER BY b", &cat);
        assert!(!has_sort(&plan.root), "eq-prefix then next-column ORDER BY must skip the Sort");
        match leaf(&plan.root) {
            PlanNode::IndexScan(IndexScan {
                index,
                op: IndexOp::Seek { eq_prefix, .. },
                direction: ScanDirection::Forward,
                ..
            }) => {
                assert_eq!(index, "iab");
                assert_eq!(eq_prefix.len(), 1, "the a=1 equality prefix is kept");
            }
            other => panic!("expected a forward IndexScan Seek on iab, got {other:?}"),
        }
    }

    #[test]
    fn order_by_full_composite_index_skips_sort() {
        // `ORDER BY a, b` matches the WHOLE (a,b) index (len == key count) -> FullScan.
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        let plan = plan_it("SELECT * FROM t ORDER BY a, b", &cat);
        assert!(!has_sort(&plan.root), "ORDER BY matching the full index must skip the Sort");
        assert!(matches!(
            leaf(&plan.root),
            PlanNode::IndexScan(IndexScan { op: IndexOp::FullScan, .. })
        ));
    }

    #[test]
    fn order_by_indexed_col_with_limit_skips_sort() {
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        let plan = plan_it("SELECT * FROM t ORDER BY a LIMIT 3", &cat);
        assert!(!has_sort(&plan.root), "ORDER BY indexed col + LIMIT must skip the Sort");
        // The Limit node still sits above the (now Sort-less) index scan.
        assert!(matches!(plan.root, PlanNode::Limit { .. }), "the LIMIT node is preserved");
        assert!(matches!(
            leaf(&plan.root),
            PlanNode::IndexScan(IndexScan { op: IndexOp::FullScan, .. })
        ));
    }

    // ------------------------------------------------------------------
    // MUST-NOT-SKIP (Sort kept) — the scan order can't be proven to match.
    // ------------------------------------------------------------------

    #[test]
    fn order_by_indexed_col_desc_keeps_sort() {
        // A reverse index scan reverses ties (rowid DESC) but the stable sort keeps them
        // rowid ASC — not byte-identical, so the Sort stays. (The rowid case is unique and
        // so IS served in DESC; a secondary index is not.)
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        let plan = plan_it("SELECT * FROM t ORDER BY a DESC", &cat);
        assert!(has_sort(&plan.root), "ORDER BY indexed col DESC must keep the Sort (tie reversal)");
    }

    #[test]
    fn order_by_nocase_keeps_sort() {
        // Column-declared NOCASE and an explicit COLLATE NOCASE: the Binary index can't
        // serve either.
        let cols =
            vec![col("a", Some("INTEGER"), None), col("c", Some("TEXT"), Some("NOCASE"))];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("ic", "t", &["c"]), idef("ia", "t", &["a"])],
        };
        assert!(
            has_sort(&plan_it("SELECT * FROM t ORDER BY c", &cat).root),
            "a NOCASE-declared column keeps the Sort"
        );
        assert!(
            has_sort(&plan_it("SELECT * FROM t ORDER BY a COLLATE NOCASE", &cat).root),
            "an explicit COLLATE NOCASE keeps the Sort"
        );
    }

    #[test]
    fn order_by_non_leading_index_column_keeps_sort() {
        // Only index is (a,b); `ORDER BY b` (no a= equality) can't use it.
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY b", &cat).root));
    }

    #[test]
    fn order_by_prefix_of_wider_index_keeps_sort_in_subcase_ii() {
        // No WHERE, `ORDER BY a`, only index is (a,b): a full (a,b) scan would tie-break by
        // `b` before the rowid, differing from SeqScan+stable-sort. So the Sort stays.
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        let plan = plan_it("SELECT * FROM t ORDER BY a", &cat);
        assert!(has_sort(&plan.root), "ORDER BY a over only a wider (a,b) index keeps the Sort");
    }

    #[test]
    fn order_by_computed_expression_keeps_sort() {
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY a+0", &cat).root));
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY abs(a)", &cat).root));
    }

    #[test]
    fn mixed_direction_keeps_sort() {
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY a ASC, b DESC", &cat).root));
    }

    #[test]
    fn nulls_last_on_asc_scan_keeps_sort() {
        // An ASC index scan is naturally NULLS FIRST; an explicit NULLS LAST contradicts it.
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY a NULLS LAST", &cat).root));
    }

    #[test]
    fn descending_index_column_keeps_sort() {
        // A DESC-declared index column stores `a` descending, so a forward scan would not
        // yield `ORDER BY a ASC`.
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef_kc("ia", "t", &[("a", None, true)], false, false)],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY a", &cat).root));
    }

    #[test]
    fn nocase_index_column_keeps_sort() {
        // An index key column declared COLLATE NOCASE is guarded against (forward-compat).
        let cols = vec![col("a", Some("INTEGER"), None), col("c", Some("TEXT"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef_kc("ic", "t", &[("c", Some("NOCASE"), false)], false, false)],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY c", &cat).root));
    }

    #[test]
    fn partial_index_keeps_sort() {
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef_kc("ia", "t", &[("a", None, false)], false, true)],
        };
        assert!(has_sort(&plan_it("SELECT * FROM t ORDER BY a", &cat).root));
    }

    #[test]
    fn aggregate_and_distinct_keep_sort() {
        let cat = IdxCatalog {
            tables: vec![tdef("t", one_col("a"), None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        assert!(
            has_sort(&plan_it("SELECT a, count(*) FROM t GROUP BY a ORDER BY a", &cat).root),
            "an aggregate query keeps the Sort"
        );
        assert!(
            has_sort(&plan_it("SELECT DISTINCT a FROM t ORDER BY a", &cat).root),
            "a DISTINCT query keeps the Sort"
        );
    }

    #[test]
    fn join_keeps_sort() {
        let cols = vec![col("a", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols.clone(), None), tdef("u", cols, None)],
            indexes: vec![idef("ia", "t", &["a"])],
        };
        assert!(has_sort(&plan_it("SELECT t.a FROM t, u ORDER BY t.a", &cat).root));
    }

    #[test]
    fn a_different_where_index_keeps_its_seek_and_the_sort() {
        // WHERE chose index (b) for `b = 1`; ORDER BY a wants a DIFFERENT index (a). We do
        // not swap the WHERE seek for an ORDER BY index — keep the narrowing seek + Sort.
        let cols = vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)];
        let cat = IdxCatalog {
            tables: vec![tdef("t", cols, None)],
            indexes: vec![idef("ib", "t", &["b"]), idef("ia", "t", &["a"])],
        };
        let plan = plan_it("SELECT * FROM t WHERE b = 1 ORDER BY a", &cat);
        assert!(has_sort(&plan.root), "a different WHERE index keeps its seek and the Sort");
        match leaf(&plan.root) {
            PlanNode::IndexScan(IndexScan { index, .. }) => {
                assert_eq!(index, "ib", "the narrowing WHERE seek on ib is kept")
            }
            other => panic!("expected the WHERE index seek on ib, got {other:?}"),
        }
    }
}
