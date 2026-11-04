//! Covering-index (index-only scan) marking — optoverview.html §"Covering Indexes".
//!
//! A post-pass over a compiled [`Plan`] that flips [`IndexScan::covering`] to `true` on
//! an index-scan leaf when the index b-tree ALREADY carries every table column the query
//! reads above that leaf, so the executor can read those values straight from the index
//! entry and SKIP the per-row by-rowid table fetch. It is a pure performance optimization:
//! the emitted row shape (`column_count + 1`) is unchanged and the values are byte-identical
//! to the table-fetch path (an index key column stores the same value, under the same
//! storage class, as the table row's column — the index is written with the column's own
//! affinity), so `covering` only elides the extra lookup.
//!
//! ## Why a post-pass (not at access-path selection)
//! The access-path chooser ([`crate::access_path`]) runs per base table with no visibility
//! into the projection / ORDER BY / GROUP BY / aggregates above it, so it cannot know which
//! columns the whole query needs. This pass runs on the FULLY assembled tree, where the
//! complete set of column references above each leaf is present. It mirrors
//! [`crate::compile::order_scan`], another late rewrite over the assembled plan.
//!
//! ## Correctness argument (proven on paper before implementing)
//! For an [`IndexScan`] leaf `L` over table `T` (`N` columns) reached from a processing
//! root through a chain of single-input row operators (no join / set-op), `L` emits the
//! base row `[c0..c_{N-1}, rowid]` (width `N+1`; register `N` is the rowid). We process
//! only BASE-0 contexts (the top-level root, a NON-correlated subplan, a CTE body), where
//! that leaf row is NOT prefixed by an outer row, so `Column(i)` for `i <= N` reads base
//! register `i` directly.
//!
//! * The LOWEST [`Project`]/[`Aggregate`] in the chain (the "boundary") reshapes the row:
//!   every node ABOVE it reads the reshaped output, whose base dependencies are exactly the
//!   boundary node's own expressions — so collecting base refs from the boundary DOWN to the
//!   leaf captures the COMPLETE set of base columns the query needs. (Ranking / windowing /
//!   sorting above the boundary read derived columns, never base registers.)
//! * `covered = {N (rowid, always in the index entry)} ∪ {register of each BARE index key
//!   column}`. If every collected base ref is in `covered`, reading from the index entry is
//!   byte-identical, so `covering = true`. The empty-ref case (`count(*)` whose whole
//!   predicate is served by the index range) is trivially covered — the important instance.
//!
//! Anything the pass cannot PROVE safe stays `covering = false` (the correct table-fetch
//! path): a subquery in a collected expression (a correlated one could read a base column
//! invisibly), a window / join / set-op / unrecognized node, a `WITHOUT ROWID` table, any
//! generated column, a partial or expression index, or a `Column` beyond the base row.
//! Incomplete-but-correct beats complete-but-wrong: a missed covering is only a slower
//! (still correct) plan.

use std::collections::BTreeSet;

use minisqlite_catalog::Catalog;
use minisqlite_expr::EvalExpr;

use crate::access::IndexScan;
use crate::access_path::col_reg;
use crate::plan::{CtePlan, Plan, PlanNode};

/// Mark every provably-covering [`IndexScan`] leaf in `plan`. Runs on the top-level root,
/// each NON-correlated subplan, and each CTE body — the contexts whose leaf row is not
/// prefixed by an outer row (base-0), so `col_reg`'s 0-based register mapping is exact. A
/// correlated subplan's leaf columns sit at an outer offset, so it is left alone (the safe
/// table-fetch path).
pub(crate) fn mark_covering(catalog: &dyn Catalog, plan: &mut Plan) {
    cover_single_table_chain(catalog, &mut plan.root);
    for sub in &mut plan.subqueries {
        // A correlated subplan's own columns are offset past the outer row it reads, so the
        // 0-based `col_reg` mapping would be wrong there. Only non-correlated subplans (which
        // the executor runs with an empty outer, leaf row at base 0) are safe to cover.
        if !sub.correlated {
            cover_single_table_chain(catalog, &mut sub.plan);
        }
    }
    for cte in &mut plan.ctes {
        match cte {
            CtePlan::Materialized { body, .. } => cover_single_table_chain(catalog, body),
            CtePlan::Recursive { seed, step, .. } => {
                // Both compile at base 0. The step typically reads the working table
                // (`RecursiveScan`) or joins, so its chain declines; the seed may be a plain
                // single-table select that can cover. Either way the analysis is conservative.
                cover_single_table_chain(catalog, seed);
                cover_single_table_chain(catalog, step);
            }
        }
    }
}

/// If `node` roots a single-input chain ending at an [`IndexScan`] whose index covers every
/// base column referenced above it, set that leaf's `covering` flag. A two-phase borrow: an
/// immutable analysis decides, then a mutable descent flips the one leaf.
fn cover_single_table_chain(catalog: &dyn Catalog, node: &mut PlanNode) {
    if decide(catalog, node) {
        if let Some(leaf) = index_leaf_mut(node) {
            leaf.covering = true;
        }
    }
}

/// The single index-scan leaf reachable from `node` through the recognized single-input
/// chain nodes, or `None` if the chain hits any other node first. Mirrors [`collect_chain`]
/// exactly so it reaches the SAME leaf the decision analyzed. Deref coercion turns each
/// `&mut Box<PlanNode>` child into the `&mut PlanNode` the recursion takes.
fn index_leaf_mut(node: &mut PlanNode) -> Option<&mut IndexScan> {
    match node {
        PlanNode::IndexScan(s) => Some(s),
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => index_leaf_mut(input),
        PlanNode::Aggregate(a) => index_leaf_mut(&mut a.input),
        _ => None,
    }
}

/// Decide whether the chain rooted at `node` should have its index leaf marked covering.
fn decide(catalog: &dyn Catalog, node: &PlanNode) -> bool {
    let mut path: Vec<&PlanNode> = Vec::new();
    if !collect_chain(node, &mut path) {
        return false; // the chain does not end at an IndexScan we can analyze
    }
    let leaf: &IndexScan = match path.last() {
        Some(PlanNode::IndexScan(s)) => s,
        // `collect_chain` returning true guarantees an IndexScan tail; this is defensive.
        _ => return false,
    };
    // The lowest Project/Aggregate is the base-row boundary: nodes above it read the
    // reshaped row (their base deps flow through it), nodes at/below it read base registers.
    let Some(boundary) =
        path.iter().rposition(|n| matches!(n, PlanNode::Project { .. } | PlanNode::Aggregate(_)))
    else {
        return false; // no projection/aggregate in the chain — not a query core we model
    };
    let n = leaf.column_count;
    let mut refs: BTreeSet<usize> = BTreeSet::new();
    for chain_node in &path[boundary..] {
        if !collect_node_base_refs(chain_node, n, &mut refs) {
            return false; // a subquery / unhandled node in the base-row region → decline
        }
    }
    match covered_registers(catalog, leaf) {
        Some(covered) => refs.iter().all(|r| covered.contains(r)),
        None => false,
    }
}

/// Walk the single-input chain from `node`, pushing each node into `path`. Returns `true`
/// iff the chain terminates at an [`IndexScan`] through only the recognized row operators;
/// any other node (join, set-op, window, a `SeqScan`/`RowidScan` leaf, DML, …) returns
/// `false` — the covering analysis does not apply. Window is deliberately NOT recognized:
/// a window query reads its base columns through the window operator, which this pass does
/// not model, so it declines to the table-fetch path.
fn collect_chain<'a>(node: &'a PlanNode, path: &mut Vec<&'a PlanNode>) -> bool {
    path.push(node);
    match node {
        PlanNode::IndexScan(_) => true,
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => collect_chain(input, path),
        PlanNode::Aggregate(a) => collect_chain(&a.input, path),
        _ => false,
    }
}

/// Collect the base-table column registers a single chain node reads into `refs`. Returns
/// `false` to abort the whole decision (a subquery or an unmodelled node was seen). Only the
/// boundary node (Project/Aggregate), any Filter below it, and the IndexScan leaf can appear
/// here; a `HAVING` predicate is intentionally skipped because it binds against the
/// post-aggregate row, not base registers.
fn collect_node_base_refs(node: &PlanNode, n: usize, refs: &mut BTreeSet<usize>) -> bool {
    match node {
        PlanNode::Project { exprs, .. } => exprs.iter().all(|e| collect_base_cols(e, n, refs)),
        PlanNode::Filter { predicate, .. } => collect_base_cols(predicate, n, refs),
        PlanNode::Aggregate(a) => {
            if !a.group_by.iter().all(|e| collect_base_cols(e, n, refs)) {
                return false;
            }
            for agg in &a.aggregates {
                if !agg.args.iter().all(|e| collect_base_cols(e, n, refs)) {
                    return false;
                }
                if let Some(f) = &agg.filter {
                    if !collect_base_cols(f, n, refs) {
                        return false;
                    }
                }
                for sk in &agg.order_by {
                    if !collect_base_cols(&sk.expr, n, refs) {
                        return false;
                    }
                }
            }
            // The single-min/max and general bare-column captures are INPUT registers the
            // operator reads from the base row, so they count as referenced base columns.
            if let Some(mm) = &a.minmax_bare {
                for &r in &mm.captured_regs {
                    if r > n {
                        return false;
                    }
                    refs.insert(r);
                }
            }
            if let Some(bare) = &a.bare_arbitrary {
                for &r in bare {
                    if r > n {
                        return false;
                    }
                    refs.insert(r);
                }
            }
            true
        }
        // The IndexScan leaf produces the base row; its seek/bound expressions reference only
        // the outer row (empty at base 0), never the base table's own columns.
        PlanNode::IndexScan(_) => true,
        // Any other node type in the base-row region is unexpected in a single-table core;
        // decline rather than guess at what it reads.
        _ => false,
    }
}

/// Collect every base-table column register (`Column(i)` with `i <= n`, `n` = the rowid
/// register) referenced by `e` into `refs`. Returns `false` to abort the decision when the
/// expression reaches a subquery (a correlated one could read a base column this walker
/// cannot see) or a `Column` beyond the base row (which should not occur below the boundary
/// in a base-0 single-table chain, so its presence means the analysis is unsound here).
///
/// A precise refinement would keep covering for a subquery by adding its
/// `SubPlan::correlated_cols` to `refs`; declining is the conservative first step.
// TODO: use `SubPlan::correlated_cols` to cover queries whose collected expressions carry
// a subquery, instead of declining on every subquery.
fn collect_base_cols(e: &EvalExpr, n: usize, refs: &mut BTreeSet<usize>) -> bool {
    match e {
        EvalExpr::Column(i) => {
            if *i > n {
                return false;
            }
            refs.insert(*i);
            true
        }
        EvalExpr::Literal(_) | EvalExpr::Param(_) | EvalExpr::Now(_) => true,
        // A subquery hides its column dependencies behind a `SubqueryId`; decline (see doc).
        EvalExpr::ScalarSubquery(_)
        | EvalExpr::ScalarSubqueryColumn { .. }
        | EvalExpr::Exists { .. }
        | EvalExpr::InSubquery { .. }
        | EvalExpr::InSubqueryRow { .. } => false,
        EvalExpr::Unary { operand, .. }
        | EvalExpr::Cast { operand, .. }
        | EvalExpr::Collate { operand, .. }
        | EvalExpr::IsNull(operand)
        | EvalExpr::NotNull(operand) => collect_base_cols(operand, n, refs),
        EvalExpr::Arith { left, right, .. }
        | EvalExpr::Concat { left, right }
        | EvalExpr::Bitwise { left, right, .. }
        | EvalExpr::Compare { left, right, .. }
        | EvalExpr::NullIf { left, right, .. } => {
            collect_base_cols(left, n, refs) && collect_base_cols(right, n, refs)
        }
        EvalExpr::And(a, b) | EvalExpr::Or(a, b) => {
            collect_base_cols(a, n, refs) && collect_base_cols(b, n, refs)
        }
        EvalExpr::Between { subject, low, high, .. } => {
            collect_base_cols(subject, n, refs)
                && collect_base_cols(low, n, refs)
                && collect_base_cols(high, n, refs)
        }
        EvalExpr::InList { subject, items, .. } => {
            collect_base_cols(subject, n, refs) && items.iter().all(|i| collect_base_cols(i, n, refs))
        }
        EvalExpr::Coalesce(items) => items.iter().all(|i| collect_base_cols(i, n, refs)),
        EvalExpr::Case { operand, whens, else_expr } => {
            operand.as_deref().map_or(true, |o| collect_base_cols(o, n, refs))
                && whens
                    .iter()
                    .all(|w| collect_base_cols(&w.when, n, refs) && collect_base_cols(&w.then, n, refs))
                && else_expr.as_deref().map_or(true, |o| collect_base_cols(o, n, refs))
        }
        EvalExpr::Like { subject, pattern, escape, .. } => {
            collect_base_cols(subject, n, refs)
                && collect_base_cols(pattern, n, refs)
                && escape.as_deref().map_or(true, |x| collect_base_cols(x, n, refs))
        }
        EvalExpr::Func { args, .. } => args.iter().all(|a| collect_base_cols(a, n, refs)),
        // RAISE(...) has no column operands.
        EvalExpr::Raise { .. } => true,
    }
}

/// The set of base-row registers the leaf's index provably carries: the rowid (register `N`,
/// always the last element of every index entry) plus the register of each BARE (non-
/// expression) key column. `None` declines covering entirely — a `WITHOUT ROWID` table
/// (index entries carry the PRIMARY KEY, not a rowid), any generated column (the base record
/// omits VIRTUAL columns; a covering read would need to recompute them), a partial index
/// (does not carry every row), a plan/schema column-count mismatch, or an index key column
/// whose name is not a table column (a catalog inconsistency).
fn covered_registers(catalog: &dyn Catalog, leaf: &IndexScan) -> Option<BTreeSet<usize>> {
    let td = catalog.table_in(leaf.db, &leaf.table).ok()??;
    if td.without_rowid {
        return None;
    }
    // A generated column (STORED or VIRTUAL) makes the index-only read unsafe: a VIRTUAL
    // column is never stored, and covering must not have to recompute it. Decline the whole
    // table conservatively.
    if td.columns.iter().any(|c| c.generated.is_some()) {
        return None;
    }
    let n = td.columns.len();
    if n != leaf.column_count {
        return None; // plan and schema disagree on the column count — fail closed
    }
    let idx = catalog.index_in(leaf.db, &leaf.index).ok()??;
    if idx.partial {
        return None;
    }
    let mut covered: BTreeSet<usize> = BTreeSet::new();
    covered.insert(n); // the rowid: the trailing element of every index entry
    for (k, name) in idx.columns.iter().enumerate() {
        // An expression key column stores a computed value, not a named table column, so it
        // covers no column by name. (Expression indexes are never chosen as a seek/scan leaf
        // today, so this is also defensive.)
        let is_expr = idx.key_exprs.get(k).is_some_and(|e| e.is_some());
        if is_expr {
            continue;
        }
        match col_reg(td, name, n) {
            Some(reg) => {
                covered.insert(reg);
            }
            None => return None, // index names a non-column: decline
        }
    }
    Some(covered)
}

#[cfg(test)]
mod tests {
    use super::*;

    use minisqlite_catalog::{ColumnDef, IndexDef, KeyColumn, TableDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::{DbIndex, Result};

    use crate::access::{IndexOp, IndexScan, ScanDirection};
    use crate::{Planner, QueryPlanner};

    // A static catalog answering table + index lookups from hand-built defs (the same shape
    // `access_path.rs` / `order_scan.rs` tests use), driving the REAL `QueryPlanner` — so
    // these assert the covering flag the whole pipeline sets, not a unit in isolation.
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

    fn col(name: &str, decl: Option<&str>) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: decl.map(str::to_string),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: None,
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
            strict: false,
        }
    }

    /// A plain ascending, inherited-collation, non-unique, non-partial, all-named-column
    /// index over `cols`.
    fn idef(name: &str, table: &str, cols: &[&str]) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: cols.iter().map(|c| c.to_string()).collect(),
            key_columns: cols
                .iter()
                .map(|_| KeyColumn { collation: None, descending: false })
                .collect(),
            key_exprs: vec![None; cols.len()],
            root_page: 3,
            unique: false,
            partial: false,
            partial_predicate: None,
        }
    }

    fn plan_it(sql: &str, cat: &dyn Catalog) -> Plan {
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, cat).expect("plan ok")
    }

    /// The single index-scan leaf reachable through the recognized single-input chain, or
    /// `None` (the planner chose a SeqScan / RowidScan, or the tree is not a single-table
    /// chain). Mirrors [`index_leaf_mut`].
    fn index_leaf(node: &PlanNode) -> Option<&IndexScan> {
        match node {
            PlanNode::IndexScan(s) => Some(s),
            PlanNode::Filter { input, .. }
            | PlanNode::Project { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::Distinct { input, .. } => index_leaf(input),
            PlanNode::Aggregate(a) => index_leaf(&a.input),
            _ => None,
        }
    }

    /// The `covering` flag of the plan's index-scan leaf. Panics if the plan has no index
    /// leaf — a test asserting on covering must have driven the planner to an index scan, so
    /// a missing leaf is a broken test premise (the index was not chosen), not a `false`.
    fn covering_of(sql: &str, cat: &dyn Catalog) -> bool {
        let plan = plan_it(sql, cat);
        index_leaf(&plan.root)
            .unwrap_or_else(|| panic!("no IndexScan leaf for `{sql}` (index not chosen?)"))
            .covering
    }

    // Two-column table `t(a, b)` plus a one-column table for single-column-index cases; a
    // classic rowid table (no INTEGER PRIMARY KEY alias) unless noted.
    fn cat_ab_index_a() -> IdxCatalog {
        IdxCatalog {
            tables: vec![tdef("t", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None)],
            indexes: vec![idef("ia", "t", &["a"])],
        }
    }

    // ---------------- MUST COVER (covering == true) ----------------

    #[test]
    fn count_star_over_index_range_is_covering() {
        // THE headline case: `count(*)` reads no column, and the WHERE range is served by the
        // index seek, so nothing above the scan reads a base column — covering engages and the
        // per-row table fetch is skipped.
        let cat = cat_ab_index_a();
        assert!(covering_of("SELECT count(*) FROM t WHERE a BETWEEN 2 AND 8", &cat));
    }

    #[test]
    fn project_only_indexed_column_range_is_covering() {
        let cat = cat_ab_index_a();
        assert!(covering_of("SELECT a FROM t WHERE a >= 5", &cat));
    }

    #[test]
    fn project_only_indexed_column_equality_is_covering() {
        let cat = cat_ab_index_a();
        assert!(covering_of("SELECT a FROM t WHERE a = 3", &cat));
    }

    #[test]
    fn indexed_column_plus_rowid_is_covering() {
        // The rowid rides in every index entry, so projecting it alongside the indexed
        // column stays covering.
        let cat = cat_ab_index_a();
        assert!(covering_of("SELECT a, rowid FROM t WHERE a = 3", &cat));
    }

    #[test]
    fn multi_column_index_covering_subset_is_covering() {
        // Composite (a, b): a query needing only a and b is fully covered.
        let cat = IdxCatalog {
            tables: vec![tdef(
                "t",
                vec![col("a", Some("INTEGER")), col("b", Some("INTEGER")), col("c", Some("INTEGER"))],
                None,
            )],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        assert!(covering_of("SELECT a, b FROM t WHERE a = 1", &cat));
    }

    #[test]
    fn where_only_reference_projecting_constant_is_covering() {
        // The projection is a literal (no base column) and the WHERE column is the index
        // column — nothing above the scan reads an uncovered base column.
        let cat = cat_ab_index_a();
        assert!(covering_of("SELECT 1 FROM t WHERE a = 3", &cat));
    }

    // ---------------- MUST NOT COVER (covering == false) ----------------

    #[test]
    fn projecting_a_non_indexed_column_is_not_covering() {
        // Index on a only; projecting b needs the table row.
        let cat = cat_ab_index_a();
        assert!(!covering_of("SELECT a, b FROM t WHERE a = 1", &cat));
    }

    #[test]
    fn residual_where_on_non_indexed_column_is_not_covering() {
        // `b = 2` cannot be served by the a-index, so it stays a residual Filter reading b —
        // an uncovered base column above the scan.
        let cat = cat_ab_index_a();
        assert!(!covering_of("SELECT a FROM t WHERE a = 1 AND b = 2", &cat));
    }

    #[test]
    fn select_star_over_wider_table_is_not_covering() {
        // `*` needs every column; the a-index covers only a (+ rowid), not b.
        let cat = cat_ab_index_a();
        assert!(!covering_of("SELECT * FROM t WHERE a = 1", &cat));
    }

    #[test]
    fn multi_column_index_needing_uncovered_column_is_not_covering() {
        // Composite (a, b) but the query needs c, which the index does not carry.
        let cat = IdxCatalog {
            tables: vec![tdef(
                "t",
                vec![col("a", Some("INTEGER")), col("b", Some("INTEGER")), col("c", Some("INTEGER"))],
                None,
            )],
            indexes: vec![idef("iab", "t", &["a", "b"])],
        };
        assert!(!covering_of("SELECT a, c FROM t WHERE a = 1", &cat));
    }

    #[test]
    fn order_by_hidden_uncovered_column_is_not_covering() {
        // `ORDER BY b` appends a HIDDEN projection of b (an uncovered column) to the SELECT
        // Project, so the covering pass — which reads that Project — sees b and declines. This
        // pins the load-bearing fact that hidden ORDER BY columns are captured.
        let cat = cat_ab_index_a();
        assert!(!covering_of("SELECT a FROM t WHERE a >= 1 ORDER BY b", &cat));
    }

    #[test]
    fn partial_index_is_not_covering() {
        // A partial index does not carry every row, so even a fully-in-index projection must
        // not read straight from it. (If the planner declines to seek a partial index at all,
        // there is no IndexScan leaf and this panics — either way covering never engages.)
        let mut cat = cat_ab_index_a();
        cat.indexes[0].partial = true;
        let plan = plan_it("SELECT a FROM t WHERE a = 1", &cat);
        // Covering must be false whether or not a partial index was chosen as the leaf.
        if let Some(leaf) = index_leaf(&plan.root) {
            assert!(!leaf.covering, "a partial index must never be covering");
        }
    }

    #[test]
    fn generated_column_table_is_not_covering() {
        // Any generated column on the table makes an index-only read unsafe (a VIRTUAL column
        // is never stored), so the whole table declines covering — even for a column the index
        // does carry.
        let mut cols = vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))];
        cols[1].generated = Some(minisqlite_catalog::GeneratedColumn {
            expr: minisqlite_sql::Expr::Literal(minisqlite_sql::Literal::Integer(1)),
            stored: false,
        });
        let cat = IdxCatalog { tables: vec![tdef("t", cols, None)], indexes: vec![idef("ia", "t", &["a"])] };
        assert!(!covering_of("SELECT a FROM t WHERE a = 1", &cat));
    }

    #[test]
    fn without_rowid_table_declines_covering() {
        // A WITHOUT ROWID table stores its rows in a PRIMARY KEY b-tree and a secondary index
        // entry carries that PRIMARY KEY as its trailing element — NOT an integer rowid. The
        // covering read surfaces the entry's last value into the rowid register `N`, so on a WR
        // table it would return the PK bytes where a rowid is expected: a wrong answer.
        // `covered_registers` fails closed on `td.without_rowid`.
        //
        // This guard is UNREACHABLE through the real planner today: `access_path` forces a
        // `SeqScan` (never an `IndexScan`) for a WR table (see `access_path.rs` `without_rowid`
        // branch), so there is no leaf for a plan-shape assertion to inspect. A DIRECT
        // `covered_registers` call is therefore the only way to pin this defensive guard, so a
        // future WR-aware index access path that drops it fails loudly here rather than
        // silently emitting wrong bytes. Sibling declines (partial / generated / expression)
        // ARE reachable and tested through the planner above.
        let leaf = IndexScan {
            table: "t".to_string(),
            db: DbIndex::MAIN,
            column_count: 2,
            index: "ia".to_string(),
            op: IndexOp::FullScan,
            direction: ScanDirection::Forward,
            covering: false,
        };
        // Control: the SAME leaf over a plain rowid table IS coverable (rowid N=2 plus the
        // register of key column `a`) — proving the assertion below isn't vacuously `None`.
        let mut cat = cat_ab_index_a();
        assert!(
            covered_registers(&cat, &leaf).is_some(),
            "control: a plain rowid table's single-column index must be coverable"
        );
        // The WR flag ALONE must flip the decision to decline.
        cat.tables[0].without_rowid = true;
        assert!(
            covered_registers(&cat, &leaf).is_none(),
            "a WITHOUT ROWID table must never be covering: its index entry's trailing value is \
             the PRIMARY KEY, not a rowid"
        );
    }
}
