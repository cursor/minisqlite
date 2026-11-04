//! The MIN/MAX index/rowid seek optimization (optoverview.html §13, "The MIN/MAX
//! Optimization"): recognize a single bare-column `MIN(col)`/`MAX(col)` over a whole
//! table and satisfy it with one O(log n) b-tree seek to the extremum entry instead of
//! an O(n) full-scan aggregate.
//!
//! [`try_optimize`] is the whole entry point. `compile::aggregate` builds the ordinary
//! [`Aggregate`] for a plain (no-window) aggregate query and hands it here; when EVERY
//! eligibility condition holds this returns a [`PlanNode::MinMaxSeek`], otherwise it
//! returns the UNTOUCHED `PlanNode::Aggregate`. The fall back is conservative and
//! byte-identical: an unproven case keeps the correct full-scan plan, so a wrong extremum
//! (the classic mis-handled NULL / collation / DESC bug) is impossible — at worst the
//! optimization simply does not fire.
//!
//! # Why each condition is a correctness condition, not a heuristic
//!
//! * BINARY comparison only. Every rowid / index b-tree is physically ordered under
//!   `Collation::Binary` (`minisqlite_btree::index_key`), regardless of a declared
//!   NOCASE/RTRIM. The scan-path accumulator compares under `arg_collations[0]` (exec
//!   `ops::aggregate`); we read that SAME value, so a non-BINARY comparison (a NOCASE
//!   column) declines rather than letting the b-tree walk disagree with the scan.
//! * Ascending, non-partial, non-expression index whose LEFT-MOST key column is the
//!   argument. A DESC or non-BINARY-declared key column, an expression key, or a partial
//!   index would not present the declared extremum first — mirrors the guards
//!   `compile::order_scan` applies for the analogous ORDER-BY-via-index rewrite.
//! * Non-BLOB affinity on the index path. A typeless column can hold two equal-comparing
//!   values in different storage classes (`Integer(2)` vs `Real(2.0)`); the scan keeps the
//!   first-seen (smallest-rowid) representation on a tie while an index walk could surface
//!   another — not byte-identical. A real affinity canonicalizes equal values to one
//!   representation, so any tied entry is byte-identical. The rowid key is unique (no ties).
//! * The register `base` is read from the FROM scope's single source, so the
//!   column/rowid mapping is correct at ANY nesting depth: a bare column binds to `base+i`
//!   and the rowid to `base+N` (`bind::Source::column_register`), so inside a correlated
//!   subquery (`base > 0`) a naive `reg == N` test would mis-map onto another column.

use minisqlite_catalog::{Catalog, IndexDef, TableDef};
use minisqlite_expr::EvalExpr;
use minisqlite_types::{Affinity, Collation};

use crate::access::SeqScan;
use crate::access_path::{col_reg, key_column_is_binary};
use crate::bind::{Scope, Source};
use crate::plan::{Aggregate, MinMaxSeek, MinMaxSource, PlanNode};

/// Replace `agg` with a [`PlanNode::MinMaxSeek`] when it is provably a single bare-column
/// `MIN`/`MAX` over a whole table served by the rowid or a usable index; otherwise return
/// `PlanNode::Aggregate(agg)` unchanged (the correct full-scan fall back).
///
/// `is_max` is `Some(true)` for `MAX`, `Some(false)` for `MIN`, and `None` when the query
/// is not the single-min/max special case at all — computed by the caller from its
/// `single_minmax_special` pre-scan (only fires when there is exactly ONE aggregate).
pub(crate) fn try_optimize(
    scope: &Scope,
    catalog: &dyn Catalog,
    agg: Aggregate,
    is_max: Option<bool>,
) -> PlanNode {
    match seek_plan(scope, catalog, &agg, is_max) {
        Some(node) => node,
        None => PlanNode::Aggregate(agg),
    }
}

/// The eligibility check + plan construction, factored out so a decline is a plain `None`
/// (the caller rebuilds `PlanNode::Aggregate(agg)` without a clone).
fn seek_plan(
    scope: &Scope,
    catalog: &dyn Catalog,
    agg: &Aggregate,
    is_max: Option<bool>,
) -> Option<PlanNode> {
    let is_max = is_max?;
    // No GROUP BY / HAVING / §2.5 bare-column capture: the whole query must reduce to a
    // single extremum over the whole table (one width-1 output row).
    if !agg.group_by.is_empty() || agg.having.is_some() {
        return None;
    }
    if agg.minmax_bare.is_some() || agg.bare_arbitrary.is_some() {
        return None;
    }
    // Exactly one aggregate, and it is a plain single-arg MIN/MAX with no modifiers
    // (DISTINCT / FILTER / aggregate-ORDER BY are all irrelevant to a bare min/max but we
    // stay conservative and decline them).
    if agg.aggregates.len() != 1 {
        return None;
    }
    let call = &agg.aggregates[0];
    if call.distinct || call.filter.is_some() || !call.order_by.is_empty() || call.args.len() != 1 {
        return None;
    }
    // The argument must be a BARE column reference (not `MAX(x+1)`, `MAX(x*1)`, …).
    let reg = match &call.args[0] {
        EvalExpr::Column(r) => *r,
        _ => return None,
    };
    // The extremum must be compared under BINARY (the physical b-tree order). This is the
    // SAME collation the scan-path accumulator uses, so the two paths cannot disagree on
    // which value is the extremum.
    if call.arg_collations.first().copied().unwrap_or(Collation::Binary) != Collation::Binary {
        return None;
    }
    // The input must be a bare full-table scan: a `SeqScan` means no WHERE (a `Filter`
    // wrapper) and a single rowid base table (a WITHOUT ROWID scan is also a `SeqScan` but
    // is excluded below; a join/derived source is a different node).
    let PlanNode::SeqScan(scan) = agg.input.as_ref() else {
        return None;
    };
    // The FROM scope must be exactly one base table; its register `base` maps `reg` back to
    // a physical column/rowid correctly at any nesting depth (see the module doc).
    let [source] = scope.sources else {
        return None;
    };
    let (table, db, base): (&TableDef, _, _) = match source {
        Source::BaseTable { table, db, base, .. } => (*table, *db, *base),
        Source::Derived { .. } => return None,
    };
    // Defensive: the single source and the scan must name the same base b-tree (always true
    // for a single-table query — the scan was built from this source).
    if db != scan.db || !table.name.eq_ignore_ascii_case(&scan.table) || table.without_rowid {
        return None;
    }
    let n = table.columns.len();
    // ROWID path: the rowid pseudo-column / INTEGER PRIMARY KEY alias binds to `base + N`.
    if reg == base + n {
        return Some(seek_node(scan, is_max, MinMaxSource::Rowid));
    }
    // Otherwise a bare ordinary column at `base + i`.
    let i = reg.checked_sub(base)?;
    if i >= n {
        return None;
    }
    if table.column_affinity(i) == Affinity::Blob {
        return None;
    }
    // A catalog `Err` here DECLINES (`.ok()?` -> keep the full-scan Aggregate) rather than
    // propagating: this is an opportunistic rewrite whose fall back is always correct, and
    // this table was already resolved when `build_from` built the `SeqScan`, so a lookup
    // error is effectively unreachable and declining loses only the optimization, never
    // correctness. (Mirrors the `.ok()` decline in `compile::order_scan`.)
    let index = catalog
        .indexes_on_in(db, &table.name)
        .ok()?
        .into_iter()
        .find(|ix| index_leftmost_serves(ix, table, n, i))?;
    Some(seek_node(scan, is_max, MinMaxSource::Index { index: index.name.clone() }))
}

/// Whether `ix` can serve the extremum of base column `want_col`: a NON-partial, ordinary
/// (non-expression) index whose LEFT-MOST key column is exactly that column, ascending, and
/// declared under BINARY collation (the b-tree's physical order). Mirrors the guards the
/// ORDER-BY-via-index optimizer (`compile::order_scan`) applies, so both decline the same
/// DESC / non-BINARY / expression / partial shapes — the cases where a plain forward walk
/// would not present the declared extremum first.
fn index_leftmost_serves(ix: &IndexDef, table: &TableDef, n: usize, want_col: usize) -> bool {
    if ix.partial {
        return false;
    }
    let (Some(name), Some(kc)) = (ix.columns.first(), ix.key_columns.first()) else {
        return false;
    };
    // An expression key column (`CREATE INDEX i ON t(a+b)`) is not a plain column reference.
    if matches!(ix.key_exprs.first(), Some(Some(_))) {
        return false;
    }
    if kc.descending || !key_column_is_binary(kc, table, name) {
        return false;
    }
    matches!(col_reg(table, name, n), Some(r) if r == want_col)
}

/// Build the seek node from the scan it replaces (reusing its resolved table + namespace).
fn seek_node(scan: &SeqScan, is_max: bool, source: MinMaxSource) -> PlanNode {
    PlanNode::MinMaxSeek(MinMaxSeek { table: scan.table.clone(), db: scan.db, is_max, source })
}

#[cfg(test)]
mod tests {
    //! Plan-shape tests for the §13 MIN/MAX seek rewrite: every ELIGIBLE case must compile
    //! to a [`PlanNode::MinMaxSeek`] (so a future regression that silently stops applying
    //! the optimization is caught), and every INELIGIBLE case must keep the ordinary
    //! [`PlanNode::Aggregate`] full-scan plan (so the conservative fall back that guarantees
    //! byte-identical results is likewise pinned). Result-equivalence across the full
    //! NULL/collation/DESC edge matrix is proven end-to-end through `minisqlite::Connection`
    //! in `minisqlite/tests/conformance_minmax_index.rs`; here we pin only the plan SHAPE.
    //!
    //! Uses a self-contained index-carrying [`Catalog`] rather than the shared `src/tests.rs`
    //! fixtures, so this optimization's tests live in its own file (own commit cell) instead
    //! of contending on that hot shared test module.

    use minisqlite_catalog::{ColumnDef, IndexDef, KeyColumn, TableDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Result;

    use crate::plan::{MinMaxSeek, MinMaxSource, PlanNode};
    use crate::{Planner, QueryPlanner};

    /// A static schema store carrying tables AND indexes — the index half the shared
    /// `TestCatalog` lacks, which the seek rewrite needs to find a usable index.
    struct Cat {
        tables: Vec<TableDef>,
        indexes: Vec<IndexDef>,
    }

    impl minisqlite_catalog::Catalog for Cat {
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
            unimplemented!("test catalog is static")
        }
        fn create_table(&mut self, _: &mut dyn Pager, _: &CreateTable, _: &str) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn create_index(&mut self, _: &mut dyn Pager, _: &CreateIndex, _: &str) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn drop_object(&mut self, _: &mut dyn Pager, _: &Drop) -> Result<()> {
            unimplemented!("test catalog is static")
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
            strict: false,
        }
    }

    /// A WITHOUT ROWID table (no integer rowid; rows live in the PRIMARY KEY b-tree).
    fn tdef_without_rowid(name: &str, columns: Vec<ColumnDef>) -> TableDef {
        TableDef { without_rowid: true, ..tdef(name, columns, None) }
    }

    /// An ordinary (non-expression) single-column index on `table(column)` with the given
    /// per-key `collation` override and `descending` flag.
    fn idef(name: &str, table: &str, column: &str, collation: Option<&str>, descending: bool) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: vec![column.to_string()],
            key_columns: vec![KeyColumn { collation: collation.map(str::to_string), descending }],
            key_exprs: vec![None],
            root_page: 3,
            unique: false,
            partial: false,
            partial_predicate: None,
        }
    }

    /// An ordinary (non-expression) multi-column index on `table(columns..)`, every key
    /// column ascending BINARY — used to exercise the "left-most column of an index" case.
    fn idef_cols(name: &str, table: &str, columns: &[&str]) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: columns.iter().map(|c| c.to_string()).collect(),
            key_columns: columns
                .iter()
                .map(|_| KeyColumn { collation: None, descending: false })
                .collect(),
            key_exprs: vec![None; columns.len()],
            root_page: 3,
            unique: false,
            partial: false,
            partial_predicate: None,
        }
    }

    fn plan_root(sql: &str, cat: &Cat) -> PlanNode {
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, cat).expect("plan ok").root
    }

    /// Descend the row-preserving wrappers (`Project` for the outer expression / result
    /// naming, and any `Filter`/`Sort`/`Limit`/`Distinct`) to the aggregate-or-seek node.
    fn agg_or_seek(node: &PlanNode) -> &PlanNode {
        match node {
            PlanNode::MinMaxSeek(_) | PlanNode::Aggregate(_) => node,
            PlanNode::Project { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::Distinct { input, .. } => agg_or_seek(input),
            other => panic!("expected an Aggregate or MinMaxSeek above the leaves, got {other:?}"),
        }
    }

    /// Takes the whole plan BY VALUE and returns the (owned) seek node, so callers can pass
    /// `plan_root(..)` directly without binding it to keep a borrow alive.
    #[track_caller]
    fn expect_seek(node: PlanNode) -> MinMaxSeek {
        match agg_or_seek(&node) {
            PlanNode::MinMaxSeek(m) => m.clone(),
            other => panic!("expected a MinMaxSeek (eligible), got {other:?}"),
        }
    }

    #[track_caller]
    fn expect_aggregate(node: &PlanNode) {
        match agg_or_seek(node) {
            PlanNode::Aggregate(_) => {}
            PlanNode::MinMaxSeek(m) => {
                panic!("expected the full-scan Aggregate (ineligible), got a MinMaxSeek {m:?}")
            }
            _ => unreachable!("agg_or_seek only yields Aggregate or MinMaxSeek"),
        }
    }

    /// `t(x INTEGER)` with a plain ascending BINARY index `ix` on `x`.
    fn cat_indexed_int() -> Cat {
        Cat {
            tables: vec![tdef("t", vec![col("x", Some("INTEGER"), None)], None)],
            indexes: vec![idef("ix", "t", "x", None, false)],
        }
    }

    // --- ELIGIBLE: rewritten to a MinMaxSeek --------------------------------------------

    #[test]
    fn min_over_indexed_int_is_a_forward_index_seek() {
        let cat = cat_indexed_int();
        let m = expect_seek(plan_root("SELECT MIN(x) FROM t", &cat));
        assert!(!m.is_max, "MIN seeks the smallest key");
        assert!(matches!(&m.source, MinMaxSource::Index { index } if index == "ix"));
    }

    #[test]
    fn max_over_indexed_int_is_a_reverse_index_seek() {
        let cat = cat_indexed_int();
        let m = expect_seek(plan_root("SELECT MAX(x) FROM t", &cat));
        assert!(m.is_max, "MAX seeks the largest key");
        assert!(matches!(&m.source, MinMaxSource::Index { index } if index == "ix"));
    }

    #[test]
    fn min_over_indexed_text_binary_is_a_seek() {
        // A TEXT column with an ascending BINARY index: TEXT affinity is non-BLOB and the
        // effective comparison collation is BINARY, so the index seek is byte-identical.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("s", Some("TEXT"), None)], None)],
            indexes: vec![idef("ix", "t", "s", None, false)],
        };
        assert!(matches!(expect_seek(plan_root("SELECT MIN(s) FROM t", &cat)).source, MinMaxSource::Index { .. }));
        assert!(matches!(expect_seek(plan_root("SELECT MAX(s) FROM t", &cat)).source, MinMaxSource::Index { .. }));
    }

    #[test]
    fn minmax_over_the_leftmost_column_of_a_composite_index_is_a_seek() {
        // §13 is explicitly about "the LEFT-MOST column of an index": a composite index
        // (x, y) serves MIN(x)/MAX(x) because x is its leading key column. This pins that
        // `index_leftmost_serves` accepts a multi-column index (via `.first()`), so a future
        // over-restriction (e.g. an added `columns.len() == 1` guard) is caught here.
        let cat = Cat {
            tables: vec![tdef(
                "t",
                vec![col("x", Some("INTEGER"), None), col("y", Some("INTEGER"), None)],
                None,
            )],
            indexes: vec![idef_cols("ixy", "t", &["x", "y"])],
        };
        let mn = expect_seek(plan_root("SELECT MIN(x) FROM t", &cat));
        assert!(!mn.is_max);
        assert!(matches!(&mn.source, MinMaxSource::Index { index } if index == "ixy"));
        let mx = expect_seek(plan_root("SELECT MAX(x) FROM t", &cat));
        assert!(mx.is_max);
        assert!(matches!(&mx.source, MinMaxSource::Index { index } if index == "ixy"));
    }

    #[test]
    fn minmax_over_rowid_alias_uses_the_table_btree() {
        // `id INTEGER PRIMARY KEY` aliases the rowid, so MIN/MAX(id) seek the table b-tree
        // directly — no separate index needed.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("id", Some("INTEGER"), None), col("y", None, None)], Some(0))],
            indexes: vec![],
        };
        let mn = expect_seek(plan_root("SELECT MIN(id) FROM t", &cat));
        assert!(!mn.is_max);
        assert!(matches!(mn.source, MinMaxSource::Rowid), "INTEGER PK MIN is a rowid seek");
        let mx = expect_seek(plan_root("SELECT MAX(id) FROM t", &cat));
        assert!(mx.is_max);
        assert!(matches!(mx.source, MinMaxSource::Rowid));
    }

    #[test]
    fn minmax_over_rowid_keyword_uses_the_table_btree() {
        // The bare `rowid`/`_rowid_`/`oid` keywords name the table b-tree even with no
        // INTEGER PK alias.
        let cat = cat_indexed_int();
        for kw in ["rowid", "_rowid_", "oid"] {
            let m = expect_seek(plan_root(&format!("SELECT MAX({kw}) FROM t"), &cat));
            assert!(matches!(m.source, MinMaxSource::Rowid), "{kw} MAX is a rowid seek");
        }
    }

    #[test]
    fn outer_expression_over_the_extremum_is_preserved_above_the_seek() {
        // `MAX(x)+1`: the seek computes the extremum VALUE; the `+1` and result naming stay
        // in the Project above it (the optimization changes only how the value is computed).
        let cat = cat_indexed_int();
        let root = plan_root("SELECT MAX(x)+1 FROM t", &cat);
        assert!(matches!(&root, PlanNode::Project { .. }), "outer expr keeps its Project");
        assert!(expect_seek(root).is_max);
    }

    // --- INELIGIBLE: the conservative full-scan Aggregate is kept -----------------------

    #[test]
    fn min_without_a_usable_index_stays_a_full_scan_aggregate() {
        // No index on `x` and `x` is not the rowid → nothing presents the extremum in order.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("x", Some("INTEGER"), None)], None)],
            indexes: vec![],
        };
        expect_aggregate(&plan_root("SELECT MIN(x) FROM t", &cat));
    }

    #[test]
    fn two_aggregates_are_ineligible() {
        let cat = cat_indexed_int();
        expect_aggregate(&plan_root("SELECT MAX(x), COUNT(*) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(x), MIN(x) FROM t", &cat));
    }

    #[test]
    fn expression_argument_is_ineligible() {
        let cat = cat_indexed_int();
        expect_aggregate(&plan_root("SELECT MAX(x+1) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(x*1) FROM t", &cat));
    }

    #[test]
    fn distinct_argument_is_ineligible() {
        let cat = cat_indexed_int();
        expect_aggregate(&plan_root("SELECT MAX(DISTINCT x) FROM t", &cat));
    }

    #[test]
    fn a_where_clause_is_ineligible() {
        let cat = cat_indexed_int();
        expect_aggregate(&plan_root("SELECT MAX(x) FROM t WHERE x < 5", &cat));
    }

    #[test]
    fn group_by_is_ineligible() {
        let cat = Cat {
            tables: vec![tdef("t", vec![col("x", Some("INTEGER"), None), col("y", Some("INTEGER"), None)], None)],
            indexes: vec![idef("ix", "t", "x", None, false)],
        };
        expect_aggregate(&plan_root("SELECT MAX(x) FROM t GROUP BY y", &cat));
    }

    #[test]
    fn nocase_column_is_ineligible_even_with_a_nocase_index() {
        // A NOCASE column's effective comparison collation is NOCASE, which the physically-
        // BINARY b-tree cannot present in order — decline to the (correct) NOCASE scan.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("c", Some("TEXT"), Some("NOCASE"))], None)],
            indexes: vec![idef("ix", "t", "c", Some("NOCASE"), false)],
        };
        expect_aggregate(&plan_root("SELECT MIN(c) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(c) FROM t", &cat));
    }

    #[test]
    fn desc_index_is_ineligible() {
        // A DESC-declared index is metadata over a b-tree still physically ascending; a
        // plain forward/reverse walk would not present the declared extremum first, so
        // decline (mirrors the ORDER-BY-via-index optimizer).
        let cat = Cat {
            tables: vec![tdef("t", vec![col("x", Some("INTEGER"), None)], None)],
            indexes: vec![idef("ix", "t", "x", None, true)],
        };
        expect_aggregate(&plan_root("SELECT MIN(x) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(x) FROM t", &cat));
    }

    #[test]
    fn nocase_declared_index_on_a_binary_column_is_ineligible() {
        // Even when the column compares BINARY (master guard passes), an index whose KEY
        // column is declared NOCASE is not used: its declared order is not the BINARY order
        // the scan accumulates under. Conservative — a missed opt, never a wrong answer.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("s", Some("TEXT"), None)], None)],
            indexes: vec![idef("ix", "t", "s", Some("NOCASE"), false)],
        };
        expect_aggregate(&plan_root("SELECT MIN(s) FROM t", &cat));
    }

    #[test]
    fn blob_affinity_column_is_ineligible() {
        // A BLOB/typeless column can hold two equal-comparing values in different storage
        // classes; the scan keeps the first-seen representation on a tie while an index walk
        // could surface another — not byte-identical, so decline.
        let cat = Cat {
            tables: vec![tdef("t", vec![col("b", Some("BLOB"), None)], None)],
            indexes: vec![idef("ix", "t", "b", None, false)],
        };
        expect_aggregate(&plan_root("SELECT MIN(b) FROM t", &cat));
        // A truly typeless column (no declared type) is likewise BLOB affinity.
        let cat2 = Cat {
            tables: vec![tdef("t", vec![col("z", None, None)], None)],
            indexes: vec![idef("ix", "t", "z", None, false)],
        };
        expect_aggregate(&plan_root("SELECT MAX(z) FROM t", &cat2));
    }

    #[test]
    fn partial_index_is_ineligible() {
        // A partial index does not cover every row, so its extremum is not the table's.
        let mut ix = idef("ix", "t", "x", None, false);
        ix.partial = true;
        let cat = Cat { tables: vec![tdef("t", vec![col("x", Some("INTEGER"), None)], None)], indexes: vec![ix] };
        expect_aggregate(&plan_root("SELECT MIN(x) FROM t", &cat));
    }

    #[test]
    fn a_non_leftmost_composite_column_is_ineligible() {
        // Only the LEFT-MOST key column is served: on index (x, y) the extremum of `y` is
        // NOT presented in order by a plain walk, so MIN(y)/MAX(y) decline to the scan.
        let cat = Cat {
            tables: vec![tdef(
                "t",
                vec![col("x", Some("INTEGER"), None), col("y", Some("INTEGER"), None)],
                None,
            )],
            indexes: vec![idef_cols("ixy", "t", &["x", "y"])],
        };
        expect_aggregate(&plan_root("SELECT MIN(y) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(y) FROM t", &cat));
    }

    #[test]
    fn without_rowid_table_is_ineligible() {
        // A WITHOUT ROWID table has no integer rowid and stores rows in its PRIMARY KEY
        // b-tree; the seek executor models neither that b-tree nor a WR secondary index
        // (whose trailing key is the PK, not a rowid), so decline to the correct scan — a
        // documented conservative boundary (a secondary BINARY index COULD serve it, a
        // possible future extension).
        let cat = Cat {
            tables: vec![tdef_without_rowid("t", vec![col("x", Some("INTEGER"), None)])],
            indexes: vec![idef("ix", "t", "x", None, false)],
        };
        expect_aggregate(&plan_root("SELECT MIN(x) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT MAX(x) FROM t", &cat));
    }

    #[test]
    fn count_and_sum_are_not_minmax() {
        let cat = cat_indexed_int();
        expect_aggregate(&plan_root("SELECT COUNT(x) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT SUM(x) FROM t", &cat));
        expect_aggregate(&plan_root("SELECT COUNT(*) FROM t", &cat));
    }
}
