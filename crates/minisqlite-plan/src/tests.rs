//! Behavioral tests for the binder + planner: name resolution, access-path choice,
//! projection / DISTINCT / ORDER BY / LIMIT assembly, bind-parameter numbering,
//! VALUES, comparison metadata (datatype3 §3.2/§4.2/§7.1), and the aggregate plan
//! shape. Most go end-to-end through the real SQL parser and [`QueryPlanner`]; a few
//! reach the internal [`compile_select`] entrypoint with a bespoke
//! [`FunctionRegistry`] so the aggregate shape can be exercised before the built-in
//! aggregate family lands.

use std::sync::Arc;

use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
use minisqlite_expr::{
    AggregateAccumulator, AggregateFunction, CmpOp, CompareMeta, EvalExpr, LikeKind, SortKey,
};
use minisqlite_functions::{Arity, FunctionRegistry};
use minisqlite_pager::Pager;
use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop, Statement};
use minisqlite_types::{Affinity, Collation, Error, Result, Value};

use crate::access::{RowidOp, RowidScan, SeqScan};
use crate::compile::compile_select;
use crate::plan::{CtePlan, Plan, PlanNode};
use crate::plan_ctx::PlanCtx;
use crate::{Planner, QueryPlanner};

// ---------------------------------------------------------------------------
// Test fixtures: a static in-memory catalog and small table builders.
// ---------------------------------------------------------------------------

/// A static schema store for tests: only reads are exercised; the write / load
/// paths of the [`Catalog`] seam are never called here.
struct TestCatalog {
    tables: Vec<TableDef>,
}

impl TestCatalog {
    fn new(tables: Vec<TableDef>) -> Self {
        TestCatalog { tables }
    }
}

impl Catalog for TestCatalog {
    fn table(&self, name: &str) -> Result<Option<&TableDef>> {
        Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
    }
    fn index(&self, _name: &str) -> Result<Option<&IndexDef>> {
        Ok(None)
    }
    fn indexes_on<'a>(&'a self, _table: &str) -> Result<Vec<&'a IndexDef>> {
        Ok(Vec::new())
    }
    fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
        unimplemented!("test catalog is static")
    }
    fn create_table(&mut self, _pager: &mut dyn Pager, _stmt: &CreateTable, _sql: &str) -> Result<()> {
        unimplemented!("test catalog is static")
    }
    fn create_index(&mut self, _pager: &mut dyn Pager, _stmt: &CreateIndex, _sql: &str) -> Result<()> {
        unimplemented!("test catalog is static")
    }
    fn drop_object(&mut self, _pager: &mut dyn Pager, _stmt: &Drop) -> Result<()> {
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

/// `t(a INTEGER, b TEXT, c TEXT COLLATE NOCASE)` — the common read fixture.
fn cat_t() -> TestCatalog {
    TestCatalog::new(vec![tdef(
        "t",
        vec![col("a", Some("INTEGER"), None), col("b", Some("TEXT"), None), col("c", Some("TEXT"), Some("NOCASE"))],
        None,
    )])
}

/// `m(i INTEGER, t TEXT, r REAL, n NUMERIC, b BLOB, x)` — one column per datatype3
/// §3.1 affinity, plus a typeless `x` (BLOB affinity by default, like `b`). Every column
/// keeps the default BINARY collation so this fixture isolates the §4.2 apply-affinity
/// rule from collation. `b` and `x` both have BLOB affinity (§3.1) — a real column affinity
/// that §4.2's comparison rule treats as the same "no affinity" class as a bare expression
/// (see `Aff`).
fn cat_m() -> TestCatalog {
    TestCatalog::new(vec![tdef(
        "m",
        vec![
            col("i", Some("INTEGER"), None),
            col("t", Some("TEXT"), None),
            col("r", Some("REAL"), None),
            col("n", Some("NUMERIC"), None),
            col("b", Some("BLOB"), None),
            col("x", None, None),
        ],
        None,
    )])
}

/// `p(u TEXT, v TEXT COLLATE NOCASE, w TEXT COLLATE RTRIM)` — one column per built-in
/// collation (§7.1). All three are TEXT affinity so this fixture isolates the §7.1
/// collation-precedence rule from the apply-affinity rule.
fn cat_coll() -> TestCatalog {
    TestCatalog::new(vec![tdef(
        "p",
        vec![
            col("u", Some("TEXT"), None),
            col("v", Some("TEXT"), Some("NOCASE")),
            col("w", Some("TEXT"), Some("RTRIM")),
        ],
        None,
    )])
}

fn plan_sql(sql: &str, cat: &dyn Catalog) -> Result<Plan> {
    let ast = parse(sql)?;
    let stmt = ast.statements.first().expect("expected one statement");
    QueryPlanner::new().plan(stmt, cat)
}

// ---------------------------------------------------------------------------
// Small structural matchers.
// ---------------------------------------------------------------------------

fn expect_project(n: &PlanNode) -> (&PlanNode, &[EvalExpr]) {
    match n {
        PlanNode::Project { input, exprs } => (input, exprs.as_slice()),
        other => panic!("expected Project, got {other:?}"),
    }
}

fn expect_seqscan(n: &PlanNode) -> &SeqScan {
    match n {
        PlanNode::SeqScan(s) => s,
        other => panic!("expected SeqScan, got {other:?}"),
    }
}

fn expect_rowidscan(n: &PlanNode) -> &RowidScan {
    match n {
        PlanNode::RowidScan(s) => s,
        other => panic!("expected RowidScan, got {other:?}"),
    }
}

fn projection_of(sql: &str, cat: &dyn Catalog) -> EvalExpr {
    let plan = plan_sql(sql, cat).expect("plan ok");
    let (_input, exprs) = expect_project(&plan.root);
    assert_eq!(exprs.len(), 1, "expected a single projected expr for {sql}");
    exprs[0].clone()
}

/// Descend the row-preserving wrappers to the first `Sort` node's keys (a hidden
/// ORDER BY expression puts a strip `Project` above the `Sort`).
fn find_sort_keys(node: &PlanNode) -> &[SortKey] {
    match node {
        PlanNode::Sort { keys, .. } => keys.as_slice(),
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Distinct { input, .. }
        | PlanNode::Limit { input, .. } => find_sort_keys(input),
        other => panic!("no Sort node found, reached {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Projection / scan / filter.
// ---------------------------------------------------------------------------

#[test]
fn select_literals_with_no_from_is_project_over_singlerow() {
    let cat = TestCatalog::new(vec![]);
    let plan = plan_sql("SELECT 1, 'a'", &cat).unwrap();
    let (input, exprs) = expect_project(&plan.root);
    assert!(matches!(input, PlanNode::SingleRow));
    assert_eq!(exprs.len(), 2);
    assert_eq!(plan.result_columns.len(), 2);
}

#[test]
fn project_columns_over_seqscan() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a, b FROM t", &cat).unwrap();
    let (input, exprs) = expect_project(&plan.root);
    assert!(matches!(exprs[0], EvalExpr::Column(0)));
    assert!(matches!(exprs[1], EvalExpr::Column(1)));
    assert_eq!(expect_seqscan(input).column_count, 3);
    assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn non_rowid_where_becomes_filter_over_seqscan() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t WHERE b = 'x'", &cat).unwrap();
    let (input, _exprs) = expect_project(&plan.root);
    match input {
        PlanNode::Filter { input, predicate } => {
            assert!(
                matches!(predicate, EvalExpr::Compare { op: CmpOp::Eq, null_safe: false, .. }),
                "plain `=` is a non-null-safe Eq compare"
            );
            expect_seqscan(input);
        }
        other => panic!("expected Filter over SeqScan, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Three-valued logic / NULL binding (datatype3 NULL handling).
//
// The parser folds `a IS NULL` into `Binary{Is, .., NULL}` (→ a null-safe Compare),
// while the postfix `ISNULL` / `NOTNULL` keywords produce dedicated IsNull / NotNull
// nodes. Each form is pinned to its exact node so a 3VL regression (e.g. dropping
// `null_safe`, or swapping the IS/IS NOT operator) cannot pass silently.
// ---------------------------------------------------------------------------

#[test]
fn plain_equality_is_a_non_null_safe_eq() {
    let cat = cat_t();
    match projection_of("SELECT a = 1 FROM t", &cat) {
        EvalExpr::Compare { op: CmpOp::Eq, null_safe, .. } => {
            assert!(!null_safe, "`=` must not be null-safe");
        }
        other => panic!("expected a Compare Eq, got {other:?}"),
    }
}

#[test]
fn isnull_keyword_lowers_to_isnull_node() {
    let cat = cat_t();
    assert!(matches!(projection_of("SELECT a ISNULL FROM t", &cat), EvalExpr::IsNull(_)));
}

#[test]
fn notnull_keyword_lowers_to_notnull_node() {
    let cat = cat_t();
    assert!(matches!(projection_of("SELECT a NOTNULL FROM t", &cat), EvalExpr::NotNull(_)));
}

#[test]
fn is_operator_binds_a_null_safe_eq() {
    let cat = cat_t();
    // `x IS y` is a null-safe equality (NULL IS NULL → 1, never NULL).
    match projection_of("SELECT a IS b FROM t", &cat) {
        EvalExpr::Compare { op: CmpOp::Eq, null_safe, .. } => {
            assert!(null_safe, "`IS` must be null-safe");
        }
        other => panic!("expected a null-safe Compare Eq, got {other:?}"),
    }
}

#[test]
fn is_not_operator_binds_a_null_safe_ne() {
    let cat = cat_t();
    match projection_of("SELECT a IS NOT b FROM t", &cat) {
        EvalExpr::Compare { op: CmpOp::Ne, null_safe, .. } => {
            assert!(null_safe, "`IS NOT` must be null-safe");
        }
        other => panic!("expected a null-safe Compare Ne, got {other:?}"),
    }
}

#[test]
fn is_null_is_a_null_safe_eq_against_null() {
    let cat = cat_t();
    // `a IS NULL` parses as `Binary{Is, a, NULL}`, which lowers to a null-safe Eq —
    // the semantics that make `NULL IS NULL` yield 1 rather than NULL.
    match projection_of("SELECT a IS NULL FROM t", &cat) {
        EvalExpr::Compare { op: CmpOp::Eq, null_safe, .. } => {
            assert!(null_safe, "`IS NULL` must be null-safe");
        }
        other => panic!("expected a null-safe Compare Eq, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Rowid access paths.
// ---------------------------------------------------------------------------

#[test]
fn rowid_eq_is_a_rowid_seek_with_no_residual() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t WHERE rowid = 5", &cat).unwrap();
    let (input, _exprs) = expect_project(&plan.root);
    let scan = expect_rowidscan(input);
    assert!(matches!(scan.op, RowidOp::Eq(_)), "expected rowid Eq seek");
    assert_eq!(scan.column_count, 3);
}

#[test]
fn integer_primary_key_eq_is_a_rowid_seek() {
    let cat = TestCatalog::new(vec![tdef(
        "u",
        vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)],
        Some(0),
    )]);
    let plan = plan_sql("SELECT v FROM u WHERE id = 42", &cat).unwrap();
    let (input, _exprs) = expect_project(&plan.root);
    assert!(matches!(expect_rowidscan(input).op, RowidOp::Eq(_)));
}

#[test]
fn rowid_range_is_a_rowid_range_scan() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t WHERE rowid >= 10", &cat).unwrap();
    let (input, _exprs) = expect_project(&plan.root);
    match &expect_rowidscan(input).op {
        RowidOp::Range { lo, hi } => {
            let lo = lo.as_ref().expect("a lower bound");
            assert!(lo.inclusive, ">= is an inclusive lower bound");
            assert!(hi.is_none(), "no upper bound");
        }
        other => panic!("expected rowid Range, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Star expansion.
// ---------------------------------------------------------------------------

#[test]
fn star_expands_all_columns_in_schema_order() {
    let cat = cat_t();
    let plan = plan_sql("SELECT * FROM t", &cat).unwrap();
    let (_input, exprs) = expect_project(&plan.root);
    assert!(matches!(exprs[0], EvalExpr::Column(0)));
    assert!(matches!(exprs[1], EvalExpr::Column(1)));
    assert!(matches!(exprs[2], EvalExpr::Column(2)));
    assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
}

// ---------------------------------------------------------------------------
// Bind-parameter numbering (lang_expr.html "Parameters").
// ---------------------------------------------------------------------------

fn param_indices(sql: &str) -> Vec<usize> {
    let cat = TestCatalog::new(vec![]);
    let plan = plan_sql(sql, &cat).unwrap();
    let (_input, exprs) = expect_project(&plan.root);
    exprs
        .iter()
        .map(|e| match e {
            EvalExpr::Param(i) => *i,
            other => panic!("expected a Param, got {other:?}"),
        })
        .collect()
}

#[test]
fn named_params_reuse_their_number() {
    assert_eq!(param_indices("SELECT :x, :y, :x"), vec![1, 2, 1]);
}

#[test]
fn mixed_anonymous_and_numbered_params() {
    // `?` = max+1 = 1; `?5` raises the max to 5; the trailing `?` = max+1 = 6.
    assert_eq!(param_indices("SELECT ?, ?5, ?"), vec![1, 5, 6]);
}

// ---------------------------------------------------------------------------
// VALUES.
// ---------------------------------------------------------------------------

#[test]
fn values_two_rows_makes_a_values_node() {
    let cat = TestCatalog::new(vec![]);
    let plan = plan_sql("VALUES (1, 2), (3, 4)", &cat).unwrap();
    match &plan.root {
        PlanNode::Values { rows } => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        other => panic!("expected Values, got {other:?}"),
    }
    assert_eq!(plan.result_columns, vec!["column1".to_string(), "column2".to_string()]);
}

// ---------------------------------------------------------------------------
// ORDER BY / LIMIT / DISTINCT assembly.
// ---------------------------------------------------------------------------

#[test]
fn order_by_output_column_then_limit() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t ORDER BY a LIMIT 5", &cat).unwrap();
    // Limit { Sort { Project { SeqScan } } }
    match &plan.root {
        PlanNode::Limit { input, limit, offset } => {
            assert!(limit.is_some());
            assert!(offset.is_none());
            match input.as_ref() {
                PlanNode::Sort { input, keys, limit } => {
                    assert_eq!(keys.len(), 1);
                    assert!(matches!(keys[0].expr, EvalExpr::Column(0)));
                    assert!(!keys[0].desc);
                    // The deterministic `LIMIT 5` is also pushed onto the Sort as its
                    // top-k retention bound, so it keeps only the first 5 rows instead of
                    // materializing the whole input; the Limit node above still does the
                    // real take.
                    match limit {
                        Some(sl) => {
                            assert!(
                                matches!(sl.limit, EvalExpr::Literal(Value::Integer(5))),
                                "Sort retention limit binds 5, got {:?}",
                                sl.limit
                            );
                            assert!(sl.offset.is_none(), "no OFFSET, got {:?}", sl.offset);
                        }
                        None => panic!("expected the Sort to carry the LIMIT retention bound"),
                    }
                    expect_project(input);
                }
                other => panic!("expected Sort under Limit, got {other:?}"),
            }
        }
        other => panic!("expected Limit at the root, got {other:?}"),
    }
}

#[test]
fn order_by_nondeterministic_limit_leaves_the_sort_unbounded() {
    // The top-k retention bound is attached to the Sort ONLY when BOTH the LIMIT and the
    // OFFSET are deterministic (`compile::select::sort_limit_from` /
    // `limit_expr_is_deterministic`). This is a CORRECTNESS gate, not just an optimization
    // choice: the Sort evaluates the bound SEPARATELY from the Limit node above it, so a
    // non-deterministic bound (a fresh RNG draw, a wall-clock read, a subquery) could make
    // the sort retain FEWER rows than the Limit ultimately takes — silently dropping answer
    // rows. So a non-deterministic bound must leave the Sort a FULL stable sort
    // (`limit: None`), while the Limit node still carries the bound and remains the single
    // source of truth for skip/take. If the determinism classifier ever regressed (e.g.
    // started treating `random()` as deterministic) every other top-k test would still pass
    // and this is the only guard that would fail.
    //
    // Each form exercises one rejection branch: `random()` a registered non-deterministic
    // builtin (`Func { deterministic() == false }`), `CURRENT_TIMESTAMP` the structurally
    // non-deterministic `EvalExpr::Now` (independent of any function registry), and the last
    // pins that a non-deterministic OFFSET alone disables the bound even when the LIMIT is a
    // constant.
    let cat = cat_t();
    for sql in [
        "SELECT a FROM t ORDER BY a LIMIT random()",
        "SELECT a FROM t ORDER BY a LIMIT CURRENT_TIMESTAMP",
        "SELECT a FROM t ORDER BY a LIMIT 5 OFFSET random()",
    ] {
        match &plan_sql(sql, &cat).unwrap().root {
            PlanNode::Limit { input, limit, .. } => {
                assert!(limit.is_some(), "{sql}: the Limit node itself always carries its bound");
                match input.as_ref() {
                    PlanNode::Sort { limit, .. } => assert!(
                        limit.is_none(),
                        "{sql}: a non-deterministic LIMIT/OFFSET must leave the sort unbounded, got {limit:?}"
                    ),
                    other => panic!("{sql}: expected a Sort directly under the Limit, got {other:?}"),
                }
            }
            other => panic!("{sql}: expected Limit at the root, got {other:?}"),
        }
    }
}

#[test]
fn order_by_ordinal_inherits_the_referenced_columns_collation() {
    // The referenced output column `c` is TEXT COLLATE NOCASE. An ORDER BY *ordinal*
    // (whose term text is a bare integer with no collation) must still sort by that
    // column's NOCASE collation, not fall back to BINARY.
    let cat = cat_t();
    match &plan_sql("SELECT c FROM t ORDER BY 1", &cat).unwrap().root {
        PlanNode::Sort { keys, .. } => {
            assert_eq!(keys.len(), 1);
            assert_eq!(keys[0].collation, Collation::NoCase, "ORDER BY 1 must inherit c's NOCASE");
        }
        other => panic!("expected Sort, got {other:?}"),
    }
    // Naming the column directly yields the same collation (parity check).
    match &plan_sql("SELECT c FROM t ORDER BY c", &cat).unwrap().root {
        PlanNode::Sort { keys, .. } => assert_eq!(keys[0].collation, Collation::NoCase),
        other => panic!("expected Sort, got {other:?}"),
    }
}

#[test]
fn order_by_output_alias_inherits_the_referenced_columns_collation() {
    // ORDER BY the alias `z` of c (COLLATE NOCASE) sorts by c's collation.
    let cat = cat_t();
    match &plan_sql("SELECT c AS z FROM t ORDER BY z", &cat).unwrap().root {
        PlanNode::Sort { keys, .. } => assert_eq!(keys[0].collation, Collation::NoCase),
        other => panic!("expected Sort, got {other:?}"),
    }
}

#[test]
fn order_by_star_column_inherits_its_collation() {
    // A `*`-expanded output column must also contribute its collation to ORDER BY —
    // by ordinal and by name. Output column 3 / `c` is TEXT COLLATE NOCASE, so both
    // sort keys must be NOCASE (not the BINARY a star column previously defaulted to).
    let cat = cat_t();
    for sql in ["SELECT * FROM t ORDER BY 3", "SELECT * FROM t ORDER BY c"] {
        match &plan_sql(sql, &cat).unwrap().root {
            PlanNode::Sort { keys, .. } => {
                assert_eq!(keys[0].collation, Collation::NoCase, "{sql}: star column c is NOCASE");
            }
            other => panic!("{sql}: expected Sort, got {other:?}"),
        }
    }
}

#[test]
fn order_by_explicit_collate_overrides_the_referenced_column() {
    // An explicit COLLATE on the ORDER BY term wins over the column's own collation.
    let cat = cat_t();
    let plan = plan_sql("SELECT c FROM t ORDER BY c COLLATE RTRIM", &cat).unwrap();
    assert_eq!(find_sort_keys(&plan.root)[0].collation, Collation::Rtrim);
}

#[test]
fn order_by_non_output_expr_appends_hidden_column_then_strips_it() {
    let cat = cat_t();
    // ORDER BY b (not selected): b is bound as a hidden projection column, sorted
    // on, then stripped back to just `a`.
    let plan = plan_sql("SELECT a FROM t ORDER BY b DESC", &cat).unwrap();
    let (strip_input, strip_exprs) = expect_project(&plan.root);
    assert_eq!(strip_exprs.len(), 1, "outer strip keeps only the SELECT output");
    assert!(matches!(strip_exprs[0], EvalExpr::Column(0)));
    match strip_input {
        PlanNode::Sort { input, keys, limit } => {
            assert_eq!(keys.len(), 1);
            assert!(keys[0].desc);
            // No LIMIT on this query, so the Sort carries no retention bound: it stays a
            // full stable sort (the else-branch of the top-k optimization).
            assert!(limit.is_none(), "ORDER BY without LIMIT leaves the sort unbounded");
            // The sort key reads the hidden column appended after the output (reg 1).
            assert!(matches!(keys[0].expr, EvalExpr::Column(1)));
            let (_scan, wide) = expect_project(input);
            assert_eq!(wide.len(), 2, "inner project carries the hidden column");
        }
        other => panic!("expected Sort under the strip Project, got {other:?}"),
    }
}

#[test]
fn distinct_wraps_the_projection() {
    let cat = cat_t();
    let plan = plan_sql("SELECT DISTINCT a FROM t", &cat).unwrap();
    match &plan.root {
        PlanNode::Distinct { input, .. } => {
            expect_project(input);
        }
        other => panic!("expected Distinct, got {other:?}"),
    }
}

#[test]
fn distinct_with_order_by_hidden_column_is_an_error() {
    let cat = cat_t();
    let err = plan_sql("SELECT DISTINCT a FROM t ORDER BY b", &cat).unwrap_err();
    assert!(matches!(err, Error::Sql(_)), "expected a SQL error, got {err:?}");
}

// ---------------------------------------------------------------------------
// Comparison metadata (datatype3 §3.2 affinity, §4.2 apply-rule, §7.1 collation).
// ---------------------------------------------------------------------------

fn compare_meta_of(expr: &EvalExpr) -> CompareMeta {
    match expr {
        EvalExpr::Compare { meta, .. } => *meta,
        other => panic!("expected a Compare, got {other:?}"),
    }
}

#[test]
fn numeric_column_vs_text_literal_applies_numeric_to_the_literal() {
    let cat = cat_t();
    // a INTEGER (numeric affinity) vs '5' (no affinity) → NUMERIC applied to the RHS.
    let meta = compare_meta_of(&projection_of("SELECT a = '5' FROM t", &cat));
    assert_eq!(meta.apply_left, None);
    assert_eq!(meta.apply_right, Some(Affinity::Numeric));
}

#[test]
fn text_column_vs_bare_literal_applies_text_to_the_literal() {
    let cat = cat_t();
    // b TEXT vs 5 (no affinity) → TEXT applied to the RHS.
    let meta = compare_meta_of(&projection_of("SELECT b = 5 FROM t", &cat));
    assert_eq!(meta.apply_left, None);
    assert_eq!(meta.apply_right, Some(Affinity::Text));
}

#[test]
fn two_literals_get_no_affinity() {
    let cat = TestCatalog::new(vec![]);
    let meta = compare_meta_of(&projection_of("SELECT 1 = 2", &cat));
    assert_eq!(meta.apply_left, None);
    assert_eq!(meta.apply_right, None);
    assert_eq!(meta.collation, Collation::Binary);
}

#[test]
fn column_collation_flows_into_the_comparison() {
    let cat = cat_t();
    // c TEXT COLLATE NOCASE vs a literal → the column's NOCASE collation is used.
    let meta = compare_meta_of(&projection_of("SELECT c = 'x' FROM t", &cat));
    assert_eq!(meta.collation, Collation::NoCase);
}

#[test]
fn explicit_collate_on_the_right_operand_wins_over_default() {
    let cat = cat_t();
    // b has BINARY; an explicit COLLATE RTRIM on the RHS wins (§7.1 rule 2).
    let meta = compare_meta_of(&projection_of("SELECT b = 'x' COLLATE RTRIM FROM t", &cat));
    assert_eq!(meta.collation, Collation::Rtrim);
}

#[test]
fn column_vs_column_collation_takes_the_left_operand() {
    // §7.1 rule 3: with a column on each side and no explicit COLLATE, the LEFT
    // operand's collation wins. c is NOCASE, b is BINARY — so the two orderings
    // must disagree, which pins the left-vs-right precedence (a right-first probe
    // would give the wrong answer for at least one of them).
    let cat = cat_t();
    let left_nocase = compare_meta_of(&projection_of("SELECT c = b FROM t", &cat));
    assert_eq!(left_nocase.collation, Collation::NoCase, "left c (NOCASE) wins");
    let left_binary = compare_meta_of(&projection_of("SELECT b = c FROM t", &cat));
    assert_eq!(left_binary.collation, Collation::Binary, "left b (BINARY) wins");
}

// ---------------------------------------------------------------------------
// Result-column naming (lang_select.html result-column naming).
// ---------------------------------------------------------------------------

#[test]
fn explicit_alias_names_the_column() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a AS x FROM t", &cat).unwrap();
    assert_eq!(plan.result_columns, vec!["x".to_string()]);
}

#[test]
fn qualified_column_is_named_by_its_bare_column_name() {
    let cat = cat_t();
    // `t.b` is named `b` (the qualifier is dropped), matching SQLite.
    let plan = plan_sql("SELECT t.b FROM t", &cat).unwrap();
    assert_eq!(plan.result_columns, vec!["b".to_string()]);
}

#[test]
fn table_star_expands_that_tables_columns() {
    let cat = cat_t();
    let plan = plan_sql("SELECT t.* FROM t", &cat).unwrap();
    let (_input, exprs) = expect_project(&plan.root);
    assert_eq!(exprs.len(), 3);
    assert!(matches!(exprs[0], EvalExpr::Column(0)));
    assert!(matches!(exprs[2], EvalExpr::Column(2)));
    assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
}

// ---------------------------------------------------------------------------
// Resolution errors.
// ---------------------------------------------------------------------------

#[test]
fn unknown_table_is_a_loud_error() {
    let cat = cat_t();
    let err = plan_sql("SELECT * FROM missing", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("no such table"), "got {m:?}"),
        other => panic!("expected Sql error, got {other:?}"),
    }
}

#[test]
fn unknown_column_is_a_loud_error() {
    let cat = cat_t();
    let err = plan_sql("SELECT nope FROM t", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("no such column"), "got {m:?}"),
        other => panic!("expected Sql error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Subquery arity. A subquery used in a single-value context (a scalar `(SELECT …)`
// or the right side of a scalar `x IN (SELECT …)`) must return exactly one column,
// matching SQLite's "sub-select returns N columns - expected 1". EXISTS ignores the
// result columns, so any width is accepted there.
// ---------------------------------------------------------------------------

#[test]
fn multi_column_in_subquery_is_a_loud_error() {
    let cat = cat_t();
    let err = plan_sql("SELECT a FROM t WHERE a IN (SELECT a, b FROM t)", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("sub-select returns 2 columns"), "got {m:?}"),
        other => panic!("expected a sub-select arity error, got {other:?}"),
    }
}

#[test]
fn multi_column_scalar_subquery_is_a_loud_error() {
    let cat = cat_t();
    let err = plan_sql("SELECT (SELECT a, b FROM t) FROM t", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("sub-select returns 2 columns"), "got {m:?}"),
        other => panic!("expected a sub-select arity error, got {other:?}"),
    }
}

#[test]
fn single_column_in_subquery_is_accepted() {
    // The arity guard must not reject the valid one-column shape.
    let cat = cat_t();
    assert!(plan_sql("SELECT a FROM t WHERE a IN (SELECT a FROM t)", &cat).is_ok());
}

#[test]
fn exists_ignores_subquery_arity() {
    // EXISTS cares only whether a row exists, so a wide (even `SELECT *`) subquery
    // is accepted — the arity guard is scoped to value subqueries.
    let cat = cat_t();
    assert!(plan_sql("SELECT a FROM t WHERE EXISTS (SELECT a, b FROM t)", &cat).is_ok());
    assert!(plan_sql("SELECT a FROM t WHERE EXISTS (SELECT * FROM t)", &cat).is_ok());
}

// ---------------------------------------------------------------------------
// Aggregates. Function resolution goes straight through the registry, so an
// aggregate the registry does not know is an honest "no such function" (never a
// hidden stub). The plan SHAPE is then exercised with a bespoke registry. Both
// use an explicit registry via the internal entrypoint so they do not depend on
// which families `builtins()` happens to register (that set grows over time).
// ---------------------------------------------------------------------------

#[test]
fn unregistered_aggregate_is_an_honest_no_such_function() {
    // `count(*)` is always classified as an aggregate; against a registry with no
    // aggregates it must surface the registry's "no such function" verbatim,
    // rather than being papered over by a stub in the binder.
    let cat = cat_t();
    let reg = FunctionRegistry::empty();
    let ast = parse("SELECT count(*) FROM t").unwrap();
    let sel = select_of(&ast.statements[0]);
    let mut ctx = PlanCtx::new(&reg, &cat);
    match compile_select(&mut ctx, sel) {
        Err(Error::Sql(m)) => assert!(m.contains("no such function"), "got {m:?}"),
        other => panic!("expected a no-such-function error, got {other:?}"),
    }
}

/// A do-nothing aggregate: enough to bake a resolved handle into the plan without
/// depending on the (later) real aggregate family. Never executed here.
#[derive(Debug)]
struct StubAgg;
impl AggregateFunction for StubAgg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        unimplemented!("plan-shape test does not execute the aggregate")
    }
}

fn stub_agg_registry() -> FunctionRegistry {
    let mut r = FunctionRegistry::empty();
    r.add_aggregate("count", Arity::Any, Arc::new(StubAgg));
    r
}

fn select_of(stmt: &Statement) -> &minisqlite_sql::Select {
    match stmt {
        Statement::Select(s) => s,
        other => panic!("expected a SELECT, got {other:?}"),
    }
}

#[test]
fn count_star_makes_an_aggregate_node() {
    let cat = cat_t();
    let reg = stub_agg_registry();
    let ast = parse("SELECT count(*) FROM t").unwrap();
    let sel = select_of(&ast.statements[0]);
    let mut ctx = PlanCtx::new(&reg, &cat);
    let (root, names) = compile_select(&mut ctx, sel).unwrap();

    let (input, exprs) = expect_project(&root);
    assert_eq!(exprs.len(), 1);
    assert!(matches!(exprs[0], EvalExpr::Column(0)), "count result is the first output register");
    match input {
        PlanNode::Aggregate(agg) => {
            assert!(agg.group_by.is_empty());
            assert_eq!(agg.aggregates.len(), 1);
        }
        other => panic!("expected an Aggregate under the Project, got {other:?}"),
    }
    assert_eq!(names.len(), 1);
}

#[test]
fn group_by_key_then_aggregate_result_layout() {
    let cat = cat_t();
    let reg = stub_agg_registry();
    let ast = parse("SELECT a, count(*) FROM t GROUP BY a").unwrap();
    let sel = select_of(&ast.statements[0]);
    let mut ctx = PlanCtx::new(&reg, &cat);
    let (root, _names) = compile_select(&mut ctx, sel).unwrap();

    let (input, exprs) = expect_project(&root);
    // Post-aggregate layout is [group_key_0, agg_result_0]; the projection reads
    // register 0 (the key) and register 1 (the aggregate).
    assert!(matches!(exprs[0], EvalExpr::Column(0)));
    assert!(matches!(exprs[1], EvalExpr::Column(1)));
    // A no-ORDER-BY GROUP BY now interposes an implicit ascending group-key Sort BETWEEN the
    // Aggregate and the Project (real SQLite groups via a sort, so groups come back sorted by
    // key) — the same plan-shape ripple the compound implicit-sort introduced for SetOp. The
    // Sort's single key is the group-key column (Column(0), ascending); peel it to assert the
    // underlying Aggregate layout. The wrap/no-wrap decision itself is pinned by the
    // compile::aggregate implicit-group-sort unit tests.
    match input {
        PlanNode::Sort { input: agg_node, keys, .. } => {
            assert_eq!(keys.len(), 1, "one implicit group-key sort key");
            assert!(matches!(keys[0].expr, EvalExpr::Column(0)));
            assert!(!keys[0].desc, "implicit group sort is ascending");
            match &**agg_node {
                PlanNode::Aggregate(agg) => {
                    assert_eq!(agg.group_by.len(), 1);
                    assert_eq!(agg.aggregates.len(), 1);
                }
                other => panic!("expected an Aggregate under the implicit Sort, got {other:?}"),
            }
        }
        other => panic!("expected an implicit Sort wrapping the Aggregate, got {other:?}"),
    }
}

// ===========================================================================
// EXHAUSTIVE comparison metadata (datatype3 §3.2 / §4.2 / §7.1).
//
// These extend the basic CompareMeta cases above with the full apply-affinity
// matrix, the collation-precedence ladder, and the per-branch metadata carried by
// BETWEEN / IN / CASE / NULLIF. Affinity and collation are the #1 source of
// disagreements with real SQLite, so every case is derived directly
// from datatype3.html and cites the rule it pins.
// ===========================================================================

// ---------------------------------------------------------------------------
// The apply-affinity matrix (§3.2 operand affinity → §4.2 apply-rule).
//
// `m` supplies one column of each §3.1 affinity (i/t/r/n = INTEGER/TEXT/REAL/NUMERIC,
// b/x = BLOB — a real column affinity, DISTINCT from a no-affinity expression, see `Aff`
// below). The operand set covers every affinity class a comparison side can have: a column
// of each affinity, an integer literal, a text literal, and `+i` (a no-affinity expression
// over a numeric column — §3.2: unary `+` strips affinity). All 81 ordered pairs are checked
// against a literal transcription of the §4.2 rule, plus three rule-independent invariants
// that hold on every pair.
// ---------------------------------------------------------------------------

/// The affinity *classes* the §3.1/§4.2 rules range over: numeric (INTEGER/REAL/NUMERIC)
/// collapse together, TEXT is its own, a BLOB/typeless COLUMN is `Blob`, and a genuine
/// no-affinity EXPRESSION is `NoAff`.
///
/// SUBTLE (the recurring regression — read before "fixing" the matrix): for the §4.2
/// COMPARISON rules, `Blob` and `NoAff` are the SAME class. datatype3 §3.1
/// (datatype3.html:294) records that BLOB affinity "used to be called 'NONE'", and §4.3's
/// worked example labels its `c BLOB` and typeless `d` columns `-- no affinity`
/// (datatype3.html:671-672). So rule (ii) ("TEXT vs no affinity → TEXT to the other") fires
/// against BOTH a bare expression AND a BLOB/typeless column: `TEXT_col = BLOB_col` applies
/// TEXT, exactly like `TEXT_col = <literal>`. `Blob` is kept as its own variant ONLY because
/// §3.1 gives such a column a real BLOB affinity; the §4.2 rule below then unifies it with
/// `NoAff`. Real sqlite matches this — its expression affinity has no separate "none" (a
/// no-affinity expression reports BLOB affinity), so the two are one class. Pinned by
/// `conformance_comparison_affinity`'s `a<d,a=d,a>d` = `[0,1,0]` on the §4.3 fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aff {
    /// INTEGER / REAL / NUMERIC affinity.
    Num,
    /// TEXT affinity.
    Txt,
    /// BLOB affinity: a BLOB-declared or typeless COLUMN (§3.1). For the §4.2 comparison
    /// rules this is the SAME class as `NoAff` ("no affinity"/NONE — §3.1/§4.3).
    Blob,
    /// No affinity: a bare EXPRESSION (literal, `+col`, arithmetic, function call).
    NoAff,
}

/// datatype3 §4.2, transcribed literally and applied in order. `Blob` and `NoAff` are ONE
/// class for these rules (§3.1/§4.3 — see `Aff`), so rule (ii) below tests both:
///  (i)   (datatype3.html:640-642): a numeric operand vs a "TEXT or BLOB or no affinity"
///        operand → NUMERIC applied to the *other* operand.
///  (ii)  (datatype3.html:644-645): a TEXT operand vs a BLOB/no-affinity operand → TEXT
///        applied to the *other* operand. (A TEXT-vs-TEXT pair needs no conversion → (iii).)
///  (iii) (datatype3.html:647-648): otherwise, nothing.
/// At most one side is ever converted, so at most one field is `Some`.
///
/// This is an INDEPENDENT §4.2 transcription (structured differently from the binder's
/// `apply_affinity_rule`), backstopped by the hand-derived corners in
/// `apply_affinity_named_cases_from_datatype3` and `text_column_vs_blob_column_applies_text_rule_ii`.
/// Do NOT delete those corners to "simplify" — they are what keeps this matrix from being a
/// circular change-detector.
fn expected_apply(l: Aff, r: Aff) -> (Option<Affinity>, Option<Affinity>) {
    if l == Aff::Num && matches!(r, Aff::Txt | Aff::Blob | Aff::NoAff) {
        (None, Some(Affinity::Numeric))
    } else if r == Aff::Num && matches!(l, Aff::Txt | Aff::Blob | Aff::NoAff) {
        (Some(Affinity::Numeric), None)
    // rule (ii): a BLOB/typeless column is "no affinity" here (§3.1/§4.3), so TEXT vs
    // either `Blob` or `NoAff` applies TEXT to that side.
    } else if l == Aff::Txt && matches!(r, Aff::Blob | Aff::NoAff) {
        (None, Some(Affinity::Text))
    } else if r == Aff::Txt && matches!(l, Aff::Blob | Aff::NoAff) {
        (Some(Affinity::Text), None)
    } else {
        (None, None)
    }
}

/// The comparison operands the matrix ranges over, each paired with its §4.2 comparison
/// class (this SQL→class mapping is hand-authored ground truth). `b`/`x` are BLOB/typeless
/// columns whose BLOB affinity (§3.1) is a real column affinity — `Aff::Blob` — which the
/// §4.2 comparison rule treats as the same "no affinity" class as the bare expressions
/// `5`/`'5'`/`+i` (`NoAff`).
fn matrix_operands() -> Vec<(&'static str, Aff)> {
    vec![
        ("i", Aff::Num),     // INTEGER column
        ("r", Aff::Num),     // REAL column
        ("n", Aff::Num),     // NUMERIC column
        ("t", Aff::Txt),     // TEXT column
        ("b", Aff::Blob),    // BLOB column: BLOB affinity (§3.1) = "no affinity" for §4.2 compares
        ("x", Aff::Blob),    // typeless column: BLOB affinity too (§3.1 rule 3)
        ("5", Aff::NoAff),   // integer literal: a no-affinity expression
        ("'5'", Aff::NoAff), // text literal: a no-affinity expression
        ("+i", Aff::NoAff),  // unary + over a numeric column: §3.2 strips affinity
    ]
}

#[test]
fn apply_affinity_matrix_matches_datatype3_section_4_2() {
    // Every one of the 81 ordered operand pairs is checked against `expected_apply`, an
    // independent §4.2 transcription. Mismatches are collected (not `assert_eq!`-per-cell) so
    // a divergence reports EVERY offending cell at once with its SQL, and all correct cells
    // are still verified. This should fully pass: `expected_apply` and the binder agree
    // on all 81 pairs, including `TEXT-col vs BLOB-col` where rule (ii) applies TEXT to the
    // BLOB/typeless side (§4.2 — a BLOB column IS "no affinity" for these compares, see `Aff`).
    let cat = cat_m();
    let ops = matrix_operands();
    let mut mismatches = Vec::new();
    for &(ltext, laff) in &ops {
        for &(rtext, raff) in &ops {
            let sql = format!("SELECT {ltext} = {rtext} FROM m");
            let meta = compare_meta_of(&projection_of(&sql, &cat));
            let got = (meta.apply_left, meta.apply_right);
            let expected = expected_apply(laff, raff);
            if got != expected {
                mismatches.push(format!(
                    "  `{sql}` (l={laff:?}, r={raff:?}): datatype3 §4.2 says {expected:?}, binder gave {got:?}"
                ));
            }

            // Invariant A: §4.2 converts "the other operand", never both — so at most
            // one side is ever `Some`.
            assert!(
                !(meta.apply_left.is_some() && meta.apply_right.is_some()),
                "both sides converted for `{sql}`"
            );

            // Invariant B: no operand here carries a non-BINARY collation, so the
            // comparison collation is always BINARY (§7.1 R2 over BINARY columns / R3).
            assert_eq!(meta.collation, Collation::Binary, "collation for `{sql}`");
        }
    }
    assert!(
        mismatches.is_empty(),
        "apply-affinity divergences from datatype3 §4.2 ({} cell(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn apply_affinity_rule_is_symmetric_under_operand_swap() {
    // §4.2 applies affinity to "the other operand"; swapping the operands must swap
    // the two applied sides. (Collation is NOT symmetric — left precedence — so only
    // the apply fields are checked here.)
    let cat = cat_m();
    let ops = matrix_operands();
    for &(ltext, _) in &ops {
        for &(rtext, _) in &ops {
            let fwd = compare_meta_of(&projection_of(&format!("SELECT {ltext} = {rtext} FROM m"), &cat));
            let rev = compare_meta_of(&projection_of(&format!("SELECT {rtext} = {ltext} FROM m"), &cat));
            assert_eq!(fwd.apply_left, rev.apply_right, "swap L↔R apply for `{ltext}` = `{rtext}`");
            assert_eq!(fwd.apply_right, rev.apply_left, "swap R↔L apply for `{ltext}` = `{rtext}`");
        }
    }
}

#[test]
fn apply_affinity_named_cases_from_datatype3() {
    // Hand-derived §4.2 corners — ground truth independent of `expected_apply` above,
    // anchoring the matrix at the specific pairs the datatype3 rule text calls out.
    let cat = cat_m();
    let case = |sql: &str| compare_meta_of(&projection_of(sql, &cat));

    // numeric column vs text column → NUMERIC applied to the TEXT side (R1).
    let m = case("SELECT i = t FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, Some(Affinity::Numeric)));
    // numeric column vs text literal (none) → NUMERIC to the literal (R1).
    let m = case("SELECT i = '5' FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, Some(Affinity::Numeric)));
    // text column vs numeric literal (none) → TEXT to the literal (R2).
    let m = case("SELECT t = 5 FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, Some(Affinity::Text)));
    // numeric vs numeric (INTEGER vs REAL) → nothing (R3).
    let m = case("SELECT i = r FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, None));
    // NUMERIC-affinity column vs text column → NUMERIC to the text side (R1).
    let m = case("SELECT n = t FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, Some(Affinity::Numeric)));
    // BLOB-affinity column (left) vs numeric column → NUMERIC applied to the BLOB (left)
    // side. R1 fires because it covers "TEXT or BLOB or no affinity" (datatype3.html:641).
    let m = case("SELECT b = i FROM m");
    assert_eq!((m.apply_left, m.apply_right), (Some(Affinity::Numeric), None));
    // typeless column (left, BLOB affinity) vs numeric column → NUMERIC to the left (R1).
    let m = case("SELECT x = i FROM m");
    assert_eq!((m.apply_left, m.apply_right), (Some(Affinity::Numeric), None));
    // `+i` (none, §3.2 strips affinity) vs text column → TEXT to the `+i` (left) side (R2).
    let m = case("SELECT +i = t FROM m");
    assert_eq!((m.apply_left, m.apply_right), (Some(Affinity::Text), None));
    // two literals → nothing (R3).
    let m = case("SELECT 1 = 2 FROM m");
    assert_eq!((m.apply_left, m.apply_right), (None, None));
}

#[test]
fn text_column_vs_blob_column_applies_text_rule_ii() {
    // The subtle §4.2 corner, pinned as a hand-derived case (independent of `expected_apply`)
    // so the matrix can never silently become a circular change-detector on it — AND so the
    // recurring regression cannot creep back in.
    //
    // Does a TEXT column vs a BLOB/typeless COLUMN apply affinity? YES — §4.2 rule (ii)
    // applies TEXT to the BLOB/typeless side. For these comparison rules a BLOB/typeless
    // column IS "no affinity": datatype3 §3.1 (datatype3.html:294) records that BLOB affinity
    // "used to be called 'NONE'", and §4.3's worked example labels its `c BLOB` / `d` columns
    // `-- no affinity` (datatype3.html:671-672). So rule (ii) fires against a BLOB/typeless
    // column exactly as against a bare literal — `TEXT_col = BLOB_col` converts the BLOB side
    // to TEXT, just like `TEXT_col = <literal>`
    // (see `text_column_vs_bare_literal_applies_text_to_the_literal`).
    //
    // Pinned end-to-end by `conformance_comparison_affinity` on the §4.3 fixture:
    // `SELECT a<d, a=d, a>d FROM t1` is `[0,1,0]` (TEXT `a` vs typeless `d` → rule (ii) makes
    // `a=d` true). DO NOT revert this to `(None, None)`: a hyper-literal reading of rule (ii)'s
    // bare "no affinity" that the spec's own §4.3 example contradicts is exactly the regression
    // that keeps bouncing.
    let cat = cat_m();
    let case = |sql: &str| compare_meta_of(&projection_of(sql, &cat));
    // TEXT column on the LEFT, BLOB/typeless on the right → TEXT applied to the right operand.
    for sql in ["SELECT t = b FROM m", "SELECT t = x FROM m"] {
        let m = case(sql);
        assert_eq!(
            (m.apply_left, m.apply_right),
            (None, Some(Affinity::Text)),
            "{sql}: rule (ii) applies TEXT to the BLOB/typeless operand (§4.2; §3.1/§4.3)"
        );
    }
    // BLOB/typeless on the LEFT, TEXT column on the right → TEXT applied to the left operand.
    for sql in ["SELECT b = t FROM m", "SELECT x = t FROM m"] {
        let m = case(sql);
        assert_eq!(
            (m.apply_left, m.apply_right),
            (Some(Affinity::Text), None),
            "{sql}: rule (ii) applies TEXT to the BLOB/typeless operand (§4.2; §3.1/§4.3)"
        );
    }
}

#[test]
fn every_comparison_operator_carries_the_same_apply_meta() {
    // All of `<  <=  >  >=  =  <>  IS  IS NOT` lower through the same `compare_meta` in the
    // binder (bind_binary), so §4.2 affinity applies identically to each — the matrix only
    // exercises `=`, so this pins the rest. A numeric column vs a no-affinity literal applies
    // NUMERIC to the literal (R1) for every operator.
    let cat = cat_m();
    for op in ["<", "<=", ">", ">=", "=", "<>", "IS", "IS NOT"] {
        let sql = format!("SELECT i {op} '5' FROM m");
        let meta = compare_meta_of(&projection_of(&sql, &cat));
        assert_eq!(
            (meta.apply_left, meta.apply_right),
            (None, Some(Affinity::Numeric)),
            "apply meta for `{sql}`"
        );
        assert_eq!(meta.collation, Collation::Binary, "collation for `{sql}`");
    }
}

#[test]
fn in_subquery_carries_the_subjects_collation() {
    // §7.1: `x IN (SELECT y ...)` is handled like `x = y`, so the comparison collation is the
    // subject's (left precedence). `v` is a NOCASE column, so the InSubquery meta collates
    // NOCASE — the same `compare_meta_subject` path as the IN-list tests, exercised here for
    // the SELECT operand form.
    //
    // Affinity is intentionally NOT asserted here (doc-grounded note, not a hard failure).
    // §3.2 (datatype3.html:466-469) says the RHS of `x IN (SELECT y)` "has the same affinity as
    // the affinity of the result set expression" — handled as `x = y` (§4.2 line 655). So over
    // two TEXT columns `v IN (SELECT u ...)` should be `v = u` (TEXT vs TEXT) → §4.2 R3 → no
    // affinity. But `compare_meta_subject` approximates the subquery side as no-affinity (like
    // an IN-list), so it applies TEXT to it. Asserting only the (unambiguous) collation keeps
    // this test honest; threading the subquery's result-set affinity is a binder feature left
    // to the crate owner. This is separate from the TEXT-vs-BLOB-column corner above.
    let cat = cat_coll();
    match projection_of("SELECT v IN (SELECT u FROM p) FROM p", &cat) {
        EvalExpr::InSubquery { meta, negated, .. } => {
            assert!(!negated, "IN, not NOT IN");
            assert_eq!(meta.collation, Collation::NoCase, "subject v's NOCASE collation");
        }
        other => panic!("expected InSubquery, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Collation precedence (§7.1): explicit COLLATE (left-first) ▷ column (left-first)
// ▷ BINARY. `p` gives a column of each built-in collation.
// ---------------------------------------------------------------------------

#[test]
fn explicit_collate_left_beats_explicit_collate_right() {
    // §7.1 R1: with an explicit postfix COLLATE on BOTH operands, the LEFT one wins.
    // Two literals so no column collation is in play; NOCASE vs RTRIM so the two
    // orderings must disagree (a right-first probe would fail one of them).
    let cat = TestCatalog::new(vec![]);
    let l_nocase = compare_meta_of(&projection_of("SELECT 'a' COLLATE NOCASE = 'b' COLLATE RTRIM", &cat));
    assert_eq!(l_nocase.collation, Collation::NoCase, "left explicit NOCASE wins");
    let l_rtrim = compare_meta_of(&projection_of("SELECT 'a' COLLATE RTRIM = 'b' COLLATE NOCASE", &cat));
    assert_eq!(l_rtrim.collation, Collation::Rtrim, "left explicit RTRIM wins");
}

#[test]
fn explicit_collate_beats_a_column_collation() {
    // §7.1 R1 (explicit) outranks R2 (column). Left operand is the NOCASE column `v`;
    // the right operand is a literal with explicit COLLATE RTRIM. The explicit RTRIM
    // wins over the left column's NOCASE — proving R1 precedes R2 even when the
    // explicit COLLATE is on the right and the column is on the left.
    let cat = cat_coll();
    let meta = compare_meta_of(&projection_of("SELECT v = 'x' COLLATE RTRIM FROM p", &cat));
    assert_eq!(meta.collation, Collation::Rtrim);
}

#[test]
fn column_collation_left_beats_column_collation_right_nocase_vs_rtrim() {
    // §7.1 R2: with a column on each side and no explicit COLLATE, the LEFT column's
    // collation wins. NOCASE vs RTRIM (neither the BINARY default) so the two
    // orderings must disagree — a stronger check than the existing NOCASE-vs-BINARY
    // case, where BINARY could also arise from the R3 fallback.
    let cat = cat_coll();
    let vw = compare_meta_of(&projection_of("SELECT v = w FROM p", &cat));
    assert_eq!(vw.collation, Collation::NoCase, "left v (NOCASE) wins over right w (RTRIM)");
    let wv = compare_meta_of(&projection_of("SELECT w = v FROM p", &cat));
    assert_eq!(wv.collation, Collation::Rtrim, "left w (RTRIM) wins over right v (NOCASE)");
}

#[test]
fn unary_plus_and_cast_keep_the_columns_collation_though_they_drop_affinity() {
    // §7.1 R2 final sentence: "a column name preceded by one or more unary + and/or
    // CAST operators is still considered a column name." So `+v` and `CAST(v AS TEXT)`
    // still contribute v's NOCASE collation — even though §3.2 gives `+v` NO affinity.
    let cat = cat_coll();
    // `+v` (none) vs a none literal: no affinity applied, but collation is still v's NOCASE.
    let plus = compare_meta_of(&projection_of("SELECT +v = 'x' FROM p", &cat));
    assert_eq!(plus.collation, Collation::NoCase, "+v is still a column for collation");
    assert_eq!((plus.apply_left, plus.apply_right), (None, None), "+v (none) vs literal (none): no affinity");
    // `CAST(v AS TEXT)` has TEXT affinity (from the type) but NOCASE collation (still a
    // column): TEXT vs a none literal → TEXT applied to the literal (R2), collation NOCASE.
    let cast = compare_meta_of(&projection_of("SELECT CAST(v AS TEXT) = 5 FROM p", &cat));
    assert_eq!(cast.collation, Collation::NoCase, "CAST(v..) is still a column for collation");
    assert_eq!((cast.apply_left, cast.apply_right), (None, Some(Affinity::Text)), "CAST carries TEXT affinity");
}

#[test]
fn two_text_literals_compare_binary() {
    // §7.1 R3: no explicit COLLATE and no column operand → BINARY.
    let cat = TestCatalog::new(vec![]);
    let meta = compare_meta_of(&projection_of("SELECT 'x' = 'y'", &cat));
    assert_eq!(meta.collation, Collation::Binary);
}

// ---------------------------------------------------------------------------
// Per-branch metadata: BETWEEN / IN / CASE / NULLIF each lower to one or more
// comparisons, and each comparison carries its own §4.2/§7.1 metadata.
// ---------------------------------------------------------------------------

fn between_metas(expr: &EvalExpr) -> (CompareMeta, CompareMeta) {
    match expr {
        EvalExpr::Between { low_meta, high_meta, .. } => (*low_meta, *high_meta),
        other => panic!("expected Between, got {other:?}"),
    }
}

fn inlist_meta(expr: &EvalExpr) -> CompareMeta {
    match expr {
        EvalExpr::InList { meta, .. } => *meta,
        other => panic!("expected InList, got {other:?}"),
    }
}

fn nullif_meta(expr: &EvalExpr) -> CompareMeta {
    match expr {
        EvalExpr::NullIf { meta, .. } => *meta,
        other => panic!("expected NullIf, got {other:?}"),
    }
}

#[test]
fn between_lowers_to_two_comparisons_each_with_its_own_meta() {
    // §4.2: "a BETWEEN b AND c" is two comparisons (a>=b AND a<=c). `i` is numeric and
    // both bounds are text literals (none) → NUMERIC applied to each bound (R1, right side).
    let cat = cat_m();
    let (low, high) = between_metas(&projection_of("SELECT i BETWEEN '5' AND '9' FROM m", &cat));
    assert_eq!((low.apply_left, low.apply_right), (None, Some(Affinity::Numeric)));
    assert_eq!((high.apply_left, high.apply_right), (None, Some(Affinity::Numeric)));
    assert_eq!(low.collation, Collation::Binary);
    assert_eq!(high.collation, Collation::Binary);
}

#[test]
fn between_branches_can_apply_different_affinities() {
    // §4.2 explicitly notes the two BETWEEN comparisons may apply DIFFERENT affinities
    // to the subject. `x` is none; the low bound is the numeric column `i` (R1 applies
    // NUMERIC to the subject `x`, on the LEFT), the high bound is a none literal (R3: nothing).
    let cat = cat_m();
    let (low, high) = between_metas(&projection_of("SELECT x BETWEEN i AND 'z' FROM m", &cat));
    assert_eq!((low.apply_left, low.apply_right), (Some(Affinity::Numeric), None), "low: x(none) vs i(numeric)");
    assert_eq!((high.apply_left, high.apply_right), (None, None), "high: x(none) vs literal(none)");
}

#[test]
fn between_carries_the_subjects_collation_on_both_branches() {
    // §7.1: BETWEEN works "as if it were two separate comparisons"; the NOCASE column
    // `v` is the left operand of both, so both branches use NOCASE.
    let cat = cat_coll();
    let (low, high) = between_metas(&projection_of("SELECT v BETWEEN 'a' AND 'b' FROM p", &cat));
    assert_eq!(low.collation, Collation::NoCase);
    assert_eq!(high.collation, Collation::NoCase);
}

#[test]
fn in_list_items_have_no_affinity_and_numeric_subject_applies_numeric() {
    // §4.2: "a IN (x,y,...)" ≡ "a=+x OR a=+y ..." — the list items have NO affinity.
    // `i` (numeric) vs a none item → NUMERIC applied to the item side (apply_right),
    // never to the subject.
    let cat = cat_m();
    let meta = inlist_meta(&projection_of("SELECT i IN ('5', '6') FROM m", &cat));
    assert_eq!((meta.apply_left, meta.apply_right), (None, Some(Affinity::Numeric)));
    assert_eq!(meta.collation, Collation::Binary);
}

#[test]
fn in_list_text_subject_applies_text_to_items() {
    // §4.2: a text subject vs none items → TEXT applied to the items (apply_right).
    let cat = cat_m();
    let meta = inlist_meta(&projection_of("SELECT t IN (1, 2) FROM m", &cat));
    assert_eq!((meta.apply_left, meta.apply_right), (None, Some(Affinity::Text)));
}

#[test]
fn in_list_blob_subject_applies_nothing() {
    // §4.2: a BLOB/typeless subject vs no-affinity IN items → nothing (rule iii). Rule (ii)
    // needs a TEXT operand; here the subject is BLOB (= "no affinity" for these rules, §3.1/
    // §4.3) and the IN items are no-affinity, so NO side is TEXT and nothing is applied — the
    // no-affinity-vs-no-affinity case, on the IN path (`compare_meta_subject`). Completes the
    // subject-class trio the rule enumerates (numeric / text / blob). `b` is BLOB, `x` is
    // typeless (both BLOB affinity). Contrast `t IN (...)` (TEXT subject) which DOES apply.
    let cat = cat_m();
    for sql in ["SELECT b IN (1, 2) FROM m", "SELECT x IN (1, 2) FROM m"] {
        let meta = inlist_meta(&projection_of(sql, &cat));
        assert_eq!(
            (meta.apply_left, meta.apply_right),
            (None, None),
            "{sql}: a BLOB/typeless subject applies nothing (R3); items are already no-affinity"
        );
    }
}

#[test]
fn in_list_uses_the_subjects_collation() {
    // §7.1: "the collating sequence used for x IN (y,z,...) is the collating sequence
    // of x." The NOCASE column `v` as subject → NOCASE, regardless of the items.
    let cat = cat_coll();
    let meta = inlist_meta(&projection_of("SELECT v IN ('a', 'b') FROM p", &cat));
    assert_eq!(meta.collation, Collation::NoCase);
}

#[test]
fn in_list_explicit_collate_on_subject_wins() {
    // §7.1: for IN, an explicit COLLATE is applied to the left (subject) operand.
    let cat = cat_coll();
    let meta = inlist_meta(&projection_of("SELECT v COLLATE RTRIM IN ('a', 'b') FROM p", &cat));
    assert_eq!(meta.collation, Collation::Rtrim);
}

#[test]
fn simple_case_arm_carries_equality_meta() {
    // A simple `CASE x WHEN a ...` compares x=a per arm (§4.2). `i` numeric vs a text
    // literal (none) → NUMERIC applied to the WHEN value (apply_right).
    let cat = cat_m();
    let cmp = match projection_of("SELECT CASE i WHEN '5' THEN 1 END FROM m", &cat) {
        EvalExpr::Case { whens, .. } => whens[0].cmp.expect("simple CASE arm carries meta"),
        other => panic!("expected Case, got {other:?}"),
    };
    assert_eq!((cmp.apply_left, cmp.apply_right), (None, Some(Affinity::Numeric)));
}

#[test]
fn simple_case_arm_carries_the_operands_collation() {
    // The NOCASE column `v` is the CASE operand, so each arm's equality uses NOCASE.
    let cat = cat_coll();
    let cmp = match projection_of("SELECT CASE v WHEN 'x' THEN 1 END FROM p", &cat) {
        EvalExpr::Case { whens, .. } => whens[0].cmp.expect("simple CASE arm carries meta"),
        other => panic!("expected Case, got {other:?}"),
    };
    assert_eq!(cmp.collation, Collation::NoCase);
}

#[test]
fn searched_case_arm_has_no_comparison_meta() {
    // A searched `CASE WHEN <cond> ...` (no operand) evaluates each WHEN as a boolean:
    // there is no per-arm equality comparison, hence no meta.
    let cat = cat_m();
    match projection_of("SELECT CASE WHEN i = 1 THEN 1 END FROM m", &cat) {
        EvalExpr::Case { operand, whens, .. } => {
            assert!(operand.is_none(), "searched CASE has no operand");
            assert!(whens[0].cmp.is_none(), "searched CASE arm carries no comparison meta");
        }
        other => panic!("expected Case, got {other:?}"),
    }
}

#[test]
fn nullif_carries_comparison_meta() {
    // NULLIF(x,y) is NULL when x=y (§4.2 comparison). `i` numeric vs text literal
    // (none) → NUMERIC applied to the right.
    let cat = cat_m();
    let meta = nullif_meta(&projection_of("SELECT nullif(i, '5') FROM m", &cat));
    assert_eq!((meta.apply_left, meta.apply_right), (None, Some(Affinity::Numeric)));
}

#[test]
fn nullif_carries_the_left_operands_collation() {
    // §7.1: NULLIF's equality uses the left operand's column collation.
    let cat = cat_coll();
    let meta = nullif_meta(&projection_of("SELECT nullif(v, 'x') FROM p", &cat));
    assert_eq!(meta.collation, Collation::NoCase);
}

// ---------------------------------------------------------------------------
// Special-form lowering: coalesce / ifnull / nullif / like / glob lower to their
// dedicated IR nodes, never to a generic EvalExpr::Func. A silent fall-through to
// Func would evaluate them by the wrong path, so each is pinned to its IR node.
// ---------------------------------------------------------------------------

#[test]
fn coalesce_lowers_to_a_coalesce_node() {
    let cat = cat_m();
    match projection_of("SELECT coalesce(i, t, x) FROM m", &cat) {
        EvalExpr::Coalesce(items) => assert_eq!(items.len(), 3),
        other => panic!("expected Coalesce, got {other:?}"),
    }
}

#[test]
fn ifnull_lowers_to_a_two_arg_coalesce() {
    // ifnull(a,b) is exactly coalesce(a,b).
    let cat = cat_m();
    match projection_of("SELECT ifnull(i, t) FROM m", &cat) {
        EvalExpr::Coalesce(items) => assert_eq!(items.len(), 2),
        other => panic!("expected Coalesce, got {other:?}"),
    }
}

#[test]
fn nullif_lowers_to_a_nullif_node_not_a_func() {
    let cat = cat_m();
    assert!(matches!(projection_of("SELECT nullif(i, 5) FROM m", &cat), EvalExpr::NullIf { .. }));
}

#[test]
fn like_function_form_lowers_to_like_with_reversed_argument_order() {
    // The 2-arg function `like(Y, X)` means `X LIKE Y`: arg0 is the PATTERN, arg1 the
    // SUBJECT (lang_expr.html). Pin the reversal: subject must be the column `t`, the
    // pattern the literal — and it is a Like node, not a Func.
    let cat = cat_m();
    match projection_of("SELECT like('abc', t) FROM m", &cat) {
        EvalExpr::Like { kind: LikeKind::Like, subject, pattern, negated, escape } => {
            assert!(!negated);
            assert!(escape.is_none());
            assert!(matches!(subject.as_ref(), EvalExpr::Column(_)), "subject is the column t");
            assert!(matches!(pattern.as_ref(), EvalExpr::Literal(_)), "pattern is the literal");
        }
        other => panic!("expected Like, got {other:?}"),
    }
}

#[test]
fn glob_function_form_lowers_to_a_glob_like_node() {
    let cat = cat_m();
    match projection_of("SELECT glob('a*', t) FROM m", &cat) {
        EvalExpr::Like { kind: LikeKind::Glob, subject, pattern, .. } => {
            assert!(matches!(subject.as_ref(), EvalExpr::Column(_)), "subject is the column t");
            assert!(matches!(pattern.as_ref(), EvalExpr::Literal(_)), "pattern is the literal");
        }
        other => panic!("expected Like(Glob), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Access-path shape with bind parameters: a parameterized rowid / INTEGER PRIMARY
// KEY equality plans as an O(log n) rowid point SEEK (sqlite's OP_SeekRowid), not a
// full scan. The rowid eq executor coerces the seek value via OP_MustBeInt
// (`eval_rowid_eq` -> `must_be_int`: NUMERIC affinity then a lossless integer, else no
// rows), so `rowid = ?` seeks by value — a text/real binding names its integer rowid, a
// non-integer / NULL binding names no row — with NO affinity gate, unlike a secondary
// index. The seek reaches at most one row, so the term is fully consumed (no residual).
// This is the discipline in `access_path::rowid_eq_value`; the mirror unit test there is
// `a_parameter_rowid_equality_is_a_rowid_seek`. (A LITERAL rowid equality is likewise a
// seek — see `rowid_eq_is_a_rowid_seek_with_no_residual` above.)
// ---------------------------------------------------------------------------

/// Assert the plan is `Project -> RowidScan(rowid/intpk `= ?` seek)`: the parameterized
/// equality plans as a rowid point seek whose key is the bound parameter, fully consumed
/// (no residual Filter).
fn assert_param_eq_is_a_rowid_seek(plan: &Plan) {
    let (input, _e) = expect_project(&plan.root);
    match input {
        PlanNode::RowidScan(RowidScan { op: RowidOp::Eq(v), .. }) => assert!(
            matches!(v, EvalExpr::Param(1)),
            "the `= ?` parameter is the rowid seek key, got {v:?}"
        ),
        other => panic!("expected Project over a rowid Eq seek (no residual), got {other:?}"),
    }
}

#[test]
fn rowid_eq_param_is_a_rowid_seek() {
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t WHERE rowid = ?", &cat).unwrap();
    assert_param_eq_is_a_rowid_seek(&plan);
}

#[test]
fn integer_primary_key_eq_param_is_a_rowid_seek() {
    let cat = TestCatalog::new(vec![tdef(
        "u",
        vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)],
        Some(0),
    )]);
    let plan = plan_sql("SELECT v FROM u WHERE id = ?", &cat).unwrap();
    assert_param_eq_is_a_rowid_seek(&plan);
}

// ---------------------------------------------------------------------------
// WITHOUT ROWID base-table plan shape. A WR table stores its rows in a PRIMARY KEY
// index b-tree with NO integer rowid, so the plan side must (a) expose width N — its
// columns, no trailing rowid register — (b) hide the rowid/oid keywords, and (c) force
// a full SeqScan (never a rowid/index SEEK), because the WR access paths are not yet
// index-aware (access_path.rs). These pin the PLAN half of the plan<->exec width
// contract (the crux); the direct `Source::width()` unit pins live in
// `bind/scope.rs`, and the end-to-end value round-trip in
// `minisqlite/tests/conformance_rowid.rs`.
// ---------------------------------------------------------------------------

/// `w(id INTEGER PRIMARY KEY, v TEXT) WITHOUT ROWID` — a WR table whose INTEGER PK is
/// NOT a rowid alias (WR tables carry `rowid_alias: None`), so an `id = <lit>` equality
/// must NOT fold into a rowid seek the way it would on a rowid table.
fn cat_wr() -> TestCatalog {
    let mut w = tdef(
        "w",
        vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)],
        None,
    );
    w.without_rowid = true;
    w.columns[0].primary_key = true;
    TestCatalog::new(vec![w])
}

#[test]
fn without_rowid_table_plans_a_seqscan_of_width_n() {
    // `SELECT *` over a WR table is a SeqScan whose `column_count` is N (2), and `*`
    // expands to exactly the N stored columns — no trailing rowid register.
    let cat = cat_wr();
    let plan = plan_sql("SELECT * FROM w", &cat).expect("plan ok");
    let (input, exprs) = expect_project(&plan.root);
    let scan = expect_seqscan(input);
    assert_eq!(scan.column_count, 2, "a WR SeqScan is width N, no rowid register");
    assert_eq!(exprs.len(), 2, "* expands to exactly the N stored columns");
}

#[test]
fn without_rowid_table_hides_rowid_keywords_in_the_planner() {
    // rowid/oid/_rowid_ are not columns of a WR table, so each must fail to resolve at
    // plan time (the full-planner mirror of `rowid_in_source` returning None for WR).
    let cat = cat_wr();
    for q in ["SELECT rowid FROM w", "SELECT oid FROM w", "SELECT _rowid_ FROM w"] {
        let err = plan_sql(q, &cat).expect_err("a rowid keyword must not resolve on a WR table");
        assert!(
            format!("{err:?}").contains("no such column"),
            "{q}: expected 'no such column', got {err:?}"
        );
    }
}

#[test]
fn without_rowid_integer_pk_equality_is_a_scan_not_a_rowid_seek() {
    // On a ROWID table an `INTEGER PRIMARY KEY = <literal>` folds into a RowidScan (see
    // `rowid_eq_is_a_rowid_seek_with_no_residual`). A WR table's INTEGER PK is NOT a
    // rowid alias, so the same predicate must stay a full SeqScan with the equality as a
    // residual Filter — pinning the `access_path.rs` WR guard. A regression that let a WR
    // table reach the rowid/index seek path would surface here as a RowidScan.
    let cat = cat_wr();
    let plan = plan_sql("SELECT v FROM w WHERE id = 5", &cat).expect("plan ok");
    let (input, _e) = expect_project(&plan.root);
    match input {
        PlanNode::Filter { input, predicate } => {
            assert!(
                matches!(**input, PlanNode::SeqScan(_)),
                "a WR INTEGER-PK equality is a full scan, not a seek, got {input:?}"
            );
            assert!(
                matches!(predicate, EvalExpr::Compare { op: CmpOp::Eq, .. }),
                "the `id = 5` equality survives as the residual filter, got {predicate:?}"
            );
        }
        other => panic!("expected Filter over SeqScan (never a RowidScan), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Result-column naming (lang_select.html): an alias overrides the label; a bare /
// qualified column keeps its bare name; any other expression is named by its
// reconstructed source text.
// ---------------------------------------------------------------------------

#[test]
fn aliased_expression_takes_the_alias() {
    // An explicit alias overrides the source-text label even for a compound expression.
    let cat = cat_m();
    let plan = plan_sql("SELECT i + 1 AS s FROM m", &cat).unwrap();
    assert_eq!(plan.result_columns, vec!["s".to_string()]);
}

#[test]
fn unaliased_expression_is_named_by_its_source_text() {
    let cat = cat_m();
    // Arithmetic over a column renders back to `i+1`.
    assert_eq!(plan_sql("SELECT i + 1 FROM m", &cat).unwrap().result_columns, vec!["i+1".to_string()]);
    // A constant arithmetic expression renders its operands and operator.
    assert_eq!(plan_sql("SELECT 1 + 2 FROM m", &cat).unwrap().result_columns, vec!["1+2".to_string()]);
    // A text literal keeps its surrounding quotes.
    assert_eq!(plan_sql("SELECT 'hi' FROM m", &cat).unwrap().result_columns, vec!["'hi'".to_string()]);
}

// ---------------------------------------------------------------------------
// Delegation-seam routing. The planner dispatches DML (INSERT / UPDATE / DELETE)
// and a compound set operation (UNION / INTERSECT / EXCEPT) each to its OWN compiler
// module, and a leading WITH to the CTE stub. These tests pin the top-level ROUTING:
// that a statement reaches the RIGHT compiler (not a copy-paste-wrong sibling) and
// yields the expected top-level plan node — or, for a still-unimplemented stub (WITH),
// its exact loud error. The feature-specific plan shape lives in each module's own
// tests (compile/insert.rs, update.rs, delete.rs, compound.rs); here we assert only
// the routing plus the `mutates` flag. When a sibling implements a remaining stub it
// should UPDATE the corresponding test to assert the new plan shape, not delete it.
// ---------------------------------------------------------------------------

#[test]
fn insert_compiles_to_an_insert_node() {
    // INSERT routes QueryPlanner::plan -> compile::compile_insert, which now compiles to a
    // mutating Insert plan (the feature landed). The Insert's internals are covered by
    // compile/insert.rs; here we pin only the routing + that the statement mutates.
    let cat = cat_t();
    let plan = plan_sql("INSERT INTO t VALUES (1, 'x', 'y')", &cat).expect("INSERT compiles");
    assert!(matches!(plan.root, PlanNode::Insert(_)), "expected an Insert root, got {:?}", plan.root);
    assert!(plan.mutates, "INSERT is a mutating statement");
}

#[test]
fn update_compiles_to_an_update_node() {
    // UPDATE routes QueryPlanner::plan -> compile::compile_update, which now compiles to a
    // mutating Update plan (the feature landed; internals covered by compile/update.rs).
    let cat = cat_t();
    let plan = plan_sql("UPDATE t SET a = 1", &cat).expect("UPDATE compiles");
    assert!(matches!(plan.root, PlanNode::Update(_)), "expected an Update root, got {:?}", plan.root);
    assert!(plan.mutates, "UPDATE is a mutating statement");
}

#[test]
fn delete_compiles_to_a_delete_node() {
    // DELETE routes QueryPlanner::plan -> compile::compile_delete, which now compiles to a
    // mutating Delete plan (the feature landed; internals covered by compile/delete.rs).
    let cat = cat_t();
    let plan = plan_sql("DELETE FROM t", &cat).expect("DELETE compiles");
    assert!(matches!(plan.root, PlanNode::Delete(_)), "expected a Delete root, got {:?}", plan.root);
    assert!(plan.mutates, "DELETE is a mutating statement");
}

#[test]
fn with_clause_compiles_to_a_cte_scan() {
    // A leading WITH routes compile_select_scoped -> compile::cte::compile_with, which
    // registers each CTE into `plan.ctes` and lowers a reference in the body's FROM to a
    // CteScan over the pre-registered id — NOT a "no such table" catalog lookup.
    let cat = cat_t();
    let plan = plan_sql("WITH x AS (SELECT 1) SELECT * FROM x", &cat).expect("WITH compiles");
    assert_eq!(plan.ctes.len(), 1, "the CTE `x` is registered, got {:?}", plan.ctes);
    let (input, _e) = expect_project(&plan.root);
    match input {
        PlanNode::CteScan { id, column_count } => {
            assert_eq!((*id, *column_count), (0, 1), "FROM x scans CTE 0, width 1");
        }
        other => panic!("expected the body's `FROM x` to be a CteScan, got {other:?}"),
    }
}

/// Walk a plan tree looking for a `RecursiveScan` leaf. Covers every child-bearing
/// `PlanNode` variant so a step's self-reference cannot hide behind a wrapper the walk
/// forgot (which would be a silent false-negative). DML/DDL/leaf nodes carry no
/// `RecursiveScan`.
fn plan_contains_recursive_scan(node: &PlanNode) -> bool {
    match node {
        PlanNode::RecursiveScan { .. } => true,
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => plan_contains_recursive_scan(input),
        PlanNode::Aggregate(a) => plan_contains_recursive_scan(&a.input),
        PlanNode::Window(w) => plan_contains_recursive_scan(&w.input),
        PlanNode::Join(j) => {
            plan_contains_recursive_scan(&j.left) || plan_contains_recursive_scan(&j.right)
        }
        PlanNode::SetOp(s) => {
            plan_contains_recursive_scan(&s.left) || plan_contains_recursive_scan(&s.right)
        }
        _ => false,
    }
}

#[test]
fn recursive_cte_builds_a_recursive_plan_whose_step_reads_the_working_table() {
    // A self-referential compound CTE compiles to CtePlan::Recursive; the step's
    // self-reference lowers to a RecursiveScan (the working table), while the seed
    // (initial-select) does not — only the recursive-select may reference the table.
    let cat = cat_t();
    let plan = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<3) SELECT n FROM c",
        &cat,
    )
    .expect("recursive WITH compiles");
    assert_eq!(plan.ctes.len(), 1, "one CTE registered");
    match &plan.ctes[0] {
        CtePlan::Recursive { name, column_count, seed, step, union_all, .. } => {
            assert_eq!(name, "c");
            assert_eq!(*column_count, 1);
            assert!(*union_all, "UNION ALL -> keep-duplicates fixpoint");
            assert!(
                !plan_contains_recursive_scan(seed),
                "the seed must not reference the working table, got {seed:?}"
            );
            assert!(
                plan_contains_recursive_scan(step),
                "the recursive step must read the working table via RecursiveScan, got {step:?}"
            );
        }
        other => panic!("expected a Recursive CTE, got {other:?}"),
    }
}

#[test]
fn recursive_cte_union_without_all_selects_the_dedup_fixpoint() {
    // `UNION` (not `UNION ALL`) between the initial- and recursive-select records
    // union_all=false, which the executor reads as whole-run dedup (cycle-safe).
    let cat = cat_t();
    let plan = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION SELECT n+1 FROM c WHERE n<3) SELECT n FROM c",
        &cat,
    )
    .expect("recursive WITH compiles");
    match &plan.ctes[0] {
        CtePlan::Recursive { union_all, .. } => assert!(!*union_all, "UNION -> dedup fixpoint"),
        other => panic!("expected a Recursive CTE, got {other:?}"),
    }
}

#[test]
fn ordinary_cte_under_recursive_keyword_is_still_materialized() {
    // §2/§5: RECURSIVE does not force recursion. A non-self-referential CTE under
    // WITH RECURSIVE is an ordinary materialized CTE, not a Recursive plan.
    let cat = cat_t();
    let plan = plan_sql("WITH RECURSIVE c(n) AS (SELECT 7) SELECT n FROM c", &cat)
        .expect("WITH RECURSIVE over a non-recursive body compiles");
    assert!(
        matches!(&plan.ctes[0], CtePlan::Materialized { name, .. } if name == "c"),
        "a non-self-referential CTE stays materialized, got {:?}",
        plan.ctes[0]
    );
}

#[test]
fn cte_column_name_list_arity_mismatch_is_a_loud_error() {
    // `WITH c(a,b) AS (SELECT 1)` declares two columns for a one-column body — SQLite
    // rejects the arity mismatch rather than silently truncating / padding.
    let cat = cat_t();
    let err = plan_sql("WITH c(a,b) AS (SELECT 1) SELECT * FROM c", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("values for") && m.contains("columns"), "got {m:?}"),
        other => panic!("expected a Sql arity error, got {other:?}"),
    }
}

#[test]
fn recursive_cte_body_limit_wraps_the_fixpoint_in_a_materialized_limit() {
    // A LIMIT/OFFSET on a recursive CTE body bounds the recursive table's rows (spec §3).
    // CtePlan::Recursive carries no limit, so it is honored by a SECOND, materialized CTE
    // that scans the raw fixpoint and applies the LIMIT/OFFSET after it. For a TERMINATING
    // recursion this is result-equivalent to SQLite (rows are produced in fixpoint order,
    // so a prefix LIMIT/OFFSET keeps the same rows — see conformance_cte's
    // `recursive_limit_zero_adds_no_rows` / `recursive_offset_skips_leading_rows`). The CTE
    // name resolves to the WRAPPER (`CteScan{id:1}`), never the raw fixpoint.
    let cat = cat_t();
    let plan = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<5 LIMIT 3) \
         SELECT n FROM c",
        &cat,
    )
    .expect("a recursive CTE with a body LIMIT compiles to the materialized-limit wrapper");
    assert_eq!(plan.ctes.len(), 2, "raw fixpoint + limit wrapper, got {:?}", plan.ctes);
    assert!(
        matches!(&plan.ctes[0], CtePlan::Recursive { name, .. } if name == "c"),
        "ctes[0] is the raw recursive fixpoint, got {:?}",
        plan.ctes[0]
    );
    match &plan.ctes[1] {
        CtePlan::Materialized { name, column_count, body } => {
            assert_eq!((name.as_str(), *column_count), ("c", 1));
            match body {
                PlanNode::Limit { input, limit, offset } => {
                    assert!(
                        matches!(limit, Some(EvalExpr::Literal(Value::Integer(3)))),
                        "the wrapper applies the exact body LIMIT 3, got {limit:?}"
                    );
                    assert!(offset.is_none(), "no OFFSET in this query");
                    assert!(
                        matches!(input.as_ref(), PlanNode::CteScan { id: 0, .. }),
                        "the wrapper scans the raw fixpoint ctes[0], got {input:?}"
                    );
                }
                other => panic!("expected the wrapper body to be a Limit, got {other:?}"),
            }
        }
        other => panic!("expected ctes[1] to be a Materialized limit wrapper, got {other:?}"),
    }
    // The body's `FROM c` resolves to the WRAPPER (id 1), so it reads the limited rows.
    let (input, _e) = expect_project(&plan.root);
    assert!(
        matches!(input, PlanNode::CteScan { id: 1, .. }),
        "the outer FROM c scans the limit wrapper (ctes[1]), got {input:?}"
    );
}

#[test]
fn recursive_cte_body_order_by_with_limit_is_a_loud_error() {
    // SQLite's ORDER BY on a recursive CTE body reorders the recursion QUEUE (which rows
    // recurse first); with a LIMIT that order decides which rows survive, and a
    // breadth-first fixpoint can't reproduce it. Rejected loud rather than returning wrong
    // rows. (ORDER BY without LIMIT is a no-op on the unordered result and is dropped.)
    let cat = cat_t();
    let err = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<5 \
         ORDER BY n LIMIT 3) SELECT n FROM c",
        &cat,
    )
    .unwrap_err();
    match err {
        Error::Sql(m) => assert!(
            m.contains("ORDER BY") && m.contains("LIMIT") && m.contains("recursive"),
            "got {m:?}"
        ),
        other => panic!("expected a Sql unsupported error, got {other:?}"),
    }
}

#[test]
fn later_cte_sees_earlier_cte_in_the_same_with() {
    // §2: a CTE is visible to the CTEs after it in the same WITH. c2 references c1, so
    // both register and c2's body compiles (a missing c1 would be "no such table").
    let cat = cat_t();
    let plan = plan_sql(
        "WITH c1 AS (SELECT 3 AS n), c2 AS (SELECT n+1 AS m FROM c1) SELECT m FROM c2",
        &cat,
    )
    .expect("a later CTE referencing an earlier one compiles");
    assert_eq!(plan.ctes.len(), 2, "both CTEs registered, got {:?}", plan.ctes);
    assert!(
        matches!(&plan.ctes[0], CtePlan::Materialized { name, .. } if name == "c1"),
        "first registered CTE is c1"
    );
    assert!(
        matches!(&plan.ctes[1], CtePlan::Materialized { name, .. } if name == "c2"),
        "second registered CTE is c2"
    );
}

#[test]
fn a_non_cte_name_still_resolves_to_the_catalog() {
    // `lookup_cte` is consulted before the catalog, but only shadows a MATCHING name:
    // with no WITH in scope the check is inert and a plain table reference resolves to the
    // catalog (a SeqScan of `t`). (Scope-leak prevention is covered separately below.)
    let cat = cat_t();
    let plan = plan_sql("SELECT a FROM t", &cat).expect("a plain table query compiles");
    assert!(plan.ctes.is_empty(), "no WITH -> no CTEs, got {:?}", plan.ctes);
    let (input, _e) = expect_project(&plan.root);
    assert!(
        matches!(input, PlanNode::SeqScan(_)),
        "FROM t is a base-table SeqScan, got {input:?}"
    );
}

#[test]
fn cte_scope_does_not_leak_into_a_following_statement() {
    // The RAII CteScopeGuard pops a WITH's CTEs when compilation returns, so a CTE from
    // one statement must NOT be visible to the next statement planned on the same thread.
    // (This actually exercises the leak-prevention the guard exists for: if the scope
    // leaked, the second `FROM c` would resolve to a CteScan instead of erroring.)
    let cat = cat_t();
    plan_sql("WITH c(n) AS (SELECT 1) SELECT n FROM c", &cat).expect("the WITH statement compiles");
    let err = plan_sql("SELECT n FROM c", &cat).unwrap_err();
    match err {
        Error::Sql(m) => assert!(m.contains("no such table: c"), "expected a leaked-scope guard, got {m:?}"),
        other => panic!("expected a no-such-table error, got {other:?}"),
    }
}

#[test]
fn from_subquery_with_leading_with_resolves_its_own_cte() {
    // A WITH-bearing subquery in FROM: `FROM (WITH c AS … SELECT … FROM c) d`. The inner
    // CTE `c` must resolve when the OUTER query computes the subquery's schema (Phase 1),
    // not only during Phase-2 compilation — otherwise `FROM c` is rejected as "no such
    // table" before Phase 2 runs. Both `c` and the derived `d` register as CtePlans.
    let cat = cat_t();
    let plan = plan_sql("SELECT n FROM (WITH c(n) AS (SELECT 2) SELECT n FROM c) d", &cat)
        .expect("a WITH-bearing FROM subquery compiles");
    assert_eq!(plan.ctes.len(), 2, "inner CTE `c` and derived subquery `d` both register, got {:?}", plan.ctes);
    let (input, _e) = expect_project(&plan.root);
    match input {
        PlanNode::CteScan { .. } => {}
        other => panic!("expected the outer FROM to scan the derived subquery, got {other:?}"),
    }
}

#[test]
fn cte_shadows_a_base_table_of_the_same_name() {
    // resolve_table consults lookup_cte BEFORE the catalog: an unqualified `FROM c` under
    // `WITH c AS (...)` resolves to the CTE (a CteScan), shadowing a base table also named
    // `c`, matching SQLite's rule that a CTE hides a same-named real table.
    let cat = TestCatalog::new(vec![tdef("c", vec![col("a", Some("INTEGER"), None)], None)]);
    let plan = plan_sql("WITH c(n) AS (SELECT 9) SELECT n FROM c", &cat)
        .expect("a CTE shadowing a base table compiles");
    let (input, _e) = expect_project(&plan.root);
    assert!(
        matches!(input, PlanNode::CteScan { .. }),
        "FROM c must scan the CTE, not the base table `c` (a SeqScan), got {input:?}"
    );
}

#[test]
fn nested_with_inside_a_cte_body_compiles() {
    // A CTE body may itself carry a leading WITH; `compile_select` re-enters the WITH
    // dispatch for the inner body, extending the scope stack for that body only. The
    // inner `d` is visible to the inner body but not to the outer body.
    let cat = cat_t();
    let plan = plan_sql(
        "WITH outer_cte AS (WITH d(n) AS (SELECT 5) SELECT n FROM d) SELECT n FROM outer_cte",
        &cat,
    )
    .expect("a nested WITH inside a CTE body compiles");
    assert!(
        plan.ctes.iter().any(|c| matches!(c, CtePlan::Materialized { name, .. } if name == "outer_cte")),
        "the outer CTE registers, got {:?}",
        plan.ctes
    );
}

#[test]
fn recursive_cte_with_intersect_connector_is_a_loud_error() {
    // A recursive CTE's initial- and recursive-selects must be joined by UNION / UNION ALL
    // (spec §3). INTERSECT/EXCEPT at that seam is rejected loudly rather than silently
    // treated as a UNION recursion (which would return wrong rows).
    let cat = cat_t();
    let err = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 INTERSECT SELECT n+1 FROM c) SELECT n FROM c",
        &cat,
    )
    .unwrap_err();
    match err {
        Error::Sql(m) => assert!(
            m.contains("UNION") && m.contains("INTERSECT"),
            "expected an INTERSECT/EXCEPT rejection, got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}

#[test]
fn recursive_cte_referenced_twice_in_one_arm_is_a_loud_error() {
    // A single recursive-select may name the working table at most once (SQLite errors
    // "recursive table may not appear more than once"). Two RecursiveScans would
    // cross-product the frontier into silently-wrong rows, so reject it up front.
    let cat = cat_t();
    let err = plan_sql(
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT a.n+1 FROM c a, c b) SELECT n FROM c",
        &cat,
    )
    .unwrap_err();
    match err {
        Error::Sql(m) => assert!(
            m.contains("more than once"),
            "expected a duplicate-self-reference rejection, got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}

#[test]
fn union_compiles_to_a_setop_node() {
    // A UNION / INTERSECT / EXCEPT body routes compile_body -> compile::compound::compile_compound,
    // which compiles to a SetOp (the feature landed; internals covered by compile/compound.rs).
    // A no-ORDER-BY DEDUP compound is additionally wrapped in an implicit ascending Sort
    // (real SQLite dedups via a sort, so the result comes back sorted), so the SetOp is the
    // Sort's input here rather than the bare root. A compound SELECT is a pure read: no mutation.
    let cat = cat_t();
    let plan = plan_sql("SELECT 1 UNION SELECT 2", &cat).expect("UNION compiles");
    match &plan.root {
        PlanNode::Sort { input, .. } => assert!(
            matches!(input.as_ref(), PlanNode::SetOp(_)),
            "expected a SetOp under the implicit Sort, got {input:?}"
        ),
        other => panic!("expected an implicit Sort wrapping a SetOp, got {other:?}"),
    }
    assert!(!plan.mutates, "a compound SELECT is a read, not a mutation");
}

// ---------------------------------------------------------------------------
// JSON table-valued function: lazy hidden `json` materialization (perf guard).
//
// The hidden `json` column is the WHOLE document; copying it into every emitted row is
// O(rows·|document|) — quadratic for a flat array. So the planner materializes it ONLY
// when the statement references `json` (`TableFunctionScan::emit_json`). These tests pin
// that flag deterministically so the O(n²) regression cannot creep back: `emit_json` MUST
// be false for the common visible-column queries and true exactly when `json` is read.
// ---------------------------------------------------------------------------

/// The `emit_json` flag of the single `TableFunctionScan` reachable through `node`'s
/// structural children, or `None` if there is none.
fn tvf_emit_json(node: &PlanNode) -> Option<bool> {
    match node {
        PlanNode::TableFunctionScan { emit_json, .. } => Some(*emit_json),
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => tvf_emit_json(input),
        PlanNode::Aggregate(a) => tvf_emit_json(&a.input),
        PlanNode::Window(w) => tvf_emit_json(&w.input),
        PlanNode::Join(j) => tvf_emit_json(&j.left).or_else(|| tvf_emit_json(&j.right)),
        PlanNode::SetOp(s) => tvf_emit_json(&s.left).or_else(|| tvf_emit_json(&s.right)),
        _ => None,
    }
}

fn emit_json_of(sql: &str) -> bool {
    let cat = TestCatalog::new(vec![]);
    let plan = plan_sql(sql, &cat).expect("the json_each query compiles");
    tvf_emit_json(&plan.root).expect("the plan has a TableFunctionScan leaf")
}

#[test]
fn json_each_visible_column_query_does_not_materialize_the_document() {
    // `SELECT value` never names the hidden `json` column, so the executor must NOT copy
    // the whole document into each row (that was the O(rows·|document|) regression). The
    // binder resolves no hidden `json`, so `emit_json` is false.
    assert!(!emit_json_of("SELECT value FROM json_each('[1,2,3]')"));
}

#[test]
fn json_each_star_query_does_not_materialize_the_document() {
    // `SELECT *` expands to the eight VISIBLE columns only (hidden `json`/`root` are
    // excluded from `*`), so it too must not copy the document per row.
    assert!(!emit_json_of("SELECT * FROM json_each('[1,2,3]')"));
}

#[test]
fn json_each_count_star_does_not_materialize_the_document() {
    // A pure `count(*)` scan references no column at all — the classic case where copying
    // the document n times would be gratuitous quadratic work.
    assert!(!emit_json_of("SELECT count(*) FROM json_each('[1,2,3]')"));
}

#[test]
fn json_each_selecting_json_materializes_the_document() {
    // When the query DOES ask for `json`, the document must be materialized — the user
    // asked for it, matching SQLite echoing the input. So `emit_json` is true.
    assert!(emit_json_of("SELECT json FROM json_each('[1,2,3]')"));
}

#[test]
fn json_each_referencing_json_in_where_materializes_the_document() {
    // The reference need not be in the SELECT list: a WHERE (bound before the leaf's
    // SELECT list) that reads `json` must still flip `emit_json` on — the signal is the
    // binder resolving the hidden column, wherever it appears.
    assert!(emit_json_of("SELECT value FROM json_each('[1,2,3]') WHERE json IS NOT NULL"));
}

#[test]
fn json_each_qualified_json_reference_materializes_the_document() {
    // A qualified `alias.json` reaches the same hidden column by name, so it too must
    // materialize the document.
    assert!(emit_json_of("SELECT je.json FROM json_each('[1,2,3]') AS je"));
}
