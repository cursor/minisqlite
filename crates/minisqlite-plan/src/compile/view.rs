//! View expansion: when a query references a `CREATE VIEW`-defined view in its FROM,
//! the view is inlined exactly like a derived table (a `FROM (SELECT …)` subquery).
//! A view owns no b-tree; its stored `CREATE VIEW … AS <select>` text
//! ([`ViewDef::sql`](minisqlite_catalog::ViewDef)) is the source of truth, re-parsed
//! and lowered here. This module is the sibling of [`crate::compile::cte`]: the FROM
//! compiler ([`crate::compile::from`]) calls into it from BOTH compilation phases so
//! the two agree on the view's schema width and its registered plan.
//!
//! * Phase 1 ([`view_schema`]) computes the view's output columns (its
//!   [`SynthCol`] list) from the parsed SELECT, honoring an explicit
//!   `CREATE VIEW v(c1,c2,…)` column-name list — the same schema an inline derived
//!   table gets, so the outer scope binds `Column(base+i)` against it.
//! * Phase 2 ([`compile_view`]) compiles the SELECT into a fresh
//!   [`CtePlan::Materialized`] and returns the [`PlanNode::CteScan`] leaf that reads
//!   it — identical lowering to `compile_derived`, so the executor needs no view-aware
//!   code (a view is a materialized CTE by the time it reaches it).
//!
//! # Recursion / circular guard (the [`ViewGuard`] view stack)
//!
//! A view body may reference other views (or, by mistake, itself, directly or
//! transitively). Expansion recurses through
//! `resolve_table → derived_schema → resolve_table` (Phase 1) and
//! `compile_view → compile_select → … → build_source_leaf → compile_view` (Phase 2),
//! so an unguarded self-reference would recurse forever. A thread-local stack of the
//! view names currently being expanded ([`VIEW_STACK`]) makes re-entry of a name
//! already in progress a loud `view X is circularly defined` error rather than a hang
//! (`build.c`'s message). A depth cap ([`MAX_VIEW_DEPTH`]) is a second, belt-and-suspenders
//! bound. The guard is held across the recursive schema/compile call and pops on drop
//! (including an early `?`), so the stack stays balanced. Like the CTE scope, planning
//! is single-threaded per statement, so the thread-local is a per-statement ambient
//! value, never shared across concurrent planners.
//!
//! # Why a view body cannot see the referencing query's CTEs
//!
//! A view is a self-contained schema object. When it is expanded inside a query that
//! has its own `WITH` clause, that query's CTEs are still on the shared CTE scope
//! stack; the view's body must NOT resolve one of them (only its own `WITH` and base
//! tables/views). Both phases hold a [`crate::compile::cte::IsolatedCteScope`] to empty
//! that stack for the duration of the view's schema computation / compilation, matching
//! how real SQLite compiles a view against a fresh context.

use std::cell::RefCell;

use minisqlite_catalog::{Catalog, ViewDef};
use minisqlite_sql::{parse, Select, Statement};
use minisqlite_types::{Error, Result};

use crate::bind::scope::SynthCol;
use crate::bind::Source;
use crate::compile::cte::IsolatedCteScope;
use crate::compile::from::derived_schema;
use crate::compile::select::compile_select;
use crate::plan::{CtePlan, PlanNode};
use crate::plan_ctx::PlanCtx;

/// The deepest chain of nested view expansions allowed before it is treated as a
/// (pathological) circular definition. Cycle detection over [`VIEW_STACK`] already
/// catches every true cycle — the stack can never exceed the number of distinct views
/// without a name repeating — so this only fires on an absurd non-cyclic nesting; it is
/// a defensive bound so a bug in the cycle check can never turn into an unbounded
/// recursion. `1000` matches SQLite's typical recursion depth limit.
const MAX_VIEW_DEPTH: usize = 1000;

thread_local! {
    /// The folded names of the views currently mid-expansion, innermost last. Empty
    /// except while a view is being expanded. See the module docs.
    static VIEW_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Marks one view as "being expanded" for its lifetime, popping it on drop. Constructed
/// by [`enter`](ViewGuard::enter), which fails closed with `view X is circularly defined`
/// if the view is already on the stack (a cycle) or the stack is at [`MAX_VIEW_DEPTH`].
pub(crate) struct ViewGuard;

impl ViewGuard {
    /// Push `name` (case-folded) onto the expansion stack, or return the circular-view
    /// error if it is already present (a direct or transitive cycle) or the depth cap is
    /// reached. On success the returned guard pops exactly this one name on drop, so a
    /// balanced enter/drop keeps the stack consistent even across an early `?`.
    fn enter(name: &str) -> Result<ViewGuard> {
        let folded = name.to_ascii_lowercase();
        VIEW_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.iter().any(|n| n == &folded) || stack.len() >= MAX_VIEW_DEPTH {
                return Err(Error::sql(format!("view {name} is circularly defined")));
            }
            stack.push(folded);
            Ok(ViewGuard)
        })
    }
}

impl Drop for ViewGuard {
    fn drop(&mut self) {
        VIEW_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Phase 1: compute the view's output schema (its [`SynthCol`] list) from its stored
/// SELECT, honoring an explicit `CREATE VIEW v(c1,…)` column-name list. This is the
/// schema the outer FROM scope binds against, so its width MUST equal what
/// [`compile_view`] later compiles the body to.
///
/// Guards recursion (a self/mutually-referential view errors here rather than hanging)
/// and isolates the CTE scope (the body sees only its own `WITH`, never the referencing
/// query's — see the module docs).
pub(crate) fn view_schema(catalog: &dyn Catalog, view: &ViewDef) -> Result<Vec<SynthCol>> {
    let _view_guard = ViewGuard::enter(&view.name)?;
    let _cte_isolation = IsolatedCteScope::enter();
    let (select, columns) = parse_view_sql(&view.sql, &view.name)?;
    let base = derived_schema(catalog, &select)?;
    apply_view_columns(base, columns.as_deref(), &view.name)
}

/// Phase 2: compile the view's SELECT into a freshly-registered
/// [`CtePlan::Materialized`] and return the [`PlanNode::CteScan`] leaf that reads it —
/// the identical lowering [`crate::compile::from::compile_derived`] gives an inline
/// `FROM (SELECT …)` subquery, so a view IS a materialized CTE by the time the executor
/// runs. The body is compiled as a standalone, NON-correlated query (no parent scope),
/// so a view can never reference the referencing query's columns.
///
/// The Phase-1 schema width (`src.width()`, from [`view_schema`]) MUST equal the
/// compiled body's width — the outer scope already bound `Column(base+i)` against it —
/// so a mismatch fails closed with a loud error, never a silently mis-sized `CteScan`
/// row (the same guard `compile_derived` holds). Guards recursion and isolates the CTE
/// scope exactly as Phase 1 does.
pub(crate) fn compile_view(
    ctx: &mut PlanCtx,
    view: &ViewDef,
    src: &Source,
) -> Result<PlanNode> {
    let _view_guard = ViewGuard::enter(&view.name)?;
    let _cte_isolation = IsolatedCteScope::enter();
    let (select, _columns) = parse_view_sql(&view.sql, &view.name)?;
    let column_count = src.width();
    let (body, names) = compile_select(ctx, &select)?;
    // Phase-1 (`view_schema`, catalog-only naming, column-list applied) and Phase-2 (the
    // real projection compiler) count the output columns by SEPARATE paths; a projection
    // feature added to one and not the other would emit a `CteScan` whose width disagrees
    // with the registers the outer scope already bound. This is once-per-reference
    // plan-time work (off the hot path), so fail closed with a hard error.
    if names.len() != column_count {
        return Err(Error::sql(format!(
            "internal error: view {} Phase-1 schema width ({column_count}) \
             != compiled body width ({})",
            view.name,
            names.len()
        )));
    }
    // Capture the id AFTER the inner compile: a view/derived table nested inside this
    // view's body registers its own (lower) CtePlan first, so the outer scope's register
    // offsets (which already used `column_count`) stay correct.
    let id = ctx.ctes.len();
    ctx.ctes.push(CtePlan::Materialized {
        name: src.exposed_name().to_string(),
        column_count,
        body,
    });
    Ok(PlanNode::CteScan { id, column_count })
}

/// Re-parse a view's stored `CREATE VIEW … AS <select>` text back into its `SELECT` and
/// its optional column-name list. The catalog stores exactly one verbatim `CREATE VIEW`
/// statement as the view's `sql`, so anything else is a corrupt schema row and fails
/// closed (never a silently-wrong expansion). Shared with [`crate::compile::instead_of`],
/// which re-derives the view's column schema for the OLD/NEW trigger scope.
pub(crate) fn parse_view_sql(sql: &str, view_name: &str) -> Result<(Select, Option<Vec<String>>)> {
    let ast = parse(sql)?;
    let mut statements = ast.statements;
    if statements.len() != 1 {
        return Err(Error::sql(format!(
            "internal error: view {view_name} does not store a single CREATE VIEW statement"
        )));
    }
    let Some(Statement::CreateView(cv)) = statements.pop() else {
        return Err(Error::sql(format!(
            "internal error: view {view_name} stored SQL is not a CREATE VIEW statement"
        )));
    };
    let cv = *cv;
    Ok((*cv.select, cv.columns))
}

/// Apply an explicit `CREATE VIEW v(c1,c2,…)` column-name list to the SELECT-derived
/// schema: RENAME each output column to the listed name (carrying its affinity/collation
/// through), enforcing the arity SQLite requires. Without a list the SELECT's own output
/// names stand. Mirrors [`crate::compile::cte`]'s `cte_schema` column-list handling.
///
/// The mismatch message matches real sqlite3's exact wording (`build.c`:
/// `expected %d columns for '%s' but got %d`), with N the declared column count and M
/// the SELECT's actual output width, to match real sqlite3's error behavior.
fn apply_view_columns(
    base: Vec<SynthCol>,
    columns: Option<&[String]>,
    view_name: &str,
) -> Result<Vec<SynthCol>> {
    let Some(list) = columns else {
        return Ok(base);
    };
    if list.len() != base.len() {
        return Err(Error::sql(format!(
            "expected {} columns for '{}' but got {}",
            list.len(),
            view_name,
            base.len()
        )));
    }
    Ok(list
        .iter()
        .zip(base)
        .map(|(name, col)| SynthCol {
            name: name.clone(),
            affinity: col.affinity,
            collation: col.collation,
            hidden: false,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef, ViewDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Result;

    use crate::plan::{CtePlan, Plan, PlanNode};
    use crate::{Planner, QueryPlanner};

    // A static, read-only test catalog with base tables AND views, so the `view()` seam
    // is exercised. Mirrors the ~25-line pattern the sibling compile modules use; kept
    // local so this file owns its own fixtures and never edits `src/tests.rs`.
    struct TestCatalog {
        tables: Vec<TableDef>,
        views: Vec<ViewDef>,
    }

    impl Catalog for TestCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, _name: &str) -> Result<Option<&IndexDef>> {
            Ok(None)
        }
        fn view(&self, name: &str) -> Result<Option<&ViewDef>> {
            Ok(self.views.iter().find(|v| v.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, _table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(Vec::new())
        }
        fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn create_table(&mut self, _p: &mut dyn Pager, _s: &CreateTable, _sql: &str) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn create_index(&mut self, _p: &mut dyn Pager, _s: &CreateIndex, _sql: &str) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn drop_object(&mut self, _p: &mut dyn Pager, _s: &Drop) -> Result<()> {
            unimplemented!("test catalog is static")
        }
    }

    fn col(name: &str, decl: &str) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: Some(decl.to_string()),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: None,
            default: None,
            default_value: None,
            generated: None,
        }
    }

    /// The common base table `t(a INTEGER, b TEXT)`.
    fn t_table() -> TableDef {
        TableDef {
            name: "t".to_string(),
            columns: vec![col("a", "INTEGER"), col("b", "TEXT")],
            root_page: 2,
            without_rowid: false,
            rowid_alias: None,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        }
    }

    fn view_def(name: &str, sql: &str) -> ViewDef {
        ViewDef { name: name.to_string(), sql: sql.to_string() }
    }

    fn plan_with(tables: Vec<TableDef>, views: Vec<ViewDef>, sql: &str) -> Result<Plan> {
        let cat = TestCatalog { tables, views };
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &cat)
    }

    /// The operator directly under the root `Project`.
    fn under_project(root: &PlanNode) -> &PlanNode {
        match root {
            PlanNode::Project { input, .. } => input,
            other => panic!("expected a Project at the root, got {other:?}"),
        }
    }

    #[test]
    fn view_lowers_to_a_ctescan_over_a_materialized_body() {
        // `SELECT * FROM v` inlines the view like a derived table: a materialized CTE
        // (compiled ONCE) scanned by a CteScan — the performance contract, not a re-scan
        // of the view text per row.
        let plan = plan_with(
            vec![t_table()],
            vec![view_def("v", "CREATE VIEW v AS SELECT a, b FROM t")],
            "SELECT * FROM v",
        )
        .expect("plan ok");

        match under_project(&plan.root) {
            PlanNode::CteScan { id, column_count } => {
                assert_eq!(*id, 0, "the sole view materializes as ctes[0]");
                assert_eq!(*column_count, 2, "view v exposes 2 columns");
            }
            other => panic!("expected the FROM leaf to be a CteScan, got {other:?}"),
        }
        assert_eq!(plan.ctes.len(), 1, "exactly one materialized body registered");
        match &plan.ctes[0] {
            CtePlan::Materialized { name, column_count, body } => {
                assert_eq!(name, "v", "the CTE takes the view's exposed name");
                assert_eq!(*column_count, 2);
                assert!(
                    matches!(under_project(body), PlanNode::SeqScan(s) if s.table == "t"),
                    "the view body scans base table t, got {body:?}"
                );
            }
            other => panic!("expected a Materialized view body, got {other:?}"),
        }
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn view_output_names_come_from_the_select() {
        let plan = plan_with(
            vec![t_table()],
            vec![view_def("v", "CREATE VIEW v AS SELECT a, b FROM t")],
            "SELECT * FROM v",
        )
        .expect("plan ok");
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn explicit_column_list_renames_the_view_output() {
        let plan = plan_with(
            vec![t_table()],
            vec![view_def("v", "CREATE VIEW v(x, y) AS SELECT a, b FROM t")],
            "SELECT * FROM v",
        )
        .expect("plan ok");
        assert_eq!(plan.result_columns, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn column_list_arity_mismatch_is_an_error() {
        let err = plan_with(
            vec![t_table()],
            vec![view_def("v", "CREATE VIEW v(x, y, z) AS SELECT a, b FROM t")],
            "SELECT * FROM v",
        )
        .expect_err("an arity mismatch must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("expected 3 columns") && msg.contains("but got 2"),
            "expected an arity-mismatch message, got: {msg}"
        );
    }

    #[test]
    fn directly_self_referential_view_errors_and_does_not_hang() {
        // The guard turns an infinite recursion into a loud error before Phase 2 runs.
        let err = plan_with(
            vec![],
            vec![view_def("v", "CREATE VIEW v AS SELECT * FROM v")],
            "SELECT * FROM v",
        )
        .expect_err("a self-referential view must be rejected");
        assert!(
            err.to_string().contains("circularly defined"),
            "expected a circular-view error, got: {err}"
        );
    }

    #[test]
    fn mutually_recursive_views_error_circular() {
        let err = plan_with(
            vec![],
            vec![
                view_def("a", "CREATE VIEW a AS SELECT * FROM b"),
                view_def("b", "CREATE VIEW b AS SELECT * FROM a"),
            ],
            "SELECT * FROM a",
        )
        .expect_err("a transitive cycle must be rejected");
        assert!(
            err.to_string().contains("circularly defined"),
            "expected a circular-view error, got: {err}"
        );
    }

    #[test]
    fn view_over_a_view_registers_both_bodies() {
        // v2 → v1 → t: two materialized bodies; the outer FROM scans the last-registered.
        let plan = plan_with(
            vec![t_table()],
            vec![
                view_def("v1", "CREATE VIEW v1 AS SELECT a FROM t"),
                view_def("v2", "CREATE VIEW v2 AS SELECT a FROM v1"),
            ],
            "SELECT * FROM v2",
        )
        .expect("plan ok");
        assert_eq!(plan.ctes.len(), 2, "v1's body and v2's body are both materialized");
        match under_project(&plan.root) {
            PlanNode::CteScan { id, column_count } => {
                assert_eq!(*id, 1, "outer FROM scans v2 = ctes[1]");
                assert_eq!(*column_count, 1);
            }
            other => panic!("expected a CteScan for the outer view, got {other:?}"),
        }
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
    }

    #[test]
    fn view_body_binds_base_table_not_an_outer_cte_of_the_same_name() {
        // Isolation proof: the view's `FROM t` must bind to the BASE table `t(a, b)`,
        // never to the referencing query's CTE `t` (which has only column `a`). If the
        // outer CTE leaked in, the view body's `SELECT a, b` would fail to resolve `b`,
        // so a SUCCESSFUL plan is itself the proof — and we further pin that the view
        // body scans the base table, not a CteScan over the outer CTE.
        let plan = plan_with(
            vec![t_table()],
            vec![view_def("v", "CREATE VIEW v AS SELECT a, b FROM t")],
            "WITH t AS (SELECT 1 AS a) SELECT a, b FROM v",
        )
        .expect("isolation lets the view body resolve `b` against base table t");
        let view_body = plan
            .ctes
            .iter()
            .find_map(|c| match c {
                CtePlan::Materialized { column_count: 2, body, .. } => Some(body),
                _ => None,
            })
            .expect("the view body is the width-2 materialized CTE");
        assert!(
            matches!(under_project(view_body), PlanNode::SeqScan(s) if s.table == "t"),
            "the view body must scan base table t, not the outer CTE t; got {:?}",
            under_project(view_body)
        );
    }
}
