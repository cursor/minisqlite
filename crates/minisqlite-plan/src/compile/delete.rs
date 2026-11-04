//! `DELETE` compilation: lower a parsed [`minisqlite_sql::Delete`] into a
//! [`PlanNode::Delete`](crate::plan::PlanNode::Delete) over the scanned rows.
//!
//! Delegation seam: [`compile_delete`] is the single entrypoint the planner routes
//! every `DELETE` through, so the feature lands here rather than in the shared
//! dispatcher.
//!
//! The scan produces `[c0..c_{N-1}, rowid]` (width `N+1`, rowid at register `N`, per
//! the shared ROW/REGISTER convention in [`crate::plan`]); the executor removes each
//! row by that trailing rowid. `WHERE` and each `RETURNING` expression bind against
//! that single-table layout. Physical row removal is the executor/storage's job — this
//! only compiles the plan.

use minisqlite_catalog::TableDef;
use minisqlite_expr::EvalExpr;
use minisqlite_sql::ResultColumn;
use minisqlite_types::{DbIndex, Error, Result};

use crate::access::SeqScan;
use crate::access_path::plan_table_access;
use crate::bind::{bind_expr, Scope, Source};
use crate::colname::result_column_name;
use crate::compile::cte::register_with;
use crate::compile::index_expr::{
    compile_table_index_key_exprs, compile_table_index_partial_predicates,
};
use crate::compile::trigger::{compile_triggers, TriggerDmlEvent};
use crate::plan::{Delete, PlanNode};
use crate::plan_ctx::PlanCtx;

/// Compile a top-level `DELETE` into its plan tree and output column names (the latter
/// empty unless a `RETURNING` clause is present). See [`compile_delete_with_parent`] for
/// the trigger-action form.
pub fn compile_delete(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Delete,
) -> Result<(PlanNode, Vec<String>)> {
    compile_delete_with_parent(ctx, stmt, None)
}

/// Compile a `DELETE`, optionally as a trigger ACTION under an enclosing OLD/NEW
/// `parent` scope. With `parent = Some(..)` the target's own columns sit at
/// `base = parent.total_width()` (= `2W`) and `NEW`/`OLD` resolve through the parent;
/// NO triggers are attached (only a TOP-LEVEL DELETE attaches its direct triggers, so
/// expansion stays one level deep). `None` is exactly the top-level path.
pub fn compile_delete_with_parent(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Delete,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    // Pull the catalog reference out of `ctx` first (it is `Copy`), so the borrowed
    // `td` carries the catalog's lifetime rather than a borrow of `ctx` — leaving `&mut
    // ctx` free for the binder below. (Same borrow shape as `compile::select`.)
    let catalog = ctx.catalog;
    // Resolve the target BEFORE the unsupported-clause gap below: SQLite reports a
    // view/missing target first (`cannot modify v because it is a view` / `no such
    // table`), so `WITH c AS (…) DELETE FROM a_view` must not report the WITH gap instead.
    // A VIEW owns no b-tree; it is updatable ONLY through a matching `INSTEAD OF DELETE`
    // trigger — when one exists, redirect the DELETE through it (`compile::instead_of`,
    // firing the body per matching view row). Without such a trigger it keeps SQLite's
    // exact `cannot modify … because it is a view`.
    // Resolve the target NAMESPACE (schema qualifier → fixed DbIndex; unqualified → search
    // order, temp shadows main), then fetch the table FROM it so the def, scan, and node all
    // name the same store. An unknown qualifier / a name in no live namespace is "no such
    // table". The `db` is reused for the source, the scan, and the Delete node below.
    let Some(db) = crate::compile::from::resolve_ref_db(catalog, &stmt.table)? else {
        return Err(Error::sql(format!(
            "no such table: {}",
            crate::compile::from::qualified_table_name(&stmt.table)
        )));
    };
    let td = match catalog.table_in(db, &stmt.table.name)? {
        Some(td) => td,
        None => {
            return match catalog.view_in(db, &stmt.table.name)? {
                Some(view) => {
                    crate::compile::instead_of::compile_view_delete(ctx, db, view, stmt, parent)
                }
                None => Err(Error::sql(format!(
                    "no such table: {}",
                    crate::compile::from::qualified_table_name(&stmt.table)
                ))),
            };
        }
    };

    // A leading `WITH` registers its CTEs (mirroring the SELECT path) so the DELETE's WHERE
    // subqueries resolve the CTE names via the shared FROM/`lookup_cte` path. The compiled
    // CtePlans ride in `ctx.ctes` -> `Plan::ctes`, which the executor materializes before the
    // DELETE (the SAME plan-level attach SELECT uses — the Delete node is unchanged). The
    // guard keeps the CTEs visible through WHERE + RETURNING binding below, then is `drop`ped
    // explicitly BEFORE `compile_triggers` (see there) so they are invisible inside trigger
    // bodies (an early `?` before that drop still unwinds the scope via RAII). Registered
    // after target resolution so a missing/view target reports its own error first. (The
    // `indexed` INDEXED BY / NOT INDEXED hint is only an access-path hint the planner is free
    // to ignore.)
    let cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    let table = td.name.clone();
    let n = td.columns.len();

    // A trigger action places the target's own columns AFTER the OLD++NEW row
    // (`base = 2W`); a top-level DELETE has no outer row (`base = 0`). Uses the shared
    // child-placement helper (identical to `total_width()` for the `grouping = None` parents
    // this is reached with) so the outer-width rule has one authority — see `compile_subplan`.
    let base_offset = parent.map_or(0, |p| p.outer_row_width_for_child());
    let src = Source::BaseTable {
        exposed_name: stmt.alias.clone().unwrap_or_else(|| td.name.clone()),
        table: td,
        db,
        base: base_offset,
    };
    let sources = [src];
    let scope = Scope::with_parent(&sources, parent);

    // Bind WHERE, then RETURNING (textual order), so anonymous `?` parameters number
    // as SQLite assigns them.
    let bound_where = stmt.where_clause.as_ref().map(|w| bind_expr(&scope, ctx, w)).transpose()?;
    let scan = build_scan(catalog, td, db, base_offset, bound_where)?;

    let (returning, result_columns) = bind_returning(&scope, ctx, &stmt.returning)?;

    // Drop the CTE name scope BEFORE compiling triggers: a trigger body is a separate
    // program compiled under a fresh `PlanCtx` that shares the thread-local CTE scope, and a
    // firing statement's CTE must be invisible inside it (`lang_with.html` §1 — see
    // `compile_insert_with_parent` for the full rationale). WHERE + RETURNING binding is
    // already complete above; dropping pops only the name->id entries, leaving `ctx.ctes`
    // (-> the Delete plan's CTEs) intact. (Regression: `conformance_dml_cte_trigger`.)
    drop(cte_guard);

    // Attach the triggers this DELETE fires — only for a TOP-LEVEL statement (a trigger
    // action does not recurse into the triggers of the table it deletes from; the
    // executor drives runtime recursion).
    let triggers = if parent.is_none() {
        compile_triggers(ctx, catalog, db, td, TriggerDmlEvent::Delete)?
    } else {
        Vec::new()
    };

    // Expression-index key programs, bound against the deleted row's `[c0..c_{N-1}, rowid]`
    // frame, so the executor removes the right entry from a `CREATE INDEX i ON t(<expr>)`
    // as rows are deleted (`lang_createindex.html` §1.2). Empty when no index on `t` has an
    // expression key column.
    let index_key_exprs = compile_table_index_key_exprs(ctx, db, td)?;

    // Partial-index WHERE predicates, bound against that SAME deleted-row frame. The executor
    // removes a partial index's entry ONLY when the predicate is TRUE for the deleted row —
    // a row that was never in the index has no entry to remove (`partialindex.html` §2).
    // Empty when no index on `t` is partial.
    let index_partial_predicates = compile_table_index_partial_predicates(ctx, db, td)?;

    let node = PlanNode::Delete(Delete {
        table,
        db,
        column_count: n,
        scan: Box::new(scan),
        returning,
        triggers,
        index_key_exprs,
        index_partial_predicates,
    });
    Ok((node, result_columns))
}

/// Build the row-source scan for the delete target. Mirrors
/// [`crate::compile::update`]'s `build_scan`: index-aware [`plan_table_access`] at
/// `base_offset == 0` (a top-level DELETE), and a plain [`SeqScan`] + residual
/// [`Filter`](PlanNode::Filter) over the whole predicate at `base_offset > 0` (a trigger
/// action, whose OLD/NEW references live in the leading `[0, 2W)` registers and would be
/// mis-folded into a seek by the 0-based `plan_table_access` — see that helper's note).
fn build_scan(
    catalog: &dyn minisqlite_catalog::Catalog,
    td: &TableDef,
    db: DbIndex,
    base_offset: usize,
    bound_where: Option<EvalExpr>,
) -> Result<PlanNode> {
    if base_offset == 0 {
        let access = plan_table_access(catalog, td, db, bound_where)?;
        return Ok(match access.residual {
            Some(pred) => PlanNode::Filter { input: Box::new(access.node), predicate: pred },
            None => access.node,
        });
    }
    let scan = PlanNode::SeqScan(SeqScan {
        table: td.name.clone(),
        db,
        column_count: td.columns.len(),
    });
    Ok(match bound_where {
        Some(pred) => PlanNode::Filter { input: Box::new(scan), predicate: pred },
        None => scan,
    })
}

/// Bind a `RETURNING` list into `(exprs, output names)` against the scanned
/// `[cols.., rowid]` row — the same projection semantics as a SELECT result column:
/// `*` / `table.*` expand in place, an aliased/bare expression takes its alias or its
/// reconstructed source-text name. An empty clause yields two empty vectors.
///
/// NOTE: an identical helper lives in the sibling `compile::update`; the two DML
/// compilers are separate contention cells (own files), so the small duplication is
/// intentional rather than a shared module both must edit.
fn bind_returning(
    scope: &Scope,
    ctx: &mut PlanCtx,
    returning: &[ResultColumn],
) -> Result<(Vec<EvalExpr>, Vec<String>)> {
    let mut exprs = Vec::new();
    let mut names = Vec::new();
    for col in returning {
        match col {
            ResultColumn::Expr { expr, alias } => {
                exprs.push(bind_expr(scope, ctx, expr)?);
                names.push(alias.clone().unwrap_or_else(|| result_column_name(expr)));
            }
            ResultColumn::Star => push_star(scope, None, &mut exprs, &mut names)?,
            ResultColumn::TableStar(t) => push_star(scope, Some(t), &mut exprs, &mut names)?,
        }
    }
    Ok((exprs, names))
}

/// Expand `*` / `table.*` into one `Column(reg)` projection per column, with its name.
fn push_star(
    scope: &Scope,
    table: Option<&str>,
    exprs: &mut Vec<EvalExpr>,
    names: &mut Vec<String>,
) -> Result<()> {
    for (reg, name) in scope.expand_star(table)? {
        exprs.push(EvalExpr::Column(reg));
        names.push(name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_expr::CmpOp;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};

    use crate::access::RowidOp;
    use crate::plan::CtePlan;
    use crate::{Plan, Planner, QueryPlanner};

    /// A static in-memory catalog for planning tests (only reads are exercised).
    struct TestCatalog {
        tables: Vec<TableDef>,
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

    /// `t(a INTEGER, b TEXT, c TEXT)` — no rowid alias, so `N = 3` and the rowid sits
    /// at register 3.
    fn cat_t() -> TestCatalog {
        TestCatalog {
            tables: vec![TableDef {
                name: "t".to_string(),
                columns: vec![col("a", Some("INTEGER")), col("b", Some("TEXT")), col("c", Some("TEXT"))],
                root_page: 2,
                without_rowid: false,
                rowid_alias: None,
                auto_indexes: Vec::new(),
                checks: Vec::new(),
                foreign_keys: Vec::new(),
                autoincrement: false,
                primary_key: Vec::new(),
                strict: false,
            }],
        }
    }

    fn plan_sql(sql: &str, cat: &dyn Catalog) -> Result<Plan> {
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("expected one statement");
        QueryPlanner::new().plan(stmt, cat)
    }

    fn expect_delete(plan: &Plan) -> &Delete {
        match &plan.root {
            PlanNode::Delete(d) => d,
            other => panic!("expected Delete at the root, got {other:?}"),
        }
    }

    #[test]
    fn non_rowid_where_becomes_a_filter_over_seqscan() {
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t WHERE a > 3", &cat).unwrap();
        assert!(plan.mutates, "DELETE mutates");
        let d = expect_delete(&plan);
        assert_eq!(d.table, "t");
        assert_eq!(d.column_count, 3);
        // `a` is a normal column (not the rowid), so it stays a residual Filter over a
        // full SeqScan.
        match d.scan.as_ref() {
            PlanNode::Filter { input, predicate } => {
                assert!(
                    matches!(predicate, EvalExpr::Compare { op: CmpOp::Gt, .. }),
                    "the residual predicate is `a > 3`"
                );
                assert!(matches!(input.as_ref(), PlanNode::SeqScan(_)));
            }
            other => panic!("expected a Filter over SeqScan, got {other:?}"),
        }
        assert!(d.returning.is_empty());
        assert!(plan.result_columns.is_empty());
    }

    #[test]
    fn no_where_delete_scans_the_whole_table() {
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t", &cat).unwrap();
        let d = expect_delete(&plan);
        assert!(matches!(d.scan.as_ref(), PlanNode::SeqScan(_)), "no WHERE => full SeqScan");
    }

    #[test]
    fn rowid_eq_delete_seeks_and_returning_star_expands() {
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t WHERE rowid = 1 RETURNING *", &cat).unwrap();
        let d = expect_delete(&plan);
        // `rowid = 1` is consumed by the access path — a bare RowidScan seek.
        match d.scan.as_ref() {
            PlanNode::RowidScan(s) => assert!(matches!(s.op, RowidOp::Eq(_)), "rowid = 1 is an Eq seek"),
            other => panic!("expected a RowidScan seek, got {other:?}"),
        }
        // `RETURNING *` expands to all three columns in schema order (the rowid is not
        // part of `*`).
        assert_eq!(d.returning.len(), 3);
        assert!(matches!(d.returning[0], EvalExpr::Column(0)));
        assert!(matches!(d.returning[1], EvalExpr::Column(1)));
        assert!(matches!(d.returning[2], EvalExpr::Column(2)));
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn returning_rowid_reads_the_trailing_register() {
        // RETURNING binds against `[c0,c1,c2, rowid]`, so `rowid` is register N = 3.
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t RETURNING rowid", &cat).unwrap();
        let d = expect_delete(&plan);
        assert_eq!(d.returning.len(), 1);
        assert!(matches!(d.returning[0], EvalExpr::Column(3)), "rowid is the trailing register N=3");
        assert_eq!(plan.result_columns, vec!["rowid".to_string()]);
    }

    #[test]
    fn rowid_range_delete_is_a_range_scan() {
        // A rowid range bound is consumed into a RowidScan range (no residual Filter).
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t WHERE rowid >= 10", &cat).unwrap();
        let d = expect_delete(&plan);
        match d.scan.as_ref() {
            PlanNode::RowidScan(s) => assert!(
                matches!(s.op, RowidOp::Range { .. }),
                "rowid >= 10 is a range seek, got {:?}",
                s.op
            ),
            other => panic!("expected a RowidScan range, got {other:?}"),
        }
    }

    #[test]
    fn aliased_target_resolves_qualified_references_and_table_star() {
        // With `DELETE FROM t AS x`, the exposed name is the alias: `x.a` resolves,
        // `x.*` expands every column, and the original `t.a` no longer resolves — the
        // DELETE twin of the UPDATE aliased-target pin.
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t AS x WHERE x.a = 2 RETURNING x.*", &cat).unwrap();
        let d = expect_delete(&plan);
        assert_eq!(d.returning.len(), 3, "`x.*` expands to all three columns");
        assert!(matches!(d.returning[0], EvalExpr::Column(0)));
        assert!(matches!(d.returning[2], EvalExpr::Column(2)));
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        // `x.a = 2` (a is not the rowid) is a residual Filter over the scan.
        assert!(matches!(d.scan.as_ref(), PlanNode::Filter { .. }));

        let err = plan_sql("DELETE FROM t AS x WHERE t.a = 2", &cat).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "`t.a` under alias `x` is unresolved, got {err:?}");
    }

    #[test]
    fn returning_alias_takes_its_name() {
        // The alias arm (`AS x`) names the RETURNING output column.
        let cat = cat_t();
        let plan = plan_sql("DELETE FROM t RETURNING a AS x", &cat).unwrap();
        let d = expect_delete(&plan);
        assert_eq!(d.returning.len(), 1);
        assert!(matches!(d.returning[0], EvalExpr::Column(0)));
        assert_eq!(plan.result_columns, vec!["x".to_string()], "explicit alias wins");
    }

    #[test]
    fn unknown_table_is_a_loud_error() {
        let cat = cat_t();
        let err = plan_sql("DELETE FROM missing", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such table"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn with_on_delete_registers_cte_and_compiles() {
        // A leading `WITH` on DELETE is supported: the CTE compiles into `plan.ctes` and the
        // WHERE subquery `(SELECT n FROM x)` resolves `x` through the shared FROM/`lookup_cte`
        // path — proving the CTE is visible to the DELETE's WHERE binding. The executor
        // materializes `plan.ctes` before the DELETE (a plan-level attach; the Delete node is
        // unchanged). Was previously a loud "not yet supported" gap.
        let cat = cat_t();
        let plan =
            plan_sql("WITH x AS (SELECT 1 AS n) DELETE FROM t WHERE a IN (SELECT n FROM x)", &cat)
                .unwrap();
        assert_eq!(plan.ctes.len(), 1, "the CTE `x` is registered, got {:?}", plan.ctes);
        assert!(
            matches!(&plan.ctes[0], CtePlan::Materialized { name, .. } if name == "x"),
            "ctes[0] is the materialized CTE `x`, got {:?}",
            plan.ctes[0]
        );
        assert_eq!(plan.subqueries.len(), 1, "the WHERE IN-subquery over CTE x is registered");
        let d = expect_delete(&plan);
        // `a IN (...)` is a residual Filter over the SeqScan (a is not the rowid).
        assert!(matches!(d.scan.as_ref(), PlanNode::Filter { .. }), "got {:?}", d.scan);
    }
}
