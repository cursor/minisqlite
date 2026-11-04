//! Index-expression compilation: bind an index's stored key EXPRESSIONS into the
//! executable [`EvalExpr`]s an index-maintenance path evaluates per row.
//!
//! Per `spec/sqlite-doc/lang_createindex.html` §1.2 an index key may be an EXPRESSION
//! over the table's own columns (`CREATE INDEX i ON t(a+b)` / `t(lower(a))`), not just a
//! named column. Such a key's value is COMPUTED by evaluating the expression against
//! each row — on the read/lookup side and when the index is maintained across
//! INSERT/UPDATE/DELETE. Those EVALUATION semantics are the executor's job; this only
//! carries the raw BOUND key expression, mirroring how [`crate::compile::check`] binds a
//! CHECK predicate the executor later evaluates.
//!
//! Binding: each key expression binds against a single base-table [`Source`] over `def`
//! at register base 0 — the SAME scope [`crate::compile::check::compile_checks`] uses for
//! CHECK — so a column reference resolves exactly as every other DML expression does:
//! column `i` → `Column(i)`, and an INTEGER PRIMARY KEY alias column → the trailing
//! rowid register `N` (where its logical value lives in the executor's `[c0..c_{N-1},
//! rowid]` row). That is EXACTLY the executor's logical-row layout, so a compiled key
//! expression evaluates correctly at index-maintenance time with no remapping.
//!
//! This helper is the planner-binder HALF of expression-index support, paired with the
//! catalog's [`minisqlite_catalog::IndexDef::key_exprs`] storage slot (which holds the
//! parsed key expressions, `None` for an ordinary named key column). The DML compilers
//! ([`crate::compile::insert`] / `update` / `delete`) call it to populate each node's
//! `index_key_exprs`, and [`crate::query_planner::QueryPlanner::index_key_programs`] calls
//! it to key a `CREATE INDEX` backfill — both feeding the executor's index maintenance.

use minisqlite_catalog::TableDef;
use minisqlite_expr::EvalExpr;
use minisqlite_sql::Expr;
use minisqlite_types::{DbIndex, Result};

use crate::bind::{bind_expr, Scope, Source};
use crate::plan_ctx::PlanCtx;

/// Bind an index's per-key-column expressions (`key_exprs`, parallel to the index's
/// `columns` / `key_columns`) into a parallel `Vec<Option<EvalExpr>>` of the SAME length:
/// `Some(expr)` binds to `Some(EvalExpr)`, an ordinary named key column (`None`) stays
/// `None`. This mirrors [`compile_checks`](crate::compile::check::compile_checks) exactly
/// — one base-table scope over `def` at base 0, then bind each entry against it.
///
/// A key expression that names a column the table does not have fails to BIND here (an
/// `Error::Sql("no such column: …")`), matching real SQLite rejecting such a CREATE INDEX
/// — the gap surfaces loudly at plan time rather than being silently dropped. The one
/// exception is a bare *double-quoted* unknown name, which SQLite's DQS legacy folds to a
/// text literal (quirks.html §8), applied uniformly here because binding goes through
/// `bind_expr`.
///
/// ASSUMPTION: an index key expression references only the indexed table's own columns —
/// `lang_createindex.html` §1.2 forbids a subquery, bind parameter, or non-deterministic
/// function in an index expression, so a well-formed `key_exprs` never carries one. That
/// keeps this bind INERT w.r.t. the planner's numbering state (`bind_expr` advances none
/// of `ctx`'s param/subquery/CTE counters here), exactly as `compile_checks` relies on for
/// CHECK. If a future path ever admits such a form, re-check that inertness.
pub(crate) fn compile_index_key_exprs(
    ctx: &mut PlanCtx,
    def: &TableDef,
    key_exprs: &[Option<Expr>],
) -> Result<Vec<Option<EvalExpr>>> {
    if key_exprs.is_empty() {
        return Ok(Vec::new());
    }
    // A single base-table scope over `def` at base 0: column `i` → `Column(i)`, and the
    // INTEGER PRIMARY KEY alias → the rowid register `N` — the executor's `[c0..c_{N-1},
    // rowid]` logical-row layout, so a bound key expr evaluates against a maintained row
    // with no remap. Identical to `compile::check`.
    let sources =
        [Source::BaseTable { exposed_name: def.name.clone(), table: def, db: DbIndex::MAIN, base: 0 }];
    let scope = Scope::new(&sources);

    let mut out = Vec::with_capacity(key_exprs.len());
    for entry in key_exprs {
        out.push(match entry {
            Some(e) => Some(bind_expr(&scope, ctx, e)?),
            None => None,
        });
    }
    Ok(out)
}

/// Compile the EXPRESSION-index key programs for every index on `def` that has at least
/// one genuine expression key column, as `(index name, per-key-column compiled exprs)`
/// pairs — the `index_key_exprs` an [`Insert`](crate::plan::Insert) /
/// [`Update`](crate::plan::Update) / [`Delete`](crate::plan::Delete) node carries so the
/// executor can evaluate each maintained row's index key. Mirrors
/// [`compile_checks`](crate::compile::check::compile_checks): one call, resolved from the
/// planning context's catalog, binding against the target table's `[c0..c_{N-1}, rowid]`
/// frame — the SAME `def` the executor's index-maintenance frame uses.
///
/// An ordinary (all-named-column) index is OMITTED: its key needs no compiled expression,
/// and the executor keys it from the row's columns directly. Only an index with a `Some`
/// in its `key_exprs` (`lang_createindex.html` §1.2) is included, so the returned list is
/// exactly the indexes whose maintenance requires an evaluated key.
///
/// `pub` (not `pub(crate)`) so the EXECUTOR can compile a table's index key programs for a
/// path that discovers its target table at RUNTIME rather than at plan time — the FK
/// `ON DELETE CASCADE`/`SET NULL` cascade into a child, which finds each child table (and
/// its indexes) dynamically as it recurses. That path constructs a [`PlanCtx`] over the
/// executor's builtin function registry and calls this, mirroring how the trigger firing
/// path recompiles a trigger action's own triggers at runtime.
pub fn compile_table_index_key_exprs(
    ctx: &mut PlanCtx,
    db: DbIndex,
    def: &TableDef,
) -> Result<Vec<(String, Vec<Option<EvalExpr>>)>> {
    // Copy the catalog reference out (it is `Copy`) so the `indexes_on` borrow is tied to
    // the catalog, not to `ctx` — leaving `ctx` free for the mutable `compile_index_key_exprs`
    // binder call in the loop. Same pattern the DML compilers use to resolve `def`.
    let catalog = ctx.catalog;
    let mut out = Vec::new();
    // `db` is the namespace `def` was resolved in. Read that store's indexes with `_in(db)`:
    // the bare `indexes_on` follows the search order, so a same-named temp/attached shadow
    // could feed the WRONG namespace's expression-index key programs (matched by index name in
    // `build_index_plans`, which reads `indexes_on_in(db)`), leaving a real expression index
    // with no compiled key. With no shadow `db == MAIN` and this equals the bare form.
    for idx in catalog.indexes_on_in(db, &def.name)? {
        if idx.key_exprs.iter().any(|e| e.is_some()) {
            let compiled = compile_index_key_exprs(ctx, def, &idx.key_exprs)?;
            out.push((idx.name.clone(), compiled));
        }
    }
    Ok(out)
}

/// Bind ONE partial-index WHERE predicate (`IndexDef::partial_predicate`) into an executable
/// [`EvalExpr`], against the SAME single base-table `[c0..c_{N-1}, rowid]` scope
/// [`compile_index_key_exprs`] and [`compile_checks`](crate::compile::check::compile_checks)
/// use — so a column reference in the predicate resolves exactly as every other DML
/// expression does (column `i` → `Column(i)`, an INTEGER PRIMARY KEY alias → the rowid
/// register `N`). The executor evaluates its truthiness per maintained row to decide index
/// membership; this only carries the bound predicate.
///
/// A predicate naming a column the table lacks fails to BIND here (a loud `no such column`),
/// matching real SQLite rejecting such a `CREATE INDEX` rather than silently dropping it.
///
/// ASSUMPTION: a partial-index predicate references only the indexed table's own columns and
/// contains no subquery / bind parameter / non-deterministic function (`partialindex.html`:
/// "The WHERE clause may not contain subqueries, references to other tables, non-deterministic
/// functions, or bound parameters"; the subquery form is rejected at `CREATE INDEX` time in
/// `index_def_from_ast`). That keeps this bind INERT w.r.t. `ctx`'s param/subquery/CTE
/// counters, exactly as `compile_index_key_exprs` / `compile_checks` rely on.
pub(crate) fn compile_index_partial_predicate(
    ctx: &mut PlanCtx,
    def: &TableDef,
    predicate: &Expr,
) -> Result<EvalExpr> {
    // A single base-table scope over `def` at base 0 — identical to `compile_index_key_exprs`
    // and `compile::check`, so the predicate binds against the executor's `[c0..c_{N-1},
    // rowid]` logical-row layout with no remap.
    let sources =
        [Source::BaseTable { exposed_name: def.name.clone(), table: def, db: DbIndex::MAIN, base: 0 }];
    let scope = Scope::new(&sources);
    bind_expr(&scope, ctx, predicate)
}

/// Compile the WHERE predicate of every PARTIAL index on `def`, as `(index name, bound
/// predicate)` pairs — the `index_partial_predicates` an [`Insert`](crate::plan::Insert) /
/// [`Update`](crate::plan::Update) / [`Delete`](crate::plan::Delete) node carries so the
/// executor can gate each index's per-row maintenance on it. Mirrors
/// [`compile_table_index_key_exprs`]: one call, resolved from the planning context's catalog,
/// binding against the target table's `[c0..c_{N-1}, rowid]` frame.
///
/// A NON-partial index (`partial_predicate == None`) is OMITTED — its maintenance is
/// unconditional — so the returned list is exactly the partial indexes, keyed by name for the
/// executor's `build_index_plans` to attach to each [`IndexPlan`]. The common (no partial
/// index) case returns an empty vec and costs nothing.
///
/// `pub` for the same reason as [`compile_table_index_key_exprs`]: the FK cascade discovers a
/// child table (and its indexes) at RUNTIME and must gate a partial child index the same way,
/// so it constructs a [`PlanCtx`] over the executor's builtin registry and calls this.
pub fn compile_table_index_partial_predicates(
    ctx: &mut PlanCtx,
    db: DbIndex,
    def: &TableDef,
) -> Result<Vec<(String, EvalExpr)>> {
    // Copy the catalog reference out (it is `Copy`) so the `indexes_on_in` borrow is tied to
    // the catalog, not to `ctx`, leaving `ctx` free for the mutable binder call in the loop —
    // the same borrow shape `compile_table_index_key_exprs` uses.
    let catalog = ctx.catalog;
    let mut out = Vec::new();
    // Read this table's indexes in the resolved namespace (`_in(db)`), for the same
    // shadow-safety reason as the key-expr compiler: `build_index_plans` matches these by name
    // against `indexes_on_in(db)`, so a bare search-order read could bind the wrong namespace's
    // predicate. With no shadow `db == MAIN` and this equals the bare form.
    for idx in catalog.indexes_on_in(db, &def.name)? {
        if let Some(predicate) = &idx.partial_predicate {
            let compiled = compile_index_partial_predicate(ctx, def, predicate)?;
            out.push((idx.name.clone(), compiled));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef};
    use minisqlite_expr::ArithOp;
    use minisqlite_functions::FunctionRegistry;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop, ResultColumn, SelectBody, SelectCore, Statement};

    /// A static in-memory catalog for planning tests (only reads are exercised). The
    /// index-expr binder resolves columns through the `def`-built scope, not the catalog,
    /// so its lookups are never hit by the `a + b` cases below — but `PlanCtx` needs one.
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

    /// `t(a INTEGER, b INTEGER)` — no rowid alias, so `a` → `Column(0)`, `b` → `Column(1)`.
    fn tdef_ab() -> TableDef {
        TableDef {
            name: "t".to_string(),
            columns: vec![col("a", "INTEGER"), col("b", "INTEGER")],
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

    /// Parse a scalar expression via a `SELECT <text>` wrapper and pluck the single
    /// projection expression — the same trick `compile::update`'s test `check_ast` uses to
    /// obtain a stored predicate's `Expr`.
    fn expr_of(text: &str) -> Expr {
        let ast = parse(&format!("SELECT {text}")).expect("expression parses");
        let [Statement::Select(select)] = ast.statements.as_slice() else {
            panic!("expected one SELECT for {text:?}");
        };
        let SelectBody::Select(SelectCore::Query { columns, .. }) = &select.body else {
            panic!("expected a query core for {text:?}");
        };
        match columns.as_slice() {
            [ResultColumn::Expr { expr, .. }] => expr.clone(),
            other => panic!("expected one projection for {text:?}, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_binds_to_empty_output() {
        // `&[]` → `[]`: no key columns, nothing to bind.
        let cat = TestCatalog { tables: vec![tdef_ab()] };
        let reg = FunctionRegistry::empty();
        let def = tdef_ab();
        let mut ctx = PlanCtx::new(&reg, &cat);
        let out = compile_index_key_exprs(&mut ctx, &def, &[]).unwrap();
        assert!(out.is_empty(), "empty key_exprs binds to an empty result");
    }

    #[test]
    fn all_none_passes_through_as_none_same_length() {
        // The all-ordinary-column case (every slot `None`): the result is the SAME length,
        // every entry `None` — nothing is bound.
        let cat = TestCatalog { tables: vec![tdef_ab()] };
        let reg = FunctionRegistry::empty();
        let def = tdef_ab();
        let mut ctx = PlanCtx::new(&reg, &cat);
        let out = compile_index_key_exprs(&mut ctx, &def, &[None, None]).unwrap();
        assert_eq!(out.len(), 2, "output is parallel to input (same length)");
        assert!(out.iter().all(|e| e.is_none()), "an ordinary named column binds to None");
    }

    #[test]
    fn genuine_expression_binds_against_the_base_row_layout() {
        // A genuine expression key `a + b` binds against `[a, b, rowid]`: `a` → Column(0),
        // `b` → Column(1) — the base-table-scope contract the DML and backfill paths rely on.
        let cat = TestCatalog { tables: vec![tdef_ab()] };
        let reg = FunctionRegistry::empty();
        let def = tdef_ab();
        let mut ctx = PlanCtx::new(&reg, &cat);
        let out = compile_index_key_exprs(&mut ctx, &def, &[Some(expr_of("a + b"))]).unwrap();
        assert_eq!(out.len(), 1);
        match out[0].as_ref().expect("the expression key binds to Some") {
            EvalExpr::Arith { op: ArithOp::Add, left, right } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(0)), "a → Column(0), got {left:?}");
                assert!(matches!(right.as_ref(), EvalExpr::Column(1)), "b → Column(1), got {right:?}");
            }
            other => panic!("expected Add over Column(0), Column(1), got {other:?}"),
        }
    }

    #[test]
    fn mixed_some_and_none_preserve_position() {
        // A composite `(a+b, c)`-style key mixing an expression slot and an ordinary column
        // slot keeps positions: index 0 binds `a + b`, index 1 stays `None`.
        let cat = TestCatalog { tables: vec![tdef_ab()] };
        let reg = FunctionRegistry::empty();
        let def = tdef_ab();
        let mut ctx = PlanCtx::new(&reg, &cat);
        let key_exprs = [Some(expr_of("a + b")), None];
        let out = compile_index_key_exprs(&mut ctx, &def, &key_exprs).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Some(EvalExpr::Arith { op: ArithOp::Add, .. })), "slot 0 is the bound expr");
        assert!(out[1].is_none(), "slot 1 is an ordinary named column, stays None");
    }

    #[test]
    fn unknown_column_in_expression_fails_closed() {
        // A key expression naming a column the table lacks fails to BIND — a loud
        // `no such column`, matching sqlite rejecting such a CREATE INDEX. Never dropped.
        let cat = TestCatalog { tables: vec![tdef_ab()] };
        let reg = FunctionRegistry::empty();
        let def = tdef_ab();
        let mut ctx = PlanCtx::new(&reg, &cat);
        let err = compile_index_key_exprs(&mut ctx, &def, &[Some(expr_of("a + zzz"))]).unwrap_err();
        match err {
            minisqlite_types::Error::Sql(m) => assert!(m.contains("no such column: zzz"), "got {m:?}"),
            other => panic!("expected a no-such-column Sql error, got {other:?}"),
        }
    }
}
