//! Trigger compilation (the PLAN half of trigger firing): given a base table that an
//! `INSERT`/`UPDATE`/`DELETE` targets, find every trigger that fires for that event and
//! compile each into a [`TriggerProgram`] (its `WHEN` condition and its body statements
//! as `Plan`s) for the DML node to carry. The firing executor (a separate module) runs
//! them; nothing here touches the executor.
//!
//! ## NEW / OLD as a correlated parent scope (the crux)
//!
//! Per the register convention on [`TriggerProgram`](crate::plan::TriggerProgram): let
//! `C` be the target's column count and `W = C + 1` its base-row width. A trigger sees
//! the OLD row in registers `[0, W)` and the NEW row in `[W, 2W)`, and its action
//! statements bind their own columns starting at `2W`. We realize this by building a
//! PARENT [`Scope`] whose two sources are the SAME target table exposed as `OLD` at base
//! `0` and `NEW` at base `W` (see [`new_old_sources`]):
//!
//! * the `WHEN` condition binds DIRECTLY against this `2W`-wide scope, so `NEW.col_k`
//!   resolves to `Column(W+k)`, `OLD.col_k` to `Column(k)`, and the rowids to
//!   `Column(W+C)` / `Column(C)`;
//! * each ACTION statement compiles as a CORRELATED statement with this scope as its
//!   PARENT — its own FROM sources sit at base `2W` and its `NEW`/`OLD` references
//!   resolve THROUGH the parent, exactly as a correlated subquery resolves outer columns
//!   (reusing [`Scope::with_parent`] and the `*_with_parent` DML/SELECT entrypoints).
//!
//! Both halves of OLD/NEW are always exposed; the executor fills the absent half (OLD
//! for INSERT, NEW for DELETE) with NULLs. Only the DIRECT triggers of the DML target
//! are compiled — a trigger action's own DML does not recurse into ITS triggers here
//! (the executor drives that runtime recursion), so compile-time expansion is one level.

use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::{parse, CreateTrigger, Statement, TriggerEvent, TriggerTiming};
use minisqlite_types::{DbIndex, Error, Result};

use crate::bind::{bind_expr, new_old_sources, Scope};
use crate::compile::delete::compile_delete_with_parent;
use crate::compile::insert::compile_insert_with_parent;
use crate::compile::select::compile_select_with_parent;
use crate::compile::update::compile_update_with_parent;
use crate::plan::{Plan, TriggerProgram};
use crate::plan_ctx::PlanCtx;

/// The DML event a set of triggers is being compiled for. `Update` carries the column
/// indices the statement assigns (`changed_cols`) so an `UPDATE OF <cols>` trigger fires
/// only when it names one of them. Folding `changed_cols` INTO the `Update` variant (rather
/// than a separate parameter) makes the invariant "changed columns are meaningful only for
/// an UPDATE" hold by construction — an `Insert`/`Delete`-with-changed-columns, or an
/// `Update` that forgot them, is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerDmlEvent<'a> {
    Insert,
    Update { changed_cols: &'a [usize] },
    Delete,
}

/// Compile every trigger that fires when `event` writes base table `target`, in the
/// catalog's deterministic firing order, into [`TriggerProgram`]s for the DML node.
///
/// For an `UPDATE`, `event` carries the assigned column indices
/// ([`TriggerDmlEvent::Update`]`{ changed_cols }`), used for the `UPDATE OF <cols>` filter:
/// such a trigger fires only when one of its named columns is among them; an `UPDATE` whose
/// trigger has no `OF` list fires on any update.
///
/// Compiles BOTH `BEFORE` and `AFTER` triggers (the executor splits by
/// [`TriggerProgram::timing`]); an omitted timing defaults to `BEFORE`
/// (`lang_createtrigger.html`). `INSTEAD OF` on a table is an error (it is only valid on
/// a view; the view path is [`compile_instead_of_triggers`], which compiles the mirror
/// case — `INSTEAD OF` only, over a synthetic view schema).
pub fn compile_triggers(
    ctx: &mut PlanCtx,
    catalog: &dyn Catalog,
    db: DbIndex,
    target: &TableDef,
    event: TriggerDmlEvent,
) -> Result<Vec<TriggerProgram>> {
    // A trigger lives in the same namespace as its target table (a trigger cannot span
    // databases), and `db` is the namespace the DML resolved `target` to. Read that store's
    // triggers with `_in(db)`: the bare `triggers_on` follows the temp->main->attached search
    // order, so under a same-named temp/attached shadow it would fire the WRONG namespace's
    // triggers (or miss the target's). With no shadow `db == MAIN` and this equals the bare form.
    let defs = catalog.triggers_on_in(db, &target.name)?;
    if defs.is_empty() {
        return Ok(Vec::new());
    }

    // The shared OLD/NEW parent scope for THIS target (OLD@0, NEW@W, width 2W): the
    // WHEN binds against it and every action uses it as parent. Built once and reused.
    let sources = new_old_sources(target);
    let trigger_scope = Scope::new(&sources);

    // The function registry, copied out of `ctx` (a `&FunctionRegistry`, `Copy`) so each
    // action can build its own fresh `PlanCtx` without borrowing the outer one.
    let registry = ctx.registry;

    let mut programs = Vec::with_capacity(defs.len());
    for def in defs {
        let ct = reparse_trigger(&def.sql, &def.name)?;

        if !event_matches(&ct.event, event, target) {
            continue;
        }

        // INSTEAD OF is a view-only timing; on a base table it is invalid. This path only
        // ever compiles base-table triggers, so reject it (fail closed) rather than emit
        // a program the executor would mis-fire.
        if matches!(ct.timing, Some(TriggerTiming::InsteadOf)) {
            return Err(Error::sql(format!(
                "cannot create INSTEAD OF trigger {} on table {} (INSTEAD OF is only valid on a view)",
                ct.name.name, target.name
            )));
        }

        // WHEN binds against the 2W-wide OLD/NEW scope. Bound with the OUTER `ctx`: a
        // `TriggerProgram` stores WHEN as a bare `EvalExpr` with no subquery table of its
        // own, so any (rare) subquery inside WHEN must register into the enclosing DML
        // statement's `Plan::subqueries` — the only home there is. The common WHEN (a
        // comparison of NEW/OLD columns) registers nothing, so this is moot in practice.
        let when = match &ct.when {
            Some(w) => Some(bind_expr(&trigger_scope, ctx, w)?),
            None => None,
        };

        // Each action is a self-contained `Plan` compiled under a FRESH `PlanCtx` (its
        // own subquery ids / parameter numbering), with the OLD/NEW scope as parent.
        let mut actions = Vec::with_capacity(ct.body.len());
        for stmt in &ct.body {
            actions.push(compile_action(registry, catalog, stmt, &trigger_scope)?);
        }

        programs.push(TriggerProgram {
            name: def.name.clone(),
            timing: ct.timing.unwrap_or(TriggerTiming::Before),
            when,
            actions,
        });
    }
    Ok(programs)
}

/// Compile the `INSTEAD OF` triggers a VIEW fires for `event` into [`TriggerProgram`]s —
/// the view counterpart of [`compile_triggers`]. `view` is the SYNTHETIC [`TableDef`]
/// standing in for the view (its columns + affinities; `root_page` 0 and no rowid alias,
/// since a view owns no b-tree — built by `compile::instead_of`), used to build the same
/// OLD/NEW parent scope a base-table trigger uses (OLD@0, NEW@W, `W = C + 1`, so a view's
/// missing rowid is a NULL placeholder register at index `C`).
///
/// UNLIKE [`compile_triggers`], this selects ONLY `INSTEAD OF` triggers — the sole timing
/// that makes a view updatable (`lang_createtrigger.html` §3). A stray `BEFORE`/`AFTER`
/// trigger on a view (which SQLite rejects at CREATE, and the catalog now does too) never
/// fires here. Returns an EMPTY vec when the view has no matching `INSTEAD OF <event>`
/// trigger, so the caller keeps the "cannot modify … because it is a view" error (a view
/// is updatable only for events that have such a trigger).
pub fn compile_instead_of_triggers(
    ctx: &mut PlanCtx,
    catalog: &dyn Catalog,
    db: DbIndex,
    view: &TableDef,
    event: TriggerDmlEvent,
) -> Result<Vec<TriggerProgram>> {
    // `db` is the namespace the view was resolved in; its INSTEAD OF triggers live there.
    // `_in(db)` reads exactly that store, so a same-named shadow in another namespace can't
    // divert the lookup (search order) to the wrong view's triggers. No shadow => equals bare.
    let defs = catalog.triggers_on_in(db, &view.name)?;
    if defs.is_empty() {
        return Ok(Vec::new());
    }

    // The shared OLD/NEW parent scope for the view (OLD@0, NEW@W, width 2W): the WHEN
    // binds against it and every action uses it as parent — exactly as a base-table
    // trigger, but over the synthetic view schema. Built once and reused.
    let sources = new_old_sources(view);
    let trigger_scope = Scope::new(&sources);
    let registry = ctx.registry;

    let mut programs = Vec::with_capacity(defs.len());
    for def in defs {
        let ct = reparse_trigger(&def.sql, &def.name)?;

        // A view is updatable ONLY through INSTEAD OF triggers; skip any other timing.
        if !matches!(ct.timing, Some(TriggerTiming::InsteadOf)) {
            continue;
        }
        if !event_matches(&ct.event, event, view) {
            continue;
        }

        let when = match &ct.when {
            Some(w) => Some(bind_expr(&trigger_scope, ctx, w)?),
            None => None,
        };

        let mut actions = Vec::with_capacity(ct.body.len());
        for stmt in &ct.body {
            actions.push(compile_action(registry, catalog, stmt, &trigger_scope)?);
        }

        programs.push(TriggerProgram {
            name: def.name.clone(),
            timing: TriggerTiming::InsteadOf,
            when,
            actions,
        });
    }
    Ok(programs)
}

/// Re-parse a stored trigger's verbatim `sql` back into its [`CreateTrigger`] AST. The
/// stored text is exactly one `CREATE TRIGGER` statement; anything else is a corrupt
/// schema row and fails closed (never a silently-dropped trigger).
fn reparse_trigger(sql: &str, name: &str) -> Result<CreateTrigger> {
    let mut ast = parse(sql)?;
    if ast.statements.len() != 1 {
        return Err(Error::sql(format!(
            "stored trigger {name} did not re-parse to a single statement"
        )));
    }
    match ast.statements.pop() {
        Some(Statement::CreateTrigger(ct)) => Ok(*ct),
        _ => Err(Error::sql(format!(
            "stored trigger {name} did not re-parse to a CREATE TRIGGER"
        ))),
    }
}

/// Whether a trigger's `event` matches the DML `event` (and, for `UPDATE OF`, whether it
/// names an assigned column). `INSERT`/`DELETE` match exactly. `UPDATE` matches an
/// `UPDATE` DML; with an `OF <cols>` list it fires only if one of those columns is among
/// the statement's assigned columns (`changed_cols`, carried on the event) —
/// `lang_createtrigger.html`. An `OF` name not found on the target simply never matches (a
/// validly-created trigger names only real columns; SQLite validates that at CREATE time).
fn event_matches(trigger_event: &TriggerEvent, dml: TriggerDmlEvent, target: &TableDef) -> bool {
    match (trigger_event, dml) {
        (TriggerEvent::Insert, TriggerDmlEvent::Insert) => true,
        (TriggerEvent::Delete, TriggerDmlEvent::Delete) => true,
        (TriggerEvent::Update { columns }, TriggerDmlEvent::Update { changed_cols }) => {
            if columns.is_empty() {
                return true;
            }
            columns.iter().any(|name| {
                target
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(name))
                    .is_some_and(|idx| changed_cols.contains(&idx))
            })
        }
        _ => false,
    }
}

/// Compile one trigger body statement into a self-contained [`Plan`] under a fresh
/// [`PlanCtx`], with `parent` (the OLD/NEW scope) as the enclosing scope so the action's
/// own sources sit after the OLD++NEW row and its `NEW`/`OLD` references resolve through
/// the parent. Trigger bodies allow only `INSERT`/`UPDATE`/`DELETE`/`SELECT`; anything
/// else is a corrupt body and fails closed.
fn compile_action(
    registry: &FunctionRegistry,
    catalog: &dyn Catalog,
    stmt: &Statement,
    parent: &Scope,
) -> Result<Plan> {
    let mut actx = PlanCtx::new(registry, catalog);
    let (root, result_columns, mutates) = match stmt {
        Statement::Insert(ins) => {
            let (root, cols) = compile_insert_with_parent(&mut actx, ins, Some(parent))?;
            (root, cols, true)
        }
        Statement::Update(upd) => {
            let (root, cols) = compile_update_with_parent(&mut actx, upd, Some(parent))?;
            (root, cols, true)
        }
        Statement::Delete(del) => {
            let (root, cols) = compile_delete_with_parent(&mut actx, del, Some(parent))?;
            (root, cols, true)
        }
        Statement::Select(sel) => {
            let (root, cols) = compile_select_with_parent(&mut actx, sel, Some(parent))?;
            (root, cols, false)
        }
        _ => {
            return Err(Error::sql(
                "trigger body statements must be INSERT, UPDATE, DELETE or SELECT",
            ))
        }
    };
    // `generated` starts empty; the top-level `populate_generated` pass recurses into this
    // action plan (via the DML node that carries it) and fills its map so the action's own
    // scans/writes see their target's generated columns.
    Ok(Plan {
        root,
        result_columns,
        ctes: actx.ctes,
        subqueries: actx.subqueries,
        mutates,
        generated: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use minisqlite_catalog::{ColumnDef, IndexDef, TriggerDef};
    use minisqlite_expr::{CmpOp, EvalExpr};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{BinaryOp, CreateIndex, CreateTable, Drop, Expr, Literal};
    use minisqlite_types::Value;

    use crate::plan::{Delete, Insert, PlanNode, Update};
    use crate::{Plan, Planner, QueryPlanner};

    /// A static catalog exposing tables AND triggers (only reads are exercised).
    struct TestCatalog {
        tables: Vec<TableDef>,
        triggers: Vec<TriggerDef>,
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
        fn triggers_on(&self, table: &str) -> Result<Vec<&TriggerDef>> {
            // Match on the trigger's TARGET table, ordered by folded name for a
            // deterministic firing order (mirrors the real store).
            let mut out: Vec<&TriggerDef> =
                self.triggers.iter().filter(|t| t.table.eq_ignore_ascii_case(table)).collect();
            out.sort_by(|a, b| a.name.to_ascii_lowercase().cmp(&b.name.to_ascii_lowercase()));
            Ok(out)
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

    fn tdef(name: &str, cols: Vec<ColumnDef>) -> TableDef {
        TableDef {
            name: name.to_string(),
            columns: cols,
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

    fn trig(name: &str, table: &str, sql: &str) -> TriggerDef {
        TriggerDef::same_store(name.to_string(), table.to_string(), sql.to_string())
    }

    /// `t(a INTEGER, b INTEGER)` (trigger target; so `C=2`, `W=3`, `2W=6`),
    /// `log(x INTEGER, y INTEGER)` (the usual action target), and `audit(z INTEGER)` — a
    /// deliberately TRIGGER-FREE table. The recursion-guard tests put a trigger ON `log`
    /// whose own action writes `audit`: that makes the one-level bound OBSERVABLE (a
    /// guard removal compiles log's trigger into the action's `.triggers`) while keeping
    /// the mutated failure a clean assertion — writing `audit` cannot re-trigger, so it
    /// terminates instead of infinitely expanding at compile time.
    fn cat(triggers: Vec<TriggerDef>) -> TestCatalog {
        TestCatalog {
            tables: vec![
                tdef("t", vec![col("a", "INTEGER"), col("b", "INTEGER")]),
                tdef("log", vec![col("x", "INTEGER"), col("y", "INTEGER")]),
                tdef("audit", vec![col("z", "INTEGER")]),
            ],
            triggers,
        }
    }

    fn plan_sql(sql: &str, c: &TestCatalog) -> Result<Plan> {
        let ast = parse(sql).expect("parse dml");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, c)
    }

    fn as_insert(node: &PlanNode) -> &Insert {
        match node {
            PlanNode::Insert(i) => i,
            other => panic!("expected Insert, got {other:?}"),
        }
    }
    fn as_update(node: &PlanNode) -> &Update {
        match node {
            PlanNode::Update(u) => u,
            other => panic!("expected Update, got {other:?}"),
        }
    }
    fn as_delete(node: &PlanNode) -> &Delete {
        match node {
            PlanNode::Delete(d) => d,
            other => panic!("expected Delete, got {other:?}"),
        }
    }
    fn values_rows(node: &PlanNode) -> &[Vec<EvalExpr>] {
        match node {
            PlanNode::Values { rows } => rows,
            other => panic!("expected Values, got {other:?}"),
        }
    }
    fn col_reg(e: &EvalExpr) -> usize {
        match e {
            EvalExpr::Column(r) => *r,
            other => panic!("expected Column, got {other:?}"),
        }
    }

    // ---- attach + NEW register (the primary shape) ------------------

    #[test]
    fn after_insert_trigger_action_reads_new_columns() {
        // NEW.a -> Column(W+0)=Column(3), NEW.b -> Column(W+1)=Column(4) (W = 2+1).
        // `log` ALSO carries an INSERT trigger, so the one-level bound is OBSERVABLE: the
        // action `INSERT INTO log` must NOT pull in log's own triggers. Without the
        // `parent.is_none()` guard in insert.rs, `compile_triggers(log, …)` would find
        // `log_ins` and `action.triggers` would be non-empty — so this assertion now
        // catches a guard removal that an empty-on-log fixture would silently pass.
        let c = cat(vec![
            trig(
                "trg",
                "t",
                "CREATE TRIGGER trg AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
            ),
            trig(
                "log_ins",
                "log",
                "CREATE TRIGGER log_ins AFTER INSERT ON log BEGIN INSERT INTO audit VALUES (NEW.x); END",
            ),
        ]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let ins = as_insert(&plan.root);
        assert_eq!(ins.triggers.len(), 1, "only t's trigger attaches (log's is not t's)");
        let tp = &ins.triggers[0];
        assert_eq!(tp.name, "trg");
        assert!(matches!(tp.timing, TriggerTiming::After), "AFTER preserved");
        assert!(tp.when.is_none(), "no WHEN");
        assert_eq!(tp.actions.len(), 1);

        let action = as_insert(&tp.actions[0].root);
        assert_eq!(action.table, "log");
        assert!(
            action.triggers.is_empty(),
            "one-level bound: the INSERT-INTO-log action does NOT compile log's own triggers"
        );
        let rows = values_rows(&action.source);
        assert_eq!(rows.len(), 1);
        assert_eq!(col_reg(&rows[0][0]), 3, "NEW.a -> Column(W+0)=3");
        assert_eq!(col_reg(&rows[0][1]), 4, "NEW.b -> Column(W+1)=4");
    }

    #[test]
    fn trigger_action_dml_on_a_checked_table_still_carries_checks() {
        // A CHECK attaches REGARDLESS of `parent` — UNLIKE triggers, which attach only to a
        // TOP-LEVEL DML (`parent.is_none()`). So a trigger action's nested INSERT into a
        // checked table must STILL carry that table's CHECK. A regression that gated
        // `compile_checks` on `parent.is_none()` (mirroring the trigger guard right above)
        // would silently drop CHECK enforcement for every trigger-driven write and — because
        // every other CHECK test uses a top-level DML — stay green. This is the test that
        // pins the deliberate difference.
        //
        // `ck(v INTEGER CHECK(v > 0))`, hand-built because the static test catalog does not
        // run the catalog builder that would populate `checks` in production.
        let mut ck = tdef("ck", vec![col("v", "INTEGER")]);
        ck.checks = vec![Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column { schema: None, table: None, name: "v".to_string(), from_dqs: false }),
            right: Box::new(Expr::Literal(Literal::Integer(0))),
        }];
        let c = TestCatalog {
            tables: vec![tdef("t", vec![col("a", "INTEGER")]), ck],
            triggers: vec![trig(
                "trg",
                "t",
                "CREATE TRIGGER trg AFTER INSERT ON t BEGIN INSERT INTO ck VALUES (NEW.a); END",
            )],
        };

        let plan = plan_sql("INSERT INTO t VALUES (1)", &c).unwrap();
        let tp = &as_insert(&plan.root).triggers[0];
        assert_eq!(tp.actions.len(), 1);
        let action = as_insert(&tp.actions[0].root);
        assert_eq!(action.table, "ck");
        assert_eq!(
            action.checks.len(),
            1,
            "trigger-action INSERT into `ck` still carries ck's CHECK (attach is not parent-gated)"
        );
    }

    #[test]
    fn omitted_timing_defaults_to_before() {
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let tp = &as_insert(&plan.root).triggers[0];
        assert!(matches!(tp.timing, TriggerTiming::Before), "omitted timing => BEFORE");
    }

    // ---- WHEN binding ------------------------------------------------------------

    #[test]
    fn when_binds_against_the_old_new_scope() {
        // WHEN NEW.a > 0 -> Compare(Gt, Column(3), Literal(0)).
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER INSERT ON t WHEN NEW.a > 0 \
             BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let tp = &as_insert(&plan.root).triggers[0];
        match tp.when.as_ref().expect("WHEN bound") {
            EvalExpr::Compare { op: CmpOp::Gt, left, right, .. } => {
                assert_eq!(col_reg(left), 3, "NEW.a is Column(W+0)=3");
                assert!(matches!(right.as_ref(), EvalExpr::Literal(Value::Integer(0))));
            }
            other => panic!("expected Compare(Gt, ..), got {other:?}"),
        }
    }

    // ---- event filtering ---------------------------------------------------------

    #[test]
    fn insert_trigger_does_not_fire_on_delete_or_update() {
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        assert!(as_delete(&plan_sql("DELETE FROM t", &c).unwrap().root).triggers.is_empty());
        assert!(as_update(&plan_sql("UPDATE t SET a = 1", &c).unwrap().root).triggers.is_empty());
        // But it DOES fire on an INSERT.
        assert_eq!(as_insert(&plan_sql("INSERT INTO t VALUES (1,2)", &c).unwrap().root).triggers.len(), 1);
    }

    #[test]
    fn delete_trigger_reads_old_columns() {
        // OLD.a -> Column(0), OLD.b -> Column(1).
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER DELETE ON t BEGIN INSERT INTO log VALUES (OLD.a, OLD.b); END",
        )]);
        let plan = plan_sql("DELETE FROM t WHERE a = 1", &c).unwrap();
        let tp = &as_delete(&plan.root).triggers[0];
        let action = as_insert(&tp.actions[0].root);
        let rows = values_rows(&action.source);
        assert_eq!(col_reg(&rows[0][0]), 0, "OLD.a -> Column(0)");
        assert_eq!(col_reg(&rows[0][1]), 1, "OLD.b -> Column(1)");
    }

    // ---- UPDATE OF filtering -----------------------------------------------------

    #[test]
    fn update_of_fires_only_when_a_named_column_is_assigned() {
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg BEFORE UPDATE OF a ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        // Assigning `a` fires it; assigning only `b` does not.
        assert_eq!(as_update(&plan_sql("UPDATE t SET a = 5", &c).unwrap().root).triggers.len(), 1);
        assert!(as_update(&plan_sql("UPDATE t SET b = 5", &c).unwrap().root).triggers.is_empty());
        // Assigning both (a among them) fires it.
        assert_eq!(as_update(&plan_sql("UPDATE t SET a = 1, b = 2", &c).unwrap().root).triggers.len(), 1);
    }

    #[test]
    fn bare_update_trigger_fires_on_any_assigned_column() {
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER UPDATE ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        assert_eq!(as_update(&plan_sql("UPDATE t SET a = 1", &c).unwrap().root).triggers.len(), 1);
        assert_eq!(as_update(&plan_sql("UPDATE t SET b = 1", &c).unwrap().root).triggers.len(), 1);
    }

    // ---- action base_offset = 2W (UPDATE / DELETE actions) -----------------------

    #[test]
    fn update_action_places_own_columns_after_the_old_new_row() {
        // Action `UPDATE log SET y = NEW.a WHERE x = OLD.a`: log's own columns sit at
        // base 2W = 6, so x -> Column(6); NEW.a -> Column(3), OLD.a -> Column(0). Because
        // the WHERE references OLD/NEW in the low registers, the scan is a plain SeqScan +
        // residual Filter (not a mis-folded seek). `log` carries its OWN update trigger so
        // the `upd.triggers.is_empty()` assertion is non-vacuous — it fails if the
        // `parent.is_none()` guard in update.rs is removed.
        let c = cat(vec![
            trig(
                "trg",
                "t",
                "CREATE TRIGGER trg AFTER UPDATE ON t BEGIN UPDATE log SET y = NEW.a WHERE x = OLD.a; END",
            ),
            trig(
                "log_upd",
                "log",
                "CREATE TRIGGER log_upd AFTER UPDATE ON log BEGIN INSERT INTO audit VALUES (NEW.x); END",
            ),
        ]);
        let plan = plan_sql("UPDATE t SET a = 1", &c).unwrap();
        let action = as_update(&plan.root).triggers[0].actions[0].root.clone();
        let upd = as_update(&action);
        assert_eq!(upd.table, "log");
        assert!(
            upd.triggers.is_empty(),
            "one-level bound: the UPDATE-log action does NOT compile log's own triggers"
        );
        assert_eq!(upd.assignments.len(), 1);
        assert_eq!(upd.assignments[0].0, 1, "SET y -> log column index 1");
        assert_eq!(col_reg(&upd.assignments[0].1), 3, "NEW.a -> Column(3)");
        match upd.scan.as_ref() {
            PlanNode::Filter { input, predicate } => {
                match predicate {
                    EvalExpr::Compare { op: CmpOp::Eq, left, right, .. } => {
                        assert_eq!(col_reg(left), 6, "log.x sits at base 2W = 6");
                        assert_eq!(col_reg(right), 0, "OLD.a -> Column(0)");
                    }
                    other => panic!("expected Eq compare, got {other:?}"),
                }
                assert!(matches!(input.as_ref(), PlanNode::SeqScan(_)), "trigger action full-scans");
            }
            other => panic!("expected Filter over SeqScan, got {other:?}"),
        }
    }

    #[test]
    fn delete_action_places_own_columns_after_the_old_new_row() {
        // Action `DELETE FROM log WHERE x = NEW.a`: log.x -> Column(6), NEW.a -> Column(3).
        // `log` carries its OWN delete trigger so the `del.triggers.is_empty()` assertion
        // below is non-vacuous — it fails if the `parent.is_none()` guard in delete.rs is
        // removed.
        let c = cat(vec![
            trig(
                "trg",
                "t",
                "CREATE TRIGGER trg AFTER INSERT ON t BEGIN DELETE FROM log WHERE x = NEW.a; END",
            ),
            trig(
                "log_del",
                "log",
                "CREATE TRIGGER log_del AFTER DELETE ON log BEGIN INSERT INTO audit VALUES (OLD.x); END",
            ),
        ]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let action = as_insert(&plan.root).triggers[0].actions[0].root.clone();
        let del = as_delete(&action);
        assert_eq!(del.table, "log");
        assert!(
            del.triggers.is_empty(),
            "one-level bound: the DELETE-log action does NOT compile log's own triggers"
        );
        match del.scan.as_ref() {
            PlanNode::Filter { input, predicate } => {
                match predicate {
                    EvalExpr::Compare { op: CmpOp::Eq, left, right, .. } => {
                        assert_eq!(col_reg(left), 6, "log.x sits at base 2W = 6");
                        assert_eq!(col_reg(right), 3, "NEW.a -> Column(3)");
                    }
                    other => panic!("expected Eq compare, got {other:?}"),
                }
                assert!(matches!(input.as_ref(), PlanNode::SeqScan(_)));
            }
            other => panic!("expected Filter over SeqScan, got {other:?}"),
        }
    }

    // ---- multiple triggers, firing order ----------------------------------------

    #[test]
    fn multiple_matching_triggers_attach_in_deterministic_order() {
        let c = cat(vec![
            trig("b_trg", "t", "CREATE TRIGGER b_trg AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END"),
            trig("a_trg", "t", "CREATE TRIGGER a_trg AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.b, NEW.a); END"),
        ]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let names: Vec<&str> = as_insert(&plan.root).triggers.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a_trg", "b_trg"], "attached in the catalog's (folded-name) order");
    }

    // ---- INSTEAD OF on a table is an error ---------------------------------------

    #[test]
    fn instead_of_on_a_table_is_a_loud_error() {
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg INSTEAD OF INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        let err = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("INSTEAD OF"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- no triggers -> empty vec ------------------------------------------------

    #[test]
    fn no_triggers_leaves_the_vec_empty() {
        let c = cat(Vec::new());
        assert!(as_insert(&plan_sql("INSERT INTO t VALUES (1,2)", &c).unwrap().root).triggers.is_empty());
    }

    // ---- rowid half of the NEW/OLD register convention ---------------------------

    #[test]
    fn new_and_old_rowid_map_to_the_trailing_registers() {
        // The column offsets are pinned elsewhere; this pins the ROWID half:
        // NEW.rowid -> Column(W+C) = Column(5), OLD.rowid -> Column(C) = Column(2)
        // (C=2, W=3). An UPDATE trigger, so both OLD and NEW rows are live.
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER UPDATE ON t BEGIN INSERT INTO log VALUES (NEW.rowid, OLD.rowid); END",
        )]);
        let plan = plan_sql("UPDATE t SET a = 1", &c).unwrap();
        let action = as_insert(&as_update(&plan.root).triggers[0].actions[0].root);
        let rows = values_rows(&action.source);
        assert_eq!(col_reg(&rows[0][0]), 5, "NEW.rowid -> Column(W+C)=5");
        assert_eq!(col_reg(&rows[0][1]), 2, "OLD.rowid -> Column(C)=2");
    }

    // ---- UPDATE OF a name not on the target --------------------------------------

    #[test]
    fn update_of_a_column_not_on_the_target_never_fires() {
        // `UPDATE OF <name>` where <name> is not a column of the target can never match:
        // event_matches maps each OF name to a target column index and requires it among
        // the assigned columns; an unknown name maps to nothing. (A validly-created
        // trigger names only real columns — SQLite validates that at CREATE time.)
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg BEFORE UPDATE OF zzz ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b); END",
        )]);
        assert!(
            as_update(&plan_sql("UPDATE t SET a = 1", &c).unwrap().root).triggers.is_empty(),
            "UPDATE OF a non-existent column never fires"
        );
    }

    // ---- fail-closed paths -------------------------------------------------------

    #[test]
    fn a_stored_trigger_that_does_not_reparse_to_create_trigger_is_a_loud_error() {
        // The catalog stores verbatim trigger SQL. If a row's SQL is not exactly one
        // CREATE TRIGGER (corruption / a bad writer), compilation fails CLOSED rather
        // than silently dropping the trigger.
        let c = cat(vec![trig("bad", "t", "SELECT 1")]);
        let err = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("did not re-parse to a CREATE TRIGGER"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn a_non_dml_trigger_body_statement_is_a_loud_error() {
        // Trigger bodies allow only INSERT/UPDATE/DELETE/SELECT; `compile_action` fails
        // CLOSED on anything else. The grammar keeps a non-DML statement out of a parsed
        // body, so this drives the defensive guard directly.
        let c = cat(Vec::new());
        let target = c.table("t").unwrap().expect("t exists");
        let sources = new_old_sources(target);
        let scope = Scope::new(&sources);
        let registry = FunctionRegistry::builtins();
        let ast = parse("CREATE TABLE z (a)").expect("parse");
        let stmt = ast.statements.first().expect("one statement");
        let err = compile_action(&registry, &c, stmt, &scope).unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("must be INSERT, UPDATE, DELETE or SELECT"), "got {m:?}")
            }
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- compile-time recursion bound (one level) --------------------------------

    #[test]
    fn self_referential_trigger_compiles_without_unbounded_expansion() {
        // A trigger on `t` whose ACTION writes `t` again. The one-level compile bound
        // means the action's own `INSERT INTO t` is compiled WITHOUT re-entering
        // compile_triggers for `t` (its `.triggers` stays empty), so there is no
        // unbounded compile-time expansion. Removing the `parent.is_none()` guard would
        // make this recurse forever at COMPILE time — the exact case the bound protects.
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN INSERT INTO t VALUES (NEW.a, NEW.b); END",
        )]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).unwrap();
        let ins = as_insert(&plan.root);
        assert_eq!(ins.triggers.len(), 1, "the trigger fires on the INSERT");
        let action = as_insert(&ins.triggers[0].actions[0].root);
        assert_eq!(action.table, "t", "the action re-inserts into t");
        assert!(
            action.triggers.is_empty(),
            "one-level bound: the action does NOT re-compile t's own triggers"
        );
    }

    // ---- SELECT-action access path at a nonzero base (mis-fold regression) --------

    #[test]
    fn select_action_over_a_base_table_at_nonzero_base_does_not_misfold_a_seek() {
        // A trigger SELECT action's FROM leaf sits at base 2W, so a low-register OLD/NEW
        // reference must NOT be mis-read by the 0-based access path as the action table's
        // rowid and folded into a bogus seek. `SELECT y FROM log WHERE OLD.rowid = 9`:
        // OLD.rowid is Column(C)=Column(2), which equals log's OWN 0-based rowid register
        // (log has 2 columns) — the exact collision. It must be a residual Filter over a
        // full SeqScan of log, never a RowidScan seek that would silently ignore OLD.rowid.
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER DELETE ON t BEGIN SELECT y FROM log WHERE OLD.rowid = 9; END",
        )]);
        let plan = plan_sql("DELETE FROM t WHERE a = 1", &c).unwrap();
        let action_root = &as_delete(&plan.root).triggers[0].actions[0].root;
        let under = match action_root {
            PlanNode::Project { input, .. } => input.as_ref(),
            other => panic!("expected Project at the action root, got {other:?}"),
        };
        match under {
            PlanNode::Filter { input, predicate } => {
                match predicate {
                    EvalExpr::Compare { op: CmpOp::Eq, left, right, .. } => {
                        assert_eq!(col_reg(left), 2, "OLD.rowid is Column(C)=Column(2)");
                        assert!(matches!(right.as_ref(), EvalExpr::Literal(Value::Integer(9))));
                    }
                    other => panic!("expected an Eq compare, got {other:?}"),
                }
                assert!(
                    matches!(input.as_ref(), PlanNode::SeqScan(s) if s.table == "log"),
                    "the action must FULL-SCAN log, not seek it: {input:?}"
                );
            }
            PlanNode::RowidScan(s) => {
                panic!("OLD.rowid was mis-folded into a bogus rowid seek on {}", s.table)
            }
            other => panic!("expected a Filter over SeqScan, got {other:?}"),
        }
    }

    // ---- multi-table FROM in a trigger action (a flat join) binds after the frame ---

    #[test]
    fn trigger_action_with_a_flat_join_from_binds_sources_after_the_frame() {
        // A trigger action whose SELECT FROM is a FLAT join now plans (previously a loud
        // gap). The action compiles at base 2W with the OLD ++ NEW frame (width 6 for t,
        // W=3) as its correlated outer; the executor contributes that frame EXACTLY ONCE
        // via the join's left spine, so the two joined sources sit after it — `log` at
        // base 6 (y => schema index 1 => reg 7), `t2` at base 6+3=9. The equijoin ON
        // `log.x = t2.a` rides the Join node (`Column(6) = Column(9)`), no residual WHERE.
        // Real SQLite runs this; it is `[frame ++ log ++ t2]`, not a doubled frame.
        let c = cat(vec![trig(
            "trg",
            "t",
            "CREATE TRIGGER trg AFTER INSERT ON t \
             BEGIN SELECT log.y FROM log JOIN t AS t2 ON log.x = t2.a; END",
        )]);
        let plan = plan_sql("INSERT INTO t VALUES (1, 2)", &c).expect("the joined FROM now plans");
        let action_root = &as_insert(&plan.root).triggers[0].actions[0].root;
        let under = match action_root {
            PlanNode::Project { exprs, input } => {
                assert_eq!(col_reg(&exprs[0]), 7, "log.y at base 6 (schema index 1) => reg 7");
                input.as_ref()
            }
            other => panic!("expected Project at the action root, got {other:?}"),
        };
        match under {
            PlanNode::Join(j) => {
                assert_eq!(j.left_width, 3, "log width 3 (x,y,rowid)");
                assert_eq!(j.right_width, 3, "t2 width 3 (a,b,rowid)");
            }
            other => panic!("expected a Join under the projection, got {other:?}"),
        }
    }
}
