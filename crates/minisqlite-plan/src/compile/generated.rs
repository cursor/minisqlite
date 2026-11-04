//! GENERATED-column compilation: bind a table's `AS (expr)` generation expressions
//! into executable [`GeneratedProgram`]s and hang them on the plan-level
//! [`Plan::generated`] map the read/write paths consume (`gencol.html`).
//!
//! Why a plan-level map (not a field on the scan / DML nodes): the read path computes
//! VIRTUAL columns in the SCAN LEAVES and the write path computes every generated
//! column in the INSERT/UPDATE executor, but neither the access-path leaf builder
//! (`access_path` / `compile::from`) nor the DML compilers carry the generation
//! programs — binding them there would ripple through construction sites an in-flight
//! sibling owns. Instead this pass runs ONCE after a statement is compiled, walks the
//! finished plan for the base tables it touches, binds their generation expressions,
//! and records them by table name. Every operator then reaches a table's programs
//! through the shared [`Plan`] its `Env` already holds — exactly how `subqueries` /
//! `ctes` ride on the plan.
//!
//! Binding model: each generation expression binds against a single base-table
//! [`Source`] over the table's schema at register base 0 — the SAME scope
//! [`crate::compile::check`] uses for CHECK — so a column reference resolves as it does
//! everywhere else: column `i` → `Column(i)`, and an INTEGER PRIMARY KEY alias column →
//! the trailing rowid register `N` (a generated column may reference the INTEGER PRIMARY
//! KEY column but not the bare ROWID — `gencol.html` §2.3).
//!
//! Compute order (why a topological sort, not column order): `gencol.html` §2.2 lets a
//! generated column reference ANY other generated column of the same row — including one
//! declared LATER ("Generated columns can occur anywhere in the table definition …
//! interspersed among ordinary columns") — forbidding only a dependency cycle. The
//! read/write paths compute generated values by iterating the returned program list in
//! order and writing each into its register, so that list MUST be a dependency order: a
//! program appears only after every generated column its expression reads. So this pass
//! binds in column order and then reorders the programs by [`toposort_generated`], and a
//! `d AS (e+1), e AS (5)` (d before e) evaluates `e` first and `d` second — correct
//! regardless of declaration order. A dependency cycle (which includes a direct
//! self-reference `b AS (b+1)`) is a schema real SQLite rejects at CREATE TABLE; the
//! catalog here does not re-validate it, so the sort fails closed with an error rather
//! than looping or computing a NULL-fed wrong value.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_expr::EvalExpr;
use minisqlite_functions::FunctionRegistry;
use minisqlite_types::{DbIndex, Error, Result};

use crate::bind::{bind_expr, Scope, Source};
use crate::plan::{CtePlan, GeneratedProgram, Plan, PlanNode, TableGenerated, TriggerProgram};
use crate::plan_ctx::PlanCtx;

/// Bind every generated column of `def` into a [`GeneratedProgram`], returned in
/// DEPENDENCY order (a column follows every generated column it references — see
/// [`toposort_generated`]). Returns an empty vec (no scope built) when the table has no
/// generated column — the allocation-free fast path callers rely on.
///
/// A generated expression that binds to an unknown column fails loudly here (an
/// `Error::Sql("no such column: …")`), matching real SQLite rejecting such a schema. A
/// subquery in a generated expression (a `gencol.html` §2.3 violation the catalog does
/// not re-validate) is refused rather than silently bound: binding would register a
/// subquery on this throwaway context that is then dropped, leaving the `EvalExpr` with a
/// dangling subquery id — so we surface it as an error instead.
///
/// `pub` (like [`compile_table_index_key_exprs`](crate::compile_table_index_key_exprs)) so
/// the EXECUTOR can bind a table's generation programs for a path that discovers its table
/// at RUNTIME rather than plan time — FOREIGN KEY enforcement, which scans a parent/child
/// table (found dynamically as the FK graph is walked) whose VIRTUAL generated columns must
/// be computed to read/compare the key. That path builds a [`PlanCtx`] over the executor's
/// builtin registry and calls this, exactly as the FK cascade compiles child index keys.
pub fn bind_generated_programs(
    ctx: &mut PlanCtx,
    def: &TableDef,
) -> Result<Vec<GeneratedProgram>> {
    if def.columns.iter().all(|c| c.generated.is_none()) {
        return Ok(Vec::new());
    }
    // A single base-table scope over `def` at base 0: column `i` → `Column(i)`, INTEGER
    // PRIMARY KEY alias → the rowid register `N`. Identical to `compile::check`.
    let sources =
        [Source::BaseTable { exposed_name: def.name.clone(), table: def, db: DbIndex::MAIN, base: 0 }];
    let scope = Scope::new(&sources);

    let before_sub = ctx.subqueries.len();
    let mut out = Vec::new();
    for (i, col) in def.columns.iter().enumerate() {
        if let Some(g) = &col.generated {
            let expr = bind_expr(&scope, ctx, &g.expr)?;
            let affinity = def.column_affinity(i);
            out.push(GeneratedProgram { col_index: i, expr, stored: g.stored, affinity });
        }
    }
    if ctx.subqueries.len() != before_sub {
        return Err(Error::sql(format!(
            "subqueries are not allowed in the generated column expressions of \"{}\"",
            def.name
        )));
    }
    // Reorder column-order → dependency order so a generated column that references another
    // (possibly later-declared) generated column is computed after it (`gencol.html` §2.2).
    toposort_generated(out, def)
}

/// Reorder generated `programs` (given in `CREATE TABLE` column order) into a DEPENDENCY
/// order: every generated column appears AFTER every generated column its expression
/// references. `gencol.html` §2.2 permits a generated column to reference ANY other
/// generated column of the same row — even one declared LATER — forbidding only a
/// dependency cycle. Both compute paths iterate this list in order and fill each register
/// as they go, so this order is what lets a referrer read an already-computed value:
///  * write path — every generated value is computed into the logical row;
///  * read path — only the VIRTUAL subset is recomputed, but a STORED dependency is
///    already materialized in the record, so the relative order of the virtual programs
///    (preserved here) is all that matters.
///
/// Ties — programs that become ready together — break by column order (the smallest source
/// position first), so a table whose generated columns already reference only earlier ones
/// keeps its original order and the common case is unperturbed.
///
/// A dependency CYCLE (including a direct self-reference `b AS (b+1)`, which real SQLite
/// rejects at CREATE TABLE) leaves at least one program permanently unready; this is
/// detected (the emitted order is shorter than the input) and returned as an error rather
/// than looping or silently computing a placeholder-fed wrong value — the catalog here does
/// not re-validate the acyclic constraint, so this is where it is enforced.
fn toposort_generated(
    programs: Vec<GeneratedProgram>,
    def: &TableDef,
) -> Result<Vec<GeneratedProgram>> {
    let n = programs.len();
    if n == 0 {
        return Ok(programs);
    }
    // NB: `n == 1` still runs the full pass — a lone generated column can be a DIRECT
    // self-reference (`b AS (b + 1)`), which is a cycle the check below must catch.
    let ncols = def.columns.len();
    // Table column index → its position in `programs` (only generated columns are present;
    // `usize::MAX` marks an ordinary column, which never contributes a dependency edge).
    let mut pos_of_col = vec![usize::MAX; ncols];
    for (p, prog) in programs.iter().enumerate() {
        pos_of_col[prog.col_index] = p;
    }
    // Build the dependency graph. `indeg[p]` = how many distinct generated columns program
    // `p` reads (must be computed first); `dependents[q]` = the programs that read `q` (so
    // emitting `q` unblocks them). A reference to an ordinary column or the rowid register
    // `N` (an INTEGER PRIMARY KEY alias, `c == ncols`) is already available — not an edge.
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    let mut refs = Vec::new();
    for (p, prog) in programs.iter().enumerate() {
        refs.clear();
        collect_column_refs(&prog.expr, &mut refs);
        let mut seen: Vec<usize> = Vec::new();
        for &c in &refs {
            if c < ncols {
                let q = pos_of_col[c];
                // A self-reference (`q == p`) is kept as a self-edge on purpose: it makes
                // `p` permanently unready, so the cycle check below rejects it.
                if q != usize::MAX && !seen.contains(&q) {
                    seen.push(q);
                    dependents[q].push(p);
                    indeg[p] += 1;
                }
            }
        }
    }
    // Kahn's algorithm, always emitting the smallest ready position first for a stable,
    // column-order-preferring result.
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&p| indeg[p] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(p)) = ready.pop() {
        order.push(p);
        for &d in &dependents[p] {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                ready.push(Reverse(d));
            }
        }
    }
    if order.len() != n {
        return Err(Error::sql(format!(
            "generated column dependency cycle in table \"{}\"",
            def.name
        )));
    }
    // Apply the permutation: `order` lists source positions in their new order. `take`
    // moves each program exactly once (every position appears once in a valid toposort).
    let mut slots: Vec<Option<GeneratedProgram>> = programs.into_iter().map(Some).collect();
    let mut result = Vec::with_capacity(n);
    for p in order {
        result.push(slots[p].take().expect("toposort emits each position exactly once"));
    }
    Ok(result)
}

/// Collect the register indices this expression reads (`EvalExpr::Column(i)`) into `out`
/// (duplicates allowed; the caller de-dups). Drives the dependency edges among a table's
/// generated columns. Exhaustive over [`EvalExpr`] with NO catch-all so a new variant that
/// can hold a column reference forces this to be revisited — a missed edge would silently
/// mis-order a generated compute.
fn collect_column_refs(expr: &EvalExpr, out: &mut Vec<usize>) {
    match expr {
        EvalExpr::Column(i) => out.push(*i),
        // Leaves with no column operand, and the subquery leaves (a subquery cannot carry a
        // same-row dependency, and is rejected in a generated expression at bind time).
        EvalExpr::Literal(_)
        | EvalExpr::Param(_)
        | EvalExpr::Now(_)
        | EvalExpr::Raise { .. }
        | EvalExpr::Exists { .. }
        | EvalExpr::ScalarSubquery(_)
        | EvalExpr::ScalarSubqueryColumn { .. } => {}
        EvalExpr::Unary { operand, .. }
        | EvalExpr::IsNull(operand)
        | EvalExpr::NotNull(operand)
        | EvalExpr::Cast { operand, .. }
        | EvalExpr::Collate { operand, .. } => collect_column_refs(operand, out),
        EvalExpr::Arith { left, right, .. }
        | EvalExpr::Concat { left, right }
        | EvalExpr::Bitwise { left, right, .. }
        | EvalExpr::Compare { left, right, .. }
        | EvalExpr::And(left, right)
        | EvalExpr::Or(left, right)
        | EvalExpr::NullIf { left, right, .. } => {
            collect_column_refs(left, out);
            collect_column_refs(right, out);
        }
        EvalExpr::Between { subject, low, high, .. } => {
            collect_column_refs(subject, out);
            collect_column_refs(low, out);
            collect_column_refs(high, out);
        }
        EvalExpr::InList { subject, items, .. } => {
            collect_column_refs(subject, out);
            for it in items {
                collect_column_refs(it, out);
            }
        }
        EvalExpr::InSubquery { subject, .. } => collect_column_refs(subject, out),
        EvalExpr::InSubqueryRow { subjects, .. } => {
            for s in subjects {
                collect_column_refs(s, out);
            }
        }
        EvalExpr::Coalesce(items) => {
            for it in items {
                collect_column_refs(it, out);
            }
        }
        EvalExpr::Case { operand, whens, else_expr } => {
            if let Some(op) = operand {
                collect_column_refs(op, out);
            }
            for w in whens {
                collect_column_refs(&w.when, out);
                collect_column_refs(&w.then, out);
            }
            if let Some(e) = else_expr {
                collect_column_refs(e, out);
            }
        }
        EvalExpr::Like { subject, pattern, escape, .. } => {
            collect_column_refs(subject, out);
            collect_column_refs(pattern, out);
            if let Some(e) = escape {
                collect_column_refs(e, out);
            }
        }
        EvalExpr::Func { args, .. } => {
            for a in args {
                collect_column_refs(a, out);
            }
        }
    }
}

/// Populate `plan.generated` for every base table this plan touches that has generated
/// columns, then recurse into any nested trigger-action plans. Called once by the
/// [`QueryPlanner`](crate::QueryPlanner) after a statement is compiled.
///
/// A table is "touched" if it appears as a scan leaf (read path — its VIRTUAL columns
/// are computed on scan) OR as a DML target (write path — INSERT/UPDATE compute every
/// generated column and omit VIRTUAL from the stored record). We collect both from the
/// operator tree, its CTE bodies, and its expression subqueries, bind each once, and
/// record it. A NESTED trigger action runs under its OWN `Plan`, so we recurse into
/// every DML node's trigger action plans and populate theirs the same way.
pub(crate) fn populate_generated(
    plan: &mut Plan,
    registry: &FunctionRegistry,
    catalog: &dyn Catalog,
) -> Result<()> {
    let mut tables: Vec<(DbIndex, String)> = Vec::new();
    collect_tables(&plan.root, &mut tables);
    for cte in &plan.ctes {
        match cte {
            CtePlan::Materialized { body, .. } => collect_tables(body, &mut tables),
            CtePlan::Recursive { seed, step, .. } => {
                collect_tables(seed, &mut tables);
                collect_tables(step, &mut tables);
            }
        }
    }
    for sub in &plan.subqueries {
        collect_tables(&sub.plan, &mut tables);
    }

    for (db, name) in &tables {
        // A table already recorded (e.g. named twice in a self-join) binds once; the key is
        // (namespace, name) so a temp/attached shadow of a same-named table is a DISTINCT
        // entry, not a collision that would reuse the wrong namespace's programs.
        if plan.generated.iter().any(|t| t.db == *db && t.table.eq_ignore_ascii_case(name)) {
            continue;
        }
        // Resolve the def in the SAME namespace the node's cursor opens on (`table_in(db)`),
        // never the bare search order — under a shadow those disagree, and the generated
        // programs are pure in-memory computation (no pager to accidentally realign them), so
        // binding the wrong namespace's def stores/computes wrong values. `db == MAIN` with no
        // temp/attach, so this is the bare lookup on the hot path.
        let td = match catalog.table_in(*db, name)? {
            Some(td) => td,
            None => continue, // Not a base table (a CTE / view name), or dropped: skip.
        };
        if td.columns.iter().all(|c| c.generated.is_none()) {
            continue;
        }
        let mut ctx = PlanCtx::new(registry, catalog);
        let programs = bind_generated_programs(&mut ctx, td)?;
        if !programs.is_empty() {
            plan.generated.push(TableGenerated { db: *db, table: name.clone(), programs });
        }
    }

    // Nested trigger / INSTEAD OF action plans run under their own `Plan`; populate each.
    populate_actions(&mut plan.root, registry, catalog)
}

/// Recurse into a node's nested trigger-action plans (each a full [`Plan`]) and populate
/// their generated maps. Trigger action plans live only under DML / INSTEAD OF nodes; the
/// walk also descends the ordinary children so a DML node nested below (e.g. an INSERT
/// under a wrapper, defensively) is still reached.
fn populate_actions(
    node: &mut PlanNode,
    registry: &FunctionRegistry,
    catalog: &dyn Catalog,
) -> Result<()> {
    match node {
        PlanNode::Insert(ins) => {
            populate_actions(&mut ins.source, registry, catalog)?;
            populate_generated_for_triggers(&mut ins.triggers, registry, catalog)?;
        }
        PlanNode::Update(upd) => {
            populate_actions(&mut upd.scan, registry, catalog)?;
            populate_generated_for_triggers(&mut upd.triggers, registry, catalog)?;
        }
        PlanNode::Delete(del) => {
            populate_actions(&mut del.scan, registry, catalog)?;
            populate_generated_for_triggers(&mut del.triggers, registry, catalog)?;
        }
        PlanNode::InsteadOf(io) => {
            populate_actions(&mut io.frame_source, registry, catalog)?;
            populate_generated_for_triggers(&mut io.programs, registry, catalog)?;
        }
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. }
        | PlanNode::Window(crate::plan::Window { input, .. }) => {
            populate_actions(input, registry, catalog)?;
        }
        PlanNode::Aggregate(agg) => populate_actions(&mut agg.input, registry, catalog)?,
        PlanNode::Join(join) => {
            populate_actions(&mut join.left, registry, catalog)?;
            populate_actions(&mut join.right, registry, catalog)?;
        }
        PlanNode::SetOp(setop) => {
            populate_actions(&mut setop.left, registry, catalog)?;
            populate_actions(&mut setop.right, registry, catalog)?;
        }
        // Scan leaves and the other terminal nodes carry no nested trigger-action plan, so
        // there is nothing to populate. Enumerated explicitly (NOT a `_` catch-all, matching
        // the sibling `collect_tables`) so a future `PlanNode` variant that DOES wrap a DML /
        // action plan is a compile error here, forcing a revisit rather than silently leaving
        // that action plan's `generated` map empty (VIRTUAL -> NULL / corrupt stored record).
        PlanNode::SeqScan(_)
        | PlanNode::RowidScan(_)
        | PlanNode::IndexScan(_)
        | PlanNode::MinMaxSeek(_)
        | PlanNode::Values { .. }
        | PlanNode::SingleRow
        | PlanNode::CteScan { .. }
        | PlanNode::RecursiveScan { .. }
        | PlanNode::TableFunctionScan { .. }
        | PlanNode::PragmaFunctionScan { .. }
        | PlanNode::CreateTable(_)
        | PlanNode::CreateIndex(_) => {}
    }
    Ok(())
}

/// Populate the generated map of every action plan carried by these trigger programs.
///
/// `pub` so the EXECUTOR can run this pass on a set of triggers it RECOMPILED at runtime.
/// [`compile_triggers`](crate::compile_triggers) deliberately leaves each action plan's
/// `generated` map empty and relies on the top-level [`populate_generated`] pass to fill it
/// — but the `recursive_triggers` recompile path (`ops::trigger::recompile_target_triggers`)
/// compiles a fresh trigger set at runtime that NEVER goes through that pass. Without this
/// call, a nested trigger action that writes a table with generated columns runs with an
/// empty program map: it takes the "no generated columns" fast path and PHYSICALLY STORES a
/// VIRTUAL column (an on-disk-format corruption) while leaving its computed value NULL. So
/// the executor calls this on the recompiled programs before caching them, exactly as the
/// top-level pass does for the compile-time-expanded first level.
pub fn populate_generated_for_triggers(
    triggers: &mut [TriggerProgram],
    registry: &FunctionRegistry,
    catalog: &dyn Catalog,
) -> Result<()> {
    for trig in triggers {
        for action in &mut trig.actions {
            populate_generated(action, registry, catalog)?;
        }
    }
    Ok(())
}

/// Collect the base tables this operator subtree reads (scan leaves) or writes (DML
/// targets) into `out` as `(namespace, name)` pairs (duplicates allowed — the caller
/// de-dups). The NAMESPACE is carried, not just the name: every base/DML node already
/// stamps the resolved `db` its cursor opens on, so a `main.t` write shadowed by a temp `t`
/// records `(MAIN, "t")` and its generated programs are later bound against `main.t` — not
/// whatever a bare, search-order name lookup would resolve to. Does NOT descend into
/// trigger action plans (handled by [`populate_actions`]) or CTE/subquery side tables
/// (handled by the caller from `plan.ctes` / `plan.subqueries`).
fn collect_tables(node: &PlanNode, out: &mut Vec<(DbIndex, String)>) {
    match node {
        PlanNode::SeqScan(s) => out.push((s.db, s.table.clone())),
        PlanNode::RowidScan(s) => out.push((s.db, s.table.clone())),
        PlanNode::IndexScan(s) => out.push((s.db, s.table.clone())),
        PlanNode::MinMaxSeek(s) => out.push((s.db, s.table.clone())),
        PlanNode::Insert(ins) => {
            out.push((ins.db, ins.table.clone()));
            collect_tables(&ins.source, out);
        }
        PlanNode::Update(upd) => {
            out.push((upd.db, upd.table.clone()));
            collect_tables(&upd.scan, out);
        }
        PlanNode::Delete(del) => {
            out.push((del.db, del.table.clone()));
            collect_tables(&del.scan, out);
        }
        PlanNode::InsteadOf(io) => collect_tables(&io.frame_source, out),
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. }
        | PlanNode::Window(crate::plan::Window { input, .. }) => collect_tables(input, out),
        PlanNode::Aggregate(agg) => collect_tables(&agg.input, out),
        PlanNode::Join(join) => {
            collect_tables(&join.left, out);
            collect_tables(&join.right, out);
        }
        PlanNode::SetOp(setop) => {
            collect_tables(&setop.left, out);
            collect_tables(&setop.right, out);
        }
        // Leaves that read no base table directly, or reference tables via the plan's
        // side tables (CTE bodies / subqueries) the caller walks separately.
        PlanNode::Values { .. }
        | PlanNode::SingleRow
        | PlanNode::CteScan { .. }
        | PlanNode::RecursiveScan { .. }
        | PlanNode::TableFunctionScan { .. }
        | PlanNode::PragmaFunctionScan { .. }
        | PlanNode::CreateTable(_)
        | PlanNode::CreateIndex(_) => {}
    }
}
