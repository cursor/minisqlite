//! The SELECT orchestrator: it wires FROM -> WHERE -> (projection / aggregate) ->
//! DISTINCT -> ORDER BY -> LIMIT into one plan tree for a single query core, and
//! compiles an expression subquery into a [`SubPlan`].
//!
//! A leading `WITH` is delegated to [`cte`](crate::compile::cte) and a compound
//! select (UNION/…) to [`compound`](crate::compile::compound), each a sibling module
//! that owns that feature. The FROM clause — including base-table joins — is compiled
//! by [`from`](crate::compile::from).
//!
//! ## Correlated subqueries (the `base_offset` / `correlated_out` seam)
//!
//! An expression subquery that references a column of an ENCLOSING query is
//! *correlated*: at runtime the executor prepends the outer row to every row the
//! subplan's leaves emit, so the subplan's own sources must be placed AFTER that
//! outer row. `outer_width = parent.outer_row_width_for_child()` is where the own
//! sources go (this is `parent.total_width()` EXCEPT under a grouping parent, where the
//! prepended row is the aggregate's POST-aggregate row — see [`compile_subplan`]), and
//! `Column(i)` for `i < outer_width` reads the outer row, `i >= outer_width` the
//! subquery's own row (see [`crate::plan`]'s ROW/REGISTER convention and
//! `minisqlite-exec`'s `open_subquery`).
//!
//! Correlation is only known AFTER binding (it depends on whether a name resolves
//! locally or via the parent), so [`compile_subquery`] uses a transactional trial:
//! it binds once at `base_offset = 0` with the parent visible and a `saw_correlated`
//! cell watching. If nothing resolved through the parent (the common, non-correlated
//! case) that single bind IS the plan; otherwise it rolls back and re-binds with the
//! own sources at `base = outer_width`. `base_offset` + `correlated_out` are threaded
//! through [`compile_body`]/[`compile_core`] (and the compound arms) so the cell is
//! set no matter where in the subquery the outer reference appears.

use std::cell::{Cell, RefCell};

use minisqlite_expr::{EvalExpr, SortKey};
use minisqlite_sql::{
    Distinct, Expr, Limit, Literal, NullsOrder, OrderingTerm, Select, SelectBody, SelectCore,
    SortOrder,
};
use minisqlite_types::{Collation, Error, Result};

use crate::bind::{
    best_effort_collation, bind_expr, defined_collation, parse_collation, Scope, Source,
    Windowing,
};
use crate::compile::aggregate::{compile_result, match_output_column, Compiled, OrderResolved};
use crate::compile::from::{build_from, reorder_comma_join, resolve_from};
use crate::compile::order_scan::satisfy_order_by;
use crate::compile::values::compile_values;
use crate::compile::window::wrap_window_if_any;
use crate::plan::{PlanNode, SortLimit, SubPlan};
use crate::plan_ctx::PlanCtx;

/// Compile a top-level SELECT into its operator tree and output column names. A
/// top-level query has no enclosing row (`base_offset = 0`) and no correlation cell.
pub fn compile_select(ctx: &mut PlanCtx, sel: &Select) -> Result<(PlanNode, Vec<String>)> {
    compile_select_scoped(ctx, sel, None, 0, None, None, None)
}

/// Compile a SELECT as a trigger ACTION (or any statement correlated to an outer
/// [`Scope`]): its own FROM sources sit AFTER the outer row
/// (`base = parent.outer_row_width_for_child()`) and `NEW`/`OLD` (or any outer column)
/// resolve through `parent`, exactly like a correlated subquery's outer columns. `None` is
/// identical to [`compile_select`] (`base_offset = 0`, no parent).
///
/// Unlike a subquery, an action is NOT trial-bound to discover correlation: it is
/// unconditionally placed at `base_offset = parent.outer_row_width_for_child()` (the executor
/// always prepends the OLD++NEW row), so `correlated_out` is `None` here — there is no
/// enclosing subquery whose correlation flag needs setting. Using the SAME child-placement
/// helper as [`compile_subplan`] (not a raw `total_width()`) keeps the outer-width rule in one
/// place; for the `grouping = None` parents this path is reached with, the two are identical.
pub fn compile_select_with_parent(
    ctx: &mut PlanCtx,
    sel: &Select,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    let base_offset = parent.map_or(0, |p| p.outer_row_width_for_child());
    compile_select_scoped(ctx, sel, parent, base_offset, None, None, None)
}

/// Compile an expression subquery under `parent`, register it, and return its id.
/// Correlation is discovered and the register base chosen by [`compile_subplan`];
/// `EXISTS (SELECT …)` uses this (any result width is fine — the columns are ignored).
pub fn compile_subquery(parent: &Scope, ctx: &mut PlanCtx, sel: &Select) -> Result<usize> {
    let (sub, _names) = compile_subplan(parent, ctx, sel)?;
    Ok(ctx.register_subquery(sub))
}

/// Compile an expression subquery used in a single-value context — a scalar
/// subquery `(SELECT …)` or the right side of a scalar `x IN (SELECT …)` — which
/// must return exactly one column. A wider result is the same error real SQLite
/// raises. (`EXISTS (SELECT …)` uses [`compile_subquery`] instead: it ignores the
/// result columns entirely, so any width — including `SELECT *` — is allowed.)
pub fn compile_value_subquery(parent: &Scope, ctx: &mut PlanCtx, sel: &Select) -> Result<usize> {
    compile_width_checked_subquery(parent, ctx, sel, 1)
}

/// Compile the subquery on the RHS of a row-value `(a, …) IN (SELECT …)` probe
/// (rowvalue.html §2.2), requiring it to return exactly `expected_cols` columns — the
/// width of the LHS tuple. A differing width is the same error real SQLite raises
/// (`sub-select returns N columns - expected M`). This is the tuple generalization of
/// [`compile_value_subquery`] (which is `expected_cols == 1`).
pub fn compile_row_subquery(
    parent: &Scope,
    ctx: &mut PlanCtx,
    sel: &Select,
    expected_cols: usize,
) -> Result<usize> {
    compile_width_checked_subquery(parent, ctx, sel, expected_cols)
}

/// Compile the RHS subquery of a column-list `UPDATE` source
/// (`SET (a, b, …) = (SELECT …)`, rowvalue.html §2.3), register it, and return its id
/// together with its OUTPUT COLUMN COUNT. Unlike [`compile_row_subquery`], the width is
/// NOT checked here: the caller compares it against the name-list length and raises the
/// SET-specific `"N columns assigned M values"` error (the same message SQLite and the
/// parenthesized row-value arm use), so the width contract stays with the `SET` binder
/// that owns it. Correlation and the register base are discovered by [`compile_subplan`]
/// exactly as for a scalar/`IN`/`EXISTS` subquery, so a correlated source (`… WHERE
/// src.k = t.k`) reads the target row through the outer-row plumbing.
pub fn compile_columnlist_subquery(
    parent: &Scope,
    ctx: &mut PlanCtx,
    sel: &Select,
) -> Result<(usize, usize)> {
    let (sub, names) = compile_subplan(parent, ctx, sel)?;
    let width = names.len();
    Ok((ctx.register_subquery(sub), width))
}

/// Shared body for [`compile_value_subquery`] / [`compile_row_subquery`]: compile the
/// subplan, enforce its output column count equals `expected_cols`, and register it.
fn compile_width_checked_subquery(
    parent: &Scope,
    ctx: &mut PlanCtx,
    sel: &Select,
    expected_cols: usize,
) -> Result<usize> {
    let (sub, names) = compile_subplan(parent, ctx, sel)?;
    if names.len() != expected_cols {
        return Err(Error::sql(format!(
            "sub-select returns {} columns - expected {}",
            names.len(),
            expected_cols
        )));
    }
    Ok(ctx.register_subquery(sub))
}

/// Compile a subquery body under `parent` into a [`SubPlan`], discovering whether it
/// is correlated and placing its own sources accordingly — the transactional
/// trial-then-maybe-rebind that is the heart of correlated-subquery support.
///
/// Correlation detection is INDEPENDENT of the register base (the `saw_correlated`
/// cell is set whenever a column resolves through the parent chain, regardless of
/// where own sources sit), so we bind ONCE at `base_offset = 0` with the parent
/// visible:
/// * **not correlated** (the common case, incl. every top-of-tree scalar/`IN`
///   subquery over its own tables): the trial never touched the parent, so its plan —
///   bound at base 0 — is already the final non-correlated plan. Single compile.
/// * **correlated**: the trial's registers are wrong (own columns overlap the outer
///   row at base 0), so roll back everything it registered (subqueries + parameter
///   numbers, via the [`Savepoint`](crate::plan_ctx::Savepoint)) and re-bind with own
///   sources at `base = outer_width`. Rolling back the parameter state first makes the
///   re-bind reproduce the IDENTICAL `?`/named numbers, so statement-wide parameter
///   numbering is unaffected by the retry.
///
/// This "trial at 0, rebind only if correlated" order is the inverse of the naive
/// "trial at outer_width, rebind if NOT correlated": it makes the common
/// non-correlated case a single compile (the naive order re-compiles every
/// non-correlated subquery, which is quadratic-to-exponential over nested
/// non-correlated subqueries). Both are correct; this one is cheaper where it counts.
///
/// Re-binding (rather than shifting the trial plan's registers by `outer_width`) is
/// deliberate: a plan can contain further nested subqueries whose own outer refs
/// point into THIS plan, and shifting would have to rewrite those too. Re-binding
/// reuses the exact binder, so it is correct by construction; the cost is planning
/// time only.
///
/// A subquery correlated to an AGGREGATE enclosing query's post-aggregate row (its
/// `parent` scope carries `grouping`) is planned with the post-aggregate row width as
/// `outer_width` (see the `parent.grouping` branch below), not the pre-aggregate
/// `total_width()`. That width is provisional in pass 1; a §2.5 bare capture, a
/// post-aggregate window call, or an ORDER BY-introduced aggregate widens the real row, and
/// `compile::aggregate` re-binds such a subquery with the corrected width so its own sources
/// sit past the prepended row. On the window path the width differs BY HOST CLAUSE — HAVING
/// reads the pre-window aggregate row, the projection / ORDER BY the wider post-window row — so
/// `compile::aggregate` re-points [`Grouping::subplan_outer_width`] around the HAVING bind and
/// each clause's correlated subquery takes its own width. Two residual shapes are still
/// LOUD-rejected in `Scope::remap_post_aggregate` — rather than mis-bound — because the value
/// they name is not present in the runtime post-aggregate row: a correlated reference to a LOCAL
/// non-GROUP-BY column, and one resolving THROUGH the aggregate to a column of a query ENCLOSING
/// it.
fn compile_subplan(
    parent: &Scope,
    ctx: &mut PlanCtx,
    sel: &Select,
) -> Result<(SubPlan, Vec<String>)> {
    // The outer row the executor prepends to this subplan's rows, and thus where the
    // subquery's own FROM sources are placed (`Column(< outer_width)` reads the outer row).
    // For a subquery whose IMMEDIATE enclosing scope is an aggregate query's post-aggregate
    // context (`parent.grouping` set — a projection or HAVING term), that outer row is the
    // aggregate operator's OUTPUT row `[group_keys.., agg_results..]` (exec `ops/project.rs`
    // / `ops/aggregate.rs` pass it as `outer`), whose width is the grouping's provisional
    // post-aggregate width — NOT `total_width()`, the enclosing query's PRE-aggregate FROM
    // width. Every OTHER site (a plain query, or a WHERE / GROUP-BY-key / aggregate-ARGUMENT
    // subquery of an aggregate query) binds against a `grouping = None` scope whose runtime
    // outer genuinely IS the FROM row, so it uses `total_width()`. The correlated
    // outer-reference REGISTERS are handled in `Scope::remap_post_aggregate` (a GROUP BY
    // column redirects to its post-aggregate key register). `outer_row_width_for_child`
    // makes exactly this choice (post-aggregate width under a grouping parent, else
    // `total_width()`) and is the SAME primitive `Scope::total_width` composes with, so a
    // deeply nested subquery under a grouping parent sizes its outer consistently.
    let outer_width = parent.outer_row_width_for_child();
    let cell = Cell::new(false);
    // Companions to `cell`, populated at the SAME binder sites (`Scope::resolve_via_parent`
    // for `cols`; the `EvalExpr::Now`/non-deterministic-`Func` bind sites for `nondet`), so
    // they are exactly as complete as the correlation flag. See `SubPlan::correlated_cols`
    // and `SubPlan::deterministic`.
    let cols = RefCell::new(Vec::new());
    let nondet = Cell::new(false);
    // The subqueries already registered: after a (re)bind, `ctx.subqueries[sub_start..]` is
    // precisely THIS subplan's directly-nested subqueries, whose determinism folds in for
    // transitivity. Captured before the savepoint so it equals `sp`'s subquery length.
    let sub_start = ctx.subqueries.len();
    let sp = ctx.savepoint();

    // Trial bind: own sources at base 0, parent visible, watching `cell`/`cols`/`nondet`.
    let (node, names) =
        compile_select_scoped(ctx, sel, Some(parent), 0, Some(&cell), Some(&cols), Some(&nondet))?;
    if !cell.get() {
        // Uncorrelated: no outer references (so `correlated_cols` is empty and `outer_width`
        // is 0, the trial's base). Determinism folds this body's own flag with every nested
        // subquery's, plus the conservative CTE/derived-table barrier (an un-analyzed
        // materialized body is treated as possibly non-deterministic).
        let deterministic = !nondet.get()
            && descendants_deterministic(ctx, sub_start)
            && !plan_reaches_materialized_body(&node);
        return Ok((
            SubPlan {
                plan: node,
                correlated: false,
                correlated_cols: Vec::new(),
                deterministic,
                outer_width: 0,
            },
            names,
        ));
    }

    // A correlated subquery whose IMMEDIATE enclosing scope is an aggregate query's
    // post-aggregate context (`parent.grouping` set — a projection / HAVING / ORDER BY term)
    // PLANS: `outer_width` above is the post-aggregate row width, and the outer references
    // resolved to their post-aggregate registers (`Scope::remap_post_aggregate` redirected a
    // GROUP BY column to its key register during the trial bind). Record it on the grouping so
    // `compile::aggregate` re-binds with the corrected `subplan_outer_width` when the real
    // post-aggregate row is wider than the provisional width (a §2.5 captured bare column, a
    // post-aggregate window function, or an ORDER BY-introduced aggregate widens it) — a
    // correlated reference to a NON-GROUP-BY column is already rejected in
    // `remap_post_aggregate`. This flag is the ONLY new effect a post-aggregate correlated
    // subquery has; an aggregate query without one leaves it false and its plan unchanged.
    if let Some(g) = parent.grouping {
        g.saw_correlated_subplan.set(true);
    }

    // Correlated: discard the trial's registrations/param numbers and re-bind with own
    // sources after the outer row so `Column(i < outer_width)` reads the prepended
    // outer row and `Column(i >= outer_width)` reads the subquery's own row. Clear the
    // collectors so they reflect ONLY the final rebind — the outer registers and the
    // nondeterminism are identical between trial and rebind (the base only shifts the
    // subquery's OWN sources, never the outer references), so this is clean-slate hygiene,
    // not a behavior change.
    ctx.restore(sp);
    cols.borrow_mut().clear();
    nondet.set(false);
    let (node, names) = compile_select_scoped(
        ctx,
        sel,
        Some(parent),
        outer_width,
        Some(&cell),
        Some(&cols),
        Some(&nondet),
    )?;
    let mut correlated_cols = cols.into_inner();
    correlated_cols.sort_unstable();
    correlated_cols.dedup();
    // Same fold as the uncorrelated path, plus the CTE/derived-table barrier: a `random()`
    // in a FROM `(SELECT …)` or a referenced `WITH` body is invisible to `nondet` /
    // `descendants_deterministic` (that body compiles cell-less into `ctx.ctes`), yet it is
    // re-drawn per outer-row re-open, so reaching one forces `deterministic = false`.
    let deterministic = !nondet.get()
        && descendants_deterministic(ctx, sub_start)
        && !plan_reaches_materialized_body(&node);
    Ok((
        SubPlan { plan: node, correlated: true, correlated_cols, deterministic, outer_width },
        names,
    ))
}

/// Whether every subquery registered at or after `start` in `ctx.subqueries` is
/// deterministic. Those are exactly the directly-nested subqueries of the subplan being
/// compiled (they were registered while binding its body), so a non-deterministic one
/// makes the enclosing subplan non-deterministic too — that is how
/// [`SubPlan::deterministic`](crate::plan::SubPlan) stays transitive through nesting
/// without a second traversal: each level already folded its own descendants in.
fn descendants_deterministic(ctx: &PlanCtx, start: usize) -> bool {
    ctx.subqueries[start..].iter().all(|s| s.deterministic)
}

/// Whether `node`'s operator tree reaches a materialized CTE / derived-table scan
/// ([`PlanNode::CteScan`]) or a recursive-CTE working-table scan
/// ([`PlanNode::RecursiveScan`]).
///
/// Such a body is compiled through the CTE / derived-table path (`compile_derived` ->
/// `compile_select`, `compile_with`), which binds it with a FRESH scope that carries no
/// nondeterminism cell, and it is stored in `ctx.ctes` — NOT `ctx.subqueries`. So neither
/// the `nondet` cell (bind sites) nor `descendants_deterministic` (which scans only
/// `ctx.subqueries[start..]`) can observe a `random()` inside it. And a correlated subplan
/// re-materializes its bodies per outer-row re-open (`ops/cte.rs` re-drains on every
/// `CteScanCursor::build`), so such a `random()` is re-drawn each outer row => the subplan
/// is genuinely non-deterministic. Because that body is not analyzed we cannot PROVE it
/// deterministic, so [`compile_subplan`] treats reaching one as a determinism barrier and
/// reports the enclosing subplan `deterministic = false` (the documented "when unsure,
/// false" rule: it only forgoes memoization, never risks a stale-cache wrong answer). This
/// catches both a FROM derived table and a reference to an enclosing `WITH` CTE (both are a
/// `CteScan` in the subplan's tree), including when the CTE was registered before this
/// subplan (a `ctx.ctes.len()` delta would miss that; walking the tree does not).
///
/// A [`PlanNode::TableFunctionScan`] is deliberately NOT a barrier: its argument
/// expressions ARE bound through the nondeterminism collector (so a `random()` arg is
/// caught the normal way), and the `json_each`/`json_tree` walk itself is a pure function
/// of its document. The match is exhaustive (no catch-all) so a new `PlanNode` variant
/// with a child plan must be classified here rather than silently escaping the barrier.
fn plan_reaches_materialized_body(node: &PlanNode) -> bool {
    match node {
        PlanNode::CteScan { .. } | PlanNode::RecursiveScan { .. } => true,
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => plan_reaches_materialized_body(input),
        PlanNode::Aggregate(a) => plan_reaches_materialized_body(&a.input),
        PlanNode::Window(w) => plan_reaches_materialized_body(&w.input),
        PlanNode::Join(j) => {
            plan_reaches_materialized_body(&j.left) || plan_reaches_materialized_body(&j.right)
        }
        PlanNode::SetOp(s) => {
            plan_reaches_materialized_body(&s.left) || plan_reaches_materialized_body(&s.right)
        }
        // Leaves with no child plan node, plus DML/DDL (a read subplan never contains one).
        PlanNode::SeqScan(_)
        | PlanNode::RowidScan(_)
        | PlanNode::IndexScan(_)
        | PlanNode::MinMaxSeek(_)
        |         PlanNode::Values { .. }
        | PlanNode::SingleRow
        | PlanNode::TableFunctionScan { .. }
        | PlanNode::PragmaFunctionScan { .. }
        | PlanNode::Insert(_)
        | PlanNode::Update(_)
        | PlanNode::Delete(_)
        | PlanNode::InsteadOf(_)
        | PlanNode::CreateTable(_)
        | PlanNode::CreateIndex(_) => false,
    }
}

/// Compile a SELECT with an optional enclosing scope: route a leading `WITH` to the
/// CTE compiler, otherwise compile the body directly. `base_offset` / `correlated_out`
/// carry the correlated-subquery seam (see the module docs), and are forwarded to
/// [`cte::compile_with`](crate::compile::cte::compile_with) verbatim so a correlated
/// WITH subquery binds its own sources after the outer row exactly like a bare one.
fn compile_select_scoped(
    ctx: &mut PlanCtx,
    sel: &Select,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
) -> Result<(PlanNode, Vec<String>)> {
    if sel.with.is_some() {
        // A WITH-bearing subquery routes to the CTE compiler with the SAME seam a bare body
        // gets: `base_offset` places its own sources after the outer row and the collectors
        // report its outer references, so a correlated WITH subquery is handled by the same
        // `compile_subplan` trial/rebind as any other. The CTEs ride in `ctx.ctes` (rolled
        // back and re-registered at identical ids across a rebind).
        crate::compile::cte::compile_with(
            ctx,
            sel,
            parent,
            base_offset,
            correlated_out,
            correlated_cols_out,
            nondet_out,
        )
    } else {
        // A non-compound top-level/subquery caller never needs the per-arm column
        // collations (only the compound compiler does), so pass `None`.
        compile_body(
            ctx,
            sel,
            parent,
            base_offset,
            correlated_out,
            correlated_cols_out,
            nondet_out,
            None,
        )
    }
}

/// Compile a SELECT body — a compound set operation (routed to the `compound`
/// module) or a single query core — ignoring any leading `WITH` (handled by
/// [`compile_select_scoped`]). `base_offset` / `correlated_out` are threaded through
/// so a correlated subquery whose body is a compound (both arms) or a plain core binds
/// its own sources after the outer row and reports correlation.
///
/// `col_collations_out`, when `Some`, receives each output column's DEFINED collation
/// for the compound-select duplicate-comparison rule: `Some(c)` when the output
/// expression defines a collation — an explicit postfix `COLLATE`, else a plain column
/// reference's declared collation `c` — else `None` (see [`defined_collation`]). Only
/// [`crate::compile::compound`] passes `Some`, to combine the two arms' collations
/// left→right; every other caller passes `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_body(
    ctx: &mut PlanCtx,
    sel: &Select,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
    col_collations_out: Option<&RefCell<Vec<Option<Collation>>>>,
) -> Result<(PlanNode, Vec<String>)> {
    // Contract: a leading `WITH` must already have been consumed by the caller (via
    // `cte::compile_with`). `compile_body` ignores `sel.with`, so a WITH-bearing
    // SELECT reaching here would silently drop the CTEs — assert loudly instead. This
    // guards future in-crate callers (e.g. the compound compiler recursing per arm).
    debug_assert!(
        sel.with.is_none(),
        "compile_body received a WITH-bearing SELECT; route it through cte::compile_with first"
    );
    match &sel.body {
        SelectBody::Compound { .. } => crate::compile::compound::compile_compound(
            ctx,
            sel,
            parent,
            base_offset,
            correlated_out,
            correlated_cols_out,
            nondet_out,
            col_collations_out,
        ),
        SelectBody::Select(core) => compile_core(
            ctx,
            core,
            sel,
            parent,
            base_offset,
            correlated_out,
            correlated_cols_out,
            nondet_out,
            col_collations_out,
        ),
    }
}

/// Compile one SELECT core (a query or a VALUES list) plus the enclosing Select's
/// ORDER BY / LIMIT. `base_offset` places this core's FROM sources (0 at top level,
/// `outer_width` for a correlated subquery); `correlated_out`, when `Some`, is the
/// cell the scope sets if any name here resolves through the enclosing `parent`.
///
/// `col_collations_out`: see [`compile_body`]. Filled with each output column's COLUMN
/// collation for the compound-select duplicate rule — `Some(c)` for a plain column
/// reference, `None` otherwise (a VALUES row or a computed expression is never a column
/// reference, so all its entries are `None`).
#[allow(clippy::too_many_arguments)]
fn compile_core(
    ctx: &mut PlanCtx,
    core: &SelectCore,
    sel: &Select,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
    col_collations_out: Option<&RefCell<Vec<Option<Collation>>>>,
) -> Result<(PlanNode, Vec<String>)> {
    match core {
        SelectCore::Values(rows) => {
            // Thread `parent` so a trigger action's `VALUES (NEW.x)` resolves NEW/OLD
            // through the enclosing OLD/NEW scope (a bare VALUES passes `None`).
            let (node, names) = compile_values(ctx, rows, parent)?;
            let node = apply_values_tail(ctx, node, &names, sel)?;
            // A VALUES row's columns are literal expressions, never column references, so
            // the compound-select comparison uses no column collation for any of them.
            if let Some(out) = col_collations_out {
                *out.borrow_mut() = vec![None; names.len()];
            }
            Ok((node, names))
        }
        SelectCore::Query { distinct, columns, from, where_clause, group_by, having, windows } => {
            // Phase 1: resolve the FROM into the visible sources (with their register
            // offsets) and the NATURAL/USING coalesced columns, so the scope — and thus
            // WHERE / ON binding and projection — sees every column and each shared join
            // column resolves once. `base_offset` shifts sources past the outer row for a
            // correlated subquery.
            let catalog = ctx.catalog;
            // Reorder a pure comma join into a connected (equijoin-edge) order before
            // resolving it, so a many-table `FROM a, b, c, ... WHERE a.x=b.y ...` does not
            // build an astronomical cartesian intermediate. A no-op for every other FROM
            // shape, and result-set-preserving (comma joins commute) -- see
            // `reorder_comma_join`. The reordered tree, when present, drives the entire
            // resolve/bind/build below so registers and bindings stay consistent.
            let reordered_from = match (from, where_clause) {
                (Some(tree), Some(w)) => {
                    reorder_comma_join(catalog, tree, w, base_offset).map(Some)
                }
                _ => None,
            };
            let from = reordered_from.as_ref().unwrap_or(from);
            let (sources, coalesced) = resolve_from(catalog, from, base_offset)?;

            // A correlated STATEMENT (or trigger action) whose FROM is a FLAT join is
            // supported: a nonzero `base_offset` places the sources AFTER the outer /
            // OLD-NEW row (`base_offset`, `base_offset + WL`, …), and the executor
            // contributes that prefix EXACTLY ONCE via the join's left spine while its
            // right side is built with an empty outer, so a correlated/trigger join emits
            // `[outer ++ left ++ right]` (see `minisqlite-exec`'s `ops::join` and
            // `from::choose_join_strategy`'s `base_offset`-aware right-key rebasing). The
            // one residual gap — a PARENTHESIZED sub-join operand under a nonzero base,
            // which `build_subjoin` places in a LOCAL 0-based space that does not compose
            // with the outer prefix — is rejected loudly in `build_from`. (A top-level
            // query and the subquery trial bind are at `base_offset == 0`, unaffected.)

            // The FROM/WHERE scope carries no windowing context: a window function in
            // WHERE / ON is a loud error, and WHERE binds before the window operator.
            let scope = Scope {
                sources: sources.as_slice(),
                coalesced: coalesced.as_slice(),
                parent,
                grouping: None,
                saw_correlated: correlated_out,
                correlated_cols: correlated_cols_out,
                nondeterministic: nondet_out,
                windowing: None,
            };

            let bound_where = match where_clause {
                Some(w) => Some(bind_expr(&scope, ctx, w)?),
                None => None,
            };

            // Phase 2: build the access/join subtree; the returned residual is the
            // WHERE predicate the leaf/join could not consume, applied here as a Filter.
            let (mut node, residual) = build_from(ctx, &scope, from, base_offset, bound_where)?;
            if let Some(pred) = residual {
                node = PlanNode::Filter { input: Box::new(node), predicate: pred };
            }

            // Phase 3: the projection and ORDER BY are the only sites a window function
            // may appear, so bind them under a windowing context that collects each
            // window call and resolves it to the column the `Window` operator appends
            // after the input row (width = the row-source width computed here).
            // `assemble_output` inserts the `Window` node once every call is collected;
            // a query with no window calls adds no node.
            //
            // The window runs over THIS core's own produced row `[outer-prefix ++ own FROM]`,
            // whose width is `base_offset + own-source width` — the max source end (sources
            // are placed at `base_offset`), or `base_offset` itself for a no-FROM `SingleRow`
            // that passes its outer through. This is DELIBERATELY not `scope.total_width()`:
            // that maxes in the PARENT width (its `outer_row_width_for_child`) for the co-row
            // / correlated-nesting cases, but a NON-correlated subquery runs with an EMPTY
            // outer at exec (`base_offset == 0`), so inflating the input width by an enclosing
            // FROM / post-aggregate width would bind the window result past the real row and
            // read out of range — and would break the SAFETY PROPERTY that an aggregate query
            // with no correlated post-aggregate subquery keeps a byte-for-byte plan. The outer
            // prefix, when the subquery IS correlated, is already carried by `base_offset` (the
            // sources sit after it), so this width is correct for the correlated case too.
            let window_input_width = scope.max_source_end().unwrap_or(base_offset);
            let windowing = Windowing::new(window_input_width, windows.as_slice());
            let wscope = Scope {
                sources: sources.as_slice(),
                coalesced: coalesced.as_slice(),
                parent,
                grouping: None,
                saw_correlated: correlated_out,
                correlated_cols: correlated_cols_out,
                nondeterministic: nondet_out,
                windowing: Some(&windowing),
            };

            let compiled =
                compile_result(ctx, &wscope, columns, group_by, having, &sel.order_by, node)?;
            // For a compound arm, record each output column's DEFINED collation (the compound
            // duplicate rule: an explicit postfix COLLATE, else the column's declared
            // collation, else `None`; NO affinity — lang_select.html "duplicate rows in a
            // compound"). The postfix COLLATE is honored FOR its own operand but assigned no
            // precedence ACROSS arms — that "greater precedence not assigned to a postfix
            // COLLATE" clause is enforced by `compile_setop_tree`, which folds arms left→right
            // (`left.or(right)`), not by dropping the COLLATE here. Computed from each output's
            // SOURCE ast against `wscope`, exactly the asts `assemble_output` consumes next;
            // `None` where the output defines no collation (a literal / expression).
            if let Some(out) = col_collations_out {
                *out.borrow_mut() = compiled
                    .result_asts
                    .iter()
                    .map(|ast| ast.as_ref().and_then(|e| defined_collation(&wscope, e)))
                    .collect();
            }
            let (mut plan, names) =
                assemble_output(ctx, &wscope, *distinct, sel, compiled, base_offset)?;
            // Every clause is now bound, so `Source::json_referenced` is final: turn the
            // per-row document copy OFF for any JSON TVF leaf whose hidden `json` column was
            // never read (the common case). Must run AFTER Phase 3 — the SELECT list binds
            // there, after the leaf was built in Phase 2 with the safe `emit_json = true`.
            finalize_tvf_emit_json(&mut plan, &sources);
            Ok((plan, names))
        }
    }
}

/// Decide, now that every clause is bound, whether each JSON table-valued-function leaf in
/// `plan` must carry the whole document in its hidden `json` column. That column is copied
/// into EVERY emitted row, so keeping it when unreferenced is `O(rows·|document|)` —
/// quadratic for a document whose text grows with its element count (see
/// [`PlanNode::TableFunctionScan`]'s `emit_json`). SQLite materializes a hidden column only
/// when selected; this reproduces that.
///
/// The signal is [`Source::json_referenced`], set the instant `column_in_source` resolves a
/// hidden `json` — the ONE by-name resolution path, so it misses no reference (a correlated
/// subquery reaching this level's TVF sets it too). It is read HERE, after WHERE/ON (Phase
/// 2) and the SELECT list / GROUP BY / HAVING / ORDER BY (Phase 3) are all bound. The leaf
/// defaults to `emit_json = true`, so any leaf this structural walk does not reach simply
/// keeps the safe materializing behavior.
pub(crate) fn finalize_tvf_emit_json(plan: &mut PlanNode, sources: &[Source]) {
    // Over-approximate across this level's sources: if ANY hidden `json` was referenced,
    // every TVF leaf here keeps materializing. Exact for the common single-TVF query; a
    // second, unreferenced TVF in the same FROM at worst over-materializes (still correct).
    let emit_json = sources.iter().any(|s| s.json_referenced());
    set_tvf_emit_json(plan, emit_json);
}

/// Set `emit_json` on every [`PlanNode::TableFunctionScan`] reachable through `node`'s
/// STRUCTURAL children. It deliberately does not descend into subquery expressions or a
/// `CteScan`/derived-table plan — those are separate plans finalized by their own
/// [`finalize_tvf_emit_json`] with their own sources. Only the read-path operators a TVF
/// leaf can sit under are matched; because the leaf default is the safe `true`, an
/// unhandled node just leaves its subtree at that default rather than being a correctness
/// risk.
fn set_tvf_emit_json(node: &mut PlanNode, emit: bool) {
    match node {
        PlanNode::TableFunctionScan { emit_json, .. } => *emit_json = emit,
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Distinct { input, .. } => set_tvf_emit_json(input, emit),
        PlanNode::Aggregate(a) => set_tvf_emit_json(&mut a.input, emit),
        PlanNode::Window(w) => set_tvf_emit_json(&mut w.input, emit),
        PlanNode::Join(j) => {
            set_tvf_emit_json(&mut j.left, emit);
            set_tvf_emit_json(&mut j.right, emit);
        }
        PlanNode::SetOp(s) => {
            set_tvf_emit_json(&mut s.left, emit);
            set_tvf_emit_json(&mut s.right, emit);
        }
        // Leaves (scans, VALUES, SingleRow, CteScan, RecursiveScan) and non-read-path
        // nodes have no TVF leaf reachable through a structural child.
        _ => {}
    }
}

/// Wrap the projection with DISTINCT, ORDER BY, and LIMIT, in that logical order.
fn assemble_output(
    ctx: &mut PlanCtx,
    scope: &Scope,
    distinct: Distinct,
    sel: &Select,
    compiled: Compiled,
    base_offset: usize,
) -> Result<(PlanNode, Vec<String>)> {
    let Compiled { input, mut projections, names, result_asts, is_agg: _, order_pre } = compiled;
    let num_out = projections.len();
    let is_distinct = matches!(distinct, Distinct::Distinct);

    let mut keys: Vec<SortKey> = Vec::with_capacity(sel.order_by.len());
    let mut has_hidden = false;
    for (i, term) in sel.order_by.iter().enumerate() {
        // An aggregate query pre-resolved each term against the post-aggregate grouping
        // context (`order_pre`); a plain query resolves it here against the FROM scope. Both
        // yield a key expression over the (possibly hidden-extended) projected row plus, for
        // a term that matched an output column, that column's index (to inherit its
        // collation).
        let (expr, out_idx) = match &order_pre {
            Some(resolved) => match &resolved[i] {
                OrderResolved::Output(idx) => (EvalExpr::Column(*idx), Some(*idx)),
                OrderResolved::Hidden(bound) => {
                    // A group key / aggregate the SELECT list omits: append it as a hidden
                    // projection column past the outputs and sort by it.
                    projections.push(bound.clone());
                    has_hidden = true;
                    (EvalExpr::Column(projections.len() - 1), None)
                }
            },
            None => resolve_order_term(
                ctx, scope, term, &result_asts, &names, num_out, &mut projections, &mut has_hidden,
            )?,
        };
        let collation = order_collation(scope, term, out_idx, &result_asts)?;
        keys.push(SortKey {
            expr,
            desc: matches!(term.order, Some(SortOrder::Desc)),
            nulls_first: term.nulls.map(|n| matches!(n, NullsOrder::First)),
            collation,
        });
    }

    // DISTINCT operates on the output columns only; an ORDER BY over a non-output
    // expression (a "hidden" column) cannot coexist with it (SQLite errors too).
    if is_distinct && has_hidden {
        return Err(Error::sql("ORDER BY term does not match any column in the result set"));
    }

    // Insert the `Window` operator (if any window calls were collected while binding the
    // projection and ORDER BY) between the row source and the projection: window
    // functions run after FROM/WHERE/GROUP BY and before DISTINCT/ORDER BY/LIMIT, and
    // the collected calls resolve to the columns this node appends.
    let input = wrap_window_if_any(scope, input);
    let mut node = PlanNode::Project { input: Box::new(input), exprs: projections };
    if is_distinct {
        // DISTINCT compares output rows with `IS DISTINCT FROM` (lang_select.html §2.6),
        // i.e. each output column under its §7.1 collation (an explicit COLLATE, else the
        // source column's declared collation, else BINARY) — exactly `best_effort_collation`
        // of that column's SOURCE expression. DISTINCT rejects hidden ORDER BY columns
        // above, so the Project here is exactly the `num_out` outputs and `result_asts`
        // (length `num_out`) lines up one-to-one with them. A `None` source AST (no
        // producer emits one today) falls back to BINARY.
        let column_collations = result_asts
            .iter()
            .map(|ast| ast.as_ref().map_or(Collation::Binary, |e| best_effort_collation(scope, e)))
            .collect();
        node = PlanNode::Distinct { input: Box::new(node), column_collations };
    }
    // Bind the LIMIT/OFFSET once (if present) so the SAME bound expressions feed both
    // the sort's retention bound and the Limit node below — binding twice would
    // double-count bind parameters. Binding HERE, after the ORDER BY terms above and
    // before the Sort/Limit nodes, preserves parameter numbering (ORDER BY params, then
    // LIMIT), matching the previous `build_limit`-at-the-end ordering.
    //
    // INVARIANT: the `Sort`'s retention bound (attached just below via `sort_limit_from`)
    // and the `Limit` node MUST carry the SAME expressions — both are derived from THIS one
    // `bound_limit` for exactly that reason. Never bind them separately: the sort evaluates
    // its bound independently of the Limit, so two different bindings could disagree on the
    // count and the sort would drop rows the Limit still needs.
    let bound_limit = match &sel.limit {
        Some(limit) => Some(bind_limit_exprs(ctx, limit)?),
        None => None,
    };
    if !keys.is_empty() {
        // Give the sort the retention bound when the LIMIT is deterministic: it then
        // keeps only the first `offset + limit` rows (a bounded top-k) instead of the
        // whole input, byte-identically to the full-sort-then-Limit below. The Limit
        // node still does the real skip/take. Any intervening hidden-column drop
        // Project (below) is row-preserving 1:1, so the sort remains the one this
        // Limit directly bounds.
        let limit = bound_limit.as_ref().and_then(sort_limit_from);
        // Opportunistic ORDER BY -> scan-order rewrite: when the single base table's
        // rowid/index b-tree walk ALREADY yields the ORDER BY order, `satisfy_order_by`
        // rewrites the access leaf and we OMIT this Sort — streaming in O(1) sort-memory
        // (and early-stopping under the LIMIT node) instead of materializing + sorting.
        // It is proven byte-identical to the full stable sort or it declines (see
        // `order_scan`). Gated to the only context where the leaf's columns sit at
        // registers [0, N] and no operator re-orders rows between the scan and this Sort:
        // a NON-DISTINCT, NON-aggregate (`order_pre.is_none()`), non-correlated /
        // top-level (`base_offset == 0`) query. Any other case keeps the Sort as before.
        let order_satisfied = if !is_distinct && order_pre.is_none() && base_offset == 0 {
            satisfy_order_by(ctx.catalog, &node, &keys)
        } else {
            None
        };
        node = match order_satisfied {
            Some(rewritten) => rewritten,
            None => PlanNode::Sort { input: Box::new(node), keys, limit },
        };
    }
    if has_hidden {
        // Drop the hidden ORDER BY columns, leaving only the SELECT outputs.
        let keep = (0..num_out).map(EvalExpr::Column).collect();
        node = PlanNode::Project { input: Box::new(node), exprs: keep };
    }
    if let Some((bound_limit, bound_offset)) = bound_limit {
        node = PlanNode::Limit {
            input: Box::new(node),
            limit: Some(bound_limit),
            offset: bound_offset,
        };
    }
    Ok((node, names))
}

/// Resolve one PLAIN (non-aggregate) query's ORDER BY term to a key expression over the
/// projected row: an output ordinal / output name / structural match to a result
/// expression (via [`match_output_column`]), or else an arbitrary expression over the
/// FROM row appended as a hidden projection column.
///
/// Returns the key expression plus, when the term resolved to an existing output column,
/// that column's output index — so the caller inherits the referenced column's collation
/// rather than reading it off the (collation-less) term text. A hidden expression returns
/// `None`. (An aggregate query never reaches here; its terms are pre-resolved against the
/// grouping context in [`crate::compile::aggregate`] and delivered via `Compiled::order_pre`.)
fn resolve_order_term(
    ctx: &mut PlanCtx,
    scope: &Scope,
    term: &OrderingTerm,
    result_asts: &[Option<Expr>],
    names: &[String],
    num_out: usize,
    projections: &mut Vec<EvalExpr>,
    has_hidden: &mut bool,
) -> Result<(EvalExpr, Option<usize>)> {
    if let Some(idx) = match_output_column(term, result_asts, names, num_out)? {
        return Ok((EvalExpr::Column(idx), Some(idx)));
    }
    // An arbitrary expression over the input row: bind it and append a hidden projection.
    let bound = bind_expr(scope, ctx, &term.expr)?;
    projections.push(bound);
    *has_hidden = true;
    Ok((EvalExpr::Column(projections.len() - 1), None))
}

/// The collation for an ORDER BY term (spec G.5 / datatype3 §7.1):
/// * an explicit postfix `COLLATE` on the term always wins;
/// * a term that resolved to an existing output column (ordinal / output name /
///   structural match) inherits THAT column's collation — computed from the
///   column's *source* expression, not from the term text (which for `ORDER BY 1`
///   is a bare integer with no collation). Star-expanded columns carry a
///   synthesized bare-column source AST, so they inherit their collation too; the
///   `None`-source arm (no producer emits it today) is reserved and falls back to
///   BINARY;
/// * a hidden ORDER BY expression takes its own (best-effort) collation.
fn order_collation(
    scope: &Scope,
    term: &OrderingTerm,
    out_idx: Option<usize>,
    result_asts: &[Option<Expr>],
) -> Result<Collation> {
    if let Some(name) = &term.collation {
        return parse_collation(name);
    }
    if let Some(idx) = out_idx {
        return Ok(match &result_asts[idx] {
            Some(src) => best_effort_collation(scope, src),
            None => Collation::Binary,
        });
    }
    Ok(best_effort_collation(scope, &term.expr))
}

/// Apply a VALUES statement's trailing ORDER BY / LIMIT. ORDER BY on VALUES may only
/// reference an output column by ordinal or by its `columnN` name.
fn apply_values_tail(
    ctx: &mut PlanCtx,
    mut node: PlanNode,
    names: &[String],
    sel: &Select,
) -> Result<PlanNode> {
    if !sel.order_by.is_empty() {
        let mut keys = Vec::with_capacity(sel.order_by.len());
        for term in &sel.order_by {
            let expr = resolve_values_order_term(term, names)?;
            // BINARY is correct here only because VALUES output columns reference no
            // catalog column, so they carry no collation to inherit; a term with an
            // explicit COLLATE still uses it. (A future ORDER-BY path that DOES
            // resolve to a real column must inherit that column's collation instead.)
            let collation = match &term.collation {
                Some(name) => parse_collation(name)?,
                None => Collation::Binary,
            };
            keys.push(SortKey {
                expr,
                desc: matches!(term.order, Some(SortOrder::Desc)),
                nulls_first: term.nulls.map(|n| matches!(n, NullsOrder::First)),
                collation,
            });
        }
        // VALUES rows are literal tuples bounded by the query text (no scale), so the
        // top-k retention bound would gain nothing; leave it a full stable sort.
        node = PlanNode::Sort { input: Box::new(node), keys, limit: None };
    }
    if let Some(limit) = &sel.limit {
        node = build_limit(ctx, node, limit)?;
    }
    Ok(node)
}

/// Resolve an ORDER BY term against a VALUES output (ordinal or `columnN` name only).
fn resolve_values_order_term(term: &OrderingTerm, names: &[String]) -> Result<EvalExpr> {
    if let Expr::Literal(Literal::Integer(k)) = &term.expr {
        if *k >= 1 && (*k as usize) <= names.len() {
            return Ok(EvalExpr::Column((*k - 1) as usize));
        }
        return Err(Error::sql("ORDER BY term out of range"));
    }
    if let Expr::Column { schema: None, table: None, name, .. } = &term.expr {
        if let Some(idx) = names.iter().position(|n| n.eq_ignore_ascii_case(name)) {
            return Ok(EvalExpr::Column(idx));
        }
    }
    Err(Error::sql("ORDER BY on VALUES must reference an output column by position or name"))
}

/// Build a `LIMIT [OFFSET]` node. Both are constant scalar expressions (no columns
/// in scope), evaluated once before iteration.
///
/// `pub(crate)` so the CTE compiler ([`crate::compile::cte`]) can wrap a recursive CTE's
/// materialized fixpoint in the same `LIMIT/OFFSET` node the SELECT tail uses, keeping one
/// limit-lowering path (constant binding, offset shape) rather than a second copy.
pub(crate) fn build_limit(ctx: &mut PlanCtx, node: PlanNode, limit: &Limit) -> Result<PlanNode> {
    let (bound_limit, bound_offset) = bind_limit_exprs(ctx, limit)?;
    Ok(PlanNode::Limit { input: Box::new(node), limit: Some(bound_limit), offset: bound_offset })
}

/// Bind the `LIMIT` and optional `OFFSET` expressions once (constant scalar
/// expressions — no columns in scope). Shared by [`build_limit`] and the
/// ORDER-BY-LIMIT assembly so the bound expressions are produced EXACTLY ONCE: binding
/// twice would double-count bind parameters (each `?` would consume two numbers). The
/// resulting `EvalExpr`s feed both the `Limit` node and — via [`sort_limit_from`] — the
/// retention bound on the `Sort` directly below it.
pub(crate) fn bind_limit_exprs(
    ctx: &mut PlanCtx,
    limit: &Limit,
) -> Result<(EvalExpr, Option<EvalExpr>)> {
    let empty = Scope::empty();
    let bound_limit = bind_expr(&empty, ctx, &limit.limit)?;
    let bound_offset = match &limit.offset {
        Some(off) => Some(bind_expr(&empty, ctx, off)?),
        None => None,
    };
    Ok((bound_limit, bound_offset))
}

/// The retention bound to attach to a `Sort` that sits directly under this `LIMIT`, or
/// `None` to leave the sort a full stable sort.
///
/// Returns `Some` ONLY when both the `LIMIT` and `OFFSET` expressions are
/// DETERMINISTIC. The sort evaluates the bound SEPARATELY from the `Limit` node below
/// which it sits, so a non-deterministic bound (e.g. `LIMIT random()`, `LIMIT (SELECT
/// …)`, `CURRENT_TIME`) could make the sort retain FEWER rows than the limit ultimately
/// takes — dropping rows the answer needs. A deterministic bound evaluates identically
/// in both places, so `retain == offset + limit` exactly and the bounded top-k is
/// byte-identical to the full sort then limit. A non-deterministic bound is simply left
/// off (the full sort is still correct, just not memory-bounded); such a `LIMIT`
/// produces a non-deterministic result anyway.
pub(crate) fn sort_limit_from(bound: &(EvalExpr, Option<EvalExpr>)) -> Option<SortLimit> {
    let (limit, offset) = bound;
    let offset_ok = offset.as_ref().map_or(true, limit_expr_is_deterministic);
    if limit_expr_is_deterministic(limit) && offset_ok {
        Some(SortLimit { limit: limit.clone(), offset: offset.clone() })
    } else {
        None
    }
}

/// Whether `e` evaluates to the same value on every evaluation within one statement
/// (given fixed bind parameters). Used only to decide whether a `LIMIT`/`OFFSET` bound
/// is safe to also hand to the `Sort` below the `Limit` (see [`sort_limit_from`]).
///
/// The only non-deterministic leaves are `CURRENT_*` ([`EvalExpr::Now`]), a
/// `ScalarFunction` reporting `!deterministic()` (`random()`, the date/time functions,
/// the connection counters), and any subquery (conservatively — its body is not
/// analyzed here). Everything else is deterministic when all of its operands are. The
/// match is EXHAUSTIVE (no catch-all) so a new [`EvalExpr`] variant must be classified
/// here rather than silently defaulting to "deterministic".
///
/// This predicate is SAFETY-CRITICAL, not merely an optimization gate: because the sort
/// re-evaluates the bound INDEPENDENTLY of the `Limit` node, a value MISCLASSIFIED as
/// deterministic (in particular an inaccurate `func.deterministic()` below) would let the
/// sort retain fewer rows than the `Limit` takes — silent row loss under `LIMIT`. When in
/// doubt, return `false`: the only cost of a false negative is the lost memory
/// optimization (a still-correct full sort), never a wrong result set.
fn limit_expr_is_deterministic(e: &EvalExpr) -> bool {
    use minisqlite_expr::EvalExpr as E;
    match e {
        E::Literal(_) | E::Column(_) | E::Param(_) => true,
        // Reads the wall clock / a fresh draw per evaluation.
        E::Now(_) => false,
        E::Func { func, args } => {
            func.deterministic() && args.iter().all(limit_expr_is_deterministic)
        }
        E::Unary { operand, .. }
        | E::Cast { operand, .. }
        | E::Collate { operand, .. }
        | E::IsNull(operand)
        | E::NotNull(operand) => limit_expr_is_deterministic(operand),
        E::Arith { left, right, .. }
        | E::Concat { left, right }
        | E::Bitwise { left, right, .. }
        | E::And(left, right)
        | E::Or(left, right)
        | E::Compare { left, right, .. }
        | E::NullIf { left, right, .. } => {
            limit_expr_is_deterministic(left) && limit_expr_is_deterministic(right)
        }
        E::Between { subject, low, high, .. } => {
            limit_expr_is_deterministic(subject)
                && limit_expr_is_deterministic(low)
                && limit_expr_is_deterministic(high)
        }
        E::InList { subject, items, .. } => {
            limit_expr_is_deterministic(subject) && items.iter().all(limit_expr_is_deterministic)
        }
        E::Coalesce(items) => items.iter().all(limit_expr_is_deterministic),
        E::Case { operand, whens, else_expr } => {
            operand.as_deref().map_or(true, limit_expr_is_deterministic)
                && whens.iter().all(|w| {
                    limit_expr_is_deterministic(&w.when) && limit_expr_is_deterministic(&w.then)
                })
                && else_expr.as_deref().map_or(true, limit_expr_is_deterministic)
        }
        E::Like { subject, pattern, escape, .. } => {
            limit_expr_is_deterministic(subject)
                && limit_expr_is_deterministic(pattern)
                && escape.as_deref().map_or(true, limit_expr_is_deterministic)
        }
        // Subqueries are conservatively non-deterministic (their bodies are not analyzed
        // here); `RAISE(...)` never appears in a `LIMIT`/`OFFSET`.
        E::InSubquery { .. }
        | E::InSubqueryRow { .. }
        | E::Exists { .. }
        | E::ScalarSubquery(_)
        | E::ScalarSubqueryColumn { .. }
        | E::Raise { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    //! Correlated-subquery binding: the `SubPlan.correlated` flag and the register
    //! layout of a subplan's bound `EvalExpr`s. These assert the CONTRACT the executor
    //! relies on (see `minisqlite-exec`'s `open_subquery` + the `with_outer` leaf
    //! prepend): a correlated subplan places its own columns at `[outer_width, ..)` and
    //! reads outer columns at `[0, outer_width)`; a non-correlated subplan places its
    //! own columns at base 0. The catalog/`col`/`tdef` fixture copies the local pattern
    //! from `compile/from.rs` so this file owns its fixtures and never edits `tests.rs`.

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_expr::EvalExpr;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::{Error, Result};

    use crate::plan::{Plan, PlanNode, SubPlan};
    use crate::{Planner, QueryPlanner};

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

    fn tdef(name: &str, columns: Vec<ColumnDef>) -> TableDef {
        TableDef {
            name: name.to_string(),
            columns,
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

    /// `t(id, b)` width 3 (`outer_width` for `FROM t` = 3) · `u(k, x, g)` width 4 ·
    /// `w(m, n)` width 3. A correlated subquery over `u` therefore places `u`'s own
    /// columns at base 3: `k`->3, `x`->4, `g`->5; the outer `t`'s columns stay at
    /// `id`->0, `b`->1 (`< outer_width`).
    fn cat() -> TestCatalog {
        TestCatalog {
            tables: vec![
                tdef("t", vec![col("id", "INTEGER"), col("b", "INTEGER")]),
                tdef("u", vec![col("k", "INTEGER"), col("x", "INTEGER"), col("g", "INTEGER")]),
                tdef("w", vec![col("m", "INTEGER"), col("n", "INTEGER")]),
            ],
        }
    }

    fn plan(sql: &str) -> Plan {
        let c = cat();
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c).expect("plan ok")
    }

    fn plan_err(sql: &str) -> Error {
        let c = cat();
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c).expect_err("expected a planning error")
    }

    /// The single registered subquery (every test SQL below has exactly one).
    fn only_sub(p: &Plan) -> &SubPlan {
        assert_eq!(p.subqueries.len(), 1, "expected exactly one registered subquery");
        &p.subqueries[0]
    }

    fn project_exprs(n: &PlanNode) -> &[EvalExpr] {
        match n {
            PlanNode::Project { exprs, .. } => exprs,
            other => panic!("expected Project, got {other:?}"),
        }
    }

    fn project_input(n: &PlanNode) -> &PlanNode {
        match n {
            PlanNode::Project { input, .. } => input,
            other => panic!("expected Project, got {other:?}"),
        }
    }

    /// The `(predicate, input)` of a Filter.
    fn filter(n: &PlanNode) -> (&EvalExpr, &PlanNode) {
        match n {
            PlanNode::Filter { predicate, input } => (predicate, input),
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    /// The `(left, right)` operands of a `Compare`.
    fn compare(e: &EvalExpr) -> (&EvalExpr, &EvalExpr) {
        match e {
            EvalExpr::Compare { left, right, .. } => (left, right),
            other => panic!("expected Compare, got {other:?}"),
        }
    }

    /// The register a `Column` reads.
    fn reg(e: &EvalExpr) -> usize {
        match e {
            EvalExpr::Column(i) => *i,
            other => panic!("expected Column, got {other:?}"),
        }
    }

    // -- Non-correlated ------------------------------------------------------

    #[test]
    fn non_correlated_scalar_binds_own_columns_at_base_zero() {
        // `x` (u's column 1) binds at base 0 => Column(1); no outer reference.
        let p = plan("SELECT (SELECT x FROM u) FROM t");
        let sub = only_sub(&p);
        assert!(!sub.correlated, "no outer reference => not correlated");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 1, "x at base 0 (schema index 1)");
    }

    #[test]
    fn non_correlated_aggregate_scalar_is_not_correlated() {
        // The canonical non-correlated example. `max(x)` (arity 1) is the
        // aggregate; its argument `x` binds at base 0 (Column(1)), proving no shift.
        let p = plan("SELECT (SELECT max(x) FROM u) FROM t");
        let sub = only_sub(&p);
        assert!(!sub.correlated);
        let agg = match project_input(&sub.plan) {
            PlanNode::Aggregate(a) => a,
            other => panic!("expected Aggregate under the projection, got {other:?}"),
        };
        assert_eq!(agg.aggregates.len(), 1, "one aggregate call");
        assert_eq!(reg(&agg.aggregates[0].args[0]), 1, "max(x): x at base 0");
    }

    #[test]
    fn non_correlated_in_select_is_not_correlated() {
        // `t.id IN (SELECT k FROM u)`: the subquery names only its own table.
        let p = plan("SELECT * FROM t WHERE t.id IN (SELECT k FROM u)");
        let sub = only_sub(&p);
        assert!(!sub.correlated);
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 0, "k at base 0 (schema index 0)");
    }

    #[test]
    fn non_correlated_subquery_over_a_join_still_compiles() {
        // A NON-correlated subquery may contain a join (base 0, so the outer row is
        // never prepended); only a CORRELATED join is rejected (see the guard test).
        let p = plan("SELECT (SELECT u.x FROM u, w WHERE u.k = w.m) FROM t");
        let sub = only_sub(&p);
        assert!(!sub.correlated);
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 1, "u.x at base 0 (schema index 1)");
        let under = project_input(&sub.plan);
        assert!(
            matches!(under, PlanNode::Join(_)),
            "the join is preserved (the u.k = w.m equijoin folds into it), got {under:?}"
        );
    }

    // -- Correlated ----------------------------------------------------------

    #[test]
    fn correlated_scalar_places_own_after_outer_and_reads_outer_low() {
        // `(SELECT u.x FROM u WHERE u.k = t.id)`: own `u.*` at base 3 (outer_width),
        // outer `t.id` at 0. Project reads u.x=Column(4); WHERE is u.k=Column(3) vs
        // t.id=Column(0).
        let p = plan("SELECT (SELECT u.x FROM u WHERE u.k = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references t.id => correlated");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 4, "u.x own column at base 3 => reg 4");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 3, "u.k own => reg 3 (>= outer_width)");
        assert_eq!(reg(r), 0, "t.id outer => reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_exists_is_correlated() {
        let p = plan("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id)");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 3, "u.k own => reg 3");
        assert_eq!(reg(r), 0, "t.id outer => reg 0");
    }

    #[test]
    fn correlated_in_select_is_correlated() {
        // `t.id IN (SELECT k FROM u WHERE u.g = t.b)`: own u.g=Column(5), outer t.b=1.
        let p = plan("SELECT * FROM t WHERE t.id IN (SELECT k FROM u WHERE u.g = t.b)");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 3, "u.k own => reg 3");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 5, "u.g own (schema index 2 at base 3) => reg 5");
        assert_eq!(reg(r), 1, "t.b outer => reg 1");
    }

    // -- Parameter numbering -------------------------------------------------

    #[test]
    fn param_numbering_unaffected_by_a_non_correlated_subquery() {
        // The inner `?` is param 1 (bound first, inside the subquery); the outer `?` is
        // param 2. A non-correlated subquery is a single compile, so nothing shifts.
        let p = plan("SELECT (SELECT x FROM u WHERE g = ?), ? FROM t");
        let sub = only_sub(&p);
        assert!(!sub.correlated);
        let (pred, _) = filter(project_input(&sub.plan));
        let (_, rhs) = compare(pred);
        assert!(matches!(rhs, EvalExpr::Param(1)), "inner ? is param 1, got {rhs:?}");
        let outer = project_exprs(&p.root);
        assert!(matches!(outer[0], EvalExpr::ScalarSubquery(0)), "first output is the subquery");
        assert!(matches!(outer[1], EvalExpr::Param(2)), "outer ? is param 2, got {:?}", outer[1]);
    }

    #[test]
    fn param_numbering_preserved_across_a_correlated_rebind() {
        // A correlated subquery is trial-bound then re-bound; the re-bind must reproduce
        // the identical `?` number (inner = 1) after the savepoint reset, and the outer
        // `?` is still 2. Regression guard for the parameter-state rollback in
        // `PlanCtx::restore`.
        let p = plan("SELECT (SELECT x FROM u WHERE u.k = t.id AND g = ?), ? FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references t.id => correlated (rebind path)");
        let (pred, _) = filter(project_input(&sub.plan));
        // WHERE is `(u.k = t.id) AND (g = ?)`: the right conjunct carries the param.
        match pred {
            EvalExpr::And(_, right) => {
                let (_, rhs) = compare(right);
                assert!(matches!(rhs, EvalExpr::Param(1)), "inner ? stays param 1, got {rhs:?}");
            }
            other => panic!("expected an AND predicate, got {other:?}"),
        }
        let outer = project_exprs(&p.root);
        assert!(matches!(outer[1], EvalExpr::Param(2)), "outer ? is param 2, got {:?}", outer[1]);
    }

    // -- Loud errors ---------------------------------------------------------

    #[test]
    fn unknown_column_in_neither_subquery_nor_parent_is_a_loud_error() {
        let err = plan_err("SELECT (SELECT nope FROM u) FROM t");
        assert!(format!("{err:?}").contains("no such column"), "got {err:?}");
    }

    #[test]
    fn correlated_subquery_over_a_flat_join_binds_sources_after_the_outer() {
        // A correlated subquery whose FROM is a FLAT join now plans (previously a loud
        // gap). The outer row `t` (width 3) is the correlated prefix, contributed ONCE,
        // and the two joined sources sit AFTER it: u at base 3 (x => reg 4, k => reg 3),
        // w at base 3+4=7. The comma is a CROSS join, so the WHERE `u.k = t.id` folds into
        // the join's ON residual (a NestedLoop, since the key reads an outer column) —
        // u.k = Column(3), outer t.id = Column(0). The executor prepends the outer prefix
        // once via the join's left spine (see `minisqlite-exec`'s `ops::join`), so the row
        // is `[outer ++ u ++ w]`, not doubled.
        let p = plan("SELECT (SELECT u.x FROM u, w WHERE u.k = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references t.id => correlated");
        assert_eq!(sub.correlated_cols, vec![0], "depends on outer t.id at reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 4, "u.x at base 3 (schema index 1) => reg 4");
        match project_input(&sub.plan) {
            PlanNode::Join(j) => {
                let (l, r) = compare(j.on.as_ref().expect("correlated equijoin is the ON residual"));
                assert_eq!(reg(l), 3, "u.k own => reg 3 (>= outer_width)");
                assert_eq!(reg(r), 0, "t.id outer => reg 0 (< outer_width)");
                assert_eq!(j.left_width, 4, "u width 4 (k,x,g,rowid)");
                assert_eq!(j.right_width, 3, "w width 3 (m,n,rowid)");
            }
            other => panic!("expected a Join, got {other:?}"),
        }
    }

    #[test]
    fn correlated_subquery_over_a_parenthesized_subjoin_is_a_loud_error() {
        // The one residual gap: a PARENTHESIZED sub-join used as an OPERAND of a larger
        // join, under a correlated outer. A sub-join is built in a LOCAL 0-based register
        // space that does not compose with the outer prefix, so `build_subjoin` rejects a
        // nonzero base loudly rather than mis-bind. Here `(w JOIN t2 …)` is the right
        // operand of `u JOIN (…)`, and the subquery is correlated on outer `t.id`. (A FLAT
        // correlated join — no parenthesized operand — is supported; see the test above. A
        // whole-FROM `FROM (a JOIN b)` is also fine, as it is a flat join with redundant
        // parens routed through `build_from`, not `build_subjoin`.)
        let err = plan_err(
            "SELECT (SELECT u.x FROM u JOIN (w JOIN t AS t2 ON w.n = t2.b) \
             ON u.k = w.m WHERE u.g = t.id) FROM t",
        );
        assert!(
            format!("{err:?}").contains("parenthesized join")
                && format!("{err:?}").contains("not yet supported"),
            "got {err:?}"
        );
    }

    // -- Aggregate-enclosing correlated subqueries ---------------------------

    #[test]
    fn correlated_subquery_in_aggregate_projection_binds_post_aggregate_registers() {
        // A subquery in an AGGREGATE query's projection, correlated to the GROUP BY key,
        // now PLANS (previously a loud error). The executor evaluates it over the
        // POST-aggregate row `[t.b_key]` (width 1 = num_keys(1) + num_aggregates(0), no
        // aggregate calls), so `outer_width = 1` — NOT `total_width()` (the pre-aggregate
        // FROM width 3). The outer ref `t.b` remaps to the post-aggregate KEY register
        // `Column(0)` (via `Scope::remap_post_aggregate`), NOT its FROM reg 1; the
        // subquery's own `u` sits at base 1 (u.k = reg 1). WHERE is `u.k = t.b` =>
        // `Column(1)` (own, >= outer_width) vs `Column(0)` (the outer key, < outer_width).
        let p = plan("SELECT t.b, (SELECT k FROM u WHERE u.k = t.b) FROM t GROUP BY t.b");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references outer t.b => correlated");
        assert_eq!(sub.outer_width, 1, "post-agg outer row width = num_keys(1) + num_aggs(0)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 1, "u.k own at base 1 => reg 1");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 1, "u.k own => reg 1 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_in_aggregate_having_binds_post_aggregate_registers() {
        // HAVING binds against the SAME post-aggregate grouping scope, so a subquery there
        // correlated to the group key plans identically: `outer_width = 1`, the outer `t.b`
        // remaps to key `Column(0)`, and `u` sits at base 1 (g = reg 3). This pins that the
        // HAVING path takes the post-aggregate outer_width/registers, not the FROM ones.
        let p =
            plan("SELECT t.b FROM t GROUP BY t.b HAVING (SELECT g FROM u WHERE u.k = t.b) > 0");
        let sub = only_sub(&p);
        assert!(sub.correlated, "HAVING subquery references outer t.b => correlated");
        assert_eq!(sub.outer_width, 1, "post-agg outer row width = num_keys(1) + num_aggs(0)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 3, "u.g own at base 1 (schema idx 2) => reg 3");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 1, "u.k own => reg 1 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_in_aggregate_with_an_aggregate_widens_outer_width() {
        // With an aggregate call in the projection, the post-aggregate outer row is
        // `[t.b_key, count_result]` (width 2 = num_keys(1) + num_aggregates(1)), so
        // `outer_width = 2` and the subquery's own `u` sits at base 2 (u.k => reg 2). The
        // outer `t.b` still remaps to the key register `Column(0)`. Pins that `outer_width`
        // counts the aggregate results, not just the keys.
        let p = plan(
            "SELECT t.b, count(*), (SELECT k FROM u WHERE u.k = t.b) FROM t GROUP BY t.b",
        );
        let sub = only_sub(&p);
        assert!(sub.correlated, "references outer t.b => correlated");
        assert_eq!(sub.outer_width, 2, "post-agg outer row width = num_keys(1) + num_aggs(1)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 2, "u.k own at base 2 => reg 2");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 2, "u.k own => reg 2 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_referencing_a_non_group_key_is_a_loud_error() {
        // RESIDUAL LOUD REJECT. The subquery correlates on `t.id`, which is NOT the GROUP
        // BY key (`t.b` is). A non-group-key outer column has no stable post-aggregate
        // register (it is a §2.5 arbitrary-row bare column in real sqlite); rather than read
        // a wrong slot, `Scope::remap_post_aggregate` rejects it loudly.
        let err = plan_err("SELECT t.b, (SELECT k FROM u WHERE u.k = t.id) FROM t GROUP BY t.b");
        assert!(
            format!("{err:?}").contains("reference only the outer query's GROUP BY columns"),
            "got {err:?}"
        );
    }

    #[test]
    fn correlated_subquery_with_a_bare_column_capture_widens_outer_width() {
        // Row-widening combination, now SUPPORTED. `t.id` is a §2.5 bare column (captured
        // after the aggregate results), widening the post-aggregate row to
        // `[t.b_key, captured_id]` (width 2 = num_keys(1) + num_aggregates(0) +
        // num_captured(1)). `compile::aggregate` re-binds the correlated subquery with
        // `outer_width = 2`, so its own `u` sits at base 2 (u.k => reg 2) — past the capture —
        // and outer `t.b` remaps to post-agg key reg 0. Pins the capture-widened bookkeeping.
        let p =
            plan("SELECT t.b, t.id, (SELECT k FROM u WHERE u.k = t.b) FROM t GROUP BY t.b");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references outer t.b => correlated");
        assert_eq!(sub.outer_width, 2, "post-agg row = num_keys(1) + aggs(0) + captured(1)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 2, "u.k own at base 2 => reg 2");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 2, "u.k own => reg 2 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_with_a_window_function_widens_outer_width() {
        // Row-widening combination, now SUPPORTED. A post-aggregate window call
        // (`count(*) OVER ()`) appends its result after the aggregate row (Aggregate ->
        // Window -> Project), so the projection's outer row is `[t.b_key, window_result]`
        // (width 2 = num_keys(1) + num_aggregates(0) + num_window(1)). The correlated subquery
        // is re-bound with `outer_width = 2`; its own `u` sits at base 2 and outer `t.b`
        // remaps to post-agg key reg 0.
        let p = plan(
            "SELECT t.b, count(*) OVER (), (SELECT k FROM u WHERE u.k = t.b) FROM t GROUP BY t.b",
        );
        let sub = only_sub(&p);
        assert!(sub.correlated, "references outer t.b => correlated");
        assert_eq!(sub.outer_width, 2, "post-window row = num_keys(1) + aggs(0) + window(1)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 2, "u.k own at base 2 => reg 2");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 2, "u.k own => reg 2 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_with_an_order_by_aggregate_widens_outer_width() {
        // Row-widening combination, now SUPPORTED. The ORDER BY introduces an aggregate
        // (`count(*)`) not in the projection, pushing the real aggregate count to 1, so the
        // post-aggregate row is `[t.b_key, count_result]` (width 2 = num_keys(1) +
        // num_aggregates(1)). The correlated subquery is re-bound with `outer_width = 2`; its
        // own `u` sits at base 2 and outer `t.b` remaps to post-agg key reg 0.
        let p = plan(
            "SELECT t.b, (SELECT k FROM u WHERE u.k = t.b) FROM t GROUP BY t.b ORDER BY count(*)",
        );
        let sub = only_sub(&p);
        assert!(sub.correlated, "references outer t.b => correlated");
        assert_eq!(sub.outer_width, 2, "post-agg row = num_keys(1) + order-by agg(1)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 2, "u.k own at base 2 => reg 2");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 2, "u.k own => reg 2 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_in_having_with_a_window_binds_at_the_pre_window_width() {
        // HAVING × window — the combination that used to be a loud "not yet
        // supported" reject. A window call puts the projection / ORDER BY on the post-WINDOW
        // row `[t.b_key, window]`, but HAVING runs INSIDE the aggregate operator over the
        // NARROWER pre-window row `[t.b_key]` (the window column is not yet in scope). Because
        // `compile::aggregate` now re-points the correlated outer width per host clause, the
        // HAVING subquery binds at the PRE-window width `num_keys(1) + aggs(0) = 1`: the outer
        // `t.b` remaps to post-agg key `Column(0)` and `u` sits at base 1 (u.k = reg 1) — NOT
        // the wider width the projection subquery would use.
        let p = plan(
            "SELECT t.b, count(*) OVER () FROM t GROUP BY t.b \
             HAVING (SELECT k FROM u WHERE u.k = t.b) > 0",
        );
        let sub = only_sub(&p);
        assert!(sub.correlated, "HAVING subquery references outer t.b => correlated");
        assert_eq!(sub.outer_width, 1, "HAVING reads the PRE-window row: num_keys(1) + aggs(0)");
        assert_eq!(sub.correlated_cols, vec![0], "outer t.b remapped to post-agg key reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 1, "u.k own at base 1 => reg 1");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 1, "u.k own => reg 1 (>= outer_width)");
        assert_eq!(reg(r), 0, "outer t.b => post-agg key reg 0 (< outer_width)");
    }

    #[test]
    fn correlated_subquery_in_where_of_an_aggregate_query_still_works() {
        // Precision of the guard: a WHERE-clause subquery of an aggregate query binds
        // against the PRE-aggregate scope (grouping = None), so its outer genuinely IS
        // the FROM row — it must stay a correct correlated bind, NOT be rejected. u.g own
        // at base 3 => reg 5, outer t.b => reg 1.
        let p =
            plan("SELECT t.b FROM t WHERE t.id IN (SELECT k FROM u WHERE u.g = t.b) GROUP BY t.b");
        let sub = only_sub(&p);
        assert!(sub.correlated, "WHERE subquery correlated on t.b (pre-aggregate)");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 5, "u.g own at base 3 => reg 5");
        assert_eq!(reg(r), 1, "t.b outer => reg 1");
    }

    #[test]
    fn correlated_aggregate_subquery_under_a_nonaggregate_parent_is_correlated() {
        // The SUBQUERY aggregates (`max`) and correlates to the outer `t`, but the
        // ENCLOSING query is NOT aggregate (the aggregate is inside the subquery). So the
        // grouping guard must NOT fire (parent.grouping = None) and the subquery binds
        // correctly: `max`'s argument `t.id` is the outer ref at Column(0). Also defends
        // the `saw_correlated` propagation into the subquery's OWN grouping scope
        // (compile/aggregate.rs) — drop it and this correlation is silently missed.
        let p = plan("SELECT (SELECT max(t.id) FROM u) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "max(t.id) references outer t.id => correlated");
        let agg = match project_input(&sub.plan) {
            PlanNode::Aggregate(a) => a,
            other => panic!("expected Aggregate under the projection, got {other:?}"),
        };
        assert_eq!(reg(&agg.aggregates[0].args[0]), 0, "max(t.id): t.id outer at reg 0");
    }

    #[test]
    fn non_correlated_subquery_in_aggregate_projection_still_compiles() {
        // Placement guard: the aggregate rejection fires only AFTER the `!cell.get()`
        // non-correlated early return, so a NON-correlated subquery in an aggregate
        // projection must still compile (correlated == false), NOT be rejected. Moving
        // the guard above the early return would wrongly reject this — this pins it.
        let p = plan("SELECT (SELECT max(x) FROM u), count(*) FROM t GROUP BY t.b");
        let sub = only_sub(&p);
        assert!(!sub.correlated, "subquery names only u => not correlated, must compile");
    }

    #[test]
    fn correlated_subquery_that_is_itself_an_aggregate_with_an_outer_ref_is_a_loud_error() {
        // The "own-aggregate" twin: the SUBQUERY aggregates and its OWN projection
        // references an outer column (`t.id`), crossing its OWN aggregate boundary. The
        // enclosing query is non-aggregate, so the grouping guard does not apply; instead
        // `t.id` — neither a group key of the subquery nor inside one of its aggregates —
        // is caught by the ungrouped-column check (bind/expr.rs). The property pinned here
        // is LOUDNESS: this correlated-across-an-aggregate shape (which real sqlite
        // supports, and a future two-pass fix would) must never silently return a wrong
        // answer. If a refactor of that check ever turns this silent, this test goes red.
        let err = plan_err("SELECT (SELECT max(x) + t.id FROM u) FROM t");
        assert!(
            format!("{err:?}").contains("GROUP BY"),
            "must stay a loud error (currently the ungrouped-column check), got {err:?}"
        );
    }

    #[test]
    fn grandparent_reference_through_an_aggregate_projection_is_a_loud_error() {
        // RESIDUAL LOUD REJECT — the enclosing-query (grandparent) case. The innermost
        // `(SELECT t.id)` resolves `t.id` THROUGH the aggregate `SELECT ... FROM u GROUP BY
        // u.g` to the grandparent `FROM t`. At exec the executor prepends ONLY the aggregate's
        // post-aggregate row `[u.g_key]` — `t.id` is NOT in it — so no register can read it.
        // `Scope::remap_post_aggregate` decides locality by NAME (a grandparent register can
        // numerically overlap the aggregate's own FROM registers), so this rejects loudly
        // rather than reading a group-key slot (a silent wrong answer). Before the name-based
        // fix, the non-local branch returned the grandparent register unchanged.
        let err = plan_err("SELECT (SELECT (SELECT t.id) FROM u GROUP BY u.g) FROM t");
        assert!(
            format!("{err:?}").contains("enclosing the aggregate"),
            "got {err:?}"
        );
    }

    #[test]
    fn grandparent_reference_through_an_aggregate_having_is_a_loud_error() {
        // Same reject via the HAVING path (the witness shape): the HAVING's EXISTS
        // subquery references the grandparent `t.id` through the aggregate over `u`. Rejected
        // loudly — the grandparent row is not present in the post-aggregate row the HAVING
        // subquery runs over.
        let err = plan_err(
            "SELECT (SELECT count(*) FROM u GROUP BY u.g \
             HAVING EXISTS (SELECT 1 FROM w WHERE w.n = t.id)) FROM t",
        );
        assert!(
            format!("{err:?}").contains("enclosing the aggregate"),
            "got {err:?}"
        );
    }

    #[test]
    fn grandparent_reference_colliding_with_a_group_key_register_is_a_loud_error() {
        // The MAXIMALLY-ADVERSARIAL grandparent case: the grandparent reference's resolved
        // register NUMERICALLY EQUALS a group-key FROM register, so the DELETED register-range
        // check would have SILENTLY SUCCEEDED with a wrong VALUE — not rejected with a different
        // message like the other grandparent tests. The inner aggregate GROUPs BY `u.k` (u's
        // FIRST column, reg 0), and the grandparent `t.id` also resolves to reg 0. Under the old
        // register-range path this bound WITHOUT error: in the base-0 correlation trial,
        // `is_local(0)` was true and `col_to_group.get(&0) == Some(0)` SILENTLY remapped t.id to
        // the u.k key (an `Ok`, marking the subplan correlated); the rebind at the outer width
        // then took the `!is_local` grandparent PASSTHROUGH — so planning succeeded and the
        // innermost `(SELECT t.id)` read the post-aggregate u.k slot instead of t.id, a silent
        // wrong answer. The name-based `names_local_column` ignores registers entirely (`t` is
        // not a local source of the `u` aggregate), so this is a loud reject. `plan_err` PANICS
        // if planning SUCCEEDS, so this test fails on the old silent path and passes on the fix.
        let err = plan_err("SELECT (SELECT (SELECT t.id) FROM u GROUP BY u.k) FROM t");
        assert!(
            format!("{err:?}").contains("enclosing the aggregate"),
            "got {err:?}"
        );
    }

    #[test]
    fn non_correlated_windowed_subquery_in_a_multi_aggregate_group_by_binds_to_own_width() {
        // SAFETY-PROPERTY regression. A NON-correlated subquery containing a window function,
        // in the SELECT list of a multi-aggregate GROUP BY, must bind its window result at its
        // OWN FROM width — never the enclosing aggregate's (wider) post-aggregate width. Outer
        // `GROUP BY t.b` has 3 aggregates, so `subplan_outer_width = num_keys(1) +
        // num_aggregates(3) = 4`; the subquery `SELECT count(*) OVER () FROM w` has own FROM
        // width 3 (m, n, rowid). A non-correlated subquery runs with an EMPTY outer at exec, so
        // `count(*) OVER ()` must land at Column(3); the pre-fix `scope.total_width()` maxed in
        // the post-agg width and bound it to Column(4) — past the real [m, n, rowid, count]
        // row (an exec out-of-range). This pins the window-input width independent of the
        // (non-prepended) outer prefix, keeping the non-correlated aggregate plan intact.
        let p = plan(
            "SELECT t.b, count(*), sum(t.id), max(t.id), (SELECT count(*) OVER () FROM w) \
             FROM t GROUP BY t.b",
        );
        let sub = only_sub(&p);
        assert!(!sub.correlated, "names only w => not correlated");
        assert_eq!(sub.outer_width, 0, "non-correlated => outer_width 0 (EMPTY exec outer)");
        assert_eq!(
            reg(&project_exprs(&sub.plan)[0]),
            3,
            "count(*) OVER () binds to w's own FROM width (3), not the post-agg width (4)"
        );
    }

    #[test]
    fn non_correlated_windowed_subquery_under_a_wider_regular_parent_binds_to_own_width() {
        // The pre-existing sibling the same fix closes: a non-correlated windowed subquery
        // under a WIDER non-aggregate parent. Parent `u` is width 4 > subquery `w` width 3, so
        // the old `total_width().max(parent)` bound `count(*) OVER ()` to Column(4) (an exec
        // out-of-range); it must bind to w's own width, Column(3). Proves the window-input
        // width is independent of ANY (non-prepended) enclosing width, not just the aggregate.
        let p = plan("SELECT (SELECT count(*) OVER () FROM w) FROM u");
        let sub = only_sub(&p);
        assert!(!sub.correlated, "names only w => not correlated");
        assert_eq!(
            reg(&project_exprs(&sub.plan)[0]),
            3,
            "window col at w's own width 3, not the parent's 4"
        );
    }

    // -- No-FROM and nested correlation --------------------------------------

    #[test]
    fn correlated_no_from_subquery_reads_outer_at_reg_zero() {
        // A correlated subquery with NO FROM (`SELECT t.id`) — a SingleRow leaf whose own
        // width is 0, so the outer ref is the whole row: t.id at Column(0). Distinct from
        // the scan-based correlated cases.
        let p = plan("SELECT (SELECT t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "references t.id => correlated");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 0, "t.id outer at reg 0");
    }

    #[test]
    fn nested_correlation_through_a_no_from_intermediate_composes() {
        // Two-level nesting where the MIDDLE subquery has no FROM and the INNER reaches
        // through it to the grandparent `t`. Exercises (a) the restore/rebind recursion
        // (the middle re-binds, re-registering the inner) and (b) total_width's no-FROM
        // fallback: the inner's outer_width is taken from the middle scope, which has no
        // sources and must fall back to the grandparent width (3) — NOT 0, else the
        // inner's own u.k and the grandparent t.id would collide at Column(0).
        let p = plan("SELECT (SELECT (SELECT u.x FROM u WHERE u.k = t.id)) FROM t");
        assert_eq!(p.subqueries.len(), 2, "inner + middle both registered exactly once");
        assert!(p.subqueries.iter().all(|s| s.correlated), "both nesting levels correlated");
        let inner = p
            .subqueries
            .iter()
            .find(|s| matches!(project_input(&s.plan), PlanNode::Filter { .. }))
            .expect("the inner subquery carries the WHERE filter");
        let (pred, _) = filter(project_input(&inner.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 3, "u.k own at base 3 (no-FROM fallback outer_width, not 0)");
        assert_eq!(reg(r), 0, "t.id grandparent outer at reg 0");
    }

    // -- SubPlan analysis fields (correlated_cols / deterministic / outer_width) ----
    //
    // These pin the foundation a memoizing evaluator reads: the exact set of
    // outer registers a correlated subplan depends on, whether it is safe to memoize, and
    // the outer width it was compiled against. `correlated_cols` completeness is a
    // correctness invariant (a missed register => two different outer rows collide on one
    // cache entry => wrong answer), so these assert the exact register sets.

    #[test]
    fn uncorrelated_subplan_has_empty_analysis() {
        // No outer reference: `correlated_cols` empty, `outer_width` 0, deterministic.
        let p = plan("SELECT (SELECT max(x) FROM u) FROM t");
        let sub = only_sub(&p);
        assert!(!sub.correlated);
        assert_eq!(sub.correlated_cols, Vec::<usize>::new(), "no outer registers");
        assert_eq!(sub.outer_width, 0, "uncorrelated => outer_width 0");
        assert!(sub.deterministic, "max(x) is deterministic");
    }

    #[test]
    fn correlated_single_column_records_the_one_outer_register() {
        // `(SELECT u.x FROM u WHERE u.k = t.id)` depends on exactly the outer `t.id` (reg 0).
        let p = plan("SELECT (SELECT u.x FROM u WHERE u.k = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        assert_eq!(sub.correlated_cols, vec![0], "depends on outer t.id at reg 0");
        assert_eq!(sub.outer_width, 3, "outer t is width 3 (id,b,rowid)");
        assert!(sub.deterministic);
    }

    #[test]
    fn correlated_multi_column_records_both_outer_registers_sorted_deduped() {
        // `WHERE u.k = t.b AND u.g = t.id` reads outer t.b (1) FIRST then t.id (0), so the
        // collector sees them in [1, 0] resolve order — this asserts the RESULT is [0, 1],
        // which genuinely bites on the `sort_unstable()` (drop it and the vec stays [1, 0]).
        let p = plan("SELECT (SELECT u.x FROM u WHERE u.k = t.b AND u.g = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        assert_eq!(sub.correlated_cols, vec![0, 1], "both outer registers, sorted ascending");
        assert_eq!(sub.outer_width, 3);
        assert!(sub.deterministic);
    }

    #[test]
    fn repeated_outer_reference_is_deduped_to_one_register() {
        // The same outer column referenced twice (`u.k = t.id AND u.x = t.id`) collapses to
        // a single register — the memo key must not list a register twice.
        let p = plan("SELECT (SELECT u.g FROM u WHERE u.k = t.id AND u.x = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        assert_eq!(sub.correlated_cols, vec![0], "t.id referenced twice => one entry");
    }

    #[test]
    fn low_cardinality_correlation_canonical_shape() {
        // The headline O(n^2) shape (adapted to the test catalog): an aggregate
        // subquery in the WHERE of an aggregate outer, correlating on a low-cardinality
        // outer column `a.g` (reg 2 of `u a`). This is exactly the case the memoizer handles.
        let p = plan("SELECT count(*) FROM u a WHERE a.k = (SELECT max(x) FROM u b WHERE b.g = a.g)");
        let sub = only_sub(&p);
        assert!(sub.correlated, "correlates on a.g");
        assert_eq!(sub.correlated_cols, vec![2], "depends only on outer a.g (reg 2 of u)");
        assert_eq!(sub.outer_width, 4, "outer u is width 4 (k,x,g,rowid)");
        assert!(sub.deterministic, "max is deterministic => memoizable");
    }

    #[test]
    fn correlated_subquery_with_random_is_not_deterministic() {
        // `random()` makes the subquery unsafe to memoize even though it is correlated: the
        // register set is still recorded, but `deterministic` is false so the memoizer falls
        // back to re-running it.
        let p = plan("SELECT (SELECT random() FROM u WHERE u.k = t.id) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "still correlated on t.id");
        assert_eq!(sub.correlated_cols, vec![0]);
        assert!(!sub.deterministic, "random() => not deterministic");
    }

    #[test]
    fn correlated_subquery_with_current_timestamp_is_not_deterministic() {
        // A `CURRENT_TIMESTAMP` (an `EvalExpr::Now`) is read per-eval, so a correlated
        // subquery containing one is non-deterministic too — the Now-node half of the
        // determinism rule.
        let p = plan("SELECT (SELECT u.x FROM u WHERE u.k = t.id AND u.g < CURRENT_TIMESTAMP) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated);
        assert!(!sub.deterministic, "CURRENT_TIMESTAMP => not deterministic");
    }

    #[test]
    fn correlated_subquery_with_random_in_a_derived_table_is_not_deterministic() {
        // The scalar subquery is correlated on t.b and contains random() inside a FROM
        // derived table. The derived table is re-materialized per outer-row re-open
        // (ops/cte.rs), so random() is re-drawn each row => the subplan is genuinely
        // non-deterministic and MUST NOT be memoized. random() here is NOT bound through
        // the subplan's nondet cell (the derived body compiles via compile_derived's
        // fresh scope) and the body lands in ctx.ctes (not ctx.subqueries), so only the
        // CTE/derived-table barrier catches it. Regression guard for that hole.
        let p = plan("SELECT (SELECT max(s.r) FROM (SELECT random() AS r FROM u) s WHERE t.b > 0) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "correlated on t.b");
        assert_eq!(sub.correlated_cols, vec![1], "t.b at reg 1");
        assert!(!sub.deterministic, "random() in the derived table => NOT deterministic");
    }

    #[test]
    fn correlated_subquery_over_a_derived_table_is_conservatively_not_deterministic() {
        // The CTE/derived-table barrier is CONSERVATIVE: a derived-table body is not
        // determinism-analyzed, so ANY correlated subplan reaching one reports
        // deterministic == false — even when the body has no impurity. This only forgoes
        // memoizing this shape (safe), and pins the deliberate over-approximation so a
        // future precise CTE-body analysis is a conscious change, not an accident.
        let p = plan("SELECT (SELECT max(s.k) FROM (SELECT k FROM u) s WHERE t.b > 0) FROM t");
        let sub = only_sub(&p);
        assert!(sub.correlated, "correlated on t.b");
        assert!(
            !sub.deterministic,
            "a derived-table body is un-analyzed => conservatively non-deterministic"
        );
    }

    #[test]
    fn nested_correlation_captures_the_grandparent_register_at_every_level() {
        // Transitivity: the inner subquery reaches through a no-FROM middle to the
        // grandparent `t.id` (reg 0). BOTH the inner and the middle must record reg 0 in
        // `correlated_cols` — the middle carries the outer row down, so it depends on it
        // too. This is the completeness guarantee the binder-collector gives for free
        // (each level's `resolve_via_parent` is walked, so each level collects the register).
        let p = plan("SELECT (SELECT (SELECT u.x FROM u WHERE u.k = t.id)) FROM t");
        assert_eq!(p.subqueries.len(), 2, "inner + middle");
        for (i, s) in p.subqueries.iter().enumerate() {
            assert!(s.correlated, "subquery {i} correlated");
            assert_eq!(s.correlated_cols, vec![0], "subquery {i} depends on grandparent t.id (reg 0)");
            assert_eq!(s.outer_width, 3, "both compiled against the width-3 outer/grandparent row");
            assert!(s.deterministic, "subquery {i} deterministic");
        }
    }

    #[test]
    fn nested_nondeterminism_propagates_to_the_enclosing_subplan() {
        // Determinism is transitive: the inner subquery holds `random()`, so BOTH it and
        // the enclosing middle (which has no non-deterministic construct of its own) report
        // `deterministic == false` — the middle folds its descendant's verdict in.
        let p = plan("SELECT (SELECT (SELECT random() FROM u WHERE u.k = t.id)) FROM t");
        assert_eq!(p.subqueries.len(), 2, "inner + middle");
        assert!(
            p.subqueries.iter().all(|s| !s.deterministic),
            "the random() in the inner subquery makes both levels non-deterministic"
        );
    }

    // -- Correlation generalizes to DML --------------------------------------

    #[test]
    fn correlated_subquery_in_delete_where_is_correlated() {
        // `resolve_via_parent` now resolves outer refs for EVERY `bind_expr` caller,
        // including DML. A DELETE WHERE correlated IN-subquery binds correlated (the
        // executor hands it the deleted row as outer, width = total_width = 3): u.g own
        // at base 3 => reg 5, outer t.id => reg 0.
        let p = plan("DELETE FROM t WHERE id IN (SELECT k FROM u WHERE u.g = t.id)");
        let sub = only_sub(&p);
        assert!(sub.correlated, "DELETE WHERE subquery correlated on t.id");
        let (pred, _) = filter(project_input(&sub.plan));
        let (l, r) = compare(pred);
        assert_eq!(reg(l), 5, "u.g own at base 3 => reg 5");
        assert_eq!(reg(r), 0, "t.id outer => reg 0");
    }

    #[test]
    fn correlated_join_subquery_in_delete_binds_sources_after_the_deleted_row() {
        // A correlated FROM-join now works in DML too (the subplan flows through the same
        // `compile_core`, previously a loud gap). The executor hands the DELETE its deleted
        // row as outer (width = total_width = 3), so the joined sources sit AFTER it: u at
        // base 3 (k => reg 3), w at base 3+4=7. Correlated on outer t.id (reg 0). The WHERE
        // splits: `u.k = w.m` is the hash equikey and the correlated `u.g = t.id` the ON.
        let p =
            plan("DELETE FROM t WHERE id IN (SELECT k FROM u, w WHERE u.k = w.m AND u.g = t.id)");
        let sub = only_sub(&p);
        assert!(sub.correlated, "DELETE WHERE subquery correlated on t.id");
        assert_eq!(sub.correlated_cols, vec![0], "depends on outer t.id at reg 0");
        assert_eq!(reg(&project_exprs(&sub.plan)[0]), 3, "u.k own at base 3 (schema index 0) => reg 3");
        match project_input(&sub.plan) {
            PlanNode::Join(j) => {
                assert_eq!(j.left_width, 4, "u width 4 (k,x,g,rowid)");
                assert_eq!(j.right_width, 3, "w width 3 (m,n,rowid)");
                // The correlated `u.g = t.id` is the ON residual: u.g = reg 5, outer t.id = reg 0.
                let (l, r) = compare(j.on.as_ref().expect("correlated residual ON"));
                assert_eq!(reg(l), 5, "u.g own at base 3 => reg 5");
                assert_eq!(reg(r), 0, "t.id outer => reg 0");
            }
            other => panic!("expected a Join, got {other:?}"),
        }
    }
}
