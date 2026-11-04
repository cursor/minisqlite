//! Window-function binding and the `Window` operator insertion.
//!
//! A window call `f(args) [FILTER (WHERE …)] OVER (PARTITION BY … ORDER BY … frame)`
//! runs AFTER the row source (FROM + WHERE) and BEFORE the projection, appending one
//! column per call to the input row. The pipeline the SELECT orchestrator assembles is:
//!
//! ```text
//! input (FROM + WHERE) → [Window if any window calls] → Project → Distinct → Sort → Limit
//! ```
//!
//! [`bind_window_function`] is the binder hook (called from [`crate::bind::expr`] when a
//! function call carries an `OVER`): it maps the call to a [`WindowFuncKind`], binds its
//! arguments / `PARTITION BY` / `ORDER BY` / frame against the PRE-window input row,
//! collects a [`WindowFunc`] into the scope's [`Windowing`] context, and hands back
//! `Column(input_width + k)` — the register the operator appends for the k-th collected
//! function (mirroring how an aggregate call resolves to its post-aggregate result
//! register). [`wrap_window_if_any`] then drops the [`PlanNode::Window`] node in between,
//! once every projection and ORDER BY window call has been collected.
//!
//! # Coverage
//!
//! All documented window functions bind here: the fold-based aggregates
//! (`sum`/`count`/…), the ranking functions (`row_number`/`rank`/`dense_rank`/
//! `percent_rank`/`cume_dist`), `ntile`, the offset functions (`lag`/`lead`), and the
//! value functions (`first_value`/`last_value`/`nth_value`), plus explicit
//! `ROWS`/`RANGE`/`GROUPS` frames with `EXCLUDE`, named windows, and window chaining
//! (`OVER (base …)`). The plan this emits is the sole contract with the executor's
//! window operator (`minisqlite-exec`), which computes each kind/frame; the binder only
//! describes them.

use minisqlite_expr::{AggregateCall, EvalExpr, SortKey};
use minisqlite_sql::{
    Expr, FrameBound as AstFrameBound, FrameExclude as AstFrameExclude, FrameUnits as AstFrameUnits,
    FunctionArgs, NullsOrder, OrderingTerm, OverClause, SortOrder, WindowFrame as AstWindowFrame,
    WindowSpec,
};
use minisqlite_types::{Error, Result, Value};

use crate::bind::{
    arg_list_collations, best_effort_collation, bind_expr, parse_collation, Scope, Windowing,
};
use crate::plan::{PlanNode, Window, WindowFunc};
use crate::plan_ctx::PlanCtx;
use crate::window::{FrameBound, FrameExclude, FrameUnits, WindowFrame, WindowFuncKind};

/// Bind a window-function call (`f(...) OVER (...)`), collecting a [`WindowFunc`] and
/// returning the register of the column the [`PlanNode::Window`] operator appends for
/// it. The binder routes here from `bind_function` whenever a function call has an
/// `OVER` clause.
pub fn bind_window_function(
    scope: &Scope,
    ctx: &mut PlanCtx,
    name: &str,
    distinct: bool,
    args: &FunctionArgs,
    filter: &Option<Box<Expr>>,
    over: &OverClause,
    order_by: &[OrderingTerm],
) -> Result<EvalExpr> {
    // A window function is only valid in a windowing context: the SELECT list or ORDER
    // BY of a query, OR the post-GROUP-BY projection of an aggregated query (which binds
    // the window over the aggregate's output row — see `compile::aggregate`). In
    // WHERE / GROUP BY / HAVING / ON — including an aggregated query's GROUP BY keys and
    // HAVING, which bind through a windowing-stripped scope — the scope carries no
    // windowing context, so this is a loud error, never a mis-bind.
    let windowing = scope
        .windowing
        .ok_or_else(|| Error::sql(format!("misuse of window function {name}()")))?;

    // Resolve the OVER target to a concrete, base-resolved window spec (a named
    // window, an inline spec, or a chain extending a base window).
    let spec = effective_spec(over, windowing)?;

    // SQLite rejects DISTINCT on any window function (aggregate or not).
    if distinct {
        return Err(Error::sql("DISTINCT is not supported for window functions"));
    }

    // Arguments / in-argument ORDER BY / FILTER / PARTITION BY / window ORDER BY / frame all
    // read the PRE-window input row and must not collect a nested window function, so bind
    // through a windowing-stripped scope. Bind in SOURCE order (args, in-arg ORDER BY, filter,
    // partition, window order, frame) so `?` parameters number correctly.
    let inner = scope.without_windowing();
    let kind = bind_window_kind(&inner, ctx, name, args, filter, order_by)?;

    let mut partition_by = Vec::with_capacity(spec.partition_by.len());
    let mut partition_collations = Vec::with_capacity(spec.partition_by.len());
    for p in &spec.partition_by {
        // PARTITION BY groups rows like GROUP BY, under each key's §7.1 collation
        // (datatype3.html §7.1): an explicit postfix COLLATE, else the key column's
        // declared collation, else BINARY. Resolved once here and carried on the plan so
        // the window operator keys partitions under it — mirroring GROUP BY's
        // `group_collations` and the window ORDER BY's per-key `SortKey.collation` (both
        // already collation-aware). `best_effort_collation` is the full §7.1 rule.
        partition_collations.push(best_effort_collation(&inner, p));
        partition_by.push(bind_expr(&inner, ctx, p)?);
    }
    let mut order_by = Vec::with_capacity(spec.order_by.len());
    for term in &spec.order_by {
        order_by.push(bind_window_order_term(&inner, ctx, term)?);
    }
    let frame = bind_frame(&inner, ctx, spec.frame.as_ref())?;
    // A window frame is validated at bind time regardless of the function (SQLite
    // validates a frame even on `row_number`/`rank`/… that ignore it at run time). The
    // executor's frame engine documents these as planner-guaranteed invariants
    // (`minisqlite-exec` `ops/window/frame.rs`) and only handles a violation defensively,
    // so an unvalidated bad frame is a silent wrong answer where real SQLite errors.
    validate_frame(&frame, order_by.len())?;

    let wf = WindowFunc { kind, partition_by, partition_collations, order_by, frame };
    let mut funcs = windowing.functions.borrow_mut();
    let idx = funcs.len();
    funcs.push(wf);
    // `input_width` is consumed ONLY here — it shifts a window call's OWN result register,
    // never what gets collected (args/PARTITION/ORDER bind through `without_windowing()`,
    // width-independent). `compile::aggregate`'s two-pass binding relies on exactly that:
    // pass 1 counts aggregates/window calls with a provisional width and pass 2 re-binds
    // with the true post-aggregate width, and the passes MUST collect identical counts. If
    // a future change makes `input_width` influence collection, that contract breaks and
    // the release build (its debug_asserts stripped) would silently misplace this register.
    Ok(EvalExpr::Column(windowing.input_width + idx))
}

/// Wrap `input` in a [`PlanNode::Window`] iff the scope's windowing context collected
/// any window functions. Called once, after the projection and ORDER BY are bound, so
/// the appended columns line up with the `Column(input_width + k)` references those
/// bindings produced. A query with no window calls (or no windowing context) is returned
/// unchanged.
pub fn wrap_window_if_any(scope: &Scope, input: PlanNode) -> PlanNode {
    if let Some(w) = scope.windowing {
        let funcs = std::mem::take(&mut *w.functions.borrow_mut());
        if !funcs.is_empty() {
            return PlanNode::Window(Window { input: Box::new(input), functions: funcs });
        }
    }
    input
}

/// Map a window call to its [`WindowFuncKind`], binding the call's arguments (and, for
/// an aggregate, its in-argument `ORDER BY` and `FILTER`) against `inner`. The dedicated
/// window built-ins are matched by name; any other name is resolved as an aggregate used
/// over a window. `order_by` is the ordered-set aggregate's in-argument `ORDER BY`
/// (`group_concat(x ORDER BY y) OVER (…)`); it is bound onto the aggregate call and is a
/// loud error for a non-aggregate/built-in window function (which has no fold order).
fn bind_window_kind(
    inner: &Scope,
    ctx: &mut PlanCtx,
    name: &str,
    args: &FunctionArgs,
    filter: &Option<Box<Expr>>,
    order_by: &[OrderingTerm],
) -> Result<WindowFuncKind> {
    let lname = name.to_ascii_lowercase();
    if is_builtin_window_name(&lname) {
        // FILTER is only valid on aggregate window functions (lang_aggfunc / §3).
        if filter.is_some() {
            return Err(Error::sql(
                "FILTER clause may only be used with aggregate window functions",
            ));
        }
        // An in-argument ORDER BY sets an aggregate's fold order; a built-in window
        // function (`row_number`/`lag`/`nth_value`/…) is not an aggregate and has no such
        // order, so reject it loudly (matching the non-window "non-aggregate" wording).
        if !order_by.is_empty() {
            return Err(Error::sql(format!(
                "ORDER BY may not be used with the non-aggregate function {name}()"
            )));
        }
        return bind_builtin_window_kind(inner, ctx, &lname, name, args);
    }

    // Not a dedicated window built-in and not an aggregate: a misuse. SQLite splits this
    // by whether the name is KNOWN: a known function used where a window function is
    // required is "<name>() may not be used as a window function", while an entirely
    // UNKNOWN name is "no such function: <name>". A known name reaching here is a scalar
    // (the window built-ins and real aggregates are both handled above), so surfacing the
    // aggregate resolver's "no such function" for it would be wrong — it names a function
    // that plainly exists.
    if !is_aggregate_window(ctx, name, args) {
        if ctx.registry.is_known(name) {
            return Err(Error::sql(format!("{name}() may not be used as a window function")));
        }
        // Unknown to both namespaces: the aggregate resolver's verbatim "no such function"
        // (an unknown name never resolves, so the Ok arm is unreachable and defensively
        // maps to the same misuse error rather than panicking).
        return match ctx.registry.resolve_aggregate(name, arg_count(args)) {
            Err(e) => Err(e),
            Ok(_) => Err(Error::sql(format!("{name}() may not be used as a window function"))),
        };
    }
    let (bound_args, argc) = bind_arg_list(inner, ctx, args)?;
    // In-argument ORDER BY of an ordered-set aggregate window function
    // (`group_concat(x ORDER BY y) OVER (…)`): bound in SOURCE order (after args, before
    // FILTER) so `?` parameters number correctly, over the same pre-window `inner` scope as
    // the arguments — an ordering key may reference any input column, not just the argument.
    // The executor's window operator sorts each frame's rows by these keys before folding
    // (`ops::window`), matching the non-window aggregate path (`ops::aggregate`).
    let mut bound_order = Vec::with_capacity(order_by.len());
    for term in order_by {
        bound_order.push(bind_window_order_term(inner, ctx, term)?);
    }
    let bound_filter = match filter {
        Some(fe) => Some(bind_expr(inner, ctx, fe)?),
        None => None,
    };
    // The registry reports unknown-name / wrong-arg-count with SQLite's wording.
    let func = ctx.registry.resolve_aggregate(name, argc)?;
    Ok(WindowFuncKind::Aggregate(AggregateCall {
        func,
        distinct: false,
        args: bound_args,
        filter: bound_filter,
        order_by: bound_order,
        // A windowed aggregate is never DISTINCT, so these are unread; carried aligned
        // with `args` for shape consistency with the AggregateCall contract.
        arg_collations: arg_list_collations(inner, args),
    }))
}

/// The dedicated window built-ins (ranking / offset / value functions). These are NOT
/// registry functions — they exist only as window functions and are recognized by name.
fn is_builtin_window_name(lname: &str) -> bool {
    matches!(
        lname,
        "row_number"
            | "rank"
            | "dense_rank"
            | "percent_rank"
            | "cume_dist"
            | "ntile"
            | "lag"
            | "lead"
            | "first_value"
            | "last_value"
            | "nth_value"
    )
}

/// Build the [`WindowFuncKind`] for a dedicated window built-in, validating its
/// argument count and binding its argument expressions (in source order). `lname` is
/// the lowercased name; `orig` is the original spelling used in error messages.
fn bind_builtin_window_kind(
    inner: &Scope,
    ctx: &mut PlanCtx,
    lname: &str,
    orig: &str,
    args: &FunctionArgs,
) -> Result<WindowFuncKind> {
    // A `*` argument (`count(*)`-style) is never valid for these built-ins.
    let list: &[Expr] = match args {
        FunctionArgs::List(l) => l.as_slice(),
        FunctionArgs::Empty => &[],
        FunctionArgs::Star => return Err(wrong_args(orig)),
    };
    let n = list.len();
    Ok(match lname {
        "row_number" | "rank" | "dense_rank" | "percent_rank" | "cume_dist" => {
            if n != 0 {
                return Err(wrong_args(orig));
            }
            match lname {
                "row_number" => WindowFuncKind::RowNumber,
                "rank" => WindowFuncKind::Rank,
                "dense_rank" => WindowFuncKind::DenseRank,
                "percent_rank" => WindowFuncKind::PercentRank,
                "cume_dist" => WindowFuncKind::CumeDist,
                _ => unreachable!("outer match limits lname to the ranking set"),
            }
        }
        "ntile" => {
            if n != 1 {
                return Err(wrong_args(orig));
            }
            WindowFuncKind::Ntile(bind_expr(inner, ctx, &list[0])?)
        }
        // lag/lead(expr [, offset [, default]]): an omitted offset is 1, an omitted
        // default is NULL. Bind present arguments in source order for `?` numbering.
        "lag" | "lead" => {
            if !(1..=3).contains(&n) {
                return Err(wrong_args(orig));
            }
            let expr = bind_expr(inner, ctx, &list[0])?;
            let offset = match list.get(1) {
                Some(o) => bind_expr(inner, ctx, o)?,
                None => EvalExpr::Literal(Value::Integer(1)),
            };
            let default = match list.get(2) {
                Some(d) => bind_expr(inner, ctx, d)?,
                None => EvalExpr::Literal(Value::Null),
            };
            if lname == "lag" {
                WindowFuncKind::Lag { expr, offset, default }
            } else {
                WindowFuncKind::Lead { expr, offset, default }
            }
        }
        "first_value" | "last_value" => {
            if n != 1 {
                return Err(wrong_args(orig));
            }
            let e = bind_expr(inner, ctx, &list[0])?;
            if lname == "first_value" {
                WindowFuncKind::FirstValue(e)
            } else {
                WindowFuncKind::LastValue(e)
            }
        }
        "nth_value" => {
            if n != 2 {
                return Err(wrong_args(orig));
            }
            let expr = bind_expr(inner, ctx, &list[0])?;
            let nth = bind_expr(inner, ctx, &list[1])?;
            WindowFuncKind::NthValue { expr, n: nth }
        }
        _ => unreachable!("is_builtin_window_name limits lname to the handled set"),
    })
}

/// Bind an aggregate window call's argument list, returning the bound exprs and the
/// argument count (for arity resolution). `count(*)` / `f()` bind zero arguments.
fn bind_arg_list(
    inner: &Scope,
    ctx: &mut PlanCtx,
    args: &FunctionArgs,
) -> Result<(Vec<EvalExpr>, usize)> {
    match args {
        FunctionArgs::Star | FunctionArgs::Empty => Ok((Vec::new(), 0)),
        FunctionArgs::List(list) => {
            let mut v = Vec::with_capacity(list.len());
            for it in list {
                v.push(bind_expr(inner, ctx, it)?);
            }
            Ok((v, list.len()))
        }
    }
}

/// The argument count of a call for arity resolution (a `*` / empty list is zero).
fn arg_count(args: &FunctionArgs) -> usize {
    match args {
        FunctionArgs::List(list) => list.len(),
        _ => 0,
    }
}

/// SQLite's "wrong number of arguments" error for a named function.
fn wrong_args(name: &str) -> Error {
    Error::sql(format!("wrong number of arguments to function {name}()"))
}

/// Lower an AST frame-spec to the resolved plan [`WindowFrame`]. An omitted frame is
/// the default (`WindowFrame::default_frame`); an omitted `AND <end>` is `CURRENT ROW`;
/// an omitted `EXCLUDE` is `NO OTHERS`. Frame-bound offset expressions bind against the
/// pre-window input row (`inner`).
fn bind_frame(
    inner: &Scope,
    ctx: &mut PlanCtx,
    frame: Option<&AstWindowFrame>,
) -> Result<WindowFrame> {
    let Some(f) = frame else {
        return Ok(WindowFrame::default_frame());
    };
    let units = match f.units {
        AstFrameUnits::Range => FrameUnits::Range,
        AstFrameUnits::Rows => FrameUnits::Rows,
        AstFrameUnits::Groups => FrameUnits::Groups,
    };
    let start = bind_frame_bound(inner, ctx, &f.start)?;
    let end = match &f.end {
        Some(b) => bind_frame_bound(inner, ctx, b)?,
        None => FrameBound::CurrentRow,
    };
    let exclude = match f.exclude {
        None | Some(AstFrameExclude::NoOthers) => FrameExclude::NoOthers,
        Some(AstFrameExclude::CurrentRow) => FrameExclude::CurrentRow,
        Some(AstFrameExclude::Group) => FrameExclude::Group,
        Some(AstFrameExclude::Ties) => FrameExclude::Ties,
    };
    Ok(WindowFrame { units, start, end, exclude })
}

/// Lower one AST frame bound, binding a `PRECEDING`/`FOLLOWING` offset expression.
fn bind_frame_bound(
    inner: &Scope,
    ctx: &mut PlanCtx,
    b: &AstFrameBound,
) -> Result<FrameBound> {
    Ok(match b {
        AstFrameBound::UnboundedPreceding => FrameBound::UnboundedPreceding,
        AstFrameBound::Preceding(e) => FrameBound::Preceding(bind_expr(inner, ctx, e)?),
        AstFrameBound::CurrentRow => FrameBound::CurrentRow,
        AstFrameBound::Following(e) => FrameBound::Following(bind_expr(inner, ctx, e)?),
        AstFrameBound::UnboundedFollowing => FrameBound::UnboundedFollowing,
    })
}

/// Reject a frame that violates the boundary rules SQLite enforces at prepare time
/// (`windowfunctions.html` §2.2), so the plan never carries a frame the executor's frame
/// engine only tolerates defensively. The four rules, all of which the DEFAULT frame
/// (`RANGE UNBOUNDED PRECEDING … CURRENT ROW`) satisfies, so validating unconditionally
/// is safe:
///
/// 1. the START boundary may not be `UNBOUNDED FOLLOWING`;
/// 2. the END boundary may not be `UNBOUNDED PRECEDING`;
/// 3. the END boundary may not appear "higher" in the boundary ordering than the START
///    (§2.2.2: order `UNBOUNDED PRECEDING` < `<expr> PRECEDING` < `CURRENT ROW` <
///    `<expr> FOLLOWING` < `UNBOUNDED FOLLOWING`); so e.g. `BETWEEN CURRENT ROW AND
///    <expr> PRECEDING` and the short form `<expr> FOLLOWING` (end defaults to
///    `CURRENT ROW`) are rejected;
/// 4. a `RANGE` frame with an `<expr> PRECEDING`/`FOLLOWING` boundary requires EXACTLY
///    one `ORDER BY` term (§2.2.1) — the band is a range around that single term's value.
///    `RANGE` with only `UNBOUNDED`/`CURRENT ROW` boundaries (the default frame included)
///    has no such requirement and works with any number of `ORDER BY` terms.
fn validate_frame(frame: &WindowFrame, num_order_by: usize) -> Result<()> {
    if matches!(frame.start, FrameBound::UnboundedFollowing) {
        return Err(Error::sql("a window frame may not start with UNBOUNDED FOLLOWING"));
    }
    if matches!(frame.end, FrameBound::UnboundedPreceding) {
        return Err(Error::sql("a window frame may not end with UNBOUNDED PRECEDING"));
    }
    if bound_rank(&frame.end) < bound_rank(&frame.start) {
        return Err(Error::sql(
            "the window frame ending boundary may not precede the starting boundary",
        ));
    }
    if frame.units == FrameUnits::Range
        && (is_offset_bound(&frame.start) || is_offset_bound(&frame.end))
        && num_order_by != 1
    {
        return Err(Error::sql(
            "RANGE with an offset PRECEDING/FOLLOWING boundary requires exactly one ORDER BY term",
        ));
    }
    Ok(())
}

/// The position of a boundary in the §2.2.2 ordering (lower = earlier / "higher in the
/// list"): the START may not sit after the END, which is exactly `rank(end) >= rank(start)`.
fn bound_rank(b: &FrameBound) -> u8 {
    match b {
        FrameBound::UnboundedPreceding => 1,
        FrameBound::Preceding(_) => 2,
        FrameBound::CurrentRow => 3,
        FrameBound::Following(_) => 4,
        FrameBound::UnboundedFollowing => 5,
    }
}

/// Whether a boundary carries an `<expr>` offset (`<expr> PRECEDING`/`FOLLOWING`) — the
/// boundary form that makes a `RANGE` frame a numeric band and so needs a single
/// `ORDER BY` term.
fn is_offset_bound(b: &FrameBound) -> bool {
    matches!(b, FrameBound::Preceding(_) | FrameBound::Following(_))
}

/// Resolve an `OVER` target to the concrete window spec to apply: an inline spec, a
/// named window from the `WINDOW` clause, or either extending a base window
/// (`windowfunctions.html` §4 window chaining).
fn effective_spec(over: &OverClause, windowing: &Windowing) -> Result<WindowSpec> {
    match over {
        OverClause::Spec(spec) => resolve_chain(windowing, spec.clone(), &mut Vec::new()),
        OverClause::WindowName(name) => {
            let spec = windowing
                .lookup(name)
                .cloned()
                .ok_or_else(|| Error::sql(format!("no such window: {name}")))?;
            resolve_chain(windowing, spec, &mut vec![name.to_ascii_lowercase()])
        }
    }
}

/// Resolve a window spec that may reference a base window into a self-contained spec,
/// applying the chaining rules (`windowfunctions.html` §4):
///
/// * the referencing window may NOT specify `PARTITION BY` (it comes from the base);
/// * if the base has an `ORDER BY`, it is inherited and the referencing window may NOT
///   specify its own; if the base has none, the referencing window may;
/// * the base window may NOT specify a frame — only the referencing window may.
///
/// `visited` (lowercased names) guards against a circular base chain, which would
/// otherwise not terminate.
fn resolve_chain(
    windowing: &Windowing,
    spec: WindowSpec,
    visited: &mut Vec<String>,
) -> Result<WindowSpec> {
    let Some(base_name) = spec.base.clone() else {
        return Ok(spec);
    };
    let key = base_name.to_ascii_lowercase();
    if visited.contains(&key) {
        return Err(Error::sql(format!("circular reference: {base_name}")));
    }
    visited.push(key);
    let base_spec = windowing
        .lookup(&base_name)
        .cloned()
        .ok_or_else(|| Error::sql(format!("no such window: {base_name}")))?;
    let base = resolve_chain(windowing, base_spec, visited)?;

    if !spec.partition_by.is_empty() {
        return Err(Error::sql(
            "PARTITION BY clause is not allowed in a window that references a base window",
        ));
    }
    if base.frame.is_some() {
        return Err(Error::sql("a base window definition may not specify a frame"));
    }
    let order_by = if base.order_by.is_empty() {
        spec.order_by
    } else {
        if !spec.order_by.is_empty() {
            return Err(Error::sql(
                "ORDER BY clause is not allowed in a window that references a base window with ORDER BY",
            ));
        }
        base.order_by
    };
    Ok(WindowSpec { base: None, partition_by: base.partition_by, order_by, frame: spec.frame })
}

/// Whether a function call is an aggregate usable as a window function: `count(*)`
/// always, else any name the registry classifies as an aggregate. (Unlike
/// `function_is_aggregate`, there is no `OVER` guard — the caller already knows this is
/// a window call — and no min/max-by-argc split: a 2+-arg `min`/`max` window call falls
/// through to the aggregate resolver, which reports its wrong-arg-count verbatim.)
fn is_aggregate_window(ctx: &PlanCtx, name: &str, args: &FunctionArgs) -> bool {
    match args {
        FunctionArgs::Star => name.eq_ignore_ascii_case("count"),
        _ => ctx.registry.is_aggregate(name),
    }
}

/// Lower one window `ORDER BY` term to a [`SortKey`] over the pre-window input row,
/// mirroring the query `ORDER BY` lowering (explicit `COLLATE` wins, else the operand's
/// column collation, else BINARY; the SQLite default NULL ordering is left implicit as
/// `None`).
fn bind_window_order_term(scope: &Scope, ctx: &mut PlanCtx, term: &OrderingTerm) -> Result<SortKey> {
    let expr = bind_expr(scope, ctx, &term.expr)?;
    let collation = match &term.collation {
        Some(name) => parse_collation(name)?,
        None => best_effort_collation(scope, &term.expr),
    };
    Ok(SortKey {
        expr,
        desc: matches!(term.order, Some(SortOrder::Desc)),
        nulls_first: term.nulls.map(|n| matches!(n, NullsOrder::First)),
        collation,
    })
}

#[cfg(test)]
mod tests {
    //! Window binding/compilation at the PLAN level: an `OVER (...)` call compiles to a
    //! [`PlanNode::Window`] between the row source and the projection, each call maps to
    //! the right [`WindowFuncKind`] with its arguments/partition/order/frame bound
    //! against the pre-window input row, and the call resolves to the appended output
    //! column `Column(input_width + k)`. These check the emitted plan (not execution),
    //! so they hold regardless of what the executor's window operator supports yet.
    //! Fixtures are local to this file so it never edits `tests.rs`.

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_expr::{CmpOp, EvalExpr};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::{Collation, Error, Result, Value};

    use crate::plan::{Plan, PlanNode, Window, WindowFunc};
    use crate::window::{FrameBound, FrameExclude, FrameUnits, WindowFuncKind};
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

    /// `t(id, grp, v)` all INTEGER: the scan row is `[id=0, grp=1, v=2, rowid=3]`, so
    /// the FROM width is 4 and a window function's appended column lands at `Column(4)`.
    fn cat() -> TestCatalog {
        TestCatalog {
            tables: vec![tdef(
                "t",
                vec![col("id", "INTEGER"), col("grp", "INTEGER"), col("v", "INTEGER")],
            )],
        }
    }

    fn plan(sql: &str) -> Plan {
        let c = cat();
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c).expect("plan ok")
    }

    fn plan_err(sql: &str) -> String {
        let c = cat();
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        match QueryPlanner::new().plan(stmt, &c) {
            Ok(_) => panic!("expected a planning error for: {sql}"),
            Err(Error::Sql(msg)) => msg,
            Err(other) => panic!("expected a SQL error, got {other:?}"),
        }
    }

    /// The `(exprs, input)` of the projection at the plan root. The test queries carry
    /// no trailing ORDER BY / DISTINCT / LIMIT, so the root is the projection directly.
    fn root_project(p: &Plan) -> (&[EvalExpr], &PlanNode) {
        match &p.root {
            PlanNode::Project { exprs, input } => (exprs, input),
            other => panic!("expected Project root, got {other:?}"),
        }
    }

    fn window_under(n: &PlanNode) -> &Window {
        match n {
            PlanNode::Window(w) => w,
            other => panic!("expected a Window node, got {other:?}"),
        }
    }

    /// The single window function of a `SELECT <one window call> FROM t` plan.
    fn only_func(p: &Plan) -> &WindowFunc {
        let (_, input) = root_project(p);
        let w = window_under(input);
        assert_eq!(w.functions.len(), 1, "expected exactly one window function");
        &w.functions[0]
    }

    // -- aggregate windows (the shape the executor already computes) --------------

    #[test]
    fn agg_over_empty_window_is_a_window_over_the_scan() {
        let p = plan("SELECT id, sum(v) OVER () FROM t");
        let (exprs, input) = root_project(&p);
        // `id` stays Column(0); the window result is the appended column Column(4).
        assert!(matches!(exprs[0], EvalExpr::Column(0)));
        assert!(matches!(exprs[1], EvalExpr::Column(4)), "window col at input_width, got {:?}", exprs[1]);
        let f = &window_under(input).functions[0];
        assert!(f.partition_by.is_empty(), "OVER () has no PARTITION BY");
        assert!(f.order_by.is_empty(), "OVER () has no ORDER BY");
        match &f.kind {
            WindowFuncKind::Aggregate(call) => {
                assert_eq!(call.args.len(), 1);
                assert!(matches!(call.args[0], EvalExpr::Column(2)), "sum(v) reads v = Column(2)");
            }
            other => panic!("expected an Aggregate kind, got {other:?}"),
        }
        // No explicit frame => the default frame.
        assert_default_frame(f);
    }

    #[test]
    fn count_star_over_empty_window_has_no_args() {
        let p = plan("SELECT count(*) OVER () FROM t");
        let f = only_func(&p);
        match &f.kind {
            WindowFuncKind::Aggregate(call) => assert!(call.args.is_empty(), "count(*) binds zero args"),
            other => panic!("expected an Aggregate kind, got {other:?}"),
        }
    }

    #[test]
    fn aggregate_window_filter_is_carried_on_the_call() {
        let p = plan("SELECT sum(v) FILTER (WHERE v > 10) OVER () FROM t");
        match &only_func(&p).kind {
            // Assert the BOUND predicate, not just that a filter exists: a bug that carried
            // the wrong expression (e.g. the aggregate arg instead of the FILTER) would
            // otherwise pass. `v > 10` binds to `Column(2) > Literal(10)`.
            WindowFuncKind::Aggregate(call) => match &call.filter {
                Some(EvalExpr::Compare { op: CmpOp::Gt, left, right, .. }) => {
                    assert!(matches!(**left, EvalExpr::Column(2)), "FILTER reads v = Column(2)");
                    assert!(matches!(**right, EvalExpr::Literal(Value::Integer(10))));
                }
                other => panic!("expected a bound `v > 10` filter, got {other:?}"),
            },
            other => panic!("expected Aggregate, got {other:?}"),
        }
    }

    #[test]
    fn partition_by_and_order_by_bind_against_the_input_row() {
        let p = plan("SELECT sum(v) OVER (PARTITION BY grp ORDER BY id) FROM t");
        let f = only_func(&p);
        // PARTITION BY grp = Column(1); ORDER BY id = Column(0), ascending.
        assert_eq!(f.partition_by.len(), 1);
        assert!(matches!(f.partition_by[0], EvalExpr::Column(1)));
        assert_eq!(f.order_by.len(), 1);
        assert!(matches!(f.order_by[0].expr, EvalExpr::Column(0)));
        assert!(!f.order_by[0].desc);
    }

    #[test]
    fn desc_window_order_by_sets_desc_on_the_sort_key() {
        let p = plan("SELECT sum(v) OVER (ORDER BY v DESC) FROM t");
        let f = only_func(&p);
        assert_eq!(f.order_by.len(), 1);
        assert!(matches!(f.order_by[0].expr, EvalExpr::Column(2)));
        assert!(f.order_by[0].desc, "ORDER BY v DESC must set desc=true");
    }

    #[test]
    fn window_order_by_nulls_first_and_last_thread_through() {
        let first = plan("SELECT sum(v) OVER (ORDER BY v NULLS FIRST) FROM t");
        assert_eq!(only_func(&first).order_by[0].nulls_first, Some(true));
        let last = plan("SELECT sum(v) OVER (ORDER BY v NULLS LAST) FROM t");
        assert_eq!(only_func(&last).order_by[0].nulls_first, Some(false));
    }

    #[test]
    fn window_order_by_explicit_collate_is_carried_on_the_sort_key() {
        // The executor honors each window SortKey's collation, so ORDER BY COLLATE is
        // carried through (PARTITION BY collation is carried on `partition_collations`).
        let p = plan("SELECT sum(v) OVER (ORDER BY v COLLATE NOCASE) FROM t");
        assert_eq!(only_func(&p).order_by[0].collation, Collation::NoCase);
    }

    #[test]
    fn multi_key_partition_and_multi_term_order_collect_every_term() {
        let p = plan("SELECT sum(v) OVER (PARTITION BY grp, id ORDER BY v, id) FROM t");
        let f = only_func(&p);
        assert_eq!(f.partition_by.len(), 2);
        assert!(matches!(f.partition_by[0], EvalExpr::Column(1)));
        assert!(matches!(f.partition_by[1], EvalExpr::Column(0)));
        assert_eq!(f.order_by.len(), 2);
        assert!(matches!(f.order_by[0].expr, EvalExpr::Column(2)));
        assert!(matches!(f.order_by[1].expr, EvalExpr::Column(0)));
    }

    #[test]
    fn two_window_calls_append_two_columns_in_order() {
        let p = plan("SELECT sum(v) OVER (), count(*) OVER () FROM t");
        let (exprs, input) = root_project(&p);
        assert!(matches!(exprs[0], EvalExpr::Column(4)));
        assert!(matches!(exprs[1], EvalExpr::Column(5)));
        assert_eq!(window_under(input).functions.len(), 2);
    }

    // -- ranking functions -------------------------------------------------------

    #[test]
    fn row_number_binds_to_the_row_number_kind() {
        let f = plan("SELECT row_number() OVER (ORDER BY id) FROM t");
        assert!(matches!(only_func(&f).kind, WindowFuncKind::RowNumber));
    }

    #[test]
    fn rank_family_maps_to_its_kinds() {
        assert!(matches!(only_func(&plan("SELECT rank() OVER (ORDER BY v) FROM t")).kind, WindowFuncKind::Rank));
        assert!(matches!(
            only_func(&plan("SELECT dense_rank() OVER (ORDER BY v) FROM t")).kind,
            WindowFuncKind::DenseRank
        ));
        assert!(matches!(
            only_func(&plan("SELECT percent_rank() OVER (ORDER BY v) FROM t")).kind,
            WindowFuncKind::PercentRank
        ));
        assert!(matches!(
            only_func(&plan("SELECT cume_dist() OVER (ORDER BY v) FROM t")).kind,
            WindowFuncKind::CumeDist
        ));
    }

    #[test]
    fn ranking_functions_reject_arguments() {
        assert!(plan_err("SELECT row_number(v) OVER () FROM t").contains("wrong number of arguments"));
        assert!(plan_err("SELECT rank(v) OVER () FROM t").contains("wrong number of arguments"));
        // percent_rank / cume_dist are also zero-arg; an argument is a wrong-arg-count error.
        assert!(plan_err("SELECT percent_rank(v) OVER () FROM t").contains("wrong number of arguments"));
        assert!(plan_err("SELECT cume_dist(v) OVER () FROM t").contains("wrong number of arguments"));
    }

    #[test]
    fn ntile_binds_its_bucket_count_argument() {
        match &only_func(&plan("SELECT ntile(4) OVER (ORDER BY id) FROM t")).kind {
            WindowFuncKind::Ntile(EvalExpr::Literal(Value::Integer(4))) => {}
            other => panic!("expected Ntile(4), got {other:?}"),
        }
        assert!(plan_err("SELECT ntile() OVER () FROM t").contains("wrong number of arguments"));
    }

    // -- offset functions --------------------------------------------------------

    #[test]
    fn lag_defaults_offset_to_one_and_default_to_null() {
        match &only_func(&plan("SELECT lag(v) OVER (ORDER BY id) FROM t")).kind {
            WindowFuncKind::Lag { expr, offset, default } => {
                assert!(matches!(expr, EvalExpr::Column(2)), "lag(v): v = Column(2)");
                assert!(matches!(offset, EvalExpr::Literal(Value::Integer(1))), "default offset 1");
                assert!(matches!(default, EvalExpr::Literal(Value::Null)), "default default NULL");
            }
            other => panic!("expected Lag, got {other:?}"),
        }
    }

    #[test]
    fn lead_carries_explicit_offset_and_default() {
        match &only_func(&plan("SELECT lead(v, 2, 0) OVER (ORDER BY id) FROM t")).kind {
            WindowFuncKind::Lead { expr, offset, default } => {
                assert!(matches!(expr, EvalExpr::Column(2)));
                assert!(matches!(offset, EvalExpr::Literal(Value::Integer(2))));
                assert!(matches!(default, EvalExpr::Literal(Value::Integer(0))));
            }
            other => panic!("expected Lead, got {other:?}"),
        }
        assert!(plan_err("SELECT lead(v, 1, 2, 3) OVER () FROM t").contains("wrong number of arguments"));
    }

    // -- value functions ---------------------------------------------------------

    #[test]
    fn first_last_value_bind_their_argument() {
        assert!(matches!(
            only_func(&plan("SELECT first_value(v) OVER (ORDER BY id) FROM t")).kind,
            WindowFuncKind::FirstValue(EvalExpr::Column(2))
        ));
        assert!(matches!(
            only_func(&plan("SELECT last_value(v) OVER (ORDER BY id) FROM t")).kind,
            WindowFuncKind::LastValue(EvalExpr::Column(2))
        ));
    }

    #[test]
    fn nth_value_binds_expr_and_n() {
        match &only_func(&plan("SELECT nth_value(v, 3) OVER (ORDER BY id) FROM t")).kind {
            WindowFuncKind::NthValue { expr, n } => {
                assert!(matches!(expr, EvalExpr::Column(2)));
                assert!(matches!(n, EvalExpr::Literal(Value::Integer(3))));
            }
            other => panic!("expected NthValue, got {other:?}"),
        }
        assert!(plan_err("SELECT nth_value(v) OVER () FROM t").contains("wrong number of arguments"));
    }

    // -- frames ------------------------------------------------------------------

    /// A window function whose plan carries the default frame.
    fn assert_default_frame(f: &WindowFunc) {
        assert_eq!(f.frame.units, FrameUnits::Range);
        assert!(matches!(f.frame.start, FrameBound::UnboundedPreceding));
        assert!(matches!(f.frame.end, FrameBound::CurrentRow));
        assert_eq!(f.frame.exclude, FrameExclude::NoOthers);
    }

    #[test]
    fn omitted_frame_is_the_default_frame() {
        assert_default_frame(only_func(&plan("SELECT sum(v) OVER (ORDER BY id) FROM t")));
    }

    #[test]
    fn explicit_rows_frame_binds_units_and_bounds() {
        let p = plan("SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t");
        let f = only_func(&p);
        assert_eq!(f.frame.units, FrameUnits::Rows);
        match &f.frame.start {
            FrameBound::Preceding(EvalExpr::Literal(Value::Integer(1))) => {}
            other => panic!("expected 1 PRECEDING, got {other:?}"),
        }
        match &f.frame.end {
            FrameBound::Following(EvalExpr::Literal(Value::Integer(1))) => {}
            other => panic!("expected 1 FOLLOWING, got {other:?}"),
        }
        assert_eq!(f.frame.exclude, FrameExclude::NoOthers);
    }

    #[test]
    fn omitted_frame_end_means_current_row() {
        // `ROWS <bound> PRECEDING` (no AND …) — the end bound defaults to CURRENT ROW.
        let p = plan("SELECT sum(v) OVER (ORDER BY id ROWS 2 PRECEDING) FROM t");
        let f = only_func(&p);
        assert_eq!(f.frame.units, FrameUnits::Rows);
        assert!(matches!(f.frame.end, FrameBound::CurrentRow), "omitted end => CURRENT ROW");
    }

    #[test]
    fn frame_exclude_is_carried() {
        let p = plan("SELECT sum(v) OVER (ORDER BY id GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE TIES) FROM t");
        let f = only_func(&p);
        assert_eq!(f.frame.units, FrameUnits::Groups);
        assert_eq!(f.frame.exclude, FrameExclude::Ties);
    }

    // -- frame boundary validation (windowfunctions.html §2.2) --------------------
    // The executor's frame engine documents these as planner-guaranteed invariants and
    // only tolerates a violation defensively, so an unvalidated bad frame would be a
    // silent wrong answer where real SQLite errors at prepare time.

    #[test]
    fn frame_end_before_start_is_rejected() {
        // §2.2.2: the ending boundary may not appear "higher" than the starting one.
        // `CURRENT ROW` (start) with `1 PRECEDING` (end) inverts that order.
        let msg = plan_err(
            "SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 1 PRECEDING) FROM t",
        );
        assert!(msg.contains("may not precede the starting boundary"), "got: {msg}");
    }

    #[test]
    fn frame_short_form_following_start_is_rejected() {
        // Short form `ROWS 1 FOLLOWING`: start `1 FOLLOWING`, end defaults to CURRENT ROW
        // — an inverted frame (FOLLOWING sits after CURRENT ROW in the ordering).
        let msg = plan_err("SELECT sum(v) OVER (ORDER BY id ROWS 1 FOLLOWING) FROM t");
        assert!(msg.contains("may not precede the starting boundary"), "got: {msg}");
    }

    #[test]
    fn frame_starting_with_unbounded_following_is_rejected() {
        let msg = plan_err(
            "SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED FOLLOWING AND UNBOUNDED FOLLOWING) FROM t",
        );
        assert!(msg.contains("may not start with UNBOUNDED FOLLOWING"), "got: {msg}");
    }

    #[test]
    fn frame_ending_with_unbounded_preceding_is_rejected() {
        let msg = plan_err(
            "SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED PRECEDING) FROM t",
        );
        assert!(msg.contains("may not end with UNBOUNDED PRECEDING"), "got: {msg}");
    }

    #[test]
    fn range_offset_frame_requires_exactly_one_order_by_term() {
        // §2.2.1: a RANGE band (an `<expr>` boundary) needs exactly one ORDER BY term.
        let two = plan_err(
            "SELECT sum(v) OVER (ORDER BY grp, id RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        );
        assert!(two.contains("exactly one ORDER BY"), "two terms: {two}");
        let zero =
            plan_err("SELECT sum(v) OVER (RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t");
        assert!(zero.contains("exactly one ORDER BY"), "zero terms: {zero}");
    }

    #[test]
    fn range_offset_frame_with_one_order_by_term_binds() {
        // The valid counterpart: exactly one ORDER BY term is accepted.
        let p =
            plan("SELECT sum(v) OVER (ORDER BY id RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t");
        let f = only_func(&p);
        assert_eq!(f.frame.units, FrameUnits::Range);
        assert!(matches!(f.frame.start, FrameBound::Preceding(_)));
    }

    #[test]
    fn range_non_offset_frame_allows_multiple_order_by_terms() {
        // A RANGE frame with only UNBOUNDED/CURRENT ROW boundaries has NO single-term
        // requirement, so multiple ORDER BY terms bind fine (the default frame shape made
        // explicit). Regression guard so the §2.2.1 rule never over-rejects.
        let p = plan(
            "SELECT sum(v) OVER (ORDER BY grp, id RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        );
        assert_eq!(only_func(&p).order_by.len(), 2);
    }

    // -- named windows and chaining ----------------------------------------------

    #[test]
    fn named_window_resolves_to_the_same_spec() {
        let p = plan("SELECT sum(v) OVER w FROM t WINDOW w AS (PARTITION BY grp ORDER BY id)");
        let f = only_func(&p);
        assert!(matches!(f.partition_by[0], EvalExpr::Column(1)));
        assert!(matches!(f.order_by[0].expr, EvalExpr::Column(0)));
    }

    #[test]
    fn window_chaining_inherits_partition_and_order_from_base() {
        // `w2 AS (w1)` inherits w1's PARTITION BY grp + ORDER BY id; the referencing
        // window adds a frame. (windowfunctions.html §4.)
        let p = plan(
            "SELECT sum(v) OVER w2 FROM t \
             WINDOW w1 AS (PARTITION BY grp ORDER BY id), \
                    w2 AS (w1 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
        );
        let f = only_func(&p);
        assert!(matches!(f.partition_by[0], EvalExpr::Column(1)), "PARTITION BY grp from base");
        assert!(matches!(f.order_by[0].expr, EvalExpr::Column(0)), "ORDER BY id from base");
        assert_eq!(f.frame.units, FrameUnits::Rows, "frame from the referencing window");
    }

    #[test]
    fn chaining_rejects_partition_by_on_the_referencing_window() {
        let msg = plan_err(
            "SELECT sum(v) OVER (w1 PARTITION BY id) FROM t WINDOW w1 AS (PARTITION BY grp)",
        );
        assert!(msg.contains("PARTITION BY"), "got: {msg}");
    }

    #[test]
    fn chaining_rejects_a_base_window_with_a_frame() {
        let msg = plan_err(
            "SELECT sum(v) OVER (w1 ORDER BY id) FROM t \
             WINDOW w1 AS (ORDER BY grp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
        );
        assert!(msg.contains("base window"), "got: {msg}");
    }

    #[test]
    fn chaining_inherits_referencing_window_order_by_when_base_has_none() {
        // §4, the other inheritance direction from
        // `window_chaining_inherits_partition_and_order_from_base`: when the BASE has no
        // ORDER BY, the referencing window may supply one, and the resolved spec carries
        // the base's PARTITION BY over that referencing ORDER BY.
        let p = plan("SELECT sum(v) OVER (w1 ORDER BY id) FROM t WINDOW w1 AS (PARTITION BY grp)");
        let f = only_func(&p);
        assert!(matches!(f.partition_by[0], EvalExpr::Column(1)), "PARTITION BY grp from base");
        assert_eq!(f.order_by.len(), 1);
        assert!(
            matches!(f.order_by[0].expr, EvalExpr::Column(0)),
            "ORDER BY id from the referencing window"
        );
    }

    #[test]
    fn chaining_rejects_order_by_when_base_has_order_by() {
        // §4: a referencing window may not add its own ORDER BY when the base already
        // defines one (the two orderings would conflict).
        let msg =
            plan_err("SELECT sum(v) OVER (w1 ORDER BY id) FROM t WINDOW w1 AS (ORDER BY grp)");
        assert!(msg.contains("ORDER BY"), "got: {msg}");
    }

    #[test]
    fn chaining_circular_base_reference_terminates_with_a_loud_error() {
        // A base chain that loops (w1 -> w2 -> w1) must be caught by the `visited` cycle
        // guard and reported loudly, never recurse without bound (a hang is a wrong
        // answer). Pins that resolution TERMINATES on a cycle.
        let msg = plan_err("SELECT sum(v) OVER w1 FROM t WINDOW w1 AS (w2), w2 AS (w1)");
        assert!(msg.contains("circular reference"), "got: {msg}");
    }

    #[test]
    fn unknown_named_window_is_a_loud_error() {
        assert!(plan_err("SELECT sum(v) OVER nope FROM t").contains("no such window: nope"));
    }

    // -- result column naming ----------------------------------------------------

    #[test]
    fn window_result_column_takes_its_as_alias() {
        // PART B step 3: an `AS alias` names the window result column. Verified at the
        // plan level (`result_columns`) so it holds independent of whether the executor
        // can yet compute the function.
        let p = plan("SELECT row_number() OVER (ORDER BY id) AS rn FROM t");
        assert_eq!(p.result_columns, vec!["rn".to_string()]);
    }

    // -- misuse / loud errors ----------------------------------------------------

    #[test]
    fn distinct_window_aggregate_is_rejected() {
        assert!(plan_err("SELECT count(DISTINCT v) OVER () FROM t")
            .contains("DISTINCT is not supported for window function"));
    }

    #[test]
    fn filter_on_a_ranking_function_is_rejected() {
        let msg = plan_err("SELECT row_number() FILTER (WHERE v > 0) OVER () FROM t");
        assert!(msg.contains("FILTER"), "got: {msg}");
    }

    #[test]
    fn scalar_function_used_as_a_window_function_is_a_loud_misuse() {
        // A KNOWN scalar function (`length`) used with OVER is not a window function:
        // SQLite reports "<name>() may not be used as a window function", NOT the "no
        // such function" reserved for an entirely unknown name (a known function plainly
        // exists). The unknown name still surfaces the registry's "no such function".
        let known = plan_err("SELECT length(v) OVER () FROM t");
        assert!(known.contains("may not be used as a window function"), "got: {known}");
        let unknown = plan_err("SELECT nope(v) OVER () FROM t");
        assert!(unknown.contains("no such function: nope"), "got: {unknown}");
    }

    #[test]
    fn window_function_in_where_is_a_loud_misuse() {
        assert!(plan_err("SELECT id FROM t WHERE sum(v) OVER () > id")
            .contains("misuse of window function"));
    }

    #[test]
    fn window_function_in_group_by_is_a_loud_misuse() {
        // A window call in GROUP BY must be a loud error like WHERE/HAVING, never a
        // collected WindowFunc resolving to an out-of-range group-key register.
        assert!(plan_err("SELECT 1 FROM t GROUP BY sum(v) OVER ()")
            .contains("misuse of window function"));
    }

    #[test]
    fn partition_by_collation_is_resolved_onto_the_plan() {
        // PARTITION BY carries each key's §7.1 collation on `partition_collations` (like
        // GROUP BY's `group_collations`), so the window operator keys partitions under it.
        // An explicit postfix COLLATE is honored (§7.1 rule 1)…
        let p = plan("SELECT sum(v) OVER (PARTITION BY v COLLATE NOCASE) FROM t");
        assert_eq!(only_func(&p).partition_collations, vec![Collation::NoCase]);
        // …and a plain BINARY column keys under BINARY.
        let p = plan("SELECT sum(v) OVER (PARTITION BY v) FROM t");
        assert_eq!(only_func(&p).partition_collations, vec![Collation::Binary]);
        // The collations vector aligns 1:1 with the partition_by keys.
        let p = plan("SELECT sum(v) OVER (PARTITION BY grp, v) FROM t");
        let f = only_func(&p);
        assert_eq!(f.partition_collations.len(), f.partition_by.len());
    }
}
