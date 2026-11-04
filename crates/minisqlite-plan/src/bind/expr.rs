//! Expression lowering: [`bind_expr`] turns a SQL AST [`Expr`] into the compiled
//! [`EvalExpr`] the executor evaluates, resolving column names to registers,
//! bind parameters to numbers, function names to resolved handles, and — the #1
//! correctness lever — comparison affinity + collation into a [`CompareMeta`]
//! ([`compare_meta`], datatype3.html §3.2/§4.2/§7.1).
//!
//! Aggregate calls are NOT lowered as ordinary expressions: inside an aggregate
//! query the [`Grouping`] context (installed on the [`Scope`]) redirects a group
//! key to its key register and collects each aggregate call, returning a reference
//! to its result register. Everything else lowers structurally.

use minisqlite_expr::{
    AggregateCall, ArithOp, BitOp, CaseWhen, CmpOp, CompareMeta, EvalExpr, LikeKind as IrLikeKind,
    NowKind, RaiseKind as IrRaiseKind, SortKey, UnaryOp as IrUnaryOp,
};
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::{
    BinaryOp, Distinct, Expr, FunctionArgs, InRhs, JoinTree, LikeKind as SqlLikeKind, Literal,
    NullsOrder, OrderingTerm, OverClause, QualifiedName, RaiseAction, ResultColumn, Select,
    SelectBody, SelectCore, SortOrder, TableOrSubquery, UnaryOp as SqlUnaryOp,
};
use minisqlite_types::{affinity_of_declared_type, Affinity, Error, Result, Value};

use crate::bind::scope::{parse_collation, Grouping, Scope};
use crate::compile::select::{compile_row_subquery, compile_subquery, compile_value_subquery};
use crate::compile::window::bind_window_function;
use crate::plan_ctx::PlanCtx;

/// Lower a scalar SQL expression to an [`EvalExpr`] against `scope`.
///
/// When `scope` carries a [`Grouping`] (an aggregate query's projection/HAVING),
/// a whole group key resolves to its key register, an aggregate call is collected
/// and replaced by its result register, and a bare column that is neither is a
/// loud "must appear in GROUP BY" error.
pub fn bind_expr(scope: &Scope, ctx: &mut PlanCtx, e: &Expr) -> Result<EvalExpr> {
    if let Some(g) = scope.grouping {
        if let Some(bound) = bind_grouped_node(scope, g, ctx, e)? {
            return Ok(bound);
        }
    }
    bind_expr_normal(scope, ctx, e)
}

/// The grouping-context dispatch: match a group key, collect an aggregate, or
/// redirect/reject a column. Returns `None` for anything else so ordinary
/// structural binding recurses (still carrying the grouping context).
fn bind_grouped_node(
    scope: &Scope,
    g: &Grouping,
    ctx: &mut PlanCtx,
    e: &Expr,
) -> Result<Option<EvalExpr>> {
    // 1. A sub-expression that IS a whole GROUP BY key reads that key register.
    for (i, key) in g.key_asts.iter().enumerate() {
        if e == key {
            return Ok(Some(EvalExpr::Column(i)));
        }
    }
    // 2. An aggregate call is collected; its result lands after the group keys.
    if let Expr::Function { name, distinct, args, filter, over, order_by } = e {
        if function_is_aggregate(ctx.registry, name, args, over.is_some()) {
            return Ok(Some(bind_aggregate(scope, g, ctx, name, *distinct, args, filter, order_by)?));
        }
    }
    // 3. A bare column normally must be a group key (matched by resolved register).
    if let Expr::Column { table, name, from_dqs, .. } = e {
        let rc = match scope.without_grouping().try_resolve_column(table.as_deref(), name) {
            Ok(Some(rc)) => rc,
            // DQS: a bare double-quoted token that names NO column is not a grouped column
            // at all — bail to ordinary binding, which applies the string-literal fallback
            // (bind_expr_normal), rather than raising the wrong "must appear in GROUP BY"
            // error. An ambiguous match is a real column reference and stays the loud
            // error (`Err(e)` below), never a silent string literal.
            Ok(None) if *from_dqs => return Ok(None),
            Ok(None) => return Err(Error::sql(format!("no such column: {name}"))),
            Err(e) => return Err(e),
        };
        if let Some(&gk) = g.col_to_group.get(&rc.reg) {
            return Ok(Some(EvalExpr::Column(gk)));
        }
        // A §2.5 bare column: it takes its value from some input row of the group — the
        // extremum row in the single-min/max special case, an arbitrary (first) row in the
        // general case. Binding is identical for both (capture the register into a slot the
        // operator fills); WHICH row the operator captures from is decided by the plan
        // marker (`minmax_bare` vs `bare_arbitrary`), not here. Capture ONLY a genuine
        // column of the aggregated relation — one whose register falls within a local FROM
        // source. An OUTER/correlated reference (resolved through the parent chain, e.g. the
        // `t.id` in `SELECT (SELECT max(x)+t.id FROM u) FROM t`) is NOT a §2.5 bare column:
        // it is a correlated-across-aggregate shape this engine does not support, and it
        // must stay the loud ungrouped-column error below — capturing it would read the
        // wrong (outer) register from the aggregated input row.
        if let Some(cap) = &g.bare_capture {
            // A NATURAL/USING coalesced column takes its VALUE from the fold over ALL its
            // component copies (`COALESCE`), not from the left copy alone — which is
            // NULL-extended on an outer-join extremum row (the same shape the plain
            // projection path fixes via `Scope::coalesced_value`). So capture EACH
            // component register from the extremum row and fold over the captured slots,
            // keeping the §2.5 capture in agreement with the projection value. Coalescing
            // applies only to an UNqualified reference; every component register is a local
            // join column of this FROM by construction, so no `is_local` exclusion applies.
            if table.is_none() {
                if let Some(regs) = scope.coalesced_regs(name) {
                    let items = regs
                        .into_iter()
                        .map(|reg| EvalExpr::Column(cap.base + cap.intern(reg)))
                        .collect();
                    return Ok(Some(EvalExpr::Coalesce(items)));
                }
            }
            // The register-range test is COLLISION-SAFE here (unlike `Scope::remap_post_aggregate`,
            // which decides locality by NAME): `rc` was resolved against THIS SAME scope
            // (`scope.without_grouping()`, local-first), so a local column's register is provably
            // inside a local FROM source `[base_offset, ..)` while a correlated/parent reference's
            // register is provably `< base_offset` (the outer prefix, exactly where local sources
            // begin) — disjoint, so no grandparent register can numerically overlap a local one.
            // (`remap_post_aggregate` range-checked a register resolved against the PARENT — a
            // different 0-based space that CAN overlap — which is why it must use the name check.)
            // If a future edit ever resolves `rc` against a different scope here, replace this
            // with a NAME-based locality check (as `Scope::remap_post_aggregate` does) to stay safe.
            let is_local =
                scope.sources.iter().any(|s| rc.reg >= s.base() && rc.reg < s.base() + s.width());
            if is_local {
                let slot = cap.intern(rc.reg);
                return Ok(Some(EvalExpr::Column(cap.base + slot)));
            }
        }
        // Otherwise it is an ungrouped reference — a loud error, never a silent wrong
        // register.
        return Err(Error::sql(format!(
            "column \"{name}\" must appear in the GROUP BY clause or be used in an aggregate function"
        )));
    }
    Ok(None)
}

/// Build and register one aggregate call, returning a reference to its result
/// register (`num_keys + index`). Its arguments and FILTER bind against the
/// pre-aggregate input row (grouping turned off).
fn bind_aggregate(
    scope: &Scope,
    g: &Grouping,
    ctx: &mut PlanCtx,
    name: &str,
    distinct: bool,
    args: &FunctionArgs,
    filter: &Option<Box<Expr>>,
    order_by: &[OrderingTerm],
) -> Result<EvalExpr> {
    let inner = scope.without_grouping();
    let (bound_args, argc) = match args {
        // `count(*)` and `f()` are argc 0; a NULL/star argument list carries none.
        FunctionArgs::Star | FunctionArgs::Empty => (Vec::new(), 0usize),
        FunctionArgs::List(list) => {
            let mut v = Vec::with_capacity(list.len());
            for it in list {
                v.push(bind_expr(&inner, ctx, it)?);
            }
            (v, list.len())
        }
    };
    let bound_filter = match filter {
        Some(fe) => Some(bind_expr(&inner, ctx, fe)?),
        None => None,
    };
    // Aggregate ORDER BY (`group_concat(x ORDER BY y)`, lang_aggfunc.html): the operator
    // orders the folded values by these keys. They bind against the SAME pre-aggregate
    // input row as the arguments (grouping off), so an ordering key may reference any input
    // column, not just the aggregated argument. `count(*)` / `f()` carry no argument list
    // and so no `ORDER BY` reaches here.
    let bound_order = bind_aggregate_order(&inner, ctx, order_by)?;
    // The registry may report "no such function" (the aggregate family is a
    // sibling); that honest error propagates rather than being papered over.
    let func = ctx.registry.resolve_aggregate(name, argc)?;
    // Per-argument collation for DISTINCT dedup, resolved against the SAME pre-aggregate
    // scope (`inner`) the arguments bound in — an explicit COLLATE / column collation /
    // BINARY per argument (§7.1). Populated for every call (harmless when not DISTINCT).
    let arg_collations = arg_list_collations(&inner, args);
    let call = AggregateCall {
        func,
        distinct,
        args: bound_args,
        filter: bound_filter,
        order_by: bound_order,
        arg_collations,
    };
    let mut aggs = g.aggregates.borrow_mut();
    let idx = aggs.len();
    // INVARIANT (single-min/max §2.5): every aggregate is pushed here in source-visit
    // order with NO dedup, so this collected order/count must match `single_minmax_special`
    // /`walk_agg_calls` (compile/aggregate.rs), which computes the bare-column capture
    // offset `num_keys + num_aggregates` and `agg_index` before binding. Teaching this to
    // dedup identical calls, or reordering the collection, would silently shift that offset
    // in release builds — keep the two traversals in step (backstopped by a debug assert).
    aggs.push(call);
    Ok(EvalExpr::Column(g.num_keys + idx))
}

/// Lower an aggregate's in-argument `ORDER BY` terms to [`SortKey`]s over the
/// pre-aggregate input row (`inner` = the grouping-off argument scope). Mirrors the
/// query / window `ORDER BY` lowering: an explicit postfix `COLLATE` wins, else the
/// operand's column collation, else BINARY; the default NULL placement stays implicit
/// (`None`) so the operator applies SQLite's ASC-low / DESC-high default.
fn bind_aggregate_order(
    inner: &Scope,
    ctx: &mut PlanCtx,
    order_by: &[OrderingTerm],
) -> Result<Vec<SortKey>> {
    let mut keys = Vec::with_capacity(order_by.len());
    for term in order_by {
        let expr = bind_expr(inner, ctx, &term.expr)?;
        let collation = match &term.collation {
            Some(name) => parse_collation(name)?,
            None => best_effort_collation(inner, &term.expr),
        };
        keys.push(SortKey {
            expr,
            desc: matches!(term.order, Some(SortOrder::Desc)),
            nulls_first: term.nulls.map(|n| matches!(n, NullsOrder::First)),
            collation,
        });
    }
    Ok(keys)
}

/// Structural lowering of every non-grouping-special expression form.
fn bind_expr_normal(scope: &Scope, ctx: &mut PlanCtx, e: &Expr) -> Result<EvalExpr> {
    let out = match e {
        Expr::Literal(lit) => {
            let ev = bind_literal(lit)?;
            // CURRENT_DATE/TIME/TIMESTAMP read the wall clock per evaluation, so a subquery
            // containing one is non-deterministic (unsafe to memoize). Every other literal
            // is a constant.
            if matches!(ev, EvalExpr::Now(_)) {
                scope.note_nondeterministic();
            }
            ev
        }

        // A NATURAL/USING coalesced column resolves to `COALESCE(left, right)` (one
        // value, unambiguous); every other column to its single register. Routed through
        // `resolve_column_expr` so the coalescing is applied uniformly to every bare
        // column reference in an expression (a qualified reference names one side and is
        // never coalesced).
        Expr::Column { schema: _, table, name, from_dqs } => {
            match scope.try_resolve_column_expr(table.as_deref(), name) {
                Ok(Some(ev)) => ev,
                // SQLite DQS legacy: a bare double-quoted token that resolves to NO column
                // is a string literal (quirks.html §8, library default on). Only a genuine
                // not-found falls back — a name that IS a column but ambiguous still errors
                // (real sqlite does not fall back on an ambiguous match), and only a bare
                // `"name"` carries `from_dqs`, so a qualified reference never falls back.
                Ok(None) if *from_dqs => EvalExpr::Literal(Value::Text(name.clone())),
                Ok(None) => return Err(Error::sql(format!("no such column: {name}"))),
                Err(e) => return Err(e),
            }
        }

        Expr::BindParam(p) => EvalExpr::Param(ctx.param_index(p)?),

        Expr::Unary { op, expr } => EvalExpr::Unary {
            op: unary_op(*op),
            operand: Box::new(bind_expr(scope, ctx, expr)?),
        },

        Expr::Binary { op, left, right } => bind_binary(scope, ctx, *op, left, right)?,

        Expr::Cast { expr, type_name } => EvalExpr::Cast {
            affinity: affinity_of_declared_type(Some(type_name)),
            operand: Box::new(bind_expr(scope, ctx, expr)?),
        },

        Expr::Collate { expr, collation } => EvalExpr::Collate {
            collation: parse_collation(collation)?,
            operand: Box::new(bind_expr(scope, ctx, expr)?),
        },

        Expr::Like { negated, kind, lhs, rhs, escape } => {
            let irkind = match kind {
                SqlLikeKind::Like => IrLikeKind::Like,
                SqlLikeKind::Glob => IrLikeKind::Glob,
                SqlLikeKind::Regexp | SqlLikeKind::Match => {
                    return Err(Error::sql("REGEXP/MATCH operators are not supported"));
                }
            };
            // Bind in source order (lhs, rhs, escape) so `?` params number correctly.
            let subject = bind_expr(scope, ctx, lhs)?;
            let pattern = bind_expr(scope, ctx, rhs)?;
            let bescape = match escape {
                Some(esc) => Some(Box::new(bind_expr(scope, ctx, esc)?)),
                None => None,
            };
            EvalExpr::Like {
                negated: *negated,
                kind: irkind,
                subject: Box::new(subject),
                pattern: Box::new(pattern),
                escape: bescape,
            }
        }

        Expr::Between { negated, expr, low, high } => {
            // A row-value BETWEEN (rowvalue.html §3.2, `(year,month,day) BETWEEN … AND …`)
            // lowers to row comparisons; a scalar BETWEEN takes the two-independent-
            // comparisons path below (datatype3 §4.2: subject vs low, subject vs high,
            // each with its own metadata).
            if row_value_elems(expr).is_some()
                || row_value_elems(low).is_some()
                || row_value_elems(high).is_some()
            {
                bind_row_between(scope, ctx, *negated, expr, low, high)?
            } else {
                let low_meta = compare_meta(scope, expr, low)?;
                let high_meta = compare_meta(scope, expr, high)?;
                let subject = bind_expr(scope, ctx, expr)?;
                let blow = bind_expr(scope, ctx, low)?;
                let bhigh = bind_expr(scope, ctx, high)?;
                EvalExpr::Between {
                    negated: *negated,
                    subject: Box::new(subject),
                    low: Box::new(blow),
                    high: Box::new(bhigh),
                    low_meta,
                    high_meta,
                }
            }
        }

        Expr::In { negated, expr, rhs } => bind_in(scope, ctx, *negated, expr, rhs)?,

        Expr::Exists { negated, select } => {
            let id = compile_subquery(scope, ctx, select)?;
            EvalExpr::Exists { negated: *negated, id }
        }

        Expr::Subquery(select) => {
            let id = compile_value_subquery(scope, ctx, select)?;
            EvalExpr::ScalarSubquery(id)
        }

        Expr::Case { operand, whens, else_expr } => bind_case(scope, ctx, operand, whens, else_expr)?,

        Expr::IsNull(x) => EvalExpr::IsNull(Box::new(bind_expr(scope, ctx, x)?)),
        Expr::NotNull(x) => EvalExpr::NotNull(Box::new(bind_expr(scope, ctx, x)?)),

        Expr::Parenthesized(list) => {
            if list.len() == 1 {
                bind_expr(scope, ctx, &list[0])?
            } else {
                return Err(Error::sql("row value misused"));
            }
        }

        // RAISE(...) — lowered to `EvalExpr::Raise` for the firing executor to evaluate
        // (lang_createtrigger.html §6). The grammar only accepts RAISE inside a trigger
        // body, so binding it here is reached only when compiling a trigger action/WHEN.
        // ABORT/FAIL/ROLLBACK carry their message; IGNORE has none.
        Expr::Raise(action) => match action {
            RaiseAction::Ignore => EvalExpr::Raise { kind: IrRaiseKind::Ignore, message: None },
            RaiseAction::Abort(m) => {
                EvalExpr::Raise { kind: IrRaiseKind::Abort, message: Some(m.clone()) }
            }
            RaiseAction::Fail(m) => {
                EvalExpr::Raise { kind: IrRaiseKind::Fail, message: Some(m.clone()) }
            }
            RaiseAction::Rollback(m) => {
                EvalExpr::Raise { kind: IrRaiseKind::Rollback, message: Some(m.clone()) }
            }
        },

        Expr::Function { name, distinct, args, filter, over, order_by } => {
            bind_function(scope, ctx, name, *distinct, args, filter, over, order_by)?
        }
    };
    Ok(out)
}

/// Lower a literal to its runtime [`Value`] (or a `CURRENT_*` marker).
fn bind_literal(lit: &Literal) -> Result<EvalExpr> {
    Ok(match lit {
        Literal::Null => EvalExpr::Literal(Value::Null),
        Literal::Integer(i) => EvalExpr::Literal(Value::Integer(*i)),
        Literal::Real(r) => EvalExpr::Literal(Value::Real(*r)),
        Literal::Text(s) => EvalExpr::Literal(Value::Text(s.clone())),
        Literal::Blob(b) => EvalExpr::Literal(Value::Blob(b.clone())),
        Literal::True => EvalExpr::Literal(Value::Integer(1)),
        Literal::False => EvalExpr::Literal(Value::Integer(0)),
        Literal::CurrentDate => EvalExpr::Now(NowKind::Date),
        Literal::CurrentTime => EvalExpr::Now(NowKind::Time),
        Literal::CurrentTimestamp => EvalExpr::Now(NowKind::Timestamp),
    })
}

/// Lower a binary operator, computing comparison metadata from the AST operands
/// before they are consumed by binding.
fn bind_binary(
    scope: &Scope,
    ctx: &mut PlanCtx,
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
) -> Result<EvalExpr> {
    use BinaryOp::*;
    // Row-value comparisons (rowvalue.html §2.1): a comparison operator with a
    // parenthesized `(a, b, …)` row value on either side lowers to a three-valued
    // boolean tree of per-element scalar comparisons. This must run BEFORE the
    // scalar arms bind the operands, because binding a multi-element `Parenthesized`
    // on its own is the "row value misused" error — a row value is legal only here,
    // as a comparison operand, never as a scalar value.
    if is_row_comparison_op(op)
        && (row_value_elems(left).is_some() || row_value_elems(right).is_some())
    {
        return bind_row_comparison(scope, ctx, op, left, right);
    }
    Ok(match op {
        Add | Sub | Mul | Div | Mod => {
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Arith { op: arith_op(op), left: Box::new(l), right: Box::new(r) }
        }
        Concat => {
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Concat { left: Box::new(l), right: Box::new(r) }
        }
        JsonArrow | JsonArrow2 => {
            // The JSON `->` / `->>` operators are implemented as scalar functions
            // registered under the sentinel names "->" / "->>" (not valid SQL
            // identifiers, so no user function collides). Bind both operands and lower
            // to a two-argument call resolved exactly like any other scalar, so the
            // operator flows through the same evaluation path (json1.html §4.10).
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            let name = if op == JsonArrow { "->" } else { "->>" };
            let func = ctx.registry.resolve_scalar(name, 2)?;
            if !func.deterministic() {
                scope.note_nondeterministic();
            }
            EvalExpr::Func { func, args: vec![l, r] }
        }
        BitAnd | BitOr | LShift | RShift => {
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Bitwise { op: bit_op(op), left: Box::new(l), right: Box::new(r) }
        }
        Lt | Le | Gt | Ge | Eq | Ne => {
            let meta = compare_meta(scope, left, right)?;
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Compare { op: cmp_op(op), null_safe: false, left: Box::new(l), right: Box::new(r), meta }
        }
        Is => {
            let meta = compare_meta(scope, left, right)?;
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Compare { op: CmpOp::Eq, null_safe: true, left: Box::new(l), right: Box::new(r), meta }
        }
        IsNot => {
            let meta = compare_meta(scope, left, right)?;
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Compare { op: CmpOp::Ne, null_safe: true, left: Box::new(l), right: Box::new(r), meta }
        }
        And => {
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::And(Box::new(l), Box::new(r))
        }
        Or => {
            let l = bind_expr(scope, ctx, left)?;
            let r = bind_expr(scope, ctx, right)?;
            EvalExpr::Or(Box::new(l), Box::new(r))
        }
    })
}

// ---------------------------------------------------------------------------
// Row-value comparisons (rowvalue.html §2.1)
// ---------------------------------------------------------------------------

/// Whether `op` is a comparison operator that may take row-value operands
/// (`rowvalue.html` §2: `< <= > >= = <> IS IS NOT`). `BETWEEN`/`IN`/`CASE` are
/// separate AST forms handled by their own binders.
fn is_row_comparison_op(op: BinaryOp) -> bool {
    use BinaryOp::*;
    matches!(op, Lt | Le | Gt | Ge | Eq | Ne | Is | IsNot)
}

/// The element list of a row-value operand: a parenthesized list of two or more
/// scalars (`rowvalue.html` §1). A single-element `(x)` is a grouped scalar, not a
/// row value, so it returns `None` and binds through the ordinary scalar path.
fn row_value_elems(e: &Expr) -> Option<&[Expr]> {
    match e {
        Expr::Parenthesized(list) if list.len() > 1 => Some(list),
        _ => None,
    }
}

/// Lower a comparison with a row-value operand into a three-valued boolean tree of
/// per-element scalar comparisons (`rowvalue.html` §2.1). The `And`/`Or` nodes
/// supply SQL's 3VL for free, so the rule "a row comparison is NULL only when a
/// constituent NULL could still swing the result" falls out of the element
/// comparisons and short-circuiting connectives rather than needing its own logic.
///
/// Only two explicit, same-size row values are lowered here. A row value against a
/// scalar or a subquery, or a size mismatch, is the "row value misused" error:
/// `rowvalue.html` §2 requires two row values of equal size, and a row-value /
/// subquery comparison probe (`(a,b) = (SELECT …)`) is deferred to the executor and
/// keeps erroring here. (Row-value `IN` is routed separately through [`bind_in`].)
fn bind_row_comparison(
    scope: &Scope,
    ctx: &mut PlanCtx,
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
) -> Result<EvalExpr> {
    let (le, re) = match (row_value_elems(left), row_value_elems(right)) {
        (Some(l), Some(r)) => (l, r),
        _ => return Err(Error::sql("row value misused")),
    };
    if le.len() != re.len() {
        return Err(Error::sql("row value misused"));
    }
    let n = le.len();
    // Per-element comparison metadata: element `i` keeps its OWN affinity + collation
    // (never one shared meta), which is exactly what makes per-element NULL 3VL come
    // out right. `compare_meta` reads only the AST and assigns no bind parameters, so
    // computing it before binding does not disturb `?` numbering.
    let mut metas = Vec::with_capacity(n);
    for i in 0..n {
        metas.push(compare_meta(scope, &le[i], &re[i])?);
    }
    // Bind every element in source order — all LHS elements, then all RHS — so an
    // anonymous `?` inside a row value receives the same 1-based number SQLite
    // assigns it by lexical position (`param_index` advances on each bind).
    let mut lhs = Vec::with_capacity(n);
    for e in le {
        lhs.push(bind_expr(scope, ctx, e)?);
    }
    let mut rhs = Vec::with_capacity(n);
    for e in re {
        rhs.push(bind_expr(scope, ctx, e)?);
    }
    Ok(assemble_row_comparison(op, &lhs, &rhs, &metas))
}

/// Assemble the 3VL boolean tree for a row-value comparison from the bound element
/// expressions and their per-element metadata. `op` is guaranteed a comparison
/// operator: the only caller reaches here through [`is_row_comparison_op`].
fn assemble_row_comparison(
    op: BinaryOp,
    lhs: &[EvalExpr],
    rhs: &[EvalExpr],
    metas: &[CompareMeta],
) -> EvalExpr {
    use BinaryOp::*;
    match op {
        // `(a…) = (b…)`  ->  a1=b1 AND … AND an=bn
        // `(a…) <> (b…)` ->  a1<>b1 OR … OR an<>bn
        Eq => row_reduce(false, CmpOp::Eq, false, lhs, rhs, metas),
        Ne => row_reduce(true, CmpOp::Ne, false, lhs, rhs, metas),
        // IS / IS NOT are the null-safe analogues; each element comparison never
        // yields NULL, so neither does their AND/OR fold.
        Is => row_reduce(false, CmpOp::Eq, true, lhs, rhs, metas),
        IsNot => row_reduce(true, CmpOp::Ne, true, lhs, rhs, metas),
        // Ordering is lexicographic: interior levels use the STRICT operator (`<`
        // or `>`) plus `=`, and the final element uses the operator itself so that
        // `<=`/`>=` stay reflexive on equal rows.
        Lt => row_lex(CmpOp::Lt, CmpOp::Lt, lhs, rhs, metas, 0),
        Le => row_lex(CmpOp::Lt, CmpOp::Le, lhs, rhs, metas, 0),
        Gt => row_lex(CmpOp::Gt, CmpOp::Gt, lhs, rhs, metas, 0),
        Ge => row_lex(CmpOp::Gt, CmpOp::Ge, lhs, rhs, metas, 0),
        _ => unreachable!("assemble_row_comparison reached with a non-comparison operator"),
    }
}

/// Fold the per-element comparisons `lhs[i] <cmp> rhs[i]` with a single connective:
/// `AND` when `disjoin` is false (`=` / `IS`), `OR` when true (`<>` / `IS NOT`).
/// `null_safe` selects `IS`/`IS NOT` element semantics.
fn row_reduce(
    disjoin: bool,
    cmp: CmpOp,
    null_safe: bool,
    lhs: &[EvalExpr],
    rhs: &[EvalExpr],
    metas: &[CompareMeta],
) -> EvalExpr {
    let mut acc = elem_compare(cmp, null_safe, lhs, rhs, metas, 0);
    for i in 1..lhs.len() {
        let next = elem_compare(cmp, null_safe, lhs, rhs, metas, i);
        acc = if disjoin {
            EvalExpr::Or(Box::new(acc), Box::new(next))
        } else {
            EvalExpr::And(Box::new(acc), Box::new(next))
        };
    }
    acc
}

/// Build the lexicographic ordering tree from element `i` onward:
///   `a_i <strict> b_i  OR  (a_i = b_i  AND  <recurse from i+1>)`
/// with the FINAL element using `last` (the actual operator, e.g. `<=`) rather than
/// the strict one, so equal rows satisfy `<=`/`>=` but not `<`/`>`. `strict` is `<`
/// for `< / <=` and `>` for `> / >=`. Terminates because `i` strictly increases and
/// a row value always has at least two elements.
fn row_lex(
    strict: CmpOp,
    last: CmpOp,
    lhs: &[EvalExpr],
    rhs: &[EvalExpr],
    metas: &[CompareMeta],
    i: usize,
) -> EvalExpr {
    if i + 1 == lhs.len() {
        return elem_compare(last, false, lhs, rhs, metas, i);
    }
    let strict_here = elem_compare(strict, false, lhs, rhs, metas, i);
    let eq_here = elem_compare(CmpOp::Eq, false, lhs, rhs, metas, i);
    let tail = row_lex(strict, last, lhs, rhs, metas, i + 1);
    EvalExpr::Or(
        Box::new(strict_here),
        Box::new(EvalExpr::And(Box::new(eq_here), Box::new(tail))),
    )
}

/// One per-element scalar comparison `lhs[i] <cmp> rhs[i]` carrying element `i`'s own
/// metadata. The bound operands are cloned because a lexicographic level references
/// the same element in both a strict (`<`/`>`) and an equality comparison; this is
/// bind-time work over small row values, not a per-row cost.
fn elem_compare(
    cmp: CmpOp,
    null_safe: bool,
    lhs: &[EvalExpr],
    rhs: &[EvalExpr],
    metas: &[CompareMeta],
    i: usize,
) -> EvalExpr {
    EvalExpr::Compare {
        op: cmp,
        null_safe,
        left: Box::new(lhs[i].clone()),
        right: Box::new(rhs[i].clone()),
        meta: metas[i],
    }
}

/// Lower `(s…) [NOT] BETWEEN (lo…) AND (hi…)` (rowvalue.html §2 / §3.2, the worked
/// `(year,month,day) BETWEEN (…) AND (…)` example) into row-value comparisons.
/// `x BETWEEN lo AND hi` ≡ `x>=lo AND x<=hi`; the negation is `x<lo OR x>hi` — the
/// same three-valued result the scalar BETWEEN's negate-then-lower produces. All
/// three operands must be row values of equal size, else "row value misused". The
/// subject is compared against `lo` and `hi` with independent per-element metadata
/// (subject-vs-low, subject-vs-high), mirroring the scalar BETWEEN's two metas.
fn bind_row_between(
    scope: &Scope,
    ctx: &mut PlanCtx,
    negated: bool,
    expr: &Expr,
    low: &Expr,
    high: &Expr,
) -> Result<EvalExpr> {
    let (se, lo_e, hi_e) = match (row_value_elems(expr), row_value_elems(low), row_value_elems(high)) {
        (Some(s), Some(l), Some(h)) => (s, l, h),
        _ => return Err(Error::sql("row value misused")),
    };
    if se.len() != lo_e.len() || se.len() != hi_e.len() {
        return Err(Error::sql("row value misused"));
    }
    let n = se.len();
    // Independent per-element metadata for the two comparisons (subject vs low,
    // subject vs high) — never shared, exactly like the scalar low_meta/high_meta.
    let mut lo_metas = Vec::with_capacity(n);
    let mut hi_metas = Vec::with_capacity(n);
    for i in 0..n {
        lo_metas.push(compare_meta(scope, &se[i], &lo_e[i])?);
        hi_metas.push(compare_meta(scope, &se[i], &hi_e[i])?);
    }
    // Bind in source order — subject, then low, then high — for correct `?` numbering.
    let mut subj = Vec::with_capacity(n);
    for e in se {
        subj.push(bind_expr(scope, ctx, e)?);
    }
    let mut lo = Vec::with_capacity(n);
    for e in lo_e {
        lo.push(bind_expr(scope, ctx, e)?);
    }
    let mut hi = Vec::with_capacity(n);
    for e in hi_e {
        hi.push(bind_expr(scope, ctx, e)?);
    }
    Ok(if negated {
        // NOT BETWEEN ≡ (s < lo) OR (s > hi).
        let below = assemble_row_comparison(BinaryOp::Lt, &subj, &lo, &lo_metas);
        let above = assemble_row_comparison(BinaryOp::Gt, &subj, &hi, &hi_metas);
        EvalExpr::Or(Box::new(below), Box::new(above))
    } else {
        // BETWEEN ≡ (s >= lo) AND (s <= hi).
        let at_least = assemble_row_comparison(BinaryOp::Ge, &subj, &lo, &lo_metas);
        let at_most = assemble_row_comparison(BinaryOp::Le, &subj, &hi, &hi_metas);
        EvalExpr::And(Box::new(at_least), Box::new(at_most))
    })
}

/// Lower a simple CASE whose operand is a row value (rowvalue.html §2) as a searched
/// CASE of row equalities: each `WHEN (w…)` becomes the boolean condition
/// `(operand) = (w…)`. Each WHEN must be a row value of the operand's size, else
/// "row value misused". The operand elements are bound once (in source order, before
/// the WHENs) and cloned into each arm's equality; note that — like any searched-CASE
/// desugaring — a side-effecting operand element would be re-evaluated per arm, which
/// is immaterial for the column/literal operands this covers.
fn bind_row_case(
    scope: &Scope,
    ctx: &mut PlanCtx,
    operand: &Expr,
    whens: &[(Expr, Expr)],
    else_expr: &Option<Box<Expr>>,
) -> Result<EvalExpr> {
    let op_elems = row_value_elems(operand).expect("caller checks the operand is a row value");
    let n = op_elems.len();
    let mut op_bound = Vec::with_capacity(n);
    for e in op_elems {
        op_bound.push(bind_expr(scope, ctx, e)?);
    }
    let mut bound_whens = Vec::with_capacity(whens.len());
    for (when, then) in whens {
        let when_elems = match row_value_elems(when) {
            Some(w) if w.len() == n => w,
            _ => return Err(Error::sql("row value misused")),
        };
        // Per-element metadata for this arm's `(operand) = (when)` equality — each
        // element keeps its own affinity + collation.
        let mut metas = Vec::with_capacity(n);
        for i in 0..n {
            metas.push(compare_meta(scope, &op_elems[i], &when_elems[i])?);
        }
        // Bind the WHEN elements in source order (after the operand, before the THEN).
        let mut when_bound = Vec::with_capacity(n);
        for e in when_elems {
            when_bound.push(bind_expr(scope, ctx, e)?);
        }
        let cond = assemble_row_comparison(BinaryOp::Eq, &op_bound, &when_bound, &metas);
        let bthen = bind_expr(scope, ctx, then)?;
        // A searched-CASE arm (`cmp: None`): `cond` is evaluated as a boolean.
        bound_whens.push(CaseWhen { when: cond, cmp: None, then: bthen });
    }
    let bound_else = match else_expr.as_deref() {
        Some(e) => Some(Box::new(bind_expr(scope, ctx, e)?)),
        None => None,
    };
    Ok(EvalExpr::Case { operand: None, whens: bound_whens, else_expr: bound_else })
}

#[cfg(test)]
mod row_value_tests {
    //! Row-value comparison lowering (rowvalue.html §2.1). Each case parses and plans
    //! a literal `SELECT <row comparison>` through the live [`QueryPlanner`], then
    //! *evaluates* the lowered [`EvalExpr`] and asserts the resulting [`Value`]. This
    //! exercises the real bind → eval path (not the tree shape), so the three-valued
    //! NULL logic is checked as behavior. Every expected value is transcribed from the
    //! spec — the worked examples in `rowvalue.html` §2.1 — never from the engine.

    use minisqlite_catalog::{Catalog, IndexDef, TableDef};
    use minisqlite_expr::{eval, CompareMeta, EvalContext, EvalExpr, FnContext, SubqueryId};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::{Collation, Result, Value};

    use crate::plan::PlanNode;
    use crate::{Planner, QueryPlanner};

    /// An empty schema — the literal row-value cases reference no columns.
    struct EmptyCatalog;

    impl Catalog for EmptyCatalog {
        fn table(&self, _name: &str) -> Result<Option<&TableDef>> {
            Ok(None)
        }
        fn index(&self, _name: &str) -> Result<Option<&IndexDef>> {
            Ok(None)
        }
        fn indexes_on<'a>(&'a self, _table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(Vec::new())
        }
        fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
            unimplemented!("empty test catalog")
        }
        fn create_table(&mut self, _pager: &mut dyn Pager, _stmt: &CreateTable, _sql: &str) -> Result<()> {
            unimplemented!("empty test catalog")
        }
        fn create_index(&mut self, _pager: &mut dyn Pager, _stmt: &CreateIndex, _sql: &str) -> Result<()> {
            unimplemented!("empty test catalog")
        }
        fn drop_object(&mut self, _pager: &mut dyn Pager, _stmt: &Drop) -> Result<()> {
            unimplemented!("empty test catalog")
        }
    }

    /// A trivial evaluation context: the literal cases never reach a parameter or a
    /// subquery, so those callbacks are unreachable.
    struct NoCtx;

    impl FnContext for NoCtx {
        fn now_unix_millis(&self) -> i64 {
            0
        }
        fn random_i64(&mut self) -> i64 {
            0
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
    }

    impl EvalContext for NoCtx {
        fn param(&self, _index: usize) -> Result<Value> {
            unimplemented!("literal row-value tests bind no parameters")
        }
        fn eval_scalar_subquery(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<Value> {
            unimplemented!("literal row-value tests use no subqueries")
        }
        fn eval_exists(&mut self, _id: SubqueryId, _regs: &[Value]) -> Result<bool> {
            unimplemented!("literal row-value tests use no subqueries")
        }
        fn eval_in_subquery(
            &mut self,
            _id: SubqueryId,
            _probe: &Value,
            _meta: &CompareMeta,
            _regs: &[Value],
        ) -> Result<Option<bool>> {
            unimplemented!("literal row-value tests use no subqueries")
        }
    }

    /// Plan `SELECT <expr>` and evaluate its single projected expression.
    fn eval_row(expr_sql: &str) -> Value {
        let sql = format!("SELECT {expr_sql}");
        let ast = parse(&sql).expect("parse should succeed");
        let plan = QueryPlanner::new()
            .plan(&ast.statements[0], &EmptyCatalog)
            .expect("plan should succeed");
        let expr = match plan.root {
            PlanNode::Project { mut exprs, .. } => {
                assert_eq!(exprs.len(), 1, "expected one projected expr for {sql}");
                exprs.pop().unwrap()
            }
            other => panic!("expected a Project root for {sql}, got {other:?}"),
        };
        eval(&expr, &[], &mut NoCtx).expect("eval should not error")
    }

    /// Whether planning `SELECT <expr>` fails (parse or bind error).
    fn plan_is_err(expr_sql: &str) -> bool {
        let sql = format!("SELECT {expr_sql}");
        match parse(&sql) {
            Ok(ast) => QueryPlanner::new().plan(&ast.statements[0], &EmptyCatalog).is_err(),
            Err(_) => true,
        }
    }

    fn is_int(v: &Value, n: i64) -> bool {
        matches!(v, Value::Integer(x) if *x == n)
    }

    #[test]
    fn equality_and_inequality_no_nulls() {
        // (a…) = (b…) ≡ a1=b1 AND … ; (a…) <> (b…) ≡ a1<>b1 OR ….
        assert!(is_int(&eval_row("(1,2,3) = (1,2,3)"), 1));
        assert!(is_int(&eval_row("(1,2) = (1,3)"), 0));
        assert!(is_int(&eval_row("(1,2) <> (1,3)"), 1));
        assert!(is_int(&eval_row("(1,2) <> (1,2)"), 0));
    }

    #[test]
    fn ordering_is_lexicographic_no_nulls() {
        // rowvalue.html §2.1 worked examples plus the reflexive <=/>= boundary.
        assert!(is_int(&eval_row("(1,2,3) < (2,3,4)"), 1)); // first element decides
        assert!(is_int(&eval_row("(1,2,3) < (1,2,4)"), 1)); // last element decides
        assert!(is_int(&eval_row("(1,2) < (1,2)"), 0)); // equal rows are not <
        assert!(is_int(&eval_row("(1,2) <= (1,2)"), 1)); // equal rows are <=
        assert!(is_int(&eval_row("(2,0) > (1,9)"), 1)); // first element decides
        assert!(is_int(&eval_row("(1,2) >= (1,2)"), 1)); // equal rows are >=
    }

    #[test]
    fn null_three_valued_logic_matches_spec() {
        // The exact NULL examples from rowvalue.html §2.1 (the substitution rule):
        // NULL where a substitution could still swing the result ⇒ NULL, else 0/1.
        assert!(eval_row("(1,2,3) = (1,NULL,3)").is_null()); // NULL could be 2 (true) or 9 (false)
        assert!(is_int(&eval_row("(1,2,3) = (1,NULL,4)"), 0)); // 3≠4 fixes it false
        assert!(is_int(&eval_row("(1,2,3) < (1,3,NULL)"), 1)); // 2<3 decides before the NULL
        assert!(eval_row("(1,2,3) < (1,2,NULL)").is_null()); // prefix equal, last is 3<NULL
        assert!(is_int(&eval_row("(1,3,5) < (1,2,NULL)"), 0)); // 3<2 false decides before the NULL
    }

    #[test]
    fn null_first_element_short_circuits() {
        // The called-out pair: a definite first element decides regardless of a
        // trailing NULL; an equal first element lets the NULL make the result unknown.
        assert!(is_int(&eval_row("(1,NULL) < (2,3)"), 1)); // 1<2 is definitely true ⇒ 1
        assert!(eval_row("(1,NULL) < (1,3)").is_null()); // 1=1, then NULL<3 ⇒ NULL
        assert!(is_int(&eval_row("(1,NULL) = (2,3)"), 0)); // 1≠2 is definitely false ⇒ 0
        assert!(eval_row("(1,NULL) = (1,2)").is_null()); // 1=1, then NULL=2 ⇒ NULL
        assert!(is_int(&eval_row("(1,NULL) <> (2,3)"), 1)); // 1<>2 is definitely true ⇒ 1
    }

    #[test]
    fn is_and_is_not_are_null_safe() {
        // IS / IS NOT never yield NULL: NULL is just another comparable value.
        assert!(is_int(&eval_row("(1,2,NULL) IS (1,2,NULL)"), 1)); // rowvalue.html §2.1
        assert!(is_int(&eval_row("(1,NULL) IS (1,NULL)"), 1));
        assert!(is_int(&eval_row("(1,NULL) IS NOT (1,NULL)"), 0));
        assert!(is_int(&eval_row("(1,2) IS NOT (1,3)"), 1));
        assert!(is_int(&eval_row("(1,2) IS (1,3)"), 0));
    }

    #[test]
    fn size_mismatch_and_scalar_operand_are_errors() {
        // rowvalue.html §2: two row values of the SAME size, or it is "row value
        // misused". A row value against a scalar is likewise rejected. `+` is not a
        // comparison, so a row-value arithmetic operand is the same error via the
        // ordinary scalar path.
        assert!(plan_is_err("(1,2) = (1,2,3)")); // arity mismatch
        assert!(plan_is_err("(1,2,3) = (1,2)")); // arity mismatch (other way)
        assert!(plan_is_err("(1,2) = 1")); // row value vs scalar
        assert!(plan_is_err("1 = (1,2)")); // scalar vs row value
        assert!(plan_is_err("(1,2) + 1")); // row value in arithmetic
    }

    #[test]
    fn three_element_ordering_and_inequality() {
        // Round out `>` / `>=` / `<>` at three elements: the recursion is symmetric
        // with the `<` / `=` cases above, but exercise it directly here too.
        assert!(is_int(&eval_row("(3,2,1) > (3,2,0)"), 1)); // last element decides: 1>0
        assert!(is_int(&eval_row("(3,2,1) >= (3,2,1)"), 1)); // equal rows are >=
        assert!(is_int(&eval_row("(3,2,1) > (3,2,1)"), 0)); // equal rows are not >
        assert!(is_int(&eval_row("(1,2,3) <> (1,2,4)"), 1)); // some element differs
        assert!(is_int(&eval_row("(1,2,3) <> (1,2,3)"), 0)); // all equal ⇒ not <>
    }

    #[test]
    fn per_element_metadata_is_not_shared_across_elements() {
        // The CRITICAL invariant: each element carries its OWN comparison
        // metadata (affinity + collation), so element 1 must NOT inherit element 0's.
        // Element 0 uses NOCASE (an explicit COLLATE), element 1 the default BINARY:
        //   ('foo' COLLATE NOCASE, 'bar') = ('FOO','BAR')
        //     ≡ ('foo'='FOO' NOCASE = true) AND ('bar'='BAR' BINARY = false) ≡ 0.
        //   A single shared NOCASE meta would wrongly make element 1 true ⇒ 1.
        assert!(is_int(&eval_row("('foo' COLLATE NOCASE, 'bar') = ('FOO', 'BAR')"), 0));
        //   ('foo' COLLATE NOCASE, 'bar') = ('FOO','bar')
        //     ≡ ('foo'='FOO' NOCASE = true) AND ('bar'='bar' BINARY = true) ≡ 1.
        //   Confirms element 0 really uses NOCASE (a shared BINARY meta would give 0).
        assert!(is_int(&eval_row("('foo' COLLATE NOCASE, 'bar') = ('FOO', 'bar')"), 1));
    }

    #[test]
    fn row_between_lowers_to_range_comparisons() {
        // rowvalue.html §3.2: (s) BETWEEN (lo) AND (hi) ≡ (s)>=(lo) AND (s)<=(hi);
        // NOT BETWEEN ≡ (s)<(lo) OR (s)>(hi). Both bounds are inclusive.
        assert!(is_int(&eval_row("(2,2) BETWEEN (1,1) AND (3,3)"), 1)); // inside
        assert!(is_int(&eval_row("(4,4) BETWEEN (1,1) AND (3,3)"), 0)); // above high
        assert!(is_int(&eval_row("(0,0) BETWEEN (1,1) AND (3,3)"), 0)); // below low
        assert!(is_int(&eval_row("(1,1) BETWEEN (1,1) AND (3,3)"), 1)); // inclusive low
        assert!(is_int(&eval_row("(3,3) BETWEEN (1,1) AND (3,3)"), 1)); // inclusive high
        assert!(is_int(&eval_row("(4,4) NOT BETWEEN (1,1) AND (3,3)"), 1));
        assert!(is_int(&eval_row("(2,2) NOT BETWEEN (1,1) AND (3,3)"), 0));
        // A NULL that could still swing the range yields NULL (3VL, as for `=`/`<`).
        assert!(eval_row("(1,NULL) BETWEEN (1,1) AND (1,3)").is_null());
        // Every operand must be a row value of the same size, else "row value misused".
        assert!(plan_is_err("(2,2) BETWEEN (1,1,1) AND (3,3)"));
        assert!(plan_is_err("(2,2) BETWEEN 1 AND (3,3)"));
    }

    #[test]
    fn row_simple_case_matches_by_row_equality() {
        // rowvalue.html §2: a simple CASE over a row-value operand compares the operand
        // to each WHEN by row equality (lowered to a searched CASE of `(op)=(when)`).
        assert!(is_int(&eval_row("CASE (1,2) WHEN (1,2) THEN 1 ELSE 0 END"), 1)); // matches
        assert!(is_int(&eval_row("CASE (1,2) WHEN (3,4) THEN 1 ELSE 0 END"), 0)); // ⇒ ELSE
        // WHENs are scanned in order; the first row-equal one wins.
        assert!(is_int(&eval_row("CASE (1,2) WHEN (9,9) THEN 1 WHEN (1,2) THEN 2 ELSE 0 END"), 2));
        // A WHEN of a different size, or a scalar WHEN, is "row value misused".
        assert!(plan_is_err("CASE (1,2) WHEN (1,2,3) THEN 1 ELSE 0 END"));
        assert!(plan_is_err("CASE (1,2) WHEN 1 THEN 1 ELSE 0 END"));
    }

    #[test]
    fn row_value_in_value_list_is_error() {
        // rowvalue.html §2.2: a row-value IN requires a SUBQUERY RHS ("the RHS must be a
        // subquery expression"). A value-list RHS is "row value misused" — SQLite, unlike
        // PostgreSQL, does NOT fold it to a boolean, whether or not it would match.
        assert!(plan_is_err("(1,2) IN ((1,2),(3,4))")); // would match, still an error
        assert!(plan_is_err("(1,2) IN ((3,4),(5,6))")); // would not match, still an error
        assert!(plan_is_err("(1,2) NOT IN ((3,4),(5,6))")); // NOT IN, same
        assert!(plan_is_err("(1,2) IN (1,2)")); // scalar items
        assert!(plan_is_err("(1,2) IN ((1,2,3))")); // wrong-size item
    }

    /// Plan `SELECT <expr>` and return its single projected (bound) expression, so a
    /// test can inspect the lowered IR shape (not just its evaluated value).
    fn plan_expr(expr_sql: &str) -> EvalExpr {
        let sql = format!("SELECT {expr_sql}");
        let ast = parse(&sql).expect("parse should succeed");
        let plan = QueryPlanner::new()
            .plan(&ast.statements[0], &EmptyCatalog)
            .expect("plan should succeed");
        match plan.root {
            PlanNode::Project { mut exprs, .. } => {
                assert_eq!(exprs.len(), 1, "expected one projected expr for {sql}");
                exprs.pop().unwrap()
            }
            other => panic!("expected a Project root for {sql}, got {other:?}"),
        }
    }

    /// The error message from planning `SELECT <expr>` (which must fail).
    fn plan_err_msg(expr_sql: &str) -> String {
        let sql = format!("SELECT {expr_sql}");
        let ast = parse(&sql).expect("parse should succeed");
        QueryPlanner::new()
            .plan(&ast.statements[0], &EmptyCatalog)
            .expect_err("plan should fail")
            .to_string()
    }

    #[test]
    fn row_value_in_subquery_binds_to_in_subquery_row() {
        // rowvalue.html §2.2: a row-value IN with a subquery RHS lowers to the tuple
        // node `InSubqueryRow`, carrying one bound subject and one comparison metadata
        // per LHS element (width == the subquery's column count). This pins the binder
        // wiring; the runtime 3VL is proven in the executor's end-to-end tests.
        match plan_expr("(1,2) IN (SELECT 1, 2)") {
            EvalExpr::InSubqueryRow { negated, subjects, metas, .. } => {
                assert!(!negated, "IN, not NOT IN");
                assert_eq!(subjects.len(), 2, "one bound subject per LHS element");
                assert_eq!(metas.len(), 2, "one comparison meta per LHS element");
            }
            other => panic!("expected InSubqueryRow, got {other:?}"),
        }
        // Three-wide tuple over a three-column subquery binds likewise.
        match plan_expr("(1,2,3) IN (SELECT 1, 2, 3)") {
            EvalExpr::InSubqueryRow { subjects, metas, .. } => {
                assert_eq!(subjects.len(), 3);
                assert_eq!(metas.len(), 3);
            }
            other => panic!("expected InSubqueryRow, got {other:?}"),
        }
    }

    #[test]
    fn row_value_in_subquery_uses_per_element_metadata() {
        // The subquery branch of `bind_in` must compute per-element metadata, not reuse
        // one shared meta — the analogue of `per_element_metadata_is_not_shared_across_elements`
        // for the IN-subquery path (that test only covers row `=`). Element 0 has an
        // explicit COLLATE NOCASE, element 1 the default BINARY; a shared meta (the loop
        // reusing element 0's) would put NOCASE on element 1 too. Inspect the bound
        // metadata directly, since the tuple's runtime 3VL lives in the executor.
        match plan_expr("('foo' COLLATE NOCASE, 'bar') IN (SELECT 'x', 'y')") {
            EvalExpr::InSubqueryRow { metas, .. } => {
                assert_eq!(metas.len(), 2, "one comparison meta per LHS element");
                assert_eq!(metas[0].collation, Collation::NoCase, "element 0 keeps its explicit NOCASE");
                assert_eq!(
                    metas[1].collation,
                    Collation::Binary,
                    "element 1 keeps its own BINARY — element 0's NOCASE is not shared onto it"
                );
            }
            other => panic!("expected InSubqueryRow, got {other:?}"),
        }
    }

    #[test]
    fn row_value_not_in_subquery_sets_negated() {
        // NOT IN flips only the `negated` flag; the tuple shape is unchanged (the
        // executor applies the three-valued negation, so NOT IN of an empty subquery is
        // TRUE and NOT IN with a blocking NULL stays NULL).
        match plan_expr("(1,2) NOT IN (SELECT 1, 2)") {
            EvalExpr::InSubqueryRow { negated, subjects, .. } => {
                assert!(negated, "NOT IN sets negated");
                assert_eq!(subjects.len(), 2);
            }
            other => panic!("expected InSubqueryRow, got {other:?}"),
        }
    }

    #[test]
    fn row_value_in_subquery_column_count_mismatch_is_an_error() {
        // The subquery must return exactly as many columns as the tuple is wide, else
        // the same error real SQLite raises. Both directions (too many / too few).
        assert!(
            plan_err_msg("(1,2) IN (SELECT 1, 2, 3)").contains("sub-select returns 3 columns - expected 2")
        );
        assert!(
            plan_err_msg("(1,2,3) IN (SELECT 1, 2)").contains("sub-select returns 2 columns - expected 3")
        );
        // A single-column subquery against a 2-tuple is also a width error, not a
        // fall-through to the scalar path.
        assert!(
            plan_err_msg("(1,2) IN (SELECT 1)").contains("sub-select returns 1 columns - expected 2")
        );
    }
}

/// Build the subquery a table / table-valued function on the RHS of `IN` desugars to.
///
/// `lang_expr.html` §8: such a name "is understood to be a subquery of the form
/// (SELECT * FROM name)". A plain table (`x IN t`, empty `args`) becomes a
/// [`TableOrSubquery::Table`]; a table-valued function (`x IN f(a, b)`) becomes a
/// [`TableOrSubquery::TableFunction`] carrying the call arguments. Both the scalar
/// (`compile_value_subquery`, one column) and the row-value (`compile_row_subquery`, tuple
/// width) `IN` paths compile the returned SELECT, so this one source of truth keeps the two
/// arms in lock-step and the column-count rule is the SAME shared arity check for a scalar
/// and a tuple subject.
fn in_table_select_star(name: &QualifiedName, args: &[Expr]) -> Select {
    let source = if args.is_empty() {
        TableOrSubquery::Table { name: name.clone(), alias: None, indexed: None }
    } else {
        TableOrSubquery::TableFunction { name: name.clone(), args: args.to_vec(), alias: None }
    };
    Select {
        with: None,
        body: SelectBody::Select(SelectCore::Query {
            distinct: Distinct::All,
            columns: vec![ResultColumn::Star],
            from: Some(JoinTree::Table(source)),
            where_clause: None,
            group_by: Vec::new(),
            having: None,
            windows: Vec::new(),
        }),
        order_by: Vec::new(),
        limit: None,
    }
}

/// Lower `subject [NOT] IN (...)`.
///
/// A row-value subject `(a, b, …)` is handled first. `rowvalue.html` §2.2 states a
/// row-value `IN`'s "RHS must be a subquery expression"; a table or table-valued function
/// NAME also qualifies, since `lang_expr.html` §8 understands it as the subquery
/// `(SELECT * FROM name)` (see [`in_table_select_star`]). Only a value-LIST RHS is the "row
/// value misused" error — SQLite, unlike PostgreSQL, does NOT fold a row value against a
/// value list to a boolean. The subquery / table form lowers to [`EvalExpr::InSubqueryRow`]:
/// each subject element is bound in source order (so `?` parameters number left-to-right),
/// the subquery is compiled and required to return exactly as many columns as the tuple is
/// wide (else the SQLite `sub-select returns N columns - expected M` error), and element `i`
/// carries its OWN `compare_meta_subject` metadata (RHS no-affinity, subject collation —
/// §4.2/§7.1), exactly as the scalar `IN` path does. A scalar subject falls through to the
/// ordinary `InList` / `InSubquery` paths below (a table / TVF among them, lowered to
/// `InSubquery`).
fn bind_in(
    scope: &Scope,
    ctx: &mut PlanCtx,
    negated: bool,
    subject_ast: &Expr,
    rhs: &InRhs,
) -> Result<EvalExpr> {
    if let Some(elems) = row_value_elems(subject_ast) {
        // Bind subjects (and their per-element metadata) FIRST, in source order, so any
        // `?` params in the tuple number before the subquery's — matching SQLite's
        // left-to-right parameter numbering. Metadata is per element (see doc above): a
        // shared meta would leak element 0's affinity/collation onto the others.
        //
        // Like the scalar `InSubquery`/`InList` paths below, `compare_meta_subject` treats
        // the RHS as no-affinity (subject collation only). datatype3.html §3.2 says the
        // subquery column's result-set affinity should apply; threading it is a larger
        // binder feature. This same approximation now lives at all THREE IN sites (here,
        // and `InList`/`InSubquery` below) — whoever threads result-set affinity must fix
        // all three, or the row path silently diverges from the scalar one.
        let mut subjects = Vec::with_capacity(elems.len());
        let mut metas = Vec::with_capacity(elems.len());
        for elem in elems {
            metas.push(compare_meta_subject(scope, elem)?);
            subjects.push(bind_expr(scope, ctx, elem)?);
        }
        // The RHS must be a subquery — an explicit `(SELECT …)`, or a table / table-valued
        // function name understood as `(SELECT * FROM name)` (lang_expr.html §8, the SAME
        // `in_table_select_star` the scalar arm uses). `compile_row_subquery` then enforces
        // that it returns exactly `subjects.len()` columns (the tuple width) — the tuple
        // analogue of the scalar arm's one-column check. A value-LIST RHS is the "row value
        // misused" error (rowvalue.html §2.2): SQLite does not fold a row value against a
        // value list to a boolean.
        //
        // This List rejection is intentionally reached AFTER the subject bind loop above: a
        // malformed subject (e.g. an unknown column in the tuple) then surfaces its OWN
        // binding error first rather than being masked by the misuse message. Every valid
        // query, and the common `(a, b) IN (list)` case, is unaffected — both still error.
        let id = match rhs {
            InRhs::Select(sel) => compile_row_subquery(scope, ctx, sel, subjects.len())?,
            InRhs::Table { name, args } => {
                let sel = in_table_select_star(name, args);
                compile_row_subquery(scope, ctx, &sel, subjects.len())?
            }
            InRhs::List(_) => return Err(Error::sql("row value misused")),
        };
        return Ok(EvalExpr::InSubqueryRow { negated, subjects, id, metas });
    }
    Ok(match rhs {
        InRhs::List(items) => {
            // §4.2: list items carry NO affinity and the collation is the subject's —
            // computed as a comparison with a no-affinity right side.
            let meta = compare_meta_subject(scope, subject_ast)?;
            let subject = bind_expr(scope, ctx, subject_ast)?;
            let mut bitems = Vec::with_capacity(items.len());
            for it in items {
                bitems.push(bind_expr(scope, ctx, it)?);
            }
            EvalExpr::InList { negated, subject: Box::new(subject), items: bitems, meta }
        }
        InRhs::Select(sel) => {
            let meta = compare_meta_subject(scope, subject_ast)?;
            let subject = bind_expr(scope, ctx, subject_ast)?;
            let id = compile_value_subquery(scope, ctx, sel)?;
            EvalExpr::InSubquery { negated, subject: Box::new(subject), id, meta }
        }
        // lang_expr.html §8: a table name or table-valued function name on the RHS of IN
        // "is understood to be a subquery of the form (SELECT * FROM name)". Synthesize that
        // SELECT (see [`in_table_select_star`], shared with the row-value arm above) and route
        // it through the IDENTICAL path as `InRhs::Select`, so `x IN t` is literally
        // `x IN (SELECT * FROM t)`. The one-column requirement ("that table must have exactly
        // one column") is the SHARED `compile_value_subquery` arity check: a wider relation —
        // a multi-column table, or a table-valued function like `json_each` whose `SELECT *`
        // is 8 columns — raises the shared "sub-select returns N columns - expected 1" error,
        // matching SQLite. No second check, no column special-casing.
        InRhs::Table { name, args } => {
            let sel = in_table_select_star(name, args);
            let meta = compare_meta_subject(scope, subject_ast)?;
            let subject = bind_expr(scope, ctx, subject_ast)?;
            let id = compile_value_subquery(scope, ctx, &sel)?;
            EvalExpr::InSubquery { negated, subject: Box::new(subject), id, meta }
        }
    })
}

/// Lower a `CASE` expression. A `Some` operand is a *simple* CASE (each WHEN is an
/// equality against the operand, with its own comparison metadata); `None` is a
/// *searched* CASE (each WHEN is a boolean condition).
fn bind_case(
    scope: &Scope,
    ctx: &mut PlanCtx,
    operand: &Option<Box<Expr>>,
    whens: &[(Expr, Expr)],
    else_expr: &Option<Box<Expr>>,
) -> Result<EvalExpr> {
    // A simple CASE whose operand is a row value (rowvalue.html §2) lowers to a
    // searched CASE whose WHEN conditions are row equalities `(operand) = (when_i)`,
    // since the scalar simple-CASE machinery compares a single operand value against
    // each WHEN and cannot express a multi-element tuple equality.
    if let Some(op) = operand.as_deref() {
        if row_value_elems(op).is_some() {
            return bind_row_case(scope, ctx, op, whens, else_expr);
        }
    }
    let bound_operand = match operand.as_deref() {
        Some(op) => Some(Box::new(bind_expr(scope, ctx, op)?)),
        None => None,
    };
    let op_ast = operand.as_deref();
    let mut bound_whens = Vec::with_capacity(whens.len());
    for (when, then) in whens {
        let cmp = match op_ast {
            Some(op) => Some(compare_meta(scope, op, when)?),
            None => None,
        };
        let bwhen = bind_expr(scope, ctx, when)?;
        let bthen = bind_expr(scope, ctx, then)?;
        bound_whens.push(CaseWhen { when: bwhen, cmp, then: bthen });
    }
    let bound_else = match else_expr.as_deref() {
        Some(e) => Some(Box::new(bind_expr(scope, ctx, e)?)),
        None => None,
    };
    Ok(EvalExpr::Case { operand: bound_operand, whens: bound_whens, else_expr: bound_else })
}

/// Lower a function call: the SQLite special forms first (never via the registry),
/// then aggregate misuse detection, then a resolved scalar call.
fn bind_function(
    scope: &Scope,
    ctx: &mut PlanCtx,
    name: &str,
    distinct: bool,
    args: &FunctionArgs,
    filter: &Option<Box<Expr>>,
    over: &Option<OverClause>,
    order_by: &[OrderingTerm],
) -> Result<EvalExpr> {
    let over_some = over.is_some();
    let lname = name.to_ascii_lowercase();
    // Aggregate `ORDER BY` (`f(x ORDER BY y)`) is honored for an aggregate in an aggregate
    // query — that path (`bind_grouped_node` → `bind_aggregate`) never reaches here — AND
    // for an ORDERED-SET WINDOW aggregate (`f(x ORDER BY y) OVER (…)`, lang_aggfunc.html:
    // the ORDER BY sets the order the aggregate folds its per-frame inputs), which routes
    // through `bind_window_function` below with the `order_by` threaded in. So an OVER call
    // with a non-empty ORDER BY is NOT rejected here; the window binder either binds it (an
    // aggregate window fn) or rejects it (a non-aggregate/built-in window fn). What remains
    // a loud error here is a NON-window, non-aggregate call with ORDER BY (a scalar/special
    // form — ORDER BY is meaningless). A bare aggregate with ORDER BY outside a grouping
    // context (e.g. in WHERE) falls through to the misuse-of-aggregate error below.
    if !order_by.is_empty()
        && !over_some
        && !function_is_aggregate(ctx.registry, name, args, false)
    {
        return Err(Error::sql(format!(
            "ORDER BY may not be used with the non-aggregate function {name}()"
        )));
    }
    if let Some(bound) = try_special_form(scope, ctx, &lname, distinct, args, filter, over_some)? {
        return Ok(bound);
    }
    if function_is_aggregate(ctx.registry, name, args, over_some) {
        // An aggregate reached ordinary binding: it is used where aggregates are
        // not allowed (e.g. WHERE) — the aggregate query path pre-extracts the
        // legitimate ones before they get here.
        return Err(Error::sql(format!("misuse of aggregate function {name}()")));
    }
    if let Some(over) = over {
        // A window call (`f(...) OVER (...)`): the window binder maps the name to its
        // WindowFuncKind (aggregate, ranking, offset, or value function), resolves the
        // PARTITION BY / ORDER BY / frame from the OVER clause, and collects a WindowFunc
        // that resolves to the column the Window operator appends. `order_by` is the
        // in-argument ORDER BY of an ordered-set aggregate window fn (empty otherwise).
        return bind_window_function(scope, ctx, name, distinct, args, filter, over, order_by);
    }
    if distinct {
        return Err(Error::sql(format!("DISTINCT is not supported for scalar function {name}()")));
    }
    if filter.is_some() {
        return Err(Error::sql(format!("FILTER is not supported for scalar function {name}()")));
    }
    let (bound_args, argc) = match args {
        FunctionArgs::Star => {
            return Err(Error::sql(format!("wrong number of arguments to function {name}()")));
        }
        FunctionArgs::Empty => (Vec::new(), 0usize),
        FunctionArgs::List(list) => {
            let mut v = Vec::with_capacity(list.len());
            for it in list {
                v.push(bind_expr(scope, ctx, it)?);
            }
            (v, list.len())
        }
    };
    // Resolution errors (no such function / wrong arg count) come from the
    // registry, verbatim, so they match SQLite's wording.
    let func = ctx.registry.resolve_scalar(name, argc)?;
    // A collation-sensitive variadic scalar (`min`/`max` with ≥2 args, lang_corefunc.html)
    // binds under the collation ITS rule selects — the first argument, source order, that
    // defines a collating function, else BINARY — baked into the handle here so the per-row
    // extremum comparison needs no lookup (mirrors the aggregate `new_accumulator` seam).
    // `specialize_collation` returns `None` for every collation-INSENSITIVE function, which
    // is all the rest, leaving the shared registry handle untouched; the resolution above is
    // pure bind-time work bounded by the argument count.
    let func = match func.specialize_collation(positional_call_collation(scope, args)) {
        Some(specialized) => specialized,
        None => func,
    };
    // A non-deterministic function (e.g. `random()`) makes the enclosing subquery unsafe
    // to memoize; record it so `SubPlan::deterministic` is `false`.
    if !func.deterministic() {
        scope.note_nondeterministic();
    }
    Ok(EvalExpr::Func { func, args: bound_args })
}

/// The SQLite special-form function calls the binder lowers itself (they are not
/// registry functions). Returns `None` when `lname` is not a special form.
fn try_special_form(
    scope: &Scope,
    ctx: &mut PlanCtx,
    lname: &str,
    distinct: bool,
    args: &FunctionArgs,
    filter: &Option<Box<Expr>>,
    over_some: bool,
) -> Result<Option<EvalExpr>> {
    let is_special =
        matches!(lname, "coalesce" | "ifnull" | "iif" | "if" | "nullif" | "like" | "glob" | "likelihood");
    if !is_special {
        return Ok(None);
    }
    if distinct || filter.is_some() || over_some {
        return Err(Error::sql(format!("{lname}() does not support DISTINCT, FILTER, or OVER")));
    }
    let items: &[Expr] = match args {
        FunctionArgs::List(list) => list.as_slice(),
        FunctionArgs::Empty => &[],
        FunctionArgs::Star => {
            return Err(Error::sql(format!("use of '*' is not allowed with {lname}()")));
        }
    };
    let out = match lname {
        "coalesce" => {
            if items.len() < 2 {
                return Err(wrong_args("coalesce"));
            }
            let mut v = Vec::with_capacity(items.len());
            for it in items {
                v.push(bind_expr(scope, ctx, it)?);
            }
            EvalExpr::Coalesce(v)
        }
        "ifnull" => {
            if items.len() != 2 {
                return Err(wrong_args("ifnull"));
            }
            let a = bind_expr(scope, ctx, &items[0])?;
            let b = bind_expr(scope, ctx, &items[1])?;
            EvalExpr::Coalesce(vec![a, b])
        }
        // `iif(B1,V1,B2,V2,...,[ELSE])` is variadic (SQLite 3.49+): arguments come in
        // (Boolean, value) pairs and the first TRUE Boolean yields its paired value. An
        // ODD trailing argument is the ELSE; an even count with every Boolean false
        // yields NULL. This lowers to a searched CASE with one WHEN per pair, exactly as
        // the 3-argument form did (one pair + ELSE), generalized to N pairs. At least
        // two arguments are required. `if` is a pure alias. Bind in source order so `?`
        // parameters number correctly.
        "iif" | "if" => {
            if items.len() < 2 {
                return Err(wrong_args(lname));
            }
            let pair_count = items.len() / 2;
            let mut whens = Vec::with_capacity(pair_count);
            for pair in 0..pair_count {
                let when = bind_expr(scope, ctx, &items[2 * pair])?;
                let then = bind_expr(scope, ctx, &items[2 * pair + 1])?;
                whens.push(CaseWhen { when, cmp: None, then });
            }
            let else_expr = if items.len() % 2 == 1 {
                Some(Box::new(bind_expr(scope, ctx, &items[items.len() - 1])?))
            } else {
                None
            };
            EvalExpr::Case { operand: None, whens, else_expr }
        }
        "nullif" => {
            if items.len() != 2 {
                return Err(wrong_args("nullif"));
            }
            // `nullif(X,Y)` applies the §4.2 comparison AFFINITY like `=`, but its COLLATION
            // follows the positional "first argument that defines a collating function" rule
            // (lang_corefunc.html nullif) — the SAME rule as multi-arg min/max, NOT `=`'s
            // left-operand precedence. They differ only when the LEFT already defines a
            // collation and the RIGHT carries an explicit postfix COLLATE: `=` would let the
            // postfix win (rule 1 over rule 2), but nullif gives it no precedence over the
            // earlier-defined collation. Take the affinity from `compare_meta`, override the
            // collation with the positional choice.
            let mut meta = compare_meta(scope, &items[0], &items[1])?;
            meta.collation = defined_collation(scope, &items[0])
                .or_else(|| defined_collation(scope, &items[1]))
                .unwrap_or(minisqlite_types::Collation::Binary);
            let left = bind_expr(scope, ctx, &items[0])?;
            let right = bind_expr(scope, ctx, &items[1])?;
            EvalExpr::NullIf { left: Box::new(left), right: Box::new(right), meta }
        }
        // The function-call form `like(Y, X [,Z])` means `X LIKE Y [ESCAPE Z]`:
        // arg0 is the pattern, arg1 the subject, arg2 the escape. Bind in source
        // (argument) order so `?` params number correctly.
        "like" => {
            if items.len() != 2 && items.len() != 3 {
                return Err(wrong_args("like"));
            }
            let pattern = bind_expr(scope, ctx, &items[0])?;
            let subject = bind_expr(scope, ctx, &items[1])?;
            let escape = match items.get(2) {
                Some(z) => Some(Box::new(bind_expr(scope, ctx, z)?)),
                None => None,
            };
            EvalExpr::Like {
                negated: false,
                kind: IrLikeKind::Like,
                subject: Box::new(subject),
                pattern: Box::new(pattern),
                escape,
            }
        }
        "glob" => {
            if items.len() != 2 {
                return Err(wrong_args("glob"));
            }
            let pattern = bind_expr(scope, ctx, &items[0])?;
            let subject = bind_expr(scope, ctx, &items[1])?;
            EvalExpr::Like {
                negated: false,
                kind: IrLikeKind::Glob,
                subject: Box::new(subject),
                pattern: Box::new(pattern),
                escape: None,
            }
        }
        // `likelihood(X,Y)` is a planner hint that returns X unchanged, but its Y
        // must be a compile-time constant in [0.0, 1.0] (lang_corefunc: "a floating
        // point constant between 0.0 and 1.0, inclusive"). A non-constant or
        // out-of-range Y is rejected at bind time, matching real SQLite's prepare-time
        // error. On success Y is dropped and X is bound and returned directly — this
        // shadows the registry `likelihood`, which is a pass-through with the same
        // value on the happy path. (`likely`/`unlikely` stay registry no-ops.)
        "likelihood" => {
            if items.len() != 2 {
                return Err(wrong_args("likelihood"));
            }
            match const_numeric_literal(&items[1]) {
                Some(y) if (0.0..=1.0).contains(&y) => {}
                _ => {
                    return Err(Error::sql(
                        "second argument to likelihood() must be a constant between 0.0 and 1.0",
                    ));
                }
            }
            bind_expr(scope, ctx, &items[0])?
        }
        _ => unreachable!("is_special limits lname to the handled set"),
    };
    Ok(Some(out))
}

/// SQLite's "wrong number of arguments" error for a named function.
fn wrong_args(name: &str) -> Error {
    Error::sql(format!("wrong number of arguments to function {name}()"))
}

/// The value of a compile-time constant numeric literal — an integer or real
/// literal, or a unary minus applied (recursively) to one — as an `f64`, or `None`
/// for anything that is not such a constant. Used by `likelihood(X,Y)` to range-check
/// its constant Y at bind time; a non-literal Y (a column, a bind parameter, an
/// arithmetic expression) is `None` and so rejected.
fn const_numeric_literal(e: &Expr) -> Option<f64> {
    match e {
        Expr::Literal(Literal::Integer(i)) => Some(*i as f64),
        Expr::Literal(Literal::Real(r)) => Some(*r),
        Expr::Unary { op: SqlUnaryOp::Negative, expr } => const_numeric_literal(expr).map(|v| -v),
        _ => None,
    }
}

/// Whether a function call is an aggregate invocation: `count(*)` always, and any
/// name the registry classifies as an aggregate (window calls are handled
/// elsewhere, so `OVER` disqualifies it here).
pub fn function_is_aggregate(
    reg: &FunctionRegistry,
    name: &str,
    args: &FunctionArgs,
    over_some: bool,
) -> bool {
    if over_some {
        return false;
    }
    match args {
        FunctionArgs::Star => name.eq_ignore_ascii_case("count"),
        FunctionArgs::Empty => reg.is_aggregate(name),
        FunctionArgs::List(list) => {
            // `min`/`max` name BOTH a one-argument aggregate (lang_aggfunc) and a
            // two-or-more-argument scalar (lang_corefunc); the registry holds both,
            // so only the argument count disambiguates. A 2+-arg call is the SCALAR
            // function and must route through `resolve_scalar`, NOT the aggregate
            // resolver — otherwise it errors "wrong number of arguments" against the
            // Exact(1) aggregate. A one-argument `min`/`max` stays the aggregate.
            if list.len() >= 2 && (name.eq_ignore_ascii_case("min") || name.eq_ignore_ascii_case("max")) {
                return false;
            }
            reg.is_aggregate(name)
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison metadata (datatype3.html §3.2 affinity, §4.2 apply-rule, §7.1 collation)
// ---------------------------------------------------------------------------

/// Comparison metadata for `left <op> right` (used by =, <>, <, <=, >, >=, IS, IS
/// NOT and each BETWEEN / simple-CASE / NULLIF branch).
pub fn compare_meta(scope: &Scope, left: &Expr, right: &Expr) -> Result<CompareMeta> {
    let lc = compare_class(operand_affinity(scope, left)?);
    let rc = compare_class(operand_affinity(scope, right)?);
    let (apply_left, apply_right) = apply_affinity_rule(lc, rc);
    let collation = resolve_collation(scope, left, Some(right))?;
    Ok(CompareMeta { apply_left, apply_right, collation })
}

/// Comparison metadata for `subject IN (...)`: the right side (the list items or
/// the subquery column) is treated as having NO affinity, and the collation comes
/// from the subject alone (COLLATE on the subject still wins). This is the §7.1 /
/// §4.2 rule for `IN` — the RHS is genuine no-affinity (`CompareClass::NoAff`), which
/// §4.2 states explicitly ("the values to the right of the IN operator ... are
/// considered to have no affinity, even if they happen to be column values or CAST
/// expressions").
pub fn compare_meta_subject(scope: &Scope, subject: &Expr) -> Result<CompareMeta> {
    let lc = compare_class(operand_affinity(scope, subject)?);
    let (apply_left, apply_right) = apply_affinity_rule(lc, CompareClass::NoAff);
    let collation = resolve_collation(scope, subject, None)?;
    Ok(CompareMeta { apply_left, apply_right, collation })
}

/// The collation of a single operand (an explicit COLLATE, else a column's
/// collation, else BINARY). Used for GROUP BY keys and, via [`best_effort_collation`],
/// ORDER BY terms. Errors if the operand references an unresolvable column.
pub fn operand_collation(scope: &Scope, e: &Expr) -> Result<minisqlite_types::Collation> {
    resolve_collation(scope, e, None)
}

/// The collating sequence an operand DEFINES, if any — the shared primitive behind the
/// compound-select duplicate rule and the positional `min`/`max` argument rule. `Some(c)`
/// when the operand carries an explicit postfix `COLLATE` c (datatype3.html §7.1 rule 1),
/// else `Some(c)` when it is a plain column reference (through unary `+` / CAST / single
/// parentheses per §7.1's "still considered a column name" note) with declared collation c
/// (rule 2, BINARY included), else `None` — a literal, arithmetic, or a function call
/// defines no collation. Best-effort: an unresolvable column yields `None`.
///
/// The `Some`/`None` distinction is load-bearing. Callers that need the "greater precedence
/// NOT assigned to a postfix COLLATE" behaviour — compound SELECT dedup (lang_select.html
/// "duplicate rows") and the positional "first argument that defines a collating function"
/// rule (`min`/`max`, lang_corefunc.html) — combine per-operand results in SOURCE order
/// (left→right, first `Some` wins). That is why this returns an `Option` and, unlike
/// [`resolve_collation`] (the `=` rule), does NOT prefer an explicit COLLATE regardless of
/// its operand position: an explicit COLLATE defines a collation for ITS operand, but wins
/// across operands only by being earlier, not by outranking a column.
pub fn defined_collation(scope: &Scope, e: &Expr) -> Option<minisqlite_types::Collation> {
    if let Ok(Some(c)) = explicit_collation(e) {
        return Some(c);
    }
    operand_column_collation(scope, e).ok().flatten()
}

/// The collation of an ORDER BY / sort / aggregate-argument operand that must never fail:
/// the operand's [`defined_collation`] (explicit COLLATE, else column collation), else
/// BINARY. A reference that does not resolve (e.g. an output alias, which is not a FROM
/// column) falls back to BINARY rather than erroring.
pub fn best_effort_collation(scope: &Scope, e: &Expr) -> minisqlite_types::Collation {
    defined_collation(scope, e).unwrap_or(minisqlite_types::Collation::Binary)
}

/// The per-argument collations for an aggregate / window call's argument list, one per
/// bound argument (a `*` / empty list yields an empty vec). Each is the argument's §7.1
/// operand collation ([`best_effort_collation`]): an explicit postfix `COLLATE`, else the
/// argument column's declared collation, else BINARY. Aggregate `DISTINCT` dedup keys on
/// these (lang_select.html §2.6); a non-`DISTINCT` call carries but never reads them.
pub fn arg_list_collations(scope: &Scope, args: &FunctionArgs) -> Vec<minisqlite_types::Collation> {
    match args {
        FunctionArgs::List(list) => list.iter().map(|a| best_effort_collation(scope, a)).collect(),
        FunctionArgs::Star | FunctionArgs::Empty => Vec::new(),
    }
}

/// The collation a collation-sensitive variadic scalar applies to all its string
/// comparisons — the multi-argument `min`/`max` rule (lang_corefunc.html max_scalar /
/// min_scalar): scan the arguments left→right for the FIRST that DEFINES a collating
/// function ([`defined_collation`]) and use it; if none of the arguments define one, BINARY.
///
/// This is the positional "defines a collating function" rule — datatype3.html §7.1 rules 1
/// and 2 taken in SOURCE order, WITHOUT rule 1's precedence over rule 2 — and is
/// deliberately distinct from [`resolve_collation`] (the `=` operator's left-operand
/// precedence): `min('a', x COLLATE NOCASE)` uses NOCASE (the first arg defines nothing, the
/// second defines NOCASE), whereas an `=` comparison would prefer the explicit COLLATE
/// regardless of position.
pub fn positional_call_collation(scope: &Scope, args: &FunctionArgs) -> minisqlite_types::Collation {
    match args {
        FunctionArgs::List(list) => list
            .iter()
            .find_map(|a| defined_collation(scope, a))
            .unwrap_or(minisqlite_types::Collation::Binary),
        FunctionArgs::Star | FunctionArgs::Empty => minisqlite_types::Collation::Binary,
    }
}

/// The §3.2 "expression affinity" of a comparison operand, as an `Option`:
/// `Some(aff)` when the operand carries a real affinity, `None` for a genuine
/// no-affinity operand.
///
/// A bare column has its column affinity — including `Some(Affinity::Blob)` for a
/// BLOB-declared or typeless column (§3.1). `CAST(e AS T)` has T's affinity; COLLATE
/// and single-element parentheses are transparent. Everything else — a `+x`, a
/// literal, arithmetic, a function call, any other operator — has **no affinity**
/// (`None`).
///
/// This reports the operand's genuine §3.2 affinity and deliberately does NOT collapse
/// `Some(Affinity::Blob)` into `None`. For the §4.2 comparison rules those two ARE one
/// class ("no affinity"/NONE — §3.1 records that BLOB affinity was historically named
/// "NONE"), but that unification lives in exactly one place — [`compare_class`] — so the
/// rule reasons over the three-way [`CompareClass`], never over this raw `Option`. Keep
/// it that way: never re-derive the class from a bare `la.is_none()` here or at a call
/// site (that is the None-vs-Blob slip that made this rule regress).
pub(crate) fn operand_affinity(scope: &Scope, e: &Expr) -> Result<Option<Affinity>> {
    Ok(match e {
        Expr::Parenthesized(list) if list.len() == 1 => operand_affinity(scope, &list[0])?,
        Expr::Collate { expr, .. } => operand_affinity(scope, expr)?,
        Expr::Cast { type_name, .. } => Some(affinity_of_declared_type(Some(type_name))),
        Expr::Column { table, name, from_dqs, .. } => match scope.try_resolve_column(table.as_deref(), name) {
            Ok(Some(rc)) => Some(rc.affinity),
            // DQS: a bare double-quoted column that names NO column becomes a text literal,
            // which has no §3.2 affinity — mirror the `_ => None` literal arm below so
            // comparison metadata reflects the literal it becomes, not a phantom column. An
            // ambiguous match is a real column reference and stays the loud error.
            Ok(None) if *from_dqs => None,
            Ok(None) => return Err(Error::sql(format!("no such column: {name}"))),
            Err(e) => return Err(e),
        },
        _ => None,
    })
}

/// The §4.2 comparison class of an operand — the three classes the comparison-affinity
/// rules actually distinguish. This is the ONE place BLOB affinity and "no affinity" are
/// unified, which is what makes the recurring None-vs-Blob regression *unrepresentable*
/// downstream: [`apply_affinity_rule`] reasons over `CompareClass`, which has no `Blob`
/// variant to confuse with `NoAff`, so the bug cannot be re-encoded in the rule itself.
///
/// Why `Some(Affinity::Blob)` maps to `NoAff` (the subtlety that made this rule regress
/// repeatedly): for the comparison rules **BLOB affinity and "no affinity" are the SAME
/// class.** datatype3 §3.1 (datatype3.html:294) records that BLOB affinity "used to be
/// called 'NONE'", and §4.3's worked example annotates its `c BLOB` and typeless `d`
/// columns as `-- no affinity` (datatype3.html:671-672). So a BLOB/typeless COLUMN and a
/// no-affinity EXPRESSION are one class here. DO NOT split them back apart to force
/// `a=d` → 0 on the §4.3 fixture: that is the recurring wrong "fix" — a hyper-literal
/// reading of rule (ii)'s bare "no affinity" that the spec's own §4.3 example (labeling a
/// BLOB column "no affinity") contradicts and that real sqlite does not implement (its
/// expression affinity has no separate "none"; a no-affinity expression reports BLOB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareClass {
    /// INTEGER / REAL / NUMERIC affinity.
    Numeric,
    /// TEXT affinity.
    Text,
    /// "No affinity"/NONE: a bare no-affinity expression (`None`) OR a BLOB/typeless
    /// column (`Some(Affinity::Blob)`) — §3.1/§4.3 make these one class (see above).
    NoAff,
}

/// Collapse an operand's raw §3.2 affinity into its §4.2 comparison class. The SOLE site
/// that unifies `Some(Affinity::Blob)` with `None` (see [`CompareClass`]); the match is
/// exhaustive with no catch-all so a future `Affinity` variant forces a decision here.
fn compare_class(a: Option<Affinity>) -> CompareClass {
    match a {
        None | Some(Affinity::Blob) => CompareClass::NoAff,
        Some(Affinity::Text) => CompareClass::Text,
        Some(Affinity::Integer | Affinity::Real | Affinity::Numeric) => CompareClass::Numeric,
    }
}

/// Apply the §4.2 comparison-affinity rule, returning `(apply_left, apply_right)`: at most
/// one side is converted, only that side is `Some`, and the result is never
/// `Some(Affinity::Blob)` — only `None` / `Some(Numeric)` / `Some(Text)`.
///
/// A total function over the 3×3 [`CompareClass`] pairs with NO catch-all `_`, so adding a
/// class forces every pair to be reconsidered. The rules, applied in order (§4.2,
/// datatype3.html:640-648):
///  (i)   one operand numeric, the other TEXT/no-affinity → NUMERIC to the other.
///  (ii)  one operand TEXT, the other no-affinity          → TEXT to the other.
///  (iii) otherwise (both numeric, TEXT vs TEXT, NoAff vs NoAff) → nothing.
///
/// Pinned by `conformance_comparison_affinity` on the §4.3 fixture
/// `t1(a TEXT, b NUMERIC, c BLOB, d)`: `SELECT a<d, a=d, a>d` is `[0,1,0]` — rule (ii)
/// applies TEXT to the typeless `d` (`NoAff`), its stored INTEGER 500 → '500' → equal to
/// a's '500' — while `SELECT c<d, c=d, c>d` is `[0,0,1]` — rule (iii), both `NoAff`.
fn apply_affinity_rule(l: CompareClass, r: CompareClass) -> (Option<Affinity>, Option<Affinity>) {
    use CompareClass::{NoAff, Numeric, Text};
    match (l, r) {
        // Rule (i): numeric vs non-numeric (TEXT / no-affinity) → NUMERIC to that side.
        (Numeric, Text | NoAff) => (None, Some(Affinity::Numeric)),
        (Text | NoAff, Numeric) => (Some(Affinity::Numeric), None),
        // Rule (ii): TEXT vs no-affinity → TEXT to the no-affinity side.
        (Text, NoAff) => (None, Some(Affinity::Text)),
        (NoAff, Text) => (Some(Affinity::Text), None),
        // Rule (iii): both numeric, TEXT vs TEXT, or NoAff vs NoAff → compared as is.
        (Numeric, Numeric) | (Text, Text) | (NoAff, NoAff) => (None, None),
    }
}

/// Resolve the §7.1 collation for a comparison: an explicit postfix COLLATE (left
/// operand winning) beats a column's collation (left winning) beats BINARY.
/// `right` is `None` for the `IN` case (only the subject contributes).
fn resolve_collation(scope: &Scope, left: &Expr, right: Option<&Expr>) -> Result<minisqlite_types::Collation> {
    if let Some(c) = explicit_collation(left)? {
        return Ok(c);
    }
    if let Some(r) = right {
        if let Some(c) = explicit_collation(r)? {
            return Ok(c);
        }
    }
    if let Some(c) = operand_column_collation(scope, left)? {
        return Ok(c);
    }
    if let Some(r) = right {
        if let Some(c) = operand_column_collation(scope, r)? {
            return Ok(c);
        }
    }
    Ok(minisqlite_types::Collation::Binary)
}

/// The leftmost explicit `COLLATE` within an operand, if any. Descends the
/// transparent wrappers and, for a nested `COLLATE`, prefers the inner (leftmost
/// in source) one.
fn explicit_collation(e: &Expr) -> Result<Option<minisqlite_types::Collation>> {
    Ok(match e {
        Expr::Collate { expr, collation } => match explicit_collation(expr)? {
            Some(c) => Some(c),
            None => Some(parse_collation(collation)?),
        },
        Expr::Parenthesized(list) => {
            for item in list {
                if let Some(c) = explicit_collation(item)? {
                    return Ok(Some(c));
                }
            }
            None
        }
        Expr::Unary { expr, .. } => explicit_collation(expr)?,
        Expr::Cast { expr, .. } => explicit_collation(expr)?,
        Expr::Binary { left, right, .. } => match explicit_collation(left)? {
            Some(c) => Some(c),
            None => explicit_collation(right)?,
        },
        _ => None,
    })
}

/// The collation of an operand that is a column, per §7.1's "a column name
/// preceded by one or more unary `+` and/or CAST operators is still a column".
fn operand_column_collation(scope: &Scope, e: &Expr) -> Result<Option<minisqlite_types::Collation>> {
    Ok(match e {
        Expr::Parenthesized(list) if list.len() == 1 => operand_column_collation(scope, &list[0])?,
        Expr::Unary { op: SqlUnaryOp::Positive, expr } => operand_column_collation(scope, expr)?,
        Expr::Cast { expr, .. } => operand_column_collation(scope, expr)?,
        Expr::Column { table, name, from_dqs, .. } => match scope.try_resolve_column(table.as_deref(), name) {
            Ok(Some(rc)) => Some(rc.collation),
            // DQS: a bare double-quoted column that names NO column is a text literal
            // (BINARY collation) — returning None lets resolve_collation fall through to
            // BINARY. An ambiguous match is a real column reference and stays the loud error.
            Ok(None) if *from_dqs => None,
            Ok(None) => return Err(Error::sql(format!("no such column: {name}"))),
            Err(e) => return Err(e),
        },
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// AST-op -> IR-op mappers
// ---------------------------------------------------------------------------

fn unary_op(op: SqlUnaryOp) -> IrUnaryOp {
    match op {
        SqlUnaryOp::Negative => IrUnaryOp::Neg,
        SqlUnaryOp::Positive => IrUnaryOp::Pos,
        SqlUnaryOp::Not => IrUnaryOp::Not,
        SqlUnaryOp::BitNot => IrUnaryOp::BitNot,
    }
}

fn arith_op(op: BinaryOp) -> ArithOp {
    match op {
        BinaryOp::Add => ArithOp::Add,
        BinaryOp::Sub => ArithOp::Sub,
        BinaryOp::Mul => ArithOp::Mul,
        BinaryOp::Div => ArithOp::Div,
        BinaryOp::Mod => ArithOp::Mod,
        _ => unreachable!("arith_op called with a non-arithmetic operator"),
    }
}

fn bit_op(op: BinaryOp) -> BitOp {
    match op {
        BinaryOp::BitAnd => BitOp::And,
        BinaryOp::BitOr => BitOp::Or,
        BinaryOp::LShift => BitOp::Shl,
        BinaryOp::RShift => BitOp::Shr,
        _ => unreachable!("bit_op called with a non-bitwise operator"),
    }
}

fn cmp_op(op: BinaryOp) -> CmpOp {
    match op {
        BinaryOp::Lt => CmpOp::Lt,
        BinaryOp::Le => CmpOp::Le,
        BinaryOp::Gt => CmpOp::Gt,
        BinaryOp::Ge => CmpOp::Ge,
        BinaryOp::Eq => CmpOp::Eq,
        BinaryOp::Ne => CmpOp::Ne,
        _ => unreachable!("cmp_op called with a non-comparison operator"),
    }
}

#[cfg(test)]
mod tests {
    //! Binder routing tests for the three function-call special cases owned by this
    //! file: scalar-vs-aggregate `min`/`max` (by argc), variadic `iif`/`if`, and the
    //! compile-time `likelihood(X,Y)` range check. Each plans real SQL through the
    //! live [`QueryPlanner`] (built-in registry) and inspects the resulting plan, so
    //! the check runs through the same binder path the conformance suite exercises.

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_expr::EvalExpr;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::{Result, Value};

    use crate::plan::PlanNode;
    use crate::{Planner, QueryPlanner};

    /// A static, read-only catalog holding a handful of tables — enough to resolve the
    /// column references in the routing tests. The write paths are never called.
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
            unimplemented!("static test catalog")
        }
        fn create_table(&mut self, _pager: &mut dyn Pager, _stmt: &CreateTable, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn create_index(&mut self, _pager: &mut dyn Pager, _stmt: &CreateIndex, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn drop_object(&mut self, _pager: &mut dyn Pager, _stmt: &Drop) -> Result<()> {
            unimplemented!("static test catalog")
        }
    }

    fn icol(name: &str) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: Some("INTEGER".to_string()),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: None,
            default: None,
            default_value: None,
            generated: None,
        }
    }

    /// `t(a INTEGER, b INTEGER)` — the fixture for the column-argument cases.
    fn cat_t() -> TestCatalog {
        TestCatalog {
            tables: vec![TableDef {
                name: "t".to_string(),
                columns: vec![icol("a"), icol("b")],
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

    fn empty_cat() -> TestCatalog {
        TestCatalog { tables: Vec::new() }
    }

    /// Plan `sql` through the real planner (built-in registry) and return the plan's
    /// root node.
    fn plan_root(sql: &str, cat: &dyn Catalog) -> Result<PlanNode> {
        let ast = parse(sql)?;
        Ok(QueryPlanner::new().plan(&ast.statements[0], cat)?.root)
    }

    /// The single projected expression of a `SELECT <expr>` plan.
    fn proj0(sql: &str, cat: &dyn Catalog) -> EvalExpr {
        match plan_root(sql, cat).expect("plan should succeed") {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 1, "expected exactly one projected expr for {sql}");
                exprs.into_iter().next().unwrap()
            }
            other => panic!("expected a Project root for {sql}, got {other:?}"),
        }
    }

    // -- FIX 1: scalar vs aggregate min/max routing by argument count -------------

    #[test]
    fn min_max_with_two_or_more_args_route_to_scalar_func() {
        let cat = empty_cat();
        // 2+ arguments are the SCALAR min/max — lowered to a Func call, not an aggregate.
        match proj0("SELECT max(1, 2)", &cat) {
            EvalExpr::Func { args, .. } => assert_eq!(args.len(), 2),
            other => panic!("expected a scalar Func for max(1,2), got {other:?}"),
        }
        match proj0("SELECT min(3, 1, 2)", &cat) {
            EvalExpr::Func { args, .. } => assert_eq!(args.len(), 3),
            other => panic!("expected a scalar Func for min(3,1,2), got {other:?}"),
        }
    }

    #[test]
    fn single_arg_min_max_stay_aggregates() {
        let cat = cat_t();
        // One argument keeps the aggregate min/max: an Aggregate node is built.
        for sql in ["SELECT max(a) FROM t", "SELECT min(a) FROM t"] {
            match plan_root(sql, &cat).expect("plan ok") {
                PlanNode::Project { input, .. } => match *input {
                    PlanNode::Aggregate(agg) => assert_eq!(agg.aggregates.len(), 1),
                    other => panic!("expected an Aggregate under Project for {sql}, got {other:?}"),
                },
                other => panic!("expected Project for {sql}, got {other:?}"),
            }
        }
    }

    #[test]
    fn scalar_min_max_over_columns_is_not_an_aggregate_query() {
        let cat = cat_t();
        // max(a, b) is the 2-arg scalar; the query stays row-wise (no Aggregate node).
        match plan_root("SELECT max(a, b) FROM t", &cat).expect("plan ok") {
            PlanNode::Project { input, exprs } => {
                assert!(matches!(exprs[0], EvalExpr::Func { .. }), "expected a scalar Func, got {:?}", exprs[0]);
                assert!(
                    !matches!(*input, PlanNode::Aggregate(_)),
                    "scalar max(a,b) must not build an Aggregate node"
                );
            }
            other => panic!("expected Project, got {other:?}"),
        }
    }

    // -- FIX 2: variadic iif / if ------------------------------------------------

    /// The searched-CASE shape an `iif` call lowers to: `(whens.len(), has_else)`.
    fn iif_shape(sql: &str) -> (usize, bool) {
        let cat = empty_cat();
        match proj0(sql, &cat) {
            EvalExpr::Case { operand, whens, else_expr } => {
                assert!(operand.is_none(), "iif must lower to a searched CASE (no operand) for {sql}");
                (whens.len(), else_expr.is_some())
            }
            other => panic!("expected a Case for {sql}, got {other:?}"),
        }
    }

    #[test]
    fn iif_lowers_to_searched_case_of_pairs_plus_optional_else() {
        // 3 args = 1 (Boolean, value) pair + trailing ELSE.
        assert_eq!(iif_shape("SELECT iif(1, 'y', 'n')"), (1, true));
        // `if` is an alias with the identical lowering.
        assert_eq!(iif_shape("SELECT if(1, 'y', 'n')"), (1, true));
        // 2 args = 1 pair, no ELSE (a false Boolean then yields NULL at eval time).
        assert_eq!(iif_shape("SELECT iif(1, 'a')"), (1, false));
        // 5 args = 2 pairs + trailing ELSE.
        assert_eq!(iif_shape("SELECT iif(0, 'a', 1, 'b', 'c')"), (2, true));
        // 4 args = 2 pairs, no ELSE.
        assert_eq!(iif_shape("SELECT iif(1, 'a', 1, 'b')"), (2, false));
    }

    #[test]
    fn iif_requires_at_least_two_arguments() {
        let cat = empty_cat();
        assert!(plan_root("SELECT iif(1)", &cat).is_err(), "iif(1) must be a bind error");
        assert!(plan_root("SELECT if(1)", &cat).is_err(), "if(1) must be a bind error");
    }

    // -- FIX 3: likelihood(X,Y) compile-time constant-range check ----------------

    #[test]
    fn likelihood_in_range_returns_first_argument_unchanged() {
        let cat = empty_cat();
        // The happy path drops Y and returns the bound X unchanged.
        assert!(matches!(proj0("SELECT likelihood(7, 0.5)", &cat), EvalExpr::Literal(Value::Integer(7))));
        match proj0("SELECT likelihood('a', 0.9375)", &cat) {
            EvalExpr::Literal(Value::Text(s)) => assert_eq!(s, "a"),
            other => panic!("expected the text 'a', got {other:?}"),
        }
        // The inclusive bounds 0.0 and 1.0 are accepted.
        assert!(matches!(proj0("SELECT likelihood(7, 0.0)", &cat), EvalExpr::Literal(Value::Integer(7))));
        assert!(matches!(proj0("SELECT likelihood(7, 1.0)", &cat), EvalExpr::Literal(Value::Integer(7))));
    }

    #[test]
    fn likelihood_rejects_out_of_range_or_non_constant_y() {
        let cat = cat_t();
        // Y > 1.0, and Y < 0.0 written as a unary-minus literal, are both out of range.
        assert!(plan_root("SELECT likelihood(7, 2.0)", &cat).is_err(), "Y=2.0 is out of range");
        assert!(plan_root("SELECT likelihood(7, -0.5)", &cat).is_err(), "Y=-0.5 is out of range");
        // A non-constant Y (a column) is not a compile-time constant → rejected.
        assert!(plan_root("SELECT likelihood(7, a) FROM t", &cat).is_err(), "a non-constant Y is rejected");
    }
}
