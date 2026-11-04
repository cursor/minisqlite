//! The SELECT output layer: the projection, and — when the query aggregates — the
//! GROUP BY / HAVING / aggregate machinery beneath it.
//!
//! [`compile_result`] decides whether the query is an aggregate query (a GROUP BY,
//! a HAVING, or any aggregate call in the select list) and produces a [`Compiled`]:
//! the node that feeds the final projection, the projection expressions, the output
//! names, and (for ORDER BY resolution) the source AST of each output column. The
//! orchestrator ([`crate::compile::select`]) assembles the Project / DISTINCT /
//! ORDER BY / LIMIT around it.
//!
//! In an aggregate query the projection and HAVING bind against the post-aggregate
//! `[group_keys.., agg_results..]` layout via a [`Grouping`] context on the scope;
//! aggregate calls are collected as they are bound and become the operator's
//! [`AggregateCall`] list. When the projection carries a window function, a
//! post-aggregate windowing stage (`Aggregate → Window`) is added so the window
//! runs once per group row over that same layout — see [`compile_aggregate`].

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use minisqlite_expr::{AggregateCall, EvalExpr, SortKey};
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::{Expr, FunctionArgs, InRhs, Literal, OrderingTerm, ResultColumn, WindowSpec};
use minisqlite_types::{Collation, Error, Result};

use crate::bind::{
    bind_expr, function_is_aggregate, operand_collation, BareCapture, Grouping, Scope, Windowing,
};
use crate::colname::result_column_name;
use crate::compile::minmax_index;
use crate::plan::{Aggregate, MinMaxBare, PlanNode, Window, WindowFunc};
use crate::plan_ctx::PlanCtx;

/// The output layer of a SELECT core, minus the DISTINCT/ORDER BY/LIMIT the
/// orchestrator wraps on top.
pub struct Compiled {
    /// The node feeding the final projection: the base access/filter for a plain
    /// query, or the [`PlanNode::Aggregate`] for an aggregate query.
    pub input: PlanNode,
    /// The projection expressions, bound against `input`'s row layout.
    pub projections: Vec<EvalExpr>,
    /// Output column names, one per projection.
    pub names: Vec<String>,
    /// The source AST of each output column: an explicit expression's own AST, or a
    /// synthesized bare-column AST for a `*`-expanded column. Used to match an ORDER
    /// BY term to a result column by structural equality AND to inherit the
    /// referenced column's collation (an ORDER BY ordinal/name that resolves to this
    /// column takes its collation from this AST). `None` is reserved for a future
    /// output with no column-like source. A star column carries a synthesized
    /// `Column { name }` so it, too, contributes its collation to ORDER BY.
    pub result_asts: Vec<Option<Expr>>,
    /// Whether this is an aggregate query (ORDER BY over non-output expressions is
    /// only supported for non-aggregate queries).
    pub is_agg: bool,
    /// For an AGGREGATE query, each ORDER BY term pre-resolved against the post-aggregate
    /// grouping context (one entry per `sel.order_by` term, in order); `None` for a plain
    /// query (whose ORDER BY the orchestrator resolves against the FROM scope). This is
    /// what lets an aggregate query ORDER BY a grouping key or aggregate the SELECT list
    /// does not project — the term is bound HERE, where the grouping context is live, so a
    /// bare group key / aggregate resolves instead of hitting the "does not match any
    /// column" error. See [`OrderResolved`].
    pub order_pre: Option<Vec<OrderResolved>>,
}

/// One aggregate-query ORDER BY term, pre-resolved against the post-aggregate relation.
pub enum OrderResolved {
    /// The term is an existing output column (a 1-based ordinal, an output name/alias, or
    /// a structural match to a result-column source AST): sort by output column `idx`, no
    /// hidden column added. Its collation is inherited from that output column.
    Output(usize),
    /// The term is an expression over the post-aggregate row that is NOT a projected
    /// output — a group key, an aggregate, or an expression thereof that the SELECT list
    /// omits. It is bound here (against the grouping context) and the orchestrator appends
    /// it as a HIDDEN projection column to sort by, then drops it.
    Hidden(EvalExpr),
}

/// Try to resolve an ORDER BY term to an EXISTING output column — the three
/// aggregate-safe cases that need no post-aggregate binding: (1) a 1-based integer
/// ordinal, (2) a bare name matching an output alias/column, (3) a structural match to a
/// result column's source AST. Returns the output column index, or `None` when the term
/// is some other expression (which the caller binds in the appropriate scope). An
/// out-of-range ordinal is a loud error (as in SQLite). Shared by the plain-query
/// resolver ([`crate::compile::select`]) and the aggregate ORDER BY binding below so both
/// classify the three cases identically.
pub(crate) fn match_output_column(
    term: &OrderingTerm,
    result_asts: &[Option<Expr>],
    names: &[String],
    num_out: usize,
) -> Result<Option<usize>> {
    // 1. A bare positive integer is a 1-based output-column ordinal.
    if let Expr::Literal(Literal::Integer(k)) = &term.expr {
        if *k >= 1 && (*k as usize) <= num_out {
            return Ok(Some((*k - 1) as usize));
        }
        return Err(Error::sql(format!(
            "ORDER BY term out of range - should be between 1 and {num_out}"
        )));
    }
    // 2. A bare unqualified name matching an output column (alias or column name).
    if let Expr::Column { schema: None, table: None, name, .. } = &term.expr {
        if let Some(idx) = names.iter().position(|n| n.eq_ignore_ascii_case(name)) {
            return Ok(Some(idx));
        }
    }
    // 3. A structural match to one of the result-column source expressions.
    for (idx, ast) in result_asts.iter().enumerate() {
        if let Some(a) = ast {
            if a == &term.expr {
                return Ok(Some(idx));
            }
        }
    }
    Ok(None)
}

/// Compile the SELECT output layer over `input` (the base access + WHERE filter).
pub fn compile_result(
    ctx: &mut PlanCtx,
    scope: &Scope,
    columns: &[ResultColumn],
    group_by: &[Expr],
    having: &Option<Expr>,
    order_by: &[OrderingTerm],
    input: PlanNode,
) -> Result<Compiled> {
    let is_agg =
        !group_by.is_empty() || having.is_some() || columns_contain_aggregate(ctx.registry, columns);
    if is_agg {
        compile_aggregate(ctx, scope, columns, group_by, having, order_by, input)
    } else {
        // A plain query resolves ORDER BY later against the FROM scope (the projection is
        // bound over that row), so nothing to pre-resolve here.
        compile_projection(ctx, scope, columns, input)
    }
}

/// The plain (non-aggregate) projection: bind each result column against the FROM
/// scope, expanding `*` / `table.*` in place.
fn compile_projection(
    ctx: &mut PlanCtx,
    scope: &Scope,
    columns: &[ResultColumn],
    input: PlanNode,
) -> Result<Compiled> {
    let mut projections = Vec::new();
    let mut names = Vec::new();
    let mut result_asts = Vec::new();
    for col in columns {
        match col {
            ResultColumn::Expr { expr, alias } => {
                projections.push(bind_expr(scope, ctx, expr)?);
                names.push(alias.clone().unwrap_or_else(|| result_column_name(expr)));
                result_asts.push(Some(expr.clone()));
            }
            ResultColumn::Star => expand_star_into(scope, None, &mut projections, &mut names, &mut result_asts)?,
            ResultColumn::TableStar(t) => {
                expand_star_into(scope, Some(t), &mut projections, &mut names, &mut result_asts)?
            }
        }
    }
    Ok(Compiled { input, projections, names, result_asts, is_agg: false, order_pre: None })
}

/// Push every column of `*` / `table.*` as a projection with its name and a
/// synthesized bare-column source AST (so an ORDER BY term that resolves to a star
/// column inherits that column's collation, not BINARY). Ordinary columns project
/// `Column(reg)`; a USING/NATURAL shared column projects `COALESCE(left, right)` via
/// [`Scope::expand_star_exprs`], so `*` yields the same value a bare reference to that
/// column does (correct on a RIGHT/FULL outer join's unmatched-right row).
fn expand_star_into(
    scope: &Scope,
    table: Option<&str>,
    projections: &mut Vec<EvalExpr>,
    names: &mut Vec<String>,
    result_asts: &mut Vec<Option<Expr>>,
) -> Result<()> {
    for (expr, name) in scope.expand_star_exprs(table)? {
        projections.push(expr);
        result_asts.push(Some(star_source_ast(&name)));
        names.push(name);
    }
    Ok(())
}

/// The synthesized source AST for a `*`-expanded column: a bare (unqualified) column
/// reference by name. Resolving it against the FROM scope recovers the column's
/// collation for ORDER BY. A bare name is unambiguous here because a genuine duplicate
/// across sources is already an error and a USING/NATURAL shared column resolves to its
/// single coalesced value (`Scope::resolve_column_expr`).
fn star_source_ast(name: &str) -> Expr {
    Expr::Column { schema: None, table: None, name: name.to_string(), from_dqs: false }
}

/// Compute the SELECT list's output column NAMES and SOURCE ASTs — expanding
/// `*` / `table.*` — WITHOUT binding the projection. This is byte-for-byte the
/// `(names, result_asts)` list [`compile_projection`] and [`bind_grouped_output`] build
/// (name = the explicit alias, else [`result_column_name`]; source AST = the expression's
/// own AST, or a star column's synthesized bare-column AST via [`star_source_ast`]), lifted
/// out so GROUP BY ordinal/alias resolution can see the output columns BEFORE the group
/// keys are bound (the keys' resolved form feeds the grouping context, so it cannot wait
/// for the projection pass). Keeping this list IDENTICAL to the projection's own is
/// load-bearing: an ordinal `K` must name the same column for GROUP BY, the projection, and
/// ORDER BY. It is a pure function of `(scope, columns)`, and this scope's grouping is off
/// (the aggregate query's pre-projection scope), so its star expansion matches the
/// `without_grouping().expand_star` [`expand_grouped_star`] uses.
fn output_columns(scope: &Scope, columns: &[ResultColumn]) -> Result<(Vec<String>, Vec<Option<Expr>>)> {
    let mut names = Vec::new();
    let mut result_asts = Vec::new();
    for col in columns {
        match col {
            ResultColumn::Expr { expr, alias } => {
                names.push(alias.clone().unwrap_or_else(|| result_column_name(expr)));
                result_asts.push(Some(expr.clone()));
            }
            ResultColumn::Star => push_star_columns(scope, None, &mut names, &mut result_asts)?,
            ResultColumn::TableStar(t) => {
                push_star_columns(scope, Some(t), &mut names, &mut result_asts)?
            }
        }
    }
    Ok((names, result_asts))
}

/// Push the NAME and synthesized bare-column source AST of each `*` / `table.*` column —
/// the name/AST half of [`expand_star_into`] / [`expand_grouped_star`] without the
/// projected value — so [`output_columns`] can list the output columns without binding.
fn push_star_columns(
    scope: &Scope,
    table: Option<&str>,
    names: &mut Vec<String>,
    result_asts: &mut Vec<Option<Expr>>,
) -> Result<()> {
    for (_reg, name) in scope.expand_star(table)? {
        result_asts.push(Some(star_source_ast(&name)));
        names.push(name);
    }
    Ok(())
}

/// Resolve one GROUP BY term to the expression to actually group by, applying the same
/// integer-ordinal / output-alias rules SQLite applies to ORDER BY: lang_select.html
/// specifies that each GROUP BY term is "evaluated ... according to the processing rules
/// stated below for ORDER BY expressions" (a bare positive integer is a 1-based result-column
/// ordinal; a bare name matching a result-column alias is that column). Returns the
/// SELECT-list SOURCE expression for a resolved ordinal/alias (to be bound over the
/// pre-aggregate input row), or the original `key` unchanged for an ordinary expression or
/// an input column.
///
/// Precedence deliberately differs from ORDER BY (see [`match_output_column`]): a name that
/// resolves as an INPUT column stays an input column — SQLite groups by the table column,
/// not a same-named result alias — so the output-alias fallback fires ONLY for a name that
/// is not an input column (checked via [`Scope::try_resolve_column`], which reports a genuine
/// miss as `Ok(None)` while still erroring on an ambiguous match). Only `Literal::Integer` is
/// an ordinal: `TRUE`/`FALSE`/`Real`/`Text`/`NULL`, a subquery, and every other constant or
/// expression are ordinary terms (a constant term groups every row into ONE group), and a
/// negative `-1` is unary-minus over `Integer(1)`, not an `Integer` node, so it too is not an
/// ordinal. An out-of-range ordinal is a loud error naming the offending term's 1-based
/// position (matching SQLite's `"Nth GROUP BY term out of range"`).
fn resolve_group_key<'a>(
    key: &'a Expr,
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &'a [Option<Expr>],
    position: usize,
) -> Result<&'a Expr> {
    let num_out = out_asts.len();
    // 1. A bare positive integer is a 1-based output-column ordinal.
    if let Expr::Literal(Literal::Integer(k)) = key {
        if *k >= 1 && (*k as usize) <= num_out {
            return output_source(out_asts, (*k - 1) as usize);
        }
        return Err(Error::sql(format!(
            "{position}{} GROUP BY term out of range - should be between 1 and {num_out}",
            ordinal_suffix(position),
        )));
    }
    // 2. A bare unqualified name that is NOT an input column, but matches a result-column
    //    alias/name, groups by that output column's source expression. An input column of
    //    the same name wins (`try_resolve_column` -> Ok(Some)); an ambiguous input match
    //    stays the loud error it raises here rather than silently falling back to the alias.
    if let Expr::Column { schema: None, table: None, name, .. } = key {
        if key_scope.try_resolve_column(None, name)?.is_none() {
            if let Some(idx) = out_names.iter().position(|n| n.eq_ignore_ascii_case(name)) {
                return output_source(out_asts, idx);
            }
        }
    }
    // 3. An ordinary expression, an input column, or a non-integer constant binds as-is.
    Ok(key)
}

/// The source expression of output column `idx`, for a resolved GROUP BY ordinal/alias.
/// Every output column carries a source AST today (an explicit expression's own AST, or a
/// star column's synthesized bare-column AST), so `None` is unreachable; it is a loud error
/// rather than a silent mis-group should a future output column ever lack one.
fn output_source(out_asts: &[Option<Expr>], idx: usize) -> Result<&Expr> {
    out_asts[idx].as_ref().ok_or_else(|| {
        Error::sql("GROUP BY term refers to a result column that has no source expression")
    })
}

/// Rewrite an aggregate query's HAVING expression, replacing each bare unqualified name that is
/// NOT an input column but DOES name a SELECT-list output column with that output column's SOURCE
/// expression — SQLite's general name-resolution fallback (`lang_select.html`: an unqualified
/// identifier matching no table column resolves against the result-column alias list). This is
/// [`resolve_group_key`]'s single-expression rule lifted to a WHOLE tree, so an alias resolves at
/// ANY position: inside AND/OR/NOT, a comparison, arithmetic, a unary op, a scalar-function
/// argument, BETWEEN, an IN value-list, CASE, CAST, or COLLATE.
///
/// Precedence matches GROUP BY, NOT ORDER BY: an INPUT column of the same name WINS, so the alias
/// fallback fires only when `key_scope.try_resolve_column(None, name)` reports a genuine miss
/// (`Ok(None)`); an ambiguous input match stays the loud error it raises, and a qualified name
/// (`t.c`) or `*` is never rewritten.
///
/// A subquery boundary — a scalar `Subquery`, `EXISTS (...)`, or an `IN (SELECT ...)` / `IN table`
/// RHS — is cloned WITHOUT descending: a name inside binds against that subquery's (or table
/// function's) own FROM plus correlated outer scope, never the outer query's SELECT-list aliases.
/// A window function's `OVER` spec is likewise cloned untouched — a window function is invalid in
/// HAVING (the binder raises "misuse of window function"), so its contents never reach a plan.
///
/// The rewrite is DETERMINISTIC and a structural no-op for any HAVING that references no alias
/// (nothing matches the rewrite arm, so every node is rebuilt identically) — that is the safety
/// guarantee: a query whose HAVING uses no alias is byte-for-byte unaffected. And because the SAME
/// resolved expression is fed to both the aggregate PRE-SCANS ([`single_minmax_special`],
/// [`count_aggregates`]) and the binder, a substituted `count(*)` alias becomes a real
/// (non-deduped) aggregate call BOTH observe, keeping the captured-bare-column / window register
/// offsets consistent (see [`compile_aggregate`]). Descending into an aggregate's own arguments is
/// safe for that count: any aggregate a substitution introduces there is a nested aggregate, which
/// the binder rejects LOUDLY before a plan is built, so it can never silently miscount.
fn resolve_having_aliases(
    expr: &Expr,
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &[Option<Expr>],
) -> Result<Expr> {
    match expr {
        // THE rewrite: a bare unqualified name that does not resolve as an input column but
        // matches an output column name/alias -> that output column's source expression.
        Expr::Column { schema: None, table: None, name, .. } => {
            if key_scope.try_resolve_column(None, name)?.is_none() {
                if let Some(idx) = out_names.iter().position(|n| n.eq_ignore_ascii_case(name)) {
                    return Ok(output_source(out_asts, idx)?.clone());
                }
            }
            Ok(expr.clone())
        }
        // A qualified column and the leaves carry no rewritable child.
        Expr::Column { .. } | Expr::Literal(_) | Expr::BindParam(_) | Expr::Raise(_) => {
            Ok(expr.clone())
        }
        // Subquery boundaries: names inside resolve against the subquery's own scope, not the
        // outer SELECT-list aliases — clone without descending.
        Expr::Exists { .. } | Expr::Subquery(_) => Ok(expr.clone()),
        Expr::Unary { op, expr: inner } => {
            Ok(Expr::Unary { op: *op, expr: rewrite_child(inner, key_scope, out_names, out_asts)? })
        }
        Expr::Binary { op, left, right } => Ok(Expr::Binary {
            op: *op,
            left: rewrite_child(left, key_scope, out_names, out_asts)?,
            right: rewrite_child(right, key_scope, out_names, out_asts)?,
        }),
        Expr::Cast { expr: inner, type_name } => Ok(Expr::Cast {
            expr: rewrite_child(inner, key_scope, out_names, out_asts)?,
            type_name: type_name.clone(),
        }),
        Expr::Collate { expr: inner, collation } => Ok(Expr::Collate {
            expr: rewrite_child(inner, key_scope, out_names, out_asts)?,
            collation: collation.clone(),
        }),
        Expr::Like { negated, kind, lhs, rhs, escape } => Ok(Expr::Like {
            negated: *negated,
            kind: *kind,
            lhs: rewrite_child(lhs, key_scope, out_names, out_asts)?,
            rhs: rewrite_child(rhs, key_scope, out_names, out_asts)?,
            escape: rewrite_child_opt(escape, key_scope, out_names, out_asts)?,
        }),
        Expr::Between { negated, expr: inner, low, high } => Ok(Expr::Between {
            negated: *negated,
            expr: rewrite_child(inner, key_scope, out_names, out_asts)?,
            low: rewrite_child(low, key_scope, out_names, out_asts)?,
            high: rewrite_child(high, key_scope, out_names, out_asts)?,
        }),
        Expr::In { negated, expr: inner, rhs } => Ok(Expr::In {
            negated: *negated,
            expr: rewrite_child(inner, key_scope, out_names, out_asts)?,
            rhs: rewrite_in_rhs(rhs, key_scope, out_names, out_asts)?,
        }),
        Expr::Case { operand, whens, else_expr } => {
            let operand = rewrite_child_opt(operand, key_scope, out_names, out_asts)?;
            let mut new_whens = Vec::with_capacity(whens.len());
            for (w, t) in whens {
                new_whens.push((
                    resolve_having_aliases(w, key_scope, out_names, out_asts)?,
                    resolve_having_aliases(t, key_scope, out_names, out_asts)?,
                ));
            }
            let else_expr = rewrite_child_opt(else_expr, key_scope, out_names, out_asts)?;
            Ok(Expr::Case { operand, whens: new_whens, else_expr })
        }
        Expr::IsNull(inner) => {
            Ok(Expr::IsNull(rewrite_child(inner, key_scope, out_names, out_asts)?))
        }
        Expr::NotNull(inner) => {
            Ok(Expr::NotNull(rewrite_child(inner, key_scope, out_names, out_asts)?))
        }
        Expr::Parenthesized(list) => {
            Ok(Expr::Parenthesized(rewrite_exprs(list, key_scope, out_names, out_asts)?))
        }
        Expr::Function { name, distinct, args, filter, over, order_by } => {
            let args = match args {
                FunctionArgs::List(list) => {
                    FunctionArgs::List(rewrite_exprs(list, key_scope, out_names, out_asts)?)
                }
                FunctionArgs::Star => FunctionArgs::Star,
                FunctionArgs::Empty => FunctionArgs::Empty,
            };
            let filter = rewrite_child_opt(filter, key_scope, out_names, out_asts)?;
            let mut new_order_by = Vec::with_capacity(order_by.len());
            for term in order_by {
                new_order_by.push(OrderingTerm {
                    expr: resolve_having_aliases(&term.expr, key_scope, out_names, out_asts)?,
                    collation: term.collation.clone(),
                    order: term.order,
                    nulls: term.nulls,
                });
            }
            Ok(Expr::Function {
                name: name.clone(),
                distinct: *distinct,
                args,
                filter,
                // A window function is invalid in HAVING and errors regardless of its OVER
                // spec's contents, so the spec is cloned untouched (no alias resolution inside a
                // PARTITION BY / ORDER BY / frame that never reaches a plan).
                over: over.clone(),
                order_by: new_order_by,
            })
        }
    }
}

/// Rewrite a boxed child expression, preserving the `Box` ([`resolve_having_aliases`]).
fn rewrite_child(
    e: &Expr,
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &[Option<Expr>],
) -> Result<Box<Expr>> {
    Ok(Box::new(resolve_having_aliases(e, key_scope, out_names, out_asts)?))
}

/// Rewrite an optional boxed child expression; a `None` stays `None`.
fn rewrite_child_opt(
    e: &Option<Box<Expr>>,
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &[Option<Expr>],
) -> Result<Option<Box<Expr>>> {
    match e {
        Some(inner) => Ok(Some(rewrite_child(inner, key_scope, out_names, out_asts)?)),
        None => Ok(None),
    }
}

/// Rewrite each expression in a list (a `Parenthesized` row / an IN value-list / a function
/// argument list).
fn rewrite_exprs(
    list: &[Expr],
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &[Option<Expr>],
) -> Result<Vec<Expr>> {
    list.iter().map(|e| resolve_having_aliases(e, key_scope, out_names, out_asts)).collect()
}

/// Rewrite the scalar operands of an IN right-hand side. A bare value LIST (`x IN (c, 2)`) has its
/// operands rewritten — it is not a subquery — while an `IN (SELECT ...)` and an `IN table(args)`
/// table/table-function reference are cloned untouched: a name inside binds against the subquery's
/// or table function's own scope, never the outer query's SELECT-list aliases.
fn rewrite_in_rhs(
    rhs: &InRhs,
    key_scope: &Scope,
    out_names: &[String],
    out_asts: &[Option<Expr>],
) -> Result<InRhs> {
    match rhs {
        InRhs::List(items) => Ok(InRhs::List(rewrite_exprs(items, key_scope, out_names, out_asts)?)),
        InRhs::Select(_) | InRhs::Table { .. } => Ok(rhs.clone()),
    }
}

/// The English ordinal suffix for a 1-based position (`1`->"st", `2`->"nd", `3`->"rd",
/// `4`->"th", and `11`/`12`/`13`->"th"), for the GROUP BY out-of-range message.
fn ordinal_suffix(n: usize) -> &'static str {
    if (11..=13).contains(&(n % 100)) {
        "th"
    } else {
        match n % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    }
}

/// The aggregate query: bind the group keys (input scope), then the projection and
/// HAVING against the post-aggregate `[group_keys.., agg_results..]` layout, collecting
/// aggregate calls.
///
/// When the projection carries a WINDOW function (`f(...) OVER (...)`), a post-aggregate
/// windowing stage is inserted so the pipeline is `Aggregate → Window → Project`: the
/// window runs once per output GROUP row, AFTER `GROUP BY`/`HAVING`, and its args /
/// `PARTITION BY` / `ORDER BY` bind against the aggregate's OUTPUT relation — a group key
/// resolves to its key register, an aggregate to an aggregate result register. (An
/// aggregate written inside an OVER spec is collected as its own `AggregateCall`, NOT
/// deduped against an identical one in the SELECT list — matching the binder's universal
/// no-dedup convention — so `rank() OVER (ORDER BY sum(x))` alongside `sum(x)` computes
/// `sum(x)` in two result registers; both hold the same value, so the result is correct.)
/// That needs the post-aggregate row width (`num_keys + num_aggregates`) to place each
/// window result register, but a window's OVER spec can itself contain aggregates (e.g.
/// `rank() OVER (ORDER BY sum(x))`), so the total aggregate count is only known AFTER
/// binding. Hence the TWO-PASS binding via [`bind_grouped_output`]: a trial pass to count,
/// then (only when a window call was found) a final pass with the true width. A query with
/// no window call binds ONCE and produces the byte-for-byte plan the single-pass path did.
fn compile_aggregate(
    ctx: &mut PlanCtx,
    scope: &Scope,
    columns: &[ResultColumn],
    group_by: &[Expr],
    having: &Option<Expr>,
    order_by: &[OrderingTerm],
    input: PlanNode,
) -> Result<Compiled> {
    // Group keys bind against the pre-aggregate input row (no grouping context) and no
    // windowing context: a window function in GROUP BY is a loud "misuse of window
    // function" error (mirroring WHERE/HAVING), never a collected WindowFunc that would
    // resolve to an out-of-range group-key register.
    let key_scope = scope.without_windowing();
    // GROUP BY ordinal/alias resolution (lang_select.html: a GROUP BY term is evaluated
    // "according to the processing rules stated below for ORDER BY expressions") needs the
    // STAR-EXPANDED output column list — the SAME (names, source-ASTs) the projection and
    // ORDER BY use — so `GROUP BY 1` names the first output column and `GROUP BY <alias>`
    // its aliased column. It is computed HERE, before the keys are bound, because a resolved
    // key's expression feeds the grouping context below (`resolved_keys`).
    let (out_names, out_asts) = output_columns(scope, columns)?;
    // `output_columns` pushes to both vectors in lockstep, and both alias consumers below
    // (`resolve_group_key`, `resolve_having_aliases`) index `out_asts` with an index taken from
    // `out_names.position(..)` — so a length skew would index past `out_asts` (see
    // `output_source`). Pin that invariant here, once, at the point both are produced.
    debug_assert_eq!(out_names.len(), out_asts.len(), "output name/source lists must stay in lockstep");
    // The RESOLVED HAVING expression: each bare unqualified name that is NOT an input column but
    // matches a SELECT-list output alias/name is replaced by that column's source expression —
    // SQLite's name-resolution fallback (`lang_select.html`), the HAVING analog of the GROUP BY
    // ordinal/alias resolution below. Computed HERE, right after the output columns are known, and
    // threaded (as `&having_resolved`) into EVERY aggregate pre-scan and bind site below, so a
    // substituted aggregate alias (`HAVING c > 1` with `c` = `count(*)`) is a real aggregate that
    // BOTH the pre-scan count and the binder observe — keeping their aggregate register offsets in
    // lockstep. It is a structural no-op for a HAVING that references no alias, so such a query is
    // byte-for-byte unaffected.
    let having_resolved: Option<Expr> = match having {
        Some(h) => Some(resolve_having_aliases(h, &key_scope, &out_names, &out_asts)?),
        None => None,
    };
    // The RESOLVED group-key expressions (an ordinal/alias replaced by its output column's
    // source expression, every other key unchanged). This — not the raw ordinal/alias AST —
    // is what the grouping context matches projection sub-expressions against
    // (`bind::bind_grouped_node` step 1: `e == key`), so `SELECT a%2 AS p ... GROUP BY 1`
    // has its projected `a%2` recognized as the group key.
    let mut resolved_keys = Vec::with_capacity(group_by.len());
    let mut group_by_bound = Vec::with_capacity(group_by.len());
    let mut group_collations = Vec::with_capacity(group_by.len());
    let mut col_to_group = HashMap::new();
    for (i, key) in group_by.iter().enumerate() {
        // Bind the RESOLVED expression (over the pre-aggregate input row) so grouping, the
        // operand collation, and the `col_to_group` map are all computed on it — `GROUP BY 1`
        // groups by the first output column's source, never the constant integer 1.
        let resolved = resolve_group_key(key, &key_scope, &out_names, &out_asts, i + 1)?;
        let bound = bind_expr(&key_scope, ctx, resolved)?;
        group_collations.push(operand_collation(&key_scope, resolved)?);
        if let EvalExpr::Column(r) = &bound {
            col_to_group.entry(*r).or_insert(i);
        }
        resolved_keys.push(resolved.clone());
        group_by_bound.push(bound);
    }

    // The single-min/max "bare columns" special case (lang_select.html §2.5): when the
    // query has exactly one min()/max() aggregate (alongside any number of OTHER
    // aggregates), bare columns are captured from that extremum's row rather than
    // rejected. It must be known BEFORE the projection is bound (so a bare column
    // resolves to a captured register instead of erroring). The captured columns follow
    // ALL aggregate results in the operator's output row, so they start at
    // `num_keys + num_aggregates` — the pre-scan yields that count up front (it
    // enumerates the same aggregates, in the same order, the binder will collect), so the
    // offset is computable now with no post-bind fixup. The pre-scan counts only the
    // projection + HAVING aggregates; a post-aggregate window call can add more (in its
    // OVER spec), so with a window call present this base can UNDERCOUNT — but that
    // combination (a captured bare column AND a window call) is rejected below before any
    // plan is built, so the potentially-wrong base never reaches a plan.
    let minmax_special = single_minmax_special(ctx.registry, columns, &having_resolved);
    let num_keys = group_by.len();
    // Bare columns (a column neither inside an aggregate nor in GROUP BY) are a documented
    // SQLite extension (lang_select.html §2.5) valid in ANY aggregate query, not just the
    // single-min/max case. Captured bare columns follow all aggregate results, so they start
    // at `num_keys + num_aggregates`; the count comes from the SAME `walk_agg_calls`
    // traversal the binder collects in (so the offset needs no post-bind fixup, exactly as
    // for the min/max pre-scan). Enabling the base for every aggregate query lets a bare
    // column bind to a captured slot instead of the "must appear in GROUP BY" error; a query
    // with NO bare column captures nothing and is unaffected.
    let num_aggregates = count_aggregates(ctx.registry, columns, &having_resolved);
    let bare_base = Some(num_keys + num_aggregates);
    // The PROVISIONAL post-aggregate row width `[group_keys.., agg_results..]` a CORRELATED
    // subquery in the projection / HAVING binds against as its `outer_width` in pass 1 (the
    // executor prepends this row; see `Grouping::subplan_outer_width`). It is exact unless a
    // §2.5 capture, a post-aggregate window call, or an ORDER BY-introduced aggregate widens
    // the real row; when such a subquery is present and the row is wider, `compile_aggregate`
    // re-binds it with the true width (below) so its own sources sit past the prepended row.
    let subplan_outer_width = num_keys + num_aggregates;

    // The MIN/MAX index/rowid seek fast path (optoverview.html §13) fires ONLY for a query
    // whose ONE aggregate is a single min()/max(); capture its direction now, before
    // `minmax_special` is consumed by `split_bare_capture` below. `None` unless there is
    // exactly one aggregate AND it is that single min/max — so a query with any other
    // aggregate shape never reaches `minmax_index::try_optimize`'s further checks.
    let minmax_is_max: Option<bool> =
        minmax_special.as_ref().filter(|s| s.num_aggregates == 1).map(|s| s.is_max);

    // A post-aggregate `OVER window-name` resolves against the query's `WINDOW` clause,
    // carried on the FROM-scope's windowing context (built in `compile::select`).
    let named: &[(String, WindowSpec)] = scope.windowing.map_or(&[], |w| w.named);

    // PASS 1 (trial): bind the projection + HAVING, collecting the aggregates, any §2.5
    // captures, and any post-aggregate window calls. The window input width is provisional
    // (`num_keys`) — it only sets a window call's OWN result register, which pass 2
    // re-derives with the true post-aggregate width; pass 1's job is to COUNT. Its
    // side-table mutations (subqueries, params) are rolled back before pass 2 via the
    // savepoint, exactly as the correlated-subquery re-bind does.
    let sp = ctx.savepoint();
    let pass1 = bind_grouped_output(
        ctx, scope, columns, &having_resolved, order_by, &resolved_keys, &col_to_group, num_keys,
        bare_base, num_keys, subplan_outer_width, subplan_outer_width, named,
    )?;

    // The output-column `(names, source-ASTs)` that GROUP BY ordinal/alias resolution used
    // (`output_columns`, above) MUST be byte-for-byte the list the projection itself just
    // built in `bind_grouped_output`: an ordinal `K` has to name the SAME column for GROUP BY
    // as for the projection and ORDER BY. They are two independent loops that agree only
    // because both derive names from the same `expand_star` / `result_column_name` / alias
    // primitives; this pins that coupling so a future divergence at one site fails LOUDLY here
    // (debug/test) instead of silently mis-grouping an ordinal. (Pass 2 is a pure re-bind of
    // the same columns, already backstopped to match pass 1's shape below, so asserting pass 1
    // covers both paths.)
    debug_assert_eq!(
        out_names, pass1.names,
        "GROUP BY output-column names diverged from the projection's own names"
    );
    debug_assert_eq!(
        out_asts, pass1.result_asts,
        "GROUP BY output-column source ASTs diverged from the projection's own"
    );

    // Pass 1 bound any CORRELATED subquery in the projection / HAVING / ORDER BY against the
    // PROVISIONAL post-aggregate width `subplan_outer_width = num_keys + num_aggregates`. That
    // width is exact only when nothing widens the row the executor prepends as the subquery's
    // outer; three documented shapes widen it — a §2.5 captured bare column, a post-aggregate
    // window call, and an ORDER BY-introduced aggregate. When such a subquery is present and
    // the real row is wider, we RE-BIND with the true width so the subquery's own FROM sources
    // sit at a non-overlapping base and its outer references still read the prepended row.
    let saw_corr = pass1.saw_correlated_subplan;
    let has_window = !pass1.window_funcs.is_empty();
    let num_captured = pass1.captured_regs.len();
    let total_aggs = pass1.aggregates.len();
    let num_window = pass1.window_funcs.len();

    // A §2.5 bare-column capture together with a post-aggregate window call is not supported:
    // the captured columns' and the window results' output offsets both depend on the
    // aggregate count, which a window OVER spec can change (the pre-scan does not see it) — a
    // fragile interaction rejected LOUDLY rather than emit a wrong register. (Wrap the bare
    // column in an aggregate, or add it to GROUP BY.) Independent of any subquery.
    if has_window && num_captured != 0 {
        return Err(Error::sql(
            "a window function together with a bare column (the lang_select.html §2.5 special case) is not yet supported",
        ));
    }

    // The TRUE widths of the rows the executor prepends as the `outer` of a correlated subquery.
    // They differ BY HOST CLAUSE on the window path (Aggregate → Window → Project; see
    // `Grouping::subplan_outer_width`):
    //   * HAVING runs in the aggregate operator over `[group_keys.., agg_results.., captured..]`
    //     — the PRE-window row (the §2.5 capture + window combination is rejected above, so on
    //     the window path `num_captured == 0`);
    //   * the projection / ORDER BY run over the post-WINDOW row
    //     `[group_keys.., agg_results.., captured.., window_results..]`.
    // Off the window path `num_window == 0`, so the two coincide (a correlated HAVING subquery in
    // a non-window aggregate query needs no special handling — it always did work). This split is
    // what closes the HAVING + window combination that used to be loud-rejected here.
    let having_outer_width = num_keys + total_aggs + num_captured;
    let projection_outer_width = having_outer_width + num_window;

    // Re-bind when the window stage is present (its result registers were provisional in pass 1)
    // OR a correlated subquery was bound against a width that now undercounts the real row (either
    // clause's true width moved off the provisional). Otherwise pass 1 is already the final,
    // correctly-bound plan — the common case, byte-for-byte the single-pass output, including a
    // PLAIN correlated subquery whose provisional width was already exact.
    let need_rebind = has_window
        || (saw_corr
            && (having_outer_width != subplan_outer_width
                || projection_outer_width != subplan_outer_width));

    // The window operator appends each result column after the aggregate row, so on the window
    // path the window-input width is the aggregate output row width `num_keys + total_aggs`;
    // off the window path pass 1 used `num_keys` and any re-bind keeps that (no window call
    // reads it).
    let window_input_width = if has_window { num_keys + total_aggs } else { num_keys };

    let final_pass = if need_rebind {
        ctx.restore(sp);
        let pass2 = bind_grouped_output(
            ctx, scope, columns, &having_resolved, order_by, &resolved_keys, &col_to_group,
            num_keys, bare_base, window_input_width, having_outer_width, projection_outer_width,
            named,
        )?;
        // The re-bind is a pure function of the same inputs (only the correlated-subquery base
        // and the window-result registers move), so its shape MUST match pass 1's counts; a
        // divergence would mean a register was placed against a width that no longer holds.
        // These asserts turn that into a loud failure (debug/test) rather than a wrong row.
        debug_assert_eq!(
            pass2.aggregates.len(), total_aggs,
            "aggregate count changed between the trial and final binding passes"
        );
        debug_assert_eq!(
            pass2.window_funcs.len(), num_window,
            "window-call count changed between the trial and final binding passes"
        );
        debug_assert_eq!(
            pass2.captured_regs.len(), num_captured,
            "§2.5 capture count changed between the trial and final binding passes"
        );
        pass2
    } else {
        pass1
    };

    if !has_window {
        // No post-aggregate window call: build the plain Aggregate. Bare columns resolve to
        // captured slots via the single-min/max extremum row (`minmax_bare`) or, in the
        // general case, an arbitrary (first) row of each group (`bare_arbitrary`) — mutually
        // exclusive, both `None` when nothing was captured.
        let (minmax_bare, bare_arbitrary) =
            split_bare_capture(minmax_special, &final_pass.aggregates, final_pass.captured_regs);
        // Implicit ascending GROUP-key order for an unordered GROUP BY. Real SQLite groups via
        // a sort (datatype3.html §6: GROUP BY grouping uses the same value comparison as
        // sorting), so an unordered `GROUP BY` — one with NO explicit outer ORDER BY — comes
        // back ASCENDING by the grouping keys even WITHOUT an ORDER BY, mirroring the
        // compound-SELECT implicit-sort rule (`compile::compound`). Build the keys now, while
        // `group_collations` is still in hand (it is moved into `agg` below); each key reuses
        // its grouping collation so the sort order matches the grouping comparison — a NOCASE
        // group key sorts case-insensitively — and `nulls_first: None` lets the Sort apply the
        // SQL default (NULLs first for ASC, datatype3.html §4.1).
        //
        // Gated on `num_keys > 0` (an empty GROUP BY is a single row — nothing to order — and
        // that same guard excludes the MIN/MAX seek below, which fires only for `num_keys == 0`)
        // and on an EMPTY `order_by`: an explicit ORDER BY takes over via `assemble_output`'s
        // Sort. RESIDUAL (a pre-existing divergence this fix intentionally does NOT close): with
        // a PARTIAL explicit order — e.g. `GROUP BY x, y ORDER BY x` — real SQLite still groups
        // by the FULL key (x, y) and layers `ORDER BY x` as a stable sort on top, so equal-x
        // groups stay ordered by y; here the explicit-ORDER-BY path sorts by x ALONE, leaving
        // equal-x groups in the operator's first-appearance order. Closing it means also sorting
        // by the group keys beneath a partial ORDER BY (a secondary refinement) — the no-ORDER-BY
        // witness is the primary fix and the explicit-ORDER-BY path is deliberately untouched.
        let implicit_sort = if order_by.is_empty() && num_keys > 0 {
            Some(implicit_group_sort_keys(&group_collations))
        } else {
            None
        };
        let agg = Aggregate {
            input: Box::new(input),
            group_by: group_by_bound,
            group_collations,
            aggregates: final_pass.aggregates,
            having: final_pass.having_bound,
            minmax_bare,
            bare_arbitrary,
        };
        // MIN/MAX index/rowid seek fast path (optoverview.html §13): when this aggregate is a
        // single bare-column MIN/MAX over a plain full-table scan whose extremum a rowid or
        // ascending BINARY index already presents in order, replace the O(n) full-scan
        // aggregate with an O(log n) seek that yields the byte-identical extremum. Declined
        // when a correlated subquery is present: that seek yields the extremum row in place of
        // the full aggregate row, and its interaction with a subquery's prepended
        // post-aggregate outer row is not proven byte-identical, so we keep the correct
        // full-scan aggregate. A query with no correlated subquery fires exactly as before
        // (byte-for-byte unaffected). It fires only for `num_keys == 0`, so it never coexists
        // with the implicit group sort below (guarded on `num_keys > 0`).
        let minmax_arg = if saw_corr { None } else { minmax_is_max };
        let agg_node = minmax_index::try_optimize(scope, ctx.catalog, agg, minmax_arg);
        // Apply the implicit group-key Sort ON TOP of the aggregate output, so it becomes the
        // projection's input: `Project { Sort { Aggregate } }`. Placing it UNDER the Project is
        // load-bearing — SQLite orders on the group-key VALUES, which the projection can drop
        // or reorder (`SELECT count(*) FROM t GROUP BY x` projects `x` away). A full stable
        // sort (`limit: None`): any enclosing LIMIT sits above the row-preserving Project in
        // `assemble_output`, so pushing a top-k retention bound down here is a pure (optional)
        // optimization, never a correctness requirement.
        let agg_node = match implicit_sort {
            Some(keys) => PlanNode::Sort { input: Box::new(agg_node), keys, limit: None },
            None => agg_node,
        };
        return Ok(Compiled {
            input: agg_node,
            projections: final_pass.projections,
            names: final_pass.names,
            result_asts: final_pass.result_asts,
            is_agg: true,
            order_pre: Some(final_pass.order_resolved),
        });
    }

    // A post-aggregate window call is present: Aggregate → Window → Project.
    //
    // RESIDUAL (intentionally not implemented): the implicit ascending group-key sort added on
    // the non-window path above is NOT applied here. When the window HAS its own ordering the
    // group key does not govern output order — e.g.
    // `SELECT x, sum(y), row_number() OVER (ORDER BY sum(y)) FROM t GROUP BY x` comes back in
    // sum(y) order, not x order — so a blanket group-key sort under the window would be wrong,
    // and the correct placement relative to the window stage is case-dependent. (For a window
    // with NO ordering — e.g. `count(*) OVER ()` — real SQLite would still present groups in
    // group-key order, so that sub-case is conservatively scoped out too rather than special-
    // cased.) An explicit outer ORDER BY is still honored (via `assemble_output`'s Sort over
    // `order_resolved`); only the WINDOW + GROUP BY + no-ORDER-BY case keeps the operator's
    // first-appearance order, which can differ from real SQLite. Scoped out here rather than
    // risk mis-ordering; the non-window case (the case that matters here) is the high-value fix.
    //
    // No captured columns on this path (the §2.5 + window combination is rejected above), so the
    // operator runs the ordinary aggregation with no per-row capture.
    let agg_node = PlanNode::Aggregate(Aggregate {
        input: Box::new(input),
        group_by: group_by_bound,
        group_collations,
        aggregates: final_pass.aggregates,
        having: final_pass.having_bound,
        minmax_bare: None,
        bare_arbitrary: None,
    });
    // The window runs over the aggregate's output rows, appending one column per call at
    // `Column(num_keys + total_aggs + k)` — the register the projection read for each call.
    let windowed = PlanNode::Window(Window {
        input: Box::new(agg_node),
        functions: final_pass.window_funcs,
    });
    Ok(Compiled {
        input: windowed,
        projections: final_pass.projections,
        names: final_pass.names,
        result_asts: final_pass.result_asts,
        is_agg: true,
        order_pre: Some(final_pass.order_resolved),
    })
}

/// The implicit ascending sort keys for an unordered `GROUP BY` (one with no explicit
/// outer `ORDER BY`): one key per group column — `Column(0)..Column(n)` for the `n` entries
/// of `group_collations` — the group-key values, which occupy the first `n` columns of the
/// aggregate's output row `[group_keys.., agg_results.., captured..]`. Deriving BOTH the key
/// count and each key's collation from the single `group_collations` slice makes
/// `keys.len() == group_collations.len()` true by construction: there is no separate length
/// to disagree, so no index can fall out of range.
///
/// Each key is ASCENDING with `nulls_first: None` (so the Sort applies SQLite's default:
/// NULLs first for ASC, datatype3.html §4.1) and reuses that key's grouping collation, so
/// the ordering matches the grouping comparison exactly — the direct analog of reusing the
/// compound dedup collation in `compile::compound`. The caller places the resulting Sort
/// over the aggregate output and UNDER the projection, because SQLite orders on the
/// group-key values before the projection can drop or reorder them.
fn implicit_group_sort_keys(group_collations: &[Collation]) -> Vec<SortKey> {
    group_collations
        .iter()
        .enumerate()
        .map(|(i, &collation)| SortKey {
            expr: EvalExpr::Column(i),
            desc: false,
            nulls_first: None,
            collation,
        })
        .collect()
}

/// Everything one binding pass over an aggregate query's projection + HAVING collects
/// ([`bind_grouped_output`]): the projected exprs / output names / source ASTs and the
/// bound HAVING, plus what the grouping and windowing contexts gathered — the aggregate
/// calls, the §2.5 captured INPUT registers, and the post-aggregate window calls.
struct GroupedOutput {
    projections: Vec<EvalExpr>,
    names: Vec<String>,
    result_asts: Vec<Option<Expr>>,
    having_bound: Option<EvalExpr>,
    aggregates: Vec<AggregateCall>,
    captured_regs: Vec<usize>,
    window_funcs: Vec<WindowFunc>,
    /// Each ORDER BY term resolved against this post-aggregate relation (same length and
    /// order as the query's ORDER BY), an output-column reference or a bound hidden
    /// expression — see [`OrderResolved`]. Empty when the query has no ORDER BY.
    order_resolved: Vec<OrderResolved>,
    /// Whether this pass compiled a CORRELATED subquery ANYWHERE in the projection / HAVING /
    /// ORDER BY against the post-aggregate grouping context (read from
    /// `Grouping::saw_correlated_subplan`). `compile_aggregate` uses it to decide whether the
    /// provisional `subplan_outer_width` must be corrected by a re-bind (the subquery's outer
    /// row is wider than `num_keys + num_aggregates` under a §2.5 capture / window call /
    /// ORDER BY-introduced aggregate), and to keep the MIN/MAX seek off a correlated query.
    saw_correlated_subplan: bool,
}

/// Bind an aggregate query's projection and HAVING against the post-aggregate
/// `[group_keys.., agg_results.. (, captured..)]` layout, ONE pass. It builds the
/// short-lived [`Grouping`] (redirecting group keys, collecting aggregate calls, and — when
/// `bare_base` is `Some` — the §2.5 bare-column capture) and, over it, a [`Windowing`]
/// context so a window call in the PROJECTION binds against that same post-aggregate
/// relation and appends its result column at `Column(window_input_width + k)`.
///
/// HAVING is bound through a windowing-STRIPPED scope: HAVING runs BEFORE the window stage,
/// so a window function there stays the loud "misuse of window function" error (matching
/// WHERE/GROUP BY/ON), never a collected call. Grouping stays on for HAVING (its aggregates
/// and group keys resolve as usual).
///
/// Called up to TWICE by [`compile_aggregate`] — a trial pass to COUNT, then (only when a
/// window call is found) a final pass with the true post-aggregate width — so the caller
/// wraps the two in a [`PlanCtx`] savepoint/restore to keep subquery ids / parameter
/// numbers identical across the retry.
#[allow(clippy::too_many_arguments)]
fn bind_grouped_output(
    ctx: &mut PlanCtx,
    scope: &Scope,
    columns: &[ResultColumn],
    having: &Option<Expr>,
    order_by: &[OrderingTerm],
    resolved_keys: &[Expr],
    col_to_group: &HashMap<usize, usize>,
    num_keys: usize,
    bare_base: Option<usize>,
    window_input_width: usize,
    having_outer_width: usize,
    projection_outer_width: usize,
    named: &[(String, WindowSpec)],
) -> Result<GroupedOutput> {
    let grouping = Grouping {
        // The RESOLVED group-key expressions (an ordinal/alias already replaced by its
        // output column's source expression in `compile_aggregate`), so `bind_grouped_node`
        // structurally matches a projection sub-expression against the expression actually
        // grouped by — `GROUP BY 1` behaves exactly like `GROUP BY <that source expr>`.
        key_asts: resolved_keys,
        // A fresh map per pass (cheap — one entry per plain-column group key); the caller
        // computes it once and both passes share the same shape.
        col_to_group: col_to_group.clone(),
        num_keys,
        aggregates: RefCell::new(Vec::new()),
        bare_capture: bare_base.map(BareCapture::new),
        // The post-aggregate width a correlated subquery uses as its outer_width. Starts at the
        // projection/ORDER BY width; the HAVING bind below re-points it to the pre-window
        // `having_outer_width` and restores it. Provisional in pass 1; `compile_aggregate`
        // re-binds with the true per-clause widths when a §2.5 capture / window call / ORDER BY
        // aggregate widens the real row.
        subplan_outer_width: Cell::new(projection_outer_width),
        saw_correlated_subplan: Cell::new(false),
    };
    let windowing = Windowing::new(window_input_width, named);
    // Build the grouped scope by struct literal: it lives only for this binding pass (a
    // lifetime shorter than the catalog `'a`), so it cannot go through a `&'a` method on
    // `Scope`. Carry `saw_correlated` / `correlated_cols` / `nondeterministic` through so an
    // outer reference or non-deterministic function inside a projection/HAVING (or an
    // aggregate ARGUMENT, bound via `without_grouping`) of a correlated aggregate subquery
    // is still recorded, and the FROM's NATURAL/USING coalesced columns so a shared join
    // column resolves the same way it does before aggregation.
    //
    // INVARIANT `grouping: Some` here is load-bearing for correlated-subquery handling in
    // another file: `compile::select::compile_subplan` uses `parent.grouping` as the exact
    // discriminator for a POST-aggregate correlated subquery — its runtime `outer` is the
    // post-aggregate `[keys.., results..]` row, so it takes `outer_width` from
    // `Grouping::subplan_outer_width` (not `total_width()`) and its outer references remap
    // through `Scope::remap_post_aggregate` to their post-aggregate key registers. The ONLY
    // post-aggregate bind sites — this projection and the HAVING below — must stay on a
    // `grouping: Some` scope, and every pre-aggregate site (group keys, and aggregate
    // ARGUMENTS via `without_grouping()`) must stay `grouping: None`. Bind a
    // projection/HAVING subexpression through a `None` scope and a correlated subquery there
    // would silently mis-bind (FROM-row width/registers instead of post-aggregate).
    // `correlated_aggregate_subquery_under_a_nonaggregate_parent_is_correlated` and the
    // aggregate correlated-subquery tests in `compile::select` pin this coupling.
    let gscope = Scope {
        sources: scope.sources,
        coalesced: scope.coalesced,
        parent: scope.parent,
        grouping: Some(&grouping),
        saw_correlated: scope.saw_correlated,
        correlated_cols: scope.correlated_cols,
        nondeterministic: scope.nondeterministic,
        // The projection is a windowing site: a window call here binds against the
        // post-aggregate row (grouping still `Some`, so its args / PARTITION BY / ORDER BY
        // resolve group keys and aggregate results) and appends its result column.
        windowing: Some(&windowing),
    };

    let mut projections = Vec::new();
    let mut names = Vec::new();
    let mut result_asts = Vec::new();
    for col in columns {
        match col {
            ResultColumn::Expr { expr, alias } => {
                projections.push(bind_expr(&gscope, ctx, expr)?);
                names.push(alias.clone().unwrap_or_else(|| result_column_name(expr)));
                result_asts.push(Some(expr.clone()));
            }
            ResultColumn::Star => expand_grouped_star(
                &gscope, &grouping, None, &mut projections, &mut names, &mut result_asts,
            )?,
            ResultColumn::TableStar(t) => expand_grouped_star(
                &gscope, &grouping, Some(t), &mut projections, &mut names, &mut result_asts,
            )?,
        }
    }

    // HAVING binds WITHOUT windowing (see the fn doc): it runs before the window stage, so
    // a window function in HAVING is the loud "misuse of window function" error, not a
    // collected call. Grouping stays on, so aggregates / group keys in HAVING resolve.
    //
    // HAVING runs INSIDE the aggregate operator, over its emitted `[keys.., aggs.., captured..]`
    // row (exec `ops/aggregate.rs`) — BEFORE the window stage — so a correlated subquery in
    // HAVING is prepended a NARROWER outer row than one in the projection / ORDER BY (which run
    // post-Window over `[keys.., aggs.., captured.., window..]`). Re-point the shared outer-width
    // Cell to the pre-window `having_outer_width` for the HAVING bind, then restore the
    // projection width for ORDER BY. Off the window path the two widths coincide, so a no-window
    // query is byte-identical. `saw_correlated_subplan` keeps accumulating across projection +
    // HAVING + ORDER BY (a correlated HAVING subquery still marks the pass for the re-bind).
    let having_scope = gscope.without_windowing();
    grouping.subplan_outer_width.set(having_outer_width);
    let having_bound = match having {
        Some(h) => Some(bind_expr(&having_scope, ctx, h)?),
        None => None,
    };
    grouping.subplan_outer_width.set(projection_outer_width);

    // ORDER BY (lang_select.html §2.4): in an aggregate query, an ORDER BY term is
    // evaluated over the post-aggregate relation just like the projection — it may name a
    // group key, an aggregate, or an expression thereof that the SELECT list does NOT
    // project (`SELECT count(*) ... GROUP BY cat ORDER BY cat`, `... ORDER BY count(*)`).
    // A term that matches an output column (ordinal / name / structural) reuses it; any
    // other term binds HERE, in the same `gscope`, so it can resolve those post-aggregate
    // names (and collect any new aggregate / window call, counted in BOTH window passes).
    // The `num_out` for the output-column match is the projection width BEFORE any hidden
    // column — hidden ORDER BY columns are appended by the orchestrator, past the outputs.
    let agg_count_before_order = grouping.aggregates.borrow().len();
    let num_out = projections.len();
    let mut order_resolved = Vec::with_capacity(order_by.len());
    for term in order_by {
        match match_output_column(term, &result_asts, &names, num_out)? {
            Some(idx) => order_resolved.push(OrderResolved::Output(idx)),
            None => {
                // Not an output column: bind against the grouping scope. A group key /
                // aggregate resolves; a bare column is captured (§2.5); a genuinely
                // ungrouped column stays the loud "must appear in GROUP BY" error.
                let bound = bind_expr(&gscope, ctx, &term.expr)?;
                order_resolved.push(OrderResolved::Hidden(bound));
            }
        }
    }

    // Drain the collectors (interior mutation, so the shared borrows of `grouping` /
    // `windowing` may still be live).
    let aggregates = std::mem::take(&mut *grouping.aggregates.borrow_mut());
    let captured_regs = grouping.bare_capture.as_ref().map(BareCapture::regs).unwrap_or_default();
    let window_funcs = std::mem::take(&mut *windowing.functions.borrow_mut());

    // Register-safety guard for the §2.5 capture + ORDER BY-aggregate corner. Captured bare
    // columns are placed at `num_keys + count_aggregates(columns, having)` — computed from
    // the projection + HAVING aggregates only. An ORDER BY that binds a NEW aggregate would
    // push the real aggregate list past that base, so the captured slots would collide with
    // an order aggregate's register. This intersection (a bare-column query whose ORDER BY
    // also introduces an aggregate) is rejected LOUDLY rather than emit a colliding register
    // — the same conservative stance as the §2.5-plus-window rejection. Ordering by a group
    // key, or by an aggregate in a query with no bare columns, adds no such conflict.
    if !captured_regs.is_empty() && aggregates.len() > agg_count_before_order {
        return Err(Error::sql(
            "an ORDER BY that introduces an aggregate together with a bare column (lang_select.html §2.5) is not yet supported",
        ));
    }

    // Whether a correlated projection/HAVING/ORDER BY subquery was compiled under this
    // grouping (`compile::select::compile_subplan` set it via the shared `&grouping`). Read
    // after binding so the caller can size the correlated subquery's outer row correctly
    // (re-binding when a §2.5 capture / window call / ORDER BY aggregate widened the row).
    let saw_correlated_subplan = grouping.saw_correlated_subplan.get();

    Ok(GroupedOutput {
        projections,
        names,
        result_asts,
        having_bound,
        aggregates,
        captured_regs,
        window_funcs,
        order_resolved,
        saw_correlated_subplan,
    })
}

/// Build the single-min/max §2.5 plan marker from the pre-scan result and what the binder
/// actually collected: `None` unless the special case applied AND a bare column was
/// captured (a plain `min`/`max` with no bare columns captures nothing, so the operator
/// runs the ordinary, unchanged aggregation path). Factored out so the no-window branch
/// keeps one definition and its correctness asserts.
/// Split a completed bind's captured bare-column registers into the two mutually-exclusive
/// plan markers: the single-min/max extremum-row capture ([`MinMaxBare`]) when that special
/// case applies, else the general arbitrary-row capture (`bare_arbitrary`, a plain register
/// list) when any bare column was captured. Returns `(None, None)` for a query with no bare
/// column (the ordinary aggregation path, unchanged).
fn split_bare_capture(
    minmax_special: Option<MinMaxScan>,
    aggregates: &[AggregateCall],
    captured_regs: Vec<usize>,
) -> (Option<MinMaxBare>, Option<Vec<usize>>) {
    if captured_regs.is_empty() {
        return (None, None);
    }
    if minmax_special.is_some() {
        (build_minmax_bare(minmax_special, aggregates, captured_regs), None)
    } else {
        // No single min/max: the bare columns take an arbitrary (first) row of each group.
        (None, Some(captured_regs))
    }
}

/// Count every aggregate call in the projection + HAVING, in the SAME `walk_agg_calls`
/// traversal (and thus the same count) the binder collects in. Used to place captured bare
/// columns at `num_keys + num_aggregates` for the general §2.5 case up front, before the
/// binder has finished collecting — the general analog of [`single_minmax_special`]'s
/// `num_aggregates` (which it equals whenever that special case also applies).
fn count_aggregates(reg: &FunctionRegistry, columns: &[ResultColumn], having: &Option<Expr>) -> usize {
    let mut n = 0usize;
    for col in columns {
        if let ResultColumn::Expr { expr, .. } = col {
            walk_agg_calls(reg, expr, &mut |_| n += 1);
        }
    }
    if let Some(h) = having {
        walk_agg_calls(reg, h, &mut |_| n += 1);
    }
    n
}

fn build_minmax_bare(
    minmax_special: Option<MinMaxScan>,
    aggregates: &[AggregateCall],
    captured_regs: Vec<usize>,
) -> Option<MinMaxBare> {
    let s = minmax_special?;
    if captured_regs.is_empty() {
        return None;
    }
    // The captured columns were bound at `num_keys + num_aggregates + slot`, which is
    // correct only if the pre-scan's count/index equal the binder's collected ones. What
    // guarantees that is the EQUIVALENCE of two SEPARATE traversals — the pre-scan's
    // `walk_agg_calls` and the binder's own descent in `bind/expr.rs` — which agree because
    // both use the same `function_is_aggregate` classifier, descend in the same source
    // order, and neither dedups. This assert backstops that equivalence in debug/test builds
    // (compiled out of release); if either traversal ever changes its order/dedup, it fires
    // for exercised queries. (A post-aggregate window call CAN add further aggregates the
    // pre-scan misses, but that path never reaches here — the §2.5 + window combination is
    // rejected before this is called.)
    debug_assert_eq!(
        aggregates.len(),
        s.num_aggregates,
        "single-min/max pre-scan and the binder's collected aggregates must agree in count"
    );
    debug_assert!(
        s.agg_index < aggregates.len(),
        "the sole min/max index must fall within the collected aggregates"
    );
    Some(MinMaxBare { agg_index: s.agg_index, is_max: s.is_max, captured_regs })
}

/// Expand `*` / `table.*` in an aggregate query: every star column must be a group
/// key, else it is an ungrouped reference — a loud error (or, in the single-min/max
/// §2.5 case, a bare column captured from the extremum row), never a silent wrong
/// value. A plain-column key is matched by its input register (`col_to_group`); a
/// USING/NATURAL COALESCED key — which binds to `COALESCE(...)`, not a single register —
/// is matched by name against the GROUP BY keys ([`coalesced_star_group_key`]).
fn expand_grouped_star(
    scope: &Scope,
    grouping: &Grouping,
    table: Option<&str>,
    projections: &mut Vec<EvalExpr>,
    names: &mut Vec<String>,
    result_asts: &mut Vec<Option<Expr>>,
) -> Result<()> {
    for (reg, name) in scope.without_grouping().expand_star(table)? {
        // A star column that is a plain-column group key reads that key's register.
        if let Some(&gk) = grouping.col_to_group.get(&reg) {
            projections.push(EvalExpr::Column(gk));
            result_asts.push(Some(star_source_ast(&name)));
            names.push(name);
            continue;
        }
        // A USING/NATURAL COALESCED group key binds to `COALESCE(...)`, never a single
        // register, so it is absent from `col_to_group` above; an unqualified `*` copy of
        // it is matched by NAME against the GROUP BY keys — the `*` analogue of the
        // binder's bare-`k`-vs-key match (`bind_grouped_node` step 1). Star-path-local: a
        // qualified `a.k` binds through `bind_expr`, where it must stay the raw left copy
        // (NULL-extended on an outer-join group), so this never runs for it.
        if table.is_none() {
            if let Some(gk) = coalesced_star_group_key(scope, grouping, reg, &name) {
                projections.push(EvalExpr::Column(gk));
                result_asts.push(Some(star_source_ast(&name)));
                names.push(name);
                continue;
            }
        }
        // A non-group-key star column is bare: captured from the extremum row in the
        // single-min/max special case (§2.5), else an ungrouped-reference error. Star
        // only ever expands LOCAL sources, so every such column is a genuine column of
        // the aggregated relation (no outer-reference case to exclude).
        match &grouping.bare_capture {
            Some(cap) => {
                // A coalesced USING/NATURAL star column folds like the bare-name /
                // projection value path: capture EVERY component copy and `COALESCE`,
                // not just the surviving left copy (NULL-extended on an outer extremum
                // row). Coalescing applies only to an unqualified `*` (`table.is_none()`,
                // mirroring `expand_star_exprs`); the match is by SURVIVING register, not
                // name, since a same-named non-coalesced column can also survive a star.
                let coalesced =
                    if table.is_none() { scope.coalesced_regs_by_left(reg) } else { None };
                let projection = match coalesced {
                    Some(regs) => EvalExpr::Coalesce(
                        regs.into_iter()
                            .map(|r| EvalExpr::Column(cap.base + cap.intern(r)))
                            .collect(),
                    ),
                    None => EvalExpr::Column(cap.base + cap.intern(reg)),
                };
                projections.push(projection);
                result_asts.push(Some(star_source_ast(&name)));
                names.push(name);
            }
            None => {
                return Err(Error::sql(format!(
                    "column \"{name}\" must appear in the GROUP BY clause or be used in an aggregate function"
                )));
            }
        }
    }
    Ok(())
}

/// If an unqualified `*` column named `name` at surviving register `reg` is BOTH a
/// USING/NATURAL coalesced column AND a GROUP BY key, return that key's group index.
///
/// A coalesced group key binds to `EvalExpr::Coalesce(...)` (never a single register), so
/// it never enters [`Grouping::col_to_group`]; [`expand_grouped_star`]'s register lookup
/// thus misses it, and this recovers it the way the binder recovers a bare `k` — by
/// matching the column NAME against a whole GROUP BY key ([`bind::bind_expr`]'s step 1).
/// Scoped two ways so it can only affect the intended case: `reg` must be a coalesced left
/// copy (so a non-coalesced star column is never mistaken for a group key), and the key
/// must be a bare unqualified column of the same name (case-insensitive, matching column
/// resolution). The caller restricts this to an unqualified `*`; a qualified `a.k` binds
/// through `bind_expr`, where it must remain the raw left copy and never coalesce.
fn coalesced_star_group_key(scope: &Scope, grouping: &Grouping, reg: usize, name: &str) -> Option<usize> {
    scope.coalesced_regs_by_left(reg)?;
    let gk = grouping.key_asts.iter().position(|key| is_bare_column_named(key, name))?;
    // Group key `i` lives at post-aggregate register `i` — the load-bearing key_asts↔
    // register convention `bind_grouped_node` step 1 and `col_to_group` also rely on. A
    // `position` into `key_asts` (whose length is `num_keys`) is that register by
    // construction; pin it locally so a future divergence fails loud at this new site.
    debug_assert!(gk < grouping.num_keys, "group-key index must be a valid group-key register");
    Some(gk)
}

/// Whether `e` is a bare (schema- and table-unqualified) column reference named `name`
/// (case-insensitive) — the AST shape a bare `GROUP BY k` key has, which resolves to a
/// coalesced shared column when one exists.
fn is_bare_column_named(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Column { schema: None, table: None, name: n, .. } if n.eq_ignore_ascii_case(name))
}

/// The result of the single-min/max pre-scan: which aggregate is the sole min/max, and
/// the total aggregate count — enough to place the captured bare columns with no
/// post-bind fixup.
#[derive(Clone, Copy)]
struct MinMaxScan {
    /// `true` for `max()`, `false` for `min()`.
    is_max: bool,
    /// Index of the sole min/max in the binder's first-seen aggregate order (projection
    /// columns left-to-right, then HAVING). Its result sits at `num_keys + agg_index`,
    /// and the operator keys the extremum-row capture off it.
    agg_index: usize,
    /// Total number of aggregates. Captured bare columns follow all results, so they
    /// start at `num_keys + num_aggregates`.
    num_aggregates: usize,
}

/// Detect the single-min/max "bare columns" special case (lang_select.html §2.5): the
/// query contains EXACTLY ONE `min()`/`max()` aggregate (the single-argument aggregate
/// form), regardless of how many OTHER aggregates accompany it. Returns the
/// [`MinMaxScan`] locating that aggregate, else `None` when the case does not apply —
/// zero min/max (no extremum to associate a bare column with) or two-or-more min/max
/// (§2.5 limitation 2: the extremum row would be ambiguous).
///
/// The scan uses [`walk_agg_calls`] — the SAME single traversal `expr_contains_aggregate`
/// uses and, crucially, the same left-to-right depth-first order (projection columns then
/// HAVING) the binder collects in, and neither dedups. So `agg_index` equals the binder's
/// collected index and `num_aggregates` equals its collected count. That equivalence is
/// what lets a bare column bind to the computable register `num_keys + num_aggregates +
/// slot` mid-projection, before the binder has finished collecting.
fn single_minmax_special(
    reg: &FunctionRegistry,
    columns: &[ResultColumn],
    having: &Option<Expr>,
) -> Option<MinMaxScan> {
    let mut kinds: Vec<Option<bool>> = Vec::new();
    for col in columns {
        if let ResultColumn::Expr { expr, .. } = col {
            walk_agg_calls(reg, expr, &mut |name| kinds.push(minmax_kind(name)));
        }
    }
    if let Some(h) = having {
        walk_agg_calls(reg, h, &mut |name| kinds.push(minmax_kind(name)));
    }
    // Exactly one min/max aggregate, among any number of aggregates.
    let mut minmax = kinds.iter().enumerate().filter_map(|(i, &k)| k.map(|is_max| (i, is_max)));
    let (agg_index, is_max) = minmax.next()?;
    if minmax.next().is_some() {
        return None; // two-or-more min/max: §2.5 does not apply.
    }
    Some(MinMaxScan { is_max, agg_index, num_aggregates: kinds.len() })
}

/// Visit every aggregate CALL reachable in `e`, in first-seen (left-to-right,
/// depth-first) order, invoking `f(name)` with each aggregate function's name. Does NOT
/// descend into an aggregate's own arguments (nested aggregates are illegal) nor into a
/// subquery (it aggregates within itself).
///
/// This is THE aggregate-reachability walk: both [`expr_contains_aggregate`] ("is there
/// any?") and [`single_minmax_special`] ("which kinds, how many, in what order") are
/// built on it, so the set they observe is identical BY CONSTRUCTION — there is no second
/// traversal to drift from. The `match` is exhaustive with no `_` arm, so a new `Expr`
/// variant carrying sub-expressions is a compile error here until its descent is added,
/// which keeps this walk in lockstep with the binder's own reachability.
fn walk_agg_calls(reg: &FunctionRegistry, e: &Expr, f: &mut impl FnMut(&str)) {
    match e {
        Expr::Function { name, args, over, .. } => {
            if function_is_aggregate(reg, name, args, over.is_some()) {
                f(name);
                return;
            }
            if let FunctionArgs::List(list) = args {
                for a in list {
                    walk_agg_calls(reg, a, f);
                }
            }
        }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Collate { expr, .. } => {
            walk_agg_calls(reg, expr, f)
        }
        Expr::Binary { left, right, .. } => {
            walk_agg_calls(reg, left, f);
            walk_agg_calls(reg, right, f);
        }
        Expr::Like { lhs, rhs, escape, .. } => {
            walk_agg_calls(reg, lhs, f);
            walk_agg_calls(reg, rhs, f);
            if let Some(x) = escape.as_deref() {
                walk_agg_calls(reg, x, f);
            }
        }
        Expr::Between { expr, low, high, .. } => {
            walk_agg_calls(reg, expr, f);
            walk_agg_calls(reg, low, f);
            walk_agg_calls(reg, high, f);
        }
        Expr::In { expr, rhs, .. } => {
            walk_agg_calls(reg, expr, f);
            if let InRhs::List(items) = rhs {
                for x in items {
                    walk_agg_calls(reg, x, f);
                }
            }
        }
        Expr::Case { operand, whens, else_expr } => {
            if let Some(o) = operand.as_deref() {
                walk_agg_calls(reg, o, f);
            }
            for (w, t) in whens {
                walk_agg_calls(reg, w, f);
                walk_agg_calls(reg, t, f);
            }
            if let Some(e) = else_expr.as_deref() {
                walk_agg_calls(reg, e, f);
            }
        }
        Expr::IsNull(x) | Expr::NotNull(x) => walk_agg_calls(reg, x, f),
        Expr::Parenthesized(list) => {
            for x in list {
                walk_agg_calls(reg, x, f);
            }
        }
        Expr::Literal(_)
        | Expr::Column { .. }
        | Expr::BindParam(_)
        | Expr::Raise(_)
        | Expr::Exists { .. }
        | Expr::Subquery(_) => {}
    }
}

/// The min/max kind of an aggregate function NAME already classified as an aggregate
/// (so a one-argument `min`/`max`, not the multi-argument scalar form): `Some(true)`
/// for `max`, `Some(false)` for `min`, `None` for any other aggregate.
fn minmax_kind(name: &str) -> Option<bool> {
    if name.eq_ignore_ascii_case("max") {
        Some(true)
    } else if name.eq_ignore_ascii_case("min") {
        Some(false)
    } else {
        None
    }
}

/// Whether any result column contains an aggregate call (not descending into
/// subqueries, whose aggregation is their own).
fn columns_contain_aggregate(reg: &FunctionRegistry, columns: &[ResultColumn]) -> bool {
    columns.iter().any(|c| match c {
        ResultColumn::Expr { expr, .. } => expr_contains_aggregate(reg, expr),
        _ => false,
    })
}

/// Whether `e` contains an aggregate call, stopping at subquery boundaries (a subquery
/// aggregates within itself). A thin wrapper over [`walk_agg_calls`] — the ONE
/// aggregate-reachability traversal in this file — so it can never disagree with the
/// pre-scan (or, transitively, the binder) about which calls are aggregates.
pub(crate) fn expr_contains_aggregate(reg: &FunctionRegistry, e: &Expr) -> bool {
    let mut found = false;
    walk_agg_calls(reg, e, &mut |_| found = true);
    found
}

#[cfg(test)]
mod tests {
    // `super::*` re-exports this module's own imports, so `EvalExpr`, `SortKey`, `Collation`,
    // `PlanNode`, `Aggregate`, and `ordinal_suffix` are already in scope; only the
    // catalog/pager fixtures and the top-level planner entrypoints are new.
    use super::*;

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};

    use crate::plan::Plan;
    use crate::{Planner, QueryPlanner};

    // -----------------------------------------------------------------------
    // Local static test catalog (a copy of the `src/tests.rs` fixture pattern;
    // that file lives in another module and is not edited here — the same
    // reason `compile::compound`'s test module keeps its own copy).
    // -----------------------------------------------------------------------

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
        fn create_table(
            &mut self,
            _pager: &mut dyn Pager,
            _stmt: &CreateTable,
            _sql: &str,
        ) -> Result<()> {
            unimplemented!("test catalog is static")
        }
        fn create_index(
            &mut self,
            _pager: &mut dyn Pager,
            _stmt: &CreateIndex,
            _sql: &str,
        ) -> Result<()> {
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

    /// `t(a INTEGER, b TEXT, c TEXT COLLATE NOCASE)` — `a`->Column(0), `b`->Column(1),
    /// `c`->Column(2). No indexes, so the MIN/MAX seek fast path never fires here.
    fn cat_t() -> TestCatalog {
        TestCatalog::new(vec![tdef(
            "t",
            vec![
                col("a", Some("INTEGER"), None),
                col("b", Some("TEXT"), None),
                col("c", Some("TEXT"), Some("NOCASE")),
            ],
            None,
        )])
    }

    fn plan_sql(sql: &str, cat: &dyn Catalog) -> Result<Plan> {
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("expected one statement");
        QueryPlanner::new().plan(stmt, cat)
    }

    /// Peel the top `Project` and return its input node (the projection's source). Every
    /// aggregate query ends in a `Project`; the implicit group sort (when present) sits
    /// directly under it.
    fn project_input(n: &PlanNode) -> &PlanNode {
        match n {
            PlanNode::Project { input, .. } => input,
            other => panic!("expected a Project at the plan root, got {other:?}"),
        }
    }

    fn expect_aggregate(n: &PlanNode) -> &Aggregate {
        match n {
            PlanNode::Aggregate(a) => a,
            other => panic!("expected an Aggregate, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Implicit ascending GROUP-key order for an unordered GROUP BY (no ORDER BY):
    // real SQLite groups via a sort, so the planner wraps the aggregate output in an
    // ascending Sort on the group-key columns, UNDER the projection.
    // -----------------------------------------------------------------------

    #[test]
    fn implicit_group_order_wraps_aggregate_in_ascending_sort() {
        // `SELECT a, count(*) FROM t GROUP BY a` (no ORDER BY): the plan is
        // `Project { Sort { Aggregate } }` with one ascending key on the group column.
        let cat = cat_t();
        let plan = plan_sql("SELECT a, count(*) FROM t GROUP BY a", &cat).unwrap();
        match project_input(&plan.root) {
            PlanNode::Sort { input, keys, limit } => {
                assert_eq!(keys.len(), 1, "one key per group column");
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "sorts group-key column 0");
                assert!(!keys[0].desc, "implicit group sort is ASCENDING");
                assert_eq!(keys[0].nulls_first, None, "implicit sort uses the SQL-default NULLS order");
                assert!(limit.is_none(), "implicit group sort is a full stable sort");
                // The Sort wraps the Aggregate directly (the projection reads the sorted rows).
                expect_aggregate(input);
            }
            other => panic!("expected a Sort under the Project, got {other:?}"),
        }
    }

    #[test]
    fn implicit_group_order_multicolumn_sorts_every_group_key() {
        // A multi-column GROUP BY sorts by every key in order: keys are Column(0), Column(1),
        // both ascending — so groups equal on `a` are still ordered by `b`.
        let cat = cat_t();
        let plan = plan_sql("SELECT a, b, count(*) FROM t GROUP BY a, b", &cat).unwrap();
        match project_input(&plan.root) {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 2, "one key per group column");
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "first key is group column 0");
                assert!(matches!(keys[1].expr, EvalExpr::Column(1)), "second key is group column 1");
                assert!(keys.iter().all(|k| !k.desc), "all group keys ascending");
                let agg = expect_aggregate(input);
                assert_eq!(agg.group_by.len(), 2, "two group keys");
            }
            other => panic!("expected a Sort under the Project, got {other:?}"),
        }
    }

    #[test]
    fn implicit_group_order_reuses_the_group_key_collation() {
        // CONTRACT: an implicit group-sort key reuses the group column's grouping collation —
        // exactly the Aggregate's `group_collations[i]` — so the order matches the grouping
        // comparison. `c` is declared COLLATE NOCASE, so the key is NOCASE, not the BINARY
        // default; asserting the RELATIONSHIP (not just the literal) pins the reuse.
        let cat = cat_t();
        let plan = plan_sql("SELECT c, count(*) FROM t GROUP BY c", &cat).unwrap();
        match project_input(&plan.root) {
            PlanNode::Sort { input, keys, .. } => {
                let agg = expect_aggregate(input);
                assert_eq!(
                    keys[0].collation, agg.group_collations[0],
                    "implicit key inherits the group column's grouping collation"
                );
                assert_eq!(keys[0].collation, Collation::NoCase, "c is COLLATE NOCASE");
            }
            other => panic!("expected a Sort under the Project, got {other:?}"),
        }
    }

    #[test]
    fn no_implicit_sort_for_empty_group_by() {
        // A whole-table aggregate (`num_keys == 0`) yields a single row: nothing to order, so
        // NO Sort is inserted — the projection reads the Aggregate directly. `expect_aggregate`
        // panics if a Sort (or anything else) sits between the Project and the Aggregate.
        let cat = cat_t();
        let plan = plan_sql("SELECT count(*) FROM t", &cat).unwrap();
        let agg = expect_aggregate(project_input(&plan.root));
        assert!(agg.group_by.is_empty(), "no group keys");
    }

    #[test]
    fn explicit_order_by_adds_no_implicit_group_sort() {
        // With an explicit ORDER BY, the implicit group sort must NOT fire: the existing
        // ORDER BY Sort sits OVER the Project (sorting the projected output), and the
        // Project's input is the Aggregate directly — no second (implicit) Sort under it.
        let cat = cat_t();
        let plan = plan_sql("SELECT a, count(*) FROM t GROUP BY a ORDER BY a", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1, "the explicit ORDER BY key");
                // The explicit Sort wraps the Project; under the Project is the bare Aggregate.
                let under_project = project_input(input);
                expect_aggregate(under_project);
            }
            other => panic!("expected the explicit ORDER BY Sort at the root, got {other:?}"),
        }
    }

    #[test]
    fn implicit_group_sort_survives_a_projection_that_drops_the_key() {
        // `SELECT count(*) FROM t GROUP BY a` projects the group key `a` AWAY, yet the sort
        // must still order by the key VALUE — which is why the Sort sits UNDER the projection,
        // on the aggregate output `[a_key, count]`, keying on Column(0) (the dropped key).
        let cat = cat_t();
        let plan = plan_sql("SELECT count(*) FROM t GROUP BY a", &cat).unwrap();
        match project_input(&plan.root) {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1, "sort on the single group key");
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "keys the aggregate-output group column, not the projected output");
                expect_aggregate(input);
            }
            other => panic!("expected a Sort under the Project, got {other:?}"),
        }
    }

    #[test]
    fn window_path_has_no_implicit_group_sort() {
        // RESIDUAL GUARD: on the post-aggregate WINDOW path (`Aggregate -> Window -> Project`)
        // the implicit group-key sort is deliberately NOT inserted — the window's own ordering
        // governs output order there (see the residual comment in `compile_aggregate`). Pin the
        // shape `Project { Window { Aggregate } }` (no interposed Sort) so a future edit cannot
        // silently start (mis)ordering this path without failing here. This locks in the
        // documented limitation as an executable invariant, not just prose.
        let cat = cat_t();
        let plan =
            plan_sql("SELECT a, row_number() OVER () FROM t GROUP BY a", &cat).unwrap();
        match project_input(&plan.root) {
            PlanNode::Window(w) => {
                // The Window reads the Aggregate directly — no implicit Sort between them.
                expect_aggregate(w.input.as_ref());
            }
            other => {
                panic!("expected a Window under the Project (no implicit group Sort), got {other:?}")
            }
        }
    }

    /// `ordinal_suffix` feeds the GROUP BY out-of-range message ("<N>th GROUP BY term ...").
    /// Real SQL rarely reaches a position past 2, so the higher branches are otherwise
    /// unexercised; pin the whole rule (unit digit, minus the 11/12/13 teens exception) here.
    #[test]
    fn ordinal_suffix_covers_units_and_the_teens() {
        // Unit digit 1/2/3 -> st/nd/rd; every other unit -> th.
        assert_eq!(ordinal_suffix(1), "st");
        assert_eq!(ordinal_suffix(2), "nd");
        assert_eq!(ordinal_suffix(3), "rd");
        for n in [4usize, 5, 6, 7, 8, 9, 10] {
            assert_eq!(ordinal_suffix(n), "th", "{n} takes th");
        }
        // The teens 11/12/13 are the exception: all th, never st/nd/rd.
        assert_eq!(ordinal_suffix(11), "th");
        assert_eq!(ordinal_suffix(12), "th");
        assert_eq!(ordinal_suffix(13), "th");
        // Past the teens the unit rule resumes; the hundreds teens (111/112/113) are th again.
        assert_eq!(ordinal_suffix(21), "st");
        assert_eq!(ordinal_suffix(22), "nd");
        assert_eq!(ordinal_suffix(23), "rd");
        assert_eq!(ordinal_suffix(101), "st");
        assert_eq!(ordinal_suffix(111), "th");
        assert_eq!(ordinal_suffix(112), "th");
        assert_eq!(ordinal_suffix(113), "th");
    }
}
