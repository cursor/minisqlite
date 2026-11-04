//! Compound SELECT (`UNION` / `UNION ALL` / `INTERSECT` / `EXCEPT`): compile the two
//! (or more) select bodies and combine them into a [`PlanNode::SetOp`](crate::plan::PlanNode::SetOp),
//! then wrap the whole compound in its outer `ORDER BY` / `LIMIT`.
//!
//! Delegation seam: [`compile_compound`] is the single entrypoint `compile_body`
//! routes a [`SelectBody::Compound`](minisqlite_sql::SelectBody::Compound) through,
//! so the set-operation feature lands here rather than in the SELECT orchestrator.
//!
//! Shape: the compound body is a left-deep tree of set operators
//! (`a UNION b UNION c` = `(a UNION b) UNION c`, left-associative per the parser).
//! `compile_setop_tree` walks it by borrowing each `&SelectBody` node — so the whole
//! chain compiles in one pass without re-cloning the growing left sub-body at every
//! level — and bottoms out at a single arm, each compiled as a standalone SELECT (no
//! ORDER BY/LIMIT of its own — those belong to the whole compound) via
//! [`compile_body`](crate::compile::select::compile_body). The two arms of each
//! operator must have equal result width; the result column *names* come from the
//! LEFT arm (the plan `SetOp` convention).
//!
//! Correlated subqueries: `base_offset` / `correlated_out` (see
//! [`crate::compile::select`]) are threaded UNCHANGED to EVERY arm, so a compound that
//! is itself a correlated subquery places each arm's own sources after the same outer
//! row and any arm referencing an outer column marks the one shared correlation cell.

use std::cell::{Cell, RefCell};

use minisqlite_expr::{EvalExpr, SortKey};
use minisqlite_sql::{CompoundOp, Expr, Literal, NullsOrder, Select, SelectBody, SortOrder};
use minisqlite_types::{Collation, Error, Result};

use crate::bind::{parse_collation, Scope};
use crate::compile::select::compile_body;
use crate::plan::{PlanNode, SetOp, SetOpKind};
use crate::plan_ctx::PlanCtx;

/// Compile a compound SELECT (a [`SelectBody::Compound`](minisqlite_sql::SelectBody::Compound)),
/// with an optional enclosing scope, into its plan tree and output column names.
///
/// The outer `sel.order_by` / `sel.limit` apply to the whole compound and wrap the
/// emitted `SetOp`; a leading `sel.with` is handled by the CTE compiler before this
/// runs, so it is ignored here (its arms carry no `WITH`).
///
/// `base_offset` / `correlated_out` carry the correlated-subquery seam
/// (see [`crate::compile::select`]) and are forwarded to every arm.
///
/// `col_collations_out`, when `Some`, receives this compound's per-output-column DEFINED
/// collation (the first-determinable-left-to-right result — see [`compile_setop_tree`]),
/// so an OUTER compound that has this one as an arm inherits it. In practice only reached
/// for a compound nested as a compound arm; every current caller passes `None`.
#[allow(clippy::too_many_arguments)]
pub fn compile_compound(
    ctx: &mut PlanCtx,
    sel: &Select,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
    col_collations_out: Option<&RefCell<Vec<Option<Collation>>>>,
) -> Result<(PlanNode, Vec<String>)> {
    // `compile_body` only routes a Compound body here; a Select body means a mis-wired
    // caller. Fail closed (in every build) rather than silently compile a lone core as
    // if it were a compound.
    if !matches!(sel.body, SelectBody::Compound { .. }) {
        return Err(Error::sql("compile_compound called on a non-compound SELECT body"));
    }

    // Compile the set-operation tree itself; the outer ORDER BY / LIMIT below apply to
    // the whole compound, not to any single arm. `col_up` is each output column's
    // first-determinable-left-to-right DEFINED collation (`None` = undetermined = BINARY).
    let (mut node, names, col_up) = compile_setop_tree(
        ctx,
        &sel.body,
        parent,
        base_offset,
        correlated_out,
        correlated_cols_out,
        nondet_out,
    )?;
    let width = names.len();
    debug_assert_eq!(col_up.len(), width, "one column collation per output column");

    // Propagate this compound's column collation up to an enclosing compound (if any):
    // SQLite's `multiSelectCollSeq` recurses through the left arm, so a compound nested
    // as an arm contributes its own first-determinable collation to the outer comparison.
    if let Some(out) = col_collations_out {
        *out.borrow_mut() = col_up.clone();
    }

    // Output-column collations for the outer ORDER BY. These mirror the OUTERMOST
    // SetOp's dedup collations: the column's determined collation, else BINARY.
    let column_collations: Vec<Collation> =
        col_up.iter().map(|c| c.unwrap_or(Collation::Binary)).collect();

    // Sort keys reference the compound's OUTPUT columns, resolved against `names` +
    // `column_collations`.
    let mut keys = order_keys(sel, &names, &column_collations)?;
    // Implicit ORDER for a DEDUPLICATING compound. Real SQLite implements the
    // UNION / INTERSECT / EXCEPT duplicate elimination as a SORT, so such a compound
    // comes back in ASCENDING full-row order even WITHOUT an explicit ORDER BY
    // (lang_select.html §3, datatype3.html §6: the operators "perform implicit
    // comparisons" and the result is ordered by them). `UNION ALL` is the exception —
    // it concatenates the left rows then the right rows in order and does NOT sort — so
    // this fires only when the OUTERMOST operator dedups.
    //
    // When there is no explicit ORDER BY, synthesize a full-row ascending sort over every
    // output column, reusing each column's dedup collation (`column_collations[i]`, the
    // OUTERMOST SetOp's per-column comparison collation) so the sort order matches the
    // duplicate comparison. `nulls_first: None` is the SQL default (NULLs sort first for
    // ASC, datatype3.html §4.1). This feeds the SAME wrap path below as an explicit ORDER
    // BY, so a `... UNION ... LIMIT k` still pushes the top-k retention bound into the sort.
    //
    // `node` is always a `PlanNode::SetOp` here (a non-compound body is rejected above), so
    // its op decides this; inspect `&node` before it is moved into the `Sort` below.
    //
    // Residual (rare; intentionally not implemented): if the OUTERMOST op is `UNION ALL`
    // over a dedup sub-compound (e.g. `(a UNION b) UNION ALL c`), the whole compound is
    // correctly left unsorted, but real SQLite's LEFT sub-result is itself sorted while ours
    // streams dedup order. Matching that would need per-subtree sort insertion.
    if keys.is_empty()
        && let PlanNode::SetOp(s) = &node
        && matches!(s.op, SetOpKind::Union | SetOpKind::Intersect | SetOpKind::Except)
    {
        keys = (0..width)
            .map(|i| SortKey {
                expr: EvalExpr::Column(i),
                desc: false,
                nulls_first: None,
                collation: column_collations[i],
            })
            .collect();
    }
    // Bind the LIMIT once so it feeds both the compound sort's retention bound and the
    // Limit node (see `select::bind_limit_exprs`). `order_keys` above binds no bind
    // parameters (a compound ORDER BY resolves only to output columns by ordinal/name),
    // so binding here keeps LIMIT parameter numbering after the arms.
    //
    // INVARIANT (same as the plain-SELECT path in `select::assemble_output`): the sort's
    // retention bound and the Limit node MUST stay co-derived from THIS one `bound_limit`,
    // else the independently-evaluated sort bound could drop rows the Limit still needs.
    let bound_limit = match &sel.limit {
        Some(limit) => Some(crate::compile::select::bind_limit_exprs(ctx, limit)?),
        None => None,
    };
    if !keys.is_empty() {
        // The sort is directly under the Limit over the whole compound, so the same
        // bounded top-k as the plain-SELECT path applies (byte-identical to full sort
        // then Limit): both consume the SAME SetOp output order, so the first-k of the
        // stable sort is identical. Attach it when the LIMIT is deterministic.
        let limit = bound_limit.as_ref().and_then(crate::compile::select::sort_limit_from);
        node = PlanNode::Sort { input: Box::new(node), keys, limit };
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

/// Compile a (possibly nested) compound body into a left-deep tree of
/// [`PlanNode::SetOp`], returning the plan, the output column names (taken from the
/// leftmost arm), and each output column's DEFINED collation for the compound dedup rule.
///
/// The third element is one `Option<Collation>` per output column: the
/// FIRST-DETERMINABLE-LEFT-TO-RIGHT DEFINED collation across this subtree's arms —
/// `Some(c)` when some arm's output expression (scanning left first) DEFINES a collation
/// `c` (an explicit postfix `COLLATE`, else a plain column reference's declared collation),
/// else `None` (BINARY for the comparison). This mirrors SQLite's `multiSelectCollSeq`: the
/// left subtree's propagated collation wins, else this level's right arm.
///
/// Per lang_select.html's "duplicate rows in a compound" rule the comparison is "as if the
/// columns were the operands of the equals (=) operator, EXCEPT that greater precedence is
/// not assigned to a collation sequence specified with the postfix COLLATE operator" (and no
/// affinity). A postfix COLLATE therefore still DEFINES its own arm's collation
/// ([`defined_collation`]) — it is not ignored — but earns no precedence OVER a column
/// collation across arms; that "no greater precedence" clause is exactly the left→right fold
/// below (`left.or(right)`), which lets an EARLIER arm's column collation beat a LATER arm's
/// postfix COLLATE, while a postfix COLLATE in the earlier arm still wins by position.
///
/// Recurses on `&SelectBody` rather than re-wrapping the left arm in a cloned
/// synthetic `Select` per level: a left-nested chain `a UNION b UNION … UNION z`
/// would otherwise deep-clone the entire remaining left sub-body at every level
/// (Σ = O(arms²) arm clones). The base case is a single (non-compound) arm, wrapped
/// as a standalone SELECT and compiled via [`compile_body`]. Arms are compiled
/// left-then-right so `ctx` (bind-param numbering, subquery registration) advances in
/// source order.
///
/// `base_offset` / `correlated_out` are forwarded UNCHANGED to every arm (and every
/// nested level): all arms of a compound share one register base and one correlation
/// cell, so a correlated compound subquery binds each arm's own sources after the same
/// outer row and any arm's outer reference marks the whole compound correlated.
fn compile_setop_tree(
    ctx: &mut PlanCtx,
    body: &SelectBody,
    parent: Option<&Scope>,
    base_offset: usize,
    correlated_out: Option<&Cell<bool>>,
    correlated_cols_out: Option<&RefCell<Vec<usize>>>,
    nondet_out: Option<&Cell<bool>>,
) -> Result<(PlanNode, Vec<String>, Vec<Option<Collation>>)> {
    let (op, left, right_core) = match body {
        SelectBody::Compound { op, left, right } => (*op, left.as_ref(), right),
        // Base case: a single arm. It carries no ORDER BY/LIMIT of its own (those
        // belong to the whole compound), so wrap it in a bare SELECT and compile it.
        SelectBody::Select(core) => {
            let arm = Select {
                with: None,
                body: SelectBody::Select(core.clone()),
                order_by: Vec::new(),
                limit: None,
            };
            // Collect this arm's per-output-column DEFINED collation (explicit postfix
            // COLLATE, else a column ref's declared collation, else `None`); `compile_body`
            // routes a Select body to `compile_core`, which fills the cell.
            let col_cells = RefCell::new(Vec::new());
            let (node, names) = compile_body(
                ctx,
                &arm,
                parent,
                base_offset,
                correlated_out,
                correlated_cols_out,
                nondet_out,
                Some(&col_cells),
            )?;
            let col_up = col_cells.into_inner();
            debug_assert_eq!(
                col_up.len(),
                names.len(),
                "one column collation per output column for a compound arm"
            );
            return Ok((node, names, col_up));
        }
    };

    // Left first (source-order ctx threading), recursing without cloning the left
    // sub-body; the right arm is always a single core.
    let (left_node, names, left_up) = compile_setop_tree(
        ctx,
        left,
        parent,
        base_offset,
        correlated_out,
        correlated_cols_out,
        nondet_out,
    )?;
    let right_sel = Select {
        with: None,
        body: SelectBody::Select(right_core.clone()),
        order_by: Vec::new(),
        limit: None,
    };
    let right_cells = RefCell::new(Vec::new());
    let (right_node, right_names) = compile_body(
        ctx,
        &right_sel,
        parent,
        base_offset,
        correlated_out,
        correlated_cols_out,
        nondet_out,
        Some(&right_cells),
    )?;
    let right_up = right_cells.into_inner();

    // Both arms must have the same result width (SQLite's compound-arity rule); the
    // output width IS that width, and the result names come from the LEFT arm.
    if names.len() != right_names.len() {
        return Err(Error::sql(format!(
            "SELECTs to the left and right of {} do not have the same number of result columns",
            op_name(op)
        )));
    }
    let width = names.len();
    debug_assert_eq!(left_up.len(), width, "left arm collation vec width");
    debug_assert_eq!(right_up.len(), width, "right arm collation vec width");

    // Per-column collation, folding this level's two operands by the compound rule: the
    // first DEFINED collation scanning left → right (the left subtree already folded
    // left-to-right, then this right arm). `left.or(right)` IS the "greater precedence not
    // assigned to a postfix COLLATE" clause — a postfix COLLATE defines its arm's collation
    // but only wins across arms by being earlier, never by outranking an earlier arm's
    // column collation. `None` propagates upward so an enclosing compound keeps scanning;
    // here it also becomes the dedup collation (`None` → BINARY). No affinity — see the doc.
    let col_up: Vec<Option<Collation>> =
        (0..width).map(|i| left_up[i].or(right_up[i])).collect();
    // The whole-row dedup comparison (`UNION`/`INTERSECT`/`EXCEPT`) applies exactly this
    // per column via `row_key` in the executor. `UNION ALL` never dedups, so its vector
    // is unused, but a correctly-sized one is still supplied.
    let column_collations: Vec<Collation> =
        col_up.iter().map(|c| c.unwrap_or(Collation::Binary)).collect();

    Ok((
        PlanNode::SetOp(SetOp {
            op: set_op_kind(op),
            left: Box::new(left_node),
            right: Box::new(right_node),
            column_collations,
        }),
        names,
        col_up,
    ))
}

/// Resolve the compound's outer `ORDER BY` into sort keys over the output row.
///
/// A compound's ORDER BY may only reference an OUTPUT column — by 1-based ordinal or
/// by name — not an arbitrary expression (SQLite forbids that here). An explicit
/// `COLLATE` on the term wins; otherwise the referenced column's dedup collation
/// (`column_collations[idx]`) is inherited.
fn order_keys(
    sel: &Select,
    names: &[String],
    column_collations: &[Collation],
) -> Result<Vec<SortKey>> {
    let num_out = names.len();
    let mut keys = Vec::with_capacity(sel.order_by.len());
    for term in &sel.order_by {
        // The parser folds `ORDER BY x COLLATE C` into `term.expr` (an `Expr::Collate`),
        // leaving `term.collation` empty in the usual case; peel the COLLATE to expose
        // the ordinal / name to match, and to capture the explicit collation. Fall back
        // to the (rare) `term.collation` field if the expr carried none.
        let (core, expr_collation) = peel_collate(&term.expr)?;
        let explicit = match expr_collation {
            Some(c) => Some(c),
            None => match &term.collation {
                Some(name) => Some(parse_collation(name)?),
                None => None,
            },
        };
        let idx = match_output_column(core, names, num_out)?;
        keys.push(SortKey {
            expr: EvalExpr::Column(idx),
            desc: matches!(term.order, Some(SortOrder::Desc)),
            nulls_first: term.nulls.map(|n| matches!(n, NullsOrder::First)),
            collation: explicit.unwrap_or(column_collations[idx]),
        });
    }
    Ok(keys)
}

/// Match one ORDER BY term core (after `COLLATE` has been peeled) to an output-column
/// index: a bare 1-based integer ordinal, or an unqualified name equal (ASCII
/// case-insensitively) to an output column name. An out-of-range ordinal and any
/// other expression form are loud errors — a compound ORDER BY resolves only to the
/// result set, never to a from-clause expression.
fn match_output_column(core: &Expr, names: &[String], num_out: usize) -> Result<usize> {
    if let Expr::Literal(Literal::Integer(k)) = core {
        if *k >= 1 && (*k as usize) <= num_out {
            return Ok((*k - 1) as usize);
        }
        return Err(Error::sql(format!(
            "ORDER BY term out of range - should be between 1 and {num_out}"
        )));
    }
    if let Expr::Column { schema: None, table: None, name, .. } = core
        && let Some(idx) = names.iter().position(|n| n.eq_ignore_ascii_case(name))
    {
        return Ok(idx);
    }
    Err(Error::sql("ORDER BY term does not match any column in the result set"))
}

/// Peel a leading (possibly nested) `COLLATE` off an ORDER BY term, returning the
/// inner expression to match and the explicit collation named. When `COLLATE` nests,
/// the innermost (leftmost in source) name wins, matching the §7.1 precedence the
/// comparison binder uses. Only `COLLATE` is peeled: a parenthesized ordinal is not
/// an ordinal in SQLite, matching the single-SELECT ORDER BY path.
fn peel_collate(e: &Expr) -> Result<(&Expr, Option<Collation>)> {
    if let Expr::Collate { expr, collation } = e {
        let (inner, inner_collation) = peel_collate(expr)?;
        let collation = match inner_collation {
            Some(c) => Some(c),
            None => Some(parse_collation(collation)?),
        };
        return Ok((inner, collation));
    }
    Ok((e, None))
}

/// SQLite's spelling of the compound operator, for the arity-mismatch error text.
fn op_name(op: CompoundOp) -> &'static str {
    match op {
        CompoundOp::Union => "UNION",
        CompoundOp::UnionAll => "UNION ALL",
        CompoundOp::Intersect => "INTERSECT",
        CompoundOp::Except => "EXCEPT",
    }
}

/// Map the parsed compound operator to the plan's [`SetOpKind`].
fn set_op_kind(op: CompoundOp) -> SetOpKind {
    match op {
        CompoundOp::Union => SetOpKind::Union,
        CompoundOp::UnionAll => SetOpKind::UnionAll,
        CompoundOp::Intersect => SetOpKind::Intersect,
        CompoundOp::Except => SetOpKind::Except,
    }
}

#[cfg(test)]
mod tests {
    // `super::*` re-exports the module's own imports, so `EvalExpr`, `Collation`,
    // `Error`, `PlanNode`, `SetOp`, and `SetOpKind` are already in scope here; only the
    // catalog/pager fixtures and the top-level planner entrypoints are new.
    use super::*;

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Value;

    use crate::plan::Plan;
    use crate::{Planner, QueryPlanner};

    // -----------------------------------------------------------------------
    // Local static test catalog (a copy of the `src/tests.rs` fixture pattern;
    // that file lives in another module and is not edited here).
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
    /// `c`->Column(2).
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

    // -----------------------------------------------------------------------
    // Matchers.
    // -----------------------------------------------------------------------

    fn expect_setop(n: &PlanNode) -> &SetOp {
        match n {
            PlanNode::SetOp(s) => s,
            other => panic!("expected SetOp, got {other:?}"),
        }
    }

    /// The SetOp at a compound's ROOT, seen THROUGH the implicit dedup sort that now wraps a
    /// no-ORDER-BY UNION / INTERSECT / EXCEPT (see the implicit-ORDER rule in
    /// `compile_compound`); a bare SetOp (e.g. UNION ALL) is returned directly. Structural
    /// tests that assert the SetOp's OWN shape (kind / nesting / names) use this so they stay
    /// independent of the wrapper; the wrap-vs-no-wrap decision itself is pinned separately by
    /// the dedicated `implicit_*` tests below.
    fn root_setop(n: &PlanNode) -> &SetOp {
        match n {
            PlanNode::Sort { input, .. } => expect_setop(input),
            other => expect_setop(other),
        }
    }

    /// A `Project`'s output width (= number of projected columns).
    fn project_width(n: &PlanNode) -> usize {
        match n {
            PlanNode::Project { exprs, .. } => exprs.len(),
            other => panic!("expected Project, got {other:?}"),
        }
    }

    fn single_column(n: &PlanNode) -> &EvalExpr {
        match n {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 1, "expected a single projected column");
                &exprs[0]
            }
            other => panic!("expected Project, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // SetOp kind + width + names.
    // -----------------------------------------------------------------------

    #[test]
    fn union_makes_a_setop_with_equal_width_children() {
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT a FROM t", &cat).unwrap();
        let s = root_setop(&plan.root);
        assert!(matches!(s.op, SetOpKind::Union));
        assert_eq!(project_width(&s.left), project_width(&s.right), "arms must be equal width");
        assert_eq!(project_width(&s.left), 1);
        assert_eq!(s.column_collations.len(), 1, "one collation per output column");
        // Result names come from the LEFT arm.
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
    }

    #[test]
    fn each_operator_maps_to_its_setop_kind() {
        let cat = cat_t();
        let cases = [
            ("SELECT a FROM t UNION SELECT a FROM t", SetOpKind::Union),
            ("SELECT a FROM t UNION ALL SELECT a FROM t", SetOpKind::UnionAll),
            ("SELECT a FROM t INTERSECT SELECT a FROM t", SetOpKind::Intersect),
            ("SELECT a FROM t EXCEPT SELECT a FROM t", SetOpKind::Except),
        ];
        for (sql, want) in cases {
            let plan = plan_sql(sql, &cat).unwrap();
            let s = root_setop(&plan.root);
            // `SetOpKind` has no `PartialEq`; compare enum discriminants.
            assert_eq!(
                std::mem::discriminant(&s.op),
                std::mem::discriminant(&want),
                "{sql}: expected {want:?}, got {:?}",
                s.op
            );
        }
    }

    // -----------------------------------------------------------------------
    // Left-associative nesting: `a UNION b UNION c` == `(a UNION b) UNION c`.
    // -----------------------------------------------------------------------

    #[test]
    fn three_way_union_nests_left_associatively() {
        let cat = cat_t();
        let plan =
            plan_sql("SELECT a FROM t UNION SELECT b FROM t UNION SELECT c FROM t", &cat).unwrap();
        let outer = root_setop(&plan.root);
        assert!(matches!(outer.op, SetOpKind::Union));
        // The OUTER right child is the last arm, `c` (Column(2)).
        assert!(
            matches!(single_column(&outer.right), EvalExpr::Column(2)),
            "outer right arm is `c` (Column 2)"
        );
        // The OUTER left child is the nested `(a UNION b)`.
        let inner = expect_setop(&outer.left);
        assert!(matches!(inner.op, SetOpKind::Union));
        assert!(matches!(single_column(&inner.left), EvalExpr::Column(0)), "inner left is `a`");
        assert!(matches!(single_column(&inner.right), EvalExpr::Column(1)), "inner right is `b`");
        // Names still come from the leftmost arm.
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
    }

    #[test]
    fn four_way_mixed_compound_nests_left_deep() {
        // All four operators have EQUAL precedence and are left-associative, so
        // `a UNION b INTERSECT c EXCEPT a` == `((a UNION b) INTERSECT c) EXCEPT a`.
        // Exercises the `&SelectBody` recursion at depth 3 with mixed operator kinds
        // and confirms the right arm at each level is that level's trailing core.
        let cat = cat_t();
        let plan = plan_sql(
            "SELECT a FROM t UNION SELECT b FROM t INTERSECT SELECT c FROM t EXCEPT SELECT a FROM t",
            &cat,
        )
        .unwrap();
        // Outermost: EXCEPT, right arm the last `a` (Column 0).
        let l3 = root_setop(&plan.root);
        assert!(matches!(l3.op, SetOpKind::Except));
        assert!(matches!(single_column(&l3.right), EvalExpr::Column(0)), "outermost right is last `a`");
        // Middle: INTERSECT, right arm `c` (Column 2).
        let l2 = expect_setop(&l3.left);
        assert!(matches!(l2.op, SetOpKind::Intersect));
        assert!(matches!(single_column(&l2.right), EvalExpr::Column(2)), "intersect right is `c`");
        // Innermost: UNION(a, b).
        let l1 = expect_setop(&l2.left);
        assert!(matches!(l1.op, SetOpKind::Union));
        assert!(matches!(single_column(&l1.left), EvalExpr::Column(0)), "innermost left is `a`");
        assert!(matches!(single_column(&l1.right), EvalExpr::Column(1)), "innermost right is `b`");
        // Names still come from the leftmost arm.
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
    }

    // -----------------------------------------------------------------------
    // Width mismatch is a loud error.
    // -----------------------------------------------------------------------

    #[test]
    fn width_mismatch_is_a_loud_error() {
        let cat = cat_t();
        let err = plan_sql("SELECT a FROM t UNION SELECT a, b FROM t", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("do not have the same number of result columns"),
                "got {m:?}"
            ),
            other => panic!("expected a SQL arity error, got {other:?}"),
        }
    }

    #[test]
    fn width_mismatch_names_the_operator() {
        // The error text carries the offending operator (SQLite's wording).
        let cat = cat_t();
        let err = plan_sql("SELECT a FROM t EXCEPT SELECT a, b FROM t", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("EXCEPT"), "operator named in error; got {m:?}"),
            other => panic!("expected a SQL arity error, got {other:?}"),
        }
    }

    #[test]
    fn width_mismatch_left_wider_is_also_a_loud_error() {
        // The arity check is symmetric on width (`!=`, not `<`): a WIDER left arm must
        // error too. This is the mirror of the narrower-left cases above and guards
        // against a `<`-instead-of-`!=` regression that would only catch one direction.
        let cat = cat_t();
        let err = plan_sql("SELECT a, b FROM t UNION SELECT a FROM t", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("do not have the same number of result columns"),
                "got {m:?}"
            ),
            other => panic!("expected a SQL arity error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Outer ORDER BY / LIMIT wrap the whole compound.
    // -----------------------------------------------------------------------

    #[test]
    fn order_by_ordinal_wraps_the_setop_in_a_sort() {
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1);
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "ordinal 1 -> output col 0");
                assert!(!keys[0].desc);
                // The Sort wraps the SetOp directly.
                expect_setop(input);
            }
            other => panic!("expected Sort at the root, got {other:?}"),
        }
    }

    #[test]
    fn order_by_name_resolves_against_output_columns() {
        // ORDER BY references the LEFT arm's output name `a`, case-insensitively.
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT b FROM t ORDER BY A DESC", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1);
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "name `a` -> output col 0");
                assert!(keys[0].desc, "DESC preserved");
                expect_setop(input);
            }
            other => panic!("expected Sort at the root, got {other:?}"),
        }
    }

    #[test]
    fn order_by_explicit_collate_on_a_compound_term_is_honored() {
        // `COLLATE NOCASE` on the ORDER BY term is peeled off the (folded) expr and
        // used as the sort-key collation, overriding the default BINARY.
        let cat = cat_t();
        let plan =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY a COLLATE NOCASE", &cat)
                .unwrap();
        match &plan.root {
            PlanNode::Sort { keys, .. } => {
                assert_eq!(keys.len(), 1);
                assert_eq!(keys[0].collation, Collation::NoCase);
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_ordinal_inherits_the_output_column_dedup_collation() {
        // CONTRACT (durable across the first-cut fix): an ORDER BY ordinal with no
        // explicit COLLATE inherits the referenced OUTPUT column's dedup collation —
        // i.e. exactly the SetOp's `column_collations[idx]`. Asserting the relationship
        // (not the current BINARY value) keeps this test correct once per-column
        // collation is threaded through: if that fix sets `column_collations[0]` to
        // NoCase but forgets to propagate it into the inherited sort-key collation,
        // this fails. `c` is declared COLLATE NOCASE, so it also exercises that path.
        let cat = cat_t();
        let plan = plan_sql("SELECT c FROM t UNION SELECT c FROM t ORDER BY 1", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                let s = expect_setop(input);
                assert_eq!(
                    keys[0].collation, s.column_collations[0],
                    "ordinal term inherits the output column's dedup collation"
                );
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_out_of_range_ordinal_is_a_loud_error() {
        let cat = cat_t();
        let err =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY 2", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("out of range"), "got {m:?}"),
            other => panic!("expected an out-of-range error, got {other:?}"),
        }
    }

    #[test]
    fn order_by_unknown_name_is_a_loud_error() {
        let cat = cat_t();
        let err =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY nope", &cat).unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("does not match any column in the result set"), "got {m:?}")
            }
            other => panic!("expected a does-not-match error, got {other:?}"),
        }
    }

    #[test]
    fn order_by_qualified_name_is_rejected() {
        // A compound ORDER BY resolves ONLY to an output column (bare name or ordinal),
        // never a table-qualified reference; `t.a` (table = Some) must miss and error.
        // Matches real SQLite, which also rejects a qualified name in a compound ORDER
        // BY ("1st ORDER BY term does not match any column in the result set").
        let cat = cat_t();
        let err =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY t.a", &cat).unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("does not match any column in the result set"), "got {m:?}")
            }
            other => panic!("expected a does-not-match error, got {other:?}"),
        }
    }

    #[test]
    fn limit_wraps_the_whole_compound() {
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT a FROM t LIMIT 5", &cat).unwrap();
        match &plan.root {
            PlanNode::Limit { input, limit, offset } => {
                // Assert the bound value, not just presence: `LIMIT 5` must bind the
                // constant 5 into the limit slot, with no offset.
                assert!(
                    matches!(limit, Some(EvalExpr::Literal(Value::Integer(5)))),
                    "LIMIT binds the constant 5, got {limit:?}"
                );
                assert!(offset.is_none(), "no OFFSET, got {offset:?}");
                // No ORDER BY, so the implicit dedup sort now sits between the Limit and the
                // SetOp; peel it to reach the compound. (The Limit-over-Sort-over-SetOp shape
                // and the retention bound are pinned by `implicit_order_then_limit_*`.)
                root_setop(input);
            }
            other => panic!("expected Limit at the root, got {other:?}"),
        }
    }

    #[test]
    fn order_by_then_limit_nest_limit_outermost() {
        // Logical order: Limit { Sort { SetOp } }.
        let cat = cat_t();
        let plan =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1 LIMIT 5 OFFSET 2", &cat)
                .unwrap();
        match &plan.root {
            PlanNode::Limit { input, limit, offset } => {
                // `LIMIT 5 OFFSET 2` must bind 5 into limit and 2 into offset (not
                // swapped, not a wrong constant).
                assert!(
                    matches!(limit, Some(EvalExpr::Literal(Value::Integer(5)))),
                    "LIMIT binds 5, got {limit:?}"
                );
                assert!(
                    matches!(offset, Some(EvalExpr::Literal(Value::Integer(2)))),
                    "OFFSET binds 2, got {offset:?}"
                );
                match input.as_ref() {
                    PlanNode::Sort { input, limit: sort_limit, .. } => {
                        // The deterministic `LIMIT 5 OFFSET 2` is also pushed onto the
                        // Sort as its top-k retention bound (retain = offset + limit),
                        // carrying the SAME bound expressions as the Limit node above.
                        match sort_limit {
                            Some(sl) => {
                                assert!(
                                    matches!(sl.limit, EvalExpr::Literal(Value::Integer(5))),
                                    "Sort retention limit binds 5, got {:?}",
                                    sl.limit
                                );
                                assert!(
                                    matches!(sl.offset, Some(EvalExpr::Literal(Value::Integer(2)))),
                                    "Sort retention offset binds 2, got {:?}",
                                    sl.offset
                                );
                            }
                            None => panic!("expected the Sort to carry the LIMIT retention bound"),
                        }
                        expect_setop(input);
                    }
                    other => panic!("expected Sort under Limit, got {other:?}"),
                }
            }
            other => panic!("expected Limit at the root, got {other:?}"),
        }
    }

    #[test]
    fn order_by_nulls_first_last_and_default_propagate_to_the_sort_key() {
        // `NULLS FIRST`/`NULLS LAST` map to `SortKey.nulls_first` Some(true)/Some(false);
        // an absent NULLS clause leaves it None (the engine applies the SQL default).
        let cat = cat_t();
        let cases = [
            ("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1 NULLS FIRST", Some(true)),
            ("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1 NULLS LAST", Some(false)),
            ("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1", None),
        ];
        for (sql, want) in cases {
            let plan = plan_sql(sql, &cat).unwrap();
            match &plan.root {
                PlanNode::Sort { keys, .. } => {
                    assert_eq!(keys[0].nulls_first, want, "{sql}");
                }
                other => panic!("{sql}: expected Sort, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Implicit ORDER for a DEDUP compound with NO explicit ORDER BY: real SQLite
    // dedups via a sort, so UNION / INTERSECT / EXCEPT come back sorted ascending.
    // UNION ALL preserves concatenation order and is NOT sorted.
    // -----------------------------------------------------------------------

    #[test]
    fn implicit_order_union_no_order_by_wraps_in_ascending_sort() {
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT a FROM t", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1, "one key per output column");
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "sorts output col 0");
                assert!(!keys[0].desc, "implicit sort is ASCENDING");
                assert_eq!(keys[0].nulls_first, None, "implicit sort uses the SQL-default NULLS order");
                // The implicit Sort wraps the SetOp directly.
                let s = expect_setop(input);
                assert!(matches!(s.op, SetOpKind::Union));
            }
            other => panic!("expected an implicit Sort at the root, got {other:?}"),
        }
    }

    #[test]
    fn implicit_order_union_all_no_order_by_is_not_sorted() {
        // UNION ALL preserves order (left rows then right rows), so no Sort is added: its
        // root stays a bare SetOp. This is the control proving the implicit sort is keyed
        // to the DEDUP kinds only.
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION ALL SELECT a FROM t", &cat).unwrap();
        match &plan.root {
            PlanNode::SetOp(s) => assert!(matches!(s.op, SetOpKind::UnionAll)),
            other => {
                panic!("expected a bare SetOp (no implicit Sort) at the root, got {other:?}")
            }
        }
    }

    #[test]
    fn implicit_order_intersect_and_except_no_order_by_wrap_in_sort() {
        // The implicit sort fires for every DEDUP operator, not just UNION.
        let cat = cat_t();
        for (sql, want) in [
            ("SELECT a FROM t INTERSECT SELECT a FROM t", SetOpKind::Intersect),
            ("SELECT a FROM t EXCEPT SELECT a FROM t", SetOpKind::Except),
        ] {
            let plan = plan_sql(sql, &cat).unwrap();
            match &plan.root {
                PlanNode::Sort { input, keys, .. } => {
                    assert_eq!(keys.len(), 1, "{sql}: one key");
                    assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "{sql}: col 0");
                    assert!(!keys[0].desc, "{sql}: ascending");
                    let s = expect_setop(input);
                    assert_eq!(
                        std::mem::discriminant(&s.op),
                        std::mem::discriminant(&want),
                        "{sql}: wrapped SetOp keeps its kind"
                    );
                }
                other => panic!("{sql}: expected an implicit Sort, got {other:?}"),
            }
        }
    }

    #[test]
    fn implicit_order_multicolumn_sorts_every_output_column() {
        // The implicit sort is a FULL-ROW sort: one ascending key per output column, in
        // order, so two rows equal in column 0 are still ordered by column 1.
        let cat = cat_t();
        let plan = plan_sql("SELECT a, b FROM t UNION SELECT a, b FROM t", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 2, "one key per output column");
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "first key is col 0");
                assert!(matches!(keys[1].expr, EvalExpr::Column(1)), "second key is col 1");
                assert!(keys.iter().all(|k| !k.desc), "all keys ascending");
                expect_setop(input);
            }
            other => panic!("expected an implicit Sort, got {other:?}"),
        }
    }

    #[test]
    fn implicit_order_inherits_output_column_dedup_collation() {
        // CONTRACT: an implicit key reuses the referenced OUTPUT column's dedup collation —
        // exactly the SetOp's `column_collations[idx]` — so the sort order matches the
        // duplicate comparison. `c` is declared COLLATE NOCASE, exercising a non-BINARY
        // collation. Assert the RELATIONSHIP (not a literal value) so it stays correct as
        // collation threading evolves, mirroring `order_by_ordinal_inherits_...`.
        let cat = cat_t();
        let plan = plan_sql("SELECT c FROM t UNION SELECT c FROM t", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                let s = expect_setop(input);
                assert_eq!(
                    keys[0].collation, s.column_collations[0],
                    "implicit key inherits the output column's dedup collation"
                );
            }
            other => panic!("expected an implicit Sort, got {other:?}"),
        }
    }

    #[test]
    fn implicit_order_is_overridden_by_explicit_order_by() {
        // An explicit ORDER BY takes precedence over the implicit ascending sort: here
        // `ORDER BY 1 DESC` yields a single DESCENDING key, not the implicit ascending one.
        let cat = cat_t();
        let plan =
            plan_sql("SELECT a FROM t UNION SELECT a FROM t ORDER BY 1 DESC", &cat).unwrap();
        match &plan.root {
            PlanNode::Sort { input, keys, .. } => {
                assert_eq!(keys.len(), 1);
                assert!(matches!(keys[0].expr, EvalExpr::Column(0)));
                assert!(keys[0].desc, "explicit DESC wins over the implicit ASC");
                expect_setop(input);
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn implicit_order_then_limit_nests_limit_outermost_and_bounds_sort() {
        // `... UNION ... LIMIT 5` with NO ORDER BY: the implicit sort still fires and, being
        // directly under the Limit, carries the same deterministic top-k retention bound.
        // Logical shape: Limit { Sort { SetOp } } (mirrors `order_by_then_limit_...`).
        let cat = cat_t();
        let plan = plan_sql("SELECT a FROM t UNION SELECT a FROM t LIMIT 5", &cat).unwrap();
        match &plan.root {
            PlanNode::Limit { input, limit, offset } => {
                assert!(
                    matches!(limit, Some(EvalExpr::Literal(Value::Integer(5)))),
                    "LIMIT binds 5, got {limit:?}"
                );
                assert!(offset.is_none(), "no OFFSET, got {offset:?}");
                match input.as_ref() {
                    PlanNode::Sort { input, keys, limit: sort_limit } => {
                        assert!(matches!(keys[0].expr, EvalExpr::Column(0)), "implicit key col 0");
                        assert!(!keys[0].desc, "ascending");
                        match sort_limit {
                            Some(sl) => {
                                assert!(
                                    matches!(sl.limit, EvalExpr::Literal(Value::Integer(5))),
                                    "Sort retention limit binds 5, got {:?}",
                                    sl.limit
                                );
                                assert!(
                                    sl.offset.is_none(),
                                    "no retention offset, got {:?}",
                                    sl.offset
                                );
                            }
                            None => panic!("expected the implicit Sort to carry the LIMIT bound"),
                        }
                        expect_setop(input);
                    }
                    other => panic!("expected Sort under Limit, got {other:?}"),
                }
            }
            other => panic!("expected Limit at the root, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // No-FROM cores still compound (the smallest end-to-end shape).
    // -----------------------------------------------------------------------

    #[test]
    fn no_from_selects_compound() {
        let cat = TestCatalog::new(vec![]);
        let plan = plan_sql("SELECT 1 UNION SELECT 2", &cat).unwrap();
        let s = root_setop(&plan.root);
        assert!(matches!(s.op, SetOpKind::Union));
        assert_eq!(s.column_collations.len(), 1);
        // The output column is named after the LEFT arm's literal.
        assert_eq!(plan.result_columns, vec!["1".to_string()]);
    }
}
