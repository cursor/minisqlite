//! `UPDATE` compilation: lower a parsed [`minisqlite_sql::Update`] into a
//! [`PlanNode::Update`](crate::plan::PlanNode::Update) over the scanned rows.
//!
//! Delegation seam: [`compile_update`] is the single entrypoint the planner routes
//! every `UPDATE` through, so the feature lands here rather than in the shared
//! dispatcher.
//!
//! The scan's LEADING columns are the target row `[c0..c_{N-1}, rowid]` (width `N+1`,
//! rowid at register `N`, per the shared ROW/REGISTER convention in [`crate::plan`]). A
//! plain `UPDATE` scan is exactly that; an `UPDATE ... FROM` scan is the target
//! cross-joined with the FROM tables, so the FROM columns TRAIL the target's rowid. `SET`
//! values and the `WHERE` predicate bind against the combined (target + FROM) scope so a
//! FROM column resolves; `RETURNING` binds against a TARGET-ONLY scope
//! (`lang_returning.html`: auxiliary FROM tables may not participate in RETURNING).
//! Assignments carry the *schema column index* (0-based), not a register — the executor
//! overlays them onto the target's slice of the old row.

use minisqlite_catalog::TableDef;
use minisqlite_expr::EvalExpr;
use minisqlite_sql::{
    ConflictClause, Expr, FromClause, JoinKind, JoinOperator, JoinTree, ResultColumn, SetClause,
    TableOrSubquery,
};
use minisqlite_types::{Affinity, DbIndex, Error, Result};

use crate::access::SeqScan;
use crate::access_path::plan_table_access;
use crate::bind::scope::is_rowid_keyword;
use crate::bind::{bind_expr, Scope, Source};
use crate::colname::result_column_name;
use crate::compile::check::compile_checks;
use crate::compile::cte::register_with;
use crate::compile::from::{build_from, resolve_from};
use crate::compile::index_expr::{
    compile_table_index_key_exprs, compile_table_index_partial_predicates,
};
use crate::compile::select::compile_columnlist_subquery;
use crate::compile::trigger::{compile_triggers, TriggerDmlEvent};
use crate::plan::{OnConflict, PlanNode, Update};
use crate::plan_ctx::PlanCtx;

/// Compile a top-level `UPDATE` into its plan tree and output column names (the latter
/// empty unless a `RETURNING` clause is present). See [`compile_update_with_parent`] for
/// the trigger-action form.
pub fn compile_update(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Update,
) -> Result<(PlanNode, Vec<String>)> {
    compile_update_with_parent(ctx, stmt, None)
}

/// Compile an `UPDATE`, optionally as a trigger ACTION under an enclosing OLD/NEW
/// `parent` scope. With `parent = Some(..)` the target's own columns sit at
/// `base = parent.total_width()` (= `2W`) and `NEW`/`OLD` resolve through the parent;
/// NO triggers are attached (only a TOP-LEVEL UPDATE attaches its direct triggers, so
/// expansion stays one level deep). `None` is exactly the top-level path.
pub fn compile_update_with_parent(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Update,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    // Pull the catalog reference out of `ctx` first (it is `Copy`), so the borrowed
    // `td` has the catalog's lifetime rather than a borrow of `ctx` — leaving `&mut
    // ctx` free for the binder below. (Same borrow shape as `compile::select`.)
    let catalog = ctx.catalog;
    // Resolve the target BEFORE the unsupported-clause gaps below: SQLite reports a
    // view/missing target first (`cannot modify v because it is a view` / `no such
    // table`), so `UPDATE a_view ... FROM t` must not report the FROM gap instead. A VIEW
    // owns no b-tree; it is updatable ONLY through a matching `INSTEAD OF UPDATE` trigger
    // (respecting `UPDATE OF <cols>` against the assigned columns) — when one exists,
    // redirect the UPDATE through it (`compile::instead_of`). Without such a trigger it
    // keeps SQLite's exact `cannot modify … because it is a view`.
    // Resolve the target NAMESPACE (schema qualifier → fixed DbIndex; unqualified → search
    // order, temp shadows main), then fetch the table FROM it so the def, scan, and node all
    // name the same store. An unknown qualifier / a name in no live namespace is "no such
    // table". The `db` is reused for the source, the scan, and the Update node below.
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
                    crate::compile::instead_of::compile_view_update(ctx, db, view, stmt, parent)
                }
                None => Err(Error::sql(format!(
                    "no such table: {}",
                    crate::compile::from::qualified_table_name(&stmt.table)
                ))),
            };
        }
    };

    // A leading `WITH` registers its CTEs (mirroring the SELECT path) so the UPDATE's WHERE
    // and SET subqueries resolve the CTE names via the shared FROM/`lookup_cte` path. The
    // compiled CtePlans ride in `ctx.ctes` -> `Plan::ctes`, which the executor materializes
    // before the UPDATE (the SAME plan-level attach SELECT uses — the Update node is
    // unchanged). The guard keeps the CTEs visible through SET/WHERE/RETURNING binding below,
    // then is `drop`ped explicitly BEFORE `compile_triggers` (see there) so they are invisible
    // inside trigger bodies (an early `?` before that drop still unwinds the scope via RAII).
    let cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    let table = td.name.clone();
    let n = td.columns.len();

    // A trigger action places the target's own columns AFTER the OLD++NEW row
    // (`base = 2W`); a top-level UPDATE has no outer row (`base = 0`). Uses the shared
    // child-placement helper (identical to `total_width()` for the `grouping = None` parents
    // this is reached with) so the outer-width rule has one authority — see `compile_subplan`.
    let base_offset = parent.map_or(0, |p| p.outer_row_width_for_child());

    // UPDATE ... FROM as a trigger ACTION (parent set) stays a loud gap: the exec-side
    // join would have to forward the OLD/NEW frame to each leaf without re-prepending it
    // (the same restriction `compile/select.rs` documents for a correlated FROM-join at a
    // nonzero base). No in-scope test needs it; reject rather than silently mis-bind.
    if stmt.from.is_some() && parent.is_some() {
        return Err(Error::sql("UPDATE ... FROM inside a trigger action is not yet supported"));
    }

    // Lower `UPDATE t ... FROM <f>` by synthesizing the combined FROM tree `t , <f>` so the
    // TARGET is source 0: its row `[c0..c_{n-1}, rowid]` lands at registers
    // `[base_offset .. base_offset+n]` with the rowid at `base_offset+n` — exactly the
    // single-table layout the executor already reads (it then ignores the trailing FROM
    // columns). The FROM tables cross-join (comma) after it, and the WHERE clause performs
    // the real join + filter (`build_from` folds equijoins into the join, any remainder as
    // the residual Filter below). A plain UPDATE (no
    // FROM) keeps the single BaseTable source unchanged (`combined_from` is `None`).
    let combined_from: Option<FromClause> = stmt.from.as_ref().map(|from_tree| {
        let target_tos = TableOrSubquery::Table {
            name: stmt.table.clone(),
            alias: stmt.alias.clone(),
            indexed: stmt.indexed.clone(),
        };
        // A `JoinTree::Join`'s `right` is one `TableOrSubquery`: a single FROM leaf is used
        // directly; a FROM that is itself a join becomes a parenthesized sub-join operand
        // (its sources still resolve to the same flat left-to-right registers).
        let right = match from_tree {
            JoinTree::Table(tos) => tos.clone(),
            join => TableOrSubquery::Join(Box::new(join.clone())),
        };
        JoinTree::Join {
            left: Box::new(JoinTree::Table(target_tos)),
            op: JoinOperator { natural: false, kind: JoinKind::Comma },
            right,
            constraint: None,
        }
    });

    // Phase 1: resolve the sources (and NATURAL/USING coalesced columns). For UPDATE...FROM
    // `resolve_from` places the target at source 0 (base `base_offset`) then the FROM tables
    // after it; a plain UPDATE is just the single target source with no coalesced columns.
    // The owned `sources`/`coalesced` locals are declared before `scope` so their borrows
    // outlive it, and they borrow the catalog (lifetime `'e`), not `ctx` — leaving `&mut
    // ctx` free for the binders / `build_from` below (the borrow shape `select.rs` relies on).
    let (sources, coalesced) = if combined_from.is_some() {
        resolve_from(catalog, &combined_from, base_offset)?
    } else {
        let src = Source::BaseTable {
            exposed_name: stmt.alias.clone().unwrap_or_else(|| td.name.clone()),
            table: td,
            db,
            base: base_offset,
        };
        (vec![src], Vec::new())
    };
    // A struct literal (not `Scope::with_parent`) so the resolved `coalesced` slice rides in
    // the scope; for the plain path `coalesced` is empty, so binding is identical to before.
    let scope = Scope {
        sources: sources.as_slice(),
        coalesced: coalesced.as_slice(),
        parent,
        grouping: None,
        saw_correlated: None,
        correlated_cols: None,
        nondeterministic: None,
        windowing: None,
    };

    // Bind in TEXTUAL order — SET, then WHERE, then RETURNING — so anonymous `?`
    // parameters receive the numbers SQLite assigns (parameter numbering follows
    // source order, and the grammar is `SET .. WHERE .. RETURNING`). A SET *target* name
    // resolves against `td` (the target's own columns, via `column_index`); SET *value*
    // exprs and WHERE bind against the combined scope, so `o.v` / `t.v` / `daily.amt` all
    // resolve to their register in the joined row.
    let assignments = bind_assignments(&scope, ctx, td, &stmt.set)?;

    let bound_where = stmt.where_clause.as_ref().map(|w| bind_expr(&scope, ctx, w)).transpose()?;

    // Phase 2: build the scan. UPDATE...FROM builds the target×FROM join and applies the
    // WHERE against it (`build_from` folds equijoins into the join, any remainder as the
    // residual Filter) — this yields inner-join semantics: an unmatched
    // target row produces no join row, so it is never buffered and thus left unchanged. A
    // plain UPDATE keeps the existing index-aware single-table access path.
    let mut scan = if combined_from.is_some() {
        let (node, residual) = build_from(ctx, &scope, &combined_from, base_offset, bound_where)?;
        match residual {
            Some(pred) => PlanNode::Filter { input: Box::new(node), predicate: pred },
            None => node,
        }
    } else {
        build_scan(catalog, td, db, base_offset, bound_where)?
    };
    // SET and WHERE (the only clauses that can reach an `UPDATE ... FROM json_each(...)`
    // auxiliary table — RETURNING is bound target-only below) are now bound, so the JSON
    // TVF leaf can drop its per-row document copy unless its hidden `json` column was read.
    // Same lazy-materialization rule as SELECT; a no-op for a plain UPDATE (no TVF leaf).
    crate::compile::select::finalize_tvf_emit_json(&mut scan, &sources);

    let column_affinities: Vec<Affinity> =
        (0..td.columns.len()).map(|i| td.column_affinity(i)).collect();

    let on_conflict = map_on_conflict(stmt.or_conflict);

    // RETURNING may reference ONLY the target table: `lang_returning.html` — "In an UPDATE
    // FROM statement, the auxiliary tables named in the FROM clause may not participate in
    // the RETURNING clause." So bind it against a TARGET-ONLY scope (`sources[0]` is always
    // the target), NOT the combined join scope: `*` then expands to the target's columns and
    // a FROM-column reference is a loud "no such column/table". This is also load-bearing for
    // exec — the executor evaluates RETURNING against the rebuilt TARGET row
    // `[c0..c_{n-1}, rowid]` (width `n+1`), never the wide join row, so a RETURNING register
    // beyond the target would be out of range. A target-only scope carries no coalesced
    // columns, so `with_parent` builds it; for a plain UPDATE `sources` is already just the
    // target, so this spans exactly the target either way.
    let returning_scope = Scope::with_parent(&sources[..1], parent);
    let (returning, result_columns) = bind_returning(&returning_scope, ctx, &stmt.returning)?;

    // Drop the CTE name scope BEFORE compiling triggers: a trigger body is a separate
    // program compiled under a fresh `PlanCtx` that shares the thread-local CTE scope, and a
    // firing statement's CTE must be invisible inside it (`lang_with.html` §1 — see
    // `compile_insert_with_parent` for the full rationale). SET/WHERE/RETURNING binding is
    // already complete above; dropping pops only the name->id entries, leaving `ctx.ctes`
    // (-> the Update plan's CTEs) intact. (Regression: `conformance_dml_cte_trigger`.)
    drop(cte_guard);

    // Attach the triggers this UPDATE fires — only for a TOP-LEVEL statement (see
    // `compile_insert_with_parent` for why a trigger action does not recurse). `UPDATE
    // OF <cols>` filtering keys off the columns actually assigned by this SET clause.
    let triggers = if parent.is_none() {
        let changed: Vec<usize> = assignments.iter().map(|(i, _)| *i).collect();
        compile_triggers(ctx, catalog, db, td, TriggerDmlEvent::Update { changed_cols: &changed })?
    } else {
        Vec::new()
    };

    // The target's CHECK constraints, bound against the row layout `[c0..c_{N-1}, rowid]`.
    // The executor evaluates these against the POST-assignment new row — the SAME layout,
    // so no register remapping is needed here. UNLIKE triggers, checks attach regardless
    // of `parent`: a CHECK is enforced on every updated row, including one a trigger action
    // updates (no re-expansion concern — the executor just evaluates the predicate).
    let checks = compile_checks(ctx, td)?;

    // Expression-index key programs, bound against the target's OWN `[c0..c_{N-1}, rowid]`
    // frame (base 0) — NOT any `UPDATE … FROM` join scope — so the executor maintains a
    // `CREATE INDEX i ON t(<expr>)` against the post-assignment new row it stores
    // (`lang_createindex.html` §1.2). Empty when no index on `t` has an expression key.
    let index_key_exprs = compile_table_index_key_exprs(ctx, db, td)?;

    // Partial-index WHERE predicates, bound against that SAME `[c0..c_{N-1}, rowid]` frame.
    // The executor evaluates each against the OLD row (to decide whether an entry exists to
    // delete) and the NEW row (to decide whether to add one), so a row moving OUT of the
    // predicate has its entry dropped and a row moving IN gains one — `partialindex.html` §2.
    // Empty when no index on `t` is partial.
    let index_partial_predicates = compile_table_index_partial_predicates(ctx, db, td)?;

    let node = PlanNode::Update(Update {
        table,
        db,
        column_count: n,
        assignments,
        scan: Box::new(scan),
        column_affinities,
        on_conflict,
        returning,
        triggers,
        checks,
        index_key_exprs,
        index_partial_predicates,
    });
    Ok((node, result_columns))
}

/// Build the row-source scan for the update target.
///
/// At `base_offset == 0` (a top-level UPDATE) this uses the index-aware
/// [`plan_table_access`] as before. At `base_offset > 0` (a trigger action) the WHERE
/// binds the target's own columns at `base_offset + i` and OLD/NEW at the leading
/// `[0, 2W)`, but `plan_table_access` assumes 0-based table registers: a low-register
/// OLD/NEW reference (`WHERE OLD.x = 5`) would collide with a 0-based column/rowid
/// register and be mis-folded into a rowid/index SEEK — a WRONG plan that touches the
/// wrong rows. So a trigger action emits a plain [`SeqScan`] with the whole predicate as
/// a residual [`Filter`](PlanNode::Filter): correct at any base. (Index-aware access at a
/// nonzero base — rebasing the predicate around `plan_table_access` — is a follow-up.)
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

/// Bind the `SET` list into `(schema column index, value expr)` pairs, evaluated
/// against the scanned row. A `col = expr` maps `col` to its schema index. A
/// column-list target (`lang_update.html`, rowvalue.html §2.3) is handled per its
/// right-hand shape:
/// * a parenthesized row value `(a, b) = (x, y)` — must be "of the same size" as the
///   name list; matched arity binds each pair, a size mismatch is a loud error;
/// * a single-name list `(a) = expr` — just `a = expr` (the parser unwraps `(x)` to the
///   scalar `x`, so `(a) = (SELECT x)` is the scalar-subquery assignment `a = (SELECT x)`);
/// * a subquery row-value source `(a, b) = (SELECT x, y …)` — the subquery is compiled
///   once and each name reads its positional column of the first result row (a size
///   mismatch is the same loud error as the parenthesized case);
/// * a multi-name list fed by any other bare scalar (`(a, b) = 1`) — a size mismatch.
///
/// A column named more than once across the `SET` list is NOT rejected: SQLite keeps
/// only the rightmost occurrence, which the executor reproduces by applying the
/// assignments in order (last write wins), so the duplicate pair is emitted as-is.
fn bind_assignments(
    scope: &Scope,
    ctx: &mut PlanCtx,
    td: &TableDef,
    set: &[SetClause],
) -> Result<Vec<(usize, EvalExpr)>> {
    let mut out = Vec::new();
    for clause in set {
        match clause {
            SetClause::Column { name, value } => {
                let idx = column_index(td, name)?;
                let bound = bind_expr(scope, ctx, value)?;
                out.push((idx, bound));
            }
            SetClause::Columns { names, value } => match value {
                // `(a, b, ...) = (x, y, ...)`: one assignment per name, in source
                // order (so the row value's `?` params number left to right).
                Expr::Parenthesized(list) if list.len() == names.len() => {
                    for (nm, item) in names.iter().zip(list.iter()) {
                        let idx = column_index(td, nm)?;
                        let bound = bind_expr(scope, ctx, item)?;
                        out.push((idx, bound));
                    }
                }
                // A parenthesized row value whose width differs from the name list is
                // an error, not an unimplemented feature: the row value must be "of the
                // same size" (`lang_update.html`).
                Expr::Parenthesized(list) => {
                    return Err(Error::sql(format!(
                        "{} columns assigned {} values",
                        names.len(),
                        list.len()
                    )))
                }
                // `(a) = expr`: a single-name column list takes a scalar right-hand
                // side. The parser unwraps `(x)` to the scalar `x`, so a valid
                // `(a) = (1)` arrives here as `a = 1` (and `(a) = (SELECT x)` as the
                // scalar-subquery assignment `a = (SELECT x)`) — checked BEFORE the
                // subquery-source arm so a single-name list never takes the row-value
                // path.
                _ if names.len() == 1 => {
                    let idx = column_index(td, &names[0])?;
                    let bound = bind_expr(scope, ctx, value)?;
                    out.push((idx, bound));
                }
                // `(a, b, …) = (SELECT x, y, …)`: a subquery ROW-VALUE source
                // (rowvalue.html §2.3). Compile the subquery ONCE (a correlated
                // scalar-style subplan over the target row — `… WHERE src.k = t.k`
                // reads the outer row through the same plumbing as a correlated scalar
                // subquery), then emit one assignment per name reading its positional
                // column of the subquery's first result row. The subquery must be "of
                // the same size" as the name list: a width mismatch is the same
                // `"N columns assigned M values"` error SQLite raises (and that the
                // parenthesized arm above uses), NOT a silent truncate/pad.
                Expr::Subquery(select) => {
                    let (id, width) = compile_columnlist_subquery(scope, ctx, select)?;
                    if width != names.len() {
                        return Err(Error::sql(format!(
                            "{} columns assigned {} values",
                            names.len(),
                            width
                        )));
                    }
                    for (i, nm) in names.iter().enumerate() {
                        let idx = column_index(td, nm)?;
                        out.push((idx, EvalExpr::ScalarSubqueryColumn { id, col: i }));
                    }
                }
                // A multi-name list fed by a bare scalar (not a parenthesized row value
                // and not a subquery) — e.g. `(a, b) = 1` — is a size mismatch: one
                // value for N columns, the same SQLite error as a too-narrow row value.
                _ => {
                    return Err(Error::sql(format!("{} columns assigned 1 values", names.len())))
                }
            },
        }
    }
    Ok(out)
}

/// Resolve a `SET`-target column name to its 0-based schema index (case-insensitive).
///
/// `rowid`/`_rowid_`/`oid` write the rowid (`rowidtable.html`: it "can be accessed or
/// changed by reading or writing to any of the 'rowid' or 'oid' or '_rowid_' columns")
/// UNLESS a real column shadows the name — so a declared column is resolved first, and
/// only then does a rowid keyword fall back to the alias. On an INTEGER PRIMARY KEY
/// table the alias column IS the rowid and the executor moves the rowid when that column
/// index is assigned (`minisqlite-exec/src/ops/update.rs` derives `new_rowid` from the
/// alias slot), so the keyword maps to `td.rowid_alias`. On a PLAIN rowid table there is
/// no alias column to carry the new rowid through the current `Update` node, so
/// `SET rowid = ...` is a loud gap (needs executor support) — never a silent no-op or
/// mis-store. This mirrors INSERT's `resolve_target_columns`/`rowid_target_index`.
fn column_index(td: &TableDef, name: &str) -> Result<usize> {
    if let Some(i) = td.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name)) {
        // A GENERATED column may not be assigned (`gencol.html` §2: "their values can not
        // be directly written"). Real SQLite's exact message is
        // `cannot UPDATE generated column "<name>"`. Every SET-target form (a plain
        // `col = ...`, a `(a, b) = ...` row value, a subquery source) resolves through here,
        // so one guard rejects them all.
        if td.columns[i].generated.is_some() {
            return Err(Error::sql(format!(
                "cannot UPDATE generated column \"{}\"",
                td.columns[i].name
            )));
        }
        return Ok(i);
    }
    if is_rowid_keyword(name) {
        return match td.rowid_alias {
            Some(i) => Ok(i),
            None => Err(Error::sql(format!(
                "UPDATE with an explicit rowid target ({name}) on a table with no INTEGER \
                 PRIMARY KEY is not yet supported"
            ))),
        };
    }
    Err(Error::sql(format!("no such column: {name}")))
}

/// Map a parsed `OR <algorithm>` (or its absence) to the plan's [`OnConflict`]. A
/// bare `UPDATE` with no `OR` clause is `ABORT` (SQLite's default).
fn map_on_conflict(c: Option<ConflictClause>) -> OnConflict {
    match c {
        None | Some(ConflictClause::Abort) => OnConflict::Abort,
        Some(ConflictClause::Rollback) => OnConflict::Rollback,
        Some(ConflictClause::Fail) => OnConflict::Fail,
        Some(ConflictClause::Ignore) => OnConflict::Ignore,
        Some(ConflictClause::Replace) => OnConflict::Replace,
    }
}

/// Bind a `RETURNING` list into `(exprs, output names)` against the scanned
/// `[cols.., rowid]` row — the same projection semantics as a SELECT result column:
/// `*` / `table.*` expand in place, an aliased/bare expression takes its alias or its
/// reconstructed source-text name. An empty clause yields two empty vectors.
///
/// NOTE: an identical helper lives in the sibling `compile::delete`; the two DML
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
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef};
    use minisqlite_expr::CmpOp;
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Value;

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

    /// `k(id INTEGER PRIMARY KEY, v TEXT)` — `id` is the rowid alias (index 0) — plus
    /// `r(rowid INTEGER, v TEXT)`, a table with a REAL column literally named `rowid`
    /// (not an INTEGER PRIMARY KEY, so `rowid_alias` is None), used to check a declared
    /// column shadows the rowid keyword.
    fn cat_k() -> TestCatalog {
        TestCatalog {
            tables: vec![
                TableDef {
                    name: "k".to_string(),
                    columns: vec![col("id", Some("INTEGER")), col("v", Some("TEXT"))],
                    root_page: 2,
                    without_rowid: false,
                    rowid_alias: Some(0),
                    auto_indexes: Vec::new(),
                    checks: Vec::new(),
                    foreign_keys: Vec::new(),
                    autoincrement: false,
                    primary_key: Vec::new(),
                    strict: false,
                },
                TableDef {
                    name: "r".to_string(),
                    columns: vec![col("rowid", Some("INTEGER")), col("v", Some("TEXT"))],
                    root_page: 3,
                    without_rowid: false,
                    rowid_alias: None,
                    auto_indexes: Vec::new(),
                    checks: Vec::new(),
                    foreign_keys: Vec::new(),
                    autoincrement: false,
                    primary_key: Vec::new(),
                    strict: false,
                },
            ],
        }
    }

    fn plan_sql(sql: &str, cat: &dyn Catalog) -> Result<Plan> {
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("expected one statement");
        QueryPlanner::new().plan(stmt, cat)
    }

    fn expect_update(plan: &Plan) -> &Update {
        match &plan.root {
            PlanNode::Update(u) => u,
            other => panic!("expected Update at the root, got {other:?}"),
        }
    }

    #[test]
    fn rowid_eq_update_is_a_rowid_seek_scan() {
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1 WHERE rowid = 5", &cat).unwrap();
        assert!(plan.mutates, "UPDATE mutates");
        let u = expect_update(&plan);
        assert_eq!(u.table, "t");
        assert_eq!(u.column_count, 3);
        assert_eq!(u.column_affinities.len(), 3, "one affinity per table column");
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 0, "column `a` is schema index 0");
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(1))));
        // `rowid = 5` is fully consumed by the access path, so the scan is a bare
        // RowidScan seek — no residual Filter.
        match u.scan.as_ref() {
            PlanNode::RowidScan(s) => {
                assert!(matches!(s.op, RowidOp::Eq(_)), "rowid = 5 is an Eq seek");
                assert_eq!(s.column_count, 3);
            }
            other => panic!("expected a RowidScan seek, got {other:?}"),
        }
        assert!(u.returning.is_empty());
        assert!(plan.result_columns.is_empty(), "no RETURNING => no result columns");
    }

    #[test]
    fn no_where_update_scans_the_whole_table() {
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET b = 'x'", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 1, "column `b` is schema index 1");
        assert!(matches!(u.scan.as_ref(), PlanNode::SeqScan(_)), "no WHERE => full SeqScan");
    }

    #[test]
    fn unknown_set_column_is_a_loud_error() {
        let cat = cat_t();
        let err = plan_sql("UPDATE t SET nope = 1", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such column"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_table_is_a_loud_error() {
        let cat = cat_t();
        let err = plan_sql("UPDATE missing SET a = 1", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such table"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    // ---- SET rowid / _rowid_ / oid (write-target rowid, mirrors INSERT's GOAL 2) -----

    #[test]
    fn set_rowid_maps_to_the_integer_primary_key_alias() {
        // `k(id INTEGER PRIMARY KEY, v)`: `SET rowid` targets the alias column (index 0) —
        // the same assignment `SET id = ...` produces — so the executor moves the rowid.
        let cat = cat_k();
        let plan = plan_sql("UPDATE k SET rowid = 10", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 0, "rowid -> alias column index 0");
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(10))));
    }

    #[test]
    fn set_rowid_keyword_variants_all_map_to_the_alias() {
        // Every spelling (case-insensitive) resolves to the alias index on an IPK table.
        let cat = cat_k();
        for kw in ["rowid", "_rowid_", "oid", "RoWiD", "OID"] {
            let sql = format!("UPDATE k SET {kw} = 7");
            let plan = plan_sql(&sql, &cat).unwrap_or_else(|e| panic!("plan {sql:?}: {e:?}"));
            assert_eq!(expect_update(&plan).assignments[0].0, 0, "{kw} -> alias index 0");
        }
    }

    #[test]
    fn set_real_column_named_rowid_shadows_the_keyword() {
        // `r(rowid INTEGER, v)`: a declared column named `rowid` is that column, NOT the
        // rowid pseudo-column, so it resolves to its own index and is not treated as a gap.
        let cat = cat_k();
        let plan = plan_sql("UPDATE r SET rowid = 5", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments[0].0, 0, "the real `rowid` column is index 0");
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(5))));
    }

    #[test]
    fn set_rowid_on_a_plain_rowid_table_is_a_loud_gap() {
        // `t` has no INTEGER PRIMARY KEY and no real `rowid` column, so the current Update
        // node cannot carry a new rowid — a loud, specific error, never a silent no-op or
        // mis-store (the executor derives new_rowid from the alias slot alone).
        let cat = cat_t();
        let err = plan_sql("UPDATE t SET rowid = 5", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("not yet supported"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn returning_binds_against_the_scanned_row() {
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1 RETURNING a", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 1);
        assert!(matches!(u.returning[0], EvalExpr::Column(0)), "`a` reads register 0");
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
    }

    #[test]
    fn returning_rowid_reads_the_trailing_register() {
        // RETURNING binds against `[c0,c1,c2, rowid]`, so `rowid` is register N = 3.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1 RETURNING rowid", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 1);
        assert!(matches!(u.returning[0], EvalExpr::Column(3)), "rowid is the trailing register N=3");
        assert_eq!(plan.result_columns, vec!["rowid".to_string()]);
    }

    #[test]
    fn column_list_row_value_assigns_each_target() {
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET (a, b) = (1, 2)", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 2);
        assert_eq!(u.assignments[0].0, 0);
        assert_eq!(u.assignments[1].0, 1);
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(1))));
        assert!(matches!(u.assignments[1].1, EvalExpr::Literal(Value::Integer(2))));
    }

    #[test]
    fn column_list_arity_mismatch_is_a_size_error() {
        // A row value must be "of the same size" as the name list (`lang_update.html`);
        // a width mismatch is a real SQLite error, not an unimplemented feature.
        let cat = cat_t();
        let err = plan_sql("UPDATE t SET (a, b) = (1, 2, 3)", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("2 columns assigned 3 values"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn single_name_column_list_is_a_plain_assignment() {
        // The parser unwraps `(1)` to the scalar `1`, so `SET (a) = (1)` is just
        // `a = 1` — SQLite accepts it; it must not fall into the subquery-source gap.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET (a) = (1)", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 0, "single-name list `(a)` targets column a");
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(1))));
    }

    #[test]
    fn column_list_from_subquery_source_binds_scalar_subquery_columns() {
        // `(a, b) = (SELECT 1, 2)`: the subquery ROW-VALUE source (rowvalue.html §2.3)
        // compiles ONCE into `plan.subqueries` and each name takes its positional column
        // of the first result row — assignment i is `ScalarSubqueryColumn { id, col: i }`,
        // all sharing the SAME subquery id (so the subplan runs once per row, not once per
        // column). Was previously a loud "not yet supported" gap.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET (a, b) = (SELECT 1, 2)", &cat).unwrap();
        assert_eq!(plan.subqueries.len(), 1, "one subplan registered for the source");
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 2, "one assignment per name");
        assert_eq!(u.assignments[0].0, 0, "first name -> column a (index 0)");
        assert_eq!(u.assignments[1].0, 1, "second name -> column b (index 1)");
        let id0 = match u.assignments[0].1 {
            EvalExpr::ScalarSubqueryColumn { id, col } => {
                assert_eq!(col, 0, "column a reads subquery column 0");
                id
            }
            ref other => panic!("expected ScalarSubqueryColumn, got {other:?}"),
        };
        match u.assignments[1].1 {
            EvalExpr::ScalarSubqueryColumn { id, col } => {
                assert_eq!(col, 1, "column b reads subquery column 1");
                assert_eq!(id, id0, "both assignments share ONE subplan id");
            }
            ref other => panic!("expected ScalarSubqueryColumn, got {other:?}"),
        }
    }

    #[test]
    fn column_list_subquery_source_width_mismatch_is_a_size_error() {
        // A subquery source must be "of the same size" as the name list (rowvalue.html
        // §2 / lang_update.html §2). A width mismatch in EITHER direction is a size error
        // — the SAME `"N columns assigned M values"` message SQLite and the parenthesized
        // row-value arm use — NOT a silent truncate/pad and NOT the old "not yet
        // supported" gap.
        let cat = cat_t();
        let wider = plan_sql("UPDATE t SET (a, b) = (SELECT 1, 2, 3)", &cat).unwrap_err();
        match wider {
            Error::Sql(m) => assert!(m.contains("2 columns assigned 3 values"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
        let narrower = plan_sql("UPDATE t SET (a, b) = (SELECT 1)", &cat).unwrap_err();
        match narrower {
            Error::Sql(m) => assert!(m.contains("2 columns assigned 1 values"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn column_list_scalar_source_to_multi_name_list_is_a_size_error() {
        // A bare scalar RHS to a multi-name list (`(a, b) = 1`, not a parenthesized row
        // value and not a subquery) is one value for two columns — a size mismatch, the
        // same error SQLite raises. Guards the final `_` arm of `bind_assignments`.
        let cat = cat_t();
        let err = plan_sql("UPDATE t SET (a, b) = 1", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("2 columns assigned 1 values"), "got {m:?}"),
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_set_target_keeps_both_assignments_in_order() {
        // SQLite does NOT reject a repeated SET column; it keeps the rightmost
        // (`lang_update.html`). The executor reproduces that by applying assignments in
        // order, so both pairs are emitted and the later write wins — pinned here so a
        // well-meaning "dedup/reject" change can't silently diverge from SQLite.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1, a = 2", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 2, "both assignments to `a` are kept");
        assert_eq!(u.assignments[0].0, 0);
        assert_eq!(u.assignments[1].0, 0);
        assert!(matches!(u.assignments[0].1, EvalExpr::Literal(Value::Integer(1))));
        assert!(matches!(u.assignments[1].1, EvalExpr::Literal(Value::Integer(2))), "rightmost wins");
    }

    #[test]
    fn set_then_where_number_params_in_textual_order() {
        // SQLite numbers `?` in source order, and the grammar is `SET .. WHERE ..`, so
        // the SET `?` is param 1 and the WHERE `?` is param 2. Binding WHERE first would
        // swap them — this pins the textual-order binding.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = ? WHERE b = ?", &cat).unwrap();
        let u = expect_update(&plan);
        assert!(matches!(u.assignments[0].1, EvalExpr::Param(1)), "the SET `?` is param 1");
        // `b = ?` (b is not the rowid) becomes the residual Filter predicate.
        match u.scan.as_ref() {
            PlanNode::Filter { predicate, .. } => match predicate {
                EvalExpr::Compare { op: CmpOp::Eq, left, right, .. } => {
                    assert!(matches!(left.as_ref(), EvalExpr::Column(1)), "left is column `b`");
                    assert!(matches!(right.as_ref(), EvalExpr::Param(2)), "the WHERE `?` is param 2");
                }
                other => panic!("expected an Eq compare, got {other:?}"),
            },
            other => panic!("expected a Filter over the scan, got {other:?}"),
        }
    }

    #[test]
    fn update_from_builds_a_join_scan() {
        // `UPDATE ... FROM` compiles into an Update whose scan joins the target (source 0)
        // with the FROM table, with the WHERE `o.b = t.b` folded into that join. The
        // target's `column_count` is unchanged (it is the TARGET's column count — the
        // executor reads the leading target row `[c0..c_{n-1}, rowid]` and ignores the
        // trailing FROM columns). Uses a self-join `t AS o` so the static test catalog
        // (one table) suffices.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = o.a FROM t AS o WHERE o.b = t.b", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.table, "t");
        assert_eq!(u.column_count, 3, "column_count is the TARGET's column count, not the join width");
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 0, "SET targets column a (index 0)");
        // t(a,b,c) has no rowid alias, so it occupies 4 registers ([a,b,c,rowid] = regs
        // 0..3); the FROM copy `o` is source 1 at base 4, so `o.a` reads register 4 —
        // beyond the target row, proving the value binds against the combined join scope.
        assert!(
            matches!(u.assignments[0].1, EvalExpr::Column(4)),
            "o.a is register 4 (FROM source at base 4), got {:?}",
            u.assignments[0].1
        );
        // The scan is the target×FROM join directly (the WHERE equijoin folds into it).
        assert!(
            matches!(u.scan.as_ref(), PlanNode::Join(_)),
            "the scan is the target×FROM join, got {:?}",
            u.scan
        );
    }

    #[test]
    fn update_from_returning_sees_only_the_target() {
        // lang_returning.html: "In an UPDATE FROM statement, the auxiliary tables named in
        // the FROM clause may not participate in the RETURNING clause." So `RETURNING *`
        // expands to the TARGET's columns only — for t(a,b,c) (no rowid alias) that is
        // registers 0,1,2 — and a reference to a FROM (auxiliary) table is a loud bind
        // error. This also keeps every RETURNING register inside the target row the
        // executor rebuilds RETURNING against (`[c0..c_{n-1}, rowid]`, width n+1).
        let cat = cat_t();
        let plan =
            plan_sql("UPDATE t SET a = o.a FROM t AS o WHERE o.b = t.b RETURNING *", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 3, "RETURNING * is the 3 TARGET columns, not the join width");
        assert!(matches!(u.returning[0], EvalExpr::Column(0)));
        assert!(matches!(u.returning[1], EvalExpr::Column(1)));
        assert!(matches!(u.returning[2], EvalExpr::Column(2)));
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        // A RETURNING reference to the FROM (auxiliary) table `o` is rejected at bind time
        // (the RETURNING scope contains only the target), matching SQLite.
        let err = plan_sql("UPDATE t SET a = o.a FROM t AS o WHERE o.b = t.b RETURNING o.a", &cat)
            .unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("no such table: o"), "expected the FROM table unresolved, got {m:?}")
            }
            other => panic!("expected a Sql error, got {other:?}"),
        }
    }

    #[test]
    fn with_on_update_registers_cte_and_compiles() {
        // A leading `WITH` on UPDATE is supported: the CTE compiles into `plan.ctes` and the
        // SET subquery `(SELECT n FROM x)` resolves `x` through the shared FROM/`lookup_cte`
        // path — proving the CTE is visible to the UPDATE's binding. The executor
        // materializes `plan.ctes` before the UPDATE (a plan-level attach; the Update node is
        // unchanged). Was previously a loud "not yet supported" gap.
        let cat = cat_t();
        let plan =
            plan_sql("WITH x AS (SELECT 1 AS n) UPDATE t SET a = (SELECT n FROM x)", &cat).unwrap();
        assert_eq!(plan.ctes.len(), 1, "the CTE `x` is registered, got {:?}", plan.ctes);
        assert!(
            matches!(&plan.ctes[0], CtePlan::Materialized { name, .. } if name == "x"),
            "ctes[0] is the materialized CTE `x`, got {:?}",
            plan.ctes[0]
        );
        let u = expect_update(&plan);
        assert_eq!(u.assignments.len(), 1);
        assert_eq!(u.assignments[0].0, 0, "SET targets column a");
        assert!(
            matches!(u.assignments[0].1, EvalExpr::ScalarSubquery(_)),
            "the SET value is the scalar subquery over CTE x, got {:?}",
            u.assignments[0].1
        );
    }

    #[test]
    fn or_conflict_algorithm_flows_into_the_plan() {
        // Every `OR <algorithm>` arm must map to its OnConflict, and a bare UPDATE
        // defaults to ABORT — pinned exhaustively so a mis-wired arm can't hide.
        // OnConflict has no PartialEq (and lives in an out-of-scope crate), so compare
        // its Debug form — a fieldless variant renders as its own name.
        let cat = cat_t();
        let cases = [
            ("UPDATE t SET a = 1", "Abort"),
            ("UPDATE OR ABORT t SET a = 1", "Abort"),
            ("UPDATE OR ROLLBACK t SET a = 1", "Rollback"),
            ("UPDATE OR FAIL t SET a = 1", "Fail"),
            ("UPDATE OR IGNORE t SET a = 1", "Ignore"),
            ("UPDATE OR REPLACE t SET a = 1", "Replace"),
        ];
        for (sql, expected) in cases {
            let plan = plan_sql(sql, &cat).unwrap();
            let got = format!("{:?}", expect_update(&plan).on_conflict);
            assert_eq!(got, expected, "on_conflict for `{sql}`");
        }
    }

    #[test]
    fn column_affinities_follow_the_declared_types() {
        // The affinity vector drives value coercion in the executor, so its CONTENT (not
        // just its length) is contract: `t(a INTEGER, b TEXT, c TEXT)` in schema order.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(
            u.column_affinities,
            vec![Affinity::Integer, Affinity::Text, Affinity::Text],
            "one affinity per column, in schema order, from the declared type"
        );
    }

    #[test]
    fn returning_alias_and_computed_expr_take_their_names() {
        // The alias arm (`AS x`) and a computed expression exercise the naming path that
        // the bare-column cases skip.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1 RETURNING a AS x, a + 1", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 2);
        assert_eq!(plan.result_columns[0], "x", "explicit alias wins");
        // The computed column's name is its reconstructed source text (SQLite derives
        // RETURNING/SELECT output names from the expression text when unaliased).
        assert_eq!(plan.result_columns.len(), 2);
        assert!(!plan.result_columns[1].is_empty(), "computed column still gets a name");
    }

    #[test]
    fn returning_table_star_through_alias_expands_all_columns() {
        // `RETURNING x.*` must resolve the alias `x` and expand to every column, in
        // schema order — this pins both the alias binding and the TableStar arm.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t AS x SET a = 1 RETURNING x.*", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 3);
        assert!(matches!(u.returning[0], EvalExpr::Column(0)));
        assert!(matches!(u.returning[1], EvalExpr::Column(1)));
        assert!(matches!(u.returning[2], EvalExpr::Column(2)));
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn aliased_target_resolves_qualified_references() {
        // With `UPDATE t AS x`, the exposed name is the alias, so `x.a` must resolve and
        // `t.a` must not — a broken exposed_name would slip through untested otherwise.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t AS x SET a = 1 WHERE x.a = 2 RETURNING x.a", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.returning.len(), 1);
        assert!(matches!(u.returning[0], EvalExpr::Column(0)), "`x.a` resolves to column a");
        // The stale table name is no longer exposed once aliased (SQLite behavior).
        let err = plan_sql("UPDATE t AS x SET a = 1 WHERE t.a = 2", &cat).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "`t.a` under alias `x` is unresolved, got {err:?}");
    }

    // ---- CHECK constraints compiled onto the node (compile::check) ----------------

    /// Parse a scalar expression (a CHECK predicate) from its text via a `SELECT <expr>`
    /// wrapper — the same trick `compile::insert::parse_default_expr` uses to re-parse a
    /// stored predicate. The DML compilers are separate contention cells, so this tiny
    /// helper is duplicated rather than shared (mirrors the `bind_returning` note above).
    fn check_ast(text: &str) -> Expr {
        use minisqlite_sql::{SelectBody, SelectCore, Statement};
        let ast = parse(&format!("SELECT {text}")).expect("check predicate parses");
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

    /// `tc(x INTEGER)` carrying the CHECK predicates in `check_texts`, so the planner's
    /// `compile_checks` binds them onto the UPDATE node. Hand-built because the test
    /// catalog is static (the catalog builder populates `checks` in production).
    fn cat_checked(check_texts: &[&str]) -> TestCatalog {
        TestCatalog {
            tables: vec![TableDef {
                name: "tc".to_string(),
                columns: vec![col("x", Some("INTEGER"))],
                root_page: 2,
                without_rowid: false,
                rowid_alias: None,
                auto_indexes: Vec::new(),
                checks: check_texts.iter().map(|t| check_ast(t)).collect(),
                foreign_keys: Vec::new(),
                autoincrement: false,
                primary_key: Vec::new(),
                strict: false,
            }],
        }
    }

    #[test]
    fn update_compiles_check_constraint_onto_the_node() {
        // tc(x INTEGER CHECK(x > 0)): the predicate binds against the row `[x, rowid]` and
        // rides on the UPDATE node; the executor evaluates it against the post-assignment
        // new row (SAME layout, no remap). `x` -> reg 0, detail is the table name.
        let cat = cat_checked(&["x > 0"]);
        let plan = plan_sql("UPDATE tc SET x = 5", &cat).unwrap();
        let u = expect_update(&plan);
        assert_eq!(u.checks.len(), 1, "the one CHECK is compiled onto the node");
        assert_eq!(u.checks[0].detail, "tc", "detail is the table name");
        match &u.checks[0].expr {
            EvalExpr::Compare { left, .. } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(0)), "x -> reg 0, got {left:?}");
            }
            other => panic!("expected a Compare node for `x > 0`, got {other:?}"),
        }
    }

    #[test]
    fn update_without_checks_carries_no_checks() {
        // A table with no CHECK carries an empty `checks` vec on its UPDATE node.
        let cat = cat_t();
        let plan = plan_sql("UPDATE t SET a = 1", &cat).unwrap();
        assert!(expect_update(&plan).checks.is_empty());
    }

    #[test]
    fn update_check_referencing_unknown_column_is_a_bind_error() {
        // A stored CHECK naming a missing column fails to BIND at plan time — a loud
        // `no such column`, matching sqlite rejecting such a CREATE TABLE. Never dropped.
        let cat = cat_checked(&["y > 0"]);
        let err = plan_sql("UPDATE tc SET x = 5", &cat).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such column: y"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }
}
