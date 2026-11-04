//! View DML redirected through `INSTEAD OF` triggers (`lang_createtrigger.html` §3).
//!
//! A view owns no b-tree, so `INSERT`/`UPDATE`/`DELETE` on it cannot touch storage
//! directly. It is updatable ONLY for an event that has a matching `INSTEAD OF <event>`
//! trigger: the DML then fires that trigger's body FOR EACH affected view row (binding
//! `NEW`/`OLD`) and performs no base write. When no such trigger exists the view stays
//! non-updatable and the caller keeps SQLite's `cannot modify X because it is a view`.
//!
//! The three entrypoints ([`compile_view_insert`] / [`compile_view_update`] /
//! [`compile_view_delete`]) are called from the DML compilers' view-reject sites. Each
//! builds a [`PlanNode::InsteadOf`] whose `frame_source` yields one `OLD ++ NEW` frame
//! per affected row and whose `programs` are the matching `INSTEAD OF` triggers.
//!
//! # The frame the planner builds (matching the executor's OLD/NEW contract)
//!
//! Trigger bodies bind against the SAME `OLD ++ NEW` register layout base-table triggers
//! use (see [`TriggerProgram`](crate::plan::TriggerProgram)): for a view of `C` columns,
//! `W = C + 1`, OLD in `[0, W)`, NEW in `[W, 2W)`, each action's own operators at `2W`.
//! A view has no rowid, so both rowid slots (register `C` and `2C+1`) are NULL. We build
//! that scope from a SYNTHETIC [`TableDef`] carrying the view's columns (so
//! [`new_old_sources`](crate::bind::new_old_sources) lays OLD/NEW at width `W` each), and
//! `frame_source` is a [`PlanNode::Project`] that emits exactly `2W` values per row:
//!
//! * INSERT — OLD half all-NULL; NEW columns from the supplied `VALUES`/`SELECT` mapped
//!   to the view's column list (an unnamed column → NULL, since a view has no DEFAULT).
//! * DELETE — OLD columns from `SELECT <view cols> FROM <view> WHERE <p>`; NEW half NULL.
//! * UPDATE — OLD columns from the view rows matching WHERE; NEW = OLD with the `SET`
//!   assignments (evaluated against OLD) overlaid.
//!
//! # Scope cut (documented, not silent)
//!
//! Only a TOP-LEVEL view DML fires here; a view DML nested inside a trigger ACTION
//! (`parent = Some`) keeps the `cannot modify` error, matching the pre-existing behavior
//! (the one-level compile bound).
//!
//! A leading `WITH` is supported: its CTEs are registered exactly as on a base-table DML
//! ([`register_with`]) and are visible to the redirected INSERT source (`VALUES`/`SELECT`
//! and any subqueries) and to the DELETE/UPDATE `WHERE`/`SET` subqueries — but NOT to the
//! fired `INSTEAD OF` bodies (a CTE lives "only for the duration of a single SQL statement",
//! `lang_with.html` §1), which is why registration happens AFTER the trigger compile below.
//!
//! `RETURNING` is supported: it binds against the affected view row's columns `[0, C)` (the
//! NEW row for INSERT/UPDATE, the OLD row for DELETE), and the executor emits one result row
//! per affected view row — the values "as seen by the top-level … statement", before any
//! fired body runs (`lang_returning.html`). The bound exprs ride on [`InsteadOf::returning`].
//!
//! `UPDATE … FROM` and `ON CONFLICT` (upsert) on a view DML are still rejected loudly rather
//! than silently ignored — and neither is merely "not coded yet":
//!
//! * UPSERT "processing happens only for uniqueness constraints" (`lang_upsert.html`), and a
//!   view has no UNIQUE / PRIMARY KEY / unique index, so there is no conflict target to
//!   resolve at all.
//! * `UPDATE … FROM` fires the trigger once per AFFECTED TARGET ROW. A base-table UPDATE…FROM
//!   dedups multiple FROM matches for one target row by the target's ROWID (updating it once,
//!   an arbitrary FROM match); a view row has no rowid to dedup by, so when the join produces
//!   several FROM rows per view row there is no rowid-keyed way to reproduce SQLite's
//!   once-per-target-row firing. Rather than fire the wrong number of times in that case (a
//!   silent correctness bug), we reject loudly until a
//!   rowid-free dedup with confirmed semantics is designed.

use minisqlite_catalog::{Catalog, ColumnDef, TableDef, ViewDef};
use minisqlite_expr::EvalExpr;
use minisqlite_sql::{Delete, Expr, Insert, InsertSource, ResultColumn, SetClause, Update};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, DbIndex, Error, Result, Value};

use crate::bind::scope::SynthCol;
use crate::bind::{bind_expr, Scope, Source};
use crate::colname::result_column_name;
use crate::compile::ctas::ctas_columns;
use crate::compile::cte::register_with;
use crate::compile::select::compile_select_with_parent;
use crate::compile::trigger::{compile_instead_of_triggers, TriggerDmlEvent};
use crate::compile::values::compile_values;
use crate::compile::view::{compile_view, parse_view_sql};
use crate::plan::{InsteadOf, InsteadOfEvent, PlanNode, TriggerProgram};
use crate::plan_ctx::PlanCtx;

/// Compile `INSERT INTO <view> …` as an INSTEAD OF firing (or the view error). See the
/// module docs. `parent` is the enclosing trigger-action scope, if any.
pub fn compile_view_insert(
    ctx: &mut PlanCtx,
    db: DbIndex,
    view: &ViewDef,
    stmt: &Insert,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    let catalog = ctx.catalog;
    let cols = view_columns(catalog, view)?;
    let synth = synth_table(view, &cols);
    let programs = select_programs(ctx, catalog, db, &synth, TriggerDmlEvent::Insert, parent)?;
    if programs.is_empty() {
        return Err(not_updatable(view));
    }
    reject_unsupported_insert(stmt, view)?;

    // A leading `WITH` registers its CTEs so the redirected source `SELECT`/`VALUES` (and
    // their subqueries) resolve them — the SAME `register_with` a base-table INSERT uses.
    // Registered AFTER the trigger compile above (a fired body must not see the firing
    // statement's CTEs) and held through source + RETURNING binding by the guard, dropped here.
    let _cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    let frame_source = build_insert_frame(ctx, &cols, stmt, view)?;
    // RETURNING binds against the NEW view row (the inserted values); the executor reads the
    // frame's NEW half. The exposed name (INSERT alias, else the view) resolves a `v.col`.
    let exposed = stmt.alias.clone().unwrap_or_else(|| view.name.clone());
    let (returning, names) = compile_view_returning(ctx, &stmt.returning, &cols, exposed)?;
    Ok((
        instead_of_node(view, InsteadOfEvent::Insert, cols.len(), frame_source, programs, returning),
        names,
    ))
}

/// Compile `DELETE FROM <view> [WHERE …]` as an INSTEAD OF firing (or the view error).
pub fn compile_view_delete(
    ctx: &mut PlanCtx,
    db: DbIndex,
    view: &ViewDef,
    stmt: &Delete,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    let catalog = ctx.catalog;
    let cols = view_columns(catalog, view)?;
    let synth = synth_table(view, &cols);
    let programs = select_programs(ctx, catalog, db, &synth, TriggerDmlEvent::Delete, parent)?;
    if programs.is_empty() {
        return Err(not_updatable(view));
    }
    // A view DELETE has no clause left to reject: `WITH` and `RETURNING` are supported below,
    // and DELETE carries no `FROM` / `ON CONFLICT`.

    // A leading `WITH` registers its CTEs (visible to the WHERE and its subqueries), after
    // the trigger compile above and held through the WHERE bind below (see the module doc).
    let _cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    // OLD = each view row matching WHERE; NEW half all-NULL.
    let exposed = stmt.alias.clone().unwrap_or_else(|| view.name.clone());
    let (scan, c) = view_scan(ctx, view, &cols, exposed.clone(), stmt.where_clause.as_ref())?;
    let old_exprs: Vec<EvalExpr> = (0..c).map(EvalExpr::Column).collect();
    let frame_source = frame_project(scan, c, Some(old_exprs), None);
    // RETURNING binds against the OLD (deleted) view row; the executor reads the OLD half.
    let (returning, names) = compile_view_returning(ctx, &stmt.returning, &cols, exposed)?;
    Ok((
        instead_of_node(view, InsteadOfEvent::Delete, c, frame_source, programs, returning),
        names,
    ))
}

/// Compile `UPDATE <view> SET … [WHERE …]` as an INSTEAD OF firing (or the view error).
pub fn compile_view_update(
    ctx: &mut PlanCtx,
    db: DbIndex,
    view: &ViewDef,
    stmt: &Update,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    let catalog = ctx.catalog;
    let cols = view_columns(catalog, view)?;
    let synth = synth_table(view, &cols);

    // `UPDATE OF <cols>` filtering keys off the columns THIS statement assigns, resolved to
    // view-column indices — the same rule a base-table UPDATE uses (`event_matches`).
    let changed = assigned_columns(&cols, stmt);
    let event = TriggerDmlEvent::Update { changed_cols: &changed };
    let programs = select_programs(ctx, catalog, db, &synth, event, parent)?;
    if programs.is_empty() {
        return Err(not_updatable(view));
    }
    reject_unsupported_update(stmt, view)?;

    // A leading `WITH` registers its CTEs (visible to the SET/WHERE and their subqueries),
    // after the trigger compile above and held through the SET/WHERE bind below.
    let _cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    // OLD = each view row matching WHERE; NEW = OLD with the SET assignments (evaluated
    // against the OLD row) overlaid. Both bind against the view-scan scope [0, C).
    let exposed = stmt.alias.clone().unwrap_or_else(|| view.name.clone());
    let scope_cols = scan_synth_cols(&cols);
    let sources = [Source::derived(exposed.clone(), scope_cols, 0)];
    let scope = Scope::new(&sources);

    // Bind each SET value against the OLD-row scope, mapping to its view-column index.
    let mut new_exprs: Vec<EvalExpr> = (0..cols.len()).map(EvalExpr::Column).collect();
    for set in &stmt.set {
        let (name, value) = match set {
            SetClause::Column { name, value } => (name, value),
            SetClause::Columns { .. } => {
                return Err(Error::sql(format!(
                    "column-list SET on view {} is not yet supported",
                    view.name
                )));
            }
        };
        let idx = view_col_index(&cols, name, view)?;
        new_exprs[idx] = bind_expr(&scope, ctx, value)?;
    }

    let bound_where = stmt.where_clause.as_ref().map(|w| bind_expr(&scope, ctx, w)).transpose()?;
    let scan = compile_view_scan(ctx, view, &sources[0], bound_where)?;

    let old_exprs: Vec<EvalExpr> = (0..cols.len()).map(EvalExpr::Column).collect();
    let frame_source = frame_project(scan, cols.len(), Some(old_exprs), Some(new_exprs));
    // RETURNING binds against the NEW (post-update) view row; the executor reads the NEW half.
    let (returning, names) = compile_view_returning(ctx, &stmt.returning, &cols, exposed)?;
    Ok((
        instead_of_node(view, InsteadOfEvent::Update, cols.len(), frame_source, programs, returning),
        names,
    ))
}

/// Assemble the [`PlanNode::InsteadOf`] from its parts.
fn instead_of_node(
    view: &ViewDef,
    event: InsteadOfEvent,
    column_count: usize,
    frame_source: PlanNode,
    programs: Vec<TriggerProgram>,
    returning: Vec<EvalExpr>,
) -> PlanNode {
    PlanNode::InsteadOf(InsteadOf {
        view: view.name.clone(),
        event,
        column_count,
        frame_source: Box::new(frame_source),
        programs,
        returning,
    })
}

/// The matching `INSTEAD OF` programs — but ONLY for a TOP-LEVEL view DML. A view DML
/// nested in a trigger action (`parent = Some`) returns none, so the caller keeps the
/// `cannot modify` error (the documented one-level cut; see the module docs).
fn select_programs(
    ctx: &mut PlanCtx,
    catalog: &dyn Catalog,
    db: DbIndex,
    synth: &TableDef,
    event: TriggerDmlEvent,
    parent: Option<&Scope>,
) -> Result<Vec<TriggerProgram>> {
    if parent.is_some() {
        return Ok(Vec::new());
    }
    compile_instead_of_triggers(ctx, catalog, db, synth, event)
}

/// The `cannot modify X because it is a view` error (SQLite's wording, view's stored
/// creation-case name) — kept identical to the DML compilers' original reject sites.
fn not_updatable(view: &ViewDef) -> Error {
    Error::sql(format!("cannot modify {} because it is a view", view.name))
}

// ---- view column schema -----------------------------------------------------------

/// One view column for the INSTEAD OF machinery: its output name (an explicit
/// `CREATE VIEW v(…)` list applied) and the declared-type text [`ctas_columns`] derives
/// (`None` = BLOB/no-affinity), from which its affinity is recovered.
///
/// Collation is NOT tracked (it defaults to BINARY in the scan/OLD/NEW scopes). A view
/// column carrying an explicit `COLLATE` that participates in a trigger comparison would
/// need it — a documented limitation; no in-scope case exercises it.
struct ViewCol {
    name: String,
    decl_type: Option<String>,
}

impl ViewCol {
    fn affinity(&self) -> Affinity {
        affinity_of_declared_type(self.decl_type.as_deref())
    }
}

/// Derive the view's columns (names + declared types) from its stored `SELECT` via
/// [`ctas_columns`] (the SAME §2.1 name+affinity derivation `CREATE TABLE AS SELECT`
/// uses), applying an explicit `CREATE VIEW v(c1,…)` name list when present. A
/// self/mutually-referential view surfaces the recursion guard's `circularly defined`
/// error through `ctas_columns` → `compile_select`.
fn view_columns(catalog: &dyn Catalog, view: &ViewDef) -> Result<Vec<ViewCol>> {
    let (select, explicit) = parse_view_sql(&view.sql, &view.name)?;
    let ctas = ctas_columns(&select, catalog)?;
    match explicit {
        None => {
            Ok(ctas.into_iter().map(|c| ViewCol { name: c.name, decl_type: c.decl_type }).collect())
        }
        Some(names) => {
            // Same arity rule and message as the FROM-side view expansion
            // (`compile::view::apply_view_columns`).
            if names.len() != ctas.len() {
                return Err(Error::sql(format!(
                    "expected {} columns for '{}' but got {}",
                    names.len(),
                    view.name,
                    ctas.len()
                )));
            }
            Ok(names
                .into_iter()
                .zip(ctas)
                .map(|(name, c)| ViewCol { name, decl_type: c.decl_type })
                .collect())
        }
    }
}

/// The SYNTHETIC [`TableDef`] standing in for the view when building the OLD/NEW scope.
/// It carries the view's columns so [`new_old_sources`](crate::bind::new_old_sources)
/// lays OLD at `[0, W)` and NEW at `[W, 2W)` (`W = C + 1`) exactly as a base table — the
/// register at index `C` / `2C+1` being a NULL-placeholder rowid a view has no value for.
/// `without_rowid = false` and `rowid_alias = None` make that placeholder exist without a
/// column aliasing it; `root_page = 0` because a view owns no b-tree (nothing reads it).
fn synth_table(view: &ViewDef, cols: &[ViewCol]) -> TableDef {
    TableDef {
        name: view.name.clone(),
        columns: cols
            .iter()
            .map(|c| ColumnDef {
                name: c.name.clone(),
                declared_type: c.decl_type.clone(),
                not_null: false,
                primary_key: false,
                unique: false,
                collation: None,
                default: None,
                default_value: None,
                generated: None,
            })
            .collect(),
        root_page: 0,
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

/// The [`SynthCol`]s the view's rows scan into for DELETE/UPDATE WHERE/SET binding: each
/// view column's name + affinity (collation defaults to BINARY; see [`ViewCol`]). The
/// caller wraps these in a [`Source::Derived`] under the view's exposed name at base 0.
fn scan_synth_cols(cols: &[ViewCol]) -> Vec<SynthCol> {
    cols.iter()
        .map(|c| SynthCol {
            name: c.name.clone(),
            affinity: c.affinity(),
            collation: Collation::Binary,
            hidden: false,
        })
        .collect()
}

// ---- frame assembly ---------------------------------------------------------------

/// Assemble the `OLD ++ NEW` frame [`PlanNode::Project`] over `input`: `2W = 2*(C+1)`
/// expressions — OLD's `C` columns then its NULL rowid slot, then NEW's `C` columns then
/// its NULL rowid slot. A `None` half is all-NULL (INSERT has no OLD, DELETE has no NEW).
/// Each supplied `Vec<EvalExpr>` MUST be length `C` and read `input`'s columns `[0, C)`.
fn frame_project(
    input: PlanNode,
    c: usize,
    old: Option<Vec<EvalExpr>>,
    new: Option<Vec<EvalExpr>>,
) -> PlanNode {
    let mut exprs = Vec::with_capacity(2 * (c + 1));
    push_half_exprs(&mut exprs, old, c);
    push_half_exprs(&mut exprs, new, c);
    PlanNode::Project { input: Box::new(input), exprs }
}

/// Push one frame half's `C` column expressions (or `C` NULLs when absent), then the
/// half's NULL rowid-placeholder slot — so each half is exactly `W = C + 1` wide.
fn push_half_exprs(exprs: &mut Vec<EvalExpr>, half: Option<Vec<EvalExpr>>, c: usize) {
    match half {
        Some(vals) => {
            debug_assert_eq!(vals.len(), c, "a frame half must be width C");
            exprs.extend(vals);
        }
        None => exprs.extend(std::iter::repeat_with(|| EvalExpr::Literal(Value::Null)).take(c)),
    }
    exprs.push(EvalExpr::Literal(Value::Null));
}

// ---- INSERT direction -------------------------------------------------------------

/// Build the INSERT `frame_source`: OLD all-NULL, NEW from the supplied row source mapped
/// onto the view's full column list (a view column no supplied value names → NULL).
fn build_insert_frame(
    ctx: &mut PlanCtx,
    cols: &[ViewCol],
    stmt: &Insert,
    view: &ViewDef,
) -> Result<PlanNode> {
    let c = cols.len();
    let (source, source_width, col_map) = compile_insert_source(ctx, cols, stmt, view)?;

    // NEW column i ← the source position mapped to it, else NULL.
    let new_exprs: Vec<EvalExpr> = (0..c)
        .map(|i| match col_map.iter().position(|&target| target == i) {
            Some(p) => {
                debug_assert!(p < source_width, "mapped source position within the row");
                EvalExpr::Column(p)
            }
            None => EvalExpr::Literal(Value::Null),
        })
        .collect();

    Ok(frame_project(source, c, None, Some(new_exprs)))
}

/// Compile the INSERT's row source (`VALUES` / `SELECT` / `DEFAULT VALUES`) and resolve
/// its target columns to view-column indices. Returns `(source_node, source_width,
/// col_map)` where `col_map[p]` is the view-column index supplied by source position `p`.
fn compile_insert_source(
    ctx: &mut PlanCtx,
    cols: &[ViewCol],
    stmt: &Insert,
    view: &ViewDef,
) -> Result<(PlanNode, usize, Vec<usize>)> {
    let c = cols.len();
    let col_map = resolve_insert_columns(cols, stmt, view)?;
    match &stmt.source {
        InsertSource::DefaultValues => {
            // A view column has no DEFAULT: every NEW column is NULL. One zero-wide row
            // (no source position maps to any column, so the frame's NEW half is all-NULL).
            Ok((PlanNode::SingleRow, 0, Vec::new()))
        }
        InsertSource::Values(rows) => {
            let width = row_arity(rows)?;
            check_insert_arity(&col_map, c, width, view)?;
            // Top-level: no OLD/NEW parent for the supplied values.
            let (node, _names) = compile_values(ctx, rows, None)?;
            Ok((node, width, col_map))
        }
        InsertSource::Select(sel) => {
            let (node, names) = compile_select_with_parent(ctx, sel, None)?;
            let width = names.len();
            check_insert_arity(&col_map, c, width, view)?;
            Ok((node, width, col_map))
        }
    }
}

/// Resolve the INSERT's target columns to view-column indices: the positional identity
/// `[0, C)` when there is no column list, else each named column's index.
fn resolve_insert_columns(cols: &[ViewCol], stmt: &Insert, view: &ViewDef) -> Result<Vec<usize>> {
    match &stmt.columns {
        None => Ok((0..cols.len()).collect()),
        Some(names) => names.iter().map(|n| view_col_index(cols, n, view)).collect(),
    }
}

/// The view-column index for `name` (case-insensitive), or SQLite's `table X has no
/// column named Y` (a view uses the same message).
fn view_col_index(cols: &[ViewCol], name: &str, view: &ViewDef) -> Result<usize> {
    cols.iter()
        .position(|c| c.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| Error::sql(format!("table {} has no column named {}", view.name, name)))
}

/// Every `VALUES` row's arity (from the first row; all rows must match).
fn row_arity(rows: &[Vec<Expr>]) -> Result<usize> {
    let width = rows.first().map_or(0, Vec::len);
    for row in rows {
        if row.len() != width {
            return Err(Error::sql("all VALUES must have the same number of terms"));
        }
    }
    Ok(width)
}

/// The INSERT arity rule, with SQLite's two distinct messages (matching
/// `compile::insert::check_arity`): a positional insert supplies one value per view
/// column; an explicit list one value per listed column.
fn check_insert_arity(col_map: &[usize], c: usize, source_width: usize, view: &ViewDef) -> Result<()> {
    if source_width == col_map.len() {
        return Ok(());
    }
    if col_map.len() == c && matches_positional(col_map) {
        Err(Error::sql(format!(
            "table {} has {c} columns but {source_width} values were supplied",
            view.name
        )))
    } else {
        Err(Error::sql(format!("{source_width} values for {} columns", col_map.len())))
    }
}

/// Whether `col_map` is the positional identity `[0, 1, …]` (no explicit column list),
/// which selects the positional arity message.
fn matches_positional(col_map: &[usize]) -> bool {
    col_map.iter().enumerate().all(|(i, &target)| i == target)
}

// ---- DELETE / UPDATE view scan ----------------------------------------------------

/// Build a streaming scan of the view's rows filtered by `where_clause`, plus the view's
/// column count `C`. The rows land at registers `[0, C)`; WHERE binds against them.
fn view_scan(
    ctx: &mut PlanCtx,
    view: &ViewDef,
    cols: &[ViewCol],
    exposed: String,
    where_clause: Option<&Expr>,
) -> Result<(PlanNode, usize)> {
    let scope_cols = scan_synth_cols(cols);
    let sources = [Source::derived(exposed, scope_cols, 0)];
    let scope = Scope::new(&sources);
    let bound_where = where_clause.map(|w| bind_expr(&scope, ctx, w)).transpose()?;
    let scan = compile_view_scan(ctx, view, &sources[0], bound_where)?;
    Ok((scan, cols.len()))
}

/// Compile the view's body into a `CteScan` (reusing the FROM-side [`compile_view`], so a
/// view is a materialized CTE by the time the executor runs) and wrap it in a `Filter`
/// when a WHERE predicate is present. `src` supplies the view's schema width and the CTE
/// name; `bound_where` is already bound against `src`'s columns `[0, C)`.
fn compile_view_scan(
    ctx: &mut PlanCtx,
    view: &ViewDef,
    src: &Source,
    bound_where: Option<EvalExpr>,
) -> Result<PlanNode> {
    let scan = compile_view(ctx, view, src)?;
    Ok(match bound_where {
        Some(predicate) => PlanNode::Filter { input: Box::new(scan), predicate },
        None => scan,
    })
}

/// The view-column indices this UPDATE's `SET` assigns, for `UPDATE OF <cols>` matching
/// (SQLite's OF-aware `tmask` — a view is updatable for UPDATE only when a matching
/// `INSTEAD OF UPDATE` trigger exists). An assignment naming an unknown column is SKIPPED
/// here (it can match no trigger's OF list, which names only real view columns), so the
/// updatability check reports `cannot modify` FIRST — as SQLite does — rather than this
/// pre-empting it with a column error; a genuine unknown column then surfaces later, only
/// once the view is confirmed updatable (during the SET/WHERE bind in
/// [`compile_view_update`]), matching SQLite's ordering.
fn assigned_columns(cols: &[ViewCol], stmt: &Update) -> Vec<usize> {
    let mut out = Vec::with_capacity(stmt.set.len());
    for set in &stmt.set {
        let names: &[String] = match set {
            SetClause::Column { name, .. } => std::slice::from_ref(name),
            SetClause::Columns { names, .. } => names,
        };
        for name in names {
            if let Some(idx) = cols.iter().position(|c| c.name.eq_ignore_ascii_case(name)) {
                out.push(idx);
            }
        }
    }
    out
}

// ---- unsupported-clause gaps (loud, not silent) -----------------------------------

fn reject_unsupported_insert(stmt: &Insert, view: &ViewDef) -> Result<()> {
    // Spec-correct reject (real sqlite also errors): UPSERT "processing happens only for
    // uniqueness constraints" (lang_upsert.html) and a view has no UNIQUE / PRIMARY KEY / unique
    // index, so there is no conflict target to resolve — sqlite rejects `INSERT INTO <view> …
    // ON CONFLICT …` for the same reason.
    if !stmt.upsert.is_empty() {
        return Err(unsupported("ON CONFLICT (upsert)", view));
    }
    Ok(())
}

fn reject_unsupported_update(stmt: &Update, view: &ViewDef) -> Result<()> {
    // CONSERVATIVE reject (real sqlite SUCCEEDS here — a known gap vs real sqlite, NOT a spec
    // error): UPDATE…FROM fires the INSTEAD OF trigger once per affected target row, and
    // lang_update.html §2.2 dedups multiple FROM matches for one target row by the target's
    // ROWID ("only one of those output rows is used … arbitrary"). A view row has no rowid, so
    // the once-per-view-row firing can't be reproduced when the join yields several FROM rows per
    // view row; rejected until a rowid-free dedup with confirmed semantics is designed, rather
    // than fire the wrong number of times (a silent correctness bug).
    if stmt.from.is_some() {
        return Err(unsupported("FROM", view));
    }
    Ok(())
}

/// Bind a view DML's `RETURNING` clause against the affected view row's columns `[0, C)`
/// (a bare view scope at base 0, same as the DELETE/UPDATE scan scope). The executor picks
/// the frame half these `EvalExpr::Column(i)` read — NEW for INSERT/UPDATE, OLD for DELETE.
/// Returns the bound output expressions and their result-column names; an empty clause
/// yields two empty vecs. `exposed` is the name a qualified `v.col` / `alias.col` resolves
/// against (the statement's alias, else the view name). Mirrors the base-table
/// `compile::insert::compile_returning`, minus the rowid register a view has no value for.
fn compile_view_returning(
    ctx: &mut PlanCtx,
    returning: &[ResultColumn],
    cols: &[ViewCol],
    exposed: String,
) -> Result<(Vec<EvalExpr>, Vec<String>)> {
    if returning.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let scope_cols = scan_synth_cols(cols);
    let sources = [Source::derived(exposed, scope_cols, 0)];
    let scope = Scope::new(&sources);

    let mut exprs = Vec::with_capacity(returning.len());
    let mut names = Vec::with_capacity(returning.len());
    for rc in returning {
        match rc {
            ResultColumn::Expr { expr, alias } => {
                exprs.push(bind_expr(&scope, ctx, expr)?);
                names.push(match alias {
                    Some(a) => a.clone(),
                    None => result_column_name(expr),
                });
            }
            ResultColumn::Star => {
                for (reg, name) in scope.expand_star(None)? {
                    exprs.push(EvalExpr::Column(reg));
                    names.push(name);
                }
            }
            ResultColumn::TableStar(t) => {
                for (reg, name) in scope.expand_star(Some(t))? {
                    exprs.push(EvalExpr::Column(reg));
                    names.push(name);
                }
            }
        }
    }
    Ok((exprs, names))
}

/// A loud, honest gap for a view-DML clause not yet redirected through INSTEAD OF (see
/// the module docs). Flagged rather than silently ignored so a caller never gets a
/// half-applied statement.
fn unsupported(clause: &str, view: &ViewDef) -> Error {
    Error::sql(format!("{clause} on a view ({}) INSTEAD OF trigger is not yet supported", view.name))
}

#[cfg(test)]
mod tests {
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef, TriggerDef, ViewDef};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};
    use minisqlite_types::Result;

    use crate::plan::{CtePlan, InsteadOfEvent, Plan, PlanNode};
    use crate::{Planner, QueryPlanner};

    /// A static read-only catalog carrying a base table, a view over it, and the view's
    /// triggers — enough to drive planning through the view-DML reject sites. Mirrors the
    /// sibling `compile::view` test catalog, plus the `triggers_on` seam.
    struct TestCatalog {
        tables: Vec<TableDef>,
        views: Vec<ViewDef>,
        triggers: Vec<TriggerDef>,
    }

    impl Catalog for TestCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, _name: &str) -> Result<Option<&IndexDef>> {
            Ok(None)
        }
        fn view(&self, name: &str) -> Result<Option<&ViewDef>> {
            Ok(self.views.iter().find(|v| v.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, _table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(Vec::new())
        }
        fn triggers_on<'a>(&'a self, table: &str) -> Result<Vec<&'a TriggerDef>> {
            Ok(self.triggers.iter().filter(|t| t.table.eq_ignore_ascii_case(table)).collect())
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

    fn base_table() -> TableDef {
        TableDef {
            name: "base".to_string(),
            columns: vec![col("id", "INTEGER"), col("val", "TEXT")],
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

    fn view_v() -> ViewDef {
        ViewDef { name: "v".to_string(), sql: "CREATE VIEW v AS SELECT id, val FROM base".to_string() }
    }

    /// An `INSTEAD OF <event>` trigger `ON v` whose full text is `sql`.
    fn trig(sql: &str) -> TriggerDef {
        TriggerDef::same_store("tr".to_string(), "v".to_string(), sql.to_string())
    }

    fn insert_trigger() -> TriggerDef {
        trig("CREATE TRIGGER tr INSTEAD OF INSERT ON v \
              BEGIN INSERT INTO base(id, val) VALUES(NEW.id, NEW.val); END")
    }

    fn plan_with(triggers: Vec<TriggerDef>, sql: &str) -> Result<Plan> {
        let cat = TestCatalog { tables: vec![base_table()], views: vec![view_v()], triggers };
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &cat)
    }

    /// A view INSERT with a matching `INSTEAD OF INSERT` trigger compiles to an `InsteadOf`
    /// node (not the "cannot modify" error), and the whole statement stays `mutates`.
    #[test]
    fn view_insert_with_matching_trigger_is_an_instead_of_node() {
        let plan = plan_with(vec![insert_trigger()], "INSERT INTO v(id, val) VALUES(1, 'x')")
            .expect("a view with an INSTEAD OF INSERT trigger is updatable");
        assert!(plan.mutates, "a view INSERT is still a mutating statement");
        match &plan.root {
            PlanNode::InsteadOf(io) => {
                assert_eq!(io.view, "v");
                assert_eq!(io.event, InsteadOfEvent::Insert);
                assert_eq!(io.column_count, 2, "view v exposes 2 columns");
                assert_eq!(io.programs.len(), 1, "one matching INSTEAD OF INSERT program");
            }
            other => panic!("expected an InsteadOf root, got {other:?}"),
        }
    }

    /// A view is updatable ONLY for an event that has a matching INSTEAD OF trigger; every
    /// other view DML keeps SQLite's `cannot modify X because it is a view`.
    #[test]
    fn view_dml_without_matching_trigger_is_not_updatable() {
        // No triggers at all.
        let e = plan_with(vec![], "INSERT INTO v(id, val) VALUES(1, 'x')")
            .expect_err("no INSTEAD OF INSERT trigger → not updatable");
        assert!(e.to_string().contains("cannot modify v because it is a view"), "got {e}");

        // An INSTEAD OF INSERT trigger does NOT make DELETE (or UPDATE) updatable.
        let e = plan_with(vec![insert_trigger()], "DELETE FROM v WHERE id = 1")
            .expect_err("only an INSERT trigger exists");
        assert!(e.to_string().contains("cannot modify v because it is a view"), "got {e}");
    }

    /// A view DELETE with a matching `INSTEAD OF DELETE` trigger compiles to an `InsteadOf`.
    #[test]
    fn view_delete_with_matching_trigger_is_an_instead_of_node() {
        let t = trig("CREATE TRIGGER tr INSTEAD OF DELETE ON v \
                      BEGIN DELETE FROM base WHERE id = OLD.id; END");
        let plan = plan_with(vec![t], "DELETE FROM v WHERE id = 1").expect("updatable for DELETE");
        match &plan.root {
            PlanNode::InsteadOf(io) => assert_eq!(io.event, InsteadOfEvent::Delete),
            other => panic!("expected an InsteadOf root, got {other:?}"),
        }
    }

    /// `UPDATE OF <col>` gates view updatability exactly as SQLite's OF-aware `tmask` does:
    /// `SET`-ting a covered column fires the trigger (`InsteadOf` node); `SET`-ting only an
    /// uncovered column matches no trigger, so the view is not updatable for that statement.
    #[test]
    fn update_of_column_gates_view_updatability() {
        let t = trig("CREATE TRIGGER tr INSTEAD OF UPDATE OF val ON v \
                      BEGIN UPDATE base SET val = NEW.val WHERE id = NEW.id; END");

        let plan = plan_with(vec![t.clone()], "UPDATE v SET val = 'y' WHERE id = 1")
            .expect("SET val is covered by UPDATE OF val");
        match &plan.root {
            PlanNode::InsteadOf(io) => assert_eq!(io.event, InsteadOfEvent::Update),
            other => panic!("expected an InsteadOf root, got {other:?}"),
        }

        let e = plan_with(vec![t], "UPDATE v SET id = 9 WHERE id = 1")
            .expect_err("SET id is not covered by UPDATE OF val");
        assert!(e.to_string().contains("cannot modify v because it is a view"), "got {e}");
    }

    /// A leading `WITH` on a view INSERT is supported (was a loud "not yet supported"): its
    /// CTE is registered on the plan and the redirected source `SELECT … FROM c` resolves
    /// `c`, compiling to an `InsteadOf` whose firing is unchanged.
    #[test]
    fn with_on_view_insert_registers_cte_and_compiles() {
        let plan = plan_with(
            vec![insert_trigger()],
            "WITH c(id, val) AS (VALUES (1, 'x')) INSERT INTO v(id, val) SELECT id, val FROM c",
        )
        .expect("WITH on a view INSERT with a matching INSTEAD OF INSERT trigger compiles");
        // INSERT builds NEW from the source directly (no view scan), so the ONLY CTE on the
        // plan is the firing statement's own `c`.
        assert_eq!(plan.ctes.len(), 1, "the CTE c is registered on the plan, got {:?}", plan.ctes);
        assert!(named_cte(&plan, "c"), "the WITH CTE `c` is on the plan");
        match &plan.root {
            PlanNode::InsteadOf(io) => {
                assert_eq!(io.event, InsteadOfEvent::Insert);
                assert_eq!(io.programs.len(), 1, "one matching INSTEAD OF INSERT program");
            }
            other => panic!("expected an InsteadOf root, got {other:?}"),
        }
    }

    /// A leading `WITH` on a view DELETE compiles: the CTE is registered and the WHERE
    /// references it through a subquery.
    #[test]
    fn with_on_view_delete_registers_cte_and_compiles() {
        let t = trig("CREATE TRIGGER tr INSTEAD OF DELETE ON v \
                      BEGIN DELETE FROM base WHERE id = OLD.id; END");
        let plan = plan_with(
            vec![t],
            "WITH ids(x) AS (VALUES (1)) DELETE FROM v WHERE id IN (SELECT x FROM ids)",
        )
        .expect("WITH on a view DELETE compiles");
        // Two CTEs: the view materialized for the OLD-row scan, plus the firing `ids`.
        assert!(named_cte(&plan, "ids"), "the WITH CTE `ids` is on the plan, got {:?}", plan.ctes);
        assert!(matches!(&plan.root, PlanNode::InsteadOf(io) if io.event == InsteadOfEvent::Delete));
    }

    /// A leading `WITH` on a view UPDATE compiles: the CTE is registered and a `SET` value
    /// references it through a scalar subquery.
    #[test]
    fn with_on_view_update_registers_cte_and_compiles() {
        let t = trig("CREATE TRIGGER tr INSTEAD OF UPDATE ON v \
                      BEGIN UPDATE base SET val = NEW.val WHERE id = NEW.id; END");
        let plan = plan_with(
            vec![t],
            "WITH names(n) AS (VALUES ('y')) UPDATE v SET val = (SELECT n FROM names) WHERE id = 1",
        )
        .expect("WITH on a view UPDATE compiles");
        assert!(named_cte(&plan, "names"), "the WITH CTE `names` is on the plan, got {:?}", plan.ctes);
        assert!(matches!(&plan.root, PlanNode::InsteadOf(io) if io.event == InsteadOfEvent::Update));
    }

    /// Whether a CTE named `name` (ASCII case-insensitive) is registered on the plan.
    fn named_cte(plan: &Plan, name: &str) -> bool {
        plan.ctes.iter().any(|c| match c {
            CtePlan::Materialized { name: n, .. } | CtePlan::Recursive { name: n, .. } => {
                n.eq_ignore_ascii_case(name)
            }
        })
    }
}
