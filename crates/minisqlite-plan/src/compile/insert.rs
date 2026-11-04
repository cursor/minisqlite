//! `INSERT` compilation: lower a parsed [`minisqlite_sql::Insert`] into a
//! [`PlanNode::Insert`](crate::plan::PlanNode::Insert) over its source rows.
//!
//! Delegation seam: [`compile_insert`] is the single entrypoint the planner routes
//! every `INSERT` through, so the feature lands here rather than in the shared
//! dispatcher.
//!
//! The emitted node speaks the executor's contract (see
//! `crates/minisqlite-exec/src/ops/insert.rs`): `column_count` is the table's column
//! count `N`; `column_affinities` has length `N` in `CREATE TABLE` order; `columns`
//! maps each source value to a target column (or `None` for a positional all-columns
//! insert) — with ONE exception: the SENTINEL `N` at the position naming
//! `rowid`/`_rowid_`/`oid` on a plain rowid table, which is NOT a real column index but
//! the explicit-rowid channel recorded in the node's `rowid_source` (see
//! [`rowid_target_index`]); and `returning` binds against the inserted row layout
//! `[c0..c_{N-1}, rowid]` (rowid at register `N`), the same shape the executor
//! materializes before evaluating `RETURNING`.
//!
//! A column the target list OMITS takes its declared `DEFAULT` here (or NULL if it has
//! none): the default expression is bound and injected into the source, so the executor
//! never has to know about defaults — it always receives a value for every column the
//! `columns` list names. The rowid alias is the one exception (see [`default_injections`]).

use minisqlite_catalog::{Catalog, IndexDef, TableDef};
use minisqlite_expr::EvalExpr;
use minisqlite_sql::{
    parse, ConflictClause, Expr, IndexedColumn, IndexedColumnTarget, InsertSource, ResultColumn,
    SelectBody, SelectCore, SetClause, Statement, Upsert, UpsertAction, UpsertTarget,
};
use minisqlite_types::{Affinity, DbIndex, Error, Result};

use crate::bind::scope::is_rowid_keyword;
use crate::bind::{bind_expr, Scope, Source};
use crate::colname::result_column_name;
use crate::compile::check::compile_checks;
use crate::compile::cte::register_with;
use crate::compile::index_expr::{
    compile_table_index_key_exprs, compile_table_index_partial_predicates,
};
use crate::compile::select::{compile_columnlist_subquery, compile_select_with_parent};
use crate::compile::trigger::{compile_triggers, TriggerDmlEvent};
use crate::compile::values::compile_values;
use crate::plan::{
    ConflictTarget, Insert, OnConflict, PlanNode, UpsertActionPlan, UpsertClause, UpsertPlan,
};
use crate::plan_ctx::PlanCtx;

/// Compile a top-level `INSERT` into its plan tree and output column names (the latter
/// empty unless a `RETURNING` clause is present). A top-level INSERT has no enclosing
/// scope; see [`compile_insert_with_parent`] for the trigger-action form.
pub fn compile_insert(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Insert,
) -> Result<(PlanNode, Vec<String>)> {
    compile_insert_with_parent(ctx, stmt, None)
}

/// Compile an `INSERT`, optionally as a trigger ACTION under an enclosing OLD/NEW
/// `parent` scope. With `parent = Some(..)` the source's `NEW`/`OLD` references resolve
/// through it (a `VALUES (NEW.x)` or an `INSERT … SELECT` correlated to OLD/NEW), and
/// NO triggers of the inserted-into table are attached — only a TOP-LEVEL INSERT
/// (`parent = None`) attaches its direct triggers, so trigger expansion stays one level
/// deep (the executor drives runtime recursion). `None` is exactly the top-level path.
pub fn compile_insert_with_parent(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Insert,
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    // Copy the catalog reference out so `td` borrows through it (lifetime tied to the
    // catalog, not to `ctx`), leaving `ctx` free for the mutable binder/source calls
    // below — the same pattern `compile_select` uses.
    let catalog = ctx.catalog;

    // Resolve the target table first, so a missing table reports "no such table" (as
    // SQLite does) regardless of any unsupported clause below. A VIEW owns no b-tree; it
    // is updatable ONLY through a matching `INSTEAD OF INSERT` trigger — when one exists,
    // redirect the INSERT through it (`compile::instead_of`, which fires the trigger body
    // FOR EACH row and makes no base write). Without such a trigger it keeps SQLite's exact
    // `cannot modify … because it is a view` (using the view's stored creation-case name).
    // Resolve the target NAMESPACE (a schema qualifier maps to its fixed DbIndex; an
    // unqualified name resolves in search order — temp shadows main), then fetch the table
    // FROM that namespace so the def and the pager the write targets agree. An unknown
    // qualifier / a name in no live namespace is "no such table".
    let Some(db) = crate::compile::from::resolve_ref_db(catalog, &stmt.table)? else {
        return Err(Error::sql(format!(
            "no such table: {}",
            crate::compile::from::qualified_table_name(&stmt.table)
        )));
    };
    let td: &TableDef = match catalog.table_in(db, &stmt.table.name)? {
        Some(td) => td,
        None => {
            return match catalog.view_in(db, &stmt.table.name)? {
                Some(view) => {
                    crate::compile::instead_of::compile_view_insert(ctx, db, view, stmt, parent)
                }
                None => Err(Error::sql(format!(
                    "no such table: {}",
                    crate::compile::from::qualified_table_name(&stmt.table)
                ))),
            };
        }
    };
    let n = td.columns.len();
    let table = td.name.clone();

    // GENERATED columns (`gencol.html`) are never user-supplied: they are excluded from an
    // implicit positional INSERT and from default injection, and naming one in an explicit
    // target list is rejected (in `resolve_target_columns`). `arity_n` is the count of
    // INSERTABLE (non-generated) columns — what a positional INSERT must supply a value for
    // — and equals `n` for a table with no generated column (the allocation-free fast path).
    let has_generated = td.columns.iter().any(|c| c.generated.is_some());
    let arity_n = if has_generated {
        td.columns.iter().filter(|c| c.generated.is_none()).count()
    } else {
        n
    };

    // A leading `WITH` registers its CTEs (mirroring the SELECT path) so the source SELECT
    // and any subqueries resolve the CTE names via the shared FROM/`lookup_cte` path. The
    // compiled CtePlans ride in `ctx.ctes` -> `Plan::ctes`, which the executor materializes
    // before the INSERT (the SAME plan-level attach SELECT uses — the Insert node is
    // unchanged). The guard keeps the CTEs visible through source + RETURNING binding below,
    // then is `drop`ped explicitly BEFORE `compile_triggers` (see there) so they are invisible
    // inside trigger bodies (an early `?` before that drop still unwinds the scope via RAII).
    // Registered here (after target resolution) so a missing/view target still reports its own
    // error first.
    let cte_guard = stmt.with.as_ref().map(|with| register_with(ctx, with)).transpose()?;

    // Per-column affinities, in `CREATE TABLE` order (length `N`).
    let column_affinities: Vec<Affinity> =
        (0..td.columns.len()).map(|i| td.column_affinity(i)).collect();

    // Target columns, the row source, and its width — with the arity check that ties
    // them together (skipped for DEFAULT VALUES, which supplies no values). Any column
    // the target list omits has its `DEFAULT` bound and injected into the source AFTER
    // the arity check, so a default never counts against the user-supplied arity.
    let (source, source_width, columns) = match &stmt.source {
        InsertSource::DefaultValues => {
            // Every column is omitted, so each takes its DEFAULT (or NULL if none, or an
            // auto rowid for the alias). Emit one row of the bound defaults and list
            // exactly the columns that got one; a column left out of `columns` is filled
            // by the executor (NULL for a plain column, an auto-assigned rowid for the
            // alias).
            let (idxs, exprs) = default_injections(ctx, td, &[])?;
            let width = exprs.len();
            (PlanNode::Values { rows: vec![exprs] }, width, Some(idxs))
        }
        InsertSource::Values(rows) => {
            let mut columns = resolve_target_columns(&stmt.columns, td, &table)?;
            let mut width = values_width(rows)?;
            check_arity(&columns, arity_n, width, &table)?;
            // A positional INSERT on a table WITH generated columns targets the insertable
            // (non-generated) columns only, so materialize that explicit list — the executor
            // then maps each value to the right column and computes the generated ones. Done
            // AFTER the arity check (which used the positional "table T has N columns …" form).
            if columns.is_none() && has_generated {
                columns = Some(positional_insertable_columns(td));
            }
            let (mut node, _names) = compile_values(ctx, rows, parent)?;
            // A positional insert (`columns == None`) supplies every column, so nothing
            // is omitted; only an explicit list can leave a column to its default.
            if let Some(cols) = columns.as_mut() {
                let (idxs, exprs) = default_injections(ctx, td, cols)?;
                if !idxs.is_empty() {
                    append_values_row_defaults(&mut node, &exprs)?;
                    width += idxs.len();
                    cols.extend(idxs);
                }
            }
            (node, width, columns)
        }
        InsertSource::Select(sel) => {
            let mut columns = resolve_target_columns(&stmt.columns, td, &table)?;
            let (mut node, names) = compile_select_with_parent(ctx, sel, parent)?;
            let mut width = names.len();
            check_arity(&columns, arity_n, width, &table)?;
            // Positional INSERT … SELECT into a table with generated columns targets the
            // insertable columns only (see the VALUES arm).
            if columns.is_none() && has_generated {
                columns = Some(positional_insertable_columns(td));
            }
            // Same default substitution, but the source is a SELECT tree: wrap it in a
            // Project that passes its `width` output columns through (`Column(0..width)`)
            // then appends the default constants.
            if let Some(cols) = columns.as_mut() {
                let (idxs, exprs) = default_injections(ctx, td, cols)?;
                if !idxs.is_empty() {
                    let mut proj: Vec<EvalExpr> = (0..width).map(EvalExpr::Column).collect();
                    proj.extend(exprs);
                    node = PlanNode::Project { input: Box::new(node), exprs: proj };
                    width += idxs.len();
                    cols.extend(idxs);
                }
            }
            (node, width, columns)
        }
    };

    // `INSERT OR <algorithm>`; a bare INSERT defaults to ABORT.
    let on_conflict = stmt.or_conflict.map_or(OnConflict::Abort, conflict_to_on_conflict);

    // The `ON CONFLICT ...` UPSERT clauses (`lang_upsert.html`), bound AFTER the source
    // and BEFORE `RETURNING` so anonymous `?` parameters in a DO UPDATE's `SET`/`WHERE`
    // number in textual order (source, then SET/WHERE, then RETURNING). `None` for a
    // plain INSERT (no `ON CONFLICT`). The CTE scope is still in force here, so a DO
    // UPDATE subquery can reference a leading `WITH`.
    let upsert = compile_upsert(ctx, td, db, &table, &stmt.upsert)?;

    // `RETURNING`, bound against the inserted row `[c0..c_{N-1}, rowid]`.
    let (returning, result_columns) = compile_returning(ctx, stmt, td, db)?;

    // Invariant: with an explicit target list the (possibly default-extended) source width
    // equals the number of target columns — the executor maps source value `k` to
    // `columns[k]`. The three source arms above each grow `source_width` and `columns` in
    // lockstep (`width += idxs.len()` alongside `cols.extend(idxs)`); pin it here so a future
    // edit that grows one without the other trips loudly in debug/tests rather than silently
    // mis-mapping the row. (A positional insert carries no list; its width is checked == N.)
    if let Some(cols) = &columns {
        debug_assert_eq!(
            source_width,
            cols.len(),
            "INSERT source_width must equal the target column count"
        );
    }

    // Drop the CTE name scope BEFORE compiling triggers. A trigger body is a SEPARATE
    // program compiled under a fresh `PlanCtx` that SHARES the thread-local CTE scope, and
    // per `lang_with.html` §1 a CTE exists "only for the duration of a single SQL statement"
    // (the firing one) — so it must be invisible inside the triggers it fires. This INSERT's
    // own CTE-referencing binding (source SELECT + RETURNING) is already complete above.
    // Dropping only pops the name->id scope entries; `ctx.ctes` (-> the Insert plan's CTEs)
    // is untouched, so the source's `CteScan` ids stay valid. Without this, a trigger body's
    // `FROM c` would bind the firing CTE `c` instead of base table `c`, emitting a `CteScan`
    // whose id indexes THIS statement's `ctes` into the trigger action's own (separate,
    // usually empty) plan — a dangling reference. (Regression: `conformance_dml_cte_trigger`.)
    drop(cte_guard);

    // Attach the triggers this INSERT fires — but ONLY for a top-level statement. A
    // trigger action's own INSERT (`parent = Some`) does NOT recurse into the triggers
    // of the table IT writes; the executor drives that runtime recursion, and compiling
    // it here would over-expand (and could loop on a self-referential trigger). `td` is
    // a base table (a view target is rejected above), so its direct triggers apply.
    let triggers = if parent.is_none() {
        compile_triggers(ctx, catalog, db, td, TriggerDmlEvent::Insert)?
    } else {
        Vec::new()
    };

    // The table's CHECK constraints, bound against the inserted row `[c0..c_{N-1}, rowid]`.
    // UNLIKE triggers, these attach REGARDLESS of `parent`: a CHECK is a property of the
    // table enforced on every inserted row, including one a trigger action inserts (there
    // is no re-expansion / recursion concern — the executor just evaluates the predicate).
    let checks = compile_checks(ctx, td)?;

    // Expression-index key programs, bound against the SAME inserted-row frame, so the
    // executor can maintain any `CREATE INDEX i ON t(<expr>)` on this table as rows are
    // inserted (`lang_createindex.html` §1.2). Empty when no index on `t` has an
    // expression key column.
    let index_key_exprs = compile_table_index_key_exprs(ctx, db, td)?;

    // Partial-index WHERE predicates, bound against that SAME inserted-row frame, so the
    // executor includes a UNIQUE/plain partial index's entry (and enforces its uniqueness)
    // ONLY for rows the predicate accepts (`partialindex.html` §2). Empty when no index on
    // `t` is partial.
    let index_partial_predicates = compile_table_index_partial_predicates(ctx, db, td)?;

    // The explicit-rowid channel: the `columns` position holding the sentinel N (set by
    // `rowid_target_index` for a plain-rowid-table rowid/_rowid_/oid target). At most one
    // (a duplicate rowid target is already a "duplicate column name" error). None for a
    // positional insert (no list) or an INTEGER PRIMARY KEY table (its alias maps to a real
    // column index < N).
    let rowid_source = columns.as_ref().and_then(|c| c.iter().position(|&t| t == n));

    let node = PlanNode::Insert(Insert {
        table,
        db,
        column_count: n,
        columns,
        source: Box::new(source),
        source_width,
        column_affinities,
        on_conflict,
        returning,
        triggers,
        checks,
        rowid_source,
        upsert,
        index_key_exprs,
        index_partial_predicates,
    });
    Ok((node, result_columns))
}

/// Resolve an explicit `(col, ...)` target list to 0-based column indices in the
/// order written (so source value `k` maps to `indices[k]`). `None` (no list) stays
/// `None` — a positional insert over all columns.
fn resolve_target_columns(
    stmt_columns: &Option<Vec<String>>,
    td: &TableDef,
    table: &str,
) -> Result<Option<Vec<usize>>> {
    let names = match stmt_columns {
        None => return Ok(None),
        Some(names) => names,
    };
    let mut indices = Vec::with_capacity(names.len());
    for name in names {
        // A real column shadows the rowid keywords (SQLite: a user column named `rowid`
        // is that column, not the rowid), so resolve a named column first and only then
        // fall back to the `rowid`/`_rowid_`/`oid` handling.
        let idx = match td.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name)) {
            Some(i) => i,
            None => rowid_target_index(td, name, table)?,
        };
        // A GENERATED column may not be written directly (`gencol.html` §2: "their values
        // can not be directly written"). Naming one in the target list is real SQLite's
        // exact `cannot INSERT into generated column "<name>"`. The sentinel `N` (the
        // explicit-rowid channel) is never a real column, so it is excluded by `idx < N`.
        if idx < td.columns.len() && td.columns[idx].generated.is_some() {
            return Err(Error::sql(format!(
                "cannot INSERT into generated column \"{}\"",
                td.columns[idx].name
            )));
        }
        // A column named twice in the target list is treated as a loud error. Real
        // SQLite's behavior here is unverified and may instead keep the last value; if
        // real sqlite later disagrees, relax this to last-wins.
        if indices.contains(&idx) {
            return Err(Error::sql(format!("duplicate column name: {name}")));
        }
        indices.push(idx);
    }
    Ok(Some(indices))
}

/// The schema indices of a table's INSERTABLE (non-generated) columns, in `CREATE TABLE`
/// order — the explicit target list a positional INSERT on a table with generated columns
/// is equivalent to (`gencol.html`: a generated column is computed, never supplied). For a
/// table with no generated column this is `0..N`, but it is only ever called when at least
/// one column IS generated (the positional-conversion guard), so the common path never
/// allocates it.
fn positional_insertable_columns(td: &TableDef) -> Vec<usize> {
    td.columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.generated.is_none())
        .map(|(i, _)| i)
        .collect()
}

/// Resolve a target-list name that is not a declared column: `rowid`/`_rowid_`/`oid`
/// (case-insensitive) name the rowid, everything else is the loud "no such column".
///
/// Three cases by table kind:
/// * INTEGER PRIMARY KEY table (`rowid_alias == Some`): that alias column IS the rowid,
///   so the keyword maps to its index — the executor derives the rowid from that column's
///   inserted value.
/// * PLAIN rowid table (`rowid_alias == None` and `without_rowid == false`): the rowid is
///   not a table column, so the keyword maps to the SENTINEL index `N` (== the column
///   count): `columns` carries `N` at that position and the caller records it in the
///   node's `rowid_source`, so the executor reads the supplied value as the explicit rowid
///   instead of storing it as a column.
/// * WITHOUT ROWID table (`without_rowid == true`): there is NO rowid, so the keyword is
///   not special — it is a plain unknown column name, the same loud "no such column" real
///   SQLite reports, never the sentinel channel (which is for rowid tables only).
///
/// `rowid_alias == None` therefore does NOT by itself mean "plain rowid table": a WITHOUT
/// ROWID table also has no alias, so `without_rowid` is what separates the last two cases.
fn rowid_target_index(td: &TableDef, name: &str, table: &str) -> Result<usize> {
    if !is_rowid_keyword(name) {
        return Err(Error::sql(format!("table {table} has no column named {name}")));
    }
    match td.rowid_alias {
        Some(i) => Ok(i),
        // A WITHOUT ROWID table has no rowid at all, so the keyword is not special there —
        // report it as an unknown column exactly as real SQLite does; never route it through
        // the plain-rowid sentinel channel below.
        None if td.without_rowid => {
            Err(Error::sql(format!("table {table} has no column named {name}")))
        }
        // Plain rowid table: the keyword names the hidden rowid. Return the SENTINEL index
        // N (== column count) — not a real column. `columns` carries N at this position and
        // `rowid_source` (computed by the caller) points the executor at the value, which it
        // reads as the rowid instead of storing as a column.
        None => Ok(td.columns.len()),
    }
}

/// Bind the `DEFAULT` of every table column NOT already in `supplied`, returning
/// `(indices, exprs)` in matching order: `exprs[k]` is the bound default for column
/// `indices[k]`, ready to append to the INSERT source (with `indices` appended to
/// `columns`). By construction `indices.len() == exprs.len()`.
///
/// A column is deliberately left OUT — for the executor to fill — in two cases:
/// * it is the rowid alias (`td.rowid_alias`): an omitted INTEGER PRIMARY KEY must
///   auto-assign its rowid, so a DEFAULT on it is ignored (never injected);
/// * it has no `DEFAULT`: the executor stores NULL for a column absent from `columns`.
///
/// SQLite column defaults are constant expressions (`lang_createtable.html`), so each
/// binds against an empty scope; a non-constant default is rejected there (an empty
/// scope resolves no columns) rather than being silently mis-bound.
fn default_injections(
    ctx: &mut PlanCtx,
    td: &TableDef,
    supplied: &[usize],
) -> Result<(Vec<usize>, Vec<EvalExpr>)> {
    // O(N) membership test over the (small) supplied set instead of a per-column scan.
    // `supplied` holds resolved column indices (`< N`) plus, possibly, the sentinel `N`
    // (the explicit-rowid channel on a plain rowid table) — not a real column, so it is
    // skipped rather than indexed.
    let mut is_supplied = vec![false; td.columns.len()];
    for &c in supplied {
        // Skip EXACTLY the sentinel N (the explicit-rowid channel): it is not a real column
        // and supplies no default. The sentinel is the ONLY legal out-of-range value; an
        // index strictly greater than N is a planner bug, so let it panic-index below (fail
        // loud) rather than silently absorbing it.
        if c == is_supplied.len() {
            continue;
        }
        is_supplied[c] = true;
    }

    let mut indices = Vec::new();
    let mut exprs = Vec::new();
    for (i, col) in td.columns.iter().enumerate() {
        // A GENERATED column is never injected — it has no DEFAULT (`gencol.html` §2.3) and
        // is computed on write, so it must not be given a value here. (`col.default` is
        // always `None` for it, so the check below would skip it anyway; this is explicit.)
        if is_supplied[i] || td.rowid_alias == Some(i) || col.generated.is_some() {
            continue;
        }
        let Some(text) = col.default.as_deref() else {
            continue;
        };
        let ast = parse_default_expr(text)?;
        exprs.push(bind_expr(&Scope::empty(), ctx, &ast)?);
        indices.push(i);
    }
    Ok((indices, exprs))
}

/// Append the bound default expressions (one per omitted defaulted column) to every
/// row of a `VALUES` source. `compile_values` always yields a `PlanNode::Values`, so
/// anything else is a malformed plan and fails closed rather than silently dropping
/// the defaults.
fn append_values_row_defaults(node: &mut PlanNode, defaults: &[EvalExpr]) -> Result<()> {
    match node {
        PlanNode::Values { rows } => {
            for row in rows.iter_mut() {
                row.extend(defaults.iter().cloned());
            }
            Ok(())
        }
        _ => Err(Error::sql("INSERT VALUES source did not compile to a Values node")),
    }
}

/// Parse the raw SQL text of a column `DEFAULT` (as stored in
/// [`ColumnDef::default`](minisqlite_catalog::ColumnDef)) back into an expression AST.
///
/// The SQL crate exposes only whole-statement parsing, so the text is parsed as the
/// single projection of `SELECT <text>` and that one expression extracted (the same
/// wrapper the expression-eval tests use). Anything that is not exactly one plain
/// scalar projection fails LOUD rather than silently mis-defaulting.
fn parse_default_expr(text: &str) -> Result<Expr> {
    let ast = parse(&format!("SELECT {text}"))?;
    let [Statement::Select(select)] = ast.statements.as_slice() else {
        return Err(Error::sql(format!("malformed column DEFAULT expression: {text}")));
    };
    let SelectBody::Select(SelectCore::Query { columns, .. }) = &select.body else {
        return Err(Error::sql(format!("malformed column DEFAULT expression: {text}")));
    };
    match columns.as_slice() {
        [ResultColumn::Expr { expr, .. }] => Ok(expr.clone()),
        _ => Err(Error::sql(format!("malformed column DEFAULT expression: {text}"))),
    }
}

/// The common width of a `VALUES` row list (`lang_insert.html`: every row must have
/// the same number of terms), validated up front so a width/arity error fires with
/// SQLite's exact wording before any row is bound. `compile_values` also rejects
/// unequal widths, but later (mid-binding) and with different wording; this pass is
/// what keeps the INSERT path fail-fast and SQLite-faithful. Consolidating the two
/// would first require aligning `compile_values`'s message to this one.
fn values_width(rows: &[Vec<Expr>]) -> Result<usize> {
    // The parser guarantees at least one row; stay defensive rather than indexing.
    let width = rows.first().map_or(0, Vec::len);
    for row in rows {
        if row.len() != width {
            return Err(Error::sql("all VALUES must have the same number of terms"));
        }
    }
    Ok(width)
}

/// The INSERT arity rule (`lang_insert.html`): with a column list, each source term
/// must supply exactly one value per listed column; without one, exactly one per table
/// column. The two mismatches use real sqlite3's two distinct messages.
fn check_arity(
    columns: &Option<Vec<usize>>,
    n: usize,
    source_width: usize,
    table: &str,
) -> Result<()> {
    match columns {
        None => {
            if source_width != n {
                return Err(Error::sql(format!(
                    "table {table} has {n} columns but {source_width} values were supplied"
                )));
            }
        }
        Some(cols) => {
            if source_width != cols.len() {
                return Err(Error::sql(format!(
                    "{source_width} values for {} columns",
                    cols.len()
                )));
            }
        }
    }
    Ok(())
}

/// Map the parsed `INSERT OR <algorithm>` clause to the plan's [`OnConflict`].
fn conflict_to_on_conflict(c: ConflictClause) -> OnConflict {
    match c {
        ConflictClause::Rollback => OnConflict::Rollback,
        ConflictClause::Abort => OnConflict::Abort,
        ConflictClause::Fail => OnConflict::Fail,
        ConflictClause::Ignore => OnConflict::Ignore,
        ConflictClause::Replace => OnConflict::Replace,
    }
}

/// Bind a `RETURNING` clause against the inserted row layout `[c0..c_{N-1}, rowid]`
/// (width `N+1`), returning the bound output expressions and their result-column
/// names. An empty clause yields two empty vecs (no output columns).
fn compile_returning(
    ctx: &mut PlanCtx,
    stmt: &minisqlite_sql::Insert,
    td: &TableDef,
    db: DbIndex,
) -> Result<(Vec<EvalExpr>, Vec<String>)> {
    if stmt.returning.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    // One source exposing the table's columns at base 0; its rowid sits at register `N`
    // (the last register of a `Source::BaseTable`), exactly the row the executor
    // evaluates RETURNING over.
    // The exposed name is the INSERT alias if present, else the table name, so a
    // qualified `t.col` / `alias.col` reference resolves.
    let exposed_name = stmt.alias.clone().unwrap_or_else(|| td.name.clone());
    let sources = [Source::BaseTable { exposed_name, table: td, db, base: 0 }];
    let scope = Scope::new(&sources);

    let mut exprs = Vec::with_capacity(stmt.returning.len());
    let mut names = Vec::with_capacity(stmt.returning.len());
    for rc in &stmt.returning {
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

/// Compile the chained `ON CONFLICT ...` UPSERT clauses of an INSERT into a
/// [`UpsertPlan`], or `None` for a plain INSERT with no `ON CONFLICT` (`lang_upsert.html`).
///
/// Each clause resolves its conflict target to the uniqueness-constraint column set it
/// matches (`None` for a target-omitted clause), and — for `DO UPDATE` — binds its `SET`
/// assignments and optional `WHERE` over the combined row `existing(W) ++ excluded(W)`,
/// `W` the target's base-row width. The PARENT-SCOPE trick makes a bare / `table.`-qualified
/// column (and `rowid`) read the EXISTING row while `excluded.col` reads the
/// would-be-inserted row, WITHOUT either being ambiguous:
///
/// * the CHILD scope is one source over the table at base 0 exposed under the table's own
///   name, so a bare column resolves there — into `[0, W)` — and never sees `excluded`;
/// * the PARENT scope is one source over the SAME table at base `W` exposed as `excluded`,
///   so only an `excluded.`-qualified reference (which misses the child) reaches it, into
///   `[W, 2W)`.
///
/// `W` is taken from [`Source::width`] (the single source of truth), so the base of
/// `excluded` and the combined-row width the executor forms can never drift from a scan's
/// row shape. Binding order across clauses is textual (SET then WHERE, clause by clause),
/// so anonymous `?` parameters number as SQLite assigns them.
///
/// The `excluded` parent is a CO-ROW half of the same physical `2W`-wide row (not an outer
/// prefix below the child), so [`Scope::total_width`] reports `2W` for this scope. That
/// makes a CORRELATED subquery in a SET value / row-value source / the `WHERE` place its own
/// sources above the full combined row, matching the `existing(W) ++ excluded(W)` outer the
/// executor forms — so correlated DO UPDATE binds exactly like the plain `UPDATE` path.
fn compile_upsert(
    ctx: &mut PlanCtx,
    td: &TableDef,
    db: DbIndex,
    table: &str,
    upserts: &[Upsert],
) -> Result<Option<UpsertPlan>> {
    if upserts.is_empty() {
        return Ok(None);
    }
    let existing =
        [Source::BaseTable { exposed_name: table.to_string(), table: td, db, base: 0 }];
    let w = existing[0].width();
    let excluded = [Source::BaseTable {
        exposed_name: "excluded".to_string(),
        table: td,
        db,
        base: w,
    }];
    let excluded_scope = Scope::new(&excluded);
    let scope = Scope::with_parent(&existing, Some(&excluded_scope));

    // Copied out once (a shared `&dyn Catalog` is `Copy`), so the per-clause `&mut ctx` binds
    // below do not conflict with reading the schema's indexes for target validation.
    let catalog = ctx.catalog;
    let mut clauses = Vec::with_capacity(upserts.len());
    for up in upserts {
        let target = resolve_conflict_target(catalog, db, td, table, up.target.as_ref())?;
        let action = match &up.action {
            UpsertAction::Nothing => UpsertActionPlan::Nothing,
            UpsertAction::Update { set, where_clause } => {
                // SET binds first, then the WHERE (textual `?` order).
                let assignments = bind_upsert_assignments(&scope, ctx, td, set)?;
                let predicate =
                    where_clause.as_ref().map(|w| bind_expr(&scope, ctx, w)).transpose()?;
                UpsertActionPlan::Update { assignments, predicate }
            }
        };
        clauses.push(UpsertClause { target, action });
    }
    Ok(Some(UpsertPlan { clauses }))
}

/// Resolve an `ON CONFLICT` target to the [`ConflictTarget`] naming the uniqueness
/// constraint it matches (`lang_upsert.html` §2), resolved against the schema at plan time:
///
/// * target omitted (`ON CONFLICT DO ...`, since 3.35.0) → [`ConflictTarget::Any`], matching
///   any uniqueness conflict;
/// * every part a bare column name → [`ConflictTarget::Columns`] (the INTEGER PRIMARY KEY
///   rowid alias, or a UNIQUE index's column SET), validated against a real uniqueness
///   constraint by [`target_matches_uniqueness`];
/// * any part an EXPRESSION (`ON CONFLICT(a+b)` / `ON CONFLICT(lower(s))`, and the mixed
///   `(a, b+c)` form) → [`ConflictTarget::ExprIndex`], pinned to the one non-partial UNIQUE
///   *expression* index whose key structurally equals the target — by its b-tree ROOT PAGE,
///   the identity the executor's `IndexPlan` carries.
///
/// Each `IndexedColumn`'s ASC/DESC/COLLATE are index-ordering details irrelevant to WHICH
/// constraint the target names, so they are ignored.
///
/// * a `WHERE` on the target names a PARTIAL unique index (`lang_upsert.html` §2 — a "unique
///   index" includes a partial one; `partialindex.html` §2.1) → [`ConflictTarget::ExprIndex`],
///   pinned to the one PARTIAL UNIQUE index whose key structurally equals the target AND whose
///   `partial_predicate` structurally equals the target's `WHERE`. Covers both a column-based
///   partial index (`… ON t(a) WHERE b>0`) and an expression-based one (`… ON t(lower(a)) WHERE
///   b>0`), via [`match_partial_conflict_index`]. This is correct now that DML index maintenance
///   honors a partial index's predicate (`exec/ops/dml_index.rs::index_admits_row` gates every
///   add / UNIQUE-probe / delete on it), so a partial unique index physically holds ONLY its
///   matching rows and UNIQUE is enforced across exactly those — routing an upsert to it is sound.
fn resolve_conflict_target(
    catalog: &dyn Catalog,
    db: DbIndex,
    td: &TableDef,
    table: &str,
    target: Option<&UpsertTarget>,
) -> Result<ConflictTarget> {
    let Some(target) = target else {
        return Ok(ConflictTarget::Any);
    };
    if let Some(where_pred) = target.where_clause.as_ref() {
        // A WHERE on the conflict target names a PARTIAL unique index (`lang_upsert.html` §2,
        // `partialindex.html` §2.1): the target's key AND its WHERE must BOTH match the partial
        // index. It resolves to that index's b-tree root — the identity the executor's `IndexPlan`
        // carries — exactly like an expression index; the executor's `decide_upsert` matches the
        // conflict on that root (`ConflictTarget::ExprIndex`). Both a column-based and an
        // expression-based partial index are handled here (via the shared key-part match), since a
        // partial index — column- OR expression-based — is named by its root, not a plain column set.
        let target_parts: Vec<KeyPart> = target.columns.iter().map(ic_key_part).collect();
        let root = match_partial_conflict_index(catalog, db, table, &target_parts, where_pred)?;
        return Ok(ConflictTarget::ExprIndex(root));
    }
    // Any expression part means the target names an INDEXED EXPRESSION, not a plain column set
    // (an expression can never be the rowid alias), so it can only match a UNIQUE expression
    // index — resolved structurally to that index's root page.
    if target.columns.iter().any(|ic| matches!(ic.target, IndexedColumnTarget::Expr(_))) {
        let target_parts: Vec<KeyPart> = target.columns.iter().map(ic_key_part).collect();
        let root = match_expression_conflict_index(catalog, db, table, &target_parts)?;
        return Ok(ConflictTarget::ExprIndex(root));
    }
    // All-column-name target: the column-set path. The any-expression branch above returned for
    // any expression part, so every part here is a `Name`. Match exhaustively (rather than an
    // `if let` that silently drops a non-name part) so the invariant is enforced by the compiler:
    // a future `IndexedColumnTarget` variant breaks the build here until it is handled.
    let mut cols = Vec::with_capacity(target.columns.len());
    for ic in &target.columns {
        match &ic.target {
            IndexedColumnTarget::Name(name) => {
                cols.push(conflict_target_column_index(td, table, name)?)
            }
            IndexedColumnTarget::Expr(_) => {
                unreachable!("expression targets are resolved by the any-expression branch above")
            }
        }
    }
    // The target must NAME a uniqueness constraint (`lang_upsert.html` §2: it "specifies ... a
    // uniqueness constraint"): the INTEGER PRIMARY KEY (rowid alias) or some UNIQUE index's
    // exact column set. A target that resolves to real columns but matches NO constraint is an
    // error in real sqlite ("ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE
    // constraint"), NOT a silent accept-then-plain-insert. Skipped for WITHOUT ROWID tables:
    // their PRIMARY KEY *is* the table b-tree, which owns no separate index for
    // `target_matches_uniqueness` to enumerate, so a valid `ON CONFLICT(pkcols)` would spuriously
    // fail here. A WR upsert's column target is instead validated at execution against the actual
    // PRIMARY KEY and secondary indexes (`exec/ops/insert.rs::decide_upsert_wr`).
    if !td.without_rowid && !target_matches_uniqueness(catalog, db, td, table, &cols)? {
        return Err(Error::sql(
            "ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint",
        ));
    }
    Ok(ConflictTarget::Columns(cols))
}

/// One key part of a conflict target or of a UNIQUE index's key, for the STRUCTURAL match
/// that resolves an expression conflict target (`ON CONFLICT(a + b)`) to its index. A bare
/// column name compares case-insensitively (SQL identifiers are case-insensitive); an indexed
/// expression compares by [`expr_key_eq`]'s case-insensitive structural equality (NOT `Expr`'s
/// derived, case-sensitive `==`). Borrows its source (the target AST or the [`IndexDef`]) — no
/// clone.
enum KeyPart<'a> {
    Name(&'a str),
    Expr(&'a Expr),
}

/// The [`KeyPart`] an `ON CONFLICT` target part names: a bare column, or an indexed
/// expression (`lang_createindex.html` §1.2). ASC/DESC/COLLATE are ignored (see
/// [`resolve_conflict_target`]).
fn ic_key_part(ic: &IndexedColumn) -> KeyPart<'_> {
    match &ic.target {
        IndexedColumnTarget::Name(name) => KeyPart::Name(name),
        IndexedColumnTarget::Expr(expr) => KeyPart::Expr(expr),
    }
}

/// The key parts of an index in key order. `IndexDef.key_exprs[i] == Some(expr)` marks key
/// column `i` as a genuine EXPRESSION (with an empty-name sentinel in `columns[i]`); else it
/// is the ordinary named column `columns[i]` (`catalog::def`, `IndexDef`). Read via `.get(i)`
/// so a schema whose `key_exprs` is shorter than `columns` degrades to the named-column path
/// rather than panicking — the same defensive read `catalog::introspect` uses.
fn index_key_parts(idx: &IndexDef) -> Vec<KeyPart<'_>> {
    idx.columns
        .iter()
        .enumerate()
        .map(|(i, name)| match idx.key_exprs.get(i) {
            Some(Some(expr)) => KeyPart::Expr(expr),
            _ => KeyPart::Name(name.as_str()),
        })
        .collect()
}

/// Whether two key parts are the same constraint part: column names case-insensitively,
/// expressions by [`expr_key_eq`]'s case-insensitive structural equality (so `a + b` matches
/// the index key `a + b` — and case-varied `A + B` / `LOWER(s)` match too — but `b + a` or
/// `a + 1` do not). A name and an expression never match.
fn key_part_eq(a: &KeyPart, b: &KeyPart) -> bool {
    match (a, b) {
        (KeyPart::Name(x), KeyPart::Name(y)) => x.eq_ignore_ascii_case(y),
        (KeyPart::Expr(x), KeyPart::Expr(y)) => expr_key_eq(x, y),
        _ => false,
    }
}

/// Structural equality of two index-key expressions under SQL's case-insensitive IDENTIFIER
/// semantics: column names, table/schema qualifiers, function names, `COLLATE` names, and
/// `CAST` type names compare with `eq_ignore_ascii_case` (SQL identifiers are case-insensitive —
/// the same fold the bare-column [`KeyPart::Name`] path applies); operators, structure, and
/// LITERAL VALUES compare exactly (a text literal `'Foo'` is NOT `'foo'`). Applied recursively,
/// so `ON CONFLICT(A + B)` matches an index key `a + b` and `ON CONFLICT(LOWER(s))` matches
/// `lower(s)`, exactly as real sqlite matches identifiers — where `Expr`'s DERIVED `==` (which
/// this replaces) wrongly rejected a target on a mere case difference.
///
/// It handles every expression shape a `CREATE INDEX` key may take (`lang_createindex.html`
/// §1.2); the forms an index key can NEVER contain (a bind parameter, a sub-select, `RAISE`)
/// fall back to exact `==` — never reached for a valid key, and fail-closed if they somehow are.
///
/// GENERALIZATION NOTE: this is currently the engine's only structural comparison of an index
/// KEY expression (the query planner still skips expression-index keys, `compile/covering.rs`).
/// When an expression-index seek/lookup path is added, hoist this into a shared helper so both
/// sites use the SAME case-insensitive notion rather than reintroducing a case-sensitive `==`.
fn expr_key_eq(a: &Expr, b: &Expr) -> bool {
    fn ci(a: &str, b: &str) -> bool {
        a.eq_ignore_ascii_case(b)
    }
    fn opt_ci(a: &Option<String>, b: &Option<String>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => x.eq_ignore_ascii_case(y),
            _ => false,
        }
    }
    fn opt_eq(a: &Option<Box<Expr>>, b: &Option<Box<Expr>>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => expr_key_eq(x, y),
            _ => false,
        }
    }
    fn exprs_eq(a: &[Expr], b: &[Expr]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| expr_key_eq(x, y))
    }
    fn fn_args_eq(a: &minisqlite_sql::FunctionArgs, b: &minisqlite_sql::FunctionArgs) -> bool {
        use minisqlite_sql::FunctionArgs::{Empty, List, Star};
        match (a, b) {
            (List(x), List(y)) => exprs_eq(x, y),
            (Star, Star) => true,
            (Empty, Empty) => true,
            _ => false,
        }
    }
    fn in_rhs_eq(a: &minisqlite_sql::InRhs, b: &minisqlite_sql::InRhs) -> bool {
        use minisqlite_sql::InRhs::List;
        match (a, b) {
            (List(x), List(y)) => exprs_eq(x, y),
            // A sub-select / table-valued `IN` rhs can't appear in an index key; compare exactly.
            _ => a == b,
        }
    }
    match (a, b) {
        (Expr::Literal(x), Expr::Literal(y)) => x == y,
        (
            Expr::Column { schema: s1, table: t1, name: n1, from_dqs: d1 },
            Expr::Column { schema: s2, table: t2, name: n2, from_dqs: d2 },
        ) => opt_ci(s1, s2) && opt_ci(t1, t2) && ci(n1, n2) && d1 == d2,
        (Expr::Unary { op: o1, expr: e1 }, Expr::Unary { op: o2, expr: e2 }) => {
            o1 == o2 && expr_key_eq(e1, e2)
        }
        (
            Expr::Binary { op: o1, left: l1, right: r1 },
            Expr::Binary { op: o2, left: l2, right: r2 },
        ) => o1 == o2 && expr_key_eq(l1, l2) && expr_key_eq(r1, r2),
        (
            Expr::Function { name: n1, distinct: d1, args: a1, filter: f1, over: o1, order_by: ob1 },
            Expr::Function { name: n2, distinct: d2, args: a2, filter: f2, over: o2, order_by: ob2 },
        ) => {
            ci(n1, n2)
                && d1 == d2
                && fn_args_eq(a1, a2)
                && opt_eq(f1, f2)
                && o1 == o2
                && ob1 == ob2
        }
        (Expr::Cast { expr: e1, type_name: t1 }, Expr::Cast { expr: e2, type_name: t2 }) => {
            ci(t1, t2) && expr_key_eq(e1, e2)
        }
        (Expr::Collate { expr: e1, collation: c1 }, Expr::Collate { expr: e2, collation: c2 }) => {
            ci(c1, c2) && expr_key_eq(e1, e2)
        }
        (
            Expr::Like { negated: g1, kind: k1, lhs: l1, rhs: r1, escape: e1 },
            Expr::Like { negated: g2, kind: k2, lhs: l2, rhs: r2, escape: e2 },
        ) => g1 == g2 && k1 == k2 && expr_key_eq(l1, l2) && expr_key_eq(r1, r2) && opt_eq(e1, e2),
        (
            Expr::Between { negated: g1, expr: x1, low: lo1, high: hi1 },
            Expr::Between { negated: g2, expr: x2, low: lo2, high: hi2 },
        ) => g1 == g2 && expr_key_eq(x1, x2) && expr_key_eq(lo1, lo2) && expr_key_eq(hi1, hi2),
        (
            Expr::In { negated: g1, expr: x1, rhs: rhs1 },
            Expr::In { negated: g2, expr: x2, rhs: rhs2 },
        ) => g1 == g2 && expr_key_eq(x1, x2) && in_rhs_eq(rhs1, rhs2),
        (
            Expr::Case { operand: op1, whens: w1, else_expr: el1 },
            Expr::Case { operand: op2, whens: w2, else_expr: el2 },
        ) => {
            opt_eq(op1, op2)
                && w1.len() == w2.len()
                && w1
                    .iter()
                    .zip(w2)
                    .all(|(p, q)| expr_key_eq(&p.0, &q.0) && expr_key_eq(&p.1, &q.1))
                && opt_eq(el1, el2)
        }
        (Expr::IsNull(x), Expr::IsNull(y)) => expr_key_eq(x, y),
        (Expr::NotNull(x), Expr::NotNull(y)) => expr_key_eq(x, y),
        (Expr::Parenthesized(x), Expr::Parenthesized(y)) => exprs_eq(x, y),
        // Forms an index key can never contain (`lang_createindex.html` §1.2): a bind parameter,
        // a scalar sub-select / EXISTS, RAISE. Compare exactly — no identifier to fold, and
        // fail-closed (never equal to a different shape) if one somehow reaches here.
        (Expr::BindParam(_), _)
        | (Expr::Exists { .. }, _)
        | (Expr::Subquery(_), _)
        | (Expr::Raise(_), _) => a == b,
        _ => false,
    }
}

/// Whether two key-part lists are equal as MULTISETS — same parts, order-insensitively (a
/// conflict target names a constraint by its key SET, `lang_upsert.html` §2). Each part of
/// `a` is matched to a distinct unused part of `b`, so duplicated parts are counted.
fn key_parts_multiset_eq(a: &[KeyPart], b: &[KeyPart]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut used = vec![false; b.len()];
    'next: for pa in a {
        for (i, pb) in b.iter().enumerate() {
            if !used[i] && key_part_eq(pa, pb) {
                used[i] = true;
                continue 'next;
            }
        }
        return false;
    }
    true
}

/// Resolve an EXPRESSION conflict target to the b-tree ROOT PAGE of the one non-partial
/// UNIQUE index whose key structurally equals `target_parts` (`lang_upsert.html` §2: the
/// target "must ... match a single uniqueness constraint"). The root page is the identity the
/// executor's `IndexPlan` carries (`exec/ops/dml_index.rs`: `IndexPlan.root == idx.root_page`),
/// so the plan and the executor name the SAME index. Fail-closed: NO match → the same "does
/// not match any PRIMARY KEY or UNIQUE constraint" error a bad column target gives; TWO
/// matching indexes (an ambiguous target) → a loud ambiguity error rather than an arbitrary
/// pick. `db` is the insert target's namespace, so a same-named shadow cannot match against
/// another namespace's indexes (as the column path also guards).
fn match_expression_conflict_index(
    catalog: &dyn Catalog,
    db: DbIndex,
    table: &str,
    target_parts: &[KeyPart],
) -> Result<u32> {
    let mut matched: Option<u32> = None;
    for idx in catalog.indexes_on_in(db, table)? {
        // Only a NON-partial UNIQUE index is named by this NO-WHERE expression path: a non-unique
        // index is not a uniqueness constraint, and a PARTIAL index is named only WITH its matching
        // WHERE — resolved by [`match_partial_conflict_index`], not here — so this no-WHERE path
        // must skip it (mirroring the [`target_matches_uniqueness`] partial skip on the column path).
        if !idx.unique || idx.partial {
            continue;
        }
        if key_parts_multiset_eq(target_parts, &index_key_parts(idx)) {
            if matched.is_some() {
                return Err(Error::sql(
                    "ON CONFLICT target matches more than one UNIQUE index (ambiguous)",
                ));
            }
            matched = Some(idx.root_page);
        }
    }
    matched.ok_or_else(|| {
        Error::sql("ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint")
    })
}

/// Resolve a PARTIAL-index conflict target (`ON CONFLICT(<cols-or-exprs>) WHERE <pred>`,
/// `lang_upsert.html` §2 + `partialindex.html` §2.1) to the b-tree ROOT PAGE of the one PARTIAL
/// UNIQUE index whose key structurally equals `target_parts` AND whose partial predicate
/// structurally equals `where_pred`. BOTH must match:
///
/// 1. the KEY — column names case-insensitively and indexed expressions by [`expr_key_eq`],
///    order-insensitively as a multiset ([`key_parts_multiset_eq`]); the SAME comparison
///    [`match_expression_conflict_index`] uses, so a column-based (`ON CONFLICT(a)`) and an
///    expression-based (`ON CONFLICT(lower(a))`) target are handled uniformly;
/// 2. the PREDICATE — the target's `WHERE` must MATCH the partial index's `WHERE` by [`expr_key_eq`]
///    structural equality (SQLite requires the target WHERE to correspond to the index WHERE), NOT
///    semantic implication: `WHERE b>0` does not match an index `WHERE b>=0`. The parser strips a
///    single-element paren (`crate::parser`), so `WHERE (b>0)` and `WHERE b>0` are already the same
///    `Expr` here and match without special-casing.
///
/// Only a PARTIAL UNIQUE index is eligible — a non-unique index is not a uniqueness constraint, and
/// a NON-partial (full) unique index is named WITHOUT a `WHERE` (the column / expression paths of
/// [`resolve_conflict_target`]), so this is the mirror of those matchers' `idx.partial` skip: they
/// exclude partial indexes, this includes ONLY partial ones. Fail-closed exactly like
/// [`match_expression_conflict_index`]: NO match → the same "does not match any PRIMARY KEY or
/// UNIQUE constraint" error; TWO matching indexes → a loud ambiguity error rather than an arbitrary
/// pick. `db` is the insert target's namespace, so a same-named shadow in another namespace cannot
/// match (as the sibling matchers also guard).
fn match_partial_conflict_index(
    catalog: &dyn Catalog,
    db: DbIndex,
    table: &str,
    target_parts: &[KeyPart],
    where_pred: &Expr,
) -> Result<u32> {
    let mut matched: Option<u32> = None;
    for idx in catalog.indexes_on_in(db, table)? {
        if !idx.unique || !idx.partial {
            continue;
        }
        // A partial index always carries its predicate (the catalog invariant `partial ==
        // partial_predicate.is_some()`); a partial index with none cannot be verified against the
        // target's WHERE, so skip it (fail-closed) rather than match a predicate we cannot compare.
        let Some(pred) = idx.partial_predicate.as_ref() else {
            continue;
        };
        // Predicate first (cheap short-circuit for the common single-partial-index table), then the
        // key set. BOTH must match for the target to name this partial index.
        if !expr_key_eq(pred, where_pred) {
            continue;
        }
        if !key_parts_multiset_eq(target_parts, &index_key_parts(idx)) {
            continue;
        }
        if matched.is_some() {
            return Err(Error::sql(
                "ON CONFLICT target matches more than one partial UNIQUE index (ambiguous)",
            ));
        }
        matched = Some(idx.root_page);
    }
    matched.ok_or_else(|| {
        Error::sql("ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint")
    })
}

/// Whether `cols` (schema column indices) is EXACTLY the column set of some uniqueness
/// constraint on `table`: the single-column INTEGER PRIMARY KEY (the rowid alias), or a UNIQUE
/// index — auto-created for a `UNIQUE`/non-integer-`PRIMARY KEY` column, or an explicit
/// `CREATE UNIQUE INDEX`. This is the SAME set the executor's `decide_upsert` can match a
/// detected conflict against (the rowid alias plus each unique index plan), so a target this
/// accepts is exactly one the executor can actually honor — the plan and exec never disagree on
/// which targets are valid.
fn target_matches_uniqueness(
    catalog: &dyn Catalog,
    db: DbIndex,
    td: &TableDef,
    table: &str,
    cols: &[usize],
) -> Result<bool> {
    // The INTEGER PRIMARY KEY is a single-column constraint on the rowid-alias column (it owns
    // no separate index — the table b-tree IS keyed by the rowid — so it is checked here).
    if let Some(ai) = td.rowid_alias {
        if cols.len() == 1 && cols[0] == ai {
            return Ok(true);
        }
    }
    // Any UNIQUE index. Map each index's column NAMES to schema indices (as the executor's
    // `build_index_plan` does) and compare order-insensitively, matching `same_column_set`.
    // `db` is the insert target's namespace: read its indexes with `_in(db)` so a same-named
    // temp/attached shadow can't validate the `ON CONFLICT` target against another namespace's
    // constraints — the executor's `build_index_plans(node.db)` maintains only THIS namespace's
    // indexes, so the two must agree on which target is honorable. No shadow => equals bare.
    for idx in catalog.indexes_on_in(db, table)? {
        if !idx.unique {
            continue;
        }
        // Skip a mixed/EXPRESSION index. An expression key column is stored as an empty-name
        // sentinel in `columns` (`key_exprs[i] == Some(_)`, `columns[i] == ""`), which the
        // `filter_map` below would silently DROP (no real column is named "") — shrinking e.g.
        // `(a, b + c)` (`columns == ["a", ""]`) down to just `(a)` and letting a column-only
        // target false-match a subset (`ON CONFLICT(a)` wrongly matching `UNIQUE(a, b + c)`).
        // Such an index is a uniqueness constraint only over its FULL key and is named solely via
        // the ExprIndex path ([`match_expression_conflict_index`]), never as a plain column set.
        if idx.key_exprs.iter().any(|e| e.is_some()) {
            continue;
        }
        // Skip a PARTIAL index. A bare `ON CONFLICT(cols)` (no WHERE) does not name a partial
        // index in real sqlite — that requires the matching `WHERE`, which the resolver resolves
        // via [`match_partial_conflict_index`] above — and the expression path skips `idx.partial`
        // too, so the two sibling matchers agree on which indexes are eligible targets.
        if idx.partial {
            continue;
        }
        let positions: Vec<usize> = idx
            .columns
            .iter()
            .filter_map(|c| td.columns.iter().position(|col| col.name.eq_ignore_ascii_case(c)))
            .collect();
        if positions.len() == cols.len() && cols.iter().all(|x| positions.contains(x)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Resolve one `ON CONFLICT` target column name to its schema index: a declared column
/// shadows the rowid keywords (as everywhere); `rowid`/`_rowid_`/`oid` name the INTEGER
/// PRIMARY KEY alias column (whose uniqueness IS the rowid's), matching how the executor
/// reports a rowid conflict's target as the alias column set. A rowid keyword on a table
/// with no INTEGER PRIMARY KEY has no column set to match and is a loud error.
fn conflict_target_column_index(td: &TableDef, table: &str, name: &str) -> Result<usize> {
    if let Some(i) = td.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name)) {
        return Ok(i);
    }
    if is_rowid_keyword(name) {
        if let Some(ai) = td.rowid_alias {
            return Ok(ai);
        }
        return Err(Error::sql(format!(
            "ON CONFLICT target {name} on a table with no INTEGER PRIMARY KEY is not supported"
        )));
    }
    Err(Error::sql(format!("table {table} has no column named {name}")))
}

/// Bind a DO UPDATE `SET` list into `(schema column index, value expr)` pairs, evaluated
/// against the combined `existing ++ excluded` row. Mirrors `compile::update`'s
/// `bind_assignments` for the same [`SetClause`] shape — a `col = expr`, a same-arity
/// `(a, b) = (x, y)` row value, a single-name `(a) = expr` (the parser unwraps `(x)` to
/// `x`), the width-mismatch error, and a `(a, b, …) = (SELECT …)` subquery ROW-VALUE
/// source (compiled once, one `ScalarSubqueryColumn` per name).
///
/// A subquery — a row-value source, a scalar SET value, or one inside the DO UPDATE
/// `WHERE` — may be CORRELATED to the conflict row, referencing both the existing row
/// (a bare / `table.`-qualified column) and the candidate (`excluded.col`). That binds
/// correctly because the combined scope reports `total_width() == 2W`: the `excluded`
/// source is a CO-ROW half of the same physical row (not an outer prefix), so
/// [`Scope::total_width`] composes it via `.max(parent)` (see there and `compile_upsert`)
/// and `compile_subplan` sizes the correlated outer at the full `2W`, placing the subplan's
/// own sources ABOVE the combined row — exactly matching the `existing(W) ++ excluded(W)`
/// outer the executor forms (`minisqlite-exec`'s `do_upsert_update`). This reuses the SAME
/// correlation machinery the plain `UPDATE` path uses. The DML compilers are separate
/// contention cells, so this small duplication with `bind_assignments` is intentional.
fn bind_upsert_assignments(
    scope: &Scope,
    ctx: &mut PlanCtx,
    td: &TableDef,
    set: &[SetClause],
) -> Result<Vec<(usize, EvalExpr)>> {
    let mut out = Vec::new();
    for clause in set {
        match clause {
            SetClause::Column { name, value } => {
                let idx = set_target_column_index(td, name)?;
                out.push((idx, bind_expr(scope, ctx, value)?));
            }
            SetClause::Columns { names, value } => match value {
                Expr::Parenthesized(list) if list.len() == names.len() => {
                    for (nm, item) in names.iter().zip(list.iter()) {
                        let idx = set_target_column_index(td, nm)?;
                        out.push((idx, bind_expr(scope, ctx, item)?));
                    }
                }
                Expr::Parenthesized(list) => {
                    return Err(Error::sql(format!(
                        "{} columns assigned {} values",
                        names.len(),
                        list.len()
                    )))
                }
                _ if names.len() == 1 => {
                    let idx = set_target_column_index(td, &names[0])?;
                    out.push((idx, bind_expr(scope, ctx, value)?));
                }
                // `(a, b, …) = (SELECT x, y, …)`: a subquery ROW-VALUE source
                // (rowvalue.html §2.3), the upsert twin of `compile::update`'s arm. Compile
                // the subquery ONCE, width-check it against the name list (the same
                // `"N columns assigned M values"` size error, NOT a silent truncate/pad),
                // then emit one assignment per name reading its positional column of the
                // subquery's first result row.
                Expr::Subquery(select) => {
                    let (id, width) = compile_columnlist_subquery(scope, ctx, select)?;
                    if width != names.len() {
                        return Err(Error::sql(format!(
                            "{} columns assigned {} values",
                            names.len(),
                            width
                        )));
                    }
                    // A source CORRELATED to the conflict row (`(a, b) = (SELECT … WHERE
                    // src.k = t.k)`) is fully supported: the combined scope reports
                    // `total_width() == 2W`, so `compile_columnlist_subquery` (via
                    // `compile_subplan`) places the subplan's own sources above the
                    // `existing(W) ++ excluded(W)` row the executor forms — same as the plain
                    // `UPDATE` path. See this fn's doc and `compile_upsert`.
                    for (i, nm) in names.iter().enumerate() {
                        let idx = set_target_column_index(td, nm)?;
                        out.push((idx, EvalExpr::ScalarSubqueryColumn { id, col: i }));
                    }
                }
                // A multi-name list fed by any other bare scalar (`(a, b) = 1`) is a size
                // mismatch — one value for N columns.
                _ => {
                    return Err(Error::sql(format!("{} columns assigned 1 values", names.len())))
                }
            },
        }
    }
    Ok(out)
}

/// Resolve a DO UPDATE `SET`-target column name to its schema index (mirrors
/// `compile::update`'s `column_index`): a declared column shadows the rowid keywords;
/// `rowid`/`_rowid_`/`oid` map to the INTEGER PRIMARY KEY alias, and are a loud gap on a
/// plain rowid table (the executor derives a moved rowid from the alias slot alone).
fn set_target_column_index(td: &TableDef, name: &str) -> Result<usize> {
    if let Some(i) = td.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name)) {
        return Ok(i);
    }
    if is_rowid_keyword(name) {
        return match td.rowid_alias {
            Some(i) => Ok(i),
            None => Err(Error::sql(format!(
                "UPSERT DO UPDATE with an explicit rowid target ({name}) on a table with no \
                 INTEGER PRIMARY KEY is not yet supported"
            ))),
        };
    }
    Err(Error::sql(format!("no such column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef};
    use minisqlite_pager::Pager;
    // `parse` is already in scope via `use super::*` (the parent module imports it); the
    // rest of the AST builders used only by tests are pulled in here.
    use minisqlite_sql::{CreateIndex, CreateTable, Drop};
    use minisqlite_types::Value;

    use crate::plan::{CtePlan, Plan};
    use crate::{Planner, QueryPlanner};

    // ---- test catalog + builders (a static schema store; only reads are exercised) --

    struct TestCatalog {
        tables: Vec<TableDef>,
        indexes: Vec<IndexDef>,
    }

    impl TestCatalog {
        fn new(tables: Vec<TableDef>) -> Self {
            TestCatalog { tables, indexes: Vec::new() }
        }
        /// Attach index defs — UPSERT conflict-target validation enumerates a table's UNIQUE
        /// indexes, so a table that carries one needs it here. Fluent, for use in `cat()`.
        fn with_indexes(mut self, indexes: Vec<IndexDef>) -> Self {
            self.indexes = indexes;
            self
        }
    }

    impl Catalog for TestCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
            Ok(self.indexes.iter().find(|i| i.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(self.indexes.iter().filter(|i| i.table.eq_ignore_ascii_case(table)).collect())
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

    /// A column carrying a `DEFAULT`, whose text is the RENDERED SQL form the catalog
    /// stores (e.g. a string literal `'zz'` keeps its quotes, an integer is bare `5`).
    fn col_def(name: &str, decl: Option<&str>, default: &str) -> ColumnDef {
        ColumnDef { default: Some(default.to_string()), ..col(name, decl) }
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

    /// A UNIQUE, non-partial index `name` on `table`'s plain `columns` (no expression keys).
    /// The column-target path reads `table`/`columns`/`unique`; `key_exprs` is empty, so
    /// [`index_key_parts`] classifies every key column as a NAME. For an index over an
    /// EXPRESSION key, use [`uidx_parts`].
    fn uidx(name: &str, table: &str, columns: &[&str]) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: columns.iter().map(|c| c.to_string()).collect(),
            key_columns: Vec::new(),
            key_exprs: Vec::new(),
            root_page: 3,
            unique: true,
            partial: false,
            partial_predicate: None,
        }
    }

    /// An index over `parts` (each a key part's SQL text), classified EXACTLY as the parser
    /// classifies an `ON CONFLICT` target part and the catalog stores an index key: a bare
    /// unqualified column name is a NAME part (`columns[i]` set, `key_exprs[i] = None`),
    /// anything else an EXPRESSION part (`key_exprs[i] = Some(expr)`, `columns[i]` an
    /// empty-name sentinel). `root_page` is the identity a resolved `ExprIndex` pins to;
    /// `unique` covers the eligibility case (a non-unique index is not a valid target). The index
    /// is always NON-partial; for a PARTIAL index carrying a real WHERE predicate use
    /// [`uidx_partial`].
    fn uidx_parts(
        name: &str,
        table: &str,
        parts: &[&str],
        root_page: u32,
        unique: bool,
    ) -> IndexDef {
        let mut columns = Vec::with_capacity(parts.len());
        let mut key_exprs = Vec::with_capacity(parts.len());
        for p in parts {
            let expr = parse_default_expr(p).expect("index key part parses");
            if let Expr::Column { schema: None, table: None, name, .. } = &expr {
                columns.push(name.clone());
                key_exprs.push(None);
            } else {
                columns.push(String::new());
                key_exprs.push(Some(expr));
            }
        }
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns,
            key_columns: Vec::new(),
            key_exprs,
            root_page,
            unique,
            partial: false,
            partial_predicate: None,
        }
    }

    /// A UNIQUE PARTIAL index over `parts` carrying the REAL partial WHERE `predicate` (SQL
    /// text, parsed the same way the parser parses a `CREATE INDEX ... WHERE` / an `ON CONFLICT
    /// ... WHERE` clause). Unlike [`uidx_parts`] (which always builds a NON-partial index with no
    /// predicate), this is for the PARTIAL-index conflict-target tests, which DO read the
    /// predicate — a WHERE target resolves only when its `WHERE` structurally equals the index's
    /// `partial_predicate`.
    fn uidx_partial(name: &str, table: &str, parts: &[&str], root_page: u32, predicate: &str) -> IndexDef {
        IndexDef {
            partial: true,
            partial_predicate: Some(
                parse_default_expr(predicate).expect("partial predicate parses"),
            ),
            ..uidx_parts(name, table, parts, root_page, true)
        }
    }

    /// `t(a INTEGER, b TEXT)`, `s(x INTEGER, y TEXT)` (INSERT..SELECT source),
    /// `k(id INTEGER PRIMARY KEY, v TEXT)` (rowid-alias / RETURNING rowid checks),
    /// `d(a INTEGER, b TEXT DEFAULT 'zz')` and `dd(a INTEGER DEFAULT 5, b TEXT DEFAULT 'z')`
    /// (DEFAULT substitution), `d3(a INTEGER, b TEXT DEFAULT 'zz', c TEXT)` (mixed: one
    /// omitted column has a default, another does not), `dt(a INTEGER, b DEFAULT
    /// CURRENT_TIMESTAMP)` (a keyword-literal default) and `dr(a INTEGER, b REAL DEFAULT
    /// 3.14)` (a REAL literal default), `kd(id INTEGER PRIMARY KEY DEFAULT 9, v TEXT)` (a
    /// DEFAULT on the rowid alias, which must still be ignored so the rowid auto-assigns),
    /// `dbad(a INTEGER, b TEXT DEFAULT a)` (a non-constant default, which must fail loud
    /// rather than mis-bind), and `tr(rowid TEXT, v INTEGER)` (a real user column named
    /// `rowid`, which must shadow the rowid keyword in the target list).
    fn cat() -> TestCatalog {
        TestCatalog::new(vec![
            tdef("t", vec![col("a", Some("INTEGER")), col("b", Some("TEXT"))], None),
            tdef("s", vec![col("x", Some("INTEGER")), col("y", Some("TEXT"))], None),
            tdef("k", vec![col("id", Some("INTEGER")), col("v", Some("TEXT"))], Some(0)),
            tdef("d", vec![col("a", Some("INTEGER")), col_def("b", Some("TEXT"), "'zz'")], None),
            tdef(
                "dd",
                vec![col_def("a", Some("INTEGER"), "5"), col_def("b", Some("TEXT"), "'z'")],
                None,
            ),
            tdef(
                "d3",
                vec![
                    col("a", Some("INTEGER")),
                    col_def("b", Some("TEXT"), "'zz'"),
                    col("c", Some("TEXT")),
                ],
                None,
            ),
            // `CURRENT_TIMESTAMP` is stored by the catalog as its bare keyword text (see
            // builder::render_literal), a non-numeric/non-string literal default path.
            tdef("dt", vec![col("a", Some("INTEGER")), col_def("b", None, "CURRENT_TIMESTAMP")], None),
            tdef("kd", vec![col_def("id", Some("INTEGER"), "9"), col("v", Some("TEXT"))], Some(0)),
            // `dbad(a INTEGER, b TEXT DEFAULT a)`: b's default TEXT references a COLUMN,
            // so it is not a constant. It exists only to exercise the loud-error path for
            // a non-constant default (see `non_constant_column_default_is_a_loud_error`).
            tdef("dbad", vec![col("a", Some("INTEGER")), col_def("b", Some("TEXT"), "a")], None),
            // `dr(a INTEGER, b REAL DEFAULT 3.14)`: a REAL literal default, exercising the
            // float-literal parse+bind path the INTEGER/TEXT/keyword default tests skip.
            tdef("dr", vec![col("a", Some("INTEGER")), col_def("b", Some("REAL"), "3.14")], None),
            // `tr(rowid TEXT, v INTEGER)`: a real user column literally named `rowid` (NOT an
            // INTEGER PRIMARY KEY), used to pin that a real column shadows the rowid keyword
            // in the target list (see `real_column_named_rowid_shadows_the_keyword`).
            tdef("tr", vec![col("rowid", Some("TEXT")), col("v", Some("INTEGER"))], None),
            // `tu(a INTEGER, b TEXT)` with a UNIQUE index on `b` (see `.with_indexes` below):
            // a plain rowid table whose non-PK column `b` DOES carry a uniqueness constraint,
            // so `ON CONFLICT(b)` is a valid target (see
            // `upsert_target_non_alias_column_resolves_to_its_index`).
            tdef("tu", vec![col("a", Some("INTEGER")), col("b", Some("TEXT"))], None),
            // Expression-index conflict-target fixtures (`ON CONFLICT(<expr>)`):
            // `te(a, b)` UNIQUE on `a + b`; `tea(a, b, c)` UNIQUE on the mixed key `(a, b+c)`;
            // `tne(a, b)` a NON-unique index on `a + b` (ineligible as a target); `tamb(a, b)`
            // TWO unique indexes on the same `a + b` key (an ambiguous target). See the
            // `.with_indexes` below and the `upsert_expression_target_*` tests.
            tdef("te", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None),
            tdef(
                "tea",
                vec![
                    col("a", Some("INTEGER")),
                    col("b", Some("INTEGER")),
                    col("c", Some("INTEGER")),
                ],
                None,
            ),
            tdef("tne", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None),
            tdef("tamb", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None),
            // `tp(a, b)` carries TWO PARTIAL unique indexes, both `WHERE b > 0`: a column-based
            // one on `a` (root 48) and an expression-based one on `lower(a)` (root 51). A bare
            // `ON CONFLICT(a)` (no WHERE) names NEITHER (real sqlite needs the matching WHERE), so
            // the column path rejects it; a `ON CONFLICT(a) WHERE b > 0` names the column-based one
            // and `ON CONFLICT(lower(a)) WHERE b > 0` the expression-based one — see the
            // `upsert_partial_*` tests.
            tdef("tp", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None),
            // `tm(a, b, c)` carries BOTH a plain single-column UNIQUE index on `a` (root 49) AND a
            // mixed expression UNIQUE index `(a, b + c)` (root 50), so the two matchers interact on
            // ONE table: `ON CONFLICT(a)` skips the mixed index and resolves to the plain sibling,
            // while `ON CONFLICT(a, b + c)` resolves to the mixed index. See the two
            // `..._sibling...` tests.
            tdef(
                "tm",
                vec![
                    col("a", Some("INTEGER")),
                    col("b", Some("INTEGER")),
                    col("c", Some("INTEGER")),
                ],
                None,
            ),
            // `tpamb(a, b)` carries TWO partial unique indexes with the SAME key `a` AND the SAME
            // predicate `b > 0` (roots 52, 53) — a degenerate schema real sqlite permits. An
            // `ON CONFLICT(a) WHERE b > 0` target matches BOTH, so the resolver must raise the
            // partial-index ambiguity error rather than arbitrarily picking one. See
            // `upsert_partial_target_matching_two_partial_indexes_is_ambiguous`.
            tdef("tpamb", vec![col("a", Some("INTEGER")), col("b", Some("INTEGER"))], None),
        ])
        .with_indexes(vec![
            uidx("tu_b", "tu", &["b"]),
            uidx_parts("te_ab", "te", &["a + b"], 42, true),
            uidx_parts("tea_abc", "tea", &["a", "b + c"], 43, true),
            uidx_parts("tamb_1", "tamb", &["a + b"], 44, true),
            uidx_parts("tamb_2", "tamb", &["a + b"], 45, true),
            uidx_parts("tne_ab", "tne", &["a + b"], 46, false),
            uidx_partial("tp_a", "tp", &["a"], 48, "b > 0"),
            uidx_partial("tp_la", "tp", &["lower(a)"], 51, "b > 0"),
            uidx_parts("tm_a", "tm", &["a"], 49, true),
            uidx_parts("tm_a_bc", "tm", &["a", "b + c"], 50, true),
            uidx_partial("tpamb_1", "tpamb", &["a"], 52, "b > 0"),
            uidx_partial("tpamb_2", "tpamb", &["a"], 53, "b > 0"),
        ])
    }

    fn plan_ins(sql: &str) -> Result<Plan> {
        let c = cat();
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c)
    }

    fn expect_insert(node: &PlanNode) -> &Insert {
        match node {
            PlanNode::Insert(i) => i,
            other => panic!("expected PlanNode::Insert, got {other:?}"),
        }
    }

    // ---- VALUES: positional, reordered column list, single column -----------------

    #[test]
    fn values_no_column_list_positional() {
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x')").unwrap();
        assert!(plan.mutates);
        assert!(plan.result_columns.is_empty(), "no RETURNING → no output columns");
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.table, "t");
        assert_eq!(ins.column_count, 2);
        assert_eq!(ins.columns, None, "no explicit column list");
        assert_eq!(ins.source_width, 2);
        assert_eq!(ins.column_affinities, vec![Affinity::Integer, Affinity::Text]);
        assert!(matches!(ins.on_conflict, OnConflict::Abort), "bare INSERT defaults to ABORT");
        assert!(ins.returning.is_empty());
        match ins.source.as_ref() {
            PlanNode::Values { rows } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 2);
            }
            other => panic!("expected Values source, got {other:?}"),
        }
    }

    #[test]
    fn values_reordered_column_list_maps_indices() {
        // `(b, a)` → source value 0 targets column b (index 1), value 1 targets a (0).
        let plan = plan_ins("INSERT INTO t(b, a) VALUES ('x', 1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![1, 0]));
        assert_eq!(ins.source_width, 2);
        // Affinities stay in table/schema order (a=INTEGER, b=TEXT) regardless of the
        // reordered target list — the executor applies them by target column, so they
        // must not be permuted to follow the column list.
        assert_eq!(ins.column_affinities, vec![Affinity::Integer, Affinity::Text]);
    }

    #[test]
    fn values_single_column_list() {
        let plan = plan_ins("INSERT INTO t(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0]));
        assert_eq!(ins.source_width, 1);
    }

    #[test]
    fn values_multi_row_builds_all_rows() {
        // Two rows of two terms each → a Values source carrying both bound rows.
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x'), (2, 'y')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, None);
        assert_eq!(ins.source_width, 2);
        match ins.source.as_ref() {
            PlanNode::Values { rows } => {
                assert_eq!(rows.len(), 2, "both rows are built");
                assert!(rows.iter().all(|r| r.len() == 2), "each row has 2 terms");
            }
            other => panic!("expected Values source, got {other:?}"),
        }
    }

    // ---- arity + equal-width errors ----------------------------------------------

    #[test]
    fn arity_error_no_column_list() {
        // t has 2 columns; only 1 value supplied.
        let err = plan_ins("INSERT INTO t VALUES (1)").unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("table t has 2 columns but 1 values were supplied"), "got {m:?}")
            }
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn arity_error_with_column_list() {
        // 1 target column but 2 values → sqlite3's "K values for M columns" form.
        let err = plan_ins("INSERT INTO t(a) VALUES (1, 2)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("2 values for 1 columns"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn values_unequal_widths_is_an_error() {
        let err = plan_ins("INSERT INTO t VALUES (1, 'x'), (2)").unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("all VALUES must have the same number of terms"), "got {m:?}")
            }
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- INSERT ... SELECT --------------------------------------------------------

    #[test]
    fn insert_from_select() {
        let plan = plan_ins("INSERT INTO t SELECT x, y FROM s").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.column_count, 2);
        assert_eq!(ins.columns, None);
        assert_eq!(ins.source_width, 2, "SELECT projects 2 columns");
        // The source is the compiled SELECT tree, not a literal Values node.
        assert!(!matches!(ins.source.as_ref(), PlanNode::Values { .. }), "SELECT source");
    }

    #[test]
    fn insert_from_select_arity_error() {
        // SELECT projects 1 column into a 2-column table (no column list).
        let err = plan_ins("INSERT INTO t SELECT x FROM s").unwrap_err();
        match err {
            Error::Sql(m) => {
                assert!(m.contains("table t has 2 columns but 1 values were supplied"), "got {m:?}")
            }
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- OR conflict ------------------------------------------------------------

    #[test]
    fn or_replace_sets_replace() {
        let plan = plan_ins("INSERT OR REPLACE INTO t VALUES (1, 'x')").unwrap();
        assert!(matches!(expect_insert(&plan.root).on_conflict, OnConflict::Replace));
    }

    #[test]
    fn or_ignore_sets_ignore() {
        let plan = plan_ins("INSERT OR IGNORE INTO t VALUES (1, 'x')").unwrap();
        assert!(matches!(expect_insert(&plan.root).on_conflict, OnConflict::Ignore));
    }

    #[test]
    fn or_fail_and_rollback() {
        assert!(matches!(
            expect_insert(&plan_ins("INSERT OR FAIL INTO t VALUES (1, 'x')").unwrap().root)
                .on_conflict,
            OnConflict::Fail
        ));
        assert!(matches!(
            expect_insert(&plan_ins("INSERT OR ROLLBACK INTO t VALUES (1, 'x')").unwrap().root)
                .on_conflict,
            OnConflict::Rollback
        ));
    }

    #[test]
    fn replace_into_is_or_replace() {
        // `REPLACE INTO` is sugar for `INSERT OR REPLACE INTO`.
        let plan = plan_ins("REPLACE INTO t VALUES (1, 'x')").unwrap();
        assert!(matches!(expect_insert(&plan.root).on_conflict, OnConflict::Replace));
    }

    // ---- RETURNING ---------------------------------------------------------------

    #[test]
    fn returning_columns_and_rowid() {
        // a → register 0; rowid → register N (=2) in the inserted `[a, b, rowid]` row.
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x') RETURNING a, rowid").unwrap();
        assert_eq!(plan.result_columns, vec!["a".to_string(), "rowid".to_string()]);
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.returning.len(), 2);
        assert!(matches!(ins.returning[0], EvalExpr::Column(0)), "a → reg 0");
        assert!(matches!(ins.returning[1], EvalExpr::Column(2)), "rowid → reg N=2");
    }

    #[test]
    fn returning_star_expands_all_columns() {
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x') RETURNING *").unwrap();
        assert_eq!(plan.result_columns, vec!["a".to_string(), "b".to_string()]);
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.returning.len(), 2);
        assert!(matches!(ins.returning[0], EvalExpr::Column(0)));
        assert!(matches!(ins.returning[1], EvalExpr::Column(1)));
    }

    #[test]
    fn returning_expression_with_alias() {
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x') RETURNING a + 1 AS s").unwrap();
        assert_eq!(plan.result_columns, vec!["s".to_string()]);
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.returning.len(), 1);
        // The alias only renames the output column; the bound expr is still `a + 1` —
        // an arithmetic node whose left operand is column a (register 0).
        match &ins.returning[0] {
            EvalExpr::Arith { left, .. } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(0)), "left operand is column a")
            }
            other => panic!("expected Arith, got {other:?}"),
        }
    }

    #[test]
    fn returning_qualified_by_insert_alias() {
        // `INSERT INTO t AS x` exposes the inserted row under alias `x`, so `RETURNING
        // x.a` resolves to column a (register 0) and names the output column `a`.
        let plan = plan_ins("INSERT INTO t AS x VALUES (1, 'x') RETURNING x.a").unwrap();
        assert_eq!(plan.result_columns, vec!["a".to_string()]);
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.returning.len(), 1);
        assert!(matches!(ins.returning[0], EvalExpr::Column(0)), "x.a → reg 0");
    }

    #[test]
    fn returning_integer_primary_key_alias_and_rowid_both_resolve_to_rowid_register() {
        // k has an INTEGER PRIMARY KEY (`id`, rowid alias). Both `id` and `rowid`
        // resolve to register N (=2), the rowid the executor fills for RETURNING.
        let plan = plan_ins("INSERT INTO k VALUES (1, 'x') RETURNING id, rowid").unwrap();
        assert_eq!(plan.result_columns, vec!["id".to_string(), "rowid".to_string()]);
        let ins = expect_insert(&plan.root);
        assert!(matches!(ins.returning[0], EvalExpr::Column(2)), "IPK id aliases rowid reg N=2");
        assert!(matches!(ins.returning[1], EvalExpr::Column(2)), "rowid → reg N=2");
    }

    // ---- resolution errors -------------------------------------------------------

    #[test]
    fn unknown_table_is_a_loud_error() {
        let err = plan_ins("INSERT INTO nope VALUES (1)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such table: nope"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_column_is_a_loud_error() {
        let err = plan_ins("INSERT INTO t(z) VALUES (1)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("table t has no column named z"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_column_in_list_is_a_loud_error() {
        // Erroring on a duplicate target column is the current behavior; real SQLite
        // may instead keep the last value. This test pins the current loud-error
        // behavior.
        let err = plan_ins("INSERT INTO t(a, a) VALUES (1, 2)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("duplicate column name: a"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- UPSERT (ON CONFLICT) compiled onto the node -----------------------------

    #[test]
    fn plain_insert_has_no_upsert() {
        // No `ON CONFLICT` → the node carries `upsert: None`, so the executor takes the
        // plain conflict path. (Was previously a loud "not yet supported" gap for any
        // ON CONFLICT; the reject is gone.)
        assert!(expect_insert(&plan_ins("INSERT INTO t VALUES (1, 'x')").unwrap().root)
            .upsert
            .is_none());
    }

    #[test]
    fn upsert_do_nothing_target_omitted_compiles_to_ir() {
        // `ON CONFLICT DO NOTHING` (target omitted, since 3.35.0) → one clause with an `Any`
        // target (matches any uniqueness conflict) and a `Nothing` action.
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x') ON CONFLICT DO NOTHING").unwrap();
        let upsert =
            expect_insert(&plan.root).upsert.as_ref().expect("ON CONFLICT → an UpsertPlan");
        assert_eq!(upsert.clauses.len(), 1);
        assert_eq!(
            upsert.clauses[0].target,
            ConflictTarget::Any,
            "target-omitted clause matches any conflict"
        );
        assert!(matches!(upsert.clauses[0].action, UpsertActionPlan::Nothing));
    }

    #[test]
    fn upsert_do_update_resolves_alias_target_and_binds_existing_vs_excluded() {
        // k(id INTEGER PRIMARY KEY, v TEXT): `ON CONFLICT(id)` resolves to the alias column
        // set `[0]`. In `SET v = excluded.v`, the LHS is column index 1 (v); the RHS
        // `excluded.v` binds to the EXCLUDED half at register `W + 1` (W = N+1 = 3) → 4 via
        // the parent-scope trick, while a bare `v` would be the EXISTING half at register 1.
        let plan =
            plan_ins("INSERT INTO k VALUES (1, 'a') ON CONFLICT(id) DO UPDATE SET v = excluded.v")
                .unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().expect("DO UPDATE → an UpsertPlan");
        assert_eq!(upsert.clauses.len(), 1);
        assert_eq!(
            upsert.clauses[0].target,
            ConflictTarget::Columns(vec![0]),
            "ON CONFLICT(id) → alias column [0]"
        );
        match &upsert.clauses[0].action {
            UpsertActionPlan::Update { assignments, predicate } => {
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, 1, "SET v → column index 1");
                assert!(matches!(assignments[0].1, EvalExpr::Column(4)), "excluded.v → reg W+1=4");
                assert!(predicate.is_none(), "no WHERE");
            }
            other => panic!("expected DO UPDATE, got {other:?}"),
        }
    }

    #[test]
    fn upsert_do_update_bare_column_is_existing_and_where_binds() {
        // `SET v = v WHERE v = excluded.v` on k: the bare RHS `v` and the WHERE's left `v`
        // bind to the EXISTING half (register 1), `excluded.v` to the EXCLUDED half
        // (register 4) — pinning that a bare column reads the existing row, not the candidate.
        let plan = plan_ins(
            "INSERT INTO k VALUES (1, 'a') ON CONFLICT(id) DO UPDATE SET v = v WHERE v = excluded.v",
        )
        .unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        match &upsert.clauses[0].action {
            UpsertActionPlan::Update { assignments, predicate } => {
                assert!(matches!(assignments[0].1, EvalExpr::Column(1)), "bare v → existing reg 1");
                match predicate.as_ref().expect("WHERE present") {
                    EvalExpr::Compare { left, right, .. } => {
                        assert!(matches!(left.as_ref(), EvalExpr::Column(1)), "WHERE bare v → 1");
                        assert!(
                            matches!(right.as_ref(), EvalExpr::Column(4)),
                            "WHERE excluded.v → 4"
                        );
                    }
                    other => panic!("expected a Compare WHERE, got {other:?}"),
                }
            }
            other => panic!("expected DO UPDATE, got {other:?}"),
        }
    }

    #[test]
    fn upsert_target_non_alias_column_resolves_to_its_index() {
        // On `tu(a INTEGER, b TEXT)` with a UNIQUE index on `b` (plain rowid table, no alias),
        // `ON CONFLICT(b)` resolves to the constraint's column set `[1]` (b) — a non-rowid
        // uniqueness target. `b` must name a REAL uniqueness constraint (`tu`'s index provides
        // it); on a plain `t(a, b)` with no such constraint it is rejected instead, see
        // `upsert_target_without_uniqueness_constraint_is_rejected`.
        let plan = plan_ins("INSERT INTO tu VALUES (1, 'x') ON CONFLICT(b) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(
            upsert.clauses[0].target,
            ConflictTarget::Columns(vec![1]),
            "ON CONFLICT(b) → [1]"
        );
    }

    #[test]
    fn upsert_target_without_uniqueness_constraint_is_rejected() {
        // `ON CONFLICT(b)` on `t(a, b)` — `b` has NO PRIMARY KEY / UNIQUE / unique index.
        // lang_upsert.html §2: the conflict target "specifies ... a uniqueness constraint", so
        // real sqlite rejects this at prepare; we must too, not silently accept it and fall
        // through to a plain insert.
        let err =
            plan_ins("INSERT INTO t VALUES (1, 'x') ON CONFLICT(b) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_chained_clauses_compile_in_written_order() {
        // Two chained clauses → two UpsertClauses in written order, each with its own
        // resolved target: `ON CONFLICT(id)` → `Columns([0])` (alias) then a target-omitted
        // `DO NOTHING` → `Any`.
        let plan = plan_ins(
            "INSERT INTO k VALUES (1, 'a') ON CONFLICT(id) DO UPDATE SET v = excluded.v ON CONFLICT DO NOTHING",
        )
        .unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses.len(), 2);
        assert_eq!(upsert.clauses[0].target, ConflictTarget::Columns(vec![0]));
        assert!(matches!(upsert.clauses[0].action, UpsertActionPlan::Update { .. }));
        assert_eq!(upsert.clauses[1].target, ConflictTarget::Any);
        assert!(matches!(upsert.clauses[1].action, UpsertActionPlan::Nothing));
    }

    #[test]
    fn upsert_expression_target_resolves_to_its_expression_index() {
        // `te(a, b)` has a UNIQUE index on `a + b` (root 42). `ON CONFLICT(a + b)` names that
        // INDEXED EXPRESSION — it cannot be a plain column set — so it resolves to
        // `ExprIndex(42)`, pinning the clause to that index by its b-tree root, the identity
        // the executor's IndexPlan carries.
        let plan =
            plan_ins("INSERT INTO te VALUES (1, 2) ON CONFLICT(a + b) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(
            upsert.clauses[0].target,
            ConflictTarget::ExprIndex(42),
            "ON CONFLICT(a + b) → the expression index's root page"
        );
    }

    #[test]
    fn upsert_mixed_name_and_expr_target_resolves_to_its_expression_index() {
        // `tea(a, b, c)` is UNIQUE on the MIXED key `(a, b + c)` (root 43). `ON CONFLICT(b + c,
        // a)` matches it part for part (name vs name, expr vs expr) order-insensitively — the
        // target names the constraint by its key SET — so it resolves to `ExprIndex(43)`.
        let plan =
            plan_ins("INSERT INTO tea VALUES (1, 2, 3) ON CONFLICT(b + c, a) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(43));
    }

    #[test]
    fn upsert_expression_target_with_no_matching_index_is_rejected() {
        // `te` indexes only `a + b`. `ON CONFLICT(a - b)` matches no UNIQUE index, so it is a
        // loud error (fail-closed) — never a silent mis-match onto `te_ab`.
        let err =
            plan_ins("INSERT INTO te VALUES (1, 2) ON CONFLICT(a - b) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_expression_target_matching_only_a_non_unique_index_is_rejected() {
        // `tne` has a NON-unique index on `a + b`; a non-unique index is not a uniqueness
        // constraint, so `ON CONFLICT(a + b)` cannot target it and is rejected.
        let err =
            plan_ins("INSERT INTO tne VALUES (1, 2) ON CONFLICT(a + b) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_expression_target_matching_two_unique_indexes_is_ambiguous() {
        // `tamb` has TWO unique indexes on the same key `a + b` (roots 44, 45): the target is
        // ambiguous, so it is a loud error rather than an arbitrary pick.
        let err =
            plan_ins("INSERT INTO tamb VALUES (1, 2) ON CONFLICT(a + b) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("ambiguous"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_column_target_matching_only_a_mixed_expression_index_is_rejected() {
        // REGRESSION: `tea` has NO single-column unique constraint on `a`; its only unique index
        // is the MIXED key `(a, b + c)` (`columns == ["a", ""]`, the second an empty-name
        // sentinel for the expression key). A bare `ON CONFLICT(a)` names no uniqueness
        // constraint, so it must be REJECTED (`lang_upsert.html` §2) — not accepted by dropping
        // that sentinel and mistaking `(a, b + c)` for an `(a)` index. Before the expression-key
        // skip in `target_matches_uniqueness`, this wrongly returned `Ok(Columns([0]))` and the
        // executor then silently plain-inserted a non-colliding row that real sqlite rejects.
        let err =
            plan_ins("INSERT INTO tea VALUES (1, 2, 3) ON CONFLICT(a) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_column_target_still_matches_a_single_column_unique_index() {
        // Guard against an over-broad expression-key skip: `tu` has a PLAIN single-column UNIQUE
        // index on `b` (no expression key, not partial), so `ON CONFLICT(b)` must STILL resolve
        // to `Columns([1])` — the mixed-index skip must not regress a genuine column target.
        let plan =
            plan_ins("INSERT INTO tu VALUES (1, 'x') ON CONFLICT(b) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::Columns(vec![1]));
    }

    #[test]
    fn upsert_column_target_does_not_match_a_partial_unique_index() {
        // `tp` has a PARTIAL unique index on the plain column `a` (root 48). A bare
        // `ON CONFLICT(a)` (no WHERE) does not name a partial index in real sqlite — it needs the
        // matching WHERE, which the resolver resolves via `match_partial_conflict_index` — so this
        // no-WHERE column path skips a partial index just as the expression path does, and rejects
        // the target rather than false-matching it.
        let err = plan_ins("INSERT INTO tp VALUES (1, 2) ON CONFLICT(a) DO NOTHING").unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_expression_target_matches_its_index_case_insensitively() {
        // SQL identifiers are case-insensitive, so `ON CONFLICT(A + B)` names the SAME expression
        // index as `CREATE UNIQUE INDEX ... ON te(a + b)` (root 42). The expression comparison
        // must fold identifier case exactly as the bare-column path does — `Expr`'s derived `==`
        // (which `expr_key_eq` replaced) wrongly rejected the target on the case difference alone.
        let plan =
            plan_ins("INSERT INTO te VALUES (1, 2) ON CONFLICT(A + B) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(
            upsert.clauses[0].target,
            ConflictTarget::ExprIndex(42),
            "case-varied ON CONFLICT(A + B) still names te's a + b index"
        );
    }

    #[test]
    fn upsert_mixed_case_target_resolves_order_and_case_insensitively() {
        // `tea` is UNIQUE on the mixed key `(a, b + c)` (root 43). `ON CONFLICT(B + C, A)` varies
        // BOTH the name part (`A`) and the expression part (`B + C`) in case AND genuinely
        // REORDERS them (expr first, name second) vs the index key order — it still names the
        // constraint by its key SET, so it resolves to `ExprIndex(43)`.
        let plan =
            plan_ins("INSERT INTO tea VALUES (1, 2, 3) ON CONFLICT(B + C, A) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(43));
    }

    #[test]
    fn expr_key_eq_folds_identifier_case_but_not_literals() {
        // The expression-target comparator folds identifier case — column names, table
        // qualifiers, and function names — but keeps operators, structure, and literal VALUES
        // exact. This pins the `LOWER` vs `lower` and `A + B` vs `a + b` contract directly.
        let e = |t: &str| parse_default_expr(t).expect("expression parses");
        // Column identifiers fold case.
        assert!(expr_key_eq(&e("A + B"), &e("a + b")), "column identifiers fold case");
        // Function names fold case.
        assert!(expr_key_eq(&e("LOWER(s)"), &e("lower(s)")), "function names fold case");
        // Table qualifiers fold case in every part.
        assert!(expr_key_eq(&e("T.A + B"), &e("t.a + b")), "table qualifier folds case");
        // CAST type names and COLLATE names are identifiers too — fold their case (and the inner
        // expression's). Both are legal index-key shapes not reachable via the plan-level fixtures.
        assert!(expr_key_eq(&e("CAST(a AS TEXT)"), &e("cast(A as text)")), "CAST type + inner fold");
        assert!(
            expr_key_eq(&e("a COLLATE NOCASE"), &e("A collate nocase")),
            "COLLATE name + inner fold"
        );
        // Case folding is not a wildcard: genuinely different expressions stay distinct.
        assert!(!expr_key_eq(&e("a + b"), &e("a + c")), "different column");
        assert!(!expr_key_eq(&e("a + b"), &e("a - b")), "different operator");
        assert!(!expr_key_eq(&e("a + b"), &e("b + a")), "operand order matters");
        assert!(!expr_key_eq(&e("lower(s)"), &e("upper(s)")), "different function");
        // The CAST type / COLLATE name are still COMPARED (folding is not "ignore"): a differing
        // type or collation must not match.
        assert!(!expr_key_eq(&e("CAST(a AS TEXT)"), &e("CAST(a AS INT)")), "different CAST type");
        assert!(
            !expr_key_eq(&e("a COLLATE NOCASE"), &e("a COLLATE RTRIM")),
            "different COLLATE name"
        );
        // String LITERALS remain case-SENSITIVE — `'Foo'` is not `'foo'`.
        assert!(
            !expr_key_eq(&e("a || 'Foo'"), &e("a || 'foo'")),
            "string literal case is significant"
        );
    }

    #[test]
    fn upsert_column_target_resolves_to_the_plain_sibling_not_the_mixed_index() {
        // `tm` carries BOTH a plain UNIQUE index on `a` (root 49) and a mixed UNIQUE index
        // `(a, b + c)` (root 50). `ON CONFLICT(a)` must SKIP the mixed index (whose empty-name
        // sentinel would otherwise false-match) and resolve to the plain sibling's column set —
        // proving the skip is PRECISE (drops the mixed-only false match, keeps the genuine one),
        // not a blanket rejection of every column target on a table that happens to have an
        // expression index.
        let plan = plan_ins("INSERT INTO tm VALUES (1, 2, 3) ON CONFLICT(a) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::Columns(vec![0]));
    }

    #[test]
    fn upsert_expression_target_resolves_to_the_mixed_index_with_a_plain_sibling_present() {
        // On the same `tm`, the full `ON CONFLICT(a, b + c)` names the MIXED key, not the plain
        // `(a)` sibling — so it resolves to the mixed index's root (50), with no ambiguity.
        let plan =
            plan_ins("INSERT INTO tm VALUES (1, 2, 3) ON CONFLICT(a, b + c) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(50));
    }

    #[test]
    fn upsert_partial_column_target_resolves_to_the_partial_index() {
        // `tp` has a column-based PARTIAL unique index on `a` WHERE `b > 0` (root 48). A target
        // that names the same column AND the same WHERE selects that partial index by root — a
        // partial index is pinned by root (like an expression index), not a plain column set.
        let plan =
            plan_ins("INSERT INTO tp VALUES (1, 2) ON CONFLICT(a) WHERE b > 0 DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(48));
    }

    #[test]
    fn upsert_partial_expression_target_resolves_to_the_partial_index() {
        // `tp` also has an EXPRESSION-based partial unique index on `lower(a)` WHERE `b > 0`
        // (root 51). A `ON CONFLICT(lower(a)) WHERE b > 0` names it via the shared key-part match.
        let plan = plan_ins(
            "INSERT INTO tp VALUES (1, 2) ON CONFLICT(lower(a)) WHERE b > 0 DO NOTHING",
        )
        .unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(51));
    }

    #[test]
    fn upsert_partial_target_matches_a_parenthesized_where() {
        // The parser strips a single-element paren, so `WHERE (b > 0)` is the SAME `Expr` as the
        // index's `WHERE b > 0` — the target still resolves to root 48 with no special handling.
        let plan =
            plan_ins("INSERT INTO tp VALUES (1, 2) ON CONFLICT(a) WHERE (b > 0) DO NOTHING").unwrap();
        let upsert = expect_insert(&plan.root).upsert.as_ref().unwrap();
        assert_eq!(upsert.clauses[0].target, ConflictTarget::ExprIndex(48));
    }

    #[test]
    fn upsert_partial_target_with_mismatched_where_is_rejected() {
        // The index's WHERE is `b > 0`; a target WHERE `b >= 0` is a structurally DIFFERENT
        // predicate (matching is structural, not semantic implication), so it names no constraint.
        let err = plan_ins("INSERT INTO tp VALUES (1, 2) ON CONFLICT(a) WHERE b >= 0 DO NOTHING")
            .unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_partial_where_target_does_not_match_a_nonpartial_index() {
        // `te` has a NON-partial unique index on `a + b`. A WHERE target names only a PARTIAL
        // index, so it must NOT bind to the full index — the mirror of the no-WHERE path skipping
        // partial indexes.
        let err = plan_ins("INSERT INTO te VALUES (1, 2) ON CONFLICT(a + b) WHERE a > 0 DO NOTHING")
            .unwrap_err();
        match err {
            Error::Sql(m) => assert!(
                m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
                "got {m:?}"
            ),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn upsert_partial_target_matching_two_partial_indexes_is_ambiguous() {
        // `tpamb` has TWO partial unique indexes with the SAME key `a` and SAME predicate `b > 0`,
        // so `ON CONFLICT(a) WHERE b > 0` matches BOTH: `match_partial_conflict_index` must raise
        // its ambiguity error (the partial mirror of the expression-index ambiguity case) rather
        // than arbitrarily pick the first-enumerated root.
        let err = plan_ins("INSERT INTO tpamb VALUES (1, 2) ON CONFLICT(a) WHERE b > 0 DO NOTHING")
            .unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("ambiguous"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn with_on_insert_registers_cte_and_compiles() {
        // A leading `WITH` on INSERT is supported: the CTE compiles into `plan.ctes` and the
        // source `SELECT * FROM c` resolves `c` through the shared FROM/`lookup_cte` path, so
        // the whole statement compiles to an INSERT over that CTE-backed source. The executor
        // materializes `plan.ctes` before running the INSERT (a plan-level attach; the Insert
        // node is unchanged). Was previously a loud "not yet supported" gap.
        let plan = plan_ins("WITH c AS (SELECT 1, 2) INSERT INTO t SELECT * FROM c").unwrap();
        assert_eq!(plan.ctes.len(), 1, "the CTE `c` is registered on the plan, got {:?}", plan.ctes);
        assert!(
            matches!(&plan.ctes[0], CtePlan::Materialized { name, .. } if name == "c"),
            "ctes[0] is the materialized CTE `c`, got {:?}",
            plan.ctes[0]
        );
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.table, "t");
        assert_eq!(ins.column_count, 2);
    }

    // ---- DEFAULT VALUES ----------------------------------------------------------

    #[test]
    fn default_values_is_one_empty_row_over_an_empty_target_list() {
        // `t` declares no defaults, so DEFAULT VALUES is still an empty target list over
        // a single zero-width row (every column left for the executor to NULL-fill).
        let plan = plan_ins("INSERT INTO t DEFAULT VALUES").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.column_count, 2);
        assert_eq!(ins.column_affinities.len(), 2);
        assert_eq!(ins.columns, Some(Vec::new()), "empty target list → all columns default");
        assert_eq!(ins.source_width, 0);
        match ins.source.as_ref() {
            PlanNode::Values { rows } => {
                assert_eq!(rows.len(), 1, "exactly one row");
                assert!(rows[0].is_empty(), "zero-width row");
            }
            other => panic!("expected Values source, got {other:?}"),
        }
    }

    // ---- column DEFAULT substitution (GOAL 1) ------------------------------------

    /// The `Values` rows of an INSERT source, or panic if the source is another node.
    fn values_rows(ins: &Insert) -> &[Vec<EvalExpr>] {
        match ins.source.as_ref() {
            PlanNode::Values { rows } => rows,
            other => panic!("expected Values source, got {other:?}"),
        }
    }

    #[test]
    fn partial_column_list_injects_omitted_default() {
        // d(a INTEGER, b TEXT DEFAULT 'zz'): omitting b appends its bound default and
        // extends the target list, so the executor stores b='zz' rather than NULL.
        let plan = plan_ins("INSERT INTO d(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "b (index 1) appended after a");
        assert_eq!(ins.source_width, 2, "width grew by the injected default");
        let rows = values_rows(ins);
        assert_eq!(rows.len(), 1);
        assert!(matches!(rows[0][0], EvalExpr::Literal(Value::Integer(1))), "a = 1");
        assert!(
            matches!(&rows[0][1], EvalExpr::Literal(Value::Text(s)) if s == "zz"),
            "b = its DEFAULT 'zz', got {:?}",
            rows[0][1]
        );
    }

    #[test]
    fn partial_column_list_without_default_stays_unset() {
        // t(a INTEGER, b TEXT): b has no default, so it is left OUT of the target list
        // (unset → executor NULLs it), exactly as before this feature.
        let plan = plan_ins("INSERT INTO t(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0]), "b not injected (no default)");
        assert_eq!(ins.source_width, 1);
        assert_eq!(values_rows(ins)[0].len(), 1, "no default appended");
    }

    #[test]
    fn multi_row_values_inject_default_into_every_row() {
        // The default is appended to EACH compiled Values row.
        let plan = plan_ins("INSERT INTO d(a) VALUES (1), (2)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]));
        assert_eq!(ins.source_width, 2);
        let rows = values_rows(ins);
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(row.len(), 2, "each row carries a and the injected default");
            assert!(matches!(&row[1], EvalExpr::Literal(Value::Text(s)) if s == "zz"));
        }
    }

    #[test]
    fn partial_list_mixes_injected_default_and_unset_no_default() {
        // d3(a, b DEFAULT 'zz', c): omitting BOTH b (has a default) and c (none) in one
        // statement injects ONLY b; c stays out of `columns` so the executor NULLs it.
        // This pins the per-column decision inside the injection loop (inject vs leave
        // unset), which the single-omitted-column tests can't distinguish.
        let plan = plan_ins("INSERT INTO d3(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "b (default) injected; c (no default) left unset");
        assert_eq!(ins.source_width, 2);
        let rows = values_rows(ins);
        assert_eq!(rows.len(), 1);
        assert!(matches!(rows[0][0], EvalExpr::Literal(Value::Integer(1))), "a = 1");
        assert!(matches!(&rows[0][1], EvalExpr::Literal(Value::Text(s)) if s == "zz"), "b default");
    }

    #[test]
    fn keyword_literal_default_binds_to_a_constant_node() {
        // A keyword-literal default (`CURRENT_TIMESTAMP`) is stored as bare text and must
        // parse+bind (against the empty scope) to a constant Now node — not error at plan
        // time and not resolve as a column reference. Exercises the literal path the
        // number/string default tests skip.
        let plan = plan_ins("INSERT INTO dt(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "the omitted CURRENT_TIMESTAMP default is injected");
        let rows = values_rows(ins);
        assert!(
            matches!(rows[0][1], EvalExpr::Now(_)),
            "b default binds to a constant Now node, got {:?}",
            rows[0][1]
        );
    }

    #[test]
    fn real_literal_default_binds_to_a_real_literal() {
        // dr(a INTEGER, b REAL DEFAULT 3.14): a REAL literal default must parse+bind to a
        // Real constant — the float path the INTEGER / TEXT / CURRENT_TIMESTAMP default
        // tests don't exercise.
        let plan = plan_ins("INSERT INTO dr(a) VALUES (1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "the omitted REAL default is injected");
        let rows = values_rows(ins);
        assert!(
            matches!(&rows[0][1], EvalExpr::Literal(Value::Real(r)) if (*r - 3.14).abs() < 1e-9),
            "b default binds to Real(3.14), got {:?}",
            rows[0][1]
        );
    }

    #[test]
    fn default_values_injects_all_declared_defaults() {
        // dd(a INTEGER DEFAULT 5, b TEXT DEFAULT 'z'): DEFAULT VALUES emits one row of
        // both bound defaults over the full target list.
        let plan = plan_ins("INSERT INTO dd DEFAULT VALUES").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]));
        assert_eq!(ins.source_width, 2);
        let rows = values_rows(ins);
        assert_eq!(rows.len(), 1);
        assert!(matches!(rows[0][0], EvalExpr::Literal(Value::Integer(5))), "a = 5");
        assert!(matches!(&rows[0][1], EvalExpr::Literal(Value::Text(s)) if s == "z"), "b = 'z'");
    }

    #[test]
    fn insert_select_wraps_source_in_project_with_default() {
        // INSERT INTO d(a) SELECT x FROM s: the SELECT (width 1) is wrapped in a Project
        // that passes its column through then appends b's default constant.
        let plan = plan_ins("INSERT INTO d(a) SELECT x FROM s").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]));
        assert_eq!(ins.source_width, 2);
        match ins.source.as_ref() {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 2);
                assert!(matches!(exprs[0], EvalExpr::Column(0)), "pass SELECT col 0 through");
                assert!(
                    matches!(&exprs[1], EvalExpr::Literal(Value::Text(s)) if s == "zz"),
                    "append b's default"
                );
            }
            other => panic!("expected Project source, got {other:?}"),
        }
    }

    #[test]
    fn insert_select_all_columns_supplied_injects_nothing() {
        // Both columns supplied → nothing omitted → no default appended. The SELECT body
        // itself compiles to a Project of its two columns, so the distinguishing fact is
        // that no THIRD (default) expr was appended and the width stayed 2.
        let plan = plan_ins("INSERT INTO d(a, b) SELECT x, y FROM s").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "both columns supplied, none appended");
        assert_eq!(ins.source_width, 2, "width unchanged (no default injected)");
        // Hard-match the source shape: a `SELECT a, b` compiles to a two-expr Project, and
        // with nothing omitted NO third default const is appended. A `match … else panic`
        // (not an `if let`) so a shape regression fails loudly instead of skipping the check.
        match ins.source.as_ref() {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 2, "no injected default constant appended to the projection");
            }
            other => panic!("expected a Project source for INSERT...SELECT, got {other:?}"),
        }
    }

    #[test]
    fn positional_insert_never_injects_defaults() {
        // A positional insert (no column list) supplies every column, so d's default is
        // not consulted and the source stays exactly the user's VALUES row.
        let plan = plan_ins("INSERT INTO d VALUES (1, 'x')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, None);
        assert_eq!(ins.source_width, 2);
        assert_eq!(values_rows(ins)[0].len(), 2);
    }

    #[test]
    fn rowid_alias_default_is_ignored_so_rowid_auto_assigns() {
        // kd(id INTEGER PRIMARY KEY DEFAULT 9, v TEXT): omitting the alias must NOT inject
        // its DEFAULT 9 — the executor auto-assigns the rowid, so id is left unset.
        let plan = plan_ins("INSERT INTO kd(v) VALUES ('a')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![1]), "only v; the alias id is left unset");
        assert_eq!(ins.source_width, 1);
        assert_eq!(values_rows(ins)[0].len(), 1, "no default appended for the alias");
    }

    #[test]
    fn non_constant_column_default_is_a_loud_error() {
        // dbad(a INTEGER, b TEXT DEFAULT a): b's default references column `a`, so it is
        // not a constant. Column defaults are bound against the EMPTY scope (a default
        // sees no row), so the reference resolves to nothing and the injection fails
        // LOUD — never a silent NULL substitution or a mis-bound register. Pins the
        // "do NOT silently substitute NULL for a real default" invariant. (The
        // real catalog renders only literal defaults to text, so this exact input does
        // not arise from it today; this guards the injector if a non-literal default
        // text ever reaches it.)
        let err = plan_ins("INSERT INTO dbad(a) VALUES (1)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such column: a"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- rowid / _rowid_ / oid in the target column list (GOAL 2) -----------------

    #[test]
    fn rowid_keyword_maps_to_the_alias_column() {
        // k(id INTEGER PRIMARY KEY, v): `rowid` names the alias column (index 0), so the
        // executor treats the supplied value as the explicit rowid.
        let plan = plan_ins("INSERT INTO k(rowid, v) VALUES (42, 'a')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "rowid → alias index 0, v → 1");
        assert_eq!(ins.source_width, 2);
    }

    #[test]
    fn oid_and_underscore_rowid_also_map_to_the_alias() {
        let plan = plan_ins("INSERT INTO k(oid, v) VALUES (7, 'a')").unwrap();
        assert_eq!(expect_insert(&plan.root).columns, Some(vec![0, 1]), "oid → alias index 0");
        // `_rowid_` in a non-leading position maps the same and preserves list order.
        let plan = plan_ins("INSERT INTO k(v, _rowid_) VALUES ('a', 7)").unwrap();
        assert_eq!(expect_insert(&plan.root).columns, Some(vec![1, 0]), "_rowid_ → alias index 0");
    }

    #[test]
    fn rowid_keyword_is_case_insensitive() {
        let plan = plan_ins("INSERT INTO k(RoWiD, v) VALUES (1, 'a')").unwrap();
        assert_eq!(expect_insert(&plan.root).columns, Some(vec![0, 1]));
    }

    #[test]
    fn real_column_named_rowid_shadows_the_keyword() {
        // tr(rowid TEXT, v INTEGER): `rowid` is a REAL user column, so the target list must
        // resolve it to THAT column (index 0), never the rowid slot. Pins the shadowing
        // ORDER in resolve_target_columns (a named column is resolved before the rowid
        // keyword); reversing it would route `rowid` to the alias/rowid path — a loud gap on
        // this plain table — instead of mapping to the real column.
        let plan = plan_ins("INSERT INTO tr(rowid) VALUES ('x')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0]), "the real `rowid` column shadows the keyword");
        assert_eq!(ins.source_width, 1, "v has no default, so nothing else is injected");
    }

    #[test]
    fn duplicate_rowid_and_alias_name_is_a_loud_error() {
        // `id` and `rowid` are the same column on k, so listing both is a duplicate.
        let err = plan_ins("INSERT INTO k(id, rowid) VALUES (1, 2)").unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("duplicate column name: rowid"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn explicit_rowid_on_a_plain_rowid_table_maps_to_the_rowid_channel() {
        // t(a INTEGER, b TEXT) has no INTEGER PRIMARY KEY, so `rowid` in the target list
        // names the hidden rowid: it maps to the SENTINEL index N (=2) in `columns`, and
        // `rowid_source` points at that position so the executor reads the supplied value (5)
        // as the rowid rather than storing it as a column. (b is omitted with no default → NULL.)
        let plan = plan_ins("INSERT INTO t(rowid, a) VALUES (5, 1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.column_count, 2);
        assert_eq!(ins.columns, Some(vec![2, 0]), "rowid → sentinel N=2, a → 0");
        assert_eq!(ins.rowid_source, Some(0), "the rowid value is source position 0");
        assert_eq!(ins.source_width, 2);
    }

    #[test]
    fn plain_insert_has_no_explicit_rowid_source() {
        assert_eq!(expect_insert(&plan_ins("INSERT INTO t VALUES (1, 'x')").unwrap().root).rowid_source, None);
    }

    #[test]
    fn rowid_keyword_on_ipk_table_uses_alias_not_the_explicit_channel() {
        // k(id INTEGER PRIMARY KEY, v): `rowid` maps to the alias column (index 0), NOT the
        // sentinel; rowid_source stays None (the executor derives the rowid from the alias).
        let plan = plan_ins("INSERT INTO k(rowid, v) VALUES (42, 'a')").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 1]), "rowid → alias index 0");
        assert_eq!(ins.rowid_source, None, "alias path, not the explicit-rowid channel");
    }

    #[test]
    fn explicit_rowid_in_a_non_leading_position_tracks_its_source_index() {
        // `rowid` LAST in the target list → sentinel N=2 lands at position 1, so rowid_source
        // must be Some(1), NOT Some(0). This is the case a hardcoded `Some(0)` would fail:
        // it pins that rowid_source is the ACTUAL `.position(|&t| t == N)` of the sentinel.
        let plan = plan_ins("INSERT INTO t(a, rowid) VALUES (1, 5)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 2]), "a → 0, rowid → sentinel N=2");
        assert_eq!(ins.rowid_source, Some(1), "rowid value is source position 1, not 0");
        assert_eq!(ins.source_width, 2);
    }

    #[test]
    fn explicit_rowid_with_injected_default_stays_correct_after_columns_grow() {
        // d(a INTEGER, b TEXT DEFAULT 'zz'): `INSERT INTO d(a, rowid)` omits b, so its default
        // is APPENDED to `columns` (→ [0, 2, 1]) AFTER the sentinel is already in place.
        // rowid_source is computed from the FINAL columns, so it correctly stays at the
        // sentinel's position (1) rather than being shifted by the appended default.
        let plan = plan_ins("INSERT INTO d(a, rowid) VALUES (1, 5)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 2, 1]), "a → 0, rowid → sentinel 2, b default → 1");
        assert_eq!(ins.rowid_source, Some(1), "sentinel still at position 1 after the default append");
        assert_eq!(ins.source_width, 3, "grew by the injected b default");
        // The injected b default rides in the trailing slot; the rowid value (5) is untouched.
        assert!(
            matches!(&values_rows(ins)[0][2], EvalExpr::Literal(Value::Text(s)) if s == "zz"),
            "b's default 'zz' is appended, got {:?}",
            values_rows(ins)[0][2]
        );
    }

    #[test]
    fn oid_and_underscore_rowid_on_a_plain_table_map_to_the_sentinel() {
        // The `_rowid_` / `oid` spellings route to the plain-table sentinel channel too (only
        // `rowid` itself is otherwise exercised on a plain table).
        let plan = plan_ins("INSERT INTO t(oid, a) VALUES (5, 1)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![2, 0]), "oid → sentinel N=2");
        assert_eq!(ins.rowid_source, Some(0));
        let plan = plan_ins("INSERT INTO t(a, _rowid_) VALUES (1, 5)").unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.columns, Some(vec![0, 2]), "_rowid_ → sentinel N=2");
        assert_eq!(ins.rowid_source, Some(1));
    }

    #[test]
    fn explicit_rowid_on_a_without_rowid_table_is_no_such_column() {
        // A WITHOUT ROWID table has NO rowid, so `rowid`/`_rowid_`/`oid` are not special —
        // they are plain unknown column names (the loud error real SQLite reports), and must
        // NOT route through the plain-rowid sentinel channel. `rowid_alias == None` alone does
        // not imply a plain rowid table; `without_rowid` is what distinguishes them.
        let w = TableDef {
            name: "w".to_string(),
            columns: vec![col("a", Some("INTEGER")), col("b", Some("TEXT"))],
            root_page: 2,
            without_rowid: true,
            rowid_alias: None,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        };
        let err = plan_against("INSERT INTO w(rowid, a) VALUES (5, 1)", w).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("table w has no column named rowid"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    // ---- CHECK constraints compiled onto the node (compile::check) ----------------

    /// A `TableDef` carrying CHECK predicates parsed from `check_texts` (each a scalar
    /// expression like `"x > 0"`). In production the catalog builder populates `checks`;
    /// the test catalog is static, so the def is hand-built and each predicate is parsed
    /// via the module's `SELECT <expr>` wrapper (`parse_default_expr`, which parses an
    /// arbitrary scalar expression — a CHECK binds the same way a DEFAULT does).
    fn tdef_ck(
        name: &str,
        columns: Vec<ColumnDef>,
        rowid_alias: Option<usize>,
        check_texts: &[&str],
    ) -> TableDef {
        let checks =
            check_texts.iter().map(|t| parse_default_expr(t).expect("check predicate parses")).collect();
        TableDef {
            name: name.to_string(),
            columns,
            root_page: 2,
            without_rowid: false,
            rowid_alias,
            auto_indexes: Vec::new(),
            checks,
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        }
    }

    /// Plan `sql` against a one-table catalog holding `table` — for the CHECK tests, which
    /// need a table whose `checks` are populated (the shared `cat()` tables carry none).
    fn plan_against(sql: &str, table: TableDef) -> Result<Plan> {
        let c = TestCatalog::new(vec![table]);
        let ast = parse(sql)?;
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c)
    }

    #[test]
    fn insert_compiles_check_constraint_onto_the_node() {
        // tc(x INTEGER CHECK(x > 0)): the planner binds the predicate against the inserted
        // row `[x, rowid]` and carries it on the node for the executor. `x` resolves to
        // register 0 and the detail is the table name.
        let tc = tdef_ck("tc", vec![col("x", Some("INTEGER"))], None, &["x > 0"]);
        let plan = plan_against("INSERT INTO tc VALUES (5)", tc).unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.checks.len(), 1, "the one CHECK is compiled onto the node");
        assert_eq!(ins.checks[0].detail, "tc", "detail is the table name");
        match &ins.checks[0].expr {
            EvalExpr::Compare { left, .. } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(0)), "x -> reg 0, got {left:?}");
            }
            other => panic!("expected a Compare node for `x > 0`, got {other:?}"),
        }
    }

    #[test]
    fn insert_compiles_multiple_checks_in_stored_order() {
        // Two checks over DISTINCT columns ride onto the node in the catalog's STORED order:
        // check[0] is the `x` predicate (binds `Column(0)`), check[1] the `y` predicate
        // (`Column(1)`). Column-level then table-level checks are collected in declaration
        // order and `compile_checks` preserves that `Vec` order — deterministic and stable.
        // (SQLite doesn't make the order observable for correctness, but a reorder here would
        // still be a regression against that stated intent, so this pins it via the operands.)
        let tc = tdef_ck(
            "tc",
            vec![col("x", Some("INTEGER")), col("y", Some("INTEGER"))],
            None,
            &["x > 0", "y > 0"],
        );
        let plan = plan_against("INSERT INTO tc VALUES (5, 6)", tc).unwrap();
        let checks = &expect_insert(&plan.root).checks;
        assert_eq!(checks.len(), 2);
        match (&checks[0].expr, &checks[1].expr) {
            (EvalExpr::Compare { left: l0, .. }, EvalExpr::Compare { left: l1, .. }) => {
                assert!(matches!(l0.as_ref(), EvalExpr::Column(0)), "check[0] is the `x` predicate");
                assert!(matches!(l1.as_ref(), EvalExpr::Column(1)), "check[1] is the `y` predicate");
            }
            other => panic!("expected two Compare nodes, got {other:?}"),
        }
    }

    #[test]
    fn insert_into_table_without_checks_carries_no_checks() {
        // A table with no CHECK carries an empty `checks` vec on its INSERT node — the
        // common case must not fabricate a predicate.
        let plan = plan_ins("INSERT INTO t VALUES (1, 'x')").unwrap();
        assert!(expect_insert(&plan.root).checks.is_empty());
    }

    #[test]
    fn insert_check_referencing_unknown_column_is_a_bind_error() {
        // A stored CHECK that names a column the table lacks fails to BIND at plan time —
        // a loud `no such column` error, matching real sqlite rejecting such a CREATE
        // TABLE (our architecture binds the predicate at plan time). Never silently dropped.
        let tc = tdef_ck("tc", vec![col("x", Some("INTEGER"))], None, &["y > 0"]);
        let err = plan_against("INSERT INTO tc VALUES (5)", tc).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such column: y"), "got {m:?}"),
            other => panic!("expected Sql error, got {other:?}"),
        }
    }

    #[test]
    fn insert_check_on_integer_primary_key_alias_binds_to_rowid_register() {
        // kc(id INTEGER PRIMARY KEY CHECK(id > 0)): the alias column resolves to the rowid
        // register N (=1 here) — the same `[c0..c_{N-1}, rowid]` layout the executor's row
        // uses — so a check on the rowid-alias column reads the right slot, not a NULL
        // schema slot.
        let kc = tdef_ck("kc", vec![col("id", Some("INTEGER"))], Some(0), &["id > 0"]);
        let plan = plan_against("INSERT INTO kc VALUES (5)", kc).unwrap();
        let ins = expect_insert(&plan.root);
        assert_eq!(ins.checks.len(), 1);
        match &ins.checks[0].expr {
            EvalExpr::Compare { left, .. } => assert!(
                matches!(left.as_ref(), EvalExpr::Column(1)),
                "id alias -> rowid reg N=1, got {left:?}"
            ),
            other => panic!("expected Compare, got {other:?}"),
        }
    }
}
