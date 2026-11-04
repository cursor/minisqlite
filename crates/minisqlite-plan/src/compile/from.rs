//! FROM-clause compilation, in two phases so an `ON`/`USING` predicate can bind
//! against the *whole* join's [`Scope`] (which needs every source's register offset,
//! which in turn needs the entire FROM walked first):
//!
//! * [`resolve_from`] (phase 1) walks the join tree and returns the [`Source`] list —
//!   the columns visible to the rest of the statement — with each source's `base`
//!   register offset. The caller builds the [`Scope`] from these and binds `WHERE`.
//! * [`build_from`] (phase 2) builds the access/join operator subtree: a constant
//!   `SingleRow` (no FROM), a single base table's access path, or a left-deep
//!   [`PlanNode::Join`] tree, binding each join's `ON` against that scope.
//!
//! # Scope: base tables and derived tables (subqueries in FROM)
//!
//! This handles base tables and `FROM (SELECT …) [AS] alias` derived tables, joined by
//! comma joins and every `JOIN` operator. A base table resolves to a catalog
//! [`crate::bind::Source::BaseTable`] and its rowid-aware access path; a derived table
//! resolves to a [`crate::bind::Source::Derived`] (its output schema computed in Phase
//! 1 without compiling) whose access leaf is a materialized `CtePlan` scanned by a
//! [`PlanNode::CteScan`] — width `column_count`, NO trailing rowid (the executor's
//! `ops::cte` contract). A derived *subquery* table is NON-correlated (SQLite has no
//! LATERAL for those), so its inner SELECT compiles at base 0 with no parent scope.
//!
//! # Table-valued functions (`json_each` / `json_tree`, `pragma_*`)
//!
//! A `json_each(X[,P])` / `json_tree(X[,P])` FROM function (json1.html §4.24) also
//! resolves to a [`crate::bind::Source::Derived`], but with a FIXED schema (eight visible
//! columns plus the hidden `json`/`root` input columns) ([`json_table_columns`]); its
//! access leaf is a [`PlanNode::TableFunctionScan`]
//! (not a `CtePlan`) whose argument expressions are bound in Phase 2. Unlike a subquery
//! derived table these ARE implicitly LATERAL: the argument may reference a preceding
//! table's columns, so when the function is a join's right operand the planner forces
//! [`JoinStrategy::IndexNestedLoop`] and the executor re-evaluates the argument per left
//! row.
//!
//! The `pragma_*` schema-introspection FROM functions (`pragma_table_info('t')` and the
//! rest, pragma.html §2) work the same way but carry each PRAGMA's fixed columns and lower
//! to a [`PlanNode::PragmaFunctionScan`]; their leaf/schema/argument handling lives in
//! [`crate::compile::pragma_tvf`], keeping this hot file thin. Any other function name in
//! FROM stays a loud "no such table-valued function" error.
//!
//! # Parenthesized joins (`FROM (a JOIN b …) …`)
//!
//! A parenthesized `( join-clause )` is a `table-or-subquery` that is itself a whole
//! sub-join, so a single FROM leaf can contribute MANY sources. [`resolve_leaf`] handles
//! it in both phases by recursing into the bracketed [`JoinTree`]: Phase 1 places the
//! sub-join's sources at the same flat left-to-right bases (parenthesization changes only
//! the tree SHAPE, never the register order), and Phase 2 ([`build_subjoin`]) builds it
//! as a self-contained [`PlanNode::Join`] subtree in its OWN local register space (its
//! leftmost leaf at register 0), which is exactly what the executor evaluates it in when
//! it uses the sub-join's rows, whole, as one operand of the surrounding join. That local
//! 0-based space composes with the enclosing join only when the executor runs the sub-join
//! with an EMPTY outer, which holds at `base_offset == 0` but not under a correlated /
//! trigger outer, so [`build_subjoin`] rejects a nonzero `base_offset` loudly (a precise
//! residual gap). A FLAT correlated / trigger join — the common shape — IS supported: its
//! leaves sit directly at `base_offset`, `base_offset + WL`, … and the executor prepends
//! the outer prefix exactly once (see `minisqlite-exec`'s `ops::join`).
//!
//! # Register layout (the shared ROW/REGISTER convention, see [`crate::plan`])
//!
//! A base table with `N` columns occupies `N+1` registers — `[c0, …, c_{N-1}, rowid]`.
//! In a join the sources are laid out left to right: the first at `base_offset`, each
//! subsequent one at `base_offset +` the combined width so far. A `Join` node emits
//! `left ++ right`, so its `left_width` is exactly the combined width of everything to
//! its left and its `on` predicate — bound through the scope — reads the same
//! registers the executor materializes. `base_offset` is `0` for a top-level query and
//! non-zero for a correlated subquery / trigger action, where it places the subquery's own
//! sources after the outer (`OLD ++ NEW`) row that the executor prepends at each FROM leaf.

use minisqlite_catalog::Catalog;
use minisqlite_expr::{CmpOp, EvalExpr};
use minisqlite_functions::JsonTableKind;
use minisqlite_sql::{
    BinaryOp, Expr, FromClause, JoinConstraint, JoinKind, JoinOperator, JoinTree, QualifiedName,
    ResultColumn, Select, SelectBody, SelectCore, TableOrSubquery,
};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, DbIndex, Error, Result};

use crate::access::{RowidOp, RowidScan, ScanDirection, SeqScan};
use crate::access_path::plan_table_access;
use crate::bind::expr::operand_affinity;
use crate::bind::scope::SynthCol;
use crate::bind::{best_effort_collation, bind_expr, Coalesced, ResolvedColumn, Scope, Source};
use crate::colname::result_column_name;
use crate::plan::{CtePlan, Join, JoinStrategy, JoinType, PlanNode};
use crate::plan_ctx::PlanCtx;

/// The register width a source contributes to the combined row: a base table is `N+1`
/// (`[c0, …, c_{N-1}, rowid]`); a derived table is `columns.len()` (no rowid).
/// Delegates to the single source of truth, [`Source::width`].
fn table_width(source: &Source) -> usize {
    source.width()
}

// ---------------------------------------------------------------------------
// Phase 1: resolve the FROM tree into the visible sources.
// ---------------------------------------------------------------------------

/// Walk the FROM join tree and return one [`Source`] per FROM leaf (a base table or a
/// derived table), in left-to-right order, each carrying the register offset (`base`)
/// where its columns begin.
///
/// `base_offset` is where the first source is placed (0 for a top-level query). The
/// `'a` catalog lifetime flows into each `BaseTable` source's borrowed `&TableDef` (a
/// `Derived` source borrows nothing). A no-FROM query yields an empty source list.
///
/// Alongside the sources it returns the NATURAL/USING [`Coalesced`] columns: each shared
/// join column appears once in the join output, so the caller's [`Scope`] resolves a bare
/// shared name to `COALESCE(left, right)` and drops the right copy from `*`. An `ON` join
/// (or no join) contributes none. Each [`Coalesced`] register is absolute — it already
/// carries `base_offset`, matching the combined row the executor materializes.
pub fn resolve_from<'a>(
    catalog: &'a dyn Catalog,
    from: &Option<FromClause>,
    base_offset: usize,
) -> Result<(Vec<Source<'a>>, Vec<Coalesced>)> {
    let mut sources = Vec::new();
    let mut coalesced = Vec::new();
    if let Some(tree) = from {
        resolve_tree(catalog, tree, base_offset, &mut sources, &mut coalesced)?;
    }
    Ok((sources, coalesced))
}

/// Append a [`Source`] for every FROM leaf in `tree` (a base/derived table, or — via
/// [`resolve_leaf`], recursively — every leaf of a parenthesized sub-join), placing the
/// first at `base_offset` and each subsequent one after the running combined width, and
/// append a [`Coalesced`] entry for each shared column of a USING/NATURAL join
/// encountered. Returns the combined register width of the whole subtree.
///
/// A parenthesization changes only the join tree SHAPE (which `ON` binds where, and the
/// NULL-extension grouping for outer joins); it does NOT change this flat left-to-right
/// register order of the sources.
fn resolve_tree<'a>(
    catalog: &'a dyn Catalog,
    tree: &JoinTree,
    base_offset: usize,
    sources: &mut Vec<Source<'a>>,
    coalesced: &mut Vec<Coalesced>,
) -> Result<usize> {
    match tree {
        JoinTree::Table(tos) => resolve_leaf(catalog, tos, base_offset, sources, coalesced),
        // `left` is a subtree (recurse); `right` is ONE leaf — a single table/subquery, or
        // a parenthesized sub-join contributing MULTIPLE sources ([`resolve_leaf`]) —
        // placed right after the left subtree's combined width.
        JoinTree::Join { left, op, right, constraint } => {
            let left_start = sources.len();
            let left_width = resolve_tree(catalog, left, base_offset, sources, coalesced)?;
            let right_start = sources.len();
            let right_width =
                resolve_leaf(catalog, right, base_offset + left_width, sources, coalesced)?;
            // Record this join's USING/NATURAL shared columns. `left_sources` is everything
            // the left subtree added (a shared name may live in any of them); `right_sources`
            // is everything the right operand added — one leaf, or the whole parenthesized
            // sub-join. Phase 2 (`build_on`) still builds the row-filtering ON equality; this
            // is purely the name-resolution / star half of the coalesce.
            collect_coalesced(
                op,
                constraint.as_ref(),
                &sources[left_start..right_start],
                &sources[right_start..],
                coalesced,
            )?;
            Ok(left_width + right_width)
        }
    }
}

/// Resolve one FROM leaf (a `table-or-subquery`) at register offset `base`, appending its
/// [`Source`](s) and any USING/NATURAL [`Coalesced`] columns, and returning the combined
/// register width it contributes (so the enclosing join places later sources after it):
///
/// * a plain base/derived table (or CTE/view) → one [`Source`] (via [`resolve_table`]);
/// * a **parenthesized join** `( join-clause )` → the WHOLE sub-join, recursing into
///   [`resolve_tree`] so each of its sources lands at the same running left-to-right base
///   (`base +` the width so far) the flat case would give, and its own NATURAL/USING
///   coalesced columns are collected too.
///
/// A parenthesization only re-shapes the join tree; it never changes this flat register
/// order (see [`resolve_tree`]).
fn resolve_leaf<'a>(
    catalog: &'a dyn Catalog,
    tos: &TableOrSubquery,
    base_offset: usize,
    sources: &mut Vec<Source<'a>>,
    coalesced: &mut Vec<Coalesced>,
) -> Result<usize> {
    match tos {
        TableOrSubquery::Join(jt) => resolve_tree(catalog, jt, base_offset, sources, coalesced),
        _ => {
            let src = resolve_table(catalog, tos, base_offset)?;
            let width = table_width(&src);
            sources.push(src);
            Ok(width)
        }
    }
}

/// Record the [`Coalesced`] entries for a USING / NATURAL join: one per shared column,
/// pairing the left side's register with the right side's, with the LEFT column's
/// affinity/collation (the coalesced value is `COALESCE(left, right)`, left-preferred).
/// The shared names are the explicit `USING (cols)` list, or — for NATURAL — every name
/// common to both sides ([`natural_columns`]); an explicit `ON` (or no constraint)
/// coalesces nothing. A `USING` name absent from a side is left for Phase 2's
/// [`build_using`] to report as a loud error, so a non-shared name is simply skipped.
///
/// `right_sources` is a SLICE, not a single source: the right operand of a join can be a
/// parenthesized sub-join contributing several sources, and a shared column is matched
/// against the first one that declares it (the join-side counterpart of an unqualified
/// lookup). A plain right leaf is just a one-element slice.
fn collect_coalesced(
    op: &JoinOperator,
    constraint: Option<&JoinConstraint>,
    left_sources: &[Source],
    right_sources: &[Source],
    coalesced: &mut Vec<Coalesced>,
) -> Result<()> {
    let names = if op.natural {
        natural_columns(left_sources, right_sources)
    } else if let Some(JoinConstraint::Using(cols)) = constraint {
        cols.clone()
    } else {
        return Ok(());
    };
    for name in &names {
        let (Some(left), Some(right)) =
            (column_in_sources(left_sources, name)?, column_in_sources(right_sources, name)?)
        else {
            continue;
        };
        coalesced.push(Coalesced {
            name: name.clone(),
            left: left.reg,
            right: right.reg,
            affinity: left.affinity,
            collation: left.collation,
        });
    }
    Ok(())
}

/// The first source in `sources` that resolves the real column `name`, as a
/// [`ResolvedColumn`] (register + affinity + collation), or `None` if none declares it.
/// The join-side counterpart of the scope's unqualified lookup, used to capture each
/// side's copy of a shared USING/NATURAL column when building [`Coalesced`] entries.
fn column_in_sources(sources: &[Source], name: &str) -> Result<Option<ResolvedColumn>> {
    for src in sources {
        if let Some(rc) = src.column(name)? {
            return Ok(Some(rc));
        }
    }
    Ok(None)
}

/// Resolve one SINGLE FROM leaf into a [`Source`] at register offset `base`. A base table
/// borrows its catalog schema; a `FROM (SELECT …)` subquery becomes a
/// [`Source::Derived`] whose output schema is computed WITHOUT compiling the inner plan
/// (Phase 1 is ctx-free — see [`derived_schema`]). A JSON table-valued function
/// (`json_each`/`json_tree`) also becomes a [`Source::Derived`], with the fixed schema
/// (eight visible columns plus the hidden `json`/`root`) ([`json_table_columns`]); any
/// other function name is a loud error. A
/// parenthesized join is routed through [`resolve_leaf`] BEFORE this and never reaches
/// here (its arm is a defensive internal error). The schema qualifier is ignored (single
/// database) and `INDEXED BY` / `NOT INDEXED` hints are honored by the access-path layer,
/// not here.
fn resolve_table<'a>(
    catalog: &'a dyn Catalog,
    tos: &TableOrSubquery,
    base: usize,
) -> Result<Source<'a>> {
    match tos {
        TableOrSubquery::Table { name, alias, indexed: _ } => {
            // A common table expression from an enclosing `WITH` shadows a base table of
            // the same name and is matched ONLY by an unqualified reference (a
            // schema-qualified `db.name` is always a real table). It resolves to a
            // derived source carrying the CTE's synthetic schema; its access leaf (a
            // `CteScan`, or a `RecursiveScan` inside the recursive step) is emitted in
            // Phase 2 by [`build_source_leaf`] from the SAME CTE-scope entry, so both
            // phases agree on the width and the pre-registered id.
            if name.schema.is_none() {
                if let Some(cte) = crate::compile::cte::lookup_cte(&name.name) {
                    let exposed_name = alias.clone().unwrap_or_else(|| name.name.clone());
                    return Ok(Source::derived(exposed_name, cte.columns, base));
                }
            }
            // Resolve WHICH namespace this reference names, then fetch the def FROM that
            // namespace. A schema qualifier (`main`/`temp`/`temporary`) maps to its fixed
            // `DbIndex`; an unqualified name resolves in SQLite SEARCH ORDER (temp shadows
            // main, first match wins). An unknown qualifier, or a name absent from every
            // live namespace, is "no such table". The resolved `db` is stamped on the plan
            // node so the executor opens the cursor on the RIGHT store — a def and the pager
            // it is read through must agree on the namespace, or the root page is read from
            // the wrong store (the whole point of shadowing/qualification).
            let Some(db) = resolve_ref_db(catalog, name)? else {
                return Err(Error::sql(format!("no such table: {}", qualified_table_name(name))));
            };
            if let Some(table) = catalog.table_in(db, &name.name)? {
                let exposed_name = alias.clone().unwrap_or_else(|| name.name.clone());
                return Ok(Source::BaseTable { exposed_name, table, db, base });
            }
            // Not a CTE and not a base table: a VIEW is expanded inline exactly like a
            // `FROM (SELECT …)` derived table. Its schema comes from the stored SELECT
            // (Phase 1, here); Phase 2's `build_source_leaf` compiles+materializes the
            // SAME SELECT into a `CtePlan`, so both phases agree on the width. The view is
            // fetched from the SAME resolved namespace (`view_in(db, …)`), and `view_schema`
            // guards against a circular definition rather than recursing forever. `columns`
            // is owned, so the borrowed `view` does not escape into the returned `Source`.
            if let Some(view) = catalog.view_in(db, &name.name)? {
                let exposed_name = alias.clone().unwrap_or_else(|| name.name.clone());
                let columns = crate::compile::view::view_schema(catalog, view)?;
                return Ok(Source::derived(exposed_name, columns, base));
            }
            Err(Error::sql(format!("no such table: {}", qualified_table_name(name))))
        }
        TableOrSubquery::Subquery { select, alias } => {
            // An unaliased derived table has no qualifier (empty exposed name); its
            // columns are still visible unqualified.
            let exposed_name = alias.clone().unwrap_or_default();
            let columns = derived_schema(catalog, select)?;
            Ok(Source::derived(exposed_name, columns, base))
        }
        TableOrSubquery::TableFunction { name, alias, args: _ } => {
            // A FROM-clause table-valued function resolves to a `Source::Derived` carrying a
            // FIXED schema — the same shape a derived table uses (no rowid, own `SynthCol`
            // list) — so the rest of the binder treats it like any other derived source. Its
            // access leaf is emitted in Phase 2 by [`build_source_leaf`] from the same
            // `TableFunction` node. Two families are recognized: the JSON TVFs
            // (`json_each`/`json_tree`, eight visible columns plus the hidden `json`/`root`
            // input columns) and the `pragma_*` introspection TVFs (each PRAGMA's fixed
            // columns; see [`crate::compile::pragma_tvf`]). The exposed name a qualified
            // reference matches is the FROM alias if given, else the function's own name
            // (SQLite allows `WHERE json_each.value …`). Any other name is a loud error.
            let columns = if json_table_kind(name).is_some() {
                json_table_columns()
            } else if let Some(kind) = crate::compile::pragma_tvf::classify(name) {
                crate::compile::pragma_tvf::pragma_columns(kind)
            } else {
                return Err(Error::sql(format!(
                    "no such table-valued function: {}",
                    name.name
                )));
            };
            let exposed_name = alias.clone().unwrap_or_else(|| name.name.clone());
            Ok(Source::derived(exposed_name, columns, base))
        }
        TableOrSubquery::Join(_) => Err(Error::sql(
            "internal error: a parenthesized join reached resolve_table; \
             it must be routed through resolve_leaf",
        )),
    }
}

/// Render a table reference for a "no such table" error the way SQLite does: a
/// schema-qualified reference keeps its qualifier (`no such table: temp.t`), an
/// unqualified one is the bare name (`no such table: t`). Shared with the DML compilers
/// (`insert`/`update`/`delete`), which resolve their target the same way.
pub(crate) fn qualified_table_name(name: &QualifiedName) -> String {
    match &name.schema {
        Some(schema) => format!("{}.{}", schema, name.name),
        None => name.name.clone(),
    }
}

/// The database namespace a table/view reference resolves to, or `None` if no live
/// namespace owns it (which the caller turns into "no such table").
///
/// This is the SINGLE resolution rule every reference site shares — the FROM binder
/// ([`resolve_table`]), its Phase-2 view re-fetch ([`build_source_leaf`]), and the DML
/// target compilers — so all agree on which store a name means:
/// * a schema qualifier maps to its [`DbIndex`] via [`Catalog::db_of_schema`] — the fixed
///   built-ins (`main`/`temp`/`temporary`) plus any ATTACH alias the catalog knows; an
///   unknown qualifier is `None`;
/// * an UNQUALIFIED `sqlite_master` / `sqlite_schema` names MAIN's schema table
///   unconditionally ([`is_main_schema_table`]) — NOT the temp store, even though temp is
///   searched first for user tables;
/// * every other UNQUALIFIED name resolves in SQLite search order (temp shadows main,
///   first match) via [`Catalog::owner_db`].
pub(crate) fn resolve_ref_db(
    catalog: &dyn Catalog,
    name: &QualifiedName,
) -> Result<Option<DbIndex>> {
    match name.schema.as_deref() {
        // A schema qualifier resolves through the catalog so an ATTACH alias (not just the
        // fixed `main`/`temp` built-ins) reaches its store: on a `MultiCatalog` this is the
        // alias-aware override; on a bare `SchemaCatalog` it is the built-ins-only default.
        Some(schema) => catalog.db_of_schema(schema),
        // The schema tables are namespace-specific NAMES in SQLite (schematab.html §2):
        // `sqlite_master`/`sqlite_schema` name MAIN's, `sqlite_temp_master`/
        // `sqlite_temp_schema` name TEMP's. But EVERY namespace's `SchemaCatalog` aliases
        // all four names to its own page-1 built-in, so the temp-first search order would
        // resolve an unqualified `sqlite_master` to the temp store once temp exists — and
        // `SELECT … FROM sqlite_master` would then list temp objects, which must be
        // INVISIBLE there. Pin `sqlite_master`/`sqlite_schema` to MAIN so temp objects
        // never leak into it; `temp.sqlite_master` (qualified) still reaches temp's table,
        // and `sqlite_temp_master`/`sqlite_temp_schema` resolve to temp via `owner_db`'s
        // ordinary search order (temp aliases them, and it is searched first).
        None if is_main_schema_table(&name.name) => Ok(Some(DbIndex::MAIN)),
        None => catalog.owner_db(&name.name),
    }
}

/// Whether `name` is one of MAIN's schema-table names — `sqlite_master` or its modern
/// synonym `sqlite_schema` (ASCII case-insensitive, the identifier folding SQLite uses).
/// These name MAIN's schema table wherever they appear unqualified, so an unqualified
/// reference to one always resolves to [`DbIndex::MAIN`] (see [`resolve_ref_db`]). The
/// temp-only names (`sqlite_temp_master`/`sqlite_temp_schema`) are deliberately absent:
/// they name the TEMP store and resolve through the ordinary search order.
fn is_main_schema_table(name: &str) -> bool {
    name.eq_ignore_ascii_case("sqlite_master") || name.eq_ignore_ascii_case("sqlite_schema")
}

/// Compute a derived table's output schema — each column's name, and its affinity +
/// collation per datatype3 §3.3 (a bare column ref inherits its column's affinity +
/// collation, a `CAST(_ AS T)` takes T's affinity, any other expression is NONE + BINARY
/// unless it carries an explicit `COLLATE`) — from the inner SELECT, **catalog-only, no
/// plan compilation** (so Phase 1 stays ctx-free and never touches `select.rs`).
///
/// The count and names MUST match what [`crate::compile::select::compile_select`] later
/// produces for the same SELECT, because the derived source's register width is what the
/// outer scope binds against; [`compile_derived`] asserts the two agree. The naming
/// mirrors the projection compiler exactly:
/// * a compound (`UNION`/…) takes its names from the leftmost arm (the `SetOp`
///   convention: result names come from the left input);
/// * a `VALUES` core names columns `column1`, `column2`, … ;
/// * a query core names each result column by its explicit alias, else the bare
///   column's name, else the rendered expression text ([`result_column_name`]), and
///   expands `*` / `table.*` against the inner FROM's catalog schemas — recursively, so
///   a derived table nested in the inner FROM resolves too. Affinity/collation flow
///   through that recursion for free: a bare reference to a nested derived column reads
///   that column's already-inherited [`SynthCol`] affinity/collation, so a chain of
///   subqueries carries the base column's affinity all the way out (§3.3 applied at each
///   layer).
///
/// It is CTE-aware transitively: an inner `*` over a CTE reference expands through
/// [`resolve_table`], which consults the active CTE scope. This is also how the CTE
/// compiler ([`crate::compile::cte`]) derives a CTE's own output schema before the
/// column-name list (`WITH t(a,b) AS …`) is applied.
///
/// A WITH-bearing inner SELECT (`FROM (WITH c AS … SELECT … FROM c)`) is handled here:
/// its own CTEs are registered (schema-only) for the duration of this computation so the
/// inner body can resolve them, mirroring the Phase-2 `compile_with` that follows. Without
/// this, Phase 1 would reject `FROM c` as an unknown table before Phase 2 ran.
pub(crate) fn derived_schema(catalog: &dyn Catalog, select: &Select) -> Result<Vec<SynthCol>> {
    // Register a WITH-bearing subquery's CTE schemas for this computation only (popped when
    // `_cte_guard` drops). Phase 2 re-registers them with real plan ids via `compile_with`.
    let _cte_guard = match &select.with {
        Some(with) => Some(crate::compile::cte::push_cte_schemas(catalog, with)?),
        None => None,
    };
    match leftmost_core(&select.body) {
        SelectCore::Values(rows) => {
            let width = rows
                .first()
                .ok_or_else(|| Error::sql("VALUES must have at least one row"))?
                .len();
            Ok((1..=width).map(|i| synth_col(format!("column{i}"))).collect())
        }
        SelectCore::Query { columns, from, .. } => {
            // Resolve the inner FROM's sources (catalog-only) so `*` / `table.*` expand
            // against the same columns Phase 2's projection will see. The coalesced list
            // makes an inner unqualified `*` over a NATURAL/USING join drop the right copy
            // of each shared column, so the derived table's width matches Phase 2 exactly.
            let (sources, coalesced) = resolve_from(catalog, from, 0)?;
            let scope = Scope {
                sources: &sources,
                coalesced: &coalesced,
                parent: None,
                grouping: None,
                saw_correlated: None,
                correlated_cols: None,
                nondeterministic: None,
                windowing: None,
            };
            let mut out = Vec::new();
            for col in columns {
                match col {
                    ResultColumn::Expr { expr, alias } => {
                        // datatype3 §3.3: the derived column's affinity/collation are the
                        // §3.2/§7.1 metadata of the inner result expression — a bare column
                        // ref inherits its column's, a CAST takes the cast type's, any other
                        // expression is NONE (Blob) + BINARY unless it carries an explicit
                        // COLLATE. The binder helpers compute exactly this; reuse them so the
                        // rule stays the single one the comparison path already applies.
                        // `operand_affinity` errors on an unresolvable/ambiguous column,
                        // surfacing that here (Phase 1) exactly as the Phase-2 projection
                        // compile would.
                        let name = alias.clone().unwrap_or_else(|| result_column_name(expr));
                        let affinity = operand_affinity(&scope, expr)?
                            .unwrap_or_else(|| affinity_of_declared_type(None));
                        let collation = best_effort_collation(&scope, expr);
                        out.push(synth_col_typed(name, affinity, collation));
                    }
                    ResultColumn::Star => {
                        // Each `*` column is a bare column reference, so it inherits its
                        // source column's affinity + collation (§3.3).
                        for (name, affinity, collation) in scope.expand_star_cols(None)? {
                            out.push(synth_col_typed(name, affinity, collation));
                        }
                    }
                    ResultColumn::TableStar(t) => {
                        for (name, affinity, collation) in scope.expand_star_cols(Some(t))? {
                            out.push(synth_col_typed(name, affinity, collation));
                        }
                    }
                }
            }
            Ok(out)
        }
    }
}

/// The leftmost query/VALUES core of a (possibly compound) SELECT body. A compound is
/// left-associative and its result names come from the leftmost arm, so walking `left`
/// until a non-compound core reaches the arm whose names the compound exposes.
fn leftmost_core(body: &SelectBody) -> &SelectCore {
    match body {
        SelectBody::Select(core) => core,
        SelectBody::Compound { left, .. } => leftmost_core(left),
    }
}

/// A derived synthetic column with NONE affinity (`affinity_of_declared_type(None)` =
/// `Affinity::Blob`) and BINARY collation — the genuine no-affinity cases: a `VALUES`
/// column (a VALUES row has no declared column types) and the JSON table-valued-function
/// input columns. A query core's result columns instead inherit their inner expression's
/// affinity/collation via [`synth_col_typed`] (datatype3 §3.3).
fn synth_col(name: String) -> SynthCol {
    SynthCol {
        name,
        affinity: affinity_of_declared_type(None),
        collation: Collation::Binary,
        hidden: false,
    }
}

/// A derived synthetic column carrying an explicit affinity + collation — the §3.2/§7.1
/// metadata of the inner result expression it maps to (datatype3 §3.3). Used for a query
/// core's result columns, where a bare column ref / `*` column inherits its source
/// column's affinity + collation and any other expression is NONE (Blob) + BINARY.
fn synth_col_typed(name: String, affinity: Affinity, collation: Collation) -> SynthCol {
    SynthCol { name, affinity, collation, hidden: false }
}

/// A HIDDEN derived column (NONE affinity, BINARY collation): resolvable by name but
/// excluded from `SELECT *`. Only the JSON TVFs' `json`/`root` input columns use this.
fn hidden_col(name: &str) -> SynthCol {
    SynthCol {
        name: name.to_string(),
        affinity: affinity_of_declared_type(None),
        collation: Collation::Binary,
        hidden: true,
    }
}

/// The output schema shared by `json_each` / `json_tree` (json1.html §4.24): the eight
/// VISIBLE columns `key, value, type, atom, id, parent, fullkey, path`, then the two
/// HIDDEN input columns `json` (the raw document argument) and `root` (the start-path
/// text). Each is NONE affinity / BINARY collation — the values are executor-produced,
/// not stored with a declared type. The hidden pair sit LAST so the visible columns keep
/// registers `base+0..base+7`; they are excluded from `*` (see [`Source::star_columns`])
/// but resolvable by name, matching SQLite's hidden-column contract. The executor
/// (`ops::table_function`) appends the hidden values to every row it emits.
fn json_table_columns() -> Vec<SynthCol> {
    let mut cols: Vec<SynthCol> =
        ["key", "value", "type", "atom", "id", "parent", "fullkey", "path"]
            .into_iter()
            .map(|n| synth_col(n.to_string()))
            .collect();
    cols.push(hidden_col("json"));
    cols.push(hidden_col("root"));
    cols
}

/// Classify a FROM table-valued function name as a JSON TVF, or `None` for any other
/// name (which the caller turns into a loud "no such table-valued function" error). The
/// match is case-insensitive (SQLite function names are); a schema-qualified name is
/// never one of these built-ins.
fn json_table_kind(name: &QualifiedName) -> Option<JsonTableKind> {
    if name.schema.is_some() {
        return None;
    }
    match name.name.to_ascii_lowercase().as_str() {
        "json_each" => Some(JsonTableKind::Each),
        "json_tree" => Some(JsonTableKind::Tree),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Phase 2: build the access / join subtree.
// ---------------------------------------------------------------------------

/// Build the access/join operator subtree for the FROM clause and return it together
/// with the residual `WHERE` the caller must still apply as a [`PlanNode::Filter`].
///
/// * No FROM → [`PlanNode::SingleRow`]; the whole `bound_where` is the residual.
/// * A single base table → the rowid-aware access path ([`plan_table_access`]), which
///   may fold a rowid constraint into a seek and hands back the leftover residual.
/// * A single derived table → its materialized `CtePlan` scanned by a
///   [`PlanNode::CteScan`]; the whole `bound_where` is the residual (a `CteScan`
///   consumes no predicate).
/// * A join → a left-deep [`PlanNode::Join`] tree (each leaf a full scan — a base
///   table's scan or a derived table's `CteScan` — except a right leaf that becomes a
///   per-left rowid seek under [`JoinStrategy::IndexNestedLoop`]); the whole
///   `bound_where` is returned as the residual (one `Filter` over the join is correct
///   and sufficient — per-table pushdown is a later optimization).
///
/// `base_offset` matches the value passed to [`resolve_from`] (0 at top level); the
/// sources in `scope` were placed there.
pub fn build_from(
    ctx: &mut PlanCtx,
    scope: &Scope,
    from: &Option<FromClause>,
    base_offset: usize,
    bound_where: Option<EvalExpr>,
) -> Result<(PlanNode, Option<EvalExpr>)> {
    let catalog = ctx.catalog;
    let Some(tree) = from else {
        return Ok((PlanNode::SingleRow, bound_where));
    };
    match tree {
        // The WHOLE FROM is a parenthesized join `(a JOIN b …)`: parenthesizing the entire
        // FROM changes nothing (there is no enclosing join to re-associate against), so it
        // is the same plan as the unparenthesized `FROM a JOIN b …` — build the bracketed
        // tree directly against the full scope at `base_offset`, whose sources are already
        // at their absolute bases (`base_offset`, `base_offset + WL`, …). This is NOT a
        // `build_subjoin` local 0-based space: at a nonzero base (a correlated subquery or
        // trigger action) the executor prepends the outer prefix ONCE via the join's left
        // spine, exactly as for a flat join, so a whole-FROM parenthesized join is
        // supported. (Only a parenthesized join used as an OPERAND of a larger join needs a
        // local space and is rejected under a nonzero base — see [`build_subjoin`].)
        JoinTree::Table(TableOrSubquery::Join(jt)) => {
            debug_assert_eq!(
                scope.sources.first().map(|s| s.base()),
                Some(base_offset),
                "resolve_from placed the first source at base_offset"
            );
            let mut next = 0usize;
            let (mut node, _width) = build_join(ctx, scope, catalog, jt, base_offset, &mut next)?;
            debug_assert_eq!(next, scope.sources.len(), "every source is consumed once");
            // Push WHERE conjuncts into the join tree (comma/inner joins only); the
            // leftover is the residual top `Filter`. A no-op for outer / sub-join shapes.
            let residual = push_where_into_join(&mut node, base_offset, bound_where);
            Ok((node, residual))
        }
        // A single source: a base table keeps its rowid-aware access path (the WHERE is
        // offered to `plan_table_access`, which may fold a rowid seek and return the
        // leftover residual); a derived table becomes a `CteScan` leaf that consumes no
        // predicate (the whole WHERE stays the residual).
        JoinTree::Table(tos) => {
            let src = scope
                .sources
                .first()
                .ok_or_else(|| Error::sql("internal error: single-table FROM produced no source"))?;
            build_source_leaf(ctx, scope, catalog, tos, src, bound_where)
        }
        // A join: build the left-deep tree (each leaf a full scan — a base scan or a
        // derived CteScan — or a rowid seek on an IndexNestedLoop right leaf), then let
        // the caller apply the whole WHERE as one Filter above it (the residual).
        JoinTree::Join { .. } => {
            debug_assert_eq!(
                scope.sources.first().map(|s| s.base()),
                Some(base_offset),
                "resolve_from placed the first source at base_offset"
            );
            let mut next = 0usize;
            let (mut node, _width) = build_join(ctx, scope, catalog, tree, base_offset, &mut next)?;
            debug_assert_eq!(next, scope.sources.len(), "every source is consumed once");
            // Push WHERE conjuncts into the join tree (comma/inner joins only); the
            // leftover is the residual top `Filter`. A no-op for outer / sub-join shapes.
            let residual = push_where_into_join(&mut node, base_offset, bound_where);
            Ok((node, residual))
        }
    }
}

/// Build the left-deep join subtree for `tree`, consuming sources from `scope` in
/// left-to-right order via `next`. Returns the node and its combined register width
/// (relative to `base_offset`, i.e. the width of the row the node emits).
///
/// Both operand positions may be a parenthesized sub-join (`TableOrSubquery::Join`): the
/// left-spine leaf via [`build_operand`], the right operand via [`build_right`]. A sub-join
/// is built as a self-contained LOCAL subtree ([`build_subjoin`]); a plain leaf keeps its
/// existing single-source path.
fn build_join(
    ctx: &mut PlanCtx,
    scope: &Scope,
    catalog: &dyn Catalog,
    tree: &JoinTree,
    base_offset: usize,
    next: &mut usize,
) -> Result<(PlanNode, usize)> {
    match tree {
        // The left-spine leaf. It is always at the running base (leftmost of its spine),
        // so a left-position sub-join's 0-based local space coincides with its absolute
        // base — [`build_operand`] handles both a plain table and a parenthesized join.
        JoinTree::Table(tos) => build_operand(ctx, scope, catalog, tos, base_offset, next),
        JoinTree::Join { left, op, right, constraint } => {
            let left_start = *next;
            let (left_node, left_width) = build_join(ctx, scope, catalog, left, base_offset, next)?;
            let join_type = join_type_of(op.kind);

            // Build the RIGHT operand (a single leaf, or a whole parenthesized sub-join),
            // its bound `on`, and its physical strategy. `next` enters at the right
            // operand's first source and leaves past its last, so the running cursor —
            // and the `left_sources`/`right_sources` slices [`build_right`] cuts from the
            // full scope — stay correct.
            let (right_node, right_width, on, strategy) = build_right(
                ctx, scope, catalog, right, op, constraint.as_ref(), &join_type, base_offset,
                left_start, left_width, next,
            )?;
            let node = PlanNode::Join(Join {
                left: Box::new(left_node),
                left_width,
                right: Box::new(right_node),
                right_width,
                join_type,
                on,
                strategy,
            });
            Ok((node, left_width + right_width))
        }
    }
}

/// Build ONE FROM operand (a `table-or-subquery`) as a plan node in `scope`'s register
/// space, consuming the source(s) it covers from `next`, and return the node with its
/// combined width. A plain leaf is a single access leaf ([`build_source_leaf`], its
/// residual discarded — the WHERE is applied once above the whole join); a parenthesized
/// join is a self-contained local subtree ([`build_subjoin`]).
///
/// This is the LEFT-operand builder (the left-spine leaf, and nested leaves inside a
/// sub-join). A join's RIGHT operand goes through [`build_right`] instead, which also
/// weighs a rowid index-nested-loop against the bound `on`.
fn build_operand(
    ctx: &mut PlanCtx,
    scope: &Scope,
    catalog: &dyn Catalog,
    tos: &TableOrSubquery,
    base_offset: usize,
    next: &mut usize,
) -> Result<(PlanNode, usize)> {
    match tos {
        TableOrSubquery::Join(jt) => build_subjoin(ctx, catalog, jt, base_offset, next),
        _ => {
            let src = source_at(scope, *next)?;
            *next += 1;
            let (node, _residual) = build_source_leaf(ctx, scope, catalog, tos, src, None)?;
            Ok((node, table_width(src)))
        }
    }
}

/// Build a parenthesized sub-join `jt` (Phase 2) as a SELF-CONTAINED join subtree in its
/// OWN 0-based local register space, advancing the enclosing `next` past the sources it
/// covers. Returns the node and its combined width.
///
/// WHY A LOCAL SPACE. A sub-join used as a join operand is evaluated by the executor
/// STANDALONE: the enclosing join builds it with the same `outer` it received — NOT with
/// the left row (see `minisqlite-exec`'s `ops::join`) — so the sub-join's rows are its own
/// `b ++ c …` with its leftmost leaf at register 0, and its internal `ON`/`USING` MUST
/// bind THERE. The enclosing join's own `ON` still binds against the ABSOLUTE scope: its
/// combined row is exactly `left ++ (b ++ c …)`, which places the sub-join's columns at
/// their absolute bases, and a hash right-key rebases by `left_width` back onto this same
/// 0-based row — so the two register spaces compose exactly. Re-resolving `jt` at base 0
/// reproduces that local layout precisely (and recurses for nested parens).
///
/// The `base_offset` here is the ENCLOSING correlated/trigger outer width (0 at the top
/// level). It is used ONLY as a guard: a sub-join's LOCAL 0-based space composes with the
/// enclosing join only when the executor evaluates it standalone with an EMPTY outer,
/// which holds at the top level but NOT under a correlated / trigger outer (a left-spine
/// sub-join would then need to carry the outer prefix at `base_offset`, which this local
/// build cannot). So a nonzero `base_offset` is a loud, precise residual gap rather than a
/// silent mis-bind; a FLAT correlated/trigger join (the common shape) is fully supported.
fn build_subjoin(
    ctx: &mut PlanCtx,
    catalog: &dyn Catalog,
    jt: &JoinTree,
    base_offset: usize,
    next: &mut usize,
) -> Result<(PlanNode, usize)> {
    if base_offset != 0 {
        return Err(Error::sql(
            "a correlated subquery or trigger action whose FROM contains a parenthesized join \
             is not yet supported",
        ));
    }
    // Phase 1 (local): RE-RESOLVE the sub-join's sources at base 0 (rather than slicing +
    // rebasing the outer scope's already-resolved copies). The re-walk is the simpler,
    // lower-risk way to get a 0-based local layout; it is cold plan-time work bounded by
    // FROM-nesting depth and never double-compiles a derived table (compilation happens
    // only at build_join leaves, each reached once — this is schema-only). The fresh scope
    // has NO parent because `build_subjoin` only proceeds at `base_offset == 0`: the guard
    // above returns a loud error for a nonzero base (a parenthesized sub-join operand under a
    // correlated / trigger outer is the one residual gap), so there is no outer row for the
    // sub-join to reference and a `parent` here could only mis-resolve, never help.
    let mut sources = Vec::new();
    let mut coalesced = Vec::new();
    let width = resolve_tree(catalog, jt, 0, &mut sources, &mut coalesced)?;
    let source_count = sources.len();
    let local_scope = Scope {
        sources: sources.as_slice(),
        coalesced: coalesced.as_slice(),
        parent: None,
        grouping: None,
        saw_correlated: None,
        correlated_cols: None,
        nondeterministic: None,
        windowing: None,
    };
    // Phase 2 (local): build the join subtree binding each ON against the local scope.
    let mut local_next = 0usize;
    let (node, built) = build_join(ctx, &local_scope, catalog, jt, 0, &mut local_next)?;
    debug_assert_eq!(local_next, source_count, "sub-join sources consumed once");
    debug_assert_eq!(built, width, "sub-join Phase-1 and Phase-2 widths agree");
    *next += source_count;
    Ok((node, width))
}

/// Build a join's RIGHT operand together with its bound `on` and physical [`JoinStrategy`],
/// consuming its source(s) from `next`. Returns `(right_node, right_width, on, strategy)`.
///
/// The operand is built first so its source span `[right_start, *next)` is known, then
/// `on` is bound against the FULL (absolute) `scope` over `left_sources ++ right_sources`,
/// then the strategy is chosen from that `on`:
///   * a single leaf → [`choose_right_and_strategy`] (may fold a rowid index-nested-loop);
///   * a parenthesized sub-join → a local subtree ([`build_subjoin`]) with a hash / nested
///     strategy — never a rowid seek, since a sub-join is not a single base table.
#[allow(clippy::too_many_arguments)]
fn build_right(
    ctx: &mut PlanCtx,
    scope: &Scope,
    catalog: &dyn Catalog,
    right: &TableOrSubquery,
    op: &JoinOperator,
    constraint: Option<&JoinConstraint>,
    join_type: &JoinType,
    base_offset: usize,
    left_start: usize,
    left_width: usize,
    next: &mut usize,
) -> Result<(PlanNode, usize, Option<EvalExpr>, JoinStrategy)> {
    let right_start = *next;
    match right {
        TableOrSubquery::Join(jt) => {
            let (right_node, right_width) = build_subjoin(ctx, catalog, jt, base_offset, next)?;
            // `left_sources` is everything this join node covers before its right operand;
            // `right_sources` is the whole sub-join's span. USING/NATURAL resolve names on
            // each side within these slices; the `on` binds against the absolute scope.
            let left_sources = &scope.sources[left_start..right_start];
            let right_sources = &scope.sources[right_start..*next];
            // Cross-check the two register spaces agree: the sub-join's own combined width
            // (from build_subjoin's base-0 re-resolve) MUST equal the width of the outer
            // scope span it consumed here. They can only diverge if source-schema
            // resolution ever became context-sensitive — fail loud at that cause rather
            // than letting a mis-sized right operand read off-by-`left_width` registers.
            debug_assert_eq!(
                right_sources.iter().map(|s| table_width(s)).sum::<usize>(),
                right_width,
                "parenthesized sub-join width matches the outer sources it spans",
            );
            let on = build_on(ctx, scope, op, constraint, left_sources, right_sources)?;
            // A parenthesized sub-join is never a single base table, so it never seeks by
            // rowid; the strategy is Hash (for an equijoin) or NestedLoop, and `on` is the
            // residual (Hash strips the consumed equi-keys) or the full predicate.
            let (strategy, on) = choose_join_strategy(on, base_offset, left_width, right_width);
            Ok((right_node, right_width, on, strategy))
        }
        _ => {
            let right_idx = right_start;
            let right_src = source_at(scope, right_idx)?;
            *next += 1;
            let right_width = table_width(right_src);

            // The left sources are everything this join node covers before its right leaf;
            // the right side is that single leaf. USING/NATURAL resolve names on each side
            // within these slices (an unqualified name may live in several tables of the
            // whole scope, so a side-scoped lookup disambiguates it).
            let left_sources = &scope.sources[left_start..right_idx];
            let right_sources = &scope.sources[right_idx..=right_idx];
            let on = build_on(ctx, scope, op, constraint, left_sources, right_sources)?;

            // Choose the right leaf and physical strategy from the bound `on`, and get back
            // the `on` to store: a rowid index-nested-loop (a per-left seek, no right
            // materialization) or a nested loop keeps the FULL `on`, which the executor
            // re-checks per pair; a hash join keeps only the RESIDUAL (its consumed
            // equi-keys are redundant with the hash match and are stripped).
            let (right_node, strategy, on) = choose_right_and_strategy(
                ctx,
                scope,
                catalog,
                right,
                right_src,
                join_type,
                on,
                base_offset,
                left_width,
                right_width,
            )?;
            Ok((right_node, right_width, on, strategy))
        }
    }
}

// ---------------------------------------------------------------------------
// Join strategy selection: a rowid index-nested-loop, else Hash for equijoins,
// else NestedLoop.
//
// Every keyed strategy is O(left + right) or better; the nested loop is O(left * right),
// catastrophic at scale. The choice is purely physical — it preserves the result set for
// EVERY join type. A keyed strategy is correct as long as it never MISSES a pair `on`
// would keep; over-matching is harmless because the `on` the executor re-checks
// (`ops::join`'s `try_pair`/`on_keeps`) filters the surplus.
//
// What each strategy stores as the node `on` differs, because the SOUNDNESS of dropping a
// pre-filter from `on` differs:
//   * Hash stores only the RESIDUAL — the `on` with its consumed equi-keys removed. Each
//     hash key is a plain `=` with NO affinity coercion, and its collation is folded into
//     the key, so a hash-matched pair (raw key + that collation) has the `=` provably TRUE
//     and a NULL key hash-excludes. Re-checking that `=` per pair would be pure
//     O(output-pairs) waste, so it is stripped; the residual still gates every matched
//     pair. (A coerced `=`, an `IS`, or any non-equality is never a hash key, so it stays
//     in the residual — it is the real filter.)
//   * A rowid index-nested-loop stores the FULL `on`: the seek key can OVER-match (a
//     coerced rowid compare seeks the coerced value), so the full-predicate recheck is a
//     required backstop, not redundant.
//   * NestedLoop stores the FULL `on` — nothing was a usable key to strip.
// The executor's hash join fills outer (Left/Right/Full) rows itself, and a Cross/comma
// join has no `on` so it stays nested.
// ---------------------------------------------------------------------------

/// Build the right leaf and pick the physical [`JoinStrategy`] for one join node, and
/// return the `on` to store on the node alongside it: the RESIDUAL for a hash join (its
/// consumed equi-keys stripped), or the FULL `on` for a rowid index-nested-loop or a
/// nested loop (see [`choose_join_strategy`]).
///
/// A rowid index-nested-loop is preferred for Inner/Left when the right side is a BASE
/// TABLE and the `on` seeks it by its rowid (see [`rowid_seek_key`]): the right leaf
/// becomes a per-left [`RowidScan`] the executor drives with `outer = left row`, so no
/// right side is ever materialized. A DERIVED right table has no rowid to seek, so it
/// never takes this path. Otherwise the right leaf is a full scan (a base table's scan or
/// a derived table's `CteScan`, via [`build_source_leaf`]) and the strategy is
/// [`choose_join_strategy`] (Hash for a register-based equijoin, else NestedLoop).
///
/// `base_offset != 0` (a correlated subquery or trigger action's flat join) never seeks:
/// the rowid/TVF IndexNestedLoop's seek register arithmetic assumes a 0-based join row, so
/// a shifted base falls through to [`choose_join_strategy`] (Hash or NestedLoop, both
/// correct at any base). A LATERAL table-valued function operand, which REQUIRES the seek,
/// is rejected loudly at a nonzero base instead (above).
#[allow(clippy::too_many_arguments)]
fn choose_right_and_strategy(
    ctx: &mut PlanCtx,
    scope: &Scope,
    catalog: &dyn Catalog,
    tos: &TableOrSubquery,
    right_src: &Source,
    join_type: &JoinType,
    on: Option<EvalExpr>,
    base_offset: usize,
    left_width: usize,
    right_width: usize,
) -> Result<(PlanNode, JoinStrategy, Option<EvalExpr>)> {
    // A table-valued function RIGHT operand is implicitly LATERAL — its argument may
    // reference the left row or the outer/OLD-NEW row — so it MUST run as an
    // IndexNestedLoop rebuilt per left row (the block just below), but that strategy is
    // gated to `base_offset == 0`. Under a correlated subquery / trigger outer a fall
    // through to Hash/NestedLoop would build the TVF ONCE with an empty outer, mis-binding
    // (or panicking on) a lateral argument. Reject it loudly instead — a precise residual
    // gap. (A base-table / derived-table join operand is unaffected and fully supported;
    // a derived table is materialized standalone, so it cannot reference the outer.)
    if base_offset != 0 && matches!(tos, TableOrSubquery::TableFunction { .. }) {
        return Err(Error::sql(
            "a table-valued function in the FROM of a correlated subquery or trigger action \
             is not yet supported",
        ));
    }
    // A JSON table-valued function right operand is implicitly LATERAL: its argument may
    // reference the LEFT row (`FROM t, json_each(t.col)`), so it MUST run as an
    // IndexNestedLoop — the executor rebuilds the right per left row with that row as
    // `outer`, which is exactly the context the TVF leaf evaluates its argument against.
    // A Hash/NestedLoop would instead build the right once against the join's own `outer`
    // (empty at top level), so a correlated argument would read out-of-range registers.
    // The executor's IndexNestedLoop supports Inner/Left/Cross (a comma join is Cross);
    // it rejects Right/Full, so those fall through to the default strategy (correct for a
    // NON-correlated argument, and the obscure correlated Right/Full case — which SQLite
    // itself restricts — is not special-cased). Gated on `base_offset == 0` like the rowid
    // seek below; a nonzero base (a correlated / trigger flat join) took the loud TVF gap
    // above, so only a top-level TVF reaches this INL path.
    if base_offset == 0
        && matches!(tos, TableOrSubquery::TableFunction { .. })
        && matches!(join_type, JoinType::Inner | JoinType::Left | JoinType::Cross)
    {
        let (right, _residual) = build_source_leaf(ctx, scope, catalog, tos, right_src, None)?;
        return Ok((right, JoinStrategy::IndexNestedLoop, on));
    }
    // IndexNestedLoop is Inner/Left over a BASE-TABLE right only (a derived table has no
    // rowid to seek; the executor also rejects INL for Right/Full, whose unmatched-right
    // tracking cannot work against a per-left-rebuilt right side). Gated on
    // `base_offset == 0`: the seek key's register arithmetic assumes a 0-based join row, so
    // a correlated / trigger flat join (shifted base) uses Hash/NestedLoop instead — both
    // correct at any base — via `choose_join_strategy` below.
    if base_offset == 0 && matches!(join_type, JoinType::Inner | JoinType::Left) {
        // A WITHOUT ROWID right table has NO integer rowid to seek, so it can never take
        // this rowid IndexNestedLoop path — it must fall through to the SeqScan-based join,
        // the same invariant `plan_table_access` enforces for a WR base table. The guard is
        // the sibling of that one: without it, `right_rowid_reg = base+N` would name a
        // register that, for a WR table, is one-past its row (its width is N, not N+1) and
        // aliases nothing — masked today only because `rowid_in_source` emits no Column
        // there, so the seek could not match. Guard explicitly rather than lean on that
        // downstream accident.
        if let Source::BaseTable { table, base, db, .. } = right_src {
            if !table.without_rowid {
                // The right table's rowid is the last register of its slice in the combined
                // (`left ++ right`) row the `on` was bound against. Borrow `on` to find the
                // seek key (an owned clone); `on` itself is returned IN FULL — an
                // IndexNestedLoop re-checks the whole predicate per seek hit, so unlike Hash
                // it strips nothing.
                let right_rowid_reg = *base + table.columns.len();
                let seek = on.as_ref().and_then(|on_ref| {
                    let mut conjuncts = Vec::new();
                    flatten_conjuncts(on_ref, &mut conjuncts);
                    conjuncts
                        .iter()
                        .copied()
                        .find_map(|c| rowid_seek_key(c, right_rowid_reg, left_width))
                });
                if let Some(seek_key) = seek {
                    let right = PlanNode::RowidScan(RowidScan {
                        table: table.name.clone(),
                        db: *db,
                        column_count: table.columns.len(),
                        op: RowidOp::Eq(seek_key),
                        direction: ScanDirection::Forward,
                    });
                    return Ok((right, JoinStrategy::IndexNestedLoop, on));
                }
            }
        }
    }

    let (right, _residual) = build_source_leaf(ctx, scope, catalog, tos, right_src, None)?;
    let (strategy, on) = choose_join_strategy(on, base_offset, left_width, right_width);
    Ok((right, strategy, on))
}

/// Build one FROM source's access leaf and the residual predicate the caller must still
/// apply as a [`PlanNode::Filter`]:
///
/// * a **base table** → the rowid-aware access path ([`plan_table_access`]), which is
///   offered `pred` and may fold a rowid seek, returning the leftover residual;
/// * a **derived table** → either a `CteScan` / `RecursiveScan` over a CTE's
///   PRE-REGISTERED `CtePlan` (a `FROM cte_name` reference), or a fresh materialized
///   `CtePlan` for an inline `FROM (SELECT …)` subquery ([`compile_derived`]); either
///   scan consumes no predicate, so the whole `pred` is returned as the residual.
///
/// A join leaf passes `pred = None` (the WHERE is applied once above the whole join) and
/// discards the (always-`None`) residual.
fn build_source_leaf(
    ctx: &mut PlanCtx,
    scope: &Scope,
    catalog: &dyn Catalog,
    tos: &TableOrSubquery,
    src: &Source,
    pred: Option<EvalExpr>,
) -> Result<(PlanNode, Option<EvalExpr>)> {
    match src {
        Source::BaseTable { table, base, db, .. } => {
            // At a NONZERO base (a correlated statement — a trigger action or a
            // correlated subquery), the bound predicate lives in the shifted register
            // space: this table's own columns are at `base + i`, while OLD/NEW (or an
            // outer query's) columns sit BELOW `base`. `plan_table_access` is strictly
            // 0-based (`rowid_reg = n`, and it reads `Column(i)` as this table's column
            // `i`), so handing it that shifted predicate lets a low outer register be
            // mis-read as this table's rowid / an indexed column and folded into a BOGUS
            // seek that silently drops rows. Emit a plain scan and keep the whole
            // predicate as the residual `Filter` (correct at any base); recovering an
            // index/rowid seek at a nonzero base is a perf follow-up (a base-offset-aware
            // `plan_table_access`). This mirrors `build_scan` in
            // `compile/{update,delete}.rs`, the two other nonzero-base access sites.
            if *base != 0 {
                let node = PlanNode::SeqScan(SeqScan {
                    table: table.name.clone(),
                    db: *db,
                    column_count: table.columns.len(),
                });
                return Ok((node, pred));
            }
            let access = plan_table_access(catalog, table, *db, pred)?;
            Ok((access.node, access.residual))
        }
        // A `Source::Derived` is one of two shapes, told apart by its FROM node:
        //  * a CTE reference (`FROM cte_name`, a `Table`) — its leaf is a `CteScan` over
        //    the PRE-REGISTERED [`CtePlan`] (Phase 1 already recorded the id/kind), or a
        //    `RecursiveScan` when it is the recursive CTE's own step self-reference;
        //  * an inline subquery (`FROM (SELECT …)`, a `Subquery`) — compiled on the spot
        //    by [`compile_derived`], which registers a fresh materialized `CtePlan`.
        // Either way the scan consumes no predicate, so the whole `pred` is the residual.
        Source::Derived { .. } => match tos {
            TableOrSubquery::Table { name, .. } => {
                // Precedence MUST match Phase 1 (`resolve_table`): an unqualified name may
                // be shadowed by a CTE, but a schema-qualified `db.name` is always a schema
                // object (a CTE has no schema), so it skips the CTE check in BOTH phases.
                // Without this gate a qualified `main.v` that also names an in-scope CTE
                // would resolve to the VIEW in Phase 1 yet scan the CTE's body here in
                // Phase 2 — a silent wrong-body scan (the width fence only guards the
                // neither-CTE-nor-view case, not the ambiguous both case).
                if name.schema.is_none() {
                    if let Some(cte) = crate::compile::cte::lookup_cte(&name.name) {
                        let column_count = src.width();
                        let node = match cte.leaf {
                            crate::compile::cte::CteLeaf::Scan(id) => {
                                PlanNode::CteScan { id, column_count }
                            }
                            crate::compile::cte::CteLeaf::Recursive => {
                                PlanNode::RecursiveScan { column_count }
                            }
                            // A schema-only entry is a Phase-1 placeholder (see `push_cte_schemas`)
                            // that must be popped before Phase 2; reaching it here means a WITH scope
                            // outlived its schema computation — a bug, not a user error.
                            crate::compile::cte::CteLeaf::SchemaOnly => {
                                return Err(Error::sql(format!(
                                    "internal error: CTE {} is a Phase-1 schema-only entry at Phase 2",
                                    name.name
                                )));
                            }
                        };
                        return Ok((node, pred));
                    }
                }
                // Not a CTE: a `Table`-shaped derived source is a VIEW (Phase 1 resolved it
                // to `Source::Derived` via `catalog.view_in`). Compile its stored SELECT into
                // a fresh materialized `CtePlan` — the identical lowering `compile_derived`
                // gives an inline `FROM (SELECT …)` — so the executor reads a plain
                // `CteScan` and needs no view-aware code. `compile_view` re-checks the
                // circular guard and isolates the CTE scope. The view is fetched from the
                // SAME namespace Phase 1 resolved (via `resolve_ref_db` + `view_in`), not a
                // bare `view()`, so a shadowed / schema-qualified `v` re-fetches the SAME
                // view here as in Phase 1 rather than the search-order-first one.
                if let Some(db) = resolve_ref_db(catalog, name)? {
                    if let Some(view) = catalog.view_in(db, &name.name)? {
                        return Ok((crate::compile::view::compile_view(ctx, view, src)?, pred));
                    }
                }
                Err(Error::sql(format!(
                    "internal error: {} resolved to a derived source in Phase 1 but is neither \
                     a CTE nor a view in Phase 2",
                    name.name
                )))
            }
            // A table-valued function: its leaf's argument expressions are bound HERE,
            // against the whole FROM `scope`, so a correlated `FROM t, json_each(t.col)` (or
            // `FROM t, pragma_index_info(t.name)`) resolves the argument to the left table's
            // register (the executor threads that left row as the leaf's `outer`). A
            // `pragma_*` name compiles to a `PragmaFunctionScan` (see
            // [`crate::compile::pragma_tvf`]); everything else is a JSON TVF
            // (`TableFunctionScan`). Like the other derived leaves it consumes no predicate,
            // so the whole `pred` is the residual.
            TableOrSubquery::TableFunction { name, args, .. } => {
                let node = match crate::compile::pragma_tvf::classify(name) {
                    Some(kind) => {
                        crate::compile::pragma_tvf::compile(ctx, scope, name, kind, args, src)?
                    }
                    None => compile_table_function(ctx, scope, name, args, src)?,
                };
                Ok((node, pred))
            }
            _ => Ok((compile_derived(ctx, tos, src)?, pred)),
        },
    }
}

/// Compile a JSON table-valued function leaf (`json_each` / `json_tree`) into a
/// [`PlanNode::TableFunctionScan`]. The argument expressions are bound against the whole
/// FROM `scope` (SQLite table-valued functions are implicitly LATERAL, so the JSON
/// argument may reference a preceding table's columns — the executor re-evaluates them
/// per outer row). Accepts `(X)` or `(X, P)`; any other arity is a loud error. The
/// carried `column_count` is the Phase-1 schema width (the eight visible columns plus the
/// hidden `json`/`root`), so the emitted rows match the register width the outer scope
/// already bound against.
fn compile_table_function(
    ctx: &mut PlanCtx,
    scope: &Scope,
    name: &QualifiedName,
    args: &[Expr],
    src: &Source,
) -> Result<PlanNode> {
    let kind = json_table_kind(name).ok_or_else(|| {
        Error::sql(format!("no such table-valued function: {}", name.name))
    })?;
    if args.is_empty() || args.len() > 2 {
        return Err(Error::sql(format!(
            "wrong number of arguments to function {}()",
            name.name
        )));
    }
    let arg = bind_expr(scope, ctx, &args[0])?;
    let path = match args.get(1) {
        Some(p) => Some(bind_expr(scope, ctx, p)?),
        None => None,
    };
    let column_count = src.width();
    debug_assert_eq!(
        column_count,
        minisqlite_functions::JSON_TABLE_COLUMN_COUNT
            + minisqlite_functions::JSON_TABLE_HIDDEN_COLUMN_COUNT,
        "the JSON TVF Phase-1 schema must be the visible columns plus the hidden json/root"
    );
    // `emit_json` defaults to `true` (materialize the document into each row's hidden
    // `json` slot). This is the correct-if-conservative value: it is flipped to `false` by
    // the post-binding pass (`compile::select::finalize_tvf_emit_json`) once the projection
    // and every other clause are bound and the binder confirms nothing referenced the
    // hidden `json` column — which is when the per-row document copy is safe to skip. The
    // argument expressions are bound above (Phase 2), but the SELECT list is bound later
    // (Phase 3), so the reference is not yet known here.
    Ok(PlanNode::TableFunctionScan { kind, arg, path, column_count, emit_json: true })
}

/// Compile a derived table (`FROM (SELECT …) [AS] alias`) into a [`PlanNode::CteScan`]
/// over a freshly-registered [`CtePlan::Materialized`]. The inner SELECT is compiled as
/// a standalone, NON-correlated query (base 0, no parent — SQLite has no LATERAL), so
/// its own `EvalExpr::Column`s index its own rows and its rows are `column_count` wide
/// with NO trailing rowid (the `CteScan` / `ops::cte` contract).
///
/// The CTE `id` is captured AFTER compiling the inner body: a derived table nested
/// inside this one registers its own (lower) `CtePlan` first, so capturing the length
/// beforehand would collide. The Phase-1 schema width ([`derived_schema`]) MUST equal the
/// compiled body's width — the outer scope already bound `Column(base+i)` against it — so a
/// mismatch fails closed with a loud error, never a silent mis-sized `CteScan` row.
fn compile_derived(ctx: &mut PlanCtx, tos: &TableOrSubquery, src: &Source) -> Result<PlanNode> {
    let TableOrSubquery::Subquery { select, .. } = tos else {
        return Err(Error::sql("internal error: a derived source is not a subquery in FROM"));
    };
    let column_count = src.width();
    let (body, names) = crate::compile::select::compile_select(ctx, select)?;
    // Phase-1 (`derived_schema`, catalog-only naming) and Phase-2 (the real projection
    // compiler) count the output columns by SEPARATE paths; they agree today because both
    // delegate to the same leaf helpers, but a projection feature added to one and not the
    // other would emit a `CteScan` whose width disagrees with the registers the outer scope
    // already bound (`Column(base+i)`), silently mis-sizing rows in a release build. This is
    // once-per-derived-table plan-time work (off the hot path), so fail closed with a hard
    // error rather than a debug-only assert.
    if names.len() != column_count {
        return Err(Error::sql(format!(
            "internal error: derived table Phase-1 schema width ({column_count}) \
             != compiled body width ({})",
            names.len()
        )));
    }
    // Capture the id AFTER the inner compile (nested derived tables pushed their own
    // CtePlans first); the outer scope's register offsets already used `column_count`.
    let id = ctx.ctes.len();
    ctx.ctes.push(CtePlan::Materialized {
        name: src.exposed_name().to_string(),
        column_count,
        body,
    });
    Ok(PlanNode::CteScan { id, column_count })
}

/// If `c` equates the right table's rowid (`right_rowid_reg`) to a bare left column
/// (`Column(i)`, `i < left_width`), return that column as the seek key. The executor
/// evaluates it against the left row (passed as `outer`) to seek the right table by
/// rowid.
///
/// Correctness (holds for EVERY left value class, coerced or not): the seek key IS the
/// `on`'s left operand, so the seek and the `on` re-check evaluate the same expression.
/// The rowid eq executor coerces that value with sqlite's `OP_SeekRowid` rule
/// (`eval_rowid_eq` → `must_be_int`: apply NUMERIC affinity, then require a lossless
/// integer, else no rows), which is EXACTLY the affinity `on` applies to the rowid
/// comparison — so `must_be_int(left) == Int(R)` iff `on`'s `right.rowid == NUMERIC(left)`
/// is true (both hold iff `left` losslessly equals the integer `R`). The seek therefore
/// hits precisely the row `on` keeps for this left row; it can never miss one, and the
/// IndexNestedLoop re-checks the full `on` after the seek (see the executor's `on_keeps`)
/// as a backstop. This is why an affinity-carrying (text/real) left column also seeks —
/// unlike a secondary-index seek (raw against a Binary-ordered key) or a hash equijoin
/// (hashes RAW, so a coerced key silently misses — see [`equijoin_key`]), the rowid seek
/// coerces, so no affinity gate is needed here.
fn rowid_seek_key(c: &EvalExpr, right_rowid_reg: usize, left_width: usize) -> Option<EvalExpr> {
    let EvalExpr::Compare { op: CmpOp::Eq, null_safe: false, left, right, .. } = c else {
        return None;
    };
    if is_column_ref(left, right_rowid_reg) {
        if let EvalExpr::Column(i) = right.as_ref() {
            if *i < left_width {
                return Some(EvalExpr::Column(*i));
            }
        }
    }
    if is_column_ref(right, right_rowid_reg) {
        if let EvalExpr::Column(i) = left.as_ref() {
            if *i < left_width {
                return Some(EvalExpr::Column(*i));
            }
        }
    }
    None
}

/// True if `e` is exactly `Column(reg)`.
fn is_column_ref(e: &EvalExpr, reg: usize) -> bool {
    matches!(e, EvalExpr::Column(r) if *r == reg)
}

/// The register windows a join node's two sides occupy in its `on` predicate's register
/// space: the left subtree covers `left` and the right subtree covers `right`. Built
/// from `base_offset` and the two widths (the `on` is bound against the combined
/// `left ++ right` row placed at `base_offset`).
struct JoinRanges {
    left: std::ops::Range<usize>,
    right: std::ops::Range<usize>,
}

impl JoinRanges {
    fn new(base_offset: usize, left_width: usize, right_width: usize) -> Self {
        let left = base_offset..base_offset + left_width;
        let right = left.end..left.end + right_width;
        JoinRanges { left, right }
    }
}

/// Which side of the join an `on` operand reads. Only a single-sided operand can be a
/// hash key: `LeftOnly` is probed against the standalone left row, `RightOnly` (after
/// rebasing) against the standalone right row.
#[derive(Clone, Copy, PartialEq, Eq)]
enum KeySide {
    /// ≥1 column, ALL within the left range (and nothing else).
    LeftOnly,
    /// ≥1 column, ALL within the right range (and nothing else).
    RightOnly,
    /// A constant (no columns), a mix of both sides, an out-of-range/outer reference,
    /// or any subquery — none of which is a usable single-sided hash key.
    Neither,
}

/// Choose the physical [`JoinStrategy`] for one join AND the `on` predicate to store on
/// the node, consuming the bound `on`.
///
/// The top-level `AND` conjuncts of `on` are partitioned in ONE pass: a hash-usable
/// equijoin conjunct (see [`equijoin_key`]) contributes one `(left_key, right_key,
/// collation)` triple to the three parallel key vectors; every other conjunct is a
/// RESIDUAL. Using the SAME [`equijoin_key`] test for both makes the strip criterion
/// identical to the key criterion by construction — a conjunct is a key XOR a residual,
/// never miscounted.
///
/// * With ≥1 key the join runs as [`JoinStrategy::Hash`] and the returned `on` is ONLY
///   the residual (re-`AND`ed, or `None` when every conjunct was a key). Storing the
///   residual rather than the full `on` is the point of this pass: a consumed equi-key is
///   a plain `=` with NO affinity coercion whose collation is carried into the hash key,
///   so a pair that hash-matches (raw key + that collation) has the `=` provably TRUE and
///   a NULL key hash-excludes. Dropping exactly those conjuncts therefore cannot change
///   which pairs pass — for INNER and every OUTER flavor alike (the hash match already
///   enforces the equality; unmatched-row null-extension is unchanged) — and it removes
///   the per-pair re-evaluation the executor's `on_keeps` would otherwise pay for every
///   hash-matched pair (its `None` fast path, or a residual-only eval).
/// * Otherwise (no `on`, no equality, or only affinity-coercing / non-single-sided
///   equalities) it stays [`JoinStrategy::NestedLoop`] and the FULL `on` is returned
///   UNCHANGED — nothing was a hash key, so there is nothing safe to strip and the nested
///   loop re-checks the whole predicate per pair.
fn choose_join_strategy(
    on: Option<EvalExpr>,
    base_offset: usize,
    left_width: usize,
    right_width: usize,
) -> (JoinStrategy, Option<EvalExpr>) {
    let Some(on) = on else {
        return (JoinStrategy::NestedLoop, None);
    };
    let ranges = JoinRanges::new(base_offset, left_width, right_width);

    // The right key is rebased into the STANDALONE right row (its leftmost column at 0).
    // In the combined `on` register space that row's columns begin at `base_offset +
    // left_width` — the outer prefix (`base_offset`) plus the left width — so a right key
    // shifts down by that whole amount. At the top level `base_offset == 0`, so the shift
    // is `left_width`, unchanged; under a correlated / trigger outer the executor builds
    // the right side with an EMPTY outer (no prefix — see `minisqlite-exec`'s `ops::join`),
    // so its columns really do start at register 0 of the standalone right row.
    let right_shift = base_offset + left_width;

    // Partition the conjuncts: hash keys vs residual. The residual is captured as borrows
    // and cloned ONLY in the Hash branch, so the common NestedLoop path returns the full
    // `on` by move with no clone.
    let mut conjuncts = Vec::new();
    flatten_conjuncts(&on, &mut conjuncts);
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut key_collations = Vec::new();
    let mut residual: Vec<&EvalExpr> = Vec::new();
    for c in conjuncts {
        match equijoin_key(c, &ranges, right_shift) {
            Some((lk, rk, coll)) => {
                left_keys.push(lk);
                right_keys.push(rk);
                key_collations.push(coll);
            }
            None => residual.push(c),
        }
    }

    if left_keys.is_empty() {
        // No conjunct was a hash key → nothing safe to strip; keep the full `on`.
        return (JoinStrategy::NestedLoop, Some(on));
    }
    // ≥1 equi-key consumed → store ONLY the residual (re-`AND`ed; `None` if empty). The
    // residual conjuncts are cloned from the ORIGINAL combined-space expressions — NOT the
    // rebased right-key copies `equijoin_key` builds — because `on_keeps` evaluates the
    // stored `on` against the combined `outer ++ left ++ right` row, so the residual must
    // keep those same column indices.
    let residual = rebuild_conjunction(residual.into_iter().cloned().collect());
    (JoinStrategy::Hash { left_keys, right_keys, key_collations }, residual)
}

/// Re-`AND` residual conjuncts into a single predicate, or `None` when there are none
/// (every conjunct was consumed as a hash key). Folds left in discovery order; `AND` is
/// associative under SQLite's three-valued logic, so the rebuilt shape evaluates
/// identically to the original conjunction minus the stripped equi-keys.
fn rebuild_conjunction(conjuncts: Vec<EvalExpr>) -> Option<EvalExpr> {
    let mut it = conjuncts.into_iter();
    let first = it.next()?;
    Some(it.fold(first, |acc, c| EvalExpr::And(Box::new(acc), Box::new(c))))
}

// ---------------------------------------------------------------------------
// WHERE-clause predicate pushdown into an inner/comma join tree.
//
// A comma join `FROM a, b, c WHERE a.x = b.y AND c.z = 5` compiles to a left-deep tree
// of CROSS joins whose ONLY filter is one `Filter` above the whole join, so the executor
// forms the FULL cartesian product before filtering: O(rows ^ tables), which is billions
// of rows -- an effective hang -- once several tables join. Real SQLite treats a comma
// join with equalities in the WHERE exactly like a `JOIN ... ON` with those equalities.
// This pass does the same: it distributes each WHERE conjunct down to the LOWEST join
// node that already produces every column the conjunct reads, folding it into that node's
// `on`. A conjunct spanning two sides becomes a join predicate there (and, via
// `choose_join_strategy`, an O(n) hash key); a single-side conjunct filters at the
// earliest level so the next table multiplies against far fewer surviving rows.
//
// The output rows are unchanged: for INNER / CROSS joins the result is the cartesian
// product filtered by (all ON conjuncts AND the WHERE), and AND is associative and
// commutative under three-valued logic, so evaluating a conjunct at any node that
// provides its inputs keeps exactly the rows where it is TRUE.
//
// SAFETY GUARD: applied ONLY to a left-deep tree of INNER/CROSS joins whose right
// operands are plain leaves (the shape a comma join builds). Any outer join
// (LEFT/RIGHT/FULL) or a parenthesized sub-join right operand disqualifies the WHOLE tree
// and the entire WHERE stays as the single top `Filter` (the prior behavior): pushing a
// predicate below an outer join's null-extended side would change results, and a sub-join
// right operand lives in its own local register space this absolute-space walk must not
// touch. A conjunct containing a subquery is likewise never pushed (its correlated refs
// live in a separate subplan that `mark_sides` reports as out-of-range), so it stays in
// the residual.

/// Distribute the conjuncts of `bound_where` into the join tree `node`, folding each into
/// the `on` of the lowest node that can evaluate it, and return the leftover residual for
/// the caller to apply as a top `Filter`. Returns `bound_where` unchanged (as the whole
/// residual) unless `node` is the eligible left-deep inner/cross shape (see the note
/// above), so it is a safe no-op for every other plan.
fn push_where_into_join(
    node: &mut PlanNode,
    base_offset: usize,
    bound_where: Option<EvalExpr>,
) -> Option<EvalExpr> {
    let where_expr = bound_where?;
    if !eligible_for_pushdown(node) {
        return Some(where_expr);
    }
    let mut conjuncts = Vec::new();
    flatten_conjuncts_owned(where_expr, &mut conjuncts);
    let mut pending: Vec<(EvalExpr, bool)> = conjuncts.into_iter().map(|e| (e, false)).collect();
    distribute_into_spine(node, base_offset, &mut pending);
    let leftover: Vec<EvalExpr> =
        pending.into_iter().filter(|(_, placed)| !placed).map(|(e, _)| e).collect();
    rebuild_conjunction(leftover)
}

/// True iff `node` is a left-deep tree of INNER/CROSS joins whose every right operand is a
/// plain leaf (not a nested `Join`) -- the only shape WHERE pushdown is proven safe for
/// here. Any other shape keeps the single top `Filter`.
fn eligible_for_pushdown(node: &PlanNode) -> bool {
    match node {
        PlanNode::Join(j) => {
            matches!(j.join_type, JoinType::Inner | JoinType::Cross)
                && !matches!(*j.right, PlanNode::Join(_))
                && eligible_for_pushdown(&j.left)
        }
        // A leaf (base scan, cte scan, ...) is a valid left-spine end.
        _ => true,
    }
}

/// Owned counterpart of [`flatten_conjuncts`]: consume `e`, splitting its top-level `AND`
/// chain into owned leaf conjuncts.
fn flatten_conjuncts_owned(e: EvalExpr, out: &mut Vec<EvalExpr>) {
    match e {
        EvalExpr::And(a, b) => {
            flatten_conjuncts_owned(*a, out);
            flatten_conjuncts_owned(*b, out);
        }
        other => out.push(other),
    }
}

/// Walk the left-deep spine, recursing into the left child FIRST so the DEEPEST node gets
/// first claim, then place each still-unplaced conjunct at THIS node in the best position:
///
///  1. A conjunct referencing ONLY this node's right leaf (a single-table filter on the
///     table just joined in) is pushed INTO that leaf's scan, as a [`PlanNode::Filter`]
///     over the `SeqScan`. This is the critical anti-blowup step: the right table is then
///     pre-filtered to its surviving rows BEFORE the nested loop multiplies the (possibly
///     large) left side against it -- without it, an unfiltered right table is scanned in
///     full for every left row (millions of allocating pair-concats). The predicate is
///     rebased into the standalone right row's 0-based space (the executor builds a right
///     leaf with an empty outer -- see `minisqlite-exec`'s `ops::join`), exactly the
///     `right_shift` [`equijoin_key`] applies to a right hash key.
///  2. Any other conjunct this node covers (a cross-side join predicate, or a filter on
///     the left spine) is `AND`ed into the node's `on`; a NestedLoop node is then re-run
///     through [`choose_join_strategy`] so a cross-side equijoin becomes an O(n) hash key.
///     A node already Hash / IndexNestedLoop keeps its strategy and just gains the conjunct
///     as an extra residual filter (correct -- the executor re-checks the residual per pair).
fn distribute_into_spine(node: &mut PlanNode, base_offset: usize, pending: &mut [(EvalExpr, bool)]) {
    let PlanNode::Join(j) = node else { return };
    distribute_into_spine(&mut j.left, base_offset, &mut *pending);
    let rbase = base_offset + j.left_width; // absolute base of the right leaf's columns
    let hi = rbase + j.right_width;

    // Step 1: push right-leaf-only single-table filters into the right SeqScan. Guarded to
    // a plain full scan: an IndexNestedLoop right is a per-left rowid seek and a non-scan
    // leaf has its own shape, so only a SeqScan (the comma-join / hash-build right) is
    // wrapped. `refs_only_within` also excludes any outer/correlated reference, which the
    // standalone right leaf (built with an empty outer) could not resolve.
    if matches!(j.right.as_ref(), PlanNode::SeqScan(_)) {
        let mut leaf_preds: Vec<EvalExpr> = Vec::new();
        for (expr, placed) in pending.iter_mut() {
            if *placed {
                continue;
            }
            if refs_only_within(expr, rbase, hi) {
                *placed = true;
                let mut rebased = expr.clone();
                rebase_columns(&mut rebased, rbase);
                leaf_preds.push(rebased);
            }
        }
        if let Some(pred) = rebuild_conjunction(leaf_preds) {
            let inner = std::mem::replace(j.right.as_mut(), PlanNode::SingleRow);
            *j.right = PlanNode::Filter { input: Box::new(inner), predicate: pred };
        }
    }

    // Step 2: fold the remaining covered conjuncts (cross-side joins, left-spine filters)
    // into this node's `on`.
    let mut claimed: Vec<EvalExpr> = Vec::new();
    for (expr, placed) in pending.iter_mut() {
        if *placed {
            continue;
        }
        if covered_below(expr, hi) {
            *placed = true;
            claimed.push(expr.clone());
        }
    }
    if claimed.is_empty() {
        return;
    }
    let fold_in = |mut on: Option<EvalExpr>, claimed: Vec<EvalExpr>| -> Option<EvalExpr> {
        for c in claimed {
            on = Some(match on {
                Some(prev) => EvalExpr::And(Box::new(prev), Box::new(c)),
                None => c,
            });
        }
        on
    };
    match j.strategy {
        // A NestedLoop node stores the FULL `on`, so fold the claimed conjuncts in and
        // re-choose the strategy: a cross-side equijoin among them upgrades the node to a
        // hash join (O(n) instead of O(n^2)); anything else stays a nested loop with the
        // conjuncts as its `on`.
        JoinStrategy::NestedLoop => {
            let combined = fold_in(j.on.take(), claimed);
            let (strategy, on) =
                choose_join_strategy(combined, base_offset, j.left_width, j.right_width);
            j.strategy = strategy;
            j.on = on;
        }
        // Hash / IndexNestedLoop already chose keys from their structural ON; just AND the
        // claimed conjuncts into the stored `on` as extra per-pair filters (no re-keying).
        _ => {
            j.on = fold_in(j.on.take(), claimed);
        }
    }
}

/// True iff every `EvalExpr::Column(i)` in `e` has `i < hi` and `e` contains no subquery /
/// RAISE. Reuses the exhaustive, tested [`mark_sides`] with a single window `[0, hi)`:
/// `saw_other` is set for any column at or above `hi` OR any subquery, so `!saw_other` is
/// exactly "fully evaluable from registers below `hi`, with nothing that must not be
/// pushed." (Outer/correlated columns sit below `base_offset <= hi`, so they count as
/// available, which is correct -- the outer prefix is present in every combined row.)
fn covered_below(e: &EvalExpr, hi: usize) -> bool {
    let ranges = JoinRanges { left: 0..hi, right: 0..0 };
    let (mut saw_left, mut saw_right, mut saw_other) = (false, false, false);
    mark_sides(e, &ranges, &mut saw_left, &mut saw_right, &mut saw_other);
    !saw_other
}

/// True iff `e` references at least one column, EVERY column lies in `[lo, hi)`, and `e`
/// contains no subquery / RAISE. Used to detect a single-table filter on the right leaf
/// `[lo, hi)`: `saw_left` (≥1 column in the window) AND `!saw_other` (nothing outside it,
/// no outer/correlated reference, no subquery). A column outside `[lo, hi)` -- including an
/// outer/correlated register below `lo` -- sets `saw_other`, so such a conjunct is NOT
/// pushed to the standalone right leaf (which is built with an empty outer and could not
/// resolve it); it stays for the `on` step instead.
fn refs_only_within(e: &EvalExpr, lo: usize, hi: usize) -> bool {
    let ranges = JoinRanges { left: lo..hi, right: 0..0 };
    let (mut saw_left, mut saw_right, mut saw_other) = (false, false, false);
    mark_sides(e, &ranges, &mut saw_left, &mut saw_right, &mut saw_other);
    saw_left && !saw_other
}

// ---------------------------------------------------------------------------
// Join reordering for a pure comma-join FROM.
//
// `FROM a, b, c, ...` is fully commutative and associative (every join is CROSS/inner),
// so ANY leaf order yields the same result SET. The order only decides how large the
// intermediate joins grow: joining along equijoin edges (each newly added table shares a
// `WHERE t.x = u.y` with an already-joined table) keeps every step bounded, while the
// textual order can force an enormous cartesian intermediate before a connecting table
// appears -- some generated queries join up to dozens of tables in random textual order, for
// which an unreordered plan is astronomically large.
//
// The leaves are reordered greedily into a connected order BEFORE the FROM is resolved,
// so the normal resolve/bind/build pipeline then runs unchanged in the better order
// (registers, scope, WHERE, and projection all bind fresh against the new order -- no
// register remapping needed). It applies ONLY to a pure comma join of plain table leaves;
// an explicit ON join, a CROSS JOIN (SQLite's "do not reorder" barrier), a NATURAL/USING
// join, a subquery/parenthesized leaf, or any outer join is left exactly as written.
// Correctness never depends on the chosen order -- commutativity guarantees the same rows
// -- so an imperfect graph analysis can only pick a slower order, never a wrong one.

/// If `from` is a pure comma join of >= 3 plain table leaves, return a leaf-reordered
/// copy whose order joins along `where_ast` equijoin edges (greedy connected order); else
/// `None` (use the FROM as written). Reordering never changes the result set.
pub(crate) fn reorder_comma_join(
    catalog: &dyn Catalog,
    from: &JoinTree,
    where_ast: &Expr,
    base_offset: usize,
) -> Option<JoinTree> {
    let leaves = comma_join_table_leaves(from)?;
    let n = leaves.len();
    // A 2-way comma join has only the identity/reverse order and no cartesian to defeat.
    if n < 3 {
        return None;
    }
    // Same ctx-free schema resolve the real pipeline runs, only to map a column reference
    // to the leaf that owns it. A shape mismatch (e.g. a coalesced source) bails safely.
    let mut sources: Vec<Source> = Vec::new();
    let mut coalesced = Vec::new();
    resolve_tree(catalog, from, base_offset, &mut sources, &mut coalesced).ok()?;
    if sources.len() != n || !coalesced.is_empty() {
        return None;
    }

    // Build the equijoin graph + per-leaf single-table filter counts from the WHERE.
    let mut conjuncts: Vec<&Expr> = Vec::new();
    flatten_and_ast(where_ast, &mut conjuncts);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut filter_score: Vec<u32> = vec![0; n];
    for c in conjuncts {
        if let Some((i, j)) = equijoin_leaves(c, &sources) {
            adjacency[i].push(j);
            adjacency[j].push(i);
            continue;
        }
        let mut refs = Vec::new();
        collect_column_leaves(c, &sources, &mut refs);
        refs.sort_unstable();
        refs.dedup();
        if refs.len() == 1 {
            filter_score[refs[0]] += 1;
        }
    }

    let order = greedy_join_order(n, &adjacency, &filter_score);
    // Identity order: nothing to gain, skip the rebuild.
    if order.iter().enumerate().all(|(k, &s)| k == s) {
        return None;
    }
    Some(rebuild_comma_join(&leaves, &order))
}

/// The plain table leaves of a PURE comma join, left-to-right, or `None` if `from` is not
/// exactly that (any non-comma operator, NATURAL, an ON/USING constraint, or a
/// subquery/function/parenthesized leaf disqualifies it -- those change the result under
/// reordering or carry a constraint that must stay attached to its tables).
fn comma_join_table_leaves(from: &JoinTree) -> Option<Vec<&TableOrSubquery>> {
    let mut leaves = Vec::new();
    fn walk<'a>(t: &'a JoinTree, out: &mut Vec<&'a TableOrSubquery>) -> Option<()> {
        match t {
            JoinTree::Table(tos) => {
                if matches!(tos, TableOrSubquery::Table { .. }) {
                    out.push(tos);
                    Some(())
                } else {
                    None
                }
            }
            JoinTree::Join { left, op, right, constraint } => {
                if op.natural || op.kind != JoinKind::Comma || constraint.is_some() {
                    return None;
                }
                walk(left, out)?;
                if matches!(right, TableOrSubquery::Table { .. }) {
                    out.push(right);
                    Some(())
                } else {
                    None
                }
            }
        }
    }
    walk(from, &mut leaves)?;
    Some(leaves)
}

/// Rebuild a left-deep pure comma-join tree from `leaves` taken in `order`.
fn rebuild_comma_join(leaves: &[&TableOrSubquery], order: &[usize]) -> JoinTree {
    let mut tree = JoinTree::Table(leaves[order[0]].clone());
    for &idx in &order[1..] {
        tree = JoinTree::Join {
            left: Box::new(tree),
            op: JoinOperator { natural: false, kind: JoinKind::Comma },
            right: leaves[idx].clone(),
            constraint: None,
        };
    }
    tree
}

/// Split a raw WHERE expression's top-level `AND` chain into its leaf conjuncts.
fn flatten_and_ast<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::Binary { op: BinaryOp::And, left, right } = e {
        flatten_and_ast(left, out);
        flatten_and_ast(right, out);
    } else {
        out.push(e);
    }
}

/// If `c` is a plain `col = col` between two DIFFERENT comma-join leaves, return their
/// leaf indices (an equijoin edge). Only a bare `Column = Column` counts -- a coerced or
/// expression equality is not treated as a graph edge (it still runs, just does not steer
/// the order).
fn equijoin_leaves(c: &Expr, sources: &[Source]) -> Option<(usize, usize)> {
    let Expr::Binary { op: BinaryOp::Eq, left, right } = c else {
        return None;
    };
    let (Expr::Column { table: lt, name: ln, .. }, Expr::Column { table: rt, name: rn, .. }) =
        (left.as_ref(), right.as_ref())
    else {
        return None;
    };
    let li = column_leaf(lt, ln, sources)?;
    let ri = column_leaf(rt, rn, sources)?;
    (li != ri).then_some((li, ri))
}

/// Resolve a raw column reference to the index of the comma-join leaf that owns it: a
/// qualified `t.col` matches the source whose exposed name is `t`; a bare `col` matches
/// the unique source that declares a column of that name (`None` if none or if ambiguous
/// -- an ambiguous or unresolved column simply does not steer the order).
fn column_leaf(table: &Option<String>, name: &str, sources: &[Source]) -> Option<usize> {
    match table {
        Some(t) => sources.iter().position(|s| s.exposed_name().eq_ignore_ascii_case(t)),
        None => {
            let mut found = None;
            for (i, s) in sources.iter().enumerate() {
                if s.column_names().iter().any(|c| c.eq_ignore_ascii_case(name)) {
                    if found.is_some() {
                        return None; // ambiguous bare name -- do not attribute it
                    }
                    found = Some(i);
                }
            }
            found
        }
    }
}

/// Best-effort collect of the leaf indices a conjunct references (for spotting a
/// single-table filter). Walks the common WHERE shapes and resolves each column; it does
/// NOT descend into a subquery (a different scope) and treats unhandled shapes as
/// contributing nothing. Under-collecting only weakens the ordering heuristic; it can
/// never make a wrong order (commutativity).
fn collect_column_leaves(e: &Expr, sources: &[Source], out: &mut Vec<usize>) {
    match e {
        Expr::Column { table, name, .. } => {
            if let Some(i) = column_leaf(table, name, sources) {
                out.push(i);
            }
        }
        Expr::Unary { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::Collate { expr, .. }
        | Expr::IsNull(expr)
        | Expr::NotNull(expr) => collect_column_leaves(expr, sources, out),
        Expr::Binary { left, right, .. } => {
            collect_column_leaves(left, sources, out);
            collect_column_leaves(right, sources, out);
        }
        Expr::Between { expr, low, high, .. } => {
            collect_column_leaves(expr, sources, out);
            collect_column_leaves(low, sources, out);
            collect_column_leaves(high, sources, out);
        }
        Expr::Like { lhs, rhs, escape, .. } => {
            collect_column_leaves(lhs, sources, out);
            collect_column_leaves(rhs, sources, out);
            if let Some(esc) = escape {
                collect_column_leaves(esc, sources, out);
            }
        }
        // The IN subject participates; the RHS (a value list or subquery) is left alone.
        Expr::In { expr, .. } => collect_column_leaves(expr, sources, out),
        Expr::Case { operand, whens, else_expr } => {
            if let Some(o) = operand {
                collect_column_leaves(o, sources, out);
            }
            for (w, t) in whens {
                collect_column_leaves(w, sources, out);
                collect_column_leaves(t, sources, out);
            }
            if let Some(el) = else_expr {
                collect_column_leaves(el, sources, out);
            }
        }
        Expr::Parenthesized(items) => {
            for it in items {
                collect_column_leaves(it, sources, out);
            }
        }
        // Literal / BindParam / Raise carry no column; Subquery / Exists / Function are not
        // walked (a subquery is a separate scope; a function's args rarely steer ordering).
        _ => {}
    }
}

/// Greedy connected join order over `n` leaves: repeatedly pick the not-yet-placed leaf
/// with the most equijoin edges to the already-placed set (ties broken by more single-
/// table filters, then higher degree, then lowest index). The first pick has no placed
/// neighbours, so it falls to the most-filtered / highest-degree leaf; a disconnected
/// component likewise restarts from its best leaf. Maximizing edges-to-placed keeps each
/// join a keyed (hash) join against the running result instead of a cartesian product.
fn greedy_join_order(n: usize, adjacency: &[Vec<usize>], filter_score: &[u32]) -> Vec<usize> {
    let mut placed = Vec::with_capacity(n);
    let mut in_placed = vec![false; n];
    while placed.len() < n {
        let mut best: Option<(usize, (u32, u32, u32))> = None;
        for j in 0..n {
            if in_placed[j] {
                continue;
            }
            let conn = adjacency[j].iter().filter(|&&k| in_placed[k]).count() as u32;
            let key = (conn, filter_score[j], adjacency[j].len() as u32);
            // `>` keeps the FIRST (lowest-index) leaf on ties -> deterministic order.
            if best.is_none_or(|(_, bk)| key > bk) {
                best = Some((j, key));
            }
        }
        let (idx, _) = best.expect("an unplaced leaf exists while placed.len() < n");
        placed.push(idx);
        in_placed[idx] = true;
    }
    placed
}

/// Flatten the top-level conjunction of `e` into its leaf conjuncts (a non-`AND` node is
/// one conjunct). Borrows, so the original `on` is left intact to retain on the node.
fn flatten_conjuncts<'e>(e: &'e EvalExpr, out: &mut Vec<&'e EvalExpr>) {
    match e {
        EvalExpr::And(a, b) => {
            flatten_conjuncts(a, out);
            flatten_conjuncts(b, out);
        }
        other => out.push(other),
    }
}

/// If `c` is a hash-usable equijoin conjunct, return its `(left_key, right_key,
/// collation)` — key expressions evaluated against the STANDALONE left / right rows the
/// executor's hash join probes / builds. Usable requires ALL of:
///
///  * a plain `=`: `CmpOp::Eq` with `null_safe: false` (an `IS` differs on NULL, whose
///    semantics the hash's NULL-excludes-match path does not reproduce);
///  * NO affinity coercion (`meta.apply_left`/`apply_right` both `None`). CRITICAL: the
///    executor hashes keys RAW and only re-checks `on` for pairs that already
///    hash-matched, so a coerced equijoin (e.g. INTEGER col = TEXT col) would hash-miss
///    and SILENTLY DROP matching rows — such a term is not a hash key;
///  * one operand referencing ONLY the left side and the other ONLY the right side (in
///    either textual order).
///
/// The left key is taken verbatim (its `Column`s already index the standalone left row,
/// which the executor builds with the same `outer`, so its outer prefix and left columns
/// line up with the combined `on` space). The right key has every `Column(i)` shifted
/// down by `right_shift` (= `base_offset + left_width`): the right subtree's columns
/// begin that many registers earlier in the standalone right row — which the executor
/// builds with an EMPTY outer, its leftmost column at register 0 — than in the combined
/// `outer ++ left ++ right` row the `on` was bound against.
fn equijoin_key(
    c: &EvalExpr,
    ranges: &JoinRanges,
    right_shift: usize,
) -> Option<(EvalExpr, EvalExpr, Collation)> {
    let EvalExpr::Compare { op: CmpOp::Eq, null_safe: false, left, right, meta } = c else {
        return None;
    };
    if meta.apply_left.is_some() || meta.apply_right.is_some() {
        return None;
    }
    let (left_op, right_op) = match (operand_side(left, ranges), operand_side(right, ranges)) {
        (KeySide::LeftOnly, KeySide::RightOnly) => (left.as_ref(), right.as_ref()),
        (KeySide::RightOnly, KeySide::LeftOnly) => (right.as_ref(), left.as_ref()),
        _ => return None,
    };
    let left_key = left_op.clone();
    let mut right_key = right_op.clone();
    rebase_columns(&mut right_key, right_shift);
    Some((left_key, right_key, meta.collation))
}

/// Classify a comparison operand by which side of the join every `Column` it reads falls
/// on. `LeftOnly`/`RightOnly` require at least one column and that EVERY column lies
/// wholly within that side's range; a constant, a mix of sides, a reference outside both
/// ranges (an outer column), or any subquery is `Neither`.
fn operand_side(e: &EvalExpr, ranges: &JoinRanges) -> KeySide {
    let mut saw_left = false;
    let mut saw_right = false;
    let mut saw_other = false;
    mark_sides(e, ranges, &mut saw_left, &mut saw_right, &mut saw_other);
    match (saw_left, saw_right, saw_other) {
        (true, false, false) => KeySide::LeftOnly,
        (false, true, false) => KeySide::RightOnly,
        _ => KeySide::Neither,
    }
}

/// Walk `e`, setting `saw_left`/`saw_right` for each `Column` in the left/right range and
/// `saw_other` for one outside both — or for any subquery, whose correlated column refs
/// live in a separate subplan this walk cannot see or rebase (so an operand containing
/// one is never a hash key). The match is exhaustive so a new [`EvalExpr`] variant is a
/// compile error here rather than a silently-misclassified operand.
fn mark_sides(
    e: &EvalExpr,
    ranges: &JoinRanges,
    saw_left: &mut bool,
    saw_right: &mut bool,
    saw_other: &mut bool,
) {
    match e {
        EvalExpr::Column(i) => {
            if ranges.left.contains(i) {
                *saw_left = true;
            } else if ranges.right.contains(i) {
                *saw_right = true;
            } else {
                *saw_other = true;
            }
        }
        EvalExpr::Literal(_) | EvalExpr::Param(_) | EvalExpr::Now(_) => {}
        EvalExpr::ScalarSubquery(_)
        | EvalExpr::ScalarSubqueryColumn { .. }
        | EvalExpr::Exists { .. } => *saw_other = true,
        EvalExpr::InSubquery { subject, .. } => {
            *saw_other = true;
            mark_sides(subject, ranges, saw_left, saw_right, saw_other);
        }
        EvalExpr::InSubqueryRow { subjects, .. } => {
            *saw_other = true;
            for s in subjects {
                mark_sides(s, ranges, saw_left, saw_right, saw_other);
            }
        }
        EvalExpr::Unary { operand, .. } => mark_sides(operand, ranges, saw_left, saw_right, saw_other),
        EvalExpr::Arith { left, right, .. }
        | EvalExpr::Concat { left, right }
        | EvalExpr::Bitwise { left, right, .. }
        | EvalExpr::Compare { left, right, .. } => {
            mark_sides(left, ranges, saw_left, saw_right, saw_other);
            mark_sides(right, ranges, saw_left, saw_right, saw_other);
        }
        EvalExpr::And(a, b) | EvalExpr::Or(a, b) => {
            mark_sides(a, ranges, saw_left, saw_right, saw_other);
            mark_sides(b, ranges, saw_left, saw_right, saw_other);
        }
        EvalExpr::IsNull(x) | EvalExpr::NotNull(x) => mark_sides(x, ranges, saw_left, saw_right, saw_other),
        EvalExpr::Between { subject, low, high, .. } => {
            mark_sides(subject, ranges, saw_left, saw_right, saw_other);
            mark_sides(low, ranges, saw_left, saw_right, saw_other);
            mark_sides(high, ranges, saw_left, saw_right, saw_other);
        }
        EvalExpr::InList { subject, items, .. } => {
            mark_sides(subject, ranges, saw_left, saw_right, saw_other);
            for it in items {
                mark_sides(it, ranges, saw_left, saw_right, saw_other);
            }
        }
        EvalExpr::Coalesce(items) => {
            for it in items {
                mark_sides(it, ranges, saw_left, saw_right, saw_other);
            }
        }
        EvalExpr::NullIf { left, right, .. } => {
            mark_sides(left, ranges, saw_left, saw_right, saw_other);
            mark_sides(right, ranges, saw_left, saw_right, saw_other);
        }
        EvalExpr::Case { operand, whens, else_expr } => {
            if let Some(o) = operand {
                mark_sides(o, ranges, saw_left, saw_right, saw_other);
            }
            for w in whens {
                mark_sides(&w.when, ranges, saw_left, saw_right, saw_other);
                mark_sides(&w.then, ranges, saw_left, saw_right, saw_other);
            }
            if let Some(e) = else_expr {
                mark_sides(e, ranges, saw_left, saw_right, saw_other);
            }
        }
        EvalExpr::Cast { operand, .. } | EvalExpr::Collate { operand, .. } => {
            mark_sides(operand, ranges, saw_left, saw_right, saw_other)
        }
        EvalExpr::Like { subject, pattern, escape, .. } => {
            mark_sides(subject, ranges, saw_left, saw_right, saw_other);
            mark_sides(pattern, ranges, saw_left, saw_right, saw_other);
            if let Some(e) = escape {
                mark_sides(e, ranges, saw_left, saw_right, saw_other);
            }
        }
        EvalExpr::Func { args, .. } => {
            for a in args {
                mark_sides(a, ranges, saw_left, saw_right, saw_other);
            }
        }
        // A RAISE(...) references no join column, but like a subquery it must never make
        // an operand a hash key: it is trigger-body-only and side-effecting, so mark it as
        // "other" so any operand containing it classifies as `Neither`.
        EvalExpr::Raise { .. } => *saw_other = true,
    }
}

/// Shift every `Column(i)` in `e` down by `shift` (= `base_offset + left_width`), in
/// place. Applied only to a `RightOnly` key operand to rebase it from the combined
/// `outer ++ left ++ right` register space into the standalone right row (whose columns
/// begin `shift` registers earlier — the outer prefix plus the left width). A `RightOnly`
/// operand's columns are all `>= base_offset + left_width == shift`, so the subtraction
/// never underflows (guarded). The match is exhaustive so a new [`EvalExpr`] variant is a
/// compile error here; the subquery arms are unreachable for a `RightOnly` operand (see
/// [`operand_side`]) but handled to keep the rewrite total.
fn rebase_columns(e: &mut EvalExpr, shift: usize) {
    match e {
        EvalExpr::Column(i) => {
            debug_assert!(*i >= shift, "a RightOnly key column is always >= base_offset + left_width");
            *i -= shift;
        }
        EvalExpr::Literal(_) | EvalExpr::Param(_) | EvalExpr::Now(_) => {}
        EvalExpr::ScalarSubquery(_) | EvalExpr::ScalarSubqueryColumn { .. } | EvalExpr::Exists { .. } => {}
        EvalExpr::InSubquery { subject, .. } => rebase_columns(subject, shift),
        EvalExpr::InSubqueryRow { subjects, .. } => {
            for s in subjects {
                rebase_columns(s, shift);
            }
        }
        EvalExpr::Unary { operand, .. } => rebase_columns(operand, shift),
        EvalExpr::Arith { left, right, .. }
        | EvalExpr::Concat { left, right }
        | EvalExpr::Bitwise { left, right, .. }
        | EvalExpr::Compare { left, right, .. } => {
            rebase_columns(left, shift);
            rebase_columns(right, shift);
        }
        EvalExpr::And(a, b) | EvalExpr::Or(a, b) => {
            rebase_columns(a, shift);
            rebase_columns(b, shift);
        }
        EvalExpr::IsNull(x) | EvalExpr::NotNull(x) => rebase_columns(x, shift),
        EvalExpr::Between { subject, low, high, .. } => {
            rebase_columns(subject, shift);
            rebase_columns(low, shift);
            rebase_columns(high, shift);
        }
        EvalExpr::InList { subject, items, .. } => {
            rebase_columns(subject, shift);
            for it in items {
                rebase_columns(it, shift);
            }
        }
        EvalExpr::Coalesce(items) => {
            for it in items {
                rebase_columns(it, shift);
            }
        }
        EvalExpr::NullIf { left, right, .. } => {
            rebase_columns(left, shift);
            rebase_columns(right, shift);
        }
        EvalExpr::Case { operand, whens, else_expr } => {
            if let Some(o) = operand {
                rebase_columns(o, shift);
            }
            for w in whens {
                rebase_columns(&mut w.when, shift);
                rebase_columns(&mut w.then, shift);
            }
            if let Some(e) = else_expr {
                rebase_columns(e, shift);
            }
        }
        EvalExpr::Cast { operand, .. } | EvalExpr::Collate { operand, .. } => rebase_columns(operand, shift),
        EvalExpr::Like { subject, pattern, escape, .. } => {
            rebase_columns(subject, shift);
            rebase_columns(pattern, shift);
            if let Some(e) = escape {
                rebase_columns(e, shift);
            }
        }
        EvalExpr::Func { args, .. } => {
            for a in args {
                rebase_columns(a, shift);
            }
        }
        // A RAISE(...) holds no columns to rebase. Unreachable for a `RightOnly` operand
        // (it is marked `saw_other`, so such an operand is never `RightOnly`), handled to
        // keep the rewrite total.
        EvalExpr::Raise { .. } => {}
    }
}

/// The `on` predicate for one join, bound against the combined `left ++ right` row:
///
/// * `ON expr` → the bound expression.
/// * `USING (cols)` → an `AND`-chain of `left.col = right.col` per column name.
/// * `NATURAL` (no explicit constraint) → `USING` over the column names common to both
///   sides (empty ⇒ a plain cross product).
/// * a comma / `CROSS` / bare `JOIN` with no constraint → `None`.
///
/// This builds ONLY the row-filtering equality. The column COALESCING a `USING`/`NATURAL`
/// join also implies — the shared column appearing once in the output, a bare reference
/// resolving to `COALESCE(left, right)`, and the right copy dropped from `*` — is a
/// separate concern handled elsewhere: [`collect_coalesced`] records the shared left/right
/// register pairs in Phase 1, and `scope.rs`'s coalesced-aware resolvers
/// (`resolve_column_expr` / `resolve_unqualified` / `expand_star`) apply them. Splitting
/// the two keeps this equality a plain table-qualified `Column = Column` (so a hash/index
/// join key is preserved) while a bare shared name still coalesces.
fn build_on(
    ctx: &mut PlanCtx,
    scope: &Scope,
    op: &JoinOperator,
    constraint: Option<&JoinConstraint>,
    left_sources: &[Source],
    right_sources: &[Source],
) -> Result<Option<EvalExpr>> {
    if op.natural {
        // A NATURAL join may not also carry an explicit ON/USING (the grammar rejects
        // it; guard here so a malformed tree is a loud error, not a silent drop).
        if constraint.is_some() {
            return Err(Error::sql("a NATURAL join may not have an ON or USING clause"));
        }
        let names = natural_columns(left_sources, right_sources);
        return build_using(ctx, scope, &names, left_sources, right_sources);
    }
    match constraint {
        Some(JoinConstraint::On(expr)) => Ok(Some(bind_expr(scope, ctx, expr)?)),
        Some(JoinConstraint::Using(cols)) => build_using(ctx, scope, cols, left_sources, right_sources),
        None => Ok(None),
    }
}

/// Build the `AND`-chain of `left.col = right.col` equalities for a `USING`/`NATURAL`
/// join. Each side of every equality is a table-qualified column AST bound through the
/// scope, so the comparison inherits the correct affinity + collation and the
/// `INTEGER PRIMARY KEY` rowid alias — the same machinery an explicit `ON` uses.
/// Returns `None` when there are no columns (an empty NATURAL match = a cross product).
fn build_using(
    ctx: &mut PlanCtx,
    scope: &Scope,
    cols: &[String],
    left_sources: &[Source],
    right_sources: &[Source],
) -> Result<Option<EvalExpr>> {
    let mut acc: Option<EvalExpr> = None;
    for name in cols {
        let left_tab = source_containing(left_sources, name).ok_or_else(|| missing_using(name))?;
        let right_tab = source_containing(right_sources, name).ok_or_else(|| missing_using(name))?;
        let eq_ast = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(column_ast(left_tab, name)),
            right: Box::new(column_ast(right_tab, name)),
        };
        let eq = bind_expr(scope, ctx, &eq_ast)?;
        acc = Some(match acc {
            Some(prev) => EvalExpr::And(Box::new(prev), Box::new(eq)),
            None => eq,
        });
    }
    Ok(acc)
}

/// The column names common to both sides of a NATURAL join, in left-side column order
/// (deduplicated). The rowid is never a natural-join column (only declared columns
/// participate).
fn natural_columns(left_sources: &[Source], right_sources: &[Source]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for lsrc in left_sources {
        for lname in lsrc.column_names() {
            let already = names.iter().any(|n| n.eq_ignore_ascii_case(lname));
            if !already && source_containing(right_sources, lname).is_some() {
                names.push(lname.to_string());
            }
        }
    }
    names
}

/// The exposed name of the first source in `sources` that declares a column named
/// `name` (case-insensitive), or `None` if none does.
fn source_containing<'s>(sources: &'s [Source], name: &str) -> Option<&'s str> {
    sources.iter().find(|s| s.has_column(name)).map(|s| s.exposed_name())
}

/// A table-qualified column reference AST (`table.name`), used to build `USING`
/// equalities that bind unambiguously through the scope.
fn column_ast(table: &str, name: &str) -> Expr {
    Expr::Column { schema: None, table: Some(table.to_string()), name: name.to_string(), from_dqs: false }
}

/// SQLite's error when a `USING` column is absent from one of the joined sides.
fn missing_using(name: &str) -> Error {
    Error::sql(format!("cannot join using column {name} - column not present in both tables"))
}

/// Map the parsed [`JoinKind`] to the plan's [`JoinType`]. A comma join is a cross join.
fn join_type_of(kind: JoinKind) -> JoinType {
    match kind {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Right => JoinType::Right,
        JoinKind::Full => JoinType::Full,
        JoinKind::Cross => JoinType::Cross,
        JoinKind::Comma => JoinType::Cross,
    }
}

/// A source by index, with a loud internal error if the tree walk and the resolved
/// source list ever disagree (they are produced from the same tree, so this cannot
/// fire in practice — it turns a would-be panic into a diagnosable error).
fn source_at<'a>(scope: &Scope<'a>, idx: usize) -> Result<&'a Source<'a>> {
    scope
        .sources
        .get(idx)
        .ok_or_else(|| Error::sql("internal error: join tree references a missing source"))
}

#[cfg(test)]
mod tests {
    use minisqlite_catalog::{Catalog, ColumnDef, IndexDef, TableDef};
    use minisqlite_expr::{CmpOp, EvalExpr};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop, FromClause, SelectBody, SelectCore, Statement};
    use minisqlite_types::Result;

    use super::resolve_from;
    use crate::access::{RowidOp, RowidScan};
    use crate::{CtePlan, Join, JoinStrategy, JoinType, Plan, Planner, PlanNode, QueryPlanner};

    // A static, read-only test catalog (the write/load paths are never exercised
    // here). Mirrors the ~25-line pattern in `src/tests.rs`; kept local so this file
    // owns its own fixtures and never edits tests.rs.
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

    /// `t(a,b)` · `u(c,d)` · `v(e,f)` · `w(a,g)` — `w` shares `a` with `t` for
    /// USING/NATURAL; each table is 2 columns (width 3 = `[c0, c1, rowid]`).
    fn cat() -> TestCatalog {
        TestCatalog {
            tables: vec![
                tdef("t", vec![col("a", "INTEGER"), col("b", "TEXT")]),
                tdef("u", vec![col("c", "INTEGER"), col("d", "TEXT")]),
                tdef("v", vec![col("e", "INTEGER"), col("f", "TEXT")]),
                tdef("w", vec![col("a", "INTEGER"), col("g", "TEXT")]),
            ],
        }
    }

    fn plan(sql: &str) -> Plan {
        let c = cat();
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, &c).expect("plan ok")
    }

    /// The operator directly under the root Project (the FROM subtree, after any WHERE
    /// Filter is stripped when present).
    fn under_project(root: &PlanNode) -> &PlanNode {
        match root {
            PlanNode::Project { input, .. } => input,
            other => panic!("expected Project at root, got {other:?}"),
        }
    }

    fn expect_join(n: &PlanNode) -> &Join {
        match n {
            PlanNode::Join(j) => j,
            other => panic!("expected Join, got {other:?}"),
        }
    }

    fn expect_rowidscan(n: &PlanNode) -> &RowidScan {
        match n {
            PlanNode::RowidScan(s) => s,
            other => panic!("expected RowidScan, got {other:?}"),
        }
    }

    /// Assert `e` is `Compare(op, Column(l), Column(r))` (a column=column equijoin ON).
    fn assert_col_cmp(e: &EvalExpr, op: CmpOp, l: usize, r: usize) {
        match e {
            EvalExpr::Compare { op: got, left, right, .. } => {
                assert_eq!(*got, op, "compare op");
                assert!(matches!(left.as_ref(), EvalExpr::Column(i) if *i == l), "left Column({l}), got {left:?}");
                assert!(matches!(right.as_ref(), EvalExpr::Column(i) if *i == r), "right Column({r}), got {right:?}");
            }
            other => panic!("expected a Column-vs-Column Compare, got {other:?}"),
        }
    }

    /// Assert `e` is exactly `Column(n)`.
    fn assert_column(e: &EvalExpr, n: usize) {
        assert!(matches!(e, EvalExpr::Column(i) if *i == n), "expected Column({n}), got {e:?}");
    }

    /// The Hash strategy's `(left_keys, right_keys, key_collations.len())`, or a panic if
    /// the join is not a hash join.
    fn expect_hash(j: &Join) -> (&[EvalExpr], &[EvalExpr], usize) {
        match &j.strategy {
            JoinStrategy::Hash { left_keys, right_keys, key_collations } => {
                assert_eq!(
                    left_keys.len(),
                    right_keys.len(),
                    "left/right key vectors are parallel"
                );
                assert_eq!(
                    left_keys.len(),
                    key_collations.len(),
                    "one collation per key pair"
                );
                (left_keys.as_slice(), right_keys.as_slice(), key_collations.len())
            }
            other => panic!("expected a Hash strategy, got {other:?}"),
        }
    }

    /// Assert `j` is a single-key Hash whose equi-key equates COMBINED-space `Column(l)`
    /// (the left key, taken verbatim) with `Column(r)` (the right operand), and that the
    /// key was STRIPPED from `on` (a pure equijoin leaves no residual). The right key is
    /// stored rebased into the standalone right row, so it reads as `Column(r -
    /// j.left_width)` (every test here is at `base_offset == 0`, where the rebase is
    /// exactly `left_width`). This pins the same registers the old full-`on` `Compare`
    /// did, now that the equi-key lives in the hash strategy and `on` is `None`.
    fn assert_hash_equikey_combined(j: &Join, l: usize, r: usize) {
        let (lk, rk, ncoll) = expect_hash(j);
        assert_eq!(lk.len(), 1, "one hash key");
        assert_eq!(ncoll, 1, "one collation");
        assert_column(&lk[0], l);
        assert_column(&rk[0], r - j.left_width);
        assert!(
            j.on.is_none(),
            "a pure equijoin strips its only conjunct → residual `on` is None, got {:?}",
            j.on
        );
    }

    #[test]
    fn inner_join_on_builds_left_deep_join() {
        // t(a,b) width 3, u(c,d) width 3. `t.a=u.c` binds to Column(0) vs Column(3).
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Inner));
        assert_eq!(j.left_width, 3);
        assert_eq!(j.right_width, 3);
        // `t.a = u.c` is a hash-safe equijoin (INTEGER = INTEGER, no affinity coercion),
        // so the planner picks Hash over the O(left*right) nested loop.
        let (lk, rk, ncoll) = expect_hash(j);
        assert_eq!(lk.len(), 1);
        assert_column(&lk[0], 0); // t.a
        assert_column(&rk[0], 0); // u.c: combined index 3, rebased by -left_width(3) → 0
        assert_eq!(ncoll, 1);
        // The consumed equi-key is STRIPPED from `on` (a pure equijoin → residual None);
        // the equality now lives in the hash keys asserted above, not a per-pair recheck.
        assert!(j.on.is_none(), "pure equijoin strips `on` to None, got {:?}", j.on);
        // SELECT * over the two 2-col tables projects a,b,c,d (rowids excluded).
        assert_eq!(p.result_columns, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn equijoin_rebases_the_right_key_by_left_width() {
        // The right key's `Column`s are shifted down by `left_width` so they index the
        // STANDALONE right row. u.d is u's SECOND column: combined index 4, rebased to
        // Column(1). This pins the general rebase offset.
        //
        // NOTE: an earlier draft used `t.a = u.d` for this, but t.a is INTEGER and u.d TEXT,
        // which applies NUMERIC affinity to the TEXT side (CompareMeta.apply_right =
        // Some) — that pair is NOT hash-safe and is the affinity-guard case
        // (`an_affinity_coercing_equijoin_stays_nested_loop`). `t.b = u.d` (TEXT = TEXT)
        // exercises the identical rebase offset with a genuinely hash-safe equijoin.
        let p = plan("SELECT * FROM t JOIN u ON t.b = u.d");
        let j = expect_join(under_project(&p.root));
        let (lk, rk, _) = expect_hash(j);
        assert_column(&lk[0], 1); // t.b (combined index 1, left side, unchanged)
        assert_column(&rk[0], 1); // u.d: combined index 4, rebased by -3 → 1
    }

    #[test]
    fn equijoin_classifies_by_operand_content_not_position() {
        // Operands reversed (`u.c = t.a`): classification is by CONTENT, so the left key
        // is still t.a and the right key still u.c regardless of textual order.
        let p = plan("SELECT * FROM t JOIN u ON u.c = t.a");
        let j = expect_join(under_project(&p.root));
        let (lk, rk, _) = expect_hash(j);
        assert_column(&lk[0], 0); // t.a
        assert_column(&rk[0], 0); // u.c rebased 3 → 0
    }

    #[test]
    fn multi_key_equijoin_builds_parallel_key_vectors() {
        // Two hash-safe equijoins (INTEGER=INTEGER and TEXT=TEXT) yield two key pairs, in
        // discovery order, each right key rebased by -left_width.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c AND t.b = u.d");
        let j = expect_join(under_project(&p.root));
        let (lk, rk, ncoll) = expect_hash(j);
        assert_eq!(lk.len(), 2);
        assert_eq!(ncoll, 2);
        assert_column(&lk[0], 0); // t.a
        assert_column(&lk[1], 1); // t.b
        assert_column(&rk[0], 0); // u.c rebased 3 → 0
        assert_column(&rk[1], 1); // u.d rebased 4 → 1
        // Both conjuncts became hash keys → the residual `on` is empty (None).
        assert!(j.on.is_none(), "multi-key equijoin strips every conjunct, got {:?}", j.on);
    }

    #[test]
    fn only_the_equality_conjunct_becomes_a_hash_key() {
        // `t.a = u.c AND t.a > u.c`: the `=` is one hash key (stripped from `on`); the `>`
        // is not an equality, so it survives as the RESIDUAL the executor re-checks.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c AND t.a > u.c");
        let j = expect_join(under_project(&p.root));
        let (lk, rk, ncoll) = expect_hash(j);
        assert_eq!(lk.len(), 1, "one hash key from the `=` only");
        assert_eq!(ncoll, 1);
        assert_column(&lk[0], 0);
        assert_column(&rk[0], 0);
        // Only the residual `>` remains on the node — the `=` moved into the hash key
        // above — so `on` is exactly the surviving `t.a > u.c` (Column(0) > Column(3)).
        assert_col_cmp(j.on.as_ref().expect("residual `>` retained"), CmpOp::Gt, 0, 3);
    }

    #[test]
    fn equijoin_and_coerced_equality_keeps_the_coerced_as_residual() {
        // `t.a = u.c AND t.a = u.d`: `t.a=u.c` is INT=INT → the hash key (stripped). The
        // SECOND `=` is INT (t.a) vs TEXT (u.d), which applies NUMERIC affinity — NOT a
        // hash key (raw hashing would MISS coerced matches), so it MUST survive as the
        // residual `on`. This is the safety net: dropping it would silently over-emit.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c AND t.a = u.d");
        let j = expect_join(under_project(&p.root));
        let (lk, rk, ncoll) = expect_hash(j);
        assert_eq!(lk.len(), 1, "only the un-coerced `=` is a hash key");
        assert_eq!(ncoll, 1);
        assert_column(&lk[0], 0); // t.a
        assert_column(&rk[0], 0); // u.c rebased 3 → 0
        // The coerced `t.a = u.d` (Column(0) = Column(4)) stays as the residual, WITH its
        // affinity coercion — the executor re-checks it, catching matches raw hashing missed.
        match j.on.as_ref().expect("coerced `=` retained as residual") {
            EvalExpr::Compare { op: CmpOp::Eq, null_safe: false, left, right, meta } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(0)), "left is t.a, got {left:?}");
                assert!(matches!(right.as_ref(), EvalExpr::Column(4)), "right is u.d, got {right:?}");
                assert!(
                    meta.apply_left.is_some() || meta.apply_right.is_some(),
                    "the retained `=` is the affinity-coercing one, got {meta:?}"
                );
            }
            other => panic!("expected a coerced Eq residual, got {other:?}"),
        }
    }

    #[test]
    fn equijoin_and_is_keeps_the_null_safe_equality_as_residual() {
        // `t.a = u.c AND t.b IS u.d`: `t.a=u.c` is the hash key (stripped). `t.b IS u.d`
        // is a NULL-safe equality (`null_safe: true`) — the hash's NULL-excludes-match
        // path cannot reproduce `NULL IS NULL = true`, so IS is never a hash key and MUST
        // survive as the residual.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c AND t.b IS u.d");
        let j = expect_join(under_project(&p.root));
        let (lk, _rk, _n) = expect_hash(j);
        assert_eq!(lk.len(), 1, "only the plain `=` is a hash key");
        assert_column(&lk[0], 0); // t.a
        // The IS survives as the residual: a null-safe Eq on t.b (Column(1)) / u.d (Column(4)).
        match j.on.as_ref().expect("IS retained as residual") {
            EvalExpr::Compare { op: CmpOp::Eq, null_safe: true, left, right, .. } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(1)), "left is t.b, got {left:?}");
                assert!(matches!(right.as_ref(), EvalExpr::Column(4)), "right is u.d, got {right:?}");
            }
            other => panic!("expected a null-safe Eq (IS) residual, got {other:?}"),
        }
    }

    #[test]
    fn a_non_equijoin_stays_nested_loop() {
        // `t.a > u.c` has no equality conjunct, so there is no hash key.
        let p = plan("SELECT * FROM t JOIN u ON t.a > u.c");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.strategy, JoinStrategy::NestedLoop), "got {:?}", j.strategy);
    }

    #[test]
    fn a_cross_join_stays_nested_loop() {
        // A comma join has no `on`, so no equijoin and thus no hash strategy.
        let p = plan("SELECT * FROM t, u");
        let j = expect_join(under_project(&p.root));
        assert!(j.on.is_none(), "a comma join has no ON predicate");
        assert!(matches!(j.strategy, JoinStrategy::NestedLoop), "got {:?}", j.strategy);
    }

    #[test]
    fn a_left_join_equijoin_is_hash_and_keeps_its_join_type() {
        // Hash is correct for LEFT (the executor null-fills unmatched left rows); the
        // strategy choice does not depend on the join type, only on the equijoin key.
        let p = plan("SELECT * FROM t LEFT JOIN u ON t.a = u.c");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Left), "join type preserved");
        let (lk, rk, _) = expect_hash(j);
        assert_column(&lk[0], 0); // t.a
        assert_column(&rk[0], 0); // u.c rebased 3 → 0
    }

    #[test]
    fn an_affinity_coercing_equijoin_stays_nested_loop() {
        // t.a is INTEGER, u.d is TEXT: the `=` applies NUMERIC affinity to the TEXT side
        // (CompareMeta.apply_right = Some), so it is NOT hash-safe — the executor hashes
        // keys RAW, so hashing this pair would silently drop matching rows. It must stay
        // a nested loop, with the coercion applied by the `on` recheck.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.d");
        let j = expect_join(under_project(&p.root));
        assert!(
            matches!(j.strategy, JoinStrategy::NestedLoop),
            "an affinity-coercing equijoin must not become a hash key, got {:?}",
            j.strategy
        );
        // The ON is still present and correct (only the physical strategy differs).
        assert_col_cmp(j.on.as_ref().expect("an ON predicate"), CmpOp::Eq, 0, 4);
    }

    #[test]
    fn rowid_equijoin_uses_index_nested_loop() {
        // `u.rowid = t.a` seeks u by rowid keyed on the left column t.a (INTEGER, no
        // affinity coercion): the right leaf is a per-left RowidScan and the strategy is
        // IndexNestedLoop — no right materialization, O(left * log right).
        let p = plan("SELECT * FROM t JOIN u ON u.rowid = t.a");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Inner));
        assert_eq!(j.left_width, 3);
        assert_eq!(j.right_width, 3);
        assert!(
            matches!(j.strategy, JoinStrategy::IndexNestedLoop),
            "a right-rowid equijoin is an index nested loop, got {:?}",
            j.strategy
        );
        // The left side stays a full scan; the right side becomes a rowid seek on u
        // whose key is the left column t.a (Column(0), read from the outer/left row).
        assert!(matches!(j.left.as_ref(), PlanNode::SeqScan(_)), "left is a scan, got {:?}", j.left);
        let scan = expect_rowidscan(j.right.as_ref());
        assert_eq!(scan.table, "u");
        assert_eq!(scan.column_count, 2);
        match &scan.op {
            RowidOp::Eq(key) => assert_column(key, 0), // t.a in the outer (left) row
            other => panic!("expected a rowid Eq seek, got {other:?}"),
        }
        // The FULL ON is retained (the executor re-checks it on every seek hit).
        assert_col_cmp(j.on.as_ref().expect("an ON predicate"), CmpOp::Eq, 5, 0);
    }

    #[test]
    fn rowid_equijoin_reversed_operands_still_index_nested_loop() {
        // Operand order does not matter: `t.a = u.rowid` seeks u by rowid just the same.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.rowid");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.strategy, JoinStrategy::IndexNestedLoop), "got {:?}", j.strategy);
        let scan = expect_rowidscan(j.right.as_ref());
        match &scan.op {
            RowidOp::Eq(key) => assert_column(key, 0),
            other => panic!("expected a rowid Eq seek, got {other:?}"),
        }
    }

    #[test]
    fn rowid_equijoin_on_a_text_key_uses_index_nested_loop() {
        // `u.rowid = t.b` keys the rowid seek on t.b (TEXT), so the comparison applies
        // NUMERIC affinity to the t.b operand (CompareMeta.apply_right = Some). This is
        // STILL a per-left rowid seek, not a full O(left*right) scan: the executor's
        // `eval_rowid_eq` coerces the seek value via OP_MustBeInt — the SAME rule `on`
        // applies — so `must_be_int(t.b) == Int(R)` iff `on`'s `u.rowid == NUMERIC(t.b)`
        // holds, and the IndexNestedLoop re-checks the full `on` after each seek. The seek
        // hits exactly the row `on` keeps. This mirrors `rowid_equijoin_uses_index_nested_loop`
        // (keyed on the INTEGER column t.a); the only difference is the left column's
        // affinity, which the executor now handles identically — so the planner must not
        // gate the seek on it (regression guard for the removed `apply_*.is_none()` check).
        let p = plan("SELECT * FROM t JOIN u ON u.rowid = t.b");
        let j = expect_join(under_project(&p.root));
        assert!(
            matches!(j.strategy, JoinStrategy::IndexNestedLoop),
            "a text-keyed rowid equijoin still seeks (executor coerces), got {:?}",
            j.strategy
        );
        assert!(matches!(j.left.as_ref(), PlanNode::SeqScan(_)), "left is a scan, got {:?}", j.left);
        let scan = expect_rowidscan(j.right.as_ref());
        assert_eq!(scan.table, "u");
        match &scan.op {
            RowidOp::Eq(key) => assert_column(key, 1), // t.b in the outer (left) row
            other => panic!("expected a rowid Eq seek, got {other:?}"),
        }
        // The FULL ON is retained; the INL re-checks it on every seek hit (the backstop).
        assert_col_cmp(j.on.as_ref().expect("an ON predicate"), CmpOp::Eq, 5, 1);
    }

    #[test]
    fn left_join_rowid_equijoin_uses_index_nested_loop() {
        // IndexNestedLoop is valid for LEFT (the executor null-fills a missing seek).
        let p = plan("SELECT * FROM t LEFT JOIN u ON u.rowid = t.a");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Left), "join type preserved");
        assert!(matches!(j.strategy, JoinStrategy::IndexNestedLoop), "got {:?}", j.strategy);
        assert_eq!(expect_rowidscan(j.right.as_ref()).table, "u");
    }

    #[test]
    fn a_right_join_on_rowid_is_not_index_nested_loop() {
        // The executor rejects IndexNestedLoop for Right/Full, so a right-rowid equijoin
        // under RIGHT JOIN must NOT pick it. It stays a keyed strategy though: `u.rowid =
        // t.a` is still a hash-safe equijoin, so it falls back to Hash (right leaf a
        // scan), never IndexNestedLoop.
        let p = plan("SELECT * FROM t RIGHT JOIN u ON u.rowid = t.a");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Right));
        assert!(
            !matches!(j.strategy, JoinStrategy::IndexNestedLoop),
            "IndexNestedLoop is forbidden for RIGHT, got {:?}",
            j.strategy
        );
        assert!(matches!(j.right.as_ref(), PlanNode::SeqScan(_)), "right stays a scan for Right join");
    }

    #[test]
    fn a_regular_column_equijoin_is_not_index_nested_loop() {
        // `t.a = u.c` equates two ordinary columns (u.c is not u's rowid), so there is no
        // rowid seek — it is a Hash join, not IndexNestedLoop.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c");
        let j = expect_join(under_project(&p.root));
        assert!(
            matches!(j.strategy, JoinStrategy::Hash { .. }),
            "a non-rowid equijoin is a hash join, got {:?}",
            j.strategy
        );
    }

    #[test]
    fn left_join_maps_to_join_type_left() {
        let p = plan("SELECT * FROM t LEFT JOIN u ON t.a = u.c");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Left));
        // The equi-key t.a=u.c is stripped into the hash strategy (residual `on` is None).
        assert_hash_equikey_combined(j, 0, 3);
    }

    #[test]
    fn comma_join_is_cross_with_no_on() {
        let p = plan("SELECT * FROM t, u");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Cross));
        assert!(j.on.is_none(), "a comma join has no ON predicate");
        assert_eq!(j.left_width, 3);
        assert_eq!(j.right_width, 3);
    }

    #[test]
    fn using_builds_an_equality_on() {
        // t(a,b) and w(a,g) share `a`: USING(a) => `t.a = w.a` => Column(0) vs Column(3).
        // That synthesized equality is a hash key (INTEGER=INTEGER), so it lives in the
        // hash strategy (`on` stripped to None) rather than a retained `on` Compare.
        let p = plan("SELECT * FROM t JOIN w USING (a)");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Inner));
        assert_hash_equikey_combined(j, 0, 3);
    }

    #[test]
    fn natural_join_builds_equality_on_common_column() {
        // t(a,b) NATURAL JOIN w(a,g): the only common column is `a` => `t.a = w.a`, a hash
        // key (INTEGER=INTEGER) that lives in the hash strategy (`on` stripped to None).
        let p = plan("SELECT * FROM t NATURAL JOIN w");
        let j = expect_join(under_project(&p.root));
        assert!(matches!(j.join_type, JoinType::Inner));
        assert_hash_equikey_combined(j, 0, 3);
    }

    #[test]
    fn natural_join_with_no_common_column_is_a_cross() {
        // t(a,b) and u(c,d) share nothing => NATURAL degrades to a cross product.
        let p = plan("SELECT * FROM t NATURAL JOIN u");
        let j = expect_join(under_project(&p.root));
        assert!(j.on.is_none(), "no common columns => no ON => cross product");
    }

    /// The FROM clause of a parsed single-SELECT query, for exercising Phase-1
    /// [`resolve_from`] directly (its `(sources, coalesced)` output) without compiling a
    /// whole plan.
    fn from_of(sql: &str) -> Option<FromClause> {
        let ast = parse(sql).expect("parse ok");
        let Statement::Select(sel) = ast.statements.into_iter().next().expect("one statement")
        else {
            panic!("not a SELECT: {sql}");
        };
        // Match by reference and clone: `SelectBody` implements `Drop` (iterative
        // teardown in `minisqlite-sql`), so its fields cannot be moved out by pattern.
        let SelectBody::Select(SelectCore::Query { from, .. }) = &sel.body else {
            panic!("not a query core: {sql}");
        };
        from.clone()
    }

    #[test]
    fn resolve_from_records_the_using_coalesced_register_pair() {
        // t(a,b)@0 [a=0,b=1,rowid=2] JOIN w(a,g)@3 [a=3,g=4,rowid=5] USING(a): the shared
        // column `a` coalesces to ONE output column pairing the LEFT copy (reg 0) with the
        // RIGHT copy (reg 3). This localizes `collect_coalesced`'s register math — a wrong
        // left/right would otherwise only surface indirectly as a wrong conformance value.
        let cat = cat();
        let (_sources, coalesced) =
            resolve_from(&cat, &from_of("SELECT * FROM t JOIN w USING (a)"), 0).expect("resolve");
        assert_eq!(coalesced.len(), 1, "one shared column, got {coalesced:?}");
        assert_eq!(coalesced[0].name, "a");
        assert_eq!((coalesced[0].left, coalesced[0].right), (0, 3));
    }

    #[test]
    fn resolve_from_records_natural_common_column_but_not_an_on_join() {
        // t(a,b) NATURAL JOIN w(a,g): `a` is the only common name -> one Coalesced pair
        // {left:0,right:3}; b/g are not shared, so nothing else coalesces.
        let cat = cat();
        let (_s, nat) = resolve_from(&cat, &from_of("SELECT * FROM t NATURAL JOIN w"), 0).unwrap();
        assert_eq!(nat.len(), 1, "only `a` is common, got {nat:?}");
        assert_eq!((nat[0].name.as_str(), nat[0].left, nat[0].right), ("a", 0, 3));
        // An explicit ON join coalesces nothing (its shared name STAYS ambiguous by design).
        let (_s2, on) =
            resolve_from(&cat, &from_of("SELECT * FROM t JOIN w ON t.a = w.a"), 0).unwrap();
        assert!(on.is_empty(), "an ON join records no coalesced columns, got {on:?}");
    }

    #[test]
    fn resolve_from_offsets_coalesced_registers_by_base_offset() {
        // The Coalesced registers are ABSOLUTE: at base_offset 10, t.a is reg 10 and w.a is
        // reg 13 (t occupies [10,13) incl. rowid). Pins that base_offset threads through so a
        // correlated/nested placement doesn't mis-register the coalesce operands.
        let cat = cat();
        let (_s, c) =
            resolve_from(&cat, &from_of("SELECT * FROM t JOIN w USING (a)"), 10).unwrap();
        assert_eq!((c[0].left, c[0].right), (10, 13), "registers carry base_offset, got {c:?}");
    }

    #[test]
    fn three_table_join_nests_left_deep() {
        // t JOIN u ON t.a=u.c JOIN v ON u.d=v.e. Top: left=(t⋈u) width 6, right=v.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c JOIN v ON u.d = v.e");
        let top = expect_join(under_project(&p.root));
        assert_eq!(top.left_width, 6, "left combined width is t(3)+u(3)");
        assert_eq!(top.right_width, 3);
        // u.d is register 4 (u@3, col 1); v.e is register 6 (v@6, col 0).
        assert_col_cmp(top.on.as_ref().expect("top ON"), CmpOp::Eq, 4, 6);
        // The left child is itself the inner t⋈u join; its `t.a=u.c` equi-key (INTEGER=
        // INTEGER) is a hash key stripped from `on` (the top `u.d=v.e` is TEXT=INTEGER, a
        // coerced compare that is NOT a hash key, so it stays on the node — asserted above).
        let inner = expect_join(top.left.as_ref());
        assert_eq!(inner.left_width, 3);
        assert_eq!(inner.right_width, 3);
        assert_hash_equikey_combined(inner, 0, 3);
    }

    #[test]
    fn where_over_join_folds_into_the_join() {
        // For an INNER join a single-table WHERE `t.b = 'x'` folds into the join node (ON
        // and WHERE are equivalent there) rather than sitting in a separate Filter above
        // it; the `t.a = u.c` equijoin is the hash key.
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c WHERE t.b = 'x'");
        let j = expect_join(under_project(&p.root));
        match j.on.as_ref().expect("WHERE folded into the join's ON") {
            EvalExpr::Compare { left, .. } => {
                assert!(matches!(left.as_ref(), EvalExpr::Column(1)), "t.b is register 1, got {left:?}");
            }
            other => panic!("expected a Compare ON, got {other:?}"),
        }
    }

    #[test]
    fn qualified_where_column_over_join_binds_to_the_right_side() {
        // A single right-table WHERE `u.d = 'x'` pushes down onto the right input as a
        // Filter, where u.d is the right table's LOCAL register 1 (u.c=0, u.d=1).
        let p = plan("SELECT * FROM t JOIN u ON t.a = u.c WHERE u.d = 'x'");
        let j = expect_join(under_project(&p.root));
        match j.right.as_ref() {
            PlanNode::Filter { predicate, .. } => match predicate {
                EvalExpr::Compare { left, .. } => {
                    assert!(matches!(left.as_ref(), EvalExpr::Column(1)), "u.d is right-local register 1, got {left:?}");
                }
                other => panic!("expected Compare, got {other:?}"),
            },
            other => panic!("expected a Filter pushed onto the right input, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_join_column_is_a_loud_error() {
        // `a` exists in both t and w: an unqualified reference is ambiguous.
        let c = cat();
        let ast = parse("SELECT a FROM t JOIN w ON t.a = w.a").unwrap();
        let stmt = ast.statements.first().unwrap();
        let err = QueryPlanner::new().plan(stmt, &c).unwrap_err();
        assert!(format!("{err:?}").contains("ambiguous"), "got {err:?}");
    }

    #[test]
    fn using_missing_column_is_a_loud_error() {
        // `z` is in neither table.
        let c = cat();
        let ast = parse("SELECT * FROM t JOIN u USING (z)").unwrap();
        let stmt = ast.statements.first().unwrap();
        let err = QueryPlanner::new().plan(stmt, &c).unwrap_err();
        assert!(format!("{err:?}").contains("cannot join using column"), "got {err:?}");
    }

    #[test]
    fn single_table_from_is_unchanged() {
        // No join: a bare table still compiles to a plain scan under the Project.
        let p = plan("SELECT a, b FROM t");
        assert!(
            matches!(under_project(&p.root), PlanNode::SeqScan(_)),
            "single table stays a SeqScan"
        );
    }

    // ----- Derived tables: FROM (SELECT ...) [AS] alias -----

    /// `p.ctes[0]` as a `Materialized { name, column_count }`, returning `(name,
    /// column_count, body)` or a panic. Pins the single-derived-table registration.
    fn only_materialized(p: &Plan) -> (&str, usize, &PlanNode) {
        assert_eq!(p.ctes.len(), 1, "expected exactly one registered CTE/derived table");
        match &p.ctes[0] {
            CtePlan::Materialized { name, column_count, body } => (name.as_str(), *column_count, body),
            other => panic!("expected a Materialized CTE, got {other:?}"),
        }
    }

    #[test]
    fn derived_sole_from_is_a_ctescan_over_a_materialized_cte() {
        // FROM (SELECT a, b FROM t) d : d exposes a,b (width 2, NO rowid). Its access
        // leaf is a CteScan over ONE Materialized CTE whose body scans t; the outer
        // projection binds d.a -> Column(0), d.b -> Column(1).
        let p = plan("SELECT a, b FROM (SELECT a, b FROM t) d");

        let (name, cc, body) = only_materialized(&p);
        assert_eq!(name, "d", "the CTE takes the derived table's exposed name");
        assert_eq!(cc, 2, "a,b -> width 2 (no trailing rowid)");
        // The body is the inner SELECT's plan: a Project over a scan of t.
        assert!(
            matches!(under_project(body), PlanNode::SeqScan(s) if s.table == "t"),
            "derived body scans t, got {body:?}"
        );

        // The FROM access leaf is a CteScan referencing ctes[0] with the same width.
        match under_project(&p.root) {
            PlanNode::CteScan { id, column_count } => {
                assert_eq!(*id, 0, "references ctes[0]");
                assert_eq!(*column_count, 2);
            }
            other => panic!("expected a CteScan leaf, got {other:?}"),
        }

        // The outer projection reads the derived row directly (base 0): a,b at Column 0,1.
        match &p.root {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 2);
                assert_column(&exprs[0], 0);
                assert_column(&exprs[1], 1);
            }
            other => panic!("expected a Project root, got {other:?}"),
        }
        assert_eq!(p.result_columns, vec!["a", "b"]);
    }

    #[test]
    fn derived_column_binds_by_its_exposed_alias_not_the_base_name() {
        // Inner aliases rename the outputs: (SELECT a AS x, b AS y FROM t) d exposes x,y.
        // Selecting `y` binds to the SECOND derived column (Column(1)), proving name
        // resolution goes through the derived schema — not the underlying table.
        let p = plan("SELECT y FROM (SELECT a AS x, b AS y FROM t) d");
        assert_eq!(only_materialized(&p).1, 2, "x,y -> width 2");
        match &p.root {
            PlanNode::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
                assert_column(&exprs[0], 1); // y is the 2nd derived column
            }
            other => panic!("expected a Project root, got {other:?}"),
        }
        assert_eq!(p.result_columns, vec!["y"]);

        // The underlying base name `a` is NOT visible through the aliased derived table.
        let c = cat();
        let ast = parse("SELECT a FROM (SELECT a AS x FROM t) d").unwrap();
        let stmt = ast.statements.first().unwrap();
        let err = QueryPlanner::new().plan(stmt, &c).unwrap_err();
        assert!(format!("{err:?}").contains("no such column"), "got {err:?}");
    }

    #[test]
    fn derived_inner_star_expands_the_inner_base_schema() {
        // (SELECT * FROM t) d : the inner `*` expands to t's a,b (rowid excluded), so the
        // derived table is width 2 and the outer `*` projects a,b.
        let p = plan("SELECT * FROM (SELECT * FROM t) d");
        assert_eq!(only_materialized(&p).1, 2);
        match under_project(&p.root) {
            PlanNode::CteScan { column_count, .. } => assert_eq!(*column_count, 2),
            other => panic!("expected a CteScan leaf, got {other:?}"),
        }
        assert_eq!(p.result_columns, vec!["a", "b"]);
    }

    #[test]
    fn derived_from_values_names_columns_positionally() {
        // A VALUES core derived table names its columns column1.. (SQLite's convention),
        // and its width is the row arity. `(VALUES (1,2),(3,4)) v` -> column1, column2.
        let p = plan("SELECT column1, column2 FROM (VALUES (1, 2), (3, 4)) v");
        assert_eq!(only_materialized(&p).1, 2, "VALUES arity 2 -> width 2");
        match &p.root {
            PlanNode::Project { exprs, .. } => {
                assert_column(&exprs[0], 0);
                assert_column(&exprs[1], 1);
            }
            other => panic!("expected a Project root, got {other:?}"),
        }
        assert_eq!(p.result_columns, vec!["column1", "column2"]);
    }

    #[test]
    fn derived_joined_to_base_table_offsets_registers() {
        // (SELECT c, d FROM u) x JOIN t ON x.c = t.a : the derived x is the LEFT leaf at
        // base 0 (width 2, a CteScan); t is the right base table at base 2 (width 3, incl
        // rowid). x.c -> Column(0), t.a -> Column(2). This pins the register offsets a
        // derived-in-join produces (the strategy is the join planner's contract, not asserted).
        let p = plan("SELECT * FROM (SELECT c, d FROM u) x JOIN t ON x.c = t.a");
        assert_eq!(only_materialized(&p).0, "x");

        let j = expect_join(under_project(&p.root));
        assert_eq!(j.left_width, 2, "derived x is width 2 (no rowid)");
        assert_eq!(j.right_width, 3, "base t is width 3 (a, b, rowid)");
        assert!(
            matches!(j.left.as_ref(), PlanNode::CteScan { id: 0, column_count: 2 }),
            "left leaf is the derived CteScan, got {:?}",
            j.left
        );
        // x.c (derived, INTEGER-inherited) = t.a (INTEGER) is a hash key; the register
        // offsets it pins now live in the hash keys (x.c=Column(0), t.a=Column(2)).
        assert_hash_equikey_combined(j, 0, 2);
        assert_eq!(p.result_columns, vec!["c", "d", "a", "b"]);
    }

    #[test]
    fn base_table_joined_to_derived_offsets_registers() {
        // Symmetric: t JOIN (SELECT c, d FROM u) x — t is the LEFT base at base 0 (width
        // 3), the derived x is the RIGHT leaf at base 3 (width 2, a CteScan). t.a ->
        // Column(0), x.c -> Column(3). A derived right side never takes the rowid
        // index-nested-loop path (it has no rowid).
        let p = plan("SELECT * FROM t JOIN (SELECT c, d FROM u) x ON t.a = x.c");
        let j = expect_join(under_project(&p.root));
        assert_eq!(j.left_width, 3);
        assert_eq!(j.right_width, 2);
        assert!(matches!(j.left.as_ref(), PlanNode::SeqScan(s) if s.table == "t"), "left base scan");
        assert!(
            matches!(j.right.as_ref(), PlanNode::CteScan { id: 0, column_count: 2 }),
            "right leaf is the derived CteScan, got {:?}",
            j.right
        );
        assert!(
            !matches!(j.strategy, JoinStrategy::IndexNestedLoop),
            "a derived right has no rowid, so never IndexNestedLoop, got {:?}",
            j.strategy
        );
        // t.a=x.c is a hash key; its register offsets (t.a=Column(0), x.c=Column(3)) now
        // live in the hash keys, `on` stripped to None.
        assert_hash_equikey_combined(j, 0, 3);
        assert_eq!(p.result_columns, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn nested_derived_tables_register_inner_cte_first() {
        // (SELECT a FROM (SELECT a, b FROM t) inner) outer : the INNER derived table
        // compiles and registers its CtePlan (id 0) BEFORE the outer captures its own id
        // (1). The outer's CteScan references id 1; its body contains the inner's CteScan
        // referencing id 0. This is the id-ordering the compile_derived doc guarantees.
        let p = plan("SELECT a FROM (SELECT a FROM (SELECT a, b FROM t) inr) outr");
        assert_eq!(p.ctes.len(), 2, "one CTE per derived table, inner then outer");
        // ctes[0] is the inner (width 2: a,b); ctes[1] is the outer (width 1: a).
        match &p.ctes[0] {
            CtePlan::Materialized { name, column_count, .. } => {
                assert_eq!(name, "inr");
                assert_eq!(*column_count, 2);
            }
            other => panic!("ctes[0] should be the inner derived table, got {other:?}"),
        }
        match &p.ctes[1] {
            CtePlan::Materialized { name, column_count, body } => {
                assert_eq!(name, "outr");
                assert_eq!(*column_count, 1);
                // The outer body scans the inner via a CteScan on id 0.
                assert!(
                    matches!(under_project(body), PlanNode::CteScan { id: 0, .. }),
                    "outer body reads the inner CTE (id 0), got {body:?}"
                );
            }
            other => panic!("ctes[1] should be the outer derived table, got {other:?}"),
        }
        // The top-level FROM leaf is the outer derived table (id 1).
        assert!(
            matches!(under_project(&p.root), PlanNode::CteScan { id: 1, column_count: 1 }),
            "top FROM leaf references the outer CTE id 1, got {:?}",
            p.root
        );
        assert_eq!(p.result_columns, vec!["a"]);
    }

    #[test]
    fn derived_table_where_becomes_a_filter_over_the_ctescan() {
        // A WHERE over a derived table is a Filter above the CteScan (a CteScan consumes
        // no predicate, so the whole WHERE is the residual). `d.a = 1` binds to Column(0).
        let p = plan("SELECT a FROM (SELECT a, b FROM t) d WHERE a = 1");
        match under_project(&p.root) {
            PlanNode::Filter { input, predicate } => {
                assert!(
                    matches!(input.as_ref(), PlanNode::CteScan { id: 0, column_count: 2 }),
                    "filter sits over the derived CteScan, got {input:?}"
                );
                match predicate {
                    EvalExpr::Compare { left, .. } => assert_column(left, 0),
                    other => panic!("expected a Compare predicate, got {other:?}"),
                }
            }
            other => panic!("expected a Filter over CteScan, got {other:?}"),
        }
    }

    // ----- Parenthesized joins in FROM: `( join-clause )` as a table-or-subquery -----
    //
    // The register bases are the #1 risk: a mis-based nested source silently reads the
    // wrong registers. These pin the Source bases (Phase 1) and the built `Join` tree's
    // `left_width` + each `on`'s register space (Phase 2). The KEY invariant is that a
    // parenthesization changes only the join-tree SHAPE, never the flat left-to-right
    // source order — and that a RIGHT-operand sub-join binds its own `on` in a LOCAL
    // 0-based space while the enclosing `on` binds in the absolute scope.

    #[test]
    fn resolve_from_paren_join_keeps_the_flat_source_bases() {
        // A parenthesization does NOT change the flat register order: whether the
        // sub-join sits on the left `(t JOIN u) JOIN v` or the right `t JOIN (u JOIN v)`,
        // the sources are t@0, u@3, v@6 — identical to the flat `t JOIN u JOIN v`.
        let cat = cat();
        for sql in [
            "SELECT * FROM t JOIN u ON t.a = u.c JOIN v ON u.d = v.e", // flat baseline
            "SELECT * FROM t JOIN (u JOIN v ON u.d = v.e) ON t.a = u.c", // paren on right
            "SELECT * FROM (t JOIN u ON t.a = u.c) JOIN v ON u.d = v.e", // paren on left
        ] {
            let (sources, _coalesced) =
                resolve_from(&cat, &from_of(sql), 0).unwrap_or_else(|e| panic!("resolve {sql}: {e:?}"));
            let bases: Vec<usize> = sources.iter().map(|s| s.base()).collect();
            let names: Vec<&str> = sources.iter().map(|s| s.exposed_name()).collect();
            assert_eq!(bases, vec![0, 3, 6], "flat source bases unchanged for `{sql}`");
            assert_eq!(names, vec!["t", "u", "v"], "flat source order unchanged for `{sql}`");
        }
    }

    #[test]
    fn paren_join_on_right_binds_nested_on_local_and_outer_on_absolute() {
        // `t JOIN (u JOIN v ON u.d = v.f) ON t.a = v.e`. Absolute layout: t@0, u@3, v@6.
        // The OUTER `on` (t.a = v.e) binds in the ABSOLUTE scope: t.a = Column(0),
        // v.e = Column(6). The NESTED `on` (u.d = v.f) binds in the sub-join's own LOCAL
        // 0-based space (u@0, v@3): u.d = Column(1), v.f = Column(4) — NOT the absolute
        // Column(4)/Column(7). This is the crux: the executor evaluates the sub-join
        // standalone, so its `on` MUST read local registers.
        let p = plan("SELECT * FROM t JOIN (u JOIN v ON u.d = v.f) ON t.a = v.e");
        let top = expect_join(under_project(&p.root));
        assert_eq!(top.left_width, 3, "left operand is the single table t (width 3)");
        assert_eq!(top.right_width, 6, "right operand is the (u JOIN v) sub-join (width 6)");
        // Outer equi-key reads ABSOLUTE registers (t.a=Column(0), v.e=Column(6)); both
        // `t.a=v.e` and the nested `u.d=v.f` are hash keys, so each `on` is stripped to None
        // and the register binding is pinned through the hash keys instead.
        assert_hash_equikey_combined(top, 0, 6);
        // The right child IS the nested join, its equi-key in LOCAL registers (u.d=1, v.f=4).
        let inner = expect_join(top.right.as_ref());
        assert_eq!(inner.left_width, 3, "nested left u (width 3)");
        assert_eq!(inner.right_width, 3, "nested right v (width 3)");
        assert_hash_equikey_combined(inner, 1, 4);
        // A parenthesized right operand is a sub-join, never a single base table, so it
        // NEVER folds into a rowid index-nested-loop (that path is single-base-table only);
        // it takes a hash/nested strategy via choose_join_strategy.
        assert!(
            !matches!(top.strategy, JoinStrategy::IndexNestedLoop),
            "a sub-join right operand is never IndexNestedLoop, got {:?}",
            top.strategy
        );
        assert_eq!(p.result_columns, vec!["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn paren_join_under_right_join_threads_join_type_and_keeps_flat_bases() {
        // LEFT/RIGHT/FULL are the join types; the paren planner path is join-type-agnostic for
        // the register/subtree layout and threads the type from `op.kind`. `(t JOIN u ON
        // t.a=u.c) RIGHT JOIN v ON t.a=v.e`: left is the (t JOIN u) sub-join (width 6),
        // right is v (width 3), join_type is Right, and the outer ON is absolute
        // t.a=Column(0), v.e=Column(6). RIGHT/FULL never uses IndexNestedLoop (the executor
        // rejects it), so the strategy is hash/nested here too.
        let p = plan("SELECT * FROM (t JOIN u ON t.a = u.c) RIGHT JOIN v ON t.a = v.e");
        let top = expect_join(under_project(&p.root));
        assert!(matches!(top.join_type, JoinType::Right), "RIGHT threaded from op.kind, got {:?}", top.join_type);
        assert_eq!(top.left_width, 6, "left (t JOIN u) sub-join width 6");
        assert_eq!(top.right_width, 3, "right v width 3");
        // The RIGHT-join equi-key is a hash key too (Hash serves Right); its absolute
        // registers (t.a=Column(0), v.e=Column(6)) are pinned through the hash keys.
        assert_hash_equikey_combined(top, 0, 6);
        assert!(
            !matches!(top.strategy, JoinStrategy::IndexNestedLoop),
            "RIGHT never uses IndexNestedLoop, got {:?}",
            top.strategy
        );
        // The left child is the (t JOIN u) sub-join; its `t.a=u.c` equi-key is a hash key
        // in the sub-join's own local register space (t.a=Column(0), u.c=Column(3)).
        let inner = expect_join(top.left.as_ref());
        assert_eq!(inner.left_width, 3);
        assert_eq!(inner.right_width, 3);
        assert_hash_equikey_combined(inner, 0, 3);
    }

    #[test]
    fn paren_join_on_left_builds_a_left_child_subjoin() {
        // `(t JOIN u ON t.a = u.c) JOIN v ON u.d = v.e`. The left operand is the sub-join,
        // so the top join's `left_width` is the COMBINED t+u width (6), and its left child
        // is itself a Join. The outer `on` (u.d = v.e) is absolute: u.d = Column(4) (u@3,
        // col 1), v.e = Column(6). A left-position sub-join's local base is 0 (it is
        // leftmost), so its inner `on` (t.a = u.c) is Column(0) = Column(3).
        let p = plan("SELECT * FROM (t JOIN u ON t.a = u.c) JOIN v ON u.d = v.e");
        let top = expect_join(under_project(&p.root));
        assert_eq!(top.left_width, 6, "left operand is the (t JOIN u) sub-join (width 6)");
        assert_eq!(top.right_width, 3, "right operand is the single table v (width 3)");
        // The outer `u.d=v.e` is TEXT=INTEGER (a coerced compare), NOT a hash key, so it
        // stays on the node in full (Column(4)=Column(6)).
        assert_col_cmp(top.on.as_ref().expect("outer ON"), CmpOp::Eq, 4, 6);
        // The inner `t.a=u.c` IS a hash key (INTEGER=INTEGER), stripped into the strategy.
        let inner = expect_join(top.left.as_ref());
        assert_eq!(inner.left_width, 3);
        assert_eq!(inner.right_width, 3);
        assert_hash_equikey_combined(inner, 0, 3);
    }

    #[test]
    fn sole_paren_join_is_the_whole_from_and_matches_the_flat_join() {
        // `FROM (t JOIN u ON t.a = u.c)` — a parenthesized join as the ENTIRE FROM.
        // Wrapping the whole FROM changes nothing, so the plan equals `FROM t JOIN u`:
        // one Join, left_width 3, right_width 3, on Column(0) = Column(3).
        let p = plan("SELECT * FROM (t JOIN u ON t.a = u.c)");
        let j = expect_join(under_project(&p.root));
        assert_eq!(j.left_width, 3);
        assert_eq!(j.right_width, 3);
        assert_hash_equikey_combined(j, 0, 3);
        assert_eq!(p.result_columns, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn both_operands_parenthesized_builds_two_subjoin_children() {
        // `(t JOIN u ON t.a=u.c) JOIN (v JOIN w ON v.e=w.a) ON t.a = v.e`. Absolute:
        // t@0, u@3, v@6, w@9. BOTH operands are sub-joins: top.left_width = 6 (t+u),
        // top.right_width = 6 (v+w). The outer `on` (t.a = v.e) is absolute: Column(0) =
        // Column(6). The RIGHT sub-join's `on` (v.e = w.a) is LOCAL (v@0, w@3):
        // Column(0) = Column(3), NOT the absolute Column(6)=Column(9).
        let p = plan("SELECT * FROM (t JOIN u ON t.a = u.c) JOIN (v JOIN w ON v.e = w.a) ON t.a = v.e");
        let top = expect_join(under_project(&p.root));
        assert_eq!(top.left_width, 6, "left (t JOIN u) width 6");
        assert_eq!(top.right_width, 6, "right (v JOIN w) width 6");
        // All three equi-keys are INTEGER=INTEGER hash keys. The outer key is pinned in
        // ABSOLUTE registers (t.a=Column(0), v.e=Column(6)); each sub-join's key is pinned
        // in its own LOCAL 0-based space (Column(0)=Column(3)).
        assert_hash_equikey_combined(top, 0, 6);
        let left = expect_join(top.left.as_ref());
        assert_hash_equikey_combined(left, 0, 3);
        let right = expect_join(top.right.as_ref());
        assert_hash_equikey_combined(right, 0, 3);
    }

    #[test]
    fn derived_schema_inherits_bare_column_affinity_and_collation() {
        // datatype3 §3.3: a subquery/view/CTE output column that is a bare column ref
        // inherits its source column's affinity + collation; a CAST takes the cast type's
        // affinity; any other expression is NONE (Blob) + BINARY. This is the plan-side
        // pin of the fix that `conformance_derived_affinity` covers end-to-end.
        use super::derived_schema;
        use minisqlite_types::{affinity_of_declared_type, Affinity, Collation};

        // t(a INTEGER, b TEXT) and tc(s TEXT COLLATE NOCASE).
        let nocase = ColumnDef {
            name: "s".to_string(),
            declared_type: Some("TEXT".to_string()),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: Some("NOCASE".to_string()),
            default: None,
            default_value: None,
            generated: None,
        };
        let c = TestCatalog {
            tables: vec![
                tdef("t", vec![col("a", "INTEGER"), col("b", "TEXT")]),
                tdef("tc", vec![nocase]),
            ],
        };
        let schema = |sql: &str| {
            let ast = parse(sql).expect("parse ok");
            let stmt = ast.statements.into_iter().next().expect("one statement");
            let Statement::Select(sel) = stmt else { panic!("expected a SELECT: {sql}") };
            derived_schema(&c, &sel).expect("derived_schema ok")
        };

        // A bare column reference inherits the base column's affinity.
        let s = schema("SELECT a, b FROM t");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].name, "a");
        assert_eq!(s[0].affinity, Affinity::Integer, "bare `a` inherits INTEGER");
        assert_eq!(s[1].name, "b");
        assert_eq!(s[1].affinity, Affinity::Text, "bare `b` inherits TEXT");

        // `SELECT *` preserves each column's affinity (each is a bare column reference).
        let s = schema("SELECT * FROM t");
        assert_eq!(s[0].affinity, Affinity::Integer);
        assert_eq!(s[1].affinity, Affinity::Text);

        // An arithmetic expression has NO affinity — NONE (Blob), the §3.2 default.
        let s = schema("SELECT a + 0 AS x FROM t");
        assert_eq!(s[0].name, "x");
        assert_eq!(s[0].affinity, affinity_of_declared_type(None), "an expression has no affinity");
        assert_eq!(s[0].affinity, Affinity::Blob);

        // `CAST(_ AS T)` takes the cast type's affinity.
        let s = schema("SELECT CAST(b AS INTEGER) AS y FROM t");
        assert_eq!(s[0].affinity, Affinity::Integer, "CAST(_ AS INTEGER) has INTEGER affinity");

        // A bare column reference inherits the base column's COLLATE (NOCASE); an
        // expression (string concat here) falls back to BINARY and NONE affinity.
        let s = schema("SELECT s FROM tc");
        assert_eq!(s[0].collation, Collation::NoCase, "bare `s` inherits NOCASE");
        let s = schema("SELECT s || '' AS z FROM tc");
        assert_eq!(s[0].collation, Collation::Binary, "an expression's collation is BINARY");
        assert_eq!(s[0].affinity, Affinity::Blob, "concat has no affinity");
    }
}
