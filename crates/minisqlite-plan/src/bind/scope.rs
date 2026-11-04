//! Name resolution: the [`Scope`] of columns visible to an expression, built from
//! the FROM clause, plus the [`Grouping`] context that redirects references to the
//! post-aggregate row layout inside an aggregate query.
//!
//! A [`Scope`] resolves a (possibly qualified) column name to a *register* in the
//! row its expression will run against, following the shared ROW/REGISTER
//! convention (see [`crate::plan`]): a *rowid* base table with `N` columns occupies
//! `[base, base+N)` and its rowid sits at `base+N`. An `INTEGER PRIMARY KEY` column
//! aliases the rowid, so it resolves to `base+N` (not its own schema slot, which
//! holds NULL for a pure alias).
//!
//! A [`Source`] is either a `BaseTable` (a catalog `TableDef`) or a `Derived` (a
//! subquery in FROM / a CTE, width `columns.len()` with NO rowid — it carries its own
//! [`SynthCol`] list because it has no catalog table). A rowid `BaseTable` is width
//! `N+1` with a trailing rowid; a WITHOUT ROWID `BaseTable` is width `N` with NO rowid
//! register (it stores its rows in a PRIMARY KEY index b-tree — withoutrowid.html), so
//! `rowid`/`_rowid_`/`oid` do not resolve against it. Every resolver dispatches on the
//! variant; a `Derived` source likewise has no rowid.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use minisqlite_catalog::{ColumnDef, TableDef};
use minisqlite_expr::{AggregateCall, EvalExpr};
use minisqlite_sql::{Expr, WindowSpec};
use minisqlite_types::{Affinity, Collation, DbIndex, Error, Result};

use crate::plan::WindowFunc;

/// One synthetic column of a derived table (a subquery in FROM) or a CTE: the output
/// name the inner SELECT produces, plus the affinity and collation an outer comparison
/// operand needs. A derived table has no catalog [`TableDef`], so its columns are
/// carried here directly rather than borrowed from the schema.
///
/// Per datatype3 §3.3, a VIEW / subquery / CTE column's affinity + collation are those of
/// the inner result expression it maps to (§3.2 / §7.1): a bare column reference inherits
/// that column's affinity + collation, a `CAST(_ AS T)` takes T's affinity, and any other
/// expression has NO affinity (`Affinity::Blob`, what `affinity_of_declared_type(None)`
/// yields) + BINARY collation unless it carries an explicit `COLLATE`. The FROM compiler
/// (`crate::compile::from::derived_schema`) fills these in per column; a `VALUES` column
/// and the JSON table-valued-function input columns are the genuine NONE/BINARY cases.
#[derive(Debug, Clone)]
pub struct SynthCol {
    pub name: String,
    pub affinity: Affinity,
    pub collation: Collation,
    /// Whether this column is HIDDEN: resolvable by name but excluded from `*` expansion
    /// and from NATURAL/USING join matching. Only the JSON table-valued functions' input
    /// columns (`json`, `root`; json1.html §4.24) are hidden; every ordinary derived /
    /// CTE column is visible (`false`).
    pub hidden: bool,
}

/// One step's pairing of a NATURAL/USING join column coalesced into a single output
/// column. Per SQLite (`lang_select.html`), a shared USING/NATURAL column appears EXACTLY
/// ONCE in the join output: its unqualified name resolves to the value of whichever side
/// is present (the left for INNER/LEFT, the right where the left is the NULL-extended
/// outer side), and the right copy is omitted from `*` expansion. A QUALIFIED reference
/// (`right_table.col`) still reaches the right copy directly; only the unqualified name
/// and unqualified `*` are affected. `affinity`/`collation` are the left column's, used
/// when the coalesced column is a comparison operand (all same-name entries share `left`,
/// so the left metadata is the single anchor).
///
/// This is ONE join step: a chain (`a JOIN b USING(x) JOIN c USING(x)`) records one entry
/// per step (`{x: a.x,b.x}`, `{x: a.x,c.x}`), so the runtime VALUE of the shared column is
/// `COALESCE` over EVERY same-name entry's registers, NOT just this one pair — see
/// [`Scope::coalesced_regs`] / [`Scope::coalesced_value`], the single home of that fold
/// that all value/capture sites route through. (A two-table join is the single-entry case,
/// `COALESCE(left, right)`.)
#[derive(Debug, Clone)]
pub struct Coalesced {
    /// The shared column name (case-insensitive).
    pub name: String,
    /// Register of this step's left side's copy (the fold's preferred operand; all
    /// same-name entries in a chain share this left register).
    pub left: usize,
    /// Register of this step's right side's copy (a fold fallback operand).
    pub right: usize,
    /// The left column's affinity, for when the coalesced column is a comparison operand.
    pub affinity: Affinity,
    /// The left column's collation, likewise.
    pub collation: Collation,
}

/// One FROM-clause source and the register offset (`base`) where its columns begin in
/// the combined row (0 for a single-table query; the running left width for the right
/// side of a join).
///
/// A rowid `BaseTable` borrows a catalog [`TableDef`] and occupies `N+1` registers
/// (`[c0, …, c_{N-1}, rowid]`); an INTEGER PRIMARY KEY column aliases the trailing
/// rowid register. A WITHOUT ROWID `BaseTable` has no rowid, so it occupies exactly `N`
/// registers (`[c0, …, c_{N-1}]`), matching its index-b-tree scan row shape. A
/// `Derived` (a subquery in FROM / a CTE) has NO catalog table and NO rowid: it carries
/// its own [`SynthCol`] list and occupies exactly `columns.len()` registers
/// (`[c0, …, c_{N-1}]`), matching the `CteScan` row shape the executor materializes
/// (`minisqlite-exec`'s `ops::cte`).
pub enum Source<'a> {
    /// A base table (a catalog `TableDef`): width `N+1` with a trailing rowid for a
    /// rowid table, or width `N` with no rowid for a WITHOUT ROWID table.
    BaseTable {
        /// The name a qualified reference (`name.col`) must match (case-insensitive).
        exposed_name: String,
        /// The table's schema (borrowed from the catalog — never copied).
        table: &'a TableDef,
        /// Which database namespace the table lives in (`main`/`temp`/attached). Set by
        /// the binder from a schema qualifier or SQLite name-resolution search order, and
        /// carried into the access-path node so the executor opens the cursor on the right
        /// store. `DbIndex::MAIN` for a binding-only scope that never builds a scan (CHECK,
        /// generated-column, index-key, UPSERT `existing`/`excluded`, trigger `OLD`/`NEW`).
        db: DbIndex,
        /// Register offset of this source's first column in the combined row.
        base: usize,
    },
    /// A derived table / CTE, width `columns.len()` with NO rowid register.
    Derived {
        /// The qualifier (`alias.col`) this source answers to (empty for an unaliased
        /// subquery, whose columns are still visible unqualified).
        exposed_name: String,
        /// The synthetic output columns, in order.
        columns: Vec<SynthCol>,
        /// Register offset of this source's first column in the combined row.
        base: usize,
        /// Set true the first time the binder resolves this source's hidden `json` column
        /// (a JSON table-valued function's whole-document input column; json1.html §4.24).
        /// It is a cheap bind-time signal that the statement will READ `json`, so the
        /// executor must copy the (potentially large) document into every row — otherwise
        /// that per-row copy is skipped (see `PlanNode::TableFunctionScan::emit_json` and
        /// `compile::select::finalize_tvf_emit_json`). Interior-mutable because resolution
        /// takes `&Source`; irrelevant (stays `false`) for ordinary derived tables / CTEs,
        /// which have no hidden columns.
        json_referenced: Cell<bool>,
    },
}

impl<'a> Source<'a> {
    /// Construct a `Derived` source (a subquery in FROM, a CTE, an inlined view, or a JSON
    /// table-valued function) exposing `columns` starting at register `base`. The single
    /// builder for the variant, so the interior-mutable `json_referenced` tracking cell
    /// (reset to `false`) has one home and no construction site can forget it.
    pub fn derived(exposed_name: String, columns: Vec<SynthCol>, base: usize) -> Source<'a> {
        Source::Derived { exposed_name, columns, base, json_referenced: Cell::new(false) }
    }

    /// The qualifier a `name.col` reference must match (case-insensitive).
    pub fn exposed_name(&self) -> &str {
        match self {
            Source::BaseTable { exposed_name, .. } | Source::Derived { exposed_name, .. } => {
                exposed_name
            }
        }
    }

    /// The register offset where this source's first column begins.
    pub fn base(&self) -> usize {
        match self {
            Source::BaseTable { base, .. } | Source::Derived { base, .. } => *base,
        }
    }

    /// The database namespace this source's rows come from. A `Derived` source has no
    /// base table (its rows are materialized in-engine), so it reports `DbIndex::MAIN` —
    /// the value is only consumed when building a `BaseTable`'s access-path node.
    pub fn db(&self) -> DbIndex {
        match self {
            Source::BaseTable { db, .. } => *db,
            Source::Derived { .. } => DbIndex::MAIN,
        }
    }

    /// Whether the binder resolved this source's hidden `json` column while binding the
    /// statement (a JSON TVF whose whole-document `json` column is actually read). `false`
    /// for base tables and for any derived table / CTE (they have no hidden `json`). Read
    /// after all clauses are bound to decide whether the TVF leaf must copy the document
    /// into each row (see `compile::select::finalize_tvf_emit_json`).
    pub fn json_referenced(&self) -> bool {
        match self {
            Source::Derived { json_referenced, .. } => json_referenced.get(),
            Source::BaseTable { .. } => false,
        }
    }

    /// The combined-row register width this source occupies: a *rowid* base table is
    /// `N+1` (its columns plus the trailing rowid register); a WITHOUT ROWID base table
    /// is `N` (no rowid register — its scan emits exactly the N stored columns); a
    /// derived table is `columns.len()` (no rowid). This is the single source of truth
    /// for source width (`from.rs::table_width` delegates here, and `total_width` sums
    /// it), so it MUST agree with the executor's per-kind scan row shape.
    pub fn width(&self) -> usize {
        match self {
            Source::BaseTable { table, .. } if table.without_rowid => table.columns.len(),
            Source::BaseTable { table, .. } => table.columns.len() + 1,
            Source::Derived { columns, .. } => columns.len(),
        }
    }

    /// Register that column index `i` reads from. A base table's normal column is
    /// `base+i`; its `INTEGER PRIMARY KEY` alias reads the rowid register `base+N`. A
    /// derived column is always `base+i` (no rowid, no alias).
    fn column_register(&self, i: usize) -> usize {
        match self {
            Source::BaseTable { table, base, .. } => {
                if table.rowid_alias == Some(i) {
                    *base + table.columns.len()
                } else {
                    *base + i
                }
            }
            Source::Derived { base, .. } => *base + i,
        }
    }

    /// The `(register, name)` pairs this source contributes to a `*` expansion, in
    /// schema order. The rowid is never included; a base table's INTEGER PRIMARY KEY
    /// column expands to its rowid register (its logical value) via `column_register`.
    fn star_columns(&self) -> Vec<(usize, String)> {
        match self {
            Source::BaseTable { table, .. } => table
                .columns
                .iter()
                .enumerate()
                .map(|(i, col)| (self.column_register(i), col.name.clone()))
                .collect(),
            // A HIDDEN derived column (a JSON TVF's `json`/`root`) is excluded from `*`,
            // exactly as in SQLite; `enumerate` before the filter keeps each surviving
            // column's original index so `column_register` still maps it to `base + i`.
            Source::Derived { columns, .. } => columns
                .iter()
                .enumerate()
                .filter(|(_, col)| !col.hidden)
                .map(|(i, col)| (self.column_register(i), col.name.clone()))
                .collect(),
        }
    }

    /// Like [`Self::star_columns`] but additionally surfacing each column's affinity +
    /// collation — the metadata a bare reference to that column carries as a comparison
    /// operand. A derived table's `*` columns are bare column references, so they inherit
    /// these (datatype3 §3.3); the FROM compiler uses this to give a `SELECT *` derived
    /// schema the inner columns' affinity/collation.
    ///
    /// The register + name + hidden-column set is IDENTICAL to [`Self::star_columns`] (both
    /// map column `i` through [`Self::column_register`] and skip hidden columns), so a `*`
    /// name list and its affinity/collation list stay in lockstep. Errors only if a base
    /// column's stored `COLLATE` name is unknown — the same validation [`column_in_source`]
    /// performs, routed through the shared [`base_column_meta`].
    fn star_columns_full(&self) -> Result<Vec<(usize, String, Affinity, Collation)>> {
        match self {
            Source::BaseTable { table, .. } => table
                .columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    let (affinity, collation) = base_column_meta(table, i)?;
                    Ok((self.column_register(i), col.name.clone(), affinity, collation))
                })
                .collect(),
            Source::Derived { columns, .. } => Ok(columns
                .iter()
                .enumerate()
                .filter(|(_, col)| !col.hidden)
                .map(|(i, col)| {
                    (self.column_register(i), col.name.clone(), col.affinity, col.collation)
                })
                .collect()),
        }
    }

    /// Whether this source declares a column named `name` (case-insensitive), for
    /// NATURAL / USING join column matching. The rowid is never a join column, and a
    /// HIDDEN column never participates in NATURAL/USING (it is not a visible column).
    pub fn has_column(&self, name: &str) -> bool {
        match self {
            Source::BaseTable { table, .. } => {
                table.columns.iter().any(|c| c.name.eq_ignore_ascii_case(name))
            }
            Source::Derived { columns, .. } => {
                columns.iter().any(|c| !c.hidden && c.name.eq_ignore_ascii_case(name))
            }
        }
    }

    /// This source's declared VISIBLE column names, in order (for building NATURAL-join
    /// equalities). The rowid and any HIDDEN column are never included.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Source::BaseTable { table, .. } => {
                table.columns.iter().map(|c| c.name.as_str()).collect()
            }
            Source::Derived { columns, .. } => {
                columns.iter().filter(|c| !c.hidden).map(|c| c.name.as_str()).collect()
            }
        }
    }

    /// Resolve a real (non-rowid) column `name` within THIS one source — the register
    /// plus affinity/collation — or `None` if absent (case-insensitive). Used by the
    /// FROM compiler to capture each side's copy of a NATURAL/USING shared column when
    /// building [`Coalesced`] entries.
    pub fn column(&self, name: &str) -> Result<Option<ResolvedColumn>> {
        column_in_source(self, name)
    }
}

/// A resolved column reference: the register to read plus the affinity and
/// collation the binder needs when this column is a comparison operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedColumn {
    pub reg: usize,
    pub affinity: Affinity,
    pub collation: Collation,
}

/// The columns visible to an expression. `parent` links to an enclosing query's
/// scope so a subquery can reference outer columns (a correlated subquery);
/// `grouping`, when set, redirects binding to the post-aggregate
/// `[group_keys.., agg_results..]` layout.
pub struct Scope<'a> {
    pub sources: &'a [Source<'a>],
    /// NATURAL/USING columns coalesced into a single output column (empty for a query
    /// with no such join). An unqualified reference to a name here resolves to
    /// `COALESCE(left, right)` and is unambiguous despite appearing in both sides, and
    /// its right copy is dropped from unqualified `*` expansion. See [`Coalesced`].
    pub coalesced: &'a [Coalesced],
    pub parent: Option<&'a Scope<'a>>,
    pub grouping: Option<&'a Grouping<'a>>,
    /// When `Some`, the flag the correlated-subquery compiler watches: it is set to
    /// `true` the moment THIS scope resolves a column through its `parent` chain — i.e.
    /// the enclosing subquery references an outer column and is therefore correlated.
    /// `None` for scopes that don't track correlation (the top-level query, DML,
    /// constant contexts). The cell lives on the compiler's stack for the duration of
    /// one subquery's binding; interior mutability lets the shared-`&` binder record
    /// the fact without threading a return value through every resolver.
    ///
    /// This also marks an *intermediate* subquery when only a deeper nested subquery
    /// reaches THROUGH it to a grandparent: that resolution walks this scope's
    /// `resolve_via_parent`, which sets this cell too, so the intermediate is correctly
    /// flagged correlated (it must carry the outer row down to the deeper subquery).
    pub saw_correlated: Option<&'a Cell<bool>>,
    /// The additive companion to [`saw_correlated`](Self::saw_correlated): when `Some`,
    /// every outer register this scope resolves through its `parent` chain is PUSHED here
    /// (at the same site the bool is set). The correlated-subquery compiler reads it back,
    /// sorts + dedups, and stores it as [`SubPlan::correlated_cols`](crate::plan::SubPlan)
    /// — the set of outer registers the subquery depends on. Collecting at the resolve
    /// site makes it inherently as complete and transitive as the `correlated` flag: a
    /// deeper subquery reaching a grandparent walks THIS scope's `resolve_via_parent` too,
    /// so the intermediate's register is captured as well. Duplicates and order are handled
    /// by the reader; this only appends. `None` wherever `saw_correlated` is `None`.
    pub correlated_cols: Option<&'a RefCell<Vec<usize>>>,
    /// When `Some`, the flag the correlated-subquery compiler watches to decide whether the
    /// subquery is DETERMINISTIC. Set to `true` while binding this scope's expressions the
    /// moment a non-deterministic construct is produced — an `EvalExpr::Now(_)` or an
    /// `EvalExpr::Func` over a `ScalarFunction` whose `deterministic()` is `false`. Read
    /// back into [`SubPlan::deterministic`](crate::plan::SubPlan) (combined with the
    /// determinism of any nested subqueries). `None` for scopes that don't track it. Like
    /// `saw_correlated`, interior mutability lets the shared-`&` binder record the fact
    /// without threading a return value through every expression form.
    pub nondeterministic: Option<&'a Cell<bool>>,
    /// When `Some`, the window-collection context active while binding the SELECT list
    /// and ORDER BY of a query with window functions. A window call (`agg(...) OVER (…)`)
    /// binds its arguments/`PARTITION BY`/`ORDER BY` against the pre-window input row,
    /// pushes a [`WindowFunc`] into this collector, and resolves to the appended output
    /// column `Column(input_width + k)` — exactly how [`Grouping`] redirects an aggregate
    /// to its post-aggregate result register. `None` everywhere a window function is not
    /// permitted (WHERE / GROUP BY / HAVING / ON), which is what makes those a loud error.
    pub windowing: Option<&'a Windowing<'a>>,
}

/// The post-aggregate binding context, active while binding the SELECT list and
/// HAVING of an aggregate query. It matches references against the group keys and
/// collects the aggregate calls encountered so the aggregation operator can run
/// them.
///
/// `aggregates` is a `RefCell` so binding (which holds the scope by shared
/// reference) can still append discovered calls; the sink is drained once after
/// the projection and HAVING are bound.
pub struct Grouping<'a> {
    /// The GROUP BY key expressions (AST), for structural matching of a projection
    /// sub-expression against a whole group key.
    pub key_asts: &'a [Expr],
    /// Map from an input column register to the group-key index it resolves to,
    /// for the common case of a simple-column group key referenced by name.
    pub col_to_group: HashMap<usize, usize>,
    /// Number of group keys (an aggregate result lands at register `num_keys + k`).
    pub num_keys: usize,
    /// Aggregate calls discovered while binding, in first-seen (output) order.
    pub aggregates: RefCell<Vec<AggregateCall>>,
    /// Bare-column capture for a §2.5 bare column (lang_select.html §2.5). `Some` for
    /// every aggregate query (see [`BareCapture`]); a bare column (neither a group key nor
    /// inside an aggregate) is then captured and resolves to its captured register instead
    /// of the "must appear in GROUP BY" error. Which INPUT row the operator captures from
    /// is a runtime concern chosen by the plan marker (the extremum row for single-min/max,
    /// an arbitrary/first row otherwise) — binding is identical either way. A query with no
    /// bare column interns nothing here, so the marker stays absent and nothing is appended.
    pub bare_capture: Option<BareCapture>,
    /// The register width a CORRELATED subquery in this aggregate query's projection /
    /// HAVING / ORDER BY must use as its `outer_width`. At runtime such a subquery is evaluated
    /// over the row the executor prepends to the subplan's own rows, NOT the pre-aggregate FROM
    /// row that [`Scope::total_width`] reports — so the subquery's own FROM sources are placed at
    /// base `subplan_outer_width` and its outer references read `Column(< subplan_outer_width)`.
    ///
    /// The prepended row differs BY HOST CLAUSE, which is why this is an interior-mutable `Cell`
    /// the binder re-points as it moves between clauses (like `saw_correlated_subplan`), not a
    /// fixed value:
    ///   * a PROJECTION / ORDER BY subquery runs over the FINAL output row — the post-WINDOW row
    ///     `[keys.., aggs.., captured.., window..]` (exec `ops/project.rs`; the window stage
    ///     appends its columns) — so it uses the projection width;
    ///   * a HAVING subquery runs INSIDE the aggregate operator, over its emitted
    ///     `[keys.., aggs.., captured..]` row (exec `ops/aggregate.rs`), BEFORE the window stage
    ///     — so it uses the narrower pre-window width. The two coincide exactly when the query
    ///     has no window function (the common case); they diverge by `num_window` on the window
    ///     path, which is why a correlated HAVING subquery alongside a window needs its own width.
    ///
    /// In pass 1 it is the PROVISIONAL width `num_keys + num_aggregates` for both; a §2.5 captured
    /// bare column, a post-aggregate window function, or an ORDER BY that adds an aggregate widens
    /// the real row, so `compile::aggregate` re-binds (via
    /// [`saw_correlated_subplan`](Self::saw_correlated_subplan)) with the TRUE per-clause widths,
    /// re-pointing this `Cell` around the HAVING bind. Read by
    /// [`crate::compile::select::compile_subplan`] via [`Scope::outer_row_width_for_child`].
    pub subplan_outer_width: Cell<usize>,
    /// Set to `true` the moment [`crate::compile::select::compile_subplan`] compiles a
    /// CORRELATED subquery under this grouping (a post-aggregate projection / HAVING / ORDER
    /// BY subquery). Read back after binding so `compile::aggregate` can re-bind with the
    /// corrected [`subplan_outer_width`](Self::subplan_outer_width) when the real row is wider
    /// than the provisional width. Interior-mutable because the binder holds the grouping by
    /// shared reference, exactly like `aggregates`.
    pub saw_correlated_subplan: Cell<bool>,
}

/// The §2.5 bare-column capture collector (lang_select.html §2.5). It interns each bare
/// column's INPUT register (in first-appearance order) to a captured output slot, so
/// repeated references to the same column share one captured column; a bare column at slot
/// `k` resolves to `base + k`, the register the [`crate::plan::PlanNode::Aggregate`]
/// operator appends the captured value at. Shared by both bare-column paths (single-min/max
/// and the general arbitrary-row case) — they differ only in which row the operator reads.
///
/// `regs` is a `RefCell` so the shared-`&` binder can record captures the same way it
/// appends discovered [`AggregateCall`]s; it is read back once, after the projection
/// and HAVING are bound, to build the plan marker.
#[derive(Debug)]
pub struct BareCapture {
    /// Output register of the first captured column: `num_keys + num_aggregates`.
    pub base: usize,
    /// Interned input registers, in first-appearance (slot) order.
    regs: RefCell<Vec<usize>>,
}

impl BareCapture {
    /// A fresh collector whose first captured column lands at `base`.
    pub fn new(base: usize) -> BareCapture {
        BareCapture { base, regs: RefCell::new(Vec::new()) }
    }

    /// Intern an input register, returning its captured slot. De-dups by register
    /// (linear over the few bare columns), so the same column referenced twice shares
    /// one captured column.
    pub fn intern(&self, reg: usize) -> usize {
        let mut regs = self.regs.borrow_mut();
        if let Some(i) = regs.iter().position(|&r| r == reg) {
            return i;
        }
        regs.push(reg);
        regs.len() - 1
    }

    /// The captured input registers in slot order (read once, after binding, to build
    /// the plan marker's `captured_regs`).
    pub fn regs(&self) -> Vec<usize> {
        self.regs.borrow().clone()
    }
}

/// The window-collection context, active while binding the projection and ORDER BY of
/// a query that may contain window functions. Mirrors [`Grouping`]: it collects each
/// window-function call as it is bound and hands back the register of the column the
/// [`crate::plan::PlanNode::Window`] operator appends for it.
///
/// The window operator emits `input_row ++ [v_0 … v_{k-1}]` (one appended value per
/// collected [`WindowFunc`]), so the k-th collected function resolves to
/// `Column(input_width + k)`. `functions` is a `RefCell` so binding (which holds the
/// scope by shared reference) can append discovered calls; it is drained once, after
/// the projection and ORDER BY are bound, to build the operator's function list.
pub struct Windowing<'a> {
    /// Width of the pre-window input row: a window function's appended value lands at
    /// register `input_width + k` (its index in `functions`).
    pub input_width: usize,
    /// Named windows from a trailing `WINDOW name AS (...)` clause, resolved by
    /// `OVER window-name`. Matched case-insensitively; empty when the query has none.
    pub named: &'a [(String, WindowSpec)],
    /// Window-function calls discovered while binding, in first-seen (output) order.
    pub functions: RefCell<Vec<WindowFunc>>,
}

impl<'a> Windowing<'a> {
    /// A fresh window context over an input row of width `input_width`, with the given
    /// named-window definitions and an empty function collector.
    pub fn new(input_width: usize, named: &'a [(String, WindowSpec)]) -> Windowing<'a> {
        Windowing { input_width, named, functions: RefCell::new(Vec::new()) }
    }

    /// Look up a named window by its `WINDOW name AS (...)` name (case-insensitive).
    pub fn lookup(&self, name: &str) -> Option<&WindowSpec> {
        self.named.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, spec)| spec)
    }
}

impl<'a> Scope<'a> {
    /// An empty scope (a SELECT with no FROM, or a constant context such as LIMIT).
    pub fn empty() -> Scope<'a> {
        Scope {
            sources: &[],
            coalesced: &[],
            parent: None,
            grouping: None,
            saw_correlated: None,
            correlated_cols: None,
            nondeterministic: None,
            windowing: None,
        }
    }

    /// A scope over the given sources, with no parent, grouping, or coalesced columns.
    /// A query whose FROM has NATURAL/USING joins builds its scope with a struct literal
    /// (in [`crate::compile::from`]) to supply the [`Coalesced`] list; this constructor
    /// serves the join-free callers (single-table DML, derived-schema probing).
    pub fn new(sources: &'a [Source<'a>]) -> Scope<'a> {
        Scope {
            sources,
            coalesced: &[],
            parent: None,
            grouping: None,
            saw_correlated: None,
            correlated_cols: None,
            nondeterministic: None,
            windowing: None,
        }
    }

    /// A scope over the given sources with an optional enclosing (parent) scope.
    pub fn with_parent(sources: &'a [Source<'a>], parent: Option<&'a Scope<'a>>) -> Scope<'a> {
        Scope {
            sources,
            coalesced: &[],
            parent,
            grouping: None,
            saw_correlated: None,
            correlated_cols: None,
            nondeterministic: None,
            windowing: None,
        }
    }

    /// The same scope with grouping turned off — used to bind an aggregate's own
    /// arguments (which see the pre-aggregate input row), never the grouped layout.
    /// Preserves `saw_correlated` so a correlated outer reference inside an aggregate
    /// argument (or the group-key column check) is still detected, and preserves
    /// `windowing` (turning off grouping is orthogonal to window collection).
    pub fn without_grouping(&self) -> Scope<'a> {
        Scope {
            sources: self.sources,
            coalesced: self.coalesced,
            parent: self.parent,
            grouping: None,
            saw_correlated: self.saw_correlated,
            correlated_cols: self.correlated_cols,
            nondeterministic: self.nondeterministic,
            windowing: self.windowing,
        }
    }

    /// The same scope with window collection turned off — used to bind a window call's
    /// own arguments / `PARTITION BY` / `ORDER BY`, which see the pre-window input row
    /// and must not themselves collect a (nested) window function. Preserves
    /// `saw_correlated` and `grouping` so an outer reference or group key inside a
    /// window argument is still resolved correctly.
    pub fn without_windowing(&self) -> Scope<'a> {
        Scope {
            sources: self.sources,
            coalesced: self.coalesced,
            parent: self.parent,
            grouping: self.grouping,
            saw_correlated: self.saw_correlated,
            correlated_cols: self.correlated_cols,
            nondeterministic: self.nondeterministic,
            windowing: None,
        }
    }

    /// Record that a non-deterministic construct was bound in this scope, so the enclosing
    /// subquery is reported non-deterministic (see [`Scope::nondeterministic`]). A no-op
    /// when this scope does not track determinism. Called by the binder at each
    /// `EvalExpr::Now(_)` and each `EvalExpr::Func` over a non-deterministic
    /// `ScalarFunction`.
    pub(crate) fn note_nondeterministic(&self) {
        if let Some(cell) = self.nondeterministic {
            cell.set(true);
        }
    }

    /// The total register width of the row this scope's operators produce — the
    /// `outer_width` a nested (correlated) subquery places its own sources after. It is
    /// the highest `base + width` over the sources, but NEVER less than the parent's own
    /// `total_width` (see the CO-ROW case below).
    ///
    /// Two ways a parent contributes:
    /// * NORMAL nesting (a correlated subquery, a trigger action): the child's sources are
    ///   placed at `base = parent.total_width()`, so a source's `base` already carries the
    ///   outer prefix and `max(base + width)` already meets or exceeds the parent — the
    ///   `.max(parent)` is a no-op there. This COMPOSES: a subquery nested two deep sees
    ///   `grandparent_width + parent_width` naturally.
    /// * CO-ROW nesting (the UPSERT `DO UPDATE` scope, `compile::insert::compile_upsert`):
    ///   the parent is NOT an outer prefix BELOW the child but a second half sharing the
    ///   SAME physical row — `existing` is the child at base 0 and `excluded` is the parent
    ///   at base `W`, and the executor forms ONE `existing(W) ++ excluded(W)` row (2W wide;
    ///   `minisqlite-exec`'s `do_upsert_update`). Here the child's own `max(base + width)` is
    ///   only `W`, so without `.max(parent)` a correlated subquery in the SET/WHERE would
    ///   size its outer at `W` and rebase its OWN sources onto the excluded half `[W, 2W)`
    ///   and read garbage. `.max(parent)` reports the true `2W`, aligning the planner's outer
    ///   width with the executor's combined row so a correlated target/`excluded.` reference
    ///   binds correctly. (This is why the composition is a `max`, not the child alone.)
    ///
    /// A scope with no FROM (empty sources) produces a `SingleRow` that passes its outer row
    /// through unchanged, so its width IS the parent's outer-row width (0 at the top level) —
    /// the `unwrap_or(parent)` fallback; without it a correlated subquery nested inside a
    /// no-FROM correlated intermediate would place its own columns over the outer prefix and
    /// silently mis-bind.
    ///
    /// The parent contributes its [`Self::outer_row_width_for_child`], NOT a bare
    /// `total_width`: when the parent is an aggregate query's POST-aggregate scope
    /// (`grouping` set), its child was placed at `base = subplan_outer_width` (the
    /// post-aggregate row the executor prepends), so the child's outer prefix — and thus this
    /// composition — must use that post-aggregate width, not the parent's PRE-aggregate FROM
    /// width. Using the FROM width there over-reports the prefix and a doubly-nested
    /// correlated subquery (one inside a post-aggregate correlated subquery) reads an
    /// out-of-range register. This keeps `total_width` consistent with the `outer_width`
    /// [`crate::compile::select::compile_subplan`] actually compiled the child against.
    pub fn total_width(&self) -> usize {
        let parent = self.parent.map(|p| p.outer_row_width_for_child()).unwrap_or(0);
        match self.max_source_end() {
            Some(w) => w.max(parent),
            None => parent,
        }
    }

    /// The maximum end register (`base + width`) across this scope's OWN FROM sources, or
    /// `None` when there are none (a no-FROM `SingleRow`). This is the register width of the
    /// row THIS scope's own sources produce, sitting above the outer prefix (sources are placed
    /// at `base_offset`, so the max end already includes it). The single source of truth for
    /// two consumers that differ only in their empty-sources fallback: [`Self::total_width`]
    /// maxes in the parent's outer-row width (the outer width a CHILD is placed after), while
    /// the window-operator input width falls back to `base_offset` (the outer prefix the exec
    /// window runs over for a no-FROM core) — see `compile::select::compile_core`.
    pub(crate) fn max_source_end(&self) -> Option<usize> {
        self.sources.iter().map(|s| s.base() + s.width()).max()
    }

    /// The register width of the row THIS scope prepends to a nested subquery — i.e. the
    /// `outer_width` such a subquery is compiled against and where its own sources begin.
    /// It is `total_width()` for an ordinary scope, but the grouping's
    /// [`Grouping::subplan_outer_width`] (the POST-aggregate row width) when this is an
    /// aggregate query's post-aggregate scope, mirroring the exact choice
    /// [`crate::compile::select::compile_subplan`] makes for a direct child. Keeping the two
    /// in one place is what makes deep nesting under a grouping parent compose correctly.
    pub(crate) fn outer_row_width_for_child(&self) -> usize {
        match self.grouping {
            Some(g) => g.subplan_outer_width.get(),
            None => self.total_width(),
        }
    }

    // NOTE: the grouping scope is built with a struct literal at the (short-lived)
    // grouping's use site (see `compile::aggregate`) rather than via a method here,
    // because the grouping outlives only the projection binding, not the whole
    // catalog lifetime `'a`.

    /// Resolve a (possibly table-qualified) column name to a register + affinity +
    /// collation. Unqualified names must match exactly one column across all
    /// sources (0 = "no such column", >1 = "ambiguous"); `rowid`/`_rowid_`/`oid`
    /// resolve to the rowid register unless a real column shadows the name.
    pub fn resolve_column(&self, table: Option<&str>, name: &str) -> Result<ResolvedColumn> {
        match table {
            Some(t) => self.resolve_qualified(t, name),
            None => self.resolve_unqualified(name),
        }
    }

    /// Resolve a (possibly qualified) column reference to the [`EvalExpr`] that reads
    /// its value. Identical to [`Self::resolve_column`] except for an UNqualified
    /// NATURAL/USING coalesced column, which resolves to `COALESCE(left, right)` — one
    /// value, the left where present else the right — so a bare shared-column reference
    /// is unambiguous and honors the outer-join rule (`lang_select.html`). Every column
    /// reference in an expression binds through here so the coalescing is applied
    /// uniformly; a qualified reference names one side directly and is never coalesced.
    pub fn resolve_column_expr(&self, table: Option<&str>, name: &str) -> Result<EvalExpr> {
        match self.try_resolve_column_expr(table, name)? {
            Some(ev) => Ok(ev),
            None => Err(Error::sql(format!("no such column: {name}"))),
        }
    }

    /// Like [`Self::resolve_column_expr`], but reports a genuine "no such column" as
    /// `Ok(None)` instead of an error while still returning `Err` for an *ambiguous*
    /// match (a name that IS a column in two+ sources) or any other hard failure.
    ///
    /// This is the resolution the SQLite DQS fallback dispatches on (`quirks.html` §8): a
    /// bare double-quoted token becomes a string literal ONLY when it matches no valid
    /// identifier. An ambiguous name still matches a valid identifier, so real sqlite
    /// errors on it rather than falling back — preserved here by keeping ambiguity an
    /// `Err`, never collapsing it into the `Ok(None)` not-found case.
    pub fn try_resolve_column_expr(
        &self,
        table: Option<&str>,
        name: &str,
    ) -> Result<Option<EvalExpr>> {
        if table.is_none() {
            if let Some(ev) = self.coalesced_value(name) {
                return Ok(Some(ev));
            }
        }
        Ok(self.try_resolve_column(table, name)?.map(|rc| EvalExpr::Column(rc.reg)))
    }

    /// The value expression an UNqualified NATURAL/USING coalesced column `name` binds
    /// to, or `None` when `name` names no coalesced column: `COALESCE` over the column's
    /// component registers ([`Self::coalesced_regs`]).
    fn coalesced_value(&self, name: &str) -> Option<EvalExpr> {
        self.coalesced_regs(name).map(fold_coalesce)
    }

    /// The distinct component registers of the unqualified NATURAL/USING coalesced column
    /// `name`, in join order (the shared left copy first, then each entry's right copy), or
    /// `None` when `name` names no coalesced column. Every same-name [`Coalesced`] entry
    /// contributes its registers, deduped.
    ///
    /// This is the ONE register-level basis every coalesced-VALUE site shares: the bare-name
    /// value ([`Self::coalesced_value`]), the `SELECT *` value ([`Self::star_column_expr`]),
    /// and the single-min/max §2.5 bare-column CAPTURE (`bind/expr.rs`,
    /// `compile/aggregate.rs`) all read a coalesced column's value as `COALESCE` over exactly
    /// these registers. Folding EVERY same-name entry — not just the first — is what makes a
    /// 3+-table USING/NATURAL chain correct: `a JOIN b USING(x) JOIN c USING(x)` records
    /// `{x: a.x, b.x}` then `{x: a.x, c.x}` (both sharing `a.x`), so the first entry alone
    /// would drop `c.x` and read NULL on an outer row present only in `c`. Operand order is
    /// irrelevant to the value (a USING/NATURAL predicate forces every present copy equal on
    /// a matched row; on an outer row only the preserved side is non-NULL), so the dedup
    /// order is free — what matters is that ALL copies are included. Centralizing this here
    /// keeps each value site from independently re-deriving (and, on a sibling path,
    /// forgetting) the fold.
    pub(crate) fn coalesced_regs(&self, name: &str) -> Option<Vec<usize>> {
        let mut regs: Vec<usize> = Vec::new();
        for c in self.coalesced.iter().filter(|c| c.name.eq_ignore_ascii_case(name)) {
            for reg in [c.left, c.right] {
                if !regs.contains(&reg) {
                    regs.push(reg);
                }
            }
        }
        (!regs.is_empty()).then_some(regs)
    }

    /// The component registers ([`Self::coalesced_regs`]) of the coalesced column whose
    /// SURVIVING unqualified-`*` copy is `reg` (its left copy — `expand_star` drops the
    /// right copies), or `None` if `reg` is not a coalesced left copy. The register-keyed
    /// entry the `*` value/capture paths use: they hold the surviving register, not the
    /// name, and — since a same-named NON-coalesced column can also survive a star (an
    /// unrelated third table's copy) — the match MUST be by register, never by name.
    pub(crate) fn coalesced_regs_by_left(&self, reg: usize) -> Option<Vec<usize>> {
        let c = self.coalesced.iter().find(|c| c.left == reg)?;
        self.coalesced_regs(&c.name)
    }

    /// Like [`Self::resolve_column`], but distinguishes "no column of that name"
    /// (`Ok(None)`) from an *ambiguous* match or any other hard failure (`Err`) — the
    /// not-found-aware entrypoint behind [`Self::try_resolve_column_expr`]. A qualified
    /// reference never participates in the DQS fallback, so a qualified miss stays the hard
    /// `Err` (`no such column: t.name` / `no such table: t`); only a bare unqualified miss
    /// yields `Ok(None)`.
    pub fn try_resolve_column(
        &self,
        table: Option<&str>,
        name: &str,
    ) -> Result<Option<ResolvedColumn>> {
        match table {
            None => self.try_resolve_unqualified(name),
            Some(t) => self.resolve_qualified(t, name).map(Some),
        }
    }

    fn resolve_qualified(&self, table: &str, name: &str) -> Result<ResolvedColumn> {
        for src in self.sources {
            if src.exposed_name().eq_ignore_ascii_case(table) {
                if let Some(rc) = column_in_source(src, name)? {
                    return Ok(rc);
                }
                if let Some(rc) = rowid_in_source(src, name) {
                    return Ok(rc);
                }
                return Err(Error::sql(format!("no such column: {table}.{name}")));
            }
        }
        // Not a local source: it may name an enclosing query's source (correlated).
        self.resolve_via_parent(Some(table), name)
    }

    fn resolve_unqualified(&self, name: &str) -> Result<ResolvedColumn> {
        // A bare name that matches no column anywhere is the loud not-found error; an
        // *ambiguous* match (or any other hard failure) propagates from the core below.
        match self.try_resolve_unqualified(name)? {
            Some(rc) => Ok(rc),
            None => Err(Error::sql(format!("no such column: {name}"))),
        }
    }

    /// The not-found-aware core of unqualified resolution: `Ok(Some)` when the name binds
    /// to exactly one column (or its coalesced / rowid form), `Ok(None)` when it matches
    /// NO column anywhere (this scope and every parent), and `Err` for an *ambiguous*
    /// match. A name present in two+ sources IS a valid identifier, so it must stay an
    /// error and never be reported as not-found — that is the exact distinction the DQS
    /// fallback rides on (a real ambiguity must error, not become a string literal).
    fn try_resolve_unqualified(&self, name: &str) -> Result<Option<ResolvedColumn>> {
        // A NATURAL/USING coalesced column is a single logical column: it resolves to
        // its LEFT copy's register/affinity/collation (the value coalesce is applied by
        // `resolve_column_expr`) and is UNAMBIGUOUS even though the name physically
        // appears in both joined sides. Checked first, so the count-based ambiguity
        // rule below never fires on a shared join column.
        if let Some(c) = self.coalesced.iter().find(|c| c.name.eq_ignore_ascii_case(name)) {
            return Ok(Some(ResolvedColumn { reg: c.left, affinity: c.affinity, collation: c.collation }));
        }
        let mut found: Option<ResolvedColumn> = None;
        let mut count = 0usize;
        for src in self.sources {
            if let Some(rc) = column_in_source(src, name)? {
                count += 1;
                found = Some(rc);
            }
        }
        if count == 1 {
            return Ok(Some(found.expect("count==1 implies a value")));
        }
        if count > 1 {
            return Err(Error::sql(format!("ambiguous column name: {name}")));
        }
        // No real column of that name: `rowid`/`_rowid_`/`oid` name the rowid.
        if is_rowid_keyword(name) {
            let mut rfound: Option<ResolvedColumn> = None;
            let mut rcount = 0usize;
            for src in self.sources {
                if let Some(rc) = rowid_in_source(src, name) {
                    rcount += 1;
                    rfound = Some(rc);
                }
            }
            if rcount == 1 {
                return Ok(Some(rfound.expect("rcount==1 implies a value")));
            }
            if rcount > 1 {
                return Err(Error::sql(format!("ambiguous column name: {name}")));
            }
        }
        self.try_resolve_via_parent(name)
    }

    /// Redirect a correlated reference to one of THIS scope's OWN columns to its
    /// POST-aggregate register, when this scope is an aggregate query's post-aggregate
    /// binding scope (its `grouping` is set). This is the correlated-subquery twin of
    /// [`bind::bind_grouped_node`](crate::bind)'s step 1: a subquery in an aggregate
    /// query's projection / HAVING is evaluated by the executor over the aggregate
    /// operator's OUTPUT row `[group_keys.., agg_results..]`, NOT the FROM row, so a
    /// correlated reference to a GROUP BY column must read that column's post-aggregate KEY
    /// register (`col_to_group`), exactly as a direct reference to it does.
    ///
    /// Applied at the point a PARENT resolves the reference (in [`Self::resolve_via_parent`]
    /// / [`Self::try_resolve_via_parent`]), so `self` here is the enclosing scope that owns
    /// the register:
    ///
    /// * No `grouping` (a plain query's FROM scope, or a WHERE / GROUP-BY-key /
    ///   aggregate-ARGUMENT scope, all built with `grouping = None`): return the register
    ///   unchanged — the runtime outer row genuinely IS this scope's FROM row.
    /// * `grouping` set and the reference names one of this scope's LOCAL FROM columns: it
    ///   must be a plain-column GROUP BY key (present in `col_to_group`) — return its key
    ///   register. A local column that is NOT a group key is a §2.5 bare column or lives
    ///   inside an aggregate; neither has a stable post-aggregate register a subquery can
    ///   correlate on, so this is a LOUD error rather than a silently wrong slot. (Real
    ///   SQLite's §2.5 arbitrary-row semantics for such a bare correlation are out of scope
    ///   here; see the residual-reject note in `compile::select::compile_subplan`.)
    /// * `grouping` set but the reference resolved THROUGH this aggregate scope to a query
    ///   ENCLOSING the aggregate (its qualifier/name matches no LOCAL source): this is a
    ///   LOUD error. The executor prepends ONLY the aggregate operator's output row
    ///   `[group_keys.., agg_results..]` — the enclosing query's row is NOT present — so that
    ///   value has no register in the runtime post-aggregate row; returning ANY register
    ///   would read a group-key/aggregate-result slot (a silently wrong answer, or an
    ///   out-of-range read), which this reject forbids. Locality is decided by NAME
    ///   ([`Self::names_local_column`]), not by register range: a grandparent's 0-based
    ///   register can numerically overlap this scope's own FROM registers, so a range test
    ///   would misclassify (and could silently remap) such a reference.
    fn remap_post_aggregate(
        &self,
        table: Option<&str>,
        name: &str,
        rc: ResolvedColumn,
    ) -> Result<ResolvedColumn> {
        let g = match self.grouping {
            Some(g) => g,
            None => return Ok(rc),
        };
        // Decide locality by NAME, not by register range: the reference resolved through this
        // scope carries a register in the RESOLVING scope's space, and a query enclosing the
        // aggregate uses the same 0-based register space as this aggregate's own FROM, so the
        // two overlap and a range test would misclassify (and could silently remap) a
        // grandparent reference. A reference is local iff its qualifier/name matches one of
        // THIS scope's own sources — exactly the test resolution itself used before it walked
        // to the parent.
        if !self.names_local_column(table, name)? {
            return Err(Error::sql(
                "a correlated subquery in the SELECT list or HAVING of an aggregate query may \
                 reference only that aggregate query's own GROUP BY columns, not a column of a \
                 query enclosing the aggregate",
            ));
        }
        match g.col_to_group.get(&rc.reg) {
            Some(&gk) => Ok(ResolvedColumn { reg: gk, affinity: rc.affinity, collation: rc.collation }),
            None => Err(Error::sql(
                "a correlated subquery in the SELECT list or HAVING of an aggregate query may \
                 reference only the outer query's GROUP BY columns",
            )),
        }
    }

    /// Does `(table, name)` name a column of one of THIS scope's OWN sources — never the
    /// parent chain? Mirrors the local half of [`Self::resolve_qualified`] /
    /// [`Self::try_resolve_unqualified`] (qualifier match, then column/coalesced/rowid
    /// lookup) WITHOUT the correlated parent fallback. This is the collision-safe basis
    /// [`Self::remap_post_aggregate`] needs: a grandparent register can numerically overlap
    /// this scope's own FROM registers, so only a NAME match distinguishes a genuine local
    /// column from a reference that merely resolved through this scope to an enclosing query.
    fn names_local_column(&self, table: Option<&str>, name: &str) -> Result<bool> {
        match table {
            // A qualified reference is local iff a local source carries the qualifier: if one
            // does, resolution bound the column there (or already errored) and never reached
            // the parent; if none does, resolution walked to the parent (a correlated ref).
            Some(t) => Ok(self.sources.iter().any(|s| s.exposed_name().eq_ignore_ascii_case(t))),
            None => {
                if self.coalesced.iter().any(|c| c.name.eq_ignore_ascii_case(name)) {
                    return Ok(true);
                }
                for src in self.sources {
                    if column_in_source(src, name)?.is_some() {
                        return Ok(true);
                    }
                }
                Ok(is_rowid_keyword(name)
                    && self.sources.iter().any(|s| rowid_in_source(s, name).is_some()))
            }
        }
    }

    /// Fall back to an enclosing scope. A successful resolution there is a *correlated*
    /// reference: the returned [`ResolvedColumn`] carries the parent's register (for a
    /// top-level parent, in `[0, outer_width)`), which is exactly where the executor places
    /// the outer row it prepends to a correlated subplan's rows. When that parent is an
    /// aggregate query's post-aggregate scope, [`Self::remap_post_aggregate`] redirects a
    /// GROUP BY column to its post-aggregate KEY register first (see there). We also mark
    /// this scope's `saw_correlated` flag so the compiler registers the enclosing subquery
    /// as correlated and places its own sources after the outer row.
    ///
    /// A failure in the parent propagates its "no such column" / "no such table" verdict
    /// unchanged; with no parent (the top level) it is that verdict itself. The flag is
    /// set ONLY on a successful parent resolve, so a genuinely unknown column stays a
    /// loud error and never marks a query correlated.
    fn resolve_via_parent(&self, table: Option<&str>, name: &str) -> Result<ResolvedColumn> {
        match self.parent {
            Some(parent) => {
                // Propagates the parent's not-found error unchanged on `?`.
                let resolved = parent.resolve_column(table, name)?;
                // Remap a GROUP BY column of an aggregate parent to its post-aggregate key
                // register (no-op for a non-grouping parent); a non-group-key LOCAL ref, or a
                // ref that resolved THROUGH the aggregate to an enclosing query, is a loud
                // error here (name-based locality — see `remap_post_aggregate`).
                let resolved = parent.remap_post_aggregate(table, name, resolved)?;
                if let Some(flag) = self.saw_correlated {
                    flag.set(true);
                }
                // Collect the outer register this correlated reference reads (companion
                // to the bool above): `resolved.reg` is the (possibly remapped) register in
                // the parent chain, which — because a subquery's outer row IS its parent's
                // row as a prefix — indexes the same slot of the outer row the executor
                // prepends. See `Scope::correlated_cols`.
                if let Some(cols) = self.correlated_cols {
                    cols.borrow_mut().push(resolved.reg);
                }
                Ok(resolved)
            }
            None => match table {
                Some(t) => Err(Error::sql(format!("no such table: {t}"))),
                None => Err(Error::sql(format!("no such column: {name}"))),
            },
        }
    }

    /// The not-found-aware sibling of [`Self::resolve_via_parent`] for the unqualified
    /// path: `Ok(None)` at the top level (the name is genuinely unknown), otherwise the
    /// parent chain's verdict. Sets `saw_correlated` exactly when a parent resolves the
    /// name (a successful `Some`), matching `resolve_via_parent`; an ambiguity raised in a
    /// parent — or a loud reject from [`Self::remap_post_aggregate`] (a non-group-key local
    /// reference into an aggregate parent, or a reference resolving THROUGH it to an enclosing
    /// query) — propagates as `Err` and marks nothing correlated.
    fn try_resolve_via_parent(&self, name: &str) -> Result<Option<ResolvedColumn>> {
        match self.parent {
            Some(parent) => {
                let resolved = match parent.try_resolve_unqualified(name)? {
                    // Remap a GROUP BY column of an aggregate parent to its post-aggregate
                    // key register (no-op for a non-grouping parent), mirroring
                    // `resolve_via_parent`; a non-group-key local ref — or a ref resolving
                    // THROUGH the aggregate to an enclosing query — is a loud `Err` here.
                    Some(rc) => Some(parent.remap_post_aggregate(None, name, rc)?),
                    None => None,
                };
                if let Some(rc) = resolved {
                    if let Some(flag) = self.saw_correlated {
                        flag.set(true);
                    }
                    // Collect the outer register, same as `resolve_via_parent`.
                    if let Some(cols) = self.correlated_cols {
                        cols.borrow_mut().push(rc.reg);
                    }
                }
                Ok(resolved)
            }
            None => Ok(None),
        }
    }

    /// Expand `*` (all sources) or `table.*` (one source) to `(register, name)`
    /// pairs in schema order. The rowid is never included; an `INTEGER PRIMARY KEY`
    /// column expands to its rowid register (its logical value).
    pub fn expand_star(&self, table: Option<&str>) -> Result<Vec<(usize, String)>> {
        let mut out = Vec::new();
        match table {
            Some(t) => {
                let src = self
                    .sources
                    .iter()
                    .find(|s| s.exposed_name().eq_ignore_ascii_case(t))
                    .ok_or_else(|| Error::sql(format!("no such table: {t}")))?;
                out.extend(src.star_columns());
            }
            None => {
                if self.sources.is_empty() {
                    return Err(Error::sql("no tables specified"));
                }
                for src in self.sources {
                    out.extend(src.star_columns());
                }
                // A NATURAL/USING shared column appears exactly once under an
                // unqualified `*`: drop each coalesced column's RIGHT copy (the left
                // copy stays, carrying the coalesced value's position). A qualified
                // `table.*` above is untouched — it names one table and shows all of its
                // columns, shared or not.
                if !self.coalesced.is_empty() {
                    out.retain(|(reg, _)| !self.coalesced.iter().any(|c| c.right == *reg));
                }
            }
        }
        Ok(out)
    }

    /// Like [`Self::expand_star`], but each expanded column carries its affinity +
    /// collation — `(name, affinity, collation)` instead of `(register, name)`. A `*`
    /// column is a bare column reference, so it inherits its source column's affinity +
    /// collation (datatype3 §3.3); [`crate::compile::from::derived_schema`] uses this so a
    /// `SELECT *` derived table / view / CTE column matches a direct reference to the inner
    /// column.
    ///
    /// The column SET and its order are exactly [`Self::expand_star`]'s — including
    /// dropping each unqualified NATURAL/USING shared column's RIGHT copy (by register), so
    /// the surviving LEFT copy's affinity/collation is the one used, consistent with the
    /// register `expand_star` keeps for that shared column.
    pub fn expand_star_cols(
        &self,
        table: Option<&str>,
    ) -> Result<Vec<(String, Affinity, Collation)>> {
        let mut out: Vec<(usize, String, Affinity, Collation)> = Vec::new();
        match table {
            Some(t) => {
                let src = self
                    .sources
                    .iter()
                    .find(|s| s.exposed_name().eq_ignore_ascii_case(t))
                    .ok_or_else(|| Error::sql(format!("no such table: {t}")))?;
                out.extend(src.star_columns_full()?);
            }
            None => {
                if self.sources.is_empty() {
                    return Err(Error::sql("no tables specified"));
                }
                for src in self.sources {
                    out.extend(src.star_columns_full()?);
                }
                if !self.coalesced.is_empty() {
                    out.retain(|(reg, ..)| !self.coalesced.iter().any(|c| c.right == *reg));
                }
            }
        }
        Ok(out.into_iter().map(|(_reg, name, aff, coll)| (name, aff, coll)).collect())
    }

    /// Expand `*` / `table.*` into the PROJECTED expressions (with output names) — the
    /// form a `SELECT` list emits. The column SET is exactly [`Self::expand_star`]'s, but
    /// each USING/NATURAL shared column projects `COALESCE(left, right)` — the SAME value
    /// a bare reference to it binds to via [`Self::resolve_column_expr`] — instead of a
    /// bare `Column(left)`. This keeps the two star/name resolution paths in agreement:
    /// on a RIGHT/FULL outer join's unmatched-right row the left copy is NULL-extended, so
    /// a plain `Column(left)` would read NULL where the coalesced value is the present
    /// right side (and diverge from real sqlite). Coalescing applies ONLY to an
    /// unqualified `*` (mirroring `resolve_column_expr`); a qualified `table.*` names one
    /// table and projects each of its columns as-is.
    pub fn expand_star_exprs(&self, table: Option<&str>) -> Result<Vec<(EvalExpr, String)>> {
        let cols = self.expand_star(table)?;
        let coalesce = table.is_none() && !self.coalesced.is_empty();
        Ok(cols.into_iter().map(|(reg, name)| (self.star_column_expr(reg, coalesce), name)).collect())
    }

    /// The projected expression for a surviving `*` column at register `reg`: the whole-
    /// chain `COALESCE(...)` when `coalesce` is set and `reg` is the LEFT copy of a
    /// coalesced shared column, else a plain `Column(reg)`. `expand_star` has already
    /// dropped every right copy, so `reg` is the single surviving copy of its shared name;
    /// we recover that name from the matching [`Coalesced`] entry and fold ALL same-name
    /// entries via [`Self::coalesced_value`] — the SAME N-ary value a bare reference binds
    /// to — so `*` and a bare name stay consistent even across a three-plus-table USING
    /// chain (where a 2-ary fold over the first entry alone would drop a later operand).
    fn star_column_expr(&self, reg: usize, coalesce: bool) -> EvalExpr {
        if coalesce {
            if let Some(regs) = self.coalesced_regs_by_left(reg) {
                return fold_coalesce(regs);
            }
        }
        EvalExpr::Column(reg)
    }
}

/// Build the `COALESCE(Column(regs[0]), Column(regs[1]), …)` value expression over a
/// coalesced column's component registers ([`Scope::coalesced_regs`]). The single home
/// that turns the register list into a direct runtime value, shared by the bare-name
/// ([`Scope::coalesced_value`]) and `*` ([`Scope::star_column_expr`]) value paths so they
/// cannot diverge. (The §2.5 CAPTURE paths do not use this: they must first remap each
/// register through the extremum-row capture slot, so they build the `COALESCE` over
/// `Column(cap.base + slot)` themselves — but over the SAME [`Scope::coalesced_regs`] set.)
pub(crate) fn fold_coalesce(regs: Vec<usize>) -> EvalExpr {
    EvalExpr::Coalesce(regs.into_iter().map(EvalExpr::Column).collect())
}

/// The two `Source`s of a trigger body's OLD/NEW parent scope over `target`, per the
/// NEW/OLD register convention (see [`crate::plan::TriggerProgram`]): the target table
/// exposed as `OLD` at register base 0 and as `NEW` at base `W`, where `W` is the target's
/// base-row width. Both borrow the SAME target [`TableDef`], so `OLD.col_k → Column(k)`,
/// `NEW.col_k → Column(W+k)`, and the resulting scope has `total_width() == 2W`.
///
/// `W` is taken from [`Source::width`] — the single source of truth — NOT hardcoded, so
/// OLD/NEW cannot drift from a scan's row shape: a ROWID target is `W = C + 1` (columns +
/// trailing rowid, `OLD.rowid → Column(C)`, `NEW.rowid → Column(W+C)`); a WITHOUT ROWID
/// target is `W = C` (no rowid register, so `rowid`/`oid` do not resolve against OLD/NEW,
/// matching [`rowid_in_source`]).
///
/// A trigger's `WHEN` binds directly against `Scope::new(&sources)`; each action
/// statement compiles as a CORRELATED statement with that scope as its PARENT (its own
/// FROM sources sit at base `2W`), reusing the correlated-subquery machinery. `OLD` and
/// `NEW` are always both exposed regardless of event — the firing executor fills the
/// absent half (OLD for INSERT, NEW for DELETE) with NULLs.
pub fn new_old_sources(target: &TableDef) -> [Source<'_>; 2] {
    let old = Source::BaseTable {
        exposed_name: "OLD".to_string(),
        table: target,
        db: DbIndex::MAIN,
        base: 0,
    };
    // Place NEW one full base-row width above OLD, derived from the SAME `width()` a scan
    // uses, so a WR target (width N, no rowid register) and a rowid target (width N+1)
    // are handled by construction rather than a duplicated `+ 1` that would misbind NEW.*.
    let w = old.width();
    let new = Source::BaseTable {
        exposed_name: "NEW".to_string(),
        table: target,
        db: DbIndex::MAIN,
        base: w,
    };
    [old, new]
}

/// The affinity + collation of base `table`'s column at index `i` — the metadata a
/// reference to that column carries as a comparison operand. An `INTEGER PRIMARY KEY`
/// alias (`rowid_alias == Some(i)`) reads as the integer rowid, so its affinity is
/// INTEGER (§3.1); any other column takes its declared-type affinity. The collation is
/// the column's declared `COLLATE`, else BINARY, in every case.
///
/// The SINGLE home of this mapping: both by-name resolution ([`column_in_source`]) and
/// `*` expansion ([`Source::star_columns_full`]) route through it, so the rowid-alias
/// INTEGER rule and the declared-type/collation lookup cannot drift between the two.
fn base_column_meta(table: &TableDef, i: usize) -> Result<(Affinity, Collation)> {
    let col = &table.columns[i];
    let affinity = if table.rowid_alias == Some(i) {
        Affinity::Integer
    } else {
        table.column_affinity(i)
    };
    Ok((affinity, collation_of(col)?))
}

/// Resolve a real (non-rowid) column named `name` within one source. A base table
/// resolves against its schema columns (an INTEGER PRIMARY KEY alias reads as an
/// integer rowid); a derived table resolves against its [`SynthCol`] list, carrying
/// that column's own affinity/collation.
fn column_in_source(src: &Source, name: &str) -> Result<Option<ResolvedColumn>> {
    match src {
        Source::BaseTable { table, .. } => {
            for (i, col) in table.columns.iter().enumerate() {
                if col.name.eq_ignore_ascii_case(name) {
                    let (affinity, collation) = base_column_meta(table, i)?;
                    return Ok(Some(ResolvedColumn {
                        reg: src.column_register(i),
                        affinity,
                        collation,
                    }));
                }
            }
            Ok(None)
        }
        Source::Derived { columns, json_referenced, .. } => {
            // Iterates ALL columns, including HIDDEN ones: a JSON TVF's `json`/`root` are
            // excluded from `*` (see `star_columns`) but resolvable by name here, which is
            // exactly SQLite's hidden-column contract.
            for (i, col) in columns.iter().enumerate() {
                if col.name.eq_ignore_ascii_case(name) {
                    // Record that the (potentially huge) hidden `json` document column is
                    // actually read, so the executor materializes it per row ONLY then (the
                    // other hidden column, `root`, is a small path string, always emitted).
                    // This is the false-negative-free reference signal the TVF leaf's
                    // `emit_json` flag is set from — every by-name resolution funnels here.
                    if col.hidden && col.name.eq_ignore_ascii_case("json") {
                        json_referenced.set(true);
                    }
                    return Ok(Some(ResolvedColumn {
                        reg: src.column_register(i),
                        affinity: col.affinity,
                        collation: col.collation,
                    }));
                }
            }
            Ok(None)
        }
    }
}

/// Resolve `name` as the rowid alias (`rowid`/`_rowid_`/`oid`) of one source. Only a
/// *rowid* base table has a rowid; a WITHOUT ROWID table has NONE (it stores its rows
/// in a PRIMARY KEY index b-tree, with no integer rowid — withoutrowid.html §2), and a
/// derived table (a subquery in FROM / a CTE) has none either. In both those cases
/// `rowid`/`_rowid_`/`oid` do not resolve here and fall through to a real column of
/// that name elsewhere, or the loud "no such column" real SQLite reports.
fn rowid_in_source(src: &Source, name: &str) -> Option<ResolvedColumn> {
    if !is_rowid_keyword(name) {
        return None;
    }
    match src {
        // A WITHOUT ROWID base table exposes no rowid register (its source width is N,
        // not N+1), so the rowid keywords are just unknown names against it.
        Source::BaseTable { table, .. } if table.without_rowid => None,
        Source::BaseTable { table, base, .. } => Some(ResolvedColumn {
            reg: base + table.columns.len(),
            affinity: Affinity::Integer,
            collation: Collation::Binary,
        }),
        Source::Derived { .. } => None,
    }
}

/// Whether `name` is one of SQLite's built-in rowid aliases (`rowid`/`_rowid_`/`oid`,
/// case-insensitive; `lang_createtable.html`).
///
/// The single source of truth for the rowid-alias vocabulary: the name resolver here
/// and the write-target resolvers (`compile::insert::resolve_target_columns`,
/// `compile::update::column_index`) all consult THIS predicate, so the set cannot
/// drift between how a rowid keyword is read and how it is written.
pub(crate) fn is_rowid_keyword(name: &str) -> bool {
    name.eq_ignore_ascii_case("rowid")
        || name.eq_ignore_ascii_case("_rowid_")
        || name.eq_ignore_ascii_case("oid")
}

/// A column's collating sequence: its declared `COLLATE`, else BINARY.
fn collation_of(col: &ColumnDef) -> Result<Collation> {
    match &col.collation {
        Some(name) => parse_collation(name),
        None => Ok(Collation::Binary),
    }
}

/// Parse a collation name (case-insensitive) to a built-in [`Collation`]. An
/// unknown name is a loud error, matching SQLite's "no such collating sequence".
pub fn parse_collation(name: &str) -> Result<Collation> {
    if name.eq_ignore_ascii_case("binary") {
        Ok(Collation::Binary)
    } else if name.eq_ignore_ascii_case("nocase") {
        Ok(Collation::NoCase)
    } else if name.eq_ignore_ascii_case("rtrim") {
        Ok(Collation::Rtrim)
    } else {
        Err(Error::sql(format!("no such collation sequence: {name}")))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `Source` register/width contract. Most exercise the
    //! `Source::Derived` half (a subquery in FROM / a CTE): a synthetic-schema source with
    //! NO rowid, whose columns resolve by name to `base + i`, pinning the derived contract
    //! the CTE support builds on. The `BaseTable` half is largely exercised end-to-end through
    //! the planner (`compile::from`'s tests and `src/tests.rs`), but the WITHOUT ROWID
    //! width shape (width N, no rowid register) is pinned DIRECTLY here too, since it is the
    //! plan side of the plan<->exec WR row-width contract and a regression to N+1 would
    //! otherwise stay green.
    use super::*;
    use minisqlite_types::affinity_of_declared_type;

    /// A derived column with NONE affinity / BINARY collation — the typeless case these
    /// register/width tests exercise (affinity/collation are irrelevant to the register
    /// mechanics under test here). Production builds a typed derived column from its inner
    /// expression via `compile::from::synth_col_typed` (§3.3), or NONE/BINARY via
    /// `synth_col` for VALUES / JSON-TVF columns.
    fn synth(name: &str) -> SynthCol {
        SynthCol {
            name: name.to_string(),
            affinity: affinity_of_declared_type(None),
            collation: Collation::Binary,
            hidden: false,
        }
    }

    /// A `Source::Derived` named `name` exposing `cols` starting at register `base`.
    /// `'static` because a derived source borrows nothing (unlike `BaseTable`'s
    /// `&TableDef`); it coerces to any shorter scope lifetime by covariance.
    fn derived(name: &str, cols: &[&str], base: usize) -> Source<'static> {
        Source::derived(name.to_string(), cols.iter().map(|c| synth(c)).collect(), base)
    }

    /// A `Coalesced` join column pairing left/right registers for `name`, with NONE
    /// affinity / BINARY collation (the metadata is irrelevant to these register-level
    /// coalescing tests), as `compile::from` builds one.
    fn coalesced(name: &str, left: usize, right: usize) -> Coalesced {
        Coalesced {
            name: name.to_string(),
            left,
            right,
            affinity: affinity_of_declared_type(None),
            collation: Collation::Binary,
        }
    }

    /// The register a `Column` expression reads (panics otherwise). `EvalExpr` carries no
    /// `PartialEq`, so column-resolution tests compare the extracted register instead.
    fn col_reg(e: &EvalExpr) -> usize {
        match e {
            EvalExpr::Column(reg) => *reg,
            other => panic!("expected an EvalExpr::Column, got {other:?}"),
        }
    }

    #[test]
    fn derived_width_is_column_count_with_no_rowid() {
        let d = derived("x", &["a", "b", "c"], 0);
        assert_eq!(d.width(), 3, "a derived source is columns.len(), NOT +1 for a rowid");
        assert_eq!(d.exposed_name(), "x");
        assert_eq!(d.base(), 0);
        assert_eq!(d.column_names(), vec!["a", "b", "c"]);
        assert!(d.has_column("A"), "column match is case-insensitive");
        assert!(!d.has_column("z"));
    }

    #[test]
    fn derived_column_resolves_by_name_to_base_plus_index() {
        // base 5, as if this derived table were the right side of a join: col 0 -> reg 5,
        // col 1 -> reg 6 (a straight `base + i`, no rowid-alias remapping).
        let d = derived("x", &["a", "b"], 5);
        let a = column_in_source(&d, "a").unwrap().expect("a resolves");
        assert_eq!(a.reg, 5);
        assert_eq!(a.affinity, affinity_of_declared_type(None), "derived column has NONE affinity");
        assert_eq!(a.collation, Collation::Binary);
        let b = column_in_source(&d, "B").unwrap().expect("case-insensitive match");
        assert_eq!(b.reg, 6);
        assert!(column_in_source(&d, "z").unwrap().is_none(), "unknown column is None");
    }

    #[test]
    fn derived_source_has_no_rowid() {
        // A derived table (a subquery in FROM) has no integer rowid, so none of the
        // rowid keywords resolve against it — they fall through to a real column or error.
        let d = derived("x", &["a"], 0);
        assert!(rowid_in_source(&d, "rowid").is_none());
        assert!(rowid_in_source(&d, "_rowid_").is_none());
        assert!(rowid_in_source(&d, "oid").is_none());
    }

    #[test]
    fn derived_star_expands_at_base_offsets_excluding_rowid() {
        // `*` over a derived source at base 3 yields each column at `base + i` with its
        // synthetic name, and no trailing rowid pair.
        let sources = [derived("x", &["a", "b"], 3)];
        let scope = Scope::new(&sources);
        let star = scope.expand_star(None).unwrap();
        assert_eq!(star, vec![(3, "a".to_string()), (4, "b".to_string())]);
        // `x.*` (qualified) expands the same source identically.
        assert_eq!(scope.expand_star(Some("x")).unwrap(), star);
    }

    #[test]
    fn rowid_keyword_does_not_resolve_against_a_derived_source() {
        let sources = [derived("x", &["a"], 0)];
        let scope = Scope::new(&sources);
        let err = scope.resolve_column(None, "rowid").unwrap_err();
        assert!(format!("{err:?}").contains("no such column"), "got {err:?}");
    }

    #[test]
    fn qualified_and_unqualified_derived_columns_resolve_to_the_same_register() {
        let sources = [derived("x", &["a", "b"], 0)];
        let scope = Scope::new(&sources);
        let q = scope.resolve_column(Some("x"), "b").expect("x.b resolves");
        let u = scope.resolve_column(None, "b").expect("bare b resolves");
        assert_eq!(q.reg, 1, "b is the 2nd derived column at base 0");
        assert_eq!(u.reg, 1);
        assert_eq!(q.affinity, affinity_of_declared_type(None));
    }

    #[test]
    fn try_resolve_column_reports_not_found_as_none_but_ambiguous_as_err() {
        // Two sources both expose `a`; `b` is unique to the first; `zzz` is in neither.
        let sources = [derived("x", &["a", "b"], 0), derived("y", &["a"], 2)];
        let scope = Scope::new(&sources);
        // A unique name resolves to `Ok(Some(..))`.
        assert!(matches!(scope.try_resolve_column(None, "b"), Ok(Some(_))), "unique name resolves");
        // A name that matches NO column anywhere is the not-found outcome the DQS fallback
        // rides on: `Ok(None)`, NOT an error.
        assert!(matches!(scope.try_resolve_column(None, "zzz"), Ok(None)), "unknown name is Ok(None)");
        // A name that IS a column in two+ sources is AMBIGUOUS: it must stay `Err`, never
        // collapse to `Ok(None)` — otherwise the DQS fallback would turn a real (ambiguous)
        // identifier into a string literal, which real sqlite never does (quirks.html §8).
        let e = scope.try_resolve_column(None, "a").unwrap_err();
        assert!(format!("{e:?}").contains("ambiguous"), "ambiguous name stays an error, got {e:?}");
        // `try_resolve_column_expr` mirrors the same three outcomes.
        assert!(matches!(scope.try_resolve_column_expr(None, "b"), Ok(Some(_))));
        assert!(matches!(scope.try_resolve_column_expr(None, "zzz"), Ok(None)));
        assert!(scope.try_resolve_column_expr(None, "a").is_err());
        // The error-producing `resolve_column` still errors on BOTH not-found and ambiguous,
        // exactly as before this refactor (behavior-preserving for the non-DQS paths).
        assert!(scope.resolve_column(None, "zzz").is_err(), "not-found still errors via resolve_column");
        assert!(scope.resolve_column(None, "a").is_err(), "ambiguous still errors via resolve_column");
    }

    #[test]
    fn total_width_sums_a_derived_source_by_its_own_width() {
        // Two derived sources placed at 0 and 2 give a combined width of 4 (2 + 2), no
        // rowid slack — this is the `base + width` the next (join) source is placed after.
        let sources = [derived("x", &["a", "b"], 0), derived("y", &["c", "d"], 2)];
        let scope = Scope::new(&sources);
        assert_eq!(scope.total_width(), 4);
    }

    /// A catalog `TableDef` for exercising the `Source::BaseTable` half directly. The WR
    /// row-width contract (plan side) needs a unit pin, not only the end-to-end planner
    /// tests; only the fields the `Source` resolvers read matter, and `without_rowid` /
    /// `rowid_alias` select the row-width shape.
    fn base_table_def(
        name: &str,
        cols: &[&str],
        without_rowid: bool,
        rowid_alias: Option<usize>,
    ) -> TableDef {
        TableDef {
            name: name.to_string(),
            columns: cols
                .iter()
                .map(|c| ColumnDef {
                    name: c.to_string(),
                    declared_type: None,
                    not_null: false,
                    primary_key: false,
                    unique: false,
                    collation: None,
                    default: None,
                    default_value: None,
                    generated: None,
                })
                .collect(),
            root_page: 2,
            without_rowid,
            rowid_alias,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        }
    }

    #[test]
    fn without_rowid_base_source_is_width_n_and_hides_rowid() {
        // The plan half of the WR row-width contract: a WITHOUT ROWID base table is width
        // N (its columns, NO trailing rowid register), and the rowid keywords do not
        // resolve against it. A regression to N+1 here would desync the plan from the
        // executor's width-N WR scan (and shift any following join source by one register).
        let wr = base_table_def("w", &["a", "b"], true, None);
        let wr_src = Source::BaseTable { exposed_name: "w".to_string(), table: &wr, db: DbIndex::MAIN, base: 0 };
        assert_eq!(wr_src.width(), 2, "a WR base source is width N (no rowid register)");
        assert!(rowid_in_source(&wr_src, "rowid").is_none(), "WR exposes no rowid");
        assert!(rowid_in_source(&wr_src, "oid").is_none(), "WR exposes no oid");

        // Contrast: a plain rowid table IS N+1, with the rowid at register base+N.
        let rowid = base_table_def("r", &["a", "b"], false, None);
        let rowid_src = Source::BaseTable { exposed_name: "r".to_string(), table: &rowid, db: DbIndex::MAIN, base: 0 };
        assert_eq!(rowid_src.width(), 3, "a rowid base source is width N+1");
        assert_eq!(
            rowid_in_source(&rowid_src, "rowid").expect("rowid resolves").reg,
            2,
            "the rowid sits at base+N",
        );
    }

    #[test]
    fn without_rowid_source_places_the_next_join_source_after_n_registers() {
        // `total_width` (= max `base + width`) is where the next join source is placed, so
        // a WR base at 0 must contribute exactly N — not N+1, which would leave a phantom
        // rowid slot and shift a join partner's registers by one.
        let wr = base_table_def("w", &["a", "b"], true, None);
        let sources = [Source::BaseTable { exposed_name: "w".to_string(), table: &wr, db: DbIndex::MAIN, base: 0 }];
        let scope = Scope::new(&sources);
        assert_eq!(scope.total_width(), 2, "a WR base source spans exactly N registers");
    }

    #[test]
    fn without_rowid_star_expands_columns_at_base_plus_i() {
        // `*` over a WR source yields each column at `base + i` (no rowid remap, no
        // trailing rowid pair), so a non-first column lands on the register the executor's
        // width-N row actually holds it in.
        let wr = base_table_def("w", &["a", "b", "c"], true, None);
        let sources = [Source::BaseTable { exposed_name: "w".to_string(), table: &wr, db: DbIndex::MAIN, base: 0 }];
        let scope = Scope::new(&sources);
        assert_eq!(
            scope.expand_star(None).unwrap(),
            vec![(0, "a".to_string()), (1, "b".to_string()), (2, "c".to_string())],
        );
    }

    #[test]
    fn upsert_co_row_parent_scope_reports_2w_and_resolves_both_halves() {
        // The UPSERT DO UPDATE scope (`compile::insert::compile_upsert`): the target table
        // is the CHILD at base 0 (exposed under its own name → the existing row) and the
        // SAME table is the PARENT at base `W` exposed as `excluded` (the candidate row).
        // The executor forms ONE combined `existing(W) ++ excluded(W)` row, so this co-row
        // scope MUST report `total_width() == 2W` — the outer width a correlated subquery in
        // the SET/WHERE places its own sources after. A regression to `W` (ignoring the
        // co-row parent) rebases such a subquery onto the excluded half and reads garbage,
        // exactly the bug the `.max(parent)` in `total_width` repairs.
        let t = base_table_def("t", &["a", "b"], false, None);
        let w = Source::BaseTable { exposed_name: "t".to_string(), table: &t, db: DbIndex::MAIN, base: 0 }.width();
        assert_eq!(w, 3, "a 2-column rowid table is width N+1 = 3");
        let existing = [Source::BaseTable { exposed_name: "t".to_string(), table: &t, db: DbIndex::MAIN, base: 0 }];
        let excluded =
            [Source::BaseTable { exposed_name: "excluded".to_string(), table: &t, db: DbIndex::MAIN, base: w }];
        let excluded_scope = Scope::new(&excluded);
        let scope = Scope::with_parent(&existing, Some(&excluded_scope));

        assert_eq!(scope.total_width(), 2 * w, "co-row existing ++ excluded is 2W wide");
        // A bare / table-qualified column reads the EXISTING half `[0, W)` (unambiguous —
        // `resolve_unqualified` only counts the child sources); `excluded.col` reaches the
        // candidate half `[W, 2W)` through the parent — the register layout `do_upsert_update`
        // fills.
        assert_eq!(scope.resolve_column(None, "a").unwrap().reg, 0, "bare a → existing col 0");
        assert_eq!(scope.resolve_column(Some("t"), "b").unwrap().reg, 1, "t.b → existing col 1");
        assert_eq!(scope.resolve_column(Some("excluded"), "a").unwrap().reg, w, "excluded.a → W");
        assert_eq!(
            scope.resolve_column(Some("excluded"), "b").unwrap().reg,
            w + 1,
            "excluded.b → W+1",
        );
    }

    /// A `Scope` over two sources sharing a column `k`, with `k` coalesced (left copy at
    /// 0, right copy at 2) — the shape `compile::from` builds for `x JOIN y USING (k)`.
    fn coalesced_join_scope<'a>(
        sources: &'a [Source<'a>],
        coalesced: &'a [Coalesced],
    ) -> Scope<'a> {
        Scope {
            sources,
            coalesced,
            parent: None,
            grouping: None,
            saw_correlated: None,
            correlated_cols: None,
            nondeterministic: None,
            windowing: None,
        }
    }

    #[test]
    fn coalesced_column_resolves_unambiguously_to_a_coalesce_expr() {
        // `x(k,a) JOIN y(k,b) USING (k)`: k appears in both sides (regs 0 and 2), but a
        // bare `k` must NOT be ambiguous — it resolves to COALESCE(Col(0), Col(2)).
        let sources = [derived("x", &["k", "a"], 0), derived("y", &["k", "b"], 2)];
        let coalesced = [coalesced("k", 0, 2)];
        let scope = coalesced_join_scope(&sources, &coalesced);

        match scope.resolve_column_expr(None, "k").expect("bare k resolves") {
            EvalExpr::Coalesce(items) => {
                let regs: Vec<usize> = items.iter().map(col_reg).collect();
                assert_eq!(regs, vec![0, 2]);
            }
            other => panic!("expected COALESCE(left,right), got {other:?}"),
        }
        // The single-register resolver picks the LEFT copy (for affinity/collation).
        assert_eq!(scope.resolve_column(None, "k").unwrap().reg, 0);
        // A non-shared column is unaffected: a bare `a` is a plain Column.
        assert_eq!(col_reg(&scope.resolve_column_expr(None, "a").unwrap()), 1);
    }

    #[test]
    fn coalesced_column_qualified_reference_reaches_each_side_directly() {
        // A qualified reference is never coalesced: `x.k` -> reg 0, `y.k` -> reg 2.
        let sources = [derived("x", &["k", "a"], 0), derived("y", &["k", "b"], 2)];
        let coalesced = [coalesced("k", 0, 2)];
        let scope = coalesced_join_scope(&sources, &coalesced);
        assert_eq!(col_reg(&scope.resolve_column_expr(Some("x"), "k").unwrap()), 0);
        assert_eq!(col_reg(&scope.resolve_column_expr(Some("y"), "k").unwrap()), 2);
    }

    #[test]
    fn star_over_a_coalesced_join_omits_the_right_copy() {
        // `SELECT *` over `x(k,a) JOIN y(k,b) USING (k)` yields k (left), a, b — the
        // right k is dropped so the shared column appears exactly once.
        let sources = [derived("x", &["k", "a"], 0), derived("y", &["k", "b"], 2)];
        let coalesced = [coalesced("k", 0, 2)];
        let scope = coalesced_join_scope(&sources, &coalesced);
        assert_eq!(
            scope.expand_star(None).unwrap(),
            vec![(0, "k".to_string()), (1, "a".to_string()), (3, "b".to_string())],
        );
        // A qualified `y.*` still shows ALL of y's columns, including its own k copy.
        assert_eq!(
            scope.expand_star(Some("y")).unwrap(),
            vec![(2, "k".to_string()), (3, "b".to_string())],
        );
    }

    #[test]
    fn star_exprs_coalesces_the_shared_column_only_when_unqualified() {
        // `SELECT *` over `x(k,a) JOIN y(k,b) USING (k)` projects the shared k as
        // COALESCE(Col(0), Col(2)) — the SAME value a bare `k` binds to via
        // `resolve_column_expr` — while a and b stay plain columns. This agreement is
        // what makes `*` correct on a RIGHT/FULL outer join's unmatched-right row (the
        // left copy NULL-extended, the right copy present); a plain `Column(left)` there
        // would read NULL and diverge from both the bare name and real sqlite.
        let sources = [derived("x", &["k", "a"], 0), derived("y", &["k", "b"], 2)];
        let coalesced = [coalesced("k", 0, 2)];
        let scope = coalesced_join_scope(&sources, &coalesced);

        let exprs = scope.expand_star_exprs(None).unwrap();
        let names: Vec<&str> = exprs.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["k", "a", "b"], "right copy of k dropped; shape unchanged");
        match &exprs[0].0 {
            EvalExpr::Coalesce(items) => {
                assert_eq!(items.iter().map(col_reg).collect::<Vec<_>>(), vec![0, 2]);
            }
            other => panic!("expected COALESCE(left,right) for the shared k, got {other:?}"),
        }
        assert_eq!(col_reg(&exprs[1].0), 1, "a is a plain column");
        assert_eq!(col_reg(&exprs[2].0), 3, "b is a plain column");

        // A qualified `x.*` names one table and never coalesces: its k is a plain column.
        let qexprs = scope.expand_star_exprs(Some("x")).unwrap();
        assert_eq!(col_reg(&qexprs[0].0), 0, "x.* projects x.k as a plain Column, not COALESCE");
        assert_eq!(col_reg(&qexprs[1].0), 1);
    }

    #[test]
    fn coalesced_column_folds_n_ary_across_a_three_table_using_chain() {
        // `x(k,a) JOIN y(k,b) USING(k) JOIN z(k,c) USING(k)` is left-associative, so the
        // FROM compiler records TWO same-name coalesced entries, both sharing x.k (reg 0)
        // as the left copy: {k: x.k(0), y.k(2)} then {k: x.k(0), z.k(4)}. A bare `k` and
        // `*`'s copy of it must fold ALL THREE distinct registers into one COALESCE, not
        // just the first pair — otherwise z.k is dropped and an outer row present only in z
        // reads NULL. Order is insertion order (left once, then each right): 0, 2, 4.
        let sources =
            [derived("x", &["k", "a"], 0), derived("y", &["k", "b"], 2), derived("z", &["k", "c"], 4)];
        let coalesced = [coalesced("k", 0, 2), coalesced("k", 0, 4)];
        let scope = coalesced_join_scope(&sources, &coalesced);

        // A bare `k` folds x.k, y.k, z.k (regs 0, 2, 4) — the whole-chain value.
        match scope.resolve_column_expr(None, "k").expect("bare k resolves") {
            EvalExpr::Coalesce(items) => {
                assert_eq!(
                    items.iter().map(col_reg).collect::<Vec<_>>(),
                    vec![0, 2, 4],
                    "bare k must COALESCE all three copies, not drop the third table's",
                );
            }
            other => panic!("expected an N-ary COALESCE for the chained k, got {other:?}"),
        }

        // `SELECT *` drops y.k and z.k (the right copies) and projects the surviving k as
        // the SAME three-way COALESCE, keeping `*` and the bare name in agreement.
        let exprs = scope.expand_star_exprs(None).unwrap();
        let names: Vec<&str> = exprs.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["k", "a", "b", "c"], "one shared k, then a, b, c");
        match &exprs[0].0 {
            EvalExpr::Coalesce(items) => {
                assert_eq!(
                    items.iter().map(col_reg).collect::<Vec<_>>(),
                    vec![0, 2, 4],
                    "the `*` copy of k must fold the whole chain too",
                );
            }
            other => panic!("expected an N-ary COALESCE for the `*` k, got {other:?}"),
        }
        assert_eq!(col_reg(&exprs[1].0), 1, "a is a plain column");
        assert_eq!(col_reg(&exprs[2].0), 3, "b is a plain column");
        assert_eq!(col_reg(&exprs[3].0), 5, "c is a plain column");
    }
}
