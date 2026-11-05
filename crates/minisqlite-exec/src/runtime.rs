//! [`Runtime`] — the per-connection state expression evaluation reaches for, threaded
//! through every [`RowCursor::next_row`](crate::RowCursor::next_row) pull.
//!
//! It holds four things a statement cannot compute from its rows alone:
//! * a deterministic PRNG for `random()` / `randomblob()` (seeded, SplitMix64, so a
//!   run is reproducible),
//! * the `last_insert_rowid` from the most recent successful `INSERT`,
//! * the per-statement `changes` and the connection-lifetime `total_changes`,
//! * the bound parameters (`?N` / named), read by `EvalExpr::Param`.
//!
//! DML operators mutate the counters (`record_insert` / `record_change`) and the
//! engine reads them back after draining the cursor; the RNG is mutated in place by
//! the function-context callbacks. Everything else (reads) leaves it untouched.
//!
//! It also carries three pieces of *per-execution* scratch, cleared/managed within a
//! statement rather than living for the connection's lifetime: the recursive-CTE
//! working-table stack, the uncorrelated-subquery result cache (see [`CachedSubquery`]
//! and [`clear_subquery_cache`](Runtime::clear_subquery_cache)), and the memo of
//! trigger sets re-compiled for runtime recursion. Finally it holds the trigger-firing
//! depth (bounding recursion) and the connection-level `recursive_triggers` flag (the
//! `PRAGMA` gate that decides whether a trigger-action DML fires further triggers).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use minisqlite_plan::TriggerProgram;
use minisqlite_types::{DbIndex, Error, Result, Row, Value};

use crate::corr_key::CorrCell;
use crate::keys::CellKey;

/// The default PRNG seed. Fixed so a `random()` sequence is reproducible across runs
/// (tests can drive a known sequence). SQLite's own `random()` is
/// not seedable from SQL; determinism here is a deliberate, testable choice.
const DEFAULT_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// The maximum number of distinct `(subquery-id, correlation-key)` entries the
/// [correlated-subquery cache](Runtime::correlated_subquery_cache) holds in TOTAL before it
/// stops accepting NEW keys within a statement. A tunable bound so peak memory stays bounded
/// even when the
/// correlating outer columns are high-cardinality (peak RSS matters here): once the
/// cache is full, a miss simply re-runs the subquery (today's behavior) rather than growing
/// the map without limit. 65,536 entries is generous for the low-cardinality correlation
/// this optimization targets (~tens of distinct keys) while capping a pathological case.
const CORRELATED_CACHE_CAP: usize = 65_536;

/// A cached result of an *uncorrelated* expression subquery, keyed by its
/// `SubqueryId` (its index in `Plan::subqueries`) in [`Runtime::subquery_cache`].
///
/// The SQL contract this exists to satisfy (lang_expr.html §12): "An uncorrelated
/// subquery is evaluated only once and the result reused as necessary." So the
/// first evaluation of an uncorrelated subquery within a statement stores its result
/// here and every later use in that same statement reads it back instead of re-running
/// the subplan — which is *correctness* (a volatile `(SELECT random())` must yield the
/// same value across the outer rows) as much as it is performance (`x IN (SELECT ...)`
/// scans the inner table once, not once per outer row). One variant per subquery
/// callback shape.
#[derive(Debug)]
pub(crate) enum CachedSubquery {
    /// A scalar subquery's value: the first column of the first row, or NULL if empty.
    Scalar(Value),
    /// A row-value subquery's FIRST row, materialized once for a column-list `UPDATE`
    /// source (`SET (a, b, …) = (SELECT …)`) whose N `ScalarSubqueryColumn` assignments
    /// all share the one subplan id and each need a positional column of the SAME first
    /// row. An EMPTY vec is the "no rows" sentinel (every column then reads back NULL via
    /// `get(col)`), so the subplan runs once even when it returns nothing — the same
    /// once-only rule as [`Scalar`](CachedSubquery::Scalar) (lang_expr.html §12).
    FirstRow(Row),
    /// An `EXISTS` subquery's truth: whether the subplan yields at least one row.
    Exists(bool),
    /// An `IN (subquery)` candidate set, materialized once for O(1) membership probes.
    /// `set` holds the non-NULL candidate keys after right-affinity coercion, folded
    /// by the comparison collation (so the probe is a plain `HashSet::contains`).
    /// `set_has_null` records whether any candidate coerced to NULL, and `any_rows`
    /// whether the subplan produced any row at all — both needed to reproduce SQL's
    /// three-valued `IN` exactly (an empty set is FALSE even for a NULL probe, whereas
    /// a non-empty set with no match but a NULL present is UNKNOWN).
    InSet { set: HashSet<CellKey>, set_has_null: bool, any_rows: bool },
    /// A row-value `(a, …) IN (subquery)` candidate list, materialized once. Unlike the
    /// scalar [`InSet`](CachedSubquery::InSet), a tuple probe cannot use a plain hash set:
    /// per-element NULLs make membership three-valued at the *row* level (a candidate can
    /// be an UNKNOWN match — every non-NULL element equal, some element NULL), so the
    /// probe scans these rows applying the tuple AND3 (rowvalue.html §2.2). Caching still
    /// runs the subplan once (correctness for a volatile subquery, and one inner scan).
    InRows(Vec<Row>),
}

/// The key a re-compiled trigger set is memoized under in [`Runtime::recompiled_triggers`]:
/// the (case-folded) target table name plus the DML event. `Update` carries the assigned
/// column indices, since `UPDATE OF <cols>` filtering makes two updates of the same table
/// with different assigned columns fire different trigger sets. Owned (unlike the
/// borrow-carrying `TriggerDmlEvent`) so it can live in the cache across firings.
///
/// The full memo key pairs this with the target's [`DbIndex`] namespace (see
/// [`Runtime::recompiled_triggers`]): under `recursive_triggers`, a temp/attached table can
/// SHADOW a same-named main table, so keying on the name alone would return the wrong
/// namespace's recompiled triggers. The `db` disambiguates them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TriggerEventKey {
    Insert,
    Delete,
    Update(Vec<usize>),
}

/// The change counters a trigger program's firing must not leak into its caller,
/// captured by [`Runtime::enter_trigger`] and restored by [`Runtime::exit_trigger`].
///
/// SQLite's `changes()` is "exclusive of statements in lower-level triggers"
/// (lang_corefunc.html) and `last_insert_rowid()` "reverts to what it was before the
/// trigger was fired" once the trigger program ends (c3ref/last_insert_rowid.html), while
/// `total_changes()` MUST keep counting trigger-caused changes. Snapshotting these two on
/// enter and restoring them on exit (leaving `total_changes` to accumulate) makes every
/// nested DML write invisible to the outer `changes()`/`last_insert_rowid()` at once, and
/// nests correctly under recursion (each level restores its own baseline).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriggerSavepoint {
    changes: i64,
    last_insert_rowid: i64,
}

/// Both per-statement subquery caches lifted out together across a trigger-action
/// re-entrancy boundary by [`take_subquery_cache`](Runtime::take_subquery_cache) and put
/// back by [`restore_subquery_cache`](Runtime::restore_subquery_cache).
///
/// Carrying BOTH the uncorrelated and the correlated map in one opaque token is a
/// correctness choice: the trigger runner treats it as a pass-through handle (take → run
/// action → restore), so a caller CANNOT scope one cache and forget the other. The
/// correlated cache is thus scoped under triggers with no extra wiring — the bad state (one
/// map scoped, the other bleeding into a nested action) is unrepresentable.
pub(crate) struct SubqueryCaches {
    uncorrelated: HashMap<usize, CachedSubquery>,
    correlated: HashMap<usize, HashMap<Vec<CorrCell>, CachedSubquery>>,
}

/// Per-connection runtime state threaded through execution. Created once per
/// connection (or per statement run) and handed to each `next_row` pull by `&mut`.
pub struct Runtime {
    /// SplitMix64 state — advanced on each draw so successive `random()` calls differ.
    rng_state: u64,
    /// Rowid of the most recent successful `INSERT` on this connection.
    last_insert_rowid: i64,
    /// Rows changed by the most recent `INSERT`/`UPDATE`/`DELETE` statement.
    changes: i64,
    /// Total rows changed since the connection was opened.
    total_changes: i64,
    /// Bound statement parameters, indexed by `EvalExpr::Param`.
    params: Vec<Value>,
    /// The recursive-CTE working-table STACK. Each recursive-CTE step round pushes
    /// the current frontier (the rows the previous round produced) so a
    /// `RecursiveScan` nested in the step can read it; the round pops it afterward.
    /// A stack (not a single slot) so nested recursive CTEs each address their OWN
    /// frontier — the innermost is always on top. Empty outside any recursive step.
    recursive_frames: Vec<Vec<Row>>,
    /// Per-statement cache of *uncorrelated* subquery results, keyed by `SubqueryId`.
    /// Populated on the first evaluation of each uncorrelated subquery in a statement
    /// and read back on every later use, so an uncorrelated subquery is evaluated
    /// exactly once (lang_expr.html §12). Cleared at the START of each statement's
    /// drain (via `StatementRoot`) so it never bleeds across statements — the same
    /// `Runtime` is reused for every statement on a connection and `SubqueryId`s
    /// restart at 0 per plan. See [`CachedSubquery`].
    subquery_cache: HashMap<usize, CachedSubquery>,
    /// Per-statement cache of *correlated* subquery results, keyed by a `SubqueryId` and, per
    /// id, by the VALUES of the outer registers that subplan depends on — a fold-free
    /// [`CorrCell`] per [`SubPlan::correlated_cols`](minisqlite_plan::SubPlan) entry, in that
    /// sorted order. A correlated subquery is re-run per outer row today; this lets an
    /// evaluator memoize the result for one DISTINCT outer-value combination so a
    /// low-cardinality correlation runs the subplan once per distinct key instead of once
    /// per row. Bounded by [`CORRELATED_CACHE_CAP`] (TOTAL entries across all ids) and
    /// cleared / scoped on the SAME per-statement + trigger-action lifecycle as
    /// [`subquery_cache`](Runtime::subquery_cache) (see
    /// [`clear_subquery_cache`](Runtime::clear_subquery_cache) /
    /// [`take_subquery_cache`](Runtime::take_subquery_cache)) so it never bleeds across
    /// statements or into a nested trigger action.
    ///
    /// Shape: `id -> (correlation-key -> result)`, a NESTED map rather than a flat
    /// `(id, key)` one, specifically so the hot HIT probe allocates nothing. The probe runs
    /// once per outer row on the low-cardinality path this optimization targets; the outer
    /// `get(&id)` is by a copy `usize`, and the inner `get(key: &[CorrCell])` borrows the
    /// caller's already-built key slice (`Vec<CorrCell>: Borrow<[CorrCell]>`) — so a hit does
    /// NOT clone the key (a flat `(id, key)` map would rebuild the whole tuple, deep-copying
    /// Text/Blob key bytes, on every probe). The correlation key is still built once per
    /// outer-row eval (unavoidable — it IS the lookup key); only the redundant per-probe
    /// clone is removed.
    ///
    /// Populated by the correlated-subquery eval path (the `EvalCtx` subquery callbacks in
    /// `context.rs`) on a miss and read on a hit, so a correlated subquery whose result is
    /// stable for a given outer-value combination runs its subplan once per DISTINCT key
    /// instead of once per outer row. Only *eligible* correlated subqueries populate it —
    /// a read-only statement, a deterministic subplan, and a non-empty correlation key
    /// (see `context::correlated_memo_eligible`); an ineligible correlated subquery re-runs
    /// per outer row and never touches this map, preserving pre-memo behavior exactly.
    correlated_subquery_cache: HashMap<usize, HashMap<Vec<CorrCell>, CachedSubquery>>,
    /// How many trigger programs are currently on the firing stack. A DML operator
    /// bumps this while it runs one trigger's actions (via [`enter_trigger`] /
    /// [`exit_trigger`]) so a trigger whose action fires the same event again is
    /// bounded: past [`MAX_TRIGGER_DEPTH`] the enter fails rather than recursing without
    /// end. Zero outside any firing; balanced (every enter has a matching exit, even
    /// when an action errors), so it starts at 0 for each new top-level statement.
    trigger_depth: usize,
    /// Whether `PRAGMA recursive_triggers` is ON. SQLite defaults it OFF: a DML
    /// statement inside a trigger body does NOT fire further triggers unless this is
    /// set (pragma.html). The DML operators gate the runtime recompile-and-fire of a
    /// nested action's own triggers on it, so a trigger-action write fires no further
    /// triggers by default. Connection-lifetime (not per-statement); set by the engine's
    /// PRAGMA handler.
    recursive_triggers: bool,
    /// Whether `PRAGMA foreign_keys` is ON. SQLite defaults it OFF (pragma.html;
    /// version 3.6.19+): FK constraints are not enforced unless this is set. It is the
    /// gate a later FK-enforcement pass consults before checking parent keys / running
    /// referential actions. Recorded here (connection-lifetime, like
    /// [`recursive_triggers`](Runtime::recursive_triggers)) so that gate has one home;
    /// the engine only stores + reports it — no enforcement reads it yet.
    foreign_keys: bool,
    /// Whether a DEFERRED foreign-key check may be deferred RIGHT NOW — i.e. an explicit
    /// or savepoint-started transaction is open to defer the check INTO. Set by the engine
    /// to [`SqlEngine::txn_active`](../../minisqlite_engine) before every mutating statement
    /// runs. It is the transaction half of the deferral predicate (see [`fk_deferred_now`]):
    /// in AUTOCOMMIT no transaction is open, so this is false and a `DEFERRABLE INITIALLY
    /// DEFERRED` constraint is checked immediately, exactly like an immediate one
    /// (`foreignkeys.html` §4.2 — "in [autocommit] deferred constraints behave the same as
    /// immediate constraints"). Per-statement scratch, not connection-lifetime: it defaults
    /// to false and is re-set each statement, so it can never leave the executor thinking a
    /// deferral is possible when no transaction is active.
    ///
    /// [`fk_deferred_now`]: crate::ops::foreign_key::fk_deferred_now
    fk_defer_active: bool,
    /// Whether `PRAGMA defer_foreign_keys` is ON — the pragma that "temporarily change[s]
    /// all foreign key constraints to deferred regardless of how they are declared"
    /// (`pragma.html` #pragma_defer_foreign_keys) for the current transaction. When true AND
    /// a transaction is open ([`fk_defer_active`](Runtime::fk_defer_active)), EVERY FK — not
    /// only the declared-deferred ones — is checked at COMMIT rather than at statement time.
    /// It is automatically switched OFF at each COMMIT and ROLLBACK (the engine resets it in
    /// `exec_commit`/`exec_rollback` and at the end of each autocommit statement's implicit
    /// transaction), so it never leaks past the transaction it applied to.
    /// SQLite defaults it OFF; a RESTRICT action stays immediate even under it
    /// (`foreignkeys.html` §4.3), which the enforcement paths handle by never routing RESTRICT
    /// through the deferral predicate.
    defer_foreign_keys: bool,
    /// STICKY "the defer pragma was ON at some point during this transaction" — the COMMIT
    /// recheck's coverage flag, distinct from the LIVE [`defer_foreign_keys`] the per-statement
    /// skip reads. It is set true whenever `defer_foreign_keys` is turned ON and, unlike the
    /// live flag, is NOT cleared when the pragma is toggled back OFF mid-transaction — only at
    /// transaction end (`reset_defer_foreign_keys`). This closes a desync: a row allowed
    /// to sit orphaned while the pragma was ON must still be RESCANNED at COMMIT even if the
    /// pragma was turned OFF before COMMIT, or an FK-violating row would commit with
    /// `foreign_keys` ON — contradicting §4.2 ("COMMIT will fail as long as foreign key
    /// constraints remain in violation"). The commit recheck reads THIS (via
    /// [`defer_foreign_keys_armed`](Runtime::defer_foreign_keys_armed)) as its `defer_all`, so
    /// coverage can only ever GROW within a txn, never shrink. Turning the pragma OFF→ON→OFF
    /// leaves it armed; a declared-DEFERRED FK is unaffected (always rescanned via `fk.deferred`).
    defer_foreign_keys_armed: bool,
    /// Per-statement memo of the trigger sets recompiled for RUNTIME recursion, keyed by
    /// (target table, event). A nested trigger-action DML carries an empty compiled
    /// `triggers` vec (the one-level compile bound), so the executor recompiles the
    /// action target's own triggers to recurse — but the result does not change within a
    /// statement (schema is stable, the action is fixed), so it is computed once per
    /// (table, event) here and reused across the firings (once per outer row) instead of
    /// re-parsing/binding/planning every row. Held behind `Arc` so a DML operator takes a
    /// cheap handle it can iterate while it mutates the rest of the runtime. Cleared per
    /// statement (same lifecycle as [`subquery_cache`](Runtime::subquery_cache)) so a
    /// `CREATE`/`DROP TRIGGER` between statements is picked up.
    recompiled_triggers: HashMap<(DbIndex, String, TriggerEventKey), Arc<Vec<TriggerProgram>>>,
    /// The ephemeral JSON value-subtype channel (json1.html §3.4), backing the four
    /// [`FnContext`](minisqlite_expr::FnContext) subtype methods the executor's
    /// `EvalCtx` implements. `arg_subtypes[i]` is the subtype of the current call's
    /// argument `i` (`0` = none) and `result_subtype` is the subtype the function in
    /// hand marked on its own result; both are set immediately before a function call
    /// and read during / right after it, so the state never outlives one function
    /// invocation. It lives on the connection-lifetime `Runtime` (not on the per-eval
    /// `EvalCtx`, which is rebuilt at ~35 operator sites) so the `arg_subtypes` buffer
    /// is allocated once and reused across every call, and so adding the channel does
    /// not perturb those construction sites. A JSON function can never re-enter the
    /// evaluator (its `FnContext` cannot run subqueries), so at most one call is live
    /// at a time and this single slot cannot be clobbered mid-call.
    arg_subtypes: Vec<u8>,
    /// The subtype the just-called function marked on its result (`0` = none); see
    /// [`arg_subtypes`](Runtime::arg_subtypes).
    result_subtype: u8,
    /// TEST/DIAGNOSTIC ONLY: when `true`, the correlated-subquery memo is bypassed and
    /// every correlated subquery re-runs its subplan per outer row (the exact pre-memo
    /// behavior). Defaults to `false` (memo active). It exists so tests
    /// can prove cache-on == cache-off through the real eval path; normal operation never
    /// sets it and it is not reachable from the facade. This is a plain runtime bool, NOT a
    /// behavior-selecting cargo feature — it selects nothing at build time and both paths
    /// compile into the one live route.
    correlated_cache_disabled: bool,
    /// `RAISE(IGNORE)` control signal (lang_createtrigger.html §RAISE). A trigger body's
    /// `RAISE(IGNORE)` cannot be an [`Error`] — it is a non-error "abandon the current
    /// row's operation and continue" signal the pinned 4-variant `Error` cannot carry — so
    /// the evaluator sets this flag (via [`EvalContext::signal_raise_ignore`]) and returns
    /// a sentinel `Err` to unwind out of the trigger body. [`fire_triggers`] then TAKES the
    /// flag and, when set, reports [`TriggerFlow::IgnoreRow`] instead of the error, and the
    /// DML operator skips the current row. It lives here (not on the per-eval `EvalCtx`,
    /// rebuilt at ~35 sites) so the one boolean survives the unwind from `eval` up to the
    /// DML row loop. Set-and-immediately-taken within one row's trigger fire, so it never
    /// bleeds across rows or statements.
    ///
    /// [`fire_triggers`]: crate::ops::trigger::fire_triggers
    /// [`TriggerFlow::IgnoreRow`]: crate::ops::trigger::TriggerFlow
    raise_ignore: bool,
}

/// The maximum trigger-recursion nesting the executor allows before it errors, so a
/// self-referential trigger (or a cyclic chain) cannot recurse without end and exhaust
/// the stack. Mirrors SQLite's default `SQLITE_MAX_TRIGGER_DEPTH` (1000); a correct
/// `WHEN` bound normally terminates the recursion long before this backstop.
const MAX_TRIGGER_DEPTH: usize = 1000;

impl Runtime {
    /// A runtime with the fixed default seed and no bound parameters.
    pub fn new() -> Runtime {
        Runtime {
            rng_state: DEFAULT_SEED,
            last_insert_rowid: 0,
            changes: 0,
            total_changes: 0,
            params: Vec::new(),
            recursive_frames: Vec::new(),
            subquery_cache: HashMap::new(),
            correlated_subquery_cache: HashMap::new(),
            trigger_depth: 0,
            recursive_triggers: false,
            foreign_keys: false,
            fk_defer_active: false,
            defer_foreign_keys: false,
            defer_foreign_keys_armed: false,
            recompiled_triggers: HashMap::new(),
            arg_subtypes: Vec::new(),
            result_subtype: 0,
            correlated_cache_disabled: false,
            raise_ignore: false,
        }
    }

    /// A runtime seeded as [`new`](Runtime::new) but carrying bound parameters.
    pub fn with_params(params: Vec<Value>) -> Runtime {
        Runtime { params, ..Runtime::new() }
    }

    /// Replace the bound parameters (e.g. re-binding a prepared statement).
    pub fn set_params(&mut self, params: Vec<Value>) {
        self.params = params;
    }

    /// Rows changed by the most recent statement (the SQL `changes()`).
    pub fn changes(&self) -> i64 {
        self.changes
    }

    /// Rowid of the most recent successful `INSERT` (the SQL `last_insert_rowid()`).
    pub fn last_insert_rowid(&self) -> i64 {
        self.last_insert_rowid
    }

    /// Total rows changed since the connection opened (the SQL `total_changes()`).
    pub fn total_changes(&self) -> i64 {
        self.total_changes
    }

    /// Record a successful `INSERT` of `rowid`: it becomes `last_insert_rowid` and
    /// both change counters advance. Used by the DML `INSERT` operator per row.
    pub fn record_insert(&mut self, rowid: i64) {
        self.last_insert_rowid = rowid;
        self.changes += 1;
        self.total_changes += 1;
    }

    /// Record one changed row that is not an `INSERT` (an `UPDATE`/`DELETE`): the
    /// change counters advance but `last_insert_rowid` is left alone.
    pub fn record_change(&mut self) {
        self.changes += 1;
        self.total_changes += 1;
    }

    /// Reset the per-statement `changes` counter to zero at the start of a statement.
    /// `total_changes` and `last_insert_rowid` persist across statements, so they are
    /// deliberately not reset here.
    pub fn reset_statement_changes(&mut self) {
        self.changes = 0;
    }

    /// Look up an uncorrelated subquery's cached result by its `SubqueryId`. `None`
    /// is a cache MISS — the first use of that subquery within a statement, or any
    /// use after [`clear_subquery_cache`](Runtime::clear_subquery_cache).
    pub(crate) fn cached_subquery(&self, id: usize) -> Option<&CachedSubquery> {
        self.subquery_cache.get(&id)
    }

    /// Store an uncorrelated subquery's once-evaluated result under its `SubqueryId`,
    /// so every later use in the same statement reuses it. Only the caller in
    /// `context.rs` populates this, and only for `SubPlan.correlated == false`.
    pub(crate) fn cache_subquery(&mut self, id: usize, entry: CachedSubquery) {
        self.subquery_cache.insert(id, entry);
    }

    /// Drop every cached subquery result — BOTH the uncorrelated and the correlated map.
    /// Invoked once at the START of each statement's drain (by `StatementRoot`, the wrapper
    /// the executor puts at the root of every statement cursor) so a cached value from one
    /// statement can never be read by the next — the same `Runtime`, hence the same cache
    /// maps, is reused for every statement on a connection, and `SubqueryId`s restart at 0
    /// for each plan. Clearing both here is why the correlated cache needs no separate
    /// statement-boundary reset.
    ///
    /// CRITICAL: this clears ONLY the two subquery caches. It must NOT touch `changes`,
    /// `total_changes`, `last_insert_rowid`, the RNG, or the bound parameters — those
    /// are connection-lifetime (or explicitly statement-reset elsewhere) and a
    /// sequence like `INSERT ...; SELECT changes();` depends on them surviving here.
    pub(crate) fn clear_subquery_cache(&mut self) {
        self.subquery_cache.clear();
        self.correlated_subquery_cache.clear();
    }

    /// Take BOTH subquery caches OUT (leaving empty ones behind), returning them as one
    /// [`SubqueryCaches`] token the caller restores later with
    /// [`restore_subquery_cache`](Runtime::restore_subquery_cache).
    ///
    /// Used to scope the caches across a trigger-action re-entrancy boundary: a trigger
    /// action is a separately-compiled `Plan` whose `SubqueryId`s restart at 0, run NESTED
    /// inside the enclosing statement's drain (not through `StatementRoot`), so without
    /// this the action's subquery id 0 would collide with the enclosing statement's cached
    /// id 0 — a variant mismatch error, or (worse) the action silently reading the
    /// enclosing statement's data. The enclosing caches are taken here, the action runs
    /// with fresh ones, and they are put back afterward — the same per-plan reset
    /// `StatementRoot` performs at the statement boundary, reopened at the action boundary.
    /// Both maps move together so the correlated cache is scoped identically (see
    /// [`SubqueryCaches`]).
    pub(crate) fn take_subquery_cache(&mut self) -> SubqueryCaches {
        SubqueryCaches {
            uncorrelated: std::mem::take(&mut self.subquery_cache),
            correlated: std::mem::take(&mut self.correlated_subquery_cache),
        }
    }

    /// Restore both subquery caches previously removed by
    /// [`take_subquery_cache`](Runtime::take_subquery_cache), discarding whatever entries
    /// the nested run left. Pairs with `take_subquery_cache`.
    pub(crate) fn restore_subquery_cache(&mut self, caches: SubqueryCaches) {
        self.subquery_cache = caches.uncorrelated;
        self.correlated_subquery_cache = caches.correlated;
    }

    /// Look up a *correlated* subquery's cached result by `(SubqueryId, correlation key)`.
    /// `None` is a cache MISS — this outer-value combination has not been computed within
    /// the statement (or the cache was cleared / is at capacity). The `key` is one
    /// [`CorrCell`] per [`SubPlan::correlated_cols`](minisqlite_plan::SubPlan) entry, in
    /// that sorted order (build it with [`corr_key`](crate::corr_key::corr_key)).
    pub(crate) fn cached_correlated_subquery(
        &self,
        id: usize,
        key: &[CorrCell],
    ) -> Option<&CachedSubquery> {
        // Alloc-free HIT probe (the hot per-outer-row path): the outer `get(&id)` is by a copy
        // `usize`, and the inner `get(key)` borrows the caller's `&[CorrCell]` directly
        // (`Vec<CorrCell>: Borrow<[CorrCell]>`) — so a hit clones NOTHING, where a flat
        // `(id, Vec<CorrCell>)` map would rebuild the whole tuple (deep-copying Text/Blob key
        // bytes) on every probe.
        self.correlated_subquery_cache.get(&id).and_then(|inner| inner.get(key))
    }

    /// Store a *correlated* subquery's result under `(SubqueryId, correlation key)`, so a
    /// later outer row with the same key reuses it. Bounded: once the cache holds
    /// [`CORRELATED_CACHE_CAP`] entries in TOTAL (across all ids), a NEW key is DROPPED (this
    /// is a no-op) rather than growing memory without limit — the evaluator simply re-runs the
    /// subquery on the next miss, exactly as it does today. An existing key is always updated
    /// (it does not grow the map), so a re-store of a live key is never dropped.
    pub(crate) fn cache_correlated_subquery(
        &mut self,
        id: usize,
        key: Vec<CorrCell>,
        entry: CachedSubquery,
    ) {
        // Cap on TOTAL entries across every id's inner map (the whole memo's memory). It is
        // summed here rather than tracked as a running counter so `take`/`restore` (which swap
        // whole maps) need no extra bookkeeping. This runs on a MISS only (~once per distinct
        // key) and #ids is tiny (the correlated subqueries in one statement), so the sum is
        // cheap; the hot HIT path never sums. An EXISTING key is always updated (no growth); a
        // NEW key is admitted only below the cap, else dropped so the caller re-runs next miss.
        let total: usize = self.correlated_subquery_cache.values().map(|inner| inner.len()).sum();
        match self.correlated_subquery_cache.get_mut(&id) {
            Some(inner) => {
                if total >= CORRELATED_CACHE_CAP && !inner.contains_key(&key) {
                    return;
                }
                inner.insert(key, entry);
            }
            None => {
                // A brand-new id always contributes a NEW key; admit only below the cap, and
                // avoid inserting an empty inner map when we would immediately drop the key.
                if total >= CORRELATED_CACHE_CAP {
                    return;
                }
                let mut inner = HashMap::new();
                inner.insert(key, entry);
                self.correlated_subquery_cache.insert(id, inner);
            }
        }
    }

    /// Whether the correlated-subquery memo is currently bypassed (see
    /// [`correlated_cache_disabled`](Runtime::correlated_cache_disabled)). Read by the
    /// `EvalCtx` subquery callbacks so a disabled memo falls back to the pre-memo re-run
    /// path (today's behavior) rather than consulting or populating the cache.
    pub(crate) fn correlated_cache_disabled(&self) -> bool {
        self.correlated_cache_disabled
    }

    /// TEST/DIAGNOSTIC hook: turn the correlated-subquery memo off (`true`) or on
    /// (`false`, the default). Tests drive cache-on vs cache-off through
    /// this to prove the memo path returns byte-identical results to the pre-memo re-run
    /// path; normal operation never calls it.
    pub fn set_correlated_cache_disabled(&mut self, disabled: bool) {
        self.correlated_cache_disabled = disabled;
    }

    /// TEST/DIAGNOSTIC hook: the number of distinct correlated-memo entries currently held
    /// (across all subquery ids). A correctness test reads it after a drain to prove the
    /// memo COLLAPSED many outer rows sharing a key into few subplan runs (a genuine hit),
    /// not merely that it returned the right value. It reflects the LAST statement's run,
    /// because the caches are cleared at the START of each statement's drain, not the end.
    pub fn correlated_cache_len(&self) -> usize {
        self.correlated_subquery_cache.values().map(|inner| inner.len()).sum()
    }

    /// Whether `PRAGMA recursive_triggers` is ON (the SQL `recursive_triggers` setting).
    /// SQLite defaults it OFF; the DML operators consult it to decide whether a nested
    /// trigger-action DML fires further triggers.
    pub fn recursive_triggers(&self) -> bool {
        self.recursive_triggers
    }

    /// Set the `recursive_triggers` flag (the engine's `PRAGMA recursive_triggers = …`
    /// handler). Connection-lifetime — it persists across statements until changed again.
    pub fn set_recursive_triggers(&mut self, on: bool) {
        self.recursive_triggers = on;
    }

    /// Whether `PRAGMA foreign_keys` is ON (the SQL `foreign_keys` setting). SQLite
    /// defaults it OFF; a later FK-enforcement pass consults it to decide whether to
    /// enforce foreign-key constraints. `PRAGMA foreign_keys` (get) reports this.
    pub fn foreign_keys(&self) -> bool {
        self.foreign_keys
    }

    /// Set the `foreign_keys` flag (the engine's `PRAGMA foreign_keys = …` handler).
    /// Connection-lifetime — it persists across statements until changed again.
    pub fn set_foreign_keys(&mut self, on: bool) {
        self.foreign_keys = on;
    }

    /// Whether a DEFERRED foreign-key check may be deferred right now — a transaction is
    /// open to defer INTO (see [`fk_defer_active`](Runtime::fk_defer_active)). Read by the
    /// child/parent FK-enforcement paths (via [`fk_deferred_now`]) to decide whether to skip
    /// the immediate check and let the COMMIT-time recheck catch a violation instead.
    ///
    /// [`fk_deferred_now`]: crate::ops::foreign_key::fk_deferred_now
    pub fn fk_defer_active(&self) -> bool {
        self.fk_defer_active
    }

    /// Set the per-statement "a transaction is open to defer into" flag. The engine calls
    /// this before every mutating statement with [`SqlEngine::txn_active`](../../minisqlite_engine),
    /// so autocommit (no active transaction) always leaves it false and deferred FKs behave
    /// like immediate ones (`foreignkeys.html` §4.2).
    pub fn set_fk_defer_active(&mut self, active: bool) {
        self.fk_defer_active = active;
    }

    /// Whether `PRAGMA defer_foreign_keys` is ON (the SQL `defer_foreign_keys` setting) — all
    /// FKs are treated as deferred for the current transaction. `PRAGMA defer_foreign_keys`
    /// (get) reports this; the COMMIT-time recheck consults it to scan EVERY FK, not only the
    /// declared-deferred ones.
    pub fn defer_foreign_keys(&self) -> bool {
        self.defer_foreign_keys
    }

    /// Set the LIVE `defer_foreign_keys` flag (the engine's `PRAGMA defer_foreign_keys = …`
    /// handler). Unlike [`set_foreign_keys`](Runtime::set_foreign_keys) this MAY be changed
    /// within a transaction — deferring all FKs for that transaction is its entire purpose
    /// (`pragma.html` #pragma_defer_foreign_keys). Turning it ON also ARMS the sticky
    /// [`defer_foreign_keys_armed`](Runtime::defer_foreign_keys_armed) coverage flag so the
    /// COMMIT recheck still rescans even if the pragma is later toggled OFF in the same txn;
    /// turning it OFF leaves the sticky flag set (coverage only grows within a txn). Use
    /// [`reset_defer_foreign_keys`](Runtime::reset_defer_foreign_keys) at transaction end
    /// to clear BOTH.
    pub fn set_defer_foreign_keys(&mut self, on: bool) {
        self.defer_foreign_keys = on;
        if on {
            self.defer_foreign_keys_armed = true;
        }
    }

    /// Whether the defer pragma has been ON at ANY point in the current transaction — the
    /// STICKY coverage flag the COMMIT recheck uses as its `defer_all` (see
    /// [`defer_foreign_keys_armed`](Runtime#structfield.defer_foreign_keys_armed)). Reading
    /// this instead of the live [`defer_foreign_keys`](Runtime::defer_foreign_keys) keeps a
    /// mid-txn toggle-OFF from silently shrinking the set of tables rescanned at COMMIT.
    pub fn defer_foreign_keys_armed(&self) -> bool {
        self.defer_foreign_keys_armed
    }

    /// Clear BOTH the live `defer_foreign_keys` flag and its sticky
    /// [`defer_foreign_keys_armed`](Runtime::defer_foreign_keys_armed) coverage flag. Called
    /// whenever a transaction ends — COMMIT, ROLLBACK, a RELEASE that commits, and the end of
    /// an autocommit statement's implicit transaction — since the pragma "is automatically
    /// switched off at each COMMIT or ROLLBACK" (`pragma.html` #pragma_defer_foreign_keys).
    /// The start of a transaction does NOT reset it, so a value set just before `BEGIN`
    /// applies to that transaction, as in sqlite3.
    pub fn reset_defer_foreign_keys(&mut self) {
        self.defer_foreign_keys = false;
        self.defer_foreign_keys_armed = false;
    }

    /// Look up the memoized re-compiled trigger set for `(db, table, event)`, returning a
    /// cheap `Arc` handle clone on a hit. `None` is a miss — the first recompile of that
    /// (db, table, event) within a statement, or any use after [`clear_recompiled_triggers`](Runtime::clear_recompiled_triggers).
    pub(crate) fn cached_recompiled_triggers(
        &self,
        key: &(DbIndex, String, TriggerEventKey),
    ) -> Option<Arc<Vec<TriggerProgram>>> {
        self.recompiled_triggers.get(key).cloned()
    }

    /// Store a re-compiled trigger set under `(db, table, event)` so later firings in the same
    /// statement reuse it instead of re-parsing/binding/planning the trigger bodies.
    pub(crate) fn cache_recompiled_triggers(
        &mut self,
        key: (DbIndex, String, TriggerEventKey),
        programs: Arc<Vec<TriggerProgram>>,
    ) {
        self.recompiled_triggers.insert(key, programs);
    }

    /// Drop every memoized re-compiled trigger set. Invoked once at the START of each
    /// statement's drain (by `StatementRoot`, alongside the subquery-cache clear) so a
    /// `CREATE`/`DROP TRIGGER` between statements is never masked by a stale entry.
    pub(crate) fn clear_recompiled_triggers(&mut self) {
        self.recompiled_triggers.clear();
    }

    /// Push `rows` as the current recursive-CTE frontier before running a step round.
    /// The matching [`pop_recursive_frame`](Runtime::pop_recursive_frame) must run
    /// after the round so the stack stays balanced (nested recursive CTEs rely on it).
    pub(crate) fn push_recursive_frame(&mut self, rows: Vec<Row>) {
        self.recursive_frames.push(rows);
    }

    /// Pop the current recursive-CTE frontier after a step round. Debug-asserts a
    /// frame was present, catching a push/pop imbalance during development; in release
    /// a stray pop is a harmless no-op rather than a panic.
    pub(crate) fn pop_recursive_frame(&mut self) {
        let popped = self.recursive_frames.pop();
        debug_assert!(
            popped.is_some(),
            "pop_recursive_frame with no active frame (push/pop imbalance)"
        );
    }

    /// A CLONE of the current recursive-CTE frontier (the top of the stack), for a
    /// [`RecursiveScan`] to stream. Cloning gives the scan a stable view even if it is
    /// rebuilt several times within one round (e.g. as a join's inner side). Reading it
    /// with no active frame is a malformed plan (a `RecursiveScan` outside a recursive
    /// step), so it errors rather than panicking.
    ///
    /// [`RecursiveScan`]: minisqlite_plan::PlanNode::RecursiveScan
    pub(crate) fn current_recursive_frame(&self) -> Result<Vec<Row>> {
        self.recursive_frames
            .last()
            .cloned()
            .ok_or_else(|| Error::sql("RecursiveScan outside a recursive CTE step"))
    }

    /// Enter one level of trigger firing, bounding the recursion AND snapshotting the
    /// change counters a trigger program must not leak. A DML operator calls this right
    /// before it runs one trigger program's actions and pairs it with
    /// [`exit_trigger`](Runtime::exit_trigger), passing back the returned
    /// [`TriggerSavepoint`]. Past [`MAX_TRIGGER_DEPTH`] it returns a clear error (the
    /// statement then aborts and the implicit transaction rolls back) rather than letting
    /// a self-referential trigger recurse until the stack is exhausted — the liveness bar
    /// (a run that never terminates is a wrong result).
    ///
    /// The returned savepoint captures `changes` and `last_insert_rowid` so `exit_trigger`
    /// can restore them: the trigger program's own writes advance `total_changes` (which
    /// must count them) but are invisible to the outer `changes()`/`last_insert_rowid()`,
    /// per lang_corefunc.html / c3ref/last_insert_rowid.html.
    pub(crate) fn enter_trigger(&mut self) -> Result<TriggerSavepoint> {
        if self.trigger_depth >= MAX_TRIGGER_DEPTH {
            return Err(Error::sql("too many levels of trigger recursion"));
        }
        self.trigger_depth += 1;
        Ok(TriggerSavepoint {
            changes: self.changes,
            last_insert_rowid: self.last_insert_rowid,
        })
    }

    /// Leave one level of trigger firing (pairs with [`enter_trigger`](Runtime::enter_trigger)),
    /// restoring the `changes`/`last_insert_rowid` captured on enter so the trigger
    /// program's own DML is excluded from the outer statement's `changes()` /
    /// `last_insert_rowid()` (while `total_changes` keeps whatever the actions added).
    /// Called even when an action errors, so the counters and depth are restored before
    /// the error propagates and the next top-level statement starts clean. A stray exit
    /// with no matching enter is a balance bug caught in debug and a harmless no-op in
    /// release.
    pub(crate) fn exit_trigger(&mut self, saved: TriggerSavepoint) {
        debug_assert!(self.trigger_depth > 0, "exit_trigger with no active trigger (enter/exit imbalance)");
        self.trigger_depth = self.trigger_depth.saturating_sub(1);
        self.changes = saved.changes;
        self.last_insert_rowid = saved.last_insert_rowid;
    }

    /// The current trigger-firing nesting depth (0 outside any firing). Used by the DML
    /// operators to decide whether a nested action's own triggers must be re-compiled and
    /// fired at run time (the compile pass expands only one level).
    pub(crate) fn trigger_depth(&self) -> usize {
        self.trigger_depth
    }

    /// Raise the `RAISE(IGNORE)` control flag (see [`raise_ignore`](Runtime::raise_ignore)).
    /// Called by the executor's `EvalContext::signal_raise_ignore` when a trigger body
    /// evaluates `RAISE(IGNORE)`; the evaluator then returns a sentinel `Err` that unwinds
    /// to the enclosing `fire_triggers`, which consumes the flag with
    /// [`take_raise_ignore`](Runtime::take_raise_ignore).
    pub(crate) fn set_raise_ignore(&mut self) {
        self.raise_ignore = true;
    }

    /// Read and CLEAR the `RAISE(IGNORE)` control flag. `fire_triggers` calls this on the
    /// error path of a fired trigger: `true` means the error was the `RAISE(IGNORE)`
    /// sentinel (skip the current row, no error), `false` a genuine error to propagate.
    /// Clearing here keeps the signal scoped to the one row that raised it.
    pub(crate) fn take_raise_ignore(&mut self) -> bool {
        std::mem::take(&mut self.raise_ignore)
    }

    /// Draw the next pseudo-random `i64` (for `random()`), advancing the PRNG.
    pub fn random_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    /// Fill `buf` with pseudo-random bytes (for `randomblob(n)`), advancing the PRNG.
    /// Draws 8 bytes at a time and copies the needed prefix, so a partial trailing
    /// chunk costs one draw, not one per byte.
    pub fn fill_random(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    /// Read bound parameter `index` (0-based into the params vector). An out-of-range
    /// index fails closed rather than defaulting to NULL, matching the evaluator's
    /// "an out-of-range parameter is an error" contract.
    pub(crate) fn param(&self, index: usize) -> Result<Value> {
        self.params
            .get(index)
            .cloned()
            .ok_or_else(|| Error::sql("bind parameter out of range"))
    }

    /// Replace the per-argument JSON subtypes for the function call about to run
    /// (json1.html §3.4). Reuses the existing buffer's capacity so the common
    /// per-row call re-set is allocation-free; an EMPTY `s` clears it, so every
    /// subsequent [`arg_subtype`](Runtime::arg_subtype) reads 0.
    pub(crate) fn set_arg_subtypes(&mut self, s: &[u8]) {
        self.arg_subtypes.clear();
        self.arg_subtypes.extend_from_slice(s);
    }

    /// The subtype of the current call's argument `i` (`0` = none, or out of range).
    pub(crate) fn arg_subtype(&self, i: usize) -> u8 {
        self.arg_subtypes.get(i).copied().unwrap_or(0)
    }

    /// Mark the subtype of the value the function in hand is about to return.
    pub(crate) fn set_result_subtype(&mut self, st: u8) {
        self.result_subtype = st;
    }

    /// Read and clear the result subtype the just-finished function set (`0` if none).
    pub(crate) fn take_result_subtype(&mut self) -> u8 {
        std::mem::take(&mut self.result_subtype)
    }

    /// SplitMix64: a fast, well-distributed 64-bit PRNG with a full 2^64 period. Each
    /// call advances `rng_state` by the golden-ratio odd constant, then applies the
    /// finalizing mix. Reproducible from the seed and free of the LCG lattice
    /// artifacts a naive `state * a + c` would show.
    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Runtime::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corr_key::CorrCell;

    #[test]
    fn prng_is_deterministic_from_seed() {
        let mut a = Runtime::new();
        let mut b = Runtime::new();
        let seq_a: Vec<i64> = (0..8).map(|_| a.random_i64()).collect();
        let seq_b: Vec<i64> = (0..8).map(|_| b.random_i64()).collect();
        assert_eq!(seq_a, seq_b, "same seed yields the same sequence");
    }

    #[test]
    fn prng_advances() {
        // Consecutive draws differ (a stuck PRNG would repeat). Not a randomness
        // proof, just a guard against a non-advancing state bug.
        let mut rt = Runtime::new();
        let first = rt.random_i64();
        let second = rt.random_i64();
        assert_ne!(first, second);
    }

    #[test]
    fn fill_random_fills_every_byte_including_partial_chunk() {
        // 13 bytes = one full 8-byte chunk + a 5-byte tail. Pin the EXACT bytes against
        // the PRNG sequence so the tail is contractually checked: a dropped tail (e.g.
        // switching `chunks_mut` to `chunks_exact_mut`) leaves bytes 8..13 zero and
        // fails here — the weaker "some byte is non-zero" check would not catch it,
        // since the first full chunk already supplies non-zero bytes.
        let mut rt = Runtime::new();
        let mut buf = [0u8; 13];
        rt.fill_random(&mut buf);

        // Independently derive the expected 13 bytes from two draws of a fresh runtime.
        // `random_i64().to_le_bytes()` is the same byte pattern `fill_random` copies
        // (`next_u64().to_le_bytes()`) — `as i64` only reinterprets the bits, it does
        // not change them.
        let mut reference = Runtime::new();
        let mut expected = Vec::with_capacity(16);
        expected.extend_from_slice(&reference.random_i64().to_le_bytes());
        expected.extend_from_slice(&reference.random_i64().to_le_bytes());
        assert_eq!(buf.as_slice(), &expected[..13], "every byte incl. the 5-byte tail");
    }

    #[test]
    fn change_counters_track_inserts_and_changes() {
        let mut rt = Runtime::new();
        assert_eq!(rt.changes(), 0);
        assert_eq!(rt.last_insert_rowid(), 0);
        assert_eq!(rt.total_changes(), 0);

        rt.record_insert(42);
        assert_eq!(rt.last_insert_rowid(), 42);
        assert_eq!(rt.changes(), 1);
        assert_eq!(rt.total_changes(), 1);

        rt.record_change();
        assert_eq!(rt.last_insert_rowid(), 42, "a non-insert change leaves last_insert_rowid");
        assert_eq!(rt.changes(), 2);
        assert_eq!(rt.total_changes(), 2);

        rt.reset_statement_changes();
        assert_eq!(rt.changes(), 0, "per-statement counter reset");
        assert_eq!(rt.total_changes(), 2, "lifetime counter persists");
        assert_eq!(rt.last_insert_rowid(), 42, "last_insert_rowid persists");
    }

    #[test]
    fn deferred_fk_flags_default_off_and_toggle_independently() {
        // Both deferral flags start OFF (autocommit, no defer pragma) so a fresh runtime
        // enforces every FK immediately — the invariant the autocommit path relies on.
        let mut rt = Runtime::new();
        assert!(!rt.fk_defer_active(), "no transaction open by default");
        assert!(!rt.defer_foreign_keys(), "defer_foreign_keys defaults OFF");

        // The per-statement transaction flag and the connection pragma flag are distinct
        // knobs (the deferral predicate ANDs them), so toggling one never moves the other.
        rt.set_fk_defer_active(true);
        assert!(rt.fk_defer_active());
        assert!(!rt.defer_foreign_keys(), "fk_defer_active must not touch defer_foreign_keys");

        rt.set_defer_foreign_keys(true);
        assert!(rt.defer_foreign_keys());
        assert!(rt.fk_defer_active(), "defer_foreign_keys must not touch fk_defer_active");

        // Both clear back to OFF (the COMMIT/ROLLBACK reset + the autocommit re-set path).
        rt.set_fk_defer_active(false);
        rt.set_defer_foreign_keys(false);
        assert!(!rt.fk_defer_active());
        assert!(!rt.defer_foreign_keys());
    }

    #[test]
    fn defer_foreign_keys_arming_is_sticky_until_reset() {
        // The sticky coverage flag closes the ON→OFF-mid-txn desync: a table deferred while
        // the pragma was ON must still be rescanned at COMMIT even after the pragma is toggled
        // OFF. So arming must survive a set(false) and clear ONLY on an explicit reset.
        let mut rt = Runtime::new();
        assert!(!rt.defer_foreign_keys_armed(), "sticky flag defaults OFF");

        rt.set_defer_foreign_keys(true);
        assert!(rt.defer_foreign_keys(), "live flag ON");
        assert!(rt.defer_foreign_keys_armed(), "turning the pragma ON arms the sticky flag");

        // Toggle the LIVE flag OFF mid-transaction: the live flag drops, the sticky stays —
        // this is exactly what keeps the COMMIT recheck scanning the deferred table.
        rt.set_defer_foreign_keys(false);
        assert!(!rt.defer_foreign_keys(), "live flag OFF after toggle-off");
        assert!(rt.defer_foreign_keys_armed(), "sticky coverage survives a mid-txn toggle-off");

        // Transaction end/start clears BOTH (the pragma is per-transaction).
        rt.reset_defer_foreign_keys();
        assert!(!rt.defer_foreign_keys(), "reset clears the live flag");
        assert!(!rt.defer_foreign_keys_armed(), "reset clears the sticky flag");
    }

    #[test]
    fn subquery_cache_insert_get_covers_every_variant() {
        let mut rt = Runtime::new();
        assert!(rt.cached_subquery(0).is_none(), "an empty cache is a miss");

        rt.cache_subquery(0, CachedSubquery::Scalar(Value::Integer(7)));
        rt.cache_subquery(1, CachedSubquery::Exists(true));
        rt.cache_subquery(
            2,
            CachedSubquery::InSet { set: HashSet::new(), set_has_null: true, any_rows: true },
        );
        rt.cache_subquery(
            3,
            CachedSubquery::InRows(vec![vec![Value::Integer(1), Value::Integer(2)]]),
        );
        rt.cache_subquery(4, CachedSubquery::FirstRow(vec![Value::Integer(5), Value::Integer(6)]));

        match rt.cached_subquery(0) {
            Some(CachedSubquery::Scalar(Value::Integer(7))) => {}
            other => panic!("expected Scalar(7), got {other:?}"),
        }
        assert!(matches!(rt.cached_subquery(1), Some(CachedSubquery::Exists(true))));
        match rt.cached_subquery(2) {
            Some(CachedSubquery::InSet { set, set_has_null, any_rows }) => {
                assert!(set.is_empty());
                assert!(*set_has_null);
                assert!(*any_rows);
            }
            other => panic!("expected InSet, got {other:?}"),
        }
        match rt.cached_subquery(3) {
            Some(CachedSubquery::InRows(rows)) => {
                assert_eq!(rows.len(), 1);
                assert!(matches!(rows[0].as_slice(), [Value::Integer(1), Value::Integer(2)]));
            }
            other => panic!("expected InRows, got {other:?}"),
        }
        match rt.cached_subquery(4) {
            Some(CachedSubquery::FirstRow(row)) => {
                assert!(matches!(row.as_slice(), [Value::Integer(5), Value::Integer(6)]));
            }
            other => panic!("expected FirstRow, got {other:?}"),
        }
        assert!(rt.cached_subquery(5).is_none(), "an id never inserted is still a miss");
    }

    #[test]
    fn clear_subquery_cache_empties_map_but_preserves_counters_and_rng() {
        // A reference runtime advances its RNG identically so we can prove the clear
        // leaves the RNG stream untouched (not reseeded, not perturbed).
        let mut rt = Runtime::new();
        let mut reference = Runtime::new();
        assert_eq!(rt.random_i64(), reference.random_i64(), "same seed, first draw agrees");

        // Seed connection-lifetime state that MUST survive a cache clear, plus BOTH caches.
        rt.record_insert(42);
        rt.record_change();
        rt.cache_subquery(0, CachedSubquery::Scalar(Value::Integer(1)));
        rt.cache_subquery(1, CachedSubquery::Exists(false));
        rt.cache_correlated_subquery(0, vec![CorrCell::Int(1)], CachedSubquery::Scalar(Value::Integer(9)));
        assert!(rt.cached_subquery(0).is_some());
        assert!(rt.cached_correlated_subquery(0, &[CorrCell::Int(1)]).is_some());

        rt.clear_subquery_cache();

        // Both caches are empty...
        assert!(rt.cached_subquery(0).is_none(), "clear empties the uncorrelated subquery cache");
        assert!(rt.cached_subquery(1).is_none());
        assert!(
            rt.cached_correlated_subquery(0, &[CorrCell::Int(1)]).is_none(),
            "clear empties the correlated subquery cache too"
        );
        // ...but the counters are exactly as set (record_insert/record_change do not
        // draw the RNG, so the reference's RNG is still in lock-step below).
        assert_eq!(rt.changes(), 2, "changes() survives a subquery-cache clear");
        assert_eq!(rt.total_changes(), 2, "total_changes() survives");
        assert_eq!(rt.last_insert_rowid(), 42, "last_insert_rowid() survives");
        // ...and the RNG stream is unaffected: rt's next draw still matches the
        // reference that never cleared anything.
        assert_eq!(rt.random_i64(), reference.random_i64(), "RNG stream unaffected by clear");
    }

    #[test]
    fn params_read_and_range_check() {
        let rt = Runtime::with_params(vec![Value::Integer(7), Value::Text("hi".into())]);
        assert!(matches!(rt.param(0).unwrap(), Value::Integer(7)));
        assert!(matches!(rt.param(1).unwrap(), Value::Text(s) if s == "hi"));
        assert!(rt.param(2).is_err(), "out-of-range param is an error, not NULL");
    }

    #[test]
    fn set_params_replaces() {
        let mut rt = Runtime::new();
        assert!(rt.param(0).is_err());
        rt.set_params(vec![Value::Null]);
        assert!(matches!(rt.param(0).unwrap(), Value::Null));
    }

    // `Value` has no `PartialEq`, so frames are compared by their (single-column)
    // integer contents.
    fn frame_ints(frame: &[Row]) -> Vec<i64> {
        frame
            .iter()
            .map(|r| match r.first() {
                Some(Value::Integer(i)) => *i,
                other => panic!("expected a single Integer column, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn recursive_frame_missing_is_an_error_not_a_panic() {
        // Reading the frontier with no active step is a malformed plan (a bare
        // RecursiveScan). It must surface an error, never panic.
        let rt = Runtime::new();
        assert!(rt.current_recursive_frame().is_err());
    }

    #[test]
    fn recursive_frame_push_returns_a_clone_and_pop_restores() {
        let mut rt = Runtime::new();
        let frame = vec![vec![Value::Integer(1)], vec![Value::Integer(2)]];
        rt.push_recursive_frame(frame.clone());

        let seen = rt.current_recursive_frame().expect("frame present");
        assert_eq!(frame_ints(&seen), vec![1, 2], "current frame is the pushed rows");
        // A second read yields an independent clone (mutating one must not touch the
        // stack), so a RecursiveScan rebuilt mid-round always sees the same frontier.
        let mut seen2 = rt.current_recursive_frame().unwrap();
        seen2.clear();
        assert_eq!(
            frame_ints(&rt.current_recursive_frame().unwrap()),
            vec![1, 2],
            "reads are independent clones"
        );

        rt.pop_recursive_frame();
        assert!(rt.current_recursive_frame().is_err(), "pop clears the frame");
    }

    #[test]
    fn trigger_depth_enters_exits_and_caps() {
        let mut rt = Runtime::new();
        assert_eq!(rt.trigger_depth(), 0, "no firing outside a trigger");

        // Enter up to the cap; each enter succeeds and bumps the depth by one. The
        // savepoints are held on a stack and paired with the exits below.
        let mut saved = Vec::new();
        for expected in 1..=MAX_TRIGGER_DEPTH {
            saved.push(rt.enter_trigger().expect("enter within the cap"));
            assert_eq!(rt.trigger_depth(), expected);
        }
        // At the cap, the next enter fails (a self-referential trigger cannot recurse
        // past the backstop) and leaves the depth unchanged.
        assert!(rt.enter_trigger().is_err(), "entering past the cap errors");
        assert_eq!(rt.trigger_depth(), MAX_TRIGGER_DEPTH, "a failed enter does not bump depth");

        // Exiting unwinds the stack symmetrically back to zero.
        while let Some(sp) = saved.pop() {
            rt.exit_trigger(sp);
        }
        assert_eq!(rt.trigger_depth(), 0, "every enter is unwound");
    }

    #[test]
    fn trigger_savepoint_hides_nested_changes_but_keeps_total() {
        // A trigger program's own DML must advance `total_changes` (it counts trigger
        // changes) but leave the outer `changes()`/`last_insert_rowid()` as they were
        // before the trigger fired (lang_corefunc.html / c3ref/last_insert_rowid.html).
        let mut rt = Runtime::new();
        rt.record_insert(1); // the "top-level" row: changes=1, total=1, lir=1
        assert_eq!((rt.changes(), rt.total_changes(), rt.last_insert_rowid()), (1, 1, 1));

        let sp = rt.enter_trigger().expect("enter");
        // The trigger action inserts elsewhere and updates a counter row.
        rt.record_insert(99); // changes=2, total=2, lir=99
        rt.record_change(); // changes=3, total=3
        assert_eq!((rt.changes(), rt.total_changes(), rt.last_insert_rowid()), (3, 3, 99));
        rt.exit_trigger(sp);

        // changes()/last_insert_rowid() revert to the pre-trigger values; total keeps all.
        assert_eq!(rt.changes(), 1, "changes() excludes the trigger's own writes");
        assert_eq!(rt.last_insert_rowid(), 1, "last_insert_rowid() reverts after the trigger");
        assert_eq!(rt.total_changes(), 3, "total_changes() counts the trigger's writes");
    }

    #[test]
    fn nested_savepoints_each_restore_their_own_baseline() {
        // Two nesting levels: the inner restore returns to the middle baseline, the
        // outer restore to the top-level baseline — robust under recursion.
        let mut rt = Runtime::new();
        rt.record_insert(1); // top level: changes=1, lir=1
        let outer = rt.enter_trigger().expect("enter outer");
        rt.record_insert(2); // changes=2, lir=2
        let inner = rt.enter_trigger().expect("enter inner");
        rt.record_insert(3); // changes=3, lir=3
        rt.exit_trigger(inner);
        assert_eq!((rt.changes(), rt.last_insert_rowid()), (2, 2), "inner restores middle baseline");
        rt.exit_trigger(outer);
        assert_eq!((rt.changes(), rt.last_insert_rowid()), (1, 1), "outer restores top baseline");
        assert_eq!(rt.total_changes(), 3, "total_changes accumulated every level");
    }

    #[test]
    fn subquery_cache_take_and_restore_round_trips() {
        // `take` lifts BOTH enclosing caches out (leaving them empty for a nested action's
        // own id-0-based subqueries) and `restore` puts them back, discarding the nested
        // entries. One token carries both maps, so neither can bleed into the action.
        let mut rt = Runtime::new();
        rt.cache_subquery(0, CachedSubquery::Scalar(Value::Integer(7)));
        rt.cache_correlated_subquery(0, vec![CorrCell::Int(5)], CachedSubquery::Exists(true));
        let saved = rt.take_subquery_cache();
        assert!(rt.cached_subquery(0).is_none(), "uncorrelated cache empty after take");
        assert!(
            rt.cached_correlated_subquery(0, &[CorrCell::Int(5)]).is_none(),
            "correlated cache empty after take"
        );
        // The nested run caches its own id 0 in BOTH caches (different subqueries).
        rt.cache_subquery(0, CachedSubquery::Exists(true));
        rt.cache_correlated_subquery(0, vec![CorrCell::Int(5)], CachedSubquery::Scalar(Value::Integer(1)));
        rt.restore_subquery_cache(saved);
        match rt.cached_subquery(0) {
            Some(CachedSubquery::Scalar(Value::Integer(7))) => {}
            other => panic!("restore should bring back the enclosing Scalar(7), got {other:?}"),
        }
        match rt.cached_correlated_subquery(0, &[CorrCell::Int(5)]) {
            Some(CachedSubquery::Exists(true)) => {}
            other => panic!("restore should bring back the enclosing correlated Exists(true), got {other:?}"),
        }
    }

    #[test]
    fn correlated_cache_hit_and_miss_by_id_and_key() {
        let mut rt = Runtime::new();
        // A miss on an empty cache.
        assert!(rt.cached_correlated_subquery(0, &[CorrCell::Int(1)]).is_none(), "empty is a miss");

        rt.cache_correlated_subquery(0, vec![CorrCell::Int(1)], CachedSubquery::Scalar(Value::Integer(10)));
        // A hit on the exact (id, key).
        match rt.cached_correlated_subquery(0, &[CorrCell::Int(1)]) {
            Some(CachedSubquery::Scalar(Value::Integer(10))) => {}
            other => panic!("expected Scalar(10), got {other:?}"),
        }
        // Same id, DIFFERENT key -> miss (the key is part of the identity).
        assert!(rt.cached_correlated_subquery(0, &[CorrCell::Int(2)]).is_none(), "different key is a miss");
        // Different id, same key -> miss (the id is part of the identity).
        assert!(rt.cached_correlated_subquery(1, &[CorrCell::Int(1)]).is_none(), "different id is a miss");
        // A multi-column key is matched element-wise, in order.
        rt.cache_correlated_subquery(
            0,
            vec![CorrCell::Int(1), CorrCell::Text(b"g".to_vec())],
            CachedSubquery::Exists(true),
        );
        assert!(
            rt.cached_correlated_subquery(0, &[CorrCell::Int(1), CorrCell::Text(b"g".to_vec())]).is_some(),
            "multi-column key hits"
        );
        assert!(
            rt.cached_correlated_subquery(0, &[CorrCell::Int(1), CorrCell::Text(b"h".to_vec())]).is_none(),
            "one differing key element is a miss"
        );
    }

    #[test]
    fn correlated_cache_cap_stops_new_keys_but_serves_and_updates_existing() {
        let mut rt = Runtime::new();
        // Fill exactly to capacity with distinct keys (id 0, key [Int(i)]).
        for i in 0..CORRELATED_CACHE_CAP as i64 {
            rt.cache_correlated_subquery(0, vec![CorrCell::Int(i)], CachedSubquery::Exists(true));
        }
        assert_eq!(rt.correlated_cache_len(), CORRELATED_CACHE_CAP, "filled to the cap");

        // A NEW key past the cap is DROPPED (no growth, no panic).
        rt.cache_correlated_subquery(
            0,
            vec![CorrCell::Int(CORRELATED_CACHE_CAP as i64)],
            CachedSubquery::Exists(false),
        );
        assert_eq!(rt.correlated_cache_len(), CORRELATED_CACHE_CAP, "a new key past the cap is dropped");
        assert!(
            rt.cached_correlated_subquery(0, &[CorrCell::Int(CORRELATED_CACHE_CAP as i64)]).is_none(),
            "the dropped new key is a miss (caller re-runs the subquery)"
        );

        // An EXISTING key is still served...
        assert!(rt.cached_correlated_subquery(0, &[CorrCell::Int(0)]).is_some(), "existing key still served at cap");
        // ...and can be UPDATED at the cap (updating a present key does not grow the map).
        rt.cache_correlated_subquery(0, vec![CorrCell::Int(0)], CachedSubquery::Scalar(Value::Integer(99)));
        assert_eq!(rt.correlated_cache_len(), CORRELATED_CACHE_CAP, "updating a live key does not grow past the cap");
        match rt.cached_correlated_subquery(0, &[CorrCell::Int(0)]) {
            Some(CachedSubquery::Scalar(Value::Integer(99))) => {}
            other => panic!("an existing key is updated in place at the cap, got {other:?}"),
        }
    }

    #[test]
    fn correlated_cache_key_is_storage_class_exact() {
        // The correlation key must NOT fold storage classes (unlike CellKey): Integer(2),
        // Real(2.0), and Text("2") are three DISTINCT cache keys, so a correlated subquery
        // whose result depends on the storage class (e.g. `typeof`) is never mis-served.
        let mut rt = Runtime::new();
        rt.cache_correlated_subquery(0, vec![CorrCell::Int(2)], CachedSubquery::Scalar(Value::Text("integer".into())));
        rt.cache_correlated_subquery(0, vec![CorrCell::Real((2.0f64).to_bits())], CachedSubquery::Scalar(Value::Text("real".into())));
        rt.cache_correlated_subquery(0, vec![CorrCell::Text(b"2".to_vec())], CachedSubquery::Scalar(Value::Text("text".into())));
        assert_eq!(rt.correlated_cache_len(), 3, "int/real/text keys do not collide");
        match rt.cached_correlated_subquery(0, &[CorrCell::Int(2)]) {
            Some(CachedSubquery::Scalar(Value::Text(s))) => assert_eq!(s, "integer"),
            other => panic!("Int(2) key -> integer, got {other:?}"),
        }
        match rt.cached_correlated_subquery(0, &[CorrCell::Real((2.0f64).to_bits())]) {
            Some(CachedSubquery::Scalar(Value::Text(s))) => assert_eq!(s, "real"),
            other => panic!("Real(2.0) key -> real, got {other:?}"),
        }
    }

    #[test]
    fn recompiled_trigger_cache_stores_keys_by_event_and_clears() {
        let mut rt = Runtime::new();
        let insert_key = (DbIndex::MAIN, "t".to_string(), TriggerEventKey::Insert);
        assert!(rt.cached_recompiled_triggers(&insert_key).is_none(), "empty cache is a miss");

        rt.cache_recompiled_triggers(insert_key.clone(), Arc::new(Vec::new()));
        assert!(rt.cached_recompiled_triggers(&insert_key).is_some(), "hit after cache");
        // A different event on the same table is a distinct key (miss).
        assert!(
            rt.cached_recompiled_triggers(&(DbIndex::MAIN, "t".to_string(), TriggerEventKey::Delete))
                .is_none(),
            "event is part of the key"
        );
        // The SAME table+event in a DIFFERENT namespace is a distinct key (miss): a temp/attached
        // table shadowing `main.t` must not read main's recompiled set (the shadowing bug fixed
        // by threading `db` into `recompile_target_triggers`).
        assert!(
            rt.cached_recompiled_triggers(&(DbIndex::TEMP, "t".to_string(), TriggerEventKey::Insert))
                .is_none(),
            "db namespace is part of the key"
        );
        // An UPDATE's changed-column set is part of the key.
        rt.cache_recompiled_triggers(
            (DbIndex::MAIN, "t".to_string(), TriggerEventKey::Update(vec![0])),
            Arc::new(Vec::new()),
        );
        assert!(rt
            .cached_recompiled_triggers(&(
                DbIndex::MAIN,
                "t".to_string(),
                TriggerEventKey::Update(vec![0])
            ))
            .is_some());
        assert!(
            rt.cached_recompiled_triggers(&(
                DbIndex::MAIN,
                "t".to_string(),
                TriggerEventKey::Update(vec![1])
            ))
            .is_none(),
            "a different assigned-column set is a distinct key"
        );

        rt.clear_recompiled_triggers();
        assert!(rt.cached_recompiled_triggers(&insert_key).is_none(), "cleared per statement");
    }

    #[test]
    fn recursive_frames_nest_as_a_stack() {
        // An inner recursive CTE pushes its own frame atop an outer one; the inner is
        // always on top, and popping it restores the outer frontier.
        let mut rt = Runtime::new();
        rt.push_recursive_frame(vec![vec![Value::Integer(10)]]);
        rt.push_recursive_frame(vec![vec![Value::Integer(20)]]);
        assert_eq!(
            frame_ints(&rt.current_recursive_frame().unwrap()),
            vec![20],
            "innermost frame on top"
        );
        rt.pop_recursive_frame();
        assert_eq!(
            frame_ints(&rt.current_recursive_frame().unwrap()),
            vec![10],
            "outer frame restored"
        );
        rt.pop_recursive_frame();
        assert!(rt.current_recursive_frame().is_err());
    }
}
