//! The `UPDATE` DML operator: eager two-phase rewrite, then stream `RETURNING`.
//!
//! Structurally identical to [`insert`](super::insert): an `UPDATE` must apply its
//! whole scan before it can report anything, so it does all its work on the FIRST
//! `next_row` pull — where the [`Runtime`] is in hand for the change counter — and
//! then streams any buffered `RETURNING` rows one per later pull.
//!
//! Two phases that cannot overlap. The scan borrows the pager `&dyn Pager` (a read
//! cursor) while the write path needs the exclusive `&mut dyn Pager`; so phase one
//! drains the ENTIRE scan under a shared reborrow and buffers it, and only once that
//! borrow is released does phase two take `&mut` and rewrite. Buffering is also
//! load-bearing for correctness, not just borrows: it is SQLite's pre-update
//! snapshot (every row's assignments see the table as it was before any write), AND
//! it is what lets the scan be an `IndexScan` on an index this statement rewrites —
//! we never mutate a b-tree while a cursor is walking it. The buffer is bounded by
//! the updated set; nothing larger is materialized.
//!
//! One `OR REPLACE`-only wrinkle to the snapshot: an earlier row's victim deletion (or
//! its rowid move onto a slot a later buffered row still points at) can leave a LATER
//! buffered row stale — the physical row at its `old_rowid` was deleted or replaced. So
//! the write phase re-reads such a row LIVE before rewriting it — exactly SQLite's second
//! UPDATE pass, which re-reads each row and skips one already deleted to make room. This
//! is gated on `OR REPLACE` and on the row's slot actually having been touched (tracked
//! in an `invalidated` set), so a non-REPLACE UPDATE and every untouched row pay nothing;
//! Abort/Ignore can never create the interference (a colliding move errors or skips).
//! CAVEAT: this closes only the staleness the REPLACE CONFLICT mechanism creates. It does
//! NOT cover a DIFFERENT pre-existing source — an FK ON UPDATE referential action (or a
//! nested trigger) that rewrites ANOTHER row in this same buffer mid-statement; that row's
//! later iteration still runs from its pre-image, a cross-cutting UPDATE/DELETE staleness
//! best fixed once for the whole operator, not just for REPLACE.
//!
//! Row shapes. The scan's LEADING columns are the target row `[c0..c_{N-1}, rowid]`
//! (width `N+1`, per the shared ROW/REGISTER convention); a plain `UPDATE` scan is
//! exactly that, while an `UPDATE ... FROM` join row is WIDER — the FROM tables'
//! columns trail after the target's rowid and are read only by the assignment/`WHERE`
//! exprs, never by the target write (which reads only the leading `[0..N]` + rowid).
//! Each assignment value expr is evaluated against that whole OLD row; `RETURNING` is
//! evaluated against the UPDATED target row (the `[cols.., rowid]` shape, width `N+1`,
//! rebuilt from the new values — never the wide join row). Affinity is applied to
//! ASSIGNED values only — an unassigned column keeps its already-stored
//! (already-affinity-applied) value.

use std::collections::HashSet;

use minisqlite_btree::{
    index_delete_with_order, index_insert_with_order, table_delete, table_insert, TableCursor,
};
use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_expr::eval;
use minisqlite_fileformat::{encode_record_enc, TextEncoding};
use minisqlite_pager::{text_encoding_of, PageId, Pager};
use minisqlite_plan::{GeneratedProgram, OnConflict, Plan, TriggerDmlEvent, Update};
use minisqlite_types::{apply_affinity, ConstraintKind, DbIndex, Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::{PagerSet, Env, Pagers};
use crate::ops::constraints::{enforce_checks_over_new_row, enforce_not_null, ConstraintOutcome};
use crate::ops::foreign_key;
use crate::ops::dml_index::{
    build_index_plans, delete_index_entries, index_keys_for_plans, insert_index_entries,
    unique_conflict, unique_conflict_rowid, wr_delete_victim_by_pk, wr_index_keys_for_plans,
    wr_unique_conflict, IndexPlan,
};
use crate::ops::generated::{compute_generated, stored_record};
use crate::ops::must_be_int::must_be_int;
use crate::ops::returning::eval_returning;
use crate::ops::trigger;
use crate::ops::without_rowid::{wr_layout, WrLayout};
use crate::row::{decode_table_row_enc, decode_table_row_skipping_virtual_enc, resolve_base_table};
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the `UPDATE` cursor. It holds the write-side pager set ([`PagerSet`]) the
/// rewrite needs and defers all work to the first pull (see the module doc), mirroring
/// `insert`.
pub(crate) fn update<'e>(
    node: &'e Update,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(UpdateCursor {
        node,
        catalog,
        pagers,
        plan,
        outer,
        built: false,
        returning: Vec::new(),
        ret_idx: 0,
    }))
}

struct UpdateCursor<'e> {
    node: &'e Update,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    /// The correlated frame this UPDATE runs under: empty at top level, or a firing
    /// trigger's `OLD ++ NEW` frame when this UPDATE is a trigger action. The scan is
    /// built with it, so each scan row is `frame ++ [c0..c_{N-1}, rowid]`; the table row
    /// (and its rowid / index key) is recovered from the trailing `n + 1` values.
    outer: &'e [Value],
    /// Whether the eager rewrite has run (it runs exactly once, on the first pull).
    built: bool,
    /// Buffered `RETURNING` rows, streamed one per `next_row` after the rewrite. Empty
    /// when the statement has no `RETURNING` clause (the cursor then yields no rows).
    returning: Vec<Row>,
    /// Index of the next buffered `RETURNING` row to emit.
    ret_idx: usize,
}

impl RowCursor for UpdateCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if !self.built {
            self.build(rt)?;
            self.built = true;
        }
        if self.ret_idx < self.returning.len() {
            // `mem::take` hands out the owned row without cloning; the slot is left an
            // empty Vec we never read again.
            let row = std::mem::take(&mut self.returning[self.ret_idx]);
            self.ret_idx += 1;
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }
}

impl UpdateCursor<'_> {
    /// Apply the whole `UPDATE` (both phases) and buffer any `RETURNING` rows.
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        // Copy out the borrow-independent context so the rest of the method only
        // touches `self.pagers` (a shared source view, then the exclusive target pager,
        // never overlapping) and `self.returning`. `catalog`/`plan`/`node` are `Copy`.
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        let n = node.column_count;

        // (1) Resolve the target table. UPDATE is WR-aware, so it resolves through
        // `resolve_base_table` (which accepts a WITHOUT ROWID table) and branches below;
        // a missing table still fails closed here. `node.db` is the target namespace
        // (`main` unless a temp/qualified target). `def` borrows through the `Copy`
        // `&dyn Catalog` (lifetime `'e`), so it does NOT pin a borrow on `self`, leaving
        // `self.pagers` free to be reborrowed below.
        let def: &TableDef = resolve_base_table(catalog, node.db, &node.table)?;

        // Validate the plan's shape against the table before indexing by it, so a
        // malformed plan fails closed here rather than panicking deep in the row loop.
        // Both hold for a WITHOUT ROWID table too (its scan emits width-N rows), so this
        // runs before the WR branch below.
        if n != def.columns.len() {
            return Err(Error::sql(format!(
                "UPDATE column_count {n} does not match table {} column count {}",
                node.table,
                def.columns.len()
            )));
        }
        if node.column_affinities.len() != n {
            return Err(Error::sql(format!(
                "UPDATE column_affinities length {} does not match column_count {n}",
                node.column_affinities.len()
            )));
        }

        // A WITHOUT ROWID table stores its rows in the PRIMARY KEY index b-tree (no rowid
        // to move or seed), so it takes a separate write path keyed by PRIMARY KEY. Branch
        // AFTER the shape validation (which applies to both kinds) and BEFORE any
        // rowid-specific machinery below, mirroring `insert::build`.
        if def.without_rowid {
            return self.apply_without_rowid(rt, def, n);
        }

        let root = def.root_page;
        let alias = def.rowid_alias;
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // The target's generated-column programs (STORED + VIRTUAL, in column order), bound
        // once (loop-invariant). Empty for a plain table — the row loop then does no generated
        // work and takes the byte-identical record path. The OLD side needs no recompute here:
        // the scan that produced each buffered `old` row already computed its VIRTUAL columns
        // (scan.rs / indexscan.rs), so `old[..n]` carries the values an index on a generated
        // column was keyed by. Only the NEW row is recomputed (6.1) and, at write time, its
        // VIRTUAL columns are omitted from the record (`stored_record`, step 8).
        let gprograms = plan.generated_programs(node.db, &node.table);

        // The triggers to fire for THIS write (see `trigger::effective_triggers`). A
        // top-level UPDATE borrows `node.triggers`; a trigger-ACTION UPDATE recompiles the
        // target's own triggers under `PRAGMA recursive_triggers` (default OFF). `UPDATE OF
        // <cols>` filtering is honored by passing the assigned column indices as the event's
        // `changed_cols`, exactly as the top-level compile does. `changed` is only consulted
        // by the recompile path; building it (O(assignments)) once per build is cheap.
        let changed: Vec<usize> = node.assignments.iter().map(|(ci, _)| *ci).collect();
        let effective = trigger::effective_triggers(
            &node.triggers,
            rt,
            catalog,
            node.db,
            def,
            TriggerDmlEvent::Update { changed_cols: &changed },
        )?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow of the whole store set — an `UPDATE ... FROM`
        // scan may read another namespace than the target). Drain the entire scan into
        // `buffered`. The scope bounds the shared source view so it is released before the
        // write phase, and the buffer is the pre-update snapshot (see the module doc).
        let mut buffered: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut src = build_cursor(&node.scan, env, self.outer)?;
            while let Some(r) = src.next_row(rt)? {
                buffered.push(r);
            }
        }

        // The database's text encoding (fileformat2 §1.3.13), read once for the whole
        // write phase: UTF-8 yields byte-identical records; a UTF-16 database stores every
        // TEXT value in that encoding so the rewritten rows stay readable by real sqlite.
        // Read under a scoped shared view; the per-row write phase re-borrows the target
        // store mutably below — re-borrowed per "piece" around each trigger fire, because a
        // fire takes the WHOLE store set to route each action to its own namespace.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);

        // (3)-(7) WRITE PHASE (exclusive borrow, one row at a time). `base` is the width
        // of the correlated frame the scan prepends (0 at top level): the table row is the
        // trailing `[c0..c_{N-1}, rowid]`, so it lives at `old[base..]`. Assignment/WHERE
        // exprs bind ABSOLUTE register indices into the WHOLE `old` row (their table columns
        // sit at `base + col`, their `NEW`/`OLD` refs in the frame below `base`), so eval
        // still runs against the full `old`; only the write reads through `base`.
        let base = self.outer.len();
        let has_returning = !node.returning.is_empty();
        // Only `OR REPLACE` can make an earlier row's write invalidate a LATER buffered row's
        // slot: a victim deletion (which includes a rowid move onto an OCCUPIED slot — the
        // occupant is deleted as a victim first). Every other policy errors or skips such a
        // collision, so the stale-snapshot bookkeeping below is REPLACE-only — a non-REPLACE
        // UPDATE never touches it.
        let is_replace = matches!(node.on_conflict, OnConflict::Replace);
        // Rowids this statement has deleted as REPLACE victims — the COMPLETE set of slots a
        // later DISTINCT buffered row can find changed (see the victim-loop note for why a
        // rowid move needs no separate entry). A later buffered row whose `old_rowid` is in
        // here was invalidated by an earlier iteration and must be re-read LIVE (the refresh
        // just below). Empty — and never inserted into — unless REPLACE actually deletes a
        // victim, so a non-REPLACE UPDATE and a conflict-free REPLACE stay zero-cost.
        let mut invalidated: HashSet<i64> = HashSet::new();
        for buffered_row in &buffered {
            // (3) The scan row starts with `frame ++ [c0..c_{N-1}, rowid]` — the target's
            // own columns and rowid at the LEADING `base + n + 1` registers. A plain UPDATE
            // row is exactly that width; an `UPDATE ... FROM` join row is WIDER (the FROM
            // tables' columns trail after the target's rowid and are read only by the
            // assignment/WHERE exprs, never by the target write). A row NARROWER than the
            // target itself is a malformed plan — the rowid register would be misread — so
            // fail closed rather than pluck a bogus rowid.
            if buffered_row.len() < base + n + 1 {
                return Err(Error::sql(format!(
                    "UPDATE scan row width {} is smaller than table {} column count {n} (+rowid, +frame {base})",
                    buffered_row.len(),
                    node.table
                )));
            }
            let old_rowid = match &buffered_row[base + n] {
                Value::Integer(r) => *r,
                other => {
                    return Err(Error::sql(format!(
                        "UPDATE scan row has a non-integer rowid register: {other:?}"
                    )))
                }
            };
            // (3.5) PASS-2 FRESHNESS (REPLACE only). If an earlier row in THIS statement
            // deleted the row at `old_rowid` as a victim (possibly then reoccupying the slot by
            // moving another row onto it), `buffered_row`'s snapshot is stale. Re-read the LIVE
            // row — exactly what real sqlite's second UPDATE pass does — so the rewrite acts on
            // the row that is actually there, never a phantom: a VANISHED row (deleted to make
            // room) is skipped, and a REOCCUPIED slot is reprocessed from its CURRENT contents.
            // This is the invariant the feature exists to hold: an index entry must never point
            // at a rowid whose row we clobbered. The target portion is spliced back into the
            // buffered frame so an `UPDATE ... FROM` join row keeps its FROM columns; the
            // refreshed row then flows through the rest of the body as `old` unchanged.
            let refreshed: Option<Row> = if is_replace && invalidated.contains(&old_rowid) {
                match read_live_row(
                    self.pagers.source().get(node.db)?,
                    root,
                    def,
                    gprograms,
                    catalog,
                    plan,
                    node.db,
                    old_rowid,
                    rt,
                    enc,
                )? {
                    None => continue,
                    Some(target) => {
                        // `read_live_row` yields the width-(N+1) `[c0..c_{N-1}, rowid]` target
                        // shape; the splice below overwrites exactly that window, so a width
                        // mismatch is a decode-helper regression, not a data condition — assert
                        // it (a bad `clone_from_slice` would otherwise panic opaquely).
                        debug_assert_eq!(
                            target.len(),
                            n + 1,
                            "read_live_row must return the [c0..c_{{N-1}}, rowid] shape (width N+1)"
                        );
                        let mut o = buffered_row.clone();
                        o[base..base + n + 1].clone_from_slice(&target);
                        Some(o)
                    }
                }
            } else {
                None
            };
            let old: &Row = refreshed.as_ref().unwrap_or(buffered_row);
            // (4) Compute the new column values into `new_vals`, which — once its alias
            // slot is normalized in (6) — doubles AS the `logical` row (the index-key +
            // RETURNING shape), so no separate per-row `old_logical`/`stored`/`logical`
            // copies are allocated. This is the hot path (one iteration per updated row),
            // so the row is materialized once, not four times. `old[..n]` is the
            // pre-update `N` columns (alias slot already = old_rowid) AND the OLD index
            // key, so it is borrowed directly for the delete in (8) rather than copied.
            // CRITICAL: every value expr is evaluated against `old` (the FULL pre-update
            // row), never the partially-updated `new_vals` — so `SET a = a + 1, b = a`
            // uses the pre-update `a` for BOTH, matching SQLite. Affinity is applied to
            // the ASSIGNED value as it is stored; unassigned columns keep their existing
            // (already-affinity-applied) values, copied through untouched.
            // Width-N new logical row. One extra slot is reserved because both
            // `enforce_checks_over_new_row` (6.6) and RETURNING (9) append the rowid as the
            // trailing register N — the `[c0..c_{N-1}, rowid]` layout their predicates bind
            // against — so neither append reallocates per row.
            let mut new_vals = Vec::with_capacity(n + 1);
            new_vals.extend_from_slice(&old[base..base + n]);
            // Evaluate the SET assignments under the shared WHOLE-SLICE source view
            // (`self.pagers.source()`), the SAME view INSERT evaluates its value exprs under
            // (ops/insert.rs): a `SET col = (SELECT … FROM aux.t)` is a scalar subquery that
            // may read ANY namespace (an ATTACHed db, `temp`), not only the target's
            // (lang_update.html: the SET value is a general `expr`; lang_attach.html /
            // lang_naming.html: cross-db qualified + unqualified resolution). A
            // single-namespace `Pagers::One { db: node.db }` view fails closed on any read
            // naming another namespace — the bug this eval-view fixes. (The later write-phase
            // eval steps — generated recompute, CHECK — DO forbid subqueries per spec and
            // stay on `Pagers::One`.) `source()` borrows `self.pagers` SHARED, so this scope
            // MUST fully close (env + ctx dropped, shared borrow released) BEFORE the write
            // phase reborrows `self.pagers` mutably via `target(node.db)` just below — the two
            // borrows cannot coexist; `new_vals` is filled with OWNED `Value`s here so nothing
            // outlives the scope. Same-namespace subqueries are unaffected: `Pagers::Set`
            // resolves `set[node.db]`, the exact store the old `Pagers::One` reborrowed, so
            // read-your-writes within the statement is preserved.
            {
                let env = Env { catalog, pagers: self.pagers.source(), plan };
                let mut ctx = EvalCtx { rt, env, outer: old };
                for (ci, expr) in &node.assignments {
                    if *ci >= n {
                        return Err(Error::sql(format!(
                            "UPDATE assignment targets column {ci} but table {} has {n} columns",
                            node.table
                        )));
                    }
                    let v = eval(expr, old, &mut ctx)?;
                    new_vals[*ci] = apply_affinity(v, node.column_affinities[*ci]);
                }
            }
            // Piece-1 target pager: the post-assignment steps that need the exclusive target
            // store (the generated-column recompute in 6.1) take it here, AFTER the shared
            // assignment-eval scope above has closed. Re-borrowed again after each trigger
            // fire below, so it never overlaps the `&mut self.pagers` a fire takes for
            // per-action routing.
            let pager = self.pagers.target(node.db)?;

            // (5) Determine the new rowid (lang_createtable.html §5). Assigning the
            // INTEGER PRIMARY KEY alias to an integer — or to a string/real that
            // losslessly converts to one, via the shared `OP_MustBeInt` coercion — moves
            // the rowid. Unlike INSERT, an UPDATE cannot null the rowid, so NULL is an
            // error too: `require_int` collapses both `MustBeInt::Null` and
            // `MustBeInt::NotInt` (a blob, a fractional or out-of-range real, or a
            // non-numeric string) to "datatype mismatch", aborting the statement (the bug
            // this replaced silently kept the old rowid, masking the error). A genuine
            // no-op (no assignment touched the alias) needs NO special case: the scan
            // fills the alias slot with `Integer(old_rowid)` (see `decode_table_row_enc`) and
            // step (4) copied it into `new_vals[ai]`, so it coerces straight back to
            // old_rowid and the row is not relocated.
            let new_rowid = match alias {
                Some(ai) => must_be_int(&new_vals[ai]).require_int()?,
                None => old_rowid,
            };

            // (6) Normalize `new_vals` into the `logical` row: the alias slot must hold
            // `Integer(new_rowid)` for the index keys and RETURNING. Step (5) already
            // guarantees it does — a non-integer alias assignment errored there, and the
            // integer / no-op cases leave `Integer(new_rowid)` in the slot — so this is a
            // defensive re-assert of that invariant, not a fixup. After it `new_vals` IS
            // the logical row; the STORED record differs from it in ONLY the alias slot
            // (NULL there — the rowid lives in the b-tree key, not a stored column), which
            // (8) applies in place around `encode_record`, so no second full-row copy is
            // built.
            if let Some(ai) = alias {
                new_vals[ai] = Value::Integer(new_rowid);
            }

            // (6.1) RECOMPUTE GENERATED columns (gencol.html) from the post-assignment row,
            // in `CREATE TABLE` column order, so a SET that touched a base column re-derives
            // its generated dependents (a generated column itself is never assigned — the
            // planner rejects that). Each program's expr binds over `[c0..c_{N-1}, new_rowid]`
            // (register N is the rowid — an INTEGER PRIMARY KEY reference resolves there);
            // `compute_generated` applies each column's affinity and fills its slot IN PLACE,
            // so a later generated column reads an earlier one already recomputed. This runs
            // BEFORE the trigger frame / NOT NULL / CHECK / uniqueness / rewrite, so the value
            // flows into NEW.*, every constraint, the indexes, and RETURNING. A VIRTUAL value
            // is used everywhere but OMITTED from the stored record (`stored_record`, step 8).
            // Empty `gprograms` (a plain table) does nothing.
            if !gprograms.is_empty() {
                new_vals.push(Value::Integer(new_rowid));
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                compute_generated(gprograms, false, &mut new_vals, env, rt)?;
                new_vals.truncate(n);
            }

            // Build the `OLD ++ NEW` trigger frame ONCE per row when this UPDATE carries
            // triggers: OLD = the pre-update `[cols, old_rowid]` (the alias slot already
            // holds `old_rowid`), NEW = the post-assignment `[cols, new_rowid]`. Reused for
            // the BEFORE fire (below, pre-rewrite) and the AFTER fire (post-rewrite). `None`
            // when there are no triggers, so a trigger-free UPDATE runs exactly as before. A
            // BEFORE trigger sees NEW as the proposed row before NOT NULL / CHECK / rewrite
            // (lang_createtrigger.html §2); we do NOT re-read a NEW the trigger body mutates
            // (an advanced BEFORE-UPDATE feature no in-scope test needs).
            let frame = if triggers.is_empty() {
                None
            } else {
                let mut old_half = Vec::with_capacity(n + 1);
                old_half.extend_from_slice(&old[base..base + n]);
                old_half.push(Value::Integer(old_rowid));
                let mut new_half = Vec::with_capacity(n + 1);
                new_half.extend_from_slice(&new_vals);
                new_half.push(Value::Integer(new_rowid));
                Some(trigger::build_frame(n, Some(old_half), Some(new_half)))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row's update: skip the
                // NOT NULL / CHECK checks, the rewrite, the AFTER triggers, and RETURNING,
                // leaving the existing row untouched and uncounted (like `OR IGNORE`).
                if let trigger::TriggerFlow::IgnoreRow = trigger::fire_triggers(
                    triggers,
                    trigger::Phase::Before,
                    frame,
                    catalog,
                    &mut self.pagers,
                    rt,
                    plan,
                )? {
                    continue;
                }
            }

            // Piece-2 target pager: re-borrowed after the BEFORE fire for CHECK / FK / index /
            // rewrite. The re-borrow addresses the same store, so a same-namespace BEFORE
            // action's writes are visible here (read-your-writes across the fire gap).
            let pager = self.pagers.target(node.db)?;

            // (6.5) Enforce column NOT NULL over the new logical row, BEFORE the
            // conflict check and the rewrite, so an ABORT leaves the row untouched and
            // an IGNORE-skipped row `continue`s without deleting/rewriting it. The alias
            // slot always holds `Integer(new_rowid)` here (steps 5-6), so the helper's
            // rowid-alias exemption never trips on it.
            match enforce_not_null(def, &new_vals, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (6.6) Enforce table/column CHECK constraints over the POST-assignment new row,
            // AFTER NOT NULL and BEFORE the conflict probe and the rewrite. The row-layout
            // choreography — the `[c0..c_{N-1}, rowid]` register-N append for an INTEGER
            // PRIMARY KEY alias, the truncate-back-to-width-N-even-on-error invariant, and the
            // scoped `EvalCtx` — lives in `enforce_checks_over_new_row` so it stays identical
            // to INSERT (5.6). The outcome resolves under `on_conflict` exactly like (6.5): OR
            // IGNORE SKIPs just the offending row (`continue`, no error — the row is left
            // untouched, not deleted/rewritten), while ABORT/FAIL/ROLLBACK/REPLACE raise
            // MID-LOOP on the first violating row and the engine's implicit-txn rollback backs
            // out every row this UPDATE already rewrote — so a partially-satisfiable ABORT
            // UPDATE applies to NONE of its rows.
            match enforce_checks_over_new_row(
                &node.checks,
                &mut new_vals,
                n,
                new_rowid,
                &node.on_conflict,
                Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan },
                rt,
            )? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (6.7) FOREIGN KEY (child side): the updated row's key columns must still
            // reference an existing parent (or be NULL) when `PRAGMA foreign_keys` is ON.
            // Same immediate check as INSERT; a violation aborts.
            foreign_key::enforce_child_foreign_keys(def, &new_vals, catalog, node.db, &*pager, rt)?;

            // (6.7b) FOREIGN KEY (parent side, ON UPDATE): if THIS row is a referenced parent
            // whose key an incoming FK points at, and this UPDATE changes that key, fire the
            // FK's ON UPDATE action on the matching children — RESTRICT/NO ACTION abort,
            // CASCADE rewrites their FK columns to the new key, SET NULL nulls them, SET
            // DEFAULT resets them to their column defaults (then re-checks the FK). A no-op
            // when the pragma is OFF, when no FK references this table, or when no referenced
            // key column changed. Runs BEFORE this row's own rewrite so children are found by
            // the OLD key still in storage; takes `&mut` because CASCADE/SET NULL/SET DEFAULT
            // write.
            foreign_key::enforce_parent_update(
                def,
                &old[base..base + n],
                &new_vals,
                catalog,
                node.db,
                &mut *pager,
                rt,
            )?;

            // (6.8) EVAL PHASE of index maintenance: compute the OLD keys (to delete this
            // row's pre-update entries) and the NEW keys (to probe UNIQUE and to write) while
            // the pager is still borrowed SHARED — an expression key needs the `EvalCtx`, which
            // the exclusive-borrow rewrite below cannot hold. Skipped when the table has NO
            // index (no dead per-row work). OLD frame is the scanned `[c0..c_{N-1}, rowid]`
            // slice at `old[base..base + n + 1]` (rowid = old_rowid, borrowed — no copy); the
            // NEW frame is formed IN PLACE by appending the rowid to `new_vals`'s reserved
            // trailing slot and truncating back (no second full-row copy). Col-only (today)
            // reads the same columns + rowid the pre-expression path did.
            let (old_keys, new_keys) = if index_plans.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let old_frame = &old[base..base + n + 1];
                new_vals.push(Value::Integer(new_rowid));
                let keys = {
                    let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                    let mut ctx = EvalCtx { rt, env, outer: &new_vals };
                    let old_keys = index_keys_for_plans(&index_plans, old_frame, old_rowid, &mut ctx)?;
                    let new_keys = index_keys_for_plans(&index_plans, &new_vals, new_rowid, &mut ctx)?;
                    (old_keys, new_keys)
                };
                new_vals.truncate(n);
                keys
            };

            // (7) Enforce uniqueness against the PRE-UPDATE state, BEFORE deleting this
            // row's own entries, under `on_conflict`. The NEW keys carry the row's proposed
            // indexed values; the probe excludes this row's own entries via `old_rowid`.
            match detect_conflict(
                &*pager,
                &node.on_conflict,
                def,
                root,
                old_rowid,
                new_rowid,
                &new_keys,
                &index_plans,
                enc,
            )? {
                Action::Proceed => {}
                Action::Skip => continue, // OnConflict::Ignore: leave the row untouched.
                Action::Replace(victims) => {
                    // OnConflict::Replace: sqlite deletes every pre-existing row causing the
                    // constraint violation BEFORE applying this row's update (lang_conflict.html).
                    // Each victim's index entries are keyed by ITS stored values, so read the
                    // row back to recompute them before deleting. The read (a shared borrow) is
                    // scoped closed before each delete takes the exclusive borrow — the operator
                    // never holds both at once — and all of it runs under the SAME piece-2 `pager`
                    // borrow as the rewrite in (8) below, so a mid-op error rolls back a
                    // consistent index+table. `victims` never contains `old_rowid` (see
                    // `detect_conflict`), so this never deletes the very row about to be
                    // rewritten. The implicit deletes are NOT counted (no `record_change`):
                    // sqlite's REPLACE does not bump the change counter for rows it removes to
                    // make room (an UPDATE counts only the rows it updates), matching insert.rs.
                    for victim in victims {
                        // Read the victim back (VIRTUAL generated columns filled) so its index
                        // entries are keyed by ITS stored values. Every victim rowid was observed
                        // LIVE by `detect_conflict` moments ago in this single-threaded statement,
                        // and the set is deduped, so a missing table row here is corruption or a
                        // logic bug — fail loud (the statement rolls back) rather than strand an
                        // entry or rewrite over a phantom. Mirrors insert.rs and the
                        // scanned-row-vanished guard in (8).
                        let victim_row = read_live_row(
                            &*pager, root, def, gprograms, catalog, plan, node.db, victim, rt, enc,
                        )?
                        .ok_or_else(|| {
                            Error::format(format!(
                                "UPDATE OR REPLACE: conflicting row at rowid {victim} not \
                                 found when deleting it to make room"
                            ))
                        })?;
                        // EVAL the victim's index keys from its width-(N+1) `[c0..c_{N-1}, rowid]`
                        // row under the SHARED borrow (an expression key needs the ctx), before
                        // the exclusive-borrow deletes below.
                        let victim_keys = {
                            let env =
                                Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                            let mut ctx = EvalCtx { rt, env, outer: &victim_row };
                            index_keys_for_plans(&index_plans, &victim_row, victim, &mut ctx)?
                        };
                        // FOREIGN KEY (parent side, ON DELETE): REPLACE removes this victim to
                        // make room, so — exactly like a standalone DELETE (ops/delete.rs) —
                        // fire every incoming FK's ON DELETE action on it FIRST: RESTRICT/NO
                        // ACTION abort, CASCADE deletes referencing children, SET NULL/SET
                        // DEFAULT rewrite them. Located by the victim's OLD key values
                        // (`victim_row[..n]`, its width-N logical row), still in storage below.
                        // A no-op when foreign_keys is OFF (default).
                        foreign_key::enforce_parent_delete(def, &victim_row[..n], catalog, node.db, &mut *pager, rt)?;
                        // Index entries BEFORE the table row (mirrors insert.rs), so a mid-op
                        // failure can't leave an index entry pointing at a deleted table row.
                        delete_index_entries(&mut *pager, &index_plans, &victim_keys, enc)?;
                        table_delete(&mut *pager, root, victim)?;
                        // This slot is now empty: a LATER buffered row that points at it must
                        // re-read live (and find it gone) instead of trusting its stale snapshot.
                        // Victim deletes are the COMPLETE set of invalidations a later distinct
                        // buffered row can observe: a rowid move's landing slot, when occupied,
                        // is deleted HERE as a victim first (so it is recorded), and when empty
                        // holds no row any buffered row references; a move's vacated SOURCE slot
                        // is the mover's own already-processed rowid, unique across the buffer.
                        // So tracking victims alone is sufficient — no separate move-target entry.
                        invalidated.insert(victim);
                    }
                }
            }

            // (8) Rewrite the row and move its index entries. Delete the OLD index
            // entries first (keyed by the pre-update row `old[base..base + n + 1]` — its
            // columns + old rowid — where `base` is nonzero for a trigger-action UPDATE).
            // Then, only if the rowid moved, delete the old table row — a same-rowid update
            // is an in-place REPLACE via `table_insert`, so deleting there would needlessly
            // churn the b-tree.
            delete_index_entries(&mut *pager, &index_plans, &old_keys, enc)?;
            if new_rowid != old_rowid {
                let existed = table_delete(&mut *pager, root, old_rowid)?;
                if !existed {
                    // The row was just scanned into `buffered`, so it must be present;
                    // its absence means storage corruption or a logic bug — fail loud.
                    return Err(Error::format(format!(
                        "UPDATE could not find scanned row at rowid {old_rowid} to rewrite"
                    )));
                }
            }
            // The STORED record is `new_vals` with the alias slot blanked to NULL (the
            // rowid is the b-tree key, not a stored column); blank it in place, encode,
            // then restore `Integer(new_rowid)` so `new_vals` is the logical row again for
            // RETURNING (the index keys were already computed at step 6.8 above and are
            // written from the precomputed `new_keys`, not rebuilt from `new_vals` here). A
            // generated-column table instead builds the record via `stored_record`, which
            // blanks the alias slot AND omits every VIRTUAL column (computed on read, never
            // stored — gencol.html).
            let record = if gprograms.is_empty() {
                if let Some(ai) = alias {
                    new_vals[ai] = Value::Null;
                    let rec = encode_record_enc(&new_vals, enc);
                    new_vals[ai] = Value::Integer(new_rowid);
                    rec
                } else {
                    encode_record_enc(&new_vals, enc)
                }
            } else {
                encode_record_enc(&stored_record(&new_vals, def), enc)
            };
            table_insert(&mut *pager, root, new_rowid, &record)?;
            insert_index_entries(&mut *pager, &index_plans, &new_keys, enc)?;
            rt.record_change();

            // Fire AFTER triggers now the row is rewritten (post-write, per row). Reuses the
            // frame built above; a skipped row (NOT NULL / CHECK / IGNORE) `continue`d before
            // reaching here, so AFTER never fires for a row left untouched.
            if let Some(frame) = &frame {
                // An AFTER trigger's `RAISE(IGNORE)` cannot un-write the row (already
                // rewritten and counted, IGNORE rolls nothing back); it only abandons this
                // row's remaining work, so skip RETURNING and move to the next row.
                if let trigger::TriggerFlow::IgnoreRow = trigger::fire_triggers(
                    triggers,
                    trigger::Phase::After,
                    frame,
                    catalog,
                    &mut self.pagers,
                    rt,
                    plan,
                )? {
                    continue;
                }
            }

            // (9) RETURNING: evaluate each expr over the UPDATED row `[cols.., rowid]`
            // and buffer it. `new_vals` (the logical row) is moved into `regs` to avoid a
            // clone; it already carries the new values with the alias slot as the rowid.
            if has_returning {
                let mut regs = new_vals;
                regs.push(Value::Integer(new_rowid));
                // Evaluate under the whole-slice shared SOURCE view (`eval_returning`), NOT a
                // single-namespace `Pagers::One`, so a RETURNING subquery may read ANY
                // namespace (an ATTACHed db / `temp`) — the same cross-namespace fix applied
                // to the SET assignments above (lang_returning.html §3 allows subqueries).
                // RETURNING is read-only and the last per-row step — the AFTER fire's `&mut
                // self.pagers` borrow (and piece-2 `pager`) already closed — so NO target
                // pager is reborrowed; `source()` borrows `self.pagers` shared and is released
                // when `eval_returning` returns the owned row, before `self.returning.push`.
                let row = eval_returning(&node.returning, &regs, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(row);
            }
        }
        Ok(())
    }

    /// Apply the whole `UPDATE` (both phases) to a WITHOUT ROWID table — keyed by PRIMARY
    /// KEY instead of an integer rowid — and buffer any `RETURNING` rows. Mirrors the rowid
    /// [`build`](Self::build) path's two phases (drain the scan under a shared borrow into the
    /// pre-update snapshot, then rewrite each row under the exclusive borrow) and the INSERT
    /// WR write path's per-row constraint choreography, but over the width-`N` WR row (no
    /// rowid register).
    ///
    /// A WR row IS the PRIMARY KEY index b-tree's key record (withoutrowid.html; fileformat2
    /// §2.4), so a rewrite is `index_delete(old_key)` + `index_insert(new_record)`: correct
    /// whether or not the PRIMARY KEY changed (a same-PK rewrite reinserts at the same b-tree
    /// slot; a PK change moves it). PRIMARY KEY + secondary-index uniqueness is enforced
    /// against the live tree under `on_conflict`; `last_insert_rowid()` is untouched (WR has no
    /// rowid), and each updated row is counted via `record_change`.
    ///
    /// Secondary indexes are maintained per row through the shared [`super::dml_index`] WR
    /// path (`[indexed cols.., trailing PK..]`, fileformat2 §2.5.1): the OLD entry (old columns
    /// + old PK) is deleted and the NEW entry (new columns + new PK) inserted. This covers BOTH
    /// an indexed-column change AND a PRIMARY KEY change — a PK change alters the trailing-PK
    /// part of EVERY entry, so both keys are recomputed unconditionally. A REPLACE victim's PK
    /// record AND all its secondary entries are removed together (`wr_delete_victim_by_pk`).
    ///
    /// SCOPE (honest fail-closed): an EXPRESSION index on a WR table is refused (its key exprs
    /// bind against a frame with no WR rowid) — that loud error is raised by `build_index_plans`
    /// here — and a WR table that carries its OWN triggers is refused (the `OLD ++ NEW` frame is
    /// rowid-shaped). Both keep the store consistent rather than maintaining a wrong structure.
    fn apply_without_rowid(&mut self, rt: &mut Runtime, def: &TableDef, n: usize) -> Result<()> {
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        // The WR b-tree root (the PRIMARY KEY index), stable across this statement's writes
        // (see the INSERT WR path). Every probe/delete/insert below targets it.
        let root = def.root_page;
        let layout = wr_layout(def)?;
        // The target's generated-column programs, loop-invariant. A WR generation expr binds
        // over `[c0..c_{N-1}]` (no rowid register); STORED values are written, VIRTUAL values
        // are omitted from the record by the virtual-aware `WrLayout` and recomputed on read.
        let gprograms = plan.generated_programs(node.db, &node.table);

        // The WR secondary-index write plans (fileformat2 §2.5.1 key shape). Building them here
        // ALSO raises the loud fail-closed error for an EXPRESSION index on a WR table (see
        // `build_wr_index_key`). Empty for a WR table with no secondary index — the per-row
        // maintenance below then does nothing.
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // The triggers to fire for THIS write (mirrors the rowid path). A WR trigger frame is
        // laid at width `W = N` (no rowid register — see `trigger::build_frame_wr`), so a body's
        // `NEW.col_k` / `OLD.col_k` binds to `Column(N+k)` / `Column(k)`. The EFFECTIVE set (not
        // bare `node.triggers`) honors `recursive_triggers` for a nested trigger-action UPDATE
        // into a WR table; `changed_cols` mirrors the top-level compile so `UPDATE OF <cols>`
        // trigger filtering is honored. An empty set fires nothing (no frame built below).
        let changed: Vec<usize> = node.assignments.iter().map(|(ci, _)| *ci).collect();
        let effective = trigger::effective_triggers(
            &node.triggers,
            rt,
            catalog,
            node.db,
            def,
            TriggerDmlEvent::Update { changed_cols: &changed },
        )?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow): drain the ENTIRE scan into `buffered` — the
        // pre-update snapshot — before any write. Load-bearing for a WR table: the scan reads
        // the very PRIMARY KEY b-tree the write phase mutates, so mutating mid-scan would
        // corrupt the live cursor; the snapshot also gives correct semantics when an UPDATE
        // changes a PK that a later scanned row's OLD PK would have matched (the REPLACE
        // freshness re-read below). The scope releases the shared view before the write phase.
        let mut buffered: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut src = build_cursor(&node.scan, env, self.outer)?;
            while let Some(r) = src.next_row(rt)? {
                buffered.push(r);
            }
        }

        // The database's text encoding, read once for the whole write phase (see the rowid
        // path). A WR row is stored in the PK index b-tree, so its TEXT columns are in this
        // encoding too.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);

        // (3)-(9) WRITE PHASE (exclusive borrow, one row at a time). `base` is the width of
        // any correlated frame the scan prepends (0 at top level, >0 for a trigger-action
        // UPDATE): the WR target row is the width-`N` slice `old[base..base + n]` (no trailing
        // rowid register, unlike the rowid path). Assignment/WHERE exprs bind ABSOLUTE register
        // indices into the WHOLE row, so eval runs against the full `old`.
        let base = self.outer.len();
        let has_returning = !node.returning.is_empty();
        // Only `OR REPLACE` can make an earlier row's victim deletion invalidate a LATER
        // buffered row's slot (every other policy errors or skips such a collision), so this
        // stale-snapshot bookkeeping is REPLACE-only. It holds the PRIMARY KEYs deleted as
        // REPLACE victims — the COMPLETE set of slots a later buffered row can find changed (a
        // rewrite lands on an occupied slot only after deleting its occupant as a victim here;
        // an empty slot holds no row any buffered snapshot references). PKs are compared under
        // `Binary`, the collation the WR b-tree is keyed on, so this agrees with its order.
        let mut invalidated: Vec<Vec<Value>> = Vec::new();
        // Reused across the (rare) REPLACE live re-reads, so a refresh allocates only the
        // returned row's spine (mirrors the scan cursor's shared scratch).
        let mut scratch: Vec<Value> = Vec::new();
        for buffered_row in &buffered {
            // (3) The scan row is `frame ++ [c0..c_{N-1}]` — the target's own columns at the
            // LEADING `base + n` registers (an `UPDATE ... FROM` join row may trail wider FROM
            // columns, read only by the assignment/WHERE exprs). A row narrower than that is a
            // malformed plan — fail closed rather than slice out of bounds.
            if buffered_row.len() < base + n {
                return Err(Error::sql(format!(
                    "UPDATE scan row width {} is smaller than WITHOUT ROWID table {} column \
                     count {n} (+frame {base})",
                    buffered_row.len(),
                    node.table
                )));
            }
            // The OLD PRIMARY KEY, from the snapshot's target columns. Computed ONCE here and
            // reused below for BOTH the REPLACE freshness check and the old-key delete probe: a
            // freshness re-read re-reads the LIVE row at THIS SAME PK, so the refreshed row's PK
            // columns equal these — there is no distinct post-refresh PK to recompute.
            let old_pk = layout.pk_values(&buffered_row[base..base + n]);

            // (3.5) REPLACE FRESHNESS. If an earlier row in THIS statement deleted the row at
            // this OLD PK as a victim, `buffered_row`'s snapshot is stale. Re-read the LIVE row
            // (read-your-writes through the shared source view, which reflects the target
            // writes) — exactly SQLite's second UPDATE pass: a VANISHED row (deleted to make
            // room) is skipped, a REOCCUPIED slot is reprocessed from its CURRENT contents (so
            // a REPLACE cascade like `UPDATE OR REPLACE t SET pk=pk+1` collapses correctly).
            // The target window is spliced back so an `UPDATE ... FROM` row keeps its FROM cols.
            let refreshed: Option<Row> = if matches!(node.on_conflict, OnConflict::Replace)
                && pk_in(&layout, &invalidated, &old_pk)
            {
                match read_live_wr_row(
                    self.pagers.source().get(node.db)?,
                    root,
                    def,
                    &layout,
                    gprograms,
                    catalog,
                    plan,
                    node.db,
                    &old_pk,
                    &mut scratch,
                    rt,
                    enc,
                )? {
                    None => continue,
                    Some(live) => {
                        debug_assert_eq!(
                            live.len(),
                            n,
                            "read_live_wr_row must return the width-N schema row"
                        );
                        let mut o = buffered_row.clone();
                        o[base..base + n].clone_from_slice(&live);
                        Some(o)
                    }
                }
            } else {
                None
            };
            let old: &Row = refreshed.as_ref().unwrap_or(buffered_row);

            // (4) Compute the NEW target columns: carry the OLD values, then overwrite each
            // assigned column with its (affinity-applied) new value. Every value expr is
            // evaluated against the FULL pre-update `old` row (so `SET a=a+1, b=a` uses the
            // pre-update `a` for both), under the whole-slice shared SOURCE view (a SET
            // subquery may read any namespace). The scope closes — releasing the shared borrow
            // — before the exclusive target pager is taken below.
            let mut new_vals: Vec<Value> = Vec::with_capacity(n + 1);
            new_vals.extend_from_slice(&old[base..base + n]);
            {
                let env = Env { catalog, pagers: self.pagers.source(), plan };
                let mut ctx = EvalCtx { rt, env, outer: old };
                for (ci, expr) in &node.assignments {
                    if *ci >= n {
                        return Err(Error::sql(format!(
                            "UPDATE assignment targets column {ci} but table {} has {n} columns",
                            node.table
                        )));
                    }
                    let v = eval(expr, old, &mut ctx)?;
                    new_vals[*ci] = apply_affinity(v, node.column_affinities[*ci]);
                }
            }

            // (6.1) Recompute GENERATED columns over the width-N post-assignment row (a WR
            // generation expr binds over `[c0..c_{N-1}]`, no rowid register). STORED and
            // VIRTUAL are both recomputed; the virtual-aware `WrLayout` omits VIRTUAL from the
            // stored record (step 8) and the scan recomputes it on read. The eval pager is
            // SCOPED and released before the BEFORE fire (which takes the whole pager set).
            if !gprograms.is_empty() {
                let pager = self.pagers.target(node.db)?;
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                compute_generated(gprograms, false, &mut new_vals, env, rt)?;
            }

            // Build the `OLD ++ NEW` WR frame ONCE per row when this UPDATE carries triggers:
            // OLD = the pre-update width-N `old[base..base+n]`, NEW = the post-assignment
            // width-N `new_vals` (NO rowid register — `build_frame_wr` lays each half at width
            // N). Reused for the BEFORE fire (pre-rewrite) and the AFTER fire (post-rewrite);
            // `None` when trigger-free, so that path stays byte-identical. A BEFORE trigger sees
            // NEW before NOT NULL / CHECK / rewrite (lang_createtrigger.html §2); a NEW the body
            // mutates is not re-read (an advanced feature no in-scope test needs).
            let frame = if triggers.is_empty() {
                None
            } else {
                Some(trigger::build_frame_wr(
                    n,
                    Some(old[base..base + n].to_vec()),
                    Some(new_vals[..n].to_vec()),
                ))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row's update: skip the
                // NOT NULL / CHECK checks, the rewrite, the AFTER fire, and RETURNING, leaving
                // the existing row untouched and uncounted (like `OR IGNORE`).
                if let trigger::TriggerFlow::IgnoreRow = trigger::fire_triggers(
                    triggers,
                    trigger::Phase::Before,
                    frame,
                    catalog,
                    &mut self.pagers,
                    rt,
                    plan,
                )? {
                    continue;
                }
            }

            // The exclusive target pager for this row's write pieces (the constraint probes, the
            // PK probe/delete/insert, the secondary-index maintenance). Re-borrowed AFTER the
            // BEFORE fire (which needs the whole set); released (NLL) before the AFTER fire +
            // RETURNING below.
            let pager = self.pagers.target(node.db)?;

            // (6.5) Enforce explicit column NOT NULL, then the PRIMARY KEY's implicit NOT NULL
            // (withoutrowid.html), both under `on_conflict` (IGNORE skips this row untouched).
            match enforce_not_null(def, &new_vals, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }
            match layout.enforce_pk_not_null(def, &new_vals, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (6.6) Enforce CHECK constraints over the post-assignment row. A WR row has no
            // rowid register; `enforce_checks_over_new_row` appends one trailing placeholder
            // for the rowid-alias CHECK layout it shares with the rowid path and truncates it
            // back — the `0` placeholder is inert here (a CHECK referencing rowid is a bind
            // error upstream on a WR table). IGNORE skips this row; the rest raise mid-loop.
            match enforce_checks_over_new_row(
                &node.checks,
                &mut new_vals,
                n,
                0,
                &node.on_conflict,
                Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan },
                rt,
            )? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (6.7) FOREIGN KEY (child side): the updated row's key columns must still
            // reference an existing parent (or be NULL) when `PRAGMA foreign_keys` is ON.
            foreign_key::enforce_child_foreign_keys(def, &new_vals, catalog, node.db, &*pager, rt)?;

            // (6.7b) FOREIGN KEY (parent side, ON UPDATE): if THIS WR row is a parent whose
            // referenced key an incoming FK points at, and this UPDATE changes that key, fire
            // the FK's ON UPDATE action on the matching children — exactly as the rowid UPDATE
            // path (`enforce_parent_update` in `build`) and the WR DELETE path
            // (`enforce_parent_delete`) do for their sides: RESTRICT/NO ACTION reject,
            // CASCADE/SET NULL/SET DEFAULT rewrite the children. The parent side reads ONLY the
            // referenced key columns of the width-N logical row (no rowid), so a WR *parent* is
            // fully supported; a WR *child* fails CLOSED inside the helper (it cannot be scanned
            // to apply an IMMEDIATE action) rather than orphaning a reference. A no-op when
            // `foreign_keys` is OFF, when nothing references this table, or when no referenced
            // key column changed. Placed here — after child FK, before the conflict resolution
            // and rewrite below — to match the rowid path's ordering, so children are still
            // located by the OLD key that is still in storage (deleted just below).
            foreign_key::enforce_parent_update(
                def,
                &old[base..base + n],
                &new_vals,
                catalog,
                node.db,
                &mut *pager,
                rt,
            )?;

            // (2') OLD key: the exact stored key-record bytes to delete, probed on the LIVE
            // tree through the exclusive pager (read-your-writes). `old_pk` was computed once
            // above from the snapshot's target columns; the freshness refresh re-reads the row
            // at that same PK, so it still names this row's stored key.
            let old_key = layout.pk_conflict_key(&*pager, root, &old_pk, enc)?.ok_or_else(|| {
                // Reached only if the OLD row vanished without being recorded as a REPLACE
                // victim (the freshness re-read above already `continue`s on a recorded one) —
                // i.e. storage corruption or a logic bug, not a data condition. Fail loud.
                Error::format(format!(
                    "UPDATE could not find scanned WITHOUT ROWID row to rewrite in table {} \
                     (PRIMARY KEY {old_pk:?})",
                    node.table
                ))
            })?;

            // (6.8) EVAL secondary-index keys: the OLD entries to remove (old columns + old PK)
            // and the NEW entries to write (new columns + new PK), both fileformat2 §2.5.1
            // shape. A WR index key is all ordinary columns (an expression WR index is
            // fail-closed at plan build); the scoped eval ctx is needed only to gate a PARTIAL
            // index on its WHERE predicate — over the OLD frame for `old_keys` (a row outside
            // the predicate then has no stale entry to drop) and the NEW frame for `new_keys`
            // (a row moving in gains one). The OLD row is `old[base..base+n]` (the refreshed
            // values under REPLACE, so the keys match the entries actually stored); the NEW row
            // is `new_vals`; both are width-N with generated columns filled. NOTE: a PRIMARY KEY
            // change alters the trailing-PK part of EVERY entry, so OLD/NEW differ even for an
            // index whose indexed columns did not change — recomputing both unconditionally
            // handles that. The ctx's shared pager+rt borrow ends before the writes below.
            let (old_keys, new_keys) = if index_plans.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                let mut ctx = EvalCtx { rt, env, outer: &new_vals };
                let old_keys = wr_index_keys_for_plans(&index_plans, &old[base..base + n], &mut ctx)?;
                let new_keys = wr_index_keys_for_plans(&index_plans, &new_vals, &mut ctx)?;
                (old_keys, new_keys)
            };

            // (3') Conflict resolution across the PRIMARY KEY and every UNIQUE secondary index,
            // on the PRE-UPDATE state (this row's own entries are still present). IGNORE leaves
            // the row untouched on the first conflict; ABORT/FAIL/ROLLBACK raise on the first;
            // REPLACE gathers every conflicting existing row (by PRIMARY KEY) to delete first.
            let new_pk = layout.pk_values(&new_vals);
            let mut victims: Vec<Vec<Value>> = Vec::new();
            let mut skip = false;
            // PRIMARY KEY: only a real PK change can collide with a DIFFERENT row (a same-PK
            // rewrite's only match is this row itself).
            if !layout.pk_equal(&old_pk, &new_pk)
                && layout.pk_conflict_key(&*pager, root, &new_pk, enc)?.is_some()
            {
                match node.on_conflict {
                    OnConflict::Ignore => skip = true,
                    OnConflict::Replace => victims.push(new_pk.clone()),
                    OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                        return Err(wr_pk_conflict_error(def, &layout));
                    }
                }
            }
            // UNIQUE secondary indexes: probe the NEW key, excluding THIS row's own still-present
            // entry via `old_pk` (its entries are keyed by the old PK until the rewrite below).
            if !skip {
                for (ip, key) in index_plans.iter().zip(&new_keys) {
                    if !ip.unique {
                        continue;
                    }
                    // A partial index not admitting the rewritten row has a `None` new key —
                    // no entry, no conflict — so skip its probe.
                    let Some(key) = key else { continue };
                    match node.on_conflict {
                        OnConflict::Replace => {
                            if let Some(vpk) =
                                wr_unique_conflict(&*pager, ip, key, Some(&old_pk), enc)?
                            {
                                victims.push(vpk);
                            }
                        }
                        OnConflict::Ignore => {
                            if wr_unique_conflict(&*pager, ip, key, Some(&old_pk), enc)?.is_some() {
                                skip = true;
                                break;
                            }
                        }
                        OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                            if wr_unique_conflict(&*pager, ip, key, Some(&old_pk), enc)?.is_some() {
                                return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
                            }
                        }
                    }
                }
            }
            if skip {
                continue; // OnConflict::Ignore hit a conflict: leave this row untouched.
            }

            // (3'.1) REPLACE victims: delete each conflicting existing row — its PK record AND
            // all its secondary-index entries — before this row's rewrite. Dedup by encoded PK.
            // A victim is a DIFFERENT row than this one (the PK probe fires only on a real PK
            // change; the unique probe excludes `old_pk`), so this never deletes the row being
            // rewritten. Record each victim PK as invalidated so a later buffered row pointing
            // at it re-reads live. Victims are NOT counted (REPLACE does not bump the change
            // counter for rows it removes to make room — lang_conflict.html).
            if !victims.is_empty() {
                let mut seen: Vec<Vec<u8>> = Vec::new();
                for vpk in &victims {
                    let enc_pk = encode_record_enc(vpk, enc);
                    if seen.contains(&enc_pk) {
                        continue;
                    }
                    seen.push(enc_pk);
                    wr_delete_victim_by_pk(
                        &mut *pager, &layout, def, root, &index_plans, gprograms, catalog, plan,
                        node.db, vpk, enc, rt,
                    )?;
                    invalidated.push(vpk.clone());
                }
            }

            // (8) Rewrite: delete this row's OLD secondary entries and OLD PK record, then
            // insert the NEW PK record and NEW secondary entries. Deleting the old key first
            // keeps a same-PK rewrite from holding a stale duplicate; moving the secondary
            // entries (old → new) reflects both an indexed-column change and a PK change.
            delete_index_entries(&mut *pager, &index_plans, &old_keys, enc)?;
            index_delete_with_order(&mut *pager, root, &old_key, layout.order())?;
            let record = encode_record_enc(&layout.storage_values(&new_vals), enc);
            index_insert_with_order(&mut *pager, root, &record, layout.order())?;
            insert_index_entries(&mut *pager, &index_plans, &new_keys, enc)?;
            rt.record_change();

            // Fire AFTER triggers now the row is rewritten (post-write). Reuses the frame; a row
            // skipped by NOT NULL / CHECK / IGNORE `continue`d earlier, so AFTER never fires for
            // a row that was not updated. An AFTER `RAISE(IGNORE)` cannot un-write the row; it
            // only abandons this row's RETURNING. The per-row `pager` was last used by
            // `insert_index_entries`, so its exclusive borrow has ended (NLL).
            if let Some(frame) = &frame {
                if let trigger::TriggerFlow::IgnoreRow = trigger::fire_triggers(
                    triggers,
                    trigger::Phase::After,
                    frame,
                    catalog,
                    &mut self.pagers,
                    rt,
                    plan,
                )? {
                    continue;
                }
            }

            // (9) RETURNING: evaluate over the NEW width-N row (UPDATE RETURNING returns the
            // post-update values — lang_returning.html). A WR base row has no trailing rowid
            // register, so `new_vals` is exactly the shape RETURNING binds against. Evaluated
            // under the shared source view (a RETURNING subquery may read any namespace); the
            // exclusive `pager` borrow has ended (NLL) after `index_insert`.
            if has_returning {
                let row =
                    eval_returning(&node.returning, &new_vals, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(row);
            }
        }
        Ok(())
    }
}

/// Read the live stored row at `rowid` as the width-`N+1` `[c0..c_{N-1}, rowid]` shape a
/// scan would produce (VIRTUAL generated columns computed, STORED ones decoded), or
/// `None` if no row exists there. Used both to clean an `OR REPLACE` victim and to refresh
/// a buffered row whose slot an earlier iteration invalidated — the latter relies on the
/// `None` case to skip a row already deleted to make room (pass-2 semantics). A plain table
/// (`gprograms` empty) takes the positional decode and does no compute; a generated-column
/// table decodes through the virtual-aware path and fills the VIRTUAL subset so an index
/// keyed on a generated column matches. Reads only, so it takes the shared pager.
#[allow(clippy::too_many_arguments)]
fn read_live_row(
    pager: &dyn Pager,
    root: PageId,
    def: &TableDef,
    gprograms: &[GeneratedProgram],
    catalog: &dyn Catalog,
    plan: &Plan,
    db: DbIndex,
    rowid: i64,
    rt: &mut Runtime,
    enc: TextEncoding,
) -> Result<Option<Row>> {
    let mut row: Row = {
        let mut tc = TableCursor::open(pager, root)?;
        if !tc.seek_exact(rowid)? {
            return Ok(None);
        }
        let payload = tc.payload()?;
        if gprograms.is_empty() {
            decode_table_row_enc(&payload, rowid, def, enc)
        } else {
            decode_table_row_skipping_virtual_enc(&payload, rowid, def, enc)
        }
    };
    if !gprograms.is_empty() {
        let env = Env { catalog, pagers: Pagers::One { db, pager }, plan };
        compute_generated(gprograms, true, &mut row, env, rt)?;
    }
    Ok(Some(row))
}

/// What to do with the row about to be rewritten. A hard conflict (Abort/Fail/Rollback)
/// still short-circuits as an `Err` from [`detect_conflict`]; Ignore yields `Skip`.
enum Action {
    /// No conflict (or none that stops the rewrite): rewrite the row (in place, or at its
    /// new rowid).
    Proceed,
    /// OnConflict::Ignore hit a conflict: leave this row untouched.
    Skip,
    /// OnConflict::Replace: delete these existing conflicting rowids (their table rows AND
    /// all their index entries) first, then rewrite this row. Non-empty and deduped, and it
    /// NEVER contains `old_rowid` — this row is rewritten, not deleted as its own victim.
    /// An empty conflict set yields `Proceed`, never `Replace(vec![])`.
    Replace(Vec<i64>),
}

/// Detect a rowid or UNIQUE-index conflict for the row about to be rewritten, on the
/// PRE-UPDATE state (this row's own old entries are still present), and decide what to
/// do under `on_conflict`. Reads only, so it takes the shared pager.
///
/// * Ignore / Abort / Fail / Rollback short-circuit on the FIRST conflict (Ignore ->
///   `Skip`; the rest -> the constraint `Err`).
/// * Replace does NOT stop at the first conflict: sqlite deletes EVERY pre-existing row
///   causing a violation before applying the update (lang_conflict.html), so it gathers
///   all of them — the row already sitting at `new_rowid` when the rowid moves, plus each
///   UNIQUE index's colliding row — into an `Action::Replace` set for the caller to delete
///   first. This row's OWN entries are excluded by probing with `old_rowid` (the identity
///   its still-present entries are keyed by), so a victim is always a DIFFERENT row.
#[allow(clippy::too_many_arguments)]
fn detect_conflict(
    pager: &dyn Pager,
    on_conflict: &OnConflict,
    def: &TableDef,
    root: PageId,
    old_rowid: i64,
    new_rowid: i64,
    keys: &[Option<Vec<Value>>],
    index_plans: &[IndexPlan],
    enc: TextEncoding,
) -> Result<Action> {
    // The precomputed UNIQUE-probe keys (the NEW row's) are 1:1 with `index_plans` (built by
    // `index_keys_for_plans` over the same plans); the index loop below pairs them positionally.
    // A `None` key is a row the rewritten values do not place in that (PARTIAL) index — no
    // entry, so it cannot conflict there.
    debug_assert_eq!(keys.len(), index_plans.len(), "precomputed keys are 1:1 with index plans");
    // REPLACE accumulates every conflicting rowid here; the other policies return on the
    // first conflict and never touch it.
    let mut victims: Vec<i64> = Vec::new();

    // Rowid uniqueness only matters when the rowid actually moves: a same-rowid update
    // REPLACEs this row's own slot (never a conflict). When it moves, a row already
    // sitting at `new_rowid` is necessarily a DIFFERENT row (ours is at `old_rowid`).
    if new_rowid != old_rowid {
        let mut tc = TableCursor::open(pager, root)?;
        if tc.seek_exact(new_rowid)? {
            match on_conflict {
                OnConflict::Ignore => return Ok(Action::Skip),
                // The row occupying the moved-to slot is the victim REPLACE deletes. It is a
                // DIFFERENT row (`new_rowid != old_rowid`), so this can never enqueue this
                // row's own slot as a victim.
                OnConflict::Replace => victims.push(new_rowid),
                OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                    return Err(rowid_conflict_error(def));
                }
            }
        }
    }

    // UNIQUE-index uniqueness. Pass `old_rowid`, NOT `new_rowid`, as the self-identity
    // to ignore: this row's own index entry is currently keyed by `old_rowid`, so an
    // entry with the same key and a rowid other than `old_rowid` is a genuine other
    // row. Using `new_rowid` here would false-positive when the rowid changes while an
    // indexed column is unchanged (the old entry [key, old_rowid] would look foreign) —
    // and, for REPLACE, would make this row its own victim.
    for (ip, key) in index_plans.iter().zip(keys) {
        if !ip.unique {
            continue;
        }
        // A partial index the rewritten row is not in has a `None` key — no entry, no
        // conflict — so skip its probe.
        let Some(key) = key else { continue };
        match on_conflict {
            // REPLACE: collect the colliding existing rowid (at most one per unique index —
            // a UNIQUE key has a single entry), EXCLUDING this row's own entry via `old_rowid`.
            OnConflict::Replace => {
                if let Some(existing) = unique_conflict_rowid(pager, ip, key, old_rowid, enc)? {
                    victims.push(existing);
                }
            }
            OnConflict::Ignore => {
                if unique_conflict(pager, ip, key, old_rowid, enc)? {
                    return Ok(Action::Skip);
                }
            }
            OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                if unique_conflict(pager, ip, key, old_rowid, enc)? {
                    return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
                }
            }
        }
    }

    if victims.is_empty() {
        Ok(Action::Proceed)
    } else {
        // A victim can be reached via BOTH the moved rowid AND a unique index, or via two
        // unique indexes: dedup so the write phase deletes each exactly once. This is
        // load-bearing — the write phase reads each victim back before deleting and fails
        // loud on a miss, so a duplicate rowid would delete the row then re-read the now-gone
        // row and error. By construction no victim is `old_rowid` (the rowid probe only fires
        // when `new_rowid != old_rowid`, pushing `new_rowid`; the unique probe excludes
        // `old_rowid`), so this row is always rewritten, never deleted as its own victim.
        debug_assert!(
            !victims.contains(&old_rowid),
            "UPDATE OR REPLACE victim set must never contain old_rowid {old_rowid}"
        );
        victims.sort_unstable();
        victims.dedup();
        Ok(Action::Replace(victims))
    }
}

/// The error for a duplicate rowid landing zone. An `INTEGER PRIMARY KEY` collision
/// is a PRIMARY KEY violation (`t.col`); a bare rowid collision is a ROWID violation.
fn rowid_conflict_error(def: &TableDef) -> Error {
    match def.rowid_alias {
        Some(ai) => Error::constraint(
            ConstraintKind::PrimaryKey,
            format!("{}.{}", def.name, def.columns[ai].name),
        ),
        None => Error::constraint(ConstraintKind::RowId, &def.name),
    }
}

/// Read the LIVE WITHOUT ROWID row at `pk_values` as the width-`N` schema-order row a WR
/// scan produces (VIRTUAL generated columns computed, STORED ones decoded), or `None` if
/// no row with that PRIMARY KEY exists. The WR analogue of [`read_live_row`], used by the
/// REPLACE freshness re-read: a `None` skips a row deleted to make room, a `Some` supplies
/// the current contents of a reoccupied slot to reprocess.
///
/// The stored key-record IS the row, so this probes the PK (reusing
/// [`WrLayout::pk_conflict_key`], which returns the exact stored bytes), decodes them back
/// to schema order via [`WrLayout::decode_row_enc`], then fills any VIRTUAL generated column
/// (`only_virtual = true`: STORED values are already in the decoded record). Reads only, so
/// it takes the shared pager; `scratch` is a caller-owned decode buffer reused across calls.
#[allow(clippy::too_many_arguments)]
fn read_live_wr_row(
    pager: &dyn Pager,
    root: PageId,
    def: &TableDef,
    layout: &WrLayout,
    gprograms: &[GeneratedProgram],
    catalog: &dyn Catalog,
    plan: &Plan,
    db: DbIndex,
    pk_values: &[Value],
    scratch: &mut Vec<Value>,
    rt: &mut Runtime,
    enc: TextEncoding,
) -> Result<Option<Row>> {
    let key = match layout.pk_conflict_key(pager, root, pk_values, enc)? {
        None => return Ok(None),
        Some(k) => k,
    };
    let mut row = layout.decode_row_enc(&key, def, scratch, enc);
    if !gprograms.is_empty() {
        let env = Env { catalog, pagers: Pagers::One { db, pager }, plan };
        compute_generated(gprograms, true, &mut row, env, rt)?;
    }
    Ok(Some(row))
}

/// Whether `pk` matches any PRIMARY KEY in `victims` (the PKs deleted as REPLACE victims
/// this statement), under [`WrLayout::pk_equal`] — the per-column collation the WITHOUT ROWID
/// b-tree is keyed on, so "same key" here agrees with which b-tree slot a key lands in.
fn pk_in(layout: &WrLayout, victims: &[Vec<Value>], pk: &[Value]) -> bool {
    victims.iter().any(|v| layout.pk_equal(v, pk))
}

/// The error for a duplicate PRIMARY KEY in a WITHOUT ROWID table: a `PRIMARY KEY`
/// constraint violation naming the PK columns (`table.col1, table.col2, …`), matching the
/// [`rowid_conflict_error`] style and SQLite's message. Kept local to the UPDATE op (the
/// INSERT op has an identical private helper) so this path stays independent of the sibling
/// WR write paths — the small duplication is deliberate, not a shared seam to factor out.
fn wr_pk_conflict_error(def: &TableDef, layout: &WrLayout) -> Error {
    let detail = layout
        .pk_columns()
        .iter()
        .map(|&i| format!("{}.{}", def.name, def.columns[i].name))
        .collect::<Vec<_>>()
        .join(", ");
    Error::constraint(ConstraintKind::PrimaryKey, detail)
}
