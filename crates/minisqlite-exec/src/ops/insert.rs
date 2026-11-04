//! The `INSERT` DML operator: eager two-phase write, then stream `RETURNING`.
//!
//! Unlike a read operator (which streams rows out of storage), an `INSERT` must
//! *apply* its whole source before it can report anything, so it does all its work on
//! the FIRST `next_row` pull — where the [`Runtime`] is in hand for the change
//! counters — and then streams any buffered `RETURNING` rows one per later pull.
//!
//! Two-phase, and why the phases cannot overlap: the read path borrows the pager
//! `&dyn Pager` (many cursors may read at once) while the write path needs the
//! exclusive `&mut dyn Pager`. So phase one reads the ENTIRE source under a shared
//! reborrow and buffers it (this also gives `INSERT INTO t SELECT ... FROM t` the
//! pre-insert snapshot real sqlite produces — the source sees the table as it was
//! before any row is written), and only once that shared borrow is released does
//! phase two take `&mut` and write. Both buffers — the source rows and the
//! `RETURNING` rows — are bounded by the inserted set; nothing larger is
//! materialized. (Streaming the source for a large, non-self-referential
//! `INSERT ... SELECT` is a later memory refinement the borrow split defers.)
//!
//! Row shape produced by the source and consumed here: the source emits `source_width`
//! values per row, which are mapped onto the table's `N` columns (`column_count`) by
//! `columns` (explicit target list) or positionally (all `N` in `CREATE TABLE` order).

use minisqlite_btree::{
    index_delete_with_order, index_insert_with_order, table_delete, table_insert, TableCursor,
};
use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_expr::{eval, truth, EvalExpr};
use minisqlite_fileformat::{encode_record_enc, TextEncoding};
use minisqlite_pager::{text_encoding_of, PageId, Pager};
use minisqlite_plan::{
    ConflictTarget, Insert, OnConflict, Plan, TriggerDmlEvent, UpsertActionPlan, UpsertPlan,
};
use minisqlite_types::{apply_affinity, ConstraintKind, Error, Result, Row, Value};

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
use crate::ops::must_be_int::{datatype_mismatch, must_be_int, MustBeInt};
use crate::ops::returning::eval_returning;
use crate::ops::sequence::{read_sequence, write_sequence};
use crate::ops::trigger;
use crate::ops::without_rowid::{wr_layout, WrLayout};
use crate::row::{
    decode_table_row_enc, decode_table_row_skipping_virtual_enc, resolve_base_table,
};
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the `INSERT` cursor. It holds the write-side pager set ([`PagerSet`]) the write
/// needs and defers all work to the first pull (see the module doc).
pub(crate) fn insert<'e>(
    node: &'e Insert,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(InsertCursor {
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

struct InsertCursor<'e> {
    node: &'e Insert,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    /// The correlated frame this INSERT runs under: empty at top level, or a firing
    /// trigger's `OLD ++ NEW` frame when this INSERT is a trigger action. The source is
    /// built with it (so a `VALUES(NEW.x)` reads the frame), and each source row is
    /// stripped back to its trailing `source_width` clean values before mapping.
    outer: &'e [Value],
    /// Whether the eager apply has run (it runs exactly once, on the first pull).
    built: bool,
    /// Buffered `RETURNING` rows, streamed one per `next_row` after the apply. Empty
    /// when the statement has no `RETURNING` clause (the cursor then yields no rows).
    returning: Vec<Row>,
    /// Index of the next buffered `RETURNING` row to emit.
    ret_idx: usize,
}

impl RowCursor for InsertCursor<'_> {
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

impl InsertCursor<'_> {
    /// Apply the whole `INSERT` (both phases) and buffer any `RETURNING` rows.
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        // Copy out the shared, borrow-independent context so the rest of the method
        // only touches `self.pagers` (a shared source view, then the exclusive target
        // pager, never overlapping) and `self.returning`. `catalog`/`plan`/`node` are
        // `Copy` references.
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        let n = node.column_count;

        // (1) Resolve the target table. INSERT is WR-aware, so it resolves through
        // `resolve_base_table` (which accepts a WITHOUT ROWID table) and branches below;
        // a missing table still fails closed here. `node.db` is the target namespace
        // (`main` unless a temp/qualified target). `def` borrows through the `Copy`
        // `&dyn Catalog`, so it carries lifetime `'e` and does NOT pin a borrow on `self`
        // — leaving `self.pagers` free to be borrowed mutably below.
        let def: &TableDef = resolve_base_table(catalog, node.db, &node.table)?;

        // Validate the plan's shape against the table before indexing by it, so a
        // malformed plan fails closed here rather than panicking on an out-of-range
        // column/affinity/alias index deep in the row loop. Both must equal `N`.
        if n != def.columns.len() {
            return Err(Error::sql(format!(
                "INSERT column_count {n} does not match table {} column count {}",
                node.table,
                def.columns.len()
            )));
        }
        if node.column_affinities.len() != n {
            return Err(Error::sql(format!(
                "INSERT column_affinities length {} does not match column_count {n}",
                node.column_affinities.len()
            )));
        }

        // A WITHOUT ROWID table stores its rows in the PRIMARY KEY index b-tree (no
        // rowid to assign/seed), so it takes a separate write path. Branch AFTER the
        // shape validation above (which applies to both kinds) and BEFORE any
        // rowid-specific machinery below.
        if def.without_rowid {
            return self.apply_without_rowid(rt, def, n);
        }

        let root = def.root_page;
        let alias = def.rowid_alias;
        // AUTOINCREMENT tables generate rowids that never reuse a freed value (autoinc.html
        // §2), tracked via `sqlite_sequence`; a plain INTEGER PRIMARY KEY seeds from the
        // table max alone and so reuses. The schema fact is the only signal (a plain and an
        // AUTOINCREMENT `INTEGER PRIMARY KEY` are identical at the row level).
        let autoincrement = def.autoincrement;
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // The target's generated-column programs (STORED + VIRTUAL, in column order), from
        // the plan-level map — loop-invariant, so bound once here. Empty for a plain table
        // (the write loop then does no generated work and takes the byte-identical record
        // path). When non-empty, every generated value is computed into the logical row so
        // constraints / indexes / triggers / RETURNING all see it, and the physical record is
        // built by `stored_record` (which OMITS VIRTUAL columns — computed on read, never
        // stored, gencol.html).
        let gprograms = plan.generated_programs(node.db, &node.table);

        // The triggers to fire for THIS write (see `trigger::effective_triggers`): a
        // top-level INSERT borrows its pre-compiled `node.triggers`; a trigger-ACTION INSERT
        // (empty vec by the one-level compile bound) recompiles the target's own triggers to
        // recurse — but only under `PRAGMA recursive_triggers` (default OFF). A trigger-free
        // top-level INSERT yields an empty set, so it is unchanged.
        let effective =
            trigger::effective_triggers(&node.triggers, rt, catalog, node.db, def, TriggerDmlEvent::Insert)?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow of the whole store set — `INSERT … SELECT` may
        // read a source in another namespace than the target). Buffer the entire source,
        // then seed the auto-rowid counter from the TARGET namespace's current maximum
        // rowid. The scope bounds the shared source view so it is released before the
        // write phase takes the exclusive target pager.
        let mut buffered: Vec<Row> = Vec::new();
        let next_auto_seed: i64;
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut src = build_cursor(&node.source, env, self.outer)?;
            while let Some(r) = src.next_row(rt)? {
                buffered.push(r);
            }
            drop(src);
            // The auto-rowid seed reads the TARGET store (a shared read; the source drain
            // above is done, so a second shared view of the set is fine).
            let target = self.pagers.source().get(node.db)?;
            let mut tc = TableCursor::open(target, root)?;
            let table_max = if tc.last()? { tc.rowid() } else { 0 };
            drop(tc);
            // AUTOINCREMENT NEVER reuses a rowid (autoinc.html §2): seed the high-water from
            // the greater of the table's current max rowid and the largest rowid EVER used
            // (which `sqlite_sequence` records), so a rowid freed by deleting the largest —
            // or every — row is not handed out again. A plain INTEGER PRIMARY KEY seeds from
            // the table max alone (and so reuses). `read_sequence` is `None` until this
            // table's first-ever insert (sqlite writes no seq row before then), which folds
            // to `0` and leaves the first generated rowid at `table_max + 1`.
            next_auto_seed = if autoincrement {
                let ever = read_sequence(target, catalog, node.db, &node.table)?.map_or(0, |s| s.seq);
                table_max.max(ever)
            } else {
                table_max
            };
        }
        let mut next_auto = next_auto_seed;
        // The database's text encoding (fileformat2 §1.3.13), read once for the whole
        // write phase. A UTF-8 database (the default) yields byte-identical records to the
        // plain codec; a UTF-16 database stores every TEXT value in that encoding so real
        // sqlite reads the file back. Read under a scoped shared view so the row loop parses
        // page 1 once; the per-row write phase re-borrows the target store mutably below —
        // re-borrowed per "piece" around each trigger fire, because a fire takes the WHOLE
        // store set to route each action to its own namespace.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);
        // Whether any row was actually written (survived NOT NULL / CHECK / conflict). Only
        // then is the AUTOINCREMENT high-water persisted below — a statement that inserts
        // nothing must not create or bump the `sqlite_sequence` row.
        let mut inserted_any = false;

        // (3)-(8) WRITE PHASE (exclusive borrow, one row at a time).
        let has_returning = !node.returning.is_empty();
        for src_row in &buffered {
            // Strip the correlated frame a trigger-action source prepends, recovering the
            // clean `source_width` values this row supplies (a no-op at top level, where
            // `outer` is empty and the row is already clean). See `clean_source_row`.
            let clean = clean_source_row(src_row, node.source_width)?;

            // (3) Map the source row onto the N target columns (unset columns NULL).
            let vals = map_row(node, clean, n)?;

            // (4) Determine the rowid (lang_createtable.html §5 "ROWID and the INTEGER
            // PRIMARY KEY"). The INTEGER PRIMARY KEY slot holds ONLY a signed 64-bit
            // integer, so the supplied value goes through the shared `OP_MustBeInt`
            // coercion (`must_be_int`): a supplied integer — or a string/real that
            // losslessly converts to one (so '123' → 123 and 2.0 → 2, matching sqlite) —
            // IS the rowid; a NULL slot, or a table with no rowid alias, auto-assigns
            // max(rowid)+1 (INSERT is the one caller for which `MustBeInt::Null` is NOT an
            // error); and a blob, a fractional or out-of-range real, or a non-numeric
            // string (`MustBeInt::NotInt`) is a "datatype mismatch" that aborts the
            // statement (NOT a silent auto-assign — the bug this replaced).
            // `explicit` records whether the rowid was supplied (vs. auto-assigned): only
            // a supplied rowid can collide with an existing one, so it gates the rowid
            // uniqueness probe in step (6). The auto high-water mark `next_auto` is NOT
            // committed here: a row skipped below (a NOT NULL violation, or an ignored
            // UNIQUE/rowid conflict) must consume no rowid, so the commit is deferred to
            // step (7) once the row is actually written — the next auto row then reuses
            // max(rowid)+1, matching sqlite.
            let coerced = match alias {
                Some(ai) => must_be_int(&vals[ai]),
                None => match node.rowid_source {
                    // A plain rowid table whose target list named rowid/_rowid_/oid: the
                    // explicit rowid is the raw source value at `rowid_source` (not stored
                    // as a column, so read the CLEAN source row, not `vals`). `must_be_int`
                    // gives the same coercion as the alias path (an integer, or a
                    // losslessly-integral string/real, is the rowid; a blob/fractional/
                    // non-numeric aborts).
                    Some(rs) => match clean.get(rs) {
                        Some(v) => must_be_int(v),
                        None => return Err(Error::sql("INSERT rowid_source index out of range")),
                    },
                    // No alias and no explicit rowid: auto-assign max+1.
                    None => MustBeInt::Null,
                },
            };
            let (mut rowid, explicit) = match coerced {
                MustBeInt::Int(i) => (i, true),
                MustBeInt::Null => {
                    let r = next_auto.checked_add(1).ok_or_else(|| {
                        Error::sql("rowid space exhausted (auto-increment past i64::MAX)")
                    })?;
                    (r, false)
                }
                MustBeInt::NotInt => return Err(datatype_mismatch()),
            };

            // (5) Apply per-column affinity to the STORED record; the alias column is
            // stored NULL (the rowid lives in the b-tree key and is refilled on read).
            // `logical` mirrors `stored` but carries the rowid in the alias slot — it
            // is the affinity-consistent row RETURNING and index keys are built from.
            let mut stored = Vec::with_capacity(n);
            for (j, v) in vals.iter().enumerate() {
                if alias == Some(j) {
                    stored.push(Value::Null);
                } else {
                    stored.push(apply_affinity(v.clone(), node.column_affinities[j]));
                }
            }
            // Width-N logical row `[c0..c_{N-1}]` (the alias slot carries the rowid). One
            // extra slot is reserved because `enforce_checks_over_new_row` (5.6) appends the
            // rowid as the trailing register N — the `[c0..c_{N-1}, rowid]` layout the CHECK
            // predicates bind against — and truncates back, so that append never reallocates.
            let mut logical = Vec::with_capacity(n + 1);
            logical.extend_from_slice(&stored);
            if let Some(ai) = alias {
                logical[ai] = Value::Integer(rowid);
            }

            // (5.4) Compute GENERATED columns (gencol.html) from the mapped row, in `CREATE
            // TABLE` column order, BEFORE the trigger frame / NOT NULL / CHECK / uniqueness /
            // storage — so a generated column participates in NEW.*, every constraint, the
            // indexes, and RETURNING exactly like a stored column. Each program's expr binds
            // over the `[c0..c_{N-1}, rowid]` eval row (register N is the rowid, so an INTEGER
            // PRIMARY KEY reference resolves there); `compute_generated` applies each column's
            // declared affinity and fills its slot IN PLACE, so a later generated column reads
            // an earlier one already computed. STORED values become part of the record built at
            // step (7); a VIRTUAL value is used by constraints/indexes/RETURNING but OMITTED
            // from the stored record (`stored_record`), matching real sqlite. Empty `gprograms`
            // (a plain table) does nothing and keeps the byte-identical record path.
            if !gprograms.is_empty() {
                // Extend `logical` to the width-(N+1) eval row (rowid at register N), compute
                // in place, then truncate back to the width-N logical row. A scoped target
                // pager backs the eval; released before the BEFORE fire (which takes the set).
                logical.push(Value::Integer(rowid));
                let pager = self.pagers.target(node.db)?;
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                compute_generated(gprograms, false, &mut logical, env, rt)?;
                logical.truncate(n);
            }

            // Build the `OLD ++ NEW` trigger frame ONCE per row when this INSERT carries
            // triggers (OLD all-NULL — an INSERT has no old row; NEW = the `[cols, rowid]`
            // being inserted). It is reused for the BEFORE fire (below, pre-write) and the
            // AFTER fire (post-write). `None` when there are no triggers, so a trigger-free
            // INSERT builds nothing and runs byte-for-byte as before. A BEFORE trigger sees
            // NEW as the row about to be inserted — before NOT NULL / CHECK / uniqueness —
            // matching sqlite's "before the database is changed" (lang_createtrigger.html §2).
            let mut frame = if triggers.is_empty() {
                None
            } else {
                let mut new_half = Vec::with_capacity(n + 1);
                new_half.extend_from_slice(&logical);
                new_half.push(Value::Integer(rowid));
                Some(trigger::build_frame(n, None, Some(new_half)))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row: skip the write, the
                // AFTER triggers, and RETURNING, and move to the next source row — nothing
                // is counted (matching `OR IGNORE`'s `continue`, and an auto rowid consumes
                // none since `next_auto` is not yet advanced for this row).
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

            // Piece-2 target pager: re-borrowed after the BEFORE fire for the TOCTOU rowid
            // re-read, NOT NULL / CHECK / FK / uniqueness, the REPLACE victim deletes, and the
            // row + index write. The re-borrow addresses the same store, so a same-namespace
            // BEFORE action's inserts (which advance the live table max) are visible to (5.1).
            let pager = self.pagers.target(node.db)?;

            // (5.1) Re-derive an auto-assigned rowid from the LIVE table max, closing a
            // TOCTOU that only trigger re-entrancy can open. A BEFORE trigger just fired (and
            // prior rows' triggers ran earlier) may have inserted into THIS same table,
            // advancing its max past the loop-local `next_auto` (which only tracks this
            // operator's OWN writes). The auto path hands out `next_auto + 1` and SKIPS the
            // rowid-uniqueness probe on the invariant "an auto rowid is max+1, so it cannot
            // collide"; `table_insert` REPLACEs on a rowid clash — so a stale `next_auto`
            // would silently OVERWRITE the trigger's row. Re-reading the live max restores the
            // invariant (`max(next_auto, live_max) + 1` is above every rowid now in the table
            // AND above every rowid this statement already handed out, so it still cannot
            // collide and the probe stays safely skipped; for AUTOINCREMENT the `next_auto`
            // term also honors the never-reuse high-water). Gated on `frame.is_some()`
            // (triggers present) and `!explicit`, so a trigger-free or explicit-rowid INSERT
            // re-reads nothing and is byte-identical to before. When the max advanced, reflect
            // the final rowid in `logical`'s alias slot and rebuild the frame's NEW half so the
            // AFTER trigger and RETURNING observe the row's real rowid. (`stored` carries NULL
            // in the alias slot — the rowid is the b-tree key, not a stored column — so it is
            // rowid-independent and needs no rebuild.)
            if !explicit && frame.is_some() {
                let live_max = {
                    let mut tc = TableCursor::open(&*pager, root)?;
                    if tc.last()? { tc.rowid() } else { 0 }
                };
                let rederived = next_auto.max(live_max).checked_add(1).ok_or_else(|| {
                    Error::sql("rowid space exhausted (auto-increment past i64::MAX)")
                })?;
                if rederived != rowid {
                    rowid = rederived;
                    if let Some(ai) = alias {
                        logical[ai] = Value::Integer(rowid);
                    }
                    // A generated column may reference the INTEGER PRIMARY KEY (the rowid),
                    // whose value just changed — recompute so the record, indexes, the AFTER
                    // frame's NEW half, and RETURNING all reflect the final rowid.
                    if !gprograms.is_empty() {
                        logical.push(Value::Integer(rowid));
                        let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                        compute_generated(gprograms, false, &mut logical, env, rt)?;
                        logical.truncate(n);
                    }
                    let mut new_half = Vec::with_capacity(n + 1);
                    new_half.extend_from_slice(&logical);
                    new_half.push(Value::Integer(rowid));
                    frame = Some(trigger::build_frame(n, None, Some(new_half)));
                }
            }

            // (5.5) Enforce column NOT NULL over the logical row, BEFORE the uniqueness
            // probe and the write, mirroring update.rs step (6.5). An ABORT/FAIL/
            // ROLLBACK (and, as a documented interim, REPLACE) raises the constraint
            // error and leaves storage untouched; OR IGNORE skips the row — and since
            // `next_auto` is not yet committed, a skipped auto row consumes no rowid.
            // The rowid-alias slot (now `Integer(rowid)`) is exempt inside the helper,
            // so a NULL supplied for an INTEGER PRIMARY KEY stays an auto-assign request
            // rather than a NOT NULL violation.
            match enforce_not_null(def, &logical, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (5.6) Enforce table/column CHECK constraints over the new row, AFTER NOT NULL
            // and BEFORE the uniqueness probe and the write. The row-layout choreography —
            // the `[c0..c_{N-1}, rowid]` register-N append for an INTEGER PRIMARY KEY alias,
            // the truncate-back-to-width-N-even-on-error invariant, and the scoped `EvalCtx`
            // — lives in `enforce_checks_over_new_row` so it stays identical to UPDATE (6.6).
            // The outcome resolves under `on_conflict` exactly like (5.5): OR IGNORE SKIPs
            // just the offending row (`continue`, no error — and since `next_auto` is not yet
            // committed, a skipped auto row consumes no rowid), while ABORT/FAIL/ROLLBACK/
            // REPLACE raise MID-LOOP so a violation on any row of a multi-row INSERT aborts
            // the statement and the engine's implicit-txn rollback backs out every row
            // already written (no per-statement savepoint — that rollback is the mechanism).
            match enforce_checks_over_new_row(
                &node.checks,
                &mut logical,
                n,
                rowid,
                &node.on_conflict,
                Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan },
                rt,
            )? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (5.7) FOREIGN KEY (child side): when `PRAGMA foreign_keys` is ON, a new row's
            // key columns must reference an existing parent (or be NULL). No-op otherwise.
            // Reads the parent through a shared reborrow of the pager (its overlay reflects
            // this statement's earlier staged rows). A violation aborts the statement — the
            // `OR IGNORE`-skips-an-FK-failure nuance is a documented follow-up (the common
            // default-ABORT case is correct). The parent-side `ON DELETE/UPDATE` actions live
            // in the DELETE/UPDATE operators (see `foreign_key`'s SCOPE note).
            foreign_key::enforce_child_foreign_keys(def, &logical, catalog, node.db, &*pager, rt)?;

            // (5.8) EVAL PHASE of index maintenance: compute this row's index keys NOW,
            // while the pager is still borrowed SHARED, so the conflict probe and the write
            // below reuse them. An index key column can be a compiled EXPRESSION (evaluated
            // through a shared-pager `EvalCtx`), which cannot be evaluated once the write
            // phase takes the exclusive `&mut` — so the keys are precomputed here exactly as
            // the CHECK constraints were evaluated above. When the table has NO index this is
            // skipped entirely: an index-free INSERT (a common bulk-write target) builds no
            // frame and clones nothing per row. Otherwise the frame is `[c0..c_{N-1}, rowid]`
            // (the scope the key exprs bind against), formed IN PLACE by appending the rowid
            // to `logical`'s reserved trailing slot (its `with_capacity(n + 1)`) and truncating
            // back — no second full-row copy. For an all-ordinary-column index each key is the
            // same `[cols.., rowid]` a single-phase write would produce; an expression index
            // evaluates its computed key columns through the ctx.
            let new_keys = if index_plans.is_empty() {
                Vec::new()
            } else {
                logical.push(Value::Integer(rowid));
                let keys = {
                    let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                    let mut ctx = EvalCtx { rt, env, outer: &logical };
                    index_keys_for_plans(&index_plans, &logical, rowid, &mut ctx)?
                };
                logical.truncate(n);
                keys
            };

            // (6) Uniqueness. An UPSERT (`ON CONFLICT ... DO ...`) makes the per-row
            // decision (lang_upsert.html §2): a no-conflict row falls through to the normal
            // write below; a matched `DO NOTHING` skips the row; a matched `DO UPDATE`
            // rewrites the existing conflicting row in place (its `WHERE` may still make it a
            // no-op); an unhandled conflict ABORTs. A plain INSERT enforces uniqueness under
            // `on_conflict` exactly as before.
            if let Some(upsert) = &node.upsert {
                match decide_upsert(
                    &*pager,
                    upsert,
                    def,
                    root,
                    alias,
                    rowid,
                    explicit,
                    &new_keys,
                    &index_plans,
                    enc,
                )? {
                    // No conflict: fall through to the normal write path below.
                    UpsertDecision::Insert => {}
                    // DO NOTHING on a conflict: skip the row (no insert, no update, no error).
                    UpsertDecision::Nothing => continue,
                    // DO UPDATE on a conflict: rewrite the existing row in place instead of
                    // inserting. `do_upsert_update` is handed the whole `&mut self.pagers` set
                    // (piece-2 `pager`'s last use was `decide_upsert` above, so the reborrow is
                    // free) — it evaluates the subquery-legal WHERE/SET/RETURNING exprs under the
                    // whole-slice shared source view (any namespace reachable) and reborrows the
                    // target store mutably only for its writes. It returns the evaluated
                    // RETURNING output row when the row was updated and `has_returning`, else
                    // `None` (no RETURNING, or a `WHERE`-gated no-op).
                    UpsertDecision::Update { existing_rowid, clause_idx } => {
                        let UpsertActionPlan::Update { assignments, predicate } =
                            &upsert.clauses[clause_idx].action
                        else {
                            // `decide_upsert` returns `Update` only for a `DO UPDATE` clause;
                            // any other action here is a logic bug, not an expected state.
                            return Err(Error::format(
                                "UPSERT: decision selected DO UPDATE for a non-UPDATE clause",
                            ));
                        };
                        let ret = do_upsert_update(
                            &mut self.pagers,
                            catalog,
                            plan,
                            rt,
                            node,
                            def,
                            root,
                            alias,
                            n,
                            &index_plans,
                            assignments,
                            predicate.as_ref(),
                            existing_rowid,
                            &logical,
                            rowid,
                            has_returning,
                            enc,
                        )?;
                        if let Some(row) = ret {
                            self.returning.push(row);
                        }
                        continue;
                    }
                }
            } else {
                match detect_conflict(
                    &*pager,
                    &node.on_conflict,
                    def,
                    root,
                    rowid,
                    explicit,
                    &new_keys,
                    &index_plans,
                    enc,
                )? {
                    Action::Proceed => {}
                    Action::Skip => continue, // OnConflict::Ignore: drop the row silently.
                    Action::Replace(victims) => {
                        // OnConflict::Replace: sqlite deletes every pre-existing row causing
                        // the constraint violation, THEN inserts the new row. Each victim's
                        // index entries are keyed by ITS stored values, so read the row back
                        // to recompute them before deleting. The read (a shared borrow) is
                        // scoped closed before the deletes take the exclusive borrow — the
                        // operator never holds both at once. A victim at `rowid` (a rowid
                        // conflict) is deleted here too, so the insert below is always a clean
                        // insert with no in-place special case.
                        for victim in victims {
                            let mut old_row: Vec<Value> = {
                                let mut tc = TableCursor::open(&*pager, root)?;
                                if tc.seek_exact(victim)? {
                                    let payload = tc.payload()?;
                                    // Width N+1 `[c0..c_{N-1}, rowid]` with the alias slot
                                    // filled — the same shape (and index-key source) the row
                                    // was inserted with, so its old entries recompute exactly.
                                    // A generated-column table's record OMITS its VIRTUAL
                                    // columns, so it decodes through the virtual-aware path
                                    // (NULL placeholders) and the compute below fills them —
                                    // otherwise a positional decode would mis-map the columns.
                                    if gprograms.is_empty() {
                                        decode_table_row_enc(&payload, victim, def, enc)
                                    } else {
                                        decode_table_row_skipping_virtual_enc(&payload, victim, def, enc)
                                    }
                                } else {
                                    // Every victim rowid was observed LIVE moments ago in this
                                    // same single-threaded statement (a rowid probe that seeked
                                    // it, or an index entry pointing at it), and the set is
                                    // deduped, so a missing table row here is not an expected
                                    // state — it means a corrupt store (a dangling index entry)
                                    // or a logic bug. Fail loud (the statement rolls back)
                                    // rather than skip the cleanup and then insert a duplicate
                                    // key over the top; mirrors update.rs's "scanned row
                                    // vanished" guard and the corrupt-index Err in dml_index.
                                    return Err(Error::format(format!(
                                        "INSERT OR REPLACE: conflicting row at rowid {victim} \
                                         not found when deleting it to make room"
                                    )));
                                }
                            };
                            // Fill the victim's VIRTUAL generated columns (absent from the
                            // record) so an index keyed on one deletes the correct entry.
                            // STORED values are already present from the decode, so only the
                            // virtual subset is computed.
                            if !gprograms.is_empty() {
                                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                                compute_generated(gprograms, true, &mut old_row, env, rt)?;
                            }
                            // EVAL PHASE: the victim's index keys, from its width-(N+1)
                            // `[c0..c_{N-1}, rowid]` row, under the SHARED borrow (an
                            // expression key needs the ctx) — before the exclusive-borrow
                            // deletes below. Col-only (today) reads the same columns + rowid
                            // the pre-expression delete did.
                            let victim_keys = {
                                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                                let mut ctx = EvalCtx { rt, env, outer: &old_row };
                                index_keys_for_plans(&index_plans, &old_row, victim, &mut ctx)?
                            };
                            // FOREIGN KEY (parent side, ON DELETE): this REPLACE deletes the
                            // victim to make room, so — exactly like a standalone DELETE
                            // (ops/delete.rs) — fire every incoming FK's ON DELETE action on it
                            // FIRST: RESTRICT/NO ACTION abort, CASCADE deletes referencing
                            // children, SET NULL/SET DEFAULT rewrite them. Located by the
                            // victim's OLD key values (`old_row[..n]`, its width-N logical row),
                            // still in storage below. A no-op when foreign_keys is OFF (default).
                            foreign_key::enforce_parent_delete(def, &old_row[..n], catalog, node.db, &mut *pager, rt)?;
                            delete_index_entries(&mut *pager, &index_plans, &victim_keys, enc)?;
                            table_delete(&mut *pager, root, victim)?;
                        }
                    }
                }
            }

            // (7) Write the row and every index entry, then record the change. Commit
            // the auto-increment high-water mark NOW — the row survived the NOT NULL and
            // uniqueness checks and is actually being written, so it (and only it)
            // consumes a rowid. The implicit deletes REPLACE performed above are NOT
            // counted: sqlite's REPLACE does not increment the change counter for rows it
            // removes to make room (spec/sqlite-doc/lang_conflict.html: "Nor does REPLACE
            // increment the change counter"), so a single-row INSERT OR REPLACE reports
            // changes() == 1 — one `record_insert`, exactly as the normal path.
            next_auto = next_auto.max(rowid);
            // The physical record: a plain table encodes `stored` directly (fast path); a
            // table with generated columns builds the record from the computed `logical` row
            // via `stored_record`, which blanks the rowid-alias slot AND OMITS every VIRTUAL
            // generated column (computed on read, never stored — gencol.html), so a real
            // sqlite3 reads the file back with the exact column layout it expects.
            let record = if gprograms.is_empty() {
                encode_record_enc(&stored, enc)
            } else {
                encode_record_enc(&stored_record(&logical, def), enc)
            };
            table_insert(&mut *pager, root, rowid, &record)?;
            insert_index_entries(&mut *pager, &index_plans, &new_keys, enc)?;
            rt.record_insert(rowid);
            inserted_any = true;

            // Fire AFTER triggers now the row is written (post-write, per row). Reuses the
            // frame built above; a skipped row (NOT NULL / CHECK / IGNORE) `continue`d before
            // reaching here, so AFTER never fires for a row that was not inserted.
            if let Some(frame) = &frame {
                // An AFTER trigger's `RAISE(IGNORE)` cannot un-write the row (it is already
                // stored and counted, and IGNORE rolls nothing back); it only abandons this
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

            // (8) RETURNING: evaluate each expr over `[cols.., rowid]` and buffer it. This is
            // the LAST use of `logical` in the iteration, so it is MOVED (not cloned) into
            // `regs` — one width-N row saved per returned row on a bulk `INSERT … RETURNING`,
            // mirroring UPDATE/UPSERT which also move their logical row here.
            if has_returning {
                let mut regs = logical;
                regs.push(Value::Integer(rowid));
                // Whole-slice shared SOURCE view (`eval_returning`): a RETURNING subquery may
                // read ANY namespace (an ATTACHed db / `temp`), exactly like the VALUES exprs
                // above — lang_returning.html §3 allows subqueries. Read-only and the last
                // per-row step (the AFTER fire's `&mut self.pagers` borrow already closed), so
                // no target pager is reborrowed.
                let row = eval_returning(&node.returning, &regs, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(row);
            }
        }

        // Persist the AUTOINCREMENT high-water once, after the write loop, so the next
        // statement — and the next reopen — never reuses a rowid. `next_auto` now holds the
        // largest rowid used or ever seen (seeded from `max(table_max, seq)`, advanced by
        // `.max(rowid)` for every written row, explicit or generated), so it is `>=` the old
        // `seq` and the upsert is monotonic. Gated on `inserted_any` so a zero-row insert
        // leaves `sqlite_sequence` untouched (no spurious row).
        if autoincrement && inserted_any {
            // Re-borrow the target store to persist the AUTOINCREMENT high-water, after the
            // per-row loop released its piece pagers.
            let pager = self.pagers.target(node.db)?;
            write_sequence(&mut *pager, catalog, node.db, &node.table, next_auto)?;
        }
        Ok(())
    }

    /// Apply an `INSERT` into a WITHOUT ROWID table: the row IS the PRIMARY KEY index
    /// b-tree's key record, so there is no rowid to assign or seed. Mirrors the rowid
    /// path's phases — buffer the source under a shared borrow, then write each row under
    /// the exclusive borrow — but keyed by PRIMARY KEY (withoutrowid.html; fileformat2
    /// §2.4) via the [`super::without_rowid`] layout/probe primitives.
    ///
    /// Per row: map → affinity → NOT NULL (explicit columns AND the implicit PK NOT NULL)
    /// → CHECK → PRIMARY KEY + secondary-index uniqueness under `on_conflict` → write the
    /// full-row record into the PK b-tree and every secondary-index entry.
    /// `last_insert_rowid()` is left UNCHANGED (SQLite does not set it for a WITHOUT ROWID
    /// insert); the change is still counted via `record_change`.
    ///
    /// Secondary indexes are maintained through the shared [`super::dml_index`] WR path,
    /// which keys each entry by `[indexed cols.., trailing PK..]` (fileformat2 §2.5.1) — the
    /// PRIMARY KEY playing the role the rowid plays for a rowid-table index. EXPRESSION
    /// indexes on a WR table remain fail-closed (the planner binds their key exprs against a
    /// frame with no WR rowid); that loud error is raised by `build_index_plans` here.
    fn apply_without_rowid(&mut self, rt: &mut Runtime, def: &TableDef, n: usize) -> Result<()> {
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        // Cache the WR b-tree root once for every probe+insert in this statement. Sound
        // because an index-b-tree's ROOT PageId is stable across inserts: growth splits
        // pages BELOW the root (or lifts a new root only via the pager, which does not
        // change `def.root_page` mid-statement), so a later iteration's probe/insert
        // targets the same tree the earlier ones grew. The within-statement PK probe
        // (step 6) then sees earlier staged rows through the pager's transaction overlay
        // (read-your-writes), which is what catches a duplicate PK inside one INSERT.
        let root = def.root_page;
        let layout = wr_layout(def)?;
        // The target's generated-column programs, loop-invariant (see the rowid path). A WR
        // generation expr binds over `[c0..c_{N-1}]` (no rowid register); STORED values are
        // written, VIRTUAL values are omitted from the record by the virtual-aware `WrLayout`.
        let gprograms = plan.generated_programs(node.db, &node.table);

        // The WR secondary-index write plans (fileformat2 §2.5.1 key shape, set on each
        // plan's `wr` field by `build_index_plan`). Building them here ALSO raises the loud
        // fail-closed error for an EXPRESSION index on a WR table (see `build_wr_index_key`),
        // so an unsupported index shape aborts before any row is written. Empty for a WR
        // table with no secondary index — the maintenance loop below then does nothing.
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // UPSERT (`ON CONFLICT ... DO ...`) on a WITHOUT ROWID table is handled per-row in the
        // write loop below (`decide_upsert_wr` / `do_upsert_update_wr`), keyed by the PRIMARY
        // KEY b-tree instead of a rowid: a matched `DO UPDATE` rewrites the existing row via the
        // same delete-old-key / insert-new-record choreography a WR `UPDATE` uses (`ops/update.rs`).

        // The triggers to fire for THIS write (mirrors the rowid path). A WR trigger frame is
        // laid at width `W = N` — a WR row has NO rowid register, and the plan side's
        // `new_old_sources` places `NEW` at `old.width()` (= N for a WR table), so a body's
        // `NEW.col_k` / `OLD.col_k` binds to `Column(N+k)` / `Column(k)` and `NEW.rowid` is a
        // bind error upstream (WR hides the rowid). `trigger::build_frame_wr` builds that width;
        // firing goes through the shared `fire_triggers`. The EFFECTIVE set (not bare
        // `node.triggers`) also honors `recursive_triggers` for a nested trigger-action INSERT
        // into a WR table; an empty set fires nothing and keeps the trigger-free path
        // byte-identical (no frame is built below).
        let effective =
            trigger::effective_triggers(&node.triggers, rt, catalog, node.db, def, TriggerDmlEvent::Insert)?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow of the whole store set): buffer the whole source
        // before any write, so an `INSERT INTO t SELECT ... FROM t` sees the pre-insert
        // snapshot. The scope bounds the shared source view so it is released before the
        // write phase takes the exclusive target pager.
        let mut buffered: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut src = build_cursor(&node.source, env, self.outer)?;
            while let Some(r) = src.next_row(rt)? {
                buffered.push(r);
            }
        }

        // The database's text encoding, read once for the whole write phase (see the rowid
        // path), under a SCOPED shared view of the target store (the temporary `source()`
        // borrow is released as this statement ends). A WITHOUT ROWID row is stored in the PK
        // index b-tree, so its record — and therefore its TEXT columns — must be in this
        // encoding too. The per-row write below reborrows the target store mutably.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);

        // (3)-(8) WRITE PHASE. Each row takes the exclusive target pager for its write pieces
        // in a borrow that CLOSES before the read-only RETURNING eval, so RETURNING runs under
        // the whole-slice shared source view (a RETURNING subquery may read any namespace).
        let has_returning = !node.returning.is_empty();
        for src_row in &buffered {
            // Strip any correlated frame prefix (a no-op at top level; see `clean_source_row`).
            let clean = clean_source_row(src_row, node.source_width)?;

            // (3) Map the source row onto the N columns (unset columns NULL).
            let vals = map_row(node, clean, n)?;

            // (5) Apply per-column affinity to the STORED record. A WR table has no rowid
            // alias, so every column stores its real (affinity-coerced) value — no blank
            // slot. `logical` is built at capacity n+1 because `enforce_checks_over_new_row`
            // appends a trailing register and truncates back (see below).
            let mut logical: Vec<Value> = Vec::with_capacity(n + 1);
            for (j, v) in vals.iter().enumerate() {
                logical.push(apply_affinity(v.clone(), node.column_affinities[j]));
            }

            // (5.4) Compute GENERATED columns over the mapped WR row (gencol.html), in column
            // order. A WR generation expr binds over `[c0..c_{N-1}]` (a WR table has no rowid
            // register), so the width-N `logical` IS the eval row. STORED values fill their
            // slot and go into the record; a VIRTUAL value fills its slot too — seen by NOT
            // NULL / PK / CHECK — but the virtual-aware `WrLayout` OMITS it from the stored key
            // record (`storage_values`, step 7) and the scan recomputes it on read. The eval
            // pager is SCOPED and released before the BEFORE fire (which takes the whole pager
            // set), mirroring the rowid path.
            if !gprograms.is_empty() {
                let pager = self.pagers.target(node.db)?;
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                compute_generated(gprograms, false, &mut logical, env, rt)?;
            }

            // Build the `OLD ++ NEW` WR frame ONCE per row when this INSERT carries triggers:
            // OLD all-NULL (an INSERT has no old row); NEW = the width-N `[c0..c_{N-1}]` being
            // inserted (NO rowid register — `build_frame_wr` lays each half at width N). Reused
            // for the BEFORE fire (pre-write) and the AFTER fire (post-write); `None` when
            // trigger-free, so that path builds nothing and stays byte-identical.
            let frame = if triggers.is_empty() {
                None
            } else {
                Some(trigger::build_frame_wr(n, None, Some(logical[..n].to_vec())))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row: skip the write, the
                // AFTER fire, and RETURNING, and move on — nothing is counted (matching the
                // `OR IGNORE` `continue` below). A BEFORE trigger sees NEW before NOT NULL /
                // CHECK / uniqueness ("before the database is changed", lang_createtrigger.html).
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

            // The exclusive target pager for THIS row's write pieces (the constraint probes,
            // the PK-uniqueness / secondary-index probes, the REPLACE victim deletes, and the
            // b-tree + index writes). Re-borrowed AFTER the BEFORE fire (which needs the whole
            // set) and released before the AFTER fire + RETURNING, so those take the shared view.
            let pager = self.pagers.target(node.db)?;

            // (5.5) Enforce explicit column NOT NULL, then the PRIMARY KEY's implicit
            // NOT NULL (withoutrowid.html), both under `on_conflict` (IGNORE skips the row).
            match enforce_not_null(def, &logical, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }
            match layout.enforce_pk_not_null(def, &logical, &node.on_conflict)? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (5.6) Enforce CHECK constraints. A WR table has no rowid register, and its
            // CHECK predicates bind against the width-N row (rowid is hidden on WR, so a
            // CHECK referencing it is a bind error upstream). `enforce_checks_over_new_row`
            // appends one trailing register for the rowid-alias CHECK layout it shares with
            // the rowid path; that slot is never read here, so a `0` placeholder is inert.
            match enforce_checks_over_new_row(
                &node.checks,
                &mut logical,
                n,
                0,
                &node.on_conflict,
                Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan },
                rt,
            )? {
                ConstraintOutcome::Proceed => {}
                ConstraintOutcome::Skip => continue,
            }

            // (5.7) FOREIGN KEY (child side) — same immediate check as the rowid path. A WR
            // child row is width-N (no rowid register), and the key columns are `0..N`, so
            // `logical` is the right shape to read.
            foreign_key::enforce_child_foreign_keys(def, &logical, catalog, node.db, &*pager, rt)?;

            // (5.8) EVAL PHASE of secondary-index maintenance: compute this row's WR index
            // keys `[indexed cols.., trailing PK..]` (fileformat2 §2.5.1) now, for the UNIQUE
            // probe and the write below. WR index key columns are ALL ordinary columns (an
            // expression WR index is fail-closed at plan build); the scoped eval ctx is needed
            // only to evaluate a PARTIAL index's WHERE predicate over `logical` (gating its
            // membership — a `None` key is a row not in that index). Empty when no secondary
            // index; the ctx's shared pager+rt borrow ends before the writes below.
            let pk_values = layout.pk_values(&logical);
            let new_keys = {
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                let mut ctx = EvalCtx { rt, env, outer: &logical };
                wr_index_keys_for_plans(&index_plans, &logical, &mut ctx)?
            };

            // (6) Uniqueness / UPSERT, resolved against the CURRENT pager state (a later row in
            // this multi-row INSERT sees an earlier staged row through the transaction overlay).
            // An UPSERT (`ON CONFLICT ... DO ...`) makes the per-row decision keyed by the
            // PRIMARY KEY b-tree (lang_upsert.html §2): a no-conflict row falls through to the
            // clean write below; a matched `DO NOTHING` skips the row; a matched `DO UPDATE`
            // rewrites the existing conflicting row in place (its `WHERE` may still make it a
            // no-op); an unhandled conflict ABORTs. A plain INSERT resolves the PRIMARY KEY then
            // each UNIQUE secondary index under `on_conflict` (IGNORE skips the row on the first
            // conflict; ABORT/FAIL/ROLLBACK raise; REPLACE gathers every conflicting existing row
            // by its PRIMARY KEY to delete first).
            if let Some(upsert) = &node.upsert {
                match decide_upsert_wr(
                    &*pager, upsert, def, &layout, root, &pk_values, &new_keys, &index_plans, enc,
                )? {
                    // No conflict: fall through to the clean write path below.
                    WrUpsertDecision::Insert => {}
                    // DO NOTHING on a conflict: skip the row (no insert, no update, no error).
                    WrUpsertDecision::Nothing => continue,
                    // DO UPDATE on a conflict: rewrite the existing row in place instead of
                    // inserting. `do_upsert_update_wr` takes the whole `&mut self.pagers` set —
                    // `decide_upsert_wr` above was this iteration's last use of the per-row
                    // exclusive `pager`, and the arm `continue`s (never reads `pager` again on
                    // this path), so NLL frees the reborrow (as the rowid path's Update arm does).
                    // It returns the RETURNING output row when the row was updated and
                    // `has_returning`, else `None` (no RETURNING, or a `WHERE`-gated no-op).
                    WrUpsertDecision::Update { existing_pk, clause_idx } => {
                        let UpsertActionPlan::Update { assignments, predicate } =
                            &upsert.clauses[clause_idx].action
                        else {
                            // `decide_upsert_wr` returns `Update` only for a `DO UPDATE` clause;
                            // any other action here is a logic bug, not an expected state.
                            return Err(Error::format(
                                "UPSERT: decision selected DO UPDATE for a non-UPDATE clause",
                            ));
                        };
                        let ret = do_upsert_update_wr(
                            &mut self.pagers,
                            catalog,
                            plan,
                            rt,
                            node,
                            def,
                            &layout,
                            root,
                            n,
                            &index_plans,
                            assignments,
                            predicate.as_ref(),
                            &existing_pk,
                            &logical,
                            has_returning,
                            enc,
                        )?;
                        if let Some(row) = ret {
                            self.returning.push(row);
                        }
                        continue;
                    }
                }
            } else {
                let mut victims: Vec<Vec<Value>> = Vec::new();
                let mut skip = false;
                if layout.pk_conflict_key(&*pager, root, &pk_values, enc)?.is_some() {
                    match &node.on_conflict {
                        OnConflict::Ignore => skip = true,
                        OnConflict::Replace => victims.push(pk_values.clone()),
                        OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                            return Err(pk_conflict_error(def, &layout));
                        }
                    }
                }
                if !skip {
                    for (ip, key) in index_plans.iter().zip(&new_keys) {
                        if !ip.unique {
                            continue;
                        }
                        // A partial index not admitting this row has a `None` key — no entry,
                        // no conflict — so skip its probe.
                        let Some(key) = key else { continue };
                        match &node.on_conflict {
                            // REPLACE: collect the conflicting row's PRIMARY KEY (no self to
                            // exclude — this row's own entry is not yet written, so `None`).
                            OnConflict::Replace => {
                                if let Some(vpk) = wr_unique_conflict(&*pager, ip, key, None, enc)? {
                                    victims.push(vpk);
                                }
                            }
                            OnConflict::Ignore => {
                                if wr_unique_conflict(&*pager, ip, key, None, enc)?.is_some() {
                                    skip = true;
                                    break;
                                }
                            }
                            OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                                if wr_unique_conflict(&*pager, ip, key, None, enc)?.is_some() {
                                    return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
                                }
                            }
                        }
                    }
                }
                if skip {
                    continue; // OnConflict::Ignore hit a conflict: drop this row silently.
                }

                // (6.1) REPLACE victim cleanup: delete every conflicting existing row — its PK
                // b-tree record AND all its secondary-index entries — before the insert, so the
                // insert is always clean. Dedup by encoded PK (one row can collide on the PK AND
                // a unique index, or two indexes can name one row); `wr_delete_victim_by_pk` is
                // idempotent, but dedup avoids a wasted re-probe. REPLACE does NOT count the rows
                // it removes (lang_conflict.html), so there is no `record_change` for a victim.
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
                    }
                }
            }

            // (7) Write the full-row record into the PRIMARY KEY index b-tree, then every
            // secondary-index entry. There is no rowid, so `last_insert_rowid()` is left
            // untouched (SQLite's WR rule); the insert is counted once via `record_change`.
            let storage = layout.storage_values(&logical);
            let record = encode_record_enc(&storage, enc);
            index_insert_with_order(&mut *pager, root, &record, layout.order())?;
            insert_index_entries(&mut *pager, &index_plans, &new_keys, enc)?;
            rt.record_change();

            // Fire AFTER triggers now the row is written (post-write). Reuses the frame; a row
            // skipped by NOT NULL / CHECK / conflict `continue`d earlier, so AFTER never fires
            // for a row that was not inserted. An AFTER `RAISE(IGNORE)` cannot un-write the row
            // (already stored + counted); it only abandons this row's RETURNING. The per-row
            // `pager` above was last used by `insert_index_entries`, so its exclusive borrow has
            // ended (NLL) and `fire_triggers` is free to take the whole pager set.
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

            // (8) RETURNING: evaluate each expr over the width-N logical row. A WR base
            // row has no trailing rowid register (rowid is hidden on WR), so nothing is
            // appended — the layout RETURNING binds against is exactly `[c0..c_{N-1}]`.
            // Evaluated under the whole-slice shared SOURCE view so a RETURNING subquery may
            // read any namespace (lang_returning.html §3), matching every other DML RETURNING
            // site. The per-row `pager` above is not used past `index_insert`, so its exclusive
            // `&mut self.pagers` borrow has ended by here (NLL) and `source()` is free to take.
            if has_returning {
                let row = eval_returning(&node.returning, &logical, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(row);
            }
        }
        Ok(())
    }
}

/// The error for a duplicate PRIMARY KEY in a WITHOUT ROWID table: a `PRIMARY KEY`
/// constraint violation naming the PK columns (`table.col1, table.col2, …`), matching
/// the `rowid_conflict_error` style and SQLite's message.
fn pk_conflict_error(def: &TableDef, layout: &WrLayout) -> Error {
    let detail = layout
        .pk_columns()
        .iter()
        .map(|&i| format!("{}.{}", def.name, def.columns[i].name))
        .collect::<Vec<_>>()
        .join(", ");
    Error::constraint(ConstraintKind::PrimaryKey, detail)
}

/// What to do with the row about to be written. A hard conflict (Abort/Fail/Rollback)
/// is not represented here — it short-circuits as an `Err` from [`detect_conflict`].
enum Action {
    /// No conflict (or none that stops the write): insert the row.
    Proceed,
    /// OnConflict::Ignore hit a conflict: drop this row silently.
    Skip,
    /// OnConflict::Replace: delete these existing conflicting rowids (their table rows
    /// AND all their index entries) first, then insert. Non-empty and deduped — an
    /// empty conflict set yields `Proceed`, never `Replace(vec![])`.
    Replace(Vec<i64>),
}

/// Recover the CLEAN source row — the `source_width` values a row supplies to the target
/// columns — from what the source cursor emitted. A trigger-action source runs with the
/// firing `OLD ++ NEW` frame as its correlated outer row: a leaf `VALUES(NEW.x, …)` source
/// emits `frame ++ values` (the `values` operator prepends the outer), while a `SELECT`/
/// projection source emits only its projected columns. In BOTH shapes the clean values are
/// the LAST `source_width` of the emitted row, so slicing the trailing `source_width` is
/// the one rule that handles either. At top level the source runs with an EMPTY frame, so
/// the emitted row is already exactly `source_width` wide and this returns it unchanged. A
/// row narrower than `source_width` is a malformed plan and fails closed rather than
/// panicking on the slice.
fn clean_source_row(src_row: &[Value], source_width: usize) -> Result<&[Value]> {
    src_row
        .len()
        .checked_sub(source_width)
        .map(|start| &src_row[start..])
        .ok_or_else(|| Error::sql("INSERT source row narrower than source_width"))
}

/// Map one source row onto the `N` target columns. With an explicit `columns` list,
/// source value `k` lands in target column `map[k]`; otherwise source value `k` is
/// target column `k` (positional, `source_width == N`). Unset columns stay NULL (real
/// sqlite would substitute the column DEFAULT — a documented later enhancement). The
/// explicit-rowid SENTINEL `N` (== `column_count`) in `map` is NOT a stored column: it
/// is skipped here, the rowid being read separately via `node.rowid_source`. A source
/// row narrower than expected, or a target index PAST the sentinel (`> N`), is a
/// malformed plan and fails closed rather than panicking on the index.
fn map_row(node: &Insert, src_row: &[Value], n: usize) -> Result<Vec<Value>> {
    let mut vals = vec![Value::Null; n];
    match &node.columns {
        Some(map) => {
            for (k, &target) in map.iter().enumerate() {
                let v = src_row.get(k).ok_or_else(|| {
                    Error::sql("INSERT source row has fewer values than the target column list")
                })?;
                // The explicit-rowid sentinel (target == N) is not a stored column — the
                // rowid is read separately via `node.rowid_source`. Skip storing it.
                if target == n {
                    continue;
                }
                if target > n {
                    return Err(Error::sql("INSERT target column index out of range"));
                }
                vals[target] = v.clone();
            }
        }
        None => {
            for (k, slot) in vals.iter_mut().enumerate() {
                let v = src_row.get(k).ok_or_else(|| {
                    Error::sql("INSERT source row has fewer values than the table columns")
                })?;
                *slot = v.clone();
            }
        }
    }
    Ok(vals)
}

/// Detect rowid / UNIQUE-index conflicts for the row about to be written and decide
/// what to do under `on_conflict`. Reads only, so it takes the shared pager.
///
/// * Ignore / Abort / Fail / Rollback short-circuit on the FIRST conflict (Ignore ->
///   `Skip`; the rest -> the constraint `Err`).
/// * Replace does NOT stop at the first conflict: sqlite must delete EVERY pre-existing
///   row causing a violation, so it gathers all of them — the row already at `rowid`
///   (only an explicit rowid can collide) plus each UNIQUE index's colliding row — into
///   an `Action::Replace` set for the caller to delete before inserting.
#[allow(clippy::too_many_arguments)]
fn detect_conflict(
    pager: &dyn Pager,
    on_conflict: &OnConflict,
    def: &TableDef,
    root: PageId,
    rowid: i64,
    explicit: bool,
    keys: &[Option<Vec<Value>>],
    index_plans: &[IndexPlan],
    enc: TextEncoding,
) -> Result<Action> {
    // The precomputed UNIQUE-probe keys are 1:1 with `index_plans` (built by
    // `index_keys_for_plans` over the same plans); pairing them positionally below relies
    // on that, so pin it. A `None` key is a row not in that (PARTIAL) index — no entry, so
    // no conflict to detect there.
    debug_assert_eq!(keys.len(), index_plans.len(), "precomputed keys are 1:1 with index plans");
    // REPLACE accumulates every conflicting rowid here; the other policies return on the
    // first conflict and never touch it.
    let mut victims: Vec<i64> = Vec::new();

    // Rowid uniqueness. An auto-assigned rowid is max+1 and cannot collide, so only an
    // explicitly supplied rowid needs the probe.
    if explicit {
        let mut tc = TableCursor::open(pager, root)?;
        if tc.seek_exact(rowid)? {
            match on_conflict {
                OnConflict::Ignore => return Ok(Action::Skip),
                OnConflict::Replace => victims.push(rowid),
                OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                    return Err(rowid_conflict_error(def));
                }
            }
        }
    }

    // UNIQUE-index uniqueness. For REPLACE, collect the colliding rowid from each unique
    // index (at most one per index — a UNIQUE key has a single entry); for the others,
    // the first conflict decides the outcome.
    for (ip, key) in index_plans.iter().zip(keys) {
        if !ip.unique {
            continue;
        }
        // A partial index that does not admit this row has a `None` key: it holds no entry
        // for the row, so it cannot conflict — skip its probe entirely.
        let Some(key) = key else { continue };
        match on_conflict {
            OnConflict::Replace => {
                if let Some(existing) = unique_conflict_rowid(pager, ip, key, rowid, enc)? {
                    victims.push(existing);
                }
            }
            OnConflict::Ignore => {
                if unique_conflict(pager, ip, key, rowid, enc)? {
                    return Ok(Action::Skip);
                }
            }
            OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                if unique_conflict(pager, ip, key, rowid, enc)? {
                    return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
                }
            }
        }
    }

    if victims.is_empty() {
        Ok(Action::Proceed)
    } else {
        // A row can collide on both the rowid and a unique index, or two unique indexes
        // can name the same existing row: dedup so each victim is deleted exactly once.
        // This is load-bearing, not mere tidiness — the write phase reads each victim's
        // stored row back before deleting it and now fails loud if that row is missing,
        // so processing one rowid twice (its second read landing after its own delete)
        // would turn a benign duplicate into a spurious "row vanished" error.
        victims.sort_unstable();
        victims.dedup();
        Ok(Action::Replace(victims))
    }
}

/// The error for a duplicate rowid. An INTEGER PRIMARY KEY duplicate is a PRIMARY KEY
/// violation (`t.col`); a bare rowid duplicate is a ROWID violation.
fn rowid_conflict_error(def: &TableDef) -> Error {
    match def.rowid_alias {
        Some(ai) => Error::constraint(
            ConstraintKind::PrimaryKey,
            format!("{}.{}", def.name, def.columns[ai].name),
        ),
        None => Error::constraint(ConstraintKind::RowId, &def.name),
    }
}

// ---------------------------------------------------------------------------
// UPSERT (`INSERT ... ON CONFLICT ... DO UPDATE / DO NOTHING`; lang_upsert.html).
//
// Owned by this operator: when `node.upsert` is set, step (6) of the write loop calls
// `decide_upsert` per candidate row (instead of the plain `detect_conflict`) and, on a
// matched `DO UPDATE`, `do_upsert_update` rewrites the existing conflicting row in place.
// The plan side (`minisqlite-plan::compile::insert`) resolved each clause's conflict target
// to a `ConflictTarget` (a column set, an expression-index root, or "any") and bound its
// SET/WHERE over the combined `existing ++ excluded` row.
// ---------------------------------------------------------------------------

/// The per-row UPSERT decision after probing the table's uniqueness constraints.
enum UpsertDecision {
    /// No matching conflict: insert the candidate row via the normal write path.
    Insert,
    /// A matched `DO NOTHING`: skip the row entirely (no insert, no update, no error).
    Nothing,
    /// A matched `DO UPDATE`: rewrite the existing conflicting row (`existing_rowid`) using
    /// clause `clause_idx`. That clause's `WHERE` may still turn the rewrite into a no-op.
    Update { existing_rowid: i64, clause_idx: usize },
}

/// One uniqueness conflict a candidate row triggers.
struct UpsertConflict {
    /// The violated constraint's column set (a UNIQUE index's columns, or the rowid-alias
    /// column). EMPTY for a bare-rowid conflict OR an expression-index conflict — a column-set
    /// (`ON CONFLICT (cols)`) target cannot name either, so only a target-omitted clause, or
    /// (for an expression index) an `ExprIndex` target keyed on `index_root`, matches it.
    target: Vec<usize>,
    /// The violated UNIQUE index's b-tree root page, when this conflict came from an index —
    /// the identity a [`ConflictTarget::ExprIndex`] clause matches (it pins the target to one
    /// index's root, the same value the plan stored). `None` for a rowid/IPK conflict, which
    /// no index-keyed target can name.
    index_root: Option<PageId>,
    /// The existing (conflicting) row's rowid, to load + rewrite for a `DO UPDATE`.
    existing_rowid: i64,
    /// Which constraint was violated, for the ABORT-fallback error if no clause matches.
    constraint: ConflictConstraint,
}

/// Which uniqueness constraint a conflict violated, to name the ABORT-fallback error when no
/// UPSERT clause matches it (an unhandled conflict is a real constraint error, as for a
/// plain INSERT).
enum ConflictConstraint {
    /// A rowid / INTEGER PRIMARY KEY collision.
    Rowid,
    /// A UNIQUE-index collision, carrying that index's `table.col, ...` detail.
    Unique(String),
}

impl ConflictConstraint {
    fn into_error(self, def: &TableDef) -> Error {
        match self {
            ConflictConstraint::Rowid => rowid_conflict_error(def),
            ConflictConstraint::Unique(detail) => Error::constraint(ConstraintKind::Unique, detail),
        }
    }
}

/// Classify a candidate row for an UPSERT: probe every uniqueness constraint WITHOUT
/// erroring, then pick the clause that handles the conflict (`lang_upsert.html` §2).
///
/// * No conflict → [`UpsertDecision::Insert`] (the normal write path runs).
/// * Otherwise the FIRST clause whose conflict target matches a violated constraint runs — a
///   [`ConflictTarget::Columns`] target matches a column set order-insensitively, a
///   [`ConflictTarget::ExprIndex`] target matches the conflict on that index's root, and a
///   [`ConflictTarget::Any`] (target-omitted) clause matches any conflict, firing on the
///   first-detected one. A matched `DO NOTHING` → [`UpsertDecision::Nothing`]; a matched
///   `DO UPDATE` → [`UpsertDecision::Update`] on that conflict's existing row.
/// * A conflict no clause matches is an unhandled uniqueness violation → the constraint
///   `Err` (ABORT), exactly as a plain INSERT would raise.
///
/// Reads only, so it takes the shared pager, and probes the CURRENT pager state — so a later
/// row in the same multi-row INSERT correctly conflicts with an earlier row this statement
/// already wrote (the exec writes incrementally).
#[allow(clippy::too_many_arguments)]
fn decide_upsert(
    pager: &dyn Pager,
    upsert: &UpsertPlan,
    def: &TableDef,
    root: PageId,
    alias: Option<usize>,
    rowid: i64,
    explicit: bool,
    keys: &[Option<Vec<Value>>],
    index_plans: &[IndexPlan],
    enc: TextEncoding,
) -> Result<UpsertDecision> {
    // The precomputed UNIQUE-probe keys are 1:1 with `index_plans` (built by
    // `index_keys_for_plans` over the same plans); the loop below pairs them positionally. A
    // `None` key is a row not in that (PARTIAL) index — it holds no entry, so no conflict.
    debug_assert_eq!(keys.len(), index_plans.len(), "precomputed keys are 1:1 with index plans");
    // Collect every uniqueness conflict this candidate row triggers, in probe order: the
    // rowid/PK collision first (only an explicitly supplied rowid can collide), then each
    // UNIQUE index's collision. Order matters only for a target-omitted clause (it fires on
    // the first-listed conflict).
    let mut conflicts: Vec<UpsertConflict> = Vec::new();

    if explicit {
        let mut tc = TableCursor::open(pager, root)?;
        if tc.seek_exact(rowid)? {
            // A rowid/IPK conflict's target is the rowid-alias column set (the column whose
            // uniqueness IS the rowid's); a bare-rowid table has no such column, so an empty
            // set that only a target-omitted clause can match.
            let target = match alias {
                Some(ai) => vec![ai],
                None => Vec::new(),
            };
            conflicts.push(UpsertConflict {
                target,
                index_root: None,
                existing_rowid: rowid,
                constraint: ConflictConstraint::Rowid,
            });
        }
    }

    for (ip, key) in index_plans.iter().zip(keys) {
        if !ip.unique {
            continue;
        }
        // A partial index that does not admit this row has a `None` key — no entry, no
        // conflict — so it contributes no `UpsertConflict`.
        let Some(key) = key else { continue };
        if let Some(existing) = unique_conflict_rowid(pager, ip, key, rowid, enc)? {
            conflicts.push(UpsertConflict {
                // A column-set (`ON CONFLICT (cols)`) target names plain columns, so an
                // expression index has an EMPTY column set here (`col_positions` is `Some` for a
                // column-only index, `None` for an expression index). An expression index is
                // instead named by an `ExprIndex` target via `index_root` below; a bare-column
                // set is still matched by a target-omitted clause.
                target: ip.col_positions().unwrap_or_default(),
                index_root: Some(ip.root),
                existing_rowid: existing,
                constraint: ConflictConstraint::Unique(ip.detail.clone()),
            });
        }
    }

    if conflicts.is_empty() {
        return Ok(UpsertDecision::Insert);
    }

    // First clause (written order) whose target matches a violated constraint wins.
    for (clause_idx, clause) in upsert.clauses.iter().enumerate() {
        let matched = match &clause.target {
            // Target omitted: fires on the first-detected conflict (probe order).
            ConflictTarget::Any => conflicts.first(),
            // A column-set target matches the violated constraint whose column set equals it,
            // order-insensitively.
            ConflictTarget::Columns(target) => {
                conflicts.iter().find(|c| same_column_set(&c.target, target))
            }
            // An expression-index target matches the conflict on THAT index, keyed by its
            // b-tree root (the identity the plan pinned) — an expression index has no column
            // set, so it can only be named this way.
            ConflictTarget::ExprIndex(root) => {
                conflicts.iter().find(|c| c.index_root == Some(*root))
            }
        };
        if let Some(c) = matched {
            return Ok(match clause.action {
                UpsertActionPlan::Nothing => UpsertDecision::Nothing,
                UpsertActionPlan::Update { .. } => {
                    UpsertDecision::Update { existing_rowid: c.existing_rowid, clause_idx }
                }
            });
        }
    }

    // No clause matched any violated constraint: an unhandled uniqueness conflict ABORTs,
    // exactly as a plain INSERT would. Report the first-detected conflict's constraint. The
    // `conflicts.is_empty()` early return above guarantees at least one, so `None` here is a
    // logic bug — surface it via the file's fail-loud-via-`Error` convention, not a panic.
    match conflicts.into_iter().next() {
        Some(c) => Err(c.constraint.into_error(def)),
        None => Err(Error::format("UPSERT: detected a conflict but had none to report")),
    }
}

/// Order-insensitive equality of two conflict-target column sets. A uniqueness constraint
/// never repeats a column, so equal length plus "every column of `a` appears in `b`" means the
/// sets are equal — an allocation-free compare over these tiny (typically 1-2 element) sets.
fn same_column_set(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x))
}

/// Run a matched `DO UPDATE` clause: rewrite the existing conflicting row in place, the same
/// choreography an `UPDATE` uses (`lang_upsert.html` §2 — the INSERT "behaves as an
/// UPDATE"). Returns the evaluated RETURNING output row when `has_returning` and the row was
/// updated, or `None` (no RETURNING, or a `WHERE`-gated no-op).
///
/// Namespace reach: the subquery-legal exprs — the `WHERE` predicate, the SET assignments,
/// and the RETURNING clause — are evaluated under the whole-slice shared SOURCE view
/// (`pagers.source()`), so a scalar subquery in any of them may read ANY namespace (an
/// ATTACHed db / `temp`), exactly like a plain UPDATE's SET (see `ops/update.rs`) and the
/// shared `ops::returning::eval_returning`. The constraint-class evals where the spec forbids
/// subqueries — the generated-column recompute, CHECK enforcement, and index-key eval — stay
/// on the fail-closed single-namespace `Pagers::One { db: node.db }` view. Borrows never
/// overlap: every shared `source()` read closes before the exclusive `pagers.target(node.db)`
/// write borrow, which itself closes before the read-only RETURNING eval.
///
/// TOCTOU invariant (KEEP if this grows): `existing_full`/`old_keys` are read while the pager
/// is borrowed SHARED, then consumed by the single exclusive `target()` write below, so they
/// must not go stale in that window. This function fires NO triggers. Its ONE nested write is
/// the parent-side FK `enforce_parent_update` on the `target()` borrow (step 6.8), which
/// rewrites rows in OTHER (child) tables — located by the OLD key VALUES passed in, not by
/// re-reading this row — so THIS table's `existing_rowid` row and its index entries are
/// untouched and the snapshot stays valid. (A self-referential FK whose child is this very
/// table is the one cross-cutting FK/DML staleness corner shared with the rowid UPDATE path; a
/// WR child fails closed inside the helper.) The plain INSERT/UPDATE/DELETE loops re-borrow
/// `target()` AFTER their BEFORE-trigger fire for the same reason. If a future change adds a
/// trigger fire or a nested write that CAN touch this table's `existing_rowid` row, it MUST
/// re-read `existing_full`/`old_keys` after that fire, or it will write a stale snapshot.
///
/// Row layout (matches `compile_upsert`'s binding): the assignment/predicate exprs bind over
/// the combined row `existing(W) ++ excluded(W)`, `W = N + 1`. The EXISTING row occupies
/// `[0, W)` (columns `[0, N)`, rowid `[N]`) and the would-be-inserted EXCLUDED row `[W, 2W)`
/// — so a bare / `table.`-qualified column and `rowid` read the existing row, an
/// `excluded.col` the candidate. New column values start from the EXISTING row (a column the
/// SET omits keeps its original value) and each assignment overwrites its column.
#[allow(clippy::too_many_arguments)]
fn do_upsert_update(
    pagers: &mut PagerSet<'_>,
    catalog: &dyn Catalog,
    plan: &Plan,
    rt: &mut Runtime,
    node: &Insert,
    def: &TableDef,
    root: PageId,
    alias: Option<usize>,
    n: usize,
    index_plans: &[IndexPlan],
    assignments: &[(usize, EvalExpr)],
    predicate: Option<&EvalExpr>,
    existing_rowid: i64,
    candidate_logical: &[Value],
    candidate_rowid: i64,
    has_returning: bool,
    enc: TextEncoding,
) -> Result<Option<Row>> {
    let w = n + 1;
    // The target's generated-column programs (see the write loop). A DO UPDATE is an in-place
    // UPDATE, so it decodes/recomputes generated columns exactly like `update.rs`.
    let gprograms = plan.generated_programs(node.db, &node.table);

    // (1) Load the existing conflicting row: width `W = [c0..c_{N-1}, rowid]`, alias slot
    // filled. `decide_upsert` saw it live moments ago in this single-threaded statement, so
    // its absence now means a corrupt store or a logic bug — fail loud. A generated-column
    // table's record OMITS its VIRTUAL columns, so it decodes through the virtual-aware path
    // and the compute below fills them (STORED values are already present) — the combined
    // `existing ++ excluded` row the SET/WHERE bind against must carry the real generated
    // values a predicate or assignment may reference.
    let mut existing_full: Vec<Value> = {
        let mut tc = TableCursor::open(pagers.source().get(node.db)?, root)?;
        if !tc.seek_exact(existing_rowid)? {
            return Err(Error::format(format!(
                "UPSERT DO UPDATE: conflicting row at rowid {existing_rowid} vanished before update"
            )));
        }
        let payload = tc.payload()?;
        if gprograms.is_empty() {
            decode_table_row_enc(&payload, existing_rowid, def, enc)
        } else {
            decode_table_row_skipping_virtual_enc(&payload, existing_rowid, def, enc)
        }
    };
    if !gprograms.is_empty() {
        // Generated-column recompute stays single-namespace (spec forbids subqueries here).
        let env =
            Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan };
        compute_generated(gprograms, true, &mut existing_full, env, rt)?;
    }

    // (2) Build the combined row `existing(W) ++ excluded(W)` the SET/WHERE bind against: the
    // existing row (columns `[0, N)`, rowid `[N]`) then the candidate columns (alias slot =
    // candidate rowid) and the candidate rowid — the exact layout `compile_upsert` bound.
    let mut combined = Vec::with_capacity(2 * w);
    combined.extend_from_slice(&existing_full);
    combined.extend_from_slice(&candidate_logical[..n]);
    combined.push(Value::Integer(candidate_rowid));
    debug_assert_eq!(combined.len(), 2 * w, "combined upsert row is existing(W) ++ excluded(W)");

    // (3) WHERE gate: a NULL or false predicate makes the DO UPDATE a no-op — the existing
    // row is left unchanged, no error, nothing inserted (lang_upsert.html §2.1).
    if let Some(pred) = predicate {
        let apply = {
            // WHERE may hold a scalar subquery over ANY namespace (a DO UPDATE is an UPDATE;
            // `lang_upsert.html` §2), so evaluate under the whole-slice shared source view,
            // NOT a single-namespace `Pagers::One` — the same cross-namespace fix as SET.
            let env = Env { catalog, pagers: pagers.source(), plan };
            let mut ctx = EvalCtx { rt, env, outer: &combined };
            truth(&eval(pred, &combined, &mut ctx)?) == Some(true)
        };
        if !apply {
            return Ok(None);
        }
    }

    // (4) Compute the new column values: start from the EXISTING row's columns (so a column
    // the SET does not mention keeps its original value), then apply each assignment,
    // evaluated over the combined row, with per-column affinity — exactly UPDATE's rule. One
    // extra slot is reserved (as at the plain-INSERT/UPDATE sites) so the later generated-col
    // recompute and the index-key EVAL can append the trailing rowid IN PLACE without a
    // per-row reallocation, then truncate back to the width-N logical row.
    let mut new_vals: Vec<Value> = Vec::with_capacity(n + 1);
    new_vals.extend_from_slice(&existing_full[..n]);
    {
        // SET may hold a scalar subquery over ANY namespace (a DO UPDATE is an UPDATE), so
        // evaluate under the whole-slice shared source view, mirroring `ops/update.rs`.
        let env = Env { catalog, pagers: pagers.source(), plan };
        let mut ctx = EvalCtx { rt, env, outer: &combined };
        for (ci, expr) in assignments {
            if *ci >= n {
                return Err(Error::sql(format!(
                    "UPSERT DO UPDATE assignment targets column {ci} but table {} has {n} columns",
                    node.table
                )));
            }
            let v = eval(expr, &combined, &mut ctx)?;
            new_vals[*ci] = apply_affinity(v, node.column_affinities[*ci]);
        }
    }

    // (5) Determine the new rowid, then fail closed on an alias-CHANGING DO UPDATE. Assigning
    // the INTEGER PRIMARY KEY alias to a NEW value would MOVE the row, and supporting that
    // safely needs the auto-rowid high-water (`next_auto`, owned by the caller) to learn the
    // moved-to rowid: otherwise a later auto-assigned row in the SAME statement can be handed
    // the just-vacated `max+1` and — because `table_insert` REPLACES on a rowid collision —
    // silently overwrite the moved row and strand its index entries (SILENT corruption, worse
    // than the "a secondary conflict may error" this corner was descoped to). Until a move
    // threads `next_auto` back (a documented follow-up), reject it LOUDLY. A same-VALUE
    // assignment (`SET id = id`, or a string/real that converts to the existing rowid) is not a
    // move and is allowed; a NULL / non-convertible value aborts via UPDATE's `require_int`.
    let new_rowid = match alias {
        Some(ai) => must_be_int(&new_vals[ai]).require_int()?,
        None => existing_rowid,
    };
    if new_rowid != existing_rowid {
        return Err(Error::sql(
            "UPSERT DO UPDATE that changes the rowid / INTEGER PRIMARY KEY is not yet supported",
        ));
    }
    // The rowid is unchanged; normalize the alias slot to `Integer(existing_rowid)` so
    // `new_vals` IS the logical row (a `SET id = '5'` may have left Text/Real in the slot).
    if let Some(ai) = alias {
        new_vals[ai] = Value::Integer(existing_rowid);
    }

    // Recompute GENERATED columns: a SET on a base column changes any generated column that
    // depends on it (a generated column is never a SET target — rejected at compile). Compute
    // over the `[c0..c_{N-1}, rowid]` eval row so NOT NULL / CHECK / the UNIQUE probe / index
    // entries / RETURNING all observe the fresh values; a VIRTUAL column is omitted from the
    // record by `stored_record` below.
    if !gprograms.is_empty() {
        new_vals.push(Value::Integer(existing_rowid));
        // Generated-column recompute stays single-namespace (spec forbids subqueries here).
        let env =
            Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan };
        compute_generated(gprograms, false, &mut new_vals, env, rt)?;
        new_vals.truncate(n);
    }

    // (5.5 / 5.6) DO UPDATE is "OR ABORT" (lang_upsert.html §3), so the rewritten row must
    // satisfy NOT NULL and CHECK exactly as an ordinary UPDATE would (ops/update.rs steps
    // 6.5/6.6): a SET that nulls a NOT NULL column or breaks a CHECK ABORTs — it does not
    // silently write bad data. Enforce over the new logical row (the alias slot is exempt in
    // `enforce_not_null`; `enforce_checks_over_new_row` handles the register-N rowid append and
    // truncates `new_vals` back to width `n`). OR ABORT never yields `Skip` (only IGNORE does);
    // a defensive `Skip` is treated as a `WHERE`-style no-op.
    match enforce_not_null(def, &new_vals, &OnConflict::Abort)? {
        ConstraintOutcome::Proceed => {}
        ConstraintOutcome::Skip => return Ok(None),
    }
    match enforce_checks_over_new_row(
        &node.checks,
        &mut new_vals,
        n,
        existing_rowid,
        &OnConflict::Abort,
        // CHECK enforcement stays single-namespace (spec forbids subqueries here).
        Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan },
        rt,
    )? {
        ConstraintOutcome::Proceed => {}
        ConstraintOutcome::Skip => return Ok(None),
    }

    // EVAL PHASE of index maintenance for the in-place rewrite: compute BOTH the OLD keys
    // (to delete the existing entries) and the NEW keys (to probe UNIQUE and to write) while
    // the pager is still borrowed SHARED — an expression key needs the `EvalCtx`, which the
    // exclusive-borrow writes below cannot hold. Skipped when the table has NO index (no dead
    // per-row work). OLD frame is `existing_full` (width N+1, `[c0..c_{N-1}, rowid]`); the NEW
    // frame is formed IN PLACE by appending the rowid to `new_vals`'s reserved trailing slot
    // and truncating back (no second full-row copy). The rowid is unchanged (the alias-move
    // rejection above), so both use `existing_rowid`. Col-only (today) reads the same columns
    // + rowid the pre-expression path did.
    let (old_keys, new_keys) = if index_plans.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        new_vals.push(Value::Integer(existing_rowid));
        let keys = {
            // Index-key eval stays single-namespace (spec forbids subqueries here).
            let env =
                Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan };
            let mut ctx = EvalCtx { rt, env, outer: &new_vals };
            let old_keys = index_keys_for_plans(index_plans, &existing_full, existing_rowid, &mut ctx)?;
            let new_keys = index_keys_for_plans(index_plans, &new_vals, existing_rowid, &mut ctx)?;
            (old_keys, new_keys)
        };
        new_vals.truncate(n);
        keys
    };

    // (6) DO UPDATE is "OR ABORT": if the updated row would collide with a DIFFERENT row on a
    // UNIQUE constraint, abort rather than write a duplicate key. The rowid is unchanged (see
    // the alias-move rejection above), so only the secondary UNIQUE indexes can newly collide;
    // probe with `existing_rowid` as the self-identity so this row's own still-present entries
    // do not false-positive. No in-scope test provokes this; the guard keeps the store honest.
    for (ip, key) in index_plans.iter().zip(&new_keys) {
        // A partial index that does not admit the rewritten row has a `None` new key — the
        // row is not in it, so it cannot collide there.
        let Some(key) = key else { continue };
        if ip.unique && unique_conflict(pagers.source().get(node.db)?, ip, key, existing_rowid, enc)? {
            return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
        }
    }

    // (6.7) FOREIGN KEY (child side): a DO UPDATE "behaves as an UPDATE" (lang_upsert.html §2)
    // and is "OR ABORT" (§3), so the REWRITTEN row's key columns must still reference an
    // existing parent (or be NULL) under `PRAGMA foreign_keys=ON` — the SAME check the rowid
    // UPDATE path runs (`ops/update.rs`). The child-FK check at the INSERT candidate site
    // (step 5.7) validated the would-be-INSERTED row, NOT this SET-rewritten existing row, so
    // without this a `DO UPDATE SET fk_col = <dangling>` would silently write an orphaned child.
    // Read under the shared source view (its overlay reflects this statement's earlier writes),
    // like the UNIQUE probe just above and BEFORE the exclusive `target()` reborrow below; a
    // no-op when FK is OFF or the table declares no FK.
    foreign_key::enforce_child_foreign_keys(
        def,
        &new_vals,
        catalog,
        node.db,
        pagers.source().get(node.db)?,
        rt,
    )?;

    // (7) Rewrite the row in place (the UPDATE choreography): delete the OLD index entries
    // (keyed by the existing columns + rowid), write the new record (alias slot blanked to NULL
    // — the rowid lives in the b-tree key) over the SAME rowid (an in-place REPLACE via
    // `table_insert`), then the new index entries; count one change. `enc` is the
    // statement-level encoding passed in by the caller (read once), not re-derived here.
    // Exclusive write reborrow: every shared `source()` read above (the last was the UNIQUE
    // probe) has closed, so `target(node.db)` takes `&mut` the target store alone.
    let pager = pagers.target(node.db)?;
    // (6.8) FOREIGN KEY (parent side, ON UPDATE): if THIS row is a referenced parent whose key
    // an incoming FK points at and this DO UPDATE changes that key, fire the FK's ON UPDATE
    // action on the matching children (RESTRICT/NO ACTION abort; CASCADE rewrites their FK
    // columns to the new key, SET NULL nulls them, SET DEFAULT resets them) — the SAME
    // parent-side enforcement the rowid UPDATE path runs (`ops/update.rs`). Runs on the
    // exclusive `target()` borrow but BEFORE this row's own delete+reinsert below, so children
    // are still located by the OLD key VALUES passed in (`existing_full[..n]`). A no-op when FK
    // is OFF, when nothing references this table, or when no referenced key column changed; an
    // IPK/rowid change was already rejected above, so only a non-rowid referenced key reaches
    // here. Takes `&mut` because CASCADE/SET NULL/SET DEFAULT write the children.
    foreign_key::enforce_parent_update(
        def,
        &existing_full[..n],
        &new_vals,
        catalog,
        node.db,
        &mut *pager,
        rt,
    )?;
    delete_index_entries(&mut *pager, index_plans, &old_keys, enc)?;
    let record = if gprograms.is_empty() {
        if let Some(ai) = alias {
            new_vals[ai] = Value::Null;
            let rec = encode_record_enc(&new_vals, enc);
            new_vals[ai] = Value::Integer(existing_rowid);
            rec
        } else {
            encode_record_enc(&new_vals, enc)
        }
    } else {
        // A generated-column table: `stored_record` blanks the alias slot AND omits VIRTUAL
        // columns (never stored — gencol.html), matching what real sqlite writes.
        encode_record_enc(&stored_record(&new_vals, def), enc)
    };
    table_insert(&mut *pager, root, existing_rowid, &record)?;
    insert_index_entries(&mut *pager, index_plans, &new_keys, enc)?;
    rt.record_change();

    // (8) RETURNING binds against the row AS IT STANDS AFTER the update, `[c0..c_{N-1},
    // rowid]`. The exclusive `&mut` write borrow above (`pager`) has ended, so evaluate under
    // the whole-slice shared SOURCE view via the shared `eval_returning` helper — a RETURNING
    // subquery may read ANY namespace (an ATTACHed db / `temp`; lang_returning.html §3),
    // exactly like SET/WHERE above. `None` = nothing to return (no RETURNING).
    if has_returning {
        let mut regs = new_vals;
        regs.push(Value::Integer(existing_rowid));
        let row = eval_returning(&node.returning, &regs, catalog, pagers.source(), plan, rt)?;
        Ok(Some(row))
    } else {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// WITHOUT ROWID UPSERT. The WR analogues of `decide_upsert` / `do_upsert_update`, keyed by the
// PRIMARY KEY b-tree (a WR row has no rowid). Called from `apply_without_rowid`'s write loop when
// `node.upsert` is set; the plan side bound each `DO UPDATE`'s SET/WHERE over the combined row
// `existing(N) ++ excluded(N)` (`W = N` for a WR table — no rowid register).
// ---------------------------------------------------------------------------

/// The per-row UPSERT decision for a WITHOUT ROWID table after probing its uniqueness
/// constraints — the WR analogue of [`UpsertDecision`], identifying the conflicting row by its
/// PRIMARY KEY (a WR row has no rowid) rather than a rowid.
enum WrUpsertDecision {
    /// No matching conflict: insert the candidate row via the normal WR write path.
    Insert,
    /// A matched `DO NOTHING`: skip the row entirely (no insert, no update, no error).
    Nothing,
    /// A matched `DO UPDATE`: rewrite the existing conflicting row (identified by its PRIMARY
    /// KEY) using clause `clause_idx`. That clause's `WHERE` may still make the rewrite a no-op.
    Update { existing_pk: Vec<Value>, clause_idx: usize },
}

/// One uniqueness conflict a WR candidate row triggers (the WR analogue of [`UpsertConflict`]).
struct WrUpsertConflict {
    /// The violated constraint's column set: the PRIMARY KEY columns for a PK conflict, or a
    /// UNIQUE secondary index's indexed columns. A `ConflictTarget::Columns` target matches this
    /// set order-insensitively.
    target: Vec<usize>,
    /// The violated UNIQUE secondary index's b-tree root, for a `ConflictTarget::ExprIndex`
    /// target. `None` for a PRIMARY KEY conflict. A WR column-based PARTIAL unique index IS named
    /// this way — `resolve_conflict_target` resolves `ON CONFLICT(cols) WHERE <pred>` to
    /// `ExprIndex(root)` and the `ExprIndex` arm below matches this `index_root` — so the field is
    /// load-bearing for WR, not just uniform with the rowid path. (Only WR *expression* indexes
    /// still fail closed at CREATE, so an expression conflict target can never reach a WR conflict.)
    index_root: Option<PageId>,
    /// The existing (conflicting) row's PRIMARY KEY, to load + rewrite for a `DO UPDATE`.
    existing_pk: Vec<Value>,
    /// Which constraint was violated, for the ABORT-fallback error when no clause matches.
    kind: WrConflictKind,
}

/// Which WR uniqueness constraint a conflict violated, to name the ABORT-fallback error when no
/// UPSERT clause matches it (an unhandled conflict is a real constraint error, as for a plain
/// INSERT).
enum WrConflictKind {
    /// A PRIMARY KEY collision.
    PrimaryKey,
    /// A UNIQUE secondary-index collision, carrying that index's `table.col, ...` detail.
    Unique(String),
}

/// Classify a candidate row for an UPSERT into a WITHOUT ROWID table: probe the PRIMARY KEY and
/// every UNIQUE secondary index WITHOUT erroring, then pick the clause that handles the conflict
/// (`lang_upsert.html` §2) — the WR analogue of [`decide_upsert`], keyed by PRIMARY KEY.
///
/// * No conflict → [`WrUpsertDecision::Insert`] (the clean WR write path runs).
/// * Otherwise the FIRST clause whose conflict target matches a violated constraint runs — a
///   [`ConflictTarget::Columns`] target matches a column set order-insensitively (the PK columns
///   or a UNIQUE index's columns), a [`ConflictTarget::ExprIndex`] target matches the conflict on
///   that index's root (a WR column-based PARTIAL unique index is named this way), and a
///   [`ConflictTarget::Any`] (target-omitted) clause matches the first-detected conflict. A matched
///   `DO NOTHING` → [`WrUpsertDecision::Nothing`]; a matched `DO UPDATE` →
///   [`WrUpsertDecision::Update`] on that conflict's existing row.
/// * A conflict no clause matches is an unhandled uniqueness violation → the constraint `Err`
///   (ABORT), exactly as a plain INSERT would raise.
///
/// Reads only, so it takes the shared pager, and probes the CURRENT pager state (read-your-writes
/// over this statement's earlier rows). Probe order — PRIMARY KEY first, then the UNIQUE indexes
/// in plan order — matches the plain WR conflict handling and the rowid `decide_upsert`.
#[allow(clippy::too_many_arguments)]
fn decide_upsert_wr(
    pager: &dyn Pager,
    upsert: &UpsertPlan,
    def: &TableDef,
    layout: &WrLayout,
    root: PageId,
    pk_values: &[Value],
    keys: &[Option<Vec<Value>>],
    index_plans: &[IndexPlan],
    enc: TextEncoding,
) -> Result<WrUpsertDecision> {
    // Keys are 1:1 with `index_plans`; a `None` key is a row not in that (PARTIAL) index —
    // it holds no entry, so no conflict.
    debug_assert_eq!(keys.len(), index_plans.len(), "precomputed keys are 1:1 with index plans");
    let mut conflicts: Vec<WrUpsertConflict> = Vec::new();

    // PRIMARY KEY conflict first: its target is the PK column set (what an `ON CONFLICT (pkcols)`
    // names), and the existing row's PK equals the candidate's PK (they collided on it).
    if layout.pk_conflict_key(pager, root, pk_values, enc)?.is_some() {
        conflicts.push(WrUpsertConflict {
            target: layout.pk_columns().to_vec(),
            index_root: None,
            existing_pk: pk_values.to_vec(),
            kind: WrConflictKind::PrimaryKey,
        });
    }

    // Then each UNIQUE secondary index, carrying the conflicting row's reconstructed PRIMARY KEY
    // (`wr_unique_conflict` returns it) so a matched `DO UPDATE` can load and rewrite that row.
    for (ip, key) in index_plans.iter().zip(keys) {
        if !ip.unique {
            continue;
        }
        // A partial index not admitting this row has a `None` key — no entry, no conflict.
        let Some(key) = key else { continue };
        if let Some(vpk) = wr_unique_conflict(pager, ip, key, None, enc)? {
            conflicts.push(WrUpsertConflict {
                target: ip.col_positions().unwrap_or_default(),
                index_root: Some(ip.root),
                existing_pk: vpk,
                kind: WrConflictKind::Unique(ip.detail.clone()),
            });
        }
    }

    if conflicts.is_empty() {
        return Ok(WrUpsertDecision::Insert);
    }

    // First clause (written order) whose target matches a violated constraint wins.
    for (clause_idx, clause) in upsert.clauses.iter().enumerate() {
        let matched = match &clause.target {
            ConflictTarget::Any => conflicts.first(),
            ConflictTarget::Columns(target) => {
                conflicts.iter().find(|c| same_column_set(&c.target, target))
            }
            ConflictTarget::ExprIndex(r) => conflicts.iter().find(|c| c.index_root == Some(*r)),
        };
        if let Some(c) = matched {
            return Ok(match clause.action {
                UpsertActionPlan::Nothing => WrUpsertDecision::Nothing,
                UpsertActionPlan::Update { .. } => {
                    WrUpsertDecision::Update { existing_pk: c.existing_pk.clone(), clause_idx }
                }
            });
        }
    }

    // No clause matched any violated constraint: an unhandled uniqueness conflict ABORTs, exactly
    // as a plain INSERT would. Report the first-detected conflict's constraint (the emptiness
    // early-return above guarantees at least one, so `None` here is a logic bug — fail loud).
    match conflicts.into_iter().next() {
        Some(c) => Err(match c.kind {
            WrConflictKind::PrimaryKey => pk_conflict_error(def, layout),
            WrConflictKind::Unique(detail) => Error::constraint(ConstraintKind::Unique, detail),
        }),
        None => Err(Error::format("UPSERT: detected a conflict but had none to report")),
    }
}

/// Run a matched `DO UPDATE` clause for a WITHOUT ROWID table: rewrite the existing conflicting
/// row in place, the same delete-old-key / insert-new-record choreography a WR `UPDATE` uses
/// (`ops/update.rs`; `lang_upsert.html` §2 — the INSERT "behaves as an UPDATE"). Returns the
/// evaluated RETURNING output row when `has_returning` and the row was updated, or `None` (no
/// RETURNING, or a `WHERE`-gated no-op). The WR analogue of [`do_upsert_update`].
///
/// Namespace reach mirrors [`do_upsert_update`]: the subquery-legal exprs (WHERE, SET, RETURNING)
/// evaluate under the whole-slice shared SOURCE view (any namespace reachable); the evals the
/// spec forbids subqueries in (generated recompute, CHECK, index-key read) stay single-namespace.
/// Borrows never overlap: every shared `source()` read closes before the ONE exclusive `target()`
/// write borrow, which closes before the read-only RETURNING eval.
///
/// Fires NO triggers (parity with the rowid `do_upsert_update`; the caller already fired the
/// statement's BEFORE INSERT triggers for this candidate row, and skips its AFTER INSERT fire on
/// the `DO UPDATE` path).
///
/// Row layout (matches the plan-side WR binding): the assignment/predicate exprs bind over the
/// combined row `existing(N) ++ excluded(N)` — a WR row has NO rowid register, so `W = N` (unlike
/// the rowid path's `N + 1`). The EXISTING row occupies `[0, N)` and the would-be-inserted
/// EXCLUDED row `[N, 2N)`, so a bare / `table.`-qualified column reads the existing row and an
/// `excluded.col` the candidate. New values start from the EXISTING row (a column the SET omits
/// keeps its value); each assignment overwrites its column. A PRIMARY KEY-changing `DO UPDATE`
/// is supported (the WR record is re-keyed, as a WR UPDATE does) — unlike the rowid path, which
/// rejects a rowid change (its reason, the auto-rowid high-water, does not exist for a WR table).
#[allow(clippy::too_many_arguments)]
fn do_upsert_update_wr(
    pagers: &mut PagerSet<'_>,
    catalog: &dyn Catalog,
    plan: &Plan,
    rt: &mut Runtime,
    node: &Insert,
    def: &TableDef,
    layout: &WrLayout,
    root: PageId,
    n: usize,
    index_plans: &[IndexPlan],
    assignments: &[(usize, EvalExpr)],
    predicate: Option<&EvalExpr>,
    existing_pk: &[Value],
    candidate_logical: &[Value],
    has_returning: bool,
    enc: TextEncoding,
) -> Result<Option<Row>> {
    let gprograms = plan.generated_programs(node.db, &node.table);

    // (1) Load the existing conflicting row (width N). A WR row IS its PK-b-tree key record, so
    // `pk_conflict_key` returns the stored bytes and `decode_row_enc` reconstructs the schema-
    // order row. `decide_upsert_wr` saw it live moments ago in this single-threaded statement, so
    // its absence now is a corrupt store / logic bug — fail loud. A generated-column table's
    // record OMITS its VIRTUAL columns (`decode_row_enc` leaves them NULL); the compute below
    // fills them so the combined `existing ++ excluded` row carries the real generated values a
    // predicate/assignment may read.
    let mut existing_full: Vec<Value> = {
        let src = pagers.source().get(node.db)?;
        let key = layout.pk_conflict_key(src, root, existing_pk, enc)?.ok_or_else(|| {
            Error::format(format!(
                "UPSERT DO UPDATE: conflicting WITHOUT ROWID row (PRIMARY KEY {existing_pk:?}) \
                 vanished before update in table {}",
                node.table
            ))
        })?;
        let mut scratch = Vec::new();
        layout.decode_row_enc(&key, def, &mut scratch, enc)
    };
    if !gprograms.is_empty() {
        // Fill VIRTUAL generated columns (STORED ones are already in the decoded record). Stays
        // single-namespace (spec forbids subqueries in a generated expr).
        let env = Env {
            catalog,
            pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? },
            plan,
        };
        compute_generated(gprograms, true, &mut existing_full, env, rt)?;
    }

    // (2) Build the combined row `existing(N) ++ excluded(N)` the SET/WHERE bind against.
    let mut combined = Vec::with_capacity(2 * n);
    combined.extend_from_slice(&existing_full);
    combined.extend_from_slice(&candidate_logical[..n]);
    debug_assert_eq!(combined.len(), 2 * n, "combined WR upsert row is existing(N) ++ excluded(N)");

    // (3) WHERE gate: a NULL/false predicate makes the DO UPDATE a no-op — the existing row is
    // left unchanged, no error, nothing inserted (lang_upsert.html §2.1).
    if let Some(pred) = predicate {
        let apply = {
            // WHERE may hold a scalar subquery over ANY namespace (a DO UPDATE is an UPDATE), so
            // evaluate under the whole-slice shared source view (same as SET below).
            let env = Env { catalog, pagers: pagers.source(), plan };
            let mut ctx = EvalCtx { rt, env, outer: &combined };
            truth(&eval(pred, &combined, &mut ctx)?) == Some(true)
        };
        if !apply {
            return Ok(None);
        }
    }

    // (4) New column values: start from the EXISTING row (a column the SET omits keeps its value),
    // then apply each assignment over the combined row with per-column affinity — UPDATE's rule.
    let mut new_vals: Vec<Value> = Vec::with_capacity(n + 1);
    new_vals.extend_from_slice(&existing_full[..n]);
    {
        // SET may hold a scalar subquery over ANY namespace (a DO UPDATE is an UPDATE), so
        // evaluate under the whole-slice shared source view, mirroring `ops/update.rs`.
        let env = Env { catalog, pagers: pagers.source(), plan };
        let mut ctx = EvalCtx { rt, env, outer: &combined };
        for (ci, expr) in assignments {
            if *ci >= n {
                return Err(Error::sql(format!(
                    "UPSERT DO UPDATE assignment targets column {ci} but table {} has {n} columns",
                    node.table
                )));
            }
            let v = eval(expr, &combined, &mut ctx)?;
            new_vals[*ci] = apply_affinity(v, node.column_affinities[*ci]);
        }
    }

    // Recompute GENERATED columns over the new width-N row (a SET on a base column changes a
    // dependent generated column; a generated column is never a SET target — rejected at compile).
    if !gprograms.is_empty() {
        let env = Env {
            catalog,
            pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? },
            plan,
        };
        compute_generated(gprograms, false, &mut new_vals, env, rt)?;
    }

    // (5) DO UPDATE is "OR ABORT" (lang_upsert.html §3): the rewritten row must satisfy NOT NULL
    // (explicit + the implicit PRIMARY KEY NOT NULL) and CHECK, exactly as an ordinary WR UPDATE.
    // `enforce_checks_over_new_row` appends one trailing register (the rowid-alias CHECK layout it
    // shares with the rowid path — inert `0` on WR) and truncates `new_vals` back to width `n`.
    match enforce_not_null(def, &new_vals, &OnConflict::Abort)? {
        ConstraintOutcome::Proceed => {}
        ConstraintOutcome::Skip => return Ok(None),
    }
    match layout.enforce_pk_not_null(def, &new_vals, &OnConflict::Abort)? {
        ConstraintOutcome::Proceed => {}
        ConstraintOutcome::Skip => return Ok(None),
    }
    match enforce_checks_over_new_row(
        &node.checks,
        &mut new_vals,
        n,
        0,
        &OnConflict::Abort,
        Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan },
        rt,
    )? {
        ConstraintOutcome::Proceed => {}
        ConstraintOutcome::Skip => return Ok(None),
    }

    // (6) EVAL the OLD index keys (to delete the existing entries) and the NEW keys (to probe
    // UNIQUE and to write). A WR index key is a pure column read (an expression WR index fails
    // closed at plan build); the scoped eval ctx is needed only to gate a PARTIAL index on its
    // WHERE predicate — evaluated against the OLD frame for `old_keys` (so a row that was outside
    // the predicate has no stale entry to drop) and the NEW frame for `new_keys` (so a row that
    // moves in gains one). A PRIMARY KEY change alters the trailing-PK part of EVERY entry, so
    // OLD/NEW are recomputed unconditionally. The ctx's shared borrow ends before the writes.
    let old_pk = existing_pk.to_vec();
    let new_pk = layout.pk_values(&new_vals);
    let (old_keys, new_keys) = if index_plans.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let env =
            Env { catalog, pagers: Pagers::One { db: node.db, pager: pagers.source().get(node.db)? }, plan };
        let mut ctx = EvalCtx { rt, env, outer: &new_vals };
        let old_keys = wr_index_keys_for_plans(index_plans, &existing_full[..n], &mut ctx)?;
        let new_keys = wr_index_keys_for_plans(index_plans, &new_vals, &mut ctx)?;
        (old_keys, new_keys)
    };

    // A PRIMARY KEY change that collides with a DIFFERENT existing row aborts (a same-PK rewrite's
    // only PK match is this row itself, so probe only on a real change).
    if !layout.pk_equal(&old_pk, &new_pk)
        && layout.pk_conflict_key(pagers.source().get(node.db)?, root, &new_pk, enc)?.is_some()
    {
        return Err(pk_conflict_error(def, layout));
    }
    // A UNIQUE secondary-index collision with a DIFFERENT row aborts (exclude THIS row's own
    // still-present entry via its OLD PK). A partial index not admitting the rewritten row has a
    // `None` new key — no entry, no collision — so skip it.
    for (ip, key) in index_plans.iter().zip(&new_keys) {
        let Some(key) = key else { continue };
        if ip.unique
            && wr_unique_conflict(pagers.source().get(node.db)?, ip, key, Some(&old_pk), enc)?
                .is_some()
        {
            return Err(Error::constraint(ConstraintKind::Unique, &ip.detail));
        }
    }

    // (6.7) FOREIGN KEY (child side): the rewritten row's key columns must still reference an
    // existing parent (or be NULL) under `PRAGMA foreign_keys=ON`, as an ordinary WR UPDATE checks.
    foreign_key::enforce_child_foreign_keys(
        def,
        &new_vals,
        catalog,
        node.db,
        pagers.source().get(node.db)?,
        rt,
    )?;

    // The exact OLD stored key-record bytes to delete, probed on the live tree (read-your-writes).
    let old_key =
        layout.pk_conflict_key(pagers.source().get(node.db)?, root, &old_pk, enc)?.ok_or_else(
            || {
                Error::format(format!(
                    "UPSERT DO UPDATE could not find WITHOUT ROWID row to rewrite in table {} \
                     (PRIMARY KEY {old_pk:?})",
                    node.table
                ))
            },
        )?;

    // (7) Rewrite under the single exclusive target borrow: fire the parent-side FK ON UPDATE
    // action (children located by the OLD key values, still stored), then delete the OLD secondary
    // entries + OLD PK record and insert the NEW PK record + NEW secondary entries. Deleting the
    // old key first keeps a same-PK rewrite from holding a stale duplicate (a WR row's b-tree key
    // is its FULL record, so even an unchanged PK needs delete-then-insert); moving the secondary
    // entries reflects an indexed-column or PRIMARY KEY change. Count one change.
    let pager = pagers.target(node.db)?;
    foreign_key::enforce_parent_update(
        def,
        &existing_full[..n],
        &new_vals,
        catalog,
        node.db,
        &mut *pager,
        rt,
    )?;
    delete_index_entries(&mut *pager, index_plans, &old_keys, enc)?;
    index_delete_with_order(&mut *pager, root, &old_key, layout.order())?;
    let record = encode_record_enc(&layout.storage_values(&new_vals), enc);
    index_insert_with_order(&mut *pager, root, &record, layout.order())?;
    insert_index_entries(&mut *pager, index_plans, &new_keys, enc)?;
    rt.record_change();

    // (8) RETURNING binds against the row AS IT STANDS AFTER the update — a WR base row is width-N
    // with no trailing rowid register, so `new_vals` is exactly the shape RETURNING binds against.
    // Evaluated under the shared source view (a RETURNING subquery may read any namespace); the
    // exclusive write borrow above has ended (NLL) after `insert_index_entries`.
    if has_returning {
        let row = eval_returning(&node.returning, &new_vals, catalog, pagers.source(), plan, rt)?;
        Ok(Some(row))
    } else {
        Ok(None)
    }
}

