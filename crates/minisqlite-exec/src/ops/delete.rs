//! The `DELETE` DML operator: eager two-phase removal, then stream `RETURNING`.
//!
//! Like [`insert`](crate::ops::insert), a `DELETE` mutates storage, so it does all
//! its work on the FIRST `next_row` pull — where the [`Runtime`] is in hand for the
//! change counter — and then streams any buffered `RETURNING` rows one per later pull.
//!
//! Two-phase, and why the phases cannot overlap: the scan reads pages through a
//! shared `&dyn Pager` (many read cursors may borrow at once), while the removal
//! needs the exclusive `&mut dyn Pager`. So phase one drains the ENTIRE scan under a
//! shared reborrow into a buffer, and only once that borrow is released does phase
//! two take `&mut` and delete. Buffering the whole scan first is also what makes a
//! `DELETE` whose `scan` is an `IndexScan` over the very index being modified safe:
//! the scan cursor is fully consumed before any entry is removed, so it never reads a
//! tree it is mutating. The buffer is bounded by the rows the scan selects (the rows
//! to delete); nothing larger is materialized.
//!
//! Removal order per row (rowid tables): index entries FIRST, then the table row, so no
//! intermediate state ever leaves an index entry pointing at a missing table row. Both
//! `index_delete` and `table_delete` are idempotent on a missing key/rowid, so a row
//! already gone is a silent no-op rather than an error (it cannot happen from a single
//! scan, but the primitives stay defensive); only a row that was actually present
//! advances the change counter. A WITHOUT ROWID table has no separate table b-tree — a row
//! IS its PRIMARY KEY entry — so its removal deletes every secondary-index entry (keyed by
//! `[indexed cols.., trailing PK..]`, fileformat2 §2.5.1) and then the PRIMARY KEY entry
//! (see [`DeleteCursor::apply_without_rowid`]).
//!
//! Row shape consumed here depends on the target. A ROWID table's `scan` subtree emits
//! `[c0..c_{N-1}, rowid]` (width `N+1`), the rowid in register `N`, and `RETURNING`
//! evaluates against that same row. A WITHOUT ROWID table has no integer rowid: its scan
//! emits the width-`N` row `[c0..c_{N-1}]`, and each row is located in — and removed from —
//! the PRIMARY KEY index b-tree by its PRIMARY KEY (see [`DeleteCursor::apply_without_rowid`]).

use minisqlite_btree::{index_delete_with_order, table_delete};
use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_pager::text_encoding_of;
use minisqlite_plan::{Delete, Plan, TriggerDmlEvent};
use minisqlite_types::{Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::{PagerSet, Env, Pagers};
use crate::ops::dml_index::{
    build_index_plans, delete_index_entries, index_keys_for_plans, wr_index_keys_for_plans,
};
use crate::ops::foreign_key;
use crate::ops::returning::eval_returning;
use crate::ops::trigger;
use crate::ops::without_rowid::wr_layout;
use crate::row::resolve_base_table;
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the `DELETE` cursor. It holds the write-side pager set ([`PagerSet`]) the
/// removal needs and defers all work to the first pull (see the module doc).
pub(crate) fn delete<'e>(
    node: &'e Delete,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(DeleteCursor {
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

struct DeleteCursor<'e> {
    node: &'e Delete,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    /// The correlated frame this DELETE runs under: empty at top level, or a firing
    /// trigger's `OLD ++ NEW` frame when this DELETE is a trigger action. The scan is built
    /// with it, so each scan row is `frame ++ <target row>`: for a ROWID table the target is
    /// `[c0..c_{N-1}, rowid]` (the deleted rowid and index key recovered from the trailing
    /// `n + 1` values); for a WITHOUT ROWID table it is the width-`N` `[c0..c_{N-1}]` with no
    /// rowid register (see [`DeleteCursor::apply_without_rowid`]).
    outer: &'e [Value],
    /// Whether the eager removal has run (it runs exactly once, on the first pull).
    built: bool,
    /// Buffered `RETURNING` rows, streamed one per `next_row` after the removal. Empty
    /// when the statement has no `RETURNING` clause (the cursor then yields no rows).
    returning: Vec<Row>,
    /// Index of the next buffered `RETURNING` row to emit.
    ret_idx: usize,
}

impl RowCursor for DeleteCursor<'_> {
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

impl DeleteCursor<'_> {
    /// Apply the whole `DELETE` (both phases) and buffer any `RETURNING` rows.
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        // Copy out the borrow-independent context (all `Copy` `'e` references) so the
        // rest of the method only touches `self.pagers` (a shared source view, then the
        // exclusive target pager, never overlapping) and `self.returning`.
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        let n = node.column_count;

        // (1) Resolve the target table. DELETE is WR-aware, so it resolves through
        // `resolve_base_table` (which accepts a WITHOUT ROWID table) and branches below;
        // a missing table still fails closed here. `node.db` is the target namespace
        // (`main` unless a temp/qualified target). `def` borrows through the `Copy`
        // `&dyn Catalog`, so it carries lifetime `'e` and does NOT pin a borrow on `self`
        // — leaving `self.pagers` free to be borrowed below.
        let def: &TableDef = resolve_base_table(catalog, node.db, &node.table)?;

        // Validate the plan's shape against the table before indexing rows by it, so a
        // malformed plan fails closed here rather than mis-slicing the row deep in the
        // loop. This applies to BOTH kinds (the planner sets `column_count = n` for each):
        // a rowid scan produces `[c0..c_{N-1}, rowid]`, a WR scan the width-`N` row.
        if n != def.columns.len() {
            return Err(Error::sql(format!(
                "DELETE column_count {n} does not match table {} column count {}",
                node.table,
                def.columns.len()
            )));
        }

        // A WITHOUT ROWID table stores its rows in the PRIMARY KEY index b-tree (there is
        // no integer rowid to seek and no rowid register in the scan row), so it takes a
        // separate delete path keyed by PRIMARY KEY. Branch AFTER the shape validation above
        // (which applies to both kinds) and BEFORE any rowid-specific machinery below.
        if def.without_rowid {
            return self.apply_without_rowid(rt, def, n);
        }

        let root = def.root_page;
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // The triggers to fire for THIS write (see `trigger::effective_triggers`). A
        // top-level DELETE borrows `node.triggers`; a trigger-ACTION DELETE recompiles the
        // target's own triggers under `PRAGMA recursive_triggers` (default OFF). A
        // trigger-free top-level DELETE yields an empty set, so it is unchanged.
        let effective =
            trigger::effective_triggers(&node.triggers, rt, catalog, node.db, def, TriggerDmlEvent::Delete)?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow of the whole store set — the scan may read
        // another namespace than the target). Drain the entire scan into `buffered`, then
        // release the shared view by closing the scope before the write phase takes the
        // exclusive target pager. Buffering gives a stable delete set even when the scan
        // reads the very index/table about to be mutated.
        let mut buffered: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut scan = build_cursor(&node.scan, env, self.outer)?;
            while let Some(r) = scan.next_row(rt)? {
                buffered.push(r);
            }
        }

        // The database's text encoding (fileformat2 §1.3.13), read ONCE for the whole
        // statement (off-56 is fixed at DB creation and never changes mid-statement) and
        // threaded into every per-row `delete_index_entries` below — so the UTF-8 hot path
        // pays no per-row page-1 header parse. Read under a scoped shared view of the target
        // store; the per-row write phase re-borrows the target mutably below.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);

        // (3) WRITE PHASE (exclusive borrow, one row at a time). `base` is the width of the
        // correlated frame the scan prepends (0 at top level): the deleted table row is the
        // trailing `[c0..c_{N-1}, rowid]` at `row[base..]`.
        let base = self.outer.len();
        let has_returning = !node.returning.is_empty();
        for row in &buffered {
            // The rowid is register `base + N`. A row missing it (or with a non-integer
            // there) is a malformed plan — fail closed rather than panic on the index.
            let rowid = match row.get(base + n) {
                Some(Value::Integer(r)) => *r,
                _ => return Err(Error::sql("DELETE scan row has no integer rowid")),
            };
            // `row.get(base + n)` succeeded, so `row.len() >= base + n + 1` and this slice
            // cannot panic. `logical` is the row's `N` column values (the index-key source).
            let logical = &row[base..base + n];

            // Build the `OLD ++ NEW` trigger frame when this DELETE carries triggers: OLD =
            // the `[cols, rowid]` about to be removed, NEW all-NULL (a DELETE has no new row).
            // Fire BEFORE triggers now, before any index/table removal (lang_createtrigger.html
            // §2). `None` (no triggers) runs exactly as before.
            let frame = if triggers.is_empty() {
                None
            } else {
                let mut old_half = Vec::with_capacity(n + 1);
                old_half.extend_from_slice(logical);
                old_half.push(Value::Integer(rowid));
                Some(trigger::build_frame(n, Some(old_half), None))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row's delete: skip FK
                // enforcement, index/table removal, the AFTER triggers, and RETURNING,
                // leaving the row in place and uncounted.
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

            // The exclusive target pager for this row's base-table removal, RE-borrowed
            // AFTER the BEFORE fire (which took the whole store set for per-action routing)
            // so the two borrows never overlap. The re-borrow addresses the same underlying
            // store, so a same-namespace BEFORE action's writes are visible here.
            let pager = self.pagers.target(node.db)?;

            // FOREIGN KEY (parent side): before removing this row, enforce every incoming
            // FK's ON DELETE action via `foreign_key::enforce_parent_delete`, which handles
            // them all — with `PRAGMA foreign_keys` ON, a NO ACTION / RESTRICT reference by a
            // surviving child aborts the DELETE, CASCADE deletes the referencing children, and
            // SET NULL / SET DEFAULT rewrite their FK columns (to NULL / the column default).
            // Takes `&mut` because those actions write; a no-op when the pragma is OFF.
            foreign_key::enforce_parent_delete(def, logical, catalog, node.db, &mut *pager, rt)?;

            // EVAL PHASE of index maintenance: compute this row's index keys from its
            // `[c0..c_{N-1}, rowid]` frame under a SHARED borrow (an expression key needs the
            // ctx), before the exclusive-borrow removal below. Col-only (today) reads the same
            // columns + rowid the pre-expression delete did.
            let keys = {
                let frame = &row[base..base + n + 1];
                let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                let mut ctx = EvalCtx { rt, env, outer: frame };
                index_keys_for_plans(&index_plans, frame, rowid, &mut ctx)?
            };

            // Index entries FIRST, then the table row (see the module doc): a
            // mid-operation view never has an index entry pointing at a missing row.
            delete_index_entries(&mut *pager, &index_plans, &keys, enc)?;
            let removed = table_delete(&mut *pager, root, rowid)?;
            if removed {
                // Count only rows that were actually present (a `DELETE`'s `changes()`).
                rt.record_change();
            }

            // Fire AFTER triggers now the row is removed (post-write, per row).
            if let Some(frame) = &frame {
                // An AFTER trigger's `RAISE(IGNORE)` cannot un-delete the row (already
                // removed and counted, IGNORE rolls nothing back); it only abandons this
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

            // RETURNING: evaluate each expr over the deleted row `[cols.., rowid]` and buffer
            // it, under the whole-slice shared SOURCE view (`eval_returning`), NOT a
            // single-namespace `Pagers::One` — a RETURNING subquery may read ANY namespace
            // (an ATTACHed db / `temp`; lang_returning.html §3 allows subqueries). RETURNING
            // is read-only and the last per-row step (the AFTER fire's `&mut self.pagers`
            // borrow already closed), so no target pager is reborrowed; `source()` is released
            // when `eval_returning` returns the owned row.
            if has_returning {
                let out = eval_returning(&node.returning, row, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(out);
            }
        }
        Ok(())
    }

    /// Apply a `DELETE` over a WITHOUT ROWID table: each row IS a key record in the
    /// PRIMARY KEY index b-tree (`withoutrowid.html`; fileformat2 §2.4), so a row is
    /// removed by deleting its PRIMARY KEY entry — there is no integer rowid and no table
    /// b-tree. Mirrors the rowid path's phases (buffer the whole scan under a shared borrow,
    /// then remove each row under the exclusive borrow) and the WR INSERT write path
    /// ([`super::insert`]'s `apply_without_rowid`), in the delete direction, via the
    /// [`super::without_rowid`] layout/probe primitives.
    ///
    /// Per row: delete every secondary-index entry (keyed by `[indexed cols.., trailing PK..]`,
    /// fileformat2 §2.5.1) via the shared [`super::dml_index`] WR path, then locate the EXACT
    /// stored key-record bytes for the row's PRIMARY KEY
    /// ([`super::without_rowid::WrLayout::pk_conflict_key`], correct even for a short record
    /// predating a later `ADD COLUMN`) and remove the PRIMARY KEY entry with `index_delete` —
    /// entries before the row, so no intermediate state strands an entry pointing at a removed
    /// row. Count the change only when the entry was actually present, then evaluate `RETURNING`
    /// over the width-`N` row. `last_insert_rowid()` is irrelevant to `DELETE` and left untouched.
    ///
    /// SCOPE (honest fail-closed): an EXPRESSION index on a WR table is refused (its key exprs
    /// bind against a frame with no WR rowid) — that loud error is raised by `build_index_plans`
    /// here — and a WR table with its OWN effective triggers is refused (the `OLD ++ NEW` frame
    /// convention is built around the rowid-table layout, so firing them here would mis-bind
    /// `OLD.rowid`; WR triggers are a documented follow-up).
    ///
    /// Parent-side `FOREIGN KEY` `ON DELETE` actions are enforced exactly as the rowid path
    /// does (a no-op unless `PRAGMA foreign_keys` is ON).
    fn apply_without_rowid(&mut self, rt: &mut Runtime, def: &TableDef, n: usize) -> Result<()> {
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        // The WR b-tree root is stable across the whole statement (see WR INSERT), and the
        // layout maps a schema-order logical row to its PRIMARY KEY values / stored record.
        // `wr_layout` fails closed only on a genuinely PK-less (corrupt) catalog — a
        // well-formed WR table always declares a PRIMARY KEY — so this delete inherits that guard.
        let root = def.root_page;
        let layout = wr_layout(def)?;
        // The WR secondary-index write plans (fileformat2 §2.5.1 key shape). Building them here
        // ALSO raises the loud fail-closed error for an EXPRESSION index on a WR table (see
        // `build_wr_index_key`). Empty for a WR table with no secondary index — the per-row
        // entry deletion below then does nothing.
        let index_plans =
            build_index_plans(catalog, node.db, def, &node.index_key_exprs, &node.index_partial_predicates)?;

        // The triggers to fire for THIS write (mirrors the rowid path). A WR trigger frame is
        // laid at width `W = N` (no rowid register — see `trigger::build_frame_wr`), so a body's
        // `OLD.col_k` binds to `Column(k)`. The EFFECTIVE set (not bare `node.triggers`) honors
        // `recursive_triggers` for a nested trigger-action DELETE into a WR table; an empty set
        // fires nothing (no frame built). The correlated `outer` frame is threaded through the
        // scan below.
        let effective =
            trigger::effective_triggers(&node.triggers, rt, catalog, node.db, def, TriggerDmlEvent::Delete)?;
        let triggers = effective.as_slice();

        // (2) READ PHASE (shared borrow of the whole store set — the scan may read another
        // namespace than the target). Drain the ENTIRE scan into `buffered` before any
        // removal: a WR scan iterates the very PRIMARY KEY b-tree the deletes then mutate, so
        // buffering first gives a stable delete set and never reads a tree it is mutating (the
        // same two-phase reason as the rowid path). The buffer is bounded by the rows the scan
        // selects; nothing larger is materialized.
        let mut buffered: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut scan = build_cursor(&node.scan, env, self.outer)?;
            while let Some(r) = scan.next_row(rt)? {
                buffered.push(r);
            }
        }

        // The database's text encoding (fileformat2 §1.3.13), read ONCE for the whole
        // statement (off-56 is fixed at DB creation) and threaded into every per-row PK probe
        // below — so the PK key bytes the probe encodes line up with the same-encoding stored
        // WR key records. Read under a scoped shared view of the target store; the per-row
        // write phase re-borrows the target mutably below.
        let enc = text_encoding_of(self.pagers.source().get(node.db)?);

        // (3) WRITE PHASE (exclusive borrow, one row at a time). `base` is the width of the
        // correlated frame the scan prepends (0 at top level): the WR row is the trailing
        // width-`N` slice `row[base..base + n]` — `[c0..c_{N-1}]`, NO rowid register.
        let base = self.outer.len();
        let has_returning = !node.returning.is_empty();
        for row in &buffered {
            // The width-`N` logical row (the index-key source). A row narrower than
            // `base + n` is a malformed plan — fail closed rather than panic on the slice.
            let logical = row.get(base..base + n).ok_or_else(|| {
                Error::sql("DELETE WITHOUT ROWID scan row narrower than the table column count")
            })?;

            // Build the `OLD ++ NEW` WR frame when this DELETE carries triggers: OLD = the
            // width-N `[c0..c_{N-1}]` about to be removed (NO rowid register — `build_frame_wr`
            // lays each half at width N), NEW all-NULL (a DELETE has no new row). Fire BEFORE
            // now, before FK enforcement / removal (lang_createtrigger.html §2). `None` (no
            // triggers) runs exactly as before.
            let frame = if triggers.is_empty() {
                None
            } else {
                Some(trigger::build_frame_wr(n, Some(logical.to_vec()), None))
            };
            if let Some(frame) = &frame {
                // A BEFORE trigger's `RAISE(IGNORE)` abandons this row's delete: skip FK
                // enforcement, removal, the AFTER fire, and RETURNING, leaving the row in place
                // and uncounted.
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

            // The exclusive target pager for this row's PK-b-tree removal, RE-borrowed AFTER
            // the BEFORE fire (which took the whole store set) so the borrows never overlap.
            let pager = self.pagers.target(node.db)?;

            // FOREIGN KEY (parent side): before removing this row, enforce every incoming
            // FK's ON DELETE action, exactly as the rowid path does — with `PRAGMA
            // foreign_keys` ON a NO ACTION/RESTRICT reference by a surviving child aborts,
            // CASCADE deletes the referencing children, SET NULL / SET DEFAULT rewrite them.
            // The parent side only reads `logical[i]` for the FK-referenced columns, so it is
            // WR-safe (a WR parent's b-tree is never touched here). A no-op when the pragma is
            // OFF (the default), so a WR delete with FK off is byte-for-byte the algorithm below.
            foreign_key::enforce_parent_delete(def, logical, catalog, node.db, &mut *pager, rt)?;

            // Locate the row's exact stored key-record bytes by its PRIMARY KEY, then remove
            // that entry. `pk_conflict_key` returns the precise on-disk bytes (correct even
            // for a short record written before a later `ADD COLUMN`), so `index_delete`
            // removes the real entry rather than a re-encoded guess. A scanned row is always
            // present in practice (buffered from this same PK b-tree, and only distinct keys
            // are removed), so `None` is a defensive guard — a row already gone is NOT counted
            // as a change we did not make, never fabricated.
            let pk = layout.pk_values(logical);
            if let Some(key) = layout.pk_conflict_key(&*pager, root, &pk, enc)? {
                // Delete every secondary-index entry FIRST (keyed by this row's `[indexed
                // cols.., trailing PK..]`), then the PRIMARY KEY record — so no intermediate
                // state leaves a secondary entry pointing at a removed row. `logical` is the
                // snapshot row; a WR DELETE never rewrites the row it is about to remove (the
                // parent-side FK cascade above touches CHILDREN, not this row), so its indexed
                // values still name the stored entries. Empty `index_plans` is a no-op. A
                // scoped eval ctx (shared pager + rt) computes the keys — gating each PARTIAL
                // index on its predicate over `logical` — then drops before the mutable delete.
                let sec_keys = {
                    let env = Env { catalog, pagers: Pagers::One { db: node.db, pager: &*pager }, plan };
                    let mut ctx = EvalCtx { rt, env, outer: logical };
                    wr_index_keys_for_plans(&index_plans, logical, &mut ctx)?
                };
                delete_index_entries(&mut *pager, &index_plans, &sec_keys, enc)?;
                index_delete_with_order(&mut *pager, root, &key, layout.order())?;
                rt.record_change();

                // Fire AFTER now the row is removed (post-write). The per-row `pager` was last
                // used by `index_delete`, so its exclusive borrow has ended (NLL). An AFTER
                // `RAISE(IGNORE)` cannot un-delete the row; it only abandons this row's
                // RETURNING.
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

                // RETURNING: evaluate each expr over the width-`N` deleted row (a WR row has
                // NO trailing rowid register — the row RETURNING binds against is exactly
                // `[c0..c_{N-1}]`), under the whole-slice shared SOURCE view so a RETURNING
                // subquery may read ANY namespace (lang_returning.html §3), matching every
                // other DML RETURNING site. The per-row `pager` above is unused past
                // `index_delete`, so its exclusive `&mut self.pagers` borrow has ended (NLL)
                // and `source()` is free. Evaluated INSIDE the `Some` arm so the rows RETURNING
                // reports are EXACTLY the rows removed (and counted) — the two can never
                // diverge, even if a future WR-FK cascade could vacate a buffered row before
                // its own delete (a `None` probe then yields no phantom RETURNING row).
                if has_returning {
                    let out =
                        eval_returning(&node.returning, logical, catalog, self.pagers.source(), plan, rt)?;
                    self.returning.push(out);
                }
            }
        }
        Ok(())
    }
}
