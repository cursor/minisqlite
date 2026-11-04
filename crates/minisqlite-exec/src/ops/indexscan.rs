//! Index access path: [`index_scan`] positions on an index b-tree and iterates the
//! matching entries in key order. For each entry it recovers the rowid from the index
//! key and, UNLESS the scan is covering, fetches the base-table row by that rowid; a
//! COVERING scan ([`IndexScan::covering`]) instead builds the row straight from the index
//! entry, skipping the fetch (optoverview.html §"Covering Indexes"). Like every table leaf
//! it emits `[c0, …, c_{N-1}, rowid]` (width `N+1`) and prepends the `outer` row when
//! non-empty — the covering and fetch paths produce the byte-identical shape.
//!
//! ## WITHOUT ROWID secondary index (fetch-by-PK)
//! A secondary index on a WITHOUT ROWID table keys each entry by
//! `[indexed cols.., trailing PK..]` (fileformat2 §2.5.1) — there is NO trailing rowid. So
//! the WR read recovers the full PRIMARY KEY from the entry (via the shared
//! [`super::dml_index`] `WrIndexKey` layout, so read and write agree) and fetches the row
//! from the WR PRIMARY KEY b-tree (`table.root_page`, itself an index b-tree) by that PK,
//! emitting the width-`N` row a WR `SeqScan` emits (no trailing rowid register). The
//! equality-prefix and range-bound classification is UNCHANGED — the indexed columns still
//! lead every entry, and the trailing PK sits past the indexed-column count the bounds apply
//! to. Covering is disabled for WR (its emit shape is width-`N`, not the rowid `N+1`).
//!
//! A [`IndexOp::Seek`] is a real b-tree SEEK, not a full index scan: the cursor is
//! positioned with `seek_ge` / `seek_le` and the walk STOPS the moment the equality
//! prefix no longer matches (the index is ordered, so no later entry can match). Only
//! the entries the op actually selects are visited — cost is proportional to the
//! matches, not the index size.
//!
//! ## Ordering / collation invariant (load-bearing)
//! The index b-tree orders keys under the index's [`KeyOrder`]: each key column's
//! collation and `DESC` flag (see `minisqlite_btree::index_key`). The cursor seeks with
//! that order, and every prefix/bound comparison here goes through the same
//! [`KeyOrder::compare_column`]: the "stop once the prefix stops matching" and "stop once
//! the bound is crossed" decisions are only sound if this comparator agrees with the order
//! the cursor seeks in. The planner only seeks a column whose comparison collation is the
//! index column's own collation, so comparing under the index collation is the query's
//! comparison too.
//!
//! Everything below therefore works in INDEX order, not value order. On a `DESC` column
//! the two differ: a value lower bound (`b > 5`) is where the scan ENDS in index order.
//! [`IndexScanCursor::prepare`] swaps the bounds of a `DESC` range column once, so `low` /
//! `high` always mean the index-order start / end, and the positioning and classification
//! code is the same for both sort orders. NULLs sort last in a `DESC` column, so a value
//! lower bound (now the index-order end) still stops the scan before them, and the
//! planner's NULL rule (keep an upper-only bound in the residual) holds unchanged.
//!
//! ## Reverse positioning (the group-top problem)
//! Forward is straightforward: `seek_ge` on the low side lands at the first matching
//! entry and `next()` walks up. Reverse is subtler because
//! `IndexCursor::seek_le(prefix)` lands strictly BELOW the prefix group (a shorter
//! record sorts `Less` than any full key sharing it), and there is no universal
//! maximum `Value` to seek to a group's TOP. So for the two cases where the reverse
//! start is the top of a value group — an *inclusive* upper bound whose value is
//! actually present, and an equality prefix with *no* upper bound — the start is found
//! by walking forward across that one boundary group (bounded by matching rows; it
//! never scans non-matching prefixes). See [`IndexScanCursor::position_reverse_start`].

use std::cmp::Ordering;

use minisqlite_btree::{IndexCursor, KeyOrder, TableCursor};
use minisqlite_catalog::TableDef;
use minisqlite_expr::eval;
use minisqlite_fileformat::{decode_record_enc, encode_record_enc, TextEncoding};
use minisqlite_pager::text_encoding_of;
use minisqlite_plan::{col_reg, GeneratedProgram, IndexOp, IndexScan, ScanDirection};
use minisqlite_types::{compare_values, Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::dml_index::{build_index_plan, index_key_fields, wr_entry_pk, IndexPlan};
use crate::ops::generated::compute_generated;
use crate::ops::with_outer;
use crate::ops::without_rowid::{wr_layout, WrLayout};
use crate::row::{decode_table_row_enc, decode_table_row_skipping_virtual_enc, resolve_base_table};
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build an index access path over `node.index` for base table `node.table`.
///
/// Opens the index cursor (positioned lazily on the first pull, since the seek key
/// may reference `outer`) and a table cursor reused for each by-rowid fetch. The
/// emitted row width is `node.column_count + 1`. For a NON-covering scan each row is
/// produced by the by-rowid table fetch through [`decode_table_row_enc`]; for a covering
/// scan ([`IndexScan::covering`]) it is built directly from the index entry (see
/// [`IndexScanCursor::emit_current`]), skipping the fetch.
pub(crate) fn index_scan<'a>(
    node: &'a IndexScan,
    env: Env<'a>,
    outer: &'a [Value],
) -> Result<Box<dyn RowCursor + 'a>> {
    let idx = env
        .catalog
        .index_in(node.db, &node.index)?
        .ok_or_else(|| Error::sql(format!("no such index: {}", node.index)))?;
    // WR-aware: `resolve_base_table` accepts a WITHOUT ROWID table. A WR secondary-index
    // scan fetches its rows from the PRIMARY KEY b-tree (below), so it must resolve here.
    let table: &'a TableDef = resolve_base_table(env.catalog, node.db, &node.table)?;

    // Reject malformed plans up front (a planner bug) rather than silently returning
    // wrong or no rows.
    if let IndexOp::Seek { eq_prefix, low, high } = &node.op {
        // An equality prefix wider than the index cannot be satisfied.
        if eq_prefix.len() > idx.columns.len() {
            return Err(Error::sql(format!(
                "index seek prefix ({}) exceeds index {} column count ({})",
                eq_prefix.len(),
                node.index,
                idx.columns.len()
            )));
        }
        // A range bound constrains the column immediately AFTER the equality prefix. If
        // the prefix already pins every indexed column there is no such column, so a
        // bound here is a contradictory plan. Fail closed rather than let `classify`'s
        // `p < key_columns` guard silently ignore it.
        if eq_prefix.len() == idx.columns.len() && (low.is_some() || high.is_some()) {
            return Err(Error::sql(format!(
                "index seek on {} carries a range bound but its equality prefix already \
                 covers all {} indexed column(s) (no column left to bound)",
                node.index,
                idx.columns.len()
            )));
        }
    }

    let pager = env.pagers.get(node.db)?;
    // Opened for the rowid by-rowid fetch. For a WR table `table.root_page` is an index
    // b-tree (the PK store), not a table b-tree, so this cursor is UNUSED on the WR path
    // (which fetches through `wr.pk_cursor` below); `TableCursor::open` does no I/O, so
    // constructing it unconditionally is free and keeps the rowid path unchanged.
    let table_cursor = TableCursor::open(pager, table.root_page)?;
    let generated = env.plan.generated_programs(node.db, &node.table);
    let has_virtual = generated.iter().any(|p| p.is_virtual());
    let enc = text_encoding_of(pager);

    // A WITHOUT ROWID table's secondary index appends the trailing PRIMARY KEY instead of a
    // rowid (fileformat2 §2.5.1), so its read recovers that PK from each entry and fetches
    // the row from the PK b-tree. Build the shared [`IndexPlan`] (its `wr` layout is the ONE
    // encoder read/write/probe share, so the PK recovered here matches what the writer
    // stored), the `WrLayout` (to decode the fetched PK record), and a second cursor over the
    // PK b-tree. `build_index_plan` fail-closes an EXPRESSION index on a WR table. `None` for
    // a rowid table (the common case), which keeps the byte-identical by-rowid path.
    //
    // The `None` partial predicate is correct for a READ: a plan's `partial_predicate` is
    // consulted ONLY by `index_admits_row` during index MAINTENANCE (INSERT/UPDATE/DELETE key
    // generation), never on a scan — a read walks whatever entries the b-tree already holds
    // (a partial index physically contains only its matching rows). The planner also declines
    // a partial index for reads, so `idx` here is a full index regardless.
    let wr = if table.without_rowid {
        let layout = wr_layout(table)?;
        let pk_cursor =
            IndexCursor::open_with_order(pager, table.root_page, layout.order().clone())?;
        let plan = build_index_plan(table, idx, &[], None)?;
        Some(WrRead { plan, layout, pk_cursor, scratch: Vec::new() })
    } else {
        None
    };
    // The index's key order (per-column collation and `DESC`): the WR plan's full order
    // (indexed columns + trailing PK), else the indexed columns followed by the ascending
    // BINARY rowid. Built from the catalog definition, so an expression index needs no
    // compiled key expressions here.
    let order = match &wr {
        Some(w) => w.plan.order.clone(),
        None => KeyOrder::new(index_key_fields(table, idx)),
    };
    let index_cursor = IndexCursor::open_with_order(pager, idx.root_page, order.clone())?;

    // Covering (index-only) read: engage ONLY when the planner marked this scan covering
    // AND the table has no generated column AND it is a rowid table. The planner already
    // declines covering for any generated-column table (a VIRTUAL column is never stored, so
    // an index-only read cannot supply it); the `generated.is_empty()` guard here is a
    // defensive second gate so a planner miss falls back to the correct by-rowid fetch rather
    // than emitting NULLs. A WR scan is never covering here — its emit shape is the width-`N`
    // WR row, not the rowid `[cols.., rowid]` the covering path builds.
    let covering = node.covering && generated.is_empty() && wr.is_none();
    // The index-entry-position -> row-register map for the covered columns, computed ONCE
    // via the SAME `col_reg` the planner's covering decision used — so the executor fills
    // exactly the registers the planner proved the query reads. An expression key column
    // (no named table column) is skipped; a key column that is the INTEGER PRIMARY KEY alias
    // maps to the rowid register `N`, which `emit_current` fills from the entry's trailing
    // rowid, so it is skipped here to keep that one authoritative write.
    let covering_map = if covering {
        build_covering_map(idx, table)
    } else {
        Vec::new()
    };

    Ok(Box::new(IndexScanCursor {
        env,
        table,
        index_cursor,
        table_cursor,
        op: &node.op,
        direction: &node.direction,
        key_columns: idx.columns.len(),
        order,
        outer,
        phase: Phase::Init,
        eq_vals: Vec::new(),
        low: None,
        high: None,
        generated,
        has_virtual,
        enc,
        covering,
        covering_map,
        wr,
    }))
}

/// The WITHOUT ROWID fetch-by-PK state for a WR secondary-index scan (see the module doc).
/// Held on [`IndexScanCursor`] only for a WR table; a rowid scan carries `None` and never
/// touches it.
struct WrRead<'a> {
    /// The index's shared write plan — its `wr` field ([`super::dml_index::WrIndexKey`]) is
    /// the ONE encoder that recovers the full PRIMARY KEY from a decoded entry, so the read
    /// resolves the same PK bytes the writer stored.
    plan: IndexPlan,
    /// The WR table's storage layout, to decode a fetched PK-record into a schema-order row.
    layout: WrLayout,
    /// A cursor over the WR PRIMARY KEY b-tree (`table.root_page`), re-seeked per emitted
    /// entry to fetch that entry's row by its PRIMARY KEY.
    pk_cursor: IndexCursor<'a>,
    /// Reused decode scratch (the WR decode drains it each call), so a scan does not churn a
    /// per-row buffer.
    scratch: Vec<Value>,
}

/// Map each covered index key column to the row register it fills, for the covering read.
///
/// An index entry is `[key_col_0, .., key_col_{m-1}, rowid]`. For each ORDINARY (non-
/// expression) key column at position `k`, `col_reg` resolves its table column to a row
/// register `reg`; the covering read then copies `entry[k]` into `local[reg]`. Skipped:
/// * an EXPRESSION key column (`key_exprs[k] == Some`) — a computed value with no named
///   table column to fill (the planner likewise does not treat it as covering a column);
/// * a key column whose `col_reg` is the rowid register `N` (an INTEGER PRIMARY KEY alias)
///   — `emit_current` writes register `N` from the entry's trailing rowid, so mapping it
///   here would be a redundant (and order-sensitive) double write.
///
/// This is the executor's half of the covering contract; the planner's `covered` set is
/// built from the same `col_reg`, so the registers filled here are exactly the ones the
/// planner proved the query reads.
fn build_covering_map(idx: &minisqlite_catalog::IndexDef, table: &TableDef) -> Vec<(usize, usize)> {
    let n = table.columns.len();
    let mut map = Vec::with_capacity(idx.columns.len());
    for (k, name) in idx.columns.iter().enumerate() {
        if idx.key_exprs.get(k).is_some_and(|e| e.is_some()) {
            continue;
        }
        if let Some(reg) = col_reg(table, name, n) {
            if reg != n {
                map.push((k, reg));
            }
        }
    }
    map
}

/// The scan's progress. `Copy` so `match self.phase` reads it out without holding a
/// borrow of `self` across the cursor moves in each arm.
#[derive(Clone, Copy)]
enum Phase {
    /// Not started: on the first pull, evaluate the bound expressions and position the
    /// cursor at the scan's start.
    Init,
    /// Positioned: the cursor sits on a candidate entry to classify (the freshly
    /// positioned one right after `Init`, or the one reached by the last `step`).
    Active,
    /// Exhausted.
    Done,
}

/// How the current index entry relates to the requested range.
enum Class {
    /// Out of range on the far side (or the prefix diverged): the ordered scan is
    /// finished.
    Stop,
    /// Out of range on the near side (the exclusive-bound group we seeked into, or an
    /// overshoot): skip it and keep stepping.
    Skip,
    /// In range: fetch the table row and emit it.
    Emit,
}

/// A streaming index access path. Holds one index cursor (its descent path) and one
/// reused table cursor — never a materialized set of entries or rows.
struct IndexScanCursor<'a> {
    env: Env<'a>,
    table: &'a TableDef,
    index_cursor: IndexCursor<'a>,
    table_cursor: TableCursor<'a>,
    op: &'a IndexOp,
    direction: &'a ScanDirection,
    /// The number of indexed columns `k` (an index key record is `[c0..c_{k-1},
    /// rowid]`), so a range bound applies only when `eq_prefix.len() < k`.
    key_columns: usize,
    /// The index's key order — the one the cursor seeks with. Every prefix / bound test
    /// compares through it (see the module doc's ordering invariant).
    order: KeyOrder,
    outer: &'a [Value],
    phase: Phase,
    /// The evaluated equality-prefix values (`eq_prefix`), computed once on `Init`.
    eq_vals: Vec<Value>,
    /// The evaluated lower bound `(value, inclusive)` on the column after the prefix, in
    /// INDEX order (for a `DESC` column this is the query's upper bound, see `prepare`).
    low: Option<(Value, bool)>,
    /// The evaluated upper bound `(value, inclusive)` on the column after the prefix, in
    /// INDEX order (for a `DESC` column this is the query's lower bound).
    high: Option<(Value, bool)>,
    /// The base table's generated-column programs (STORED + VIRTUAL, column order); empty
    /// for a non-generated table. The read path computes only the VIRTUAL subset after the
    /// by-rowid table fetch.
    generated: &'a [GeneratedProgram],
    /// Whether any program is VIRTUAL — selects the virtual-aware decode + on-read compute.
    has_virtual: bool,
    /// The database's TEXT encoding (§1.3.13), threaded into both the index-key decode and
    /// the by-rowid base-row decode so a UTF-16 file's TEXT transcodes to UTF-8. NOTE: a
    /// point/range SEEK on a TEXT-typed index column in a UTF-16 database is a documented
    /// deferred gap — the b-tree seek compares the (UTF-8) search-key record against the
    /// stored UTF-16 key bytes directly (`minisqlite_btree::index_key`), so positioning is
    /// not reliable there. Numeric-keyed seeks and the emitted rows decode correctly.
    enc: TextEncoding,
    /// `true` for a covering (index-only) read: `emit_current` builds the emitted row from
    /// the index entry and SKIPS the by-rowid table fetch. Set from [`IndexScan::covering`]
    /// gated by `generated.is_empty()` (see [`index_scan`]).
    covering: bool,
    /// For a covering read, the `(index_entry_position, row_register)` pairs to copy from
    /// each index entry into the emitted row (see [`build_covering_map`]). Empty when not
    /// covering. The rowid register is filled separately from the entry's trailing rowid.
    covering_map: Vec<(usize, usize)>,
    /// `Some` for a WITHOUT ROWID secondary-index scan: the fetch-by-PK state (see
    /// [`WrRead`]). When set, [`Self::emit_current`] recovers the PRIMARY KEY from the entry
    /// and fetches the width-`N` row from the PK b-tree instead of the by-rowid fetch.
    wr: Option<WrRead<'a>>,
}

impl RowCursor for IndexScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        loop {
            match self.phase {
                Phase::Done => return Ok(None),
                Phase::Init => {
                    self.prepare(rt)?;
                    if !self.position_start()? {
                        self.phase = Phase::Done;
                        return Ok(None);
                    }
                    // Classify the freshly positioned entry below WITHOUT stepping.
                    self.phase = Phase::Active;
                }
                Phase::Active => {
                    if !self.step()? {
                        self.phase = Phase::Done;
                        return Ok(None);
                    }
                }
            }
            let entry = self.current_key()?;
            match self.classify(&entry) {
                Class::Stop => {
                    self.phase = Phase::Done;
                    return Ok(None);
                }
                Class::Skip => continue,
                Class::Emit => return self.emit_current(&entry, rt).map(Some),
            }
        }
    }
}

impl IndexScanCursor<'_> {
    /// Evaluate the equality-prefix and bound expressions once, at the start of the
    /// scan. They may reference `outer` (a correlated / index-nested-loop drive), so
    /// evaluation needs the runtime and cannot happen at build time. A `FullScan`
    /// leaves the defaults (empty prefix, no bounds), which the rest of the code treats
    /// exactly like a `Seek` with no constraints.
    ///
    /// The plan's bounds are in VALUE order. On a `DESC` range column they are swapped
    /// here into INDEX order (the value upper bound is where the walk starts), so the rest
    /// of the cursor never looks at the sort order again.
    fn prepare(&mut self, rt: &mut Runtime) -> Result<()> {
        let op = self.op;
        if let IndexOp::Seek { eq_prefix, low, high } = op {
            let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
            let mut eq = Vec::with_capacity(eq_prefix.len());
            for e in eq_prefix {
                eq.push(eval(e, self.outer, &mut ctx)?);
            }
            let low = match low {
                Some(b) => Some((eval(&b.value, self.outer, &mut ctx)?, b.inclusive)),
                None => None,
            };
            let high = match high {
                Some(b) => Some((eval(&b.value, self.outer, &mut ctx)?, b.inclusive)),
                None => None,
            };
            let p = eq.len();
            self.eq_vals = eq;
            if p < self.key_columns && self.order.field(p).descending {
                self.low = high;
                self.high = low;
            } else {
                self.low = low;
                self.high = high;
            }
        }
        Ok(())
    }

    /// Position the cursor at the scan's starting entry; `Ok(false)` if no entry can be
    /// in range (leaving the cursor unpositioned).
    fn position_start(&mut self) -> Result<bool> {
        match self.direction {
            ScanDirection::Forward => self.position_forward_start(),
            ScanDirection::Reverse => self.position_reverse_start(),
        }
    }

    /// Forward start: seek to `eq_vals ++ [low]` (or `eq_vals`, or the whole index).
    /// A prefix key sorts before every full key sharing it, so `seek_ge` lands on the
    /// first entry whose prefix is `>= eq_vals` and (if a low bound is given) whose
    /// next column is `>= low` — exactly the first candidate. An exclusive low is
    /// handled by [`Self::classify`] skipping the `== low` group the seek landed in.
    fn position_forward_start(&mut self) -> Result<bool> {
        let mut seek_vals = self.eq_vals.clone();
        if let Some((lv, _incl)) = &self.low {
            seek_vals.push(lv.clone());
        }
        if seek_vals.is_empty() {
            self.index_cursor.first()
        } else {
            self.index_cursor.seek_ge(&encode_record_enc(&seek_vals, self.enc))
        }
    }

    /// Reverse start: position at the GREATEST in-range entry, then the caller walks
    /// down with `prev`. See the module doc for why the top of a value group cannot
    /// always be reached with a single seek; the `walk_to_top` fallback handles the two
    /// cases that need it.
    fn position_reverse_start(&mut self) -> Result<bool> {
        match &self.high {
            Some((hv, inclusive)) => {
                let mut hk = self.eq_vals.clone();
                hk.push(hv.clone());
                let hk = encode_record_enc(&hk, self.enc);
                if *inclusive {
                    // Want the greatest entry with prefix == eq and col_p <= high.
                    if !self.index_cursor.seek_ge(&hk)? {
                        // Nothing is >= [eq, high]; every entry sorts below it, so the
                        // greatest in-range entry (if any) is the global maximum.
                        return self.index_cursor.last();
                    }
                    let entry = self.current_key()?;
                    if self.prefix_matches(&entry)
                        && self.col_after_prefix_cmp(&entry, hv) == Ordering::Equal
                    {
                        // Landed on the BOTTOM of the `col_p == high` group; its top is
                        // the reverse start.
                        self.walk_to_top(Some(hv.clone()))
                    } else {
                        // The landing is strictly above the inclusive range, so the
                        // greatest in-range entry is the one just below it.
                        self.index_cursor.prev()
                    }
                } else {
                    // Exclusive high: `seek_le([eq, high])` lands on the greatest entry
                    // strictly below the `[eq, high]` prefix, i.e. the greatest with
                    // col_p < high (or a smaller prefix) — exactly the reverse start.
                    self.index_cursor.seek_le(&hk)
                }
            }
            None => {
                if self.eq_vals.is_empty() {
                    // Pure range with no upper bound, or a full scan: start at the max.
                    self.index_cursor.last()
                } else {
                    // Equality prefix, no upper bound: the reverse start is the top of
                    // the prefix group. `seek_ge(eq)` lands on its bottom.
                    if !self.index_cursor.seek_ge(&encode_record_enc(&self.eq_vals, self.enc))? {
                        return Ok(false);
                    }
                    let entry = self.current_key()?;
                    if self.prefix_matches(&entry) {
                        self.walk_to_top(None)
                    } else {
                        // The first entry >= eq has a greater prefix: no prefix group.
                        Ok(false)
                    }
                }
            }
        }
    }

    /// From the BOTTOM of a value group the cursor is currently on, reposition to the
    /// group's TOP and return `Ok(true)`. The group is "prefix matches `eq_vals`" and,
    /// when `pin` is `Some(v)`, additionally "the column after the prefix `== v`".
    ///
    /// Stepping forward off the group's end lands on either an out-of-group entry
    /// (`prev` back one to the top) or the end of the index (`last` is the top, since
    /// the group ran to the final entry). Bounded by the group size.
    fn walk_to_top(&mut self, pin: Option<Value>) -> Result<bool> {
        loop {
            if !self.index_cursor.next()? {
                // Ran off the end: the group's top is the index's last entry.
                return self.index_cursor.last();
            }
            let entry = self.current_key()?;
            let in_group = self.prefix_matches(&entry)
                && match &pin {
                    Some(v) => self.col_after_prefix_cmp(&entry, v) == Ordering::Equal,
                    None => true,
                };
            if !in_group {
                // Stepped out: the previous entry was the group's top.
                return self.index_cursor.prev();
            }
        }
    }

    /// Advance one entry in the scan direction.
    fn step(&mut self) -> Result<bool> {
        match self.direction {
            ScanDirection::Forward => self.index_cursor.next(),
            ScanDirection::Reverse => self.index_cursor.prev(),
        }
    }

    /// Decode the current index entry's full key record (`[c0..c_{k-1}, rowid]`). TEXT
    /// columns decode in the database's text encoding (§1.3.13); the trailing rowid is an
    /// integer regardless of encoding.
    fn current_key(&self) -> Result<Vec<Value>> {
        let bytes = self.index_cursor.key()?;
        Ok(decode_record_enc(&bytes, self.enc))
    }

    /// Classify the current entry against the equality prefix and range bounds. The
    /// prefix decision is symmetric (a mismatch ends the scan in either direction,
    /// since the index is ordered); the bound decision is direction-dependent — the far
    /// bound stops the scan, the near bound skips the boundary group.
    fn classify(&self, entry: &[Value]) -> Class {
        let p = self.eq_vals.len();
        for i in 0..p {
            match entry.get(i) {
                // This prefix column still matches: keep checking the rest of the prefix.
                Some(v) if self.order.compare_column(i, v, &self.eq_vals[i]) == Ordering::Equal => {}
                // The prefix DIVERGED at this column. The index is ordered and we are
                // walking away from the near end of the prefix group, so no later entry
                // in this direction can match — the ordered scan is finished.
                Some(_) => return Class::Stop,
                // Entry shorter than the equality prefix. Unreachable via a seek:
                // `seek_ge` / `seek_le` position PAST any entry with fewer columns than
                // the `p`-column seek key (a shorter record sorts below it), so the walk
                // never lands on one; a truly diverged first column is caught by the
                // `Some(_)` arm above before we index this far. Treated as end-of-scan.
                None => return Class::Stop,
            }
        }
        // A bound constrains the column right after the prefix, and only when that
        // column is a real indexed column (`p < k`, not the trailing rowid).
        if p < self.key_columns {
            if let Some(col) = entry.get(p) {
                if let Some((hv, incl)) = &self.high {
                    let c = self.order.compare_column(p, col, hv);
                    if c == Ordering::Greater || (c == Ordering::Equal && !incl) {
                        return match self.direction {
                            ScanDirection::Forward => Class::Stop,
                            ScanDirection::Reverse => Class::Skip,
                        };
                    }
                }
                if let Some((lv, incl)) = &self.low {
                    let c = self.order.compare_column(p, col, lv);
                    if c == Ordering::Less || (c == Ordering::Equal && !incl) {
                        return match self.direction {
                            ScanDirection::Forward => Class::Skip,
                            ScanDirection::Reverse => Class::Stop,
                        };
                    }
                }
            }
        }
        Class::Emit
    }

    /// Whether the entry's leading columns equal `eq_vals` (under the index collations).
    fn prefix_matches(&self, entry: &[Value]) -> bool {
        let p = self.eq_vals.len();
        if entry.len() < p {
            return false;
        }
        (0..p).all(|i| self.order.compare_column(i, &entry[i], &self.eq_vals[i]) == Ordering::Equal)
    }

    /// Compare the entry's column right after the equality prefix against `v`, in index
    /// order. A short
    /// entry missing that column sorts `Less` (never `Equal`), so it is not treated as
    /// in a value group.
    fn col_after_prefix_cmp(&self, entry: &[Value], v: &Value) -> Ordering {
        match entry.get(self.eq_vals.len()) {
            Some(col) => self.order.compare_column(self.eq_vals.len(), col, v),
            None => Ordering::Less,
        }
    }

    /// Build the emitted `[c0..c_{N-1}, rowid]` row (width `N+1`) for the current index
    /// entry. The rowid is the last value of the decoded index key.
    ///
    /// * COVERING read: the index already carries every column the query needs, so the row
    ///   is assembled straight from the index `entry` — no table fetch. This is byte-for-byte
    ///   identical to the fetch path for the columns the query reads, because an index key
    ///   column stores the SAME `Value` (under the same storage class, via the same
    ///   `decode_record_enc`) the table row's column decodes to (see `ops::dml_index`). The
    ///   non-covered registers stay `Value::Null`; the planner proved they are never read.
    /// * NON-COVERING read: seek the table b-tree by rowid and decode the full row — the
    ///   correct path for every index and the only one for a table with generated columns.
    fn emit_current(&mut self, entry: &[Value], rt: &mut Runtime) -> Result<Row> {
        // WITHOUT ROWID: the entry carries the trailing PRIMARY KEY, not a rowid, so fetch
        // the width-`N` row from the PK b-tree by that PK (see [`Self::emit_current_wr`]).
        if self.wr.is_some() {
            return self.emit_current_wr(entry, rt);
        }
        let rowid = match entry.last() {
            Some(Value::Integer(r)) => *r,
            other => {
                return Err(Error::format(format!(
                    "index entry has no integer rowid: {other:?}"
                )))
            }
        };
        if self.covering {
            // Index-only read: fill the covered columns from the entry, the rowid from the
            // entry's trailing value, and leave every other register NULL. Width is
            // `N + 1` — the SAME shape the by-rowid decode below produces.
            let n = self.table.columns.len();
            let mut local = vec![Value::Null; n + 1];
            for &(k, reg) in &self.covering_map {
                // `k` indexes the key columns (before the trailing rowid) and `reg <= n` by
                // construction; guard both so a malformed entry/map degrades to a NULL slot
                // rather than panicking mid-scan.
                if let (Some(v), true) = (entry.get(k), reg < local.len()) {
                    local[reg] = v.clone();
                }
            }
            local[n] = Value::Integer(rowid);
            return Ok(with_outer(self.outer, local));
        }
        if !self.table_cursor.seek_exact(rowid)? {
            // A dangling index entry (index points at a rowid with no table row) is
            // corruption / an index-maintenance bug — fail loud rather than skip.
            return Err(Error::format(format!(
                "index entry points to missing table rowid {rowid}"
            )));
        }
        // The base table record never stores a VIRTUAL generated column (even one this index
        // keys on — its value lives in the index entry, not the table row), so the table
        // fetch decodes it as a NULL placeholder and the on-read compute fills it. A
        // non-generated table takes the byte-identical fast decode and skips the compute.
        let mut local = {
            let payload = self.table_cursor.payload()?;
            if self.has_virtual {
                decode_table_row_skipping_virtual_enc(&payload, rowid, self.table, self.enc)
            } else {
                decode_table_row_enc(&payload, rowid, self.table, self.enc)
            }
        };
        if self.has_virtual {
            compute_generated(self.generated, true, &mut local, self.env, rt)?;
        }
        Ok(with_outer(self.outer, local))
    }

    /// Build the emitted width-`N` WITHOUT ROWID row `[c0..c_{N-1}]` (no trailing rowid) for
    /// the current WR index entry: recover its full PRIMARY KEY (fileformat2 §2.5.1, via the
    /// shared `WrIndexKey` layout so read and write agree), seek the PK b-tree, and decode.
    ///
    /// The seek lands on the first key `>= pk`; a WR index entry must reference a live PK row,
    /// so a landing whose PRIMARY KEY is NOT exactly `pk` means the referenced row is gone — a
    /// corrupt/unmaintained index, surfaced loud rather than emitting a wrong or phantom row
    /// (the rowid path's dangling-entry guard, in the WR key space). VIRTUAL generated columns
    /// are computed on read exactly as the by-rowid path does.
    fn emit_current_wr(&mut self, entry: &[Value], rt: &mut Runtime) -> Result<Row> {
        // Copy out the `Copy` context so the `&mut self.wr` borrow below does not clash with
        // the shared field reads (`table`/`enc`/`env`/… are all cheap references).
        let enc = self.enc;
        let table = self.table;
        let has_virtual = self.has_virtual;
        let generated = self.generated;
        let env = self.env;
        let outer = self.outer;
        // Dispatched only from `emit_current` under `self.wr.is_some()`, so `None` here is an
        // engine bug — surface it loud (matching the `wr_*` helpers in `ops::dml_index`)
        // rather than panicking, so the `is_some()` gate and this fetch cannot drift.
        let Some(wr) = self.wr.as_mut() else {
            return Err(Error::format(
                "emit_current_wr called on a non-WITHOUT ROWID index scan (engine bug)",
            ));
        };

        // Recover the full PRIMARY KEY from the index entry, then fetch its row from the PK
        // b-tree. `seek_ge` on the encoded PK prefix lands on the first entry whose PK is
        // `>= pk`; the exact-match check below turns a "row gone" landing into a loud error.
        let pk = wr_entry_pk(&wr.plan, entry)?;
        let prefix = encode_record_enc(&pk, enc);
        let missing = || {
            Error::format(format!(
                "WITHOUT ROWID index entry points to a missing PRIMARY KEY row {pk:?} \
                 (corrupt/unmaintained index)"
            ))
        };
        if !wr.pk_cursor.seek_ge(&prefix)? {
            return Err(missing());
        }
        let key_bytes = wr.pk_cursor.key()?.into_owned();
        let mut local = wr.layout.decode_row_enc(&key_bytes, table, &mut wr.scratch, enc);
        // Verify the landed row's PRIMARY KEY is EXACTLY the one the entry named (seek_ge can
        // land on a greater key when the target row is absent). Compare under each PK column's
        // collation — the WR PK b-tree's own ordering (see `WrLayout::pk_conflict_key`).
        let found_pk = wr.layout.pk_values(&local);
        if found_pk.len() != pk.len()
            || found_pk.iter().zip(&pk).enumerate().any(|(i, (a, b))| {
                compare_values(a, b, wr.layout.pk_field(i).collation) != Ordering::Equal
            })
        {
            return Err(missing());
        }
        // A WR base record never stores a VIRTUAL generated column, so decode left it NULL;
        // compute it on read (byte-identical to the by-rowid path). Non-generated: no-op.
        if has_virtual {
            compute_generated(generated, true, &mut local, env, rt)?;
        }
        Ok(with_outer(outer, local))
    }
}
