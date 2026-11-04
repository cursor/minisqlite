//! Table access leaves: [`seq_scan`] (full b-tree scan) and [`rowid_scan`] (point-eq or
//! range on the rowid key). A ROWID table scan emits `[c0, …, c_{N-1}, rowid]` (width
//! `N+1`); a WITHOUT ROWID table scan emits `[c0, …, c_{N-1}]` (width `N`, NO rowid — it
//! walks the PRIMARY KEY index b-tree and decodes each key record, see
//! [`super::without_rowid`]). Both prepend the `outer` row when it is non-empty (a
//! correlated context), per the shared ROW/REGISTER convention.

use minisqlite_btree::{IndexCursor, TableCursor};
use minisqlite_catalog::TableDef;
use minisqlite_expr::{eval, to_integer, EvalExpr};
use minisqlite_fileformat::TextEncoding;
use minisqlite_pager::text_encoding_of;
use minisqlite_plan::{GeneratedProgram, RangeBound, RowidOp, RowidScan, ScanDirection, SeqScan};
use minisqlite_types::{Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::generated::compute_generated;
use crate::ops::must_be_int::{must_be_int, MustBeInt};
use crate::ops::with_outer;
use crate::ops::without_rowid::{wr_layout, WrLayout};
use crate::row::{decode_table_row_enc, decode_table_row_skipping_virtual_enc, resolve_base_table, resolve_table};
use crate::runtime::Runtime;
use crate::RowCursor;

/// Count the rows a cursor would still yield by pulling and discarding each — exactly the
/// work the default [`RowCursor::count_rows`] does. The scan leaves below use it as the
/// fallback when their decode-free entry-count fast path does not apply (a point or bounded
/// rowid op, or a scan already in flight), so the returned count is identical to a drain.
fn drain_count(cursor: &mut dyn RowCursor, rt: &mut Runtime) -> Result<usize> {
    let mut n = 0usize;
    while cursor.next_row(rt)?.is_some() {
        n += 1;
    }
    Ok(n)
}

/// Build a full-table scan over `node.table`. A ROWID table walks its table b-tree in
/// ascending rowid order; a WITHOUT ROWID table walks its PRIMARY KEY index b-tree in
/// key order (`ScanDirection` is always forward for a [`SeqScan`]).
pub(crate) fn seq_scan<'a>(
    node: &'a SeqScan,
    env: Env<'a>,
    outer: &'a [Value],
) -> Result<Box<dyn RowCursor + 'a>> {
    // The table's generated-column programs (empty for the common non-generated table).
    // `has_virtual` is the one bit that selects the virtual-aware decode + on-read compute
    // over the byte-identical fast decode, so a table with no VIRTUAL column pays nothing.
    let generated = env.plan.generated_programs(node.db, &node.table);
    let has_virtual = generated.iter().any(|p| p.is_virtual());
    // The store backing this node's namespace (`node.db`): a `SeqScan` cursor is bound to
    // exactly one b-tree in exactly one pager, and a cross-namespace query resolves each
    // base leaf against its own `db`. A single-store connection resolves `db = main` here,
    // the same pager as before.
    let pager = env.pagers.get(node.db)?;
    // The database's TEXT encoding (§1.3.13), read once per scan from the page-1 header.
    // A UTF-16 file's stored TEXT is transcoded to the engine's UTF-8 during decode; a
    // UTF-8 file yields `TextEncoding::Utf8` and the decode is byte-identical to before.
    let enc = text_encoding_of(pager);
    // WR-aware gate: a WITHOUT ROWID table is read through the index-b-tree path (no
    // fabricated rowid), so it resolves through `resolve_base_table` and branches here.
    let def: &'a TableDef = resolve_base_table(env.catalog, node.db, &node.table)?;
    if def.without_rowid {
        let layout = wr_layout(def)?;
        let cursor = IndexCursor::open_with_order(pager, def.root_page, layout.order().clone())?;
        return Ok(Box::new(WrSeqScanCursor {
            env,
            def,
            layout,
            cursor,
            outer,
            started: false,
            scratch: Vec::new(),
            generated,
            has_virtual,
            enc,
        }));
    }
    let cursor = TableCursor::open(pager, def.root_page)?;
    Ok(Box::new(SeqScanCursor { env, def, cursor, outer, started: false, generated, has_virtual, enc }))
}

/// A streaming WITHOUT ROWID full-table scan: iterate the PRIMARY KEY index b-tree in
/// key order, decoding each key record into a schema-order row of width `N` (no trailing
/// rowid). Holds only the index cursor (one b-tree descent path) plus the row layout and
/// a decode scratch buffer — never the whole table.
struct WrSeqScanCursor<'a> {
    env: Env<'a>,
    def: &'a TableDef,
    layout: WrLayout,
    cursor: IndexCursor<'a>,
    outer: &'a [Value],
    started: bool,
    /// Reused across every row: `decode_row` decodes the stored (PK-first) values into
    /// this buffer and drains them out by move, so a scan of M rows allocates this spine
    /// once, not once per row.
    scratch: Vec<Value>,
    /// This table's generated-column programs (STORED + VIRTUAL, column order), from the
    /// plan-level map. Empty for a non-generated table; the read path uses only the
    /// VIRTUAL subset (`only_virtual = true`).
    generated: &'a [GeneratedProgram],
    /// Whether any program is VIRTUAL — selects the virtual-aware decode + on-read compute.
    has_virtual: bool,
    /// The database's TEXT encoding (§1.3.13), so a UTF-16 WR table's key-record TEXT
    /// columns transcode to UTF-8 on decode.
    enc: TextEncoding,
}

impl RowCursor for WrSeqScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        let positioned = if self.started {
            self.cursor.next()?
        } else {
            self.started = true;
            self.cursor.first()?
        };
        if !positioned {
            return Ok(None);
        }
        // `decode_row` already leaves each VIRTUAL column a NULL placeholder (its schema
        // slot is absent from the storage permutation — see `WrLayout`), so the width-N row
        // is ready for the on-read compute below. A non-generated table takes the same decode
        // and skips the compute entirely.
        let mut local = {
            let key = self.cursor.key()?;
            self.layout.decode_row_enc(key.as_ref(), self.def, &mut self.scratch, self.enc)
        };
        if self.has_virtual {
            // Fill each VIRTUAL column in column order (a WR row is width N, no rowid
            // register — the generation exprs bound against `[c0..c_{N-1}]`).
            compute_generated(self.generated, true, &mut local, self.env, rt)?;
        }
        Ok(Some(with_outer(self.outer, local)))
    }

    /// Fast `count(*)`: a full WITHOUT ROWID scan yields exactly one row per PRIMARY KEY
    /// index entry, so return the index b-tree's entry count without decoding any key — the
    /// row count of the table. A scan already in flight (`started`) instead drains the
    /// remaining rows so a resume (pull-then-count) stays exact; the row shape and any
    /// generated columns are irrelevant to a pure row count.
    fn count_rows(&mut self, rt: &mut Runtime) -> Result<usize> {
        if self.started {
            return drain_count(self, rt);
        }
        self.started = true;
        self.cursor.entry_count()
    }
}

/// A streaming full-table scan: `first()` on the first pull, then `next()`, decoding
/// each positioned row. Holds only the cursor (one b-tree descent path) — never the
/// whole table.
struct SeqScanCursor<'a> {
    env: Env<'a>,
    def: &'a TableDef,
    cursor: TableCursor<'a>,
    outer: &'a [Value],
    started: bool,
    /// This table's generated-column programs (STORED + VIRTUAL, column order), from the
    /// plan-level map. Empty for a non-generated table.
    generated: &'a [GeneratedProgram],
    /// Whether any program is VIRTUAL — selects the virtual-aware decode + on-read compute.
    has_virtual: bool,
    /// The database's TEXT encoding (§1.3.13), so a UTF-16 file's TEXT columns transcode
    /// to UTF-8 on decode; `Utf8` leaves the decode byte-identical to before.
    enc: TextEncoding,
}

impl RowCursor for SeqScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        let positioned = if self.started {
            self.cursor.next()?
        } else {
            self.started = true;
            self.cursor.first()?
        };
        if !positioned {
            return Ok(None);
        }
        // Fast path: a table with no VIRTUAL column decodes byte-identically to before. With
        // a VIRTUAL column, decode leaves its slot a NULL placeholder (the physical record
        // never stored it) and the on-read compute fills it. Either way `local` is width N+1
        // (`[c0..c_{N-1}, rowid]`) — the shape the generation exprs are bound against.
        let mut local = {
            let payload = self.cursor.payload()?;
            if self.has_virtual {
                decode_table_row_skipping_virtual_enc(&payload, self.cursor.rowid(), self.def, self.enc)
            } else {
                decode_table_row_enc(&payload, self.cursor.rowid(), self.def, self.enc)
            }
        };
        if self.has_virtual {
            compute_generated(self.generated, true, &mut local, self.env, rt)?;
        }
        Ok(Some(with_outer(self.outer, local)))
    }

    /// Fast `count(*)`: a bare full-table scan yields exactly one row per table b-tree
    /// entry, so return the b-tree's entry count without decoding any payload — identical to
    /// the row count a `first`/`next` drain produces. A scan already in flight (`started`)
    /// instead drains the remaining rows so a resume stays exact.
    ///
    /// Unlike the drain, this path does NOT decode rows or evaluate generated-column
    /// expressions (`compute_generated`). That is correct for a pure row COUNT — a generated
    /// column, STORED or VIRTUAL, never adds or removes a row — and mirrors sqlite counting
    /// b-tree entries without materializing values. A filtered `count(*) ... WHERE` still
    /// drains (it is not on this fast path), so any per-row generation there runs as usual.
    fn count_rows(&mut self, rt: &mut Runtime) -> Result<usize> {
        if self.started {
            return drain_count(self, rt);
        }
        self.started = true;
        self.cursor.entry_count()
    }
}

/// Build a rowid access path: a point lookup (`RowidOp::Eq`) or a range scan
/// (`RowidOp::Range`) walking the table b-tree in `node.direction`.
pub(crate) fn rowid_scan<'a>(
    node: &'a RowidScan,
    env: Env<'a>,
    outer: &'a [Value],
) -> Result<Box<dyn RowCursor + 'a>> {
    let pager = env.pagers.get(node.db)?;
    let def: &'a TableDef = resolve_table(env.catalog, node.db, &node.table)?;
    let cursor = TableCursor::open(pager, def.root_page)?;
    let generated = env.plan.generated_programs(node.db, &node.table);
    let has_virtual = generated.iter().any(|p| p.is_virtual());
    let enc = text_encoding_of(pager);
    Ok(Box::new(RowidScanCursor {
        env,
        def,
        cursor,
        op: &node.op,
        direction: &node.direction,
        outer,
        phase: Phase::Init,
        generated,
        has_virtual,
        enc,
    }))
}

/// A rowid scan's progress. `Copy` so `match self.phase` copies out its data and no
/// borrow of `self` is held across the cursor/emit calls in each arm.
#[derive(Clone, Copy)]
enum Phase {
    /// Not started: evaluate the constraint on the first pull.
    Init,
    /// Range scan in flight over the inclusive integer bounds `[lo, hi]`; `started`
    /// is false until the cursor has been positioned at the range's near end.
    Range { lo: i64, hi: i64, started: bool },
    /// Exhausted.
    Done,
}

struct RowidScanCursor<'a> {
    env: Env<'a>,
    def: &'a TableDef,
    cursor: TableCursor<'a>,
    op: &'a RowidOp,
    direction: &'a ScanDirection,
    outer: &'a [Value],
    phase: Phase,
    /// This table's generated-column programs (STORED + VIRTUAL, column order); empty for a
    /// non-generated table. The read path computes only the VIRTUAL subset.
    generated: &'a [GeneratedProgram],
    /// Whether any program is VIRTUAL — selects the virtual-aware decode + on-read compute.
    has_virtual: bool,
    /// The database's TEXT encoding (§1.3.13), threaded into the by-rowid row decode.
    enc: TextEncoding,
}

impl<'a> RowCursor for RowidScanCursor<'a> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        // Copy the two 'a references out so no borrow of `self` is held while the
        // arms below take `&mut self` (cursor moves, emit, phase reassignment).
        let op = self.op;
        match self.phase {
            Phase::Done => Ok(None),
            Phase::Init => match op {
                RowidOp::Eq(expr) => {
                    // A point lookup reaches at most one row: seek exactly, emit it if
                    // present, then we are done regardless.
                    let target = self.eval_rowid_eq(rt, expr)?;
                    self.phase = Phase::Done;
                    match target {
                        Some(t) if self.cursor.seek_exact(t)? => Ok(Some(self.emit(rt)?)),
                        _ => Ok(None),
                    }
                }
                RowidOp::Range { lo, hi } => {
                    match self.eval_rowid_range(rt, lo.as_ref(), hi.as_ref())? {
                        Some((lo_i, hi_i)) => {
                            self.phase = Phase::Range { lo: lo_i, hi: hi_i, started: false };
                            // Recurse once to position and emit the first in-range row.
                            self.next_row(rt)
                        }
                        None => {
                            self.phase = Phase::Done;
                            Ok(None)
                        }
                    }
                }
            },
            Phase::Range { lo, hi, started } => {
                let positioned = if started {
                    self.step()?
                } else {
                    self.phase = Phase::Range { lo, hi, started: true };
                    self.position_range_start(lo, hi)?
                };
                if !positioned {
                    self.phase = Phase::Done;
                    return Ok(None);
                }
                let rid = self.cursor.rowid();
                // The walk is monotonic from the near end, so the FAR bound is what
                // ends the scan: forward stops once past `hi`, reverse once below `lo`.
                let in_range = match self.direction {
                    ScanDirection::Forward => rid <= hi,
                    ScanDirection::Reverse => rid >= lo,
                };
                if in_range {
                    Ok(Some(self.emit(rt)?))
                } else {
                    self.phase = Phase::Done;
                    Ok(None)
                }
            }
        }
    }

    /// Fast `count(*)` — ONLY for a bare, unbounded full-table scan. A rowid scan that is an
    /// equality point lookup or a bounded range selects a SUBSET of the table, so its count
    /// is NOT the b-tree entry count and must be produced by draining (which runs the
    /// seek/range logic). The entry-count shortcut is therefore gated on BOTH:
    ///   * `Phase::Init` — the scan has not started (a resume must drain the remainder), and
    ///   * `RowidOp::Range { lo: None, hi: None }` — unbounded on both ends, i.e. the whole
    ///     table in rowid order (the shape a bare `ORDER BY rowid [DESC]` compiles to).
    /// Scan direction is irrelevant to a count. Gated out, it falls back to the exact drain.
    ///
    /// This taken branch is currently NOT reached by `count(*)` SQL: a bare `count(*)`
    /// compiles to a `SeqScan`, and the only producer of an unbounded `RowidScan` is the
    /// `ORDER BY rowid` rewrite, which a row-count aggregate drops. It is kept as defensive,
    /// forward-safe code for a future planner that could route here; its correctness (entry
    /// count == full drain) is pinned by the `minisqlite-btree` unit tests if it is reached.
    fn count_rows(&mut self, rt: &mut Runtime) -> Result<usize> {
        if matches!(self.phase, Phase::Init)
            && matches!(self.op, RowidOp::Range { lo: None, hi: None })
        {
            self.phase = Phase::Done;
            return self.cursor.entry_count();
        }
        drain_count(self, rt)
    }
}

impl RowidScanCursor<'_> {
    /// Decode and emit the row the cursor is currently positioned on. A rowid scan always
    /// resolves a rowid table (`resolve_table` rejects WITHOUT ROWID), so the decoded row is
    /// width N+1 (`[c0..c_{N-1}, rowid]`) — the shape the generation exprs bind against.
    ///
    /// Fast path unchanged for a non-generated table; with a VIRTUAL column the row decodes
    /// with a NULL placeholder in the virtual slot and the on-read compute fills it.
    fn emit(&self, rt: &mut Runtime) -> Result<Row> {
        let mut local = {
            let payload = self.cursor.payload()?;
            if self.has_virtual {
                decode_table_row_skipping_virtual_enc(&payload, self.cursor.rowid(), self.def, self.enc)
            } else {
                decode_table_row_enc(&payload, self.cursor.rowid(), self.def, self.enc)
            }
        };
        if self.has_virtual {
            compute_generated(self.generated, true, &mut local, self.env, rt)?;
        }
        Ok(with_outer(self.outer, local))
    }

    /// Evaluate an `Eq` target to a concrete rowid, or `None` when the value names no
    /// row (a fractional / out-of-range / non-finite real, a non-numeric or fractional
    /// text, a blob, or NULL).
    ///
    /// This is sqlite's `OP_SeekRowid` coercion — apply NUMERIC affinity then require a
    /// lossless integer, the shared [`must_be_int`] (`OP_MustBeInt`) rule — so a runtime
    /// seek value (a `?` param, a numeric string, an integral real) is handled by value:
    /// `rowid = '5'` reaches rowid 5, `rowid = '2.0'` and `rowid = 5.0` reach rowid 2 / 5,
    /// while `rowid = 5.5`, `rowid = 'abc'` and `rowid = NULL` name no row. The planner
    /// need not pre-coerce a runtime value; a literal it already folds to an `Integer`
    /// seeks unchanged (`must_be_int` is idempotent on an integer).
    fn eval_rowid_eq(&mut self, rt: &mut Runtime, expr: &EvalExpr) -> Result<Option<i64>> {
        let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
        let v = eval(expr, self.outer, &mut ctx)?;
        // OP_SeekRowid: apply NUMERIC affinity then require a lossless integer; anything
        // else (fractional/out-of-range real, non-numeric text, blob, NULL) names no row.
        Ok(match must_be_int(&v) {
            MustBeInt::Int(i) => Some(i),
            MustBeInt::Null | MustBeInt::NotInt => None,
        })
    }

    /// Evaluate the range bounds to inclusive integer `[lo, hi]`. `None` means the
    /// range is empty (an exclusive bound at the i64 extreme, or `lo > hi`).
    ///
    /// DOCUMENTED REFINEMENT: bounds are coerced with `to_integer` (a non-integer or
    /// NULL bound is truncated / treated as 0). Ceil/floor for a fractional real
    /// bound and NULL-excludes-all semantics are a follow-up; the planner emits
    /// integer rowid bounds in practice.
    fn eval_rowid_range(
        &mut self,
        rt: &mut Runtime,
        lo: Option<&RangeBound>,
        hi: Option<&RangeBound>,
    ) -> Result<Option<(i64, i64)>> {
        let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
        let lo_i = match lo {
            None => i64::MIN,
            Some(b) => {
                let x = to_integer(&eval(&b.value, self.outer, &mut ctx)?);
                if b.inclusive {
                    x
                } else {
                    // Exclusive `> x` starts at x+1; `> i64::MAX` selects nothing.
                    match x.checked_add(1) {
                        Some(y) => y,
                        None => return Ok(None),
                    }
                }
            }
        };
        let hi_i = match hi {
            None => i64::MAX,
            Some(b) => {
                let x = to_integer(&eval(&b.value, self.outer, &mut ctx)?);
                if b.inclusive {
                    x
                } else {
                    // Exclusive `< y` ends at y-1; `< i64::MIN` selects nothing.
                    match x.checked_sub(1) {
                        Some(y) => y,
                        None => return Ok(None),
                    }
                }
            }
        };
        if lo_i > hi_i {
            return Ok(None);
        }
        Ok(Some((lo_i, hi_i)))
    }

    /// Position the cursor at the near end of the range for the scan direction:
    /// forward at the first rowid `>= lo`, reverse at the largest rowid `<= hi`.
    fn position_range_start(&mut self, lo: i64, hi: i64) -> Result<bool> {
        match self.direction {
            ScanDirection::Forward => self.cursor.seek_ge(lo),
            ScanDirection::Reverse => self.position_reverse_start(hi),
        }
    }

    /// Find the largest rowid `<= hi` (the start of a reverse range scan). The cursor
    /// has no `seek_le`, so: an unbounded `hi` starts at `last()`; otherwise
    /// `seek_ge(hi)` lands on the first rowid `>= hi` — if that is exactly `hi` it is
    /// the start, if it overshoots we step back one to the largest rowid `< hi`, and
    /// if nothing is `>= hi` every rowid is `< hi` so `last()` is the largest `<= hi`.
    fn position_reverse_start(&mut self, hi: i64) -> Result<bool> {
        if hi == i64::MAX {
            return self.cursor.last();
        }
        if self.cursor.seek_ge(hi)? {
            if self.cursor.rowid() == hi {
                Ok(true)
            } else {
                // Overshoot: the largest rowid strictly below `hi` (which is `<= hi`).
                self.cursor.prev()
            }
        } else {
            self.cursor.last()
        }
    }

    /// Advance one step in the scan direction.
    fn step(&mut self) -> Result<bool> {
        match self.direction {
            ScanDirection::Forward => self.cursor.next(),
            ScanDirection::Reverse => self.cursor.prev(),
        }
    }
}
