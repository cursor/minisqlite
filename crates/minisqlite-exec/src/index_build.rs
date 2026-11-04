//! `CREATE INDEX` backfill: populate a freshly created (empty) index b-tree from the
//! table rows that already exist.
//!
//! The catalog's `create_index` only allocates an empty index root and writes the
//! `sqlite_schema` row; it does NOT copy existing rows into the index (an index is a
//! redundant copy of ALL the table's rows — `spec/sqlite-doc/lang_createindex.html`).
//! This module is the executor half the engine calls right after `create_index`, in
//! the SAME write transaction, so the schema registration and the backfill commit (or
//! roll back) as one atomic unit.
//!
//! It reuses the SHARED DML index-maintenance path ([`build_index_plan`] +
//! [`insert_index_entries`] + [`unique_conflict`]) rather than re-deriving key
//! encoding, so a backfilled entry is byte-for-byte the same key an `INSERT` after the
//! index would have written — collation, `DESC`, and rowid-suffix all come out
//! identical for free, and the two paths cannot drift.
//!
//! Memory is bounded: the table is STREAMED one row at a time (never collected into a
//! `Vec`), so `CREATE INDEX` on a large table stays proportional to a single row plus
//! the index descent, not the table size.

use minisqlite_btree::{IndexCursor, TableCursor};
use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_expr::EvalExpr;
use minisqlite_fileformat::TextEncoding;
use minisqlite_pager::{text_encoding_of, Pager};
use minisqlite_plan::{GeneratedProgram, Plan, PlanNode};
use minisqlite_types::{ConstraintKind, DbIndex, Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::{Env, Pagers};
use crate::ops::dml_index::{
    build_index_plan, index_key_values, insert_index_entries, unique_conflict,
    wr_index_key_values, wr_unique_conflict, IndexPlan,
};
use crate::ops::generated::compute_generated;
use crate::ops::without_rowid::wr_layout;
use crate::row::{decode_table_row_enc, decode_table_row_skipping_virtual_enc, resolve_base_table};
use crate::runtime::Runtime;

/// Backfill the index named `index_name` from the existing rows of its table.
///
/// Runs inside the caller's (`CREATE INDEX`) write transaction — it neither begins nor
/// commits — so any error it returns propagates to the engine, which rolls the whole
/// transaction back, leaving neither the index b-tree, the `sqlite_schema` row, nor a
/// half-populated index behind.
///
/// For a `UNIQUE` index, a duplicate key among the existing rows is a
/// `UNIQUE constraint failed: …` error (the identical error and probe the INSERT path
/// raises — `<table>.<col>` for an ordinary index, `index '<name>'` for an expression
/// index), which aborts the create. NULL keys are exempt: a UNIQUE index
/// treats every NULL as distinct, so any number of NULL-keyed rows backfill without
/// conflict (`lang_createindex.html` §1.1).
///
/// The scan is bounded in memory: rows are streamed from the table b-tree one at a
/// time via a re-seek from the last processed rowid, so nothing table-sized is
/// materialized. (A bounded-batch scan — read K rows per descent instead of one — is a
/// possible future refinement to cut the per-row re-descent when `CREATE INDEX` at
/// scale is profiled as a bottleneck; it would not change the memory bound's shape.)
///
/// `generated` is the target table's GENERATED-column programs (bound by the planner;
/// empty for a table with none). An index over a VIRTUAL generated column must COMPUTE
/// that column to key its entries — the value is not in the stored record (`gencol.html`)
/// — so when any program is VIRTUAL each row is decoded through the virtual-aware path
/// and its virtual columns are computed before the index key is built. A STORED
/// generated column is already in the record, so a plain or STORED-only table keeps the
/// byte-identical fast decode with no per-row compute.
///
/// `index_key_exprs` is the NEW index's compiled key expressions (aligned with its
/// `columns` / `key_exprs`; `Some` = an expression key column, `None`/absent = an ordinary
/// named column). The engine binds them from the index's key expressions and passes them
/// here (see [`minisqlite_plan::QueryPlanner::index_key_programs`]); an expression index
/// (`CREATE INDEX i ON t(a+b)`) backfills by EVALUATING each key column per row. For an
/// ordinary all-named-column index the slice is empty and the backfill key is byte-identical
/// to a column-only index.
///
/// `partial_predicate` is the NEW index's compiled WHERE predicate when it is a PARTIAL index
/// (`CREATE INDEX i ON t(a) WHERE p`), `None` for a full index. The engine binds it from the
/// `CREATE INDEX` statement and passes it here; the backfill inserts an entry for a row only
/// when the predicate is TRUE over that row ([`index_key_values`] returns `None` otherwise), so
/// a partial index backfills exactly the matching rows — never phantom entries for rows outside
/// it — and a partial UNIQUE index's create errors only on a duplicate AMONG those matching rows
/// (`partialindex.html` §2, §4). A full index passes `None` and is byte-for-byte unchanged.
pub fn build_index(
    pager: &mut dyn Pager,
    catalog: &dyn Catalog,
    index_name: &str,
    generated: &[GeneratedProgram],
    index_key_exprs: &[Option<EvalExpr>],
    partial_predicate: Option<EvalExpr>,
) -> Result<()> {
    // Resolve the just-created index and its table. `create_index` cached both before
    // this runs, so a missing index here is a logic/corruption error, not an expected
    // state — fail loud rather than silently skip the backfill.
    let idx = catalog.index(index_name)?.ok_or_else(|| {
        Error::format(format!("build_index: index {index_name} not found in catalog"))
    })?;
    // `resolve_base_table` is the WR-aware base-table gate (it accepts BOTH a rowid table and
    // a WITHOUT ROWID table). The `def.without_rowid` branch below routes a WR target to the
    // PRIMARY-KEY-b-tree backfill; a rowid target keeps the integer-rowid streaming path.
    //
    // `catalog` here is the target namespace's CONCRETE `SchemaCatalog` (the engine passes
    // the single namespace being indexed, not a `MultiCatalog`), so its `table_in` ignores
    // the `db` argument — `DbIndex::MAIN` resolves the table in that one namespace.
    let def = resolve_base_table(catalog, DbIndex::MAIN, &idx.table)?;
    // Build the shared index-key plan. For a WR table this ALSO raises the loud fail-closed
    // error for an EXPRESSION index (its key exprs bind against a frame with no WR rowid). A
    // PARTIAL index carries its compiled predicate so the backfill gates each row on it.
    let plan = build_index_plan(def, idx, index_key_exprs, partial_predicate)?;
    let table_root = def.root_page;
    let plans = std::slice::from_ref(&plan);

    // A VIRTUAL generated column is not physically stored (`gencol.html`), so an index
    // over one must compute the value to key each entry; a STORED generated column is in
    // the record, so a plain / STORED-only table needs neither the virtual-aware decode
    // nor the compute. `has_virtual` gates only that per-row decode+compute below.
    let has_virtual = generated.iter().any(|p| p.is_virtual());
    // A throwaway eval context, built UNCONDITIONALLY because it now serves BOTH the
    // generated-column compute (when `has_virtual`) AND the index-key EVAL phase (an
    // expression index key is evaluated through it). Both kinds of expression are subquery-
    // and parameter-free (enforced at bind time), so an EMPTY plan (never indexed for a
    // subquery) and a fresh `Runtime` (RNG/params unused) suffice; for an ordinary
    // all-named-column index no key part is an expression, so the ctx is not read for keying.
    // A generated- or index-key expr using a non-deterministic function would be a schema real
    // sqlite rejects at CREATE, and a generated column is recomputed on read regardless — out
    // of scope here.
    let backfill_plan = Plan {
        root: PlanNode::SingleRow,
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    let mut rt = Runtime::new();

    // The database's text encoding (fileformat2 §1.3.13), read ONCE for the whole backfill
    // (off-56 is fixed at DB creation) and threaded into every per-row decode and index
    // write below — so the UTF-8 backfill of an N-row table pays no per-row header parse,
    // and a UTF-16 table's stored rows decode (and its new index keys encode) correctly.
    let enc = text_encoding_of(&*pager);

    // A WITHOUT ROWID target stores its rows in the PRIMARY KEY index b-tree (no integer
    // rowid), so it backfills through a separate PK-keyed streaming path that decodes each
    // stored row via `WrLayout` and writes the WR-shaped secondary key (`[indexed cols..,
    // trailing PK..]`, fileformat2 §2.5.1). Branch here — AFTER the shared plan build (which
    // fail-closes an expression WR index) and the once-per-statement encoding read, BEFORE the
    // rowid-only streaming loop below (which decodes a `[cols.., rowid]` frame a WR row lacks).
    if def.without_rowid {
        return build_index_wr(
            pager, catalog, def, &plan, plans, generated, has_virtual, &backfill_plan, &mut rt, enc,
        );
    }

    // Stream the table b-tree in ascending rowid order. Each pass opens a short-lived
    // read cursor (a shared reborrow of the pager), fetches the next row after the one
    // last indexed, then releases that borrow so the write phase can take the exclusive
    // `&mut` to insert the entry — the same read-then-write borrow split the DML
    // operators use, here per row so no table-sized buffer is held.
    //
    // INVARIANT (why re-seeking across the write is safe): the table b-tree is
    // unchanged between reading rowid R and re-seeking to the next rowid, so the scan
    // visits every row exactly once and skips none. This holds because the write phase
    // touches ONLY the (disjoint) index b-tree; `allocate_page` never frees a page, so
    // an index insert/split/overflow cannot move or clobber a live table page;
    // `table_root` is stable (a b-tree root id does not change under insert); and the
    // single writer holds the write lock for the whole transaction, so no other
    // connection mutates the table. A future edit that lets the write phase touch the
    // table (CREATE INDEX firing triggers, or a vacuum that relocates pages) would void
    // this and must revisit the streaming strategy.
    let mut last: Option<i64> = None;
    loop {
        // READ PHASE (shared borrow, scoped): fetch (rowid, decoded row) after `last`.
        let fetched = {
            let mut tc = TableCursor::open(&*pager, table_root)?;
            let positioned = match last {
                None => tc.first()?,
                // The next row is the first rowid strictly greater than the last one
                // indexed. `checked_add` guards the degenerate case where the previous
                // rowid was i64::MAX: no rowid can follow it, so the scan is done.
                Some(prev) => match prev.checked_add(1) {
                    Some(next) => tc.seek_ge(next)?,
                    None => false,
                },
            };
            if positioned {
                let rowid = tc.rowid();
                let row = {
                    let payload = tc.payload()?;
                    // A table with a VIRTUAL column stores a record that OMITS it, so it
                    // must decode through the virtual-aware path (NULL placeholders) and
                    // compute below — a positional decode would mis-map the columns.
                    if has_virtual {
                        decode_table_row_skipping_virtual_enc(&payload, rowid, def, enc)
                    } else {
                        decode_table_row_enc(&payload, rowid, def, enc)
                    }
                };
                Some((rowid, row))
            } else {
                None
            }
        };
        let Some((rowid, mut row)) = fetched else { break };

        // Fill the VIRTUAL generated columns so the index key reflects the COMPUTED value
        // (STORED columns are already present from the decode, so only the virtual subset
        // is computed). The eval row is `[c0..c_{N-1}, rowid]` (width N+1), so a generated
        // expression referencing the INTEGER PRIMARY KEY resolves to the rowid register.
        if has_virtual {
            // Single-namespace backfill: `Pagers::One` over the one store being indexed.
            // The generated exprs carry no table scan, so `get` is never called and the
            // `db` tag is inert; `MAIN` is a safe placeholder (a stray scan would fail
            // closed rather than read another store).
            let env =
                Env { catalog, pagers: Pagers::One { db: DbIndex::MAIN, pager: &*pager }, plan: &backfill_plan };
            compute_generated(generated, true, &mut row, env, &mut rt)?;
        }

        // EVAL PHASE of index maintenance: compute this row's index key ([cols.., rowid])
        // through a shared-pager `EvalCtx` (an expression key needs it), before the
        // exclusive-borrow write below. `row` is the `[c0..c_{N-1}, rowid]` frame the key
        // exprs bind against; Col-only (today) reads the same columns + rowid the
        // pre-expression backfill did.
        let key = {
            let env =
                Env { catalog, pagers: Pagers::One { db: DbIndex::MAIN, pager: &*pager }, plan: &backfill_plan };
            let mut ctx = EvalCtx { rt: &mut rt, env, outer: &row };
            index_key_values(&plan, &row, rowid, &mut ctx)?
        };
        let keys = std::slice::from_ref(&key);

        // WRITE PHASE (exclusive borrow). A PARTIAL index yields `None` for a row its WHERE
        // predicate does not admit: the row is not in the index, so it neither conflicts nor is
        // inserted (`insert_index_entries` skips a `None`), and the UNIQUE probe is skipped too —
        // a partial UNIQUE create errors only on a duplicate among the MATCHING rows. For a full
        // index (or a partial one that admits the row) the key is `Some`: for a UNIQUE index,
        // probe the entries inserted SO FAR (every earlier row) before adding this one — the
        // current row's entry is not yet in the index, so a matching key at another rowid is a
        // genuine duplicate among the pre-existing data. `unique_conflict` returns no conflict
        // for a key with any NULL column (NULLs are distinct in a UNIQUE index).
        if plan.unique {
            if let Some(k) = &key {
                if unique_conflict(&*pager, &plan, k, rowid, enc)? {
                    return Err(Error::constraint(ConstraintKind::Unique, &plan.detail));
                }
            }
        }
        insert_index_entries(&mut *pager, plans, keys, enc)?;
        last = Some(rowid);
    }
    Ok(())
}

/// Backfill a secondary index over a WITHOUT ROWID table by streaming its PRIMARY KEY
/// b-tree (the table's rows ARE its entries — `withoutrowid.html`; fileformat2 §2.4) and
/// writing a WR-shaped entry (`[indexed cols.., trailing PK..]`, fileformat2 §2.5.1) for
/// each row through the shared [`super::ops::dml_index`] encoder — the SAME key an `INSERT`
/// after the index would write, so backfill and DML cannot drift.
///
/// Memory is bounded the same way as the rowid path: rows are STREAMED one at a time by
/// re-seeking past the last processed PRIMARY KEY key each pass, never collected into a
/// `Vec`. The re-seek is safe for the identical reason (the write phase touches ONLY the
/// disjoint NEW index b-tree; `allocate_page` never frees, so the PK b-tree pages are stable
/// across the read→write→re-seek cycle) — a `seek_ge` on the exact stored key bytes re-finds
/// the last row, and `next` advances to the following one.
///
/// UNIQUE enforcement matches the rowid path and `INSERT`: probe the entries written SO FAR
/// (every EARLIER row) with `self_pk = None` (this row's own entry is not yet present, so any
/// matching indexed-prefix entry is a genuine pre-existing duplicate) and error
/// `UNIQUE constraint failed: …` on a hit — the same error and probe an `INSERT` raises. A
/// key with any NULL indexed column is exempt (NULLs are distinct in a UNIQUE index).
///
/// A VIRTUAL generated indexed column is computed per row before keying (the value is not in
/// the stored record); a plain/STORED table skips that compute. An EXPRESSION index on a WR
/// table never reaches here — [`build_index_plan`] already fail-closed above.
#[allow(clippy::too_many_arguments)]
fn build_index_wr(
    pager: &mut dyn Pager,
    catalog: &dyn Catalog,
    def: &TableDef,
    plan: &IndexPlan,
    plans: &[IndexPlan],
    generated: &[GeneratedProgram],
    has_virtual: bool,
    backfill_plan: &Plan,
    rt: &mut Runtime,
    enc: TextEncoding,
) -> Result<()> {
    // The PK b-tree root and the schema→storage layout, both stable across the statement.
    // `wr_layout` fails closed only on a genuinely PK-less (corrupt) catalog — a
    // well-formed WR table always declares a PRIMARY KEY — so the backfill inherits that guard.
    let table_root = def.root_page;
    let layout = wr_layout(def)?;
    // Reused across rows so a large table's backfill does not churn a per-row scratch Vec.
    let mut scratch: Vec<Value> = Vec::new();
    // The last processed entry's exact key bytes; `None` starts at the first entry. Re-seeking
    // by these bytes (then stepping past) streams the whole tree without a table-sized buffer.
    let mut last_key: Option<Vec<u8>> = None;
    loop {
        // READ PHASE (shared borrow, scoped): fetch the next (key bytes, decoded row).
        let fetched: Option<(Vec<u8>, Row)> = {
            let mut cur = IndexCursor::open_with_order(&*pager, table_root, layout.order().clone())?;
            let positioned = match &last_key {
                None => cur.first()?,
                // `seek_ge` re-finds the exact last key (the PK b-tree is unchanged across the
                // write), and `next` advances to the following row. If the last key somehow no
                // longer resolves the tree is done rather than risking a skipped/looped row.
                Some(k) => cur.seek_ge(k)? && cur.next()?,
            };
            if positioned {
                let key_bytes = cur.key()?.into_owned();
                let row = layout.decode_row_enc(&key_bytes, def, &mut scratch, enc);
                Some((key_bytes, row))
            } else {
                None
            }
        };
        let Some((key_bytes, mut row)) = fetched else { break };

        // Fill VIRTUAL generated columns so the index key reflects the COMPUTED value. The
        // eval row is the width-N WR row `[c0..c_{N-1}]` (no rowid register — a WR generation
        // expr binds over the columns alone). STORED columns are already present from decode.
        if has_virtual {
            let env = Env {
                catalog,
                pagers: Pagers::One { db: DbIndex::MAIN, pager: &*pager },
                plan: backfill_plan,
            };
            compute_generated(generated, true, &mut row, env, rt)?;
        }

        // EVAL: the WR secondary key for this row (`[indexed cols.., trailing PK..]`). A WR
        // index key is all ordinary columns (an expression WR index is fail-closed at plan
        // build); the scoped eval ctx is needed only to gate a PARTIAL index on its WHERE
        // predicate over the width-N WR row (a WR table has no rowid register, so the predicate
        // reads only `row`). A full WR index never touches the ctx. The shared pager borrow the
        // ctx holds ends before the exclusive-borrow write below.
        let key = {
            let env =
                Env { catalog, pagers: Pagers::One { db: DbIndex::MAIN, pager: &*pager }, plan: backfill_plan };
            let mut ctx = EvalCtx { rt, env, outer: &row };
            wr_index_key_values(plan, &row, &mut ctx)?
        };
        let keys = std::slice::from_ref(&key);

        // WRITE PHASE (exclusive borrow). A PARTIAL index yields `None` for a row it does not
        // admit: no entry, so no UNIQUE probe and nothing inserted (`insert_index_entries`
        // skips a `None`). For a full index (or an admitted row) the key is `Some`: UNIQUE
        // probe over the earlier rows, then insert.
        if plan.unique {
            if let Some(k) = &key {
                if wr_unique_conflict(&*pager, plan, k, None, enc)?.is_some() {
                    return Err(Error::constraint(ConstraintKind::Unique, &plan.detail));
                }
            }
        }
        insert_index_entries(&mut *pager, plans, keys, enc)?;
        last_key = Some(key_bytes);
    }
    Ok(())
}
