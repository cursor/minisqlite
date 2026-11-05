//! `MinMaxSeek` — the MIN/MAX index/rowid seek fast path (optoverview.html §13). Compute a
//! single `MIN(col)`/`MAX(col)` over a whole table with ONE O(log n) b-tree descent to the
//! extremum entry, instead of the O(n) full-scan aggregate the planner would otherwise emit.
//!
//! The planner ([`minisqlite_plan::compile::minmax_index`]) only chooses this node when the
//! extremum is provably byte-identical to the full-scan aggregate (BINARY comparison served
//! by the rowid or an ascending BINARY index whose left-most key column is the argument, a
//! non-BLOB affinity so ties are representation-stable). So this operator only has to WALK
//! to the extremum and apply the NULL rule — no comparison/collation decision is made here.
//!
//! Emits EXACTLY ONE width-1 row `[value]` — the extremum, or `[Null]` for an empty or
//! all-NULL table — then nothing, matching the single-aggregate no-`GROUP BY` [`Aggregate`]
//! output it replaces (which is likewise one width-1 row and carries no `outer` prefix).

use minisqlite_btree::{IndexCursor, TableCursor};
use minisqlite_fileformat::{decode_record_enc, TextEncoding};
use minisqlite_pager::text_encoding_of;
use minisqlite_plan::{MinMaxSeek, MinMaxSource};
use minisqlite_types::{Error, Result, Row, Value};

use crate::env::Env;
use crate::row::{real_affinity_on_read, resolve_table};
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the seek. Holds only the node + [`Env`]; the single b-tree descent happens lazily
/// on the first pull (it needs no [`Runtime`]). `outer` is irrelevant — like the aggregate
/// it replaces, this leaf never carries a correlated prefix into its output row.
pub(crate) fn minmax_seek<'a>(
    node: &'a MinMaxSeek,
    env: Env<'a>,
    _outer: &'a [Value],
) -> Result<Box<dyn RowCursor + 'a>> {
    Ok(Box::new(MinMaxSeekCursor { node, env, done: false }))
}

struct MinMaxSeekCursor<'a> {
    node: &'a MinMaxSeek,
    env: Env<'a>,
    /// The one row is produced on the first pull; every later pull returns `None`.
    done: bool,
}

impl RowCursor for MinMaxSeekCursor<'_> {
    fn next_row(&mut self, _rt: &mut Runtime) -> Result<Option<Row>> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        Ok(Some(vec![compute_extremum(self.node, self.env)?]))
    }
}

/// Seek the one extremum value: O(log n) descent, then O(1) for MAX / MIN-with-no-leading-
/// NULL and O(k) for MIN skipping `k` leading NULLs — never a full scan.
fn compute_extremum(node: &MinMaxSeek, env: Env<'_>) -> Result<Value> {
    let pager = env.pagers.get(node.db)?;
    match &node.source {
        MinMaxSource::Rowid => {
            // The rowid b-tree is ordered by integer rowid: MAX = last, MIN = first. A rowid
            // is never NULL, so there is no NULL skip; an empty table yields NULL.
            let table = resolve_table(env.catalog, node.db, &node.table)?;
            let mut cur = TableCursor::open(pager, table.root_page)?;
            let positioned = if node.is_max { cur.last()? } else { cur.first()? };
            Ok(if positioned { Value::Integer(cur.rowid()) } else { Value::Null })
        }
        MinMaxSource::Index { index } => {
            let def = env
                .catalog
                .index_in(node.db, index)?
                .ok_or_else(|| Error::sql(format!("no such index: {index}")))?;
            // The planner (`minisqlite_plan::compile::minmax_index`) builds an Index seek
            // ONLY over a non-partial, non-expression, ascending, BINARY-declared left-most
            // key column — the exact shape a plain forward/reverse b-tree walk presents in
            // extremum order. A seek over any other shape (a partial index misses rows; a
            // DESC or NOCASE-declared key walks in a different order than it accumulates
            // under) would yield a SILENTLY WRONG extremum on a core SQL path.
            // That invariant lives only in the planner's guards and is not encoded in
            // `MinMaxSource::Index` (a bare name), so re-check it loudly here rather than
            // trust a producer blindly — a debug guard that turns a future planner bug (or a
            // `MinMaxSeek` built outside `try_optimize`) into a test-time panic at the cause.
            debug_assert!(
                !def.partial
                    && !matches!(def.key_exprs.first(), Some(Some(_)))
                    && def.key_columns.first().is_some_and(|kc| {
                        !kc.descending
                            && kc.collation.as_deref().is_none_or(|c| c.eq_ignore_ascii_case("BINARY"))
                    }),
                "MinMaxSeek over index {index}: expected a non-partial, non-expression, \
                 ascending, BINARY left-most key (the planner's minmax_index invariant); a \
                 plain b-tree walk over any other shape returns a wrong extremum"
            );
            let enc = text_encoding_of(pager);
            let mut cur = IndexCursor::open(pager, def.root_page)?;
            let mut v = index_extremum(&mut cur, node.is_max, enc)?;
            // The key column is a plain table column (asserted above), so its stored value
            // gets the column's REAL affinity on read, the same as a table scan applies.
            let table = resolve_table(env.catalog, node.db, &node.table)?;
            if let Some(col) = def.columns.first().and_then(|name| {
                table.columns.iter().find(|c| c.name.eq_ignore_ascii_case(name))
            }) {
                real_affinity_on_read(&mut v, col);
            }
            Ok(v)
        }
    }
}

/// The extremum left-most-key value of an ascending BINARY index b-tree.
///
/// NULLs sort FIRST (`minisqlite_btree::index_key`), so:
/// * **MAX** = the LAST entry's key column — non-NULL unless the tree is empty or every
///   entry's key column is NULL (both -> NULL).
/// * **MIN** = the FIRST NON-NULL entry's key column — walk forward from the smallest key
///   and skip the leading NULL run (bounded, O(k)); empty or all-NULL -> NULL.
fn index_extremum(cur: &mut IndexCursor, is_max: bool, enc: TextEncoding) -> Result<Value> {
    if is_max {
        if !cur.last()? {
            return Ok(Value::Null);
        }
        return leftmost_key(cur, enc);
    }
    if !cur.first()? {
        return Ok(Value::Null);
    }
    loop {
        let v = leftmost_key(cur, enc)?;
        if !matches!(v, Value::Null) {
            return Ok(v);
        }
        if !cur.next()? {
            return Ok(Value::Null);
        }
    }
}

/// Decode the left-most key column of the entry the cursor is on. An index key is a record
/// `[key_cols.., rowid]`; the extremum is its first column. Decoded with the database's TEXT
/// encoding (§1.3.13) so a UTF-16 file's TEXT transcodes to UTF-8 — byte-identical to the
/// value the table row decodes to (same stored bytes, same encoding); the caller then applies
/// the column's REAL affinity on read, as the table decode does.
fn leftmost_key(cur: &IndexCursor, enc: TextEncoding) -> Result<Value> {
    let key = cur.key()?;
    Ok(decode_record_enc(key.as_ref(), enc).into_iter().next().unwrap_or(Value::Null))
}
