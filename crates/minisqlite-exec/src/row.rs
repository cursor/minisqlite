//! Decode a stored table record into the executor's row layout.
//!
//! The shared ROW/REGISTER convention (see `minisqlite_plan::plan`): a *ROWID* base-table
//! scan of an `N`-column table emits width `N+1` — the `N` columns then the rowid as
//! `Value::Integer` in register `N`, and [`decode_table_row_enc`] is the one place that shape
//! is produced. A *WITHOUT ROWID* base-table scan emits width `N` (no rowid register); its
//! shape is produced separately in [`crate::ops::without_rowid::WrLayout::decode_row`],
//! because a WR row is keyed by its PRIMARY KEY and has no integer rowid to append.

use minisqlite_catalog::{Catalog, ColumnDef, TableDef};
use minisqlite_fileformat::{decode_record_into_enc, TextEncoding};
use minisqlite_types::{affinity_of_declared_type, Affinity, DbIndex, Error, Result, Row, Value};

/// Whether `col` is a VIRTUAL generated column — the executor-side alias for
/// [`ColumnDef::is_virtual_generated`], which is THE source of truth for "does this column
/// take a stored record slot?" (a VIRTUAL column takes none — `gencol.html`; a STORED
/// generated column and an ordinary column are both physically present). Kept as a free
/// function so the many `is_virtual_generated(&def.columns[i])` call sites across `row`,
/// `ops::without_rowid`, and `ops::generated` read uniformly; it delegates so the predicate
/// cannot drift from the catalog's `ALTER TABLE DROP COLUMN` slot mapping, which uses the
/// same method.
pub(crate) fn is_virtual_generated(col: &ColumnDef) -> bool {
    col.is_virtual_generated()
}

/// Apply REAL affinity on read to one decoded column value: an INTEGER read from a column
/// with REAL affinity becomes the equal REAL.
///
/// sqlite3 stores a REAL-affinity value that has no fractional part as an integer to save
/// space and converts it back when the column is read (datatype3.html §3, `OP_RealAffinity`
/// after `OP_Column` in `sqlite3ColumnDefault`), so a file it wrote holds integer serial
/// types in REAL columns. Every place that turns stored bytes into a column value (table
/// rows, WITHOUT ROWID rows, covering index keys, the MIN/MAX index seek) routes through
/// here. A short row's `ADD COLUMN` default gets the same treatment, as in sqlite3.
pub(crate) fn real_affinity_on_read(v: &mut Value, col: &ColumnDef) {
    if let Value::Integer(i) = *v {
        if affinity_of_declared_type(col.declared_type.as_deref()) == Affinity::Real {
            *v = Value::Real(i as f64);
        }
    }
}

/// [`real_affinity_on_read`] over a schema-order row prefix `[c0..]` of table `def`.
pub(crate) fn real_affinity_on_read_row(row: &mut [Value], def: &TableDef) {
    for (v, col) in row.iter_mut().zip(&def.columns) {
        real_affinity_on_read(v, col);
    }
}

/// Build the `[c0, …, c_{N-1}, rowid]` row (width `N+1`) for one stored record, decoding
/// TEXT columns in the database's declared text encoding `enc` (fileformat2 §1.3.13).
///
/// Encoding is threaded, never assumed: a UTF-16 file's TEXT columns transcode to the
/// engine's UTF-8 here, and a UTF-8 file passes `TextEncoding::Utf8` for the byte-identical
/// fast path. There is deliberately NO bare UTF-8 wrapper — every caller (read scans and
/// the DML/DDL read-back sites) reads the DB encoding ONCE per statement (`text_encoding_of`)
/// and passes it, so a decode can never silently assume UTF-8 on a UTF-16 database.
///
/// This is the ROWID-table decode: it unconditionally appends the integer rowid
/// register, which a `WITHOUT ROWID` table (keyed by its primary key, no integer rowid)
/// has no basis for. A rowid-only operator (`rowid_scan`, the min/max seek) gates its
/// table through [`resolve_table`] (which rejects a WR table outright); a WR-aware
/// operator (`seq_scan`, `index_scan`, INSERT / UPDATE / DELETE, the `CREATE INDEX`
/// backfill) resolves through [`resolve_base_table`] and branches to the WR index-b-tree
/// decode ([`crate::ops::without_rowid::WrLayout::decode_row_enc`]) BEFORE ever calling
/// this — so a WR table never reaches here. The `debug_assert!` below pins that invariant.
///
/// * Missing trailing columns (a record stored with fewer than `N` values — a "short"
///   row that predates a later `ADD COLUMN`) decode to that column's DEFAULT, per
///   SQLite: a short row reads a newly-added column as its default, not always NULL.
///   The default is the constant `ColumnDef::default_value` the catalog materialized at
///   build time (NULL when the column has no constant default), so this stays a lookup,
///   never a per-row re-parse.
/// * Extra trailing values (more than `N`) are truncated — a record must not widen
///   the logical row past the schema's column count.
/// * If the table has an `INTEGER PRIMARY KEY` rowid alias, that column's register
///   reflects the rowid (a stored NULL there shows as the rowid), per the convention.
pub(crate) fn decode_table_row_enc(
    payload: &[u8],
    rowid: i64,
    def: &TableDef,
    enc: TextEncoding,
) -> Row {
    // INVARIANT: every caller resolves the table through `resolve_table` first, which
    // rejects WITHOUT ROWID tables — so a base-table row decoded here always has a real
    // integer rowid to append in register `N`. Assert it, so a future leaf operator that
    // opens a table cursor without going through the gate fails loudly at its cause
    // rather than fabricating a bogus rowid register for a keyless table.
    debug_assert!(
        !def.without_rowid,
        "decode_table_row_enc on WITHOUT ROWID table {} — resolve via resolve_table first",
        def.name
    );
    let n = def.columns.len();
    // Runs once per scanned row (the hot path), so size the buffer to the known
    // final width `N + 1` up front: `decode_record_into_enc` clears then appends the
    // stored columns (keeping our capacity), and the trailing rowid `push` then
    // also fits — no growth realloc in the common case.
    let mut row = Vec::with_capacity(n + 1);
    decode_record_into_enc(payload, enc, &mut row);
    row.truncate(n);
    // Fill any columns the record did not store (a short row from `ADD COLUMN`) with
    // each column's materialized DEFAULT, falling back to NULL. This slice is empty for
    // a full-width row, so the common path pays nothing; the `clone` runs only on the
    // short-row tail. The subsequent `rowid_alias` fixup still overrides its register,
    // and an INTEGER PRIMARY KEY can never be an added column, so it is never short here.
    for col in &def.columns[row.len()..] {
        row.push(col.default_value.clone().unwrap_or(Value::Null));
    }
    real_affinity_on_read_row(&mut row, def);
    row.push(Value::Integer(rowid));
    if let Some(i) = def.rowid_alias {
        // The INTEGER PRIMARY KEY column aliases the rowid: its stored value is NULL
        // (the rowid lives in the b-tree key), so surface the rowid in that register.
        // `rowid_alias` is a column index (catalog invariant `i < N`); assert it so a
        // malformed catalog fails at its cause, not as a bare out-of-bounds later.
        debug_assert!(i < n, "rowid_alias {i} out of range for {n}-column table");
        row[i] = Value::Integer(rowid);
    }
    row
}

/// Build the `[c0, …, c_{N-1}, rowid]` row (width `N+1`) for a stored record whose table
/// has VIRTUAL generated columns — the virtual-aware sibling of [`decode_table_row_enc`],
/// decoding TEXT columns in the database's text encoding `enc` (§1.3.13) the same way.
///
/// A VIRTUAL generated column is NOT written to the physical record (`gencol.html`), so a
/// naive positional decode would shift every value after it. This walks the schema columns
/// in order and maps each NON-virtual column (ordinary or STORED) to the NEXT stored value,
/// leaving each VIRTUAL column a `Value::Null` PLACEHOLDER for the scan/DML step to compute
/// (it has no access to the generation expressions — decoding is pure layout, using `def`
/// only). The short-row DEFAULT fill (a non-virtual column the record did not store), the
/// extra-value truncation, and the rowid / INTEGER PRIMARY KEY alias fixup match
/// [`decode_table_row_enc`] exactly, so the only difference is the virtual-skip mapping.
///
/// Callers select this over the plain [`decode_table_row_enc`] on a per-cursor `has_virtual`
/// flag, so a table with no virtual column never pays for the branch.
pub(crate) fn decode_table_row_skipping_virtual_enc(
    payload: &[u8],
    rowid: i64,
    def: &TableDef,
    enc: TextEncoding,
) -> Row {
    debug_assert!(
        !def.without_rowid,
        "decode_table_row_skipping_virtual_enc on WITHOUT ROWID table {} — WR uses WrLayout::decode_row",
        def.name
    );
    let n = def.columns.len();
    // Decode the stored (non-virtual) values, then distribute them across the logical row:
    // each non-virtual column consumes the next stored value; a virtual column is a NULL
    // placeholder. `into_iter().next()` yields the short-row DEFAULT when the record stored
    // fewer non-virtual values than the schema has (e.g. an `ADD COLUMN` VIRTUAL predating
    // rows), and silently drops any extra trailing values (truncation).
    // `n` is an upper bound on the stored-value count (a table can only have FEWER stored
    // values than columns once a VIRTUAL column omits its slot), so this one capacity hint
    // removes the growth reallocations `decode_record_into_enc` would otherwise incur as it
    // appends — this runs once per row on a full scan of a virtual-column table.
    let mut stored = Vec::with_capacity(n);
    decode_record_into_enc(payload, enc, &mut stored);
    let mut stored = stored.into_iter();
    let mut row = Vec::with_capacity(n + 1);
    for col in &def.columns {
        if is_virtual_generated(col) {
            row.push(Value::Null);
        } else {
            row.push(stored.next().unwrap_or_else(|| col.default_value.clone().unwrap_or(Value::Null)));
        }
    }
    real_affinity_on_read_row(&mut row, def);
    row.push(Value::Integer(rowid));
    if let Some(i) = def.rowid_alias {
        debug_assert!(i < n, "rowid_alias {i} out of range for {n}-column table");
        // An INTEGER PRIMARY KEY can never be generated, so this never overwrites a virtual
        // placeholder — it surfaces the rowid in the alias column exactly as the fast path does.
        row[i] = Value::Integer(rowid);
    }
    row
}

/// Resolve a base table by name, accepting BOTH rowid and `WITHOUT ROWID` tables. The
/// gate every operator that can handle either kind (`seq_scan`, `index_scan`, INSERT /
/// UPDATE / DELETE, and the `CREATE INDEX` backfill) routes through; each then branches on
/// `def.without_rowid` to pick the rowid-keyed or the PRIMARY KEY index-b-tree path. For a
/// WR secondary index that means keying entries by `[indexed cols.., trailing PK..]` and
/// fetching rows by PRIMARY KEY. Errors `no such table: {name}` when absent.
///
/// A rowid-only operator (`rowid_scan`, the min/max seek) must use [`resolve_table`]
/// instead, so a `WITHOUT ROWID` table can never reach a path that fabricates an integer
/// rowid it does not have.
pub(crate) fn resolve_base_table<'a>(
    catalog: &'a dyn Catalog,
    db: DbIndex,
    name: &str,
) -> Result<&'a TableDef> {
    catalog.table_in(db, name)?.ok_or_else(|| Error::sql(format!("no such table: {name}")))
}

/// Resolve a base table by name for a ROWID-ONLY path — the gate the rowid-keyed
/// operators (`rowid_scan` and the min/max seek) route through, so their shared
/// fail-closed checks cannot drift between call sites.
///
/// * Errors `no such table: {name}` when the table is absent (the exact text the operators
///   produced individually before this was centralized).
/// * Errors loudly on a `WITHOUT ROWID` table. Such a table stores its rows in the
///   primary-key index b-tree, with no integer rowid, but these rowid-keyed paths assume
///   one: [`decode_table_row_enc`] appends an integer rowid register and the rowid
///   range/point seek walks a table b-tree by rowid. Rather than fabricate a bogus rowid
///   and silently read the wrong structure, fail closed. (The WR-aware operators —
///   `seq_scan`, `index_scan`, INSERT / UPDATE / DELETE, the `CREATE INDEX` backfill — use
///   [`resolve_base_table`] and take the index-b-tree path instead, which for a WR
///   secondary index fetches each row by its recovered PRIMARY KEY.)
pub(crate) fn resolve_table<'a>(
    catalog: &'a dyn Catalog,
    db: DbIndex,
    name: &str,
) -> Result<&'a TableDef> {
    let def = resolve_base_table(catalog, db, name)?;
    if def.without_rowid {
        return Err(Error::sql(format!("WITHOUT ROWID table {name} is not supported")));
    }
    Ok(def)
}
