//! WITHOUT ROWID (WR) table storage: the layout and index-b-tree primitives the WR
//! read/write path shares, kept in one file so INSERT and the scan cannot disagree on
//! how a WR row is encoded, ordered, or probed.
//!
//! A WR table stores each row in an INDEX b-tree keyed by its PRIMARY KEY
//! (`withoutrowid.html`; fileformat2 §2.4) — there is NO integer rowid. The on-disk key
//! record is the PRIMARY KEY columns (in PK-declared order) followed by every remaining
//! column (in CREATE TABLE order); its value encoding is otherwise a normal record. The
//! table's own b-tree root (`TableDef.root_page`) IS that PK index, so the executor
//! reads/writes it with the same [`IndexCursor`] / `index_insert` primitives a secondary
//! index uses — the row simply plays the part of the index key record.
//!
//! This module owns:
//! * [`WrLayout`] — the permutation between the executor's schema-order logical row
//!   `[c0..c_{N-1}]` and the storage-order key record (PK columns first);
//! * the PRIMARY KEY column derivation from a [`TableDef`] (which carries no explicit
//!   PK-order field — see [`wr_pk_columns`]);
//! * encode (schema → storage) / decode (storage → schema) so a scan reconstructs the
//!   original column order without a fabricated rowid;
//! * the PRIMARY KEY's per-column sort order and collation ([`WrLayout::order`]), the
//!   [`KeyOrder`] every read and write of the PK b-tree must use;
//! * the PRIMARY KEY uniqueness probe ([`WrLayout::pk_conflict_key`]);
//! * the implicit `NOT NULL` on every PRIMARY KEY column ([`WrLayout::enforce_pk_not_null`]).

use std::cmp::Ordering;

use minisqlite_btree::{IndexCursor, KeyField, KeyOrder};
use minisqlite_catalog::TableDef;
use minisqlite_fileformat::{
    decode_record_enc, decode_record_into_enc, encode_record_enc, TextEncoding,
};
use minisqlite_pager::{PageId, Pager};
use minisqlite_plan::OnConflict;
use minisqlite_types::{compare_values, ConstraintKind, Error, Result, Row, Value};

use crate::ops::constraints::ConstraintOutcome;
use crate::ops::dml_index::collation_of;
use crate::row::is_virtual_generated;

/// The storage layout of a WITHOUT ROWID table's b-tree key record: a permutation from
/// storage position to schema column, with the leading `pk_len` positions being the
/// PRIMARY KEY columns (the uniqueness-determining, ordering prefix).
///
/// `perm[j]` is the SCHEMA column index stored at storage slot `j`. Every PHYSICALLY
/// STORED column appears exactly once, so encode/decode is a total, lossless reordering
/// of the stored columns. The first `pk_len` slots are the PRIMARY KEY columns in PK
/// order (deduped, §2.4.1); the rest are the non-PK STORED columns in CREATE TABLE order.
///
/// A VIRTUAL generated column is NOT stored in the record (`gencol.html`), so it is
/// OMITTED from `perm` entirely — it takes no storage slot, and decode leaves its schema
/// register a `Value::Null` placeholder for the scan to compute. A generated column can
/// never be part of the PRIMARY KEY, so the PK prefix is always physically stored. When
/// the table has no virtual column `perm.len() == N` and the layout is byte-identical to
/// before, so a non-generated WR table pays nothing.
pub(crate) struct WrLayout {
    perm: Vec<usize>,
    pk_len: usize,
    /// The b-tree key order: one [`KeyField`] per PRIMARY KEY column (its `DESC` flag and
    /// effective collation), in PK order. The non-PK columns after them never decide an
    /// order (the PK is unique), so they are left at the ascending BINARY default.
    order: KeyOrder,
}

impl WrLayout {
    /// The schema-column indices of the PRIMARY KEY columns, in PK (storage-prefix)
    /// order. These are exactly the columns that determine b-tree order and uniqueness,
    /// and the columns a WR table implicitly makes `NOT NULL`.
    pub(crate) fn pk_columns(&self) -> &[usize] {
        &self.perm[..self.pk_len]
    }

    /// The PK b-tree's key order (each PRIMARY KEY column's sort order and collation,
    /// sqlite3's `KeyInfo` for the PK index). Every cursor, insert, and delete on the WR
    /// table's own b-tree must use it, so the tree stays in the order sqlite3 expects.
    pub(crate) fn order(&self) -> &KeyOrder {
        &self.order
    }

    /// The ordering of PRIMARY KEY column `i` (in PK order).
    pub(crate) fn pk_field(&self, i: usize) -> KeyField {
        self.order.field(i)
    }

    /// Reorder a schema-order logical row `[c0..c_{N-1}]` into the storage-order values
    /// of the on-disk key record (PK columns first), OMITTING any VIRTUAL generated column
    /// (absent from `perm`, so never stored — `gencol.html`). `logical` must be width `N`
    /// with every STORED generated value already computed; the result is exactly the
    /// record a real `sqlite3` reads back.
    pub(crate) fn storage_values(&self, logical: &[Value]) -> Vec<Value> {
        self.perm.iter().map(|&i| logical[i].clone()).collect()
    }

    /// The PRIMARY KEY values of a schema-order logical row, in PK order — the prefix a
    /// uniqueness probe / seek is built from.
    pub(crate) fn pk_values(&self, logical: &[Value]) -> Vec<Value> {
        self.perm[..self.pk_len].iter().map(|&i| logical[i].clone()).collect()
    }

    /// Decode one stored WR key record back into a schema-order row `[c0..c_{N-1}]`
    /// (width `N`, NO trailing rowid). The record is stored in PK-first storage order,
    /// so each stored slot `j` is MOVED back to schema column `perm[j]`.
    ///
    /// This runs once per scanned row on a full WR scan (linear in table size — the
    /// hot read path), so it mirrors the discipline of the rowid decode
    /// ([`crate::row::decode_table_row_enc`]): no value is cloned, and only the columns the
    /// record did NOT store get a default. `scratch` is a caller-owned buffer decoded
    /// into and drained each call (the scan cursor reuses one across all rows), so the
    /// only per-row allocation is the returned row's spine — the stored TEXT/BLOB
    /// payloads are moved through, never copied.
    ///
    /// A short record (fewer stored values than the STORED-column count — a row written
    /// before a later `ADD COLUMN`) leaves the uncovered SCHEMA columns at their
    /// materialized DEFAULT (NULL when none), matching the rowid decode's short-row rule.
    /// Added columns are always non-PK and land in the trailing storage slots, so
    /// `perm[take..]` are exactly those uncovered columns and a short record never drops a
    /// PK column.
    ///
    /// A VIRTUAL generated column is absent from `perm`, so its schema register is never
    /// assigned here — it stays the `Value::Null` placeholder the spine is initialized to,
    /// which the scan cursor then overwrites with the computed value.
    ///
    /// TEXT columns decode in the database's text encoding `enc` (§1.3.13), so a UTF-16
    /// WITHOUT ROWID table's stored columns transcode to the engine's internal UTF-8. A
    /// UTF-8 database passes `TextEncoding::Utf8` and the decode is byte-identical to before.
    pub(crate) fn decode_row_enc(
        &self,
        payload: &[u8],
        def: &TableDef,
        scratch: &mut Vec<Value>,
        enc: TextEncoding,
    ) -> Row {
        let n = def.columns.len();
        // Decode the PK-first storage values into the reusable scratch, then move each
        // into its schema slot. `drain(..).take(take)` yields the covered values by move
        // and (Drain-on-drop) empties the rest, so scratch keeps its capacity for the
        // next row. The `Value::Null` fill sizes the spine; every STORED slot is then
        // overwritten (moved value or default), and any VIRTUAL column (not in `perm`)
        // is deliberately left NULL as the scan's compute placeholder.
        decode_record_into_enc(payload, enc, scratch);
        // Bound by the number of STORED columns (`perm.len()`), which is `N` minus the
        // virtual columns — never `N` when the table has a VIRTUAL column, so a record's
        // values map only onto physically-stored slots.
        let take = scratch.len().min(self.perm.len());
        let mut row: Row = vec![Value::Null; n];
        for (j, v) in scratch.drain(..).take(take).enumerate() {
            row[self.perm[j]] = v;
        }
        // Empty for a full-width record (`take == perm.len()`), so the common path pays
        // nothing; only a short row's uncovered tail materializes a default (a per-row
        // clone that runs solely on that tail, and only when the DEFAULT is a heap value).
        for &schema_i in &self.perm[take..] {
            row[schema_i] = def.columns[schema_i].default_value.clone().unwrap_or(Value::Null);
        }
        row
    }

    /// Whether the WR b-tree at `root` already holds a row whose PRIMARY KEY equals
    /// `pk_values`, returning that existing row's FULL key-record bytes (so a REPLACE can
    /// delete it) or `None` when there is no conflict.
    ///
    /// The probe seeks the PK prefix under the PK b-tree's [`order`](Self::order) and
    /// compares the landed entry's leading PK columns value-wise under each column's
    /// collation — the same ordering the b-tree itself is built on, so "is this PK
    /// present?" here can never disagree with where an insert would place the key. A
    /// `COLLATE NOCASE` PK column therefore treats `'a'` and `'A'` as the same key, as
    /// sqlite3 does. WR PK columns are implicitly `NOT NULL` (enforced before this call), so
    /// there is no NULL-distinctness subtlety.
    pub(crate) fn pk_conflict_key(
        &self,
        pager: &dyn Pager,
        root: PageId,
        pk_values: &[Value],
        enc: TextEncoding,
    ) -> Result<Option<Vec<u8>>> {
        // The PK probe key is encoded in the database's text encoding so its raw-byte
        // compare in `seek_ge` lines up with the stored (same-encoding) WR key records.
        // `enc` is passed by the caller (read once per statement), not re-derived per row.
        let prefix = encode_record_enc(pk_values, enc);
        let mut cur = IndexCursor::open_with_order(pager, root, self.order.clone())?;
        // A prefix key sorts before every full record sharing that prefix, so `seek_ge`
        // lands on the first entry whose PK columns are >= the target PK.
        if !cur.seek_ge(&prefix)? {
            return Ok(None);
        }
        let found = cur.key()?;
        let decoded = decode_record_enc(found.as_ref(), enc);
        if decoded.len() < pk_values.len() {
            // A record shorter than the PK is a corrupt WR entry (every stored row holds
            // at least its PK); do not manufacture a false conflict.
            return Ok(None);
        }
        for (i, want) in pk_values.iter().enumerate() {
            if compare_values(&decoded[i], want, self.pk_field(i).collation) != Ordering::Equal {
                return Ok(None); // The landed entry has a different PK — no conflict.
            }
        }
        Ok(Some(found.into_owned()))
    }

    /// Whether two PRIMARY KEY value tuples (in PK order) are the SAME key: equal column by
    /// column under each PK column's collation, the order the b-tree is keyed on. So under a
    /// `COLLATE NOCASE` PK, `'a'` and `'A'` are one key, and rewriting one to the other is not
    /// a PRIMARY KEY change.
    pub(crate) fn pk_equal(&self, a: &[Value], b: &[Value]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).enumerate().all(|(i, (x, y))| {
                compare_values(x, y, self.pk_field(i).collation) == Ordering::Equal
            })
    }

    /// Enforce the implicit `NOT NULL` on every PRIMARY KEY column of a WITHOUT ROWID
    /// table (`withoutrowid.html`: "NOT NULL is enforced on every column of the PRIMARY
    /// KEY in a WITHOUT ROWID table"). A NULL in any PK column resolves under
    /// `on_conflict` exactly as [`super::constraints::enforce_not_null`] does an explicit
    /// `NOT NULL`: `IGNORE` skips the row, every other policy raises the `NOT NULL`
    /// constraint error naming the first offending PK column (in PK order).
    pub(crate) fn enforce_pk_not_null(
        &self,
        def: &TableDef,
        logical: &[Value],
        on_conflict: &OnConflict,
    ) -> Result<ConstraintOutcome> {
        for &j in self.pk_columns() {
            if logical[j].is_null() {
                return match on_conflict {
                    OnConflict::Ignore => Ok(ConstraintOutcome::Skip),
                    OnConflict::Replace
                    | OnConflict::Abort
                    | OnConflict::Fail
                    | OnConflict::Rollback => Err(Error::constraint(
                        ConstraintKind::NotNull,
                        format!("{}.{}", def.name, def.columns[j].name),
                    )),
                };
            }
        }
        Ok(ConstraintOutcome::Proceed)
    }
}

/// Build the [`WrLayout`] for a WITHOUT ROWID table, or a loud error when its PRIMARY
/// KEY column order cannot be recovered from the [`TableDef`] (see [`wr_pk_columns`];
/// a WR table always has a PRIMARY KEY, so an empty derivation means "unrecoverable
/// catalog", never "no PK", and failing closed beats guessing the key columns).
pub(crate) fn wr_layout(def: &TableDef) -> Result<WrLayout> {
    debug_assert!(def.without_rowid, "wr_layout on a rowid table {}", def.name);
    let n = def.columns.len();
    let pk = wr_pk_columns(def)?;
    if pk.is_empty() {
        return Err(Error::sql(format!(
            "WITHOUT ROWID table {} has no recoverable PRIMARY KEY column order \
             (unrecoverable catalog: a WITHOUT ROWID table always has a PRIMARY KEY)",
            def.name
        )));
    }
    // perm = PK columns (deduped, §2.4.1) followed by the remaining STORED columns in
    // schema order. `in_pk` both dedups the PK prefix and marks which columns to append
    // after. `wr_pk_columns` already resolved each PK entry to an in-range schema index
    // (failing loud on an unknown name), so `p < n` here is an invariant, not a runtime case.
    let mut in_pk = vec![false; n];
    let mut perm: Vec<usize> = Vec::with_capacity(n);
    for &p in &pk {
        debug_assert!(p < n, "wr_pk_columns returned out-of-range column {p} for {n}-col {}", def.name);
        // A generated column can never be part of the PRIMARY KEY (`gencol.html`), so a PK
        // column is always physically stored. Assert it, so a catalog that ever admitted a
        // VIRTUAL PK column fails loud here rather than encoding a slot with no stored value.
        debug_assert!(
            !is_virtual_generated(&def.columns[p]),
            "WR PRIMARY KEY column {p} of {} is a VIRTUAL generated column (forbidden)",
            def.name
        );
        if !in_pk[p] {
            in_pk[p] = true;
            perm.push(p);
        }
    }
    let pk_len = perm.len();
    // Append the non-PK columns in schema order, SKIPPING VIRTUAL generated columns: a
    // VIRTUAL column is not written to the record (`gencol.html`), so it takes no storage
    // slot. STORED generated columns and ordinary columns are physically present and DO get
    // one. A table with no VIRTUAL column keeps `perm.len() == N` — byte-identical layout.
    for i in 0..n {
        if !in_pk[i] && !is_virtual_generated(&def.columns[i]) {
            perm.push(i);
        }
    }
    let stored_cols = def.columns.iter().filter(|c| !is_virtual_generated(c)).count();
    debug_assert_eq!(
        perm.len(),
        stored_cols,
        "WR storage permutation must cover every STORED column exactly once"
    );
    let order = KeyOrder::new(perm[..pk_len].iter().map(|&p| wr_pk_field(def, p)).collect());
    Ok(WrLayout { perm, pk_len, order })
}

/// The sort order and effective collation of PRIMARY KEY column `p` (a schema index) of a
/// WITHOUT ROWID table, as sqlite3 keys its PK b-tree: the PRIMARY KEY clause's `DESC` and
/// explicit `COLLATE`, else the column's declared collation, else BINARY. Read from the
/// PK's auto-index spec (`emit_row == false`), the same metadata `PRAGMA index_xinfo`
/// reports. An INTEGER PRIMARY KEY has no such spec and is keyed ascending under its
/// column collation.
fn wr_pk_field(def: &TableDef, p: usize) -> KeyField {
    let col = &def.columns[p];
    let kc = def.auto_indexes.iter().find(|s| !s.emit_row).and_then(|spec| {
        let pos = spec.columns.iter().position(|c| c.eq_ignore_ascii_case(&col.name))?;
        spec.key_columns.get(pos)
    });
    let explicit = kc.and_then(|k| k.collation.as_deref());
    KeyField {
        collation: collation_of(explicit.or(col.collation.as_deref())),
        descending: kc.is_some_and(|k| k.descending),
    }
}

/// The PRIMARY KEY columns of a WITHOUT ROWID table, as schema-column indices in PK
/// (storage-prefix) order (§2.4.1). Sourced in priority order:
///
/// 0. [`TableDef::primary_key`] — the catalog builder's explicit, ordered PK-column list,
///    derived uniformly for EVERY PK form: column-level `PRIMARY KEY`, table-level
///    composite `PRIMARY KEY(c1, c2, ...)`, AND the table-level single-column INTEGER
///    PRIMARY KEY (`PRIMARY KEY(a)` with `a INTEGER`). That last shape is the one the
///    reconstruction below CANNOT recover — a table-level integer PK reserves no
///    auto-index ordinal (integer PKs are excluded) and sets no per-column flag (a
///    table-level constraint is not folded into per-column flags) — so preferring the
///    explicit list is what closes that hole. Used whenever populated (its members are
///    already resolved, in-range schema indices).
/// 1. Else the PRIMARY KEY's `auto_indexes` spec (`emit_row == false`): a WR table's
///    non-integer PK reserves an auto-index ordinal but owns no separate index (the
///    table's own b-tree IS the PK index); its `columns` are the PK columns in key order.
/// 2. Else the per-column `primary_key` flag: a column-level `INTEGER PRIMARY KEY` gets no
///    auto-index spec (schematab.html excludes an integer PK) but the flag marks it.
///
/// Fallbacks 1/2 are retained for any `TableDef` built without the explicit list, so no
/// construction path silently loses a PK it could still describe. An ALL-empty result
/// means "unrecoverable" (never "no PK" — a WR table always has one), and [`wr_layout`]
/// fails closed there.
///
/// A PK spec that names a column NOT present in `def.columns` is a corrupt/unsupported
/// catalog, not a recoverable shape: it is a loud [`Error::format`], never a silent drop.
/// Dropping it would yield a too-short `pk_len` and a mis-ordered permutation — an on-disk
/// record layout that diverges from what real sqlite writes (a silent cross-read
/// corruption), so this fails fast at its cause instead.
fn wr_pk_columns(def: &TableDef) -> Result<Vec<usize>> {
    // The authoritative, ordered list covers all PK forms (including the table-level
    // single INTEGER PK the reconstruction below misses), so prefer it when populated.
    if !def.primary_key.is_empty() {
        return Ok(def.primary_key.clone());
    }
    if let Some(spec) = def.auto_indexes.iter().find(|s| !s.emit_row) {
        return spec
            .columns
            .iter()
            .map(|name| {
                def.columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(name))
                    .ok_or_else(|| {
                        Error::format(format!(
                            "WITHOUT ROWID table {} PRIMARY KEY names unknown column {name:?}",
                            def.name
                        ))
                    })
            })
            .collect();
    }
    Ok(def
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.primary_key)
        .map(|(i, _)| i)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::{index_insert, init_database};
    use minisqlite_catalog::{Catalog, SchemaCatalog};
    use minisqlite_fileformat::encode_record;
    use minisqlite_pager::MemPager;
    use minisqlite_sql::{parse, Statement};

    /// Build a WR `TableDef` through the real catalog path (so `auto_indexes` /
    /// `primary_key` flags are populated exactly as production does).
    fn wr_def(create_sql: &str) -> (MemPager, SchemaCatalog, String) {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let mut cat = SchemaCatalog::new();
        let ast = parse(create_sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TABLE");
        };
        let name = stmt.name.name.clone();
        cat.create_table(&mut pager, stmt, create_sql).unwrap();
        (pager, cat, name)
    }

    fn layout_of(cat: &SchemaCatalog, name: &str) -> WrLayout {
        let def = cat.table(name).unwrap().unwrap();
        wr_layout(def).unwrap()
    }

    #[test]
    fn single_column_pk_layout_is_identity() {
        let (_p, cat, t) = wr_def("CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[0], "the lone PK column leads the record");
        // a is PK (slot 0), b follows (slot 1): storage order == schema order here.
        let logical = vec![Value::Integer(1), Value::Text("x".into())];
        let stored = l.storage_values(&logical);
        assert!(matches!(stored[0], Value::Integer(1)));
        assert!(matches!(&stored[1], Value::Text(s) if s == "x"));
    }

    #[test]
    fn pk_not_first_is_permuted_to_the_front() {
        // PK is the SECOND column: storage order must put it first, then the rest in
        // schema order — and decode must invert that exactly.
        let (_p, cat, t) = wr_def("CREATE TABLE t(a, b, PRIMARY KEY(b)) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[1], "b (schema index 1) is the PK");
        let logical = vec![Value::Text("aval".into()), Value::Integer(7)];
        let stored = l.storage_values(&logical);
        assert!(matches!(stored[0], Value::Integer(7)), "PK (b) stored first");
        assert!(matches!(&stored[1], Value::Text(s) if s == "aval"), "non-PK (a) second");
        // Round-trip: encode the storage record, decode back to schema order.
        let def = cat.table(&t).unwrap().unwrap();
        let record = encode_record(&stored);
        let back = l.decode_row_enc(&record, def, &mut Vec::new(), TextEncoding::Utf8);
        assert_eq!(back.len(), 2, "decoded row is width N, no rowid");
        assert!(matches!(&back[0], Value::Text(s) if s == "aval"), "schema col a restored");
        assert!(matches!(back[1], Value::Integer(7)), "schema col b restored");
    }

    #[test]
    fn composite_pk_orders_by_key_then_appends_rest() {
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a, b, c, PRIMARY KEY(c, a)) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[2, 0], "PK key order (c, a) → schema indices 2,0");
        let logical =
            vec![Value::Integer(10), Value::Integer(20), Value::Integer(30)];
        let stored = l.storage_values(&logical);
        // storage = [c, a, b] = [30, 10, 20]
        assert!(matches!(stored[0], Value::Integer(30)));
        assert!(matches!(stored[1], Value::Integer(10)));
        assert!(matches!(stored[2], Value::Integer(20)));
        let def = cat.table(&t).unwrap().unwrap();
        let back = l.decode_row_enc(&encode_record(&stored), def, &mut Vec::new(), TextEncoding::Utf8);
        assert!(matches!(back[0], Value::Integer(10)), "a");
        assert!(matches!(back[1], Value::Integer(20)), "b");
        assert!(matches!(back[2], Value::Integer(30)), "c");
    }

    #[test]
    fn virtual_generated_column_takes_no_storage_slot() {
        // gencol.html: a VIRTUAL generated column is NOT written to the record. It must be
        // absent from the storage permutation, so `storage_values` omits it and `decode_row`
        // leaves its schema register a NULL placeholder for the scan to compute. A record a
        // real sqlite3 writes for this table holds only a and b.
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a PRIMARY KEY, b, c AS (b + 1)) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let l = wr_layout(def).unwrap();
        assert_eq!(l.pk_columns(), &[0], "a is the PK; a generated column is never PK");
        // Only a (PK) and b (ordinary) are stored; c (VIRTUAL) is omitted.
        let logical = vec![Value::Text("k".into()), Value::Integer(5), Value::Integer(6)];
        let stored = l.storage_values(&logical);
        assert_eq!(stored.len(), 2, "the VIRTUAL column c takes no storage slot");
        assert!(matches!(&stored[0], Value::Text(s) if s == "k"), "PK a stored first");
        assert!(matches!(stored[1], Value::Integer(5)), "ordinary b stored, not virtual c");
        // Decode: c is a NULL placeholder (scan computes it); a and b are restored.
        let back = l.decode_row_enc(&encode_record(&stored), def, &mut Vec::new(), TextEncoding::Utf8);
        assert_eq!(back.len(), 3, "decoded row is width N, virtual column included as a slot");
        assert!(matches!(&back[0], Value::Text(s) if s == "k"), "a restored");
        assert!(matches!(back[1], Value::Integer(5)), "b restored");
        assert!(
            matches!(back[2], Value::Null),
            "virtual c is a NULL placeholder for the scan to fill, got {:?}",
            back[2]
        );
    }

    #[test]
    fn stored_generated_column_keeps_its_storage_slot() {
        // gencol.html: a STORED generated column IS physically written, so it keeps a
        // storage slot and round-trips like an ordinary column (no NULL placeholder).
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a PRIMARY KEY, b, c AS (b * 2) STORED) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let l = wr_layout(def).unwrap();
        let logical = vec![Value::Text("k".into()), Value::Integer(5), Value::Integer(10)];
        let stored = l.storage_values(&logical);
        assert_eq!(stored.len(), 3, "a STORED generated column keeps its storage slot");
        let back = l.decode_row_enc(&encode_record(&stored), def, &mut Vec::new(), TextEncoding::Utf8);
        assert!(matches!(back[2], Value::Integer(10)), "STORED c round-trips from the record");
    }

    #[test]
    fn decode_short_record_defaults_the_uncovered_tail() {
        // A record stored before a later `ADD COLUMN` holds fewer values than N. The
        // covered slots must decode by move; the uncovered TAIL column must materialize
        // its DEFAULT, not stay NULL — the branch the move-decode rewrite must preserve.
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a PRIMARY KEY, b, c DEFAULT 7) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let l = wr_layout(def).unwrap();
        // Store only [a, b] (2 of 3 columns): storage order is PK-first = [a, b], so a
        // short record simply omits the trailing non-PK column c.
        let record = encode_record(&[Value::Text("k".into()), Value::Integer(1)]);
        let back = l.decode_row_enc(&record, def, &mut Vec::new(), TextEncoding::Utf8);
        assert_eq!(back.len(), 3, "decoded row is width N even for a short record");
        assert!(matches!(&back[0], Value::Text(s) if s == "k"), "PK a moved from storage");
        assert!(matches!(back[1], Value::Integer(1)), "b moved from storage");
        assert!(
            matches!(back[2], Value::Integer(7)),
            "uncovered c takes its DEFAULT (7), not NULL, got {:?}",
            back[2]
        );
    }

    #[test]
    fn decode_reuses_scratch_without_bleeding_between_rows() {
        // The scan cursor reuses ONE scratch buffer across every row; a decode must
        // fully consume/clear it so row N's values never leak into row N+1. Decode two
        // different records through the same scratch and assert each is exact.
        let (_p, cat, t) = wr_def("CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let l = wr_layout(def).unwrap();
        let mut scratch = Vec::new();
        let r1 = l.decode_row_enc(
            &encode_record(&[Value::Text("k".into()), Value::Integer(1)]),
            def,
            &mut scratch,
            TextEncoding::Utf8,
        );
        let r2 = l.decode_row_enc(
            &encode_record(&[Value::Text("z".into()), Value::Integer(2)]),
            def,
            &mut scratch,
            TextEncoding::Utf8,
        );
        assert!(matches!(&r1[0], Value::Text(s) if s == "k") && matches!(r1[1], Value::Integer(1)));
        assert!(matches!(&r2[0], Value::Text(s) if s == "z") && matches!(r2[1], Value::Integer(2)));
        assert!(scratch.is_empty(), "scratch is drained empty after each decode");
    }

    #[test]
    fn column_level_integer_pk_recovers_via_flag() {
        // No emit_row=false spec exists for an INTEGER PK; the per-column flag carries it.
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[0], "the INTEGER PK column recovered via primary_key flag");
    }

    #[test]
    fn table_level_single_integer_pk_recovers_via_primary_key_list() {
        // The shape the older auto-index/flag reconstruction could NOT recover: a
        // TABLE-LEVEL single-column INTEGER PRIMARY KEY reserves no auto-index ordinal
        // (integer PKs are excluded) and sets no per-column flag (a table-level constraint
        // is not folded into per-column flags). `TableDef.primary_key` carries it, so
        // `wr_layout` now builds instead of failing "no recoverable PRIMARY KEY".
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a INTEGER, b TEXT, PRIMARY KEY(a)) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[0], "table-level INTEGER PK column recovered");
        // Storage round-trips: a (PK) leads the record, then b in schema order.
        let logical = vec![Value::Integer(42), Value::Text("x".into())];
        let stored = l.storage_values(&logical);
        assert!(matches!(stored[0], Value::Integer(42)), "PK a stored first");
        assert!(matches!(&stored[1], Value::Text(s) if s == "x"), "b second");
        let def = cat.table(&t).unwrap().unwrap();
        let back = l.decode_row_enc(&encode_record(&stored), def, &mut Vec::new(), TextEncoding::Utf8);
        assert!(matches!(back[0], Value::Integer(42)), "a restored");
        assert!(matches!(&back[1], Value::Text(s) if s == "x"), "b restored");
    }

    #[test]
    fn table_level_composite_integer_pk_recovers_in_key_order() {
        // A table-level composite PK with an INTEGER member, key order (b, a): the
        // explicit list preserves declaration order so storage puts b before a.
        let (_p, cat, t) =
            wr_def("CREATE TABLE t(a INTEGER, b INTEGER, c TEXT, PRIMARY KEY(b, a)) WITHOUT ROWID");
        let l = layout_of(&cat, &t);
        assert_eq!(l.pk_columns(), &[1, 0], "PK key order (b, a) -> schema indices 1,0");
    }

    #[test]
    fn pk_conflict_key_detects_and_misses_correctly() {
        let (mut pager, cat, t) =
            wr_def("CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let root = def.root_page;
        let l = wr_layout(def).unwrap();

        pager.begin().unwrap();
        // Store one row with PK 'k'. The stored record is [a, b] (a is PK, first).
        let row = vec![Value::Text("k".into()), Value::Integer(1)];
        index_insert(&mut pager, root, &encode_record(&l.storage_values(&row))).unwrap();

        // Same PK → conflict, returning the existing full key bytes.
        let hit =
            l.pk_conflict_key(&pager, root, &[Value::Text("k".into())], TextEncoding::Utf8).unwrap();
        assert!(hit.is_some(), "an existing PK is a conflict");
        // Different PK → no conflict.
        let miss = l
            .pk_conflict_key(&pager, root, &[Value::Text("other".into())], TextEncoding::Utf8)
            .unwrap();
        assert!(miss.is_none(), "an absent PK is not a conflict");
        pager.commit().unwrap();
    }

    #[test]
    fn enforce_pk_not_null_flags_null_pk() {
        let (_p, cat, t) = wr_def("CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID");
        let def = cat.table(&t).unwrap().unwrap();
        let l = wr_layout(def).unwrap();
        // A NULL PK under ABORT is a NOT NULL violation naming t.a.
        let err = l
            .enforce_pk_not_null(def, &[Value::Null, Value::Integer(1)], &OnConflict::Abort)
            .unwrap_err();
        match err {
            Error::Constraint(m) => {
                assert!(m.starts_with("NOT NULL constraint failed"), "kind, got {m:?}");
                assert!(m.contains("t.a"), "names the PK column, got {m:?}");
            }
            other => panic!("expected a NOT NULL Constraint error, got {other:?}"),
        }
        // Under IGNORE it is a Skip, not an error.
        assert!(matches!(
            l.enforce_pk_not_null(def, &[Value::Null, Value::Integer(1)], &OnConflict::Ignore)
                .unwrap(),
            ConstraintOutcome::Skip
        ));
        // A non-NULL PK proceeds.
        assert!(matches!(
            l.enforce_pk_not_null(def, &[Value::Integer(5), Value::Null], &OnConflict::Abort)
                .unwrap(),
            ConstraintOutcome::Proceed
        ));
    }
}
