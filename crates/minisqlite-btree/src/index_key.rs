//! Index-key comparison and cell-payload access — the ordering the whole index
//! b-tree is built on (fileformat2 §2.2 record sort order, §2.5 index keys).
//!
//! An index key is a *record*: the indexed columns followed by the table row key
//! (the rowid). Two keys compare column-by-column in the storage-class order
//! (`minisqlite_types::compare_values`): the first non-Equal column decides, and if
//! one record runs out of columns while every compared column was Equal, the SHORTER
//! record is `Less`. That last rule is load-bearing: a *prefix* search key (the
//! indexed columns without a trailing rowid) then sorts before every full key that
//! shares that prefix, which is exactly what a point/range index seek needs to land
//! on the first matching entry.
//!
//! Sort order and collation: each column compares under the collating sequence and
//! sort order the index declares for it, carried in a [`KeyOrder`] (sqlite3's
//! `KeyInfo`). A `DESC` column reverses that column's whole comparison, storage class
//! included, so its NULLs sort last (fileformat2 §2.2 record sort order, applied per
//! column in the index's declared direction). A TEXT column under `NOCASE` / `RTRIM` compares
//! its raw body bytes under that sequence. Columns past the end of the [`KeyOrder`]
//! compare ascending under BINARY, which is the rule for the trailing rowid of a
//! rowid-table index. The caller (the executor) builds the [`KeyOrder`] from the index
//! definition, since this layer does not see the schema; every seek, insert, and
//! delete on one b-tree must use the same one, or keys land out of order.

use std::borrow::Cow;
use std::cmp::Ordering;

use minisqlite_fileformat::{read_serial_value, RecordCursor};
use minisqlite_pager::Pager;
use minisqlite_types::{compare_text_bytes, compare_values, Collation, Result, Value};

/// How one key column is ordered: its collating sequence (consulted only when both
/// sides are TEXT) and whether the column is `DESC`. The default is ascending BINARY.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyField {
    pub collation: Collation,
    pub descending: bool,
}

/// The per-column ordering of an index b-tree's key records (sqlite3's `KeyInfo`).
/// Field `i` orders record column `i`; record columns past the last field compare
/// ascending under BINARY. [`KeyOrder::default`] (no fields) is the all-ascending
/// BINARY order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeyOrder {
    fields: Vec<KeyField>,
}

impl KeyOrder {
    /// A key order with one [`KeyField`] per leading record column.
    pub fn new(fields: Vec<KeyField>) -> KeyOrder {
        KeyOrder { fields }
    }

    /// The ordering of record column `i` (ascending BINARY past the last field).
    #[inline]
    pub fn field(&self, i: usize) -> KeyField {
        self.fields.get(i).copied().unwrap_or_default()
    }

    /// Compare two decoded values of record column `i` the way the b-tree orders that
    /// column: [`compare_values`] under the column's collation, reversed for `DESC`.
    /// Agrees with [`compare_index_keys`] column by column, so a caller walking a cursor
    /// can decide "still in range" with the order the cursor seeks in.
    pub fn compare_column(&self, i: usize, a: &Value, b: &Value) -> Ordering {
        let f = self.field(i);
        let ord = compare_values(a, b, f.collation);
        if f.descending { ord.reverse() } else { ord }
    }
}

/// The storage-class sort rank of a serial type — NULL(0) < numeric(1) < TEXT(2) <
/// BLOB(3) — matching `minisqlite_types::storage_class_rank` without decoding a
/// `Value`. Serials 10/11 are reserved and read back as NULL; integers (1-6, 8, 9)
/// and REAL (7) are one numeric class.
#[inline]
fn serial_class(serial: u64) -> u8 {
    match serial {
        0 | 10 | 11 => 0,     // NULL (10/11 are reserved and decode to NULL)
        1..=9 => 1,           // numeric: integers (1-6, 8, 9) and REAL (7)
        n if n & 1 == 0 => 3, // even >= 12 -> BLOB
        _ => 2,               // odd >= 13 -> TEXT
    }
}

/// Compare two encoded index-key records under the index sort order `order`. Iterates
/// both records column-by-column and returns the first non-Equal result (reversed for a
/// `DESC` column). A record that ends first, all compared columns equal, is `Less` (so a
/// prefix key sorts before any longer key sharing that prefix, whatever the sort order).
/// Equal only when both end together with every column Equal.
///
/// Hot path — this is the comparator inside every interior/leaf binary search, so it
/// allocates nothing. Column bodies are compared borrowed: two TEXT columns are compared
/// on their raw body bytes under the column's collation (`memcmp` for BINARY, exactly
/// what `compare_values` computes, and the sqlite-faithful raw-byte order even for
/// non-UTF-8 TEXT), two BLOB columns are `memcmp`, and only the numeric classes are
/// decoded — an integer/real decode is itself alloc-free — then routed through
/// `compare_values` for the exact cross-type numeric ordering. Cross-class pairs order by
/// storage rank.
pub(crate) fn compare_index_keys(a: &[u8], b: &[u8], order: &KeyOrder) -> Ordering {
    let mut ca = RecordCursor::new(a);
    let mut cb = RecordCursor::new(b);
    let mut i = 0usize;
    loop {
        match (ca.next(), cb.next()) {
            (Some((sa, ba)), Some((sb, bb))) => {
                let field = order.field(i);
                i += 1;
                let (cla, clb) = (serial_class(sa), serial_class(sb));
                let ord = if cla != clb {
                    cla.cmp(&clb)
                } else {
                    match cla {
                        // Both NULL: equal, keep comparing later columns.
                        0 => Ordering::Equal,
                        // Both TEXT: raw-byte compare under the column's collation, no alloc.
                        2 => compare_text_bytes(ba, bb, field.collation),
                        // Both BLOB: raw-byte compare (collation never applies to a BLOB).
                        3 => ba.cmp(bb),
                        // Both numeric: decode (alloc-free) for the exact int/real order.
                        _ => compare_values(
                            &read_serial_value(sa, ba),
                            &read_serial_value(sb, bb),
                            Collation::Binary,
                        ),
                    }
                };
                match ord {
                    Ordering::Equal => continue,
                    non_eq if field.descending => return non_eq.reverse(),
                    non_eq => return non_eq,
                }
            }
            // The shorter record is Less once every shared column compared Equal.
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (None, None) => return Ordering::Equal,
        }
    }
}

/// The full comparison bytes for an index cell's key payload.
///
/// A cell fully inline on its page has its whole key record as the local payload,
/// returned `Cow::Borrowed` with no copy. An overflowed key keeps only its inline
/// prefix on the page, so its spilled tail is reassembled from the overflow chain via
/// [`crate::overflow_io::read_payload`] (returned `Cow::Owned`). Reassembly is
/// mandatory wherever a stored key is compared: the inline prefix alone can tie two
/// distinct keys, so comparing on it would misroute inserts and seeks.
pub(crate) fn cell_key_bytes<'p>(
    pager: &'p dyn Pager,
    local: &'p [u8],
    payload_len: u64,
    overflow_page: Option<u32>,
    usable: usize,
) -> Result<Cow<'p, [u8]>> {
    crate::overflow_io::read_payload(pager, local, payload_len, overflow_page, usable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_fileformat::{encode_record, CellKind};
    use minisqlite_pager::MemPager;
    use minisqlite_types::Value;
    use std::cmp::Ordering::*;

    fn cmp(a: &[Value], b: &[Value]) -> Ordering {
        compare_index_keys(&encode_record(a), &encode_record(b), &KeyOrder::default())
    }

    fn cmp_with(a: &[Value], b: &[Value], fields: &[KeyField]) -> Ordering {
        compare_index_keys(&encode_record(a), &encode_record(b), &KeyOrder::new(fields.to_vec()))
    }

    const DESC: KeyField = KeyField { collation: Collation::Binary, descending: true };
    const ASC: KeyField = KeyField { collation: Collation::Binary, descending: false };
    const NOCASE: KeyField = KeyField { collation: Collation::NoCase, descending: false };

    #[test]
    fn first_differing_column_decides() {
        assert_eq!(cmp(&[Value::Integer(1), Value::Integer(9)], &[Value::Integer(2), Value::Integer(0)]), Less);
        assert_eq!(cmp(&[Value::Integer(5), Value::Integer(1)], &[Value::Integer(5), Value::Integer(2)]), Less);
        assert_eq!(cmp(&[Value::Integer(5), Value::Integer(3)], &[Value::Integer(5), Value::Integer(2)]), Greater);
        assert_eq!(cmp(&[Value::Integer(5), Value::Integer(2)], &[Value::Integer(5), Value::Integer(2)]), Equal);
    }

    #[test]
    fn shorter_prefix_sorts_less_than_full_key() {
        // A prefix (indexed columns without the trailing rowid) is Less than any
        // full key sharing the prefix — the property index seeks rely on.
        assert_eq!(cmp(&[Value::Integer(7)], &[Value::Integer(7), Value::Integer(-100)]), Less);
        assert_eq!(cmp(&[Value::Integer(7), Value::Integer(-100)], &[Value::Integer(7)]), Greater);
        // But a prefix is still ordered by its own first column vs the other's.
        assert_eq!(cmp(&[Value::Integer(7)], &[Value::Integer(6), Value::Integer(9999)]), Greater);
        assert_eq!(cmp(&[Value::Integer(7)], &[Value::Integer(8), Value::Integer(-9999)]), Less);
    }

    #[test]
    fn storage_class_order_across_types() {
        // NULL < numeric < text < blob (datatype3 §4).
        assert_eq!(cmp(&[Value::Null], &[Value::Integer(-9)]), Less);
        assert_eq!(cmp(&[Value::Integer(9)], &[Value::Text("a".into())]), Less);
        assert_eq!(cmp(&[Value::Text("z".into())], &[Value::Blob(vec![0])]), Less);
        // Two NULLs in the first column tie, then the rowid breaks it.
        assert_eq!(cmp(&[Value::Null, Value::Integer(1)], &[Value::Null, Value::Integer(2)]), Less);
    }

    #[test]
    fn text_and_numeric_within_class() {
        assert_eq!(cmp(&[Value::Text("abc".into())], &[Value::Text("abd".into())]), Less);
        // Numeric compares by value even across integer/real.
        assert_eq!(cmp(&[Value::Integer(2)], &[Value::Real(2.5)]), Less);
        assert_eq!(cmp(&[Value::Real(2.0)], &[Value::Integer(2)]), Equal);
    }

    #[test]
    fn desc_column_reverses_its_comparison_including_storage_class() {
        let t = |s: &str| Value::Text(s.into());
        assert_eq!(cmp_with(&[Value::Integer(1)], &[Value::Integer(2)], &[DESC]), Greater);
        assert_eq!(cmp_with(&[t("b")], &[t("a")], &[DESC]), Less);
        // NULL sorts last in a DESC column (the whole comparison is reversed).
        assert_eq!(cmp_with(&[Value::Null], &[Value::Integer(-9)], &[DESC]), Greater);
        // Only the DESC column is reversed: column 1 (ASC) and the trailing rowid
        // (past the fields, ascending BINARY) keep their natural order.
        let k = |a: i64, b: i64, r: i64| [Value::Integer(a), Value::Integer(b), Value::Integer(r)];
        assert_eq!(cmp_with(&k(5, 1, 0), &k(5, 2, 0), &[DESC, ASC]), Less);
        assert_eq!(cmp_with(&k(5, 1, 1), &k(5, 1, 2), &[DESC, ASC]), Less);
        assert_eq!(cmp_with(&k(6, 1, 1), &k(5, 1, 2), &[DESC, ASC]), Less);
        // A prefix still sorts before every full key that extends it.
        assert_eq!(cmp_with(&[Value::Integer(7)], &k(7, 0, 0), &[DESC]), Less);
    }

    #[test]
    fn text_column_compares_under_its_collation() {
        let t = |s: &str| Value::Text(s.into());
        assert_eq!(cmp_with(&[t("B")], &[t("a")], &[NOCASE]), Greater);
        assert_eq!(cmp(&[t("B")], &[t("a")]), Less);
        // Equal under NOCASE, so the rowid decides.
        let (abc2, abc1) = ([t("abc"), Value::Integer(2)], [t("ABC"), Value::Integer(1)]);
        assert_eq!(cmp_with(&abc2, &abc1, &[NOCASE]), Greater);
        let rtrim = KeyField { collation: Collation::Rtrim, descending: false };
        assert_eq!(cmp_with(&[t("a  ")], &[t("a")], &[rtrim]), Equal);
        // A collation never applies across storage classes or to BLOBs.
        assert_eq!(cmp_with(&[Value::Integer(9)], &[t("A")], &[NOCASE]), Less);
        let (b_upper, a_lower) = (Value::Blob(b"B".to_vec()), Value::Blob(b"a".to_vec()));
        assert_eq!(cmp_with(&[b_upper], &[a_lower], &[NOCASE]), Less);
    }

    #[test]
    fn cell_key_bytes_borrows_inline() {
        let rec = encode_record(&[Value::Integer(1), Value::Integer(2)]);
        let pager = MemPager::new(4096);
        let got = cell_key_bytes(&pager, &rec, rec.len() as u64, None, 4096).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)), "an inline key is borrowed, not copied");
        assert_eq!(got.as_ref(), rec.as_slice());
    }

    #[test]
    fn cell_key_bytes_reassembles_overflowed_key() {
        // A long TEXT key spills on a 512-byte page; cell_key_bytes must rebuild the
        // full record from its inline prefix + overflow chain, byte-exact.
        let usable = 512usize;
        let mut pager = MemPager::new(usable as u32);
        pager.allocate_page().unwrap(); // reserve page 1 so overflow ids are realistic
        let rec = encode_record(&[Value::Text("z".repeat(400)), Value::Integer(1)]);
        let (inline_len, first) =
            crate::overflow_io::write_overflow_chain(&mut pager, &rec, CellKind::Index, usable)
                .unwrap();
        let first = first.expect("a 400-char text key must overflow a 512-byte page");
        let got =
            cell_key_bytes(&pager, &rec[..inline_len], rec.len() as u64, Some(first), usable)
                .unwrap();
        assert!(matches!(got, Cow::Owned(_)), "a spilled key is reassembled into an owned buffer");
        assert_eq!(got.as_ref(), rec.as_slice());
    }
}
