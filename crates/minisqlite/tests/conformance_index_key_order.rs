//! Conformance battery: an index b-tree is ordered by each key column's declared **sort
//! order and collating sequence**, on disk, exactly as sqlite3 orders it.
//!
//! `CREATE INDEX` lets each key column say `ASC` / `DESC` and `COLLATE <name>`, and a key
//! column with no `COLLATE` inherits the table column's declared collation
//! (`spec/sqlite-doc/lang_createindex.html`; `spec/sqlite-doc/datatype3.html` §7). The
//! b-tree keeps its entries in that order (`spec/sqlite-doc/fileformat2.html` §2.5), and
//! sqlite3 seeks and verifies the tree under it: a file whose index is stored in plain
//! ascending BINARY order instead is one sqlite3 reports as corrupt (`PRAGMA
//! integrity_check`: "row N missing from index") and answers wrongly from. A WITHOUT ROWID
//! table's own b-tree is ordered by its PRIMARY KEY the same way
//! (`spec/sqlite-doc/withoutrowid.html`).
//!
//! A self round-trip cannot see this: an engine that writes and reads in the same wrong
//! order answers its own queries correctly. So each case checks the PHYSICAL order of the
//! entries in the file (an in-order walk of the raw b-tree pages, decoded here
//! independently of the engine) against the order sqlite3 uses, and the read case hands
//! the engine an index page laid out the way sqlite3 lays it out. The queries are then
//! checked too, with results worked out from the data by hand.
//!
//! Each case is its own `#[test]`, so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness (the same RAII guard the other on-disk conformance files keep
// locally; only the temp-name prefix differs).
// ---------------------------------------------------------------------------

/// A unique temporary database path, deleted (with its `-journal` / `-wal` / `-shm`
/// sidecars) on drop, so nothing is left behind even when an assertion panics.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_idxord_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let _ = std::fs::remove_file(self.sidecar(suffix));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}

// ---------------------------------------------------------------------------
// Raw page access, written from fileformat2 §1.6 (b-tree pages), §2.1 (records), and
// the varint encoding, so the order checked here does not come from the engine.
// ---------------------------------------------------------------------------

/// The root page number of schema object `name`, and the database page size.
fn root_and_page_size(db: &mut Connection, name: &str) -> (usize, usize) {
    let root = query(db, &format!("SELECT rootpage FROM sqlite_schema WHERE name = '{name}'"));
    let size = query(db, "PRAGMA page_size");
    match (&root.rows[0][0], &size.rows[0][0]) {
        (Value::Integer(r), Value::Integer(s)) => (*r as usize, *s as usize),
        other => panic!("rootpage/page_size of {name}: {other:?}"),
    }
}

fn read_varint(b: &[u8]) -> (u64, usize) {
    let mut v: u64 = 0;
    for (i, &byte) in b.iter().enumerate().take(9) {
        if i == 8 {
            return ((v << 8) | byte as u64, 9);
        }
        v = (v << 7) | (byte & 0x7f) as u64;
        if byte & 0x80 == 0 {
            return (v, i + 1);
        }
    }
    panic!("truncated varint");
}

fn put_varint(out: &mut Vec<u8>, v: u64) {
    assert!(v < 0x80, "this test only writes one-byte varints");
    out.push(v as u8);
}

/// Decode a record (fileformat2 §2.1). Only the serial types these tests store occur.
fn decode_record(rec: &[u8]) -> Vec<Value> {
    let (hdr_len, mut p) = read_varint(rec);
    let mut serials = Vec::new();
    while p < hdr_len as usize {
        let (s, n) = read_varint(&rec[p..]);
        serials.push(s);
        p += n;
    }
    let mut body = hdr_len as usize;
    let mut out = Vec::new();
    for s in serials {
        let int_len = match s {
            1..=4 => s as usize,
            5 => 6,
            6 => 8,
            _ => 0,
        };
        let v = match s {
            0 => Value::Null,
            8 => Value::Integer(0),
            9 => Value::Integer(1),
            1..=6 => {
                let bytes = &rec[body..body + int_len];
                let mut x: i64 = if bytes[0] & 0x80 != 0 { -1 } else { 0 };
                for &b in bytes {
                    x = (x << 8) | b as i64;
                }
                body += int_len;
                Value::Integer(x)
            }
            n if n >= 13 && n % 2 == 1 => {
                let len = ((n - 13) / 2) as usize;
                let t = String::from_utf8(rec[body..body + len].to_vec()).unwrap();
                body += len;
                Value::Text(t)
            }
            other => panic!("serial type {other} not used by these tests"),
        };
        out.push(v);
    }
    out
}

/// Encode a record of small integers and short TEXT values (fileformat2 §2.1).
fn encode_record(values: &[Value]) -> Vec<u8> {
    let mut serials = Vec::new();
    let mut body = Vec::new();
    for v in values {
        match v {
            Value::Integer(x) if (-128..=127).contains(x) => {
                serials.push(1);
                body.push(*x as i8 as u8);
            }
            Value::Integer(x) => {
                let x = i16::try_from(*x).expect("this test only writes 1- or 2-byte integers");
                serials.push(2);
                body.extend_from_slice(&x.to_be_bytes());
            }
            Value::Text(t) => {
                serials.push(13 + 2 * t.len() as u64);
                body.extend_from_slice(t.as_bytes());
            }
            other => panic!("{other:?} not used by these tests"),
        }
    }
    let mut rec = Vec::new();
    put_varint(&mut rec, 1 + serials.len() as u64);
    for s in serials {
        put_varint(&mut rec, s);
    }
    rec.extend_from_slice(&body);
    rec
}

/// Every key record of the index-format b-tree rooted at `root`, in b-tree (in-order)
/// order: for an interior page each divider's left subtree, then the divider itself, and
/// finally the right-most subtree. Keys here are short, so none overflows.
fn btree_keys(file: &[u8], page_size: usize, root: usize) -> Vec<Vec<Value>> {
    fn walk(file: &[u8], page_size: usize, pgno: usize, out: &mut Vec<Vec<Value>>) {
        let page = &file[(pgno - 1) * page_size..pgno * page_size];
        let kind = page[0];
        let cells = u16::from_be_bytes([page[3], page[4]]) as usize;
        let (ptr_base, interior) = match kind {
            0x0a => (8, false),
            0x02 => (12, true),
            other => panic!("page {pgno} has type {other:#x}, not an index b-tree page"),
        };
        for i in 0..cells {
            let off = u16::from_be_bytes([page[ptr_base + 2 * i], page[ptr_base + 2 * i + 1]]) as usize;
            let mut p = off;
            if interior {
                let child = u32::from_be_bytes(page[p..p + 4].try_into().unwrap()) as usize;
                walk(file, page_size, child, out);
                p += 4;
            }
            let (len, n) = read_varint(&page[p..]);
            p += n;
            out.push(decode_record(&page[p..p + len as usize]));
        }
        if interior {
            let right = u32::from_be_bytes(page[8..12].try_into().unwrap()) as usize;
            walk(file, page_size, right, out);
        }
    }
    let mut out = Vec::new();
    walk(file, page_size, root, &mut out);
    out
}

/// Close-then-read helper: the stored key records of `name` (an index, or a WITHOUT ROWID
/// table) in the file's b-tree order.
fn stored_keys(tmp: &TempDb, name: &str) -> Vec<Vec<Value>> {
    let (root, page_size) = root_and_page_size(&mut tmp.open(), name);
    let file = std::fs::read(&tmp.path).unwrap();
    btree_keys(&file, page_size, root)
}

/// Replace the (leaf) root page of index `name` with one holding `keys`, in the given
/// order, laid out as an index leaf page (fileformat2 §1.6): this is the content sqlite3
/// writes for that index.
fn write_index_leaf(tmp: &TempDb, name: &str, keys: &[Vec<Value>]) {
    let (root, page_size) = root_and_page_size(&mut tmp.open(), name);
    assert!(root > 1, "an index root is never page 1");
    let mut page = vec![0u8; page_size];
    let mut content = page_size;
    for (i, key) in keys.iter().enumerate() {
        let rec = encode_record(key);
        let mut cell = Vec::new();
        put_varint(&mut cell, rec.len() as u64);
        cell.extend_from_slice(&rec);
        content -= cell.len();
        page[content..content + cell.len()].copy_from_slice(&cell);
        page[8 + 2 * i..10 + 2 * i].copy_from_slice(&(content as u16).to_be_bytes());
    }
    assert!(8 + 2 * keys.len() <= content, "the keys must fit on one leaf page");
    page[0] = 0x0a;
    page[3..5].copy_from_slice(&(keys.len() as u16).to_be_bytes());
    page[5..7].copy_from_slice(&(content as u16).to_be_bytes());
    let mut file = std::fs::read(&tmp.path).unwrap();
    file[(root - 1) * page_size..root * page_size].copy_from_slice(&page);
    std::fs::write(&tmp.path, file).unwrap();
}

fn assert_keys(what: &str, got: &[Vec<Value>], want: &[Vec<Value>]) {
    let row_eq = |g: &Vec<Value>, w: &Vec<Value>| {
        g.len() == w.len() && g.iter().zip(w).all(|(a, b)| value_eq(a, b))
    };
    let same = got.len() == want.len() && got.iter().zip(want).all(|(g, w)| row_eq(g, w));
    assert!(same, "{what}: stored key order differs\n  expected: {want:?}\n  actual:   {got:?}");
}

fn ikey(b: i64, rowid: i64) -> Vec<Value> {
    vec![int(b), int(rowid)]
}

fn tkey(c: &str, rowid: i64) -> Vec<Value> {
    vec![text(c), int(rowid)]
}

// ---------------------------------------------------------------------------
// DESC index
// ---------------------------------------------------------------------------

/// Reading a `DESC` index stored the way sqlite3 stores it: entries in descending `b`
/// order, ties by ascending rowid. Point and range seeks must find the rows.
#[test]
fn reads_a_desc_index_in_sqlite3_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        exec(&mut db, "CREATE INDEX tb ON t(b DESC)");
        exec(
            &mut db,
            "WITH RECURSIVE s(v) AS (SELECT 1 UNION ALL SELECT v+1 FROM s WHERE v<200) \
             INSERT INTO t(a, b) SELECT v, v FROM s",
        );
    }
    let sqlite3_order: Vec<Vec<Value>> = (1..=200).rev().map(|v| ikey(v, v)).collect();
    write_index_leaf(&tmp, "tb", &sqlite3_order);

    let mut db = tmp.open();
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE b > 150", int(50));
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE b >= 150", int(51));
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE b < 20", int(19));
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE b BETWEEN 100 AND 199", int(100));
    assert_rows(&mut db, "SELECT b FROM t WHERE b = 77", &[vec![int(77)]]);
    assert_rows(
        &mut db,
        "SELECT a FROM t WHERE b > 196 ORDER BY b DESC",
        &[vec![int(200)], vec![int(199)], vec![int(198)], vec![int(197)]],
    );
}

/// Inserting into a `DESC` index (here one sqlite3 laid out, holding `5`) keeps the entries
/// in descending order, so sqlite3 finds every row through the index.
#[test]
fn writes_into_a_desc_index_in_sqlite3_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        exec(&mut db, "CREATE INDEX tb ON t(b DESC)");
        exec(&mut db, "INSERT INTO t(b) VALUES (5)");
    }
    write_index_leaf(&tmp, "tb", &[ikey(5, 1)]);
    {
        let mut db = tmp.open();
        exec(&mut db, "INSERT INTO t(b) VALUES (1), (3), (9)");
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE b > 2", int(3));
        assert_rows(&mut db, "SELECT a FROM t WHERE b = 3", &[vec![int(3)]]);
    }
    assert_keys("tb", &stored_keys(&tmp, "tb"), &[ikey(9, 4), ikey(5, 1), ikey(3, 3), ikey(1, 2)]);
}

/// A `DESC` index minisqlite creates and fills itself, large enough to need interior pages,
/// with duplicate keys and NULLs: stored in sqlite3's order (descending `b`, NULLs last, ties
/// by ascending rowid), and seeks over it return the right rows.
#[test]
fn a_multi_page_desc_index_is_stored_and_searched_in_sqlite3_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        exec(&mut db, "CREATE INDEX tb ON t(b DESC)");
        // b = a / 2 for a in 1..=2000 (each value twice, 0 once, 1000 once), then two NULLs.
        exec(
            &mut db,
            "WITH RECURSIVE s(v) AS (SELECT 1 UNION ALL SELECT v+1 FROM s WHERE v<2000) \
             INSERT INTO t(a, b) SELECT v, v / 2 FROM s",
        );
        exec(&mut db, "INSERT INTO t(a, b) VALUES (2001, NULL), (2002, NULL)");
        exec(&mut db, "DELETE FROM t WHERE a % 7 = 0");
        exec(&mut db, "UPDATE t SET b = b + 1 WHERE a % 5 = 0");

        // Expected answers, computed from the same data without the index.
        let expect = |db: &mut Connection, sql: &str| query(db, sql).rows[0][0].clone();
        for (with_index, without_index) in [
            ("SELECT count(*) FROM t WHERE b > 750", "SELECT count(*) FROM t WHERE b + 0 > 750"),
            ("SELECT count(*) FROM t WHERE b < 13", "SELECT count(*) FROM t WHERE b + 0 < 13"),
            (
                "SELECT count(*) FROM t WHERE b BETWEEN 100 AND 199",
                "SELECT count(*) FROM t WHERE b + 0 BETWEEN 100 AND 199",
            ),
            ("SELECT count(*) FROM t WHERE b = 500", "SELECT count(*) FROM t WHERE b + 0 = 500"),
            ("SELECT count(*) FROM t WHERE b IS NULL", "SELECT count(*) FROM t WHERE b + 0 IS NULL"),
        ] {
            let want = expect(&mut db, without_index);
            assert_scalar(&mut db, with_index, want);
        }
    }
    let mut rows: Vec<(Option<i64>, i64)> = Vec::new();
    for a in 1..=2002i64 {
        if a % 7 == 0 {
            continue;
        }
        let mut b = if a > 2000 { None } else { Some(a / 2) };
        if a % 5 == 0 {
            b = b.map(|v| v + 1);
        }
        rows.push((b, a));
    }
    // sqlite3's DESC order: larger b first, NULL (the smallest value) last, rowid ascending.
    rows.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    let want: Vec<Vec<Value>> =
        rows.iter().map(|(b, a)| vec![b.map(int).unwrap_or_else(null), int(*a)]).collect();
    assert_keys("tb", &stored_keys(&tmp, "tb"), &want);
}

// ---------------------------------------------------------------------------
// Collation
// ---------------------------------------------------------------------------

/// An index on a `COLLATE NOCASE` column inherits NOCASE (lang_createindex.html), so its
/// entries are in case-insensitive order and NOCASE lookups find them.
#[test]
fn an_index_on_a_nocase_column_is_stored_in_nocase_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, c COLLATE NOCASE)");
        exec(&mut db, "CREATE INDEX tc ON t(c)");
        exec(&mut db, "INSERT INTO t(c) VALUES ('m')");
        exec(&mut db, "INSERT INTO t(c) VALUES ('B'), ('a'), ('Z'), ('y')");
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c = 'b'", int(1));
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c > 'x'", int(2));
        assert_rows(&mut db, "SELECT a FROM t WHERE c = 'z'", &[vec![int(4)]]);
    }
    assert_keys(
        "tc",
        &stored_keys(&tmp, "tc"),
        &[tkey("a", 3), tkey("B", 2), tkey("m", 1), tkey("y", 5), tkey("Z", 4)],
    );
}

/// `COLLATE NOCASE` on the index column of a BINARY column orders the index under NOCASE,
/// while the column's own (BINARY) comparisons still answer under BINARY.
#[test]
fn a_collate_nocase_index_column_is_stored_in_nocase_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, c)");
        exec(&mut db, "CREATE INDEX tc ON t(c COLLATE NOCASE)");
        exec(&mut db, "INSERT INTO t(c) VALUES ('m')");
        exec(&mut db, "INSERT INTO t(c) VALUES ('B'), ('a'), ('Z'), ('y')");
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c = 'b'", int(0));
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c = 'b' COLLATE NOCASE", int(1));
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c > 'x'", int(1));
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE c > 'x' COLLATE NOCASE", int(2));
    }
    assert_keys(
        "tc",
        &stored_keys(&tmp, "tc"),
        &[tkey("a", 3), tkey("B", 2), tkey("m", 1), tkey("y", 5), tkey("Z", 4)],
    );
}

/// A UNIQUE NOCASE index still rejects a value equal to an existing one under NOCASE.
#[test]
fn a_unique_nocase_index_rejects_a_case_variant() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, c COLLATE NOCASE UNIQUE)");
    exec(&mut db, "INSERT INTO t(c) VALUES ('m'), ('B'), ('a'), ('Z'), ('y')");
    assert_exec_error(&mut db, "INSERT INTO t(c) VALUES ('b')");
    assert_exec_error(&mut db, "UPDATE t SET c = 'z' WHERE c = 'y'");
    exec(&mut db, "UPDATE t SET c = 'M' WHERE c = 'm'");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(5));
}

// ---------------------------------------------------------------------------
// WITHOUT ROWID
// ---------------------------------------------------------------------------

/// A WITHOUT ROWID table's b-tree is ordered by its PRIMARY KEY, column by column in the
/// declared direction: `PRIMARY KEY (y DESC, x)` stores larger `y` first, ties by ascending
/// `x`. The stored record is the PK columns, then the rest (fileformat2 §2.4).
#[test]
fn a_without_rowid_desc_primary_key_is_stored_in_sqlite3_order() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(x, y, z, PRIMARY KEY (y DESC, x)) WITHOUT ROWID");
        exec(&mut db, "INSERT INTO t VALUES (1, 10, 'a'), (2, 30, 'b'), (3, 20, 'c')");
        exec(&mut db, "INSERT INTO t VALUES (4, 30, 'd'), (0, 30, 'e'), (5, 10, 'f')");
        exec(&mut db, "UPDATE t SET y = 25 WHERE x = 3");
        exec(&mut db, "DELETE FROM t WHERE x = 4");
        assert_exec_error(&mut db, "INSERT INTO t VALUES (0, 30, 'dup')");
        assert_rows(&mut db, "SELECT z FROM t WHERE y = 30 AND x = 2", &[vec![text("b")]]);
    }
    let row = |y: i64, x: i64, z: &str| vec![int(y), int(x), text(z)];
    assert_keys(
        "t",
        &stored_keys(&tmp, "t"),
        &[row(30, 0, "e"), row(30, 2, "b"), row(25, 3, "c"), row(10, 1, "a"), row(10, 5, "f")],
    );
}

/// A secondary index on a WITHOUT ROWID table appends the PRIMARY KEY columns, ordered as
/// the PRIMARY KEY orders them (fileformat2 §2.5.1): here `y` descending after `z`.
#[test]
fn a_without_rowid_secondary_index_orders_its_pk_suffix_like_the_pk() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(x, y, z, PRIMARY KEY (y DESC, x)) WITHOUT ROWID");
        exec(&mut db, "CREATE INDEX tz ON t(z)");
        exec(&mut db, "INSERT INTO t VALUES (1, 10, 'a'), (2, 30, 'a'), (3, 20, 'b'), (1, 20, 'a')");
        assert_rows(&mut db, "SELECT x, y FROM t WHERE z = 'b'", &[vec![int(3), int(20)]]);
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE z = 'a'", int(3));
    }
    let key = |z: &str, y: i64, x: i64| vec![text(z), int(y), int(x)];
    assert_keys(
        "tz",
        &stored_keys(&tmp, "tz"),
        &[key("a", 30, 2), key("a", 20, 1), key("a", 10, 1), key("b", 20, 3)],
    );
}

/// A `COLLATE NOCASE` PRIMARY KEY of a WITHOUT ROWID table treats `'a'` and `'A'` as the
/// same key, like sqlite3 (the b-tree is keyed under the PK column's collation).
#[test]
fn a_without_rowid_nocase_primary_key_rejects_a_case_variant() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k TEXT COLLATE NOCASE PRIMARY KEY, v) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES ('a', 1), ('B', 2)");
    assert_exec_error(&mut db, "INSERT INTO w VALUES ('A', 3)");
    exec(&mut db, "INSERT OR REPLACE INTO w VALUES ('b', 4)");
    // Rewriting a key to a case variant of itself is not a PRIMARY KEY change.
    exec(&mut db, "UPDATE w SET k = 'A' WHERE k = 'a'");
    exec(&mut db, "INSERT INTO w VALUES ('x', 5)");
    exec(&mut db, "INSERT INTO w VALUES ('X', 7) ON CONFLICT(k) DO UPDATE SET k = 'x', v = 8");
    assert_rows(
        &mut db,
        "SELECT k, v FROM w ORDER BY k",
        &[vec![text("A"), int(1)], vec![text("b"), int(4)], vec![text("x"), int(8)]],
    );
}
