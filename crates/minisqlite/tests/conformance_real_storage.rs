//! Conformance battery: REAL values that a file stores as integers read back as REAL.
//!
//! sqlite3 writes a REAL-affinity value with no fractional part as an integer to save
//! space and converts it back to floating point when the column is read
//! (`spec/sqlite-doc/datatype3.html` §3, the note on "small floating point values with no
//! fractional component and stored in columns with REAL affinity"). A file sqlite3 wrote
//! therefore holds integer serial types in REAL columns, in table rows and in index keys,
//! and every read of such a column must yield a REAL.
//!
//! The files here get that layout without invoking sqlite3: each REAL-affinity column is
//! first declared with a same-length type name of NUMERIC affinity (`DATE` for `REAL`,
//! `NUMBER` for `DOUBLE`, `MONEY` for `FLOAT`, and `ANY` padded to four bytes in a STRICT
//! table), so integral values are stored as integers, exactly as sqlite3 stores them. The
//! connection is closed, the declared type is swapped in place in the stored `CREATE TABLE`
//! text, and the file is reopened. The expected values come from sqlite3 3.51.3 running
//! the same statements with the REAL types declared from the start, and sqlite3 3.51.3
//! returns the same results on the swapped files.

mod conformance;
use conformance::*;

use minisqlite::Connection;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A unique temporary database path; `Drop` removes the file and its sidecars.
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
        path.push(format!("minisqlite_real_storage_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        Connection::open(&self.path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {:?}: {e:?}", self.path))
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let mut s = self.path.as_os_str().to_os_string();
            s.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(s));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}

/// Run `setup` on a fresh file, close it, then replace each `(from, to)` declared type in
/// the stored schema text (same byte length, so the record layout is unchanged). Every
/// `from` must occur at least once, and the setup must not use it in data.
fn file_with_types(setup: &str, swaps: &[(&str, &str)]) -> TempDb {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, setup);
    }
    let mut bytes = std::fs::read(&tmp.path).expect("read db file");
    for (from, to) in swaps {
        assert_eq!(from.len(), to.len(), "a type swap must keep the length: {from:?} -> {to:?}");
        let (from, to) = (from.as_bytes(), to.as_bytes());
        let mut hits = 0;
        let mut i = 0;
        while i + from.len() <= bytes.len() {
            if &bytes[i..i + from.len()] == from {
                bytes[i..i + from.len()].copy_from_slice(to);
                hits += 1;
                i += from.len();
            } else {
                i += 1;
            }
        }
        assert!(hits > 0, "type {:?} not found in the file", String::from_utf8_lossy(from));
    }
    std::fs::write(&tmp.path, bytes).expect("write db file");
    tmp
}

/// `DATE` has NUMERIC affinity and the same length as `REAL`.
const REAL: &[(&str, &str)] = &[("DATE", "REAL")];

/// The table from the issue: a REAL column and a NUMERIC control column.
fn prices() -> TempDb {
    file_with_types(
        "CREATE TABLE m(id INTEGER PRIMARY KEY, price DATE, n NUMERIC);
         INSERT INTO m VALUES (1, 5.0, 5.0), (2, 2.5, 2.5), (3, 0, 0), (4, -3.0, -3.0);",
        REAL,
    )
}

/// A REAL column with an index, plus a TEXT column the index does not cover.
fn indexed() -> TempDb {
    file_with_types(
        "CREATE TABLE ix(id INTEGER PRIMARY KEY, r DATE, t TEXT);
         INSERT INTO ix VALUES (1, 1, 'a'), (2, 2, 'b'), (3, 3.5, 'c'), (4, 4, 'd'),
                               (5, 'abc', 'e'), (6, NULL, 'f');
         CREATE INDEX ir ON ix(r);",
        REAL,
    )
}

#[test]
fn table_scan_reads_integer_stored_real_as_real() {
    let tmp = prices();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT id, price, typeof(price), price / 2 FROM m ORDER BY id",
        &[
            vec![int(1), real(5.0), text("real"), real(2.5)],
            vec![int(2), real(2.5), text("real"), real(1.25)],
            vec![int(3), real(0.0), text("real"), real(0.0)],
            vec![int(4), real(-3.0), text("real"), real(-1.5)],
        ],
    );
}

#[test]
fn numeric_column_keeps_the_integer() {
    let tmp = prices();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT n, typeof(n) FROM m ORDER BY id",
        &[
            vec![int(5), text("integer")],
            vec![real(2.5), text("real")],
            vec![int(0), text("integer")],
            vec![int(-3), text("integer")],
        ],
    );
}

#[test]
fn text_and_null_in_a_real_column_are_unchanged() {
    let tmp = indexed();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT id, r, typeof(r) FROM ix ORDER BY id",
        &[
            vec![int(1), real(1.0), text("real")],
            vec![int(2), real(2.0), text("real")],
            vec![int(3), real(3.5), text("real")],
            vec![int(4), real(4.0), text("real")],
            vec![int(5), text("abc"), text("text")],
            vec![int(6), null(), text("null")],
        ],
    );
}

#[test]
fn covering_index_scan_reads_key_as_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r, typeof(r) FROM ix INDEXED BY ir WHERE r > 1 ORDER BY r",
        &[
            vec![real(2.0), text("real")],
            vec![real(3.5), text("real")],
            vec![real(4.0), text("real")],
            vec![text("abc"), text("text")],
        ],
    );
    assert_rows(
        &mut db,
        "SELECT id, r, typeof(r) FROM ix INDEXED BY ir WHERE r = 2",
        &[vec![int(2), real(2.0), text("real")]],
    );
}

#[test]
fn index_scan_with_table_fetch_reads_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r, typeof(r), t FROM ix INDEXED BY ir WHERE r > 1 ORDER BY r",
        &[
            vec![real(2.0), text("real"), text("b")],
            vec![real(3.5), text("real"), text("c")],
            vec![real(4.0), text("real"), text("d")],
            vec![text("abc"), text("text"), text("e")],
        ],
    );
}

#[test]
fn min_max_over_indexed_real_column() {
    let tmp = file_with_types(
        "CREATE TABLE mm(id INTEGER PRIMARY KEY, r DATE);
         INSERT INTO mm VALUES (1, 1), (2, 2), (3, 3.5), (4, 4), (5, NULL);
         CREATE INDEX imm ON mm(r);",
        REAL,
    );
    let mut db = tmp.open();
    assert_scalar(&mut db, "SELECT max(r) FROM mm", real(4.0));
    assert_scalar(&mut db, "SELECT min(r) FROM mm", real(1.0));
    assert_rows(
        &mut db,
        "SELECT max(r), typeof(max(r)), min(r), typeof(min(r)) FROM mm",
        &[vec![real(4.0), text("real"), real(1.0), text("real")]],
    );
}

#[test]
fn arithmetic_cast_and_quote_see_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r * 1, typeof(r * 1), r + 0, typeof(r + 0), CAST(r AS TEXT), quote(r)
         FROM ix WHERE id <= 4 ORDER BY id",
        &[
            vec![real(1.0), text("real"), real(1.0), text("real"), text("1.0"), text("1.0")],
            vec![real(2.0), text("real"), real(2.0), text("real"), text("2.0"), text("2.0")],
            vec![real(3.5), text("real"), real(3.5), text("real"), text("3.5"), text("3.5")],
            vec![real(4.0), text("real"), real(4.0), text("real"), text("4.0"), text("4.0")],
        ],
    );
}

#[test]
fn grouping_and_comparison_see_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT id, r FROM ix NOT INDEXED WHERE r = 2", &[vec![int(2), real(2.0)]]);
    assert_scalar(
        &mut db,
        "SELECT group_concat(r, ',') FROM (SELECT r FROM ix WHERE id <= 4 ORDER BY id)",
        text("1.0,2.0,3.5,4.0"),
    );
    assert_rows(
        &mut db,
        "SELECT r, typeof(r), count(*) FROM ix NOT INDEXED WHERE id <= 4 GROUP BY r ORDER BY r",
        &[
            vec![real(1.0), text("real"), int(1)],
            vec![real(2.0), text("real"), int(1)],
            vec![real(3.5), text("real"), int(1)],
            vec![real(4.0), text("real"), int(1)],
        ],
    );
    assert_rows(&mut db, "SELECT DISTINCT typeof(r) FROM ix WHERE id <= 4", &[vec![text("real")]]);
}

#[test]
fn without_rowid_real_primary_key_reads_real() {
    let tmp = file_with_types(
        "CREATE TABLE w(k DATE PRIMARY KEY, v TEXT) WITHOUT ROWID;
         INSERT INTO w VALUES (1.0, 'a'), (2.5, 'b'), (3.0, 'c');",
        REAL,
    );
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT k, typeof(k), v FROM w ORDER BY k",
        &[
            vec![real(1.0), text("real"), text("a")],
            vec![real(2.5), text("real"), text("b")],
            vec![real(3.0), text("real"), text("c")],
        ],
    );
    assert_rows(&mut db, "SELECT k, v FROM w WHERE k = 3", &[vec![real(3.0), text("c")]]);
}

#[test]
fn strict_real_column_reads_real() {
    // `ANY` padded to four bytes is a valid STRICT type that stores the integer as given.
    let tmp = file_with_types(
        "CREATE TABLE s(id INTEGER PRIMARY KEY, r ANY ) STRICT;
         INSERT INTO s VALUES (1, 2), (2, 2.5);",
        &[("ANY ", "REAL")],
    );
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r, typeof(r) FROM s ORDER BY id",
        &[vec![real(2.0), text("real")], vec![real(2.5), text("real")]],
    );
}

#[test]
fn other_real_affinity_type_names() {
    let tmp = file_with_types(
        "CREATE TABLE f(id INTEGER PRIMARY KEY, d NUMBER, fl MONEY);
         INSERT INTO f VALUES (1, 2, 3);",
        &[("NUMBER", "DOUBLE"), ("MONEY", "FLOAT")],
    );
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT d, typeof(d), fl, typeof(fl) FROM f",
        &[vec![real(2.0), text("real"), real(3.0), text("real")]],
    );
}

#[test]
fn stored_generated_real_column_reads_real() {
    let tmp = file_with_types(
        "CREATE TABLE g(id INTEGER PRIMARY KEY, r DATE, gs DATE AS (r + 1) STORED);
         INSERT INTO g(id, r) VALUES (1, 2);",
        REAL,
    );
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r, gs, typeof(gs) FROM g",
        &[vec![real(2.0), real(3.0), text("real")]],
    );
}

#[test]
fn update_of_another_column_keeps_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    exec(&mut db, "UPDATE ix SET t = 'x' WHERE id = 2");
    assert_rows(
        &mut db,
        "SELECT r, typeof(r), t FROM ix WHERE id = 2",
        &[vec![real(2.0), text("real"), text("x")]],
    );
    drop(db);
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT r, typeof(r), t FROM ix WHERE id = 2",
        &[vec![real(2.0), text("real"), text("x")]],
    );
}

#[test]
fn upsert_and_trigger_see_the_stored_row_as_real() {
    let tmp = indexed();
    let mut db = tmp.open();
    exec(
        &mut db,
        "INSERT INTO ix VALUES (2, 9, 'z')
         ON CONFLICT(id) DO UPDATE SET t = typeof(r) || ':' || (r / 4)",
    );
    assert_scalar(&mut db, "SELECT t FROM ix WHERE id = 2", text("real:0.5"));
    exec(
        &mut db,
        "CREATE TABLE log(x);
         CREATE TRIGGER d AFTER DELETE ON ix BEGIN
           INSERT INTO log VALUES (typeof(OLD.r) || ':' || (OLD.r / 8));
         END;
         DELETE FROM ix WHERE id = 4;",
    );
    assert_scalar(&mut db, "SELECT x FROM log", text("real:0.5"));
}

#[test]
fn added_real_column_default_reads_real_on_old_rows() {
    // In memory: the rows that predate `ADD COLUMN` read the default, which is put
    // through REAL affinity like a stored value.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE ad(id INTEGER PRIMARY KEY);
         INSERT INTO ad VALUES (1), (2);
         ALTER TABLE ad ADD COLUMN z REAL DEFAULT 3;",
    );
    assert_rows(
        &mut db,
        "SELECT id, z, typeof(z) FROM ad ORDER BY id",
        &[vec![int(1), real(3.0), text("real")], vec![int(2), real(3.0), text("real")]],
    );
}

#[test]
fn added_real_column_default_reads_real_from_file() {
    let tmp = file_with_types(
        "CREATE TABLE ad(id INTEGER PRIMARY KEY);
         INSERT INTO ad VALUES (1), (2);
         ALTER TABLE ad ADD COLUMN z DATE DEFAULT 3;",
        REAL,
    );
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT id, z, typeof(z) FROM ad ORDER BY id",
        &[vec![int(1), real(3.0), text("real")], vec![int(2), real(3.0), text("real")]],
    );
}
