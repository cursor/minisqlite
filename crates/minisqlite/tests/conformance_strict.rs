//! Conformance battery: **STRICT tables** — the runtime datatype rule on INSERT and
//! UPDATE, and the no-affinity `ANY` column.
//!
//! The create-time half (every column must declare one of INT / INTEGER / REAL / TEXT /
//! BLOB / ANY) is pinned in the catalog's own unit tests; this file covers what a STRICT
//! table does with the values written into it.
//!
//! Spec source: `stricttables.html`.
//!   * §2: "SQLite attempts to coerce the data into the appropriate type using the usual
//!     affinity rules ... If the value cannot be losslessly converted in the specified
//!     datatype, then an SQLITE_CONSTRAINT_DATATYPE error is raised." Each declared type
//!     admits exactly one storage class after affinity (INT / INTEGER: integer, REAL:
//!     real, TEXT: text, BLOB: blob), and NULL is always allowed unless the column is
//!     NOT NULL. A REAL column converts an integer to a real.
//!   * §3: "an ANY datatype in a STRICT table ... any kind of data can be stored ... and
//!     the content is kept exactly as it is received" (no affinity), unlike an `ANY`
//!     column in an ordinary table, which gets NUMERIC affinity.
//!
//! Every expected error message is the one real sqlite 3.51.3 prints for the same
//! statement: `cannot store <VALUE TYPE> value in <COLUMN TYPE> column <table>.<col>`,
//! where the value type is NULL / INT / REAL / TEXT / BLOB and the column type is the
//! declared STRICT type in upper case. Each violation test also checks that the table is
//! left unchanged, which is what the ABORT the error implies guarantees.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Error};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Assert `sql` fails with a datatype constraint error carrying exactly `message`.
fn assert_datatype_error(db: &mut Connection, sql: &str, message: &str) {
    match assert_exec_error(db, sql) {
        Error::Constraint(m) => assert_eq!(m, message, "sql: {sql}"),
        other => {
            panic!("expected Error::Constraint({message:?})\n  sql: {sql}\n  actual: {other:?}")
        }
    }
}

// ===========================================================================
// Wrong storage class after affinity is rejected (stricttables.html §2).
// ===========================================================================

#[test]
fn real_column_rejects_text() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(r REAL) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES ('abc')",
        "cannot store TEXT value in REAL column s.r",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn int_column_rejects_fractional_real() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INT) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (1.5)",
        "cannot store REAL value in INT column s.i",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn text_column_rejects_blob() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(t TEXT) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (x'00')",
        "cannot store BLOB value in TEXT column s.t",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn blob_column_rejects_integer_and_text() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(b BLOB) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (5)",
        "cannot store INT value in BLOB column s.b",
    );
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES ('a')",
        "cannot store TEXT value in BLOB column s.b",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn column_type_in_message_is_upper_case() {
    // The declared type is written in lower case; sqlite names it in upper case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i int) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES ('a')",
        "cannot store TEXT value in INT column s.i",
    );
}

#[test]
fn multi_row_insert_is_aborted_as_a_whole() {
    // The second row is invalid, so the statement aborts and the first row is not kept.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INTEGER) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (1), ('x')",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn or_ignore_and_or_replace_do_not_skip_a_datatype_error() {
    // sqlite raises the datatype error under every conflict algorithm.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INTEGER) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT OR IGNORE INTO s VALUES ('x')",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_datatype_error(
        &mut db,
        "INSERT OR REPLACE INTO s VALUES ('x')",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn insert_select_is_checked() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(v)");
    exec(&mut db, "INSERT INTO src VALUES ('x')");
    exec(&mut db, "CREATE TABLE s(i INTEGER) STRICT");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s SELECT v FROM src",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

#[test]
fn without_rowid_strict_table_is_checked() {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE w(k INT PRIMARY KEY, v INT) STRICT, WITHOUT ROWID",
    );
    assert_datatype_error(
        &mut db,
        "INSERT INTO w VALUES (1, 'z')",
        "cannot store TEXT value in INT column w.v",
    );
    assert_datatype_error(
        &mut db,
        "INSERT INTO w VALUES ('q', 1)",
        "cannot store TEXT value in INT column w.k",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM w", int(0));
}

#[test]
fn added_column_default_is_checked() {
    // An omitted column takes its DEFAULT, and that value is checked like any other.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INTEGER) STRICT");
    exec(&mut db, "ALTER TABLE s ADD COLUMN z INT DEFAULT 'x'");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s(i) VALUES (1)",
        "cannot store TEXT value in INT column s.z",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(0));
}

// ===========================================================================
// UPDATE and UPSERT are checked too.
// ===========================================================================

#[test]
fn update_rejects_text_in_integer_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INTEGER) STRICT");
    exec(&mut db, "INSERT INTO s VALUES (1)");
    assert_datatype_error(
        &mut db,
        "UPDATE s SET i = 'x'",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_rows(
        &mut db,
        "SELECT i, typeof(i) FROM s",
        &[vec![int(1), text("integer")]],
    );
    // A value that converts losslessly is accepted.
    exec(&mut db, "UPDATE s SET i = '42'");
    assert_rows(
        &mut db,
        "SELECT i, typeof(i) FROM s",
        &[vec![int(42), text("integer")]],
    );
}

#[test]
fn update_or_ignore_does_not_skip_a_datatype_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INT) STRICT");
    exec(&mut db, "INSERT INTO s VALUES (1)");
    assert_datatype_error(
        &mut db,
        "UPDATE OR IGNORE s SET i = x'00'",
        "cannot store BLOB value in INT column s.i",
    );
    assert_rows(&mut db, "SELECT i FROM s", &[vec![int(1)]]);
}

#[test]
fn update_of_without_rowid_strict_table_is_checked() {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE w(k TEXT PRIMARY KEY, v REAL) STRICT, WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO w VALUES ('a', 1)");
    assert_datatype_error(
        &mut db,
        "UPDATE w SET v = 'zz'",
        "cannot store TEXT value in REAL column w.v",
    );
    assert_rows(&mut db, "SELECT k, v FROM w", &[vec![text("a"), real(1.0)]]);
}

#[test]
fn upsert_do_update_is_checked() {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE s(k INTEGER PRIMARY KEY, i INTEGER) STRICT",
    );
    exec(&mut db, "INSERT INTO s VALUES (1, 1)");
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (1, 2) ON CONFLICT(k) DO UPDATE SET i = 'y'",
        "cannot store TEXT value in INTEGER column s.i",
    );
    assert_rows(&mut db, "SELECT k, i FROM s", &[vec![int(1), int(1)]]);
}

// ===========================================================================
// Values that are valid after affinity, and NULL, are accepted.
// ===========================================================================

#[test]
fn null_is_allowed_in_every_strict_type() {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE s(i INTEGER, n INT, r REAL, t TEXT, b BLOB, a ANY) STRICT",
    );
    exec(
        &mut db,
        "INSERT INTO s VALUES (NULL, NULL, NULL, NULL, NULL, NULL)",
    );
    exec(&mut db, "INSERT INTO s VALUES (1, 1, 1.0, 'x', x'01', 1)");
    exec(
        &mut db,
        "UPDATE s SET i = NULL, n = NULL, r = NULL, t = NULL, b = NULL, a = NULL",
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM s WHERE coalesce(i, n, r, t, b, a) IS NULL",
        int(2),
    );
    // NULL is exempt, but the check itself is still in force on the same table.
    assert_datatype_error(
        &mut db,
        "UPDATE s SET b = 1",
        "cannot store INT value in BLOB column s.b",
    );
}

#[test]
fn real_column_converts_integers_to_real() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(r REAL) STRICT");
    exec(
        &mut db,
        "INSERT INTO s VALUES (3), ('4'), ('4.5'), (9223372036854775807)",
    );
    assert_rows(
        &mut db,
        "SELECT r, typeof(r) FROM s ORDER BY rowid",
        &[
            vec![real(3.0), text("real")],
            vec![real(4.0), text("real")],
            vec![real(4.5), text("real")],
            vec![real(9223372036854775807.0), text("real")],
        ],
    );
    // Text that does not look like a number stays text, and is rejected.
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES ('4x')",
        "cannot store TEXT value in REAL column s.r",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM s", int(4));
}

#[test]
fn affinity_is_applied_before_the_check() {
    // '7' and 2.0 convert losslessly to integers, 12 and 1.5 to text.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(i INTEGER, t TEXT) STRICT");
    exec(
        &mut db,
        "INSERT INTO s VALUES ('7', 12), (2.0, 1.5), (' 5 ', 'x')",
    );
    assert_rows(
        &mut db,
        "SELECT i, typeof(i), t, typeof(t) FROM s ORDER BY rowid",
        &[
            vec![int(7), text("integer"), text("12"), text("text")],
            vec![int(2), text("integer"), text("1.5"), text("text")],
            vec![int(5), text("integer"), text("x"), text("text")],
        ],
    );
    // A value affinity cannot convert keeps its class and is rejected.
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES ('7a', 'x')",
        "cannot store TEXT value in INTEGER column s.i",
    );
}

// ===========================================================================
// ANY keeps the value exactly as given (stricttables.html §3).
// ===========================================================================

#[test]
fn any_column_keeps_values_exactly_as_given() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(a ANY) STRICT");
    exec(
        &mut db,
        "INSERT INTO s VALUES ('12'), (12), (1.0), ('1.0'), (x'02')",
    );
    assert_rows(
        &mut db,
        "SELECT a, typeof(a) FROM s ORDER BY rowid",
        &[
            vec![text("12"), text("text")],
            vec![int(12), text("integer")],
            vec![real(1.0), text("real")],
            vec![text("1.0"), text("text")],
            vec![blob(&[0x02]), text("blob")],
        ],
    );
    // Only a STRICT table drops the affinity: an `ANY` column in an ordinary table is
    // NUMERIC, so the same '12' is stored as an integer there.
    exec(&mut db, "CREATE TABLE n(a ANY)");
    exec(&mut db, "INSERT INTO n VALUES ('12')");
    assert_rows(
        &mut db,
        "SELECT a, typeof(a) FROM n",
        &[vec![int(12), text("integer")]],
    );
}

#[test]
fn any_column_has_no_affinity_in_comparisons() {
    // With no affinity on the column, neither side is converted, so a text '12' does not
    // equal the integer 12 (and vice versa). Matches sqlite 3.51.3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(a ANY) STRICT");
    exec(&mut db, "INSERT INTO s VALUES (12), ('12')");
    assert_rows(
        &mut db,
        "SELECT typeof(a), a = 12, a = '12' FROM s ORDER BY rowid",
        &[
            vec![text("integer"), int(1), int(0)],
            vec![text("text"), int(0), int(1)],
        ],
    );
}

// ===========================================================================
// A STRICT table read back from a file. The catalog is rebuilt from the stored
// `CREATE TABLE ... STRICT` text in sqlite_schema, which is the same text sqlite3
// stores, so this covers a file sqlite3 wrote as well.
// ===========================================================================

/// A unique temporary database path; `Drop` removes the file and its sidecars.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_strict_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
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

#[test]
fn strict_table_reopened_from_file_is_enforced() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(
            &mut db,
            "CREATE TABLE s(id INTEGER PRIMARY KEY, qty INTEGER, price REAL, tag ANY) STRICT",
        );
        exec(&mut db, "INSERT INTO s VALUES (1, 2, 9.5, '12')");
    }
    let mut db = tmp.open();
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (2, 'two', 'cheap', NULL)",
        "cannot store TEXT value in INTEGER column s.qty",
    );
    assert_datatype_error(
        &mut db,
        "INSERT INTO s VALUES (2, 2, 'cheap', NULL)",
        "cannot store TEXT value in REAL column s.price",
    );
    assert_datatype_error(
        &mut db,
        "UPDATE s SET qty = 2.5",
        "cannot store REAL value in INTEGER column s.qty",
    );
    exec(&mut db, "INSERT INTO s VALUES (2, 3, 4, '07')");
    assert_rows(
        &mut db,
        "SELECT id, qty, price, typeof(price), tag, typeof(tag) FROM s ORDER BY id",
        &[
            vec![
                int(1),
                int(2),
                real(9.5),
                text("real"),
                text("12"),
                text("text"),
            ],
            vec![
                int(2),
                int(3),
                real(4.0),
                text("real"),
                text("07"),
                text("text"),
            ],
        ],
    );
}
