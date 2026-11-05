//! Conformance battery: **ROWID semantics** — the rowid aliases, the
//! `last_insert_rowid()` counter, rowid reuse vs. AUTOINCREMENT, rowid ordering,
//! and WITHOUT ROWID.
//!
//! Every expected value here is TRANSCRIBED FROM THE SQLITE DOCS, never from what
//! the engine currently returns — a failing case is the intended signal that the
//! engine diverges from the spec. A spec-correct case is NEVER weakened, deleted, or
//! `#[ignore]`d to make the suite pass; a real divergence is left as a genuine failing
//! assertion.
//!
//! Scope (this file deliberately does NOT re-test coverage that already exists):
//!   * `conformance_dml_insert.rs` covers basic auto-assignment (empty table → 1,
//!     next = largest+1, explicit NULL auto-assigns, INTEGER PRIMARY KEY basics).
//!   * `conformance_ddl.rs` covers `rowid` auto-increment and one lowercase alias
//!     row on a plain table, plus explicit-rowid insert.
//!   * `conformance_func_misc.rs` covers `last_insert_rowid()` after the first /
//!     next / multi-row insert, that UPDATE/DELETE leave it unchanged, and that a
//!     fresh connection reads 0.
//! This file covers the UNCOVERED core: the three rowid ALIASES (case-independence,
//! INTEGER PRIMARY KEY as a fourth alias, and user-column SHADOWING),
//! `last_insert_rowid()` for explicit / INTEGER-PRIMARY-KEY / failed inserts, rowid
//! REUSE-after-delete vs. AUTOINCREMENT no-reuse, and rowid ORDERING.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createtable.html` §5 "ROWIDs and the INTEGER PRIMARY KEY":
//!     "The rowid value can be accessed using one of the special case-independent
//!     names 'rowid', 'oid', or '_rowid_' in place of a column name. If a table
//!     contains a user defined column named 'rowid', 'oid' or '_rowid_', then that
//!     name always refers [to] the explicitly declared column". A single-column
//!     PRIMARY KEY whose declared type is exactly INTEGER "becomes an alias for the
//!     rowid". "The data for rowid tables is stored as a B-Tree structure ... using
//!     the rowid value as the key" (so a scan visits rows in rowid order). The
//!     rowid "is omitted in WITHOUT ROWID tables". A rowid is a "64-bit signed
//!     integer".
//!   * `autoinc.html`: the normal algorithm gives "a ROWID that is one larger than
//!     the largest ROWID in the table prior to the insert. If the table is
//!     initially empty, then a ROWID of 1 is used" — so a deleted largest rowid is
//!     reused and emptying the table resets to 1. With AUTOINCREMENT the new rowid
//!     "is at least one larger than the largest ROWID that has ever before existed
//!     in that same table" and such rowids are "guaranteed to be monotonically
//!     increasing" and never reused; SQLite tracks the largest in the internal
//!     `sqlite_sequence` table. "an INSERT statement may provide a value to use as
//!     the rowid" and negative rowids may be inserted explicitly.
//!   * `lang_corefunc.html#last_insert_rowid`: "The last_insert_rowid() function
//!     returns the ROWID of the last row insert from the database connection which
//!     invoked the function." A failed (constraint) INSERT inserts no row, so it
//!     leaves the value at the last SUCCESSFUL insert.
//!
//! Each case is its own `#[test]` so an unsupported feature fails exactly that case
//! and never masks the rest. `Value` has no `PartialEq`, so every value comparison
//! goes through the harness helpers (`assert_scalar`, `assert_rows`, …).

mod conformance;
use conformance::*;

// `Connection` types the `exec_probe` helper below. `Error` is matched to pin that
// a failed INSERT is specifically a constraint violation. Neither is re-exported
// through `conformance::*` (the harness imports them privately), so pull them in.
use minisqlite::{Connection, Error};

/// Execute a statement the SPEC requires to succeed but a partial engine may not
/// implement yet (AUTOINCREMENT, WITHOUT ROWID). This is `exec` with a richer,
/// spec-citing panic message: it is `try_exec` plus a `panic!` that names the
/// missing `feature`, so a gap reads clearly in the failure output. (It provides no
/// tolerance or isolation of its own — each case is its own `#[test]`, so any panic
/// here fails only that one test.)
fn exec_probe(db: &mut Connection, sql: &str, feature: &str) {
    if let Err(e) = try_exec(db, sql) {
        panic!(
            "spec requires this statement to succeed, but the engine errored\n  \
             feature: {feature}\n  sql: {sql}\n  error: {e:?}\n  \
             (a spec-correct case left as a genuine failing assertion)"
        );
    }
}

// ===========================================================================
// 1) ROWID ALIASES — lang_createtable.html §5, autoinc.html §2.
// "The rowid value can be accessed using one of the special case-independent
// names 'rowid', 'oid', or '_rowid_' in place of a column name."
// ===========================================================================

#[test]
fn rowid_aliases_are_case_independent() {
    // §5: the names are "case-independent", so every capitalization of each of the
    // three built-in aliases returns the same integer rowid (1 for the first row
    // of a fresh plain table — autoinc.html).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('x')");
    assert_scalar(&mut db, "SELECT rowid FROM t", int(1));
    assert_scalar(&mut db, "SELECT ROWID FROM t", int(1));
    assert_scalar(&mut db, "SELECT RoWiD FROM t", int(1));
    assert_scalar(&mut db, "SELECT oid FROM t", int(1));
    assert_scalar(&mut db, "SELECT OID FROM t", int(1));
    assert_scalar(&mut db, "SELECT _rowid_ FROM t", int(1));
    assert_scalar(&mut db, "SELECT _ROWID_ FROM t", int(1));
}

#[test]
fn rowid_aliases_agree_on_every_row() {
    // §5: on a table with no user column of those names, rowid == oid == _rowid_
    // for EVERY row (not just the first). Two rows pin that they track together.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('a'), ('b')");
    assert_rows(
        &mut db,
        "SELECT rowid, oid, _rowid_ FROM t ORDER BY rowid",
        &[vec![int(1), int(1), int(1)], vec![int(2), int(2), int(2)]],
    );
}

#[test]
fn rowid_alias_usable_in_where_clause() {
    // autoinc.html §2: the alias names "work equally well in any context" — here as
    // a WHERE predicate that selects exactly the row with that rowid.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('a'), ('b'), ('c')");
    assert_scalar(&mut db, "SELECT v FROM t WHERE rowid = 2", text("b"));
    assert_scalar(&mut db, "SELECT v FROM t WHERE _rowid_ = 3", text("c"));
}

#[test]
fn integer_primary_key_is_alias_for_rowid() {
    // §5 + autoinc.html §2: a lone INTEGER PRIMARY KEY column is a fourth alias for
    // the rowid, so the declared name and all three built-ins return the same value
    // (auto-assigned 1 for the first row).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('x')");
    assert_scalar(&mut db, "SELECT a FROM t", int(1));
    assert_scalar(&mut db, "SELECT rowid FROM t", int(1));
    assert_scalar(&mut db, "SELECT oid FROM t", int(1));
    assert_scalar(&mut db, "SELECT _rowid_ FROM t", int(1));
}

#[test]
fn integer_primary_key_alias_with_explicit_value() {
    // §5: the INTEGER PRIMARY KEY column IS the rowid, so an explicitly supplied
    // value is readable through every alias.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (50, 'x')");
    assert_scalar(&mut db, "SELECT a FROM t", int(50));
    assert_scalar(&mut db, "SELECT rowid FROM t", int(50));
    assert_scalar(&mut db, "SELECT oid FROM t", int(50));
}

#[test]
fn integer_primary_key_alias_usable_in_where_clause() {
    // §5: because `a` and `rowid` are the same key, a lookup by either selects the
    // same row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (10, 'x'), (20, 'y')");
    assert_scalar(&mut db, "SELECT v FROM t WHERE a = 20", text("y"));
    assert_scalar(&mut db, "SELECT v FROM t WHERE rowid = 20", text("y"));
}

#[test]
fn user_column_named_rowid_shadows_builtin() {
    // §5: "If a table contains a user defined column named 'rowid' ... then that
    // name always refers [to] the explicitly declared column" — so `rowid` returns
    // the user column's TEXT value, not the integer key.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(rowid TEXT, v)");
    exec(&mut db, "INSERT INTO t VALUES ('hello', 1)");
    assert_scalar(&mut db, "SELECT rowid FROM t", text("hello"));
}

#[test]
fn user_column_named_oid_shadows_builtin() {
    // §5: a user column named `oid` shadows the `oid` alias.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(oid TEXT, v)");
    exec(&mut db, "INSERT INTO t VALUES ('hello', 1)");
    assert_scalar(&mut db, "SELECT oid FROM t", text("hello"));
}

#[test]
fn user_column_named_underscore_rowid_shadows_builtin() {
    // §5: a user column named `_rowid_` shadows the `_rowid_` alias.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(_rowid_ TEXT, v)");
    exec(&mut db, "INSERT INTO t VALUES ('hello', 1)");
    assert_scalar(&mut db, "SELECT _rowid_ FROM t", text("hello"));
}

#[test]
fn shadowing_is_case_independent() {
    // §5: the special names are case-independent, so a user column named `rowid`
    // shadows `ROWID` (any capitalization) as well.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(rowid TEXT, v)");
    exec(&mut db, "INSERT INTO t VALUES ('hello', 1)");
    assert_scalar(&mut db, "SELECT ROWID FROM t", text("hello"));
}

#[test]
fn shadowing_one_name_leaves_other_aliases_reaching_the_key() {
    // §5 / autoinc.html §2: only the DECLARED name is captured — "the use of that
    // name will refer to the declared column not to the internal ROWID". A user
    // column named `rowid` still leaves `oid` and `_rowid_` reaching the true
    // integer key (auto-assigned 1 for the first row), while `rowid` returns the
    // user column's text.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(rowid TEXT, v)");
    exec(&mut db, "INSERT INTO t VALUES ('hello', 1)");
    assert_scalar(&mut db, "SELECT rowid FROM t", text("hello"));
    assert_scalar(&mut db, "SELECT oid FROM t", int(1));
    assert_scalar(&mut db, "SELECT _rowid_ FROM t", int(1));
}

#[test]
fn select_star_excludes_rowid_for_plain_table() {
    // §5: the rowid is accessed by a special name "in place of a column name" — it
    // is not one of the table's declared columns, so `SELECT *` (which expands to
    // the declared columns) does NOT include it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('x')");
    assert_columns(&mut db, "SELECT * FROM t", &["v"]);
}

#[test]
fn select_star_includes_integer_primary_key_column() {
    // §5: an INTEGER PRIMARY KEY is a real DECLARED column that merely aliases the
    // rowid, so unlike the implicit rowid it IS expanded by `SELECT *`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "v"]);
}

#[test]
fn non_integer_int_primary_key_is_not_rowid_alias() {
    // §5: "A PRIMARY KEY column only becomes an integer primary key if the declared
    // type name is exactly 'INTEGER'. Other integer type names like 'INT' ... cause
    // the primary key column to behave as an ordinary table column ... not as an
    // alias for the rowid." So `a INT PRIMARY KEY` is a normal indexed column: an
    // explicit `a` of 5 is stored in `a`, while the rowid is auto-assigned (1) and
    // stays independent of `a`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (5, 'x')");
    assert_rows(&mut db, "SELECT rowid, a FROM t", &[vec![int(1), int(5)]]);
}

#[test]
fn integer_primary_key_desc_is_not_rowid_alias() {
    // §5 (the DESC quirk): "if the declaration of a column with declared type
    // 'INTEGER' includes an 'PRIMARY KEY DESC' clause, it does not become an alias
    // for the rowid and is not classified as an integer primary key." So
    // `a INTEGER PRIMARY KEY DESC` is an ordinary indexed column: an explicit
    // a=100 is stored in `a`, while the rowid is auto-assigned (1) independently.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY DESC, v)");
    exec(&mut db, "INSERT INTO t VALUES (100, 'x')");
    assert_rows(&mut db, "SELECT rowid, a FROM t", &[vec![int(1), int(100)]]);
}

// ===========================================================================
// 2) last_insert_rowid() — lang_corefunc.html#last_insert_rowid.
// "returns the ROWID of the last row insert from the database connection".
// (The basic auto-increment cases live in conformance_func_misc.rs; these pin
// the UNCOVERED explicit / INTEGER-PRIMARY-KEY / failed-insert behavior.)
// ===========================================================================

#[test]
fn last_insert_rowid_after_explicit_rowid() {
    // The value is the rowid of the last inserted row, so an explicitly supplied
    // rowid is what the function reports.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t(rowid, v) VALUES (50, 'x')");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(50));
}

#[test]
fn last_insert_rowid_reflects_integer_primary_key_value() {
    // The INTEGER PRIMARY KEY column is the rowid (lang_createtable.html §5), so an
    // explicit value inserted into it is the last_insert_rowid().
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (50, 'x')");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(50));
}

#[test]
fn last_insert_rowid_unchanged_by_failed_insert() {
    // last_insert_rowid() is the rowid of the last ROW INSERT: a constraint-failed
    // INSERT inserts no row, so the value stays at the last SUCCESSFUL insert (6).
    // The last successful rowid (6) is deliberately DIFFERENT from the rowid the
    // failing INSERT attempts (5), so this also catches a bug that set the counter
    // to the ATTEMPTED rowid — `int(6)` fails if it became 5.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (5, 'x')");
    exec(&mut db, "INSERT INTO t VALUES (6, 'y')");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(6));
    // Duplicate PRIMARY KEY (attempts rowid 5) → constraint violation, no row inserted.
    let e = assert_exec_error(&mut db, "INSERT INTO t VALUES (5, 'dup')");
    assert!(
        matches!(e, Error::Constraint(_)),
        "duplicate rowid should be Error::Constraint(_); got {e:?}"
    );
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(6));
}

// ===========================================================================
// 3) ROWID REUSE (normal) vs AUTOINCREMENT (no reuse) — autoinc.html.
// THE KEY CONTRAST between the two rowid-selection algorithms.
// ===========================================================================

#[test]
fn normal_rowid_reused_after_deleting_largest() {
    // autoinc.html §2: the normal algorithm gives "one larger than the largest
    // ROWID in the table". After deleting the largest (3), the largest present is
    // 2, so the next insert reuses rowid 3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t WHERE a = 3");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT a FROM t WHERE v = 'd'", int(3));
}

#[test]
fn normal_rowid_plain_table_reused_after_deleting_largest() {
    // autoinc.html §2: the same reuse holds for an implicit rowid (no INTEGER
    // PRIMARY KEY column).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('a'), ('b'), ('c')"); // rowids 1,2,3
    exec(&mut db, "DELETE FROM t WHERE v = 'c'"); // removes rowid 3
    exec(&mut db, "INSERT INTO t VALUES ('d')");
    assert_scalar(&mut db, "SELECT rowid FROM t WHERE v = 'd'", int(3));
}

#[test]
fn normal_rowid_deleting_middle_still_uses_max_plus_one() {
    // autoinc.html §2: the next rowid is "one larger than the largest ROWID in the
    // table" — NOT the smallest free gap. Deleting the middle (2) leaves {1,3}, so
    // the next insert is 4 (largest present 3, +1), and 2 is NOT reused.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t WHERE a = 2");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT a FROM t WHERE v = 'd'", int(4));
}

#[test]
fn normal_rowid_resets_to_one_after_deleting_all() {
    // autoinc.html §2: "If the table is initially empty, then a ROWID of 1 is
    // used." After deleting every row the table is empty, so the next rowid is 1
    // (a normal table does NOT remember the old maximum).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT a FROM t", int(1));
}

#[test]
fn normal_rowid_plain_table_resets_to_one_after_deleting_all() {
    // Same reset-to-1 for an implicit rowid table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t");
    exec(&mut db, "INSERT INTO t VALUES ('d')");
    assert_scalar(&mut db, "SELECT rowid FROM t", int(1));
}

#[test]
fn autoincrement_does_not_reuse_after_deleting_largest() {
    // autoinc.html §1/§3: with AUTOINCREMENT the new rowid is "at least one larger
    // than the largest ROWID that has ever before existed" — deleting the largest
    // (3) does NOT free it, so the next insert is 4, not 3.
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)",
        "AUTOINCREMENT",
    );
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t WHERE a = 3");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT a FROM t WHERE v = 'd'", int(4));
}

#[test]
fn autoincrement_does_not_reset_after_deleting_all() {
    // autoinc.html §3: the "largest ever" is remembered across an emptying, so even
    // after deleting every row the next AUTOINCREMENT rowid continues (4), never
    // resetting to 1 the way a normal table does.
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)",
        "AUTOINCREMENT",
    );
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT a FROM t", int(4));
}

#[test]
fn autoincrement_sqlite_sequence_tracks_max_ever() {
    // autoinc.html §3: SQLite tracks the largest rowid ever used in the internal
    // `sqlite_sequence(name, seq)` table. After inserting 1,2,3 the row for `t`
    // holds seq=3; after deleting the largest and inserting again (→4) it holds 4.
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)",
        "AUTOINCREMENT",
    );
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    assert_scalar(&mut db, "SELECT seq FROM sqlite_sequence WHERE name = 't'", int(3));
    exec(&mut db, "DELETE FROM t WHERE a = 3");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')"); // → 4
    assert_scalar(&mut db, "SELECT seq FROM sqlite_sequence WHERE name = 't'", int(4));
}

#[test]
fn autoincrement_generated_rowids_are_monotonic() {
    // autoinc.html §3: automatically generated AUTOINCREMENT rowids "are guaranteed
    // to be monotonically increasing". Emptying the table then inserting yields 3,4
    // (continuing past the deleted 1,2), never restarting.
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)",
        "AUTOINCREMENT",
    );
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b')"); // 1,2
    exec(&mut db, "DELETE FROM t");
    exec(&mut db, "INSERT INTO t(v) VALUES ('x'), ('y')"); // 3,4
    assert_rows(
        &mut db,
        "SELECT a, v FROM t ORDER BY a",
        &[vec![int(3), text("x")], vec![int(4), text("y")]],
    );
}

#[test]
fn normal_reuse_confirmed_by_last_insert_rowid() {
    // Cross-check: on a normal table, the reused rowid (3) is also what
    // last_insert_rowid() reports (autoinc.html §2 + lang_corefunc.html).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t WHERE a = 3");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(3));
}

#[test]
fn autoincrement_no_reuse_confirmed_by_last_insert_rowid() {
    // Cross-check: with AUTOINCREMENT the non-reused rowid (4) is what
    // last_insert_rowid() reports (autoinc.html §3 + lang_corefunc.html).
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)",
        "AUTOINCREMENT",
    );
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // 1,2,3
    exec(&mut db, "DELETE FROM t WHERE a = 3");
    exec(&mut db, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(4));
}

// ===========================================================================
// 4) ROWID ORDERING — lang_createtable.html §5.
// Rowid tables are "stored as a B-Tree structure ... using the rowid value as
// the key", so a table scan visits rows in ascending rowid order.
// ===========================================================================

#[test]
fn rowid_table_scan_visits_rows_in_rowid_order() {
    // §5: because the rows live in a B-tree keyed by rowid, an unqualified scan
    // returns them in ascending rowid order regardless of INSERT order. The rows
    // are inserted with rowids 3,1,2; the scan yields rowid order 1,2,3 → a,b,c.
    // NOTE: scan order without ORDER BY is engine-observable (the rowid B-tree),
    // not a portable SQL guarantee — if a future planner picks a different access
    // path this may change, so read a break here as a plan change to weigh against
    // the file-format rule, not automatically a regression. The `ORDER BY rowid`
    // [DESC] cases below pin the contractual ordering.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (3, 'c'), (1, 'a'), (2, 'b')");
    assert_rows(
        &mut db,
        "SELECT v FROM t",
        &[vec![text("a")], vec![text("b")], vec![text("c")]],
    );
}

#[test]
fn select_order_by_rowid_ascending() {
    // §5: an explicit ORDER BY rowid produces the same ascending rowid order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (3, 'c'), (1, 'a'), (2, 'b')");
    assert_rows(
        &mut db,
        "SELECT v FROM t ORDER BY rowid",
        &[vec![text("a")], vec![text("b")], vec![text("c")]],
    );
}

#[test]
fn select_order_by_rowid_descending() {
    // §5: ORDER BY rowid DESC reverses to descending rowid order → c,b,a.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (3, 'c'), (1, 'a'), (2, 'b')");
    assert_rows(
        &mut db,
        "SELECT v FROM t ORDER BY rowid DESC",
        &[vec![text("c")], vec![text("b")], vec![text("a")]],
    );
}

// ===========================================================================
// 5) EXPLICIT rowid + the max-plus-one edge — autoinc.html §2.
// ===========================================================================

#[test]
fn explicit_rowid_then_auto_is_max_plus_one() {
    // autoinc.html §2: after an explicit rowid 100, the next auto rowid is one
    // larger than the largest in the table → 101.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v)");
    exec(&mut db, "INSERT INTO t(rowid, v) VALUES (100, 'x')");
    exec(&mut db, "INSERT INTO t(v) VALUES ('y')");
    assert_scalar(&mut db, "SELECT rowid FROM t WHERE v = 'y'", int(101));
}

#[test]
fn auto_rowid_uses_max_not_last_inserted() {
    // autoinc.html §2: the auto rowid is one larger than the LARGEST rowid in the
    // table, not one larger than the most recently inserted one. Through the
    // INTEGER PRIMARY KEY alias (explicit values 100 then 5), the largest present
    // is 100, so the next auto rowid is 101 (not 6).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (100, 'x')");
    exec(&mut db, "INSERT INTO t VALUES (5, 'y')");
    exec(&mut db, "INSERT INTO t(v) VALUES ('z')");
    assert_scalar(&mut db, "SELECT a FROM t WHERE v = 'z'", int(101));
}

#[test]
fn explicit_negative_rowid_on_integer_primary_key_is_stored() {
    // §5: a rowid is a "64-bit signed integer", so a negative value supplied
    // through the INTEGER PRIMARY KEY alias is stored verbatim and reachable via
    // both the declared name and `rowid`. (autoinc.html §2's "auto rowids stay
    // positive" guarantee only holds when no negative rowid is inserted explicitly.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (-5, 'x')");
    assert_scalar(&mut db, "SELECT a FROM t", int(-5));
    assert_scalar(&mut db, "SELECT rowid FROM t", int(-5));
}

// ===========================================================================
// 5b) Auto rowid once the largest rowid is i64::MAX — autoinc.html.
// "If the largest ROWID is equal to the largest possible integer
// (9223372036854775807) then the database engine starts picking positive
// candidate ROWIDs at random until it finds one that is not previously used."
// With AUTOINCREMENT the insert fails with SQLITE_FULL instead. The random value
// cannot be asserted, so these pin what sqlite3 3.51.3 guarantees: the row goes
// in under a positive, unused rowid, and `last_insert_rowid()` reports it.
// ===========================================================================

#[test]
fn integer_primary_key_auto_rowid_after_max_picks_an_unused_positive_rowid() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775807, 'max')");
    exec(&mut db, "INSERT INTO t(v) VALUES ('next')");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_scalar(&mut db, "SELECT id > 0 AND id < 9223372036854775807 FROM t WHERE v = 'next'", int(1));
    assert_scalar(&mut db, "SELECT last_insert_rowid() = (SELECT id FROM t WHERE v = 'next')", int(1));
    assert_scalar(&mut db, "SELECT v FROM t WHERE id = 9223372036854775807", text("max"));
}

#[test]
fn repeated_auto_rowids_after_max_all_succeed_and_stay_unique() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775807, 'max')");
    for i in 0..50 {
        exec(&mut db, &format!("INSERT INTO t(v) VALUES ('s{i}')"));
        assert_scalar(&mut db, &format!("SELECT last_insert_rowid() = (SELECT id FROM t WHERE v = 's{i}')"), int(1));
    }
    // A multi-row INSERT keeps drawing random rowids for every row.
    exec(&mut db, "INSERT INTO t(v) VALUES ('m0'), ('m1'), ('m2'), ('m3'), ('m4')");
    assert_scalar(&mut db, "SELECT last_insert_rowid() = (SELECT id FROM t WHERE v = 'm4')", int(1));
    assert_rows(
        &mut db,
        "SELECT count(*), count(DISTINCT id), min(id) > 0, max(id) FROM t",
        &[vec![int(56), int(56), int(1), int(9223372036854775807)]],
    );
}

#[test]
fn plain_rowid_table_auto_rowid_after_max_picks_an_unused_positive_rowid() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(v)");
    exec(&mut db, "INSERT INTO r(rowid, v) VALUES (9223372036854775807, 'max')");
    exec(&mut db, "INSERT INTO r(v) VALUES ('a')");
    exec(&mut db, "INSERT INTO r(v) VALUES ('b')");
    assert_rows(&mut db, "SELECT count(*), count(DISTINCT rowid), min(rowid) > 0 FROM r", &[vec![int(3), int(3), int(1)]]);
    assert_scalar(&mut db, "SELECT last_insert_rowid() = (SELECT rowid FROM r WHERE v = 'b')", int(1));
}

#[test]
fn auto_rowid_reaches_max_then_switches_to_random_within_one_statement() {
    // The first auto row takes max+1 = i64::MAX; the next one in the same statement
    // has no max+1 left and gets a random unused rowid (sqlite3 3.51.3: count 3, max(id)
    // i64::MAX).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775806, 'a')");
    exec(&mut db, "INSERT INTO t(v) VALUES ('b'), ('c')");
    assert_scalar(&mut db, "SELECT v FROM t WHERE id = 9223372036854775807", text("b"));
    assert_scalar(&mut db, "SELECT id > 0 AND id < 9223372036854775806 FROM t WHERE v = 'c'", int(1));
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

#[test]
fn auto_rowid_after_max_with_before_trigger_present() {
    // A table with a BEFORE INSERT trigger goes through the re-derive step after the
    // trigger fires; the random rowid must survive it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE log(x)");
    exec(&mut db, "CREATE TRIGGER tb BEFORE INSERT ON t BEGIN INSERT INTO log VALUES (NEW.v); END");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775807, 'max')");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a'), ('b')");
    assert_rows(&mut db, "SELECT count(*), count(DISTINCT id), min(id) > 0 FROM t", &[vec![int(3), int(3), int(1)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(3));
}

#[test]
fn auto_rowids_after_max_do_not_repeat_across_connections() {
    // Each new connection must draw fresh candidates. With the same candidate sequence
    // per connection, the 101st connection would find its first 100 candidates taken and
    // fail with "database or disk is full".
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let _ = std::fs::remove_file(self.0.with_extension("db-journal"));
        }
    }
    let path = std::env::temp_dir().join(format!("minisqlite_rowid_max_{}.db", std::process::id()));
    let guard = Cleanup(path.clone());
    drop(Cleanup(path.clone())); // clear a stale file from an earlier run
    let open = || Connection::open(&path).unwrap_or_else(|e| panic!("open {path:?}: {e:?}"));
    let mut db = open();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775807, 'max')");
    drop(db);
    for i in 0..120 {
        let mut db = open();
        exec(&mut db, &format!("INSERT INTO t(v) VALUES ('c{i}')"));
    }
    let mut db = open();
    assert_rows(&mut db, "SELECT count(*), count(DISTINCT id), min(id) > 0 FROM t", &[vec![int(121), int(121), int(1)]]);
    drop(db);
    drop(guard);
}

#[test]
fn autoincrement_after_max_rowid_is_database_full() {
    // autoinc.html: with AUTOINCREMENT, once the largest rowid has been used "the next
    // INSERT attempt fails with an SQLITE_FULL error". sqlite3 3.51.3 reports
    // "database or disk is full", also after that row is deleted (sqlite_sequence keeps
    // the high-water mark).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
    exec(&mut db, "INSERT INTO t VALUES (9223372036854775807, 'max')");
    let e = assert_exec_error(&mut db, "INSERT INTO t(v) VALUES ('next')");
    assert!(e.to_string().contains("database or disk is full"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    exec(&mut db, "DELETE FROM t");
    let e = assert_exec_error(&mut db, "INSERT INTO t(v) VALUES ('next')");
    assert!(e.to_string().contains("database or disk is full"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

// ===========================================================================
// 6) WITHOUT ROWID — lang_createtable.html §5 (PROBE).
// "The rowid (and 'oid' and '_rowid_') is omitted in WITHOUT ROWID tables", so
// referencing it is an error (no such column). WITHOUT ROWID may be
// unimplemented; the CREATE is a probe and a failure is recorded, not hidden.
// ===========================================================================

#[test]
fn without_rowid_table_has_no_rowid_column() {
    // §5: a WITHOUT ROWID table omits the rowid, so `SELECT rowid` is a
    // no-such-column error (SQLITE_ERROR / Error::Sql). The CREATE and INSERT are
    // gated with `exec_probe` on purpose: they MUST succeed for the assertion to be
    // meaningful. If WITHOUT ROWID is unimplemented (the engine can't create or
    // insert), the probe fails loudly HERE — rather than the `SELECT rowid` error
    // being satisfied for the wrong reason (an "unsupported table" error that is
    // also `Error::Sql`, which would be a false pass).
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a PRIMARY KEY) WITHOUT ROWID",
        "WITHOUT ROWID",
    );
    exec_probe(&mut db, "INSERT INTO t VALUES ('k')", "WITHOUT ROWID");
    let e = assert_query_error(&mut db, "SELECT rowid FROM t");
    assert!(
        matches!(e, Error::Sql(_)),
        "referencing the omitted rowid should be Error::Sql (no such column); got {e:?}"
    );
}

#[test]
fn without_rowid_table_has_no_oid_column() {
    // §5: the `oid` alias is likewise omitted in a WITHOUT ROWID table. As above,
    // the CREATE/INSERT probes gate the assertion so an unsupported WITHOUT ROWID
    // table cannot produce a false pass on the `SELECT oid` error.
    let mut db = mem();
    exec_probe(
        &mut db,
        "CREATE TABLE t(a PRIMARY KEY) WITHOUT ROWID",
        "WITHOUT ROWID",
    );
    exec_probe(&mut db, "INSERT INTO t VALUES ('k')", "WITHOUT ROWID");
    let e = assert_query_error(&mut db, "SELECT oid FROM t");
    assert!(
        matches!(e, Error::Sql(_)),
        "referencing the omitted oid should be Error::Sql (no such column); got {e:?}"
    );
}

#[test]
fn without_rowid_scan_round_trips_values_in_primary_key_order() {
    // withoutrowid.html: a WITHOUT ROWID table stores each row in a b-tree KEYED BY its
    // PRIMARY KEY, so a plain scan visits rows in PK order ('a' before 'k') regardless of
    // INSERT order, and every column reads back its stored value. This is the END-TO-END
    // half of the plan<->exec row-width contract's crux: a MULTI-column
    // table read through the real Connection (binder → planner → WR executor scan). A
    // width disagreement (plan expecting N+1 while exec emits N) would read the wrong
    // register — dropping `b` or shifting it — so selecting the NON-first column is what
    // makes an off-by-one surface instead of being masked. (Scan order without ORDER BY is
    // the engine-observable WR b-tree order, the documented storage property; the
    // `ORDER BY`/`WHERE` cases below pin the same order contractually.)
    let mut db = mem();
    exec_probe(&mut db, "CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID", "WITHOUT ROWID");
    exec_probe(&mut db, "INSERT INTO t VALUES ('k', 'v'), ('a', 'w')", "WITHOUT ROWID");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![text("a"), text("w")], vec![text("k"), text("v")]],
    );
    // The non-first column, selected alone, still reads its OWN value (a width/register
    // slip would return `a`'s value, a fabricated rowid, or nothing here).
    assert_rows(&mut db, "SELECT b FROM t", &[vec![text("w")], vec![text("v")]]);
}

#[test]
fn without_rowid_select_star_is_exactly_the_declared_columns() {
    // §5: a WITHOUT ROWID table has no rowid, so `SELECT *` expands to EXACTLY its N
    // declared columns (width N) — no phantom trailing rowid column. Two columns in, two
    // column names and two values out, so a width-N+1 plan/exec slip surfaces as a wrong
    // column count or a stray value rather than staying masked.
    let mut db = mem();
    exec_probe(&mut db, "CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID", "WITHOUT ROWID");
    exec_probe(&mut db, "INSERT INTO t VALUES ('k', 'v')", "WITHOUT ROWID");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "b"]);
    assert_rows(&mut db, "SELECT * FROM t", &[vec![text("k"), text("v")]]);
}

#[test]
fn without_rowid_lookup_by_primary_key_returns_the_matching_row() {
    // §5: the PRIMARY KEY is the key of the WR b-tree, so a WHERE on it selects exactly
    // the matching row (today via the residual Filter over the WR scan; a PK seek is a
    // documented perf follow-up). Pins that the equality predicate resolves against the
    // right (non-rowid) register and returns the paired non-key column.
    let mut db = mem();
    exec_probe(&mut db, "CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID", "WITHOUT ROWID");
    exec_probe(&mut db, "INSERT INTO t VALUES ('k', 'v'), ('a', 'w')", "WITHOUT ROWID");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 'k'", text("v"));
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 'a'", text("w"));
}
