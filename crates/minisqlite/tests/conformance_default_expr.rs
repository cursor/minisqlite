//! Conformance battery: a column `DEFAULT (expr)` written in parentheses.
//!
//! Spec source: `lang_createtable.html` §3.2 "The DEFAULT clause" — "An explicit
//! DEFAULT clause may specify that the default value is NULL, a string constant, a
//! blob constant, a signed-number, or any constant expression enclosed in
//! parentheses." and "If the default value of a column is an expression in
//! parentheses, then the expression is evaluated once for each row inserted and the
//! results used in the new row." `PRAGMA table_info` reports the expression's text
//! in `dflt_value` (sqlite3 shows it without the surrounding parentheses).

mod conformance;
use conformance::*;

use minisqlite::Connection;

#[test]
fn parenthesized_constant_defaults_are_applied() {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE d(x, a DEFAULT (2), b DEFAULT (1+1), c DEFAULT ('a'||'b'), e DEFAULT (abs(-2)))",
    );
    exec(&mut db, "INSERT INTO d(x) VALUES (9)");
    assert_rows(
        &mut db,
        "SELECT x, a, b, c, e FROM d",
        &[vec![int(9), int(2), int(2), text("ab"), int(2)]],
    );
}

#[test]
fn parenthesized_default_with_default_values_and_insert_select() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(x, y DEFAULT (3*4))");
    exec(&mut db, "INSERT INTO d DEFAULT VALUES");
    exec(&mut db, "INSERT INTO d(x) SELECT 7");
    assert_rows(
        &mut db,
        "SELECT x, y FROM d ORDER BY rowid",
        &[vec![null(), int(12)], vec![int(7), int(12)]],
    );
}

#[test]
fn parenthesized_default_satisfies_not_null() {
    // §3.8 + §3.2: the omitted NOT NULL column takes its default, so this succeeds.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(x, y NOT NULL DEFAULT (1+1))");
    exec(&mut db, "INSERT INTO d(x) VALUES (9)");
    assert_rows(&mut db, "SELECT x, y FROM d", &[vec![int(9), int(2)]]);
}

#[test]
fn table_info_reports_the_default_expression_text() {
    // sqlite3 3.51: `1+1` and `'a' ||  'b'` (outer whitespace dropped, inner kept).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(x, y DEFAULT (1+1), z DEFAULT (  'a' ||  'b' ))");
    assert_rows(
        &mut db,
        "SELECT name, dflt_value FROM pragma_table_info('d') ORDER BY cid",
        &[
            vec![text("x"), null()],
            vec![text("y"), text("1+1")],
            vec![text("z"), text("'a' ||  'b'")],
        ],
    );
}

#[test]
fn parenthesized_default_survives_reopening_the_file() {
    // The default is rebuilt from the CREATE TABLE text in sqlite_schema on open,
    // which is also how a table written by sqlite3 is loaded.
    let dir = std::env::temp_dir().join(format!("msq-default-expr-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("d.db");
    let _ = std::fs::remove_file(&path);
    {
        let mut db = Connection::open(&path).unwrap();
        exec(&mut db, "CREATE TABLE d(x, y DEFAULT ('n' || 'ew'))");
    }
    {
        let mut db = Connection::open(&path).unwrap();
        exec(&mut db, "INSERT INTO d(x) VALUES (1)");
        assert_rows(&mut db, "SELECT x, y FROM d", &[vec![int(1), text("new")]]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
