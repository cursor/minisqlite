//! Conformance tests: `x IN (SELECT y ...)` compares under the collation sqlite3 picks
//! for `x = y`, including the collation of the subquery's result column.
//!
//! `spec/sqlite-doc/datatype3.html` §7.1: "The expression "x IN (SELECT y ...)" is
//! handled in the same way as the expression "x = y" for the purposes of determining
//! the collating sequence." For `x = y` the order is: an explicit postfix `COLLATE` on
//! the left operand, else one on the right, else the left operand's column collation,
//! else the right operand's, else BINARY. So a literal or computed `x` takes the
//! collation of the subquery column `y`, while a column `x` keeps its own.
//!
//! Every expected value was produced by running the same SQL in sqlite3 3.51.3.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// `users.email` is NOCASE and holds `'Alice@Example.com'`. `b.addr` is a BINARY
/// column holding the lower-case address, `ev` is the fixture from the report.
fn db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE users(email TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO users VALUES ('Alice@Example.com')");
    exec(&mut db, "CREATE TABLE ev(addr TEXT, n INT)");
    exec(&mut db, "INSERT INTO ev VALUES (' alice@example.com', 1), ('bob@example.com', 2)");
    exec(&mut db, "CREATE TABLE b(addr TEXT)");
    exec(&mut db, "INSERT INTO b VALUES ('alice@example.com')");
    db
}

// -----------------------------------------------------------------------------
// The subject defines no collation: the subquery column's collation applies.
// -----------------------------------------------------------------------------

#[test]
fn literal_subject_uses_the_subquery_columns_collation() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN (SELECT email FROM users)", int(1));
    assert_scalar(&mut db, "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT email FROM users)", int(1));
}

#[test]
fn not_in_uses_the_subquery_columns_collation() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'alice@example.com' NOT IN (SELECT email FROM users)", int(0));
    assert_rows(
        &mut db,
        "SELECT n FROM ev WHERE trim(addr) NOT IN (SELECT email FROM users)",
        &[vec![int(2)]],
    );
}

#[test]
fn function_and_operator_subjects_use_the_subquery_columns_collation() {
    let mut db = db();
    assert_rows(
        &mut db,
        "SELECT n FROM ev WHERE trim(addr) IN (SELECT email FROM users)",
        &[vec![int(1)]],
    );
    assert_scalar(
        &mut db,
        "SELECT ('alice@example.com' || '') IN (SELECT email FROM users)",
        int(1),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE trim(addr) IN (SELECT email FROM users)",
        int(1),
    );
}

#[test]
fn scalar_subquery_subject_uses_the_subquery_columns_collation() {
    let mut db = db();
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE (SELECT addr) IN (SELECT email FROM users)",
        int(1),
    );
}

#[test]
fn explicit_collate_in_the_subquery_applies_to_a_literal_subject() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'a' IN (SELECT 'A' COLLATE NOCASE)", int(1));
}

#[test]
fn unary_plus_and_cast_keep_the_subquery_columns_collation() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN (SELECT +email FROM users)", int(1));
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN (SELECT (email) FROM users)", int(1));
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' IN (SELECT CAST(email AS TEXT) FROM users)",
        int(1),
    );
}

#[test]
fn star_and_table_forms_use_the_columns_collation() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT * FROM users)", int(1));
    assert_scalar(&mut db, "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT users.* FROM users)", int(1));
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN users", int(1));
}

#[test]
fn derived_table_cte_and_view_columns_carry_the_collation() {
    let mut db = db();
    exec(&mut db, "CREATE VIEW vu AS SELECT email AS e FROM users");
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' IN (SELECT e FROM (SELECT email AS e FROM users))",
        int(1),
    );
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' IN (WITH c AS (SELECT email AS e FROM users) SELECT e FROM c)",
        int(1),
    );
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN (SELECT e FROM vu)", int(1));
}

#[test]
fn grouped_distinct_ordered_and_limited_subqueries_keep_the_collation() {
    let mut db = db();
    for sql in [
        "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT email FROM users GROUP BY email)",
        "SELECT 'alice@example.com' IN (SELECT DISTINCT email FROM users)",
        "SELECT 'alice@example.com' IN (SELECT email FROM users ORDER BY 1)",
        "SELECT 'alice@example.com' IN (SELECT email FROM users LIMIT 1)",
        "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT email FROM users JOIN ev ON 1)",
    ] {
        assert_scalar(&mut db, sql, int(1));
    }
}

#[test]
fn correlated_outer_column_in_the_subquery_result_carries_its_collation() {
    let mut db = db();
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM users WHERE 'ALICE@EXAMPLE.COM' IN (SELECT users.email FROM b)",
        int(1),
    );
}

#[test]
fn compound_subquery_uses_its_last_arm() {
    // sqlite3 takes the collation from the result expression of the rightmost arm.
    let mut db = db();
    for (sql, want) in [
        ("SELECT 'alice@example.com' IN (SELECT 'zz' UNION ALL SELECT email FROM users)", 1),
        ("SELECT 'alice@example.com' IN (SELECT email FROM users UNION ALL SELECT 'zz')", 0),
        ("SELECT 'alice@example.com' IN (SELECT 'zz' UNION SELECT email FROM users)", 1),
        ("SELECT 'alice@example.com' IN (SELECT email FROM users UNION SELECT 'zz')", 0),
        (
            "SELECT 'alice@example.com' IN \
             (SELECT email FROM users UNION ALL SELECT 'zz' UNION ALL SELECT email FROM users)",
            1,
        ),
        (
            "SELECT 'alice@example.com' IN \
             (SELECT email FROM users INTERSECT SELECT email FROM users)",
            1,
        ),
    ] {
        assert_scalar(&mut db, sql, int(want));
    }
}

#[test]
fn row_value_in_uses_each_subquery_columns_collation() {
    let mut db = db();
    exec(&mut db, "CREATE TABLE r(a TEXT, b TEXT COLLATE RTRIM)");
    exec(&mut db, "INSERT INTO r VALUES ('x', 'y  ')");
    assert_scalar(&mut db, "SELECT ('x', 'y') IN (SELECT a, b FROM r)", int(1));
    assert_scalar(&mut db, "SELECT ('X', 'y') IN (SELECT a, b FROM r)", int(0));
    assert_scalar(
        &mut db,
        "SELECT ('alice@example.com', 1) IN (SELECT email, 1 FROM users)",
        int(1),
    );
    assert_scalar(
        &mut db,
        "SELECT (trim(' alice@example.com'), 1) IN (SELECT email, 1 FROM users)",
        int(1),
    );
}

#[test]
fn nulls_in_a_nocase_subquery_column() {
    let mut db = db();
    exec(&mut db, "CREATE TABLE nn(v TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO nn VALUES ('A'), (NULL)");
    assert_rows(
        &mut db,
        "SELECT 'a' IN (SELECT v FROM nn), 'b' IN (SELECT v FROM nn), \
         'a' NOT IN (SELECT v FROM nn), 'b' NOT IN (SELECT v FROM nn)",
        &[vec![int(1), null(), int(0), null()]],
    );
}

// -----------------------------------------------------------------------------
// Cases where the subquery column's collation does NOT apply.
// -----------------------------------------------------------------------------

#[test]
fn column_subject_keeps_its_own_collation() {
    let mut db = db();
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE addr IN (SELECT email FROM users)",
        int(0),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE +addr IN (SELECT email FROM users)",
        int(0),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE addr NOT IN (SELECT email FROM users)",
        int(1),
    );
}

#[test]
fn explicit_collate_on_the_subject_wins() {
    let mut db = db();
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' COLLATE BINARY IN (SELECT email FROM users)",
        int(0),
    );
    assert_scalar(
        &mut db,
        "SELECT ('alice@example.com' COLLATE BINARY, 1) IN (SELECT email, 1 FROM users)",
        int(0),
    );
}

#[test]
fn explicit_collate_in_the_subquery_beats_a_column_subject() {
    let mut db = db();
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM b WHERE addr IN (SELECT 'ALICE@EXAMPLE.COM' COLLATE NOCASE)",
        int(1),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM users WHERE email IN (SELECT 'alice@example.com' COLLATE BINARY)",
        int(0),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM users WHERE email IN (SELECT 'alice@example.com')",
        int(1),
    );
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' IN (SELECT email COLLATE BINARY FROM users)",
        int(0),
    );
}

#[test]
fn computed_subquery_results_define_no_collation() {
    let mut db = db();
    for sql in [
        "SELECT 'ALICE@EXAMPLE.COM' IN (SELECT lower(email) FROM users)",
        "SELECT 'alice@example.com' IN (SELECT max(email) FROM users)",
        "SELECT 'alice@example.com' IN (SELECT email || '' FROM users)",
        "SELECT 'alice@example.com' IN (VALUES ('Alice@Example.com'))",
    ] {
        assert_scalar(&mut db, sql, int(0));
    }
    assert_scalar(
        &mut db,
        "SELECT 'alice@example.com' IN (SELECT lower(email) FROM users)",
        int(1),
    );
}

#[test]
fn value_list_uses_only_the_subjects_collation() {
    let mut db = db();
    assert_scalar(&mut db, "SELECT 'alice@example.com' IN ('Alice@Example.com')", int(0));
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM users WHERE 'alice@example.com' IN (email, 'zz')",
        int(0),
    );
}
