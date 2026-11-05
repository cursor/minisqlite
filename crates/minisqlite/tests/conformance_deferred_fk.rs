//! Conformance battery: **DEFERRED foreign-key constraint timing** — a
//! `DEFERRABLE INITIALLY DEFERRED` constraint (and any constraint under
//! `PRAGMA defer_foreign_keys`) is checked at COMMIT, not at statement time.
//!
//! Every expectation is TRANSCRIBED FROM THE SPEC (`spec/sqlite-doc/foreignkeys.html`
//! §4.2 / §4.3, `spec/sqlite-doc/pragma.html` #pragma_defer_foreign_keys), never from what
//! the engine currently returns. A case that reveals an engine bug is left as a genuine
//! failing assertion rather than weakened to pass.
//!
//! Spec anchors:
//!   * §4.2 "Deferred Foreign Key Constraints": "Deferred foreign key constraints are not
//!     checked until the transaction tries to COMMIT. For as long as the user has an open
//!     transaction, the database is allowed to exist in a state that violates any number of
//!     deferred foreign key constraints. However, COMMIT will fail as long as foreign key
//!     constraints remain in violation."
//!   * §4.2: "If the current statement is not inside an explicit transaction ... deferred
//!     constraints behave the same as immediate constraints." (AUTOCOMMIT: deferred ==
//!     immediate — the immediate check is already correct there.)
//!   * §4.2: on a COMMIT that fails because of a deferred violation, "The transaction
//!     remains open."
//!   * §4.2 deferrable spellings: only `DEFERRABLE INITIALLY DEFERRED` is deferred; a bare
//!     `DEFERRABLE`, `DEFERRABLE INITIALLY IMMEDIATE`, and `NOT DEFERRABLE` are IMMEDIATE.
//!   * §4.3: a RESTRICT action "causes SQLite to return an error immediately ... Even if the
//!     foreign key constraint it is attached to is deferred." So RESTRICT stays IMMEDIATE.
//!   * pragma.html #pragma_defer_foreign_keys: "can be used to temporarily change all foreign
//!     key constraints to deferred regardless of how they are declared"; it "is automatically
//!     switched off at each COMMIT or ROLLBACK" and "defaults to OFF".
//!
//! The exact reject error text matches real sqlite: `FOREIGN KEY constraint failed`.
//!
//! Commit-point coverage (all exercised below): the deferred recheck runs at `COMMIT`, at a
//! `RELEASE` of a transaction-started SAVEPOINT (§4.2: a RELEASE that commits the transaction
//! is "subject to the same restrictions as a COMMIT"), and over BOTH rowid and WITHOUT ROWID
//! children. `SET DEFAULT` — whose post-action re-check can leave a child-side violation — is
//! deferred like NO ACTION (only RESTRICT stays immediate). `PRAGMA defer_foreign_keys` is
//! switched off at each COMMIT/ROLLBACK, including the implicit one that ends an autocommit
//! statement, but not at the START of a transaction (a value set just before `BEGIN` applies
//! to it, as in sqlite3 3.51.3), and its coverage is STICKY — toggling it OFF mid-transaction
//! still rescans a row it deferred while ON.
//!
//! REMAINING honest gaps (fail-closed or pre-existing, IDENTICAL in the immediate and deferred
//! paths, so deferred is never weaker than immediate):
//!   * An FK that references a WITHOUT ROWID PARENT is rejected with a loud "not supported"
//!     error by both the immediate check and the recheck (`parent_key_exists`, fail-closed).
//!   * `SET NULL` / `SET DEFAULT` over a generated-column CHILD is left as-is by both the
//!     immediate action and (therefore) the recheck — a pre-existing referential-action gap,
//!     not a deferral gap.

mod conformance;

use conformance::*;

/// Shared schema: a parent keyed by its INTEGER PRIMARY KEY and a child whose `pid`
/// references it with `DEFERRABLE INITIALLY DEFERRED`. `PRAGMA foreign_keys = ON` first,
/// since enforcement is OFF by default (foreignkeys.html §2).
fn deferred_schema(db: &mut Connection) {
    exec(db, "PRAGMA foreign_keys = ON");
    exec(db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(
        db,
        "CREATE TABLE cf(id INTEGER PRIMARY KEY, \
         pid REFERENCES pf(id) DEFERRABLE INITIALLY DEFERRED)",
    );
}

use minisqlite::Connection;

// ===========================================================================
// §4.2 — deferred constraint checked at COMMIT, not at statement time.
// ===========================================================================

#[test]
fn deferred_violation_allowed_midtxn_then_resolved_commits() {
    // §4.2 worked example: inside an explicit transaction a deferred child may reference a
    // not-yet-existing parent; resolving the violation before COMMIT lets COMMIT succeed.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    // 99 has no parent yet. A DEFERRED constraint permits the violation mid-transaction —
    // an IMMEDIATE FK (or autocommit) would error right here.
    exec(&mut db, "INSERT INTO cf VALUES (10, 99)");
    // Resolve the violation before COMMIT.
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    // No deferred constraint remains in violation → COMMIT succeeds.
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(99)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

#[test]
fn deferred_unresolved_fails_commit_and_leaves_transaction_open() {
    // §4.2: "COMMIT will fail as long as foreign key constraints remain in violation" and,
    // on that failure, "The transaction remains open."
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    // The INSERT SUCCEEDS (deferred), unlike an immediate FK which would error here.
    exec(&mut db, "INSERT INTO cf VALUES (11, 88)");
    // COMMIT fails: the deferred constraint is still violated.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");

    // The transaction is STILL OPEN. Prove it three ways:
    // (a) a read inside the transaction still sees the staged, uncommitted child row,
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(11), int(88)]]);
    // (b) a fresh BEGIN errors because a transaction is already active,
    let e2 = assert_exec_error(&mut db, "BEGIN");
    assert!(e2.to_string().contains("within a transaction"), "got: {e2}");
    // (c) a ROLLBACK succeeds and undoes the whole transaction (the child never lands).
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

#[test]
fn deferred_unresolved_commit_open_then_fixed_commits() {
    // Companion to the ROLLBACK case: after a COMMIT fails, the still-open transaction can
    // resolve the violation and COMMIT again successfully — the reference §4.2 workflow.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (11, 88)");
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Fix it inside the still-open transaction, then COMMIT succeeds.
    exec(&mut db, "INSERT INTO pf VALUES (88)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(11), int(88)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

#[test]
fn deferred_null_child_key_exempt_at_commit() {
    // MATCH SIMPLE (§4.1/§6): a NULL in the child key needs no parent, so it never violates
    // — including at the deferred COMMIT recheck.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (10, NULL)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), null()]]);
}

// ===========================================================================
// §4.3 — deferred parent-side NO ACTION defers; RESTRICT stays immediate.
// ===========================================================================

#[test]
fn deferred_parent_delete_no_action_orphan_caught_at_commit() {
    // The default ON DELETE is NO ACTION; declared deferred, a parent DELETE that orphans a
    // child is allowed mid-transaction (an immediate NO ACTION would error at the DELETE),
    // and the COMMIT recheck — a child-side scan — catches the dangling reference.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    // Orphans the child; deferred NO ACTION allows it now.
    exec(&mut db, "DELETE FROM pf WHERE id = 1");
    // Still orphaned at COMMIT → the child fails the commit.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Fix within the still-open transaction (re-add the parent), then COMMIT succeeds.
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(1)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

#[test]
fn deferred_restrict_still_errors_immediately() {
    // §4.3: RESTRICT is enforced IMMEDIATELY even on a deferred FK. The parent DELETE is
    // rejected AT THE STATEMENT, never deferred to COMMIT.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE cf(id INTEGER PRIMARY KEY, \
         pid REFERENCES pf(id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED)",
    );
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    // Immediate rejection at the DELETE, despite DEFERRABLE INITIALLY DEFERRED.
    let e = assert_exec_error(&mut db, "DELETE FROM pf WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected DELETE left the parent in place; the transaction can still commit clean.
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(1)]]);
}

#[test]
fn deferred_reintroduced_violation_fails_commit() {
    // A deferred violation resolved and then RE-introduced across statements is still caught
    // at COMMIT — the commit recheck reads the final state, not a running counter.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (10, 5)"); // orphan (deferred, allowed)
    exec(&mut db, "INSERT INTO pf VALUES (5)"); // resolves it
    exec(&mut db, "DELETE FROM pf WHERE id = 5"); // re-orphans it (deferred NO ACTION)
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(0));
}

// ===========================================================================
// AUTOCOMMIT: a deferred FK behaves exactly like an immediate one (§4.2).
// ===========================================================================

#[test]
fn deferred_in_autocommit_is_immediate() {
    // §4.2: outside an explicit transaction the implicit transaction commits as soon as the
    // statement finishes, so a deferred constraint "behave[s] the same as immediate" — the
    // orphan INSERT errors right here, with no BEGIN.
    let mut db = mem();
    deferred_schema(&mut db);
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (10, 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

#[test]
fn immediate_fk_inside_txn_still_errors_immediately() {
    // REGRESSION GUARD: an ordinary (NOT DEFERRABLE) FK is checked at statement time even
    // inside an explicit transaction — the orphan INSERT errors at the INSERT, not at COMMIT.
    // Deferring must NOT change this pre-existing immediate behavior.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (10, 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

#[test]
fn deferred_not_enforced_when_foreign_keys_off() {
    // OFF-COST + correctness: with `PRAGMA foreign_keys` OFF (the default) the commit recheck
    // early-returns, so even a deferred orphan commits untouched (constraints are DECLARED
    // but not enforced — foreignkeys.html §2/§5).
    let mut db = mem();
    // NB: deliberately do NOT turn foreign_keys ON.
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE cf(id INTEGER PRIMARY KEY, \
         pid REFERENCES pf(id) DEFERRABLE INITIALLY DEFERRED)",
    );
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (10, 99)"); // orphan, but FK enforcement is OFF
    exec(&mut db, "COMMIT"); // commits fine — no recheck when the pragma is OFF
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(99)]]);
}

// ===========================================================================
// PRAGMA defer_foreign_keys — defer ALL constraints for the current transaction.
// ===========================================================================

#[test]
fn defer_foreign_keys_defaults_off() {
    // pragma.html: `defer_foreign_keys` "defaults to OFF".
    let mut db = mem();
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
}

#[test]
fn defer_foreign_keys_pragma_defers_immediate_fk_and_autoclears_at_commit() {
    // pragma.html: defer_foreign_keys "temporarily change[s] all foreign key constraints to
    // deferred regardless of how they are declared" and "is automatically switched off at
    // each COMMIT". Here an ordinary NOT-DEFERRABLE FK is deferred to COMMIT by the pragma.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    // With the pragma ON, even this immediate FK is deferred: the orphan INSERT is allowed.
    exec(&mut db, "INSERT INTO cf VALUES (10, 77)");
    // Resolve before COMMIT, which then succeeds.
    exec(&mut db, "INSERT INTO pf VALUES (77)");
    exec(&mut db, "COMMIT");
    // Auto-switched OFF at COMMIT.
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(77)]]);
    // Proof the deferral really reverted: the SAME orphan INSERT now errors immediately in
    // autocommit (the immediate behavior is back).
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (20, 999)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
}

#[test]
fn defer_foreign_keys_pragma_unresolved_fails_commit() {
    // Under the defer pragma, an UNRESOLVED violation still fails COMMIT and leaves the
    // transaction open (§4.2), the same as a declared-deferred constraint.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "INSERT INTO cf VALUES (10, 77)"); // orphan, deferred by the pragma
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Transaction still open → ROLLBACK undoes it.
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
    // Auto-cleared at ROLLBACK too.
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
}

#[test]
fn defer_foreign_keys_pragma_autoclears_at_rollback() {
    // pragma.html: the pragma "is automatically switched off at each COMMIT or ROLLBACK".
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "INSERT INTO cf VALUES (10, 77)"); // deferred, allowed
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

// ===========================================================================
// §4.3 — SET DEFAULT defers like NO ACTION. Its post-action FK re-check CAN fail
// (the default may reference no surviving parent), so — unlike CASCADE (copies a
// live key) and SET NULL (MATCH-SIMPLE exempt) — a deferred SET DEFAULT must NOT
// raise at statement time; it fires the action and defers the rejection to COMMIT.
// ===========================================================================

/// A parent + a child whose `pid` DEFAULTs to 99 and references the parent with the given
/// `ON DELETE`/`ON UPDATE` action, `DEFERRABLE INITIALLY DEFERRED`. `foreign_keys` ON first.
fn set_default_schema(db: &mut Connection, action_clause: &str) {
    exec(db, "PRAGMA foreign_keys = ON");
    exec(db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(
        db,
        &format!(
            "CREATE TABLE cf(id INTEGER PRIMARY KEY, \
             pid INTEGER DEFAULT 99 REFERENCES pf(id) {action_clause} \
             DEFERRABLE INITIALLY DEFERRED)"
        ),
    );
}

#[test]
fn deferred_on_delete_set_default_defers_to_commit() {
    // Parent-DELETE SET DEFAULT re-check (enforce_parent_delete). ON DELETE SET DEFAULT sets the
    // child key to its column DEFAULT (99), which has no surviving parent — a normal FK violation.
    // On a DEFERRED FK the DELETE must SUCCEED mid-transaction (the action fires) and the COMMIT
    // must fail, txn left open.
    let mut db = mem();
    set_default_schema(&mut db, "ON DELETE SET DEFAULT");
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (1, 1)");
    exec(&mut db, "BEGIN");
    // Deferred → the DELETE succeeds and the SET DEFAULT action fires: cf.pid becomes 99.
    exec(&mut db, "DELETE FROM pf WHERE id = 1");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(0));
    // 99 has no parent → COMMIT fails, transaction stays open.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Resolve inside the still-open transaction (give 99 a parent), then COMMIT succeeds.
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

#[test]
fn deferred_on_update_set_default_defers_to_commit() {
    // Parent-UPDATE SET DEFAULT re-check (enforce_parent_update). ON UPDATE SET DEFAULT of a
    // referenced parent key resets the child key to its DEFAULT (99); deferred, the UPDATE must
    // SUCCEED and the COMMIT fail.
    let mut db = mem();
    set_default_schema(&mut db, "ON UPDATE SET DEFAULT");
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (1, 1)");
    exec(&mut db, "BEGIN");
    // Deferred → the UPDATE succeeds; SET DEFAULT fires: cf.pid becomes 99 (no parent 99).
    exec(&mut db, "UPDATE pf SET id = 2 WHERE id = 1");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
    assert_rows(&mut db, "SELECT id FROM pf", &[vec![int(2)]]);
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
}

#[test]
fn set_default_under_defer_pragma_is_deferred_not_raised() {
    // Proves the defer pragma is NOT silently ignored for a SET DEFAULT FK: an ordinary
    // NOT-DEFERRABLE `ON DELETE SET DEFAULT` FK, under `PRAGMA defer_foreign_keys = ON`, must
    // defer its post-action rejection to COMMIT rather than raising at the DELETE.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE cf(id INTEGER PRIMARY KEY, \
         pid INTEGER DEFAULT 99 REFERENCES pf(id) ON DELETE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (1, 1)");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    // Without the pragma this DELETE would raise at the parent-DELETE SET DEFAULT re-check; the
    // pragma defers it.
    exec(&mut db, "DELETE FROM pf WHERE id = 1");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Transaction still open → ROLLBACK restores the pre-DELETE state.
    exec(&mut db, "ROLLBACK");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(1)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

// ===========================================================================
// WITHOUT ROWID children — the deferred recheck walks the PK-index b-tree, so a
// deferred FK on a WR child is enforced at COMMIT exactly as the immediate child
// check enforces it at INSERT/UPDATE (deferred is not weaker than immediate).
// ===========================================================================

#[test]
fn deferred_without_rowid_child_violation_fails_commit() {
    // A WITHOUT ROWID child with a DEFERRABLE INITIALLY DEFERRED FK: the orphan INSERT is
    // skipped at statement time (deferred) and MUST be caught by the commit recheck — real
    // sqlite errors at COMMIT. (Previously the recheck early-returned on `without_rowid`,
    // silently committing the dangling reference.)
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(a TEXT PRIMARY KEY, \
         pid REFERENCES p(id) DEFERRABLE INITIALLY DEFERRED) WITHOUT ROWID",
    );
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO c VALUES ('x', 99)"); // deferred: skipped at INSERT
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Transaction still open → resolve and COMMIT (the WR child round-trips its PK-first record).
    exec(&mut db, "INSERT INTO p VALUES (99)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT a, pid FROM c", &[vec![text("x"), int(99)]]);
}

#[test]
fn immediate_without_rowid_child_still_errors_at_insert() {
    // Baseline pairing the deferred WR case: the SAME WR child with an IMMEDIATE FK errors at
    // the INSERT in autocommit. Proves the engine can enforce a WR child FK — so the deferred
    // recheck matching it (above) closes an asymmetry, not a uniform WR limitation.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(a TEXT PRIMARY KEY, pid REFERENCES p(id)) WITHOUT ROWID",
    );
    let e = assert_exec_error(&mut db, "INSERT INTO c VALUES ('x', 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));
}

// ===========================================================================
// PRAGMA defer_foreign_keys — sticky coverage + per-transaction lifetime.
// ===========================================================================

#[test]
fn defer_pragma_toggled_off_midtxn_still_fails_commit() {
    // ON→OFF toggle: a row deferred while the pragma was ON must STILL be rescanned at COMMIT
    // even after the pragma is turned OFF in the same transaction, or an FK-violating row would
    // commit with foreign_keys ON (contradicting §4.2). The sticky coverage flag guarantees it.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = 1");
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)"); // deferred by the pragma
    exec(&mut db, "PRAGMA defer_foreign_keys = 0"); // toggled back OFF mid-transaction
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0)); // live flag reports OFF
    // Coverage is sticky → COMMIT still rescans cf and finds the orphan.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

/// Parent/child pair with an ordinary (NOT DEFERRABLE) FK, `foreign_keys` ON.
fn defer_pragma_schema(db: &mut Connection) {
    exec(db, "PRAGMA foreign_keys = ON");
    exec(db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
}

#[test]
fn defer_pragma_set_before_begin_applies_to_the_transaction() {
    // pragma.html: the pragma "is automatically switched off at each COMMIT or ROLLBACK";
    // BEGIN does not switch it off. sqlite3 3.51.3 keeps a value set before BEGIN and defers
    // the orphan INSERT to COMMIT.
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "BEGIN");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)"); // deferred, resolved below
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
}

#[test]
fn defer_pragma_set_before_savepoint_applies_to_the_transaction() {
    // The SAVEPOINT-started analog: the value set before it applies, and the RELEASE that
    // commits the transaction switches it off (sqlite3 3.51.3).
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "SAVEPOINT s");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)");
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "RELEASE s");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(1));
}

#[test]
fn defer_pragma_set_before_begin_is_switched_off_by_rollback() {
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "BEGIN");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
}

#[test]
fn defer_pragma_switched_off_by_autocommit_insert() {
    // An autocommit INSERT commits its own implicit transaction, which switches the pragma
    // off (sqlite3 3.51.3 reports 0), so it no longer applies to a later BEGIN.
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "INSERT INTO pf VALUES (5)");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    exec(&mut db, "BEGIN");
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (1, 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
}

#[test]
fn defer_pragma_switched_off_by_autocommit_update_delete_and_ddl() {
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "UPDATE pf SET id = 1");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "DELETE FROM pf WHERE 0");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "CREATE TABLE z(a)");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "CREATE TABLE z2 AS SELECT 1 AS a");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
}

#[test]
fn defer_pragma_switched_off_by_failed_autocommit_statement() {
    // A statement that fails while running rolls back its implicit transaction, which also
    // switches the pragma off (sqlite3 3.51.3). In autocommit the FK is still checked at the
    // statement, so the orphan is rejected.
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (1, 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));

    exec(&mut db, "CREATE TABLE u(x UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES (1)");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    let e = assert_exec_error(&mut db, "INSERT INTO u VALUES (1)");
    assert!(e.to_string().contains("UNIQUE constraint failed"), "got: {e}");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
}

#[test]
fn defer_pragma_kept_by_statements_that_open_no_transaction() {
    // sqlite3 3.51.3 keeps the value across a statement that touches no table and across a
    // statement rejected before it runs (no such table).
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    assert_scalar(&mut db, "PRAGMA foreign_keys", int(1));
    assert_scalar(&mut db, "SELECT 1", int(1));
    assert_exec_error(&mut db, "INSERT INTO nosuch VALUES (1)");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
}

#[test]
fn defer_pragma_on_then_off_in_autocommit_defers_nothing() {
    // ON then OFF before BEGIN leaves nothing deferred: the orphan errors at the INSERT, and
    // a row that violated the FK before `foreign_keys` was turned on does not fail COMMIT
    // (sqlite3 3.51.3).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "INSERT INTO cf VALUES (1, 42)"); // foreign_keys still OFF
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "PRAGMA defer_foreign_keys = OFF");
    exec(&mut db, "BEGIN");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    let e = assert_exec_error(&mut db, "INSERT INTO cf VALUES (2, 99)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "INSERT INTO pf VALUES (2)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM pf", int(1));
}

#[test]
fn defer_pragma_kept_after_failed_commit() {
    // A COMMIT that fails on a deferred violation leaves the transaction open (§4.2) and,
    // in sqlite3 3.51.3, leaves the pragma ON; the COMMIT that succeeds switches it off.
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)");
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(1));
}

#[test]
fn defer_pragma_kept_after_failed_release() {
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "SAVEPOINT s");
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)");
    let e = assert_exec_error(&mut db, "RELEASE s");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

#[test]
fn defer_pragma_kept_by_rollback_to_and_nested_release() {
    // ROLLBACK TO and a nested RELEASE do not end the transaction, so the pragma stays ON
    // until the real COMMIT (sqlite3 3.51.3). The pragma is not itself rolled back by
    // ROLLBACK TO a savepoint opened before it was set.
    let mut db = mem();
    defer_pragma_schema(&mut db);
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT s");
    exec(&mut db, "PRAGMA defer_foreign_keys = ON");
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)");
    exec(&mut db, "ROLLBACK TO s");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "RELEASE s");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1));
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

#[test]
fn nested_savepoint_does_not_drop_enclosing_txn_deferral() {
    // Guard on the `!txn_active()` condition of the savepoint-start reset: a NESTED savepoint
    // (a transaction already active) must NOT reset a deferral the enclosing transaction turned
    // on — otherwise a row that should defer would be checked immediately and error.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE cf(id INTEGER PRIMARY KEY, pid REFERENCES pf(id))");
    exec(&mut db, "BEGIN");
    exec(&mut db, "PRAGMA defer_foreign_keys = 1"); // defer for this transaction
    exec(&mut db, "SAVEPOINT s"); // NESTED (txn already active) → must keep the deferral
    assert_scalar(&mut db, "PRAGMA defer_foreign_keys", int(1)); // still ON after the nested savepoint
    // The deferral survived the nested SAVEPOINT → the orphan INSERT is allowed mid-txn.
    exec(&mut db, "INSERT INTO cf VALUES (1, 99)");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(1), int(99)]]);
    exec(&mut db, "RELEASE s"); // nested release (inside BEGIN) is not a commit point
    // The orphan is caught at the real COMMIT.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

// ===========================================================================
// RELEASE of a transaction-started SAVEPOINT is a COMMIT point for the recheck
// (§4.2: "subject to the same restrictions as a COMMIT").
// ===========================================================================

#[test]
fn deferred_release_of_savepoint_transaction_is_a_commit_point() {
    // A bare SAVEPOINT (no BEGIN) starts a transaction; RELEASEing its outermost savepoint
    // commits it. A deferred violation open at that RELEASE must fail it and leave the
    // transaction (and the savepoint) OPEN, exactly like a failed COMMIT.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "SAVEPOINT s");
    exec(&mut db, "INSERT INTO cf VALUES (10, 99)"); // deferred (txn active via savepoint)
    let e = assert_exec_error(&mut db, "RELEASE s");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Still open: the savepoint is addressable and the staged row is visible.
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(99)]]);
    // Resolve, then RELEASE again → this time it commits.
    exec(&mut db, "INSERT INTO pf VALUES (99)");
    exec(&mut db, "RELEASE s");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(99)]]);
    // The RELEASE committed and ended the transaction → a ROLLBACK now has none to roll back.
    let e2 = assert_exec_error(&mut db, "ROLLBACK");
    assert!(e2.to_string().contains("no transaction"), "got: {e2}");
}

#[test]
fn deferred_release_inside_explicit_begin_is_not_a_commit_point() {
    // Guard against over-triggering: a RELEASE of a savepoint nested inside an explicit BEGIN
    // does NOT end the transaction, so it must NOT run the recheck — the deferred orphan is
    // caught only at the real COMMIT.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT s");
    exec(&mut db, "INSERT INTO cf VALUES (10, 99)"); // deferred orphan
    exec(&mut db, "RELEASE s"); // NOT a commit point (explicit BEGIN wraps) → succeeds
    // The BEGIN transaction is still open; the orphan is caught at COMMIT.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM cf", int(0));
}

// ===========================================================================
// Coverage for modified-but-previously-untested deferred paths.
// ===========================================================================

#[test]
fn deferred_parent_update_no_action_orphan_caught_at_commit() {
    // The parent-UPDATE NO ACTION deferral (companion to the tested parent-DELETE one):
    // updating a referenced parent key inside a deferred transaction is allowed mid-txn and
    // the resulting orphan is caught at COMMIT.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    // Deferred NO ACTION → the parent-key UPDATE is allowed even though it orphans cf(10).
    exec(&mut db, "UPDATE pf SET id = 2 WHERE id = 1");
    assert_rows(&mut db, "SELECT id FROM pf", &[vec![int(2)]]);
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(1)]]);
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // Re-add the parent key inside the still-open transaction, then COMMIT succeeds.
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(1)]]);
}

#[test]
fn deferred_child_update_orphan_caught_at_commit() {
    // The deferred CHILD path via UPDATE (previously exercised only via INSERT). Updating a
    // child FK column to a non-existent parent is allowed mid-txn and caught at COMMIT.
    let mut db = mem();
    deferred_schema(&mut db);
    exec(&mut db, "INSERT INTO pf VALUES (1)");
    exec(&mut db, "INSERT INTO cf VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    exec(&mut db, "UPDATE cf SET pid = 88 WHERE id = 10"); // deferred: 88 has no parent
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(88)]]);
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "ROLLBACK");
    assert_rows(&mut db, "SELECT id, pid FROM cf", &[vec![int(10), int(1)]]);
}

#[test]
fn deferred_composite_fk_orphan_caught_at_commit() {
    // A deferred COMPOSITE FK (non-rowid parent key → the general parent-scan probe): the
    // multi-column orphan is allowed mid-txn and caught at COMMIT.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE pf(a, b, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE cf(id INTEGER PRIMARY KEY, x, y, \
         FOREIGN KEY(x, y) REFERENCES pf(a, b) DEFERRABLE INITIALLY DEFERRED)",
    );
    exec(&mut db, "INSERT INTO pf VALUES (1, 2)");
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO cf VALUES (10, 1, 3)"); // (1,3) absent from pf; deferred
    assert_rows(&mut db, "SELECT id, x, y FROM cf", &[vec![int(10), int(1), int(3)]]);
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    exec(&mut db, "INSERT INTO pf VALUES (1, 3)"); // resolve the composite key
    exec(&mut db, "COMMIT");
    assert_rows(&mut db, "SELECT id, x, y FROM cf", &[vec![int(10), int(1), int(3)]]);
}
