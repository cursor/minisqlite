//! Shared column-constraint enforcement for the DML write operators (`INSERT` and
//! `UPDATE`).
//!
//! `NOT NULL` is a column-level constraint the catalog records per column
//! (`ColumnDef.not_null`), exactly the way the `UNIQUE` / `PRIMARY KEY` facts drive
//! index maintenance in [`super::dml_index`]. The semantics — which columns are
//! checked, the rowid-alias exemption, and how each `ON CONFLICT` policy resolves a
//! violation — must be IDENTICAL across `INSERT` and `UPDATE`, so the one authoritative
//! copy lives here and both operators call it, rather than a per-operator copy that can
//! drift.
//!
//! `CHECK` is a table/column constraint the plan carries as already-bound expressions
//! ([`CheckConstraint`], one per `CHECK` clause). [`enforce_checks`] evaluates each
//! against the new row; a result that casts to numeric zero (lang_createtable.html §3.7)
//! is a violation, resolved under the statement's `ON CONFLICT` policy exactly as
//! `NOT NULL` is (skip the row on `OR IGNORE`, else the constraint error). Like `NOT NULL`,
//! its behavior must be identical across `INSERT` and `UPDATE`, so it lives here too and
//! both operators call it.
//!
//! Effect surface: [`enforce_not_null`] is PURE — it inspects the row values and the
//! schema ([`TableDef`]) already in hand, reads no pager and writes nothing, and
//! returns a [`ConstraintOutcome`] (proceed / skip the row) or the constraint error.
//! [`enforce_checks`] is NOT pure: a `CHECK` is an arbitrary expression, so it takes an
//! [`EvalCtx`] to evaluate each check against the row — but it still only reads (through
//! the context) and writes nothing, returning the same [`ConstraintOutcome`] (or the
//! constraint error) on the first violation. The caller acts on the result. `DELETE` is
//! deliberately not a caller of either: removing a row cannot violate a column `NOT NULL`
//! or a `CHECK`.
//!
//! The STRICT datatype rule (`stricttables.html` §2) lives here for the same reason:
//! [`enforce_strict_types`] is pure like [`enforce_not_null`], runs right after it on the
//! affinity-applied row, and is shared by every INSERT / UPDATE / UPSERT write path.

use minisqlite_catalog::TableDef;
use minisqlite_expr::{eval, truth};
use minisqlite_plan::{CheckConstraint, OnConflict};
use minisqlite_types::{ConstraintKind, Error, Result, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::runtime::Runtime;

/// Outcome of a per-row constraint check ([`enforce_not_null`] or [`enforce_checks`])
/// under an `ON CONFLICT` policy: either the row may proceed to the rest of the write, or
/// the policy (`IGNORE`) says to skip it silently — the caller `continue`s to the next
/// row. A hard violation (`ABORT` / `FAIL` / `ROLLBACK`, plus `REPLACE` — which folds to
/// ABORT for a `CHECK`, and is a documented interim for `NOT NULL`) is not represented
/// here: it short-circuits as an `Err` from the helper, mirroring how `detect_conflict` in
/// the two operators returns the constraint error directly rather than an `Action`.
///
/// Shared by both helpers rather than a per-constraint copy: `NOT NULL` and `CHECK` resolve
/// an `IGNORE` violation identically (skip the one offending row, no error, prior rows
/// preserved), so one `Proceed`/`Skip` outcome keeps that rule in a single place instead of
/// two parallel enums that could drift.
#[derive(Debug)]
pub(crate) enum ConstraintOutcome {
    Proceed,
    Skip,
}

/// Enforce every column's `NOT NULL` constraint over the logical row
/// `[c0..c_{N-1}, ..]` (the alias slot already holds the rowid). Only the first
/// `def.columns.len()` values are inspected; a trailing rowid register the caller may
/// have appended is ignored. `logical[j]` for `j` in `0..N` is guaranteed valid: both
/// operators build the logical row to width `N` (== `def.columns.len()`, which they
/// validate before this call), matching the direct-indexing convention `dml_index`
/// uses for index-key columns.
///
/// The `INTEGER PRIMARY KEY` column (the rowid alias, `def.rowid_alias == Some(j)`) is
/// EXEMPT: its logical slot always carries `Integer(rowid)`, and a NULL supplied there
/// means "auto-assign a rowid", not a violation — so it is skipped even when it is
/// declared `NOT NULL`.
///
/// On the first NULL in a checked `NOT NULL` column the outcome follows `on_conflict`:
/// `IGNORE` yields [`ConstraintOutcome::Skip`]; every other policy raises the `NOT NULL`
/// constraint error. Columns are scanned in declaration order, so the first violating
/// column (lowest index) is the one reported, as SQLite does.
pub(crate) fn enforce_not_null(
    def: &TableDef,
    logical: &[Value],
    on_conflict: &OnConflict,
) -> Result<ConstraintOutcome> {
    for (j, col) in def.columns.iter().enumerate() {
        // Skip columns without the constraint, and the rowid-alias column: its slot is
        // the rowid (a NULL there is an auto-assign request, never a violation).
        if !col.not_null || def.rowid_alias == Some(j) {
            continue;
        }
        if logical[j].is_null() {
            match on_conflict {
                // IGNORE: skip the offending row silently. The caller `continue`s; in
                // INSERT that happens before the `next_auto` high-water commit, so a
                // skipped auto-rowid row consumes no rowid (same rule as an ignored
                // UNIQUE conflict).
                OnConflict::Ignore => return Ok(ConstraintOutcome::Skip),
                // REPLACE substitutes the column DEFAULT for the NULL, falling back to
                // ABORT when the column has no DEFAULT (lang_conflict.html). This
                // executor cannot evaluate a column DEFAULT yet — the plan carries no
                // parsed default exprs (a separate, known gap) — so both cases raise the
                // constraint error here rather than ever storing a NULL:
                //   * `col.default == None`  => the error is CORRECT SQLite behavior
                //     (no default => ABORT), not a narrowing.
                //   * `col.default == Some(_)` => the same error is a DOCUMENTED interim
                //     NARROWING: SQLite would substitute the default value. It becomes
                //     correct for free once defaults are threaded through the plan; the
                //     one-line fix belongs on this arm (branch on `col.default`).
                // ABORT / FAIL / ROLLBACK all raise the error too (the FAIL-vs-ABORT
                // statement-atomicity distinction is handled outside this helper).
                // Spelled out with no catch-all so adding an `OnConflict` variant forces
                // a decision here at compile time.
                OnConflict::Replace
                | OnConflict::Abort
                | OnConflict::Fail
                | OnConflict::Rollback => return Err(not_null_error(def, j)),
            }
        }
    }
    Ok(ConstraintOutcome::Proceed)
}

/// The error for a `NOT NULL` violation on `def.columns[j]`: `NOT NULL constraint
/// failed: <table>.<column>`, matching SQLite's message and the `rowid_conflict_error`
/// style in `insert.rs` / `update.rs`.
fn not_null_error(def: &TableDef, j: usize) -> Error {
    Error::constraint(ConstraintKind::NotNull, format!("{}.{}", def.name, def.columns[j].name))
}

/// Enforce the STRICT datatype rule (`stricttables.html` §2) over the logical row
/// `[c0..c_{N-1}, ..]`, whose values have already had the column affinity applied. A
/// no-op unless [`TableDef::strict`]. For each column, a non-NULL value must be of the
/// declared type: INT / INTEGER an integer, REAL a real (REAL affinity has already turned
/// an integer into one), TEXT a text, BLOB a blob; ANY accepts anything. NULL is left to
/// [`enforce_not_null`], which the caller runs first, as sqlite does.
///
/// Two kinds of column are skipped, both matching sqlite 3.51.3: the rowid alias (its slot
/// holds the rowid, already forced to an integer or rejected with "datatype mismatch"),
/// and a GENERATED column (sqlite does not type-check a generated value).
///
/// A violation is an error under every `ON CONFLICT` policy (sqlite does not let `OR
/// IGNORE` skip it), so this returns no [`ConstraintOutcome`]. The message is sqlite's:
/// `cannot store TEXT value in INTEGER column t.c`. It has no "constraint failed" phrase,
/// so it is built directly rather than through [`Error::constraint`]; it is still an
/// `Error::Constraint` (SQLITE_CONSTRAINT, extended code SQLITE_CONSTRAINT_DATATYPE).
pub(crate) fn enforce_strict_types(def: &TableDef, logical: &[Value]) -> Result<()> {
    if !def.strict {
        return Ok(());
    }
    for (j, col) in def.columns.iter().enumerate() {
        if def.rowid_alias == Some(j) || col.generated.is_some() {
            continue;
        }
        // The builder has already rejected a STRICT column whose type is missing or not
        // one of the six names, so the trimmed declared type is one of them here.
        let declared = col.declared_type.as_deref().unwrap_or("").trim();
        let ok = match &logical[j] {
            Value::Null => true,
            Value::Integer(_) => {
                declared.eq_ignore_ascii_case("INT") || declared.eq_ignore_ascii_case("INTEGER")
            }
            Value::Real(_) => declared.eq_ignore_ascii_case("REAL"),
            Value::Text(_) => declared.eq_ignore_ascii_case("TEXT"),
            Value::Blob(_) => declared.eq_ignore_ascii_case("BLOB"),
        } || declared.eq_ignore_ascii_case("ANY");
        if !ok {
            let value_type = match &logical[j] {
                Value::Null => "NULL",
                Value::Integer(_) => "INT",
                Value::Real(_) => "REAL",
                Value::Text(_) => "TEXT",
                Value::Blob(_) => "BLOB",
            };
            return Err(Error::Constraint(format!(
                "cannot store {value_type} value in {} column {}.{}",
                declared.to_ascii_uppercase(),
                def.name,
                col.name
            )));
        }
    }
    Ok(())
}

/// Enforce every table/column `CHECK` constraint over the new row the caller passes as
/// `logical`, in the layout each `check.expr` was bound against: `[c0..c_{N-1}, rowid]`,
/// register `i` == column `i`, with an INTEGER PRIMARY KEY alias column (and a bare
/// `rowid`) resolving to the TRAILING register `N` (see `minisqlite_plan::compile::check`
/// and `bind::scope`). The INSERT/UPDATE callers build exactly that row — the same one
/// RETURNING evaluates against — so a `CHECK(id > 0)` on the rowid alias reads register
/// `N`, not the column-`i` slot. (This is one register wider than the `[c0..c_{N-1}]`
/// slice [`enforce_not_null`] inspects, which never needs the rowid register.)
///
/// Unlike [`enforce_not_null`], this is NOT pure: a `CHECK` is an ARBITRARY expression
/// (it may reference any column or call a function), so evaluating it needs the full
/// [`EvalCtx`]. It still writes nothing — it only reads through the context — and the
/// caller acts on the outcome before the row is written, so a violation leaves storage
/// untouched.
///
/// VIOLATION RULE (lang_createtable.html §3.7): each `check.expr` "is evaluated and
/// cast to a NUMERIC value in the same way as a CAST expression. If the result is zero
/// (integer value 0 or real value 0.0), then a constraint violation has occurred. If
/// the CHECK expression evaluates to NULL, or any other non-zero value, it is not a
/// constraint violation." That is exactly [`check_violated`] over the evaluated value.
/// Checks run in slice order; the first violation decides the outcome (below), and all
/// checks passing returns [`ConstraintOutcome::Proceed`].
///
/// CONFLICT RESOLUTION mirrors [`enforce_not_null`]: a `CHECK` participates in the
/// STATEMENT-level conflict algorithm, so `INSERT OR IGNORE` / `UPDATE OR IGNORE` override
/// the default and SKIP the one offending row rather than failing the whole statement
/// (lang_conflict.html: "The algorithm specified in the OR clause of an INSERT or UPDATE
/// overrides any algorithm specified in a CREATE TABLE"; its FAIL entry lists CHECK among
/// the constraints the clause applies to). So on the first violation: `IGNORE` returns
/// [`ConstraintOutcome::Skip`] (no error — the caller drops the row and prior rows stand);
/// `ABORT` / `FAIL` / `ROLLBACK` — and `REPLACE`, which folds to ABORT for a CHECK
/// (lang_conflict.html) — raise the [`ConstraintKind::Check`] error carrying the plan's
/// `detail`. (§4.1's "a conflict clause on a CHECK has no effect / defaults to ABORT" is
/// the constraint-DEFINITION default in the schema, NOT the statement-level `OR` clause,
/// which DOES apply.) FAIL vs ABORT differ only in
/// whether PRIOR rows of the same statement survive; that distinction is the engine txn
/// wrapper's, not this helper's (a pre-existing limit shared with NOT NULL/UNIQUE), so
/// both raise here.
pub(crate) fn enforce_checks(
    checks: &[CheckConstraint],
    logical: &[Value],
    on_conflict: &OnConflict,
    ctx: &mut EvalCtx<'_>,
) -> Result<ConstraintOutcome> {
    for c in checks {
        let v = eval(&c.expr, logical, ctx)?;
        if check_violated(&v) {
            return match on_conflict {
                // OR IGNORE: skip only the offending row (no error, prior rows preserved),
                // exactly as enforce_not_null resolves an IGNORE NULL — the caller `continue`s.
                OnConflict::Ignore => Ok(ConstraintOutcome::Skip),
                // REPLACE folds to ABORT for a CHECK (lang_conflict.html); FAIL/ROLLBACK
                // raise like ABORT here (FAIL's prior-row atomicity is the txn wrapper's job,
                // not this helper's). Spelled out with no catch-all so a new OnConflict
                // variant forces a decision at compile time.
                OnConflict::Replace | OnConflict::Abort | OnConflict::Fail | OnConflict::Rollback => {
                    Err(Error::constraint(ConstraintKind::Check, &c.detail))
                }
            };
        }
    }
    Ok(ConstraintOutcome::Proceed)
}

/// Enforce `checks` over a freshly-built width-`n` DML row (`row`), resolving a violation
/// under `on_conflict`. This is the row-layout choreography wrapped around [`enforce_checks`],
/// shared by INSERT (step 5.6) and UPDATE (step 6.6) so the ONE subtle invariant below stays
/// identical across them — the same reason [`enforce_checks`] itself lives here, and the exact
/// invariant whose drift (an alias `CHECK` reading register `N` on the wrong-width row) was a
/// real bug once.
///
/// LAYOUT: each CHECK predicate is bound over `[c0..c_{N-1}, rowid]`, where an INTEGER PRIMARY
/// KEY alias column (and a bare `rowid`) resolves to the TRAILING register `N` (see
/// `minisqlite_plan::compile::check`). So this appends `Integer(rowid)` as register `N`, runs
/// the checks, then truncates `row` back to width `n` — the shape the write path needs. `row`
/// MUST be built at capacity `n + 1` so the append never reallocates per row. The `truncate`
/// runs BEFORE the result is propagated, so the width-`n` invariant holds on the error/`Skip`
/// path too. The [`EvalCtx`] (a CHECK is an arbitrary expression) is scoped so its `env`/`rt`
/// reborrow closes before the caller's exclusive write.
///
/// Returns [`ConstraintOutcome::Proceed`] when all checks pass — including the no-checks fast
/// path, which does no push/eval, so a table without CHECKs pays nothing — and otherwise defers
/// to [`enforce_checks`] for the per-`on_conflict` outcome (`IGNORE` → `Skip`, else the error).
pub(crate) fn enforce_checks_over_new_row(
    checks: &[CheckConstraint],
    row: &mut Vec<Value>,
    n: usize,
    rowid: i64,
    on_conflict: &OnConflict,
    env: Env<'_>,
    rt: &mut Runtime,
) -> Result<ConstraintOutcome> {
    if checks.is_empty() {
        return Ok(ConstraintOutcome::Proceed);
    }
    row.push(Value::Integer(rowid));
    let outcome = {
        let mut ctx = EvalCtx { rt, env, outer: &*row };
        enforce_checks(checks, &*row, on_conflict, &mut ctx)
    };
    row.truncate(n);
    outcome
}

/// Whether a `CHECK` expression's evaluated result `v` is a constraint VIOLATION
/// (lang_createtable.html §3.7): the value cast to NUMERIC is zero (`0` or `0.0`).
///
/// This is exactly the engine's boolean coercion [`minisqlite_expr::truth`] reading
/// `Some(false)` — the SAME rule `WHERE` / `CASE WHEN` use — so a NULL result (`None`,
/// "not a violation") and any non-zero value (`Some(true)`) both pass, and only a value
/// that coerces to numeric zero fails. Pure and total over [`Value`], so it is
/// exhaustively testable without an evaluator; [`enforce_checks`] layers the expression
/// eval and the error on top.
///
/// `truth`'s CAST-style leading-prefix numeric coercion — NOT column-affinity coercion
/// — is what makes this match §3.7's "cast … in the same way as a CAST expression" on
/// the text/blob edges: affinity coercion leaves a non-numeric `TEXT` and every `BLOB`
/// untouched, so `CHECK('abc')` / `CHECK(x'00')` would wrongly pass; casting them to
/// numeric yields `0`, so both correctly fail, matching real sqlite.
fn check_violated(v: &Value) -> bool {
    truth(v) == Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::init_database;
    use minisqlite_catalog::{ColumnDef, SchemaCatalog};
    use minisqlite_expr::EvalExpr;
    use minisqlite_pager::MemPager;
    use minisqlite_plan::{Plan, PlanNode};
    // `Env` and `Runtime` come through `use super::*` (both are now module-level imports for
    // `enforce_checks_over_new_row`), so they are not re-imported here.

    // `Value` derives only Debug + Clone (no PartialEq), so NULL-ness is checked via
    // `is_null()` and the error is matched by its message string.

    /// A bare column with the given name and `not_null` flag (other fields defaulted).
    fn col(name: &str, not_null: bool) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: None,
            not_null,
            primary_key: false,
            unique: false,
            collation: None,
            default: None,
            default_value: None,
            generated: None,
        }
    }

    /// A minimal `TableDef` over the given columns and optional rowid alias.
    fn table(columns: Vec<ColumnDef>, rowid_alias: Option<usize>) -> TableDef {
        TableDef {
            name: "t".to_string(),
            columns,
            root_page: 2,
            without_rowid: false,
            rowid_alias,
            auto_indexes: Vec::new(),
            // Catalog-side CHECK predicates: irrelevant to enforce_not_null (which reads
            // the per-column not_null flags), and enforce_checks is driven from the plan
            // node's bound `checks`, not this catalog field — so an empty list here.
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            // Not an AUTOINCREMENT table: these constraint-helper tests never seed rowids.
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        }
    }

    #[test]
    fn proceeds_when_no_column_is_null() {
        let def = table(vec![col("a", true), col("b", false)], None);
        let logical = [Value::Integer(1), Value::Null]; // b is nullable, so its NULL is fine
        assert!(matches!(
            enforce_not_null(&def, &logical, &OnConflict::Abort).unwrap(),
            ConstraintOutcome::Proceed
        ));
    }

    #[test]
    fn abort_on_null_in_not_null_column_reports_that_column() {
        // Column b (index 1) is NOT NULL and NULL; a (index 0) is fine. The error names
        // the offending column, not the first column.
        let def = table(vec![col("a", false), col("b", true)], None);
        let logical = [Value::Integer(1), Value::Null];
        let err = enforce_not_null(&def, &logical, &OnConflict::Abort).unwrap_err();
        match err {
            Error::Constraint(m) => {
                assert!(m.starts_with("NOT NULL constraint failed"), "kind, got {m:?}");
                assert!(m.contains("t.b"), "names the offending column b, got {m:?}");
            }
            other => panic!("expected Constraint, got {other:?}"),
        }
    }

    #[test]
    fn ignore_yields_skip() {
        let def = table(vec![col("a", true)], None);
        let logical = [Value::Null];
        assert!(matches!(
            enforce_not_null(&def, &logical, &OnConflict::Ignore).unwrap(),
            ConstraintOutcome::Skip
        ));
    }

    #[test]
    fn replace_falls_back_to_error_no_default() {
        // No DEFAULT threaded, so REPLACE fails closed (correct sqlite ABORT fallback).
        let def = table(vec![col("a", true)], None);
        let logical = [Value::Null];
        assert!(matches!(
            enforce_not_null(&def, &logical, &OnConflict::Replace),
            Err(Error::Constraint(_))
        ));
    }

    #[test]
    fn rowid_alias_column_is_exempt_even_when_declared_not_null() {
        // Column a is the rowid alias AND declared NOT NULL. A NULL in its slot means
        // "auto-assign a rowid", so it must NOT trip the check.
        let def = table(vec![col("a", true), col("b", false)], Some(0));
        let logical = [Value::Null, Value::Integer(9)];
        assert!(matches!(
            enforce_not_null(&def, &logical, &OnConflict::Abort).unwrap(),
            ConstraintOutcome::Proceed
        ));
    }

    #[test]
    fn first_violating_column_wins() {
        // Both a and b are NOT NULL and NULL; the lowest index (a) is reported.
        let def = table(vec![col("a", true), col("b", true)], None);
        let logical = [Value::Null, Value::Null];
        let err = enforce_not_null(&def, &logical, &OnConflict::Abort).unwrap_err();
        match err {
            Error::Constraint(m) => assert!(m.contains("t.a"), "first column a wins, got {m:?}"),
            other => panic!("expected Constraint, got {other:?}"),
        }
    }

    #[test]
    fn only_the_alias_column_is_exempt_not_the_whole_table() {
        // The critical distinction: the exemption is per-column (`rowid_alias == Some(j)`),
        // NOT per-table. Here col 0 IS the rowid alias and col 1 is a SEPARATE NOT NULL
        // column; a NULL in col 1 must still error (naming col 1), even though the table
        // has an alias. This kills the mutation that widens the exemption to
        // `rowid_alias.is_some()` (which would disable NOT NULL for every column whenever
        // the table has an INTEGER PRIMARY KEY) — the `CREATE TABLE t(id INTEGER PRIMARY
        // KEY, name TEXT NOT NULL)` shape is extremely common.
        let def = table(vec![col("a", true), col("b", true)], Some(0));
        let logical = [Value::Null, Value::Null];
        let err = enforce_not_null(&def, &logical, &OnConflict::Abort).unwrap_err();
        match err {
            Error::Constraint(m) => {
                assert!(m.contains("t.b"), "the non-alias NOT NULL column b is enforced, got {m:?}");
                assert!(!m.contains("t.a"), "the alias column a is exempt, got {m:?}");
            }
            other => panic!("expected Constraint, got {other:?}"),
        }
    }

    // ---- CHECK constraint enforcement (lang_createtable.html §3.7) -----------
    //
    // `check_violated` is the pure §3.7 decision (result cast to NUMERIC is zero); it
    // is exercised exhaustively here without an evaluator. `enforce_checks` (below) is
    // then driven through a real `EvalCtx` with literal check exprs to pin the loop,
    // the `Err(Check)`/`Ok` wrapping, the detail, and the short-circuit.

    #[test]
    fn check_violated_integer_zero_is_violation() {
        assert!(check_violated(&Value::Integer(0)));
    }

    #[test]
    fn check_violated_real_zero_is_violation() {
        assert!(check_violated(&Value::Real(0.0)));
        // Negative zero is still zero.
        assert!(check_violated(&Value::Real(-0.0)));
    }

    #[test]
    fn check_violated_null_passes() {
        // §3.7: "If the CHECK expression evaluates to NULL ... it is not a constraint
        // violation." NULL coerces to `truth == None`, so it is not a violation.
        assert!(!check_violated(&Value::Null));
    }

    #[test]
    fn check_violated_nonzero_numbers_pass() {
        assert!(!check_violated(&Value::Integer(1)));
        // "any other non-zero value" — a NEGATIVE value passes (only zero fails).
        assert!(!check_violated(&Value::Integer(-3)));
        assert!(!check_violated(&Value::Real(0.5)));
        assert!(!check_violated(&Value::Real(-2.5)));
    }

    #[test]
    fn check_violated_text_casting_to_zero_is_violation() {
        // §3.7 casts to NUMERIC "in the same way as a CAST expression": '0'/'0.0' cast
        // to 0 -> violation.
        assert!(check_violated(&Value::Text("0".into())));
        assert!(check_violated(&Value::Text("0.0".into())));
        // A leading-zero prefix and NON-NUMERIC text also CAST to 0 -> violation:
        // 'abc'/'0abc'/'' all become numeric 0. This is the CAST semantics §3.7
        // mandates, and the DELIBERATE difference from a lossless-affinity reading
        // (which would leave these as TEXT and wrongly pass) — matching real sqlite.
        assert!(check_violated(&Value::Text("0abc".into())));
        assert!(check_violated(&Value::Text("abc".into())));
        assert!(check_violated(&Value::Text("".into())));
    }

    #[test]
    fn check_violated_text_casting_to_nonzero_passes() {
        assert!(!check_violated(&Value::Text("1".into())));
        assert!(!check_violated(&Value::Text("2.5".into())));
        // Leading numeric prefix drives the value: '1english' -> 1 -> non-zero -> pass.
        assert!(!check_violated(&Value::Text("1english".into())));
    }

    #[test]
    fn check_violated_blob_casting_to_zero_is_violation() {
        // A BLOB is read as its bytes-as-text then cast to numeric; a zero/junk blob
        // casts to 0 -> violation. Affinity coercion never touches a BLOB, so it would
        // wrongly pass these — another reason `check_violated` uses the CAST-style
        // `truth`, not affinity.
        assert!(check_violated(&Value::Blob(vec![0x00])));
        assert!(check_violated(&Value::Blob(b"abc".to_vec())));
        // A blob whose bytes read as a non-zero number passes.
        assert!(!check_violated(&Value::Blob(b"7".to_vec())));
    }

    /// Run `enforce_checks` over a real (literal-driven) [`EvalCtx`] under `on_conflict`.
    /// The check exprs are [`EvalExpr::Literal`]s, so `eval` returns them without touching
    /// the catalog/pager/plan — a minimal in-memory [`Env`] suffices — which isolates the
    /// loop + the outcome/error resolution from the (separately proven) evaluator.
    /// `logical` is empty because a literal reads no register.
    fn run_checks(checks: &[CheckConstraint], on_conflict: &OnConflict) -> Result<ConstraintOutcome> {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        let cat = SchemaCatalog::new();
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let mut rt = Runtime::new();
        let logical: [Value; 0] = [];
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: minisqlite_types::DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let mut ctx = EvalCtx { rt: &mut rt, env, outer: &logical };
        enforce_checks(checks, &logical, on_conflict, &mut ctx)
    }

    /// A `CHECK` whose expression is the literal `v`, tagged with `detail`.
    fn lit_check(v: Value, detail: &str) -> CheckConstraint {
        CheckConstraint { expr: EvalExpr::Literal(v), detail: detail.to_string() }
    }

    /// Assert the checks PASS under the default ABORT with the `Proceed` outcome (not
    /// merely `Ok`, so a stray `Skip` would be caught).
    fn assert_proceeds(checks: &[CheckConstraint]) {
        assert!(matches!(
            run_checks(checks, &OnConflict::Abort),
            Ok(ConstraintOutcome::Proceed)
        ));
    }

    #[test]
    fn enforce_checks_empty_slice_proceeds() {
        assert_proceeds(&[]);
    }

    #[test]
    fn enforce_checks_zero_result_errors_with_check_kind_and_detail() {
        // Under the default ABORT, a zero result is the CHECK constraint error.
        match run_checks(&[lit_check(Value::Integer(0), "t.ck")], &OnConflict::Abort) {
            Err(Error::Constraint(m)) => {
                assert!(m.starts_with("CHECK constraint failed"), "CHECK kind, got {m:?}");
                assert!(m.contains("t.ck"), "carries the plan's detail, got {m:?}");
            }
            other => panic!("expected a CHECK Constraint error, got {other:?}"),
        }
    }

    #[test]
    fn enforce_checks_real_zero_result_errors() {
        assert!(matches!(
            run_checks(&[lit_check(Value::Real(0.0), "c")], &OnConflict::Abort),
            Err(Error::Constraint(_))
        ));
    }

    #[test]
    fn enforce_checks_null_result_passes() {
        // §3.7: a NULL check result is NOT a violation, so the row proceeds.
        assert_proceeds(&[lit_check(Value::Null, "c")]);
    }

    #[test]
    fn enforce_checks_nonzero_result_passes() {
        assert_proceeds(&[lit_check(Value::Integer(1), "c")]);
        assert_proceeds(&[lit_check(Value::Integer(-3), "c")]);
    }

    #[test]
    fn enforce_checks_short_circuits_at_the_first_violation() {
        // (a) A leading PASS does not mask a later violation: [1, 0] fails, naming the
        // failing (second) check, not the passing one.
        match run_checks(
            &[lit_check(Value::Integer(1), "ok"), lit_check(Value::Integer(0), "bad")],
            &OnConflict::Abort,
        ) {
            Err(Error::Constraint(m)) => {
                assert!(m.contains("bad") && !m.contains("ok"), "the failing check is named, got {m:?}");
            }
            other => panic!("expected the second check to fail, got {other:?}"),
        }
        // (b) The loop STOPS at the FIRST violation: [0("first"), 0("second")] reports
        // "first" and never reaches "second" — this distinguishes true short-circuit from
        // "evaluate all, report last" (which a single-violation case cannot).
        match run_checks(
            &[lit_check(Value::Integer(0), "first"), lit_check(Value::Integer(0), "second")],
            &OnConflict::Abort,
        ) {
            Err(Error::Constraint(m)) => {
                assert!(
                    m.contains("first") && !m.contains("second"),
                    "the FIRST violation short-circuits, got {m:?}"
                );
            }
            other => panic!("expected the first check to fail, got {other:?}"),
        }
    }

    #[test]
    fn enforce_checks_all_passing_proceeds() {
        assert_proceeds(&[
            lit_check(Value::Integer(1), "a"),
            lit_check(Value::Null, "b"),
            lit_check(Value::Text("5".into()), "c"),
        ]);
    }

    // ---- ON CONFLICT resolution (lang_conflict.html: a statement OR clause overrides
    // the constraint-definition default; CHECK participates like NOT NULL/UNIQUE) --------

    #[test]
    fn enforce_checks_or_ignore_skips_the_violating_row() {
        // OR IGNORE resolves a CHECK violation like enforce_not_null resolves an IGNORE
        // NULL: SKIP the one offending row with NO error (the caller `continue`s, prior
        // rows stand). This is the fix for `INSERT OR IGNORE` / `UPDATE OR IGNORE` past a
        // CHECK-violating row, which previously aborted the whole statement.
        assert!(matches!(
            run_checks(&[lit_check(Value::Integer(0), "c")], &OnConflict::Ignore),
            Ok(ConstraintOutcome::Skip)
        ));
        // A real-zero result skips under IGNORE too (same §3.7 violation rule).
        assert!(matches!(
            run_checks(&[lit_check(Value::Real(0.0), "c")], &OnConflict::Ignore),
            Ok(ConstraintOutcome::Skip)
        ));
    }

    #[test]
    fn enforce_checks_or_ignore_non_violating_row_still_proceeds() {
        // IGNORE only skips a VIOLATION; a passing row proceeds normally (never Skip).
        assert!(matches!(
            run_checks(&[lit_check(Value::Integer(1), "c")], &OnConflict::Ignore),
            Ok(ConstraintOutcome::Proceed)
        ));
    }

    #[test]
    fn enforce_checks_replace_fail_rollback_error_like_abort() {
        // REPLACE folds to ABORT for a CHECK (lang_conflict.html), and FAIL/ROLLBACK raise
        // here too (their prior-row atomicity is the txn wrapper's concern). Only IGNORE
        // skips — every other policy is the constraint error on a zero result.
        for oc in [OnConflict::Replace, OnConflict::Fail, OnConflict::Rollback, OnConflict::Abort] {
            assert!(
                matches!(run_checks(&[lit_check(Value::Integer(0), "c")], &oc), Err(Error::Constraint(_))),
                "policy {oc:?} must raise a CHECK error on a zero result"
            );
        }
    }
}
