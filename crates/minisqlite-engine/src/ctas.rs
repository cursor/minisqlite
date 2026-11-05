//! `CREATE TABLE ... AS SELECT` (CTAS) orchestration (`lang_createtable.html` §2.1).
//!
//! The catalog builds only a plain `Columns` table; the SELECT-populated form is the
//! engine's job because it spans three seams (plan the SELECT for its schema, register a
//! plain table, run an `INSERT ... SELECT` to fill it). [`crate::dispatch`] routes a
//! `CreateTableBody::AsSelect` here BEFORE it reaches the catalog's `AsSelect` reject.
//!
//! The whole thing is CREATE + populate as one atomic unit (§2.1): in autocommit it runs
//! in one implicit transaction that is rolled back — table and all — on any failure
//! (including a SELECT error), so a failed CTAS leaves NO table. `IF NOT EXISTS` on an
//! existing table short-circuits FIRST, before the SELECT runs, so the existing table is
//! untouched and its (unrelated) SELECT never executes.

use minisqlite_catalog::{Catalog, MultiCatalog};
use minisqlite_plan::{ctas_columns, CtasColumn, Planner};
use minisqlite_sql::{
    ColumnDef, CreateTable, CreateTableBody, Insert, InsertSource, Select, Statement, TableOptions,
};
use minisqlite_types::{DbIndex, Error, QueryResult, Result};

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Execute `CREATE TABLE [IF NOT EXISTS] <name> AS <sel>` (§2.1). Returns `Ok(None)`
    /// (CTAS produces no result set), or the error of a failed create/populate.
    ///
    /// `source` (the verbatim `CREATE TABLE ... AS SELECT ...` text) is deliberately
    /// unused: real SQLite does NOT store it — it stores a synthesized plain-columns
    /// `CREATE TABLE name(col type, ...)` as the schema `sql` (built by
    /// [`render_ctas_ddl`] below), so a reopen re-parses the resolved shape, not the
    /// SELECT. Kept in the signature to match the DDL dispatch convention.
    pub(crate) fn exec_ctas(
        &mut self,
        ct: &CreateTable,
        sel: &Select,
        _source: &str,
    ) -> Result<Option<QueryResult>> {
        // Route to the target namespace: `CREATE TEMP TABLE … AS SELECT` (or a `temp.`
        // qualifier) builds the table in the temp store (materialized here if absent), so
        // the existence check, the transaction bracket, and the create all target that
        // store — otherwise the copy would silently land in `main` (the very temp-flag bug
        // fixed here). Non-temp unqualified → main (index 0), unchanged.
        let db = self.create_target_db(ct.temp, &ct.name)?;
        let di = db.index();

        // Existence check FIRST — before the SELECT runs. `IF NOT EXISTS` on an existing
        // table is a no-op (the SELECT must NOT execute and the table must be left
        // untouched); without it, the standard "table already exists" error. Only a real
        // TABLE collision is decided here; a name held by an index/view/trigger falls
        // through to `create_table`, which reports the correct cross-type error. Checked in
        // the TARGET namespace: a temp CTAS collides only with a temp table, so it can
        // shadow a same-named main one.
        if self.catalogs[di].table(&ct.name.name)?.is_some() {
            if ct.if_not_exists {
                return Ok(None);
            }
            return Err(Error::sql(format!("table {} already exists", ct.name.name)));
        }

        // Derive the new table's columns from the SELECT (names + declared types). A bad
        // SELECT surfaces here, before anything is created — keeping CREATE atomic. The
        // SELECT may read any live namespace, so plan it against a MultiCatalog (search
        // order: temp, main, attached); with only `main` live this is the main catalog.
        let cols = {
            let mc = MultiCatalog::new(&self.catalogs, &self.namespaces);
            ctas_columns(sel, &mc)?
        };

        // Synthesize the plain table: §2.1 gives it NO primary key, NO constraints, and a
        // NULL default for every column — so each column is a bare `name [type]`.
        let synth_columns: Vec<ColumnDef> = cols
            .iter()
            .map(|col| ColumnDef {
                name: col.name.clone(),
                type_name: col.decl_type.clone(),
                constraints: Vec::new(),
            })
            .collect();
        let synth_ct = CreateTable {
            temp: ct.temp,
            if_not_exists: ct.if_not_exists,
            name: ct.name.clone(),
            body: CreateTableBody::Columns {
                columns: synth_columns,
                constraints: Vec::new(),
                options: TableOptions::default(),
            },
        };
        // The schema `sql` stored for the plain table: a re-parseable plain-columns DDL,
        // NOT the original `AS SELECT` text (mirroring real sqlite's `sqlite_master.sql`).
        let synth_source = render_ctas_ddl(&ct.name.name, &cols);

        // Populate the copy with the SELECT's rows via `INSERT INTO <name> <sel>`. The
        // clone reuses the whole SELECT verbatim, so projection / WHERE / GROUP BY / JOIN
        // / UNION / ORDER BY all copy in exactly their result set, in result order (so an
        // ORDER BY in the source fixes the contiguous 1..N rowids §2.1 assigns).
        let insert_stmt = Statement::Insert(Box::new(Insert {
            with: None,
            or_conflict: None,
            table: ct.name.clone(),
            alias: None,
            columns: None,
            source: InsertSource::Select(Box::new(sel.clone())),
            upsert: Vec::new(),
            returning: Vec::new(),
        }));

        // Atomic CREATE + populate. In autocommit, wrap in one implicit transaction so a
        // failure (create or SELECT) rolls back the whole unit — leaving no table; inside
        // an open transaction, join it (its owner controls commit/rollback), matching how
        // `run_mutating_collect` / `with_write_txn` treat DML and DDL.
        let wrap = !self.txn_active();
        if wrap {
            self.pagers[di].begin()?;
            // The implicit transaction ends below either way, which switches `PRAGMA
            // defer_foreign_keys` off (as at COMMIT / ROLLBACK).
            self.rt.reset_defer_foreign_keys();
        }
        match self.ctas_create_and_populate(db, &synth_ct, &synth_source, &insert_stmt) {
            Ok(()) => {
                if wrap {
                    self.pagers[di].commit()?;
                }
                Ok(None)
            }
            Err(e) => {
                if wrap {
                    // Revert the create + any partial rows, then resync the schema cache
                    // to the rolled-back page 1 so the table `create_table` cached before
                    // the failure does not survive (the same fix `with_write_txn` applies
                    // on its implicit-transaction error path). The rollback/reload results
                    // are ignored so a storage-level failure cannot mask the real error.
                    let _ = self.pagers[di].rollback();
                    let _ = self.catalogs[di].load(&*self.pagers[di]);
                }
                Err(e)
            }
        }
    }

    /// The fallible CREATE + populate body, run inside the caller's transaction. Register
    /// the plain table (caching its def so the planner can resolve the INSERT target),
    /// then plan and run the `INSERT ... SELECT`. Each step's field borrows
    /// (`catalog`/`pager`, then `planner`/`catalog`, then `rt`, then the whole executor)
    /// are sequential, so this stays one `&mut self` method rather than a closure.
    fn ctas_create_and_populate(
        &mut self,
        db: DbIndex,
        synth_ct: &CreateTable,
        synth_source: &str,
        insert_stmt: &Statement,
    ) -> Result<()> {
        let di = db.index();
        self.catalogs[di].create_table(&mut *self.pagers[di], synth_ct, synth_source)?;
        // Plan the INSERT against all live namespaces; the just-created target table lives
        // in namespace `db`, and the SELECT source may read any namespace (search order).
        // The synth INSERT's target name resolves in that same search order, so its DML
        // node's `db` matches `db` and the write routes to this store. The MultiCatalog
        // borrow ends before `execute_plan_collect` reborrows `self`.
        let plan = {
            let mc = MultiCatalog::new(&self.catalogs, &self.namespaces);
            self.planner.plan(insert_stmt, &mc)?
        };
        // The INSERT is a mutating statement, so reset the per-statement change counter
        // just as `run_plannable` does before a mutating plan (the DML operator then
        // advances it); CTAS itself reports no rows.
        self.rt.reset_statement_changes();
        let _ = self.execute_plan_collect(&plan)?;
        Ok(())
    }
}

/// Render the re-parseable schema `sql` for a CTAS-created plain table:
/// `CREATE TABLE "name"("c1" T1, "c2", ...)`. Identifiers are double-quoted (and any
/// embedded `"` doubled) so a name that is a keyword or holds punctuation still parses;
/// a column with no declared type ("" / `None`) is rendered as a bare quoted name.
fn render_ctas_ddl(name: &str, cols: &[CtasColumn]) -> String {
    let mut sql = String::from("CREATE TABLE ");
    push_quoted_ident(&mut sql, name);
    sql.push('(');
    for (i, col) in cols.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        push_quoted_ident(&mut sql, &col.name);
        if let Some(decl_type) = &col.decl_type {
            sql.push(' ');
            sql.push_str(decl_type);
        }
    }
    sql.push(')');
    sql
}

/// Append `ident` to `out` as a double-quoted SQL identifier, doubling any embedded `"`
/// so the tokenizer decodes it back to the exact original (the standard SQL quoting rule
/// the lexer's `quoted_ident` reverses).
fn push_quoted_ident(out: &mut String, ident: &str) {
    out.push('"');
    for ch in ident.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_types::Value;

    fn col(name: &str, decl_type: Option<&str>) -> CtasColumn {
        CtasColumn { name: name.to_string(), decl_type: decl_type.map(str::to_string) }
    }

    /// The synthesized schema `sql` is a plain-columns `CREATE TABLE` with double-quoted
    /// identifiers and the declared type (omitted for a "" / `None`-typed column) — and
    /// it re-parses through the real parser back to a `Columns` body with the same column
    /// names and declared types, so a reopened database recovers the identical schema.
    #[test]
    fn render_ctas_ddl_round_trips_through_the_parser() {
        let cols = [col("i", Some("INT")), col("t", Some("TEXT")), col("x", None)];
        let sql = render_ctas_ddl("t2", &cols);
        assert_eq!(sql, r#"CREATE TABLE "t2"("i" INT, "t" TEXT, "x")"#);

        let ast = minisqlite_sql::parse(&sql).expect("synthesized CTAS DDL must re-parse");
        let Statement::CreateTable(ct) = &ast.statements[0] else {
            panic!("expected a CREATE TABLE, got {:?}", ast.statements[0]);
        };
        assert_eq!(ct.name.name, "t2");
        let CreateTableBody::Columns { columns, constraints, options } = &ct.body else {
            panic!("expected a plain Columns body, got AS SELECT");
        };
        assert!(constraints.is_empty(), "CTAS table carries no constraints (§2.1)");
        assert_eq!(*options, TableOptions::default());
        let got: Vec<(&str, Option<&str>)> =
            columns.iter().map(|c| (c.name.as_str(), c.type_name.as_deref())).collect();
        assert_eq!(got, vec![("i", Some("INT")), ("t", Some("TEXT")), ("x", None)]);
    }

    /// An identifier that is a keyword or holds punctuation (a `SELECT expr AS "odd name"`
    /// alias) still round-trips: quoting + doubling any embedded `"` keeps the stored DDL
    /// parseable and the recovered column name byte-identical.
    #[test]
    fn render_ctas_ddl_quotes_awkward_identifiers() {
        let cols = [col("select", Some("NUM")), col(r#"a"b"#, None)];
        let sql = render_ctas_ddl("from", &cols);
        assert_eq!(sql, r#"CREATE TABLE "from"("select" NUM, "a""b")"#);

        let ast = minisqlite_sql::parse(&sql).expect("awkward-identifier DDL must re-parse");
        let Statement::CreateTable(ct) = &ast.statements[0] else { panic!("expected CREATE TABLE") };
        assert_eq!(ct.name.name, "from");
        let CreateTableBody::Columns { columns, .. } = &ct.body else { panic!("expected Columns") };
        let names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["select", r#"a"b"#]);
    }

    /// Atomicity — pre-create ordering. A CTAS whose SELECT references a missing table
    /// errors, and NO table is left behind. `exec_ctas` compiles the SELECT
    /// (`ctas_columns`) BEFORE `create_table`, so a bad SELECT fails before anything is
    /// created. This pins that ordering: a future reorder that created the table first
    /// would leak `t2`, which the two "absent afterward" checks below would then catch.
    #[test]
    fn ctas_invalid_select_errors_and_leaves_no_table() {
        let mut e = SqlEngine::open_in_memory().expect("open in-memory db");
        assert!(
            e.run_program("CREATE TABLE t2 AS SELECT * FROM no_such_table").is_err(),
            "CTAS over a missing source table must error",
        );
        assert!(
            e.catalogs[0].table("t2").unwrap().is_none(),
            "a failed CTAS must register no table in the catalog",
        );
        assert!(
            e.run_program("SELECT * FROM t2").is_err(),
            "the never-created table must be absent (no such table: t2)",
        );
    }

    /// `CREATE TABLE <existing> AS SELECT ...` WITHOUT `IF NOT EXISTS` raises the standard
    /// "table already exists" error (the non-IF-NOT-EXISTS branch of the existence check),
    /// and leaves the existing table untouched — its SELECT never runs.
    #[test]
    fn ctas_duplicate_without_if_not_exists_errors_and_preserves_table() {
        let mut e = SqlEngine::open_in_memory().expect("open in-memory db");
        e.run_program("CREATE TABLE t2(a)").unwrap();
        e.run_program("INSERT INTO t2 VALUES (1)").unwrap();

        let result = e.run_program("CREATE TABLE t2 AS SELECT 99 AS a");
        match &result {
            Err(Error::Sql(m)) => {
                assert!(m.contains("already exists"), "expected 'already exists', got: {m}")
            }
            Err(other) => panic!("expected an Sql 'already exists' error, got: {other}"),
            Ok(_) => panic!("expected the duplicate CTAS to error, but it succeeded"),
        }
        // The pre-existing table is intact: still exactly its one original row holding `1`,
        // proving the SELECT's `99` was never written (the SELECT must not run on collision).
        let q = e.run_program("SELECT a FROM t2").unwrap().expect("SELECT yields a result set");
        assert_eq!(q.rows.len(), 1, "exactly the one original row must survive");
        assert!(
            matches!(q.rows[0].as_slice(), [Value::Integer(1)]),
            "the existing table must keep its original row (1), not the SELECT's 99: got {:?}",
            q.rows[0],
        );
    }

    /// Atomicity — POST-create rollback (the subtle branch). This is a genuine black-box
    /// post-`create_table` failure: `zeroblob(N)` for `N` over the blob-length cap
    /// (`MAX_BLOB_LEN` = 1e9) COMPILES cleanly but raises "string or blob too big" at
    /// RUNTIME. Feeding it a column (`zeroblob(a)`, a = 9e9) forces evaluation during the
    /// `INSERT ... SELECT` population — after `create_table` has already cached the table
    /// def and written page 1 — so the failure lands in `exec_ctas`'s implicit-txn error
    /// path, not before it.
    ///
    /// The `e.catalogs[0].table("t2").is_none()` check is the load-bearing one: it pins the
    /// rollback AND the `catalog.load` resync (rolling back page 1 alone would leave the
    /// phantom def cached). Replacing the error arm with a bare `Err(e) => Err(e)` (skipping
    /// the rollback + resync) leaves `t2` cached, turning exactly this assertion red.
    #[test]
    fn ctas_runtime_error_after_create_rolls_back_the_table() {
        let mut e = SqlEngine::open_in_memory().expect("open in-memory db");
        e.run_program("CREATE TABLE t1(a)").unwrap();
        e.run_program("INSERT INTO t1 VALUES (9000000000)").unwrap();

        assert!(
            e.run_program("CREATE TABLE t2 AS SELECT zeroblob(a) AS b FROM t1").is_err(),
            "a CTAS whose SELECT errors at runtime must fail",
        );
        assert!(
            e.catalogs[0].table("t2").unwrap().is_none(),
            "a post-create CTAS failure must roll back + resync the catalog, leaving no def",
        );
        assert!(
            e.run_program("SELECT * FROM t2").is_err(),
            "the rolled-back table must be absent (no such table: t2)",
        );
    }
}
