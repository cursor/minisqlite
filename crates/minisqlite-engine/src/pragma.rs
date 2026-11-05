//! `PRAGMA` handling at the engine level. A few families live here:
//!
//! * The page-1 header-field pragmas the engine reads/writes directly —
//!   `user_version` (off 60), `application_id` (off 68), `schema_version`
//!   (off 40, the schema cookie), `default_cache_size` (off 48), and `page_size`
//!   (off 16). All but `page_size` are a direct read/re-encode of one 32-bit field;
//!   `page_size` rebuilds the pager so the file is physically laid out in the new
//!   size (the header off-16 must match the pager's real page size). `encoding`
//!   (off 56) reports its true state on GET and, on SET while the database is still
//!   empty, actually changes the header — the record codec is encoding-aware, so a
//!   UTF-16 database's schema/data/index bytes are written in that encoding and real
//!   sqlite reads them back (the encoding is fixed at creation, so SET is a no-op once
//!   any object exists). `auto_vacuum` (off 52/64) reports its true state on GET and, on
//!   SET while the database is still empty, records the vacuum mode: the pager then lays
//!   out real pointer-map pages (§1.8) and finalizes off 52 at each commit, so the file
//!   real sqlite reads back is a valid auto_vacuum database. Enabling/disabling is
//!   honored only before any table exists (pragma.html); a full<->incremental switch is
//!   allowed anytime (see [`SqlEngine::set_auto_vacuum_of`]).
//!
//!   These header fields live in EACH database file's page-1 header, so every one of
//!   these pragmas honors the schema qualifier (`PRAGMA aux.user_version`,
//!   `PRAGMA temp.page_count`): the target file is resolved by [`header_pragma_db`]
//!   (`SqlEngine::header_pragma_db`) and read/written through the `_of(db)` helpers.
//!   Per pragma.html, "if the optional schema name is omitted, main is assumed", so an
//!   UNQUALIFIED header pragma targets `main` (NOT the temp→main→attached search order
//!   the OBJECT pragmas below use — those resolve an object NAME; a header pragma has no
//!   object argument). An unknown/unattached qualifier resolves to no database and
//!   yields the empty result (columns, zero rows), the same "no such database"
//!   convention the introspection pragmas use — never a silent read of `main`.
//! * The schema-introspection pragmas (`table_info`, `table_xinfo`, `index_list`,
//!   `index_info`, `index_xinfo`, `foreign_key_list`) — pure reads over the
//!   [`Catalog`] seam. They resolve their target namespace (`main`/`temp`/attached)
//!   from the pragma's schema qualifier and the argument name via a per-statement
//!   [`MultiCatalog`], then build the rows with the shared
//!   [`minisqlite_catalog::pragma_rows`] — the SAME builder the `pragma_*` table-valued
//!   functions use, so the statement and TVF forms cannot diverge (one home for the
//!   resolution rule, the row shape, and the empty-when-absent behavior). They never
//!   touch the planner/executor and never scan data pages: each is O(columns) /
//!   O(indexes) over the cached schema (see `spec/sqlite-doc/pragma.html`).
//! * WAL control — `journal_mode` reads or switches the page-1 file-format versions
//!   (off 18/19, the field [`minisqlite_pager::DiskPager`] reads at open to choose the WAL
//!   vs rollback backing), and `wal_checkpoint` drives a checkpoint through the
//!   [`Pager`](minisqlite_pager::Pager) seam. Both always return one row, as SQLite does,
//!   and both stay MAIN-ONLY here (a schema qualifier is ignored). This is a deliberate
//!   SCOPE boundary, not a claim the bytes aren't per-database: the file-format version
//!   bytes ARE a per-file header field, but per-database `journal_mode`/checkpoint needs an
//!   in-connection backing swap and cross-database WAL wiring that is main-only today (the
//!   two-connection WAL rung), so it is deferred. By contrast the connection-scoped
//!   settings (`foreign_keys`, `recursive_triggers`) and the whole-connection listings
//!   (`database_list`, `integrity_check`, `quick_check`) are genuinely NOT per-database
//!   header fields — connection-wide by design and rightly qualifier-blind.
//!
//! An unknown pragma is a no-op that returns no rows (as in SQLite — an unknown
//! pragma is NOT an error). The match is written to be easy to extend one pragma
//! at a time.

use minisqlite_catalog::{pragma_rows, Catalog, MultiCatalog, PragmaFunction};
use minisqlite_fileformat::{
    DatabaseHeader, HEADER_SIZE, TEXT_ENCODING_UTF16BE, TEXT_ENCODING_UTF16LE, TEXT_ENCODING_UTF8,
};
use minisqlite_pager::CheckpointMode;
use minisqlite_sql::{Literal, PragmaArg, PragmaValue, QualifiedName};
use minisqlite_types::{DbIndex, Error, QueryResult, Result, Value};
use std::path::Path;

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Run a `PRAGMA`. Returns a result set for a get form that produces rows
    /// (`PRAGMA user_version`, the schema-introspection pragmas), `None` for a set
    /// form or an unknown pragma.
    ///
    /// The pragma name is matched case-insensitively (SQLite folds identifiers over
    /// ASCII). The schema qualifier (`name.schema`) is honored by BOTH the
    /// schema-introspection pragmas (which route to the named namespace by object name —
    /// temp→main→attached search order; see the six `pragma_*` handlers) AND the
    /// per-database HEADER pragmas (`user_version`, `page_size`, `encoding`, …), which
    /// read/write the resolved database's own page-1 header via [`header_pragma_db`]
    /// (`SqlEngine::header_pragma_db`) and the `_of(db)` helpers. Per pragma.html an
    /// UNQUALIFIED header pragma targets `main` ("if the optional schema name is omitted,
    /// main is assumed"), and an unknown/unattached qualifier yields the empty result.
    pub(crate) fn run_pragma(
        &mut self,
        name: &QualifiedName,
        arg: Option<&PragmaArg>,
    ) -> Result<Option<QueryResult>> {
        // The schema qualifier the header pragmas resolve their target database from.
        // `None` (unqualified) means `main` per pragma.html; a qualifier names a
        // database (`main`/`temp`/an ATTACH alias) or, if unknown, no database at all.
        let schema = name.schema.as_deref();
        match name.name.to_ascii_lowercase().as_str() {
            "user_version" => match arg {
                // Get: one row, one column named `user_version` (empty when the qualifier
                // names no database). The value is the resolved db's header field.
                None => self.header_int_get("user_version", schema, |h| h.user_version as i32 as i64),
                // Set: store the value in the resolved db's page-1 header; returns no rows.
                Some(a) => {
                    let v = pragma_arg_to_i64(a)?;
                    self.header_int_set(schema, |h| h.user_version = (v as i32) as u32)?;
                    Ok(None)
                }
            },

            // `application_id` (fileformat2 §1.3.15, off 68) + pragma.html
            // "application_id": a 32-bit *signed* integer an application stamps into
            // page 1 to mark the file as its own. Read and written exactly like
            // `user_version` (off 60), differing only in which header field it targets.
            "application_id" => match arg {
                None => {
                    self.header_int_get("application_id", schema, |h| h.application_id as i32 as i64)
                }
                Some(a) => {
                    let v = pragma_arg_to_i64(a)?;
                    self.header_int_set(schema, |h| h.application_id = (v as i32) as u32)?;
                    Ok(None)
                }
            },

            // `schema_version` (pragma.html "schema_version") force-writes the schema
            // cookie (fileformat2 §1.3.9, off 40) — an EXPERT pragma whose defined
            // effect is to set that header field directly (normal schema changes bump
            // the cookie themselves; this overrides it). 32-bit signed, like
            // `user_version`. Only the header byte is observable here.
            "schema_version" => match arg {
                None => {
                    self.header_int_get("schema_version", schema, |h| h.schema_cookie as i32 as i64)
                }
                Some(a) => {
                    let v = pragma_arg_to_i64(a)?;
                    self.header_int_set(schema, |h| h.schema_cookie = (v as i32) as u32)?;
                    Ok(None)
                }
            },

            // `default_cache_size` (pragma.html "default_cache_size", fileformat2
            // §1.3.11, off 48): the suggested cache size in pages, stored as a signed
            // integer (a positive N is written verbatim). DEPRECATED in SQLite, but its
            // header-byte effect is well-defined and written directly, like
            // `schema_version`.
            "default_cache_size" => match arg {
                None => self
                    .header_int_get("default_cache_size", schema, |h| h.default_cache_size as i32 as i64),
                Some(a) => {
                    let v = pragma_arg_to_i64(a)?;
                    self.header_int_set(schema, |h| h.default_cache_size = (v as i32) as u32)?;
                    Ok(None)
                }
            },

            // `page_size` (pragma.html "page_size", fileformat2 §1.3.2, off 16). GET
            // reports the current size (the codec decodes the on-disk sentinel 1 back
            // to 65536, so 65536 reports as 65536). SET rebuilds the RESOLVED db's pager
            // at the new size — but only while that database is still empty and N is a
            // power of two in [512, 65536]; otherwise it is a silent no-op that leaves
            // the size unchanged (see [`SqlEngine::set_page_size_of`]). Unlike the fields
            // above, the SET path must reconfigure the pager, not patch a header byte:
            // off 16 must match the size the pager actually lays the file out in.
            "page_size" => match arg {
                None => self.header_int_get("page_size", schema, |h| i64::from(h.page_size)),
                Some(a) => {
                    let v = pragma_arg_to_i64(a)?;
                    // An unknown qualifier resolves to no database → no-op (never resize
                    // `main` for a `bogus.page_size = N`).
                    if let Some(db) = self.header_pragma_db(schema) {
                        self.set_page_size_of(db, v)?;
                    }
                    Ok(None)
                }
            },

            // `encoding` (fileformat2 §1.3.13, off 56). GET honestly reports the
            // header's current text encoding (1 UTF-8 / 2 UTF-16le / 3 UTF-16be); a
            // file real sqlite wrote in UTF-16 decodes to UTF-16le/be here. SET records
            // the encoding the resolved database is CREATED with — "it is not possible to
            // change the text encoding of a database after it has been created"
            // (pragma.html "encoding"). The engine's record codec is encoding-aware
            // (`serial`/`decode_record_*_enc`), so writing 2/3 into off 56 while the
            // database is still empty is honored end to end: the schema and every data
            // row are then stored in that encoding and real sqlite reads them back. Once
            // any object exists the file's encoding is fixed, so SET becomes a silent
            // no-op (as in sqlite), reported by `is_freshly_created`.
            "encoding" => match arg {
                None => match self.header_pragma_db(schema) {
                    None => Ok(Some(pragma_empty("encoding"))),
                    Some(db) => {
                        let label = encoding_label(self.read_page1_header_of(db)?.text_encoding);
                        Ok(Some(text_pragma_row("encoding", label)))
                    }
                },
                Some(a) => {
                    if let Some(db) = self.header_pragma_db(schema) {
                        self.set_encoding_of(db, a)?;
                    }
                    Ok(None)
                }
            },

            // `auto_vacuum` (fileformat2 §1.3.12, off 52 largest-root-btree + off 64
            // incremental-vacuum flag). GET honestly decodes the current mode from the
            // resolved db's header (0 none / 1 full / 2 incremental). SET records the mode
            // in the resolved db's header (see [`SqlEngine::set_auto_vacuum_of`]); the
            // pager then lays out real ptrmap pages (§1.8) and finalizes off 52 at each
            // commit, so the file real sqlite reads back is a valid auto_vacuum database.
            // An unknown qualifier resolves to no database → no-op.
            "auto_vacuum" => match arg {
                None => self.header_int_get("auto_vacuum", schema, auto_vacuum_mode),
                Some(a) => {
                    if let Some(db) = self.header_pragma_db(schema) {
                        self.set_auto_vacuum_of(db, a)?;
                    }
                    Ok(None)
                }
            },

            // `incremental_vacuum` (pragma.html "incremental_vacuum"): in
            // auto_vacuum=incremental mode, remove up to N pages from the freelist and
            // truncate the file by the same amount. N omitted / `< 1` / more than the
            // freelist holds all mean "clear the entire freelist". A no-op (returns no
            // rows) when the database is not in incremental mode or the freelist is empty.
            // Reclamation runs its own transaction, so it is honored only in autocommit.
            "incremental_vacuum" => {
                self.pragma_incremental_vacuum(schema, arg)?;
                Ok(None)
            }

            // `page_count` (pragma.html #pragma_page_count): the total number of pages in
            // the RESOLVED database, read live from that db's pager (one integer row, as
            // SQLite yields — NOT the empty set the catch-all would otherwise give).
            // Read-only: SQLite has no SET form, so any argument is ignored and the
            // current count is returned (never a fabricated smaller value).
            "page_count" => self.header_page_count(schema),

            // `freelist_count` (pragma.html #pragma_freelist_count): the number of unused
            // pages on the free-list, from the resolved db's page-1 header (fileformat2
            // §1.3.5, off 36). Read-only, one integer row.
            "freelist_count" => {
                self.header_int_get("freelist_count", schema, |h| i64::from(h.freelist_count))
            }

            // Schema-introspection pragmas. Each takes the pragma's schema qualifier
            // (`name.schema`, e.g. `temp` in `PRAGMA temp.table_info(t)`) and the arg
            // object name, which the `(name)` call form or the `= name` form both carry.
            // `pragma_introspect` resolves WHICH namespace to read and builds the rows via
            // the shared `minisqlite_catalog::pragma_rows` (the same code the `pragma_*`
            // table-valued functions run). The name is case-folded by the catalog on
            // lookup, so it is passed verbatim. Each always returns its fixed column set,
            // with zero rows when the object is absent (matching SQLite, whose prepared
            // statement still declares the columns but steps no rows).
            "table_info" => self.pragma_introspect(PragmaFunction::TableInfo, name, arg),
            "table_xinfo" => self.pragma_introspect(PragmaFunction::TableXInfo, name, arg),
            "index_list" => self.pragma_introspect(PragmaFunction::IndexList, name, arg),
            "index_info" => self.pragma_introspect(PragmaFunction::IndexInfo, name, arg),
            "index_xinfo" => self.pragma_introspect(PragmaFunction::IndexXInfo, name, arg),
            "foreign_key_list" => self.pragma_introspect(PragmaFunction::ForeignKeyList, name, arg),

            // `database_list` (pragma.html #pragma_database_list): one row per open
            // database — `seq` (0-based, index order), `name` (`main`/`temp`/an ATTACH
            // alias), `file` (the backing file's absolute path, empty for an in-memory or
            // temporary database). Always at least the `main` row (seq 0); `temp` (seq 1)
            // appears once materialized and each ATTACHed database at seq 2.. — see
            // `pragma_database_list`.
            "database_list" => Ok(Some(self.pragma_database_list())),

            // `integrity_check` / `quick_check` (pragma.html): SQLite runs a battery of
            // structural checks and returns one row per problem found, or the single row
            // `ok` when the database is consistent. This engine only ever produces
            // well-formed databases (every write goes through the b-tree/pager invariants),
            // so it reports `ok` — the same answer real sqlite gives for a sound file. The
            // optional integer argument (a max-error cap) is accepted and ignored. Returning
            // one row is required: SQLite's `integrity_check` always yields at least a row,
            // so an empty result would itself be a divergence. `quick_check` is the same
            // interface with the (skipped) index cross-checks, so it shares the answer, only
            // the column name differs (SQLite names the result column after the pragma).
            "integrity_check" => Ok(Some(QueryResult {
                columns: vec!["integrity_check".to_string()],
                rows: vec![vec![Value::Text("ok".to_string())]],
            })),
            "quick_check" => Ok(Some(QueryResult {
                columns: vec!["quick_check".to_string()],
                rows: vec![vec![Value::Text("ok".to_string())]],
            })),

            // WAL / journalling control. `journal_mode` reads or switches the page-1
            // file-format versions (off 18/19, which select the WAL vs rollback backing at
            // open); `wal_checkpoint` drives a checkpoint through the pager seam. Both
            // always return one row, as SQLite does, and both stay MAIN-ONLY (the schema
            // qualifier is ignored). Those version bytes ARE per-database, but per-db
            // journal_mode/checkpoint needs an in-connection backing swap + cross-db WAL
            // wiring that is main-only today (the two-connection WAL rung), so this is a
            // deliberate scope boundary — deferred, not a claim the bytes aren't per-db.
            "journal_mode" => self.pragma_journal_mode(arg),
            "wal_checkpoint" => self.pragma_wal_checkpoint(arg),

            // `recursive_triggers` (pragma.html): whether a DML statement inside a trigger
            // body fires further triggers. SQLite DEFAULTS it OFF. GET reports 0/1; SET
            // records the boolean on the connection [`Runtime`], which the DML operators
            // consult to gate runtime trigger recursion. It is a connection-level setting,
            // not a page-1 header byte, so it lives on the Runtime rather than the header.
            "recursive_triggers" => match arg {
                None => Ok(Some(int_pragma_row(
                    "recursive_triggers",
                    self.rt.recursive_triggers() as i64,
                ))),
                Some(a) => {
                    let on = pragma_arg_to_bool(a)?;
                    self.rt.set_recursive_triggers(on);
                    Ok(None)
                }
            },

            // `foreign_keys` (pragma.html): whether foreign-key constraints are ENFORCED.
            // SQLite DEFAULTS it OFF (version 3.6.19+). GET reports 0/1; SET records the
            // boolean on the connection [`Runtime`], the gate a later FK-enforcement pass
            // will consult. Like `recursive_triggers` it is a connection-level setting, not
            // a page-1 header byte. One SQLite-specific rule: the SET form is a silent
            // no-op WITHIN a transaction (it may only change with no pending BEGIN/
            // SAVEPOINT), because it would otherwise change how already-prepared statements
            // compile mid-transaction; the GET form always reports the current value.
            "foreign_keys" => match arg {
                None => Ok(Some(int_pragma_row("foreign_keys", self.rt.foreign_keys() as i64))),
                Some(a) => {
                    let on = pragma_arg_to_bool(a)?;
                    if !self.txn_active() {
                        self.rt.set_foreign_keys(on);
                    }
                    Ok(None)
                }
            },

            // `defer_foreign_keys` (pragma.html #pragma_defer_foreign_keys): when ON, "all
            // foreign key constraints [are changed] to deferred regardless of how they are
            // declared" for the current transaction — every FK is then checked at COMMIT
            // instead of at statement time. UNLIKE `foreign_keys`, the SET form IS honored
            // inside a transaction (deferring all FKs for that transaction is its entire
            // purpose), and it is AUTOMATICALLY switched OFF at each COMMIT and ROLLBACK
            // (reset in `exec_commit`/`exec_rollback` and when an autocommit statement's
            // implicit transaction ends). A value set before `BEGIN` applies to that
            // transaction. SQLite defaults it OFF. GET reports
            // 0/1; it is a connection/transaction-level flag on the [`Runtime`], not a page-1
            // header byte. It only has an observable enforcement effect while a transaction
            // is open (in autocommit there is nothing to defer into), matching real sqlite.
            "defer_foreign_keys" => match arg {
                None => Ok(Some(int_pragma_row(
                    "defer_foreign_keys",
                    self.rt.defer_foreign_keys() as i64,
                ))),
                Some(a) => {
                    let on = pragma_arg_to_bool(a)?;
                    if !on && !self.txn_active() {
                        // Outside a transaction nothing has been deferred yet, so OFF also
                        // drops the sticky commit-coverage flag; otherwise an ON then OFF
                        // here would still widen the next transaction's COMMIT recheck.
                        self.rt.reset_defer_foreign_keys();
                    } else {
                        self.rt.set_defer_foreign_keys(on);
                    }
                    Ok(None)
                }
            },

            // Any other pragma is a no-op returning no rows. SQLite silently ignores
            // an unrecognized pragma rather than erroring, so we must not error here.
            _ => Ok(None),
        }
    }

    /// A per-statement composite read view over this connection's live schema stores
    /// (`main`, `temp`, and any attached databases), lent to the schema-introspection
    /// pragmas so they can resolve and read a namespace-qualified object. A pure borrow —
    /// [`MultiCatalog`] holds slice references and copies no schema — so building one
    /// fresh per handler is O(1), mirroring how the planner/executor take a catalog view
    /// per statement. Keep the returned view and any reads off it under one borrow.
    fn multi_catalog(&self) -> MultiCatalog<'_> {
        MultiCatalog::new(&self.catalogs, &self.namespaces)
    }

    /// Run one schema-introspection pragma (`table_info`/`table_xinfo`/`index_list`/
    /// `index_info`/`index_xinfo`/`foreign_key_list`) in its GET (row-returning) form.
    ///
    /// The pragma's schema qualifier (`name.schema`) and object argument (the `(name)` call
    /// or the `= name` form) select the target, and the rows are built over the live
    /// [`MultiCatalog`] by the shared [`minisqlite_catalog::pragma_rows`] — the SAME builder
    /// the `pragma_*` table-valued functions use, so `PRAGMA table_info(t)` and
    /// `SELECT * FROM pragma_table_info('t')` cannot diverge (the resolution rule, the
    /// per-object row shape, and the empty-when-absent behavior all live in one place). It
    /// always returns the pragma's fixed columns, with zero rows when the object is absent
    /// (matching SQLite, whose prepared statement still declares the columns but steps no
    /// rows).
    fn pragma_introspect(
        &self,
        kind: PragmaFunction,
        name: &QualifiedName,
        arg: Option<&PragmaArg>,
    ) -> Result<Option<QueryResult>> {
        let mc = self.multi_catalog();
        let object = pragma_arg_to_name(arg);
        let rows = pragma_rows(&mc, kind, name.schema.as_deref(), object.as_deref())?;
        Ok(Some(QueryResult { columns: kind.columns(), rows }))
    }

    /// `PRAGMA database_list` (pragma.html #pragma_database_list): every open database as
    /// `seq | name | file`, walking the namespace registry in index order — `main` (seq 0),
    /// `temp` (seq 1), then attached databases (seq 2..) in attach order.
    ///
    /// `file` is the on-disk path (absolute form when it resolves, else the path as opened)
    /// or the empty string for a backing with no file — an in-memory/temporary attached db
    /// or `temp`, exactly as SQLite reports for `:memory:`.
    ///
    /// The `temp` row appears only once a USER temp object has been materialized
    /// ([`SqlEngine::temp_user_materialized`]) — matching SQLite, which lists `temp` only
    /// after its schema exists, NOT merely because the store was reserved (the first ATTACH
    /// reserves the temp slot without materializing user content). Attached rows always
    /// appear. O(#databases); no data-page access.
    fn pragma_database_list(&self) -> QueryResult {
        let mut rows = Vec::with_capacity(self.namespaces.len());
        for (i, ns) in self.namespaces.iter().enumerate() {
            // Hide the reserved-but-unmaterialized temp slot; every other namespace lists.
            if i == DbIndex::TEMP.index() && !self.temp_user_materialized {
                continue;
            }
            rows.push(vec![
                Value::Integer(i as i64),
                Value::Text(ns.name.clone()),
                Value::Text(display_backing_file(ns.file.as_deref())),
            ]);
        }
        QueryResult { columns: string_columns(&["seq", "name", "file"]), rows }
    }

    /// `PRAGMA journal_mode [= mode]` (pragma.html #pragma_journal_mode). The GET
    /// form reports the current mode; the SET form changes it and reports the
    /// resulting mode. SQLite always returns exactly one row, one column named
    /// `journal_mode`, naming the mode.
    ///
    /// The mode lives in the page-1 header's file-format versions: 2/2 selects WAL,
    /// 1/1 the rollback journal (fileformat2 §1.3). Setting `WAL` writes a version-2
    /// header through the normal (rollback-mode) commit path; the switch takes effect
    /// at the NEXT open, when [`minisqlite_pager::DiskPager`] reads the header and
    /// builds a `WalStore`. A live in-connection swap of the storage backing is
    /// deliberately not attempted — the backing is fixed for an open handle — so the
    /// current connection keeps its rollback backing until reopened. A
    /// rollback-family target writes a version-1 header. An in-memory database is
    /// always `memory` and ignores the change (there is no file to reopen), matching
    /// SQLite.
    fn pragma_journal_mode(&mut self, arg: Option<&PragmaArg>) -> Result<Option<QueryResult>> {
        if self.in_memory() {
            return Ok(Some(journal_mode_row("memory")));
        }
        let resulting = match arg {
            None => self.journal_mode_str()?,
            Some(a) => {
                let requested = pragma_arg_to_name(Some(a)).unwrap_or_default().to_ascii_lowercase();
                match requested.as_str() {
                    "wal" => {
                        self.set_journal_versions(2)?;
                        "wal".to_string()
                    }
                    // The rollback-journal family. This engine implements DELETE-style
                    // journaling for all of them (their crash-recovery guarantee is
                    // identical in observable behavior — committed data survives, a
                    // rolled-back transaction leaves no trace); the version byte only
                    // records "not WAL". Echo the requested name as SQLite does.
                    "delete" | "truncate" | "persist" | "memory" | "off" => {
                        self.set_journal_versions(1)?;
                        requested
                    }
                    // An unrecognized token changes nothing and reports the current
                    // mode (SQLite likewise leaves the mode unchanged).
                    _ => self.journal_mode_str()?,
                }
            }
        };
        Ok(Some(journal_mode_row(&resulting)))
    }

    /// The mode string for the current page-1 header: `wal` when both file-format
    /// versions are 2, else `delete` (the rollback-journal default). Reused by the
    /// GET form and as the fallback for an unrecognized SET value.
    fn journal_mode_str(&self) -> Result<String> {
        let mode = if self.read_page1_header()?.is_wal() { "wal" } else { "delete" };
        Ok(mode.to_string())
    }

    /// Surgically write both page-1 file-format versions (write @18, read @19) to
    /// `version` (1 = rollback journal, 2 = WAL) through [`SqlEngine::mutate_page1_header`],
    /// preserving every other header field and the b-tree region past byte 100. That
    /// helper's `write_page` auto-commits outside a transaction — a normal rollback-mode
    /// commit, exactly the mechanism the WAL switch relies on — and stages inside one.
    /// Already being in the target mode is a no-op, so a redundant `PRAGMA journal_mode=…`
    /// neither rewrites the header nor bumps the change counter.
    fn set_journal_versions(&mut self, version: u8) -> Result<()> {
        // Already in the target mode: a no-op, so a redundant `PRAGMA journal_mode=…`
        // neither rewrites the header nor bumps the change counter.
        let h = self.read_page1_header()?;
        if h.write_version == version && h.read_version == version {
            return Ok(());
        }
        self.mutate_page1_header(|header| {
            header.write_version = version;
            header.read_version = version;
        })
    }

    /// `PRAGMA wal_checkpoint[(MODE)]` (pragma.html #pragma_wal_checkpoint): run a WAL
    /// checkpoint through the pager seam in the requested MODE and report the result the
    /// way SQLite does — one row of three integers `(busy, log, checkpointed)`.
    ///
    /// The optional argument selects the mode: bare / `PASSIVE` / `FULL` / `RESTART` /
    /// `TRUNCATE` / `NOOP` (an unrecognized argument is PASSIVE, as in SQLite). The
    /// [`CheckpointMode`] and the mode-specific `busy`/`log`/`checkpointed` semantics
    /// live at the pager seam (`WalStore::checkpoint`), which returns a
    /// `CheckpointReport`; this handler only parses the mode name and maps the report
    /// to the row.
    ///
    /// The `log`/`checkpointed` columns are `-1` outside WAL mode (pragma.html). WAL mode
    /// is decided from the page-1 header: a connection whose backing is still rollback
    /// (e.g. `journal_mode=WAL` set but not yet reopened) reports `0`/`0` for the empty
    /// log rather than `-1`, matching real SQLite for a freshly-switched, empty WAL. A
    /// not-live namespace reads the fresh-empty (rollback) default header, so the header
    /// read only errors on a genuinely unreadable/corrupt page 1 — which is propagated
    /// (fail closed) rather than silently reported as "not WAL".
    fn pragma_wal_checkpoint(&mut self, arg: Option<&PragmaArg>) -> Result<Option<QueryResult>> {
        let mode = CheckpointMode::from_pragma_arg(pragma_arg_to_name(arg).as_deref());
        let in_wal = self.read_page1_header()?.is_wal();
        let report = self.pagers[0].checkpoint(mode)?;
        let busy = i64::from(report.busy);
        let (log, checkpointed) = if in_wal {
            (report.log.map_or(0, i64::from), report.checkpointed.map_or(0, i64::from))
        } else {
            (-1, -1)
        };
        Ok(Some(QueryResult {
            columns: vec!["busy".to_string(), "log".to_string(), "checkpointed".to_string()],
            rows: vec![vec![Value::Integer(busy), Value::Integer(log), Value::Integer(checkpointed)]],
        }))
    }

    /// Resolve the target DATABASE for a whole-db HEADER pragma from its schema
    /// qualifier. Per pragma.html "if the optional schema name is omitted, main is
    /// assumed", so an UNQUALIFIED header pragma targets [`DbIndex::MAIN`]. A qualifier
    /// resolves through [`db_of_schema`](SqlEngine::db_of_schema) — the built-ins
    /// (`main`/`temp`/`temporary`) plus any ATTACH alias — and an unknown/unattached one
    /// is `None`, which the callers render as the empty result (columns, zero rows).
    ///
    /// This is DELIBERATELY different from the OBJECT pragmas' name resolution (the shared
    /// [`minisqlite_catalog::pragma_rows`] builder): those resolve an object NAME by the
    /// temp→main→attached SEARCH ORDER, but a header pragma has no object argument, so its
    /// unqualified default is simply `main`.
    fn header_pragma_db(&self, schema: Option<&str>) -> Option<DbIndex> {
        match schema {
            Some(s) => self.db_of_schema(s),
            None => Some(DbIndex::MAIN),
        }
    }

    /// One integer-valued header GET, honoring the schema qualifier. `field` reads the
    /// integer from the resolved database's decoded header. A resolved database yields
    /// one row; an unknown/unattached qualifier yields the columns with ZERO rows (the
    /// "no such database" convention shared with the introspection pragmas), never a
    /// silent read of `main`. A not-live namespace (only `temp` before its first object)
    /// reads the fresh-empty default header (see [`read_page1_header_of`]).
    fn header_int_get(
        &self,
        column: &str,
        schema: Option<&str>,
        field: impl FnOnce(&DatabaseHeader) -> i64,
    ) -> Result<Option<QueryResult>> {
        match self.header_pragma_db(schema) {
            None => Ok(Some(pragma_empty(column))),
            Some(db) => Ok(Some(int_pragma_row(column, field(&self.read_page1_header_of(db)?)))),
        }
    }

    /// One integer-valued header SET, honoring the schema qualifier: apply `f` to the
    /// resolved database's page-1 header (via [`mutate_page1_header_of`]). An
    /// unknown/unattached qualifier resolves to no database and is a silent no-op (never
    /// writes `main` for a `bogus.user_version = N`). SET forms return no rows.
    fn header_int_set(
        &mut self,
        schema: Option<&str>,
        f: impl FnOnce(&mut DatabaseHeader),
    ) -> Result<()> {
        match self.header_pragma_db(schema) {
            None => Ok(()),
            Some(db) => self.mutate_page1_header_of(db, f),
        }
    }

    /// `PRAGMA [schema.]page_count`: the total page count of the RESOLVED database, read
    /// live from that db's pager. An unknown qualifier yields the empty result; a
    /// not-live namespace (only `temp` before its first object) reports the fresh-empty
    /// default of one page — a valid database always contains at least page 1
    /// (fileformat2 §1.3, the header occupies the first 100 bytes of page 1), so this
    /// matches the count a just-materialized empty store would report, with no panic.
    fn header_page_count(&self, schema: Option<&str>) -> Result<Option<QueryResult>> {
        match self.header_pragma_db(schema) {
            None => Ok(Some(pragma_empty("page_count"))),
            Some(db) if self.namespace_live(db) => Ok(Some(int_pragma_row(
                "page_count",
                i64::from(self.pagers[db.index()].page_count()?),
            ))),
            Some(_) => Ok(Some(int_pragma_row("page_count", 1))),
        }
    }

    /// `PRAGMA [schema.]encoding = '…'` — record the RESOLVED database's text encoding at
    /// offset 56 (fileformat2 §1.3.13). The encoding is fixed at database creation
    /// (pragma.html "encoding"), so this is honored ONLY while that database is still
    /// empty (no `sqlite_schema` row has ever been persisted); once any object exists it
    /// is a silent no-op, exactly as sqlite does. The accepted spellings follow
    /// pragma.html — `UTF-8`, `UTF-16`, `UTF-16le`, `UTF-16be` (case-insensitive, and
    /// the hyphen is optional) — with bare `UTF-16` meaning the machine's native byte
    /// order, which on the little-endian reference platform is UTF-16le. An
    /// unrecognized spelling is ignored (sqlite treats a bad `encoding` value as a
    /// no-op, not an error).
    ///
    /// A not-live namespace (only `temp` before its first object) is fresh-empty by
    /// definition, so it passes the freshness gate; the write then materializes it (see
    /// [`mutate_page1_header_of`]).
    fn set_encoding_of(&mut self, db: DbIndex, arg: &PragmaArg) -> Result<()> {
        let Some(name) = pragma_arg_to_name(Some(arg)) else {
            return Ok(());
        };
        let enc = match name.to_ascii_lowercase().as_str() {
            "utf-8" | "utf8" => TEXT_ENCODING_UTF8,
            "utf-16le" | "utf16le" => TEXT_ENCODING_UTF16LE,
            "utf-16be" | "utf16be" => TEXT_ENCODING_UTF16BE,
            // Bare "UTF-16" is the native byte order; the reference platform is
            // little-endian, so it resolves to UTF-16le (matching sqlite there).
            "utf-16" | "utf16" => TEXT_ENCODING_UTF16LE,
            _ => return Ok(()),
        };
        // Fixed at creation: only a brand-new, empty database may still choose its
        // encoding. A LIVE store must hold no object yet; a not-live namespace is
        // fresh-empty by definition, so it passes (and the write below materializes it).
        // OOB-safety rests on `&&` short-circuit ORDER: `catalogs[db.index()]` is reached
        // only after `namespace_live(db)` is true, so a not-live temp never indexes the
        // absent catalog slot — keep the liveness check first (cf. `set_page_size_of`,
        // which guards the same hazard with an explicit if/else).
        if self.namespace_live(db) && !self.catalogs[db.index()].is_freshly_created() {
            return Ok(());
        }
        // Already at the target: skip the page-1 rewrite so a redundant PRAGMA neither
        // touches page 1 nor bumps the change counter.
        if self.read_page1_header_of(db)?.text_encoding == enc {
            return Ok(());
        }
        self.mutate_page1_header_of(db, |header| header.text_encoding = enc)
    }

    /// `PRAGMA [schema.]auto_vacuum = 0|1|2|NONE|FULL|INCREMENTAL` — record the RESOLVED
    /// database's vacuum mode in its page-1 header (fileformat2 §1.3.12): offset 52
    /// (largest root b-tree page) becomes a non-zero "on" flag so the pager begins
    /// laying out pointer-map pages (§1.8) and finalizes it to the true largest root at
    /// each commit; offset 64 (the incremental-vacuum flag) is 1 for INCREMENTAL, 0 for
    /// FULL.
    ///
    /// ENABLING or DISABLING auto-vacuum (crossing between NONE and FULL/INCREMENTAL) is
    /// only honored while the database is still empty — "auto-vacuuming must be turned on
    /// before any tables are created" (pragma.html "auto_vacuum"); once any object exists
    /// that transition needs a full VACUUM rewrite this engine does not yet perform, so
    /// it is a silent no-op there (an honest gap, not a corrupt half-written header).
    /// SWITCHING between FULL and INCREMENTAL only flips offset 64 (no page layout
    /// changes), so it is allowed at any time. An unrecognized argument is ignored
    /// (sqlite treats a bad `auto_vacuum` value as a no-op, not an error).
    ///
    /// A not-live namespace (only `temp` before its first object) is empty by
    /// definition, so an enable passes the freshness gate; the write then materializes it
    /// (via [`mutate_page1_header_of`]). The `namespace_live` short-circuit ORDER keeps a
    /// not-live temp from indexing the absent catalog slot (cf. `set_encoding_of`).
    fn set_auto_vacuum_of(&mut self, db: DbIndex, arg: &PragmaArg) -> Result<()> {
        let Some(mode) = parse_auto_vacuum_mode(arg) else {
            return Ok(());
        };
        let cur = self.read_page1_header_of(db)?;
        let currently_on = cur.largest_root_btree != 0;
        let want_on = mode != 0;
        // Enabling/disabling requires an empty database; a full<->incremental switch
        // (both "on") does not, since it changes no page layout. The `namespace_live`
        // check comes first so a not-live temp never indexes the absent catalog slot.
        if currently_on != want_on
            && self.namespace_live(db)
            && !self.catalogs[db.index()].is_freshly_created()
        {
            return Ok(());
        }
        // Offset 52: a non-zero flag when enabling on an empty database (the pager
        // finalizes it to the true largest root at commit); keep the existing value when
        // already enabled so a mode switch never resets it; 0 when disabling.
        let target_root = if want_on { cur.largest_root_btree.max(1) } else { 0 };
        let target_incr = u32::from(mode == 2);
        // Skip a redundant rewrite so `PRAGMA auto_vacuum = <current>` neither touches
        // page 1 nor bumps the change counter.
        if cur.largest_root_btree == target_root && cur.incremental_vacuum == target_incr {
            return Ok(());
        }
        self.mutate_page1_header_of(db, |h| {
            h.largest_root_btree = target_root;
            h.incremental_vacuum = target_incr;
        })
    }

    /// `PRAGMA [schema.]incremental_vacuum(N)` (pragma.html "incremental_vacuum"): reclaim
    /// up to `N` free pages from the RESOLVED database and truncate the file by the same
    /// amount (fileformat2 §1.8, §1.5). Only meaningful in auto_vacuum=incremental mode
    /// (off52 != 0 AND off64 == 1); a silent no-op otherwise, or when the freelist is
    /// empty, exactly as sqlite. `N` omitted / `< 1` clears the entire freelist.
    ///
    /// Reclamation relocates live tail pages into freed slots and truncates, which the
    /// pager does in its OWN transaction; that cannot nest inside an open one, so this is
    /// honored only in autocommit (a no-op inside a `BEGIN`, matching that a pragma-driven
    /// truncation there has nothing to commit into). A relocation that moves a table/index
    /// ROOT changes `sqlite_schema.rootpage`, so the resolved namespace's cached catalog is
    /// reloaded from the compacted page 1 when that happens.
    fn pragma_incremental_vacuum(
        &mut self,
        schema: Option<&str>,
        arg: Option<&PragmaArg>,
    ) -> Result<()> {
        let Some(db) = self.header_pragma_db(schema) else {
            return Ok(());
        };
        if self.txn_active() || !self.namespace_live(db) {
            return Ok(());
        }
        let h = self.read_page1_header_of(db)?;
        // Incremental mode only: off52 != 0 (auto_vacuum on) AND off64 == 1 (incremental).
        if h.largest_root_btree == 0 || h.incremental_vacuum == 0 {
            return Ok(());
        }
        let outcome = self.pagers[db.index()].incremental_vacuum(incremental_vacuum_count(arg))?;
        if outcome.root_moved {
            self.catalogs[db.index()].load(&*self.pagers[db.index()])?;
        }
        Ok(())
    }

    /// Reload any namespace's cached catalog whose last commit RELOCATED a b-tree root to
    /// satisfy §1.8 roots-first (see [`minisqlite_pager::Pager::take_root_moved`]). A
    /// `CREATE INDEX` on a populated table (or a second `CREATE TABLE` after data) creates
    /// its root above the existing pages; the commit moves that root to the front, which
    /// rewrites `sqlite_schema.rootpage` on disk. An in-commit FULL compaction can likewise
    /// relocate a root. The in-memory catalog still holds the pre-move page, so it must be
    /// reloaded. Called once a statement settles to autocommit.
    ///
    /// Cheap on the common path: one bool poll per live namespace (a field read that is
    /// `false` for every non-vacuum commit, and for every auto_vacuum commit whose roots
    /// were already first), and an actual catalog `load` only when a root really moved
    /// (rare — root-creating/removing DDL, or a compaction that relocated a root, in an
    /// auto_vacuum database). Polls EVERY live namespace, so an auto_vacuum `temp`/attached
    /// store's catalog stays consistent too.
    pub(crate) fn reload_catalogs_after_root_move(&mut self) -> Result<()> {
        for i in 0..self.pagers.len() {
            if self.pagers[i].take_root_moved() {
                self.catalogs[i].load(&*self.pagers[i])?;
            }
        }
        Ok(())
    }

    /// Decode the typed page-1 [`DatabaseHeader`] (magic-checked) of `main`. Kept as the
    /// `main`-only shorthand [`read_page1_header_of`] with [`DbIndex::MAIN`], so the
    /// existing internal callers (journal-mode reads, the `is_wal` checks) stay byte-for-
    /// byte main-only.
    fn read_page1_header(&self) -> Result<DatabaseHeader> {
        self.read_page1_header_of(DbIndex::MAIN)
    }

    /// Decode the typed page-1 [`DatabaseHeader`] (magic-checked) of database `db`. The
    /// single place the engine reads a namespace's header fields, so the on-disk layout
    /// is not re-derived here.
    ///
    /// A NOT-LIVE namespace — only `temp` before its first object, or any absent store —
    /// has no backing pager yet, so its header is the FRESH-EMPTY default
    /// ([`DatabaseHeader::default`]: `page_size` DEFAULT_PAGE_SIZE, UTF-8, and every
    /// header integer field — `user_version`/`application_id`/`schema_cookie`/
    /// `freelist_count`/`default_cache_size`/auto-vacuum — zero, per fileformat2 §1.3).
    /// This is the value a just-materialized empty store would report, so it MUST stay
    /// consistent with what [`init_database`](crate::engine) formats a fresh store to; the
    /// same fresh-empty answer also backs [`header_page_count`](Self::header_page_count)'s
    /// not-live `1` and [`set_page_size_of`](Self::set_page_size_of)'s
    /// `(1, DEFAULT_PAGE_SIZE)` (a conformance test pins a fresh materialized store to
    /// these same values so the two cannot drift apart silently). And it means a `temp.`
    /// header GET never index-panics on the reserved-but-unmaterialized slot.
    fn read_page1_header_of(&self, db: DbIndex) -> Result<DatabaseHeader> {
        if !self.namespace_live(db) {
            return Ok(DatabaseHeader::default());
        }
        let page1 = self.pagers[db.index()].read_page(1)?;
        let head: &[u8; HEADER_SIZE] = page1
            .get(..HEADER_SIZE)
            .ok_or_else(|| Error::format("page 1 is shorter than the database header"))?
            .try_into()
            .expect("a HEADER_SIZE-length slice converts to &[u8; HEADER_SIZE]");
        DatabaseHeader::read(head)
    }

    /// Decode `main`'s page-1 [`DatabaseHeader`], apply `f`, and write it back — the
    /// `main`-only shorthand for [`mutate_page1_header_of`] with [`DbIndex::MAIN`], kept
    /// for the existing unqualified setters (`journal_mode`).
    fn mutate_page1_header(&mut self, f: impl FnOnce(&mut DatabaseHeader)) -> Result<()> {
        self.mutate_page1_header_of(DbIndex::MAIN, f)
    }

    /// Decode database `db`'s page-1 [`DatabaseHeader`], apply `f` to it, and write the
    /// re-encoded 100-byte header back over that db's page 1, preserving the bytes past
    /// the header (the page's `sqlite_schema` b-tree). The single home for the surgical
    /// page-1 read-modify-write — the borrow-release copy, the fail-closed short-page
    /// check, and the header codec round-trip — shared by the `user_version` /
    /// `journal_mode` / `encoding` setters. `write_page` auto-commits outside a
    /// transaction and stages inside one, so the update is transactional in both modes.
    ///
    /// A resolvable-but-not-live target (only `temp` before its first object) is
    /// MATERIALIZED first ([`ensure_header_target_live`](SqlEngine::ensure_header_target_live)),
    /// so a `PRAGMA temp.<field> = V` persists in the (auto-created) temp database for
    /// this connection — matching sqlite — instead of index-panicking on the absent slot.
    fn mutate_page1_header_of(
        &mut self,
        db: DbIndex,
        f: impl FnOnce(&mut DatabaseHeader),
    ) -> Result<()> {
        self.ensure_header_target_live(db)?;
        let i = db.index();
        // Copy page 1 first (releasing the read borrow) so the write borrow that
        // follows does not overlap it.
        let mut page = self.pagers[i].read_page(1)?.to_vec();
        let head: &mut [u8; HEADER_SIZE] = page
            .get_mut(..HEADER_SIZE)
            .ok_or_else(|| Error::format("page 1 is shorter than the database header"))?
            .try_into()
            .expect("a HEADER_SIZE-length slice converts to &mut [u8; HEADER_SIZE]");
        let mut header = DatabaseHeader::read(head)?;
        f(&mut header);
        header.write(head);
        self.pagers[i].write_page(1, &page)
    }
}

/// Build the fixed column-name list of a pragma result from string literals.
fn string_columns(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// The `file` column for one `PRAGMA database_list` row: the absolute (canonicalized) path
/// of a file-backed database, the path as opened if canonicalization fails (e.g. the file
/// was removed underneath us), or the empty string for a backing with no file (`:memory:`,
/// a temporary database, or `temp`) — matching SQLite's empty `file` for an in-memory db.
fn display_backing_file(file: Option<&Path>) -> String {
    match file {
        Some(path) => std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned(),
        None => String::new(),
    }
}

/// The single-row, single-column (`journal_mode` = `mode`) result both the GET and
/// SET forms of `PRAGMA journal_mode` return.
fn journal_mode_row(mode: &str) -> QueryResult {
    QueryResult {
        columns: vec!["journal_mode".to_string()],
        rows: vec![vec![Value::Text(mode.to_string())]],
    }
}

/// The single-row, single-column result a header-integer GET pragma returns: one
/// column named for the pragma, one integer value. Shared by `user_version`,
/// `application_id`, `schema_version`, `default_cache_size`, and `page_size`, whose
/// GET forms differ only in the column name and which header field they read.
fn int_pragma_row(column: &str, value: i64) -> QueryResult {
    QueryResult {
        columns: vec![column.to_string()],
        rows: vec![vec![Value::Integer(value)]],
    }
}

/// The single-row, single-column TEXT result a header-string GET pragma returns (one
/// column named for the pragma, one text value) — used by `encoding`.
fn text_pragma_row(column: &str, value: &str) -> QueryResult {
    QueryResult {
        columns: vec![column.to_string()],
        rows: vec![vec![Value::Text(value.to_string())]],
    }
}

/// The empty result a header GET returns when its schema qualifier names no live
/// database (`PRAGMA bogus.user_version`): the pragma's single column, ZERO rows. This
/// is the "no such database" convention the schema-introspection pragmas already use —
/// no error, no panic, and NOT a silent read of `main`.
fn pragma_empty(column: &str) -> QueryResult {
    QueryResult { columns: vec![column.to_string()], rows: Vec::new() }
}

/// The `encoding` label for a header text-encoding code (fileformat2 §1.3.13): 2 →
/// `UTF-16le`, 3 → `UTF-16be`, and 1 (UTF-8) or 0 (unset — SQLite's UTF-8 default) →
/// `UTF-8`.
fn encoding_label(code: u32) -> &'static str {
    match code {
        TEXT_ENCODING_UTF16LE => "UTF-16le",
        TEXT_ENCODING_UTF16BE => "UTF-16be",
        _ => "UTF-8",
    }
}

/// The `auto_vacuum` mode a header encodes (fileformat2 §1.3.12): 0 (none) when the
/// largest-root-btree field (off 52) is zero; else 1 (full) when the incremental-vacuum
/// flag (off 64) is zero, else 2 (incremental). The inverse of the mapping
/// [`SqlEngine::set_auto_vacuum_of`] writes, and the decoder for a file real sqlite wrote.
fn auto_vacuum_mode(h: &DatabaseHeader) -> i64 {
    if h.largest_root_btree == 0 {
        0
    } else if h.incremental_vacuum == 0 {
        1
    } else {
        2
    }
}

/// Parse a `PRAGMA auto_vacuum = …` argument into a mode code — 0 (NONE), 1 (FULL), or
/// 2 (INCREMENTAL) — accepting the integer forms `0`/`1`/`2` and the keyword spellings
/// `none`/`full`/`incremental` (case-insensitive), exactly the vocabulary pragma.html
/// documents. `None` for any other value, which the caller treats as a silent no-op
/// (sqlite likewise ignores an unrecognized `auto_vacuum` argument rather than erroring).
fn parse_auto_vacuum_mode(arg: &PragmaArg) -> Option<i64> {
    let value = match arg {
        PragmaArg::Equals(v) | PragmaArg::Call(v) => v,
    };
    match value {
        PragmaValue::Literal(Literal::Integer(i @ (0 | 1 | 2))) => Some(*i),
        PragmaValue::Name(s) | PragmaValue::Literal(Literal::Text(s)) => {
            match s.to_ascii_lowercase().as_str() {
                "none" => Some(0),
                "full" => Some(1),
                "incremental" => Some(2),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract the object name a schema-introspection pragma carries. Both the call
/// form `PRAGMA p(name)` and the assignment form `PRAGMA p = name` are accepted, and
/// the name may be a bare identifier or a quoted string literal. A numeric/boolean
/// literal or a missing argument yields `None` (the pragma then returns no rows).
fn pragma_arg_to_name(arg: Option<&PragmaArg>) -> Option<String> {
    let value = match arg? {
        PragmaArg::Equals(v) | PragmaArg::Call(v) => v,
    };
    match value {
        PragmaValue::Name(s) => Some(s.clone()),
        PragmaValue::Literal(Literal::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Extract the boolean a `PRAGMA name = value` / `PRAGMA name(value)` carries. Accepts the
/// SQLite boolean spellings: the integers `1`/`0` (any nonzero is true), the `true`/`false`
/// keyword literals, and the barewords `on`/`off`, `yes`/`no` (case-insensitive). Anything
/// else is rejected rather than silently coerced to the wrong setting.
fn pragma_arg_to_bool(arg: &PragmaArg) -> Result<bool> {
    let value = match arg {
        PragmaArg::Equals(v) | PragmaArg::Call(v) => v,
    };
    match value {
        PragmaValue::Literal(Literal::Integer(i)) => Ok(*i != 0),
        PragmaValue::Literal(Literal::True) => Ok(true),
        PragmaValue::Literal(Literal::False) => Ok(false),
        PragmaValue::Name(s) | PragmaValue::Literal(Literal::Text(s)) => {
            match s.to_ascii_lowercase().as_str() {
                "on" | "yes" | "true" => Ok(true),
                "off" | "no" | "false" => Ok(false),
                other => Err(Error::sql(format!("unsupported PRAGMA boolean value: {other}"))),
            }
        }
        other => Err(Error::sql(format!("unsupported PRAGMA boolean value: {other:?}"))),
    }
}

/// Map a `PRAGMA incremental_vacuum` argument to a reclaim budget (pragma.html): an
/// omitted argument, a value `< 1`, or a non-integer all mean "clear the entire freelist"
/// (`None`); a positive `N` caps reclamation at `N` pages (`Some(N)`, saturating into the
/// page-id width). "Fewer than N on the freelist" needs no special case here — the
/// reclaimer simply stops when the freelist is exhausted.
fn incremental_vacuum_count(arg: Option<&PragmaArg>) -> Option<u32> {
    match arg.and_then(|a| pragma_arg_to_i64(a).ok()) {
        Some(n) if n >= 1 => Some(n.min(u32::MAX as i64) as u32),
        _ => None,
    }
}

/// Extract the integer a `PRAGMA name = value` / `PRAGMA name(value)` carries.
/// Numeric and boolean literals map to the obvious integer (a real is truncated);
/// anything else is rejected rather than silently coerced to a wrong value.
fn pragma_arg_to_i64(arg: &PragmaArg) -> Result<i64> {
    let value = match arg {
        PragmaArg::Equals(v) | PragmaArg::Call(v) => v,
    };
    match value {
        PragmaValue::Literal(Literal::Integer(i)) => Ok(*i),
        PragmaValue::Literal(Literal::Real(r)) => Ok(*r as i64),
        PragmaValue::Literal(Literal::True) => Ok(1),
        PragmaValue::Literal(Literal::False) => Ok(0),
        other => Err(Error::sql(format!("unsupported PRAGMA integer value: {other:?}"))),
    }
}
