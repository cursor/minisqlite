//! `SchemaCatalog` — the unified schema store: the `sqlite_schema` b-tree at page 1
//! is the source of truth, and an in-memory case-insensitive cache over it makes
//! lookups O(1). The same code serves an in-memory database (a valid SQLite image
//! with a real page-1 schema) and an on-disk file.
//!
//! Write path (`create_table` / `create_index`): allocate the object's b-tree root,
//! encode its `sqlite_schema` row and insert it on page 1, bump the schema cookie in
//! the database header, then cache the def. The cache write is last, so a failed
//! persistence never leaves a cached-but-unpersisted object.
//!
//! Read path: `load` scans page 1 and rebuilds the cache by parsing each stored
//! `sql`; lookups hit the HashMaps. It runs in two passes — tables and views first,
//! then indexes and triggers — because an index or trigger row resolves its target
//! table from the cache and must not depend on the on-disk row order (a view depends
//! only on its own `sql`, so it caches in pass 1). `next_schema_rowid` is tracked in
//! memory (never re-scanned) so each DDL costs O(1) beyond the b-tree insert.
//!
//! Case-insensitivity is the load-bearing invariant: SQLite identifiers fold over
//! ASCII, so `SELECT * FROM T` must find a table created as `t`. Every map is keyed
//! by the ASCII-lowercased name (`norm`), while the def stored inside keeps the
//! original spelling for reporting.

use std::collections::{HashMap, HashSet};

use minisqlite_btree::{
    create_index_btree, create_table_btree, table_delete, table_insert, TableCursor,
};
use minisqlite_fileformat::{decode_record_enc, DatabaseHeader, HEADER_SIZE};
use minisqlite_pager::{text_encoding_of, PageId, Pager};
use minisqlite_sql::{
    parse, AlterAction, AlterTable, ColumnConstraintKind, ColumnDef as SqlColumnDef, CreateIndex,
    CreateTable, CreateTableBody, CreateTrigger, CreateView, DefaultValue, Drop, DropKind,
    IndexedColumnTarget, Literal, Statement, TriggerTiming,
};
use minisqlite_types::{Error, Result};

use crate::alter_data;
use crate::alter_rewrite;
use crate::builder::{has_autoincrement, index_def_from_ast, table_def_from_ast};
use crate::catalog::Catalog;
use crate::def::{ColumnDef, IndexDef, TableDef, TriggerDef, TriggerTargetDb, ViewDef};
use crate::drop_checks;
use crate::schema_row::SchemaRow;
use crate::TriggerTarget;

/// Page 1 is always the root of the `sqlite_schema` b-tree.
const SCHEMA_ROOT: PageId = 1;

/// Fold an identifier to its case-insensitive lookup key. SQLite folds only ASCII
/// (a Unicode `İ` is not equal to `i`), so this is `to_ascii_lowercase`, not the
/// Unicode `to_lowercase` — matching the reference exactly.
fn norm(name: &str) -> String {
    name.to_ascii_lowercase()
}

/// The unified `Catalog`. Tables, indexes, views, and triggers are owned here and
/// borrowed out on lookup, so a read never copies the schema.
///
/// `table_indexes` maps a table's folded name to its indexes' folded names in
/// creation order, so `indexes_on` costs O(#indexes on the table). Its invariant —
/// every name it lists is a key in `indexes` — holds by construction: the two are
/// only ever written together.
///
/// `views` / `triggers` are keyed by the object's folded name. They are load-bearing
/// beyond persistence: tables, indexes, views, and triggers share ONE schema
/// namespace, so caching views/triggers is what lets a `CREATE TABLE`/`CREATE INDEX`
/// reject a name already taken by a view or trigger (via [`name_in_use`](Self::name_in_use)).
///
/// `next_schema_rowid` is the rowid to assign the next persisted `sqlite_schema`
/// row. It is the max stored rowid + 1 (or 1 for an empty schema), tracked in
/// memory so a DDL never re-scans page 1 to find it.
pub struct SchemaCatalog {
    tables: HashMap<String, TableDef>,
    indexes: HashMap<String, IndexDef>,
    table_indexes: HashMap<String, Vec<String>>,
    views: HashMap<String, ViewDef>,
    triggers: HashMap<String, TriggerDef>,
    next_schema_rowid: i64,
}

impl SchemaCatalog {
    /// A new store holding only the built-in `sqlite_schema` table (page 1). The
    /// engine constructs it this way and either fills it as DDL runs or calls
    /// [`load`](Catalog::load) to rebuild it from an existing database image.
    pub fn new() -> SchemaCatalog {
        let mut cat = SchemaCatalog {
            tables: HashMap::new(),
            indexes: HashMap::new(),
            table_indexes: HashMap::new(),
            views: HashMap::new(),
            triggers: HashMap::new(),
            next_schema_rowid: 1,
        };
        cat.register_builtin();
        cat
    }

    /// Register the built-in `sqlite_schema` table def (root page 1). This is NOT a
    /// persisted page-1 row — the format has no `sqlite_schema` row for itself — so
    /// it is a purely in-memory registration that lets a query read the schema
    /// table through the ordinary table path (`SELECT * FROM sqlite_master`).
    fn register_builtin(&mut self) {
        self.tables.insert(norm(SQLITE_SCHEMA_NAME), sqlite_schema_def());
    }

    /// True when no `sqlite_schema` row has ever been persisted — the database is
    /// brand-new and holds no user table/index/view/trigger. The text encoding and
    /// auto-vacuum mode are fixed at database creation ("it is not possible to change
    /// the text encoding of a database after it has been created", pragma.html), so
    /// this is the gate the engine uses to decide whether `PRAGMA encoding = …` may
    /// still rewrite the header. `next_schema_rowid` is 1 exactly until the first
    /// object is persisted (and is rebuilt as max-rowid+1 on `load`), so a reopened
    /// database that already has objects reports `false`.
    pub fn is_freshly_created(&self) -> bool {
        self.next_schema_rowid == 1
    }
}

impl Default for SchemaCatalog {
    fn default() -> Self {
        SchemaCatalog::new()
    }
}

impl Catalog for SchemaCatalog {
    fn table(&self, name: &str) -> Result<Option<&TableDef>> {
        Ok(self.tables.get(resolve_table_key(&norm(name))))
    }

    fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
        Ok(self.indexes.get(&norm(name)))
    }

    fn view(&self, name: &str) -> Result<Option<&ViewDef>> {
        Ok(self.views.get(&norm(name)))
    }

    fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>> {
        let Some(names) = self.table_indexes.get(&norm(table)) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let def = self.indexes.get(name).expect(
                "SchemaCatalog invariant: an index name linked to a table must be present in the index map",
            );
            out.push(def);
        }
        Ok(out)
    }

    fn triggers_on<'a>(&'a self, table: &str) -> Result<Vec<&'a TriggerDef>> {
        let key = norm(table);
        let mut out: Vec<&TriggerDef> =
            self.triggers.values().filter(|t| norm(&t.table) == key).collect();
        // SQLite fires a table's same-event triggers in creation (sqlite_schema rowid)
        // order. The trigger cache is keyed by folded NAME and carries no per-trigger
        // rowid, so we order by folded trigger name — a stable total order (names are
        // unique in the shared schema namespace) that is reproducible run to run.
        // Faithful creation-order firing would need a rowid/ordinal on `TriggerDef`;
        // that is deferred, so this order is deterministic but not rowid-faithful.
        // `sort_by_cached_key` folds each name once (O(k) allocations), not once per
        // comparison as a `sort_by` closure would.
        out.sort_by_cached_key(|t| norm(&t.name));
        Ok(out)
    }

    fn tables<'a>(&'a self) -> Result<Vec<&'a TableDef>> {
        // Every table lives in `self.tables` keyed by its folded name, so an internal
        // table is exactly one whose key carries the reserved `sqlite_` prefix — the
        // built-in `sqlite_schema`, the auto-created `sqlite_sequence`, and the
        // `ANALYZE`-created `sqlite_stat1`. That is the
        // same test `create_table`/`drop_object` use, and it is case-insensitive by
        // construction (keys are already `norm`-folded), so no user table is ever
        // filtered out (a user name cannot begin with `sqlite_`). Order is unspecified.
        Ok(self
            .tables
            .iter()
            .filter(|(key, _)| !key.starts_with("sqlite_"))
            .map(|(_, def)| def)
            .collect())
    }

    fn load(&mut self, pager: &dyn Pager) -> Result<()> {
        // Reset to a bare store (keeping the built-in), then repopulate from page 1.
        self.tables.clear();
        self.indexes.clear();
        self.table_indexes.clear();
        self.views.clear();
        self.triggers.clear();
        self.register_builtin();

        // Pass 1: scan page 1 once. Load every table AND view into the cache, and set
        // aside the index and trigger rows — each resolves its target table from the
        // cache, so it must be processed only after ALL tables are loaded, regardless
        // of on-disk row order. A view depends only on its own `sql`, so it caches
        // here in pass 1. `max_rowid` counts ALL rows (including the deferred index and
        // trigger rows) so `next_schema_rowid` stays correct even for rows a later pass
        // reads.
        let mut max_rowid: i64 = 0;
        let mut pending_indexes: Vec<SchemaRow> = Vec::new();
        let mut pending_triggers: Vec<SchemaRow> = Vec::new();
        // The schema's TEXT columns (object names and `sql`) are stored in the database's
        // declared encoding (§1.3.13), so a UTF-16 database keeps its schema as UTF-16;
        // decode each row in that encoding to recover UTF-8 names/SQL.
        let enc = text_encoding_of(pager);
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT)?;
        let mut positioned = cursor.first()?;
        while positioned {
            let rowid = cursor.rowid();
            if rowid > max_rowid {
                max_rowid = rowid;
            }
            let row = {
                let payload = cursor.payload()?;
                SchemaRow::from_values(&decode_record_enc(&payload, enc))?
            };
            match row.obj_type.as_str() {
                "table" => self.load_table_row(&row)?,
                "index" => pending_indexes.push(row),
                "view" => self.load_view_row(&row)?,
                "trigger" => pending_triggers.push(row),
                other => {
                    return Err(Error::Format(format!(
                        "sqlite_schema has unknown object type {other:?}"
                    )));
                }
            }
            positioned = cursor.next()?;
        }

        // Pass 2: every table AND view is cached now, so each index row can resolve its
        // target table and each trigger row can resolve its target — a table (BEFORE/AFTER)
        // or a view (INSTEAD OF).
        for row in &pending_indexes {
            self.load_index_row(row)?;
        }
        for row in &pending_triggers {
            self.load_trigger_row(row)?;
        }

        // The next persisted rowid is one past the largest seen (1 for empty schema).
        self.next_schema_rowid = max_rowid + 1;
        Ok(())
    }

    fn create_table(&mut self, pager: &mut dyn Pager, stmt: &CreateTable, sql: &str) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);

        // SQLite reserves the `sqlite_` name prefix for internal schema objects; a
        // user CREATE TABLE may not take one (this also prevents shadowing the
        // built-in `sqlite_schema`). Checked first, before the namespace lookup.
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("object name reserved for internal use: {name}")));
        }

        // Tables, indexes, views, and triggers share ONE namespace (lang_createindex.html
        // / lang_createview.html / lang_createtrigger.html). A collision with a DIFFERENT
        // object kind is always a hard error — IF NOT EXISTS suppresses ONLY a same-type
        // (table) duplicate, handled in the `"table"` arm. The index wording is preserved
        // verbatim for the landed tests; view/trigger reuse the same "already a <kind>
        // named" shape.
        match self.name_in_use(&key) {
            Some("table") => {
                if stmt.if_not_exists {
                    return Ok(());
                }
                return Err(Error::Sql(format!("table {name} already exists")));
            }
            Some("index") => {
                return Err(Error::Sql(format!("there is already an index named {name}")));
            }
            Some(other) => {
                return Err(Error::Sql(format!("there is already a {other} named {name}")));
            }
            None => {}
        }

        // Allocate the table's b-tree root, then build its schema def. Building can
        // still fail (`CREATE TABLE ... AS SELECT` is deferred); DDL runs inside the
        // caller's transaction, so the just-allocated root is discarded on rollback.
        //
        // A WITHOUT ROWID table stores its rows in an INDEX b-tree keyed by its PRIMARY
        // KEY (fileformat2 §2.4), not a rowid table b-tree — the whole row is the key
        // record. Allocate the matching root page kind up front so the index-cursor
        // read/write path sees the right page type (and a real sqlite3 can cross-read
        // the file). A rowid table keeps the table-btree root as before.
        let without_rowid = matches!(
            &stmt.body,
            CreateTableBody::Columns { options, .. } if options.without_rowid
        );
        let root =
            if without_rowid { create_index_btree(pager)? } else { create_table_btree(pager)? };
        let def = table_def_from_ast(stmt, root)?;

        // Persist the table's own row FIRST (a table precedes its auto-indexes on page
        // 1), then a NULL-`sql` row + empty b-tree root for each auto-index the table's
        // UNIQUE / PRIMARY KEY constraints imply. All rows for this single `CREATE
        // TABLE` are ONE schema change, so they use `insert_schema_row` (no cookie bump)
        // and the cookie is bumped exactly once, after the last row. The def is cached
        // only after every write succeeds.
        let table_row = SchemaRow {
            obj_type: "table".to_string(),
            name: name.clone(),
            tbl_name: name.clone(),
            rootpage: root as i64,
            sql: Some(sql.to_string()),
        };
        self.insert_schema_row(pager, &table_row)?;

        // A `WITHOUT ROWID` PK spec has `emit_row == false`: it reserves its N (so later
        // UNIQUE indexes number past it) but owns no separate index — the table's own
        // b-tree IS that key — so it gets no root and no row. Every other spec is a real
        // auto-index: an empty b-tree root and a NULL-`sql` `sqlite_autoindex_*` row.
        let mut auto_index_defs: Vec<IndexDef> = Vec::new();
        for spec in &def.auto_indexes {
            if !spec.emit_row {
                continue;
            }
            let index_root = create_index_btree(pager)?;
            let index_row = SchemaRow {
                obj_type: "index".to_string(),
                name: spec.name.clone(),
                tbl_name: name.clone(),
                rootpage: index_root as i64,
                sql: None,
            };
            self.insert_schema_row(pager, &index_row)?;
            auto_index_defs.push(IndexDef::from_auto_spec(
                spec.name.clone(),
                name.clone(),
                spec,
                index_root,
            ));
        }

        // autoinc.html §3: creating a table with an AUTOINCREMENT column auto-creates
        // the internal `sqlite_sequence` bookkeeping table if it does not already
        // exist, as part of this SAME schema change (persisted before the single cookie
        // bump below). The builder has already guaranteed any AUTOINCREMENT is a valid
        // INTEGER PRIMARY KEY, so the AST flag alone is a sufficient trigger. It is
        // written LAST (after the table's own auto-index rows) so the common
        // no-auto-index case orders the rows as `t` then `sqlite_sequence`.
        let seq_created = if has_autoincrement(stmt) {
            self.ensure_sqlite_sequence(pager)?
        } else {
            None
        };

        // One `CREATE TABLE` is one schema change: bump the cookie exactly once, after
        // the table row, every auto-index row, and any `sqlite_sequence` row are written.
        bump_schema_cookie(pager)?;

        // Cache last (persistence succeeded): the table, then its auto-indexes in
        // N-order, so `indexes_on` lists them in creation order like the explicit path.
        self.tables.insert(key, def);
        if let Some((seq_key, seq_def)) = seq_created {
            self.tables.insert(seq_key, seq_def);
        }
        for index_def in auto_index_defs {
            let index_key = norm(&index_def.name);
            self.indexes.insert(index_key.clone(), index_def);
            self.table_indexes.entry(norm(name)).or_default().push(index_key);
        }
        Ok(())
    }

    fn create_index(&mut self, pager: &mut dyn Pager, stmt: &CreateIndex, sql: &str) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);

        // An index name is reserved under the same `sqlite_` rule as a table name and
        // shares the one schema namespace (lang_createindex.html): it may collide with
        // no existing table, index, view, or trigger.
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("object name reserved for internal use: {name}")));
        }
        // Same shared-namespace rule as create_table, with the index as the same-type
        // kind here: IF NOT EXISTS softens ONLY an index-vs-index duplicate; a collision
        // with a table, view, or trigger is always a hard error. The table wording is
        // preserved verbatim for the landed tests.
        match self.name_in_use(&key) {
            Some("index") => {
                if stmt.if_not_exists {
                    return Ok(());
                }
                return Err(Error::Sql(format!("index {name} already exists")));
            }
            Some("table") => {
                return Err(Error::Sql(format!("there is already a table named {name}")));
            }
            Some(other) => {
                return Err(Error::Sql(format!("there is already a {other} named {name}")));
            }
            None => {}
        }

        // Resolve the target table and snapshot what the def build needs (canonical
        // name + columns) BEFORE any mutation, so the immutable borrow of `self` ends
        // before the b-tree writes below take `self`/`pager` mutably.
        let tkey = norm(&stmt.table);
        if tkey.starts_with("sqlite_") {
            return Err(Error::Sql("table sqlite_master may not be indexed".into()));
        }
        let (table_name, table_columns) = {
            let Some(tbl) = self.tables.get(resolve_table_key(&tkey)) else {
                return Err(Error::Sql(format!("no such table: {}", stmt.table)));
            };
            (tbl.name.clone(), tbl.columns.clone())
        };

        // Allocate the (empty) index b-tree root, then build the def. Building can still
        // fail (e.g. an unknown named key column); DDL runs inside the caller's
        // transaction, so the just-allocated root is discarded on rollback. A genuine
        // expression key is captured, not rejected (its refs are bound by the planner).
        // The catalog does NOT backfill existing rows into the new index — the
        // executor does that after create_index returns.
        let root = create_index_btree(pager)?;
        let def = index_def_from_ast(stmt, &table_name, &table_columns, root)?;

        // Persist the schema row on page 1 (advancing the rowid and bumping the cookie
        // as one unit), the exact sequence create_table uses.
        let row = SchemaRow {
            obj_type: "index".to_string(),
            name: name.clone(),
            tbl_name: table_name.clone(),
            rootpage: root as i64,
            sql: Some(sql.to_string()),
        };
        self.persist_schema_row(pager, &row)?;

        // Cache last, appended to the table's list in creation order so the
        // `table_indexes` -> `indexes` invariant (every listed name is a key in
        // `indexes`) holds.
        self.indexes.insert(key.clone(), def);
        self.table_indexes.entry(norm(&table_name)).or_default().push(key);
        Ok(())
    }

    fn create_view(&mut self, pager: &mut dyn Pager, stmt: &CreateView, sql: &str) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);

        // Reserved-name rule first, then the shared namespace (lang_createview.html).
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("object name reserved for internal use: {name}")));
        }
        // A same-type duplicate view is a silent no-op under IF NOT EXISTS; every other
        // collision (a duplicate view without IF NOT EXISTS, or a cross-type collision
        // with a table/index/trigger) is a hard error. SQLite reuses the "already
        // exists" wording for a view (the create path is shared with CREATE TABLE), so
        // one message covers all the error cases here.
        if let Some(kind) = self.name_in_use(&key) {
            if kind == "view" && stmt.if_not_exists {
                return Ok(());
            }
            return Err(Error::Sql(format!("table {name} already exists")));
        }

        // A view owns no b-tree: persist ONE `type='view'` row with rootpage 0 and the
        // verbatim `sql`, as one whole schema change (row insert + cookie bump). No
        // b-tree root is allocated, unlike create_table/create_index.
        let row = SchemaRow {
            obj_type: "view".to_string(),
            name: name.clone(),
            tbl_name: name.clone(),
            rootpage: 0,
            sql: Some(sql.to_string()),
        };
        self.persist_schema_row(pager, &row)?;

        // Cache last: persistence succeeded, so the name is now taken in the namespace.
        self.views.insert(key, ViewDef { name: name.clone(), sql: sql.to_string() });
        Ok(())
    }

    fn create_trigger(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &CreateTrigger,
        sql: &str,
        target: TriggerTarget,
    ) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);

        // Reserved-name rule first, then the shared namespace (lang_createtrigger.html).
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("object name reserved for internal use: {name}")));
        }
        // A same-type duplicate trigger is a no-op under IF NOT EXISTS; any other
        // collision is a hard error.
        if let Some(kind) = self.name_in_use(&key) {
            if kind == "trigger" && stmt.if_not_exists {
                return Ok(());
            }
            return Err(Error::Sql(format!("trigger {name} already exists")));
        }

        // The ON-target's BARE (unqualified) name. Any schema qualifier already selected the
        // store the engine resolved into `target`; the stored `tbl_name` is always bare, like
        // real sqlite, so a DROP TABLE / DROP VIEW cascade matches it by name.
        let target_name = stmt.table.name.clone();
        let target_key = norm(&target_name);

        // Resolve the target's kind (table vs view) and the binding to RECORD, per the
        // engine's resolved verdict:
        //  * SameStore — the common case — validates existence + kind against THIS store's own
        //    cache exactly as before namespaces existed, and records `SameStore`.
        //  * Foreign — a TEMP trigger on a `main`/attached object (lang_createtrigger.html §7)
        //    — trusts the engine's already-resolved existence + kind (the foreign object is not
        //    visible here) and records the target's SCHEMA NAME (or "unqualified") so fire-time
        //    discovery re-resolves it against the live registry (never a cached, DETACH-fragile
        //    numeric index).
        let (target_is_view, target) = match target {
            TriggerTarget::SameStore => {
                let target_is_table = self.tables.get(resolve_table_key(&target_key)).is_some();
                let target_is_view = self.views.contains_key(&target_key);
                // A name is only ever one kind (the shared namespace, enforced by `name_in_use`
                // on every create and fail-closed on load), so at most one of these is true.
                debug_assert!(
                    !(target_is_table && target_is_view),
                    "shared-namespace invariant: {target_name:?} is both a table and a view"
                );
                // The trigger fires on an existing target — a table (BEFORE/AFTER) or a view
                // (INSTEAD OF) — so it must exist at create time (lang_createtrigger.html).
                // Validate before persisting so a stored trigger never dangles. Checked only
                // after the name checks, so IF NOT EXISTS on an existing trigger short-circuits.
                if !target_is_table && !target_is_view {
                    return Err(Error::Sql(format!("no such table: {target_name}")));
                }
                (target_is_view, TriggerTargetDb::SameStore)
            }
            TriggerTarget::Foreign { schema: Some(schema), is_view } => {
                (is_view, TriggerTargetDb::ForeignSchema(schema))
            }
            TriggerTarget::Foreign { schema: None, is_view } => {
                (is_view, TriggerTargetDb::ForeignUnqualified)
            }
        };

        // A trigger may not target an internal/system table (lang_createtrigger.html): the
        // schema table `sqlite_schema`/`sqlite_master` and its aliases, `sqlite_sequence`,
        // `sqlite_stat*`, … — every one carries the reserved `sqlite_` prefix on its folded
        // `target_key`. Checked AFTER a same-store existence check (an absent target reports
        // `no such table` first, matching real sqlite) and for a foreign target too (a TEMP
        // trigger on a system table is equally illegal). Fails closed before any page-1 write.
        if target_key.starts_with("sqlite_") {
            return Err(Error::Sql("cannot create trigger on system table".into()));
        }

        // Timing must match the target kind (lang_createtrigger.html §3): INSTEAD OF
        // fires ONLY on a view; BEFORE/AFTER (and an omitted timing, which defaults to
        // BEFORE) fire ONLY on a table. Reject the two illegal combos, failing closed
        // exactly where real sqlite does, before any page-1 write.
        let is_instead_of = matches!(stmt.timing, Some(TriggerTiming::InsteadOf));
        if target_is_view && !is_instead_of {
            let timing_kw =
                if matches!(stmt.timing, Some(TriggerTiming::After)) { "AFTER" } else { "BEFORE" };
            return Err(Error::Sql(format!(
                "cannot create {timing_kw} trigger on view: {target_name}"
            )));
        }
        if !target_is_view && is_instead_of {
            return Err(Error::Sql(format!(
                "cannot create INSTEAD OF trigger {name} on table {target_name} (INSTEAD OF is only valid on a view)"
            )));
        }

        // Like a view, a trigger owns no b-tree: persist ONE `type='trigger'` row with
        // rootpage 0, its `tbl_name` the bare TARGET (table or view; so a DROP TABLE / DROP
        // VIEW cascade finds it), and the verbatim `sql`. A cross-namespace TEMP trigger is
        // written to the (in-memory) TEMP store just like every temp object — so it lists in
        // `temp.sqlite_master` and shares the temp store's transactional commit/rollback — and
        // is NEVER written to the main file (temp is transient by construction).
        let row = SchemaRow {
            obj_type: "trigger".to_string(),
            name: name.clone(),
            tbl_name: target_name.clone(),
            rootpage: 0,
            sql: Some(sql.to_string()),
        };
        self.persist_schema_row(pager, &row)?;

        // Cache last: the name is now taken, and the target (+ its namespace for a
        // cross-namespace temp trigger) is recorded so fire-time discovery matches the right
        // (db, table) and a DROP TABLE/VIEW cascade finds it.
        self.triggers.insert(
            key,
            TriggerDef { name: name.clone(), table: target_name, sql: sql.to_string(), target },
        );
        Ok(())
    }

    fn alter_table(&mut self, pager: &mut dyn Pager, stmt: &AlterTable, sql: &str) -> Result<()> {
        // One dispatch point per action; each writes its page-1 row(s) in place, bumps
        // the schema cookie once, and updates the cache last.
        match &stmt.action {
            AlterAction::AddColumn(coldef) => self.alter_add_column(pager, stmt, sql, coldef),
            AlterAction::RenameTo(new_name) => self.alter_rename_to(pager, stmt, new_name),
            AlterAction::RenameColumn { from, to } => {
                self.alter_rename_column(pager, stmt, from, to)
            }
            AlterAction::DropColumn(col) => self.alter_drop_column(pager, stmt, col),
        }
    }

    fn drop_object(&mut self, pager: &mut dyn Pager, stmt: &Drop) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);

        // TODO(freelist): reclaim the dropped object's root+descendant pages once a
        // btree free-tree/freelist helper exists. Today `table_delete` frees no pages,
        // so a dropped table (with its indexes' b-trees) or index leaks its pages; the
        // b-tree layer's stance is that a leaked page never affects query correctness,
        // only file size. Views and triggers own no b-tree (rootpage 0), so dropping one
        // leaks nothing beyond its page-1 row, which `table_delete` does remove.
        //
        // Ordering mirrors the create path's cache-last discipline in reverse: delete
        // the page-1 row(s) and bump the cookie FIRST, then mutate the cache, so an
        // early error (before the cache changes) leaves the cache consistent with the
        // rolled-back transaction. `next_schema_rowid` is never decremented — a freed
        // rowid must not be reused by a live row; a later `load` recomputes it as max+1.
        match stmt.kind {
            DropKind::Table => {
                // Internal `sqlite_`-prefixed tables (the schema table and its aliases,
                // sqlite_sequence, sqlite_stat*, …) may never be dropped. Checked before
                // existence per the reserved-name rule, so it fires even under IF EXISTS.
                if key.starts_with("sqlite_") {
                    return Err(Error::Sql(format!("table {name} may not be dropped")));
                }
                // Raw `key`, NOT resolve_table_key: a (non-reserved) user table is matched
                // as itself, never folded to the built-in schema table's key.
                if !self.tables.contains_key(&key) {
                    if stmt.if_exists {
                        return Ok(());
                    }
                    return Err(Error::Sql(format!("no such table: {name}")));
                }
                // CASCADE: the table's own row plus every dependent index/trigger row.
                // Reading page 1 (not the cache) makes the cascade faithful for a
                // loaded-from-disk schema, whose triggers are not modelled in the cache.
                // EXCLUDE any cross-namespace TEMP trigger only bound to a same-named table
                // in ANOTHER db (lang_createtrigger.html §7): in the temp store a trigger on
                // `main.t` shares this bare `tbl_name` "t" but must survive a `DROP TABLE
                // temp.t`. The engine cascades those from their OWN (db, table) separately.
                let foreign = self.foreign_trigger_names_folded();
                let rowids = self.schema_rows_to_delete(&*pager, |row| {
                    (row.obj_type == "table" && norm(&row.name) == key)
                        || (row.obj_type == "index" && norm(&row.tbl_name) == key)
                        || (row.obj_type == "trigger"
                            && norm(&row.tbl_name) == key
                            && !foreign.contains(&norm(&row.name)))
                })?;
                for rowid in rowids {
                    table_delete(pager, SCHEMA_ROOT, rowid)?;
                }
                bump_schema_cookie(pager)?;
                // Cache last: drop the table, every index the cache linked to it, and
                // every SAME-STORE cached trigger that fires on it (a cross-namespace temp
                // trigger bound elsewhere, `t.target.is_foreign()`, is kept). The page-1 index
                // and trigger rows were already deleted above; this keeps the cache in step.
                // A VIEW referencing the dropped table is deliberately left in place — SQLite
                // does not cascade views on DROP TABLE.
                self.tables.remove(&key);
                if let Some(index_names) = self.table_indexes.remove(&key) {
                    for index_name in index_names {
                        self.indexes.remove(&index_name);
                    }
                }
                self.triggers.retain(|_, t| !(norm(&t.table) == key && t.target.is_same_store()));
                Ok(())
            }
            DropKind::Index => {
                // An auto-index (`sqlite_autoindex_*`) — or any `sqlite_`-prefixed index —
                // backs a UNIQUE / PRIMARY KEY constraint and cannot be dropped directly;
                // it goes away only with its table.
                if key.starts_with("sqlite_") {
                    return Err(Error::Sql(
                        "index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped"
                            .into(),
                    ));
                }
                // Resolve the owning table from the cached def BEFORE any mutation, so its
                // `table_indexes` list can be pruned after the def is removed below.
                let Some(def) = self.indexes.get(&key) else {
                    if stmt.if_exists {
                        return Ok(());
                    }
                    return Err(Error::Sql(format!("no such index: {name}")));
                };
                let owning_table = norm(&def.table);
                let rowids = self
                    .schema_rows_to_delete(&*pager, |row| {
                        row.obj_type == "index" && norm(&row.name) == key
                    })?;
                for rowid in rowids {
                    table_delete(pager, SCHEMA_ROOT, rowid)?;
                }
                bump_schema_cookie(pager)?;
                // Cache last: drop the def and unlink it from its table's ordered list.
                self.indexes.remove(&key);
                if let Some(names) = self.table_indexes.get_mut(&owning_table) {
                    names.retain(|n| n.as_str() != key);
                }
                Ok(())
            }
            DropKind::View => {
                // Existence is the view cache (mirrors DROP TABLE), so a name that is not
                // a view errors "no such view" even if an (impossible-in-a-valid-schema)
                // orphan trigger row shared it. The rows to delete are still found by
                // scanning page 1, so a loaded-from-disk view drops as faithfully as a
                // freshly-created one.
                if !self.views.contains_key(&key) {
                    if stmt.if_exists {
                        return Ok(());
                    }
                    return Err(Error::Sql(format!("no such view: {name}")));
                }
                // CASCADE: the view's own row plus every INSTEAD OF trigger whose target
                // (`tbl_name`) is this view — the view mirror of DROP TABLE's trigger
                // cascade. Leaving a trigger row behind would orphan it and make the DB
                // unreadable on the next `load` (its target view would be gone). EXCLUDE a
                // cross-namespace temp trigger only bound to a same-named view in another db
                // (kept for a temp-store `DROP VIEW temp.v`; the engine cascades those from
                // their own (db, view) separately) — the DROP TABLE mirror.
                let foreign = self.foreign_trigger_names_folded();
                let rowids = self.schema_rows_to_delete(&*pager, |row| {
                    (row.obj_type == "view" && norm(&row.name) == key)
                        || (row.obj_type == "trigger"
                            && norm(&row.tbl_name) == key
                            && !foreign.contains(&norm(&row.name)))
                })?;
                for rowid in rowids {
                    table_delete(pager, SCHEMA_ROOT, rowid)?;
                }
                bump_schema_cookie(pager)?;
                // Cache last: free the view name AND every SAME-STORE cached trigger that
                // fires on it (a cross-namespace temp trigger bound elsewhere is kept). The
                // page-1 trigger rows were already deleted above; this keeps the cache in step.
                self.views.remove(&key);
                self.triggers.retain(|_, t| !(norm(&t.table) == key && t.target.is_same_store()));
                Ok(())
            }
            DropKind::Trigger => {
                // DROP TRIGGER removes exactly one trigger and cascades nothing. The
                // page-1 row is found by scanning page 1 (not the cache), so a
                // loaded-from-disk trigger drops as faithfully as a freshly-created one.
                let rowids = self.schema_rows_to_delete(&*pager, |row| {
                    row.obj_type == "trigger" && norm(&row.name) == key
                })?;
                if rowids.is_empty() {
                    if stmt.if_exists {
                        return Ok(());
                    }
                    return Err(Error::Sql(format!("no such trigger: {name}")));
                }
                for rowid in rowids {
                    table_delete(pager, SCHEMA_ROOT, rowid)?;
                }
                bump_schema_cookie(pager)?;
                // Cache last: free the dropped name so a later CREATE can reuse it.
                self.triggers.remove(&key);
                Ok(())
            }
        }
    }
}

impl SchemaCatalog {
    /// Does this store define a TRIGGER named `name` (case-insensitively)? The `Catalog`
    /// seam looks triggers up by their TARGET table ([`triggers_on`](Catalog::triggers_on)),
    /// never by the trigger's own name, but `DROP TRIGGER name` and namespace routing need
    /// the by-name check — so the engine asks this of each concrete per-namespace store to
    /// find which one owns the trigger. (Tables/indexes/views already have by-name lookups
    /// on the seam.)
    pub fn has_trigger_named(&self, name: &str) -> bool {
        self.triggers.contains_key(&norm(name))
    }

    /// Does this store hold any CROSS-NAMESPACE (foreign-bound) TEMP trigger — one whose
    /// [`target`](crate::TriggerDef::target) is foreign (`lang_createtrigger.html` §7)? Only
    /// the temp store ever does. It gates BOTH the fire path
    /// ([`triggers_on_in`](Catalog::triggers_on_in) skips the cross-namespace pass when this
    /// is `false`) and the engine's `DROP TABLE`/`DROP VIEW` cross-namespace cascade, so a
    /// connection with no such trigger (the overwhelmingly common case) pays nothing.
    pub fn has_cross_namespace_triggers(&self) -> bool {
        self.triggers.values().any(|t| t.target.is_foreign())
    }

    /// The folded names of this store's cross-namespace (foreign-bound) triggers — those a
    /// same-named `DROP TABLE`/`DROP VIEW` in THIS store must NOT cascade (they target another
    /// db's object by bare-name coincidence). Empty for every non-temp store, so the cascade
    /// predicates that consult it are byte-for-byte unchanged there.
    fn foreign_trigger_names_folded(&self) -> HashSet<String> {
        self.triggers
            .iter()
            .filter(|(_, t)| t.target.is_foreign())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Drop the triggers named by `folded_names` (already [`norm`]-folded) from THIS store —
    /// the temp-side half of a `DROP TABLE`/`DROP VIEW` cross-namespace cascade
    /// (`lang_createtrigger.html` §7). The ENGINE decides which cross-namespace TEMP triggers
    /// are bound to the dropped `(db, object)` — resolution needs the connection's namespace
    /// registry and search order, which a single store lacks — and passes their names here;
    /// the store just deletes those page-1 rows and cache entries. A no-op when empty.
    pub fn drop_triggers_by_folded_name(
        &mut self,
        pager: &mut dyn Pager,
        folded_names: &[String],
    ) -> Result<()> {
        if folded_names.is_empty() {
            return Ok(());
        }
        let victims: HashSet<&String> = folded_names.iter().collect();
        let rowids = self.schema_rows_to_delete(&*pager, |row| {
            row.obj_type == "trigger" && victims.contains(&norm(&row.name))
        })?;
        for rowid in rowids {
            table_delete(pager, SCHEMA_ROOT, rowid)?;
        }
        bump_schema_cookie(pager)?;
        for k in folded_names {
            self.triggers.remove(k);
        }
        Ok(())
    }

    /// If `key` (an already-[`norm`]-folded name) names an existing schema object,
    /// return that object's kind (`"table"` / `"index"` / `"view"` / `"trigger"`),
    /// else `None`. Tables, indexes, views, and triggers share ONE namespace
    /// (lang_createindex.html / lang_createview.html / lang_createtrigger.html), so
    /// every `create_*` consults this to reject a cross-type collision — a hard error
    /// even under `IF NOT EXISTS`, which suppresses only a same-type duplicate. A name
    /// is only ever in one cache (this very check keeps it so on write, and the load
    /// path fails closed on a schema that violates it), so the order below is not a
    /// precedence: it is simply the first — and only — cache that can match.
    fn name_in_use(&self, key: &str) -> Option<&'static str> {
        if self.tables.contains_key(key) {
            Some("table")
        } else if self.indexes.contains_key(key) {
            Some("index")
        } else if self.views.contains_key(key) {
            Some("view")
        } else if self.triggers.contains_key(key) {
            Some("trigger")
        } else {
            None
        }
    }

    /// Insert one built `SchemaRow` on page 1 at the next schema rowid and advance that
    /// counter — WITHOUT bumping the schema cookie. The rowid MUST advance with the
    /// insert (reusing it would REPLACE the row just written). Used by a DDL that writes
    /// SEVERAL rows for ONE schema change (a `CREATE TABLE` plus its auto-indexes): the
    /// caller bumps the cookie once for the whole change. A single-row DDL uses
    /// [`persist_schema_row`](Self::persist_schema_row) instead.
    fn insert_schema_row(&mut self, pager: &mut dyn Pager, row: &SchemaRow) -> Result<()> {
        let record = row.to_record_enc(text_encoding_of(pager));
        table_insert(pager, SCHEMA_ROOT, self.next_schema_rowid, &record)?;
        self.next_schema_rowid += 1;
        Ok(())
    }

    /// Persist one built `SchemaRow` as a whole single-object schema change: insert it
    /// (advancing the rowid) and bump the schema cookie so a reopen or another
    /// connection sees the schema changed. Used by a DDL that writes exactly one row
    /// (`CREATE INDEX`). Callers cache the def only after this returns, so a failed
    /// persist never leaves a cached-but-unpersisted object.
    fn persist_schema_row(&mut self, pager: &mut dyn Pager, row: &SchemaRow) -> Result<()> {
        self.insert_schema_row(pager, row)?;
        bump_schema_cookie(pager)
    }

    /// Auto-create the internal `sqlite_sequence(name,seq)` table if it does not
    /// already exist (`autoinc.html` §3), returning the `(cache-key, def)` to cache
    /// on success or `None` when it already existed (the idempotent case). The caller
    /// invokes this from within a `CREATE TABLE` that has an AUTOINCREMENT column, so
    /// the row is persisted with [`insert_schema_row`](Self::insert_schema_row) —
    /// sharing that statement's single schema-cookie bump — and cached only after that
    /// bump succeeds, exactly like the enclosing table's own def.
    ///
    /// This is an INTERNAL create: it deliberately BYPASSES the `sqlite_`-reserved-name
    /// rejection that blocks a *user* `CREATE TABLE`, because SQLite creates this table
    /// itself. The stored `sql` is the verbatim string real SQLite writes, so
    /// `SELECT sql FROM sqlite_master WHERE name='sqlite_sequence'` matches and a plain
    /// [`load`](Catalog::load) rebuilds the table by parsing it — there is no
    /// `sqlite_sequence`-specific load path.
    fn ensure_sqlite_sequence(
        &mut self,
        pager: &mut dyn Pager,
    ) -> Result<Option<(String, TableDef)>> {
        const SEQ_SQL: &str = "CREATE TABLE sqlite_sequence(name,seq)";
        let key = norm("sqlite_sequence");
        // Presence check against the cache: a second AUTOINCREMENT table (or a reopened
        // database that already has the table loaded) must not write a duplicate row.
        if self.tables.contains_key(&key) {
            return Ok(None);
        }

        let root = create_table_btree(pager)?;
        let row = SchemaRow {
            obj_type: "table".to_string(),
            name: "sqlite_sequence".to_string(),
            tbl_name: "sqlite_sequence".to_string(),
            rootpage: root as i64,
            sql: Some(SEQ_SQL.to_string()),
        };
        self.insert_schema_row(pager, &row)?;
        // Build the def from the SAME verbatim `sql` a reload parses, so the cached def
        // equals what `load` would reconstruct (the one-builder invariant).
        let def = table_def_from_sql(SEQ_SQL, root)?;
        Ok(Some((key, def)))
    }

    /// Ensure the internal `sqlite_stat1(tbl,idx,stat)` table exists, returning the
    /// page of its b-tree root (whether it already existed or was just created). The
    /// `ANALYZE` command calls this before writing statistics
    /// (`fileformat2.html` §2.6.4, `lang_analyze.html`).
    ///
    /// Like [`ensure_sqlite_sequence`](Self::ensure_sqlite_sequence) this is an
    /// INTERNAL create that deliberately BYPASSES the `sqlite_`-reserved-name
    /// rejection blocking a *user* `CREATE TABLE` — SQLite creates `sqlite_stat1`
    /// itself. The stored `sql` is the verbatim string real SQLite writes, so a plain
    /// [`load`](Catalog::load) rebuilds it by parsing that text (no `sqlite_stat1`-
    /// specific load path) and `SELECT sql FROM sqlite_master WHERE name='sqlite_stat1'`
    /// matches. The three columns are declared WITHOUT type names, exactly as SQLite's
    /// schema shows them, so the on-disk `sql` byte-matches.
    ///
    /// Unlike `ensure_sqlite_sequence` (which shares its enclosing `CREATE TABLE`'s
    /// single cookie bump and defers caching to the caller), an `ANALYZE` is a
    /// standalone statement, so this is a whole one-row schema change on its own:
    /// [`persist_schema_row`](Self::persist_schema_row) writes the row AND bumps the
    /// schema cookie, and the def is cached here. When the table is already present it
    /// is a pure lookup — return the cached root and write nothing.
    pub fn ensure_sqlite_stat1(&mut self, pager: &mut dyn Pager) -> Result<PageId> {
        const STAT1_SQL: &str = "CREATE TABLE sqlite_stat1(tbl,idx,stat)";
        let key = norm("sqlite_stat1");
        if let Some(def) = self.tables.get(&key) {
            return Ok(def.root_page);
        }

        let root = create_table_btree(pager)?;
        let row = SchemaRow {
            obj_type: "table".to_string(),
            name: "sqlite_stat1".to_string(),
            tbl_name: "sqlite_stat1".to_string(),
            rootpage: root as i64,
            sql: Some(STAT1_SQL.to_string()),
        };
        // A standalone schema change: persist the one row and bump the cookie together.
        self.persist_schema_row(pager, &row)?;
        // Cache last (persist+bump succeeded), from the SAME verbatim `sql` a reload
        // parses, so the cached def equals what `load` reconstructs.
        let def = table_def_from_sql(STAT1_SQL, root)?;
        self.tables.insert(key, def);
        Ok(root)
    }

    /// Scan page 1 once and collect, in ascending rowid order, the rowid of every
    /// `sqlite_schema` row matching `predicate`. This is how the drop path finds the
    /// physical rows to delete: a def does not store its own `sqlite_schema` rowid
    /// (that would couple the hot create/load paths), so the rowid is recovered by a
    /// scan only when an object is dropped. Decoding reuses the same
    /// `SchemaRow::from_values(decode_record(..))` the `load` path uses, so a corrupt
    /// row fails closed identically here.
    ///
    /// It only READS: all matching rowids are collected first, and the caller deletes
    /// them after the cursor is dropped — deleting during the scan would invalidate the
    /// open cursor. Thin projection over [`collect_schema_rows`](Self::collect_schema_rows)
    /// so there is ONE page-1 scanner/decoder to keep fail-closed, not two that can drift.
    fn schema_rows_to_delete(
        &self,
        pager: &dyn Pager,
        predicate: impl Fn(&SchemaRow) -> bool,
    ) -> Result<Vec<i64>> {
        Ok(self.collect_schema_rows(pager, predicate)?.into_iter().map(|(rowid, _)| rowid).collect())
    }

    /// Scan page 1 once and collect, in ascending rowid order, `(rowid, SchemaRow)`
    /// for every row matching `predicate`. The one page-1 scanner: it decodes each
    /// row with the same `SchemaRow::from_values(decode_record(..))` the `load` path
    /// uses (so a corrupt row fails closed identically), returns the decoded row so an
    /// ALTER can rewrite it and re-insert it at the SAME rowid (`table_insert`
    /// replaces an existing rowid in place), and `schema_rows_to_delete` projects it to
    /// bare rowids. Read-only: matches are collected first and the caller mutates after
    /// the cursor is dropped, since writing during the scan would invalidate the cursor.
    fn collect_schema_rows(
        &self,
        pager: &dyn Pager,
        predicate: impl Fn(&SchemaRow) -> bool,
    ) -> Result<Vec<(i64, SchemaRow)>> {
        let mut out = Vec::new();
        // Decode schema TEXT (names, `sql`) in the database's encoding (§1.3.13), the
        // same as the `load` path, so a UTF-16 database's rows decode identically here.
        let enc = text_encoding_of(pager);
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT)?;
        let mut positioned = cursor.first()?;
        while positioned {
            let row = {
                let payload = cursor.payload()?;
                SchemaRow::from_values(&decode_record_enc(&payload, enc))?
            };
            if predicate(&row) {
                out.push((cursor.rowid(), row));
            }
            positioned = cursor.next()?;
        }
        Ok(out)
    }

    /// `ALTER TABLE <t> ADD [COLUMN] <coldef>`: append the column to the table's
    /// stored `CREATE TABLE` text and rebuild its cached def. Existing rows are NOT
    /// rewritten — a now-short row reads its new trailing column as NULL
    /// (`lang_altertable.html`), so there is no data migration. SQLite's ADD COLUMN
    /// restrictions are enforced BEFORE any write, so a rejected ALTER persists
    /// nothing.
    fn alter_add_column(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &AlterTable,
        alter_sql: &str,
        coldef: &SqlColumnDef,
    ) -> Result<()> {
        let name = &stmt.name.name;
        let key = norm(name);
        // Reserved internal tables may not be altered (checked before existence, like
        // the create/drop paths), matching sqlite's "table X may not be altered".
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("table {name} may not be altered")));
        }
        // Resolve the target (raw key; the reserved case was handled above).
        let (root_page, existing_columns) = {
            let Some(tbl) = self.tables.get(&key) else {
                return Err(Error::Sql(format!("no such table: {name}")));
            };
            (tbl.root_page, tbl.columns.clone())
        };

        // Validate the new column against SQLite's ADD COLUMN restrictions BEFORE any
        // mutation, so a rejected ALTER writes nothing.
        check_add_column_allowed(coldef, &existing_columns)?;

        // The table's page-1 row is the source of truth for its stored `sql`.
        let mut table_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "table" && norm(&row.name) == key
        })?;
        if table_rows.len() != 1 {
            return Err(Error::Format(format!(
                "sqlite_schema has {} table rows for {name:?}, expected exactly one",
                table_rows.len()
            )));
        }
        let (rowid, table_row) = table_rows.pop().expect("checked len == 1 above");
        let old_sql = table_row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema table row {name:?} has NULL sql"))
        })?;

        // Rewrite: splice the verbatim column definition (recovered from the ALTER
        // text) into the stored column list, then rebuild the def from the NEW sql so
        // the cache equals exactly what a fresh `load` from page 1 would produce (the
        // table's b-tree root is preserved).
        let coldef_text = alter_rewrite::extract_add_column_def(alter_sql)?;
        let new_sql = alter_rewrite::splice_column_into_create(old_sql, &coldef_text)?;
        let new_def = table_def_from_sql(&new_sql, root_page)?;

        // Persist: overwrite the same page-1 rowid with the new sql (only the `sql`
        // column changes — keep the table's original name spelling), then bump the
        // schema cookie once.
        let new_row = SchemaRow {
            obj_type: "table".to_string(),
            name: table_row.name.clone(),
            tbl_name: table_row.tbl_name.clone(),
            rootpage: table_row.rootpage,
            sql: Some(new_sql),
        };
        let enc = text_encoding_of(pager);
        table_insert(pager, SCHEMA_ROOT, rowid, &new_row.to_record_enc(enc))?;
        bump_schema_cookie(pager)?;

        // Cache last (persistence succeeded). The name/key are unchanged and no
        // auto-index can be added (PK/UNIQUE are rejected above), so `indexes` /
        // `table_indexes` need no update — only the table def is replaced.
        self.tables.insert(key, new_def);
        Ok(())
    }

    /// `ALTER TABLE <old> RENAME TO <new>`: rename the table and cascade the new name
    /// into EVERY dependent index row (explicit + auto) AND every dependent TRIGGER row
    /// whose TARGET is this table. Both cascades are required, not cosmetic: `load`
    /// validates that each index row's target table exists, and `load_trigger_row`
    /// rejects a trigger whose `tbl_name` names no table (and one whose stored
    /// `ON <table>` disagrees with its `tbl_name`) — so a dependent index or trigger
    /// left pointing at the old name would make the database fail to reload. For a
    /// trigger, both the row's `tbl_name` and the `ON <table>` token in its `sql` move
    /// to the new name (they then agree); the trigger's own NAME is unchanged.
    ///
    /// References to the renamed table INSIDE a TRIGGER body / `WHEN`, inside a VIEW's
    /// stored `SELECT`, or in another table's FOREIGN KEY `REFERENCES` clause are a
    /// deliberate, load-tolerant deferral: the load path does not validate any of
    /// those, so leaving them stale does not break reload, and a faithful rewrite there
    /// needs scope-aware identifier resolution (a bare textual swap would clobber a
    /// same-named column, alias, or string literal).
    fn alter_rename_to(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &AlterTable,
        new_name: &str,
    ) -> Result<()> {
        let old = &stmt.name.name;
        let old_key = norm(old);
        if old_key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("table {old} may not be altered")));
        }
        if !self.tables.contains_key(&old_key) {
            return Err(Error::Sql(format!("no such table: {old}")));
        }
        let new_key = norm(new_name);
        if new_key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("object name reserved for internal use: {new_name}")));
        }
        // The target name must be free across the whole schema namespace; this also
        // rejects a rename onto the table's own name (it is still "in use").
        if self.name_in_use(&new_key).is_some() {
            return Err(Error::Sql(format!(
                "there is already another table or index with this name: {new_name}"
            )));
        }

        let root_page = self.tables.get(&old_key).expect("existence checked above").root_page;

        // The table's own page-1 row (exactly one).
        let mut table_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "table" && norm(&row.name) == old_key
        })?;
        if table_rows.len() != 1 {
            return Err(Error::Format(format!(
                "sqlite_schema has {} table rows for {old:?}, expected exactly one",
                table_rows.len()
            )));
        }
        let (table_rowid, table_row) = table_rows.pop().expect("checked len == 1 above");
        let old_name = table_row.name.clone();
        let old_table_sql = table_row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema table row {old:?} has NULL sql"))
        })?;

        // Rewrite the table's own sql and rebuild its def (root preserved) up front, so
        // a bad rewrite aborts before any write.
        let new_table_sql = alter_rewrite::rewrite_create_table_name(old_table_sql, new_name)?;
        let new_def = table_def_from_sql(&new_table_sql, root_page)?;

        // Every index row whose target is the old table (explicit, auto, even an
        // expression index `load` skips) must follow the rename so the file stays
        // consistent and reloads.
        let index_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "index" && norm(&row.tbl_name) == old_key
        })?;

        // Compute the new index page-1 rows and cache updates (no pager mutation here).
        let mut index_persists: Vec<(i64, SchemaRow)> = Vec::with_capacity(index_rows.len());
        // (old cache key, new cache key, rebuilt def) for the indexes that are cached.
        let mut index_cache_updates: Vec<(String, String, IndexDef)> = Vec::new();
        for (rowid, row) in &index_rows {
            match row.sql.as_deref() {
                None => {
                    // Auto-index: its own name embeds the table's original spelling, so
                    // swap the table portion (keeping the same ordinal N) and tbl_name.
                    let old_auto = &row.name;
                    let prefix = format!("sqlite_autoindex_{old_name}_");
                    let suffix = old_auto.strip_prefix(&prefix).ok_or_else(|| {
                        Error::Format(format!(
                            "auto-index {old_auto:?} does not match its table {old_name:?}"
                        ))
                    })?;
                    let new_auto = format!("sqlite_autoindex_{new_name}_{suffix}");
                    index_persists.push((
                        *rowid,
                        SchemaRow {
                            obj_type: "index".to_string(),
                            name: new_auto.clone(),
                            tbl_name: new_name.to_string(),
                            rootpage: row.rootpage,
                            sql: None,
                        },
                    ));
                    let old_ik = norm(old_auto);
                    if self.indexes.contains_key(&old_ik) {
                        // Rebuild the def exactly as `load_auto_index_row` would: from
                        // the renamed table's derived auto-index spec (matched by name).
                        let spec = new_def
                            .auto_indexes
                            .iter()
                            .find(|s| s.name == new_auto && s.emit_row)
                            .ok_or_else(|| {
                                Error::Format(format!(
                                    "renamed table {new_name:?} does not imply auto-index {new_auto:?}"
                                ))
                            })?;
                        let def = IndexDef::from_auto_spec(
                            new_auto.clone(),
                            new_def.name.clone(),
                            spec,
                            checked_root_page(row)?,
                        );
                        index_cache_updates.push((old_ik, norm(&new_auto), def));
                    }
                }
                Some(idx_sql) => {
                    // Explicit index: only its target table changes (its own name stays).
                    let new_idx_sql = alter_rewrite::rewrite_index_table_name(idx_sql, new_name)?;
                    index_persists.push((
                        *rowid,
                        SchemaRow {
                            obj_type: "index".to_string(),
                            name: row.name.clone(),
                            tbl_name: new_name.to_string(),
                            rootpage: row.rootpage,
                            sql: Some(new_idx_sql.clone()),
                        },
                    ));
                    let old_ik = norm(&row.name);
                    // Expression indexes are not cached (load skips them); rebuild a
                    // cache entry only for an index that is actually cached.
                    if self.indexes.contains_key(&old_ik) {
                        let def = index_def_from_sql(
                            &new_idx_sql,
                            &new_def.name,
                            &new_def.columns,
                            checked_root_page(row)?,
                        )?;
                        index_cache_updates.push((old_ik.clone(), old_ik, def));
                    }
                }
            }
        }

        // Every trigger row whose TARGET is the old table must follow the rename too,
        // or `load` fails closed: `load_trigger_row` rejects a trigger whose `tbl_name`
        // names no table AND a trigger whose stored `ON <table>` disagrees with its
        // `tbl_name`. So both the row's `tbl_name` and the `ON <table>` token in its
        // `sql` are rewritten to the new name (they then agree exactly). A trigger's
        // own NAME is unchanged, so — unlike an index — its cache key never moves.
        let trigger_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "trigger" && norm(&row.tbl_name) == old_key
        })?;

        // Compute the new trigger page-1 rows and cache updates (no pager mutation).
        // (name cache-key, rewritten sql) for the in-memory `triggers` cache update.
        let mut trigger_persists: Vec<(i64, SchemaRow)> = Vec::with_capacity(trigger_rows.len());
        let mut trigger_cache_updates: Vec<(String, String)> = Vec::with_capacity(trigger_rows.len());
        for (rowid, row) in &trigger_rows {
            let trg_sql = row.sql.as_deref().ok_or_else(|| {
                Error::Format(format!("sqlite_schema trigger row {:?} has NULL sql", row.name))
            })?;
            let new_trg_sql = alter_rewrite::rewrite_trigger_table_name(trg_sql, new_name)?;
            trigger_cache_updates.push((norm(&row.name), new_trg_sql.clone()));
            trigger_persists.push((
                *rowid,
                SchemaRow {
                    obj_type: "trigger".to_string(),
                    name: row.name.clone(),
                    tbl_name: new_name.to_string(),
                    rootpage: row.rootpage,
                    sql: Some(new_trg_sql),
                },
            ));
        }

        // The ordered index-key list for the renamed table, preserving creation order,
        // each old key mapped to its (possibly renamed) new key.
        let old_index_keys = self.table_indexes.get(&old_key).cloned().unwrap_or_default();
        let new_index_keys: Vec<String> = old_index_keys
            .iter()
            .map(|old_ik| {
                index_cache_updates
                    .iter()
                    .find(|(o, _, _)| o == old_ik)
                    .map(|(_, n, _)| n.clone())
                    .unwrap_or_else(|| old_ik.clone())
            })
            .collect();

        // --- commit phase: overwrite page-1 rows in place, then bump the cookie once. ---
        let new_table_row = SchemaRow {
            obj_type: "table".to_string(),
            name: new_name.to_string(),
            tbl_name: new_name.to_string(),
            rootpage: table_row.rootpage,
            sql: Some(new_table_sql),
        };
        let enc = text_encoding_of(pager);
        table_insert(pager, SCHEMA_ROOT, table_rowid, &new_table_row.to_record_enc(enc))?;
        for (rowid, row) in &index_persists {
            table_insert(pager, SCHEMA_ROOT, *rowid, &row.to_record_enc(enc))?;
        }
        for (rowid, row) in &trigger_persists {
            table_insert(pager, SCHEMA_ROOT, *rowid, &row.to_record_enc(enc))?;
        }
        bump_schema_cookie(pager)?;

        // --- cache phase (last): rekey the table, its indexes, and the index list. ---
        self.tables.remove(&old_key);
        self.tables.insert(new_key.clone(), new_def);
        for old_ik in &old_index_keys {
            self.indexes.remove(old_ik);
        }
        for (_old_ik, new_ik, def) in index_cache_updates {
            self.indexes.insert(new_ik, def);
        }
        self.table_indexes.remove(&old_key);
        if !new_index_keys.is_empty() {
            self.table_indexes.insert(new_key, new_index_keys);
        }
        // Update each dependent trigger's cache entry in place: its NAME (the cache key)
        // is unchanged, so only its target `table` and rewritten `sql` move to the new
        // name — keeping an in-session `DROP TABLE <new>` cascade (which matches on
        // `norm(TriggerDef.table)`) and name-in-use checks correct. `table` is stored in
        // the same spelling `create_trigger`/`load_trigger_row` use (the original name).
        // Every trigger row on `old_key` has a cached def (create/load always cache, and
        // load fails closed on a bad trigger row), so the entry must be present. Unlike
        // an expression index — persisted on page 1 but deliberately uncached, which is
        // why the index cascade guards with `contains_key` — a trigger has no
        // load-skipped category, so this is a genuine assert, not a silent skip.
        //
        // This leans on cache⟷page-1 consistency: a DDL runs against a catalog whose
        // cache reflects the current page 1. True today (the cache changes only through
        // these DDL paths). A future per-statement schema reload (for two-connection WAL)
        // must therefore reload the catalog BEFORE a DDL runs, never after — otherwise a
        // peer's just-committed trigger row could sit on page 1 yet be absent here.
        for (name_key, new_sql) in trigger_cache_updates {
            let def = self.triggers.get_mut(&name_key).expect(
                "SchemaCatalog invariant: a persisted trigger row on the renamed table must have a cached def",
            );
            def.table = new_name.to_string();
            def.sql = new_sql;
        }
        Ok(())
    }

    /// `ALTER TABLE <t> RENAME COLUMN <from> TO <to>`: rename a column within the
    /// table's stored `CREATE TABLE` (its definition AND every reference in a table
    /// constraint / generated-column expression) and cascade the rename into every
    /// dependent index so the database still reloads (`lang_altertable.html` §3). No
    /// row data is rewritten — records store values positionally, not column names —
    /// so this touches only page 1.
    ///
    /// Dependent-index handling mirrors [`alter_rename_to`](Self::alter_rename_to)'s
    /// auto-vs-explicit split, but here the table name is unchanged: an auto-index's
    /// name (`sqlite_autoindex_<table>_<N>`) and its NULL-`sql` page-1 row are
    /// untouched — only its cached column list is rebuilt from the renamed table def —
    /// while an explicit index has the column renamed inside its stored `CREATE INDEX`
    /// text (indexed columns AND a partial-index `WHERE`) and its cached def rebuilt.
    ///
    /// A reference to the renamed COLUMN inside a TRIGGER body / `WHEN`, a VIEW's
    /// stored `SELECT`, or a `CHECK` / FOREIGN KEY expression on ANOTHER object is the
    /// same load-tolerant deferral as [`alter_rename_to`](Self::alter_rename_to)'s: the
    /// load path does not validate those, so leaving them stale does not break reload,
    /// and a faithful rewrite needs scope-aware identifier resolution.
    fn alter_rename_column(
        &mut self,
        pager: &mut dyn Pager,
        stmt: &AlterTable,
        from: &str,
        to: &str,
    ) -> Result<()> {
        let tname = &stmt.name.name;
        let key = norm(tname);
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("table {tname} may not be altered")));
        }
        // Snapshot the target's root + columns before any mutation.
        let (root_page, columns) = {
            let Some(tbl) = self.tables.get(&key) else {
                return Err(Error::Sql(format!("no such table: {tname}")));
            };
            (tbl.root_page, tbl.columns.clone())
        };
        // `from` must name an existing column; `to` must not collide with a DIFFERENT
        // existing column. A case-only change of the very column being renamed
        // (`b` -> `B`) is allowed, so the collision test excludes `from`'s own index.
        let from_idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(from))
            .ok_or_else(|| Error::Sql(format!("no such column: {from}")))?;
        if columns
            .iter()
            .enumerate()
            .any(|(i, c)| i != from_idx && c.name.eq_ignore_ascii_case(to))
        {
            return Err(Error::Sql(format!("duplicate column name: {to}")));
        }

        // The table's own page-1 row (exactly one).
        let mut table_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "table" && norm(&row.name) == key
        })?;
        if table_rows.len() != 1 {
            return Err(Error::Format(format!(
                "sqlite_schema has {} table rows for {tname:?}, expected exactly one",
                table_rows.len()
            )));
        }
        let (table_rowid, table_row) = table_rows.pop().expect("checked len == 1 above");
        let old_sql = table_row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema table row {tname:?} has NULL sql"))
        })?;

        // Rewrite the table sql and rebuild its def (root preserved) up front, so a
        // bad rewrite aborts before any write.
        let new_table_sql = alter_rewrite::rename_column_in_create(old_sql, from, to)?;
        let new_def = table_def_from_sql(&new_table_sql, root_page)?;

        // Every dependent index row (auto or explicit) follows the rename.
        let index_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "index" && norm(&row.tbl_name) == key
        })?;
        // Explicit index rows to overwrite on page 1 (auto-index rows are unchanged —
        // their name embeds the table, not the column, and their sql is NULL).
        let mut index_persists: Vec<(i64, SchemaRow)> = Vec::new();
        // (cache key, rebuilt def) — the key is unchanged (index names do not change),
        // only the def's column list does.
        let mut index_cache_updates: Vec<(String, IndexDef)> = Vec::new();
        for (rowid, row) in &index_rows {
            let ik = norm(&row.name);
            match row.sql.as_deref() {
                None => {
                    // Auto-index: its NULL-`sql` page-1 row does not change. Rebuild the
                    // cached def's columns from the renamed table's derived spec (matched
                    // by the unchanged name), exactly as `load_auto_index_row` would.
                    if self.indexes.contains_key(&ik) {
                        let spec = new_def
                            .auto_indexes
                            .iter()
                            .find(|s| norm(&s.name) == ik && s.emit_row)
                            .ok_or_else(|| {
                                Error::Format(format!(
                                    "renamed table {tname:?} does not imply auto-index {:?}",
                                    row.name
                                ))
                            })?;
                        let def = IndexDef::from_auto_spec(
                            row.name.clone(),
                            new_def.name.clone(),
                            spec,
                            checked_root_page(row)?,
                        );
                        index_cache_updates.push((ik, def));
                    }
                }
                Some(idx_sql) => {
                    // Explicit index: rename the column inside its column list / partial
                    // WHERE (its own name and target table are left untouched).
                    let new_idx_sql = alter_rewrite::rename_column_in_index(idx_sql, from, to)?;
                    index_persists.push((
                        *rowid,
                        SchemaRow {
                            obj_type: "index".to_string(),
                            name: row.name.clone(),
                            tbl_name: row.tbl_name.clone(),
                            rootpage: row.rootpage,
                            sql: Some(new_idx_sql.clone()),
                        },
                    ));
                    // Expression indexes are not cached (load skips them); rebuild a
                    // cache entry only for an index that is actually cached, against the
                    // renamed table's new column names.
                    if self.indexes.contains_key(&ik) {
                        let def = index_def_from_sql(
                            &new_idx_sql,
                            &new_def.name,
                            &new_def.columns,
                            checked_root_page(row)?,
                        )?;
                        index_cache_updates.push((ik, def));
                    }
                }
            }
        }

        // --- commit phase: overwrite page-1 rows in place, then bump the cookie once. ---
        let new_table_row = SchemaRow {
            obj_type: "table".to_string(),
            name: table_row.name.clone(),
            tbl_name: table_row.tbl_name.clone(),
            rootpage: table_row.rootpage,
            sql: Some(new_table_sql),
        };
        let enc = text_encoding_of(pager);
        table_insert(pager, SCHEMA_ROOT, table_rowid, &new_table_row.to_record_enc(enc))?;
        for (rowid, row) in &index_persists {
            table_insert(pager, SCHEMA_ROOT, *rowid, &row.to_record_enc(enc))?;
        }
        bump_schema_cookie(pager)?;

        // --- cache phase (last): replace the table def and the affected index defs. ---
        // Index names and the `table_indexes` ordering are unchanged, so only the defs
        // themselves (their column lists) are replaced, in place, by their same keys.
        self.tables.insert(key, new_def);
        for (ik, def) in index_cache_updates {
            self.indexes.insert(ik, def);
        }
        Ok(())
    }

    /// `ALTER TABLE <t> DROP [COLUMN] <col>` (`lang_altertable.html` §5): remove the
    /// column from the table definition AND purge its stored value from every row.
    ///
    /// Validation (all before any write, failing closed — a schema-only change that
    /// left the data misaligned would corrupt every read): the table may not be a
    /// reserved `sqlite_` table; it must exist; `col` must exist; and the drop must be
    /// permitted. SQLite forbids dropping the table's only column, a PRIMARY KEY /
    /// UNIQUE column, any indexed column (explicit or auto), a column named in a partial
    /// index's `WHERE`, or one referenced by a surviving CHECK / FOREIGN KEY /
    /// generated-column expression. The definition-level restrictions are decided by
    /// [`drop_checks`] (auto-indexes are covered there: one exists only for a UNIQUE /
    /// PRIMARY KEY constraint, which that check rejects); the explicit-index and
    /// partial-`WHERE` restrictions are decided here against the page-1 index rows.
    ///
    /// WITHOUT ROWID tables are refused with an honest error: their rows live in the
    /// primary-key index b-tree, so purging a column means rewriting index-style key
    /// records — a different primitive that is a documented, deferred gap. (Dropping a
    /// PRIMARY KEY column of such a table is already rejected above, so only a non-PK
    /// drop reaches this refusal.)
    ///
    /// Then: rewrite the stored `CREATE TABLE` (remove the column's list item), rewrite
    /// each row's record by dropping the value at the column's position, persist the
    /// schema row + the rewritten data rows in place (rowids and roots preserved, so
    /// every surviving index stays valid), bump the cookie once, and update the cache
    /// last.
    fn alter_drop_column(&mut self, pager: &mut dyn Pager, stmt: &AlterTable, col: &str) -> Result<()> {
        let tname = &stmt.name.name;
        let key = norm(tname);
        if key.starts_with("sqlite_") {
            return Err(Error::Sql(format!("table {tname} may not be altered")));
        }
        // Snapshot the target's root, columns, and rowid-ness before any mutation.
        let (root_page, columns, without_rowid) = {
            let Some(tbl) = self.tables.get(&key) else {
                return Err(Error::Sql(format!("no such table: {tname}")));
            };
            (tbl.root_page, tbl.columns.clone(), tbl.without_rowid)
        };
        // `col` must name an existing column; `drop_idx` is its 0-based SCHEMA ordinal,
        // used for the def-level drop checks and the `CREATE TABLE` text rewrite. It is
        // NOT necessarily the physical record slot: a VIRTUAL generated column takes no
        // storage slot, so a schema ordinal maps to a physical slot only after subtracting
        // the virtual columns at or before it. The physical slot to purge is derived
        // separately (`physical_drop_idx` below) once the drop is known to proceed.
        let drop_idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(col))
            .ok_or_else(|| Error::Sql(format!("no such column: {col}")))?;
        // A table must keep at least one column (spec §5).
        if columns.len() == 1 {
            return Err(Error::Sql(format!(
                "cannot drop column \"{col}\": no other columns exist"
            )));
        }

        // The table's own page-1 row (exactly one) and its stored CREATE TABLE sql.
        let mut table_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "table" && norm(&row.name) == key
        })?;
        if table_rows.len() != 1 {
            return Err(Error::Format(format!(
                "sqlite_schema has {} table rows for {tname:?}, expected exactly one",
                table_rows.len()
            )));
        }
        let (table_rowid, table_row) = table_rows.pop().expect("checked len == 1 above");
        let old_sql = table_row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema table row {tname:?} has NULL sql"))
        })?;

        // Definition-level restrictions (PK / UNIQUE / surviving CHECK / FK / generated).
        let ast = parse(old_sql).map_err(|e| {
            Error::Format(format!("sqlite_schema table sql for {tname:?} failed to parse: {e}"))
        })?;
        let create = match ast.statements.as_slice() {
            [Statement::CreateTable(ct)] => ct,
            _ => {
                return Err(Error::Format(format!(
                    "sqlite_schema table row {tname:?} is not a single CREATE TABLE"
                )))
            }
        };
        if let Some(reason) = drop_checks::create_table_drop_block(create, col, drop_idx) {
            return Err(Error::Sql(reason));
        }

        // Explicit-index restrictions: the column may not be indexed by an explicit
        // index nor named in a partial index's WHERE. Scanning the page-1 index rows is
        // faithful to disk (expression / partial indexes are not all cached). Auto-index
        // rows have NULL sql; the constraint that implies them was already rejected by
        // `create_table_drop_block` if it names `col`, so they need no separate check.
        let index_rows = self.collect_schema_rows(&*pager, |row| {
            row.obj_type == "index" && norm(&row.tbl_name) == key
        })?;
        for (_rowid, row) in &index_rows {
            let Some(idx_sql) = row.sql.as_deref() else { continue };
            let idx_ast = parse(idx_sql).map_err(|e| {
                Error::Format(format!("sqlite_schema index sql {:?} failed to parse: {e}", row.name))
            })?;
            let ci = match idx_ast.statements.as_slice() {
                [Statement::CreateIndex(ci)] => ci,
                _ => {
                    return Err(Error::Format(format!(
                        "sqlite_schema index row {:?} is not a single CREATE INDEX",
                        row.name
                    )))
                }
            };
            let indexed = ci.columns.iter().any(|ic| match &ic.target {
                IndexedColumnTarget::Name(n) => n.eq_ignore_ascii_case(col),
                IndexedColumnTarget::Expr(e) => drop_checks::expr_references_column(e, col),
            });
            let in_where = ci
                .where_clause
                .as_ref()
                .is_some_and(|w| drop_checks::expr_references_column(w, col));
            if indexed || in_where {
                return Err(Error::Sql(format!(
                    "cannot drop column \"{col}\": it is used by index \"{}\"",
                    ci.name.name
                )));
            }
        }

        // Rows of a WITHOUT ROWID table live in the PK-index b-tree keyed by their PK,
        // not as rowid records, so purging a column needs a different rewrite primitive.
        // Refuse honestly rather than corrupt the key records (documented deferral).
        if without_rowid {
            return Err(Error::Sql(
                "DROP COLUMN on WITHOUT ROWID tables is not yet supported".into(),
            ));
        }

        // Rewrite the stored CREATE TABLE (remove the column's list item) and rebuild
        // the def up front, so a bad rewrite aborts before any write. `rowid_alias` is
        // re-derived by the builder, which correctly handles dropping a column before an
        // INTEGER PRIMARY KEY.
        let new_table_sql = alter_rewrite::remove_column_from_create(old_sql, col)?;
        let new_def = table_def_from_sql(&new_table_sql, root_page)?;

        // --- commit phase: overwrite the schema row, purge the dropped column's value
        // from every data row in place (same rowid + root, so secondary indexes stay
        // valid), then bump the cookie once. The data rewrite streams one row at a time
        // ([`alter_data::rewrite_drop_column_data`], O(1) extra memory — never buffers
        // the table); the whole thing runs under the caller's write txn, so a mid-rewrite
        // failure rolls back both the schema row and the data. ---
        let new_table_row = SchemaRow {
            obj_type: "table".to_string(),
            name: table_row.name.clone(),
            tbl_name: table_row.tbl_name.clone(),
            rootpage: table_row.rootpage,
            sql: Some(new_table_sql),
        };
        let enc = text_encoding_of(pager);
        table_insert(pager, SCHEMA_ROOT, table_rowid, &new_table_row.to_record_enc(enc))?;
        // Map the schema ordinal to the PHYSICAL record slot. A stored record holds only
        // the non-virtual columns in table-definition order (a VIRTUAL generated column is
        // never stored — `gencol.html`), so the physical slot of the dropped column is the
        // count of non-virtual columns strictly before it. Dropping a VIRTUAL column itself
        // touches NO stored bytes, so the data rewrite is skipped entirely (only the schema
        // text / def change). Without this, a schema ordinal used as a physical slot would
        // delete the wrong stored value whenever a virtual column precedes/is the target.
        if !columns[drop_idx].is_virtual_generated() {
            let physical_drop_idx =
                columns[..drop_idx].iter().filter(|c| !c.is_virtual_generated()).count();
            alter_data::rewrite_drop_column_data(pager, root_page, physical_drop_idx, enc)?;
        }
        bump_schema_cookie(pager)?;

        // --- cache phase (last): replace the table def. The index set is unchanged (we
        // rejected dropping any indexed column), so `indexes` / `table_indexes` need no
        // update — their cached column lists still name only surviving columns. ---
        self.tables.insert(key, new_def);
        Ok(())
    }

    /// Rebuild a `TableDef` from one decoded `type='table'` row and cache it. A table
    /// row whose `sql` is NULL or does not parse to a single `CREATE TABLE`, whose
    /// `CREATE TABLE` is a deferred-gap shape (`AS SELECT`), whose `rootpage` is not a
    /// valid page id, or whose name is already held by another object (or a duplicate
    /// table row), is a corrupt schema and fails closed (`Error::Format`).
    fn load_table_row(&mut self, row: &SchemaRow) -> Result<()> {
        let sql = row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema table row {:?} has NULL sql", row.name))
        })?;
        let ast = parse(sql).map_err(|e| {
            Error::Format(format!("sqlite_schema sql for {:?} failed to parse: {e}", row.name))
        })?;
        let create = match ast.statements.as_slice() {
            [Statement::CreateTable(create)] => create,
            _ => {
                return Err(Error::Format(format!(
                    "sqlite_schema sql for {:?} is not a single CREATE TABLE",
                    row.name
                )));
            }
        };
        // Map the builder's `Sql` error to `Format` so the load path speaks one
        // corrupt-schema vocabulary, symmetric with load_index_row. Every build failure
        // reachable here is a hand-tampered/foreign image real sqlite never persists: the
        // `CREATE TABLE ... AS SELECT` deferred gap (sqlite stores AS SELECT as an explicit
        // column list), and any create-time structural rule `table_def_from_ast` enforces
        // (e.g. a PK-less WITHOUT ROWID table, a duplicate column, a STRICT type violation).
        // Each fails closed rather than caching an invalid def.
        let def = table_def_from_ast(create, checked_root_page(row)?).map_err(|e| {
            Error::Format(format!("sqlite_schema table row {:?} is invalid: {e}", row.name))
        })?;
        // Shared namespace: a stored table whose name is also an index/view/trigger's —
        // or a duplicate table row — is corrupt schema, the load-side mirror of the
        // create guards. Whichever of two colliding rows loads SECOND trips this, so with
        // every loader consulting `name_in_use` detection is order-independent across all
        // four object kinds (a table/view collision is caught here or in `load_view_row`,
        // whichever loads later in pass 1; an index/trigger collision when its pass-2 row
        // loads).
        let key = norm(&def.name);
        if let Some(kind) = self.name_in_use(&key) {
            return Err(Error::Format(format!(
                "sqlite_schema table {:?} collides with a {kind} of the same name",
                row.name
            )));
        }
        self.tables.insert(key, def);
        Ok(())
    }

    /// Rebuild an `IndexDef` from one decoded `type='index'` row and cache it. Must
    /// run after every table row is loaded, since it resolves the index's target
    /// table from the cache.
    ///
    /// A NULL-`sql` row is an auto-created index (`sqlite_autoindex_*`): its indexed
    /// columns are not stored, so they are reconstructed from the target table's derived
    /// `auto_indexes` specs — the SAME `builder::auto_indexes_for` derivation the create
    /// path used — in [`load_auto_index_row`](Self::load_auto_index_row), so the two
    /// paths can never disagree.
    ///
    /// An expression index (a `Some`-`sql` `CREATE INDEX` over an expression,
    /// `lang_createindex.html` §1.2) reconstructs like any other: `index_def_from_ast`
    /// reparses the stored `sql` and captures its key expressions (`IndexDef::key_exprs`)
    /// with the stored `root_page`, so it comes back on load exactly as create built it.
    ///
    /// Anything else that is wrong — unparseable `sql`, not a single CREATE INDEX, a
    /// name already held by another object (shared namespace), a missing target table,
    /// an unknown column, an auto-index the table's constraints do not imply, or an
    /// out-of-range `rootpage` — is a corrupt/inconsistent schema and fails closed.
    fn load_index_row(&mut self, row: &SchemaRow) -> Result<()> {
        let Some(sql) = row.sql.as_deref() else {
            // Auto-created index (NULL sql): reconstruct its columns from the table's
            // derived specs, so the create and load paths can never disagree.
            return self.load_auto_index_row(row);
        };
        let ast = parse(sql).map_err(|e| {
            Error::Format(format!("sqlite_schema sql for {:?} failed to parse: {e}", row.name))
        })?;
        let ci = match ast.statements.as_slice() {
            [Statement::CreateIndex(ci)] => ci,
            _ => {
                return Err(Error::Format(format!(
                    "sqlite_schema sql for {:?} is not a single CREATE INDEX",
                    row.name
                )));
            }
        };
        // Shared namespace: a stored index whose name is also a table/view/trigger's (or
        // another index's) is corrupt schema (real sqlite never writes such an image),
        // the load-side mirror of the create guards. `name_in_use` consults all four
        // caches — safe now, since pass 1 loaded every table AND view before pass 2 runs,
        // so an index colliding with either fails closed here rather than silently landing
        // in two caches.
        let key = norm(&row.name);
        if let Some(kind) = self.name_in_use(&key) {
            return Err(Error::Format(format!(
                "sqlite_schema index {:?} collides with a {kind} of the same name",
                row.name
            )));
        }
        let root = checked_root_page(row)?;
        let (table_key, def) = {
            let Some(tbl) = self.tables.get(resolve_table_key(&norm(&ci.table))) else {
                return Err(Error::Format(format!(
                    "sqlite_schema index {:?} references missing table {:?}",
                    row.name, ci.table
                )));
            };
            // A stored index that names a column its table lacks is corrupt schema
            // (create_index validated columns, so only a tampered DB reaches here).
            // Map the builder's `Sql` error to `Format` so the load path speaks one
            // corrupt-schema vocabulary. For an expression key column the builder does no
            // existence check (the planner validates its refs when binding), so the only
            // failure left here is an ordinary named column the table lacks.
            let def = index_def_from_ast(ci, &tbl.name, &tbl.columns, root).map_err(|e| {
                Error::Format(format!(
                    "sqlite_schema index {:?} is inconsistent with table {:?}: {e}",
                    row.name, ci.table
                ))
            })?;
            (norm(&tbl.name), def)
        };
        // Same cache writes as create_index, keyed by the index's folded name and
        // appended to the table's list so reload preserves creation order.
        self.indexes.insert(key.clone(), def);
        self.table_indexes.entry(table_key).or_default().push(key);
        Ok(())
    }

    /// Reconstruct an auto-created index (a NULL-`sql` `sqlite_autoindex_*` row) and
    /// cache it. Its indexed columns are NOT stored on disk, so they are re-derived from
    /// the target table's `auto_indexes` specs — the SAME `builder::auto_indexes_for`
    /// derivation the create path used — matched by the row's (folded) name. Both paths
    /// reading the one spec list is what keeps them from ever drifting.
    ///
    /// Fails closed (`Error::Format`) on an inconsistent schema: a name that also names
    /// another object (shared namespace), a missing target table, a `sqlite_autoindex_*`
    /// name the table's constraints do not imply, or one that matches only a reserved
    /// WITHOUT ROWID PRIMARY KEY spec (which owns no row). Real sqlite re-derives and
    /// checks these on open, so any mismatch is a corrupt or foreign image. The rootpage
    /// is validated with the same `checked_root_page` helper the explicit index and table
    /// paths use.
    fn load_auto_index_row(&mut self, row: &SchemaRow) -> Result<()> {
        let root = checked_root_page(row)?;
        let key = norm(&row.name);
        // Shared namespace, the same guard load_index_row applies: an auto-index row whose
        // name is also a table/view/trigger's (or another index's) is corrupt schema (real
        // sqlite never writes such an image). `name_in_use` consults all four caches — safe
        // here, since pass 1 loaded every table and view before pass 2 runs.
        if let Some(kind) = self.name_in_use(&key) {
            return Err(Error::Format(format!(
                "sqlite_schema auto-index {:?} collides with a {kind} of the same name",
                row.name
            )));
        }
        let (table_key, def) = {
            let Some(tbl) = self.tables.get(resolve_table_key(&norm(&row.tbl_name))) else {
                return Err(Error::Format(format!(
                    "sqlite_schema auto-index {:?} references missing table {:?}",
                    row.name, row.tbl_name
                )));
            };
            // The create path wrote this row's name from one EMITTED spec, so a real
            // image has exactly one spec whose folded name matches AND that spec owns a
            // row (`emit_row == true`). No match, or a match to a reserved WITHOUT ROWID
            // PRIMARY KEY spec (`emit_row == false`, for which real sqlite writes no row),
            // is a corrupt/foreign image — fail closed, symmetric with the explicit path.
            let Some(spec) = tbl.auto_indexes.iter().find(|s| norm(&s.name) == key) else {
                return Err(Error::Format(format!(
                    "sqlite_schema auto-index {:?} is not implied by table {:?}'s constraints",
                    row.name, row.tbl_name
                )));
            };
            if !spec.emit_row {
                return Err(Error::Format(format!(
                    "sqlite_schema auto-index {:?} matches table {:?}'s reserved PRIMARY KEY, which owns no row",
                    row.name, row.tbl_name
                )));
            }
            let def = IndexDef::from_auto_spec(
                row.name.clone(),
                tbl.name.clone(),
                spec,
                root,
            );
            (norm(&tbl.name), def)
        };
        // Same cache writes as the explicit-index path, keyed by the index's folded
        // name and appended to the table's list so reload preserves creation order.
        self.indexes.insert(key.clone(), def);
        self.table_indexes.entry(table_key).or_default().push(key);
        Ok(())
    }

    /// Rebuild a [`ViewDef`] from one decoded `type='view'` row and cache it. Runs in
    /// pass 1: a view depends only on its own `sql`, not on any table being loaded.
    ///
    /// Fails closed (`Error::Format`) on an inconsistent schema, symmetric with the
    /// index load paths: a non-zero `rootpage` (a view owns no b-tree — so this uses a
    /// direct `== 0` check, NOT `checked_root_page`, which demands a real page id >= 2),
    /// a NULL `sql`, `sql` that does not parse to exactly one `CREATE VIEW`, a parsed
    /// name that disagrees with the row's `name`, or a name already held by another
    /// object (shared namespace). Real sqlite re-derives and checks these on open, so
    /// any mismatch is a corrupt or foreign image.
    fn load_view_row(&mut self, row: &SchemaRow) -> Result<()> {
        if row.rootpage != 0 {
            return Err(Error::Format(format!(
                "sqlite_schema view {:?} has a non-zero rootpage {} (a view owns no b-tree)",
                row.name, row.rootpage
            )));
        }
        let sql = row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema view row {:?} has NULL sql", row.name))
        })?;
        let ast = parse(sql).map_err(|e| {
            Error::Format(format!("sqlite_schema sql for {:?} failed to parse: {e}", row.name))
        })?;
        let cv = match ast.statements.as_slice() {
            [Statement::CreateView(cv)] => cv,
            _ => {
                return Err(Error::Format(format!(
                    "sqlite_schema sql for {:?} is not a single CREATE VIEW",
                    row.name
                )));
            }
        };
        // The stored `sql`'s view name must match the row's `name` column (a corrupt
        // image otherwise), case-insensitively like every identifier match.
        if !cv.name.name.eq_ignore_ascii_case(&row.name) {
            return Err(Error::Format(format!(
                "sqlite_schema view row name {:?} disagrees with its CREATE VIEW name {:?}",
                row.name, cv.name.name
            )));
        }
        // Shared namespace: a stored view whose name is also another object's is corrupt
        // schema (real sqlite never writes such an image), the load-side mirror of the
        // create guards. Pass 1 has loaded every table and prior view by now, so a
        // view/table or view/view collision fails closed here; an index or trigger
        // sharing the name is caught symmetrically when its pass-2 row loads (those
        // loaders also consult `name_in_use`).
        let key = norm(&row.name);
        if let Some(kind) = self.name_in_use(&key) {
            return Err(Error::Format(format!(
                "sqlite_schema view {:?} collides with a {kind} of the same name",
                row.name
            )));
        }
        self.views.insert(key, ViewDef { name: row.name.clone(), sql: sql.to_string() });
        Ok(())
    }

    /// Rebuild a [`TriggerDef`] from one decoded `type='trigger'` row and cache it.
    /// Must run after every table AND view row is loaded (pass 2), since it validates the
    /// trigger's target (a table for BEFORE/AFTER, a view for INSTEAD OF) against the cache.
    ///
    /// Fails closed (`Error::Format`) on a non-zero `rootpage` (a trigger owns no
    /// b-tree — direct `== 0` check, not `checked_root_page`), a NULL `sql`, `sql` that
    /// does not parse to exactly one `CREATE TRIGGER`, a parsed name disagreeing with
    /// the row's `name`, a `tbl_name` disagreeing with the parsed target, a target that
    /// matches neither a table nor a view, a stored timing that contradicts the target
    /// kind (INSTEAD OF on a table, or BEFORE/AFTER on a view — a combo real sqlite never
    /// writes), or a name already held by another object (shared namespace).
    fn load_trigger_row(&mut self, row: &SchemaRow) -> Result<()> {
        if row.rootpage != 0 {
            return Err(Error::Format(format!(
                "sqlite_schema trigger {:?} has a non-zero rootpage {} (a trigger owns no b-tree)",
                row.name, row.rootpage
            )));
        }
        let sql = row.sql.as_deref().ok_or_else(|| {
            Error::Format(format!("sqlite_schema trigger row {:?} has NULL sql", row.name))
        })?;
        let ast = parse(sql).map_err(|e| {
            Error::Format(format!("sqlite_schema sql for {:?} failed to parse: {e}", row.name))
        })?;
        let ct = match ast.statements.as_slice() {
            [Statement::CreateTrigger(ct)] => ct,
            _ => {
                return Err(Error::Format(format!(
                    "sqlite_schema sql for {:?} is not a single CREATE TRIGGER",
                    row.name
                )));
            }
        };
        if !ct.name.name.eq_ignore_ascii_case(&row.name) {
            return Err(Error::Format(format!(
                "sqlite_schema trigger row name {:?} disagrees with its CREATE TRIGGER name {:?}",
                row.name, ct.name.name
            )));
        }
        // The row's `tbl_name` is the trigger's target's BARE name; the parsed CREATE
        // TRIGGER's `ON [db.]<table>` must name the same table (a corrupt image otherwise).
        if !ct.table.name.eq_ignore_ascii_case(&row.tbl_name) {
            return Err(Error::Format(format!(
                "sqlite_schema trigger {:?} tbl_name {:?} disagrees with its CREATE TRIGGER target {:?}",
                row.name, row.tbl_name, ct.table.name
            )));
        }
        // Resolve the target against THIS store (pass 2 runs after pass 1 has loaded all
        // tables and views). A name is only ever one kind (shared namespace), so these are
        // exclusive for a valid image.
        let target_key = norm(&row.tbl_name);
        let target_is_table = self.tables.get(resolve_table_key(&target_key)).is_some();
        let target_is_view = self.views.contains_key(&target_key);
        debug_assert!(
            !(target_is_table && target_is_view),
            "shared-namespace invariant: {:?} is both a table and a view",
            row.tbl_name
        );
        let is_instead_of = matches!(ct.timing, Some(TriggerTiming::InsteadOf));
        let target = if target_is_table || target_is_view {
            // Same-store target: the stored timing must match the target kind
            // (lang_createtrigger.html §3): INSTEAD OF only on a view, BEFORE/AFTER (and an
            // omitted timing) only on a table. Real sqlite only ever writes a consistent
            // combo, so a mismatch is a corrupt image — fail closed rather than cache a
            // trigger that could never fire through the ordinary DML path.
            if target_is_view && !is_instead_of {
                return Err(Error::Format(format!(
                    "sqlite_schema trigger {:?} on view {:?} is not INSTEAD OF (BEFORE/AFTER triggers are table-only)",
                    row.name, row.tbl_name
                )));
            }
            if target_is_table && is_instead_of {
                return Err(Error::Format(format!(
                    "sqlite_schema INSTEAD OF trigger {:?} targets table {:?} (INSTEAD OF is view-only)",
                    row.name, row.tbl_name
                )));
            }
            TriggerTargetDb::SameStore
        } else if ct.temp {
            // A cross-namespace TEMP trigger (lang_createtrigger.html §7): its target lives in
            // another store this single-store load cannot see. Reconstruct the binding from the
            // ON-target's qualifier AS WRITTEN, never a numeric index — fire-time discovery
            // re-resolves it against the connection's live registry, so this load carries no
            // registry and needs none. A QUALIFIED `ON aux.u` reconstructs exactly (the alias
            // resolves at fire time, including after DETACH remaps indices); an UNQUALIFIED
            // `ON u` becomes ForeignUnqualified, re-resolved by search order at fire time (the
            // doc's "reattach to a same-named table on schema change"). Reached ONLY on the
            // in-memory TEMP store via the engine's rollback-resync reload — real sqlite never
            // persists a TEMP trigger to a file, so a main/attached row is never `ct.temp`, and
            // a non-temp missing target below stays the corrupt-image error it always was.
            match ct.table.schema.as_deref() {
                Some(schema) => TriggerTargetDb::ForeignSchema(schema.to_string()),
                None => TriggerTargetDb::ForeignUnqualified,
            }
        } else {
            return Err(Error::Format(format!(
                "sqlite_schema trigger {:?} references missing table {:?}",
                row.name, row.tbl_name
            )));
        };
        // Shared namespace: reject a name already held by a table/index/view/trigger.
        let key = norm(&row.name);
        if let Some(kind) = self.name_in_use(&key) {
            return Err(Error::Format(format!(
                "sqlite_schema trigger {:?} collides with a {kind} of the same name",
                row.name
            )));
        }
        self.triggers.insert(
            key,
            TriggerDef {
                name: row.name.clone(),
                table: row.tbl_name.clone(),
                sql: sql.to_string(),
                target,
            },
        );
        Ok(())
    }
}

/// Validate and narrow a stored `rootpage` to a real b-tree root page id. A table or
/// index root always fits `u32` and is >= 2 (page 1 is the schema itself). A negative
/// or oversized value would otherwise truncate silently under an `as` cast and
/// surface far later as a bad page read, so fail closed here at the boundary that
/// admitted it.
fn checked_root_page(row: &SchemaRow) -> Result<PageId> {
    u32::try_from(row.rootpage).ok().filter(|&r| r >= 2).ok_or_else(|| {
        Error::Format(format!(
            "sqlite_schema row {:?} has an out-of-range rootpage {}",
            row.name, row.rootpage
        ))
    })
}

/// The canonical name of the schema table.
const SQLITE_SCHEMA_NAME: &str = "sqlite_schema";

/// Fold the documented `sqlite_schema` aliases to the built-in's key. `key` must
/// already be `norm`-folded. Returns a `&str` borrowing `key` for the non-alias
/// path and a static string for the aliases, so a lookup allocates nothing beyond
/// the one `norm`.
fn resolve_table_key(key: &str) -> &str {
    match key {
        "sqlite_master" | "sqlite_temp_master" | "sqlite_temp_schema" => SQLITE_SCHEMA_NAME,
        other => other,
    }
}

/// The built-in `sqlite_schema` table def: five text/integer columns rooted at
/// page 1. It carries no constraint flags and is not a rowid alias.
fn sqlite_schema_def() -> TableDef {
    fn col(name: &str, declared_type: &str) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: Some(declared_type.to_string()),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: None,
            default: None,
            default_value: None,
            generated: None,
        }
    }
    TableDef {
        name: SQLITE_SCHEMA_NAME.to_string(),
        columns: vec![
            col("type", "TEXT"),
            col("name", "TEXT"),
            col("tbl_name", "TEXT"),
            col("rootpage", "INTEGER"),
            col("sql", "TEXT"),
        ],
        root_page: SCHEMA_ROOT,
        without_rowid: false,
        rowid_alias: None,
        auto_indexes: Vec::new(),
        checks: Vec::new(),
        foreign_keys: Vec::new(),
        autoincrement: false,
        primary_key: Vec::new(),
    }
}

/// Increment the schema cookie in page 1's database header (fileformat2 §1.3),
/// leaving the rest of the page (the just-inserted b-tree content) untouched. Read
/// the full post-insert page, overlay the re-encoded 100-byte header, and write it
/// back. `file_change_counter`/`database_size_pages` are the storage layer's to
/// maintain and are deliberately left alone here.
fn bump_schema_cookie(pager: &mut dyn Pager) -> Result<()> {
    let mut page = pager.read_page(SCHEMA_ROOT)?.to_vec();
    let header_bytes: [u8; HEADER_SIZE] = page
        .get(0..HEADER_SIZE)
        .ok_or_else(|| Error::Format("page 1 is shorter than the database header".into()))?
        .try_into()
        .expect("a slice of length HEADER_SIZE converts to [u8; HEADER_SIZE]");
    let mut header = DatabaseHeader::read(&header_bytes)?;
    // wrapping_add matches SQLite's counter wrap and avoids a debug overflow panic;
    // reaching u32::MAX schema changes is not physically possible in practice.
    header.schema_cookie = header.schema_cookie.wrapping_add(1);
    let mut new_header = [0u8; HEADER_SIZE];
    header.write(&mut new_header);
    page[0..HEADER_SIZE].copy_from_slice(&new_header);
    pager.write_page(SCHEMA_ROOT, &page)
}

/// Enforce SQLite's `ALTER TABLE ADD COLUMN` restrictions (`lang_altertable.html`
/// §4) against a parsed new column and the target table's existing columns. Checks
/// run in SQLite's own order (`alter.c`), so the reported error matches when several
/// apply: a duplicate column name is detected first (as the column is added), then
/// PRIMARY KEY, UNIQUE, a NOT NULL without a non-NULL default (a literal `NULL`
/// default counts as none), a non-constant default (`CURRENT_TIME`/`CURRENT_DATE`/
/// `CURRENT_TIMESTAMP` or a parenthesized expression), and finally a `GENERATED
/// ALWAYS ... STORED` column (VIRTUAL generated columns are allowed).
///
/// A `REFERENCES` (foreign key) column is NOT rejected here: foreign keys are not
/// modelled and are off by default, so — matching real sqlite with
/// `foreign_keys=OFF` — adding one is permitted.
fn check_add_column_allowed(coldef: &SqlColumnDef, existing: &[ColumnDef]) -> Result<()> {
    // Duplicate column name: sqlite detects this first, while adding the column.
    if existing.iter().any(|c| c.name.eq_ignore_ascii_case(&coldef.name)) {
        return Err(Error::Sql(format!("duplicate column name: {}", coldef.name)));
    }

    let mut has_primary_key = false;
    let mut has_unique = false;
    let mut has_not_null = false;
    let mut has_stored_generated = false;
    let mut default: Option<&DefaultValue> = None;
    for cons in &coldef.constraints {
        match &cons.kind {
            ColumnConstraintKind::PrimaryKey { .. } => has_primary_key = true,
            ColumnConstraintKind::Unique { .. } => has_unique = true,
            ColumnConstraintKind::NotNull { .. } => has_not_null = true,
            ColumnConstraintKind::Default(d) => default = Some(d),
            ColumnConstraintKind::Generated { stored, .. } => has_stored_generated |= *stored,
            ColumnConstraintKind::Null { .. }
            | ColumnConstraintKind::Check(_)
            | ColumnConstraintKind::Collate(_)
            | ColumnConstraintKind::ForeignKey(_) => {}
        }
    }

    if has_primary_key {
        return Err(Error::Sql("Cannot add a PRIMARY KEY column".into()));
    }
    if has_unique {
        return Err(Error::Sql("Cannot add a UNIQUE column".into()));
    }
    // A literal NULL default is treated as "no usable default" for this check, exactly
    // as sqlite folds `DEFAULT NULL` to no default expression.
    let default_is_effectively_null =
        matches!(default, None | Some(DefaultValue::Literal(Literal::Null)));
    if has_not_null && default_is_effectively_null {
        return Err(Error::Sql("Cannot add a NOT NULL column with default value NULL".into()));
    }
    if let Some(d) = default {
        let non_constant = match d {
            DefaultValue::Expr(..) => true,
            DefaultValue::Literal(lit) => matches!(
                lit,
                Literal::CurrentTime | Literal::CurrentDate | Literal::CurrentTimestamp
            ),
        };
        if non_constant {
            return Err(Error::Sql("Cannot add a column with non-constant default".into()));
        }
    }
    if has_stored_generated {
        return Err(Error::Sql("cannot add a STORED column".into()));
    }
    Ok(())
}

/// Parse a known-good internal `CREATE TABLE` and rebuild its [`TableDef`] at the
/// given root, so the cached def equals exactly what a fresh `load` from page 1 would
/// produce (the one-builder invariant `create_table` relies on). Three internal callers
/// pass sql they control: `ALTER TABLE` after it rewrites a table's stored `sql`,
/// `ensure_sqlite_sequence` with the verbatim `CREATE TABLE sqlite_sequence(name,seq)`,
/// and `ensure_sqlite_stat1` with the verbatim `CREATE TABLE sqlite_stat1(tbl,idx,stat)`.
/// A parse/build failure therefore means an internal invariant broke (a bad rewrite,
/// or a malformed constant) — surfaced as an error rather than silently cached.
fn table_def_from_sql(sql: &str, root: PageId) -> Result<TableDef> {
    let ast = parse(sql)?;
    match ast.statements.as_slice() {
        [Statement::CreateTable(ct)] => table_def_from_ast(ct, root),
        _ => Err(Error::Sql(format!(
            "ALTER TABLE produced sql that is not a single CREATE TABLE: {sql:?}"
        ))),
    }
}

/// Parse an internally-rewritten `CREATE INDEX` and rebuild its [`IndexDef`] against
/// the (renamed) target table's name and columns. Same one-builder rationale as
/// [`table_def_from_sql`]; only ever called for a cached (non-expression) index.
fn index_def_from_sql(
    sql: &str,
    table_name: &str,
    columns: &[ColumnDef],
    root: PageId,
) -> Result<IndexDef> {
    let ast = parse(sql)?;
    match ast.statements.as_slice() {
        [Statement::CreateIndex(ci)] => index_def_from_ast(ci, table_name, columns, root),
        _ => Err(Error::Sql(format!(
            "ALTER TABLE produced sql that is not a single CREATE INDEX: {sql:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_btree::init_database;
    use minisqlite_fileformat::{decode_record, encode_record};
    use minisqlite_pager::MemPager;
    use minisqlite_types::Value;

    /// A pager with a freshly formatted, empty database (page 1 = empty schema).
    fn fresh_pager() -> MemPager {
        let mut pager = MemPager::new(4096);
        init_database(&mut pager).unwrap();
        pager
    }

    /// Parse `sql` (one CREATE TABLE) and create the table through the real path.
    fn create(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_create(cat, pager, sql).unwrap();
    }

    /// Like `create` but returns the `Result`, for asserting error / namespace paths.
    fn try_create(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TABLE: {sql}");
        };
        cat.create_table(pager, stmt, sql)
    }

    /// Number of rows physically stored in the page-1 schema b-tree.
    fn page1_row_count(pager: &MemPager) -> usize {
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT).unwrap();
        let mut count = 0;
        let mut positioned = cursor.first().unwrap();
        while positioned {
            count += 1;
            positioned = cursor.next().unwrap();
        }
        count
    }

    fn read_schema_cookie(pager: &MemPager) -> u32 {
        let page = pager.read_page(SCHEMA_ROOT).unwrap();
        let header_bytes: [u8; HEADER_SIZE] = page[0..HEADER_SIZE].try_into().unwrap();
        DatabaseHeader::read(&header_bytes).unwrap().schema_cookie
    }

    #[test]
    fn foreign_keys_and_generated_columns_survive_create_then_reload() {
        // The captured FK + generated-column facts must be recovered on reopen. Load
        // re-parses the stored verbatim `sql` through the SAME `table_def_from_ast` builder
        // the create path used, so a fresh catalog loaded from page 1 sees identical
        // metadata — proving the round-trip with no load-path-specific code.
        use crate::ReferentialAction;
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a INTEGER, g INTEGER AS (a+1) STORED, \
             FOREIGN KEY(a) REFERENCES p(x) ON DELETE CASCADE)",
        );

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let t = reloaded.table("t").unwrap().expect("table t reloads from page 1");

        assert_eq!(t.foreign_keys.len(), 1, "the FK survives reload");
        let fk = &t.foreign_keys[0];
        assert_eq!(fk.child_columns, ["a"]);
        assert_eq!(fk.parent_table, "p");
        assert_eq!(fk.parent_columns, ["x"]);
        assert_eq!(fk.on_delete, ReferentialAction::Cascade);
        assert_eq!(fk.on_update, ReferentialAction::NoAction);

        let g = t.columns[1].generated.as_ref().expect("generated column survives reload");
        assert!(g.stored, "the STORED flag survives reload");
    }

    // --- persistence round trip ------------------------------------------------

    #[test]
    fn create_persists_and_reload_rebuilds_defs() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let sql_t = "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c)";
        let sql_u = "CREATE TABLE u(x)";
        create(&mut cat, &mut pager, sql_t);
        create(&mut cat, &mut pager, sql_u);

        // Freshly created def is present and correct.
        let (t_root, t_alias, t_cols) = {
            let t = cat.table("t").unwrap().unwrap();
            let cols: Vec<String> = t.columns.iter().map(|c| c.name.clone()).collect();
            (t.root_page, t.rowid_alias, cols)
        };
        assert_eq!(t_cols, ["a", "b", "c"]);
        assert!(t_root >= 2, "user table root must be >= 2, got {t_root}");
        assert_eq!(t_alias, Some(0));

        // A fresh store that only loads from page 1 rebuilds identical defs.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();

        let t2 = reloaded.table("t").unwrap().unwrap();
        assert_eq!(t2.name, "t");
        assert_eq!(t2.root_page, t_root, "root page survives the round trip");
        assert_eq!(t2.rowid_alias, Some(0));
        let typed: Vec<(String, Option<String>)> =
            t2.columns.iter().map(|c| (c.name.clone(), c.declared_type.clone())).collect();
        assert_eq!(
            typed,
            vec![
                ("a".to_string(), Some("INTEGER".to_string())),
                ("b".to_string(), Some("TEXT".to_string())),
                ("c".to_string(), None),
            ]
        );

        let u2 = reloaded.table("u").unwrap().unwrap();
        assert_eq!(u2.columns.len(), 1);
        assert_eq!(u2.columns[0].name, "x");

        // Case-insensitive lookup still holds after a reload.
        assert!(reloaded.table("T").unwrap().is_some());
        assert!(reloaded.table("U").unwrap().is_some());
    }

    #[test]
    fn create_writes_exact_schema_row_bytes() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let sql = "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c)";
        create(&mut cat, &mut pager, sql);
        let root = cat.table("t").unwrap().unwrap().root_page as i64;

        let mut cursor = TableCursor::open(&pager, SCHEMA_ROOT).unwrap();
        assert!(cursor.first().unwrap(), "one schema row was written");
        let values = decode_record(&cursor.payload().unwrap());
        assert_eq!(values.len(), 5);
        assert!(matches!(&values[0], Value::Text(s) if s == "table"));
        assert!(matches!(&values[1], Value::Text(s) if s == "t"));
        assert!(matches!(&values[2], Value::Text(s) if s == "t"));
        assert!(matches!(&values[3], Value::Integer(i) if *i == root));
        assert!(matches!(&values[4], Value::Text(s) if s == sql), "sql stored verbatim");
    }

    #[test]
    fn create_bumps_schema_cookie_by_one() {
        let mut pager = fresh_pager();
        let before = read_schema_cookie(&pager);
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        assert_eq!(read_schema_cookie(&pager), before + 1);
        create(&mut cat, &mut pager, "CREATE TABLE u(a)");
        assert_eq!(read_schema_cookie(&pager), before + 2);

        // A CREATE TABLE that writes SEVERAL auto-index rows is still ONE schema change:
        // the cookie advances by exactly 1, not once per auto-index row.
        create(&mut cat, &mut pager, "CREATE TABLE v(a UNIQUE, b UNIQUE, c UNIQUE)");
        assert_eq!(
            read_schema_cookie(&pager),
            before + 3,
            "three auto-indexes in one CREATE TABLE, still a single cookie bump"
        );
    }

    #[test]
    fn if_not_exists_is_idempotent_without_a_second_row() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        assert_eq!(page1_row_count(&pager), 1);

        // IF NOT EXISTS on an existing table: Ok, and no new page-1 row.
        let sql = "CREATE TABLE IF NOT EXISTS t(a)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        cat.create_table(&mut pager, stmt, sql).unwrap();
        assert_eq!(page1_row_count(&pager), 1, "IF NOT EXISTS must not add a row");

        // A second distinct table does add a row.
        create(&mut cat, &mut pager, "CREATE TABLE u(x)");
        assert_eq!(page1_row_count(&pager), 2);
    }

    #[test]
    fn duplicate_table_without_if_not_exists_errors_and_keeps_first() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");

        // Case-insensitive duplicate (as `T`) is rejected, first def untouched.
        let sql = "CREATE TABLE T(z)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "expected Sql error, got {err:?}");
        assert_eq!(page1_row_count(&pager), 1);
        let cols: Vec<String> =
            cat.table("t").unwrap().unwrap().columns.iter().map(|c| c.name.clone()).collect();
        assert_eq!(cols, ["a"], "the rejected duplicate did not overwrite the original");
    }

    #[test]
    fn create_table_rejects_sqlite_prefix() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let sql = "CREATE TABLE sqlite_foo(x)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "expected Sql error, got {err:?}");
        // Nothing was persisted.
        assert_eq!(page1_row_count(&pager), 0);
        assert!(cat.table("sqlite_foo").unwrap().is_none());
    }

    #[test]
    fn create_table_rejects_sqlite_prefix_case_insensitively() {
        // The reservation check folds the name first, so a mixed-case `SQLite_Foo` is
        // rejected exactly like `sqlite_foo`. Guards against a regression to a
        // non-normalized (case-sensitive) prefix test.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let sql = "CREATE TABLE SQLite_Foo(x)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "expected Sql error, got {err:?}");
        assert_eq!(page1_row_count(&pager), 0);
    }

    #[test]
    fn create_table_as_select_is_a_reported_gap_through_create_table() {
        // AS SELECT reaches create_table only after a b-tree root is already
        // allocated; it must surface the deferred-gap error (not fabricate a table),
        // and cache nothing. The stranded root is reclaimed by the caller's txn on
        // rollback (create_table itself opens no transaction).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let sql = "CREATE TABLE t AS SELECT * FROM u";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "expected Sql error, got {err:?}");
        assert!(cat.table("t").unwrap().is_none(), "no table cached on the error path");
        assert_eq!(page1_row_count(&pager), 0, "no schema row written on the error path");
    }

    #[test]
    fn create_table_colliding_with_an_index_is_rejected() {
        // Tables and indexes share one namespace: after CREATE INDEX i, a CREATE TABLE
        // i is rejected (real sqlite: "there is already an index named i") even though
        // `i` is not in the tables map. Standing guard for the sibling of
        // create_index's table-collision branch — this class was dormant until indexes
        // populated the cache, so it must not silently reappear.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        let rows_before = page1_row_count(&pager);

        let sql = "CREATE TABLE i(x)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "table colliding with index -> Sql, got {err:?}");

        // No shadow object, and no page-1 row written for the rejected table.
        assert!(cat.table("i").unwrap().is_none(), "a table must not shadow index `i`");
        assert!(cat.index("i").unwrap().is_some(), "the original index is untouched");
        assert_eq!(page1_row_count(&pager), rows_before, "rejected CREATE TABLE wrote no row");
    }

    #[test]
    fn create_table_if_not_exists_still_rejects_an_index_name() {
        // IF NOT EXISTS suppresses only a same-type (table) collision; a name already
        // held by an INDEX is still a hard error.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        let rows_before = page1_row_count(&pager);

        let sql = "CREATE TABLE IF NOT EXISTS i(x)";
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else { panic!() };
        let err = cat.create_table(&mut pager, stmt, sql).unwrap_err();
        assert!(
            matches!(err, Error::Sql(_)),
            "IF NOT EXISTS must not suppress an index collision, got {err:?}"
        );
        assert!(cat.table("i").unwrap().is_none(), "no table created");
        assert_eq!(page1_row_count(&pager), rows_before, "rejected CREATE TABLE wrote no row");
    }

    // --- built-in sqlite_schema / aliases -------------------------------------

    #[test]
    fn builtin_schema_table_and_aliases_resolve_to_page_one() {
        let cat = SchemaCatalog::new();
        for name in ["sqlite_schema", "sqlite_master", "SQLITE_MASTER", "sqlite_temp_schema"] {
            let def = cat.table(name).unwrap().unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(def.root_page, 1, "{name} roots at page 1");
            let cols: Vec<&str> = def.columns.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(cols, ["type", "name", "tbl_name", "rootpage", "sql"]);
        }
    }

    #[test]
    fn builtin_schema_is_not_persisted_and_survives_reload() {
        // A fresh, empty (but formatted) database has zero page-1 rows, yet the
        // built-in schema table is visible — it is in-memory only, never a row.
        let pager = fresh_pager();
        assert_eq!(page1_row_count(&pager), 0);
        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.table("sqlite_master").unwrap().is_some());
    }

    // --- create_index ----------------------------------------------------------

    /// Parse `sql` (one CREATE INDEX) and create the index through the real path.
    fn create_index(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_create_index(cat, pager, sql).unwrap();
    }

    /// Like `create_index` but returns the `Result`, for asserting error paths.
    fn try_create_index(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::CreateIndex(stmt) = &ast.statements[0] else {
            panic!("not a CREATE INDEX: {sql}");
        };
        cat.create_index(pager, stmt, sql)
    }

    /// The decoded 5-value records of every `type='index'` row physically on page 1,
    /// in rowid order.
    fn index_rows(pager: &MemPager) -> Vec<Vec<Value>> {
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT).unwrap();
        let mut out = Vec::new();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            let values = decode_record(&cursor.payload().unwrap());
            if matches!(&values[0], Value::Text(s) if s == "index") {
                out.push(values);
            }
            positioned = cursor.next().unwrap();
        }
        out
    }

    #[test]
    fn create_index_persists_one_index_row_and_bumps_cookie() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        let cookie_before = read_schema_cookie(&pager);

        let sql = "CREATE INDEX i ON t(a)";
        create_index(&mut cat, &mut pager, sql);

        let root = cat.index("i").unwrap().unwrap().root_page as i64;
        let rows = index_rows(&pager);
        assert_eq!(rows.len(), 1, "exactly one type='index' row");
        let v = &rows[0];
        assert_eq!(v.len(), 5);
        assert!(matches!(&v[0], Value::Text(s) if s == "index"));
        assert!(matches!(&v[1], Value::Text(s) if s == "i"));
        assert!(matches!(&v[2], Value::Text(s) if s == "t"), "tbl_name is the indexed table");
        assert!(matches!(&v[3], Value::Integer(r) if *r == root && *r >= 2), "root >= 2");
        assert!(matches!(&v[4], Value::Text(s) if s == sql), "sql stored verbatim");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "cookie bumped by 1");
    }

    #[test]
    fn create_index_caches_def_and_lists_in_creation_order() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i1 ON t(a)");
        create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX i2 ON t(b, a)");

        let i1 = cat.index("i1").unwrap().unwrap();
        assert_eq!(i1.name, "i1");
        assert_eq!(i1.table, "t");
        assert_eq!(i1.columns, ["a"]);
        assert!(!i1.unique);
        assert!(!i1.partial);
        assert!(i1.root_page >= 2);

        let i2 = cat.index("i2").unwrap().unwrap();
        assert_eq!(i2.columns, ["b", "a"]);
        assert!(i2.unique);

        let listed: Vec<String> =
            cat.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["i1", "i2"], "indexes_on lists in creation order");
    }

    #[test]
    fn create_index_error_cases_are_reported_and_persist_nothing() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");

        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX i ON nope(a)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "absent table -> Sql, got {e:?}");
        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(zzz)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "unknown column -> Sql, got {e:?}");
        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX sqlite_i ON t(a)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "reserved name -> Sql, got {e:?}");
        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX t ON t(a)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "name collides with a table -> Sql, got {e:?}");
        assert_eq!(index_rows(&pager).len(), 0, "no failed attempt wrote a page-1 row");

        // A real index, then a duplicate without IF NOT EXISTS: rejected, no 2nd row.
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        assert_eq!(index_rows(&pager).len(), 1);
        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "duplicate -> Sql, got {e:?}");
        assert_eq!(index_rows(&pager).len(), 1, "the rejected duplicate wrote no row");

        // IF NOT EXISTS on the existing index is a silent no-op: Ok, still one row.
        create_index(&mut cat, &mut pager, "CREATE INDEX IF NOT EXISTS i ON t(a)");
        assert_eq!(index_rows(&pager).len(), 1, "IF NOT EXISTS added no row");
    }

    #[test]
    fn unique_index_round_trips() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX i ON t(a)");
        assert!(cat.index("i").unwrap().unwrap().unique);

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.index("i").unwrap().unwrap().unique, "unique survives reload");
    }

    #[test]
    fn partial_index_persists_sql_and_round_trips() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        let sql = "CREATE INDEX i ON t(a) WHERE a>0";
        create_index(&mut cat, &mut pager, sql);
        assert!(cat.index("i").unwrap().unwrap().partial);
        assert!(
            matches!(&index_rows(&pager)[0][4], Value::Text(s) if s == sql),
            "partial-index sql stored verbatim"
        );

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.index("i").unwrap().unwrap().partial, "partial survives reload");
    }

    #[test]
    fn expression_index_is_created_persisted_and_round_trips_through_load() {
        // A GENUINE expression index (`lang_createindex.html` §1.2) is created, persisted,
        // and RECONSTRUCTED on load — the same round-trip a COLLATE/DESC index gets. The
        // cached def carries the empty-name sentinel plus a captured key expression, and a
        // fresh catalog rebuilt from page 1 alone brings back the same shape and root page.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        let sql = "CREATE INDEX i ON t(a+b)";
        create_index(&mut cat, &mut pager, sql);

        let created = cat.index("i").unwrap().expect("expression index is created");
        assert_eq!(created.columns, [""], "an expression key stores the empty-name sentinel");
        assert_eq!(created.key_exprs.len(), 1, "one key_expr slot");
        assert!(created.key_exprs[0].is_some(), "the a+b key expression is captured");
        assert!(created.root_page >= 2, "index has a real b-tree root");
        let root = created.root_page;
        // The verbatim CREATE INDEX text is stored so it can be reparsed on load.
        assert!(matches!(&index_rows(&pager)[0][4], Value::Text(s) if s == sql), "sql stored verbatim");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let idx = reloaded.index("i").unwrap().expect("expression index reconstructs on load");
        assert_eq!(idx.columns, [""]);
        assert_eq!(idx.key_exprs.len(), 1);
        assert!(idx.key_exprs[0].is_some(), "key expression survives reload");
        assert_eq!(idx.root_page, root, "reload preserves the stored root page");
    }

    #[test]
    fn index_lookup_is_case_insensitive() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        assert!(cat.index("I").unwrap().is_some(), "created as `i`, found by `I`");
    }

    // --- per-column COLLATE / DESC in explicit indexes (the fixed gap) ----------

    #[test]
    fn create_index_collate_over_column_succeeds_and_caches_collation() {
        // The core fix: `CREATE INDEX i ON t(x COLLATE NOCASE)` no longer errors (the
        // parser folds the COLLATE into an expr target, which used to be refused). It is
        // created, persisted, and its cached def carries the collation override.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x)");
        let sql = "CREATE INDEX i ON t(x COLLATE NOCASE)";
        create_index(&mut cat, &mut pager, sql);

        let idx = cat.index("i").unwrap().unwrap();
        assert_eq!(idx.columns, ["x"]);
        assert_eq!(idx.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(!idx.key_columns[0].descending);
        // The verbatim sql (with COLLATE) is stored, so it round-trips as text too.
        assert!(matches!(&index_rows(&pager)[0][4], Value::Text(s) if s == sql));
    }

    #[test]
    fn collate_and_desc_unique_index_round_trips_through_load() {
        // Build a UNIQUE index carrying a per-column COLLATE and DESC on a MemPager,
        // drop the cache, reload from page 1 alone, and confirm the rebuilt def still
        // carries a->NOCASE and b->descending with unique=true (it is NOT skipped).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX i ON t(a COLLATE NOCASE, b DESC)");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let idx = reloaded.index("i").unwrap().expect("COLLATE/DESC index is not skipped on load");
        assert_eq!(idx.columns, ["a", "b"]);
        assert_eq!(idx.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(!idx.key_columns[0].descending);
        assert_eq!(idx.key_columns[1].collation, None);
        assert!(idx.key_columns[1].descending, "b DESC survives reload");
        assert!(idx.unique, "UNIQUE survives reload");
    }

    #[test]
    fn load_reconstructs_both_collate_and_genuine_expression_indexes() {
        // Both a `COLLATE`-over-column index and a GENUINE expression index reconstruct on
        // load from their stored verbatim `sql`. Inject both directly onto page 1 and
        // confirm each comes back: the COLLATE index as a named key column carrying its
        // collation, the expression index with an empty-name sentinel and a captured
        // key expression (`lang_createindex.html` §1.2).
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        let collate_row = SchemaRow {
            obj_type: "index".into(),
            name: "ic".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX ic ON t(a COLLATE NOCASE)".into()),
        };
        let expr_row = SchemaRow {
            obj_type: "index".into(),
            name: "ie".into(),
            tbl_name: "t".into(),
            rootpage: 4,
            sql: Some("CREATE INDEX ie ON t(a+1)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &collate_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 3, &expr_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        let ic = cat.index("ic").unwrap().expect("COLLATE index reconstructed on load");
        assert_eq!(ic.key_columns[0].collation.as_deref(), Some("NOCASE"));
        let ie = cat.index("ie").unwrap().expect("expression index reconstructed on load");
        assert_eq!(ie.columns, [""], "expression key stores the empty-name sentinel");
        assert!(ie.key_exprs[0].is_some(), "expression key captured on load");
        assert_eq!(ie.root_page, 4, "stored root page preserved");
    }

    // --- auto-indexes on CREATE TABLE (UNIQUE / PRIMARY KEY) --------------------

    #[test]
    fn create_table_unique_writes_null_sql_autoindex_row() {
        // CREATE TABLE t(a UNIQUE, b): page 1 has the table row + one auto-index row.
        // The index row is NULL-sql, named sqlite_autoindex_t_1, on column a; the cached
        // def is UNIQUE and non-partial and appears in indexes_on.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE, b)");

        assert_eq!(page1_row_count(&pager), 2, "table row + one auto-index row");
        let rows = index_rows(&pager);
        assert_eq!(rows.len(), 1, "exactly one type='index' row");
        let v = &rows[0];
        assert_eq!(v.len(), 5);
        assert!(matches!(&v[0], Value::Text(s) if s == "index"));
        assert!(matches!(&v[1], Value::Text(s) if s == "sqlite_autoindex_t_1"));
        assert!(matches!(&v[2], Value::Text(s) if s == "t"), "tbl_name is the table");
        assert!(matches!(&v[3], Value::Integer(r) if *r >= 2), "root >= 2");
        assert!(matches!(&v[4], Value::Null), "auto-index stores NULL sql");

        let idx = cat.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(idx.columns, ["a"]);
        assert!(idx.unique);
        assert!(!idx.partial);
        let listed: Vec<String> =
            cat.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["sqlite_autoindex_t_1"]);
    }

    #[test]
    fn create_table_integer_primary_key_writes_no_autoindex_row() {
        // An INTEGER PRIMARY KEY is the rowid alias: no separate index, no auto-index
        // row, no N consumed. Page 1 holds only the table row.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
        assert_eq!(page1_row_count(&pager), 1, "only the table row, no auto-index");
        assert_eq!(index_rows(&pager).len(), 0, "no auto-index row");
        assert!(cat.indexes_on("t").unwrap().is_empty());
    }

    #[test]
    fn create_table_unique_collate_writes_autoindex_row_and_round_trips() {
        // The table-level UNIQUE gap: `UNIQUE(x COLLATE NOCASE)` used to be DROPPED (no
        // sqlite_autoindex row, so its uniqueness silently vanished — a complete-but-wrong
        // divergence from the .db real sqlite writes). It must now emit the auto-index
        // row, carry the collation on the cached def, and round-trip through load.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x TEXT, UNIQUE(x COLLATE NOCASE))");

        // Page 1: the table row + one auto-index row (the row was dropped entirely before).
        assert_eq!(page1_row_count(&pager), 2, "table row + the COLLATE auto-index row");
        let rows = index_rows(&pager);
        assert_eq!(rows.len(), 1, "the COLLATE UNIQUE emitted its auto-index row");
        assert!(matches!(&rows[0][1], Value::Text(s) if s == "sqlite_autoindex_t_1"));
        assert!(matches!(&rows[0][4], Value::Null), "auto-index stores NULL sql");

        let idx = cat.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(idx.columns, ["x"]);
        assert_eq!(idx.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(idx.unique);

        // Round-trip: reload from page 1 and confirm the auto-index (with its collation)
        // is re-derived, not lost.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let r =
            reloaded.index("sqlite_autoindex_t_1").unwrap().expect("auto-index re-derived on load");
        assert_eq!(r.key_columns[0].collation.as_deref(), Some("NOCASE"));
        assert!(r.unique);
    }

    // --- AUTOINCREMENT: sqlite_sequence auto-creation (autoinc.html §3) ---------

    /// Scan page 1 and return the decoded five-column schema row whose `name`
    /// (column 1) equals `name`, or `None`. Used to assert the exact on-disk bytes of
    /// the auto-created `sqlite_sequence` row.
    fn find_schema_row(pager: &MemPager, name: &str) -> Option<Vec<Value>> {
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT).unwrap();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            let values = decode_record(&cursor.payload().unwrap());
            if matches!(&values[1], Value::Text(s) if s == name) {
                return Some(values);
            }
            positioned = cursor.next().unwrap();
        }
        None
    }

    #[test]
    fn autoincrement_table_auto_creates_sqlite_sequence() {
        // CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT) auto-creates the ordinary
        // internal table sqlite_sequence(name,seq). Prove both the cached def and the
        // exact persisted page-1 row (type/name/tbl_name/rootpage/sql).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");

        // Cached def: two untyped columns name, seq.
        let seq = cat.table("sqlite_sequence").unwrap().expect("sqlite_sequence table exists");
        let col_names: Vec<String> = seq.columns.iter().map(|c| c.name.clone()).collect();
        assert_eq!(col_names, ["name", "seq"]);
        assert!(seq.root_page >= 2, "sqlite_sequence root must be a real page, got {}", seq.root_page);
        assert!(!seq.without_rowid);
        assert_eq!(seq.rowid_alias, None);

        // Page 1: exactly the two table rows (t, then sqlite_sequence); no auto-index.
        assert_eq!(page1_row_count(&pager), 2, "t + sqlite_sequence, nothing else");
        let row = find_schema_row(&pager, "sqlite_sequence").expect("sqlite_sequence row on page 1");
        assert_eq!(row.len(), 5);
        assert!(matches!(&row[0], Value::Text(s) if s == "table"), "type='table'");
        assert!(matches!(&row[1], Value::Text(s) if s == "sqlite_sequence"));
        assert!(matches!(&row[2], Value::Text(s) if s == "sqlite_sequence"), "tbl_name=name");
        assert!(matches!(&row[3], Value::Integer(r) if *r >= 2), "positive rootpage");
        assert!(
            matches!(&row[4], Value::Text(s) if s == "CREATE TABLE sqlite_sequence(name,seq)"),
            "verbatim sql real sqlite stores, got {:?}",
            row[4]
        );
    }

    #[test]
    fn autoincrement_with_auto_index_orders_sequence_row_last() {
        // A table with BOTH an autoincrement PK and a UNIQUE column writes three page-1
        // rows. sqlite_sequence is written LAST — after the table's own auto-index — so
        // the b-tree (rowid) order is: t, sqlite_autoindex_t_1, sqlite_sequence. This
        // pins that deliberate insert order so a reordering regression is caught. NOTE:
        // whether this exact interleaving matches real sqlite for the combined case is a
        // separate on-disk-fidelity to-do (the no-auto-index case `t`,
        // `sqlite_sequence` above is the verified-correct order).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT, y UNIQUE)",
        );

        let names: Vec<String> = {
            let mut cursor = TableCursor::open(&pager, SCHEMA_ROOT).unwrap();
            let mut out = Vec::new();
            let mut positioned = cursor.first().unwrap();
            while positioned {
                let values = decode_record(&cursor.payload().unwrap());
                match &values[1] {
                    Value::Text(s) => out.push(s.clone()),
                    other => panic!("schema row name must be text, got {other:?}"),
                }
                positioned = cursor.next().unwrap();
            }
            out
        };
        assert_eq!(names, ["t", "sqlite_autoindex_t_1", "sqlite_sequence"]);
    }

    #[test]
    fn sqlite_sequence_round_trips_through_load_without_special_casing() {
        // Build on a MemPager, then rebuild a fresh catalog from page 1 alone. Because
        // the row stores an ordinary `CREATE TABLE sqlite_sequence(name,seq)` sql, the
        // generic table-load path reconstructs it with no sqlite_sequence-specific code.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();

        let seq = reloaded.table("sqlite_sequence").unwrap().expect("rebuilt on load");
        let col_names: Vec<String> = seq.columns.iter().map(|c| c.name.clone()).collect();
        assert_eq!(col_names, ["name", "seq"], "load re-derives (name, seq) from the stored sql");
        // Case-insensitive lookup holds for the internal table too.
        assert!(reloaded.table("SQLITE_SEQUENCE").unwrap().is_some());
        // The user table came back as well.
        assert!(reloaded.table("t").unwrap().is_some());
    }

    #[test]
    fn second_autoincrement_table_does_not_duplicate_sqlite_sequence() {
        // Idempotence: a second AUTOINCREMENT table must NOT write a second
        // sqlite_sequence row, and each CREATE TABLE bumps the schema cookie EXACTLY
        // once (the auto-create shares the statement's single bump).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();

        let c0 = read_schema_cookie(&pager);
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");
        assert_eq!(read_schema_cookie(&pager), c0 + 1, "first CREATE TABLE: one bump");
        // t + sqlite_sequence.
        assert_eq!(page1_row_count(&pager), 2);

        create(&mut cat, &mut pager, "CREATE TABLE u(y INTEGER PRIMARY KEY AUTOINCREMENT)");
        assert_eq!(read_schema_cookie(&pager), c0 + 2, "second CREATE TABLE: still one bump");
        // t, sqlite_sequence, u — NOT a second sqlite_sequence.
        assert_eq!(page1_row_count(&pager), 3, "no duplicate sqlite_sequence row");

        let seq_rows = {
            let mut cursor = TableCursor::open(&pager, SCHEMA_ROOT).unwrap();
            let mut n = 0;
            let mut positioned = cursor.first().unwrap();
            while positioned {
                let values = decode_record(&cursor.payload().unwrap());
                if matches!(&values[1], Value::Text(s) if s == "sqlite_sequence") {
                    n += 1;
                }
                positioned = cursor.next().unwrap();
            }
            n
        };
        assert_eq!(seq_rows, 1, "exactly one sqlite_sequence row after two AUTOINCREMENT tables");
    }

    #[test]
    fn non_autoincrement_create_table_does_not_create_sqlite_sequence() {
        // A plain INTEGER PRIMARY KEY (no AUTOINCREMENT) and an ordinary table must not
        // conjure sqlite_sequence.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY, y)");
        create(&mut cat, &mut pager, "CREATE TABLE u(a, b)");
        assert!(cat.table("sqlite_sequence").unwrap().is_none(), "no sqlite_sequence in cache");
        assert!(find_schema_row(&pager, "sqlite_sequence").is_none(), "no sqlite_sequence on page 1");
    }

    #[test]
    fn drop_table_sqlite_sequence_is_rejected() {
        // sqlite_sequence is `sqlite_`-prefixed, so DROP refuses it (reserved) even
        // though it is an ordinary table — matching real sqlite. It stays present.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");
        assert!(cat.table("sqlite_sequence").unwrap().is_some());

        let err = try_drop(&mut cat, &mut pager, "DROP TABLE sqlite_sequence").unwrap_err();
        assert!(
            matches!(&err, Error::Sql(m) if m == "table sqlite_sequence may not be dropped"),
            "reserved DROP -> exact reserved-name error, got {err:?}"
        );
        assert!(cat.table("sqlite_sequence").unwrap().is_some(), "still present after refused DROP");
    }

    #[test]
    fn create_table_without_rowid_writes_unique_autoindex_not_pk() {
        // WITHOUT ROWID: the UNIQUE(a) emits a real auto-index row, the PRIMARY KEY does
        // NOT (the table's own b-tree IS the PK index). Numbering: UNIQUE(a)=N1 (row),
        // PK(b)=N2 (reserved, no row), so the ONLY row is sqlite_autoindex_t_1.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a UNIQUE, b TEXT PRIMARY KEY) WITHOUT ROWID",
        );

        let names: Vec<String> = index_rows(&pager)
            .iter()
            .map(|v| match &v[1] {
                Value::Text(s) => s.clone(),
                other => panic!("index name must be text, got {other:?}"),
            })
            .collect();
        assert_eq!(names, ["sqlite_autoindex_t_1"], "only the UNIQUE's row, not the PK's");
        let idx = cat.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(idx.columns, ["a"]);
        assert!(
            cat.index("sqlite_autoindex_t_2").unwrap().is_none(),
            "the WITHOUT ROWID PK reserves N=2 but owns no index row"
        );
    }

    #[test]
    fn without_rowid_integer_pk_autoindex_names_and_round_trips() {
        // Regression for the WITHOUT ROWID INTEGER PRIMARY KEY case (schematab.html: the
        // name is never allocated for an INTEGER PK, even WITHOUT ROWID). So for
        // `t(a INTEGER PRIMARY KEY, b UNIQUE) WITHOUT ROWID` the UNIQUE(b) auto-index is
        // sqlite_autoindex_t_1 on [b] — NOT _2 on [a]. Proven end-to-end: the persisted
        // NULL-sql row, the cached def, and the reload reconstruction must all agree.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a INTEGER PRIMARY KEY, b UNIQUE) WITHOUT ROWID",
        );

        let names: Vec<String> = index_rows(&pager)
            .iter()
            .map(|v| match &v[1] {
                Value::Text(s) => s.clone(),
                other => panic!("index name must be text, got {other:?}"),
            })
            .collect();
        assert_eq!(names, ["sqlite_autoindex_t_1"], "only the UNIQUE's row, named _1 not _2");
        let created = cat.index("sqlite_autoindex_t_1").unwrap().unwrap().clone();
        assert_eq!(created.columns, ["b"], "the auto-index covers the UNIQUE column b, not a");
        assert!(
            cat.index("sqlite_autoindex_t_2").unwrap().is_none(),
            "no _2: the INTEGER PRIMARY KEY reserved no N"
        );

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let loaded = reloaded.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(loaded.columns, created.columns, "reload re-derives columns [b], not [a]");
        assert_eq!(loaded.root_page, created.root_page);
        assert!(loaded.unique && !loaded.partial);
    }

    #[test]
    fn autoindexes_round_trip_through_load() {
        // A UNIQUE column plus a composite PRIMARY KEY: two auto-indexes. Snapshot every
        // field, reload from page 1, and assert the reconstructed defs match exactly, in
        // creation order via indexes_on.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE, b, c, PRIMARY KEY(b, c))");

        type Snap = (String, String, Vec<String>, bool, bool, u32);
        let snap = |cat: &SchemaCatalog, n: &str| -> Snap {
            let d = cat.index(n).unwrap().unwrap();
            (d.name.clone(), d.table.clone(), d.columns.clone(), d.unique, d.partial, d.root_page)
        };
        let want_names: Vec<String> =
            cat.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(
            want_names,
            ["sqlite_autoindex_t_1", "sqlite_autoindex_t_2"],
            "UNIQUE(a)=N1, composite PK(b,c)=N2, both real auto-indexes"
        );
        let want: Vec<Snap> = want_names.iter().map(|n| snap(&cat, n)).collect();
        assert_eq!(want[0].2, ["a"], "N1 indexes the UNIQUE column");
        assert_eq!(want[1].2, ["b", "c"], "N2 indexes the composite PK columns in order");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let got_names: Vec<String> =
            reloaded.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        let got: Vec<Snap> = got_names.iter().map(|n| snap(&reloaded, n)).collect();
        assert_eq!(got_names, want_names, "same auto-indexes, same creation order after reload");
        assert_eq!(got, want, "every reconstructed IndexDef field matches the created one");
    }

    #[test]
    fn without_rowid_autoindexes_round_trip_through_load() {
        // WITHOUT ROWID end-to-end: only the UNIQUE emits a row; on reload it is
        // reconstructed, and the PK's emit_row=false spec (no row on disk) causes no
        // trouble — nothing references it.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a UNIQUE, b TEXT PRIMARY KEY) WITHOUT ROWID",
        );
        let created = cat.index("sqlite_autoindex_t_1").unwrap().unwrap().clone();

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let loaded = reloaded.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(loaded.columns, created.columns, "UNIQUE(a) columns survive the round trip");
        assert_eq!(loaded.root_page, created.root_page, "root page survives");
        assert!(loaded.unique && !loaded.partial);
        let listed: Vec<String> =
            reloaded.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["sqlite_autoindex_t_1"], "only the UNIQUE auto-index; the PK owns none");
    }

    // --- migrated case-insensitivity / borrow behaviour ------------------------

    #[test]
    fn table_lookup_is_case_insensitive_and_keeps_original_case() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        for probe in ["t", "T"] {
            let got = cat.table(probe).unwrap();
            assert!(got.is_some(), "lookup {probe:?} should find table `t`");
            assert_eq!(got.unwrap().name, "t", "original spelling is preserved");
        }
    }

    #[test]
    fn original_case_is_reported_not_the_probe() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE MyTable(x)");
        let got = cat.table("mytable").unwrap().unwrap();
        assert_eq!(got.name, "MyTable");
    }

    #[test]
    fn absent_lookups_return_none_not_error() {
        let cat = SchemaCatalog::new();
        assert!(cat.table("ghost").unwrap().is_none());
        assert!(cat.index("ghost").unwrap().is_none());
    }

    #[test]
    fn indexes_on_is_empty_for_table_without_indexes_and_absent_table() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        assert!(cat.indexes_on("t").unwrap().is_empty());
        assert!(cat.indexes_on("T").unwrap().is_empty(), "case-insensitive, still empty");
        assert!(cat.indexes_on("ghost").unwrap().is_empty());
    }

    #[test]
    fn tables_enumerates_user_tables_and_excludes_internal_sqlite_tables() {
        // `tables()` is the enumeration seam FK enforcement walks to find child tables.
        // It must return every USER table and never an internal `sqlite_*` one: the
        // built-in `sqlite_schema` is always present, and AUTOINCREMENT auto-creates the
        // internal `sqlite_sequence` — neither may leak into the result.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create(&mut cat, &mut pager, "CREATE TABLE u(b)");
        create(&mut cat, &mut pager, "CREATE TABLE s(x INTEGER PRIMARY KEY AUTOINCREMENT)");

        // Both internal tables really exist in the store...
        assert!(cat.table("sqlite_schema").unwrap().is_some());
        assert!(cat.table("sqlite_sequence").unwrap().is_some(), "AUTOINCREMENT made it");

        // ...but the enumeration returns only the user tables (order is unspecified).
        let mut names: Vec<String> = cat.tables().unwrap().iter().map(|t| t.name.clone()).collect();
        names.sort();
        assert_eq!(names, ["s", "t", "u"], "every user table, no sqlite_* internals");
        assert!(
            !names.iter().any(|n| n.starts_with("sqlite_")),
            "no internal sqlite_* table leaked into tables(): {names:?}"
        );
    }

    #[test]
    fn tables_excludes_any_sqlite_prefixed_internal_not_just_a_known_name() {
        // Guard the exclusion as a PREFIX rule, not a hardcoded {sqlite_schema,
        // sqlite_sequence} list: load a schema image carrying an internal `sqlite_stat1`
        // (what ANALYZE writes) beside user tables. It folds to a `sqlite_`-prefixed key
        // like every internal, so `tables()` must drop it too. This uses the load path
        // because `create_table` refuses a `sqlite_`-prefixed name, yet a real `.db`
        // legitimately stores such a row — so a filter keyed to the two create-path
        // internals alone would wrongly leak it, and this test catches that.
        let mut pager = fresh_pager();
        let rows = [
            SchemaRow {
                obj_type: "table".into(),
                name: "t".into(),
                tbl_name: "t".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE t(a)".into()),
            },
            SchemaRow {
                obj_type: "table".into(),
                name: "u".into(),
                tbl_name: "u".into(),
                rootpage: 3,
                sql: Some("CREATE TABLE u(b)".into()),
            },
            SchemaRow {
                obj_type: "table".into(),
                name: "sqlite_stat1".into(),
                tbl_name: "sqlite_stat1".into(),
                rootpage: 4,
                sql: Some("CREATE TABLE sqlite_stat1(tbl,idx,stat)".into()),
            },
        ];
        for (i, row) in rows.iter().enumerate() {
            table_insert(&mut pager, SCHEMA_ROOT, (i + 1) as i64, &row.to_record()).unwrap();
        }
        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();

        // The internal table really loaded (non-vacuous), yet a name that is neither
        // sqlite_schema nor sqlite_sequence is still excluded by the prefix rule.
        assert!(cat.table("sqlite_stat1").unwrap().is_some(), "internal table loaded");
        let mut names: Vec<String> = cat.tables().unwrap().iter().map(|t| t.name.clone()).collect();
        names.sort();
        assert_eq!(names, ["t", "u"], "sqlite_stat1 excluded by prefix, not by known-name list");
    }

    #[test]
    fn default_matches_new() {
        // Default is new(): both carry the built-in schema table and nothing else.
        let cat = SchemaCatalog::default();
        assert!(cat.table("sqlite_schema").unwrap().is_some());
        assert!(cat.table("t").unwrap().is_none());
    }

    // --- load edge cases -------------------------------------------------------

    #[test]
    fn load_loads_index_and_view() {
        // A named-column index row loads into the cache; a view row now ALSO loads (it
        // is no longer skipped) and takes its place in the shared namespace, even though
        // there is no public `view()` accessor. `table("v")` stays None — a view is not
        // a table — so the proof the view loaded is the namespace rejecting a table `v`.
        let mut pager = fresh_pager();
        let rows = [
            SchemaRow {
                obj_type: "table".into(),
                name: "t".into(),
                tbl_name: "t".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE t(a)".into()),
            },
            SchemaRow {
                obj_type: "index".into(),
                name: "i".into(),
                tbl_name: "t".into(),
                rootpage: 3,
                sql: Some("CREATE INDEX i ON t(a)".into()),
            },
            SchemaRow {
                obj_type: "view".into(),
                name: "v".into(),
                tbl_name: "v".into(),
                rootpage: 0,
                sql: Some("CREATE VIEW v AS SELECT 1".into()),
            },
        ];
        for (i, row) in rows.iter().enumerate() {
            table_insert(&mut pager, SCHEMA_ROOT, (i + 1) as i64, &row.to_record()).unwrap();
        }

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.table("t").unwrap().is_some());
        let i = cat.index("i").unwrap().expect("named-column index loads");
        assert_eq!(i.columns, ["a"]);
        assert_eq!(i.root_page, 3);
        // The view is not a table, but IS now in the namespace: a table named `v` errors.
        assert!(cat.table("v").unwrap().is_none(), "a view is not a table");
        let err = try_create(&mut cat, &mut pager, "CREATE TABLE v(x)").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "the loaded view occupies the namespace, got {err:?}");
    }

    #[test]
    fn full_round_trip_reloads_indexes_in_order() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i1 ON t(a)");
        create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX i2 ON t(c, b)");

        // Snapshot every IndexDef field before the reload.
        type Snap = (String, String, Vec<String>, bool, bool, u32);
        let snap = |cat: &SchemaCatalog, n: &str| -> Snap {
            let d = cat.index(n).unwrap().unwrap();
            (d.name.clone(), d.table.clone(), d.columns.clone(), d.unique, d.partial, d.root_page)
        };
        let want = [snap(&cat, "i1"), snap(&cat, "i2")];

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let got = [snap(&reloaded, "i1"), snap(&reloaded, "i2")];
        assert_eq!(got, want, "every IndexDef field survives the round trip");

        let listed: Vec<String> =
            reloaded.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["i1", "i2"], "creation order survives reload");
    }

    #[test]
    fn load_is_order_independent_index_before_table() {
        // Write the INDEX row at rowid 1 and its TABLE row at rowid 2, so the index
        // physically precedes its table on page 1. The two-pass load must still
        // resolve it (it does not depend on on-disk row order).
        let mut pager = fresh_pager();
        let index_row = SchemaRow {
            obj_type: "index".into(),
            name: "i".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX i ON t(a)".into()),
        };
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &index_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &table_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.table("t").unwrap().is_some(), "table loads");
        let i = cat.index("i").unwrap().expect("index loads despite preceding its table");
        assert_eq!(i.columns, ["a"]);
        assert_eq!(i.root_page, 3);
    }

    #[test]
    fn load_reconstructs_null_sql_autoindex_row() {
        // An auto-created index stores NULL sql (name sqlite_autoindex_*). Load
        // reconstructs its columns from the table's UNIQUE / PRIMARY KEY constraints
        // (the same derivation the create path used), NOT from the absent sql — this
        // replaces the old "skip" behaviour.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a UNIQUE)".into()),
        };
        let auto_row = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_t_1".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: None,
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &auto_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.table("t").unwrap().is_some(), "table still loads");
        let idx = cat
            .index("sqlite_autoindex_t_1")
            .unwrap()
            .expect("auto-index (NULL sql) is reconstructed from the table's constraints");
        assert_eq!(idx.columns, ["a"], "columns re-derived from UNIQUE(a)");
        assert_eq!(idx.table, "t");
        assert!(idx.unique, "every auto-index is UNIQUE");
        assert!(!idx.partial, "auto-indexes are never partial");
        assert_eq!(idx.root_page, 3, "the row's own rootpage is used");
        let listed: Vec<String> =
            cat.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["sqlite_autoindex_t_1"]);
    }

    #[test]
    fn load_fails_closed_on_autoindex_missing_table() {
        // A NULL-sql auto-index row whose target table is absent is inconsistent schema:
        // fail closed rather than silently dropping it.
        let mut pager = fresh_pager();
        let auto_row = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_gone_1".into(),
            tbl_name: "gone".into(),
            rootpage: 3,
            sql: None,
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &auto_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an auto-index referencing a missing table must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_autoindex_name_not_in_constraints() {
        // A NULL-sql auto-index whose name the table's constraints do not imply (the
        // table declares no UNIQUE / PRIMARY KEY at all) is corrupt/foreign schema: fail
        // closed rather than fabricating an index from thin air.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a, b)".into()),
        };
        let auto_row = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_t_1".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: None,
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &auto_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an auto-index name not implied by the table's constraints must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_autoindex_row_for_reserved_without_rowid_pk() {
        // A WITHOUT ROWID non-integer PK reserves an N (emit_row=false) but real sqlite
        // writes NO sqlite_schema row for it (the table b-tree IS the key). A tampered or
        // foreign image that DOES carry a NULL-sql row for that reserved name must fail
        // closed, not fabricate a UNIQUE index. For `t(a UNIQUE, b TEXT PRIMARY KEY)
        // WITHOUT ROWID`: _1=UNIQUE(a) is real, _2=PK(b) is reserved with no row — so a
        // planted _2 row is inconsistent.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a UNIQUE, b TEXT PRIMARY KEY) WITHOUT ROWID".into()),
        };
        let spurious = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_t_2".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: None,
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &spurious.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "a row for a reserved (emit_row=false) WITHOUT ROWID PK must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_pk_less_without_rowid_table() {
        // The LOAD-side mirror of the create-time WITHOUT ROWID -> PRIMARY KEY rule: real
        // sqlite never persists a WITHOUT ROWID table without a PRIMARY KEY, so a stored
        // PK-less WITHOUT ROWID row is a hand-tampered/foreign image. Load re-runs
        // `table_def_from_ast`, whose `validate_without_rowid_has_primary_key` rejects it, and
        // the load path maps that `Sql` error to `Format` (the corrupt-schema vocabulary).
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a, b) WITHOUT ROWID".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        match cat.load(&pager).unwrap_err() {
            Error::Format(m) => assert!(
                m.contains("PRIMARY KEY missing on table t"),
                "load must fail closed naming the create-time rule, got {m:?}"
            ),
            other => panic!("a stored PK-less WITHOUT ROWID table must fail as Format, got {other:?}"),
        }
    }

    #[test]
    fn load_fails_closed_on_autoindex_name_colliding_with_a_table() {
        // Shared-namespace guard on the auto-index load path, uniform with the explicit
        // path's `load_fails_closed_on_index_name_colliding_with_a_table`: a foreign image
        // carrying a table AND a NULL-sql auto-index row of the same folded name must fail
        // closed rather than caching both (which would leave table(x) and index(x) Some).
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "sqlite_autoindex_t_1".into(),
            tbl_name: "sqlite_autoindex_t_1".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE sqlite_autoindex_t_1(a)".into()),
        };
        let auto_row = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_t_1".into(),
            tbl_name: "sqlite_autoindex_t_1".into(),
            rootpage: 3,
            sql: None,
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &auto_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an auto-index name colliding with a table must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_reconstructs_expression_index_row() {
        // An expression index reconstructs on load from its stored verbatim `sql`
        // (`lang_createindex.html` §1.2), like any other index — the table and the index
        // both come back.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        let expr_row = SchemaRow {
            obj_type: "index".into(),
            name: "i".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX i ON t(a+1)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &expr_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.table("t").unwrap().is_some(), "table loads");
        let idx = cat.index("i").unwrap().expect("expression index reconstructs on load");
        assert_eq!(idx.columns, [""], "expression key stores the empty-name sentinel");
        assert!(idx.key_exprs[0].is_some(), "key expression captured on load");
        assert_eq!(idx.root_page, 3, "stored root page preserved");
    }

    #[test]
    fn load_fails_closed_on_invalid_index_rootpage() {
        // The index load path shares checked_root_page with the table path: a stored
        // index row with a bogus (negative) rootpage must fail closed, not truncate
        // under an `as` cast.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        let index_row = SchemaRow {
            obj_type: "index".into(),
            name: "i".into(),
            tbl_name: "t".into(),
            rootpage: -1,
            sql: Some("CREATE INDEX i ON t(a)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &index_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an out-of-range index rootpage must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_index_referencing_missing_table() {
        // A stored index whose target table is absent is inconsistent schema: fail
        // closed rather than silently dropping the index.
        let mut pager = fresh_pager();
        let index_row = SchemaRow {
            obj_type: "index".into(),
            name: "i".into(),
            tbl_name: "gone".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX i ON gone(a)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &index_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an index referencing a missing table must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_index_with_unknown_column() {
        // A stored named-column index referencing a column its table lacks is corrupt
        // schema (create_index validated columns, so only a tampered DB hits it). It
        // fails closed with Format, matching the other corrupt-schema load paths.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        let index_row = SchemaRow {
            obj_type: "index".into(),
            name: "i".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX i ON t(zzz)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &index_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an unknown column in a stored index must fail closed as Format, got {err:?}"
        );
    }

    #[test]
    fn load_reconstructs_expression_index_among_named_indexes() {
        // A [named, expr, named] layout: all three indexes reconstruct on load, in
        // creation order, with the expression index carrying its captured key expression.
        let mut pager = fresh_pager();
        let rows = [
            SchemaRow {
                obj_type: "table".into(),
                name: "t".into(),
                tbl_name: "t".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE t(a, b)".into()),
            },
            SchemaRow {
                obj_type: "index".into(),
                name: "i1".into(),
                tbl_name: "t".into(),
                rootpage: 3,
                sql: Some("CREATE INDEX i1 ON t(a)".into()),
            },
            SchemaRow {
                obj_type: "index".into(),
                name: "iexpr".into(),
                tbl_name: "t".into(),
                rootpage: 4,
                sql: Some("CREATE INDEX iexpr ON t(a+1)".into()),
            },
            SchemaRow {
                obj_type: "index".into(),
                name: "i2".into(),
                tbl_name: "t".into(),
                rootpage: 5,
                sql: Some("CREATE INDEX i2 ON t(b)".into()),
            },
        ];
        for (i, row) in rows.iter().enumerate() {
            table_insert(&mut pager, SCHEMA_ROOT, (i + 1) as i64, &row.to_record()).unwrap();
        }
        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert!(cat.index("i1").unwrap().is_some(), "first named index loads");
        let iexpr = cat.index("iexpr").unwrap().expect("expression index reconstructs");
        assert!(iexpr.key_exprs[0].is_some(), "expression key captured on load");
        assert!(cat.index("i2").unwrap().is_some(), "named index after the expr one still loads");
        let listed: Vec<String> =
            cat.indexes_on("t").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["i1", "iexpr", "i2"], "all three indexes, in creation order");
    }

    #[test]
    fn load_fails_closed_on_as_select_table_sql() {
        // A stored table row whose verbatim sql is `CREATE TABLE ... AS SELECT` is a
        // deferred-gap build error; on the load path it must fail closed as `Format`
        // (corrupt/foreign schema), one vocabulary with the other load failures — not
        // leak the builder's `Sql`. Real sqlite persists AS SELECT as explicit columns,
        // so this is only reachable from a hand-tampered/foreign DB.
        let mut pager = fresh_pager();
        let row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t AS SELECT 1".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "AS SELECT in a stored table row must fail closed as Format, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_index_name_colliding_with_a_table() {
        // Shared namespace on the load side: a corrupt DB carrying a table and an index
        // of the same folded name must fail closed rather than caching both (which
        // would leave table("x") and index("x") both Some). The load-side mirror of the
        // create_table/create_index write guards.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "x".into(),
            tbl_name: "x".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE x(a)".into()),
        };
        let index_row = SchemaRow {
            obj_type: "index".into(),
            name: "x".into(),
            tbl_name: "x".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX x ON x(a)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &index_row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an index name colliding with a table must fail closed, got {err:?}"
        );
    }

    #[test]
    fn load_fails_closed_on_unparseable_table_sql() {
        let mut pager = fresh_pager();
        let row = SchemaRow {
            obj_type: "table".into(),
            name: "bad".into(),
            tbl_name: "bad".into(),
            rootpage: 2,
            sql: Some("this is not valid sql".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "corrupt schema sql must fail closed, got {err:?}");
    }

    #[test]
    fn load_fails_closed_on_invalid_table_rootpage() {
        // A negative rootpage would silently truncate under `as u32` to a bogus page
        // that only fails much later on a page read. load must reject it at the
        // boundary (a real table root is always a valid page id >= 2).
        let mut pager = fresh_pager();
        let row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: -1,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &row.to_record()).unwrap();
        let mut cat = SchemaCatalog::new();
        let err = cat.load(&pager).unwrap_err();
        assert!(
            matches!(err, Error::Format(_)),
            "an out-of-range table rootpage must fail closed, got {err:?}"
        );
    }

    #[test]
    fn next_rowid_continues_after_reload() {
        // After creating three tables, dropping the store and reloading must resume
        // rowid assignment at 4 (max seen + 1), so a new table does not REPLACE row 1.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        for name in ["a", "b", "c"] {
            create(&mut cat, &mut pager, &format!("CREATE TABLE {name}(x)"));
        }
        assert_eq!(page1_row_count(&pager), 3);

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        create(&mut reloaded, &mut pager, "CREATE TABLE d(x)");
        assert_eq!(page1_row_count(&pager), 4, "the new table appended a 4th row, not a REPLACE");
        assert!(reloaded.table("a").unwrap().is_some(), "existing rows untouched");
        assert!(reloaded.table("d").unwrap().is_some());
    }

    // --- drop_object -----------------------------------------------------------

    /// Parse `sql` (one DROP ...) and drop through the real path, unwrapping.
    fn drop_stmt(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_drop(cat, pager, sql).unwrap();
    }

    /// Like `drop_stmt` but returns the `Result`, for asserting error / IF EXISTS paths.
    fn try_drop(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::Drop(stmt) = &ast.statements[0] else {
            panic!("not a DROP: {sql}");
        };
        cat.drop_object(pager, stmt)
    }

    #[test]
    fn drop_table_removes_table_and_reload_agrees() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        assert_eq!(page1_row_count(&pager), 1);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE t");
        assert!(cat.table("t").unwrap().is_none(), "table gone from the cache");
        assert_eq!(page1_row_count(&pager), 0, "its page-1 row was deleted");

        // A fresh store loading only from page 1 also does not see `t`.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("t").unwrap().is_none(), "still gone after a reload");
    }

    #[test]
    fn drop_table_cascades_auto_and_explicit_indexes() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
        // table row + auto-index row (sqlite_autoindex_t_1) + explicit index row.
        assert_eq!(page1_row_count(&pager), 3);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE t");
        assert_eq!(page1_row_count(&pager), 0, "table + both index rows all removed");
        assert!(cat.table("t").unwrap().is_none());
        assert!(cat.index("i").unwrap().is_none(), "explicit index cascaded away");
        assert!(
            cat.index("sqlite_autoindex_t_1").unwrap().is_none(),
            "auto-index cascaded away"
        );
        assert!(cat.indexes_on("t").unwrap().is_empty());

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("t").unwrap().is_none());
        assert!(reloaded.index("i").unwrap().is_none());
        assert!(reloaded.index("sqlite_autoindex_t_1").unwrap().is_none());
    }

    #[test]
    fn drop_table_if_exists_absent_is_noop_else_errors() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE keep(a)");
        let cookie_before = read_schema_cookie(&pager);
        let rows_before = page1_row_count(&pager);

        // IF EXISTS on an absent table: Ok, and nothing written (no row, no cookie bump).
        drop_stmt(&mut cat, &mut pager, "DROP TABLE IF EXISTS ghost");
        assert_eq!(page1_row_count(&pager), rows_before, "IF EXISTS wrote no row");
        assert_eq!(
            read_schema_cookie(&pager),
            cookie_before,
            "IF EXISTS on an absent table did not bump the cookie"
        );
        assert!(cat.table("keep").unwrap().is_some(), "the unrelated table is untouched");

        // Without IF EXISTS an absent table is an error, and still writes nothing.
        let err = try_drop(&mut cat, &mut pager, "DROP TABLE ghost").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "absent table -> Sql, got {err:?}");
        assert_eq!(page1_row_count(&pager), rows_before, "the error path wrote no row");
    }

    #[test]
    fn drop_table_rejects_internal_sqlite_name() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        // The built-in schema table (and its alias) exist, so real sqlite reports "may
        // not be dropped" rather than "no such table"; the reserved check fires even
        // under IF EXISTS.
        let err = try_drop(&mut cat, &mut pager, "DROP TABLE sqlite_schema").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "internal table -> Sql, got {err:?}");
        let err = try_drop(&mut cat, &mut pager, "DROP TABLE IF EXISTS sqlite_master").unwrap_err();
        assert!(
            matches!(err, Error::Sql(_)),
            "IF EXISTS must not suppress the reserved-name rejection, got {err:?}"
        );
    }

    #[test]
    fn drop_index_removes_explicit_index_only() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        assert_eq!(page1_row_count(&pager), 2, "table row + index row");

        drop_stmt(&mut cat, &mut pager, "DROP INDEX i");
        assert!(cat.index("i").unwrap().is_none(), "index gone from the cache");
        assert!(cat.indexes_on("t").unwrap().is_empty(), "unlinked from its table");
        assert!(cat.table("t").unwrap().is_some(), "the table itself remains");
        assert_eq!(page1_row_count(&pager), 1, "only the index row was deleted");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.index("i").unwrap().is_none(), "gone after a reload");
        assert!(reloaded.table("t").unwrap().is_some());

        // IF EXISTS on an absent index is Ok; without it, an error.
        drop_stmt(&mut cat, &mut pager, "DROP INDEX IF EXISTS gone");
        let err = try_drop(&mut cat, &mut pager, "DROP INDEX gone").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "absent index -> Sql, got {err:?}");
    }

    #[test]
    fn drop_index_rejects_auto_index() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE)");
        let rows_before = page1_row_count(&pager);

        let err = try_drop(&mut cat, &mut pager, "DROP INDEX sqlite_autoindex_t_1").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "auto-index may not be dropped -> Sql, got {err:?}");
        assert!(
            cat.index("sqlite_autoindex_t_1").unwrap().is_some(),
            "the auto-index is untouched by the rejected DROP"
        );
        assert_eq!(page1_row_count(&pager), rows_before, "nothing was deleted");
    }

    #[test]
    fn drop_table_is_case_insensitive() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        // Created as `t`, dropped via mixed-case `T`.
        drop_stmt(&mut cat, &mut pager, "DROP TABLE T");
        assert!(cat.table("t").unwrap().is_none());
        assert_eq!(page1_row_count(&pager), 0);
    }

    #[test]
    fn drop_table_bumps_schema_cookie_once_even_when_cascading() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
        let before = read_schema_cookie(&pager);

        // One DROP TABLE that cascades three rows is still a single schema change.
        drop_stmt(&mut cat, &mut pager, "DROP TABLE t");
        assert_eq!(
            read_schema_cookie(&pager),
            before + 1,
            "a cascading DROP TABLE bumps the cookie exactly once"
        );
    }

    #[test]
    fn drop_index_bumps_schema_cookie_by_one() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        let before = read_schema_cookie(&pager);
        drop_stmt(&mut cat, &mut pager, "DROP INDEX i");
        assert_eq!(read_schema_cookie(&pager), before + 1);
    }

    #[test]
    fn drop_one_of_two_tables_reloads_survivor_and_rowid_stays_monotonic() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE a(x)");
        create(&mut cat, &mut pager, "CREATE TABLE b(y)");
        assert_eq!(page1_row_count(&pager), 2);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE a");
        assert!(cat.table("a").unwrap().is_none());
        assert!(cat.table("b").unwrap().is_some());
        assert_eq!(page1_row_count(&pager), 1, "only b's row remains");

        // `next_schema_rowid` stays monotonic across a drop: a table created afterwards
        // APPENDS a new row (it does not REPLACE the survivor by reusing a freed rowid).
        create(&mut cat, &mut pager, "CREATE TABLE c(z)");
        assert_eq!(page1_row_count(&pager), 2, "b + c: c appended, b not overwritten");
        assert!(cat.table("b").unwrap().is_some(), "b untouched by c's insert");
        assert!(cat.table("c").unwrap().is_some());

        // A fresh load from page 1 sees exactly the survivors.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("a").unwrap().is_none(), "dropped table absent after reload");
        assert!(reloaded.table("b").unwrap().is_some());
        assert!(reloaded.table("c").unwrap().is_some());
    }

    #[test]
    fn drop_view_deletes_loaded_row_else_if_exists_or_errors() {
        // A loaded database carries its view rows on page 1, and `load` caches them in
        // `self.views` (which DROP VIEW's existence check consults). Seed a table row and
        // a view row directly, load, then DROP VIEW must delete just the view row.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a)".into()),
        };
        let view_row = SchemaRow {
            obj_type: "view".into(),
            name: "v".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: Some("CREATE VIEW v AS SELECT 1".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &view_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert_eq!(page1_row_count(&pager), 2);

        // A non-existent view: IF EXISTS is Ok and writes nothing; plain DROP errors.
        let cookie_before = read_schema_cookie(&pager);
        drop_stmt(&mut cat, &mut pager, "DROP VIEW IF EXISTS nope");
        assert_eq!(page1_row_count(&pager), 2, "IF EXISTS on an absent view wrote nothing");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no cookie bump for the no-op");
        let err = try_drop(&mut cat, &mut pager, "DROP VIEW nope").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "absent view -> Sql, got {err:?}");

        // The real view row is deleted; the table row is left intact.
        drop_stmt(&mut cat, &mut pager, "DROP VIEW v");
        assert_eq!(page1_row_count(&pager), 1, "only the view row was removed");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "the successful drop bumped once");
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("t").unwrap().is_some(), "the table survives the DROP VIEW");
    }

    // --- create_view / create_trigger ------------------------------------------

    /// Parse `sql` (one CREATE VIEW) and create the view through the real path.
    fn create_view(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_create_view(cat, pager, sql).unwrap();
    }

    /// Like `create_view` but returns the `Result`, for asserting error paths.
    fn try_create_view(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::CreateView(stmt) = &ast.statements[0] else {
            panic!("not a CREATE VIEW: {sql}");
        };
        cat.create_view(pager, stmt, sql)
    }

    /// Parse `sql` (one CREATE TRIGGER) and create the trigger through the real path.
    fn create_trigger(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_create_trigger(cat, pager, sql).unwrap();
    }

    /// Like `create_trigger` but returns the `Result`, for asserting error paths. These unit
    /// tests exercise a SINGLE `SchemaCatalog`, so every target is same-store.
    fn try_create_trigger(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::CreateTrigger(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TRIGGER: {sql}");
        };
        cat.create_trigger(pager, stmt, sql, crate::TriggerTarget::SameStore)
    }

    /// The decoded 5-value records of every page-1 row whose `type` column equals
    /// `obj_type`, in rowid order.
    fn rows_of_type(pager: &MemPager, obj_type: &str) -> Vec<Vec<Value>> {
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT).unwrap();
        let mut out = Vec::new();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            let values = decode_record(&cursor.payload().unwrap());
            if matches!(&values[0], Value::Text(s) if s == obj_type) {
                out.push(values);
            }
            positioned = cursor.next().unwrap();
        }
        out
    }

    /// A short CREATE TRIGGER on table `t` (helper for the many trigger tests).
    fn trg_on(table: &str) -> String {
        format!("CREATE TRIGGER trg AFTER INSERT ON {table} BEGIN SELECT 1; END")
    }

    #[test]
    fn create_view_persists_and_reload_sees_it() {
        // 1: a view row is added (count +1), the cookie bumps by 1, and a fresh store
        // that only loads page 1 sees the view — proven via the shared namespace, since
        // there is no public view() accessor.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t");
        assert_eq!(page1_row_count(&pager), rows_before + 1, "one view row added");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "cookie bumped by 1");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let err = try_create(&mut reloaded, &mut pager, "CREATE TABLE v(x)").unwrap_err();
        assert!(matches!(err, Error::Sql(_)), "reloaded view holds the namespace, got {err:?}");
    }

    #[test]
    fn create_view_writes_exact_schema_row() {
        // 2: the exact 5-column view row — type/name/tbl_name(=name)/rootpage 0/sql.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        let sql = "CREATE VIEW v AS SELECT a FROM t";
        create_view(&mut cat, &mut pager, sql);

        let rows = rows_of_type(&pager, "view");
        assert_eq!(rows.len(), 1, "exactly one type='view' row");
        let v = &rows[0];
        assert_eq!(v.len(), 5);
        assert!(matches!(&v[0], Value::Text(s) if s == "view"));
        assert!(matches!(&v[1], Value::Text(s) if s == "v"));
        assert!(matches!(&v[2], Value::Text(s) if s == "v"), "tbl_name equals the view name");
        assert!(matches!(&v[3], Value::Integer(0)), "rootpage 0 (a view owns no b-tree)");
        assert!(matches!(&v[4], Value::Text(s) if s == sql), "sql stored verbatim");
    }

    #[test]
    fn create_trigger_writes_row_with_target_tbl_name() {
        // 3: the exact 5-column trigger row — tbl_name is the TARGET table, rootpage 0.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        let cookie_before = read_schema_cookie(&pager);
        let sql = trg_on("t");
        create_trigger(&mut cat, &mut pager, &sql);

        let rows = rows_of_type(&pager, "trigger");
        assert_eq!(rows.len(), 1, "exactly one type='trigger' row");
        let v = &rows[0];
        assert_eq!(v.len(), 5);
        assert!(matches!(&v[0], Value::Text(s) if s == "trigger"));
        assert!(matches!(&v[1], Value::Text(s) if s == "trg"));
        assert!(matches!(&v[2], Value::Text(s) if s == "t"), "tbl_name is the TARGET table");
        assert!(matches!(&v[3], Value::Integer(0)), "rootpage 0 (a trigger owns no b-tree)");
        assert!(matches!(&v[4], Value::Text(s) if *s == sql), "sql stored verbatim");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "cookie bumped by 1");
    }

    #[test]
    fn triggers_on_returns_target_matched_triggers_in_deterministic_order() {
        // The read accessor the trigger compiler/executor consults: it returns every
        // trigger whose TARGET table is `table` (case-insensitively), excludes triggers
        // on other tables, and is deterministically ordered (by folded name here) so a
        // write's compiled plan is reproducible. It also survives a reload from page 1.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create(&mut cat, &mut pager, "CREATE TABLE other(b)");

        let make = |name: &str, table: &str| {
            format!("CREATE TRIGGER {name} AFTER INSERT ON {table} BEGIN SELECT 1; END")
        };
        // Insert in an order that differs from the sorted order, so the sort is proven
        // (not merely the incidental HashMap iteration order). `beta`'s target is written
        // `ON T` (mixed case) while the table is `t`, so its stored `TriggerDef.table` is
        // "T" — matching it under a `t` query proves the filter folds the STORED target,
        // not just the query argument (a bare `t.table == key` would miss it).
        create_trigger(&mut cat, &mut pager, &make("zebra", "t"));
        create_trigger(&mut cat, &mut pager, &make("alpha", "t"));
        create_trigger(&mut cat, &mut pager, &make("mango", "t"));
        create_trigger(&mut cat, &mut pager, &make("beta", "T"));
        create_trigger(&mut cat, &mut pager, &make("on_other", "other"));

        let names = |c: &SchemaCatalog, tbl: &str| -> Vec<String> {
            c.triggers_on(tbl).unwrap().iter().map(|t| t.name.clone()).collect()
        };

        assert_eq!(
            names(&cat, "t"),
            ["alpha", "beta", "mango", "zebra"],
            "only t's triggers (incl. the mixed-case `ON T` target), in folded-name order"
        );
        assert_eq!(
            names(&cat, "T"),
            ["alpha", "beta", "mango", "zebra"],
            "match folds BOTH the query key and the stored target"
        );
        assert_eq!(names(&cat, "other"), ["on_other"], "other table's trigger is separate");
        assert!(names(&cat, "t").iter().all(|n| n != "on_other"), "other's trigger never leaks into t");

        // Every returned def's target really is the queried table (case-folded).
        assert!(cat.triggers_on("t").unwrap().iter().all(|d| norm(&d.table) == norm("t")));

        // A table with no triggers, and an absent table, both return empty (not an error).
        create(&mut cat, &mut pager, "CREATE TABLE lonely(c)");
        assert!(cat.triggers_on("lonely").unwrap().is_empty(), "no triggers -> empty");
        assert!(cat.triggers_on("ghost").unwrap().is_empty(), "absent table -> empty, not an error");

        // A fresh store loaded only from page 1 reports the same set in the same order.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert_eq!(
            names(&reloaded, "t"),
            ["alpha", "beta", "mango", "zebra"],
            "same order after reload (mixed-case target survives page-1 round-trip)"
        );
        assert_eq!(names(&reloaded, "other"), ["on_other"], "reload keeps target separation");
    }

    #[test]
    fn shared_namespace_rejects_cross_type_creates() {
        // 4: all four object kinds share one namespace, so a name held by any one blocks
        // a CREATE of any other kind — both directions.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        let rows_before = page1_row_count(&pager);

        // A view name blocks a table and an index of that name.
        let e = try_create(&mut cat, &mut pager, "CREATE TABLE v(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "table colliding with view -> Sql, got {e:?}");
        let e = try_create_index(&mut cat, &mut pager, "CREATE INDEX v ON t(a)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "index colliding with view -> Sql, got {e:?}");
        // A trigger name blocks a table of that name.
        let e = try_create(&mut cat, &mut pager, "CREATE TABLE trg(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "table colliding with trigger -> Sql, got {e:?}");
        // Conversely a table name and an index name each block a view of that name.
        let e = try_create_view(&mut cat, &mut pager, "CREATE VIEW t AS SELECT 1").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "view colliding with table -> Sql, got {e:?}");
        let e = try_create_view(&mut cat, &mut pager, "CREATE VIEW i AS SELECT 1").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "view colliding with index -> Sql, got {e:?}");

        assert_eq!(page1_row_count(&pager), rows_before, "no rejected CREATE wrote a row");
    }

    #[test]
    fn if_not_exists_suppresses_only_same_type_view_and_trigger() {
        // 5: IF NOT EXISTS on an existing view/trigger is a silent no-op; the same
        // CREATE without IF NOT EXISTS is a hard error.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        create_view(&mut cat, &mut pager, "CREATE VIEW IF NOT EXISTS v AS SELECT a FROM t");
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER IF NOT EXISTS trg AFTER INSERT ON t BEGIN SELECT 1; END",
        );
        assert_eq!(page1_row_count(&pager), rows_before, "IF NOT EXISTS added no row");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "IF NOT EXISTS did not bump cookie");

        let e = try_create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "duplicate view (no INE) -> Sql, got {e:?}");
        let e = try_create_trigger(&mut cat, &mut pager, &trg_on("t")).unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "duplicate trigger (no INE) -> Sql, got {e:?}");
    }

    #[test]
    fn create_trigger_on_missing_table_errors() {
        // 6: the target table must exist at create time.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let err = try_create_trigger(&mut cat, &mut pager, &trg_on("missing")).unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("no such table"), "got {m:?}"),
            other => panic!("expected a no-such-table Sql error, got {other:?}"),
        }
        assert_eq!(page1_row_count(&pager), 0, "nothing persisted for the failed trigger");
    }

    #[test]
    fn create_trigger_on_schema_table_is_rejected() {
        // A trigger may not target a system table. `sqlite_master` resolves (via
        // resolve_table_key) to the built-in schema table, so it CLEARS the existence check
        // and hits the system-table guard — real sqlite's `cannot create trigger on system
        // table`. Nothing is persisted.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let err = try_create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER x AFTER INSERT ON sqlite_master BEGIN SELECT 1; END",
        )
        .unwrap_err();
        match err {
            Error::Sql(m) => assert_eq!(m, "cannot create trigger on system table"),
            other => panic!("expected a system-table Sql error, got {other:?}"),
        }
        assert_eq!(page1_row_count(&pager), 0, "nothing persisted for the rejected trigger");
    }

    #[test]
    fn create_trigger_on_sqlite_sequence_is_rejected() {
        // The guard covers an internal table beyond the schema table: an AUTOINCREMENT
        // `CREATE TABLE` auto-creates `sqlite_sequence`, which then EXISTS (clearing the
        // existence check) but is still `sqlite_`-prefixed, so a trigger on it is rejected.
        // Proven end-to-end — the sequence table really exists first.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT)");
        assert!(
            cat.table("sqlite_sequence").unwrap().is_some(),
            "precondition: the autoincrement table created sqlite_sequence"
        );
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        let err = try_create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER x AFTER INSERT ON sqlite_sequence BEGIN SELECT 1; END",
        )
        .unwrap_err();
        match err {
            Error::Sql(m) => assert_eq!(m, "cannot create trigger on system table"),
            other => panic!("expected a system-table Sql error, got {other:?}"),
        }
        assert_eq!(page1_row_count(&pager), rows_before, "rejected trigger wrote no page-1 row");
        assert_eq!(
            read_schema_cookie(&pager),
            cookie_before,
            "rejected trigger did not bump the schema cookie"
        );
    }

    #[test]
    fn create_trigger_on_user_table_is_allowed() {
        // Positive control: the system-table guard is narrow — a trigger on an ordinary
        // user table (a non-`sqlite_` name) still persists exactly one row.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        assert_eq!(rows_of_type(&pager, "trigger").len(), 1, "the user-table trigger persisted");
    }

    #[test]
    fn create_trigger_on_absent_sqlite_prefixed_target_reports_no_such_table() {
        // Ordering contract: existence is checked BEFORE the system-table guard, matching real
        // sqlite (an absent target reports `no such table` first). `sqlite_nope` is `sqlite_`-
        // prefixed but is not registered and is not one of `resolve_table_key`'s aliases, so it
        // resolves to nothing — existence fails and wins over the guard. This pins the order the
        // guard's comment claims: a mutant moving the guard ahead of existence would instead
        // report `cannot create trigger on system table` here and fail this assertion.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let err = try_create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER x AFTER INSERT ON sqlite_nope BEGIN SELECT 1; END",
        )
        .unwrap_err();
        match err {
            Error::Sql(m) => assert_eq!(m, "no such table: sqlite_nope"),
            other => panic!("expected a no-such-table Sql error, got {other:?}"),
        }
        assert_eq!(page1_row_count(&pager), 0, "nothing persisted for the rejected trigger");
    }

    // --- INSTEAD OF triggers on views (lang_createtrigger.html §3) --------------

    #[test]
    fn create_instead_of_trigger_on_view_persists_and_is_found() {
        // INSTEAD OF INSERT/UPDATE/DELETE triggers are valid on a view. Each persists a
        // type='trigger' page-1 row whose tbl_name is the VIEW name (rootpage 0), and
        // triggers_on(view) returns them (an unrelated name does not).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE base(a, b)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a, b FROM base");

        let io = |name: &str, event: &str| {
            format!("CREATE TRIGGER {name} INSTEAD OF {event} ON v BEGIN SELECT 1; END")
        };
        create_trigger(&mut cat, &mut pager, &io("t_ins", "INSERT"));
        create_trigger(&mut cat, &mut pager, &io("t_upd", "UPDATE"));
        create_trigger(&mut cat, &mut pager, &io("t_del", "DELETE"));

        let rows = rows_of_type(&pager, "trigger");
        assert_eq!(rows.len(), 3, "three INSTEAD OF trigger rows persisted");
        for r in &rows {
            assert!(matches!(&r[2], Value::Text(s) if s == "v"), "tbl_name is the view: {:?}", r[2]);
            assert!(matches!(&r[3], Value::Integer(0)), "rootpage 0 (a trigger owns no b-tree)");
        }
        // The sql column (r[4]) is stored verbatim — compared as a set since row order is
        // by rowid, not name.
        let mut stored_sql: Vec<String> = rows
            .iter()
            .map(|r| match &r[4] {
                Value::Text(s) => s.clone(),
                other => panic!("trigger sql column is not text: {other:?}"),
            })
            .collect();
        stored_sql.sort();
        let mut expected_sql = vec![io("t_ins", "INSERT"), io("t_upd", "UPDATE"), io("t_del", "DELETE")];
        expected_sql.sort();
        assert_eq!(stored_sql, expected_sql, "each row stores its verbatim CREATE TRIGGER sql");

        let names: Vec<String> =
            cat.triggers_on("v").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, ["t_del", "t_ins", "t_upd"], "all three, folded-name order");
        assert_eq!(cat.triggers_on("V").unwrap().len(), 3, "the view name folds in triggers_on");
        assert!(cat.triggers_on("base").unwrap().is_empty(), "the base table has no triggers");
        assert!(cat.triggers_on("nope").unwrap().is_empty(), "an unrelated name -> empty");
    }

    #[test]
    fn instead_of_trigger_on_view_survives_reload() {
        // Before this change, load fail-closed (Error::Format) because a trigger's target
        // was a view, not a table — which aborted the WHOLE database load. It must now
        // round-trip: drop the cache, rebuild from the same page-1 image, still there.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE base(a, b)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a, b FROM base");
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER t_ins INSTEAD OF INSERT ON v BEGIN SELECT 1; END",
        );
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER t_del INSTEAD OF DELETE ON v BEGIN SELECT 1; END",
        );

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("load must succeed for an INSTEAD OF trigger on a view");
        let names: Vec<String> =
            reloaded.triggers_on("v").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, ["t_del", "t_ins"], "both view triggers cached after reload");
    }

    #[test]
    fn load_reads_foreign_instead_of_trigger_on_view() {
        // A `.db` real sqlite wrote: a base table, a view over it, and an INSTEAD OF
        // trigger whose tbl_name is the view. Seed the page-1 rows directly (no create
        // path), then load — proving we can READ a schema real sqlite wrote, the exact
        // on-disk-read gap this closes.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "base".into(),
            tbl_name: "base".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE base(a, b)".into()),
        };
        let view_row = SchemaRow {
            obj_type: "view".into(),
            name: "v".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: Some("CREATE VIEW v AS SELECT a, b FROM base".into()),
        };
        let trigger_row = SchemaRow {
            obj_type: "trigger".into(),
            name: "vt".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: Some("CREATE TRIGGER vt INSTEAD OF INSERT ON v BEGIN SELECT 1; END".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &view_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 3, &trigger_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).expect("a foreign image with an INSTEAD OF trigger on a view must load");
        let trigs = cat.triggers_on("v").unwrap();
        assert_eq!(trigs.len(), 1, "the foreign trigger is cached");
        assert_eq!(trigs[0].name, "vt");
        assert_eq!(norm(&trigs[0].table), "v", "its target is the view");
    }

    #[test]
    fn create_trigger_rejects_illegal_timing_target_combos() {
        // lang_createtrigger.html §3: INSTEAD OF is view-only; BEFORE/AFTER (and an
        // omitted timing, which defaults to BEFORE) are table-only. Each illegal combo
        // errors, writes NO page-1 row, and does NOT bump the schema cookie.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE base(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM base");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        let cases: [(&str, &str); 4] = [
            ("CREATE TRIGGER x INSTEAD OF INSERT ON base BEGIN SELECT 1; END", "INSTEAD OF"),
            ("CREATE TRIGGER x BEFORE INSERT ON v BEGIN SELECT 1; END", "view"),
            ("CREATE TRIGGER x AFTER INSERT ON v BEGIN SELECT 1; END", "view"),
            ("CREATE TRIGGER x INSERT ON v BEGIN SELECT 1; END", "view"),
        ];
        for (sql, needle) in cases {
            let err = try_create_trigger(&mut cat, &mut pager, sql).unwrap_err();
            match err {
                Error::Sql(m) => {
                    assert!(m.contains(needle), "message {m:?} should mention {needle:?} for {sql}")
                }
                other => panic!("expected an Sql error for {sql}, got {other:?}"),
            }
            assert_eq!(page1_row_count(&pager), rows_before, "rejected combo wrote a row: {sql}");
            assert_eq!(
                read_schema_cookie(&pager),
                cookie_before,
                "rejected combo bumped the cookie: {sql}"
            );
        }
    }

    #[test]
    fn load_rejects_corrupt_trigger_timing_target_combos() {
        // A stored INSTEAD OF whose target is a TABLE, or a BEFORE/AFTER whose target is
        // a VIEW, is a combo real sqlite never writes — load fails closed (Error::Format)
        // rather than caching a trigger that could never fire through the DML path.

        // (a) INSTEAD OF on a table.
        {
            let mut pager = fresh_pager();
            let table_row = SchemaRow {
                obj_type: "table".into(),
                name: "t".into(),
                tbl_name: "t".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE t(a)".into()),
            };
            let bad = SchemaRow {
                obj_type: "trigger".into(),
                name: "bad".into(),
                tbl_name: "t".into(),
                rootpage: 0,
                sql: Some("CREATE TRIGGER bad INSTEAD OF INSERT ON t BEGIN SELECT 1; END".into()),
            };
            table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
            table_insert(&mut pager, SCHEMA_ROOT, 2, &bad.to_record()).unwrap();
            let mut cat = SchemaCatalog::new();
            let err = cat.load(&pager).unwrap_err();
            assert!(
                matches!(err, Error::Format(_)),
                "INSTEAD OF on a table must fail load closed, got {err:?}"
            );
        }

        // (b) BEFORE on a view.
        {
            let mut pager = fresh_pager();
            let table_row = SchemaRow {
                obj_type: "table".into(),
                name: "base".into(),
                tbl_name: "base".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE base(a)".into()),
            };
            let view_row = SchemaRow {
                obj_type: "view".into(),
                name: "v".into(),
                tbl_name: "v".into(),
                rootpage: 0,
                sql: Some("CREATE VIEW v AS SELECT a FROM base".into()),
            };
            let bad = SchemaRow {
                obj_type: "trigger".into(),
                name: "bad".into(),
                tbl_name: "v".into(),
                rootpage: 0,
                sql: Some("CREATE TRIGGER bad BEFORE INSERT ON v BEGIN SELECT 1; END".into()),
            };
            table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
            table_insert(&mut pager, SCHEMA_ROOT, 2, &view_row.to_record()).unwrap();
            table_insert(&mut pager, SCHEMA_ROOT, 3, &bad.to_record()).unwrap();
            let mut cat = SchemaCatalog::new();
            let err = cat.load(&pager).unwrap_err();
            assert!(
                matches!(err, Error::Format(_)),
                "BEFORE on a view must fail load closed, got {err:?}"
            );
        }
    }

    #[test]
    fn drop_view_cascades_instead_of_triggers_created() {
        // DROP VIEW removes the view row AND its INSTEAD OF trigger rows (page-1 count
        // drops by 2 for one view + one trigger), frees both names, and leaves an image
        // that reloads cleanly (no orphan trigger row).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE base(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM base");
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER vt INSTEAD OF INSERT ON v BEGIN SELECT 1; END",
        );
        let before = page1_row_count(&pager);
        assert_eq!(cat.triggers_on("v").unwrap().len(), 1, "the view trigger exists pre-drop");

        drop_stmt(&mut cat, &mut pager, "DROP VIEW v");
        assert_eq!(page1_row_count(&pager), before - 2, "view row + its trigger row both removed");
        assert_eq!(rows_of_type(&pager, "view").len(), 0, "no view rows left");
        assert_eq!(rows_of_type(&pager, "trigger").len(), 0, "the view's trigger cascaded away");
        assert!(cat.triggers_on("v").unwrap().is_empty(), "trigger gone from the cache too");

        // Both names are free for reuse, and the resulting image reloads.
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM base");
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER vt INSTEAD OF INSERT ON v BEGIN SELECT 1; END",
        );
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("image reloads cleanly after the DROP VIEW cascade");
        assert_eq!(reloaded.triggers_on("v").unwrap().len(), 1, "re-created trigger reloads");
    }

    #[test]
    fn drop_view_cascades_instead_of_triggers_loaded() {
        // The same cascade, but the view + trigger are LOADED from a page-1 image (not
        // freshly created) — proving the drop scans page 1, not the cache.
        let mut pager = fresh_pager();
        let table_row = SchemaRow {
            obj_type: "table".into(),
            name: "base".into(),
            tbl_name: "base".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE base(a)".into()),
        };
        let view_row = SchemaRow {
            obj_type: "view".into(),
            name: "v".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: Some("CREATE VIEW v AS SELECT a FROM base".into()),
        };
        let trigger_row = SchemaRow {
            obj_type: "trigger".into(),
            name: "vt".into(),
            tbl_name: "v".into(),
            rootpage: 0,
            sql: Some("CREATE TRIGGER vt INSTEAD OF INSERT ON v BEGIN SELECT 1; END".into()),
        };
        table_insert(&mut pager, SCHEMA_ROOT, 1, &table_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 2, &view_row.to_record()).unwrap();
        table_insert(&mut pager, SCHEMA_ROOT, 3, &trigger_row.to_record()).unwrap();

        let mut cat = SchemaCatalog::new();
        cat.load(&pager).unwrap();
        assert_eq!(page1_row_count(&pager), 3);
        let cookie_before = read_schema_cookie(&pager);

        drop_stmt(&mut cat, &mut pager, "DROP VIEW v");
        assert_eq!(page1_row_count(&pager), 1, "only the base table row remains");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "exactly one cookie bump");
        assert!(cat.triggers_on("v").unwrap().is_empty(), "the loaded trigger cascaded from the cache");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("post-cascade image reloads (no orphan trigger row)");
        assert!(reloaded.table("base").unwrap().is_some(), "base survives DROP VIEW");
        assert!(reloaded.triggers_on("v").unwrap().is_empty());
    }

    #[test]
    fn drop_view_cascade_only_touches_the_dropped_views_triggers() {
        // Isolation: with TWO views each owning an INSTEAD OF trigger (plus a base-table
        // trigger), DROP VIEW v1 must cascade ONLY v1's trigger — not v2's, and not the
        // base table's. Guards against an over-broad cascade predicate (deleting every
        // trigger row, or matching by name rather than by the view `tbl_name`).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE base(a)");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER bt AFTER INSERT ON base BEGIN SELECT 1; END");
        create_view(&mut cat, &mut pager, "CREATE VIEW v1 AS SELECT a FROM base");
        create_view(&mut cat, &mut pager, "CREATE VIEW v2 AS SELECT a FROM base");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER t1 INSTEAD OF INSERT ON v1 BEGIN SELECT 1; END");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER t2 INSTEAD OF INSERT ON v2 BEGIN SELECT 1; END");
        let before = page1_row_count(&pager);

        drop_stmt(&mut cat, &mut pager, "DROP VIEW v1");

        // Exactly two rows removed: v1 and its trigger t1 — not v2/t2, not base/bt.
        assert_eq!(page1_row_count(&pager), before - 2, "only v1 + t1 removed");
        assert!(cat.triggers_on("v1").unwrap().is_empty(), "v1's trigger cascaded");
        let v2_trigs: Vec<String> =
            cat.triggers_on("v2").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(v2_trigs, ["t2"], "v2's trigger is untouched");
        let base_trigs: Vec<String> =
            cat.triggers_on("base").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(base_trigs, ["bt"], "the base table's trigger is untouched");
        assert!(cat.view("v2").unwrap().is_some(), "v2 itself survives");

        // A fresh load agrees: v2/t2 and base/bt persist, v1/t1 are gone.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("post-drop image reloads");
        assert_eq!(
            reloaded.triggers_on("v2").unwrap().iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
            ["t2"],
            "v2's trigger persists across reload"
        );
        assert!(reloaded.triggers_on("v1").unwrap().is_empty(), "no orphan v1 trigger after reload");
    }

    #[test]
    fn table_before_after_triggers_regress_and_instead_of_on_table_rejected() {
        // Regression: BEFORE/AFTER triggers on a TABLE create, reload, and DROP TRIGGER
        // exactly as before; and INSTEAD OF on a table is still rejected, writing nothing.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER b_ins BEFORE INSERT ON t BEGIN SELECT 1; END",
        );
        create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER a_del AFTER DELETE ON t BEGIN SELECT 1; END",
        );

        let rows_before = page1_row_count(&pager);
        let err = try_create_trigger(
            &mut cat,
            &mut pager,
            "CREATE TRIGGER io INSTEAD OF INSERT ON t BEGIN SELECT 1; END",
        )
        .unwrap_err();
        match err {
            Error::Sql(m) => assert!(m.contains("INSTEAD OF"), "got {m:?}"),
            other => panic!("expected an Sql error, got {other:?}"),
        }
        assert_eq!(page1_row_count(&pager), rows_before, "rejected INSTEAD OF on a table wrote nothing");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let names: Vec<String> =
            reloaded.triggers_on("t").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, ["a_del", "b_ins"], "both table triggers reload, folded-name order");

        drop_stmt(&mut cat, &mut pager, "DROP TRIGGER b_ins");
        let names: Vec<String> =
            cat.triggers_on("t").unwrap().iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, ["a_del"], "DROP TRIGGER removes exactly one, no cascade");
    }

    #[test]
    fn drop_table_cascades_cached_triggers_and_frees_the_name() {
        // 7a: DROP TABLE removes the table row AND its trigger rows, and frees the
        // trigger name from the cache (so re-creating it on another table does not clash).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create(&mut cat, &mut pager, "CREATE TABLE other(b)");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        assert_eq!(rows_of_type(&pager, "trigger").len(), 1);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE t");
        assert!(cat.table("t").unwrap().is_none(), "table gone");
        assert_eq!(rows_of_type(&pager, "trigger").len(), 0, "trigger row cascaded away");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("t").unwrap().is_none(), "table absent after reload");
        // The trigger name was freed: re-creating `trg` on `other` is not a collision.
        create_trigger(&mut cat, &mut pager, &trg_on("other"));
        assert_eq!(rows_of_type(&pager, "trigger").len(), 1, "trg re-created on `other`");
    }

    #[test]
    fn drop_table_does_not_cascade_views() {
        // SQLite leaves a view referencing a dropped table in place (unlike triggers and
        // indexes, views are NOT cascaded). The view row survives and its name stays held.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW vv AS SELECT a FROM t");
        assert_eq!(rows_of_type(&pager, "view").len(), 1);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE t");
        assert!(cat.table("t").unwrap().is_none(), "table dropped");
        assert_eq!(rows_of_type(&pager, "view").len(), 1, "the view row survives DROP TABLE");
        assert!(
            matches!(
                try_create(&mut cat, &mut pager, "CREATE TABLE vv(x)").unwrap_err(),
                Error::Sql(_)
            ),
            "view name still held after DROP TABLE (views are not cascaded)"
        );
    }

    #[test]
    fn drop_view_frees_the_name() {
        // 7b: DROP VIEW removes the view row and frees the name for reuse.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t");
        let e = try_create(&mut cat, &mut pager, "CREATE TABLE v(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "a table named v is blocked while v is a view");

        drop_stmt(&mut cat, &mut pager, "DROP VIEW v");
        assert_eq!(rows_of_type(&pager, "view").len(), 0, "view row deleted");
        create(&mut cat, &mut pager, "CREATE TABLE v(x)");
        assert!(cat.table("v").unwrap().is_some(), "v is now a table (name was freed)");
    }

    #[test]
    fn drop_trigger_frees_the_name() {
        // 7c: DROP TRIGGER removes the trigger row and frees the name for reuse.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        let e = try_create(&mut cat, &mut pager, "CREATE TABLE trg(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "a table named trg is blocked while trg is a trigger");

        drop_stmt(&mut cat, &mut pager, "DROP TRIGGER trg");
        assert_eq!(rows_of_type(&pager, "trigger").len(), 0, "trigger row deleted");
        create(&mut cat, &mut pager, "CREATE TABLE trg(x)");
        assert!(cat.table("trg").unwrap().is_some(), "trg is now a table (name was freed)");
    }

    #[test]
    fn drop_table_cascade_is_scoped_to_the_target_table() {
        // The cascade predicate must delete ONLY the dropped table's dependents. With two
        // tables each carrying its own index AND trigger, DROP TABLE t1 removes t1's row
        // plus t1's index and trigger, and leaves every t2 dependent intact — checked via
        // the cache, physically on page 1, and after a reload. A predicate that lost its
        // `norm(tbl_name) == key` scope would take t2's dependents too and fail here.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t1(a)");
        create(&mut cat, &mut pager, "CREATE TABLE t2(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i1 ON t1(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX i2 ON t2(a)");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER g1 AFTER INSERT ON t1 BEGIN SELECT 1; END");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER g2 AFTER INSERT ON t2 BEGIN SELECT 1; END");
        // 2 tables + 2 indexes + 2 triggers (neither table has an auto-index) = 6 rows.
        assert_eq!(page1_row_count(&pager), 6);

        drop_stmt(&mut cat, &mut pager, "DROP TABLE t1");

        // t1 and exactly its dependents are gone from the cache.
        assert!(cat.table("t1").unwrap().is_none(), "the dropped table is gone");
        assert!(cat.index("i1").unwrap().is_none(), "t1's index cascaded away");
        assert!(cat.indexes_on("t1").unwrap().is_empty(), "t1's index links are cleared");
        // t2 and ALL of its dependents survive the cascade.
        assert!(cat.table("t2").unwrap().is_some(), "the untargeted table survives");
        assert!(cat.index("i2").unwrap().is_some(), "t2's index is NOT cascaded");
        assert_eq!(cat.indexes_on("t2").unwrap().len(), 1, "t2 keeps exactly its own index link");

        // Physically: only t1's three rows were deleted; one of each type remains (t2/i2/g2).
        assert_eq!(page1_row_count(&pager), 3, "only t1's table + index + trigger rows were deleted");
        assert_eq!(rows_of_type(&pager, "index").len(), 1, "only i2's index row remains");
        assert_eq!(rows_of_type(&pager, "trigger").len(), 1, "only g2's trigger row remains");

        // A reload from page 1 alone agrees: t2 intact, t1 absent.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert!(reloaded.table("t1").unwrap().is_none(), "t1 absent after reload");
        assert!(reloaded.table("t2").unwrap().is_some(), "t2 present after reload");
        assert!(reloaded.index("i1").unwrap().is_none(), "t1's index absent after reload");
        assert!(reloaded.index("i2").unwrap().is_some(), "t2's index present after reload");
    }

    #[test]
    fn drop_error_messages_match_sqlite_wording() {
        // The wording is part of the observable SQLite contract, so pin the substrings
        // that encode a behavioral distinction (reserved-vs-absent, constraint-index, and
        // the per-kind "no such <kind>") rather than only that it is some Error::Sql.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a UNIQUE)");

        let cases: &[(&str, &str)] = &[
            // A reserved name is "may not be dropped", NOT "no such table".
            ("DROP TABLE sqlite_schema", "may not be dropped"),
            // An auto-index backs a constraint: "cannot be dropped", NOT "no such index".
            ("DROP INDEX sqlite_autoindex_t_1", "cannot be dropped"),
            // Absent objects report a per-kind "no such <kind>".
            ("DROP TABLE ghost", "no such table"),
            ("DROP INDEX ghost", "no such index"),
            ("DROP VIEW ghost", "no such view"),
            ("DROP TRIGGER ghost", "no such trigger"),
        ];
        for (sql, needle) in cases {
            let err = try_drop(&mut cat, &mut pager, sql).unwrap_err();
            match err {
                Error::Sql(m) => {
                    assert!(m.contains(needle), "`{sql}` -> {m:?}, expected to contain {needle:?}")
                }
                other => panic!("`{sql}` expected Error::Sql, got {other:?}"),
            }
        }
    }

    #[test]
    fn load_round_trip_repopulates_namespace_and_rowid() {
        // 8: a table + a view + a trigger survive a fresh reload (namespace repopulated),
        // and next_schema_rowid continues past every row (a later create appends).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_view(&mut cat, &mut pager, "CREATE VIEW v AS SELECT a FROM t");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        let rows_before = page1_row_count(&pager);

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        let e = try_create(&mut reloaded, &mut pager, "CREATE TABLE v(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "view name held after reload, got {e:?}");
        let e = try_create(&mut reloaded, &mut pager, "CREATE TABLE trg(x)").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "trigger name held after reload, got {e:?}");
        assert_eq!(page1_row_count(&pager), rows_before, "rejected creates persisted nothing");

        // A distinct new table appends a row (rowid continues past all rows, no REPLACE).
        create(&mut reloaded, &mut pager, "CREATE TABLE w(x)");
        assert_eq!(page1_row_count(&pager), rows_before + 1, "the new table appended a row");
        assert!(reloaded.table("t").unwrap().is_some(), "existing rows untouched");
        assert!(reloaded.table("w").unwrap().is_some());
    }

    /// Seed `rows` onto page 1 (rowids 1..) of a fresh pager and try to load a fresh
    /// catalog from it, returning the load result (for the fail-closed load tests).
    fn load_rows(rows: &[SchemaRow]) -> Result<()> {
        let mut pager = fresh_pager();
        for (i, row) in rows.iter().enumerate() {
            table_insert(&mut pager, SCHEMA_ROOT, (i + 1) as i64, &row.to_record()).unwrap();
        }
        let mut cat = SchemaCatalog::new();
        cat.load(&pager)
    }

    fn view_row(name: &str, rootpage: i64, sql: Option<&str>) -> SchemaRow {
        SchemaRow {
            obj_type: "view".into(),
            name: name.into(),
            tbl_name: name.into(),
            rootpage,
            sql: sql.map(str::to_string),
        }
    }

    fn table_row(name: &str) -> SchemaRow {
        SchemaRow {
            obj_type: "table".into(),
            name: name.into(),
            tbl_name: name.into(),
            rootpage: 2,
            sql: Some(format!("CREATE TABLE {name}(a)")),
        }
    }

    fn trigger_row(name: &str, target: &str, rootpage: i64, sql: Option<&str>) -> SchemaRow {
        SchemaRow {
            obj_type: "trigger".into(),
            name: name.into(),
            tbl_name: target.into(),
            rootpage,
            sql: sql.map(str::to_string),
        }
    }

    #[test]
    fn load_fails_closed_on_bad_view_rows() {
        // 9 (views): a view row with rootpage != 0, NULL sql, or sql that is not a
        // CREATE VIEW is corrupt schema and fails closed as Format.
        let bad_root = view_row("v", 2, Some("CREATE VIEW v AS SELECT 1"));
        assert!(
            matches!(load_rows(&[bad_root]).unwrap_err(), Error::Format(_)),
            "a view rootpage != 0 must fail closed"
        );

        let null_sql = view_row("v", 0, None);
        assert!(
            matches!(load_rows(&[null_sql]).unwrap_err(), Error::Format(_)),
            "a view with NULL sql must fail closed"
        );

        let not_view = view_row("v", 0, Some("CREATE TABLE v(a)"));
        assert!(
            matches!(load_rows(&[not_view]).unwrap_err(), Error::Format(_)),
            "a view whose sql is not a CREATE VIEW must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_bad_trigger_rows() {
        // 9 (triggers): rootpage != 0, NULL sql, sql that is not a CREATE TRIGGER, or a
        // missing target table all fail closed as Format. A table `t` is seeded so only
        // the specific defect (not an absent target) causes each of the first three.
        let t = table_row("t");

        let bad_root = trigger_row("trg", "t", 2, Some("CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END"));
        assert!(
            matches!(load_rows(&[t.clone(), bad_root]).unwrap_err(), Error::Format(_)),
            "a trigger rootpage != 0 must fail closed"
        );

        let null_sql = trigger_row("trg", "t", 0, None);
        assert!(
            matches!(load_rows(&[t.clone(), null_sql]).unwrap_err(), Error::Format(_)),
            "a trigger with NULL sql must fail closed"
        );

        let not_trg = trigger_row("trg", "t", 0, Some("CREATE VIEW trg AS SELECT 1"));
        assert!(
            matches!(load_rows(&[t.clone(), not_trg]).unwrap_err(), Error::Format(_)),
            "a trigger whose sql is not a CREATE TRIGGER must fail closed"
        );

        // Missing target table: no table seeded, so the ON-table lookup fails.
        let missing_target =
            trigger_row("trg", "gone", 0, Some("CREATE TRIGGER trg AFTER INSERT ON gone BEGIN SELECT 1; END"));
        assert!(
            matches!(load_rows(&[missing_target]).unwrap_err(), Error::Format(_)),
            "a trigger whose target table is absent must fail closed"
        );
    }

    // --- load-time shared-namespace collisions across ALL loaders --------------
    // Every loader (table/index/auto-index/view/trigger) consults `name_in_use`, so a
    // corrupt/foreign image that puts one name in two namespaces fails closed no matter
    // which pair collides or which row loads second. Real sqlite never writes such an
    // image, but the invariant `name_in_use`'s doc asserts must hold on the load side too.

    #[test]
    fn load_fails_closed_on_index_colliding_with_a_view() {
        // A view `x` (cached in pass 1) and an explicit index `x` (pass 2): the index
        // loader consults name_in_use (all four caches), so `x` cannot silently land in
        // BOTH self.views and self.indexes.
        let t = table_row("t");
        let v = view_row("x", 0, Some("CREATE VIEW x AS SELECT a FROM t"));
        let idx = SchemaRow {
            obj_type: "index".into(),
            name: "x".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: Some("CREATE INDEX x ON t(a)".into()),
        };
        assert!(
            matches!(load_rows(&[t, v, idx]).unwrap_err(), Error::Format(_)),
            "an index colliding with a view must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_auto_index_colliding_with_a_view() {
        // A view stealing an auto-index's name: the NULL-sql auto-index row (pass 2) must
        // fail closed against the view cached in pass 1. The table legitimately implies
        // the auto-index (UNIQUE(a)), so the ONLY defect is the name collision — this
        // isolates load_auto_index_row's namespace guard, which runs before the spec
        // lookup.
        let t = SchemaRow {
            obj_type: "table".into(),
            name: "t".into(),
            tbl_name: "t".into(),
            rootpage: 2,
            sql: Some("CREATE TABLE t(a UNIQUE)".into()),
        };
        let v = view_row(
            "sqlite_autoindex_t_1",
            0,
            Some("CREATE VIEW sqlite_autoindex_t_1 AS SELECT 1"),
        );
        let auto = SchemaRow {
            obj_type: "index".into(),
            name: "sqlite_autoindex_t_1".into(),
            tbl_name: "t".into(),
            rootpage: 3,
            sql: None,
        };
        assert!(
            matches!(load_rows(&[t, v, auto]).unwrap_err(), Error::Format(_)),
            "an auto-index colliding with a view must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_table_colliding_with_a_view() {
        // Order-dependent variant proving load_table_row now has a namespace guard: a
        // view `x` row precedes a table `x` row (both pass 1). The view caches first, then
        // the table load fails closed (previously it had no check and cached both).
        let v = view_row("x", 0, Some("CREATE VIEW x AS SELECT 1"));
        let t = table_row("x");
        assert!(
            matches!(load_rows(&[v, t]).unwrap_err(), Error::Format(_)),
            "a table colliding with a view must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_duplicate_table_rows() {
        // name_in_use is consulted before each insert, so a second table row of the same
        // name fails closed rather than silently overwriting the first (last-write-wins).
        // A valid image never carries duplicate names.
        let first = table_row("t");
        let second = table_row("t");
        assert!(
            matches!(load_rows(&[first, second]).unwrap_err(), Error::Format(_)),
            "a duplicate table row must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_view_name_disagreeing_with_sql() {
        // The view row's `name` column must match the parsed CREATE VIEW name (pins the
        // defensive name-agreement branch in load_view_row).
        let bad = view_row("v", 0, Some("CREATE VIEW other AS SELECT 1"));
        assert!(
            matches!(load_rows(&[bad]).unwrap_err(), Error::Format(_)),
            "a view whose row name disagrees with its sql name must fail closed"
        );
    }

    #[test]
    fn load_fails_closed_on_trigger_tbl_name_disagreeing_with_target() {
        // The trigger row's `tbl_name` must match the parsed CREATE TRIGGER target table
        // (pins the defensive tbl_name-vs-target branch in load_trigger_row; the mismatch
        // is caught before the target-existence check, so seeding only `t` suffices).
        let t = table_row("t");
        let bad =
            trigger_row("trg", "t", 0, Some("CREATE TRIGGER trg AFTER INSERT ON u BEGIN SELECT 1; END"));
        assert!(
            matches!(load_rows(&[t, bad]).unwrap_err(), Error::Format(_)),
            "a trigger whose tbl_name disagrees with its sql target must fail closed"
        );
    }

    // --- ALTER TABLE -----------------------------------------------------------

    /// Parse `sql` (one ALTER TABLE) and run it through the real catalog path.
    fn try_alter(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<()> {
        let ast = parse(sql).unwrap();
        let Statement::AlterTable(stmt) = &ast.statements[0] else {
            panic!("not an ALTER TABLE: {sql}");
        };
        cat.alter_table(pager, stmt, sql)
    }

    fn alter(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
        try_alter(cat, pager, sql).unwrap();
    }

    /// The verbatim stored `sql` of the `type='table'` row named `name` (page 1).
    fn table_sql(pager: &MemPager, name: &str) -> Option<String> {
        let mut cursor = TableCursor::open(pager, SCHEMA_ROOT).unwrap();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            let row = SchemaRow::from_values(&decode_record(&cursor.payload().unwrap())).unwrap();
            if row.obj_type == "table" && row.name.eq_ignore_ascii_case(name) {
                return row.sql;
            }
            positioned = cursor.next().unwrap();
        }
        None
    }

    /// A cached table def's column names, in order.
    fn col_names(cat: &SchemaCatalog, name: &str) -> Vec<String> {
        cat.table(name).unwrap().unwrap().columns.iter().map(|c| c.name.clone()).collect()
    }

    #[test]
    fn add_column_updates_cache_persists_and_reloads() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER, b TEXT)");
        let cookie_before = read_schema_cookie(&pager);

        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c REAL");

        // Cache gained the column with the right name/type.
        assert_eq!(col_names(&cat, "t"), ["a", "b", "c"]);
        assert_eq!(
            cat.table("t").unwrap().unwrap().columns[2].declared_type.as_deref(),
            Some("REAL")
        );
        // Stored sql contains the new column and the cookie bumped exactly once.
        assert!(table_sql(&pager, "t").unwrap().contains("c REAL"), "stored sql keeps the new column");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "one cookie bump");

        // Drop the cache and reload from page 1: the reloaded def must match (round-trip).
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "b", "c"], "reloaded def matches after ADD COLUMN");
        assert_eq!(
            cat.table("t").unwrap().unwrap().columns[2].declared_type.as_deref(),
            Some("REAL")
        );
    }

    #[test]
    fn add_column_with_default_round_trips() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 5");
        assert_eq!(cat.table("t").unwrap().unwrap().columns[1].default.as_deref(), Some("5"));
        cat.load(&pager).unwrap();
        assert_eq!(cat.table("t").unwrap().unwrap().columns[1].default.as_deref(), Some("5"));
    }

    #[test]
    fn add_column_splices_before_the_right_paren_with_nested_parens() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a DECIMAL(10,2), CHECK(a > 0))");
        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN b TEXT");
        // The new column goes AFTER the last column def but BEFORE the table-level
        // CHECK — a column def after a table constraint is a syntax error real sqlite
        // rejects when it re-reads the schema. The nested `(10,2)`/`(a > 0)` parens
        // must not confuse the scan.
        assert_eq!(
            table_sql(&pager, "t").unwrap(),
            "CREATE TABLE t(a DECIMAL(10,2), b TEXT, CHECK(a > 0))"
        );
        // The CHECK is a table-level constraint, so the columns are just [a, b].
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "b"]);
    }

    #[test]
    fn add_column_lands_before_table_constraints() {
        // Witness for the ordering bug: a table with a trailing table constraint must
        // keep that constraint LAST after ADD COLUMN, or the stored `sqlite_schema.sql`
        // is a column-def-after-table-constraint that real sqlite refuses to re-parse
        // ("malformed database schema"). Covers PK / UNIQUE / CHECK, each reloading.
        for (create_sql, expect) in [
            (
                "CREATE TABLE t(a, b, PRIMARY KEY(a))",
                "CREATE TABLE t(a, b, c, PRIMARY KEY(a))",
            ),
            (
                "CREATE TABLE t(a, b, UNIQUE(a))",
                "CREATE TABLE t(a, b, c, UNIQUE(a))",
            ),
            (
                "CREATE TABLE t(a, b, CHECK(a > 0))",
                "CREATE TABLE t(a, b, c, CHECK(a > 0))",
            ),
        ] {
            let mut pager = fresh_pager();
            let mut cat = SchemaCatalog::new();
            create(&mut cat, &mut pager, create_sql);
            alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c");
            assert_eq!(table_sql(&pager, "t").unwrap(), expect, "from {create_sql:?}");
            // Reload from page 1: the rewritten schema is well-formed for our loader,
            // and the new column `c` is present in order (columns precede constraints).
            cat.load(&pager).unwrap();
            assert_eq!(col_names(&cat, "t"), ["a", "b", "c"], "from {create_sql:?}");
        }
    }

    #[test]
    fn add_column_after_column_level_pk_still_appends_at_end() {
        // A column-level PRIMARY KEY is NOT a table constraint, so the new column is
        // simply appended before the `)` — and the result stays valid sqlite.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c REAL");
        assert_eq!(
            table_sql(&pager, "t").unwrap(),
            "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c REAL)"
        );
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "b", "c"]);
    }

    #[test]
    fn add_column_restrictions_are_rejected_and_persist_nothing() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER, b TEXT)");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        for bad in [
            "ALTER TABLE t ADD COLUMN c INTEGER PRIMARY KEY",
            "ALTER TABLE t ADD COLUMN c INTEGER UNIQUE",
            "ALTER TABLE t ADD COLUMN c INTEGER NOT NULL",
            "ALTER TABLE t ADD COLUMN c NOT NULL DEFAULT NULL",
            "ALTER TABLE t ADD COLUMN c DEFAULT CURRENT_TIMESTAMP",
            "ALTER TABLE t ADD COLUMN c DEFAULT (1 + 2)",
            "ALTER TABLE t ADD COLUMN a INTEGER",
            "ALTER TABLE t ADD COLUMN c INTEGER GENERATED ALWAYS AS (a + 1) STORED",
        ] {
            let e = try_alter(&mut cat, &mut pager, bad).unwrap_err();
            assert!(matches!(e, Error::Sql(_)), "{bad:?} -> Sql, got {e:?}");
        }
        // No rejected ALTER wrote a row, bumped the cookie, or changed the column set.
        assert_eq!(page1_row_count(&pager), rows_before, "no rejected ALTER wrote a row");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no cookie bump on rejection");
        assert_eq!(col_names(&cat, "t"), ["a", "b"], "columns unchanged after rejections");
    }

    #[test]
    fn add_column_error_messages_match_sqlite_wording() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        let cases = [
            ("ALTER TABLE t ADD COLUMN A INT", "duplicate column name: A"),
            ("ALTER TABLE t ADD COLUMN c PRIMARY KEY", "Cannot add a PRIMARY KEY column"),
            ("ALTER TABLE t ADD COLUMN c UNIQUE", "Cannot add a UNIQUE column"),
            ("ALTER TABLE t ADD COLUMN c INT NOT NULL", "Cannot add a NOT NULL column with default value NULL"),
            ("ALTER TABLE t ADD COLUMN c DEFAULT CURRENT_DATE", "Cannot add a column with non-constant default"),
            ("ALTER TABLE t ADD COLUMN c GENERATED ALWAYS AS (a) STORED", "cannot add a STORED column"),
        ];
        for (sql, want) in cases {
            match try_alter(&mut cat, &mut pager, sql).unwrap_err() {
                Error::Sql(m) => assert_eq!(m, want, "{sql:?}"),
                other => panic!("{sql:?} -> {other:?}"),
            }
        }
    }

    #[test]
    fn add_column_virtual_generated_is_allowed() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER)");
        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c GENERATED ALWAYS AS (a + 1) VIRTUAL");
        assert_eq!(col_names(&cat, "t"), ["a", "c"], "VIRTUAL generated column is allowed");
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "c"]);
    }

    #[test]
    fn add_column_to_strict_table_enforces_the_datatype_rules() {
        // ADD COLUMN rebuilds the whole table def from the rewritten `CREATE ... STRICT`
        // sql through `table_def_from_ast`, so the SAME create-time STRICT datatype check
        // rejects a new column with no type or a disallowed type — proving the single
        // validation site covers the ADD-COLUMN-to-STRICT path for free. `check_add_column_
        // allowed` does not look at the type, so the rejection is the builder's; it fires
        // before the persist, so page 1 and the schema cookie are untouched.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a INT) STRICT");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        for (sql, want) in [
            ("ALTER TABLE t ADD COLUMN c", "missing datatype for t.c"),
            ("ALTER TABLE t ADD COLUMN c FLOAT", "unknown datatype for t.c: \"FLOAT\""),
        ] {
            match try_alter(&mut cat, &mut pager, sql).unwrap_err() {
                Error::Sql(m) => assert_eq!(m, want, "{sql:?}"),
                other => panic!("{sql:?} -> {other:?}"),
            }
        }
        assert_eq!(page1_row_count(&pager), rows_before, "a rejected ALTER persists nothing");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "a rejected ALTER bumps no cookie");
        assert_eq!(col_names(&cat, "t"), ["a"], "the cache still has only the original column");

        // A valid-typed ADD COLUMN on the STRICT table succeeds and reloads cleanly.
        alter(&mut cat, &mut pager, "ALTER TABLE t ADD COLUMN c TEXT");
        assert_eq!(col_names(&cat, "t"), ["a", "c"]);
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();
        assert_eq!(col_names(&reloaded, "t"), ["a", "c"]);
    }

    #[test]
    fn strict_table_round_trips_through_load() {
        // A legitimately-written STRICT table must reload. Load re-parses the stored
        // verbatim `sql` through the SAME `table_def_from_ast` that now re-validates the
        // STRICT datatypes, so this proves the create-time check does NOT reject a valid
        // STRICT table on reopen — real sqlite never persists a table these rules reject, so
        // a foreign STRICT `.db` cross-read here reloads all the same. STRICT + WITHOUT ROWID
        // is exercised too (both trailing options survive the round-trip).
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a INTEGER, b TEXT, c REAL, d BLOB, e ANY) STRICT",
        );
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE w(k INTEGER PRIMARY KEY, v TEXT) STRICT, WITHOUT ROWID",
        );

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).unwrap();

        assert_eq!(col_names(&reloaded, "t"), ["a", "b", "c", "d", "e"]);
        let w = reloaded.table("w").unwrap().expect("STRICT WITHOUT ROWID table reloaded");
        assert!(w.without_rowid);
        assert_eq!(col_names(&reloaded, "w"), ["k", "v"]);
    }

    #[test]
    fn rejected_create_table_persists_nothing() {
        // A CREATE TABLE the create-time structural rules reject must write no page-1 row and
        // bump no schema cookie: `table_def_from_ast` validates BEFORE `create_table` reaches
        // `insert_schema_row` / `bump_schema_cookie`, so the whole statement is a no-op (the
        // pre-allocated b-tree root is discarded on the enclosing transaction's rollback in
        // the engine). This gives the create path the same persist-nothing guarantee the
        // ADD-COLUMN-to-STRICT test proves on the ALTER path, across all five rules.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        for bad in [
            "CREATE TABLE t(a, a)",                         // duplicate column
            "CREATE TABLE t(a PRIMARY KEY, b PRIMARY KEY)", // more than one PRIMARY KEY
            "CREATE TABLE t(a, PRIMARY KEY(b))",            // table constraint unknown column
            "CREATE TABLE t(a FLOAT) STRICT",               // STRICT unknown datatype
            "CREATE TABLE t(a, b) WITHOUT ROWID",           // WITHOUT ROWID missing PRIMARY KEY
        ] {
            let e = try_create(&mut cat, &mut pager, bad).unwrap_err();
            assert!(matches!(e, Error::Sql(_)), "{bad:?} -> Sql, got {e:?}");
            assert!(cat.table("t").unwrap().is_none(), "{bad:?}: a rejected CREATE cached no table");
        }
        assert_eq!(page1_row_count(&pager), rows_before, "no rejected CREATE wrote a page-1 row");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no rejected CREATE bumped the cookie");
    }

    #[test]
    fn add_column_rejects_reserved_and_missing_tables() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        let e = try_alter(&mut cat, &mut pager, "ALTER TABLE nope ADD COLUMN x INT").unwrap_err();
        assert!(matches!(&e, Error::Sql(m) if m == "no such table: nope"), "got {e:?}");
        let e =
            try_alter(&mut cat, &mut pager, "ALTER TABLE sqlite_master ADD COLUMN x INT").unwrap_err();
        assert!(matches!(e, Error::Sql(_)), "altering a sqlite_ table -> Sql, got {e:?}");
    }

    // --- RENAME TO -------------------------------------------------------------

    #[test]
    fn rename_table_cascades_indexes_and_reloads_cleanly() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        // `b UNIQUE` gives an auto-index (sqlite_autoindex_t_1); `ix` is explicit.
        create(&mut cat, &mut pager, "CREATE TABLE t(a INTEGER, b TEXT UNIQUE)");
        create_index(&mut cat, &mut pager, "CREATE INDEX ix ON t(a)");
        let old_root = cat.table("t").unwrap().unwrap().root_page;
        let cookie_before = read_schema_cookie(&pager);

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME TO t2");

        // Old name gone; new name resolves and keeps the same b-tree root (data intact).
        assert!(cat.table("t").unwrap().is_none(), "old name is gone");
        assert_eq!(cat.table("t2").unwrap().unwrap().root_page, old_root, "root preserved");

        // Explicit index: target table updated, its own name kept.
        assert_eq!(cat.index("ix").unwrap().unwrap().table, "t2");
        // Auto-index: renamed to sqlite_autoindex_t2_1; the old auto name is gone.
        assert!(cat.index("sqlite_autoindex_t_1").unwrap().is_none(), "old auto-index name gone");
        let auto = cat.index("sqlite_autoindex_t2_1").unwrap().unwrap();
        assert_eq!(auto.table, "t2");
        assert_eq!(auto.columns, ["b"]);

        // indexes_on lists under the new table in creation order (auto, then ix).
        let listed: Vec<String> =
            cat.indexes_on("t2").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["sqlite_autoindex_t2_1", "ix"]);
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "one cookie bump for the rename");
        assert!(table_sql(&pager, "t2").unwrap().starts_with("CREATE TABLE t2"));

        // Reload from page 1: a missing index cascade would make THIS fail, because
        // `load` validates that each index row's target table exists.
        cat.load(&pager).unwrap();
        assert!(cat.table("t").unwrap().is_none());
        assert!(cat.table("t2").unwrap().is_some());
        assert_eq!(cat.index("ix").unwrap().unwrap().table, "t2");
        assert_eq!(cat.index("sqlite_autoindex_t2_1").unwrap().unwrap().table, "t2");
        let listed: Vec<String> =
            cat.indexes_on("t2").unwrap().iter().map(|d| d.name.clone()).collect();
        assert_eq!(listed, ["sqlite_autoindex_t2_1", "ix"], "index order survives reload");
    }

    #[test]
    fn rename_index_rows_on_page_one_point_at_the_new_table() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b UNIQUE)");
        create_index(&mut cat, &mut pager, "CREATE INDEX ix ON t(a)");
        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME TO t2");

        for row in index_rows(&pager) {
            let name = match &row[1] {
                Value::Text(s) => s.clone(),
                v => panic!("index name column: {v:?}"),
            };
            let tbl = match &row[2] {
                Value::Text(s) => s.clone(),
                v => panic!("index tbl_name column: {v:?}"),
            };
            assert_eq!(tbl, "t2", "index {name} must point at the renamed table");
            match &row[4] {
                Value::Text(sql) => assert!(sql.contains("ON t2"), "explicit index sql: {sql}"),
                Value::Null => assert_eq!(name, "sqlite_autoindex_t2_1", "auto-index row renamed"),
                v => panic!("index sql column: {v:?}"),
            }
        }
    }

    #[test]
    fn rename_errors_leave_schema_unchanged() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create(&mut cat, &mut pager, "CREATE TABLE u(a)");
        create_index(&mut cat, &mut pager, "CREATE INDEX ix ON t(a)");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        for bad in [
            "ALTER TABLE t RENAME TO u",         // onto an existing table
            "ALTER TABLE t RENAME TO ix",        // onto an existing index (shared namespace)
            "ALTER TABLE t RENAME TO t",         // onto its own name (still "in use")
            "ALTER TABLE nope RENAME TO x",      // missing source
            "ALTER TABLE t RENAME TO sqlite_x",  // reserved target name
            "ALTER TABLE sqlite_master RENAME TO x", // reserved source name
        ] {
            let e = try_alter(&mut cat, &mut pager, bad).unwrap_err();
            assert!(matches!(e, Error::Sql(_)), "{bad:?} -> Sql, got {e:?}");
        }
        assert_eq!(page1_row_count(&pager), rows_before, "no rejected rename changed page 1");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no cookie bump on rejection");
        assert!(cat.table("t").unwrap().is_some(), "t still exists after failed renames");
    }

    #[test]
    fn rename_missing_table_uses_sqlite_wording() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        let e = try_alter(&mut cat, &mut pager, "ALTER TABLE nope RENAME TO x").unwrap_err();
        assert!(matches!(&e, Error::Sql(m) if m == "no such table: nope"), "got {e:?}");
    }

    #[test]
    fn rename_table_cascades_dependent_triggers_and_reloads_cleanly() {
        // Regression: a trigger's page-1 row records its TARGET table both as `tbl_name`
        // and inside its stored `ON <table>`. Before the trigger cascade, RENAME TO left
        // both at the OLD name, so on the next open `load_trigger_row` fail-closed (the
        // target table was gone) and the WHOLE database failed to reload (Error::Format).
        // The rename must move `tbl_name` AND the `ON` token so the schema reloads.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_trigger(&mut cat, &mut pager, &trg_on("t"));
        let cookie_before = read_schema_cookie(&pager);
        // The rewritten trigger text is exactly `trg_on` against the new name.
        let want_sql = trg_on("t2");

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME TO t2");

        // One cookie bump for the whole rename (table + trigger rows share it).
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "one cookie bump for the rename");

        // In-session cache: the trigger keeps its name key but now targets t2 with the
        // rewritten sql.
        let def = cat.triggers.get(&norm("trg")).expect("trigger still cached under its name");
        assert_eq!(norm(&def.table), norm("t2"), "cached TriggerDef.table follows the rename");
        assert_eq!(def.sql, want_sql, "cached trigger sql rewritten to target t2");

        // The page-1 trigger row moved too: tbl_name = t2 and its sql says ON t2.
        let rows = rows_of_type(&pager, "trigger");
        assert_eq!(rows.len(), 1, "still exactly one trigger row");
        assert!(matches!(&rows[0][2], Value::Text(s) if s == "t2"), "trigger tbl_name moved to t2");
        assert!(
            matches!(&rows[0][4], Value::Text(sql) if sql == &want_sql),
            "stored trigger sql rewritten to target t2",
        );

        // The whole point: a FRESH catalog loading only from page 1 SUCCEEDS (this
        // returned Error::Format before the cascade), and the reloaded trigger targets
        // t2 while the old table name is gone.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("reload must succeed after the trigger cascade");
        assert!(reloaded.table("t2").unwrap().is_some(), "renamed table present after reload");
        assert!(reloaded.table("t").unwrap().is_none(), "old table name absent after reload");
        let rdef = reloaded.triggers.get(&norm("trg")).expect("trigger reloaded under its name");
        assert_eq!(norm(&rdef.table), norm("t2"), "reloaded TriggerDef.table is the new table");
        assert_eq!(rdef.sql, want_sql, "reloaded trigger sql targets the new table");

        // An in-session DROP TABLE t2 still cascades the trigger from BOTH the cache and
        // page 1 — which only works because the cached target moved to t2.
        drop_stmt(&mut cat, &mut pager, "DROP TABLE t2");
        assert!(cat.triggers.get(&norm("trg")).is_none(), "trigger cascaded from the cache on DROP TABLE");
        assert_eq!(rows_of_type(&pager, "trigger").len(), 0, "trigger row cascaded from page 1");
    }

    #[test]
    fn rename_table_cascades_all_dependent_triggers_and_leaves_unrelated_ones_alone() {
        // Guards two regressions the single-trigger test cannot: (1) an OVER-BROAD
        // collect predicate (dropping the `norm(tbl_name) == old_key` clause) would
        // wrongly rewrite a trigger on an UNRELATED table; (2) a loop that rewrote only
        // the FIRST collected trigger would leave a second one stale. Both would still
        // pass the single-trigger test.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create(&mut cat, &mut pager, "CREATE TABLE other(b)");
        // Two triggers fire on `t`; one fires on `other`.
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER trg1 AFTER INSERT ON t BEGIN SELECT 1; END");
        create_trigger(&mut cat, &mut pager, "CREATE TRIGGER trg2 AFTER DELETE ON t BEGIN SELECT 2; END");
        let other_sql = "CREATE TRIGGER trg_other AFTER INSERT ON other BEGIN SELECT 3; END";
        create_trigger(&mut cat, &mut pager, other_sql);

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME TO t2");

        // Both triggers on `t` followed the rename (cache target + rewritten sql).
        for (name, event, body) in [("trg1", "INSERT", "1"), ("trg2", "DELETE", "2")] {
            let def = cat.triggers.get(&norm(name)).expect("trigger still cached");
            assert_eq!(norm(&def.table), norm("t2"), "{name} target follows the rename");
            assert_eq!(
                def.sql,
                format!("CREATE TRIGGER {name} AFTER {event} ON t2 BEGIN SELECT {body}; END"),
                "{name} sql rewritten to ON t2",
            );
        }
        // The unrelated trigger is untouched — target and stored sql unchanged.
        let od = cat.triggers.get(&norm("trg_other")).expect("unrelated trigger still cached");
        assert_eq!(norm(&od.table), norm("other"), "unrelated trigger target unchanged");
        assert_eq!(od.sql, other_sql, "unrelated trigger sql unchanged");

        // Page 1 agrees: two trigger rows now name t2, one still names other.
        let tbls: Vec<String> = rows_of_type(&pager, "trigger")
            .iter()
            .map(|r| match &r[2] {
                Value::Text(s) => s.clone(),
                v => panic!("trigger tbl_name column: {v:?}"),
            })
            .collect();
        assert_eq!(tbls.iter().filter(|s| s.as_str() == "t2").count(), 2, "two triggers moved to t2");
        assert_eq!(tbls.iter().filter(|s| s.as_str() == "other").count(), 1, "unrelated trigger stayed");

        // A fresh reload succeeds with all three triggers resolving their targets.
        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("reload must succeed with mixed dependent/unrelated triggers");
        assert_eq!(norm(&reloaded.triggers.get(&norm("trg1")).unwrap().table), norm("t2"));
        assert_eq!(norm(&reloaded.triggers.get(&norm("trg2")).unwrap().table), norm("t2"));
        assert_eq!(norm(&reloaded.triggers.get(&norm("trg_other")).unwrap().table), norm("other"));
    }

    #[test]
    fn rename_table_cascade_matches_trigger_target_case_insensitively() {
        // The collect predicate folds names with `norm`, and a trigger's stored
        // `ON <TABLE>` may differ in case from the table's own spelling. A trigger
        // created `ON T` must still cascade when `t` is renamed, and reload (which checks
        // ON<->tbl_name agreement case-insensitively) must succeed.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a)");
        create_trigger(&mut cat, &mut pager, &trg_on("T"));

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME TO t2");

        let def = cat.triggers.get(&norm("trg")).expect("trigger still cached");
        assert_eq!(norm(&def.table), norm("t2"), "case-differing target still cascaded");
        assert_eq!(def.sql, trg_on("t2"), "the ON token is rewritten to the new name");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("reload must succeed after a case-insensitive cascade");
        assert!(reloaded.table("t2").unwrap().is_some(), "renamed table present after reload");
        assert_eq!(norm(&reloaded.triggers.get(&norm("trg")).unwrap().table), norm("t2"));
    }

    #[test]
    fn rename_table_cascades_trigger_on_a_quoted_table_name_and_reloads() {
        // A trigger whose stored `ON` target is a QUOTED identifier (a table name that
        // needs quotes) must have that quoted token rewritten to the new name and still
        // round-trip through `load` — exercising the quote-aware token replacement and
        // the `ON`<->`tbl_name` agreement across the quoting boundary.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE \"t x\"(a)");
        create_trigger(&mut cat, &mut pager, &trg_on("\"t x\""));

        alter(&mut cat, &mut pager, "ALTER TABLE \"t x\" RENAME TO t2");

        let def = cat.triggers.get(&norm("trg")).expect("trigger still cached");
        assert_eq!(norm(&def.table), norm("t2"));
        assert_eq!(def.sql, trg_on("t2"), "the quoted ON target is rewritten to plain t2");

        let mut reloaded = SchemaCatalog::new();
        reloaded.load(&pager).expect("reload must succeed for a quoted-source-name cascade");
        assert!(reloaded.table("t2").unwrap().is_some(), "renamed table present after reload");
        assert!(reloaded.table("t x").unwrap().is_none(), "old quoted name absent after reload");
        assert_eq!(norm(&reloaded.triggers.get(&norm("trg")).unwrap().table), norm("t2"));
    }

    // --- RENAME COLUMN ---------------------------------------------------------

    #[test]
    fn rename_column_updates_def_stored_sql_and_reloads() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b TEXT)");
        let cookie_before = read_schema_cookie(&pager);

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO c");

        assert_eq!(col_names(&cat, "t"), ["a", "c"], "cache shows the renamed column");
        assert!(table_sql(&pager, "t").unwrap().contains("c TEXT"), "stored sql renamed the column");
        assert!(!table_sql(&pager, "t").unwrap().contains("b TEXT"), "old column name gone from sql");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "one cookie bump");

        // Reload from page 1: the rebuilt def must show the new name (round-trip).
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "c"], "reloaded def matches after RENAME COLUMN");
    }

    #[test]
    fn rename_column_cascades_into_table_level_check_and_foreign_key_and_reloads() {
        // A table-level CHECK and FOREIGN KEY column list that NAME the renamed column
        // must follow the rename in the STORED sql — end to end through
        // `alter_rename_column`, not just the pure `rename_column_in_create` unit tests.
        // The FK's REFERENCES target column (of another table) must be left untouched.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE u(x)");
        create(
            &mut cat,
            &mut pager,
            "CREATE TABLE t(a, b, CHECK(b > 0), FOREIGN KEY(b) REFERENCES u(x))",
        );

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO c");

        let sql = table_sql(&pager, "t").unwrap();
        assert!(sql.contains("CHECK(c > 0)"), "table-level CHECK follows the rename: {sql}");
        assert!(sql.contains("FOREIGN KEY(c)"), "FK column list follows the rename: {sql}");
        assert!(sql.contains("REFERENCES u(x)"), "the FK target column is untouched: {sql}");
        assert!(!sql.contains("(b)"), "no stale reference to the old column name: {sql}");

        // Reload from page 1: the whole rewritten body must round-trip to the new column
        // set. (A regression that renamed only the definition and left the CHECK/FK
        // naming `b` would still parse here, but the stored-sql asserts above pin the
        // cascade directly.)
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "c"], "reloaded def shows the renamed column");
    }

    #[test]
    fn rename_column_optional_column_keyword() {
        // `RENAME <from> TO <to>` (no COLUMN word) renames a column too.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME b TO c");
        assert_eq!(col_names(&cat, "t"), ["a", "c"]);
    }

    #[test]
    fn rename_column_cascades_to_explicit_index_and_reloads() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX ix ON t(b)");

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO c");

        // The cached index def follows the rename; its stored sql references `c`.
        assert_eq!(cat.index("ix").unwrap().unwrap().columns, ["c"]);
        let idx_row = index_rows(&pager)
            .into_iter()
            .find(|v| matches!(&v[1], Value::Text(s) if s == "ix"))
            .expect("the explicit index row is present");
        match &idx_row[4] {
            Value::Text(sql) => assert!(sql.contains("(c)"), "index sql renamed the column: {sql}"),
            v => panic!("index sql column: {v:?}"),
        }

        // Reload: an index left pointing at the old column name would make `load` fail
        // closed (unknown column). Clean round-trip proves the cascade.
        cat.load(&pager).unwrap();
        assert_eq!(cat.index("ix").unwrap().unwrap().columns, ["c"]);
        assert_eq!(col_names(&cat, "t"), ["a", "c"]);
    }

    #[test]
    fn rename_column_cascades_to_auto_index_columns_and_reloads() {
        // A UNIQUE column's auto-index derives its columns from the table's
        // constraints. Its name (sqlite_autoindex_t_1) does not change on RENAME COLUMN
        // (only the table name would change that), but its cached columns must follow.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b TEXT UNIQUE)");

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO c");

        let auto = cat.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(auto.columns, ["c"], "auto-index columns follow the rename");
        assert_eq!(auto.table, "t", "auto-index still points at the (unrenamed) table");

        cat.load(&pager).unwrap();
        assert_eq!(cat.index("sqlite_autoindex_t_1").unwrap().unwrap().columns, ["c"]);
    }

    #[test]
    fn rename_column_case_only_change_is_allowed() {
        // Renaming a column to a name that differs only in case is a rename of that
        // same column, not a collision with itself.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO B");
        assert_eq!(col_names(&cat, "t"), ["a", "B"]);
    }

    #[test]
    fn rename_column_errors_leave_schema_unchanged() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        for bad in [
            "ALTER TABLE t RENAME COLUMN nope TO z", // no such column
            "ALTER TABLE t RENAME COLUMN a TO b",    // onto an existing column
            "ALTER TABLE nope RENAME COLUMN a TO z", // missing table
            "ALTER TABLE sqlite_master RENAME COLUMN name TO x", // reserved table
        ] {
            let e = try_alter(&mut cat, &mut pager, bad).unwrap_err();
            assert!(matches!(e, Error::Sql(_)), "{bad:?} -> Sql, got {e:?}");
        }
        assert_eq!(page1_row_count(&pager), rows_before, "no rejected rename changed page 1");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no cookie bump on rejection");
        assert_eq!(col_names(&cat, "t"), ["a", "b"], "columns unchanged after rejections");
    }

    #[test]
    fn rename_column_error_messages_match_sqlite_wording() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b)");
        let e = try_alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN nope TO z").unwrap_err();
        assert!(matches!(&e, Error::Sql(m) if m == "no such column: nope"), "got {e:?}");
        let e = try_alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN a TO b").unwrap_err();
        assert!(matches!(&e, Error::Sql(m) if m == "duplicate column name: b"), "got {e:?}");
    }

    #[test]
    fn rename_column_preserves_row_data() {
        // RENAME COLUMN rewrites only the schema text — no data-b-tree writes — so an
        // existing row is byte-identical afterward, before and after a page-1 reload.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c)");
        let row = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        insert_row(&cat, &mut pager, "t", 1, &row);

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO renamed");

        assert_eq!(col_names(&cat, "t"), ["a", "renamed", "c"]);
        assert_data(&cat, &pager, "t", std::slice::from_ref(&row));
        cat.load(&pager).unwrap();
        assert_data(&cat, &pager, "t", std::slice::from_ref(&row));
    }

    #[test]
    fn rename_column_cascades_to_table_level_pk_auto_index() {
        // A table-level PRIMARY KEY(col) on a rowid table builds a UNIQUE auto-index
        // (sqlite_autoindex_t_1) whose columns derive from the constraint. Renaming the
        // key column must carry through to the rebuilt auto-index columns and reload —
        // the table-level-PK shape, distinct from the column-level UNIQUE case above.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, PRIMARY KEY(b))");

        alter(&mut cat, &mut pager, "ALTER TABLE t RENAME COLUMN b TO c");

        let auto = cat.index("sqlite_autoindex_t_1").unwrap().unwrap();
        assert_eq!(auto.columns, ["c"], "table-level PK auto-index columns follow the rename");
        cat.load(&pager).unwrap();
        assert_eq!(cat.index("sqlite_autoindex_t_1").unwrap().unwrap().columns, ["c"]);
    }

    // --- DROP COLUMN -----------------------------------------------------------

    /// Insert a raw data record (a row's stored column values) into table `name`'s
    /// data b-tree at `rowid`. Used to seed rows the DROP COLUMN data rewrite must fix.
    fn insert_row(cat: &SchemaCatalog, pager: &mut MemPager, name: &str, rowid: i64, values: &[Value]) {
        let root = cat.table(name).unwrap().unwrap().root_page;
        table_insert(pager, root, rowid, &encode_record(values)).unwrap();
    }

    /// Every stored data record of table `name`, decoded, in rowid order.
    fn data_rows(cat: &SchemaCatalog, pager: &MemPager, name: &str) -> Vec<Vec<Value>> {
        let root = cat.table(name).unwrap().unwrap().root_page;
        let mut cursor = TableCursor::open(pager, root).unwrap();
        let mut out = Vec::new();
        let mut positioned = cursor.first().unwrap();
        while positioned {
            out.push(decode_record(&cursor.payload().unwrap()));
            positioned = cursor.next().unwrap();
        }
        out
    }

    /// Assert table `name`'s stored records equal `want`. `Value` has no `PartialEq`
    /// (NULL and REAL lack a total equality), so compare the `Debug` rendering — exact
    /// for the Null / Integer records these tests use.
    fn assert_data(cat: &SchemaCatalog, pager: &MemPager, name: &str, want: &[Vec<Value>]) {
        let got = data_rows(cat, pager, name);
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "stored data rows of {name}");
    }

    #[test]
    fn drop_column_removes_middle_column_and_rewrites_row_data() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c)");
        insert_row(&cat, &mut pager, "t", 1, &[Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let cookie_before = read_schema_cookie(&pager);

        alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN b");

        assert_eq!(col_names(&cat, "t"), ["a", "c"], "cache drops the column");
        // Assert the *stored* sql reparses to exactly the survivors — a bare
        // `!contains('b')` would spuriously fire on any survivor whose name or type
        // embeds the letter (e.g. a `BLOB` column), so pin the reparsed column set.
        let stored = table_sql(&pager, "t").unwrap();
        let stored_cols: Vec<String> = table_def_from_sql(&stored, SCHEMA_ROOT)
            .unwrap()
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        assert_eq!(stored_cols, ["a", "c"], "stored sql reparses without the dropped column");
        assert_eq!(read_schema_cookie(&pager), cookie_before + 1, "one cookie bump");
        // The middle column's data is purged; the survivors stay aligned as [1, 3].
        assert_data(&cat, &pager, "t", &[vec![Value::Integer(1), Value::Integer(3)]]);

        // Reload from page 1: rebuilt def and data still agree (on-disk round trip).
        cat.load(&pager).unwrap();
        assert_eq!(col_names(&cat, "t"), ["a", "c"], "reloaded def matches after DROP COLUMN");
        assert_data(&cat, &pager, "t", &[vec![Value::Integer(1), Value::Integer(3)]]);
    }

    #[test]
    fn drop_column_optional_column_keyword() {
        // `DROP <col>` (no COLUMN word) drops a column too.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c)");
        alter(&mut cat, &mut pager, "ALTER TABLE t DROP b");
        assert_eq!(col_names(&cat, "t"), ["a", "c"]);
    }

    #[test]
    fn drop_column_handles_short_post_add_column_records() {
        // A record materialized before later columns existed is "short" (fewer stored
        // values than the schema's column count). Dropping an unmaterialized trailing
        // column leaves it untouched; dropping a still-stored column removes its slot.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c)");
        // Only a, b materialized; c is implicitly NULL (drop_idx for c is past the end).
        insert_row(&cat, &mut pager, "t", 1, &[Value::Integer(1), Value::Integer(2)]);

        alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN c");
        assert_eq!(col_names(&cat, "t"), ["a", "b"]);
        assert_data(&cat, &pager, "t", &[vec![Value::Integer(1), Value::Integer(2)]]);

        // Now drop the materialized middle column b: its slot is removed -> [1].
        alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN b");
        assert_eq!(col_names(&cat, "t"), ["a"]);
        assert_data(&cat, &pager, "t", &[vec![Value::Integer(1)]]);
    }

    #[test]
    fn drop_column_before_rowid_alias_realigns_data_and_def() {
        // Dropping a column positioned BEFORE an INTEGER PRIMARY KEY shifts the alias
        // down one; the builder re-derives `rowid_alias` and the record's NULL alias
        // placeholder shifts with it, so the two stay aligned.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(x, id INTEGER PRIMARY KEY, c)");
        assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(1));
        // Stored record: [x, NULL(alias id), c], b-tree key (rowid) = 5.
        insert_row(&cat, &mut pager, "t", 5, &[Value::Integer(9), Value::Null, Value::Integer(7)]);

        alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN x");

        assert_eq!(col_names(&cat, "t"), ["id", "c"]);
        assert_eq!(
            cat.table("t").unwrap().unwrap().rowid_alias,
            Some(0),
            "alias index shifts down with the dropped leading column"
        );
        // Slot 0 dropped: [NULL(alias), 7], still keyed by rowid 5.
        assert_data(&cat, &pager, "t", &[vec![Value::Null, Value::Integer(7)]]);

        cat.load(&pager).unwrap();
        assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(0), "reload agrees");
    }

    #[test]
    fn drop_column_restrictions_are_rejected_and_persist_nothing() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE keyed(id INTEGER PRIMARY KEY, b)");
        create(&mut cat, &mut pager, "CREATE TABLE uniq(a UNIQUE, b)");
        create(&mut cat, &mut pager, "CREATE TABLE idx(a, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX ix ON idx(b)");
        create(&mut cat, &mut pager, "CREATE TABLE one(a)");
        create(&mut cat, &mut pager, "CREATE TABLE chk(a, b, CHECK(b > 0))");
        create(&mut cat, &mut pager, "CREATE TABLE gen(a, b, g AS (b + a))");
        create(&mut cat, &mut pager, "CREATE TABLE fk(a, b, FOREIGN KEY(b) REFERENCES keyed(id))");
        create(&mut cat, &mut pager, "CREATE TABLE part(a, b)");
        create_index(&mut cat, &mut pager, "CREATE INDEX pix ON part(a) WHERE b > 0");
        let rows_before = page1_row_count(&pager);
        let cookie_before = read_schema_cookie(&pager);

        // Each rejection is pinned to its REASON substring, not merely "some Sql error":
        // a "blocks for the WRONG reason" bug (e.g. a UNIQUE drop reported as a PK drop,
        // or a FK column mistaken for an index) passes a bare `is_err()` but fails here.
        // Substrings (not `==`) because some messages embed an object name (the index
        // cases) or the table/column name.
        for (bad, want) in [
            ("ALTER TABLE keyed DROP COLUMN id", "it is a PRIMARY KEY"), // INTEGER PRIMARY KEY
            ("ALTER TABLE uniq DROP COLUMN a", "it has a UNIQUE constraint"),
            ("ALTER TABLE idx DROP COLUMN b", "it is used by index \"ix\""), // explicit index
            ("ALTER TABLE one DROP COLUMN a", "no other columns exist"),     // the only column
            ("ALTER TABLE chk DROP COLUMN b", "it is used in a CHECK constraint"),
            ("ALTER TABLE gen DROP COLUMN b", "referenced by column \"g\""), // generated column
            ("ALTER TABLE fk DROP COLUMN b", "it is used in a FOREIGN KEY constraint"),
            ("ALTER TABLE part DROP COLUMN b", "it is used by index \"pix\""), // partial-WHERE
            ("ALTER TABLE idx DROP COLUMN nope", "no such column: nope"),
            ("ALTER TABLE nope DROP COLUMN a", "no such table: nope"),
            ("ALTER TABLE sqlite_master DROP COLUMN name", "may not be altered"), // reserved
        ] {
            let e = try_alter(&mut cat, &mut pager, bad).unwrap_err();
            assert!(
                matches!(&e, Error::Sql(m) if m.contains(want)),
                "{bad:?} -> Sql error containing {want:?}, got {e:?}"
            );
        }
        assert_eq!(page1_row_count(&pager), rows_before, "no rejected drop changed page 1");
        assert_eq!(read_schema_cookie(&pager), cookie_before, "no cookie bump on rejection");
    }

    #[test]
    fn drop_column_on_a_table_level_primary_key_member_is_rejected() {
        // A column named in a table-level composite PRIMARY KEY cannot be dropped.
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c, PRIMARY KEY(a, b))");
        assert!(try_alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN b").is_err());
        // A non-key column of the same table is still droppable.
        alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN c");
        assert_eq!(col_names(&cat, "t"), ["a", "b"]);
    }

    #[test]
    fn drop_column_on_without_rowid_table_is_refused() {
        let mut pager = fresh_pager();
        let mut cat = SchemaCatalog::new();
        create(&mut cat, &mut pager, "CREATE TABLE t(a, b, c, PRIMARY KEY(a)) WITHOUT ROWID");
        let e = try_alter(&mut cat, &mut pager, "ALTER TABLE t DROP COLUMN b").unwrap_err();
        assert!(
            matches!(&e, Error::Sql(m) if m.contains("WITHOUT ROWID")),
            "WITHOUT ROWID drop is an honest gap, got {e:?}"
        );
        assert_eq!(col_names(&cat, "t"), ["a", "b", "c"], "schema unchanged after refusal");
    }
}
