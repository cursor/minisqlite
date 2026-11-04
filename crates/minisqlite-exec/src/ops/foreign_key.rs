//! FOREIGN KEY enforcement — the CHILD side (`foreignkeys.html` §3, `PRAGMA
//! foreign_keys`).
//!
//! On INSERT or UPDATE of a row in a table that declares outgoing foreign keys, verify
//! each referenced PARENT key exists — the immediate check `PRAGMA foreign_keys=ON` turns
//! on. This is a no-op when the pragma is OFF (SQLite's default) or the table has no FK,
//! so a non-FK workload pays nothing.
//!
//! SCOPE (honest, deliberate):
//! * The CHILD side only: a new/updated child row must point at an existing parent row, or
//!   carry a NULL in any key column (MATCH SIMPLE — SQLite's only supported match mode:
//!   "if any of the child key columns are NULL, then there is no requirement for a
//!   corresponding row in the parent table").
//! * The PARENT side, ON DELETE — `NO ACTION` / `RESTRICT` reject (a surviving child
//!   reference raises `FOREIGN KEY constraint failed`); `CASCADE` deletes the matching
//!   child rows (maintaining their indexes and recursing so a cascaded child that is
//!   itself a parent cascades on to its own children); `SET NULL` rewrites each matching
//!   child, setting its FK columns to NULL (re-encoding the record and moving its index
//!   entries). `SET DEFAULT` sets the child FK columns to their column defaults and then
//!   re-checks the FK (a default that references no surviving parent row raises the
//!   violation — `foreignkeys.html` §4.3). `SET NULL` / `SET DEFAULT` are skipped (the
//!   child is left as-is) for a child that has generated columns (its STORED generated
//!   values would need recomputing from the rewritten row — machinery not wired here yet).
//!   A CASCADE/SET-NULL child rewrite does NOT yet fire the child's triggers, and (per
//!   `sqlite3_changes()`, which excludes FK-action rows) does not advance `changes()`.
//! * The PARENT side, ON UPDATE — when an UPDATE changes a REFERENCED parent key,
//!   `NO ACTION` / `RESTRICT` reject a surviving child reference; `CASCADE` copies the new
//!   key into the matching children's FK columns (recursing); `SET NULL` nulls them and
//!   `SET DEFAULT` resets them to their column defaults (then re-checks the FK), both skipped
//!   for a generated-column child. An UPDATE that leaves every referenced key unchanged fires
//!   no action (`foreignkeys.html` §4.3). Same trigger / `changes()` narrowing as ON DELETE.
//! * DEFERRED timing (`foreignkeys.html` §4.2). A `DEFERRABLE INITIALLY DEFERRED`
//!   constraint — or ANY constraint while `PRAGMA defer_foreign_keys` is ON — is checked at
//!   COMMIT, not at statement time, but ONLY while an explicit (or savepoint-started)
//!   transaction is open to defer into. The predicate is [`fk_deferred_now`]: when it holds,
//!   the child INSERT/UPDATE check, the parent-side NO ACTION rejection, and the parent-side
//!   SET DEFAULT re-check are all SKIPPED (the database is allowed to sit in violation
//!   mid-transaction), and [`check_deferred_foreign_keys`] re-verifies every deferred FK at
//!   COMMIT — one child-side full scan (ROWID and WITHOUT ROWID children alike) that catches a
//!   child orphan and a parent-side NO-ACTION / SET DEFAULT orphan alike. In AUTOCOMMIT no
//!   transaction is open, so `fk_deferred_now` is false and deferred behaves exactly like
//!   immediate (§4.2). RESTRICT is the exception: it is ALWAYS immediate, even on a deferred FK
//!   (§4.3), so it never routes through the deferral predicate. CASCADE and SET NULL fire
//!   immediately and cannot leave a child-side violation (CASCADE copies a live key; SET NULL
//!   is MATCH-SIMPLE exempt), so their timing is moot; SET DEFAULT fires its action
//!   immediately too, but its post-action FK re-check CAN fail (the default may reference no
//!   surviving parent), so — like NO ACTION — that rejection defers.
//! * The check runs BEFORE the row is written (like NOT NULL/CHECK here), so a row that
//!   references ITSELF by the same key (`INSERT INTO t(id,self) VALUES(1,1)` on a
//!   self-referential FK) is rejected where real sqlite accepts it (it writes then checks).
//!   The common self-ref shape — a NULL root plus children pointing at already-inserted
//!   parents — is unaffected (NULL satisfies, prior rows are present).
//! * WITHOUT ROWID narrowings. A WR table's rows live in a PK-index b-tree, not a rowid
//!   table, so a rowid-table scan cannot decode it. Two paths therefore fail CLOSED (a loud
//!   "not supported" error, never a silent pass or skip): (a) an FK whose PARENT is WITHOUT
//!   ROWID — the child-side existence check [`parent_key_exists`] cannot scan the parent; and
//!   (b) the IMMEDIATE parent-side ON DELETE/UPDATE action on a WITHOUT ROWID CHILD —
//!   [`enforce_parent_delete`] / [`enforce_parent_update`] cannot scan the child to
//!   reject/cascade/null it, so they OVER-reject: they error whenever a WR child merely
//!   DECLARES the FK, even for a parent key no child row references (which real sqlite allows)
//!   — a wider, but still loud (never silent), divergence than (a). A DEFERRED NO ACTION check
//!   is the exception: it defers to the COMMIT recheck. The COMMIT-time deferred recheck
//!   ([`check_table_deferred_fks`]) DOES walk a WR child's PK-index b-tree, so a DEFERRED FK on
//!   a WR child is fully enforced. Correct IMMEDIATE enforcement (seeking the WR parent's
//!   b-tree; scanning + writing back a WR child for CASCADE/SET NULL/SET DEFAULT — or a
//!   read-only PK-index scan to reject only a real dependent for NO ACTION/RESTRICT) is a
//!   follow-up.

use std::cmp::Ordering;

use minisqlite_btree::{table_delete, table_insert, IndexCursor, TableCursor};
use minisqlite_catalog::{Catalog, ForeignKeyDef, ReferentialAction, TableDef};
use minisqlite_expr::EvalExpr;
use minisqlite_fileformat::{encode_record_enc, TextEncoding};
use minisqlite_pager::{text_encoding_of, Pager};
use minisqlite_plan::{
    bind_generated_programs, compile_table_index_key_exprs, compile_table_index_partial_predicates,
    GeneratedProgram, Plan, PlanCtx,
    PlanNode,
};
use minisqlite_types::{
    apply_affinity, compare_for_eq, Affinity, Collation, ConstraintKind, DbIndex, Error,
    Result, Row, Value,
};

use crate::context::EvalCtx;
use crate::env::{Env, Pagers};
use crate::ops::dml_index::{
    build_index_plans, delete_index_entries, index_keys_for_plans, insert_index_entries, IndexPlan,
};
use crate::ops::generated::{compute_generated, stored_record};
use crate::ops::trigger::builtin_registry;
use crate::ops::without_rowid::wr_layout;
use crate::row::{decode_table_row_enc, decode_table_row_skipping_virtual_enc};
use crate::runtime::Runtime;

/// Whether THIS foreign key's check must be DEFERRED to COMMIT rather than raised now.
///
/// True exactly when a transaction is open to defer into ([`Runtime::fk_defer_active`], set
/// by the engine to `txn_active()` before the statement) AND the FK is deferred — either it
/// was declared `DEFERRABLE INITIALLY DEFERRED` (`fk.deferred`) or `PRAGMA defer_foreign_keys`
/// forces every FK deferred for this transaction ([`Runtime::defer_foreign_keys`]). In
/// AUTOCOMMIT `fk_defer_active` is false, so this is false and the FK is checked immediately —
/// deferred behaves like immediate (`foreignkeys.html` §4.2). RESTRICT is ALWAYS immediate
/// (§4.3) and its callers reject it BEFORE consulting this predicate, so a deferred RESTRICT
/// still errors at statement time.
pub(crate) fn fk_deferred_now(fk: &ForeignKeyDef, rt: &Runtime) -> bool {
    rt.fk_defer_active() && (fk.deferred || rt.defer_foreign_keys())
}

/// Enforce every outgoing FK of `child` against the new/updated logical row
/// (`[c0..c_{N-1}, rowid]`, or width-N for a WITHOUT ROWID child — only the key columns
/// `0..N` are read). No-op unless `PRAGMA foreign_keys` is ON and the table has an FK.
/// Errors `FOREIGN KEY constraint failed` on the first violation (the exact phrase real
/// sqlite uses, with no object detail).
///
/// A FK whose check is currently deferred ([`fk_deferred_now`]) is SKIPPED here — the row is
/// allowed to reference a missing parent while the transaction is open, and
/// [`check_deferred_foreign_keys`] re-verifies it at COMMIT (`foreignkeys.html` §4.2).
pub(crate) fn enforce_child_foreign_keys(
    child: &TableDef,
    logical: &[Value],
    catalog: &dyn Catalog,
    db: DbIndex,
    pager: &dyn Pager,
    rt: &Runtime,
) -> Result<()> {
    if !rt.foreign_keys() || child.foreign_keys.is_empty() {
        return Ok(());
    }
    for fk in &child.foreign_keys {
        // Deferred to COMMIT for this transaction: do not raise now — the commit-time
        // recheck ([`check_deferred_foreign_keys`]) will catch an unresolved violation.
        if fk_deferred_now(fk, rt) {
            continue;
        }
        // The child key values, in FK column order. A key column NULL satisfies the
        // constraint outright (MATCH SIMPLE), so gather + null-check in one pass.
        let mut child_vals: Vec<Value> = Vec::with_capacity(fk.child_columns.len());
        let mut any_null = false;
        for cn in &fk.child_columns {
            let idx = child
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(cn))
                .ok_or_else(|| {
                    Error::sql(format!("foreign key child column {cn} not found in {}", child.name))
                })?;
            let v = logical[idx].clone();
            any_null |= v.is_null();
            child_vals.push(v);
        }
        if any_null {
            continue;
        }

        // A cross-database FK is illegal (catalog.rs), so the parent lives in the SAME
        // namespace `db` the child row is written to: look it up there via `table_in(db)`, not
        // the bare search order (which under a same-named shadow could resolve, and then probe,
        // another namespace's parent). With no shadow `db == MAIN` and this equals bare `table`.
        let parent = catalog.table_in(db, &fk.parent_table)?.ok_or_else(|| {
            // A missing parent table is a schema error SQLite reports as "no such table"
            // when the FK fires.
            Error::sql(format!("no such table: {}", fk.parent_table))
        })?;
        let parent_key = parent_key_columns(parent, fk)?;

        if !parent_key_exists(pager, catalog, parent, &parent_key, &child_vals)? {
            return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
        }
    }
    Ok(())
}

/// Re-verify every DEFERRED foreign key across namespace `db` at COMMIT time
/// (`foreignkeys.html` §4.2: deferred constraints "are not checked until the transaction
/// tries to COMMIT … COMMIT will fail as long as foreign key constraints remain in
/// violation"). For each table in `db` that declares an FK which is deferred — declared
/// `DEFERRABLE INITIALLY DEFERRED`, or ANY FK when `defer_all` is set by
/// `PRAGMA defer_foreign_keys` — stream its rows and confirm each non-NULL child key still
/// references an existing parent (MATCH SIMPLE: a row with ANY NULL in the FK columns is
/// exempt). The FIRST unresolved violation returns `Err(FOREIGN KEY constraint failed)`, and
/// the caller (the engine COMMIT) then leaves the transaction OPEN and commits nothing.
///
/// A single CHILD-side scan covers BOTH deferred child violations (an INSERT/UPDATE that
/// pointed at a missing parent) AND deferred parent-side NO-ACTION orphans (a DELETE/UPDATE
/// of a referenced parent key that left a child dangling): both manifest as a child row whose
/// FK no longer resolves, so re-checking the child side finds them all. IMMEDIATE FKs are NOT
/// re-scanned — they were already enforced at statement time and cannot be violated at COMMIT
/// — so a table with only immediate FKs is skipped (the `defer_all == false` common case).
/// The caller gates the whole pass on `PRAGMA foreign_keys` being ON, so an FK-off connection
/// pays nothing. Reuses the child-side parent-existence probe ([`parent_key_exists`]) so the
/// commit check and the immediate check agree on what "the parent exists" means; streams
/// row-by-row so no table is materialized.
pub fn check_deferred_foreign_keys(
    catalog: &dyn Catalog,
    db: DbIndex,
    pager: &dyn Pager,
    defer_all: bool,
) -> Result<()> {
    for table in catalog.tables_in(db)? {
        check_table_deferred_fks(table, catalog, db, pager, defer_all)?;
    }
    Ok(())
}

/// One resolved deferred-FK probe: the parent table, its key column indices (parent side),
/// and the child's key column indices — pre-resolved ONCE so the per-row commit-recheck
/// loop does no schema lookup. Shared by the ROWID and WITHOUT ROWID child scans.
type DeferredProbe<'a> = (&'a TableDef, Vec<usize>, Vec<usize>);

/// The per-table half of [`check_deferred_foreign_keys`]: re-verify `child`'s deferred FKs.
/// Off-cost early-outs keep the commit path cheap — a table with no FK, or none deferred
/// (and `defer_all` off), or an empty table, needs no scan.
///
/// A ROWID child walks its table b-tree; a WITHOUT ROWID child walks its PRIMARY KEY index
/// b-tree and decodes each key record via the shared [`wr_layout`] — the SAME split the
/// read-path scan uses (`ops/scan.rs`). This matters for correctness, not just coverage:
/// the child-side IMMEDIATE check ([`enforce_child_foreign_keys`]) enforces a WR child's FK
/// at INSERT/UPDATE (it reads the logical row and never had a `without_rowid` exemption), so
/// if the deferred recheck skipped WR children a `DEFERRABLE INITIALLY DEFERRED` FK on a WR
/// child would be enforced by NEITHER path — the immediate check is skipped because it is
/// deferred, and the commit check would be skipped because it is WR — silently committing a
/// dangling reference real sqlite rejects. Scanning WR here keeps deferred == immediate.
fn check_table_deferred_fks(
    child: &TableDef,
    catalog: &dyn Catalog,
    db: DbIndex,
    pager: &dyn Pager,
    defer_all: bool,
) -> Result<()> {
    if child.foreign_keys.is_empty() {
        return Ok(());
    }
    // Pre-resolve each DEFERRED FK (declared, or every FK under the defer pragma) to its
    // parent + parent-key columns + child-key indices ONCE, so the per-row loop does no
    // schema lookup. An empty result means every FK on this table is immediate → no scan
    // (they were enforced at statement time and cannot fail at COMMIT).
    let mut probes: Vec<DeferredProbe<'_>> = Vec::new();
    for fk in &child.foreign_keys {
        if !(defer_all || fk.deferred) {
            continue;
        }
        let parent = catalog
            .table_in(db, &fk.parent_table)?
            .ok_or_else(|| Error::sql(format!("no such table: {}", fk.parent_table)))?;
        let parent_key = parent_key_columns(parent, fk)?;
        let child_idx = child_key_indices(child, fk)?;
        probes.push((parent, parent_key, child_idx));
    }
    if probes.is_empty() {
        return Ok(());
    }

    // Shared decode context, derived ONCE (not per row): the DB text encoding (§1.3.13, so a
    // UTF-16 child's TEXT keys decode correctly) and the child's generated-column programs
    // (empty — fast decode — for a table with no generated column).
    let programs = table_generated_programs(catalog, child)?;
    let throwaway_plan = throwaway_eval_plan();
    let mut throwaway_rt = Runtime::new();
    let enc = text_encoding_of(pager);

    if child.without_rowid {
        // WITHOUT ROWID: the rows live in the PRIMARY KEY index b-tree (the table's own root
        // IS that index), so walk it with an IndexCursor and decode each key record via the
        // shared WrLayout — the identical machinery the read-path WR scan uses. `decode_row_enc`
        // leaves any VIRTUAL generated column a NULL placeholder, so compute the generated
        // columns when the table has one (exactly as the rowid path's `decode_scanned_row`
        // does), so an FK column that IS — or follows — a generated column reads its real value.
        let layout = wr_layout(child)?;
        let has_virtual = programs.iter().any(|p| !p.stored);
        let mut scratch: Vec<Value> = Vec::new();
        let mut cur = IndexCursor::open(pager, child.root_page)?;
        if !cur.first()? {
            return Ok(());
        }
        loop {
            // Scope the key borrow to just the decode so the mutable `cur.next()` below is
            // free (mirrors `WrSeqScanCursor::next_row` in `ops/scan.rs`).
            let mut row = {
                let key = cur.key()?;
                layout.decode_row_enc(key.as_ref(), child, &mut scratch, enc)
            };
            if has_virtual {
                // FK enforcement is single-namespace (SQLite forbids cross-database FKs) and
                // this Env only computes generated columns (subquery/table-free), so the `db`
                // tag is inert — MAIN is a safe placeholder, matching `decode_scanned_row`.
                let env = Env {
                    catalog,
                    pagers: Pagers::One { db: DbIndex::MAIN, pager },
                    plan: &throwaway_plan,
                };
                compute_generated(&programs, true, &mut row, env, &mut throwaway_rt)?;
            }
            check_child_row_against_probes(&row, &probes, pager, catalog)?;
            if !cur.next()? {
                return Ok(());
            }
        }
    }

    // ROWID child: walk the table b-tree, decoding each row virtual-aware, one at a time (no
    // whole-table materialization). The `without_rowid` block above always returns from inside
    // (empty-table `return`, else a `loop` with no `break`), so reaching here means a rowid
    // table; the assert makes that exclusion explicit so a future edit that added a
    // non-returning path to the WR block can't silently open a rowid TableCursor on a WR
    // (index-rooted) table.
    debug_assert!(
        !child.without_rowid,
        "WITHOUT ROWID child {} must be scanned by the IndexCursor path above",
        child.name
    );
    let mut tc = TableCursor::open(pager, child.root_page)?;
    if !tc.first()? {
        return Ok(());
    }
    loop {
        let rowid = tc.rowid();
        let payload = tc.payload()?;
        let row = decode_scanned_row(
            &payload,
            rowid,
            child,
            &programs,
            catalog,
            pager,
            &mut throwaway_rt,
            &throwaway_plan,
            enc,
        )?;
        check_child_row_against_probes(&row, &probes, pager, catalog)?;
        if !tc.next()? {
            return Ok(());
        }
    }
}

/// Check one decoded child row (schema-order, width `N`) against every deferred-FK probe:
/// gather each FK's child-key values, EXEMPT a row with ANY NULL key column (MATCH SIMPLE),
/// else require the parent to still exist. Err `FOREIGN KEY constraint failed` on the first
/// unresolved violation. Shared by the ROWID and WITHOUT ROWID scans so both enforce
/// identically (and identically to the immediate child check, which reuses `parent_key_exists`).
fn check_child_row_against_probes(
    row: &[Value],
    probes: &[DeferredProbe<'_>],
    pager: &dyn Pager,
    catalog: &dyn Catalog,
) -> Result<()> {
    for (parent, parent_key, child_idx) in probes {
        let mut child_vals: Vec<Value> = Vec::with_capacity(child_idx.len());
        let mut any_null = false;
        for &ci in child_idx.iter() {
            let v = row[ci].clone();
            any_null |= v.is_null();
            child_vals.push(v);
        }
        if any_null {
            continue;
        }
        if !parent_key_exists(pager, catalog, parent, parent_key, &child_vals)? {
            return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
        }
    }
    Ok(())
}

/// The parent key column indices this FK references, in the same order as the child key.
/// An explicit `REFERENCES p(a, b)` names them; a bare `REFERENCES p` (empty
/// `parent_columns`) targets the parent PRIMARY KEY — the INTEGER PRIMARY KEY rowid alias
/// when there is one, else the parent's declared PRIMARY KEY columns in declaration order.
///
/// The bare-`REFERENCES` PK column set/order comes from [`TableDef::primary_key`], which
/// records every PK form uniformly (column-level, table-level, and composite). It must NOT be
/// rebuilt from the per-column `ColumnDef::primary_key` flag: the builder sets that flag ONLY
/// for a column-level `PRIMARY KEY`, so a table-level or composite `PRIMARY KEY(...)` leaves it
/// unset on every column — a flag-based filter then finds an empty set and wrongly rejects a
/// valid child of such a parent as "referencing ... which has no primary key".
fn parent_key_columns(parent: &TableDef, fk: &ForeignKeyDef) -> Result<Vec<usize>> {
    if !fk.parent_columns.is_empty() {
        let mut out = Vec::with_capacity(fk.parent_columns.len());
        for pn in &fk.parent_columns {
            let idx = parent
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(pn))
                .ok_or_else(|| {
                    Error::sql(format!(
                        "foreign key mismatch - referencing missing column {}.{pn}",
                        parent.name
                    ))
                })?;
            out.push(idx);
        }
        return Ok(out);
    }
    if let Some(i) = parent.rowid_alias {
        return Ok(vec![i]);
    }
    let pk = parent.primary_key.clone();
    if pk.is_empty() {
        return Err(Error::sql(format!(
            "foreign key mismatch - referencing \"{}\" which has no primary key",
            parent.name
        )));
    }
    Ok(pk)
}

/// Whether a parent row exists whose `key_idxs` columns equal `child_vals` (positionally).
/// The child value takes the PARENT key column's affinity before comparison (SQLite
/// compares an FK under the parent index's affinity/collation), and text compares under
/// that column's collation. A parent with a VIRTUAL generated column is decoded virtual-
/// aware and its virtual columns computed (see [`decode_scanned_row`]) so a key column that
/// follows a virtual column — or that IS a virtual generated column — reads its real value.
fn parent_key_exists(
    pager: &dyn Pager,
    catalog: &dyn Catalog,
    parent: &TableDef,
    key_idxs: &[usize],
    child_vals: &[Value],
) -> Result<bool> {
    // Fast path: the key IS the parent's INTEGER PRIMARY KEY (rowid). Coerce the child
    // value to an integer and seek the b-tree directly — O(log n) instead of a scan. A
    // child value that is not an integer rowid (after INTEGER affinity) can match no row.
    if key_idxs.len() == 1 && parent.rowid_alias == Some(key_idxs[0]) {
        return match apply_affinity(child_vals[0].clone(), Affinity::Integer) {
            Value::Integer(k) => {
                let mut tc = TableCursor::open(pager, parent.root_page)?;
                tc.seek_exact(k)
            }
            _ => Ok(false),
        };
    }

    // A WITHOUT ROWID parent stores its rows in a PK-index b-tree, not a rowid table, so the
    // rowid-table scan below cannot decode it. Fail CLOSED with a loud "not supported" error
    // (see the module SCOPE note): a silent `Ok(true)` here would report a VIOLATED
    // FK-to-WR-parent as satisfied — a fail-OPEN path. (The IMMEDIATE parent-side ON
    // DELETE/UPDATE action on a WR *child* likewise fails closed in `enforce_parent_delete` /
    // `enforce_parent_update`; the COMMIT-time deferred recheck [`check_table_deferred_fks`]
    // is the one path that actually scans a WR child.) Correctly enforcing such an FK (seeking
    // the parent's PK-index b-tree) is a follow-up.
    if parent.without_rowid {
        return Err(Error::sql(
            "FOREIGN KEY referencing a WITHOUT ROWID parent table is not supported",
        ));
    }

    // General path: scan the parent, comparing the key columns. Correct for any key shape;
    // O(n) per check, acceptable because FK enforcement is off by default and this only
    // runs when it is on. A parent PK / UNIQUE index could make this a seek — a follow-up.
    let affinities: Vec<Affinity> = key_idxs
        .iter()
        .map(|&i| parent.column_affinity(i))
        .collect();
    let collations: Vec<Collation> =
        key_idxs.iter().map(|&i| collation_of(parent.columns[i].collation.as_deref())).collect();

    // Bind the parent's generation programs ONCE (empty — fast decode — for a parent with no
    // generated column). A VIRTUAL column is not in the stored record, so each row is decoded
    // virtual-aware and its virtual columns computed before the key comparison.
    let programs = table_generated_programs(catalog, parent)?;
    let throwaway_plan = throwaway_eval_plan();
    let mut throwaway_rt = Runtime::new();
    // Read the DB's text encoding ONCE for the whole parent scan (off-56 is fixed at DB
    // creation), not per row: a UTF-16 parent row's TEXT key columns then decode correctly.
    let enc = text_encoding_of(pager);

    let mut tc = TableCursor::open(pager, parent.root_page)?;
    if !tc.first()? {
        return Ok(false);
    }
    loop {
        let rowid = tc.rowid();
        let payload = tc.payload()?;
        let row = decode_scanned_row(
            &payload,
            rowid,
            parent,
            &programs,
            catalog,
            pager,
            &mut throwaway_rt,
            &throwaway_plan,
            enc,
        )?;
        let matched = key_idxs.iter().enumerate().all(|(k, &pidx)| {
            let cv = apply_affinity(child_vals[k].clone(), affinities[k]);
            compare_for_eq(&cv, &row[pidx], collations[k]) == Some(Ordering::Equal)
        });
        if matched {
            return Ok(true);
        }
        if !tc.next()? {
            return Ok(false);
        }
    }
}

/// Map a column's declared `COLLATE` name to a built-in collating sequence (default
/// BINARY). Only text comparison consults it.
fn collation_of(name: Option<&str>) -> Collation {
    match name {
        Some(n) if n.eq_ignore_ascii_case("NOCASE") => Collation::NoCase,
        Some(n) if n.eq_ignore_ascii_case("RTRIM") => Collation::Rtrim,
        _ => Collation::Binary,
    }
}

/// A throwaway [`Plan`] whose only purpose is to satisfy the [`Env`] a runtime evaluation
/// needs. Every FK runtime-eval path (generated-column compute, index-key evaluation) runs
/// deterministic, subquery/parameter-free expressions (`gencol.html` §2.3 /
/// `lang_createindex.html` §1.2), so the plan's `subqueries` / `ctes` / `generated` maps are
/// never consulted — an empty `SingleRow` plan suffices.
fn throwaway_eval_plan() -> Plan {
    Plan {
        root: PlanNode::SingleRow,
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

/// Bind `def`'s GENERATED-column programs for a runtime FK scan, or an empty vec when the
/// table has no generated column (the common case — no [`PlanCtx`] is built). Mirrors
/// [`compile_child_index_key_exprs`]: a runtime bind over the executor's builtin function
/// registry, because FK enforcement discovers the scanned parent/child table dynamically as
/// it walks the FK graph and so has no plan-time node carrying its programs. A generated
/// expression calling a CONNECTION-registered custom function is the same documented
/// follow-up as for a child index expression (only builtins resolve here); an unknown column
/// fails loudly (a corrupt catalog), never silently dropping a program.
fn table_generated_programs(catalog: &dyn Catalog, def: &TableDef) -> Result<Vec<GeneratedProgram>> {
    if def.columns.iter().all(|c| c.generated.is_none()) {
        return Ok(Vec::new());
    }
    let mut ctx = PlanCtx::new(builtin_registry(), catalog);
    bind_generated_programs(&mut ctx, def)
}

/// Decode a scanned base-table row of `def` with its VIRTUAL generated columns computed.
///
/// A table with a VIRTUAL generated column stores a record that OMITS it
/// ([`stored_record`]), so a positional [`decode_table_row_enc`] mis-maps every column AFTER
/// the virtual one — an FK key column that follows it reads the wrong slot (typically NULL),
/// and a key that IS a virtual generated column reads NULL. This decodes through the virtual-
/// aware [`decode_table_row_skipping_virtual_enc`] (each virtual column a NULL placeholder in
/// a full-width row) and then fills those placeholders with [`compute_generated`]
/// (`only_virtual`), so both cases read the right value. A table with no VIRTUAL column
/// (`programs` empty, or every generated column STORED — a STORED value is materialized in
/// the record) takes the plain [`decode_table_row_enc`], so a non-generated table
/// pays nothing. Both decode in the DB's text encoding `enc` (passed in, read once).
///
/// The generation expressions are deterministic and subquery/parameter-free (`gencol.html`
/// §2.3), so the throwaway `rt` / `plan` supply an eval context that is never really read —
/// the same pattern [`child_index_keys`] uses for a child index expression. `def` is a rowid
/// table here (the WITHOUT ROWID parent/child paths bail out before this), so the decoded row
/// is `[c0..c_{N-1}, rowid]` and an INTEGER PRIMARY KEY reference in a generated expression
/// resolves to the trailing rowid register.
#[allow(clippy::too_many_arguments)]
fn decode_scanned_row(
    payload: &[u8],
    rowid: i64,
    def: &TableDef,
    programs: &[GeneratedProgram],
    catalog: &dyn Catalog,
    pager: &dyn Pager,
    rt: &mut Runtime,
    plan: &Plan,
    enc: TextEncoding,
) -> Result<Row> {
    // `all(stored)` holds when `programs` is empty (no generated column) OR every generated
    // column is STORED — both store a positionally-complete record, so the fast decode is
    // correct and no compute is needed. `enc` is the DB's text encoding, read once by the
    // caller (before its scan loop) so a UTF-16 parent/child row decodes its TEXT columns
    // correctly and the UTF-8 path pays no per-row header parse.
    if programs.iter().all(|p| p.stored) {
        return Ok(decode_table_row_enc(payload, rowid, def, enc));
    }
    let mut row = decode_table_row_skipping_virtual_enc(payload, rowid, def, enc);
    // FK enforcement is single-namespace (SQLite forbids cross-database FKs) and this Env
    // only computes generated columns (subquery/table-free), so `pagers.get` is never
    // called and the `db` tag is inert; `MAIN` is a safe placeholder.
    let env = Env { catalog, pagers: Pagers::One { db: DbIndex::MAIN, pager }, plan };
    compute_generated(programs, true, &mut row, env, rt)?;
    Ok(row)
}

/// The parent-key columns' affinity + collation, used to compare a candidate child key
/// against a parent key (SQLite compares an FK under the PARENT index's affinity and
/// collation — the same rule the child-side check applies, so a child that passed INSERT
/// is found here).
fn parent_key_meta(parent: &TableDef, key_idxs: &[usize]) -> (Vec<Affinity>, Vec<Collation>) {
    let affinities = key_idxs
        .iter()
        .map(|&i| parent.column_affinity(i))
        .collect();
    let collations =
        key_idxs.iter().map(|&i| collation_of(parent.columns[i].collation.as_deref())).collect();
    (affinities, collations)
}

/// The child's DEFAULT values for `fk`'s key columns, in FK order — a column's pre-evaluated
/// constant `DEFAULT` ([`ColumnDef::default_value`]) or NULL when it has none (SQLite: a
/// column with no explicit default defaults to NULL). Used by the `SET DEFAULT` referential
/// action (`foreignkeys.html` §4.3). A non-constant default (`DEFAULT (expr)` / `CURRENT_*`)
/// is not folded (`default_value` is `None`) and so reads as NULL here — a documented
/// narrowing, since a non-constant FK-column default is a degenerate schema.
fn child_default_key(child: &TableDef, cidx: &[usize]) -> Vec<Value> {
    cidx.iter().map(|&ci| child.columns[ci].default_value.clone().unwrap_or(Value::Null)).collect()
}

/// Whether two key tuples are positionally equal under the given per-column affinity +
/// collation (the same rule the FK match uses). Length-mismatched tuples are never equal.
fn keys_equal(a: &[Value], b: &[Value], affinities: &[Affinity], collations: &[Collation]) -> bool {
    a.len() == b.len()
        && (0..a.len()).all(|i| {
            let av = apply_affinity(a[i].clone(), affinities[i]);
            let bv = apply_affinity(b[i].clone(), affinities[i]);
            compare_for_eq(&av, &bv, collations[i]) == Some(Ordering::Equal)
        })
}

/// Enforce the PARENT side of every INCOMING foreign key when a row of `parent` is
/// deleted. For each user table that declares an FK naming `parent`, resolve the parent
/// key values from the deleted row (`logical` is `[c0..c_{N-1}]`, and the rowid-alias
/// column already carries the rowid — see [`crate::row::decode_table_row_enc`]), find the
/// referencing child rows, then apply the FK's `ON DELETE` action:
///
/// * `NO ACTION` / `RESTRICT` — error `FOREIGN KEY constraint failed` if any child row
///   still references the deleted parent key (both reject; the immediate-vs-end-of-
///   statement timing difference between them is not observable for a single-statement
///   DELETE, which is the whole in-scope surface).
/// * `CASCADE` — delete each referencing child row (its index entries then its record),
///   recursing FIRST so a cascaded child that is itself a parent cascades on to its own
///   children before it is removed.
/// * `SET NULL` — rewrite each referencing child row, setting its FK columns to NULL
///   (re-encoding the record and MOVING its index entries), so the child survives detached.
///   Skipped (child left as-is) for a child that has generated columns — a documented
///   follow-up (its STORED generated values would need recomputing from the nulled row).
/// * `SET DEFAULT` — rewrite each referencing child, setting its FK columns to their column
///   defaults, then re-check the FK (a default that references no surviving parent row raises
///   the violation — `foreignkeys.html` §4.3). Skipped (child left as-is) for a
///   generated-column child, the same as SET NULL.
///
/// No-op unless `PRAGMA foreign_keys` is ON. A parent key column that is NULL in the
/// deleted row can have no child referencing it (MATCH SIMPLE: a child NULL key was never
/// constrained), so that FK is skipped. Takes `&mut Pager` because CASCADE / SET NULL /
/// SET DEFAULT write.
pub(crate) fn enforce_parent_delete(
    parent: &TableDef,
    logical: &[Value],
    catalog: &dyn Catalog,
    db: DbIndex,
    pager: &mut dyn Pager,
    rt: &Runtime,
) -> Result<()> {
    if !rt.foreign_keys() {
        return Ok(());
    }
    // The DB's text encoding (fileformat2 §1.3.13), read ONCE per cascade invocation (off-56
    // is fixed at DB creation) — NOT per cascaded row — and threaded into every child-row
    // decode / index-maintenance / record rewrite below, so a bulk cascade pays no per-row
    // header parse and a UTF-16 DB's child rows round-trip correctly.
    let enc = text_encoding_of(&*pager);
    // A cross-database FK is illegal (catalog.rs), so every child referencing this parent lives
    // in the SAME namespace `db` as the parent being deleted: enumerate that store's tables with
    // `tables_in(db)`, not the cross-namespace union (which would drag in a coincidentally-named
    // child from another namespace and try to cascade it on `db`'s pager). The whole cascade —
    // its `build_index_plans` and recursion below — stays within `db`. No shadow => equals bare.
    for child in catalog.tables_in(db)? {
        for fk in &child.foreign_keys {
            if !fk.parent_table.eq_ignore_ascii_case(&parent.name) {
                continue;
            }
            let key_idxs = parent_key_columns(parent, fk)?;
            let key_vals: Vec<Value> = key_idxs.iter().map(|&i| logical[i].clone()).collect();
            if key_vals.iter().any(Value::is_null) {
                continue;
            }
            // A WITHOUT ROWID child cannot be scanned here to apply the parent-side ON DELETE
            // action (its rows live in a PK-index b-tree, not a rowid table, which the rowid
            // `matching_child_rows` scan below cannot walk). A DEFERRED NO ACTION check may
            // skip now — the COMMIT-time deferred recheck ([`check_table_deferred_fks`]) DOES
            // scan WR children and catches a resulting orphan — but any IMMEDIATE action (NO
            // ACTION now, RESTRICT, CASCADE, SET NULL, SET DEFAULT) fires at once and cannot be
            // honored on a WR child, so fail CLOSED rather than silently skip it and orphan the
            // child (fail OPEN). CAVEAT (honest over-rejection): because we cannot scan the
            // child, this fires whenever a WR child merely DECLARES this FK — even for a parent
            // key that NO child row references, which real sqlite deletes fine — so it
            // OVER-rejects (a wider divergence than the WR-parent case, but still loud, never
            // silent). Correct immediate enforcement — and the narrowing that would reject ONLY
            // a real dependent for NO ACTION/RESTRICT via a read-only PK-index scan (the
            // machinery [`check_table_deferred_fks`] already uses) — is a follow-up.
            if child.without_rowid {
                if matches!(fk.on_delete, ReferentialAction::NoAction) && fk_deferred_now(fk, rt) {
                    continue;
                }
                return Err(Error::sql(
                    "ON DELETE FOREIGN KEY enforcement on a WITHOUT ROWID child table is not supported",
                ));
            }
            let (affinities, collations) = parent_key_meta(parent, &key_idxs);
            let cidx = child_key_indices(child, fk)?;
            let matches = matching_child_rows(
                &*pager, catalog, child, &cidx, &key_vals, &affinities, &collations, enc,
            )?;
            if matches.is_empty() {
                continue;
            }
            match fk.on_delete {
                // RESTRICT rejects IMMEDIATELY even for a deferred FK (foreignkeys.html
                // §4.3: "configuring a RESTRICT action causes SQLite to return an error
                // immediately … Even if the foreign key constraint it is attached to is
                // deferred"). So it never consults the deferral predicate.
                ReferentialAction::Restrict => {
                    return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                }
                // NO ACTION (the default) is the only parent-side rejection that DEFERS:
                // while a deferred check is active, leave the now-orphaned child in place
                // and let the COMMIT-time recheck (a child-side scan) catch it if it is
                // still orphaned; otherwise reject immediately as before.
                ReferentialAction::NoAction => {
                    if fk_deferred_now(fk, rt) {
                        continue;
                    }
                    return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                }
                ReferentialAction::Cascade => {
                    // Compile the child's expression-index key programs (empty for an
                    // ordinary all-named-column child index) so a child carrying an
                    // EXPRESSION index — `CREATE INDEX ci ON child(a+b)` — has its entries
                    // maintained across the cascade delete instead of hitting the loud 2b
                    // guard in `build_index_plans`.
                    let ckey_exprs = compile_child_index_key_exprs(catalog, db, child)?;
                    let cpreds = compile_child_index_partial_predicates(catalog, db, child)?;
                    let cplans = build_index_plans(catalog, db, child, &ckey_exprs, &cpreds)?;
                    for (crowid, crow) in &matches {
                        // Recurse before removing this child: it may itself be a parent
                        // (grandchildren) whose ON DELETE must fire first.
                        enforce_parent_delete(child, crow, catalog, db, pager, rt)?;
                        // EVAL then WRITE: compute the child row's index keys under a shared
                        // borrow before the exclusive-borrow deletes below.
                        let ckeys = child_index_keys(&cplans, crow, *crowid, catalog, &*pager)?;
                        delete_index_entries(pager, &cplans, &ckeys, enc)?;
                        table_delete(pager, child.root_page, *crowid)?;
                    }
                }
                ReferentialAction::SetNull | ReferentialAction::SetDefault => {
                    // Rewrite each matching child, setting its FK columns to NULL (SET NULL)
                    // or to their column DEFAULTs (SET DEFAULT). A generated-column child is
                    // skipped (its STORED generated values would need recomputing from the
                    // rewritten row — a documented follow-up), left as-is rather than mis-stored.
                    if child.columns.iter().any(|c| c.generated.is_some()) {
                        continue;
                    }
                    let set_default = matches!(fk.on_delete, ReferentialAction::SetDefault);
                    // The values to write into the child FK columns, in FK order: NULL for SET
                    // NULL, else each column's default under the child column's affinity.
                    let new_vals: Vec<Value> = if set_default {
                        child_default_key(child, &cidx)
                            .into_iter()
                            .enumerate()
                            .map(|(k, v)| {
                                apply_affinity(v, child.column_affinity(cidx[k]))
                            })
                            .collect()
                    } else {
                        vec![Value::Null; cidx.len()]
                    };
                    // SET DEFAULT still must satisfy the FK (foreignkeys.html §4.3: its
                    // DEFAULT-0 example violates when parent 0 is absent). The POST-DELETE
                    // parent no longer holds `key_vals`, so the defaults are valid iff any is
                    // NULL (MATCH SIMPLE) OR they reference a parent row OTHER than the one
                    // being deleted. Checked ONCE before the loop (defaults are per-FK).
                    //
                    // Unlike CASCADE (copies a live key) and SET NULL (MATCH-SIMPLE exempt),
                    // SET DEFAULT CAN leave a child-side violation, so it is a normal
                    // (non-RESTRICT) FK violation that a DEFERRED FK — or any FK under
                    // `PRAGMA defer_foreign_keys` — DEFERS to COMMIT, exactly like NO ACTION
                    // (foreignkeys.html §4.2); only RESTRICT stays always-immediate (§4.3).
                    // `!fk_deferred_now` short-circuits so a deferred FK skips this immediate
                    // RAISE (the action below still sets the default); the COMMIT recheck
                    // ([`check_table_deferred_fks`]) catches a still-dangling default and
                    // leaves the transaction open. In autocommit the predicate is false, so
                    // this raises exactly as before.
                    if !fk_deferred_now(fk, rt)
                        && set_default
                        && !new_vals.iter().any(Value::is_null)
                        && (keys_equal(&new_vals, &key_vals, &affinities, &collations)
                            || !parent_key_exists(&*pager, catalog, parent, &key_idxs, &new_vals)?)
                    {
                        return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                    }
                    // As the CASCADE arm: compile the child's expression-index key programs
                    // so an expression-indexed child's entries are MOVED (old key deleted,
                    // new key inserted) as the FK columns change.
                    let ckey_exprs = compile_child_index_key_exprs(catalog, db, child)?;
                    let cpreds = compile_child_index_partial_predicates(catalog, db, child)?;
                    let cplans = build_index_plans(catalog, db, child, &ckey_exprs, &cpreds)?;
                    for (crowid, crow) in &matches {
                        let mut newrow = crow.clone();
                        for (k, &ci) in cidx.iter().enumerate() {
                            newrow[ci] = new_vals[k].clone();
                        }
                        // EVAL then WRITE: OLD keys (from `crow`) to delete, NEW keys (from the
                        // rewritten row) to insert — both computed under a shared borrow before
                        // the exclusive-borrow writes below. Move index entries and overwrite
                        // the record in place at the same rowid.
                        let old_keys = child_index_keys(&cplans, crow, *crowid, catalog, &*pager)?;
                        let new_keys = child_index_keys(&cplans, &newrow, *crowid, catalog, &*pager)?;
                        delete_index_entries(pager, &cplans, &old_keys, enc)?;
                        let record = encode_record_enc(&stored_record(&newrow, child), enc);
                        table_insert(pager, child.root_page, *crowid, &record)?;
                        insert_index_entries(pager, &cplans, &new_keys, enc)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Enforce the PARENT side of every INCOMING foreign key when a row of `parent` is UPDATEd
/// and a REFERENCED key column changes value (`foreignkeys.html` §4.3: an ON UPDATE action
/// fires only when the update actually modifies a parent key the child references). For each
/// user table that declares an FK naming `parent`, compare that FK's parent-key columns
/// between the OLD row (`old_logical`) and the NEW row (`new_logical`); an unchanged key
/// fires nothing. When it changes, find the child rows that referenced the OLD key and apply
/// the FK's `ON UPDATE` action:
///
/// * `NO ACTION` / `RESTRICT` — error `FOREIGN KEY constraint failed` if any child still
///   references the OLD parent key (both reject; the immediate-vs-end-of-statement timing
///   difference is not observable for a single-statement UPDATE).
/// * `CASCADE` — rewrite each referencing child, copying the NEW parent key values into its
///   FK columns (each under the child column's affinity), re-encoding the record and MOVING
///   its index entries, then RECURSE so a cascaded child that is itself a referenced parent
///   propagates on.
/// * `SET NULL` / `SET DEFAULT` — rewrite each referencing child, setting its FK columns to
///   NULL (SET NULL) or to their column defaults (SET DEFAULT). Skipped (child left as-is)
///   for a child with generated columns — a documented follow-up, as in
///   [`enforce_parent_delete`]. SET DEFAULT still requires the resulting key to reference an
///   existing parent (`foreignkeys.html` §4.3), so a default that matches no post-update
///   parent row raises the FK violation.
///
/// No-op unless `PRAGMA foreign_keys` is ON. Mirrors [`enforce_parent_delete`]: a cascaded
/// child rewrite does NOT fire the child's triggers and does not advance `changes()`
/// (`sqlite3_changes()` excludes FK-action rows). `old_logical` / `new_logical` are the
/// parent's `[c0..c_{N-1}]` logical rows — the rowid-alias column already carries the rowid,
/// so an FK that references the parent's INTEGER PRIMARY KEY (rowid) cascades when the rowid
/// moves. Takes `&mut Pager` because CASCADE / SET NULL / SET DEFAULT write.
pub(crate) fn enforce_parent_update(
    parent: &TableDef,
    old_logical: &[Value],
    new_logical: &[Value],
    catalog: &dyn Catalog,
    db: DbIndex,
    pager: &mut dyn Pager,
    rt: &Runtime,
) -> Result<()> {
    if !rt.foreign_keys() {
        return Ok(());
    }
    // The DB's text encoding, read ONCE per cascade invocation (NOT per cascaded row) and
    // threaded into every child decode / index write / record rewrite below — see the same
    // hoist in `enforce_parent_delete`.
    let enc = text_encoding_of(&*pager);
    // Same namespace scoping as `enforce_parent_delete`: a cross-database FK is illegal, so the
    // children referencing this parent are all in `db`. Enumerate that store with `tables_in(db)`
    // and keep the whole cascade (its `build_index_plans` and recursion) within `db`.
    for child in catalog.tables_in(db)? {
        for fk in &child.foreign_keys {
            if !fk.parent_table.eq_ignore_ascii_case(&parent.name) {
                continue;
            }
            let key_idxs = parent_key_columns(parent, fk)?;
            let old_key: Vec<Value> = key_idxs.iter().map(|&i| old_logical[i].clone()).collect();
            let new_key: Vec<Value> = key_idxs.iter().map(|&i| new_logical[i].clone()).collect();
            let (affinities, collations) = parent_key_meta(parent, &key_idxs);

            // The referenced key is unchanged → no ON UPDATE action fires for this FK
            // (an UPDATE that touches only non-referenced columns leaves children alone).
            let unchanged = (0..key_idxs.len())
                .all(|k| compare_for_eq(&old_key[k], &new_key[k], collations[k]) == Some(Ordering::Equal));
            if unchanged {
                continue;
            }
            // A NULL in the OLD parent key means no child was constrained to it (MATCH
            // SIMPLE: a child NULL key was never required to match), so nothing references
            // the value that is changing — skip.
            if old_key.iter().any(Value::is_null) {
                continue;
            }
            // A WITHOUT ROWID child cannot be scanned here to apply the parent-side ON UPDATE
            // action (see the ON DELETE twin above, including the OVER-rejection caveat: this
            // fires whenever a WR child merely DECLARES this FK, even for a key no child row
            // references, which real sqlite updates fine). A DEFERRED NO ACTION check may skip
            // now (the COMMIT-time deferred recheck scans WR children and catches a resulting
            // orphan); any IMMEDIATE action fires at once and cannot be honored on a WR child,
            // so fail CLOSED rather than silently skip it and leave a dangling reference (fail
            // OPEN).
            if child.without_rowid {
                if matches!(fk.on_update, ReferentialAction::NoAction) && fk_deferred_now(fk, rt) {
                    continue;
                }
                return Err(Error::sql(
                    "ON UPDATE FOREIGN KEY enforcement on a WITHOUT ROWID child table is not supported",
                ));
            }
            let cidx = child_key_indices(child, fk)?;
            let matches = matching_child_rows(
                &*pager, catalog, child, &cidx, &old_key, &affinities, &collations, enc,
            )?;
            if matches.is_empty() {
                continue;
            }
            match fk.on_update {
                // RESTRICT rejects IMMEDIATELY even on a deferred FK (foreignkeys.html §4.3);
                // it never consults the deferral predicate.
                ReferentialAction::Restrict => {
                    return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                }
                // NO ACTION (the default) is the only parent-side rejection that DEFERS:
                // while a deferred check is active, leave the child referencing the OLD key
                // and let the COMMIT recheck catch it; otherwise reject immediately.
                ReferentialAction::NoAction => {
                    if fk_deferred_now(fk, rt) {
                        continue;
                    }
                    return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                }
                ReferentialAction::Cascade
                | ReferentialAction::SetNull
                | ReferentialAction::SetDefault => {
                    let cascade = matches!(fk.on_update, ReferentialAction::Cascade);
                    let set_default = matches!(fk.on_update, ReferentialAction::SetDefault);
                    // SET NULL / SET DEFAULT over a generated-column child is a follow-up (its
                    // STORED generated values would need recomputing from the rewritten row),
                    // left as-is — the same narrowing as the DELETE path. CASCADE rewrites
                    // regardless (its cascaded FK column is not itself generated).
                    if !cascade && child.columns.iter().any(|c| c.generated.is_some()) {
                        continue;
                    }
                    // The values to write into the child FK columns, in FK order (the same for
                    // every matching child): the NEW parent key (CASCADE), NULL (SET NULL), or
                    // each column's DEFAULT (SET DEFAULT), all under the child column's affinity.
                    let fk_vals: Vec<Value> = (0..cidx.len())
                        .map(|k| {
                            let raw = if cascade {
                                new_key[k].clone()
                            } else if set_default {
                                child.columns[cidx[k]].default_value.clone().unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            };
                            apply_affinity(raw, child.column_affinity(cidx[k]))
                        })
                        .collect();
                    // SET DEFAULT still must satisfy the FK (foreignkeys.html §4.3). After this
                    // UPDATE the parent holds `new_key` (not `old_key`), so the defaults are
                    // valid iff any is NULL (MATCH SIMPLE), OR they equal the new key, OR they
                    // reference some other existing parent row (and are not the vanished old
                    // key). Checked ONCE before the loop (defaults are per-FK).
                    //
                    // As on the DELETE side, SET DEFAULT can leave a child-side violation, so a
                    // DEFERRED FK (or any FK under `PRAGMA defer_foreign_keys`) DEFERS this
                    // rejection to COMMIT like NO ACTION (foreignkeys.html §4.2); only RESTRICT
                    // is always-immediate (§4.3). `!fk_deferred_now` skips the immediate RAISE
                    // for a deferred FK (the action below still sets the default); the COMMIT
                    // recheck catches a still-dangling default. Autocommit → predicate false →
                    // raises exactly as before.
                    if !fk_deferred_now(fk, rt)
                        && set_default
                        && !fk_vals.iter().any(Value::is_null)
                        && !keys_equal(&fk_vals, &new_key, &affinities, &collations)
                        && (keys_equal(&fk_vals, &old_key, &affinities, &collations)
                            || !parent_key_exists(&*pager, catalog, parent, &key_idxs, &fk_vals)?)
                    {
                        return Err(Error::constraint(ConstraintKind::ForeignKey, ""));
                    }
                    // As the DELETE cascade arm: compile the child's expression-index key
                    // programs so an expression-indexed child's entries are MOVED as its FK
                    // columns change (old key deleted, new key inserted).
                    let ckey_exprs = compile_child_index_key_exprs(catalog, db, child)?;
                    let cpreds = compile_child_index_partial_predicates(catalog, db, child)?;
                    let cplans = build_index_plans(catalog, db, child, &ckey_exprs, &cpreds)?;
                    for (crowid, crow) in &matches {
                        let mut newrow = crow.clone();
                        for (k, &ci) in cidx.iter().enumerate() {
                            newrow[ci] = fk_vals[k].clone();
                        }
                        // When the child's FK column IS its own INTEGER PRIMARY KEY (rowid
                        // alias) — the classic recursive shape where a table's PK also
                        // references its parent — a CASCADE changes the child's ROWID, so the
                        // row must be RELOCATED (deleted at the old rowid, inserted at the new)
                        // rather than rewritten in place. For an ordinary FK column the rowid
                        // is unchanged and this collapses to an in-place rewrite at `crowid`.
                        let new_rowid = match child.rowid_alias {
                            Some(ai) if cidx.contains(&ai) => match &newrow[ai] {
                                Value::Integer(r) => *r,
                                _ => *crowid,
                            },
                            _ => *crowid,
                        };
                        // EVAL then WRITE: OLD keys (from `crow`, old rowid) to delete, NEW keys
                        // (from the rewritten row, new rowid) to insert — both computed under a
                        // shared borrow before the exclusive-borrow writes below.
                        let old_keys = child_index_keys(&cplans, crow, *crowid, catalog, &*pager)?;
                        let new_keys =
                            child_index_keys(&cplans, &newrow, new_rowid, catalog, &*pager)?;
                        delete_index_entries(pager, &cplans, &old_keys, enc)?;
                        if new_rowid != *crowid {
                            table_delete(pager, child.root_page, *crowid)?;
                        }
                        let record = encode_record_enc(&stored_record(&newrow, child), enc);
                        table_insert(pager, child.root_page, new_rowid, &record)?;
                        insert_index_entries(pager, &cplans, &new_keys, enc)?;
                        // Recurse: the child's own referenced key may have changed (only when
                        // its FK columns overlap a key grandchildren reference — otherwise the
                        // recursion sees an unchanged key and returns at once), so propagate a
                        // multi-level cascade the same way the DELETE path recurses.
                        enforce_parent_update(child, crow, &newrow, catalog, db, pager, rt)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Compile the expression-index key programs for every index on `child` that has a genuine
/// expression key (`CREATE INDEX ci ON child(a+b)`), so an FK cascade / SET-NULL rewrite can
/// MAINTAIN that index. An ordinary all-named-column index contributes nothing (it is keyed
/// from the row's columns directly); the result is the same `(index name, per-key-column
/// compiled exprs)` shape the plan-time DML compilers put on an INSERT/UPDATE/DELETE node.
///
/// Unlike those compilers this runs at RUNTIME: the cascade discovers `child` (and its
/// indexes) as it walks the FK graph, so there is no plan-time node to carry the programs.
/// It compiles over the executor's process-wide builtin function registry — the same
/// runtime-compile mechanism [`crate::ops::trigger`] uses to recompile a trigger action's
/// own triggers. A child index expression calling a CONNECTION-registered custom function is
/// the same documented follow-up as for triggers (only builtins resolve here); a builtin such
/// as `lower()` resolves. Binding an unknown column would fail loudly (a corrupt catalog),
/// never silently drop the key.
fn compile_child_index_key_exprs(
    catalog: &dyn Catalog,
    db: DbIndex,
    child: &TableDef,
) -> Result<Vec<(String, Vec<Option<EvalExpr>>)>> {
    let mut ctx = PlanCtx::new(builtin_registry(), catalog);
    // `db` is the cascade's single namespace (`child` was found via `tables_in(db)`); pass it so
    // the index-expr compile reads `indexes_on_in(db)` and can't pick up a shadow's exprs.
    compile_table_index_key_exprs(&mut ctx, db, child)
}

/// Compile the WHERE predicate of every PARTIAL index on a cascade `child` at RUNTIME, so the
/// cascade honors partial-index membership: a CASCADE delete removes a child's index entry
/// only for a partial index the child row is actually IN, and a SET NULL / SET DEFAULT rewrite
/// moves an entry only across the predicate boundary (drop the OLD entry iff the OLD row was in
/// the index, add the NEW entry iff the rewritten row is). Without this, a cascade would probe
/// and delete phantom entries for a row a partial index never held — the parent-side twin of
/// the DML over-enforcement this fix removes (`partialindex.html` §2).
///
/// The sibling of [`compile_child_index_key_exprs`]: same runtime [`PlanCtx`] over the
/// executor's builtin registry, same `_in(db)` shadow-safe namespace read, feeding the same
/// `(index name, …)`-keyed lists [`build_index_plans`] attaches to each [`IndexPlan`]. A child
/// with no partial index yields an empty vec (the common case, no cost).
fn compile_child_index_partial_predicates(
    catalog: &dyn Catalog,
    db: DbIndex,
    child: &TableDef,
) -> Result<Vec<(String, EvalExpr)>> {
    let mut ctx = PlanCtx::new(builtin_registry(), catalog);
    compile_table_index_partial_predicates(&mut ctx, db, child)
}

/// Compute the precomputed index keys for a child row during an FK cascade / SET NULL
/// rewrite. `crow` is the child's `[c0..c_{N-1}]` logical row (width N, from
/// [`matching_child_rows`]); the index key expressions bind against `[cols.., rowid]`, so
/// the rowid is appended to form the frame.
///
/// A `None` slot is a PARTIAL child index the row is not in (its WHERE predicate — compiled by
/// [`compile_child_index_partial_predicates`] — is not TRUE over `crow`): the cascade's
/// `delete_index_entries` / `insert_index_entries` skip it, so the cascade touches only the
/// entries the row actually has, exactly like the DML paths.
///
/// For an ORDINARY-column child index every key part is a `Col` and the throwaway
/// `Runtime` / `Plan` wrapped in the `EvalCtx` are never read. For an EXPRESSION child index
/// (`build_index_plans` was fed the compiled key programs from
/// [`compile_child_index_key_exprs`]) `index_key_values` evaluates each `IndexKeyPart::Expr`
/// against the frame — as it does a partial index's predicate — and the throwaway ctx still
/// suffices, because a legal index expression (`lang_createindex.html` §1.2) or partial-index
/// predicate (`partialindex.html`) references only the row's own columns/rowid and calls only
/// DETERMINISTIC functions, whose implementation is already baked into the compiled
/// `EvalExpr::Func` (no subquery, bind parameter, or RNG that would need a live plan/runtime).
/// Constructing the throwaway per row is cheap (`Runtime::new` is a fixed-seed struct init)
/// and this path is off by default (`PRAGMA foreign_keys`), so it is not a hot loop.
fn child_index_keys(
    plans: &[IndexPlan],
    crow: &[Value],
    crowid: i64,
    catalog: &dyn Catalog,
    pager: &dyn Pager,
) -> Result<Vec<Option<Vec<Value>>>> {
    let mut frame = Vec::with_capacity(crow.len() + 1);
    frame.extend_from_slice(crow);
    frame.push(Value::Integer(crowid));
    let throwaway_plan = throwaway_eval_plan();
    let mut throwaway_rt = Runtime::new();
    // Single-namespace FK helper; the Env only evaluates index-key exprs (table-free), so
    // `pagers.get` is never called and the `db` tag is inert (see `decode_scanned_row`).
    let env = Env { catalog, pagers: Pagers::One { db: DbIndex::MAIN, pager }, plan: &throwaway_plan };
    let mut ctx = EvalCtx { rt: &mut throwaway_rt, env, outer: &frame };
    index_keys_for_plans(plans, &frame, crowid, &mut ctx)
}

/// The child column indices holding `fk`'s key, in FK order. Errors if the schema names a
/// child column the table does not have (a malformed catalog).
fn child_key_indices(child: &TableDef, fk: &ForeignKeyDef) -> Result<Vec<usize>> {
    let mut cidx = Vec::with_capacity(fk.child_columns.len());
    for cn in &fk.child_columns {
        let i = child
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(cn))
            .ok_or_else(|| Error::sql(format!("foreign key child column {cn} not found in {}", child.name)))?;
        cidx.push(i);
    }
    Ok(cidx)
}

/// Every row of `child` whose `cidx` columns equal the parent key `key_vals`, returned as
/// `(rowid, [c0..c_{N-1}])` pairs (the logical row is the index-key source a delete needs).
/// Compares under the PARENT key's affinity/collation, so the match agrees with the
/// child-side INSERT check. A child key column that is NULL never matches (MATCH SIMPLE),
/// which `compare_for_eq` yields naturally (NULL compares to nothing). The whole result is
/// materialized before any caller mutates, so a CASCADE delete never walks a b-tree it is
/// removing from. A WITHOUT ROWID child is rejected up front (its parent-side callers fail
/// CLOSED before calling this — see the module SCOPE note), never scanned as a rowid table. A
/// child with a VIRTUAL generated column is decoded virtual-aware and its virtual columns
/// computed (see [`decode_scanned_row`]) so the FK column — and any generated column the
/// returned row later feeds (index keys, a recursed cascade) — reads its real value.
#[allow(clippy::too_many_arguments)]
fn matching_child_rows(
    pager: &dyn Pager,
    catalog: &dyn Catalog,
    child: &TableDef,
    cidx: &[usize],
    key_vals: &[Value],
    affinities: &[Affinity],
    collations: &[Collation],
    enc: TextEncoding,
) -> Result<Vec<(i64, Row)>> {
    // A WITHOUT ROWID child's rows live in a PK-index b-tree, not a rowid table, so the rowid
    // `TableCursor` scan below cannot walk it. The parent-side callers
    // ([`enforce_parent_delete`] / [`enforce_parent_update`]) already fail CLOSED on a WR child
    // BEFORE reaching here (a deferred NO ACTION defers to the COMMIT recheck; any immediate
    // action errors), so this is an unreachable defensive guard: it must never silently return
    // an empty match set (which would fail OPEN — skip the action and orphan the child).
    if child.without_rowid {
        return Err(Error::sql(
            "FOREIGN KEY enforcement on a WITHOUT ROWID child table is not supported",
        ));
    }
    let mut out = Vec::new();
    let n = child.columns.len();
    // Bind the child's generation programs ONCE (empty — fast decode — for a child with no
    // generated column); a VIRTUAL column is not in the stored record, so each row is decoded
    // virtual-aware and its virtual columns computed before the FK-column comparison.
    let programs = table_generated_programs(catalog, child)?;
    let throwaway_plan = throwaway_eval_plan();
    let mut throwaway_rt = Runtime::new();
    let mut tc = TableCursor::open(pager, child.root_page)?;
    if !tc.first()? {
        return Ok(out);
    }
    loop {
        let rowid = tc.rowid();
        let payload = tc.payload()?;
        let row = decode_scanned_row(
            &payload,
            rowid,
            child,
            &programs,
            catalog,
            pager,
            &mut throwaway_rt,
            &throwaway_plan,
            enc,
        )?;
        let matched = cidx.iter().enumerate().all(|(k, &ci)| {
            let cv = apply_affinity(row[ci].clone(), affinities[k]);
            let pv = apply_affinity(key_vals[k].clone(), affinities[k]);
            compare_for_eq(&cv, &pv, collations[k]) == Some(Ordering::Equal)
        });
        if matched {
            out.push((rowid, row[..n].to_vec()));
        }
        if !tc.next()? {
            return Ok(out);
        }
    }
}
