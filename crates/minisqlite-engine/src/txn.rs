//! Transaction control at the engine level: `BEGIN` / `COMMIT` / `ROLLBACK`,
//! `SAVEPOINT` / `RELEASE` / `ROLLBACK TO`, and the shared
//! [`SqlEngine::with_write_txn`] wrapper that runs a catalog mutation (DDL) inside
//! the right transaction. These drive the [`Pager`] seam and the connection's
//! transaction state.
//!
//! A transaction can be started two ways: an explicit `BEGIN` (sets
//! [`SqlEngine::in_explicit_txn`]) or a `SAVEPOINT` issued in autocommit (starts a
//! transaction in the pager and pushes onto [`SqlEngine::savepoints`], leaving
//! `in_explicit_txn` false). So the one predicate for "a transaction is active" is
//! [`SqlEngine::txn_active`], NOT `in_explicit_txn` alone — every autocommit vs.
//! join-the-open-transaction decision routes through it.

use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_pager::Pager;
use minisqlite_sql::TransactionMode;
use minisqlite_types::{DbIndex, Error, Result};

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Whether a transaction is currently active — started by an explicit `BEGIN`
    /// OR implicitly by an open `SAVEPOINT`. This is the single source of truth for
    /// autocommit decisions across the engine (a bare statement wraps its own
    /// implicit transaction only when none is active).
    pub(crate) fn txn_active(&self) -> bool {
        self.in_explicit_txn || !self.savepoints.is_empty()
    }

    /// `BEGIN`: open an explicit transaction. Errors if one is already active — an
    /// explicit `BEGIN` or an implicit savepoint-started transaction (SQLite: "cannot
    /// start a transaction within a transaction").
    ///
    /// The mode governs WHEN the single write lock is taken (lang_transaction §2), which
    /// is what makes the two-connection (BUSY) behavior correct:
    /// - **DEFERRED** (the default `BEGIN`): take NO write lock at BEGIN — pin a read
    ///   snapshot only. The first write upgrades the transaction to a write transaction
    ///   (acquire the write lock, or `SQLITE_BUSY` if another connection holds it or has
    ///   committed since the snapshot). So a read-only DEFERRED transaction never blocks
    ///   a concurrent connection's commit and keeps its historic snapshot.
    /// - **IMMEDIATE** / **EXCLUSIVE**: start a write transaction at BEGIN (BUSY if
    ///   another write transaction is active). EXCLUSIVE is the same as IMMEDIATE in WAL
    ///   mode. Both map to the eager `pager.begin()`.
    ///
    /// For a single connection (or the in-memory / rollback backings) DEFERRED and the
    /// eager begin are observably identical — the lazy write lock only changes behavior
    /// under two WAL connections.
    pub(crate) fn exec_begin(&mut self, mode: TransactionMode) -> Result<()> {
        if self.txn_active() {
            return Err(Error::sql("cannot start a transaction within a transaction"));
        }
        // `PRAGMA defer_foreign_keys` is NOT cleared here: a value set just before `BEGIN`
        // applies to this transaction, as in sqlite3. It is switched off when a transaction
        // ends (pragma.html #pragma_defer_foreign_keys: "at each COMMIT or ROLLBACK"),
        // including the implicit one around an autocommit statement, so no stale value from
        // an earlier transaction can reach here.
        // An explicit transaction spans EVERY live namespace so a later COMMIT/ROLLBACK
        // covers temp writes too. `main` (index 0) governs WAL concurrency via the mode
        // and is begun first, so an IMMEDIATE `BEGIN` that hits SQLITE_BUSY there aborts
        // before any other store is begun. The in-memory temp store holds no write lock,
        // so its begin is eager regardless (its `begin_deferred` == `begin`). With only
        // `main` live this loop is exactly the single-pager begin it replaces.
        for i in 0..self.pagers.len() {
            match mode {
                TransactionMode::Deferred => self.pagers[i].begin_deferred()?,
                TransactionMode::Immediate | TransactionMode::Exclusive => self.pagers[i].begin()?,
            }
        }
        self.in_explicit_txn = true;
        Ok(())
    }

    /// `COMMIT` / `END`: commit the open transaction, releasing every savepoint —
    /// including a transaction that a `SAVEPOINT` (not `BEGIN`) started
    /// (lang_savepoint §2: "The COMMIT command may be used to release all savepoints
    /// and commit the transaction even if the transaction was originally started by a
    /// SAVEPOINT"). Errors if none is active. The state flips only after a successful
    /// commit, so a failed commit leaves the transaction open (as SQLite does).
    ///
    /// Sequential per-file commit — DELIBERATE, and correct on a normal commit. Each
    /// live namespace is committed one after another in the loop below (`main` first).
    /// On a clean (non-crash) commit this matches real SQLite: a multi-database
    /// transaction is permitted to commit and every file ends up updated, so the RESULT
    /// is correct. Rejecting multi-file commits with a loud/hard error was DELIBERATELY
    /// NOT chosen — it would diverge from real sqlite,
    /// which accepts the commit — so this path raises no such error and must not.
    ///
    /// Fully crash-atomic for a single durable file plus any in-memory/temp/`:memory:`
    /// stores (those commits cannot fail and leave no on-disk trace). The one limitation
    /// is crash-atomicity across TWO OR MORE durable FILE databases, and it is a KNOWN,
    /// SILENT-AT-RUNTIME gap — explicitly NOT a loud or runtime-guarded one. Real SQLite
    /// uses a super-journal (master journal) so a crash mid-COMMIT rolls every file back
    /// together; here a crash BETWEEN two per-file `commit()` calls can leave earlier
    /// files updated and later ones not, with NO signal at runtime. lang_attach.html §2
    /// contrasts the master-journal case with the ":memory: main / WAL" case. The
    /// super-journal is crash/fault-injection territory, deliberately out of scope
    /// here; this path assumes no crash mid-commit.
    /// The behavior is pinned only by `conformance_attach`'s multi-file commit test — a
    /// PERSISTENCE (non-crash) check, NOT a crash-atomicity test and NOT a loud-error test.
    pub(crate) fn exec_commit(&mut self) -> Result<()> {
        if !self.txn_active() {
            return Err(Error::sql("cannot commit - no transaction is active"));
        }
        // DEFERRED foreign-key recheck (foreignkeys.html §4.2): verify every deferred FK
        // BEFORE any pager commits and BEFORE the txn state is cleared, so an unresolved
        // violation fails the COMMIT with the transaction LEFT OPEN. The real logic lives in
        // `crate::deferred_fk` / the executor; this is the single hook point.
        self.check_deferred_foreign_keys()?;
        // Commit every live namespace. `main` is committed first (its durable store is
        // the one that matters; the in-memory temp store's commit cannot fail), matching
        // the single-pager commit this replaces when only `main` is live.
        for i in 0..self.pagers.len() {
            self.pagers[i].commit()?;
        }
        self.in_explicit_txn = false;
        self.savepoints.clear();
        // `PRAGMA defer_foreign_keys` is per-transaction and auto-clears at COMMIT
        // (pragma.html #pragma_defer_foreign_keys). Reset BOTH the live flag and its sticky
        // commit-coverage flag so neither leaks into the next transaction.
        self.rt.reset_defer_foreign_keys();
        Ok(())
    }

    /// `ROLLBACK`: roll back the open transaction and discard every savepoint
    /// (lang_savepoint §3: "The ROLLBACK command without a TO clause rolls backs all
    /// transactions and leaves the transaction stack empty."). Errors if none is
    /// active.
    ///
    /// Any DDL in the transaction updated the in-memory catalog cache in place, but
    /// the pager just reverted page 1 to its pre-transaction image. The two would
    /// disagree (a created table still cached, a dropped one still gone), so we
    /// reload the cache from the rolled-back page 1 to resync — the load-bearing fix
    /// that keeps the schema cache consistent with storage across a rollback.
    pub(crate) fn exec_rollback(&mut self) -> Result<()> {
        if !self.txn_active() {
            return Err(Error::sql("cannot rollback - no transaction is active"));
        }
        // Roll back every live namespace so a temp write made under this transaction is
        // undone too, then resync EACH namespace's schema cache to its rolled-back page 1
        // (the load-bearing fix — see the single-pager version this generalizes): a temp
        // DDL cached its def in the temp catalog, but the temp pager just reverted page 1.
        for i in 0..self.pagers.len() {
            self.pagers[i].rollback()?;
        }
        self.in_explicit_txn = false;
        self.savepoints.clear();
        // `PRAGMA defer_foreign_keys` is per-transaction and auto-clears at ROLLBACK
        // (pragma.html #pragma_defer_foreign_keys), the same as at COMMIT — clear BOTH the
        // live flag and its sticky commit-coverage flag.
        self.rt.reset_defer_foreign_keys();
        for i in 0..self.catalogs.len() {
            self.catalogs[i].load(&*self.pagers[i])?;
        }
        Ok(())
    }

    // ---- Autocommit (implicit single-statement) transaction bracket ------------
    //
    // These three drive ONLY the pager seam across EVERY live namespace and, unlike
    // `exec_begin`/`exec_commit`/`exec_rollback`, never touch `in_explicit_txn` /
    // `savepoints` — an autocommit statement's implicit transaction is not the
    // connection's explicit-transaction state, so the autocommit path
    // ([`SqlEngine::run_mutating_collect`]) must NOT route through those. A bare
    // mutating statement can reach namespaces beyond its top-level target at RUNTIME
    // (a trigger / INSTEAD-OF action, a nested-trigger recursion, an FK cascade all
    // pick their `node.db` during execution), so the implicit transaction must span
    // all live stores for the statement to be one atomic unit. With only `main` live
    // each is exactly the single-pager begin/commit/rollback it generalizes.

    /// Begin the implicit (autocommit) transaction on every live namespace, so a bare
    /// mutating statement's writes — including a fired action that reaches a different
    /// store — are all staged in one atomic unit. `main` (index 0) is begun FIRST, so a
    /// BUSY there aborts before any other store is begun; if a later begin fails, the
    /// already-begun pagers are rolled back so a partial failure leaves NO open
    /// transaction (the next statement's begin must not hit "a transaction is already
    /// active"). Eager `begin()` on each is safe: `temp` is in-memory and attached dbs
    /// are read-write here.
    pub(crate) fn implicit_begin_all(&mut self) -> Result<()> {
        for i in 0..self.pagers.len() {
            if let Err(e) = self.pagers[i].begin() {
                for j in 0..i {
                    let _ = self.pagers[j].rollback();
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Commit the implicit (autocommit) transaction on every live namespace opened by
    /// [`SqlEngine::implicit_begin_all`], making the statement's staged writes durable
    /// across every store it may have reached. `main` (index 0) is committed first. A
    /// store the statement never actually wrote commits an EMPTY overlay — a near-no-op.
    ///
    /// On a mid-loop commit failure the already-committed pagers stay durable (a multi-file
    /// partial commit — the crash-atomicity gap across two or more durable FILE databases,
    /// the same pre-existing limitation as [`SqlEngine::exec_commit`]; super-journal, out of
    /// scope here). The pagers NOT yet committed still hold an open implicit transaction, so
    /// they are rolled back before the error propagates — symmetric with
    /// [`SqlEngine::implicit_begin_all`]'s partial-failure cleanup — leaving NO pager
    /// stranded open. This matters because these helpers bypass `txn_active()`: a stranded
    /// open pager would make the next statement's `implicit_begin_all` hit "a transaction is
    /// already active" and wedge the connection. (A pager's own `commit` closes its
    /// transaction before the durable apply, so the pager whose commit failed is already
    /// closed; only the pagers after it remain open.) Per-pager rollback errors are swallowed
    /// so they cannot mask the original commit error.
    pub(crate) fn implicit_commit_all(&mut self) -> Result<()> {
        // The implicit transaction ends here (committed, or rolled back on failure), so
        // `PRAGMA defer_foreign_keys` switches off, as at an explicit COMMIT.
        self.rt.reset_defer_foreign_keys();
        for i in 0..self.pagers.len() {
            if let Err(e) = self.pagers[i].commit() {
                for j in (i + 1)..self.pagers.len() {
                    let _ = self.pagers[j].rollback();
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Roll back the implicit (autocommit) transaction on every live namespace, backing
    /// out the statement's staged writes across every store it may have reached (the
    /// atomic-abort half of the bracket). EVERY live pager is rolled back regardless of
    /// an individual failure — none may be left with an open transaction for the next
    /// statement — so a per-pager rollback error is swallowed here; the caller already
    /// holds the statement's own error to return, and a rollback of an un-dirtied pager
    /// cannot fail anyway (it just drops an empty overlay and releases the write lock).
    pub(crate) fn implicit_rollback_all(&mut self) {
        // Ending the implicit transaction switches `PRAGMA defer_foreign_keys` off, as at an
        // explicit ROLLBACK.
        self.rt.reset_defer_foreign_keys();
        for i in 0..self.pagers.len() {
            let _ = self.pagers[i].rollback();
        }
    }

    /// `SAVEPOINT name`: open a savepoint. If no transaction is active, one is started
    /// implicitly (in the pager) — the transaction then persists until the matching
    /// `RELEASE`, or a full `COMMIT`/`ROLLBACK`. Duplicate names are allowed and
    /// shadow: a later `RELEASE`/`ROLLBACK TO` resolves to the MOST RECENT savepoint
    /// with that name.
    pub(crate) fn exec_savepoint(&mut self, name: &str) -> Result<()> {
        // Like `BEGIN`, a SAVEPOINT that starts a transaction keeps a `PRAGMA
        // defer_foreign_keys` set just before it, and a nested one keeps the enclosing
        // transaction's value.
        // Open the savepoint on EVERY live namespace so a later RELEASE / ROLLBACK TO
        // spans temp too. The per-pager stacks are kept in lockstep — every savepoint op
        // is applied to all live pagers, and a temp store created mid-transaction replays
        // the current depth in `ensure_temp` — so each returns the SAME depth. `main`'s is
        // the one stored (paired with the name); the others are asserted to agree.
        let depth = self.pagers[0].savepoint()?;
        for i in 1..self.pagers.len() {
            let d = self.pagers[i].savepoint()?;
            debug_assert_eq!(d, depth, "namespace {i} savepoint stack drifted from main");
        }
        self.savepoints.push((name.to_string(), depth));
        Ok(())
    }

    /// `RELEASE [SAVEPOINT] name`: release the most recent savepoint with `name` and
    /// every savepoint above it, keeping their changes ("like a COMMIT for a
    /// SAVEPOINT"). An unknown name is an error and leaves the state unchanged.
    ///
    /// If the released savepoint is the outermost AND a `SAVEPOINT` (not `BEGIN`)
    /// started the transaction, the pager's release commits the transaction; the
    /// engine's autocommit state then follows from the now-empty savepoint stack with
    /// `in_explicit_txn` already false (see [`SqlEngine::txn_active`]).
    ///
    /// That outermost-savepoint release IS a COMMIT point (foreignkeys.html §4.2: a RELEASE
    /// that commits the transaction is "subject to the same restrictions as a COMMIT"), so the
    /// DEFERRED foreign-key recheck runs here too — BEFORE the pager release makes it durable
    /// and BEFORE the savepoint stack is truncated — and an unresolved deferred violation fails
    /// the RELEASE with the transaction (and its savepoints) LEFT OPEN, exactly like a failed
    /// `COMMIT`. A nested release, or a release inside an explicit `BEGIN`, does not end the
    /// transaction and so is not a commit point.
    pub(crate) fn exec_release(&mut self, name: &str) -> Result<()> {
        let pos = self
            .savepoints
            .iter()
            .rposition(|(n, _)| n == name)
            .ok_or_else(|| Error::sql(format!("no such savepoint: {name}")))?;
        let depth = self.savepoints[pos].1;
        // Releasing the OUTERMOST savepoint (pos 0) of a SAVEPOINT-started transaction (no
        // wrapping explicit `BEGIN`) ends the transaction — a COMMIT point. Run the deferred-FK
        // recheck first, so an unresolved violation returns Err here (before the release and
        // truncation below) and leaves the whole transaction open.
        let is_commit_point = pos == 0 && !self.in_explicit_txn;
        if is_commit_point {
            self.check_deferred_foreign_keys()?;
        }
        // Release at this depth on every live namespace (their stacks are in lockstep).
        // An outermost release (depth 0) of a savepoint-started transaction commits that
        // pager — temp's outermost savepoint is marked started-by-savepoint in
        // `ensure_temp` for exactly that case, so main and temp commit together.
        for i in 0..self.pagers.len() {
            self.pagers[i].release_savepoint(depth)?;
        }
        // Drop the released savepoint and every savepoint above it. Ordering: mutate
        // engine state only AFTER the pager succeeds, so a pager error leaves the
        // name stack intact and the savepoint still addressable.
        self.savepoints.truncate(pos);
        // A RELEASE that just committed the transaction ends it, so clear the per-transaction
        // `PRAGMA defer_foreign_keys` (both live + sticky), matching exec_commit/exec_rollback.
        if is_commit_point {
            self.rt.reset_defer_foreign_keys();
        }
        Ok(())
    }

    /// `ROLLBACK [TRANSACTION] TO [SAVEPOINT] name`: roll the transaction back to the
    /// most recent savepoint with `name`, KEEPING that savepoint (lang_savepoint §2:
    /// `ROLLBACK TO` "does not cancel the transaction"). Savepoints above it are
    /// discarded. An unknown name is an error and leaves the database unchanged
    /// (lang_savepoint §3).
    ///
    /// The pager reverted page 1 to the savepoint's image, so — as in
    /// [`SqlEngine::exec_rollback`] — reload the catalog cache to resync any DDL made
    /// (and now undone) since the savepoint. The transaction stays open.
    pub(crate) fn exec_rollback_to(&mut self, name: &str) -> Result<()> {
        let pos = self
            .savepoints
            .iter()
            .rposition(|(n, _)| n == name)
            .ok_or_else(|| Error::sql(format!("no such savepoint: {name}")))?;
        let depth = self.savepoints[pos].1;
        // Roll back to this depth on every live namespace, then resync each namespace's
        // schema cache to the reverted page 1 (like `exec_rollback`) — a temp DDL made
        // since the savepoint is undone in both the temp pager and its catalog cache.
        for i in 0..self.pagers.len() {
            self.pagers[i].rollback_to_savepoint(depth)?;
        }
        // Keep the named savepoint; discard only those above it.
        self.savepoints.truncate(pos + 1);
        for i in 0..self.catalogs.len() {
            self.catalogs[i].load(&*self.pagers[i])?;
        }
        Ok(())
    }

    /// Run a catalog mutation (a DDL registration/removal) inside the correct
    /// transaction: within the open transaction if one is active (explicit `BEGIN` or
    /// savepoint-started), otherwise as an implicit single-statement transaction
    /// (begin → op → commit) rolled back on error so a rejected DDL leaves no partial
    /// write.
    ///
    /// The closure receives the catalog and pager as disjoint borrows (they are
    /// distinct fields), which is why the mutation is expressed as a closure rather
    /// than a `&mut self` method call — the latter would borrow all of `self` at
    /// once and prevent lending both fields.
    ///
    /// On the implicit-transaction ERROR path the catalog cache is reloaded from the
    /// rolled-back page 1 (like [`SqlEngine::exec_rollback`]). A DDL closure may cache
    /// a def and THEN fail a later step in the same transaction — `CREATE INDEX`
    /// caches the new `IndexDef`, then its backfill can hit a UNIQUE violation among
    /// the existing rows — so the reload keeps the cache from disagreeing with the
    /// reverted storage (an orphaned index def whose b-tree was just discarded).
    pub(crate) fn with_write_txn<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut SchemaCatalog, &mut dyn Pager) -> Result<()>,
    {
        self.with_write_txn_in(DbIndex::MAIN, f)
    }

    /// Run a schema/DDL mutation against a SPECIFIC namespace `db`'s `(catalog, pager)` —
    /// the namespace form of [`with_write_txn`], used when the target is `temp` (a
    /// `CREATE TEMP …` / a `temp.`-qualified object) or an explicit `main`. The wrapping
    /// rule is unchanged: within an active transaction the mutation just runs; in autocommit
    /// it is bracketed by that namespace's own implicit `begin`/`commit`, rolled back (and
    /// its cache reloaded) on error so a rejected DDL leaves no partial write in that store.
    ///
    /// `catalogs[i]` and `pagers[i]` are distinct fields, so the closure still receives them
    /// as disjoint borrows (why the mutation is a closure, not a `&mut self` call).
    ///
    /// This runs against a LIVE namespace only, and MUST NOT materialize one. `temp`/
    /// `temporary` resolves to [`DbIndex::TEMP`] (index 1) unconditionally — even before the
    /// lazy store exists — so a naive `temp.`-qualified DROP/ALTER/CREATE INDEX could hand a
    /// DEAD index here and index a missing `pagers[1]`. Those non-live cases never reach
    /// this path: a temp `CREATE` materializes temp FIRST (in
    /// [`create_target_db`](SqlEngine::create_target_db)), and a `temp.`-qualified
    /// DROP/ALTER/CREATE INDEX issued before temp exists is run against a throwaway empty
    /// store by the dispatch arms ([`with_absent_namespace`](SqlEngine::with_absent_namespace))
    /// — those statements must not create the temp schema, so reserving temp
    /// here (as a "make every temp write safe" chokepoint) would be WRONG. The guard below is
    /// the standing backstop for any FUTURE mis-route, not a materialization point.
    pub(crate) fn with_write_txn_in<F>(&mut self, db: DbIndex, f: F) -> Result<()>
    where
        F: FnOnce(&mut SchemaCatalog, &mut dyn Pager) -> Result<()>,
    {
        let i = db.index();
        // Defensive invariant: `db` MUST be a live namespace here. The write-side DDL
        // routers ([`crate::namespace`] / the DDL arms in [`crate::dispatch`]) resolve the
        // one reachable non-live case — a `temp.` qualifier before the temp store is
        // materialized (see [`SqlEngine::namespace_live`]) — and run it against a throwaway
        // empty store BEFORE reaching here. Guard so any FUTURE non-live routing fails LOUDLY
        // in tests (the assert) and DEGRADES cleanly in release (a returned error, not the
        // out-of-bounds `pagers[i]`/`catalogs[i]` panic that this whole fix removes).
        debug_assert!(
            self.namespace_live(db),
            "with_write_txn_in: namespace {i} is not live (pagers.len() = {}); a DDL router \
             resolved a dead index — see SqlEngine::namespace_live",
            self.pagers.len()
        );
        if !self.namespace_live(db) {
            return Err(Error::sql("no such database"));
        }
        if self.txn_active() {
            return f(&mut self.catalogs[i], &mut *self.pagers[i]);
        }
        self.pagers[i].begin()?;
        // The implicit transaction ends below either way, which switches `PRAGMA
        // defer_foreign_keys` off (as at COMMIT / ROLLBACK).
        self.rt.reset_defer_foreign_keys();
        match f(&mut self.catalogs[i], &mut *self.pagers[i]) {
            Ok(()) => self.pagers[i].commit(),
            Err(e) => {
                // Revert the pages, then resync the cache to the rolled-back page 1 so
                // a def cached before a later same-transaction failure does not survive.
                // The closure's own error is what the caller needs; a rollback/reload
                // failure (storage-level) must not mask it.
                let _ = self.pagers[i].rollback();
                let _ = self.catalogs[i].load(&*self.pagers[i]);
                Err(e)
            }
        }
    }
}
