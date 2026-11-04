//! Integration tests for SECONDARY INDEXES on WITHOUT ROWID (WR) tables, end to end:
//! hand-built [`Plan`] trees run over a real [`MemPager`] + [`SchemaCatalog`] through the
//! [`Executor`]/[`RowCursor`] seam, seeded via the real WR `INSERT` operator and read back
//! through the real WR scan and a hand-built WR `IndexScan` (not a mock).
//!
//! A secondary index on a WR table keys each b-tree entry by `[indexed cols.., trailing
//! PRIMARY KEY cols not already indexed]` (fileformat2 §2.5.1) — the PRIMARY KEY playing the
//! role the rowid plays for a rowid-table index (it keeps entries distinct AND identifies the
//! row to fetch). The trailing PK is APPENDED in PK order, EXCEPT a PK column already among
//! the indexed columns with a matching collation is suppressed. These tests derive expected
//! behavior from `withoutrowid.html` + `fileformat2.html` §2.5 + `lang_createindex.html`.
//!
//! Coverage:
//! 1. non-unique index round-trip (write → hand-built IndexScan read, index order);
//! 2. UNIQUE index: duplicate errors `UNIQUE constraint failed: t.b`; `OR IGNORE` skips;
//!    `OR REPLACE` deletes the victim (PK row + index entry) and inserts;
//! 3. UPDATE of an INDEXED column moves the entry (findable under the new value);
//! 4. UPDATE of a PK column rewrites the trailing-PK part (findable under the unchanged
//!    indexed value, now carrying the new PK);
//! 5. DELETE removes the index entry (a re-insert of the freed indexed value succeeds);
//! 6. `CREATE INDEX` backfills existing rows; a pre-existing UNIQUE duplicate errors;
//! 7. a COMPOSITE PRIMARY KEY appends BOTH trailing PK columns (round-trip);
//! 8. an EXPRESSION index on a WR table stays fail-closed (the boundary kept out of scope).
//! Case 9 (no regression to the rowid index path) is covered by the existing suites
//! (`tests/indexscan.rs`, the DML tests) staying correct, not re-pinned here.
//!
//! Plus edge coverage beyond the enumerated cases: a NOCASE UNIQUE index (a conflict probe
//! that compares under the key column's collation) and UPDATE OR REPLACE / OR IGNORE resolving a
//! secondary-index UNIQUE conflict (the UPDATE side of the shared victim-cleanup path).
//!
//! DML assumes an open write transaction (the engine opens one), so each apply is wrapped in
//! `begin`/`commit`; read-back scans need no transaction.

use minisqlite_btree::{init_database, IndexCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{build_index, Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_expr::{ArithOp, CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{
    Delete, IndexOp, IndexScan, Insert, OnConflict, Plan, PlanNode, ScanDirection, SeqScan, Update,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, DbIndex, Error, Value};

// ----- fixtures ------------------------------------------------------------

/// A fresh in-memory database with one table created through the real catalog path.
fn db_with_table(create_sql: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(&mut pager, stmt, create_sql).unwrap();
    (pager, cat)
}

/// Create an index through the real catalog path (allocates an EMPTY index b-tree + writes
/// the schema row; does NOT backfill — that is [`build_index`]).
fn create_index(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateIndex(stmt) = &ast.statements[0] else {
        panic!("not a CREATE INDEX: {sql}");
    };
    cat.create_index(pager, stmt, sql).unwrap();
}

/// The per-column affinities a planner would attach, derived from the declared types.
fn affinities(cat: &SchemaCatalog, table: &str) -> Vec<Affinity> {
    cat.table(table)
        .unwrap()
        .unwrap()
        .columns
        .iter()
        .map(|c| affinity_of_declared_type(c.declared_type.as_deref()))
        .collect()
}

/// The root page of a named index's b-tree (to count its entries at the storage layer).
fn index_root(cat: &SchemaCatalog, index: &str) -> PageId {
    cat.index(index).unwrap().unwrap().root_page
}

/// Count entries in an index b-tree directly (independent of any operator).
fn entry_count(pager: &MemPager, root: PageId) -> usize {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut n = 0;
    if cur.first().unwrap() {
        n = 1;
        while cur.next().unwrap() {
            n += 1;
        }
    }
    n
}

// ----- expression / node helpers -------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}
fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}
fn txt(s: &str) -> EvalExpr {
    lit(Value::Text(s.into()))
}
fn intv(i: i64) -> EvalExpr {
    lit(Value::Integer(i))
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}
fn text(v: &Value) -> &str {
    match v {
        Value::Text(s) => s,
        other => panic!("expected Text, got {other:?}"),
    }
}

/// `left = right` under BINARY collation (the WHERE predicate for the filtered DML tests).
fn eq(left: EvalExpr, right: EvalExpr) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(right),
        meta: CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary },
    }
}

fn seqscan(table: &str, n: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n })
}

fn filtered(table: &str, n: usize, predicate: EvalExpr) -> PlanNode {
    PlanNode::Filter { input: Box::new(seqscan(table, n)), predicate }
}

// ----- plan builders -------------------------------------------------------

fn insert_plan(
    table: &str,
    n: usize,
    rows: Vec<Vec<EvalExpr>>,
    affs: Vec<Affinity>,
    on_conflict: OnConflict,
) -> Plan {
    let source_width = rows.first().map(|r| r.len()).unwrap_or(n);
    Plan {
        root: PlanNode::Insert(Insert {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            columns: None,
            source: Box::new(PlanNode::Values { rows }),
            source_width,
            column_affinities: affs,
            on_conflict,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: Vec::new(),
            rowid_source: None,
            upsert: None,
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    }
}

fn update_plan(
    table: &str,
    n: usize,
    assignments: Vec<(usize, EvalExpr)>,
    scan: PlanNode,
    affs: Vec<Affinity>,
    on_conflict: OnConflict,
) -> Plan {
    Plan {
        root: PlanNode::Update(Update {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            assignments,
            scan: Box::new(scan),
            column_affinities: affs,
            on_conflict,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: Vec::new(),
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    }
}

fn delete_plan(table: &str, n: usize, scan: PlanNode) -> Plan {
    Plan {
        root: PlanNode::Delete(Delete {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            scan: Box::new(scan),
            returning: Vec::new(),
            triggers: Vec::new(),
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    }
}

/// A width-N WR `IndexScan` plan over `index` (FullScan or an equality Seek), forward.
fn index_scan_plan(table: &str, n: usize, index: &str, op: IndexOp) -> Plan {
    Plan {
        root: PlanNode::IndexScan(IndexScan {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            index: index.to_string(),
            op,
            direction: ScanDirection::Forward,
            covering: false,
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

// ----- runners -------------------------------------------------------------

fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut out = Vec::new();
    {
        let mut cur = exec
            .execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })
            .unwrap();
        while let Some(row) = cur.next_row(rt).unwrap() {
            out.push(row);
        }
    }
    pager.commit().unwrap();
    out
}

/// Apply a mutating plan expected to FAIL: roll the statement back (as the engine does on a
/// constraint violation) and return the error.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Error {
    let mut rt = Runtime::new();
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut err = None;
    {
        match exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }) {
            Ok(mut cur) => loop {
                match cur.next_row(&mut rt) {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            },
            Err(e) => err = Some(e),
        }
    }
    pager.rollback().unwrap();
    err.expect("expected the statement to error")
}

/// A full read-back through the real WR scan: width-N rows `[c0..c_{N-1}]` in PRIMARY KEY
/// (storage) order.
fn scan_all(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
    let plan = Plan {
        root: seqscan(table, n),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    read_plan(&plan, cat, pager)
}

/// A read-back through a hand-built WR `IndexScan` over `index` (FullScan, forward): the
/// entries are visited in index-key order (`[indexed cols.., trailing PK..]`), and each row
/// is FETCHED from the PK b-tree by the recovered PRIMARY KEY, so the returned width-N rows
/// prove the whole write→read WR secondary-index round-trip.
fn index_scan_all(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize, index: &str) -> Vec<Vec<Value>> {
    read_plan(&index_scan_plan(table, n, index, IndexOp::FullScan), cat, pager)
}

/// A read-back through a WR `IndexScan` equality Seek on the single indexed column == `v`.
fn index_scan_eq(
    cat: &SchemaCatalog,
    pager: &mut MemPager,
    table: &str,
    n: usize,
    index: &str,
    v: Value,
) -> Vec<Vec<Value>> {
    let op = IndexOp::Seek { eq_prefix: vec![lit(v)], low: None, high: None };
    read_plan(&index_scan_plan(table, n, index, op), cat, pager)
}

fn read_plan(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec
        .execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })
        .unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

// ===== (1) non-unique secondary index: write → IndexScan read round-trip =====

#[test]
fn nonunique_index_round_trips_write_to_indexscan() {
    // INDEX i ON t(b) over t(a TEXT PRIMARY KEY, b INTEGER). Duplicate b values are allowed
    // (the trailing PK keeps entries distinct). A hand-built IndexScan returns the rows in
    // index-key order `[b, a]` — b ascending, ties by the trailing PK a — with the FULL row
    // contents fetched from the PK b-tree by the recovered PK.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");

    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![
                vec![txt("a1"), intv(10)],
                vec![txt("a2"), intv(20)],
                vec![txt("a3"), intv(10)], // duplicate b=10, distinct PK
                vec![txt("a4"), intv(30)],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 4, "all four rows insert (duplicate b is fine for a non-unique index)");
    assert_eq!(entry_count(&pager, index_root(&cat, "i")), 4, "one index entry per row");

    // IndexScan over i: entries ordered by [b, a] → (10,a1),(10,a3),(20,a2),(30,a4).
    let rows = index_scan_all(&cat, &mut pager, "t", 2, "i");
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(
        got,
        vec![("a1", 10), ("a3", 10), ("a2", 20), ("a4", 30)],
        "IndexScan yields rows in [b, trailing-PK a] order with full contents fetched by PK"
    );

    // An equality Seek on b=10 returns exactly the two duplicate-b rows (PK order).
    let hit = index_scan_eq(&cat, &mut pager, "t", 2, "i", Value::Integer(10));
    let hit_got: Vec<(&str, i64)> = hit.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(hit_got, vec![("a1", 10), ("a3", 10)], "Seek b=10 finds both rows via the index");
}

// ===== (2) UNIQUE secondary index: ABORT / IGNORE / REPLACE ==================

#[test]
fn unique_index_duplicate_aborts_with_the_column_detail() {
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER UNIQUE) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("k"), intv(5)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut rt,
    );
    // A second row with the same b (different PK) violates the UNIQUE secondary index.
    let err = run_err(
        &insert_plan("t", 2, vec![vec![txt("m"), intv(5)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
    );
    match err {
        Error::Constraint(m) => {
            assert!(m.contains("UNIQUE constraint failed"), "names UNIQUE, got {m:?}");
            assert!(m.contains("t.b"), "names the offending column t.b, got {m:?}");
        }
        other => panic!("expected a UNIQUE Constraint error, got {other:?}"),
    }
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 1, "the duplicate did not land");
}

#[test]
fn unique_index_or_ignore_skips_the_duplicate() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("k"), intv(5)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut rt,
    );
    let mut ig_rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("m"), intv(5)]], affinities(&cat, "t"), OnConflict::Ignore),
        &cat,
        &mut pager,
        &mut ig_rt,
    );
    assert_eq!(ig_rt.changes(), 0, "OR IGNORE consumed nothing on the UNIQUE conflict");
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "only the original row remains");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 5));
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 1, "index unchanged by the ignored row");
}

#[test]
fn unique_index_or_replace_deletes_victim_and_inserts() {
    // OR REPLACE on a UNIQUE secondary-index conflict deletes the CONFLICTING existing row —
    // its PK-b-tree row AND its index entry — then inserts the new row.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(6)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    // INSERT OR REPLACE ('n', 5): b=5 collides with 'k'; 'k' is evicted, 'n' inserted.
    let mut rep_rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("n"), intv(5)]], affinities(&cat, "t"), OnConflict::Replace),
        &cat,
        &mut pager,
        &mut rep_rt,
    );

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![("m", 6), ("n", 5)], "'k' evicted; 'm' and the new 'n' remain (PK order)");

    // The index reflects the replacement exactly: entries for b=5 (n) and b=6 (m), no stale 'k'.
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 2, "no orphaned index entry for evicted 'k'");
    let idx_rows = index_scan_all(&cat, &mut pager, "t", 2, "u");
    let idx_got: Vec<(&str, i64)> = idx_rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(idx_got, vec![("n", 5), ("m", 6)], "IndexScan shows b=5→n, b=6→m (k's entry gone)");
}

// ===== (3) UPDATE of an INDEXED column moves the entry =======================

#[test]
fn update_indexed_column_moves_the_index_entry() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(6)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    // UPDATE t SET b = 99 WHERE a = 'k'.
    let mut up_rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, intv(99))],
            filtered("t", 2, eq(col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut up_rt,
    );
    assert_eq!(up_rt.changes(), 1);

    // The OLD entry (b=5) is gone; the NEW entry (b=99) is present and findable via the index.
    assert!(index_scan_eq(&cat, &mut pager, "t", 2, "i", Value::Integer(5)).is_empty(), "old b=5 entry gone");
    let hit = index_scan_eq(&cat, &mut pager, "t", 2, "i", Value::Integer(99));
    assert_eq!(hit.len(), 1, "the row is findable under the new indexed value");
    assert_eq!((text(&hit[0][0]), int(&hit[0][1])), ("k", 99), "and carries the updated contents");
    assert_eq!(entry_count(&pager, index_root(&cat, "i")), 2, "still one entry per row (moved, not duplicated)");
}

// ===== (4) UPDATE of a PK column rewrites the trailing-PK part ===============

#[test]
fn update_pk_column_rewrites_trailing_pk_in_index_entry() {
    // The index entry is `[b, a]` (a is the trailing PK). Changing the PK column `a` — WITHOUT
    // touching the indexed column `b` — must still rewrite EVERY index entry for that row (its
    // trailing-PK part changed). The row stays findable by the unchanged b and now carries the
    // new PK, and a fetch-by-PK through the index resolves the moved PK row.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("old"), intv(7)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut rt,
    );

    // UPDATE t SET a = 'new' WHERE a = 'old' (b unchanged).
    let mut up_rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(0, txt("new"))],
            filtered("t", 2, eq(col(0), txt("old"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut up_rt,
    );
    assert_eq!(up_rt.changes(), 1);

    // Findable by the unchanged indexed value; the fetched row carries the NEW PK (proving the
    // trailing PK in the entry was rewritten AND the fetch-by-PK follows it to the moved row).
    let hit = index_scan_eq(&cat, &mut pager, "t", 2, "i", Value::Integer(7));
    assert_eq!(hit.len(), 1, "still exactly one entry for b=7");
    assert_eq!((text(&hit[0][0]), int(&hit[0][1])), ("new", 7), "entry now carries PK 'new'");
    // The old PK row is gone from the table; only the new PK exists.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "new", "PK moved old→new");
}

// ===== (5) DELETE removes the index entry ===================================

#[test]
fn delete_removes_index_entry_and_frees_the_unique_value() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(6)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    // DELETE FROM t WHERE a = 'k'.
    let mut del_rt = Runtime::new();
    run(&delete_plan("t", 2, filtered("t", 2, eq(col(0), txt("k")))), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1);

    // The index no longer returns the deleted row, and its b value is free to reuse under
    // the UNIQUE index — only possible if the delete removed the index entry, not just the row.
    assert!(index_scan_eq(&cat, &mut pager, "t", 2, "u", Value::Integer(5)).is_empty(), "b=5 entry gone");
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 1, "one index entry remains (m)");
    let mut re_rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("z"), intv(5)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut re_rt,
    );
    assert_eq!(re_rt.changes(), 1, "re-inserting the freed b=5 succeeds under the UNIQUE index");
    let hit = index_scan_eq(&cat, &mut pager, "t", 2, "u", Value::Integer(5));
    assert_eq!((text(&hit[0][0]), int(&hit[0][1])), ("z", 5), "b=5 now maps to the new row");
}

// ===== (6) CREATE INDEX backfill over an existing WR table ===================

#[test]
fn create_index_backfills_existing_wr_rows() {
    // Rows exist BEFORE the index. `create_index` allocates an empty index; `build_index`
    // backfills every existing row. A hand-built IndexScan then returns them all, in index
    // order, fetched by PK.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("a1"), intv(30)], vec![txt("a2"), intv(10)], vec![txt("a3"), intv(20)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
    pager.begin().unwrap();
    build_index(&mut pager, &cat, "i", &[], &[], None).unwrap();
    pager.commit().unwrap();
    assert_eq!(entry_count(&pager, index_root(&cat, "i")), 3, "every existing row was backfilled");

    let rows = index_scan_all(&cat, &mut pager, "t", 2, "i");
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(
        got,
        vec![("a2", 10), ("a3", 20), ("a1", 30)],
        "backfilled entries are in [b, PK] order, contents fetched by PK"
    );
}

#[test]
fn create_unique_index_backfill_errors_on_a_preexisting_duplicate() {
    // A pre-existing duplicate under a UNIQUE index is a `UNIQUE constraint failed` during
    // backfill (matching sqlite), aborting the create.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(5)]], // duplicate b=5
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    pager.begin().unwrap();
    let res = build_index(&mut pager, &cat, "u", &[], &[], None);
    pager.rollback().unwrap();
    match res {
        Err(Error::Constraint(m)) => {
            assert!(m.contains("UNIQUE constraint failed"), "names the UNIQUE violation, got {m:?}");
            assert!(m.contains("t.b"), "names the column t.b, got {m:?}");
        }
        other => panic!("expected a UNIQUE Constraint error from backfill, got {other:?}"),
    }
}

// ===== (7) composite PRIMARY KEY appends BOTH trailing PK columns ============

#[test]
fn composite_pk_appends_both_trailing_pk_columns() {
    // t(a, b, c) PRIMARY KEY(a, c), INDEX i ON t(b). The index entry is `[b, a, c]` — the
    // trailing PK is a,c (neither is the indexed column b), appended in PK order. A round-trip
    // through the IndexScan (fetch-by-PK reconstructs (a,c) from the entry) proves BOTH trailing
    // PK columns are carried and used.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT, b INTEGER, c TEXT, PRIMARY KEY(a, c)) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            3,
            vec![
                vec![txt("a1"), intv(20), txt("c1")],
                vec![txt("a2"), intv(10), txt("c2")],
                vec![txt("a1"), intv(10), txt("c9")], // same a, different c — a distinct composite PK
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 3);

    // IndexScan over i: entries ordered by [b, a, c] → (10,a1,c9),(10,a2,c2),(20,a1,c1).
    let rows = index_scan_all(&cat, &mut pager, "t", 3, "i");
    let got: Vec<(&str, i64, &str)> =
        rows.iter().map(|r| (text(&r[0]), int(&r[1]), text(&r[2]))).collect();
    assert_eq!(
        got,
        vec![("a1", 10, "c9"), ("a2", 10, "c2"), ("a1", 20, "c1")],
        "entries ordered by [b, trailing-PK a, c]; full 3-column rows fetched by the composite PK"
    );

    // An equality Seek on b=10 finds exactly the two b=10 rows, each fetched by its (a,c) PK.
    let hit = index_scan_eq(&cat, &mut pager, "t", 3, "i", Value::Integer(10));
    let hit_got: Vec<(&str, i64, &str)> =
        hit.iter().map(|r| (text(&r[0]), int(&r[1]), text(&r[2]))).collect();
    assert_eq!(hit_got, vec![("a1", 10, "c9"), ("a2", 10, "c2")], "both b=10 composite-PK rows found");
}

// ===== (8) expression index on a WR table stays fail-closed =================

#[test]
fn expression_index_on_wr_table_fails_closed() {
    // The planner binds an index key expression against a `[c0.., rowid]` frame a WR row has
    // no rowid for, so an EXPRESSION index on a WITHOUT ROWID table cannot be encoded here and
    // is refused LOUD (kept out of scope). The shared index-key builder raises it
    // for every path; here we drive the CREATE INDEX backfill with a compiled key expression.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("k"), intv(1)]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut rt,
    );

    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(b + 1)");
    // Supply the compiled key expression (`b + 1`) so the builder resolves an EXPRESSION key
    // part and reaches the WR-specific fail-closed guard (rather than the uncompiled-expr one).
    let compiled = vec![Some(EvalExpr::Arith {
        op: ArithOp::Add,
        left: Box::new(EvalExpr::Column(1)),
        right: Box::new(EvalExpr::Literal(Value::Integer(1))),
    })];
    pager.begin().unwrap();
    let res = build_index(&mut pager, &cat, "i", &[], &compiled, None);
    pager.rollback().unwrap();
    match res {
        Err(Error::Sql(m)) => {
            assert!(
                m.contains("expression index") && m.contains("WITHOUT ROWID"),
                "the guard names the expression-index-on-WITHOUT-ROWID limitation, got {m:?}"
            );
        }
        other => panic!("expected the WR expression-index fail-closed Sql error, got {other:?}"),
    }
}

// ===== (extra) collation-aware WR UNIQUE probe (NOCASE) =====================

#[test]
fn unique_index_nocase_collation_detects_case_folded_duplicate() {
    // A UNIQUE secondary index with a NOCASE key column is ordered under NOCASE, so the
    // conflict probe's prefix seek must use that order and compare under NOCASE: a BINARY
    // seek or compare would miss the case-folded duplicate. 'abc' and 'ABC' are a NOCASE
    // duplicate → the second insert errors; OR IGNORE skips it; a genuinely distinct value
    // inserts. Pins the collated WR probe that every other WR UNIQUE test (all `b INTEGER`,
    // BINARY) leaves unexercised.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b TEXT) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b COLLATE NOCASE)");
    let mut rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("k"), txt("abc")]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut rt,
    );

    // 'ABC' collides with 'abc' under NOCASE (different PK) → UNIQUE violation via the full scan.
    let err = run_err(
        &insert_plan("t", 2, vec![vec![txt("m"), txt("ABC")]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
    );
    match err {
        Error::Constraint(m) => {
            assert!(m.contains("UNIQUE constraint failed"), "names UNIQUE, got {m:?}");
            assert!(m.contains("t.b"), "names t.b, got {m:?}");
        }
        other => panic!("expected a UNIQUE Constraint error from the NOCASE probe, got {other:?}"),
    }
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 1, "the case-folded duplicate did not land");

    // OR IGNORE on the same NOCASE duplicate consumes nothing.
    let mut ig_rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("m"), txt("ABC")]], affinities(&cat, "t"), OnConflict::Ignore),
        &cat,
        &mut pager,
        &mut ig_rt,
    );
    assert_eq!(ig_rt.changes(), 0, "OR IGNORE skipped the NOCASE duplicate");
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 1, "index still has just the one entry");

    // A NOCASE-distinct value (not merely a case fold) is not a conflict and inserts.
    let mut ok_rt = Runtime::new();
    run(
        &insert_plan("t", 2, vec![vec![txt("m"), txt("xyz")]], affinities(&cat, "t"), OnConflict::Abort),
        &cat,
        &mut pager,
        &mut ok_rt,
    );
    assert_eq!(ok_rt.changes(), 1, "a NOCASE-distinct value is accepted");
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 2, "two distinct NOCASE entries");
}

// ===== (extra) UPDATE resolving a WR secondary UNIQUE conflict ==============

#[test]
fn update_or_replace_resolves_a_secondary_unique_conflict() {
    // UPDATE OR REPLACE that moves an indexed value onto ANOTHER row's value collides on the
    // UNIQUE secondary index (the updated row keeps its PK, so this is NOT a PK conflict).
    // REPLACE evicts the victim — its PK row AND its index entry — via the shared
    // `wr_delete_victim_by_pk`, then the update lands. Exercises the UPDATE secondary-conflict
    // REPLACE path (INSERT covers its own; ABORT is covered elsewhere).
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(6)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    // UPDATE OR REPLACE t SET b = 6 WHERE a = 'k' — b=6 collides with 'm'; 'm' is evicted.
    let mut up_rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, intv(6))],
            filtered("t", 2, eq(col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut up_rt,
    );
    assert_eq!(up_rt.changes(), 1, "the update counts once; the evicted victim does not");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![("k", 6)], "the REPLACE victim 'm' is gone; 'k' now holds b=6");
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 1, "one index entry (no orphan for 'm')");
    assert!(index_scan_eq(&cat, &mut pager, "t", 2, "u", Value::Integer(5)).is_empty(), "old b=5 gone");
    let hit = index_scan_eq(&cat, &mut pager, "t", 2, "u", Value::Integer(6));
    assert_eq!((text(&hit[0][0]), int(&hit[0][1])), ("k", 6), "b=6 now maps to 'k'");
}

#[test]
fn update_or_ignore_skips_a_secondary_unique_conflict() {
    // UPDATE OR IGNORE that would move an indexed value onto another row's value hits the
    // UNIQUE secondary index and SKIPS the offending row, leaving both rows untouched (the
    // IGNORE branch of the UPDATE secondary-conflict path).
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_index(&mut cat, &mut pager, "CREATE UNIQUE INDEX u ON t(b)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            vec![vec![txt("k"), intv(5)], vec![txt("m"), intv(6)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    // UPDATE OR IGNORE t SET b = 6 WHERE a = 'k' — collides with 'm'; 'k' is left unchanged.
    let mut up_rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, intv(6))],
            filtered("t", 2, eq(col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Ignore,
        ),
        &cat,
        &mut pager,
        &mut up_rt,
    );
    assert_eq!(up_rt.changes(), 0, "the conflicting row was skipped, nothing updated");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![("k", 5), ("m", 6)], "both rows unchanged after the ignored update");
    assert_eq!(entry_count(&pager, index_root(&cat, "u")), 2, "both index entries intact");
}
