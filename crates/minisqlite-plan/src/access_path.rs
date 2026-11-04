//! Access-path selection: [`plan_table_access`] chooses how a single base table is
//! reached — a [`RowidScan`] point-lookup / range on the integer rowid key, an
//! [`IndexScan`] seek / range over a secondary index, or a full [`SeqScan`] of last
//! resort — and returns the leftover ("residual") predicate the caller must still
//! apply as a [`Filter`].
//!
//! The choice is a small cost model over the constraints the bound WHERE places on
//! the table. Selectivity, strongest first: a rowid `=` (one row) ▷ a full-equality
//! seek of a UNIQUE index (one row) ▷ an index equality-prefix ▷ a rowid range ▷ an
//! index range ▷ the full scan. A table uses at most ONE index (no OR / index merge);
//! any constraint the chosen path does not consume stays in the residual.
//!
//! ## Correctness at the executor boundary (load-bearing)
//! The index executor ([`minisqlite_exec`]'s `index_scan`) seeks the index b-tree
//! which is ordered under [`Collation::Binary`] only, and it evaluates each seek /
//! bound expression **raw** — it applies neither operand affinity nor a NULL guard.
//! So a term is only turned into an index seek when that raw seek is provably right:
//!
//! * the comparison's resolved collation is `Binary` (else the index order can't
//!   serve it);
//! * the comparison does not force an affinity onto the COLUMN side. The index keys
//!   were written under the column's own storage affinity and ordered under Binary; a
//!   comparison that coerces the column to a *foreign* class (datatype3 §4.2 applies
//!   NUMERIC to a TEXT/NONE column facing a numeric operand, or TEXT to a NONE column
//!   facing a text one — e.g. `text_col = CAST(5 AS INTEGER)`) would need the index
//!   rebuilt under that coercion, so a raw seek would land in the wrong storage class
//!   and silently drop rows. A non-`None` column-side affinity therefore declines the
//!   index (the term falls to the residual, where the Filter applies the affinity);
//! * a **literal** value has its operand affinity applied HERE, at plan time, into a
//!   concrete non-null value, so the seek key matches exactly what the index stores
//!   — such a term is fully consumed;
//! * a **runtime** value (a bind parameter, …) is used only when no affinity coercion
//!   is required of it (otherwise the raw seek could miss rows the index stored under
//!   the coerced value); and because it might be NULL at run time, the term is ALSO
//!   kept in the residual so `col = NULL` (empty) is filtered correctly rather than
//!   matching the index's NULL entries. (A subquery is never a seek value — it is
//!   row-dependent per [`has_column_ref`], since a correlated subplan is invisible
//!   here, so it declines at the row-independence gate before this branch.)
//!
//! The rowid access path mirrors this shape (its helpers `rowid_eq_value` /
//! `rowid_bound` mirror the index matchers), but the EQUALITY path is stronger than the
//! index one: the rowid eq executor (`eval_rowid_eq`) coerces its seek value with
//! sqlite's `OP_SeekRowid` rule (`must_be_int`: NUMERIC affinity, then a lossless
//! integer, else no rows), so a rowid `=` seeks any value by that rule. A literal is
//! still folded to a concrete `Integer` HERE (and a NULL literal declines, `rowid =
//! NULL` being empty), while a runtime value (a `?` param, an expression) is emitted RAW
//! and coerced at run time — no affinity gate on the eq path. The rowid RANGE path stays
//! weaker: its executor coerces a bound with a lossy `to_integer` (NULL → 0, a fractional
//! real truncated), so a rowid *range* bound is used ONLY when it resolves here to a
//! concrete `Integer` (a runtime or non-integer bound is left to the residual).
//!
//! Consuming FEWER terms (leaving more residual) is always safe — it only adds a
//! redundant filter — so every uncertain case falls back to the residual.
//!
//! [`Filter`]: crate::PlanNode::Filter
//! [`Collation::Binary`]: minisqlite_types::Collation

use minisqlite_catalog::{Catalog, IndexDef, KeyColumn, TableDef};
use minisqlite_expr::{CmpOp, EvalExpr};
use minisqlite_types::{apply_affinity, Affinity, Collation, DbIndex, Result, Value};

use crate::access::{IndexOp, IndexScan, RangeBound, RowidOp, RowidScan, ScanDirection, SeqScan};
use crate::bind::parse_collation;
use crate::plan::PlanNode;

/// The chosen access path for one table: the leaf operator plus the predicate
/// that the leaf did NOT satisfy and so must still be filtered.
pub struct TableAccess {
    /// The access-path leaf node (`SeqScan`, `RowidScan`, or `IndexScan`).
    pub node: PlanNode,
    /// The predicate the leaf could not consume (`None` = fully consumed / no WHERE).
    pub residual: Option<EvalExpr>,
}

/// Choose the access path for `td` given the already-bound WHERE predicate
/// (`None` = no WHERE). The predicate references the single-table layout
/// `[c0..c_{N-1}, rowid]`, so a rowid reference is `Column(N)` and a column `i`
/// reference is `Column(i)` (an `INTEGER PRIMARY KEY` alias binds to `Column(N)`).
///
/// `catalog` is consulted for the table's secondary indexes (`indexes_on`) so an
/// equality / range with a usable index becomes a seek rather than a full scan.
///
/// `db` is the database namespace `td` was resolved in (`main`/`temp`/attached); it is
/// stamped onto every access node so the executor opens its cursor on the right store.
/// `td`'s table and any index it uses are always in the SAME namespace (SQLite forbids
/// a cross-database index), so one `db` covers both b-tree cursors.
pub fn plan_table_access(
    catalog: &dyn Catalog,
    td: &TableDef,
    db: DbIndex,
    bound_where: Option<EvalExpr>,
) -> Result<TableAccess> {
    let n = td.columns.len();
    let table = td.name.clone();

    // A WITHOUT ROWID table has no integer rowid and stores its rows in a PRIMARY KEY
    // index b-tree (withoutrowid.html). The rowid access path (`RowidScan`) has no
    // integer key to seek, and the secondary-index path (`IndexScan`) reads a trailing
    // rowid out of each index entry — a WR secondary index carries the PRIMARY KEY there
    // instead, so that executor would misread it. Until a WR-aware PK-seek / index access
    // path exists, force a full `SeqScan` and keep the whole WHERE as the residual Filter.
    // That is always correct — a superset scan the Filter narrows — just not yet
    // index-optimized. The scan is served by the executor's WR index-b-tree iteration and
    // emits exactly `n` columns (no rowid register), matching `Source::width()` for a WR
    // base table. FOLLOW-UP: a PK-prefix seek into the WR b-tree and a WR-aware IndexScan.
    if td.without_rowid {
        return Ok(TableAccess {
            node: PlanNode::SeqScan(SeqScan { table, db, column_count: n }),
            residual: bound_where,
        });
    }

    let rowid_reg = n;

    let Some(pred) = bound_where else {
        return Ok(TableAccess {
            node: PlanNode::SeqScan(SeqScan { table, db, column_count: n }),
            residual: None,
        });
    };

    let mut conjuncts = Vec::new();
    flatten_and(pred, &mut conjuncts);
    let conjuncts = expand_betweens(conjuncts);

    // A rowid equality is the strongest path: at most one row via a single seek.
    if let Some((i, value)) =
        conjuncts.iter().enumerate().find_map(|(i, c)| rowid_eq_value(c, rowid_reg).map(|v| (i, v)))
    {
        let mut conjuncts = conjuncts;
        conjuncts.remove(i);
        let node = PlanNode::RowidScan(RowidScan {
            table,
            db,
            column_count: n,
            op: RowidOp::Eq(value),
            direction: ScanDirection::Forward,
        });
        return Ok(TableAccess { node, residual: rebuild_and(conjuncts) });
    }

    // Otherwise weigh the best index path against a rowid range, by the cost order.
    // `db` is the target's resolved namespace (stamped on SeqScan/RowidScan above): read its
    // indexes with `_in(db)` so a same-named temp/attached shadow can't feed the planner
    // another namespace's indexes. With no shadow `db == MAIN` and this equals the bare form.
    let index_defs = catalog.indexes_on_in(db, &td.name)?;
    let index_cand = choose_best_index(&index_defs, &conjuncts, td, n, &table, db);
    let rowid_cand = compute_rowid_range(&conjuncts, rowid_reg, &table, n, db);

    // Weigh the two candidates on the shared tier scale. An index beats a rowid range
    // only when it is an equality path (tier above `TIER_ROWID_RANGE`); a bare index
    // range (`TIER_INDEX_RANGE`) loses to a rowid range. Matching the OWNED options and
    // moving the winner's `node`/`remove` out per arm makes "chose an index but have no
    // index candidate" unrepresentable — no `use_index` bool, no `.expect()`.
    let chosen: Option<(PlanNode, Vec<usize>)> = match (index_cand, rowid_cand) {
        (Some(ic), Some(rc)) => {
            if ic.score.0 > TIER_ROWID_RANGE {
                Some((ic.node, ic.remove))
            } else {
                Some((rc.node, rc.remove))
            }
        }
        (Some(ic), None) => Some((ic.node, ic.remove)),
        (None, Some(rc)) => Some((rc.node, rc.remove)),
        (None, None) => None,
    };

    match chosen {
        Some((node, remove)) => {
            let residual = residual_without(conjuncts, &remove);
            Ok(TableAccess { node, residual })
        }
        None => Ok(TableAccess {
            node: PlanNode::SeqScan(SeqScan { table, db, column_count: n }),
            residual: rebuild_and(conjuncts),
        }),
    }
}

// ---------------------------------------------------------------------------
// Index access-path candidate selection.
// ---------------------------------------------------------------------------

/// Access-path cost tiers on a single shared scale (higher = fewer rows visited). An
/// `IndexCandidate` stores its tier in `score.0`; the rowid-range path has no candidate
/// struct, so its tier is implicit and only ever compared against an index tier in
/// [`plan_table_access`]. Keeping all four on one scale is what lets that comparison
/// (`index_tier > TIER_ROWID_RANGE`) stay correct if a tier is ever added.
const TIER_UNIQUE_FULL_EQ: u8 = 5; // every column pinned on a UNIQUE index → ≤ 1 row
const TIER_INDEX_EQ_PREFIX: u8 = 4; // an equality prefix (possibly a trailing range)
const TIER_ROWID_RANGE: u8 = 3; // a rowid range seek (implicit; no IndexCandidate)
const TIER_INDEX_RANGE: u8 = 2; // a pure leading-column index range

/// A candidate index access path plus its cost score and the conjunct indices it
/// fully consumed (removed from the residual). Terms used only as a *superset* seek
/// are NOT in `remove` — they stay in the residual as a filter. Two kinds are only a
/// superset: a runtime value that might be NULL at run time, and an upper-bound-only
/// range (`a <= k` with no lower bound), whose scan would otherwise emit the
/// index's leading NULL-keyed rows (`NULL <= k` is NULL, not true).
struct IndexCandidate {
    node: PlanNode,
    /// `(tier, equality_prefix_len, has_range)`, compared lexicographically. Higher
    /// is better; ties keep the first (creation-order) index.
    score: (u8, usize, bool),
    remove: Vec<usize>,
}

/// A candidate rowid range access path (`lo <= rowid <= hi`, either side optional).
struct RowidRangeCandidate {
    node: PlanNode,
    remove: Vec<usize>,
}

/// Pick the best usable index for `conjuncts`, or `None` if none applies. Iterates in
/// catalog (creation) order and keeps the first index at the best score.
fn choose_best_index(
    indexes: &[&IndexDef],
    conjuncts: &[EvalExpr],
    td: &TableDef,
    n: usize,
    table: &str,
    db: DbIndex,
) -> Option<IndexCandidate> {
    let mut best: Option<IndexCandidate> = None;
    for idx in indexes {
        // A partial index (`CREATE INDEX ... WHERE p`) covers only the rows matching
        // its unmodelled predicate, so it can NEVER stand in for a general scan.
        if idx.partial {
            continue;
        }
        if let Some(cand) = build_index_candidate(idx, conjuncts, td, n, table, db) {
            if best.as_ref().is_none_or(|b| cand.score > b.score) {
                best = Some(cand);
            }
        }
    }
    best
}

/// Build the tightest access this one index can offer for `conjuncts`, or `None` if
/// it constrains nothing usable. The equality prefix is the longest CONTIGUOUS run of
/// index columns (from the first) each pinned by a usable `=` / `IS`; the column right
/// after the prefix may then take a lower and/or upper range bound (the "sandwich").
fn build_index_candidate(
    idx: &IndexDef,
    conjuncts: &[EvalExpr],
    td: &TableDef,
    n: usize,
    table: &str,
    db: DbIndex,
) -> Option<IndexCandidate> {
    let mut used = vec![false; conjuncts.len()];
    let mut eq_prefix: Vec<EvalExpr> = Vec::new();
    let mut remove: Vec<usize> = Vec::new();

    // Longest contiguous equality prefix, index column 0, 1, 2, …
    let mut p = 0usize;
    while p < idx.columns.len() {
        let Some(reg) = col_reg(td, &idx.columns[p], n) else { break };
        let mut matched: Option<(usize, EvalExpr, bool)> = None;
        for (ci, c) in conjuncts.iter().enumerate() {
            if used[ci] {
                continue;
            }
            if let Some((value, consume)) = match_index_eq(c, reg) {
                matched = Some((ci, value, consume));
                break;
            }
        }
        match matched {
            Some((ci, value, consume)) => {
                used[ci] = true;
                eq_prefix.push(value);
                if consume {
                    remove.push(ci);
                }
                p += 1;
            }
            None => break,
        }
    }

    // A lower and/or upper range bound on the single column after the equality prefix.
    let mut low: Option<RangeBound> = None;
    let mut high: Option<RangeBound> = None;
    // A consumable upper bound's conjunct index, resolved into `remove` only AFTER the
    // loop. An upper bound is fully consumed ONLY when a lower bound is also present to
    // seek the scan past the NULL keys (see the NULL-exclusion note below); the lower
    // bound may appear later in conjunct order (`a <= 5 AND a >= 2`), so the decision
    // cannot be made in-loop.
    let mut consumable_high: Option<usize> = None;
    if p < idx.columns.len() {
        if let Some(reg) = col_reg(td, &idx.columns[p], n) {
            for (ci, c) in conjuncts.iter().enumerate() {
                if used[ci] {
                    continue;
                }
                if let Some((side, bound, consume)) = match_index_range(c, reg) {
                    match side {
                        Side::Lo if low.is_none() => {
                            low = Some(bound);
                            used[ci] = true;
                            if consume {
                                remove.push(ci);
                            }
                        }
                        Side::Hi if high.is_none() => {
                            high = Some(bound);
                            used[ci] = true;
                            if consume {
                                consumable_high = Some(ci);
                            }
                        }
                        _ => continue,
                    }
                }
            }
        }
    }
    // NULL-exclusion for an UPPER-bound-only range (lang_select §2.3; datatype3 §3).
    // NULL keys sort BEFORE every value in the index b-tree, so a range with an upper
    // bound but NO lower bound starts its scan on the NULL-keyed entries (a forward
    // scan begins at the group's first entry; a reverse scan reaches them last), and
    // the executor's `index_scan` reads each bound RAW — it treats `NULL <= k` as
    // in-range because NULL sorts below `k`. But `NULL <= k` is NULL, not true, so those
    // rows must be excluded. When a lower bound is present the seek starts PAST the
    // NULLs and the upper bound is exact; without one, keep the upper comparison in the
    // residual so its Filter drops the NULL rows. This is a "superset seek": the upper
    // bound still limits how far the scan runs, and the residual Filter removes the
    // leading NULL-keyed rows the raw seek could not.
    if let Some(ci) = consumable_high {
        if low.is_some() {
            remove.push(ci);
        }
    }

    let has_range = low.is_some() || high.is_some();
    if p == 0 && !has_range {
        return None; // the index constrains nothing
    }
    let tier: u8 = if p == idx.columns.len() && idx.unique {
        TIER_UNIQUE_FULL_EQ
    } else if p >= 1 {
        TIER_INDEX_EQ_PREFIX
    } else {
        TIER_INDEX_RANGE // p == 0 && has_range → a pure leading-column range
    };
    let node = PlanNode::IndexScan(IndexScan {
        table: table.to_string(),
        db,
        column_count: n,
        index: idx.name.clone(),
        op: IndexOp::Seek { eq_prefix, low, high },
        direction: ScanDirection::Forward,
        // Covering is decided later, by the whole-plan post-pass
        // [`crate::compile::covering`], which alone can see every column the query reads
        // above this leaf. Access-path selection builds the leaf `covering: false` (the
        // correct table-fetch path); the post-pass flips it to `true` only when provably safe.
        covering: false,
    });
    Some(IndexCandidate { node, score: (tier, p, has_range), remove })
}

/// The row register a named index column reads from, mirroring the binder's
/// [`Scope`](crate::Scope) mapping: column `i` is register `i`, unless it is the
/// `INTEGER PRIMARY KEY` alias, which reads the rowid register `N`. `None` if the
/// name is not a column of `td` (a catalog inconsistency — decline the index).
///
/// `pub` so the ORDER-BY scan optimizer ([`crate::compile::order_scan`]), the covering-index
/// post-pass ([`crate::compile::covering`]), and the executor's index-only read all map an
/// index column to its register through the SAME function the WHERE path uses — one source of
/// truth for that register mapping, so the planner's covering decision and the executor's
/// covering fill cannot disagree. (The binder's `column_register` in [`crate::bind`] encodes
/// the same rowid-alias->N rule independently, keyed by index/base-offset over a `Source`
/// rather than by name over a `TableDef`; it is a separate copy, not funneled through here.)
pub fn col_reg(td: &TableDef, name: &str, n: usize) -> Option<usize> {
    let pos = td.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name))?;
    Some(if td.rowid_alias == Some(pos) { n } else { pos })
}

/// Whether an index key column's DECLARED collation resolves to [`Collation::Binary`]
/// (`None` inherits, taken as Binary; an explicit `BINARY` also qualifies; NOCASE / RTRIM
/// decline). The one source of truth for the "is this index key column usable for a
/// BINARY-ordered walk" rule, shared by BOTH b-tree-order optimizers — the ORDER-BY scan
/// rewrite ([`crate::compile::order_scan`]) and the MIN/MAX seek
/// ([`crate::compile::minmax_index`]) — so the two cannot disagree about which indexes are
/// eligible. Today the index b-tree is physically Binary-ordered regardless of this field
/// (see `minisqlite_btree::index_key`); guarding on the DECLARED collation keeps both
/// optimizations correct if per-column index collation is ever honored, and keeps that
/// future relaxation a ONE-place change rather than two copies that must move in lockstep.
pub(crate) fn key_column_is_binary(kc: &KeyColumn) -> bool {
    match &kc.collation {
        None => true,
        Some(name) => matches!(parse_collation(name), Ok(Collation::Binary)),
    }
}

/// If `e` is a usable equality (`col = value` / `col IS value`) on the index column
/// at register `col_reg`, return `(seek_value, consume)`: the value to place in the
/// seek prefix and whether the term is fully consumed (removed from the residual).
///
/// Usable requires a `Binary` collation (the index's key order) and no forced
/// column-side affinity (either would make the Binary-ordered index keys the wrong
/// thing to seek). A literal value gets its operand affinity applied here and is
/// consumed exactly; a runtime value is used raw only when it needs no affinity, and
/// is kept in the residual (so a NULL binding is still filtered — `col = NULL` is
/// empty, not the index's NULL rows).
fn match_index_eq(e: &EvalExpr, col_reg: usize) -> Option<(EvalExpr, bool)> {
    let EvalExpr::Compare { op: CmpOp::Eq, null_safe, left, right, meta } = e else {
        return None;
    };
    if meta.collation != Collation::Binary {
        return None;
    }
    let (value, value_aff, col_aff) = if is_column(left, col_reg) && !has_column_ref(right) {
        (right.as_ref(), meta.apply_right, meta.apply_left)
    } else if is_column(right, col_reg) && !has_column_ref(left) {
        (left.as_ref(), meta.apply_left, meta.apply_right)
    } else {
        return None;
    };
    // A comparison that forces an affinity onto the COLUMN side (datatype3 §4.2 rule 1
    // NUMERIC-onto-non-numeric, or rule 2 TEXT-onto-none) coerces it out of the storage
    // class the Binary-ordered index stored it under, so a raw seek would miss rows.
    // Decline; the residual filter applies the affinity correctly.
    if col_aff.is_some() {
        return None;
    }
    match value {
        EvalExpr::Literal(v) => {
            let vv = apply_opt_affinity(v, value_aff);
            if vv.is_null() {
                if *null_safe {
                    // `col IS NULL`: the index's NULL entries are exactly the matches.
                    Some((EvalExpr::Literal(Value::Null), true))
                } else {
                    // `col = NULL` is always empty — let the residual filter yield it.
                    None
                }
            } else {
                Some((EvalExpr::Literal(vv), true))
            }
        }
        other => {
            // A runtime value: usable raw only if no affinity coercion is needed.
            if value_aff.is_some() {
                return None;
            }
            Some((other.clone(), false))
        }
    }
}

/// If `e` is a usable range comparison on the index column at register `col_reg`,
/// return `(side, bound, consume)`. The operator is normalized so the value is
/// compared against the column on the left. Same usability rules as [`match_index_eq`].
fn match_index_range(e: &EvalExpr, col_reg: usize) -> Option<(Side, RangeBound, bool)> {
    let EvalExpr::Compare { op, null_safe: false, left, right, meta } = e else {
        return None;
    };
    if meta.collation != Collation::Binary {
        return None;
    }
    let (value, value_aff, col_aff, effective) = if is_column(left, col_reg) && !has_column_ref(right) {
        (right.as_ref(), meta.apply_right, meta.apply_left, *op)
    } else if is_column(right, col_reg) && !has_column_ref(left) {
        (left.as_ref(), meta.apply_left, meta.apply_right, flip(*op))
    } else {
        return None;
    };
    // See `match_index_eq`: a forced column-side affinity means the Binary-ordered index
    // keys are the wrong storage class to seek, so decline the index for this term.
    if col_aff.is_some() {
        return None;
    }
    let (side, inclusive) = match effective {
        CmpOp::Gt => (Side::Lo, false),
        CmpOp::Ge => (Side::Lo, true),
        CmpOp::Lt => (Side::Hi, false),
        CmpOp::Le => (Side::Hi, true),
        CmpOp::Eq | CmpOp::Ne => return None,
    };
    match value {
        EvalExpr::Literal(v) => {
            let vv = apply_opt_affinity(v, value_aff);
            if vv.is_null() {
                // A NULL bound matches nothing (`col > NULL` is NULL) — leave it as a
                // residual filter rather than seeking with it.
                return None;
            }
            Some((side, RangeBound { value: EvalExpr::Literal(vv), inclusive }, true))
        }
        other => {
            if value_aff.is_some() {
                return None;
            }
            Some((side, RangeBound { value: other.clone(), inclusive }, false))
        }
    }
}

/// Consume at most one lower and one upper rowid bound into a range access path.
/// Each bound is a plan-time-coerced concrete `Integer` literal (see [`rowid_bound`]);
/// non-integer, NULL, and runtime bounds are declined there and fall to the residual,
/// so the executor's lossy `to_integer` never sees a value it would mangle.
fn compute_rowid_range(
    conjuncts: &[EvalExpr],
    rowid_reg: usize,
    table: &str,
    n: usize,
    db: DbIndex,
) -> Option<RowidRangeCandidate> {
    let mut lo: Option<RangeBound> = None;
    let mut hi: Option<RangeBound> = None;
    let mut remove: Vec<usize> = Vec::new();
    for (ci, c) in conjuncts.iter().enumerate() {
        if let Some((side, rb)) = rowid_bound(c, rowid_reg) {
            match side {
                Side::Lo if lo.is_none() => {
                    lo = Some(rb);
                    remove.push(ci);
                }
                Side::Hi if hi.is_none() => {
                    hi = Some(rb);
                    remove.push(ci);
                }
                _ => {}
            }
        }
    }
    if lo.is_none() && hi.is_none() {
        return None;
    }
    let node = PlanNode::RowidScan(RowidScan {
        table: table.to_string(),
        db,
        column_count: n,
        op: RowidOp::Range { lo, hi },
        direction: ScanDirection::Forward,
    });
    Some(RowidRangeCandidate { node, remove })
}

/// Rebuild the residual from `conjuncts`, dropping the indices in `remove`.
fn residual_without(conjuncts: Vec<EvalExpr>, remove: &[usize]) -> Option<EvalExpr> {
    let keep: Vec<EvalExpr> = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| if remove.contains(&i) { None } else { Some(c) })
        .collect();
    rebuild_and(keep)
}

/// Apply an optional operand affinity to a literal value (`None` = leave untouched).
/// This is the plan-time counterpart of the per-row `apply_left`/`apply_right` the
/// comparison evaluator applies, so a literal seek key matches what the index stores.
fn apply_opt_affinity(v: &Value, a: Option<Affinity>) -> Value {
    match a {
        Some(af) => apply_affinity(v.clone(), af),
        None => v.clone(),
    }
}

// ---------------------------------------------------------------------------
// Shared conjunct helpers.
// ---------------------------------------------------------------------------

/// Which end of a range a bound constrains.
enum Side {
    Lo,
    Hi,
}

/// Flatten a top-level conjunction (`a AND b AND c`) into its conjuncts.
fn flatten_and(e: EvalExpr, out: &mut Vec<EvalExpr>) {
    match e {
        EvalExpr::And(a, b) => {
            flatten_and(*a, out);
            flatten_and(*b, out);
        }
        other => out.push(other),
    }
}

/// Rewrite each `x BETWEEN a AND b` over a bare column into `x >= a AND x <= b`, so
/// the rowid and index range logic sees uniform comparisons. This is exactly SQLite's
/// definition of BETWEEN (`lang_expr.html`) except that BETWEEN evaluates `x` once —
/// which is immaterial here because `x` is a bare column (no cost, no side effect). A
/// negated BETWEEN, or one whose subject is any non-column expression, is left whole.
/// Each branch keeps its own comparison metadata (BETWEEN may apply different affinity
/// to the two sides).
fn expand_betweens(conjuncts: Vec<EvalExpr>) -> Vec<EvalExpr> {
    let mut out = Vec::with_capacity(conjuncts.len());
    for c in conjuncts {
        match c {
            EvalExpr::Between { negated: false, subject, low, high, low_meta, high_meta }
                if matches!(subject.as_ref(), EvalExpr::Column(_)) =>
            {
                out.push(EvalExpr::Compare {
                    op: CmpOp::Ge,
                    null_safe: false,
                    left: subject.clone(),
                    right: low,
                    meta: low_meta,
                });
                out.push(EvalExpr::Compare {
                    op: CmpOp::Le,
                    null_safe: false,
                    left: subject,
                    right: high,
                    meta: high_meta,
                });
            }
            other => out.push(other),
        }
    }
    out
}

/// Re-combine conjuncts into a single `AND` chain (`None` when empty).
fn rebuild_and(conjuncts: Vec<EvalExpr>) -> Option<EvalExpr> {
    let mut it = conjuncts.into_iter();
    let mut acc = it.next()?;
    for c in it {
        acc = EvalExpr::And(Box::new(acc), Box::new(c));
    }
    Some(acc)
}

/// True if `e` is exactly `Column(reg)`.
fn is_column(e: &EvalExpr, reg: usize) -> bool {
    matches!(e, EvalExpr::Column(r) if *r == reg)
}

/// If `e` is a usable `rowid = value` (either operand order), the seek value for
/// [`RowidOp::Eq`]. The rowid executor coerces the seek value with sqlite's
/// `OP_SeekRowid` rule (`eval_rowid_eq` → `must_be_int`: apply NUMERIC affinity, then
/// require a lossless integer, else no rows), so:
///
/// * a **literal** is still coerced to its operand affinity HERE — folding it once at
///   plan time keeps the emitted key a concrete `Integer` (`rowid = '5'` seeks
///   `Integer(5)`), and a NULL result declines the seek, since `rowid = NULL` is empty;
///   and
/// * a **runtime** value (a `?` param, or any row-independent non-literal expression) is
///   emitted RAW regardless of the affinity the comparison would apply — the executor's
///   OP_MustBeInt handles it by value: a text/real binding seeks the integer it names
///   (`rowid = '5'` → rowid 5), and a non-integer or NULL binding seeks nothing, the
///   correct answer.
///
/// `rowid IS NULL` (`null_safe`) is never a seek — a rowid is never NULL — so it is
/// left whole for the residual. (A subquery is never a seek value: it is row-dependent
/// per [`has_column_ref`] and declines at the row-independence gate above.)
fn rowid_eq_value(e: &EvalExpr, rowid_reg: usize) -> Option<EvalExpr> {
    let EvalExpr::Compare { op: CmpOp::Eq, null_safe: false, left, right, meta } = e else {
        return None;
    };
    let (value, value_aff) = if is_column(left, rowid_reg) && !has_column_ref(right) {
        (right.as_ref(), meta.apply_right)
    } else if is_column(right, rowid_reg) && !has_column_ref(left) {
        (left.as_ref(), meta.apply_left)
    } else {
        return None;
    };
    match value {
        EvalExpr::Literal(v) => {
            let vv = apply_opt_affinity(v, value_aff);
            // `rowid = NULL` is always empty — leave it to the residual filter.
            if vv.is_null() {
                None
            } else {
                Some(EvalExpr::Literal(vv))
            }
        }
        other => {
            // The executor coerces the seek value via OP_MustBeInt (NUMERIC affinity +
            // lossless-integer-or-no-rows, in scan.rs `eval_rowid_eq`), so a runtime value
            // seeks regardless of the affinity the comparison would apply.
            Some(other.clone())
        }
    }
}

/// If `e` is a usable rowid range comparison (`rowid < c`, `c >= rowid`, …), the side
/// it bounds and the bound value. The operator is normalized so the value is compared
/// against the rowid on the left.
///
/// The rowid range executor coerces each bound with a lossy `to_integer` (NULL → 0, a
/// fractional real truncated), so — unlike the equality path — only a bound that
/// resolves HERE to a concrete `Integer` is safe to seek: a literal is coerced to its
/// operand affinity and must land on an `Integer`, and any runtime value is declined
/// (kept in the residual, where the Filter compares with full real / NULL semantics).
fn rowid_bound(e: &EvalExpr, rowid_reg: usize) -> Option<(Side, RangeBound)> {
    let EvalExpr::Compare { op, null_safe: false, left, right, meta } = e else {
        return None;
    };
    let (value, value_aff, effective) = if is_column(left, rowid_reg) && !has_column_ref(right) {
        (right.as_ref(), meta.apply_right, *op)
    } else if is_column(right, rowid_reg) && !has_column_ref(left) {
        (left.as_ref(), meta.apply_left, flip(*op))
    } else {
        return None;
    };
    let (side, inclusive) = match effective {
        CmpOp::Gt => (Side::Lo, false),
        CmpOp::Ge => (Side::Lo, true),
        CmpOp::Lt => (Side::Hi, false),
        CmpOp::Le => (Side::Hi, true),
        CmpOp::Eq | CmpOp::Ne => return None,
    };
    // Only a literal that coerces to an integer is a safe raw rowid bound; a non-integer
    // literal (real / text / NULL) or any runtime value would be mangled by the
    // executor's `to_integer`, so it falls to the residual filter instead.
    let EvalExpr::Literal(v) = value else {
        return None;
    };
    match apply_opt_affinity(v, value_aff) {
        iv @ Value::Integer(_) => {
            Some((side, RangeBound { value: EvalExpr::Literal(iv), inclusive }))
        }
        _ => None,
    }
}

/// Flip a comparison so `c OP rowid` becomes `rowid OP' c`.
fn flip(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
    }
}

/// Whether a compiled expression is row-DEPENDENT — evaluating it could read the
/// current row (a `Column`) or, for a subquery, an outer row it is correlated to. A
/// rowid/index seek or range bound is evaluated ONCE before the scan with an EMPTY
/// outer, so a candidate value is usable as a seek key ONLY when this returns `false`.
///
/// A subquery (`ScalarSubquery` / `ScalarSubqueryColumn` / `Exists` / `InSubquery`) is
/// treated as row-dependent unconditionally. This expression-only view holds just a
/// `SubqueryId`, not the
/// `SubPlan`, so it cannot tell a subplan CORRELATED to the scanned table from a
/// constant one — and hoisting a correlated subquery into a seek key evaluates it with
/// the wrong (empty) outer, corrupting it into a single constant that pins the column
/// and drops the rows the residual can never add back. Declining is always safe: the
/// subquery stays in the residual filter, which re-evaluates it per row. (A genuinely
/// uncorrelated subquery is thereby left un-hoisted — a missed optimization, not a
/// correctness loss; hoisting it soundly would need `SubPlan.correlated` here.)
fn has_column_ref(e: &EvalExpr) -> bool {
    match e {
        EvalExpr::Column(_) => true,
        EvalExpr::Literal(_) | EvalExpr::Param(_) | EvalExpr::Now(_) => false,
        EvalExpr::ScalarSubquery(_)
        | EvalExpr::ScalarSubqueryColumn { .. }
        | EvalExpr::Exists { .. }
        | EvalExpr::InSubquery { .. }
        | EvalExpr::InSubqueryRow { .. } => true,
        EvalExpr::Unary { operand, .. } => has_column_ref(operand),
        EvalExpr::Arith { left, right, .. }
        | EvalExpr::Concat { left, right }
        | EvalExpr::Bitwise { left, right, .. }
        | EvalExpr::Compare { left, right, .. } => has_column_ref(left) || has_column_ref(right),
        EvalExpr::And(a, b) | EvalExpr::Or(a, b) => has_column_ref(a) || has_column_ref(b),
        EvalExpr::IsNull(x) | EvalExpr::NotNull(x) => has_column_ref(x),
        EvalExpr::Between { subject, low, high, .. } => {
            has_column_ref(subject) || has_column_ref(low) || has_column_ref(high)
        }
        EvalExpr::InList { subject, items, .. } => {
            has_column_ref(subject) || items.iter().any(has_column_ref)
        }
        EvalExpr::Coalesce(items) => items.iter().any(has_column_ref),
        EvalExpr::NullIf { left, right, .. } => has_column_ref(left) || has_column_ref(right),
        EvalExpr::Case { operand, whens, else_expr } => {
            operand.as_deref().is_some_and(has_column_ref)
                || whens.iter().any(|w| has_column_ref(&w.when) || has_column_ref(&w.then))
                || else_expr.as_deref().is_some_and(has_column_ref)
        }
        EvalExpr::Cast { operand, .. } | EvalExpr::Collate { operand, .. } => has_column_ref(operand),
        EvalExpr::Like { subject, pattern, escape, .. } => {
            has_column_ref(subject)
                || has_column_ref(pattern)
                || escape.as_deref().is_some_and(has_column_ref)
        }
        EvalExpr::Func { args, .. } => args.iter().any(has_column_ref),
        // A RAISE(...) has no column operands, but it is side-effecting (evaluating it
        // aborts the statement), so it must never be hoisted out of the per-row residual
        // into a seek key. Report it row-dependent, exactly as subqueries are declined
        // above. It only appears inside a trigger body, so this arm is defensive.
        EvalExpr::Raise { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use minisqlite_catalog::{ColumnDef, KeyColumn};
    use minisqlite_pager::Pager;
    use minisqlite_sql::{parse, CreateIndex, CreateTable, Drop};

    use crate::plan::Plan;
    use crate::{Planner, QueryPlanner};

    // -----------------------------------------------------------------------
    // A static catalog that answers table + index lookups from hand-built defs.
    // It exercises the exact `plan_table_access` path the real `SchemaCatalog`
    // drives (which only reads `table` / `indexes_on`), without a pager/b-tree.
    // -----------------------------------------------------------------------
    struct IdxCatalog {
        tables: Vec<TableDef>,
        indexes: Vec<IndexDef>,
    }

    impl Catalog for IdxCatalog {
        fn table(&self, name: &str) -> Result<Option<&TableDef>> {
            Ok(self.tables.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
        }
        fn index(&self, name: &str) -> Result<Option<&IndexDef>> {
            Ok(self.indexes.iter().find(|i| i.name.eq_ignore_ascii_case(name)))
        }
        fn indexes_on<'a>(&'a self, table: &str) -> Result<Vec<&'a IndexDef>> {
            Ok(self.indexes.iter().filter(|i| i.table.eq_ignore_ascii_case(table)).collect())
        }
        fn load(&mut self, _pager: &dyn Pager) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn create_table(&mut self, _p: &mut dyn Pager, _s: &CreateTable, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn create_index(&mut self, _p: &mut dyn Pager, _s: &CreateIndex, _sql: &str) -> Result<()> {
            unimplemented!("static test catalog")
        }
        fn drop_object(&mut self, _p: &mut dyn Pager, _s: &Drop) -> Result<()> {
            unimplemented!("static test catalog")
        }
    }

    fn col(name: &str, decl: Option<&str>, collation: Option<&str>) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            declared_type: decl.map(str::to_string),
            not_null: false,
            primary_key: false,
            unique: false,
            collation: collation.map(str::to_string),
            default: None,
            default_value: None,
            generated: None,
        }
    }

    fn tdef(name: &str, columns: Vec<ColumnDef>, rowid_alias: Option<usize>) -> TableDef {
        TableDef {
            name: name.to_string(),
            columns,
            root_page: 2,
            without_rowid: false,
            rowid_alias,
            auto_indexes: Vec::new(),
            checks: Vec::new(),
            foreign_keys: Vec::new(),
            autoincrement: false,
            primary_key: Vec::new(),
            strict: false,
        }
    }

    fn idef(name: &str, table: &str, columns: &[&str], unique: bool, partial: bool) -> IndexDef {
        IndexDef {
            name: name.to_string(),
            table: table.to_string(),
            columns: columns.iter().map(|c| c.to_string()).collect(),
            key_columns: columns
                .iter()
                .map(|_| KeyColumn { collation: None, descending: false })
                .collect(),
            // Ordinary named-column index (no expression key): every key-expr slot is
            // `None`, parallel to `columns` / `key_columns`.
            key_exprs: vec![None; columns.len()],
            root_page: 3,
            unique,
            partial,
            // Preserve `partial == partial_predicate.is_some()`: an always-true placeholder for
            // a partial fixture (the access-path tests only check that a partial index is
            // DECLINED via `.partial`, never evaluate the predicate).
            partial_predicate: partial
                .then(|| minisqlite_sql::Expr::Literal(minisqlite_sql::Literal::Integer(1))),
        }
    }

    fn plan_it(sql: &str, cat: &dyn Catalog) -> Plan {
        let ast = parse(sql).expect("parse ok");
        let stmt = ast.statements.first().expect("one statement");
        QueryPlanner::new().plan(stmt, cat).expect("plan ok")
    }

    /// Descend the row-preserving wrappers (`Project`/`Filter`/`Sort`/…) to the
    /// access-path leaf.
    fn access_leaf(node: &PlanNode) -> &PlanNode {
        match node {
            PlanNode::Project { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Distinct { input, .. }
            | PlanNode::Limit { input, .. } => access_leaf(input),
            leaf => leaf,
        }
    }

    /// The first `Filter` predicate found descending from the root, or `None` if the
    /// path to the leaf carries no `Filter` (i.e. the WHERE was fully consumed).
    fn filter_predicate(node: &PlanNode) -> Option<&EvalExpr> {
        match node {
            PlanNode::Filter { predicate, .. } => Some(predicate),
            PlanNode::Project { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Distinct { input, .. }
            | PlanNode::Limit { input, .. } => filter_predicate(input),
            _ => None,
        }
    }

    fn index_seek(node: &PlanNode) -> (&str, &[EvalExpr], &Option<RangeBound>, &Option<RangeBound>) {
        match access_leaf(node) {
            PlanNode::IndexScan(IndexScan { index, op: IndexOp::Seek { eq_prefix, low, high }, .. }) => {
                (index.as_str(), eq_prefix.as_slice(), low, high)
            }
            other => panic!("expected IndexScan Seek, got {other:?}"),
        }
    }

    /// The seek value of a `RowidScan { op: Eq(..) }`, or panic.
    fn rowid_eq_target(node: &PlanNode) -> &EvalExpr {
        match access_leaf(node) {
            PlanNode::RowidScan(RowidScan { op: RowidOp::Eq(v), .. }) => v,
            other => panic!("expected RowidScan Eq, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE INDEX flips the plan.
    // -----------------------------------------------------------------------

    #[test]
    fn without_index_equality_is_a_seqscan_but_with_index_it_is_an_indexscan() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None), col("b", Some("TEXT"), None)], None)];

        // No index on `a`: a full scan with the equality left as a residual Filter.
        let no_idx = IdxCatalog { tables: tables.clone(), indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5", &no_idx);
        assert!(matches!(access_leaf(&plan.root), PlanNode::SeqScan(_)), "no index → SeqScan");
        assert!(filter_predicate(&plan.root).is_some(), "the equality is the residual filter");

        // Add an index on `a`: the same query becomes an index seek with no residual.
        let with_idx = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5", &with_idx);
        let (name, eq, low, high) = index_seek(&plan.root);
        assert_eq!(name, "ia");
        assert_eq!(eq.len(), 1, "single equality prefix");
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Integer(5))), "seek key is the literal 5");
        assert!(low.is_none() && high.is_none(), "a pure equality has no range bound");
        assert!(filter_predicate(&plan.root).is_none(), "the equality is fully consumed");
    }

    // -----------------------------------------------------------------------
    // Rowid access paths still win (kept from the original logic).
    // -----------------------------------------------------------------------

    #[test]
    fn rowid_eq_beats_an_index_and_leaves_the_index_term_as_residual() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE rowid = 7 AND a = 9", &cat);
        match access_leaf(&plan.root) {
            PlanNode::RowidScan(RowidScan { op: RowidOp::Eq(_), .. }) => {}
            other => panic!("expected RowidScan Eq (strongest), got {other:?}"),
        }
        // `a = 9` was not consumed by the rowid seek, so it remains as a filter.
        assert!(filter_predicate(&plan.root).is_some(), "the a=9 term is the residual");
    }

    #[test]
    fn rowid_between_is_a_rowid_range_scan() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid BETWEEN 1 AND 9", &cat);
        match access_leaf(&plan.root) {
            PlanNode::RowidScan(RowidScan { op: RowidOp::Range { lo, hi }, .. }) => {
                let lo = lo.as_ref().expect("lower bound from BETWEEN");
                let hi = hi.as_ref().expect("upper bound from BETWEEN");
                assert!(lo.inclusive && hi.inclusive, "BETWEEN bounds are inclusive");
            }
            other => panic!("expected RowidScan Range, got {other:?}"),
        }
        assert!(filter_predicate(&plan.root).is_none(), "both BETWEEN halves are consumed");
    }

    // -----------------------------------------------------------------------
    // Multi-column index: equality prefix and the trailing range "sandwich".
    // -----------------------------------------------------------------------

    #[test]
    fn two_equalities_fill_a_two_column_equality_prefix() {
        let tables =
            vec![tdef("t", vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("iab", "t", &["a", "b"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 AND b = 2", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert_eq!(eq.len(), 2, "both equalities pin the two-column prefix");
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Integer(1))));
        assert!(matches!(eq[1], EvalExpr::Literal(Value::Integer(2))));
        assert!(low.is_none() && high.is_none());
        assert!(filter_predicate(&plan.root).is_none(), "fully consumed");
    }

    #[test]
    fn equality_prefix_then_a_range_on_the_next_column() {
        let tables =
            vec![tdef("t", vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("iab", "t", &["a", "b"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 AND b > 2", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert_eq!(eq.len(), 1, "only `a` is an equality prefix");
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Integer(1))));
        let low = low.as_ref().expect("a low bound on b");
        assert!(!low.inclusive, "`>` is an exclusive lower bound");
        assert!(matches!(low.value, EvalExpr::Literal(Value::Integer(2))));
        assert!(high.is_none(), "no upper bound");
        assert!(filter_predicate(&plan.root).is_none(), "eq + range consume both terms");
    }

    #[test]
    fn between_on_a_single_indexed_column_is_a_two_sided_range_seek() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a BETWEEN 3 AND 9", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert!(eq.is_empty(), "no equality — a pure leading-column range");
        assert!(low.as_ref().is_some_and(|b| b.inclusive), "inclusive low from BETWEEN");
        assert!(high.as_ref().is_some_and(|b| b.inclusive), "inclusive high from BETWEEN");
        assert!(filter_predicate(&plan.root).is_none(), "both BETWEEN halves consumed");
    }

    // -----------------------------------------------------------------------
    // Upper-bound-only NULL exclusion: an `a <= k` index range (upper bound, NO lower
    // bound) starts its scan on the index's leading NULL-keyed entries, so the
    // comparison must stay in the residual to drop them (`NULL <= k` is NULL, not
    // true). A lower bound seeks PAST the NULLs, so it is fully consumed — the
    // asymmetry these tests pin.
    // -----------------------------------------------------------------------

    #[test]
    fn upper_bound_only_index_range_keeps_the_comparison_as_a_residual_filter() {
        // `a <= 5`: the scan begins at the index's first entry, and NULL keys sort
        // BEFORE all values, so a raw seek would emit the NULL-keyed rows even though
        // `NULL <= 5` is NULL (not true). The planner keeps the comparison in the
        // residual (a superset seek); the upper bound still bounds how far the scan
        // runs, and the Filter drops the NULL rows.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a <= 5", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert!(eq.is_empty(), "no equality prefix — a pure leading-column range");
        assert!(low.is_none(), "no lower bound");
        assert!(high.as_ref().is_some_and(|b| b.inclusive), "an inclusive upper bound from <=");
        let pred = filter_predicate(&plan.root).expect("a<=5 stays as the residual");
        assert!(
            matches!(pred, EvalExpr::Compare { op: CmpOp::Le, .. }),
            "residual is the leftover a<=5 comparison, got {pred:?}"
        );
    }

    #[test]
    fn strict_upper_bound_only_index_range_also_keeps_the_residual() {
        // The exclusive twin (`a < 5`): same NULL-exclusion reasoning, so the term is
        // likewise kept in the residual rather than consumed.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a < 5", &cat);
        let (_name, _eq, low, high) = index_seek(&plan.root);
        assert!(low.is_none(), "no lower bound");
        assert!(high.as_ref().is_some_and(|b| !b.inclusive), "an exclusive upper bound from <");
        let pred = filter_predicate(&plan.root).expect("a<5 stays as the residual");
        assert!(
            matches!(pred, EvalExpr::Compare { op: CmpOp::Lt, .. }),
            "residual is the leftover a<5 comparison, got {pred:?}"
        );
    }

    #[test]
    fn lower_bound_only_index_range_is_fully_consumed() {
        // The asymmetry vs the upper-bound case: `a >= 5` seeks to the first entry with
        // `a >= 5`, which is PAST every NULL key (NULLs sort first), so no NULL-keyed
        // row is ever visited — the bound is exact and the term is fully consumed.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a >= 5", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert!(eq.is_empty());
        assert!(low.as_ref().is_some_and(|b| b.inclusive), "an inclusive lower bound from >=");
        assert!(high.is_none(), "no upper bound");
        assert!(filter_predicate(&plan.root).is_none(), "a lower-bound-only range is fully consumed");
    }

    #[test]
    fn a_lower_bound_lets_the_upper_bound_be_consumed() {
        // With BOTH bounds the seek starts past the NULLs (via the lower bound), so the
        // upper bound is exact and fully consumed even though it alone would be a
        // superset seek. `a >= 2 AND a <= 5` leaves no residual.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a >= 2 AND a <= 5", &cat);
        let (_name, _eq, low, high) = index_seek(&plan.root);
        assert!(low.is_some() && high.is_some(), "both bounds present");
        assert!(
            filter_predicate(&plan.root).is_none(),
            "both bounds consumed when a lower bound is present"
        );
    }

    #[test]
    fn an_upper_bound_before_its_lower_bound_is_still_consumed() {
        // The lower bound may appear AFTER the upper in conjunct order; the consume
        // decision is deferred until both are seen, so `a <= 5 AND a >= 2` fully
        // consumes the upper bound (no residual), the same as the reversed order.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a <= 5 AND a >= 2", &cat);
        let (_name, _eq, low, high) = index_seek(&plan.root);
        assert!(low.is_some() && high.is_some(), "both bounds present regardless of order");
        assert!(
            filter_predicate(&plan.root).is_none(),
            "upper consumed even when it precedes the lower in conjunct order"
        );
    }

    #[test]
    fn trailing_upper_bound_only_after_an_equality_prefix_keeps_the_residual() {
        // The composite-index instance of the same NULL leak: `a = 1 AND b <= 15`
        // seeks the a=1 group, then range-scans b upward. b=NULL sorts first WITHIN
        // the a=1 group, so the raw scan would emit it; `NULL <= 15` is NULL, so
        // `b <= 15` must stay in the residual to drop it. The equality `a = 1` is
        // still consumed.
        let tables = vec![tdef(
            "t",
            vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)],
            None,
        )];
        let cat = IdxCatalog { tables, indexes: vec![idef("iab", "t", &["a", "b"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 AND b <= 15", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert_eq!(eq.len(), 1, "a=1 is the equality prefix");
        assert!(low.is_none(), "no lower bound on b");
        assert!(high.is_some(), "an upper bound on b");
        let pred = filter_predicate(&plan.root).expect("b<=15 stays as the residual");
        assert!(
            matches!(pred, EvalExpr::Compare { op: CmpOp::Le, .. }),
            "residual is the leftover b<=15, got {pred:?}"
        );
    }

    #[test]
    fn a_gap_in_the_equality_prefix_stops_it_contiguously() {
        // Index (a,b,c) with `a = 1 AND c = 3` (nothing constrains b): only `a` is a
        // contiguous equality prefix. `c` must NOT jump the `b` gap into the seek key —
        // it stays in the residual. A skip-the-gap bug would make eq_prefix `[1, 3]`,
        // seeking the wrong key.
        let tables = vec![tdef(
            "t",
            vec![
                col("a", Some("INTEGER"), None),
                col("b", Some("INTEGER"), None),
                col("c", Some("INTEGER"), None),
            ],
            None,
        )];
        let cat =
            IdxCatalog { tables, indexes: vec![idef("iabc", "t", &["a", "b", "c"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 AND c = 3", &cat);
        let (_name, eq, low, high) = index_seek(&plan.root);
        assert_eq!(eq.len(), 1, "the prefix stops at the b gap — only `a` is pinned");
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Integer(1))), "the seek key is a=1, not c");
        assert!(low.is_none() && high.is_none(), "c is an equality, not a range on b");
        let pred = filter_predicate(&plan.root).expect("c=3 is the residual");
        assert!(
            matches!(pred, EvalExpr::Compare { op: CmpOp::Eq, .. }),
            "residual is the leftover c=3, got {pred:?}"
        );
    }

    #[test]
    fn a_column_on_both_sides_is_never_a_seek_key() {
        // `a = b` over two UNTYPED columns (so no affinity check can mask the guard):
        // the RHS is row-dependent, so it can never be a seek key. Dropping the
        // `!has_column_ref` guard would wrongly seek with `Column(b)`.
        let tables = vec![tdef("t", vec![col("a", None, None), col("b", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = b", &cat);
        assert!(
            matches!(access_leaf(&plan.root), PlanNode::SeqScan(_)),
            "a column-dependent RHS cannot drive an index seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "a=b stays as the residual filter");
    }

    #[test]
    fn a_subquery_rhs_is_never_a_seek_key() {
        // A scalar subquery on the compared side of an UNTYPED (Blob-affinity) indexed
        // column must NOT be hoisted into a seek key. `has_column_ref` holds only a
        // `SubqueryId`, so it cannot tell whether the subplan is CORRELATED to the
        // scanned row; a correlated subquery hoisted into a once-evaluated seek key runs
        // with an empty outer and collapses to one corrupted constant that pins the
        // column and drops rows the residual can never add back. So EVERY subquery is
        // treated as row-dependent and left in the residual. Untyped is the trigger — a
        // typed column would force affinity onto the Blob subquery and decline via the
        // affinity gate, masking this row-independence hole. This case is NON-correlated
        // (the pre-fix code DID hoist it into an IndexScan); the correlated case hits the
        // same `=> true` arm — see `a_correlated_subquery_rhs_is_never_a_seek_key`.
        let tables = vec![
            tdef("t", vec![col("x", None, None)], None),
            tdef("t2", vec![col("a", None, None)], None),
        ];
        let cat = IdxCatalog { tables, indexes: vec![idef("ix", "t", &["x"], false, false)] };
        let plan = plan_it("SELECT x FROM t WHERE x = (SELECT a FROM t2)", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::IndexScan(_)),
            "a subquery seek key must not be hoisted into an index seek"
        );
        assert!(
            filter_predicate(&plan.root).is_some(),
            "the subquery equality stays as the residual filter"
        );
    }

    #[test]
    fn a_correlated_subquery_rhs_is_never_a_seek_key() {
        // The actual bug scenario: a subquery CORRELATED to the scanned row (it reads
        // `t.x` inside the subplan) on the compared side of an UNTYPED indexed column.
        // Hoisting it into a seek key evaluates it ONCE with an empty outer, collapsing a
        // per-row-varying predicate into one corrupted constant and dropping rows. It must
        // decline to the residual, which re-evaluates the correlated subplan per outer row.
        // (Correlated subqueries now bind in canonical, so this plans end-to-end.)
        let tables = vec![
            tdef("t", vec![col("x", None, None)], None),
            tdef("t2", vec![col("a", None, None)], None),
        ];
        let cat = IdxCatalog { tables, indexes: vec![idef("ix", "t", &["x"], false, false)] };
        let plan =
            plan_it("SELECT x FROM t WHERE x = (SELECT count(*) FROM t2 WHERE t2.a <= t.x)", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::IndexScan(_)),
            "a correlated subquery must not be hoisted into an index seek"
        );
        assert!(
            filter_predicate(&plan.root).is_some(),
            "the correlated subquery equality stays as the residual filter"
        );
    }

    // -----------------------------------------------------------------------
    // Cost model: equality index beats a rowid range; a bare index range loses.
    // -----------------------------------------------------------------------

    #[test]
    fn index_equality_beats_a_competing_rowid_range() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5 AND rowid > 100", &cat);
        let (_name, eq, _low, _high) = index_seek(&plan.root);
        assert_eq!(eq.len(), 1, "the equality index is chosen over the rowid range");
        // The rowid range is not consumed by the index seek → it is the residual.
        assert!(filter_predicate(&plan.root).is_some(), "rowid>100 stays as a filter");
    }

    #[test]
    fn a_bare_index_range_loses_to_a_rowid_range() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a > 5 AND rowid > 100", &cat);
        match access_leaf(&plan.root) {
            PlanNode::RowidScan(RowidScan { op: RowidOp::Range { .. }, .. }) => {}
            other => panic!("a rowid range should beat a bare index range, got {other:?}"),
        }
        assert!(filter_predicate(&plan.root).is_some(), "a>5 stays as the residual");
    }

    #[test]
    fn a_unique_full_equality_beats_a_rowid_range() {
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ua", "t", &["a"], true, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5 AND rowid > 100", &cat);
        let (name, eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(name, "ua");
        assert_eq!(eq.len(), 1, "the unique full-equality seek (one row) wins");
    }

    // -----------------------------------------------------------------------
    // Cost model across MULTIPLE candidate indexes on one table: the tightest wins.
    // (These are the tests that actually exercise `choose_best_index`'s comparison.)
    // -----------------------------------------------------------------------

    #[test]
    fn among_two_indexes_the_longer_equality_prefix_wins() {
        // `ia` on (a) offers an eq_prefix of length 1; `iab` on (a,b) offers length 2
        // for the same query. The cost model must pick the tighter `iab` and fully
        // consume both terms — picking `ia` would leave `b = 2` as a residual filter.
        let tables = vec![tdef(
            "t",
            vec![col("a", Some("INTEGER"), None), col("b", Some("INTEGER"), None)],
            None,
        )];
        let cat = IdxCatalog {
            tables,
            indexes: vec![
                idef("ia", "t", &["a"], false, false),
                idef("iab", "t", &["a", "b"], false, false),
            ],
        };
        let plan = plan_it("SELECT * FROM t WHERE a = 1 AND b = 2", &cat);
        let (name, eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(name, "iab", "the two-column index is tighter than the single-column one");
        assert_eq!(eq.len(), 2, "both equalities pin the two-column prefix");
        assert!(filter_predicate(&plan.root).is_none(), "iab consumes both terms");
    }

    #[test]
    fn a_unique_index_beats_an_equally_long_non_unique_one() {
        // Both indexes offer a full single-column equality, but the UNIQUE one is a
        // one-row seek (tier 5) and must outrank the non-unique (tier 4) — regardless
        // of catalog order (unique listed second here).
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog {
            tables,
            indexes: vec![
                idef("ia", "t", &["a"], false, false),
                idef("ua", "t", &["a"], true, false),
            ],
        };
        let plan = plan_it("SELECT * FROM t WHERE a = 5", &cat);
        let (name, _eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(name, "ua", "the unique single-row seek outranks the non-unique one");
    }

    #[test]
    fn equal_score_indexes_keep_the_first_in_catalog_order() {
        // Two interchangeable non-unique indexes on (a): a score tie must resolve to
        // the FIRST in creation order (the strict `>` keeps the incumbent).
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog {
            tables,
            indexes: vec![
                idef("ia_first", "t", &["a"], false, false),
                idef("ia_second", "t", &["a"], false, false),
            ],
        };
        let plan = plan_it("SELECT * FROM t WHERE a = 5", &cat);
        let (name, _eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(name, "ia_first", "an equal-score tie keeps the creation-order first index");
    }

    // -----------------------------------------------------------------------
    // Hard constraints: NOCASE collation and partial indexes are not usable.
    // -----------------------------------------------------------------------

    #[test]
    fn a_nocase_column_comparison_does_not_use_the_index() {
        // The index b-tree is Binary-ordered; a NOCASE comparison cannot seek it.
        let tables =
            vec![tdef("t", vec![col("a", Some("INTEGER"), None), col("c", Some("TEXT"), Some("NOCASE"))], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ic", "t", &["c"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE c = 'x'", &cat);
        assert!(matches!(access_leaf(&plan.root), PlanNode::SeqScan(_)), "NOCASE term → SeqScan");
        assert!(filter_predicate(&plan.root).is_some(), "the NOCASE term is the residual");
    }

    #[test]
    fn a_partial_index_is_never_used() {
        // A partial index's predicate is unmodelled, so it cannot stand in for a scan.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, true)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5", &cat);
        assert!(matches!(access_leaf(&plan.root), PlanNode::SeqScan(_)), "partial index → SeqScan");
    }

    // -----------------------------------------------------------------------
    // Residual: an unindexed conjunct becomes the Filter above the seek.
    // -----------------------------------------------------------------------

    #[test]
    fn an_unindexed_conjunct_survives_as_the_residual_filter() {
        let tables =
            vec![tdef("t", vec![col("a", Some("INTEGER"), None), col("b", Some("TEXT"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = 5 AND b = 'z'", &cat);
        let (_name, eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(eq.len(), 1, "only `a` is served by the index");
        let pred = filter_predicate(&plan.root).expect("b='z' is the residual");
        // The residual is exactly the leftover `b = 'z'` equality.
        assert!(
            matches!(pred, EvalExpr::Compare { op: CmpOp::Eq, .. }),
            "residual is the b='z' comparison, got {pred:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Affinity at the seek boundary: literals are coerced (usable); a runtime
    // value is used raw only when no coercion is needed.
    // -----------------------------------------------------------------------

    #[test]
    fn a_text_literal_against_an_integer_column_is_coerced_into_the_seek_key() {
        // `a = '5'` on an INTEGER column: NUMERIC affinity is applied at plan time so
        // the seek key is Integer(5), matching what the index stores.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = '5'", &cat);
        let (_name, eq, _l, _h) = index_seek(&plan.root);
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Integer(5))), "text '5' coerced to Integer(5)");
        assert!(filter_predicate(&plan.root).is_none(), "the coerced literal is fully consumed");
    }

    #[test]
    fn a_parameter_on_an_untyped_column_seeks_but_stays_in_the_residual() {
        // An untyped column has NO affinity, so a bind parameter needs no coercion and
        // can drive the seek raw — but it might be NULL at runtime, so the term is kept
        // in the residual (`col = NULL` must yield nothing, not the index's NULL rows).
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = ?", &cat);
        let (_name, eq, _l, _h) = index_seek(&plan.root);
        assert!(matches!(eq[0], EvalExpr::Param(1)), "the parameter drives the seek");
        assert!(filter_predicate(&plan.root).is_some(), "kept as a residual filter for NULL-safety");
    }

    #[test]
    fn a_parameter_needing_affinity_falls_back_to_a_seqscan() {
        // `a = ?` on an INTEGER column needs NUMERIC affinity on the parameter, which
        // cannot be applied to a runtime value at plan time (the raw seek could miss
        // rows), so the safe choice is a full scan.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = ?", &cat);
        assert!(matches!(access_leaf(&plan.root), PlanNode::SeqScan(_)), "typed-column param → SeqScan");
        assert!(filter_predicate(&plan.root).is_some(), "the a=? term is the residual");
    }

    #[test]
    fn a_cast_forcing_numeric_onto_a_text_column_declines_the_index() {
        // `x = CAST(5 AS INTEGER)` on a TEXT column forces NUMERIC affinity onto `x`
        // (datatype3 §4.2 rule 1). The index stored `x` as TEXT under Binary order, so a
        // raw Integer(5) seek would miss the Text("5") entry — the index must be declined
        // and the term left to the residual filter (which applies the affinity).
        // Otherwise CREATE INDEX would silently change the query's result.
        let tables = vec![tdef("t", vec![col("x", Some("TEXT"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ix", "t", &["x"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE x = CAST(5 AS INTEGER)", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::IndexScan(_)),
            "a forced column-side affinity must not become a raw index seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "the comparison stays as the residual");
    }

    #[test]
    fn a_text_literal_on_a_text_column_still_seeks_the_index() {
        // Contrast to the CAST case: `x = 'foo'` applies TEXT affinity to the LITERAL
        // (rule 2) and leaves the column side untouched, so the index stays usable — the
        // column-side guard must not over-reject ordinary text equalities.
        let tables = vec![tdef("t", vec![col("x", Some("TEXT"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ix", "t", &["x"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE x = 'foo'", &cat);
        let (name, eq, _l, _h) = index_seek(&plan.root);
        assert_eq!(name, "ix");
        assert!(
            matches!(&eq[0], EvalExpr::Literal(Value::Text(s)) if s == "foo"),
            "the seek key is the text literal 'foo'"
        );
        assert!(filter_predicate(&plan.root).is_none(), "the text equality is fully consumed");
    }

    #[test]
    fn a_range_cast_forcing_numeric_onto_a_text_column_declines_the_index() {
        // The RANGE analog of the CAST decline: `x > CAST(5 AS INTEGER)` on a TEXT column
        // forces NUMERIC onto `x`, so the Binary-ordered TEXT index cannot serve it — the
        // range must decline to the residual, not seek.
        let tables = vec![tdef("t", vec![col("x", Some("TEXT"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ix", "t", &["x"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE x > CAST(5 AS INTEGER)", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::IndexScan(_)),
            "a forced column-side affinity must not become a raw index range seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "the range comparison stays as the residual");
    }

    // -----------------------------------------------------------------------
    // NULL handling: `IS NULL` (null-safe) seeks the index's NULL entries; a plain
    // `= NULL` (always empty) must NOT seek them.
    // -----------------------------------------------------------------------

    #[test]
    fn is_null_seeks_the_indexs_null_entries() {
        // `a IS NULL` is a null-safe equality; the index's NULL entries are exactly the
        // matches, so it is a NULL-keyed seek with no residual.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a IS NULL", &cat);
        let (_name, eq, _l, _h) = index_seek(&plan.root);
        assert!(matches!(eq[0], EvalExpr::Literal(Value::Null)), "seek key is NULL");
        assert!(filter_predicate(&plan.root).is_none(), "IS NULL is fully consumed");
    }

    #[test]
    fn plain_equals_null_does_not_use_the_index() {
        // `a = NULL` (not null-safe) is always empty; it must not seek the index's NULL
        // entries, so it stays a residual filter over a scan.
        let tables = vec![tdef("t", vec![col("a", Some("INTEGER"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![idef("ia", "t", &["a"], false, false)] };
        let plan = plan_it("SELECT * FROM t WHERE a = NULL", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::IndexScan(_)),
            "a plain = NULL must not become an index seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "= NULL stays as the residual");
    }

    // -----------------------------------------------------------------------
    // INTEGER PRIMARY KEY alias: a comparison on the alias column takes the rowid
    // path, and an index whose column IS the alias maps to the rowid register.
    // -----------------------------------------------------------------------

    #[test]
    fn integer_primary_key_equality_is_a_rowid_seek() {
        // `id INTEGER PRIMARY KEY` aliases the rowid (register N), so `id = 42` is a
        // rowid point seek, not an index scan.
        let tables = vec![tdef(
            "u",
            vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)],
            Some(0),
        )];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM u WHERE id = 42", &cat);
        match access_leaf(&plan.root) {
            PlanNode::RowidScan(RowidScan { op: RowidOp::Eq(_), .. }) => {}
            other => panic!("expected a rowid seek on the intpk alias, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Rowid path affinity / NULL discipline (mirrors the index path): the rowid
    // executor reads seek/bound values raw, so the planner coerces literals and gates
    // NULL / affinity-needing runtime values.
    // -----------------------------------------------------------------------

    #[test]
    fn a_text_literal_rowid_equality_is_coerced_to_an_integer_seek() {
        // `rowid = '5'`: the rowid executor reads the seek value raw and a Text("5")
        // names no row, so NUMERIC affinity is applied HERE to make the key Integer(5).
        // Without this, `WHERE rowid = '5'` would wrongly return zero rows.
        let tables = vec![tdef("t", vec![col("v", Some("TEXT"), None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid = '5'", &cat);
        assert!(
            matches!(rowid_eq_target(&plan.root), EvalExpr::Literal(Value::Integer(5))),
            "text '5' is coerced to an Integer(5) rowid seek key"
        );
    }

    #[test]
    fn a_text_literal_intpk_equality_is_coerced_to_an_integer_seek() {
        // Same coercion through the INTEGER PRIMARY KEY alias: `id = '5'` seeks rowid 5.
        let tables = vec![tdef(
            "u",
            vec![col("id", Some("INTEGER"), None), col("v", Some("TEXT"), None)],
            Some(0),
        )];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM u WHERE id = '5'", &cat);
        assert!(
            matches!(rowid_eq_target(&plan.root), EvalExpr::Literal(Value::Integer(5))),
            "the intpk alias coerces '5' to an Integer(5) rowid seek"
        );
    }

    #[test]
    fn a_null_rowid_range_bound_declines_the_seek() {
        // `rowid > NULL` is always empty. The rowid range executor would coerce a NULL
        // bound to 0 (scanning everything), so the planner must NOT seek it — the term
        // falls to the residual filter, which yields no rows.
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid > NULL", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::RowidScan(_)),
            "a NULL rowid bound must not become a range scan"
        );
        assert!(filter_predicate(&plan.root).is_some(), "rowid > NULL stays as the residual");
    }

    #[test]
    fn a_parameter_rowid_equality_is_a_rowid_seek() {
        // `rowid = ?` is a rowid point seek: the eq executor coerces the value via
        // OP_MustBeInt (`eval_rowid_eq`), so the parameter drives the seek by value (a
        // text/real binding names its integer rowid, a non-integer / NULL binding names
        // no row) — no affinity gate, unlike a secondary index. The seek reaches at most
        // one row, so the term is fully consumed (no residual).
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid = ?", &cat);
        assert!(
            matches!(rowid_eq_target(&plan.root), EvalExpr::Param(1)),
            "the parameter drives the rowid seek"
        );
        assert!(filter_predicate(&plan.root).is_none(), "the seek fully consumes rowid = ?");
    }

    #[test]
    fn a_plain_null_rowid_equality_declines_the_seek() {
        // `rowid = NULL` (not null-safe) is always empty; the coerced key is NULL, which
        // must decline (the residual filter yields the empty result). This pins the
        // eq-side NULL decline, mirroring the range-side NULL decline that
        // `a_null_rowid_range_bound_declines_the_seek` already pins.
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid = NULL", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::RowidScan(_)),
            "rowid = NULL must not become a rowid seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "rowid = NULL stays as the residual");
    }

    #[test]
    fn a_fractional_real_rowid_bound_declines_the_range() {
        // `rowid > 5.5` coerces to Real(5.5), which the range executor's `to_integer`
        // would truncate (mis-rounding an upper bound), so it must decline to the
        // residual rather than seek a mangled integer bound.
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid > 5.5", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::RowidScan(_)),
            "a non-integer rowid bound must not become a range seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "rowid > 5.5 stays as the residual");
    }

    #[test]
    fn a_parameter_rowid_bound_declines_the_range() {
        // `rowid > ?` is a runtime bound the planner cannot resolve to a concrete integer,
        // so it declines. Unlike the eq path (which now SEEKS a runtime `rowid = ?`,
        // because `eval_rowid_eq` coerces via OP_MustBeInt), the range executor still uses
        // a lossy `to_integer` that cannot resolve a runtime bound — so the range stays
        // weaker and declines (see the module doc's eq-vs-range asymmetry).
        let tables = vec![tdef("t", vec![col("a", None, None)], None)];
        let cat = IdxCatalog { tables, indexes: vec![] };
        let plan = plan_it("SELECT * FROM t WHERE rowid > ?", &cat);
        assert!(
            !matches!(access_leaf(&plan.root), PlanNode::RowidScan(_)),
            "a parameter rowid bound must not become a range seek"
        );
        assert!(filter_predicate(&plan.root).is_some(), "rowid > ? stays as the residual");
    }
}
