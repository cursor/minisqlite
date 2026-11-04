//! `index_delete`: remove a key from an index b-tree, keeping the tree a valid,
//! balanced CLASSIC B-tree (the shape real sqlite writes and reads back).
//!
//! This is the index analogue of `delete` (the table b-tree DELETE), but the
//! structural rules differ because an index is a CLASSIC B-tree, not the B+-style
//! rowid table: an interior divider IS a real index entry, present exactly ONCE in
//! the whole tree, sitting in in-order position strictly between its two child
//! subtrees. So a divider can never be silently dropped the way a table b-tree's
//! rowid separator can — every divider is a live key that must be preserved.
//!
//! ## Shape of the algorithm
//! 1. **Locate** the key. It is either a LEAF cell or an interior DIVIDER (the
//!    descent stops at an interior page whose divider equals the key).
//! 2. A LEAF key is removed directly; an interior DIVIDER is removed by the classic
//!    predecessor swap (below).
//! 3. **Remove the cell** from its leaf — an O(cell) in-place splice
//!    (`Pager::page_mut` and `fileformat::delete_cell`) that leaves a freeblock (NOT
//!    defragmented) on the transactional fast route, else a whole-page rebuild that
//!    streams survivors into a fresh defragmented page on the auto-commit fallback
//!    (see `drop_leaf_cell_in_place`). The root leaf is always left as-is; a non-root
//!    leaf that UNDERFLOWS — it empties, or the drop leaves it under ~1/3 full by TOTAL
//!    free space (freeblocks included, so the two routes agree) — is rebalanced, so a
//!    delete-heavy workload keeps the page density real sqlite maintains
//!    (see `leaf_underflows`).
//!
//! ## Deleting an interior divider (predecessor swap, with split on overflow)
//! To delete divider `K` at `(pid, ptr)`: find its in-order predecessor `P` — the
//! maximum key of `K`'s left subtree = the last cell of that subtree's right-most leaf
//! `pred_leaf`. **Install `P` over `K`'s slot** (a real replace-in-place, so `P`'s
//! overflow chain becomes the divider's and `K`'s old chain is freed), then **remove
//! `P` from `pred_leaf`**. `P < K`, so after the swap `P` is still `>` every remaining
//! key of the left subtree and `<` every key of the right subtree — the divider
//! invariant holds.
//!
//! `P` can encode WIDER than `K` (key size and key order are independent — a long TEXT
//! predecessor, a large-magnitude integer), so installing it can OVERFLOW `pid`. That
//! is not an error: `pid` is split and the promoted divider propagated up exactly as
//! an insert split does (`index::propagate_split` / `balance_deeper_interior`, the one
//! shared implementation). Because the leaf side (`pred_leaf`) is handled *after* the
//! interior is made consistent, and its leaf-cell drop needs no path, a split cannot
//! corrupt it. When `pred_leaf` also underflows, its rebalance needs a root→leaf path;
//! if the swap split `pid` that recorded path is stale, so we re-descend for a fresh one
//! (rare²: only a wide-predecessor swap that overflows AND underflows its leaf).
//!
//! ## Rebalancing is a classic-B underflow fix (rotate / merge / redistribute)
//! `index_insert` keeps every non-root interior at K>=2 (>=3 children) and every
//! non-root leaf at >=1 cell (fileformat2 §1.6 — the only K<2 interior it allows is
//! page 1, which an index root never is). So the textbook invariant holds BEFORE each
//! delete, and this module RE-ESTABLISHES it after. When a non-root leaf underflows (or,
//! climbing, a non-root interior falls to K<2) we GATHER the underflowing node with an
//! adjacent sibling and the real divider between them, then REDISTRIBUTE the gathered
//! sequence into the fewest valid pages: one page is a MERGE (the parent loses a child
//! + its bordering divider); two-or-more is a REDISTRIBUTE that promotes fresh boundary
//! divider(s) back to the parent. A merge can push the parent below K>=2 in turn, so we
//! climb one level and repeat (bounded by tree height, no re-descent). Variable-width
//! keys make a naive two-node merge unsafe — it can overflow, or leave two halves that
//! cannot both reach K>=2 — so when two nodes cannot repack into all-valid pages we fold
//! in a further adjacent sibling and repack across the wider window (real sqlite's
//! balance uses up to three; the gather widens as far as it must). A REDISTRIBUTE (or a
//! net-growing MERGE) promotes fresh boundary dividers into the parent, and a promoted
//! index key can encode WIDER than the one it replaced — so rewriting a near-full parent
//! can OVERFLOW one page. That is NOT an error: the parent then SPLITS and propagates up
//! exactly as an insert split does (`split_interior_and_propagate`, the same growth the
//! predecessor swap above uses), so the DELETE succeeds like real sqlite instead of
//! failing closed. The `split_index_interior` it calls fails closed only on the
//! pre-existing large-page m<5 edge (see its docs), unreachable at the <=8 KiB pages an
//! index interior overflows at (m>=5 there). The `index_balance` planners choose the group
//! boundaries; see `rebalance_after_leaf_underflow`.
//!
//! ## Space reclamation
//! The deleted key's overflow chain (freed in phases 1-3) and every page a merge or a
//! root collapse makes redundant are returned to the freelist, so a delete-heavy
//! workload stays bounded (as real sqlite does). No DIVIDER chain is freed during a
//! rebalance: gather/redistribute preserves every surviving key exactly once (a divider
//! is a live key), so its chain is always still referenced. Redundant pages are freed
//! AFTER all writes complete (no live page-view borrow), so a following insert reuses
//! them and a delete never grows the file.

use std::cmp::Ordering;

use minisqlite_fileformat::{
    delete_cell, encode_index_leaf_cell, leaf_free_space, PageBuilder, PageType, PageView,
};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::index::{balance_deeper_interior, propagate_split};
use crate::index_balance::{plan_index_interior_groups, plan_index_leaf_groups};
use crate::index_build::{build_index_interior, build_index_leaf, split_index_interior, IndexCell};
use crate::index_key::{cell_key_bytes, compare_index_keys, KeyOrder};
use crate::index_nav::{index_choose_child, index_leaf_search, index_pointer_at};
use crate::overflow_io::free_overflow_chain;
use crate::tree::{put_page, usable_of};

/// Delete the encoded index key `key` from the index b-tree rooted at `root`.
/// Returns `Ok(true)` if the key was present and removed, `Ok(false)` if it was
/// absent (the tree is left unchanged). Runs within whatever transaction the caller
/// has open (it does not begin/commit). O(tree height).
///
/// The tree is taken as all-ascending BINARY: shorthand for
/// [`index_delete_with_order`] with [`KeyOrder::default`].
pub fn index_delete(pager: &mut dyn Pager, root: PageId, key: &[u8]) -> Result<bool> {
    index_delete_with_order(pager, root, key, &KeyOrder::default())
}

/// [`index_delete`] on a tree ordered by `order` (its per-column sort order and
/// collation), which must be the order the tree was written in.
pub fn index_delete_with_order(
    pager: &mut dyn Pager,
    root: PageId,
    key: &[u8],
    order: &KeyOrder,
) -> Result<bool> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);

    // Locate the key, recording the interior descent as `(page_id, ptr)` so a fix-up
    // can reach each parent. For a leaf hit `path` runs root..parent(leaf); for an
    // interior-divider hit it runs root..parent(pid) (the divider page is not pushed).
    let mut path: Vec<(PageId, u32)> = Vec::new();
    match locate(pager, root, key, usable, order, &mut path)? {
        None => Ok(false),
        Some(Target::Leaf { leaf, idx }) => {
            delete_leaf_cell(pager, root, path, leaf, idx, page_size, usable)?;
            Ok(true)
        }
        Some(Target::Interior { pid, ptr }) => {
            delete_interior_divider(pager, root, path, pid, ptr as usize, page_size, usable, order)?;
            Ok(true)
        }
    }
}

/// Where the key to delete lives in the tree.
enum Target {
    /// Cell `idx` of leaf page `leaf`.
    Leaf { leaf: PageId, idx: usize },
    /// Divider cell `ptr` of interior page `pid`.
    Interior { pid: PageId, ptr: u32 },
}

/// Descend from `root` to locate `key`, recording the interior path. A descent that
/// routes past an interior divider EQUAL to `key` stops there — that divider IS the
/// key (a real classic-B entry), so the target is the interior cell, not a leaf.
/// Returns `None` if the key is absent.
fn locate(
    pager: &dyn Pager,
    root: PageId,
    key: &[u8],
    usable: usize,
    order: &KeyOrder,
    path: &mut Vec<(PageId, u32)>,
) -> Result<Option<Target>> {
    let mut current = root;
    loop {
        let view = PageView::new(pager.read_page(current)?, current, usable)?;
        if !view.page_type().is_index() {
            return Err(Error::format("index_delete reached a non-index b-tree page"));
        }
        if view.page_type().is_leaf() {
            return match index_leaf_search(pager, &view, key, usable, order)? {
                Ok(idx) => Ok(Some(Target::Leaf { leaf: current, idx })),
                Err(_) => Ok(None),
            };
        }
        let (child, ptr) = index_choose_child(pager, &view, key, usable, order)?;
        if (ptr as usize) < view.cell_count() as usize {
            let cell = view.index_interior_cell(ptr as usize)?;
            let dkey =
                cell_key_bytes(pager, cell.local_payload, cell.payload_len, cell.overflow_page, usable)?;
            if compare_index_keys(key, dkey.as_ref(), order) == Ordering::Equal {
                return Ok(Some(Target::Interior { pid: current, ptr }));
            }
        }
        path.push((current, ptr));
        current = child;
    }
}

/// Remove cell `idx` from leaf `leaf` (with `path` = root..parent(leaf)). The leaf's
/// own overflow chain is freed. The cell drop goes through `drop_leaf_cell_in_place` (an
/// O(cell) in-place freeblock splice on the transactional fast route, else a defragmenting
/// whole-page rebuild); then the root leaf is left as-is (an under-full or empty root leaf
/// is a valid index), while a non-root leaf that now UNDERFLOWS (empties, or drops under
/// ~1/3 full) is rebalanced so the tree stays dense and every non-root leaf keeps >= 1 cell.
fn delete_leaf_cell(
    pager: &mut dyn Pager,
    root: PageId,
    path: Vec<(PageId, u32)>,
    leaf: PageId,
    idx: usize,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let (chain_to_free, count_before) = {
        let view = PageView::new(pager.read_page(leaf)?, leaf, usable)?;
        let count = view.cell_count() as usize;
        if idx >= count {
            return Err(Error::format("index_delete: leaf removal index past the leaf's cells"));
        }
        (view.index_leaf_cell(idx)?.overflow_page, count)
    };

    // Drop the one cell: an O(cell) in-place splice on the transactional fast route, else a
    // defragmenting whole-page rebuild (see `drop_leaf_cell_in_place`). A removal only
    // shrinks, so it always fits — down to a valid 0-cell page when the last cell goes.
    drop_leaf_cell_in_place(pager, leaf, idx, page_size, usable)?;
    // Either route must drop EXACTLY one cell — the underflow/rebalance decision below (and
    // the fallback) assume it. Assert the post-condition at its cause (debug-only, so no
    // release cost); `count_before` is the already-read pre-drop count.
    debug_assert_eq!(
        PageView::new(pager.read_page(leaf).expect("re-read edited leaf"), leaf, usable)
            .expect("re-parse edited leaf")
            .cell_count() as usize,
        count_before - 1,
        "index delete dropped != 1 cell from the leaf"
    );
    if let Some(first) = chain_to_free {
        free_overflow_chain(pager, first, usable)?;
    }

    // The root leaf has no siblings, so any fill (including empty) is the valid index.
    // A non-root leaf that is still adequately full needs nothing further; one that
    // underflowed is rebalanced up its path.
    if path.is_empty() || !leaf_underflows(pager, leaf, usable)? {
        return Ok(());
    }
    rebalance_after_leaf_underflow(pager, path, root, page_size, usable)
}

/// Delete the interior divider `K` at `(pid, ptr)` (with `path_above` =
/// root..parent(pid)) via the predecessor swap. See the module docs for the full
/// argument; briefly: install the predecessor `P` over `K`'s slot (splitting `pid` and
/// propagating up if it overflows), then remove `P` from its leaf.
#[allow(clippy::too_many_arguments)]
fn delete_interior_divider(
    pager: &mut dyn Pager,
    root: PageId,
    path_above: Vec<(PageId, u32)>,
    pid: PageId,
    ptr: usize,
    page_size: usize,
    usable: usize,
    order: &KeyOrder,
) -> Result<()> {
    // Descend the RIGHT spine of the divider's left subtree (child pointer `ptr`) to
    // the predecessor leaf, recording each interior at its right-most pointer index.
    let mut spine: Vec<(PageId, u32)> = Vec::new();
    let child0 = {
        let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
        index_pointer_at(&view, ptr as u32)?
    };
    let pred_leaf = descend_right_spine(pager, child0, usable, &mut spine)?;

    // The predecessor `P` is the last cell of `pred_leaf`.
    let (pred_idx, pred_cell, p_key) = {
        let view = PageView::new(pager.read_page(pred_leaf)?, pred_leaf, usable)?;
        let count = view.cell_count() as usize;
        if count == 0 {
            return Err(Error::format("index_delete: predecessor leaf is empty"));
        }
        let idx = count - 1;
        let c = view.index_leaf_cell(idx)?;
        let cell = IndexCell::from_parsed(c.payload_len, c.local_payload, c.overflow_page);
        let p_key =
            cell_key_bytes(pager, c.local_payload, c.payload_len, c.overflow_page, usable)?.into_owned();
        (idx, cell, p_key)
    };

    // `K`'s old overflow chain, freed once `P` has replaced it in the divider slot.
    let old_divider_overflow = {
        let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
        view.index_interior_cell(ptr)?.overflow_page
    };

    // The predecessor must be strictly less than the divider (it is the max of the
    // left subtree). A violation would mean the right-spine descent was wrong and we
    // are about to write an out-of-order divider — catch it loudly in debug.
    #[cfg(debug_assertions)]
    {
        let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
        let c = view.index_interior_cell(ptr)?;
        let k_key = cell_key_bytes(pager, c.local_payload, c.payload_len, c.overflow_page, usable)?;
        assert_eq!(
            compare_index_keys(&p_key, k_key.as_ref(), order),
            Ordering::Less,
            "index_delete: predecessor must be strictly less than the divider it replaces",
        );
    }

    // Install `P` over `K`'s slot. Returns whether it had to split `pid` (which makes
    // the recorded path to `pred_leaf` stale).
    let split = install_divider(pager, root, &path_above, pid, ptr, pred_cell, page_size, usable)?;

    // `K` is gone from the interior; reclaim its old chain (before the leaf fix-up so a
    // reinsert can reuse the pages).
    if let Some(first) = old_divider_overflow {
        free_overflow_chain(pager, first, usable)?;
    }

    // Remove `P` from `pred_leaf`. `P`'s chain now belongs to the divider, so it is NOT
    // freed here. The leaf-cell drop needs no path, so a split above cannot invalidate
    // it.
    drop_leaf_cell_in_place(pager, pred_leaf, pred_idx, page_size, usable)?;
    if !leaf_underflows(pager, pred_leaf, usable)? {
        return Ok(());
    }

    // `pred_leaf` underflowed. Its rebalance needs a root→pred_leaf path; if the swap
    // split `pid`, re-descend for a fresh one, else the recorded spine is still valid.
    let path = if split {
        fresh_path_to_pred_leaf(pager, root, &p_key, pred_leaf, usable, order)?
    } else {
        let mut p = path_above;
        p.push((pid, ptr as u32));
        p.extend(spine);
        p
    };
    // Both branches yield a path whose deepest entry is `pred_leaf`'s parent. Verify it
    // still reaches `pred_leaf` before rebalancing — a real (release) fail-closed, not a
    // debug-only guard: if a future change let `install_divider` restructure while
    // returning `false`, or a split touched the leaf, this errors instead of silently
    // corrupting the tree.
    if let Some(&(parent, c)) = path.last() {
        let landed = {
            let view = PageView::new(pager.read_page(parent)?, parent, usable)?;
            index_pointer_at(&view, c)?
        };
        if landed != pred_leaf {
            return Err(Error::format(
                "index_delete: rebalance path no longer reaches the underflowed predecessor leaf",
            ));
        }
    }
    rebalance_after_leaf_underflow(pager, path, root, page_size, usable)
}

/// Replace divider `ptr` of interior page `pid` with `new_cell`. If the rebuilt page
/// fits, write it and return `Ok(false)`. If it overflows, split `pid` and propagate
/// the promoted divider up `path_above` (or balance-deeper if `pid` is the root),
/// returning `Ok(true)` — the signal that the tree above `pid` was restructured.
fn install_divider(
    pager: &mut dyn Pager,
    root: PageId,
    path_above: &[(PageId, u32)],
    pid: PageId,
    ptr: usize,
    new_cell: IndexCell,
    page_size: usize,
    usable: usize,
) -> Result<bool> {
    let (children, mut dividers) = match read_node(pager, pid, usable)? {
        NodeContent::Interior { children, dividers } => (children, dividers),
        NodeContent::Leaf(_) => {
            return Err(Error::format("index_delete: divider install target is not an interior page"));
        }
    };
    if ptr >= dividers.len() {
        return Err(Error::format("index_delete: divider index past the interior's dividers"));
    }
    dividers[ptr] = new_cell;

    if let Some(bytes) = build_index_interior(pid, &children, &dividers, page_size, usable)? {
        put_page(pager, pid, bytes)?;
        return Ok(false);
    }
    // The swap overflowed `pid`. Grow the tree exactly as an insert split would, through
    // the one shared delete-side interior-overflow split path.
    split_interior_and_propagate(pager, root, path_above, pid, &children, &dividers, page_size, usable)?;
    Ok(true)
}

/// Split an OVERFLOWED interior `pid` — whose intended content `children`/`dividers` no
/// longer fits one page — and grow the tree exactly as an insert split does: when `pid`
/// is the root (`path_above` empty) balance-deeper in place so the root id never changes;
/// otherwise move its right half onto a freshly allocated page and propagate the promoted
/// divider up `path_above` (root..parent(pid)).
///
/// This is the ONE delete-side interior-overflow growth path, shared by `install_divider`
/// (the predecessor swap) and `rebalance_child` (the redistribute/merge parent rewrite):
/// both grow a too-wide interior the same way real sqlite does, so the "a wider promoted
/// divider overflows a near-full interior — split, don't fail closed" fix lives in ONE
/// place. Keep it that way: a missed *second copy* of exactly this class was the original
/// bug, so any future change here (e.g. the large-page 3-way rebalance for the
/// `split_index_interior` m<5 edge) must not fix one caller and silently miss the other.
fn split_interior_and_propagate(
    pager: &mut dyn Pager,
    root: PageId,
    path_above: &[(PageId, u32)],
    pid: PageId,
    children: &[u32],
    dividers: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    if path_above.is_empty() {
        balance_deeper_interior(pager, root, children, dividers, page_size, usable)?;
    } else {
        let right_id = pager.allocate_page()?;
        let split = split_index_interior(children, dividers, pid, right_id, page_size, usable)?;
        put_page(pager, pid, split.left)?;
        put_page(pager, right_id, split.right)?;
        propagate_split(pager, path_above, root, split.sep, right_id, page_size, usable)?;
    }
    Ok(())
}

/// Re-descend from `root` to `pred_leaf` after an interior split moved it, returning a
/// fresh root→pred_leaf path. The predecessor `P` (`p_key`) is now an interior divider
/// (just installed); find it, then follow the right spine of its left subtree back
/// down to `pred_leaf`.
fn fresh_path_to_pred_leaf(
    pager: &dyn Pager,
    root: PageId,
    p_key: &[u8],
    pred_leaf: PageId,
    usable: usize,
    order: &KeyOrder,
) -> Result<Vec<(PageId, u32)>> {
    let mut path: Vec<(PageId, u32)> = Vec::new();
    let (pid, ptr) = match locate(pager, root, p_key, usable, order, &mut path)? {
        Some(Target::Interior { pid, ptr }) => (pid, ptr),
        _ => {
            return Err(Error::format(
                "index_delete: reinstalled predecessor divider not found after an interior split",
            ));
        }
    };
    path.push((pid, ptr));
    let child0 = {
        let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
        index_pointer_at(&view, ptr)?
    };
    let leaf = descend_right_spine(pager, child0, usable, &mut path)?;
    if leaf != pred_leaf {
        return Err(Error::format(
            "index_delete: predecessor leaf moved unexpectedly after an interior split",
        ));
    }
    Ok(path)
}

/// Descend the RIGHT spine from `start_child` down to the leaf at the bottom, pushing
/// each interior passed as `(page, right_most_pointer_index)` onto `path`. Returns the
/// bottom leaf id. One page view per level. Shared by the initial predecessor descent
/// and the post-split re-descent so the two record the spine identically.
fn descend_right_spine(
    pager: &dyn Pager,
    start_child: PageId,
    usable: usize,
    path: &mut Vec<(PageId, u32)>,
) -> Result<PageId> {
    let mut current = start_child;
    loop {
        let view = PageView::new(pager.read_page(current)?, current, usable)?;
        if !view.page_type().is_index() {
            return Err(Error::format("index_delete: non-index page on the predecessor spine"));
        }
        if view.page_type().is_leaf() {
            return Ok(current);
        }
        let count = view.cell_count() as u32;
        let next = index_pointer_at(&view, count)?;
        path.push((current, count));
        current = next;
    }
}

/// Drop cell `drop_idx` from leaf `leaf`. FAST route: edit the page bytes in place
/// ([`Pager::page_mut`] + [`delete_cell`](minisqlite_fileformat::delete_cell)) — an
/// O(cell) splice that leaves a freeblock (the page is not defragmented). A returned
/// mutable borrow cannot auto-commit, so `page_mut` requires an active transaction and
/// fails closed otherwise; then this FALLS BACK to rebuilding the whole leaf from scratch
/// through `PageBuilder` (streaming survivors straight from the borrowed page, one reused
/// buffer — the same hot-path discipline as `delete::rebuild_leaf_dropping`), which
/// defragments. Both leave the identical surviving cell set, cell-pointer array,
/// cell_count, and total free space; they differ only in physical layout (freeblock vs
/// defragmented), which reads and the [`leaf_underflows`] check below never key off (the
/// check uses `leaf_free_space` total, not the raw contiguous gap). A removal only shrinks
/// the page, so it always fits — a fallback overflow is corruption and fails closed.
fn drop_leaf_cell_in_place(
    pager: &mut dyn Pager,
    leaf: PageId,
    drop_idx: usize,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    // FAST route: drop the one cell in place under an active transaction. The mutable
    // borrow lives only inside the `Ok` arm; on `Err` nothing is borrowed, so the
    // fallback below can re-borrow the pager (mirrors insert's in-place splice).
    match pager.page_mut(leaf) {
        Ok(page) => return delete_cell(page, leaf, usable, drop_idx),
        Err(_) => { /* no active transaction: fall back to the whole-page rebuild. */ }
    }
    let bytes = {
        let view = PageView::new(pager.read_page(leaf)?, leaf, usable)?;
        let count = view.cell_count() as usize;
        let mut b = PageBuilder::new(page_size, usable, leaf, PageType::LeafIndex);
        let mut tmp = Vec::new();
        let mut overflowed = false;
        for i in 0..count {
            if i == drop_idx {
                continue;
            }
            let cell = view.index_leaf_cell(i)?;
            tmp.clear();
            encode_index_leaf_cell(cell.payload_len, cell.local_payload, cell.overflow_page, &mut tmp);
            if !b.add_cell(&tmp) {
                overflowed = true;
                break;
            }
        }
        if overflowed {
            return Err(Error::format("index_delete: leaf overflowed dropping a cell (corruption)"));
        }
        b.finish()
    };
    put_page(pager, leaf, bytes)
}

/// Does a non-root leaf underflow — i.e. must it be rebalanced? True when it holds no
/// cells (an empty non-root leaf is invalid and MUST be merged/rotated away), or when it
/// is under ~1/3 full (the density heuristic: real sqlite rebalances a leaf whose free
/// space exceeds 2/3 of the usable page, so a scattered delete-heavy index stays as
/// compact as the one sqlite leaves, holding scan cost and peak RSS to a bounded factor).
///
/// Fill is measured from TOTAL reclaimable free space (`leaf_free_space` total: gap +
/// freeblocks + fragments), NOT the raw contiguous gap. This is required for fast/fallback
/// parity: the FAST [`drop_leaf_cell_in_place`] leaves a freeblock (the page is not
/// defragmented), so the gap alone understates free space and would make the two routes
/// disagree on whether to rebalance. Total is layout-independent — identical for a
/// freeblock-bearing page and the defragmented rebuild of the same cells — so both routes
/// rebalance at the same fill. Walking the (short, bounded) freeblock chain is
/// O(freeblocks); index leaves are never page 1, so the header offset is 0.
fn leaf_underflows(pager: &dyn Pager, leaf: PageId, usable: usize) -> Result<bool> {
    let page = pager.read_page(leaf)?;
    let view = PageView::new(page, leaf, usable)?;
    if !view.page_type().is_index() || !view.page_type().is_leaf() {
        return Err(Error::format("index_delete: leaf underflow check on a non-leaf-index page"));
    }
    if view.cell_count() == 0 {
        return Ok(true);
    }
    let free = leaf_free_space(page, leaf, usable)?.total;
    // free > (2/3) * usable  ⟺  used < (1/3) * usable. Integer form avoids rounding.
    Ok(free * 3 > usable * 2)
}

/// A non-root leaf UNDERFLOWED (emptied, or dropped under ~1/3 full) — or, on the
/// recursive climb, a non-root interior fell below K>=2. Re-establish the classic-B
/// invariants bottom-up along the recorded `path` (root..parent-of-underflowing-node)
/// by GATHER + REDISTRIBUTE: combine the underflowing node with an adjacent sibling and
/// the real divider between them, then repack into the FEWEST valid pages — a MERGE when
/// it all fits one page (the parent loses a child and its bordering divider), else a
/// REDISTRIBUTE that promotes fresh boundary divider(s) back to the parent (the parent's
/// fan-out is unchanged). A merge can push the parent below K>=2, so climb one level and
/// repeat; a redistribute leaves the parent valid, so stop. When the root interior falls
/// to a single child, its content collapses up into the (stable) root id, dropping the
/// tree's height.
///
/// `path` runs root..parent(underfull_leaf), and its deepest entry's child index
/// identifies the underflowing leaf — the caller has ALREADY rewritten that leaf to its
/// correct post-removal cells (0 or more) via `drop_leaf_cell_in_place`, so the gather
/// reads it directly and this function never wipes it (the leaf id is therefore implicit
/// in `path`, not a separate argument). NO key is lost: each gathered leaf cell and each
/// pulled-down divider is placed on exactly one output page or promoted exactly once, so
/// every surviving key stays present exactly once, dividers stay real keys strictly
/// between their subtrees, and all subtrees keep equal height. Pages that disappear (a
/// merged-away node, a collapsed pass-through child) are freed AFTER all writes with no
/// live page-view borrow, so a later insert reuses them and `page_count` stays bounded.
/// O(tree height): one bounded rebalance per level, no re-descent from the root.
fn rebalance_after_leaf_underflow(
    pager: &mut dyn Pager,
    path: Vec<(PageId, u32)>,
    root: PageId,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    // A non-root caller always passes a non-empty path; this guard is defensive (a root
    // leaf is valid at any fill and needs no rebalance).
    if path.is_empty() {
        return Ok(());
    }

    // Pages made redundant during the climb, freed once at the end (after all writes,
    // no live borrow) so a later insert reuses them — like the table-b-tree delete.
    let mut orphans: Vec<PageId> = Vec::new();

    let mut level = path.len();
    while level > 0 {
        level -= 1;
        let (pid, i) = path[level];
        // The path ABOVE `pid` (root..parent(pid)) is everything shallower than `level`;
        // a redistribute that overflows `pid` splits it and propagates up that path.
        match rebalance_child(pager, pid, i as usize, root, &path[0..level], page_size, usable, &mut orphans)? {
            Rebalanced::Redistributed => break, // `pid` fan-out unchanged: it stays valid.
            Rebalanced::SplitPropagated => break, // `pid` split; the path at/above it is resolved.
            Rebalanced::Merged { children } => {
                if level == 0 {
                    // `pid` is the root. One child left (K==0) collapses up into the
                    // stable root id (height drops); 2 children (K==1) is a VALID root
                    // (a 2-child root cannot be K>=2, so never force it) — stop.
                    if children == 1 {
                        collapse_pass_through_root(pager, root, page_size, usable, &mut orphans)?;
                    }
                    break;
                }
                if children >= 3 {
                    break; // `pid` (non-root) still has K>=2 dividers — valid, stop.
                }
                // `pid` (non-root) fell to K<2; climb and rebalance it in turn.
            }
        }
    }

    for id in orphans {
        // The root id is stable and never freed (index roots are never page 1).
        if id != root {
            pager.free_page(id)?;
        }
    }
    Ok(())
}

/// The effect of rebalancing one child on its parent's fan-out.
enum Rebalanced {
    /// A redistribute/rotate: the parent kept the same number of children (the
    /// underflow was cured by moving keys across the sibling boundary). Underflow
    /// cannot propagate up, so the climb stops.
    Redistributed,
    /// A merge: the parent now has exactly `children` children (fewer than before).
    /// If that drops it below K>=2 the climb continues.
    Merged { children: usize },
    /// Rewriting the parent with the promoted (possibly wider) boundary divider(s)
    /// OVERFLOWED one page, so the parent was SPLIT and the promoted divider propagated
    /// up `path_above` (or the root was balance-deepened) — exactly as an insert split
    /// does. The whole path at and above the parent is now consistent (a split only grows
    /// ancestors, never leaves one underfull), so the climb must STOP.
    SplitPropagated,
}

/// A planned redistribution of a gathered window: the owned gathered items plus the
/// group boundaries [`plan_index_leaf_groups`] / [`plan_index_interior_groups`]
/// returned (see `index_balance` for the `(s, e)` convention).
enum Plan {
    Leaf { flat: Vec<IndexCell>, groups: Vec<(usize, usize)> },
    Interior { children: Vec<u32>, dividers: Vec<IndexCell>, groups: Vec<(usize, usize)> },
}

/// Rebalance child `i` of interior page `pid` after that child underflowed. Gather `i`
/// with an adjacent sibling (widening to a third when two cannot form valid pages) and
/// the divider(s) between them, redistribute into the fewest valid pages reusing the
/// gathered pages' own ids, and rewrite `pid` to point at the result. Redundant window
/// pages go to `orphans` (freed by the caller after all writes).
///
/// The redistribute REPLACES `pid`'s window-internal dividers with freshly PROMOTED
/// group-boundary dividers. A promoted divider is a real index key of arbitrary width, so
/// it can encode WIDER than the one it replaced — and a near-full `pid` can then OVERFLOW
/// one page when rewritten, even when its fan-out held (a redistribute) or shrank (a
/// net-growing merge). That is not an error: `pid` is SPLIT and the promoted divider
/// propagated up `path_above` (or the root balance-deepened), exactly as an insert split
/// does — the DELETE succeeds like real sqlite instead of failing closed. `root` and
/// `path_above` (root..parent(pid), i.e. everything above `pid`) are threaded in for that
/// split, which reuses the SAME redistribute result: its output pages and orphan frees are
/// already applied, so the split only restructures `pid` and above. See [`Rebalanced`] for
/// how each outcome drives the caller's climb (a split resolves everything above `pid`, so
/// it stops the climb).
fn rebalance_child(
    pager: &mut dyn Pager,
    pid: PageId,
    i: usize,
    root: PageId,
    path_above: &[(PageId, u32)],
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<Rebalanced> {
    let (children, dividers) = match read_node(pager, pid, usable)? {
        NodeContent::Interior { children, dividers } => (children, dividers),
        NodeContent::Leaf(_) => {
            return Err(Error::format("index_delete: rebalance parent is not an interior page"));
        }
    };
    let c = children.len();
    if i >= c {
        return Err(Error::format("index_delete: underflow child index past the parent's children"));
    }
    // All children of `pid` share a height (the tree is balanced), so the underflow
    // child's kind decides the whole window's gather shape.
    let child_is_leaf = {
        let view = PageView::new(pager.read_page(children[i])?, children[i], usable)?;
        view.page_type().is_leaf()
    };

    // Window of adjacent children including `i`, initially two (prefer a left sibling).
    // A non-root interior always has K>=2 (>=3 children) and the root parent here has
    // >=2 children, so a two-node window always exists.
    let mut w_lo = if i > 0 { i - 1 } else { i };
    let mut w_hi = w_lo + 1;
    if w_hi >= c {
        return Err(Error::format("index_delete: rebalance parent has no sibling for its child"));
    }

    // Gather + plan; on an infeasible interior plan (large spilled dividers that the
    // current window cannot repack into all-K>=2 pages) fold in one more adjacent sibling
    // and retry. Real sqlite's balance uses up to three nodes; this widens as far as the
    // parent's fan-out allows before failing closed, so no valid shape is left unhandled.
    let plan = loop {
        let planned = if child_is_leaf {
            let flat = gather_leaf_window(pager, &children, &dividers, w_lo, w_hi, usable)?;
            plan_index_leaf_groups(&flat, page_size, usable)?.map(|groups| Plan::Leaf { flat, groups })
        } else {
            let (gc, gd) = gather_interior_window(pager, &children, &dividers, w_lo, w_hi, usable)?;
            plan_index_interior_groups(&gc, &gd, page_size, usable)?
                .map(|groups| Plan::Interior { children: gc, dividers: gd, groups })
        };
        if let Some(p) = planned {
            break p;
        }
        if w_hi + 1 < c {
            w_hi += 1;
        } else if w_lo > 0 {
            w_lo -= 1;
        } else {
            return Err(Error::format(
                "index_delete: cannot redistribute the window even after folding in every sibling",
            ));
        }
    };

    let win_count = w_hi - w_lo + 1;
    let g = match &plan {
        Plan::Leaf { groups, .. } | Plan::Interior { groups, .. } => groups.len(),
    };
    if g > win_count {
        // A rebalance must never need more pages than it gathered (that would force an
        // allocation and break the bounded-page-count contract). The gathered content
        // is < win_count full pages (the underflow node is below minimum), so this is
        // unreachable — fail closed rather than allocate and corrupt the accounting.
        return Err(Error::format("index_delete: redistribution needs more pages than gathered"));
    }

    // Reuse the first `g` window child ids for the `g` output pages; the rest vanish.
    let reuse: Vec<PageId> = children[w_lo..w_lo + g].to_vec();
    orphans.extend_from_slice(&children[w_lo + g..=w_hi]);

    // Write the output pages and collect the g-1 dividers promoted between them.
    let mut promoted: Vec<IndexCell> = Vec::with_capacity(g.saturating_sub(1));
    match &plan {
        Plan::Leaf { flat, groups } => {
            for (k, &(s, e)) in groups.iter().enumerate() {
                write_leaf(pager, reuse[k], &flat[s..e], page_size, usable, "leaf redistribute")?;
                if k + 1 < g {
                    promoted.push(flat[e].clone());
                }
            }
        }
        Plan::Interior { children: gc, dividers: gd, groups } => {
            for (k, &(s, e)) in groups.iter().enumerate() {
                write_interior(pager, reuse[k], &gc[s..=e], &gd[s..e], page_size, usable, "interior redistribute")?;
                if k + 1 < g {
                    promoted.push(gd[e].clone());
                }
            }
        }
    }

    // Rebuild `pid`: replace window children [w_lo..=w_hi] with the reused output ids,
    // and the window-internal dividers [w_lo..w_hi] with the promoted ones. Everything
    // outside the window is unchanged.
    let mut new_children: Vec<u32> = Vec::with_capacity(c - win_count + g);
    new_children.extend_from_slice(&children[0..w_lo]);
    new_children.extend_from_slice(&reuse);
    new_children.extend_from_slice(&children[w_hi + 1..]);

    let mut new_dividers: Vec<IndexCell> =
        Vec::with_capacity(dividers.len() - (w_hi - w_lo) + promoted.len());
    new_dividers.extend_from_slice(&dividers[0..w_lo]);
    new_dividers.append(&mut promoted);
    new_dividers.extend_from_slice(&dividers[w_hi..]);

    debug_assert_eq!(
        new_children.len(),
        new_dividers.len() + 1,
        "index_delete: rebuilt parent must keep children == dividers + 1",
    );
    // Rewrite `pid` — written EXACTLY ONCE, here, from the snapshot read at the top of the
    // fn. INVARIANT (keep it): `pid` must not be written earlier in this fn — the
    // redistribute output writes above land only on `reuse` (child) ids, never on `pid` —
    // so rebuilding from the snapshot `new_children`/`new_dividers` faithfully reflects the
    // live page. An early partial `pid` write would make this rebuild clobber it.
    //
    // The promoted boundary dividers can encode WIDER than the window-internal ones they
    // replaced (see the fn docs), so a near-full `pid` may OVERFLOW one page here even
    // though its fan-out held or shrank. If it fits, write it and report the fan-out effect
    // as before. If it overflows, SPLIT `pid` and propagate up — exactly as an insert split
    // does — rather than failing closed (an Err where real sqlite's DELETE succeeds).
    if let Some(bytes) = build_index_interior(pid, &new_children, &new_dividers, page_size, usable)? {
        put_page(pager, pid, bytes)?;
        return Ok(if g < win_count {
            Rebalanced::Merged { children: new_children.len() }
        } else {
            Rebalanced::Redistributed
        });
    }
    // `pid` overflowed. Grow the tree from the SAME redistribute result (its output pages
    // and orphan frees are already applied, so splitting restructures only `pid` and
    // above), through the one shared delete-side interior-overflow split path.
    split_interior_and_propagate(pager, root, path_above, pid, &new_children, &new_dividers, page_size, usable)?;
    Ok(Rebalanced::SplitPropagated)
}

/// Gather adjacent LEAF children `[w_lo..=w_hi]` of a parent into one strictly-ascending
/// in-order cell sequence, splicing each pulled-down divider (a real key) between the
/// leaves it separates. A spilled cell keeps its overflow linkage (via [`IndexCell`]),
/// so a rebuilt page never truncates a key.
fn gather_leaf_window(
    pager: &dyn Pager,
    children: &[u32],
    dividers: &[IndexCell],
    w_lo: usize,
    w_hi: usize,
    usable: usize,
) -> Result<Vec<IndexCell>> {
    let mut flat: Vec<IndexCell> = Vec::new();
    for idx in w_lo..=w_hi {
        match read_node(pager, children[idx], usable)? {
            NodeContent::Leaf(cells) => flat.extend(cells),
            NodeContent::Interior { .. } => {
                return Err(Error::format("index_delete: expected a leaf sibling in the gather"));
            }
        }
        if idx < w_hi {
            flat.push(dividers[idx].clone());
        }
    }
    Ok(flat)
}

/// Gather adjacent INTERIOR children `[w_lo..=w_hi]` of a parent into one
/// `(children, dividers)` pair: concatenate the child-interiors' children, and pull
/// each parent divider DOWN between successive child-interiors' own dividers (classic-B:
/// the divider rejoins the level below, keeping the subtree order intact).
fn gather_interior_window(
    pager: &dyn Pager,
    children: &[u32],
    dividers: &[IndexCell],
    w_lo: usize,
    w_hi: usize,
    usable: usize,
) -> Result<(Vec<u32>, Vec<IndexCell>)> {
    let mut gc: Vec<u32> = Vec::new();
    let mut gd: Vec<IndexCell> = Vec::new();
    for idx in w_lo..=w_hi {
        match read_node(pager, children[idx], usable)? {
            NodeContent::Interior { children: sc, dividers: sd } => {
                gc.extend(sc);
                gd.extend(sd);
            }
            NodeContent::Leaf(_) => {
                return Err(Error::format("index_delete: expected an interior sibling in the gather"));
            }
        }
        if idx < w_hi {
            gd.push(dividers[idx].clone());
        }
    }
    Ok((gc, gd))
}

/// Collapse a root that is a single-child (K==0) interior — a redundant pass-through —
/// by pulling its child's content up into the (stable) root page id, dropping tree
/// height. Repeats while the pulled-up content is itself a single-child interior. Index
/// roots are never page 1, so the child's content always fits the root page. Bypassed
/// child pages go to `orphans` (freed by the caller after all writes).
fn collapse_pass_through_root(
    pager: &mut dyn Pager,
    root: PageId,
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<()> {
    loop {
        // A pass-through root is an interior page with 0 dividers whose sole child is
        // the right-most pointer. Any other shape (leaf, or >= 1 divider) is not one.
        let child = {
            let view = PageView::new(pager.read_page(root)?, root, usable)?;
            if view.page_type().is_leaf() || view.cell_count() != 0 {
                return Ok(());
            }
            view.right_most_pointer().ok_or_else(|| {
                Error::format("index_delete: pass-through root missing its right-most pointer")
            })?
        };
        match read_node(pager, child, usable)? {
            NodeContent::Leaf(cells) => {
                write_leaf(pager, root, &cells, page_size, usable, "root collapse leaf")?;
                orphans.push(child);
                return Ok(());
            }
            NodeContent::Interior { children, dividers } => {
                write_interior(pager, root, &children, &dividers, page_size, usable, "root collapse interior")?;
                orphans.push(child);
                // Loop: the pulled-up content might itself be a 1-child interior.
            }
        }
    }
}

/// An index page read into owned lists, with the page-view borrow dropped so a
/// following `&mut pager` write is legal. A leaf yields its cells; an interior yields
/// parallel `(children, dividers)` where `children.len() == dividers.len() + 1`. Each
/// cell keeps its exact payload_len / inline bytes / overflow head via
/// `IndexCell::from_parsed`, so a rebuild never truncates a spilled key or drops its
/// chain. Used on the interior/rebalance paths that genuinely mutate the owned lists;
/// the leaf-removal path never builds one (`drop_leaf_cell_in_place` splices the cell out
/// in place on the fast route, else streams survivors straight from the page).
enum NodeContent {
    Leaf(Vec<IndexCell>),
    Interior { children: Vec<u32>, dividers: Vec<IndexCell> },
}

fn read_node(pager: &dyn Pager, pid: PageId, usable: usize) -> Result<NodeContent> {
    let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
    if !view.page_type().is_index() {
        return Err(Error::format("index_delete reached a non-index b-tree page"));
    }
    if view.page_type().is_leaf() {
        let count = view.cell_count() as usize;
        let mut cells = Vec::with_capacity(count);
        for i in 0..count {
            let c = view.index_leaf_cell(i)?;
            cells.push(IndexCell::from_parsed(c.payload_len, c.local_payload, c.overflow_page));
        }
        Ok(NodeContent::Leaf(cells))
    } else {
        let (children, dividers) = interior_lists_from(&view)?;
        Ok(NodeContent::Interior { children, dividers })
    }
}

/// Read an interior index page's parallel `(children, dividers)` from an open view.
fn interior_lists_from(view: &PageView) -> Result<(Vec<u32>, Vec<IndexCell>)> {
    if !view.page_type().is_interior() || !view.page_type().is_index() {
        return Err(Error::format("index_delete: expected an interior index page"));
    }
    let count = view.cell_count() as usize;
    let mut children = Vec::with_capacity(count + 1);
    let mut dividers = Vec::with_capacity(count);
    for i in 0..count {
        let c = view.index_interior_cell(i)?;
        children.push(c.left_child);
        dividers.push(IndexCell::from_parsed(c.payload_len, c.local_payload, c.overflow_page));
    }
    children.push(
        view.right_most_pointer()
            .ok_or_else(|| Error::format("index_delete: interior page missing its right-most pointer"))?,
    );
    Ok((children, dividers))
}

/// Build a leaf page from `cells` into `id` and write it, failing closed (with a
/// `where`-tagged message) if it unexpectedly overflows.
fn write_leaf(
    pager: &mut dyn Pager,
    id: PageId,
    cells: &[IndexCell],
    page_size: usize,
    usable: usize,
    where_: &str,
) -> Result<()> {
    let bytes = build_index_leaf(id, cells, page_size, usable)?
        .ok_or_else(|| Error::format(format!("index_delete: {where_} overflowed a page")))?;
    put_page(pager, id, bytes)
}

/// Build an interior page from `children`/`dividers` into `id` and write it, failing
/// closed if it unexpectedly overflows.
fn write_interior(
    pager: &mut dyn Pager,
    id: PageId,
    children: &[u32],
    dividers: &[IndexCell],
    page_size: usize,
    usable: usize,
    where_: &str,
) -> Result<()> {
    let bytes = build_index_interior(id, children, dividers, page_size, usable)?
        .ok_or_else(|| Error::format(format!("index_delete: {where_} overflowed a page")))?;
    put_page(pager, id, bytes)
}
