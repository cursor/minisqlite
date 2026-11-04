//! Pure navigation over one parsed index b-tree page. Like the table `nav`, these
//! read nothing from the pager — they operate on a borrowed [`PageView`] the caller
//! already opened — so the writer (descent) and the reader (cursor) share one
//! correct implementation of the separator rule.
//!
//! The index-b-tree separator invariant (fileformat2 §1.6, CLASSIC B-tree): for an
//! interior cell `(left_child, key)` the divider `key` is a REAL index entry present
//! exactly once in the whole tree; every key in `left_child`'s subtree is STRICTLY
//! LESS than `key`, and every key in the subtree to its right (the next cell's
//! `left_child`, or the right-most pointer for the last divider) is STRICTLY GREATER
//! than `key` (index keys are unique, so the bounds are strict). A search for `S`
//! takes the first cell whose `key >= S` and descends its `left_child` — unless
//! `S == key` exactly, in which case `S` *is* that interior divider (the caller
//! decides: a cursor is positioned on it, an insert treats it as a duplicate) — else,
//! past every divider, the right-most pointer. Comparison is by [`compare_index_keys`],
//! so a *prefix* search key routes to the first subtree that can hold a key with that
//! prefix.

use std::cmp::Ordering;

use minisqlite_fileformat::PageView;
use minisqlite_pager::Pager;
use minisqlite_types::{Error, Result};

use crate::index_key::{cell_key_bytes, compare_index_keys, KeyOrder};

/// Choose the child to descend into for `search_key` on an interior index page.
/// Returns `(child_page, pointer_index)` where `pointer_index` is the position of
/// the pointer taken (`0..=cell_count`), which the descent records so a later
/// sibling step (the cursor's `next`/`prev`) can resume. Binary search: the first
/// cell whose divider key `K` satisfies `search_key <= K`; if none, the right-most
/// pointer. O(log fanout).
///
/// `pager`/`usable` are threaded through only so an overflowed divider key is
/// reassembled (via `cell_key_bytes`) before comparison — comparing a spilled
/// divider on its inline prefix alone could tie two distinct keys and misroute.
pub(crate) fn index_choose_child(
    pager: &dyn Pager,
    view: &PageView,
    search_key: &[u8],
    usable: usize,
    order: &KeyOrder,
) -> Result<(u32, u32)> {
    let count = view.cell_count() as usize;
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cell = view.index_interior_cell(mid)?;
        let key = cell_key_bytes(pager, cell.local_payload, cell.payload_len, cell.overflow_page, usable)?;
        // First cell with search_key <= K, i.e. compare(search_key, K) is not Greater.
        if compare_index_keys(search_key, key.as_ref(), order) != Ordering::Greater {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo < count {
        Ok((view.index_interior_cell(lo)?.left_child, lo as u32))
    } else {
        Ok((right_pointer(view)?, count as u32))
    }
}

/// The child page at `pointer_index` on an interior index page: cell `i`'s left
/// child for `i < cell_count`, else (`== cell_count`) the right-most pointer.
pub(crate) fn index_pointer_at(view: &PageView, pointer_index: u32) -> Result<u32> {
    let count = view.cell_count() as u32;
    if pointer_index < count {
        Ok(view.index_interior_cell(pointer_index as usize)?.left_child)
    } else if pointer_index == count {
        right_pointer(view)
    } else {
        Err(Error::format("pointer index past the end of an interior index page"))
    }
}

/// The right-most child pointer, or a format error if the page is not interior.
fn right_pointer(view: &PageView) -> Result<u32> {
    view.right_most_pointer()
        .ok_or_else(|| Error::format("interior index page is missing its right-most pointer"))
}

/// Binary search a leaf index page for `search_key`. `Ok(i)` = an exact full-key
/// match at cell `i`; `Err(pos)` = absent, and `pos` is the first cell index whose
/// key is `> search_key` (the insertion point, and the landing point for a `>=`
/// seek). Leaf cells are ascending under [`compare_index_keys`] with `order` (the
/// tree's [`KeyOrder`]). `pager`/`usable`
/// let an overflowed leaf key be reassembled before comparison (see
/// `index_choose_child`).
pub(crate) fn index_leaf_search(
    pager: &dyn Pager,
    view: &PageView,
    search_key: &[u8],
    usable: usize,
    order: &KeyOrder,
) -> Result<core::result::Result<usize, usize>> {
    let count = view.cell_count() as usize;
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cell = view.index_leaf_cell(mid)?;
        let key = cell_key_bytes(pager, cell.local_payload, cell.payload_len, cell.overflow_page, usable)?;
        match compare_index_keys(search_key, key.as_ref(), order) {
            Ordering::Equal => return Ok(Ok(mid)),
            Ordering::Less => hi = mid,
            Ordering::Greater => lo = mid + 1,
        }
    }
    Ok(Err(lo))
}
