//! The index b-tree: create an empty index root, and insert an encoded index key
//! record with page splitting and root balance-deeper — the index analogue of the
//! table `tree`/`insert` modules.
//!
//! Shape of the insert algorithm (mirrors `table_insert`):
//! 1. Descend from `root` to the target leaf, recording the interior path as
//!    `(page_id, pointer_index)` so a split can be propagated to each parent. In a
//!    CLASSIC B-tree an interior divider is itself a real, unique index entry, so the
//!    descent checks each divider it routes past: if the key being inserted EQUALS a
//!    divider it already exists in the tree (in the interior, not a leaf), and the
//!    insert is a no-op — the classic-B analogue of step 2's leaf duplicate check.
//! 2. If the exact full key is already present in the landing leaf, do nothing (index
//!    keys are unique — an index key includes the rowid — so there is no REPLACE; a
//!    re-insert of the identical key is an idempotent no-op).
//! 3. Otherwise splice the new cell into key order. The FAST route edits the leaf bytes
//!    IN PLACE (`page_mut` + `insert_cell`) — an O(cell) splice, no whole-page re-encode
//!    — so a bulk index build does not re-encode a whole page per key. When no
//!    transaction is open (`page_mut` cannot hand out a committable borrow) it falls
//!    back to the O(page) whole-page `PageBuilder` rebuild.
//! 4. Otherwise split the leaf and propagate the promoted divider up the path,
//!    splitting interior pages as needed.
//! 5. When the page that must split is the root, **balance-deeper** instead of
//!    promoting: move the root's content down to a new child and turn the root into
//!    an interior page — so the root page id never changes (`IndexDef.root_page`
//!    references depend on it).
//!
//! Every split/rebuild write goes through `crate::tree::put_page` for uniformity
//! (index roots are never page 1, but routing all writes one way keeps the
//! header-preserving guarantee in one place); the in-place fast route edits the page
//! through `page_mut` and never rebuilds it. The initial empty-root write in
//! `create_index_btree` is written straight through `write_page` (a brand-new root has
//! no existing header to preserve), mirroring `create_table_btree`.

use std::cmp::Ordering;

use minisqlite_fileformat::{
    encode_index_leaf_cell, insert_cell, leaf_free_space, CellKind, PageBuilder, PageType, PageView,
};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::index_build::{
    build_index_interior, build_index_leaf, split_index_interior, split_index_leaf, IndexCell,
};
use crate::index_key::{cell_key_bytes, compare_index_keys, KeyOrder};
use crate::index_nav::{index_choose_child, index_leaf_search};
use crate::tree::{put_page, usable_of};

/// Create a new, empty index b-tree and return its root page id (>= 2). Formats a
/// freshly allocated page as an empty leaf index page; the id is what
/// `IndexDef.root_page` stores. Must be called after `init_database` (page 1 is
/// `sqlite_schema`, so an index root is never page 1).
pub fn create_index_btree(pager: &mut dyn Pager) -> Result<PageId> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);
    let id = pager.allocate_page()?;
    let page = PageBuilder::new(page_size, usable, id, PageType::LeafIndex).finish();
    pager.write_page(id, &page)?;
    Ok(id)
}

/// Insert the encoded index key `key_record` (indexed columns followed by the
/// rowid) into an all-ascending BINARY index b-tree rooted at `root`: shorthand for
/// [`index_insert_with_order`] with [`KeyOrder::default`].
pub fn index_insert(pager: &mut dyn Pager, root: PageId, key_record: &[u8]) -> Result<()> {
    index_insert_with_order(pager, root, key_record, &KeyOrder::default())
}

/// Insert the encoded index key `key_record` (indexed columns followed by the
/// rowid) into the index b-tree rooted at `root`, keeping cells in ascending
/// `compare_index_keys` order under `order` (the tree's per-column sort order and
/// collation). A re-insert of an identical full key is a no-op (index keys are
/// unique). A key past the inline threshold spills its tail onto a chain of overflow
/// pages. Runs within whatever transaction the caller has open.
pub fn index_insert_with_order(
    pager: &mut dyn Pager,
    root: PageId,
    key_record: &[u8],
    order: &KeyOrder,
) -> Result<()> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);

    // (1) Descend to the target leaf, recording the interior path. `pager`/`usable`
    // are threaded so a spilled divider key is reassembled before comparison.
    let mut path: Vec<(PageId, u32)> = Vec::new();
    let mut current = root;
    let leaf_id = loop {
        let data = pager.read_page(current)?;
        let view = PageView::new(data, current, usable)?;
        if !view.page_type().is_index() {
            return Err(Error::format("index_insert reached a non-index b-tree page"));
        }
        if view.page_type().is_leaf() {
            break current;
        }
        let (child, pointer_index) = index_choose_child(pager, &view, key_record, usable, order)?;
        // Classic-B duplicate check: `index_choose_child` returns the first divider
        // `K` with `key_record <= K`. An EXACT match means `key_record` is that
        // interior divider — a real, unique entry that lives here, not in the child
        // subtree we would descend — so it is already present and the insert is a
        // no-op. (A strict `key_record < K` routes into `children[pointer_index]` as
        // usual; `pointer_index == cell_count` means the key is past every divider.)
        if (pointer_index as usize) < view.cell_count() as usize {
            let cell = view.index_interior_cell(pointer_index as usize)?;
            let dkey =
                cell_key_bytes(pager, cell.local_payload, cell.payload_len, cell.overflow_page, usable)?;
            if compare_index_keys(key_record, dkey.as_ref(), order) == Ordering::Equal {
                return Ok(());
            }
        }
        path.push((current, pointer_index));
        current = child;
    };

    // (2) Locate the insertion point. A re-insert of the identical unique key is a
    // no-op — checked BEFORE writing any overflow chain so a duplicate neither leaks
    // overflow pages nor builds a cell for nothing.
    let pos = {
        let data = pager.read_page(leaf_id)?;
        let view = PageView::new(data, leaf_id, usable)?;
        match index_leaf_search(pager, &view, key_record, usable, order)? {
            Ok(_) => return Ok(()),
            Err(pos) => pos,
        }
    };

    // (3) The key is new. Spill its tail onto overflow pages if it exceeds the inline
    // threshold (nothing is allocated when it fits) and encode the leaf cell with the
    // inline prefix + chain head. `write_overflow_chain` only allocates fresh pages,
    // so the leaf read below re-reads the same unchanged page and `pos` stays valid.
    let (inline_len, overflow_page) =
        crate::overflow_io::write_overflow_chain(pager, key_record, CellKind::Index, usable)?;
    let mut new_cell = Vec::new();
    encode_index_leaf_cell(
        key_record.len() as u64,
        &key_record[..inline_len],
        overflow_page,
        &mut new_cell,
    );

    // (4) Decide, under a READ borrow of the leaf, whether the new cell fits in place.
    // The fit test — total reclaimable free `>= new_cell.len() + 2` (one cell plus its
    // 2-byte pointer) — is exactly `build_index_leaf_spliced`'s `PageBuilder` capacity
    // (re-encoding a cell preserves its length), so the in-place route and the rebuild
    // fallback split at the same boundary. Index keys are unique (no REPLACE), so a plain
    // insert is the only case.
    let plan = {
        let data = pager.read_page(leaf_id)?;
        let free = leaf_free_space(data, leaf_id, usable)?;
        if free.total >= new_cell.len() + 2 {
            IndexPlan::InPlace
        } else {
            let view = PageView::new(data, leaf_id, usable)?;
            IndexPlan::Split(collect_index_cells(&view, pos, key_record, inline_len, overflow_page)?)
        }
    };
    let cells = match plan {
        // (Fast route) Splice the cell into the leaf in place (falling back to a
        // whole-page rebuild only when in-place mutation is unavailable).
        IndexPlan::InPlace => {
            return insert_index_cell(pager, leaf_id, pos, &new_cell, page_size, usable);
        }
        IndexPlan::Split(cells) => cells,
    };

    // (5)/(6) The leaf overflowed. If it is the root, balance-deeper; else split and
    // propagate the divider up.
    if path.is_empty() {
        return balance_deeper_leaf(pager, root, &cells, page_size, usable);
    }
    let right_id = pager.allocate_page()?;
    let split = split_index_leaf(&cells, leaf_id, right_id, page_size, usable)?;
    put_page(pager, leaf_id, split.left)?;
    put_page(pager, right_id, split.right)?;
    propagate_split(pager, &path, root, split.sep, right_id, page_size, usable)
}

/// How to place the new index cell on the leaf: splice it in place, or split the leaf.
enum IndexPlan {
    /// The cell fits: insert it at key-order `pos` (computed by `index_leaf_search`).
    InPlace,
    /// The cell does not fit one page: the ordered cell list to split.
    Split(Vec<IndexCell>),
}

/// Place `new_cell` on the index leaf `leaf_id` at key-order `pos`. The FAST route edits
/// the page bytes in place through [`Pager::page_mut`] — an O(cell) splice. Because a
/// returned `&mut` borrow cannot auto-commit, `page_mut` requires an active transaction
/// and fails closed otherwise; then this FALLS BACK to rebuilding the whole leaf with
/// [`build_index_leaf_spliced`] and writing it via [`put_page`], the original O(page)
/// behavior. The caller has already verified the cell fits (see step 4), so a fallback
/// rebuild that overflows is corruption and fails loud rather than silently splitting.
fn insert_index_cell(
    pager: &mut dyn Pager,
    leaf_id: PageId,
    pos: usize,
    new_cell: &[u8],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    // The mutable borrow lives ONLY inside the `Ok` arm; on `Err` nothing is borrowed,
    // so the fallback below can re-borrow the pager.
    match pager.page_mut(leaf_id) {
        Ok(page) => {
            insert_cell(page, leaf_id, usable, pos, new_cell)?;
            return Ok(());
        }
        Err(_) => { /* no active transaction: fall back to the whole-page rebuild. */ }
    }
    let bytes = {
        let data = pager.read_page(leaf_id)?;
        let view = PageView::new(data, leaf_id, usable)?;
        build_index_leaf_spliced(&view, leaf_id, pos, new_cell, page_size, usable)?.ok_or_else(|| {
            Error::format("index leaf rebuild overflowed after the in-place fit check accepted it")
        })?
    };
    put_page(pager, leaf_id, bytes)
}

/// Rebuild `view` (a leaf) with `new_cell` spliced in at cell index `pos` (the
/// insertion point from `index_leaf_search`). `Ok(None)` if the result overflows one
/// page. Streams straight into a `PageBuilder`, re-encoding existing cells from their
/// borrowed bytes (preserving overflow linkage) — no owned cell list on the fast
/// path.
fn build_index_leaf_spliced(
    view: &PageView,
    leaf_id: PageId,
    pos: usize,
    new_cell: &[u8],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    let mut b = PageBuilder::new(page_size, usable, leaf_id, PageType::LeafIndex);
    let count = view.cell_count() as usize;
    let mut tmp = Vec::new();
    for i in 0..count {
        if i == pos && !b.add_cell(new_cell) {
            return Ok(None);
        }
        let cell = view.index_leaf_cell(i)?;
        tmp.clear();
        encode_index_leaf_cell(cell.payload_len, cell.local_payload, cell.overflow_page, &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(None);
        }
    }
    if pos >= count && !b.add_cell(new_cell) {
        return Ok(None);
    }
    Ok(Some(b.finish()))
}

/// Collect a leaf's cells as an owned, ordered list with the new key inserted at
/// `pos`. The new key is rebuilt from its `inline_len` prefix + `overflow_page` chain
/// head, so a spilled new key is stored with the exact inline/overflow split the
/// in-place path uses; existing cells keep their exact payload_len / inline bytes /
/// overflow head — nothing is truncated. Used only on the split path, so the per-cell
/// copies are amortized across a whole page's worth of inserts.
fn collect_index_cells(
    view: &PageView,
    pos: usize,
    key_record: &[u8],
    inline_len: usize,
    overflow_page: Option<u32>,
) -> Result<Vec<IndexCell>> {
    let count = view.cell_count() as usize;
    let mut out: Vec<IndexCell> = Vec::with_capacity(count + 1);
    for i in 0..count {
        if i == pos {
            out.push(IndexCell::new_key(key_record, inline_len, overflow_page));
        }
        let cell = view.index_leaf_cell(i)?;
        out.push(IndexCell::from_parsed(cell.payload_len, cell.local_payload, cell.overflow_page));
    }
    if pos >= count {
        out.push(IndexCell::new_key(key_record, inline_len, overflow_page));
    }
    Ok(out)
}

/// Propagate a promoted `(sep, new_right)` up the recorded interior `path`, from the
/// deepest parent toward the root. At each interior we insert the new child pointer
/// and divider; if the page still fits we write it and stop; if it overflows we
/// split it (promoting again) and continue; if the overflowing page is the root we
/// balance-deeper and stop.
///
/// `pub(crate)` so `index_delete` can reuse the exact same up-propagation when its
/// interior-divider swap overflows a page and must split it (the delete analogue of an
/// insert split). Keeping ONE propagation implementation means both write paths grow
/// the tree identically.
pub(crate) fn propagate_split(
    pager: &mut dyn Pager,
    path: &[(PageId, u32)],
    root: PageId,
    mut sep: IndexCell,
    mut new_right: PageId,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    for level in (0..path.len()).rev() {
        let (pid, pointer_index) = path[level];
        let (mut children, mut dividers) = read_index_interior_lists(pager, pid, usable)?;
        // The child at `pointer_index` split into (itself = left, new_right). Insert
        // the new right pointer after it and the promoted divider before it. In a
        // classic B-tree `sep` is the distinct entry MOVED UP out of the split child
        // (strictly greater than every key now under the left child, strictly less
        // than every key under `new_right`), so it slots in as `dividers[j]` between
        // the two pointers. See index_build::split_index_leaf/interior for the moved-up
        // (never copied) divider invariant.
        let j = pointer_index as usize;
        children.insert(j + 1, new_right);
        dividers.insert(j, sep);

        if let Some(bytes) = build_index_interior(pid, &children, &dividers, page_size, usable)? {
            return put_page(pager, pid, bytes);
        }
        // pid overflowed.
        if level == 0 {
            return balance_deeper_interior(pager, root, &children, &dividers, page_size, usable);
        }
        let right_id = pager.allocate_page()?;
        let split = split_index_interior(&children, &dividers, pid, right_id, page_size, usable)?;
        put_page(pager, pid, split.left)?;
        put_page(pager, right_id, split.right)?;
        sep = split.sep;
        new_right = right_id;
    }
    // Unreachable: a non-empty path always resolves at level 0 (fit or balance-deeper).
    Err(Error::format("index split propagation ran past the root"))
}

/// Read an interior index page into owned `(children, dividers)` lists (see
/// index_build docs). Each divider preserves its exact
/// payload_len/local_payload/overflow_page so a rebuilt or re-split parent never
/// truncates a (possibly overflowing) divider key.
fn read_index_interior_lists(
    pager: &dyn Pager,
    pid: PageId,
    usable: usize,
) -> Result<(Vec<u32>, Vec<IndexCell>)> {
    let data = pager.read_page(pid)?;
    let view = PageView::new(data, pid, usable)?;
    if !view.page_type().is_interior() || !view.page_type().is_index() {
        return Err(Error::format("expected an interior index page during split propagation"));
    }
    let count = view.cell_count() as usize;
    let mut children = Vec::with_capacity(count + 1);
    let mut dividers = Vec::with_capacity(count);
    for i in 0..count {
        let cell = view.index_interior_cell(i)?;
        children.push(cell.left_child);
        dividers.push(IndexCell::from_parsed(cell.payload_len, cell.local_payload, cell.overflow_page));
    }
    children.push(
        view.right_most_pointer()
            .ok_or_else(|| Error::format("interior index page missing its right-most pointer"))?,
    );
    Ok((children, dividers))
}

/// Grow the tree one level while keeping `root`'s page id: move the root's
/// (overflowing) leaf content to a new child and make the root an interior page. If
/// the content fits one child page the root gets a single right pointer to it (0
/// cells); otherwise the child is split and the root points to both halves.
fn balance_deeper_leaf(
    pager: &mut dyn Pager,
    root: PageId,
    cells: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let child = pager.allocate_page()?;
    if let Some(child_bytes) = build_index_leaf(child, cells, page_size, usable)? {
        // Fits in one child (only reachable when the root's capacity is smaller than
        // a full child's, which does not happen for index roots — they are never
        // page 1 — but kept for uniformity with the table balance-deeper).
        put_page(pager, child, child_bytes)?;
        let root_bytes = build_index_interior(root, &[child], &[], page_size, usable)?
            .ok_or_else(|| Error::format("root index interior unexpectedly overflowed with no cells"))?;
        return put_page(pager, root, root_bytes);
    }
    let right = pager.allocate_page()?;
    let split = split_index_leaf(cells, child, right, page_size, usable)?;
    put_page(pager, child, split.left)?;
    put_page(pager, right, split.right)?;
    write_root_over_split(pager, root, child, split.sep, right, page_size, usable)
}

/// Balance-deeper for an interior root: move the root's (overflowing) interior
/// content to a new child interior page and make the root point to it, splitting the
/// child if it does not fit one page. Mirrors `balance_deeper_leaf`.
///
/// `pub(crate)` for the same reason as `propagate_split`: when `index_delete`'s
/// divider swap overflows the ROOT interior, it grows the tree through this one
/// implementation rather than a second copy.
pub(crate) fn balance_deeper_interior(
    pager: &mut dyn Pager,
    root: PageId,
    children: &[u32],
    dividers: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let child = pager.allocate_page()?;
    if let Some(child_bytes) = build_index_interior(child, children, dividers, page_size, usable)? {
        put_page(pager, child, child_bytes)?;
        let root_bytes = build_index_interior(root, &[child], &[], page_size, usable)?
            .ok_or_else(|| Error::format("root index interior unexpectedly overflowed with no cells"))?;
        return put_page(pager, root, root_bytes);
    }
    let right = pager.allocate_page()?;
    let split = split_index_interior(children, dividers, child, right, page_size, usable)?;
    put_page(pager, child, split.left)?;
    put_page(pager, right, split.right)?;
    write_root_over_split(pager, root, child, split.sep, right, page_size, usable)
}

/// Write `root` as an interior page with a single divider: left child `left`,
/// divider `sep`, right pointer `right` (children `[left, right]`, dividers `[sep]`).
fn write_root_over_split(
    pager: &mut dyn Pager,
    root: PageId,
    left: PageId,
    sep: IndexCell,
    right: PageId,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let root_bytes = build_index_interior(root, &[left, right], &[sep], page_size, usable)?
        .ok_or_else(|| Error::format("root index interior unexpectedly overflowed with one cell"))?;
    put_page(pager, root, root_bytes)
}
