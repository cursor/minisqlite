//! `IndexCursor` — a borrowing forward/reverse/seek reader over an index b-tree,
//! the index analogue of `TableCursor`.
//!
//! This index b-tree is a CLASSIC B-tree (like real sqlite's): every key appears
//! exactly ONCE, and an interior cell's divider is itself a real index entry that
//! sits in-order BETWEEN its two child subtrees. So a full in-order scan must
//! interleave interior dividers with leaf keys — a leaf-only scan (as a B+-tree
//! cursor does) would silently drop every entry that lives in an interior page.
//!
//! The cursor holds a descent **stack** of `(page_id, idx)` from the root to the
//! current position. The TOP frame is the current entry and its page may be a LEAF
//! (`idx` = the current cell) or an INTERIOR page (`idx` = the current divider cell
//! we are positioned AT, having just yielded it). Every NON-top frame is an interior
//! ancestor whose `idx` is the child-pointer index descended to reach the frame
//! below. These two readings of an interior `idx` coincide by construction: after
//! exhausting the subtree under pointer `i` (going forward) the next entry is divider
//! `i`, and to advance off divider `i` we descend pointer `i+1` — so `idx` is simply
//! bumped from `i` to `i+1` at that transition (and the mirror for `prev`).
//!
//! That stack makes `next`/`prev` cost O(tree height) amortized: a step advances or
//! retreats within a leaf, climbs to the nearest ancestor divider not yet yielded, or
//! descends one child's left (`next`) / right (`prev`) spine — never re-seeking from
//! the root — so a full scan is linear.
//!
//! `key` returns the FULL index record for the current position — the indexed columns
//! followed by the trailing rowid — reading a LEAF cell or an INTERIOR divider cell
//! depending on the top frame's page, as `Cow::Borrowed` of the cell's bytes with no
//! copy when the key has no overflow (an overflowed key is reassembled into an owned
//! buffer). The executor consumes an index entry by reading `key`, decoding its LAST
//! column to recover the rowid, and then seeking the table b-tree by that rowid, so
//! `key` must return the whole record including the rowid — it does not strip it.

use std::borrow::Cow;

use minisqlite_fileformat::PageView;
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::index_key::{cell_key_bytes, compare_index_keys};
use crate::index_nav::{index_choose_child, index_leaf_search, index_pointer_at};
use crate::tree::usable_of;

/// A read cursor over the index b-tree rooted at `root`. Position it with `first`,
/// `last`, `seek_ge`, or `seek_le`, then read `key` and move with `next` / `prev`.
/// `key` is only meaningful while positioned (a positioning call returned `true`).
pub struct IndexCursor<'p> {
    pager: &'p dyn Pager,
    usable: usize,
    root: PageId,
    /// `(page_id, idx)` from root to the current position. Empty means unpositioned.
    /// While `positioned`, the last frame is the current entry: a leaf cell, or (in
    /// this classic B-tree) an interior divider cell the cursor is sitting on. Every
    /// frame below the top is an interior ancestor whose `idx` is the child-pointer
    /// index descended.
    stack: Vec<(PageId, u32)>,
    positioned: bool,
}

impl<'p> IndexCursor<'p> {
    /// Open a cursor over the tree rooted at `root`. Does not position it — call
    /// `first` / `last` / `seek_ge` / `seek_le` first.
    pub fn open(pager: &'p dyn Pager, root: PageId) -> Result<IndexCursor<'p>> {
        let usable = usable_of(pager);
        Ok(IndexCursor { pager, usable, root, stack: Vec::new(), positioned: false })
    }

    /// Position at the smallest key (descend the left spine). Returns `false`
    /// (leaving the cursor unpositioned) if the tree is empty.
    pub fn first(&mut self) -> Result<bool> {
        self.stack.clear();
        self.positioned = false;
        self.descend_left_spine(self.root)
    }

    /// Position at the largest key (descend the right spine). Returns `false` if the
    /// tree is empty.
    pub fn last(&mut self) -> Result<bool> {
        self.stack.clear();
        self.positioned = false;
        self.descend_right_spine(self.root)
    }

    /// Position at the first key `>= key` under `compare_index_keys`. `key` may be a
    /// PREFIX (fewer columns than a full entry), which lands on the first entry whose
    /// indexed columns are `>= key`. Returns `false` if every key is `< key`.
    pub fn seek_ge(&mut self, key: &[u8]) -> Result<bool> {
        let leaf = self.seek_descend(key)?;
        // On the leaf `key` descends to, the first cell `>= key` is the exact match
        // or the insertion point; if every cell there is `< key` the answer (if any)
        // is the first cell of the next leaf, reached via `next`.
        let landing = {
            let view = PageView::new(self.pager.read_page(leaf)?, leaf, self.usable)?;
            let count = view.cell_count() as usize;
            if count == 0 {
                GeLanding::EmptyTree
            } else {
                match index_leaf_search(self.pager, &view, key, self.usable)? {
                    Ok(idx) => GeLanding::At(idx),
                    Err(pos) if pos < count => GeLanding::At(pos),
                    Err(_) => GeLanding::PastLeaf(count - 1),
                }
            }
        };
        match landing {
            GeLanding::EmptyTree => {
                self.stack.clear();
                Ok(false)
            }
            GeLanding::At(idx) => {
                self.stack.push((leaf, idx as u32));
                self.positioned = true;
                Ok(true)
            }
            GeLanding::PastLeaf(last) => {
                // Park on the last cell of this leaf, then advance to the next leaf.
                self.stack.push((leaf, last as u32));
                self.positioned = true;
                self.next()
            }
        }
    }

    /// Position at the last key `<= key` under `compare_index_keys` (for descending
    /// range scans). `key` may be a PREFIX. Returns `false` if every key is `> key`.
    ///
    /// Defined in terms of the (correct) [`seek_ge`](Self::seek_ge) plus one
    /// adjustment, NOT a leaf-only landing. A leaf-only `seek_le` is WRONG in this
    /// classic B-tree: when `key` equals a divider `K` that lives only in an interior
    /// page, `index_choose_child` routes the descent into `K`'s LEFT subtree (every
    /// key there is strictly `< K`), so the leaf search lands on `K`'s predecessor and
    /// never sees `K` itself. `seek_ge` already handles that case — its past-leaf
    /// `next()` climbs back up to yield the divider — so build the answer from it:
    ///   * `seek_ge` finds nothing `>= key`: every entry is `< key`, so the overall
    ///     maximum (`last`) is the greatest entry `<= key` (and `false` on an empty
    ///     tree);
    ///   * otherwise it lands on the first entry `E >= key`; if `E == key` exactly then
    ///     `E` is the answer (keys are unique, so `E` is the greatest `<= key`) — this
    ///     is the interior-divider case `seek_ge` recovers;
    ///   * else `E > key`, so the entry just before it (`prev`) is the greatest entry
    ///     `< key` = the greatest `<= key`; `prev` interleaves interior dividers
    ///     correctly and returns `false` (unpositioned) when nothing is `<= key`.
    /// A PREFIX `key` never compares `Equal` to a full entry (a shorter record sorts
    /// `Less`), so it always takes the `prev` path — the greatest entry ordered before
    /// the prefix group, the intended prefix `seek_le`. Cost stays O(log n).
    pub fn seek_le(&mut self, key: &[u8]) -> Result<bool> {
        if !self.seek_ge(key)? {
            // Nothing is >= key, so every entry is < key: the overall maximum (if the
            // tree is non-empty) is the greatest entry <= key.
            return self.last();
        }
        // Positioned at the first entry E >= key. If E == key exactly, E is the answer
        // (largest <= key, since keys are unique). This is the divider fix: when key is
        // an interior divider, seek_ge climbed to it via next(), so key() == key here.
        if compare_index_keys(self.key()?.as_ref(), key) == std::cmp::Ordering::Equal {
            return Ok(true);
        }
        // Otherwise E > key, so the entry just before E is the largest entry < key
        // (= largest <= key, since nothing equals key). prev() yields it, correctly
        // interleaving interior dividers; if there is none it returns false and leaves
        // the cursor unpositioned (nothing <= key).
        self.prev()
    }

    /// Advance to the next key in ascending in-order position, interleaving interior
    /// dividers with leaf keys (classic B-tree `BtreeNext`). Returns `false` at the
    /// end (leaving the cursor unpositioned).
    pub fn next(&mut self) -> Result<bool> {
        if !self.positioned {
            return Ok(false);
        }
        let &(page_id, idx) = self.stack.last().expect("a positioned cursor has a top frame");
        let (is_leaf, count) = self.page_kind(page_id)?;
        if is_leaf {
            // (a) Within the leaf: step right if there is another cell.
            if idx + 1 < count {
                self.stack.last_mut().expect("leaf frame").1 = idx + 1;
                return Ok(true);
            }
            // Leaf exhausted: pop it and climb. An ancestor `(pid, pi)` with
            // `pi < cell_count` has an unvisited divider `pi` immediately after the
            // subtree we just finished — position AT it (leave `pi` unchanged, it now
            // reads as "current divider" rather than "pointer descended"). `pi == count`
            // means we finished the right-most subtree with no divider after it, so keep
            // climbing.
            self.stack.pop();
            loop {
                let Some(&(pid, pi)) = self.stack.last() else {
                    self.positioned = false;
                    return Ok(false);
                };
                let (_pleaf, pcount) = self.page_kind(pid)?;
                if pi < pcount {
                    return Ok(true);
                }
                self.stack.pop();
            }
        } else {
            // (b) At interior divider `idx`: the next entry is the leftmost key of the
            // child to the RIGHT of the divider, i.e. child pointer `idx + 1`. Bump the
            // frame to that pointer (making it an ancestor again) and descend its left
            // spine to a leaf.
            let child = {
                let view = PageView::new(self.pager.read_page(page_id)?, page_id, self.usable)?;
                index_pointer_at(&view, idx + 1)?
            };
            self.stack.last_mut().expect("interior frame").1 = idx + 1;
            self.descend_left_spine(child)
        }
    }

    /// Retreat to the previous key in descending in-order position — the exact mirror
    /// of [`next`](Self::next) (classic B-tree `BtreePrev`). Returns `false` at the
    /// beginning (leaving the cursor unpositioned).
    pub fn prev(&mut self) -> Result<bool> {
        if !self.positioned {
            return Ok(false);
        }
        let &(page_id, idx) = self.stack.last().expect("a positioned cursor has a top frame");
        let (is_leaf, _count) = self.page_kind(page_id)?;
        if is_leaf {
            // (a) Within the leaf: step left if there is a preceding cell.
            if idx > 0 {
                self.stack.last_mut().expect("leaf frame").1 = idx - 1;
                return Ok(true);
            }
            // At the leaf's first cell: pop and climb. An ancestor `(pid, pi)` with
            // `pi > 0` has divider `pi - 1` immediately before the subtree we just
            // finished — position AT it (set the frame to `pi - 1`). `pi == 0` means we
            // finished the left-most subtree with no divider before it, so keep climbing.
            self.stack.pop();
            loop {
                let Some(&(_pid, pi)) = self.stack.last() else {
                    self.positioned = false;
                    return Ok(false);
                };
                if pi > 0 {
                    self.stack.last_mut().expect("interior frame").1 = pi - 1;
                    return Ok(true);
                }
                self.stack.pop();
            }
        } else {
            // (b) At interior divider `idx`: the previous entry is the rightmost key of
            // the child to the LEFT of the divider, i.e. child pointer `idx` (the left
            // child of divider `idx`). Keep the frame at `idx` (it becomes an ancestor
            // again, pointer `idx` descended) and descend its right spine to a leaf.
            let child = {
                let view = PageView::new(self.pager.read_page(page_id)?, page_id, self.usable)?;
                index_pointer_at(&view, idx)?
            };
            self.descend_right_spine(child)
        }
    }

    /// The current entry's full index record bytes (indexed columns + rowid). The top
    /// frame's page may be a LEAF (read the leaf cell) or, in this classic B-tree, an
    /// INTERIOR page the cursor is positioned on (read the divider cell). A key with no
    /// overflow is borrowed from the page with no copy (`Cow::Borrowed`); an overflowed
    /// key is reassembled from its chain into an owned buffer (`Cow::Owned`), byte-exact
    /// including the trailing rowid. Only valid while positioned.
    pub fn key(&self) -> Result<Cow<'_, [u8]>> {
        let &(page_id, idx) = self
            .stack
            .last()
            .ok_or_else(|| Error::format("index cursor key requested while unpositioned"))?;
        let pager: &'p dyn Pager = self.pager;
        let data: &'p [u8] = pager.read_page(page_id)?;
        let view = PageView::new(data, page_id, self.usable)?;
        // Read the cell at the top frame: a leaf cell, or an interior divider cell
        // (both expose the same local_payload / payload_len / overflow_page fields).
        let (local_payload, payload_len, overflow_page) = if view.page_type().is_leaf() {
            let cell = view.index_leaf_cell(idx as usize)?;
            (cell.local_payload, cell.payload_len, cell.overflow_page)
        } else {
            let cell = view.index_interior_cell(idx as usize)?;
            (cell.local_payload, cell.payload_len, cell.overflow_page)
        };
        cell_key_bytes(pager, local_payload, payload_len, overflow_page, self.usable)
    }

    /// The number of entries in this cursor's index b-tree, counted without decoding any
    /// key (see [`crate::index_entry_count`]). Independent of the cursor's position, and
    /// equal to a full `first`/`next` drain's entry count — so it is the row count of a
    /// WITHOUT ROWID table (its rows live in this PRIMARY KEY index b-tree). It counts
    /// interior divider cells too, because this is a classic B-tree.
    pub fn entry_count(&self) -> Result<usize> {
        crate::index_entry_count(self.pager, self.root)
    }

    /// Read a page's `(is_leaf, cell_count)` for a stepping decision, erroring if it
    /// is not an index b-tree page (the cursor must never wander into a table page).
    fn page_kind(&self, page_id: PageId) -> Result<(bool, u32)> {
        let view = PageView::new(self.pager.read_page(page_id)?, page_id, self.usable)?;
        if !view.page_type().is_index() {
            return Err(Error::format("index cursor reached a non-index b-tree page"));
        }
        Ok((view.page_type().is_leaf(), u32::from(view.cell_count())))
    }

    /// Descend from `start` taking the left-most pointer at each interior page until
    /// a leaf, pushing each page onto the stack and positioning at the leaf's first
    /// cell. Returns `false` only if `start` is an empty leaf (the empty-tree root).
    fn descend_left_spine(&mut self, start: PageId) -> Result<bool> {
        let mut current = start;
        let page_limit = u64::from(self.pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let view = PageView::new(self.pager.read_page(current)?, current, self.usable)?;
            if !view.page_type().is_index() {
                return Err(Error::format("index cursor reached a non-index b-tree page"));
            }
            if view.page_type().is_leaf() {
                if view.cell_count() == 0 {
                    // Only the root of an empty tree is a childless leaf; any leaf
                    // reached by descending a real pointer must be non-empty.
                    if self.stack.is_empty() {
                        self.positioned = false;
                        return Ok(false);
                    }
                    return Err(Error::format("empty index leaf reached below an interior pointer"));
                }
                self.stack.push((current, 0));
                self.positioned = true;
                return Ok(true);
            }
            let child = index_pointer_at(&view, 0)?;
            self.stack.push((current, 0));
            current = child;
        }
    }

    /// Descend from `start` taking the right-most pointer at each interior page until
    /// a leaf, pushing each page onto the stack and positioning at the leaf's last
    /// cell. Returns `false` only if `start` is an empty leaf (the empty-tree root).
    fn descend_right_spine(&mut self, start: PageId) -> Result<bool> {
        let mut current = start;
        let page_limit = u64::from(self.pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let view = PageView::new(self.pager.read_page(current)?, current, self.usable)?;
            if !view.page_type().is_index() {
                return Err(Error::format("index cursor reached a non-index b-tree page"));
            }
            let count = view.cell_count() as u32;
            if view.page_type().is_leaf() {
                if count == 0 {
                    if self.stack.is_empty() {
                        self.positioned = false;
                        return Ok(false);
                    }
                    return Err(Error::format("empty index leaf reached below an interior pointer"));
                }
                self.stack.push((current, count - 1));
                self.positioned = true;
                return Ok(true);
            }
            // The right-most pointer sits at pointer index == cell_count.
            let child = index_pointer_at(&view, count)?;
            self.stack.push((current, count));
            current = child;
        }
    }

    /// Descend from the root choosing the child for `key` at each interior page,
    /// pushing the interior path, and return the leaf page reached. Leaves the leaf
    /// entry for the caller to push after searching it.
    fn seek_descend(&mut self, key: &[u8]) -> Result<PageId> {
        self.stack.clear();
        self.positioned = false;
        let mut current = self.root;
        let page_limit = u64::from(self.pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let step = {
                let view = PageView::new(self.pager.read_page(current)?, current, self.usable)?;
                if !view.page_type().is_index() {
                    return Err(Error::format("index cursor reached a non-index b-tree page"));
                }
                if view.page_type().is_leaf() {
                    None
                } else {
                    Some(index_choose_child(self.pager, &view, key, self.usable)?)
                }
            };
            match step {
                None => return Ok(current),
                Some((child, pointer_index)) => {
                    self.stack.push((current, pointer_index));
                    current = child;
                }
            }
        }
    }
}

/// Where a `seek_ge` descent landed within its leaf.
enum GeLanding {
    /// The whole tree is empty.
    EmptyTree,
    /// The answer is cell `idx` on this leaf.
    At(usize),
    /// Every cell on this leaf is `< key`; park on cell `last` and advance.
    PastLeaf(usize),
}
