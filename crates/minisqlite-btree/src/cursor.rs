//! `TableCursor` — a borrowing forward/reverse/seek reader over a table b-tree.
//!
//! The cursor holds a descent **stack** of `(page_id, index)` from the root to the
//! current leaf: for an interior entry `index` is the pointer index descended, for
//! the leaf entry it is the current cell index. That stack is what makes `next` and
//! `prev` cost O(tree height) amortized — a step moves within the leaf, or climbs to
//! the nearest ancestor with an unvisited child on the correct side (`next` then
//! descends that child's left spine, `prev` its right spine) — instead of re-seeking
//! from the root each time, so a full scan in either direction is linear.
//!
//! Reads borrow pages from the pager (`&'p dyn Pager`). `payload` returns
//! `Cow::Borrowed` of the leaf cell's bytes with no copy for an inline row, and a
//! reassembled `Cow::Owned` for a row that spilled onto overflow pages. The pager is
//! shared and immutable for the cursor's life, so the tree does not change under an
//! open scan.

use std::borrow::Cow;

use minisqlite_fileformat::PageView;
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::nav::{choose_child, leaf_search, pointer_at};
use crate::tree::usable_of;

/// A read cursor over the table b-tree rooted at `root`. Position it with `first`,
/// `last`, `seek_exact`, or `seek_ge`, then read `rowid` / `payload` and step with
/// `next` (ascending) or `prev` (descending). `rowid` / `payload` are only
/// meaningful while positioned (a successful positioning call returned `true`).
pub struct TableCursor<'p> {
    pager: &'p dyn Pager,
    usable: usize,
    root: PageId,
    /// `(page_id, index)` from root to leaf. Empty means unpositioned. The last
    /// entry is always the current leaf and its cell index while `positioned`.
    stack: Vec<(PageId, u32)>,
    /// Cached rowid of the current cell, so `rowid()` is infallible and allocation
    /// free (set on every successful positioning).
    rowid: i64,
    positioned: bool,
}

impl<'p> TableCursor<'p> {
    /// Open a cursor over the tree rooted at `root`. Does not position it — call
    /// `first` / `last` / `seek_exact` / `seek_ge` first.
    pub fn open(pager: &'p dyn Pager, root: PageId) -> Result<TableCursor<'p>> {
        let usable = usable_of(pager);
        Ok(TableCursor { pager, usable, root, stack: Vec::new(), rowid: 0, positioned: false })
    }

    /// Position at the smallest rowid. Returns `false` (leaving the cursor
    /// unpositioned) if the tree is empty. Descends the left spine.
    pub fn first(&mut self) -> Result<bool> {
        self.stack.clear();
        self.positioned = false;
        self.descend_left_spine(self.root)
    }

    /// Position at the largest rowid. Returns `false` (leaving the cursor
    /// unpositioned) if the tree is empty. Descends the right spine — the mirror of
    /// `first` — so a following `prev` walks the whole tree in descending order.
    pub fn last(&mut self) -> Result<bool> {
        self.stack.clear();
        self.positioned = false;
        self.descend_right_spine(self.root)
    }

    /// Position exactly at `rowid`. Returns `false` if it is absent.
    pub fn seek_exact(&mut self, rowid: i64) -> Result<bool> {
        let leaf = self.seek_descend(rowid)?;
        let found = {
            let pager = self.pager;
            let view = PageView::new(pager.read_page(leaf)?, leaf, self.usable)?;
            match leaf_search(&view, rowid)? {
                Ok(idx) => Some((idx, view.table_leaf_cell(idx)?.rowid)),
                Err(_) => None,
            }
        };
        match found {
            Some((idx, rid)) => {
                self.stack.push((leaf, idx as u32));
                self.rowid = rid;
                self.positioned = true;
                Ok(true)
            }
            None => {
                self.stack.clear();
                Ok(false)
            }
        }
    }

    /// Position at the first rowid `>= rowid`. Returns `false` if none exists (the
    /// key is greater than every rowid in the tree). Used for range scans.
    pub fn seek_ge(&mut self, rowid: i64) -> Result<bool> {
        let leaf = self.seek_descend(rowid)?;
        // On the leaf `rowid` descends to, the first cell `>= rowid` is either the
        // exact match or the insertion point; if every cell there is `< rowid` the
        // answer (if any) is the first cell of the next leaf, reached via `next`.
        let landing = {
            let pager = self.pager;
            let view = PageView::new(pager.read_page(leaf)?, leaf, self.usable)?;
            let count = view.cell_count() as usize;
            if count == 0 {
                Landing::EmptyTree
            } else {
                match leaf_search(&view, rowid)? {
                    Ok(idx) => Landing::At(idx, view.table_leaf_cell(idx)?.rowid),
                    Err(pos) if pos < count => Landing::At(pos, view.table_leaf_cell(pos)?.rowid),
                    Err(_) => Landing::PastLeaf(count - 1),
                }
            }
        };
        match landing {
            Landing::EmptyTree => {
                self.stack.clear();
                Ok(false)
            }
            Landing::At(idx, rid) => {
                self.stack.push((leaf, idx as u32));
                self.rowid = rid;
                self.positioned = true;
                Ok(true)
            }
            Landing::PastLeaf(last) => {
                // Park on the last cell of this leaf, then advance to the next leaf.
                self.stack.push((leaf, last as u32));
                self.positioned = true;
                self.next()
            }
        }
    }

    /// Advance to the next rowid in ascending order, across leaves. Returns `false`
    /// at the end (leaving the cursor unpositioned).
    pub fn next(&mut self) -> Result<bool> {
        if !self.positioned {
            return Ok(false);
        }
        // Advance within the current leaf if possible.
        let advanced = {
            let pager = self.pager;
            let &(leaf_id, idx) = self.stack.last().expect("a positioned cursor has a leaf entry");
            let view = PageView::new(pager.read_page(leaf_id)?, leaf_id, self.usable)?;
            let count = view.cell_count() as u32;
            if idx + 1 < count {
                Some(view.table_leaf_cell((idx + 1) as usize)?.rowid)
            } else {
                None
            }
        };
        if let Some(rid) = advanced {
            let last = self.stack.last_mut().expect("leaf entry");
            last.1 += 1;
            self.rowid = rid;
            return Ok(true);
        }

        // Leaf exhausted: climb to the nearest ancestor with an unvisited child.
        self.stack.pop();
        loop {
            let Some(&(pid, pointer_index)) = self.stack.last() else {
                self.positioned = false;
                return Ok(false);
            };
            let next_child = {
                let pager = self.pager;
                let view = PageView::new(pager.read_page(pid)?, pid, self.usable)?;
                let count = view.cell_count() as u32;
                if pointer_index < count {
                    Some(pointer_at(&view, pointer_index + 1)?)
                } else {
                    None
                }
            };
            match next_child {
                Some(child) => {
                    self.stack.last_mut().expect("interior entry").1 = pointer_index + 1;
                    return self.descend_left_spine(child);
                }
                None => {
                    self.stack.pop();
                }
            }
        }
    }

    /// Move to the previous rowid in descending order, across leaves. Returns
    /// `false` at the start (leaving the cursor unpositioned). The mirror of `next`.
    pub fn prev(&mut self) -> Result<bool> {
        if !self.positioned {
            return Ok(false);
        }
        // Step back within the current leaf if there is an earlier cell.
        let stepped_back = {
            let &(leaf_id, idx) = self.stack.last().expect("a positioned cursor has a leaf entry");
            if idx > 0 {
                let pager = self.pager;
                let view = PageView::new(pager.read_page(leaf_id)?, leaf_id, self.usable)?;
                Some(view.table_leaf_cell((idx - 1) as usize)?.rowid)
            } else {
                None
            }
        };
        if let Some(rid) = stepped_back {
            let last = self.stack.last_mut().expect("leaf entry");
            last.1 -= 1;
            self.rowid = rid;
            return Ok(true);
        }

        // At the leaf's first cell: climb to the nearest ancestor that still has an
        // earlier child (pointer index > 0), step to it, and descend its right spine.
        self.stack.pop();
        loop {
            let Some(&(pid, pointer_index)) = self.stack.last() else {
                self.positioned = false;
                return Ok(false);
            };
            if pointer_index > 0 {
                let child = {
                    let pager = self.pager;
                    let view = PageView::new(pager.read_page(pid)?, pid, self.usable)?;
                    pointer_at(&view, pointer_index - 1)?
                };
                self.stack.last_mut().expect("interior entry").1 = pointer_index - 1;
                return self.descend_right_spine(child);
            }
            self.stack.pop();
        }
    }

    /// The current rowid. Only meaningful while positioned.
    pub fn rowid(&self) -> i64 {
        self.rowid
    }

    /// The number of rows in this cursor's table b-tree, counted without decoding any
    /// payload (see [`crate::table_entry_count`]). Independent of the cursor's position,
    /// and equal to a full `first`/`next` drain's row count — the `count(*)` fast path.
    pub fn entry_count(&self) -> Result<usize> {
        crate::table_entry_count(self.pager, self.root)
    }

    /// The current row's record bytes. An inline row is `Cow::Borrowed` straight
    /// from the leaf page with no copy; a row that spilled onto overflow pages is
    /// reassembled from its chain and returned `Cow::Owned`. Only valid while
    /// positioned.
    pub fn payload(&self) -> Result<Cow<'_, [u8]>> {
        let &(leaf_id, idx) = self
            .stack
            .last()
            .ok_or_else(|| Error::format("cursor payload requested while unpositioned"))?;
        let pager: &'p dyn Pager = self.pager;
        let data: &'p [u8] = pager.read_page(leaf_id)?;
        let view = PageView::new(data, leaf_id, self.usable)?;
        let cell = view.table_leaf_cell(idx as usize)?;
        crate::overflow_io::read_payload(
            pager,
            cell.local_payload,
            cell.payload_len,
            cell.overflow_page,
            self.usable,
        )
    }

    /// Descend from `start` taking the left-most pointer at each interior page until
    /// a leaf, pushing each page onto the stack and positioning at the leaf's first
    /// cell. Returns `false` only if `start` is an empty leaf (the empty-tree root).
    fn descend_left_spine(&mut self, start: PageId) -> Result<bool> {
        let pager = self.pager;
        let mut current = start;
        let page_limit = u64::from(pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let view = PageView::new(pager.read_page(current)?, current, self.usable)?;
            if !view.page_type().is_table() {
                return Err(Error::format("cursor reached a non-table b-tree page"));
            }
            if view.page_type().is_leaf() {
                if view.cell_count() == 0 {
                    // Only the root of an empty tree is a childless leaf; any leaf
                    // reached by descending a real pointer must be non-empty.
                    if self.stack.is_empty() {
                        self.positioned = false;
                        return Ok(false);
                    }
                    return Err(Error::format("empty leaf reached below an interior pointer"));
                }
                self.rowid = view.table_leaf_cell(0)?.rowid;
                self.stack.push((current, 0));
                self.positioned = true;
                return Ok(true);
            }
            let child = pointer_at(&view, 0)?;
            self.stack.push((current, 0));
            current = child;
        }
    }

    /// Descend from `start` taking the right-most pointer at each interior page
    /// until a leaf, pushing each interior with the pointer index taken (so `prev`
    /// can resume) and positioning at the leaf's last cell. Returns `false` only if
    /// `start` is an empty leaf (the empty-tree root). The mirror of
    /// `descend_left_spine`.
    fn descend_right_spine(&mut self, start: PageId) -> Result<bool> {
        let pager = self.pager;
        let mut current = start;
        let page_limit = u64::from(pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let view = PageView::new(pager.read_page(current)?, current, self.usable)?;
            if !view.page_type().is_table() {
                return Err(Error::format("cursor reached a non-table b-tree page"));
            }
            if view.page_type().is_leaf() {
                let count = view.cell_count();
                if count == 0 {
                    // Only the root of an empty tree is a childless leaf; any leaf
                    // reached by descending a real pointer must be non-empty.
                    if self.stack.is_empty() {
                        self.positioned = false;
                        return Ok(false);
                    }
                    return Err(Error::format("empty leaf reached below an interior pointer"));
                }
                let last_idx = count as u32 - 1;
                self.rowid = view.table_leaf_cell(last_idx as usize)?.rowid;
                self.stack.push((current, last_idx));
                self.positioned = true;
                return Ok(true);
            }
            // The right-most pointer's index is `cell_count` (one past the last cell).
            let ptr_index = view.cell_count() as u32;
            let child = pointer_at(&view, ptr_index)?;
            self.stack.push((current, ptr_index));
            current = child;
        }
    }

    /// Descend from the root choosing the child for `rowid` at each interior page,
    /// pushing the interior path, and return the leaf page reached. Leaves the leaf
    /// entry for the caller to push after searching it.
    fn seek_descend(&mut self, rowid: i64) -> Result<PageId> {
        self.stack.clear();
        self.positioned = false;
        let pager = self.pager;
        let mut current = self.root;
        let page_limit = u64::from(pager.page_count()?);
        let mut steps: u64 = 0;
        loop {
            steps += 1;
                        if steps > page_limit {
                            return Err(Error::format(
                                "cursor descent exceeded the database page count (cyclic or corrupt tree)",
                            ));
                        }
            let step = {
                let view = PageView::new(pager.read_page(current)?, current, self.usable)?;
                if !view.page_type().is_table() {
                    return Err(Error::format("cursor reached a non-table b-tree page"));
                }
                if view.page_type().is_leaf() {
                    None
                } else {
                    Some(choose_child(&view, rowid)?)
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
enum Landing {
    /// The whole tree is empty.
    EmptyTree,
    /// The answer is cell `idx` on this leaf, with the given rowid.
    At(usize, i64),
    /// Every cell on this leaf is `< key`; park on cell `last` and advance.
    PastLeaf(usize),
}
