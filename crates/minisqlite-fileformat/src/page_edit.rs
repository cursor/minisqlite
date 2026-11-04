//! In-place, single-cell editing of a b-tree LEAF page (fileformat2 §1.6) — the
//! O(cell) counterpart of rebuilding a whole page with [`PageBuilder`](crate::PageBuilder).
//!
//! The b-tree's hot insert path splices ONE cell into a leaf. Rebuilding the page
//! (re-encoding every existing cell into a fresh `PageBuilder` buffer) costs
//! O(cells) per row; these editors mutate the page bytes directly, so a single
//! insert is O(cell) amortized: shift the 2-byte cell-pointer array by one slot,
//! drop the new cell into the unallocated gap (defragmenting first only when the gap
//! is fragmented), and patch the b-tree header fields. Every edit leaves a
//! SPEC-VALID SQLite leaf page that real sqlite reads back (invariant I below).
//!
//! Scope is LEAF pages only (table `0x0d`, index `0x0a`) — the only pages the insert
//! hot path mutates one cell at a time. Interior pages are still rebuilt through
//! `PageBuilder`, so these functions fail closed on a non-leaf page.
//!
//! ## Page layout (offsets relative to the header offset `H`, = 100 on page 1 else 0)
//! - `H+0` type · `H+1..3` first-freeblock (0 = none) · `H+3..5` cell count `N` ·
//!   `H+5..7` cell-content-area start `C` (stored 0 means 65536) · `H+7` fragmented
//!   free bytes.
//! - Cell pointer array (CPA) at `H+8`: `N` big-endian 2-byte offsets, in KEY order.
//! - Unallocated gap `[H+8+2N, C)`; content area `[C, U)` holding cells, freeblocks,
//!   and fragments (NOT necessarily contiguous or in key order — that is impl policy,
//!   not a format requirement, so appending a cell into the gap is correct).
//!
//! ## Invariant (I), preserved by every editor here
//! After any call the page decodes (via [`PageView`](crate::PageView)) to exactly the
//! intended ordered cell set and is spec-valid: `N` matches the CPA length; every CPA
//! offset lies in `[C, U)` and points at a whole cell; the freeblock chain is
//! offset-sorted, each block `>= 4` bytes and within `[C, U)`, and terminates; and
//! `C` / first-freeblock / fragmented-bytes stay mutually consistent. [`defragment`]
//! additionally restores the fully-packed shape `PageBuilder` produces — after it the
//! contiguous gap equals the page's total free space.

use crate::bytes::{be16, write_be16};
use crate::overflow::{payload_split, CellKind};
use crate::page::{header_offset_for, PageType};
use crate::varint::read_varint;
use minisqlite_types::{Error, Result};

/// Fixed leaf b-tree page-header length (fileformat2 §1.6): 8 bytes, before the cell
/// pointer array. (Interior pages use 12; the editors here reject interior pages.)
const LEAF_HEADER_LEN: usize = 8;

// Header field offsets, relative to the header offset `H`.
const HDR_FIRST_FREEBLOCK: usize = 1;
const HDR_CELL_COUNT: usize = 3;
const HDR_CONTENT_START: usize = 5;
const HDR_FRAG: usize = 7;

/// The format's ceiling on total fragmented (sub-4-byte) free bytes on a page
/// (fileformat2 §1.6): once a delete would push the total past this, the page is
/// defragmented instead of accumulating more fragments.
const MAX_FRAGMENTED_BYTES: usize = 60;

/// The free space available on a leaf page for placing another cell, split into the
/// total reclaimable free bytes and the contiguous unallocated gap.
///
/// `total` counts the unallocated gap plus every freeblock plus the fragmented
/// bytes: the space an insert can use *after* a [`defragment`]. `gap` is only the
/// contiguous unallocated region between the cell-pointer array and the content area:
/// the space an insert can use *without* defragmenting. A cell of `S` content bytes
/// plus its 2-byte pointer fits in place iff `total >= S + 2`, and needs no defrag
/// iff `gap >= S + 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSpace {
    /// All reclaimable free bytes: gap + freeblocks + fragments.
    pub total: usize,
    /// The contiguous unallocated gap `[cell-pointer-array-end, content-start)`.
    pub gap: usize,
}

/// Read `C` (the cell-content-area start), mapping the stored 0 to its 65536 meaning.
fn read_content_start(page: &[u8], h: usize) -> usize {
    let v = be16(page, h + HDR_CONTENT_START);
    if v == 0 { 65536 } else { v as usize }
}

/// Write `C`, using the format's 0 sentinel for a content area that starts at 65536.
fn write_content_start(page: &mut [u8], h: usize, c: usize) {
    let field = if c >= 65536 { 0 } else { c as u16 };
    write_be16(page, h + HDR_CONTENT_START, field);
}

/// Validate `page` as a LEAF b-tree page for `page_number` with usable size `usable`,
/// returning its header offset and type. Fails closed on a short page, a non-b-tree
/// type byte, or an interior page (the in-place editors handle leaves only).
fn leaf_header(page: &[u8], page_number: u32, usable: usize) -> Result<(usize, PageType)> {
    if page.len() < usable || usable < LEAF_HEADER_LEN + 1 {
        return Err(Error::format("page shorter than its usable size"));
    }
    let h = header_offset_for(page_number);
    if h + LEAF_HEADER_LEN > usable {
        return Err(Error::format("page too small for a leaf b-tree header"));
    }
    let page_type = PageType::from_byte(page[h])
        .ok_or_else(|| Error::format("in-place edit target is not a b-tree page"))?;
    if !page_type.is_leaf() {
        return Err(Error::format("in-place cell edit requires a leaf b-tree page"));
    }
    Ok((h, page_type))
}

/// On-page byte length of the cell at `off` on a leaf page of type `pt`, usable size
/// `usable`. A table-leaf cell is `varint(P), varint(rowid), inline payload,
/// [overflow ptr]`; an index-leaf cell drops the rowid varint. The inline length and
/// overflow presence come from the §1.6 spill split, so this matches exactly what
/// [`crate::PageView`] would parse and what the cell encoders wrote.
fn cell_len_at(page: &[u8], off: usize, pt: PageType, usable: usize) -> Result<usize> {
    let rest = page.get(off..).ok_or_else(|| Error::format("cell offset past end of page"))?;
    let (payload_len, n1) =
        read_varint(rest).ok_or_else(|| Error::format("bad payload varint in leaf cell"))?;
    let (kind, header_len) = if pt.is_table() {
        let after = page
            .get(off + n1..)
            .ok_or_else(|| Error::format("leaf cell truncated before its rowid"))?;
        let (_rowid, n2) =
            read_varint(after).ok_or_else(|| Error::format("bad rowid varint in leaf cell"))?;
        (CellKind::TableLeaf, n1 + n2)
    } else {
        (CellKind::Index, n1)
    };
    let split = payload_split(usable, payload_len, kind);
    let overflow = if split.has_overflow { 4 } else { 0 };
    Ok(header_len + split.local + overflow)
}

/// Public wrapper of [`cell_len_at`]: the on-page byte length of the leaf cell whose
/// content offset is `off`. Lets the b-tree size a REPLACE's old cell without
/// re-implementing the §1.6 cell layout.
pub fn leaf_cell_len(page: &[u8], page_number: u32, usable: usize, off: usize) -> Result<usize> {
    let (_h, pt) = leaf_header(page, page_number, usable)?;
    cell_len_at(page, off, pt, usable)
}

/// Total bytes in the freeblock chain, validating it is offset-sorted, each block
/// `>= 4` bytes and inside `[content_start, usable)`, and terminates. Fails closed on
/// any malformed link so a corrupt page cannot loop or over-count.
fn freeblock_total(page: &[u8], h: usize, content_start: usize, usable: usize) -> Result<usize> {
    let mut total = 0usize;
    let mut off = be16(page, h + HDR_FIRST_FREEBLOCK) as usize;
    let mut prev = 0usize;
    // Each freeblock is >= 4 bytes at a strictly increasing offset, so the chain has
    // at most usable/4 links; the guard turns a corrupt cycle into an error.
    let mut guard = usable / 4 + 1;
    while off != 0 {
        if guard == 0 {
            return Err(Error::format("freeblock chain too long (corrupt page)"));
        }
        guard -= 1;
        if off <= prev {
            return Err(Error::format("freeblock chain not offset-sorted (corrupt page)"));
        }
        if off < content_start || off + 4 > usable {
            return Err(Error::format("freeblock offset outside the content area (corrupt page)"));
        }
        let size = be16(page, off + 2) as usize;
        if size < 4 || off + size > usable {
            return Err(Error::format("freeblock size invalid (corrupt page)"));
        }
        total += size;
        prev = off;
        off = be16(page, off) as usize;
    }
    Ok(total)
}

/// Free-space summary of a leaf page (see [`FreeSpace`]). Reads only; the b-tree uses
/// it to decide, under a read borrow, whether a cell fits in place before taking the
/// mutable page.
pub fn leaf_free_space(page: &[u8], page_number: u32, usable: usize) -> Result<FreeSpace> {
    let (h, _pt) = leaf_header(page, page_number, usable)?;
    let n = be16(page, h + HDR_CELL_COUNT) as usize;
    let content_start = read_content_start(page, h);
    let ptr_end = h + LEAF_HEADER_LEN + 2 * n;
    if content_start < ptr_end || content_start > usable {
        return Err(Error::format("corrupt leaf page: content start outside the usable region"));
    }
    let gap = content_start - ptr_end;
    let frag = page[h + HDR_FRAG] as usize;
    let freeblocks = freeblock_total(page, h, content_start, usable)?;
    Ok(FreeSpace { total: gap + frag + freeblocks, gap })
}

/// Insert the already-encoded `cell` at cell-pointer index `pos` (`0..=N`), keeping
/// the pointer array in key order. `pos` is the caller's key-order insertion index.
///
/// Placement uses the contiguous gap; when the gap is too small the page is
/// [`defragment`]ed first (which coalesces every freeblock and fragment into the gap).
/// The caller MUST have verified the cell fits — total free `>= cell.len() + 2` — via
/// [`leaf_free_space`]; if it does not, this fails closed rather than corrupt the page
/// (the caller should split instead). For sequential appends into a fresh page this is
/// byte-for-byte what `PageBuilder` produces.
pub fn insert_cell(
    page: &mut [u8],
    page_number: u32,
    usable: usize,
    pos: usize,
    cell: &[u8],
) -> Result<()> {
    let (h, _pt) = leaf_header(page, page_number, usable)?;
    let n = be16(page, h + HDR_CELL_COUNT) as usize;
    if pos > n {
        return Err(Error::format("insert position past the cell count"));
    }
    let s = cell.len();
    if s == 0 {
        return Err(Error::format("cannot insert an empty cell"));
    }
    let cpa = h + LEAF_HEADER_LEN;
    let mut content_start = read_content_start(page, h);
    let ptr_end = cpa + 2 * n;
    if content_start < ptr_end || content_start > usable {
        return Err(Error::format("corrupt leaf page: content start outside the usable region"));
    }
    if content_start - ptr_end < s + 2 {
        // Not enough CONTIGUOUS space; reclaim freeblocks/fragments into the gap. The
        // caller guarantees total free >= s + 2, so the gap suffices after defrag.
        defragment(page, page_number, usable)?;
        content_start = read_content_start(page, h);
        if content_start - ptr_end < s + 2 {
            return Err(Error::format(
                "cell does not fit the leaf page after defragment (caller must split)",
            ));
        }
    }
    let new_content = content_start - s;
    // Drop the cell into the reclaimed gap.
    page[new_content..new_content + s].copy_from_slice(cell);
    // Open a 2-byte slot at `pos` by shifting the tail of the pointer array right.
    page.copy_within(cpa + 2 * pos..cpa + 2 * n, cpa + 2 * pos + 2);
    write_be16(page, cpa + 2 * pos, new_content as u16);
    // One more cell; the content area now starts at the placed cell.
    write_be16(page, h + HDR_CELL_COUNT, (n + 1) as u16);
    write_content_start(page, h, new_content);
    Ok(())
}

/// Delete the cell at index `pos` (`0..N`): drop its pointer (shifting the tail of the
/// pointer array left) and free its content region. The freed region joins the
/// contiguous gap when it sits at the content-area start, else becomes a freeblock
/// (>= 4 bytes) or fragmented bytes (< 4). Deleting the last cell resets the page to
/// the canonical empty shape. Leaves a spec-valid page (invariant I).
pub fn delete_cell(page: &mut [u8], page_number: u32, usable: usize, pos: usize) -> Result<()> {
    let (h, pt) = leaf_header(page, page_number, usable)?;
    let n = be16(page, h + HDR_CELL_COUNT) as usize;
    if pos >= n {
        return Err(Error::format("delete position past the cell count"));
    }
    let cpa = h + LEAF_HEADER_LEN;
    let off = be16(page, cpa + 2 * pos) as usize;
    let content_start = read_content_start(page, h);
    let sz = cell_len_at(page, off, pt, usable)?;
    if off < content_start || off + sz > usable {
        return Err(Error::format("corrupt leaf page: cell outside the content area"));
    }
    // Remove the pointer: shift the tail of the pointer array left by one slot.
    page.copy_within(cpa + 2 * (pos + 1)..cpa + 2 * n, cpa + 2 * pos);
    let new_n = n - 1;
    write_be16(page, h + HDR_CELL_COUNT, new_n as u16);
    if new_n == 0 {
        // Empty page: reset to the canonical clean shape (no freeblocks/fragments).
        write_be16(page, h + HDR_FIRST_FREEBLOCK, 0);
        write_content_start(page, h, usable);
        page[h + HDR_FRAG] = 0;
        return Ok(());
    }
    free_region(page, page_number, h, off, sz, content_start, usable)
}

/// Return the region `[off, off+sz)` of a just-unlinked cell to the page's free
/// space, updating the header fields. See [`delete_cell`].
fn free_region(
    page: &mut [u8],
    page_number: u32,
    h: usize,
    off: usize,
    sz: usize,
    content_start: usize,
    usable: usize,
) -> Result<()> {
    if off == content_start {
        // At the top of the content area: fold it back into the contiguous gap by
        // raising the content-area start, then swallow any freeblock that now sits
        // exactly at the new start (it can only be the chain head — the smallest
        // freeblock offset — since nothing lies below the new start).
        let mut new_content = content_start + sz;
        let mut fb = be16(page, h + HDR_FIRST_FREEBLOCK) as usize;
        let mut guard = usable / 4 + 1;
        while fb != 0 && fb == new_content && guard > 0 {
            guard -= 1;
            let fb_size = be16(page, fb + 2) as usize;
            let next = be16(page, fb) as usize;
            new_content += fb_size;
            fb = next;
        }
        write_content_start(page, h, new_content);
        write_be16(page, h + HDR_FIRST_FREEBLOCK, fb as u16);
        return Ok(());
    }
    if sz >= 4 {
        return insert_freeblock(page, h, off, sz, usable);
    }
    // A sub-4-byte hole cannot be a freeblock; count it as fragmented, defragmenting
    // if that would push the total past the format's 60-byte ceiling.
    let frag = page[h + HDR_FRAG] as usize;
    if frag + sz > MAX_FRAGMENTED_BYTES {
        defragment(page, page_number, usable)
    } else {
        page[h + HDR_FRAG] = (frag + sz) as u8;
        Ok(())
    }
}

/// Splice a freeblock of `size` bytes at `off` into the offset-sorted freeblock chain,
/// merging it with a neighbouring freeblock that it touches or that is at most 3 bytes
/// away. Bytes between merged blocks were fragments, so they leave the fragmented-byte
/// count. This is what sqlite3's `freeSpace()` does, and sqlite3's page checks reject a
/// chain where one freeblock starts within 3 bytes of the end of the previous one, so
/// an unmerged pair makes the page corrupt for sqlite3. `size >= 4` is the caller's
/// contract (a smaller hole is fragmented, not chained).
fn insert_freeblock(page: &mut [u8], h: usize, off: usize, size: usize, usable: usize) -> Result<()> {
    // Find the neighbours: `prev` is the last freeblock below `off` (0 = none, the
    // header slot links the chain) and `next` the first one above it.
    let mut prev = 0usize;
    let mut next = be16(page, h + HDR_FIRST_FREEBLOCK) as usize;
    let mut guard = usable / 4 + 1;
    while next != 0 && next < off {
        if guard == 0 {
            return Err(Error::format("freeblock chain too long (corrupt page)"));
        }
        guard -= 1;
        prev = next;
        next = be16(page, next) as usize;
    }
    if next == off {
        return Err(Error::format("freeing an already-free block (double free, corrupt page)"));
    }

    let mut start = off;
    let mut end = off + size;
    let mut absorbed = 0usize;
    if next != 0 {
        if next < end {
            return Err(Error::format("freed cell overlaps a freeblock (corrupt page)"));
        }
        if next <= end + 3 {
            absorbed += next - end;
            end = next + be16(page, next + 2) as usize;
            next = be16(page, next) as usize;
        }
    }
    if prev != 0 {
        let prev_end = prev + be16(page, prev + 2) as usize;
        if prev_end > start {
            return Err(Error::format("freed cell overlaps a freeblock (corrupt page)"));
        }
        if prev_end + 3 >= start {
            absorbed += start - prev_end;
            start = prev;
        }
    }
    let frag = page[h + HDR_FRAG] as usize;
    if absorbed > frag {
        return Err(Error::format("fragmented-byte count too small for the gap between freeblocks (corrupt page)"));
    }
    page[h + HDR_FRAG] = (frag - absorbed) as u8;

    write_be16(page, start, next as u16);
    write_be16(page, start + 2, (end - start) as u16);
    if start != prev {
        // Not folded into `prev`: link the new block in after it (or as the head).
        if prev == 0 {
            write_be16(page, h + HDR_FIRST_FREEBLOCK, start as u16);
        } else {
            write_be16(page, prev, start as u16);
        }
    }
    Ok(())
}

/// Compact a leaf page: repack every live cell against the end of the usable region
/// in key order and clear all freeblocks and fragmented bytes. After this the
/// contiguous gap equals the page's total free space, so an insert that fits at all
/// fits in the gap. The result is byte-for-byte what a `PageBuilder` rebuild of the
/// same cells would produce (unallocated space is zeroed), so a defragmented page and
/// a rebuilt page are indistinguishable on disk.
pub fn defragment(page: &mut [u8], page_number: u32, usable: usize) -> Result<()> {
    let (h, pt) = leaf_header(page, page_number, usable)?;
    let n = be16(page, h + HDR_CELL_COUNT) as usize;
    let cpa = h + LEAF_HEADER_LEN;
    let ptr_end = cpa + 2 * n;
    // Snapshot so cell reads never alias the compacted writes (defrag is off the hot
    // path, so a one-page copy here is fine).
    let src = page.to_vec();
    let mut content_top = usable;
    for i in 0..n {
        let off = be16(&src, cpa + 2 * i) as usize;
        let sz = cell_len_at(&src, off, pt, usable)?;
        if off + sz > usable {
            return Err(Error::format("corrupt leaf page: cell extends past the usable region"));
        }
        let new_off = content_top
            .checked_sub(sz)
            .filter(|&x| x >= ptr_end)
            .ok_or_else(|| Error::format("defragment: cells exceed the usable page space"))?;
        page[new_off..new_off + sz].copy_from_slice(&src[off..off + sz]);
        write_be16(page, cpa + 2 * i, new_off as u16);
        content_top = new_off;
    }
    // Zero the reclaimed gap so the page matches a PageBuilder rebuild byte-for-byte.
    for b in &mut page[ptr_end..content_top] {
        *b = 0;
    }
    write_be16(page, h + HDR_FIRST_FREEBLOCK, 0);
    write_content_start(page, h, content_top);
    page[h + HDR_FRAG] = 0;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{
        encode_index_leaf_cell, encode_table_leaf_cell, PageBuilder, PageType, PageView,
    };
    use crate::serial::encode_record;
    use minisqlite_types::Value;

    const PAGE: usize = 4096;
    const USABLE: usize = 4096;

    /// One entry in a reference model of a leaf page's contents: the decoded key
    /// (rowid for a table leaf, ignored for an index leaf) and the EXACT encoded cell
    /// bytes. `assert_valid` checks the page against this byte-for-byte, so it catches
    /// any corruption of a cell's content, not just its position.
    #[derive(Clone)]
    struct Entry {
        rowid: i64,
        cell: Vec<u8>,
    }

    /// Encode a table-leaf cell for `rowid` carrying an integer marker, no overflow.
    fn table_cell(rowid: i64, marker: i64) -> Vec<u8> {
        let payload = encode_record(&[Value::Integer(marker)]);
        let mut cell = Vec::new();
        encode_table_leaf_cell(payload.len() as u64, rowid, &payload, None, &mut cell);
        cell
    }

    /// Encode a table-leaf cell whose payload is `len` filler bytes (for sizing).
    fn table_cell_sized(rowid: i64, len: usize) -> Vec<u8> {
        let payload = vec![0xABu8; len];
        let mut cell = Vec::new();
        encode_table_leaf_cell(payload.len() as u64, rowid, &payload, None, &mut cell);
        cell
    }

    /// A fresh, empty leaf table page (page 3, so header offset 0).
    fn empty_table_page() -> Vec<u8> {
        PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable).finish()
    }

    /// Insert a table cell in key order into both the page and the reference model.
    fn model_insert(page: &mut [u8], page_number: u32, model: &mut Vec<Entry>, rowid: i64, cell: Vec<u8>) {
        let pos = model.partition_point(|e| e.rowid < rowid);
        insert_cell(page, page_number, USABLE, pos, &cell).unwrap();
        model.insert(pos, Entry { rowid, cell });
    }

    /// Decode the rowids currently on a leaf table page, in stored (key) order.
    fn rowids(page: &[u8], page_number: u32) -> Vec<i64> {
        let view = PageView::new(page, page_number, USABLE).unwrap();
        (0..view.cell_count() as usize)
            .map(|i| view.table_leaf_cell(i).unwrap().rowid)
            .collect()
    }

    /// Strong spec-validity oracle for a leaf TABLE page against a reference model:
    /// the page decodes to exactly the model's cells (byte-for-byte, in key order),
    /// cells and freeblocks occupy disjoint regions inside `[content_start, usable)`,
    /// rowids strictly ascend, and the freeblock chain is well-formed. This is the
    /// "real sqlite reads it back" check used across the tests.
    fn assert_valid(page: &[u8], page_number: u32, model: &[Entry]) {
        let h = header_offset_for(page_number);
        let view = PageView::new(page, page_number, USABLE).unwrap();
        assert_eq!(view.cell_count() as usize, model.len(), "cell count matches model");
        let content_start = read_content_start(page, h);
        let ptr_end = h + LEAF_HEADER_LEN + 2 * model.len();
        assert!(content_start >= ptr_end, "content start is past the pointer array");
        assert!(content_start <= USABLE, "content start within usable region");

        // Occupancy map of the content area: cells and freeblocks must be in range
        // and pairwise disjoint (no overlap).
        let mut occupied = vec![false; USABLE];
        for (i, entry) in model.iter().enumerate() {
            let cell = view.table_leaf_cell(i).unwrap();
            assert_eq!(cell.rowid, entry.rowid, "cell {i} rowid in key order");
            let off = view.cell_pointer(i);
            let sz = cell_len_at(page, off, PageType::LeafTable, USABLE).unwrap();
            assert!(off >= content_start && off + sz <= USABLE, "cell {i} within content area");
            assert_eq!(&page[off..off + sz], entry.cell.as_slice(), "cell {i} bytes match model");
            for b in &mut occupied[off..off + sz] {
                assert!(!*b, "cell {i} overlaps another cell/freeblock");
                *b = true;
            }
        }
        for w in model.windows(2) {
            assert!(w[0].rowid < w[1].rowid, "model rowids strictly ascending");
        }
        // Freeblock chain: well-formed, disjoint from cells, inside the content area.
        let mut fb = be16(page, h + HDR_FIRST_FREEBLOCK) as usize;
        let mut prev = 0usize;
        let mut prev_end = 0usize;
        let mut guard = USABLE;
        while fb != 0 {
            assert!(guard > 0, "freeblock chain terminates");
            guard -= 1;
            assert!(fb > prev, "freeblocks offset-sorted");
            // sqlite3 treats a freeblock that starts within 3 bytes of the end of the
            // previous one as a corrupt page (`free space corruption`), so neighbours
            // must be merged.
            assert!(prev == 0 || fb > prev_end + 3, "freeblock at {fb} not merged with the one ending at {prev_end}");
            let size = be16(page, fb + 2) as usize;
            assert!(size >= 4, "freeblock at least 4 bytes");
            assert!(fb >= content_start && fb + size <= USABLE, "freeblock within content area");
            for b in &mut occupied[fb..fb + size] {
                assert!(!*b, "freeblock overlaps a cell/another freeblock");
                *b = true;
            }
            prev = fb;
            prev_end = fb + size;
            fb = be16(page, fb) as usize;
        }
    }

    /// Five sequential rowids, packed from the end of the page, so cells for rowids 2
    /// and 3 sit next to each other below the cell for rowid 1.
    fn five_cell_page() -> (Vec<u8>, Vec<Entry>) {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        for rowid in 1..=5i64 {
            model_insert(&mut page, 3, &mut model, rowid, table_cell(rowid, rowid * 10));
        }
        (page, model)
    }

    /// Delete the cells for `rowids` one at a time and check that their space ends up
    /// as ONE freeblock covering both.
    fn assert_neighbours_merge(rowids: [i64; 2]) {
        let (mut page, mut model) = five_cell_page();
        let freed: usize = model
            .iter()
            .filter(|e| rowids.contains(&e.rowid))
            .map(|e| e.cell.len())
            .sum();
        for rowid in rowids {
            let pos = model.iter().position(|e| e.rowid == rowid).unwrap();
            delete_cell(&mut page, 3, USABLE, pos).unwrap();
            model.remove(pos);
        }
        assert_valid(&page, 3, &model);
        let h = header_offset_for(3);
        let fb = be16(&page, h + HDR_FIRST_FREEBLOCK) as usize;
        assert_ne!(fb, 0, "the freed cells form a freeblock");
        assert_eq!(be16(&page, fb), 0, "one freeblock, not a chain of two");
        assert_eq!(be16(&page, fb + 2) as usize, freed, "the freeblock covers both cells");
        assert_eq!(page[h + HDR_FRAG], 0, "no fragmented bytes");
    }

    #[test]
    fn freeing_a_cell_below_a_freeblock_merges_with_it() {
        // Rowid 3's cell ends where the freeblock left by rowid 2 starts.
        assert_neighbours_merge([2, 3]);
    }

    #[test]
    fn freeing_a_cell_above_a_freeblock_merges_with_it() {
        // Rowid 2's cell starts where the freeblock left by rowid 3 ends.
        assert_neighbours_merge([3, 2]);
    }

    #[test]
    fn insert_into_empty_page() {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        model_insert(&mut page, 3, &mut model, 1, table_cell(1, 100));
        assert_eq!(rowids(&page, 3), vec![1]);
        assert_valid(&page, 3, &model);
    }

    #[test]
    fn sequential_appends_match_page_builder_byte_for_byte() {
        // The hot path: append rowids 1..=20 one at a time into a fresh page. The
        // in-place result must be IDENTICAL to a PageBuilder rebuild of the same
        // cells, since the byte-exact on-disk write fixtures insert sequential rowids.
        let mut page = empty_table_page();
        let mut cells = Vec::new();
        for rowid in 1..=20i64 {
            let cell = table_cell(rowid, rowid * 10);
            let pos = PageView::new(&page, 3, USABLE).unwrap().cell_count() as usize; // append
            insert_cell(&mut page, 3, USABLE, pos, &cell).unwrap();
            cells.push(cell);
        }
        let mut b = PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable);
        for c in &cells {
            assert!(b.add_cell(c));
        }
        let built = b.finish();
        assert_eq!(page, built, "in-place sequential appends == PageBuilder rebuild");
    }

    #[test]
    fn insert_before_first_and_in_middle_keeps_key_order() {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        // Insert 10, 30, then 20 (middle) and 5 (before first) via key-order splice.
        for (rowid, marker) in [(10, 1), (30, 3), (20, 2), (5, 0)] {
            model_insert(&mut page, 3, &mut model, rowid, table_cell(rowid, marker));
        }
        assert_eq!(rowids(&page, 3), vec![5, 10, 20, 30]);
        assert_valid(&page, 3, &model);
    }

    #[test]
    fn delete_middle_creates_freeblock_then_page_still_valid() {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        for rowid in 1..=5i64 {
            model_insert(&mut page, 3, &mut model, rowid, table_cell(rowid, rowid));
        }
        // Delete rowid 3 (index 2). It is not at the content-area top, so it becomes
        // a freeblock; the page must remain spec-valid and carry a freeblock.
        let free_before = leaf_free_space(&page, 3, USABLE).unwrap();
        delete_cell(&mut page, 3, USABLE, 2).unwrap();
        model.remove(2);
        assert_eq!(rowids(&page, 3), vec![1, 2, 4, 5]);
        assert_valid(&page, 3, &model);
        let free_after = leaf_free_space(&page, 3, USABLE).unwrap();
        assert!(free_after.total > free_before.total, "delete increased total free space");
        assert!(be16(&page, HDR_FIRST_FREEBLOCK) != 0, "a mid-page delete leaves a freeblock");
    }

    #[test]
    fn insert_after_deletes_reuses_reclaimed_space_via_defragment() {
        // Build a page, delete a few middle cells (leaving freeblocks), then insert a
        // cell that fits only in the reclaimed (non-contiguous) space. The insert must
        // succeed by defragmenting, and the page stays valid byte-for-byte.
        let mut page = empty_table_page();
        let mut model = Vec::new();
        for rowid in 1..=40i64 {
            model_insert(&mut page, 3, &mut model, rowid, table_cell_sized(rowid, 80));
        }
        for _ in 0..3 {
            delete_cell(&mut page, 3, USABLE, 10).unwrap(); // rowids 11, 12, 13
            model.remove(10);
        }
        let fs = leaf_free_space(&page, 3, USABLE).unwrap();
        assert!(fs.total >= fs.gap, "freeblocks add to total beyond the gap");
        assert_valid(&page, 3, &model);

        // Insert rowid 12 back, sized to require reclaimed space.
        let cell = table_cell_sized(12, 80);
        assert!(leaf_free_space(&page, 3, USABLE).unwrap().total >= cell.len() + 2);
        model_insert(&mut page, 3, &mut model, 12, cell);
        assert_eq!(rowids(&page, 3).len(), model.len());
        assert_valid(&page, 3, &model);
    }

    #[test]
    fn replace_via_delete_then_insert_keeps_order_and_validity() {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        for rowid in 1..=5i64 {
            model_insert(&mut page, 3, &mut model, rowid, table_cell(rowid, rowid * 10));
        }
        // REPLACE rowid 3 (index 2) with a new, larger payload: delete then insert.
        delete_cell(&mut page, 3, USABLE, 2).unwrap();
        model.remove(2);
        let replacement = table_cell_sized(3, 200);
        insert_cell(&mut page, 3, USABLE, 2, &replacement).unwrap();
        model.insert(2, Entry { rowid: 3, cell: replacement });
        assert_eq!(rowids(&page, 3), vec![1, 2, 3, 4, 5]);
        assert_valid(&page, 3, &model);
        assert_eq!(PageView::new(&page, 3, USABLE).unwrap().table_leaf_cell(2).unwrap().payload_len, 200);
    }

    #[test]
    fn delete_all_cells_resets_to_empty_page() {
        let mut page = empty_table_page();
        let mut model = Vec::new();
        for rowid in 1..=4i64 {
            model_insert(&mut page, 3, &mut model, rowid, table_cell(rowid, rowid));
        }
        while PageView::new(&page, 3, USABLE).unwrap().cell_count() > 0 {
            delete_cell(&mut page, 3, USABLE, 0).unwrap();
        }
        // Deleting the last cell restores the canonical empty HEADER: no cells, no
        // freeblocks, no fragments, and the content area back at the usable end. The
        // unallocated bytes below it may hold stale data — like real sqlite, delete
        // does not zero freed space (that is `secure_delete`, off by default) — but
        // that region is invisible to a reader, so the page still decodes to empty and
        // is spec-valid. A defragment (or any rebuild) would zero it; delete need not.
        assert_valid(&page, 3, &[]);
        let empty = empty_table_page();
        assert_eq!(&page[..LEAF_HEADER_LEN], &empty[..LEAF_HEADER_LEN], "canonical empty header");
        assert_eq!(read_content_start(&page, 0), USABLE, "content area back at the usable end");
        assert_eq!(be16(&page, HDR_FIRST_FREEBLOCK), 0, "no freeblocks");
        assert_eq!(page[HDR_FRAG], 0, "no fragmented bytes");
        // A defragment of the empty page reproduces the fully-zeroed canonical page.
        defragment(&mut page, 3, USABLE).unwrap();
        assert_eq!(page, empty, "defragment of an empty page == fresh PageBuilder page");
    }

    #[test]
    fn defragment_equals_page_builder_rebuild() {
        // Insert then delete to fragment the page, defragment, and confirm the bytes
        // equal a PageBuilder rebuild of exactly the remaining cells.
        let mut page = empty_table_page();
        for rowid in 1..=12i64 {
            let pos = PageView::new(&page, 3, USABLE).unwrap().cell_count() as usize;
            insert_cell(&mut page, 3, USABLE, pos, &table_cell_sized(rowid, 60)).unwrap();
        }
        delete_cell(&mut page, 3, USABLE, 3).unwrap(); // rowid 4
        delete_cell(&mut page, 3, USABLE, 6).unwrap(); // rowid 8 (index shifted)
        let remaining = rowids(&page, 3);
        // Capture the exact cells still present, in order, to rebuild.
        let view = PageView::new(&page, 3, USABLE).unwrap();
        let mut cells = Vec::new();
        for i in 0..view.cell_count() as usize {
            let c = view.table_leaf_cell(i).unwrap();
            let mut bytes = Vec::new();
            encode_table_leaf_cell(c.payload_len, c.rowid, c.local_payload, c.overflow_page, &mut bytes);
            cells.push(bytes);
        }
        drop(view);

        defragment(&mut page, 3, USABLE).unwrap();
        assert_eq!(rowids(&page, 3), remaining, "defragment preserves cells in order");

        let mut b = PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable);
        for c in &cells {
            assert!(b.add_cell(c));
        }
        assert_eq!(page, b.finish(), "defragment output == PageBuilder rebuild of the same cells");
    }

    #[test]
    fn free_space_matches_page_builder_capacity_exactly() {
        // Pin the fit-equivalence the b-tree relies on: total free >= S + 2 iff a
        // PageBuilder can pack all existing cells plus the new one. Fill a page until
        // free_space says the next cell will not fit, and confirm PageBuilder agrees.
        let mut page = empty_table_page();
        let mut cells: Vec<Vec<u8>> = Vec::new();
        let mut rowid = 1i64;
        loop {
            let cell = table_cell_sized(rowid, 100);
            let fs = leaf_free_space(&page, 3, USABLE).unwrap();
            let fits = fs.total >= cell.len() + 2;

            // Independently: does a PageBuilder rebuild of all cells + this one fit?
            let mut b = PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable);
            let mut builder_fits = true;
            for c in &cells {
                if !b.add_cell(c) {
                    builder_fits = false;
                }
            }
            if builder_fits {
                builder_fits = b.add_cell(&cell);
            }
            assert_eq!(fits, builder_fits, "free-space fit decision == PageBuilder capacity (rowid {rowid})");

            if !fits {
                break;
            }
            insert_cell(&mut page, 3, USABLE, cells.len(), &cell).unwrap();
            cells.push(cell);
            rowid += 1;
            assert!(rowid < 1000, "page must fill before 1000 small cells");
        }
        assert!(cells.len() > 5, "several cells fit before the page filled");
    }

    #[test]
    fn page_one_edits_preserve_the_database_header_region() {
        // On page 1 the b-tree header is at offset 100; in-place edits must never
        // touch the first 100 bytes (the database header).
        let mut page = PageBuilder::new(PAGE, USABLE, 1, PageType::LeafTable).finish();
        for b in page.iter_mut().take(100) {
            *b = 0x5A; // stand-in database header
        }
        let header_before = page[..100].to_vec();
        for rowid in 1..=6i64 {
            let pos = PageView::new(&page, 1, USABLE).unwrap().cell_count() as usize;
            insert_cell(&mut page, 1, USABLE, pos, &table_cell(rowid, rowid)).unwrap();
        }
        delete_cell(&mut page, 1, USABLE, 2).unwrap();
        defragment(&mut page, 1, USABLE).unwrap();
        assert_eq!(&page[..100], &header_before[..], "database header region untouched");
        assert_eq!(rowids(&page, 1), vec![1, 2, 4, 5, 6]);
    }

    #[test]
    fn index_leaf_insert_and_delete_are_valid() {
        // The editors work on index-leaf pages too (no rowid varint in the cell).
        let mut page = PageBuilder::new(PAGE, USABLE, 4, PageType::LeafIndex).finish();
        let keys: Vec<Vec<u8>> = (1..=6i64)
            .map(|k| encode_record(&[Value::Integer(k), Value::Integer(k * 100)]))
            .collect();
        for (i, key) in keys.iter().enumerate() {
            let mut cell = Vec::new();
            encode_index_leaf_cell(key.len() as u64, key, None, &mut cell);
            insert_cell(&mut page, 4, USABLE, i, &cell).unwrap();
        }
        let view = PageView::new(&page, 4, USABLE).unwrap();
        assert_eq!(view.cell_count(), 6);
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(view.index_leaf_cell(i).unwrap().local_payload, key.as_slice());
        }
        drop(view);
        // Delete an index cell; the page stays readable and correctly ordered.
        delete_cell(&mut page, 4, USABLE, 2).unwrap();
        let view = PageView::new(&page, 4, USABLE).unwrap();
        assert_eq!(view.cell_count(), 5);
        let expected: Vec<&Vec<u8>> = keys.iter().enumerate().filter(|(i, _)| *i != 2).map(|(_, k)| k).collect();
        for (i, key) in expected.iter().enumerate() {
            assert_eq!(view.index_leaf_cell(i).unwrap().local_payload, key.as_slice());
        }
    }

    #[test]
    fn leaf_cell_len_matches_encoded_size_including_overflow() {
        // Non-overflow: length equals the encoded cell bytes.
        let mut page = empty_table_page();
        let cell = table_cell(7, 12345);
        insert_cell(&mut page, 3, USABLE, 0, &cell).unwrap();
        let off = PageView::new(&page, 3, USABLE).unwrap().cell_pointer(0);
        assert_eq!(leaf_cell_len(&page, 3, USABLE, off).unwrap(), cell.len());

        // Overflow: a table-leaf cell whose payload spilled (inline prefix + 4-byte
        // pointer). Build it directly and confirm the size includes the 4 bytes.
        let usable = 512usize;
        let payload_total = 2000u64;
        let split = payload_split(usable, payload_total, CellKind::TableLeaf);
        assert!(split.has_overflow);
        let inline = vec![0u8; split.local];
        let mut ov_cell = Vec::new();
        encode_table_leaf_cell(payload_total, 9, &inline, Some(123), &mut ov_cell);
        let mut opage = PageBuilder::new(usable, usable, 3, PageType::LeafTable).finish();
        insert_cell(&mut opage, 3, usable, 0, &ov_cell).unwrap();
        let ooff = PageView::new(&opage, 3, usable).unwrap().cell_pointer(0);
        assert_eq!(leaf_cell_len(&opage, 3, usable, ooff).unwrap(), ov_cell.len());
        assert_eq!(
            PageView::new(&opage, 3, usable).unwrap().table_leaf_cell(0).unwrap().overflow_page,
            Some(123)
        );
    }

    #[test]
    fn interior_page_is_rejected() {
        let page = PageBuilder::new(PAGE, USABLE, 5, PageType::InteriorTable).finish();
        let mut p = page.clone();
        assert!(insert_cell(&mut p, 5, USABLE, 0, &table_cell(1, 1)).is_err());
        assert!(delete_cell(&mut p, 5, USABLE, 0).is_err());
        assert!(defragment(&mut p, 5, USABLE).is_err());
        assert!(leaf_free_space(&page, 5, USABLE).is_err());
    }

    #[test]
    fn model_check_random_insert_delete_sequences() {
        // The strongest guard: apply a long, deterministic pseudo-random sequence of
        // inserts and deletes, mirror them in a byte-exact reference model, and after
        // EVERY op assert the page decodes to exactly the model and is spec-valid.
        // Exercises key-order splicing, freeblock creation, defragment-on-insert, and
        // delete-to-empty across many interleavings.
        let mut page = empty_table_page();
        let mut model: Vec<Entry> = Vec::new();
        // Small deterministic LCG (no external rand dependency).
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let (mut inserts, mut deletes, mut defrag_triggered) = (0u32, 0u32, false);

        for _ in 0..4000 {
            let do_insert = model.is_empty() || (next() % 3 != 0);
            if do_insert {
                let rowid = (next() % 500) as i64 + 1;
                if model.iter().any(|e| e.rowid == rowid) {
                    continue;
                }
                let size = (next() % 120) as usize + 1;
                let cell = table_cell_sized(rowid, size);
                if leaf_free_space(&page, 3, USABLE).unwrap().total < cell.len() + 2 {
                    continue; // would need a split; out of scope for the page editor
                }
                // Note when an insert must reclaim non-contiguous space (defrag path).
                let fs = leaf_free_space(&page, 3, USABLE).unwrap();
                if fs.gap < cell.len() + 2 {
                    defrag_triggered = true;
                }
                model_insert(&mut page, 3, &mut model, rowid, cell);
                inserts += 1;
            } else {
                let idx = (next() as usize) % model.len();
                delete_cell(&mut page, 3, USABLE, idx).unwrap();
                model.remove(idx);
                deletes += 1;
            }
            // The invariant, re-checked after EVERY op: the page decodes to exactly
            // the model and is spec-valid.
            assert_valid(&page, 3, &model);
        }
        // Confirm the sequence actually exercised all three paths, so the per-step
        // invariant check above was not vacuous.
        assert!(inserts > 100, "exercised many inserts (got {inserts})");
        assert!(deletes > 100, "exercised many deletes (got {deletes})");
        assert!(defrag_triggered, "at least one insert reclaimed non-contiguous free space");
    }
}
