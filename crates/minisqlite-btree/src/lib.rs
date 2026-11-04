//! The **table** b-tree: create, rowid insert with page splitting, and a
//! forward/seek read cursor. It turns the pager + on-disk codec into a usable
//! rowid-keyed store — the storage half of a rowid table (fileformat2 §1.6 B-tree
//! pages, §2.3 rowid tables). It builds ON `minisqlite-fileformat` (cell/record
//! codec, `PageView`/`PageBuilder`) and `minisqlite-pager` (page access +
//! copy-on-write transactions); it does not reimplement varints, records, or cells.
//!
//! Scope is the table and index b-trees, including rowid-table `DELETE` (which
//! reclaims the pages and overflow chains it frees, rebalances an interior that a
//! removal pushes below the on-disk minimum so every non-root interior keeps `K >= 2`,
//! and reuses freed pages on the next insert). A table row or index key past the inline
//! threshold spills its tail onto a chain of overflow pages (written on insert,
//! reassembled on read).
//!
//! This root is a thin re-export hub. The real code lives in submodules so a
//! feature lands in its own file rather than contending on `lib.rs`:
//!
//! - [`tree`] — `init_database`, `create_table_btree`, and the header-preserving
//!   page write.
//! - [`build`] — building and splitting leaf / interior pages from cell lists.
//! - [`nav`] — pure navigation over a `PageView` (child choice, binary search).
//! - [`insert`] — `table_insert`: descent, in-leaf insert/REPLACE, split
//!   propagation, and balance-deeper (the root page id never changes).
//! - [`delete`] — `table_delete`: descent, in-leaf removal, separator retargeting,
//!   empty-subtree collapse with root height-decrease (the root page id never
//!   changes), rebalance-on-underflow (rotate / merge to keep non-root interiors at
//!   `K >= 2`), and reclaiming the freed pages and overflow chains to the freelist.
//! - [`cursor`] — [`TableCursor`], a borrowing forward/seek reader.
//!
//! The index b-tree mirrors that structure for key-record (not rowid) pages:
//!
//! - [`index_key`] — index-key record comparison (the ordering seeks rely on) under
//!   a per-column [`KeyOrder`] (sort order and collation).
//! - [`index_nav`] — pure navigation over an index `PageView`.
//! - [`index_build`] — building and splitting index leaf / interior pages.
//! - [`index`] — `create_index_btree` and `index_insert` (descent, splice, split
//!   propagation, balance-deeper; the root page id never changes).
//! - [`index_delete`] — `index_delete`: classic-B key removal (leaf cell or the
//!   predecessor-swap for an interior divider) followed by a bottom-up underflow
//!   rebalance (merge / redistribute, keeping every non-root interior K>=2), root
//!   collapse, and reclaiming freed pages / overflow chains.
//! - [`index_balance`] — pure planners that pack a gathered leaf/interior sequence
//!   into the fewest valid pages for that rebalance.
//! - [`index_cursor`] — [`IndexCursor`], a borrowing forward/reverse/seek reader.

mod build;
mod count;
mod cursor;
mod delete;
mod insert;
mod nav;
mod overflow_io;
mod tree;

mod index;
mod index_balance;
mod index_build;
mod index_cursor;
mod index_delete;
mod index_key;
mod index_nav;

pub use count::{index_entry_count, table_entry_count};
pub use cursor::TableCursor;
pub use delete::table_delete;
pub use insert::table_insert;
pub use tree::{create_table_btree, init_database};

pub use index::{create_index_btree, index_insert, index_insert_with_order};
pub use index_cursor::IndexCursor;
pub use index_delete::{index_delete, index_delete_with_order};
pub use index_key::{KeyField, KeyOrder};
