//! Shared value vocabulary and semantics for the engine — the five SQLite storage
//! classes, the row/result types, the error type, and the value-semantics layer
//! (affinity, numeric coercion, collation, comparison, CAST) that every other
//! crate builds on. A small, std-only crate on purpose: it is depended on
//! everywhere, so it must stay cheap to compile and free of the engine.
//!
//! This root is a thin re-export hub — the real code lives in the submodule files
//! (`value`, `error`, `affinity`, `numeric`, `collation`, `compare`, `cast`) so a
//! feature lands in its own file rather than contending on `lib.rs`.

mod affinity;
mod cast;
mod collation;
mod compare;
mod dbindex;
mod error;
mod namespace_meta;
mod numeric;
mod value;

pub use dbindex::DbIndex;
pub use namespace_meta::NamespaceMeta;

pub use value::{QueryResult, Row, Value};

pub use error::{code, ConstraintKind, Error, Result};

pub use affinity::{affinity_of_declared_type, apply_affinity, Affinity};

pub use numeric::{
    integer_to_text, looks_like_integer, looks_like_real, numeric_prefix, parse_int_prefix,
    parse_real_prefix, real_to_int_if_exact, real_to_int_trunc, real_to_text, text_to_numeric,
    NumericPrefix,
};

pub use collation::{compare_text, compare_text_bytes, Collation};

pub use compare::{compare_for_eq, compare_values, extremum_wins, storage_class_rank};

pub use cast::cast_to;
