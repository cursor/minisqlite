//! Column type affinity: deriving a column's affinity from its declared type, and
//! coercing a value toward a column's affinity on the way into storage.
//!
//! This is datatype3.html §3. Affinity is the #1 source of subtle disagreements
//! with real sqlite, so the rules are transcribed literally and exercised against
//! the spec's own worked examples in the tests below.

use crate::numeric::{integer_to_text, numeric_affinity_text, real_to_int_if_exact, real_to_text};
use crate::value::Value;

/// A column's type affinity (datatype3.html §3). `Blob` is what the docs now call
/// the "BLOB" affinity and historically called "NONE": no preferred storage class,
/// no coercion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affinity {
    Blob,
    Text,
    Numeric,
    Integer,
    Real,
}

/// Derive a column's affinity from its declared type (datatype3.html §3.1). The
/// rules are applied *in order* against the upper-cased declared type, and the
/// order matters: "CHARINT" and "FLOATING POINT" both hit rule 1 (they contain
/// "INT") before any later rule. `None`/empty declared type is BLOB (rule 3).
pub fn affinity_of_declared_type(decl: Option<&str>) -> Affinity {
    let decl = match decl {
        Some(d) => d,
        None => return Affinity::Blob,
    };
    // Case-insensitive substring test without allocating an upper-cased copy: the row
    // decoder calls this per stored INTEGER value to apply REAL affinity on read.
    let has = |pat: &str| {
        decl.as_bytes().windows(pat.len()).any(|w| w.eq_ignore_ascii_case(pat.as_bytes()))
    };
    if has("INT") {
        Affinity::Integer
    } else if has("CHAR") || has("CLOB") || has("TEXT") {
        Affinity::Text
    } else if has("BLOB") || decl.trim().is_empty() {
        Affinity::Blob
    } else if has("REAL") || has("FLOA") || has("DOUB") {
        Affinity::Real
    } else {
        Affinity::Numeric
    }
}

/// Coerce a value toward a column's affinity as it is stored (datatype3.html
/// §3.2-3.4). NULL and BLOB storage-class values are never coerced by affinity;
/// only numbers and text move. Coercion only happens when it is lossless (an
/// unconvertible TEXT stays TEXT) — that is the difference from a CAST, which
/// always converts.
pub fn apply_affinity(v: Value, a: Affinity) -> Value {
    // NULL and BLOB are never touched by affinity, whatever the column.
    if matches!(v, Value::Null | Value::Blob(_)) {
        return v;
    }
    match a {
        // BLOB (NONE) affinity: no preference, no coercion.
        Affinity::Blob => v,

        // TEXT affinity: numbers become their text form; text stays text.
        Affinity::Text => match v {
            Value::Integer(i) => Value::Text(integer_to_text(i)),
            Value::Real(r) => Value::Text(real_to_text(r)),
            other => other,
        },

        // NUMERIC and INTEGER affinity behave identically for storage (they differ
        // only inside CAST): convert numeric-looking text to a number, and demote a
        // real that is exactly integral to an integer.
        Affinity::Numeric | Affinity::Integer => match v {
            Value::Text(s) => numeric_affinity_text(&s).unwrap_or(Value::Text(s)),
            Value::Real(r) => real_to_int_if_exact(r).map(Value::Integer).unwrap_or(Value::Real(r)),
            other => other, // Integer stays Integer
        },

        // REAL affinity: like NUMERIC, but then force any integer to floating point.
        Affinity::Real => match v {
            Value::Integer(i) => Value::Real(i as f64),
            Value::Real(r) => Value::Real(r),
            Value::Text(s) => match numeric_affinity_text(&s) {
                Some(Value::Integer(i)) => Value::Real(i as f64),
                Some(other) => other, // already Real
                None => Value::Text(s),
            },
            other => other,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinity_derivation_examples() {
        use Affinity::*;
        // datatype3.html §3.1.1 example table (a representative subset).
        assert_eq!(affinity_of_declared_type(Some("INT")), Integer);
        assert_eq!(affinity_of_declared_type(Some("INTEGER")), Integer);
        assert_eq!(affinity_of_declared_type(Some("BIGINT")), Integer);
        assert_eq!(affinity_of_declared_type(Some("VARCHAR(10)")), Text);
        assert_eq!(affinity_of_declared_type(Some("CHARACTER(20)")), Text);
        assert_eq!(affinity_of_declared_type(Some("TEXT")), Text);
        assert_eq!(affinity_of_declared_type(Some("CLOB")), Text);
        assert_eq!(affinity_of_declared_type(Some("BLOB")), Blob);
        assert_eq!(affinity_of_declared_type(None), Blob);
        assert_eq!(affinity_of_declared_type(Some("")), Blob);
        assert_eq!(affinity_of_declared_type(Some("REAL")), Real);
        assert_eq!(affinity_of_declared_type(Some("DOUBLE PRECISION")), Real);
        assert_eq!(affinity_of_declared_type(Some("FLOAT")), Real);
        assert_eq!(affinity_of_declared_type(Some("NUMERIC")), Numeric);
        assert_eq!(affinity_of_declared_type(Some("DECIMAL(10,5)")), Numeric);
        assert_eq!(affinity_of_declared_type(Some("BOOLEAN")), Numeric);
        assert_eq!(affinity_of_declared_type(Some("DATETIME")), Numeric);
    }

    #[test]
    fn affinity_rule_order_and_case() {
        use Affinity::*;
        // "FLOATING POINT" -> INTEGER (contains "INT" via POINT), per the doc note.
        assert_eq!(affinity_of_declared_type(Some("FLOATING POINT")), Integer);
        // "STRING" -> NUMERIC (not TEXT), per the doc note.
        assert_eq!(affinity_of_declared_type(Some("STRING")), Numeric);
        // "CHARINT" -> INTEGER (rule 1 precedence over rule 2).
        assert_eq!(affinity_of_declared_type(Some("CHARINT")), Integer);
        // Case-insensitive.
        assert_eq!(affinity_of_declared_type(Some("varchar")), Text);
        assert_eq!(affinity_of_declared_type(Some("real")), Real);
    }

    // datatype3.html §3.4: inserting the same literals into columns of each
    // affinity yields these stored storage classes.
    fn stored(v: Value, decl: &str) -> &'static str {
        apply_affinity(v, affinity_of_declared_type(Some(decl))).type_name()
    }

    #[test]
    fn column_affinity_behavior_text_literal() {
        // INSERT '500.0' -> TEXT, INTEGER, INTEGER, REAL, TEXT
        assert_eq!(stored(Value::Text("500.0".into()), "TEXT"), "text");
        assert_eq!(stored(Value::Text("500.0".into()), "NUMERIC"), "integer");
        assert_eq!(stored(Value::Text("500.0".into()), "INTEGER"), "integer");
        assert_eq!(stored(Value::Text("500.0".into()), "REAL"), "real");
        assert_eq!(stored(Value::Text("500.0".into()), "BLOB"), "text");
    }

    #[test]
    fn column_affinity_behavior_real_literal() {
        // INSERT 500.0 -> TEXT, INTEGER, INTEGER, REAL, REAL
        assert_eq!(stored(Value::Real(500.0), "TEXT"), "text");
        assert_eq!(stored(Value::Real(500.0), "NUMERIC"), "integer");
        assert_eq!(stored(Value::Real(500.0), "INTEGER"), "integer");
        assert_eq!(stored(Value::Real(500.0), "REAL"), "real");
        assert_eq!(stored(Value::Real(500.0), "BLOB"), "real");
    }

    #[test]
    fn column_affinity_behavior_integer_literal() {
        // INSERT 500 -> TEXT, INTEGER, INTEGER, REAL, INTEGER
        assert_eq!(stored(Value::Integer(500), "TEXT"), "text");
        assert_eq!(stored(Value::Integer(500), "NUMERIC"), "integer");
        assert_eq!(stored(Value::Integer(500), "INTEGER"), "integer");
        assert_eq!(stored(Value::Integer(500), "REAL"), "real");
        assert_eq!(stored(Value::Integer(500), "BLOB"), "integer");
    }

    #[test]
    fn blobs_and_nulls_are_never_coerced() {
        for decl in ["TEXT", "NUMERIC", "INTEGER", "REAL", "BLOB"] {
            assert_eq!(stored(Value::Blob(vec![5, 0]), decl), "blob");
            assert_eq!(stored(Value::Null, decl), "null");
        }
    }

    #[test]
    fn text_affinity_renders_numbers() {
        assert!(matches!(apply_affinity(Value::Integer(42), Affinity::Text), Value::Text(s) if s == "42"));
        assert!(matches!(apply_affinity(Value::Real(1.5), Affinity::Text), Value::Text(s) if s == "1.5"));
    }

    #[test]
    fn numeric_affinity_leaves_unconvertible_text() {
        // Non-numeric text is not coerced (lossless-only rule).
        assert!(matches!(
            apply_affinity(Value::Text("hello".into()), Affinity::Numeric),
            Value::Text(s) if s == "hello"
        ));
        // Hex is not a well-formed literal for runtime conversion -> stays text.
        assert!(matches!(
            apply_affinity(Value::Text("0x10".into()), Affinity::Numeric),
            Value::Text(s) if s == "0x10"
        ));
    }

    #[test]
    fn real_affinity_forces_integers_to_real() {
        assert!(matches!(apply_affinity(Value::Integer(7), Affinity::Real), Value::Real(r) if r == 7.0));
        assert!(matches!(apply_affinity(Value::Text("7".into()), Affinity::Real), Value::Real(r) if r == 7.0));
        assert!(matches!(apply_affinity(Value::Text("7.5".into()), Affinity::Real), Value::Real(r) if r == 7.5));
    }

    #[test]
    fn real_affinity_leaves_non_numeric_text() {
        // REAL affinity, like NUMERIC, only coerces convertible text; junk stays TEXT.
        assert!(matches!(
            apply_affinity(Value::Text("hello".into()), Affinity::Real),
            Value::Text(s) if s == "hello"
        ));
    }

    #[test]
    fn numeric_affinity_real_to_int_range_guard() {
        // A large integral real still inside i64 range demotes to INTEGER exactly...
        let two_pow_53 = 9_007_199_254_740_992.0_f64; // 2^53, exactly representable
        assert!(matches!(
            apply_affinity(Value::Real(two_pow_53), Affinity::Numeric),
            Value::Integer(9_007_199_254_740_992)
        ));
        // ...but 2^63 is one past i64::MAX, so the guard must keep it REAL rather than
        // mis-demoting it to i64::MAX.
        let two_pow_63 = 9_223_372_036_854_775_808.0_f64;
        assert!(matches!(
            apply_affinity(Value::Real(two_pow_63), Affinity::Numeric),
            Value::Real(r) if r == two_pow_63
        ));
    }
}
