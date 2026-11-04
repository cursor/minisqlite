//! The three built-in collating sequences and text comparison under them
//! (datatype3.html §7). Collation only ever affects *text* comparison; numbers
//! and blobs never consult a collation.

use std::cmp::Ordering;

/// A built-in collating sequence. `Binary` is the default (datatype3.html §7.1:
/// "If no collating function is explicitly defined, then the collating function
/// defaults to BINARY").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Collation {
    /// Byte-wise `memcmp`, regardless of text encoding.
    #[default]
    Binary,
    /// ASCII-only case-insensitive: the 26 upper-case ASCII letters fold to lower
    /// case; all other bytes (including non-ASCII) compare as-is.
    NoCase,
    /// Binary, but trailing space (0x20) characters are ignored.
    Rtrim,
}

/// Compare two text values under a collating sequence (datatype3.html §7).
///
/// * `Binary` compares the raw UTF-8 bytes (`memcmp`).
/// * `NoCase` folds ASCII `A`-`Z` to `a`-`z` and compares byte-wise; only ASCII is
///   folded (SQLite does not do full Unicode case folding). NOTE: SQLite's NOCASE
///   also treats an embedded NUL as a string terminator; that nuance is not
///   replicated here (we compare the full length), which only differs for the rare
///   text value containing an embedded `\0`.
/// * `Rtrim` strips trailing 0x20 bytes from both operands, then compares binary.
pub fn compare_text(a: &str, b: &str, c: Collation) -> Ordering {
    compare_text_bytes(a.as_bytes(), b.as_bytes(), c)
}

/// [`compare_text`] over raw text bytes, for callers that hold an encoded TEXT body and
/// must not allocate (the index-key comparator compares record bodies borrowed). The
/// rules are identical: each collating sequence only ever looks at bytes.
pub fn compare_text_bytes(a: &[u8], b: &[u8], c: Collation) -> Ordering {
    match c {
        Collation::Binary => a.cmp(b),
        Collation::NoCase => nocase_cmp(a, b),
        Collation::Rtrim => rstrip_spaces(a).cmp(rstrip_spaces(b)),
    }
}

#[inline]
fn ascii_lower(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b + 32 } else { b }
}

fn nocase_cmp(a: &[u8], b: &[u8]) -> Ordering {
    let n = a.len().min(b.len());
    for i in 0..n {
        match ascii_lower(a[i]).cmp(&ascii_lower(b[i])) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
    }
    a.len().cmp(&b.len())
}

fn rstrip_spaces(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., b' '] = s {
        s = rest;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering::*;

    #[test]
    fn binary_is_bytewise() {
        assert_eq!(compare_text("abc", "abc", Collation::Binary), Equal);
        // Upper-case ASCII sorts before lower-case ('A'=65 < 'a'=97).
        assert_eq!(compare_text("ABC", "abc", Collation::Binary), Less);
        assert_eq!(compare_text("abc", "abd", Collation::Binary), Less);
        assert_eq!(compare_text("ab", "abc", Collation::Binary), Less);
    }

    #[test]
    fn nocase_folds_ascii_only() {
        assert_eq!(compare_text("abc", "ABC", Collation::NoCase), Equal);
        assert_eq!(compare_text("Abc", "aBC", Collation::NoCase), Equal);
        assert_eq!(compare_text("abc", "abd", Collation::NoCase), Less);
        // Non-ASCII bytes are not folded.
        assert_eq!(compare_text("é", "É", Collation::NoCase), compare_text("é", "É", Collation::Binary));
        assert_ne!(compare_text("é", "É", Collation::NoCase), Equal);
    }

    #[test]
    fn nocase_length_tiebreak() {
        // Equal folded prefix, unequal length: the shorter string sorts first (the
        // length tiebreak after the case-insensitive prefix compares equal).
        assert_eq!(compare_text("abc", "ABCD", Collation::NoCase), Less);
        assert_eq!(compare_text("ABCD", "abc", Collation::NoCase), Greater);
    }

    #[test]
    fn rtrim_ignores_trailing_spaces() {
        assert_eq!(compare_text("abc", "abc  ", Collation::Rtrim), Equal);
        assert_eq!(compare_text("abc ", "abc", Collation::Rtrim), Equal);
        // Only trailing spaces are ignored; leading/interior are significant.
        assert_eq!(compare_text(" abc", "abc", Collation::Rtrim), Less);
        assert_ne!(compare_text("a bc", "abc", Collation::Rtrim), Equal);
        // Only 0x20 is trimmed, not tabs.
        assert_ne!(compare_text("abc\t", "abc", Collation::Rtrim), Equal);
    }

    // datatype3.html §7.2 worked example: ORDER BY c (RTRIM) over
    // {'abc  ', 'abc', 'abc ', 'ABC'} sorts 'ABC' first, then the equal 'abc's.
    #[test]
    fn rtrim_sort_example() {
        let mut rows = ["abc  ", "abc", "abc ", "ABC"];
        rows.sort_by(|x, y| compare_text(x, y, Collation::Rtrim));
        assert_eq!(rows[0], "ABC"); // 'A' < 'a'
        // The remaining three all trim to "abc" and compare equal.
        for r in &rows[1..] {
            assert_eq!(compare_text(r, "abc", Collation::Rtrim), Equal);
        }
    }

    #[test]
    fn default_is_binary() {
        assert_eq!(Collation::default(), Collation::Binary);
    }
}
