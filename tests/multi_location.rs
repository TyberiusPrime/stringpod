//! Tests for [`stringpod::DualStringPodMultiLocation`] — the ragged alias dual
//! pod used as a tag column's frozen seq+qual snapshot.
//!
//! The contract: rows alias sub-ranges of a source read pod (multiple per row
//! for multi-hit reads, none for no-hit reads); the bytes stay frozen no matter
//! how the source is later edited (overlay cut, rebuild, in-place reverse); and
//! row-axis ops keep the snapshot aligned with the reads.

use bstr::BStr;
use std::borrow::Cow;
use stringpod::{DualStringPod, DualStringPodBuilder, StringPod, StringPodBuilder};

fn read_pod(entries: &[(&str, &str)]) -> DualStringPod {
    let mut bld = DualStringPodBuilder::with_capacity(0, entries.len());
    for (s, q) in entries {
        bld.push(s.as_bytes(), q.as_bytes());
    }
    bld.finish()
}

#[test]
fn aliases_multi_hit_and_no_hit_rows() {
    // read 0 has two hits, read 1 has none, read 2 has one.
    let source = read_pod(&[
        ("ACGTACGTACGT", "IIIIFFFFJJJJ"),
        ("TTTTTTTT", "########"),
        ("GGGGCCCC", "11112222"),
    ]);

    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(0, 4), (8, 4)]); // "ACGT"/"IIII" and "ACGT"/"JJJJ"
        b.push_row(&[]); // no hit
        b.push_row(&[(4, 4)]); // "CCCC"/"2222"
        b.finish()
    };

    assert_eq!(snap.row_count(), 3);
    assert_eq!(snap.loc_count_in(0), 2);
    assert!(snap.row_is_empty(1));
    assert_eq!(snap.loc_count_in(2), 1);

    assert_eq!(snap.seq(0, 0), BStr::new("ACGT"));
    assert_eq!(snap.qual(0, 0), BStr::new("IIII"));
    assert_eq!(snap.seq(0, 1), BStr::new("ACGT"));
    assert_eq!(snap.qual(0, 1), BStr::new("JJJJ"));
    assert_eq!(snap.pair(2, 0), (BStr::new("CCCC"), BStr::new("2222")));

    // joined: borrow for single, allocate (with separator) for multi.
    assert_eq!(&*snap.joined_seq(0, Some(b"-")), BStr::new("ACGT-ACGT"));
    assert_eq!(&*snap.joined_qual(0, None), BStr::new("IIIIJJJJ"));
    assert_eq!(&*snap.joined_seq(2, None), BStr::new("CCCC"));

    let row0: Vec<_> = snap.iter_row(0).collect();
    assert_eq!(row0.len(), 2);
}

#[test]
fn loc_region_returns_captured_read_relative_coords() {
    // `loc_region` must hand back the *read-relative* (start, len) — the offset
    // within the source entry's visible bytes — regardless of where that entry
    // sits in the shared buffer. That coordinate is what the read pod's edit log
    // lifts; it is independent of the base offset used to slice the frozen bytes.
    let source = read_pod(&[("ACGTACGTACGT", "IIIIFFFFJJJJ"), ("TTTTGGGG", "11112222")]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(0, 4), (8, 4)]);
        b.push_row(&[(4, 4)]); // entry 1 starts mid-buffer; offset is still 4
        b.finish()
    };

    assert_eq!(snap.loc_region(0, 0), (0, 4));
    assert_eq!(snap.loc_region(0, 1), (8, 4));
    assert_eq!(snap.loc_region(1, 0), (4, 4));

    let row1: Vec<_> = snap.row_regions(1).collect();
    assert_eq!(row1, vec![(4, 4)]);

    // and the coordinate still slices the right frozen bytes.
    assert_eq!(snap.seq(1, 0), BStr::new("GGGG"));
    assert_eq!(snap.qual(1, 0), BStr::new("2222"));
}

#[test]
fn iter_seq_qual_pair_yield_per_row_captured_values() {
    let source = read_pod(&[
        ("ACGTACGTACGT", "IIIIFFFFJJJJ"), // two hits
        ("TTTTTTTT", "########"),         // no hit
        ("GGGGCCCC", "11112222"),         // one hit
    ]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(0, 4), (8, 4)]); // "ACGT"+"ACGT" / "IIII"+"JJJJ"
        b.push_row(&[]);
        b.push_row(&[(4, 4)]); // "CCCC" / "2222"
        b.finish()
    };

    // no-hit row 1 → `None`; hit rows → `Some(joined)`.
    let seqs: Vec<Cow<BStr>> = snap.iter_seq().collect();
    assert_eq!(
        seqs,
        vec![
            BStr::new("ACGTACGT"),
            BStr::new(""),
            BStr::new("CCCC")
        ]
    );

    let quals: Vec<Cow<BStr>> = snap.iter_qual().collect();
    assert_eq!(
        quals,
        vec![
            BStr::new("IIIIJJJJ"),
            BStr::new(""),
            BStr::new("2222")
        ]
    );

    // `iter()` is ExactSize over rows; `&snap` (IntoIterator) yields one
    // `Option<(seq, qual)>` per row — `None` for the no-hit row.
    assert_eq!(snap.iter().len(), 3);
    let mut pairs: Vec<(Cow<BStr>, Cow<BStr>)> = Vec::new();
    dbg!(&snap);
    for pair in &snap {
        pairs.push( pair)
    }
    assert_eq!(pairs.len(), 3);
    assert_eq!(
        pairs[0],
        (Cow::Borrowed(BStr::new("ACGTACGT")), Cow::Borrowed(BStr::new("IIIIJJJJ")))
    );
    assert_eq!(pairs[1], 
        (Cow::Borrowed(BStr::new("")), Cow::Borrowed(BStr::new(""))));

    assert_eq!(
        pairs[2],
        ((Cow::Borrowed(BStr::new("CCCC")), Cow::Borrowed(BStr::new("2222"))))
    );
}

#[test]
fn snapshot_is_frozen_across_source_edits() {
    let mut source = read_pod(&[("ACGTACGT", "IIIIFFFF")]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(2, 4)]); // "GTAC" / "IIFF"
        b.finish()
    };
    assert_eq!(snap.pair(0, 0), (BStr::new("GTAC"), BStr::new("IIFF")));

    // Overlay cut (metadata only, same Arc).
    source.cut_start(3, None);
    assert_eq!(snap.pair(0, 0), (BStr::new("GTAC"), BStr::new("IIFF")));

    // Rebuild (prefix → fresh Arc; snapshot keeps the old one).
    source = source.prefix(b"XX", b"##");
    assert_eq!(snap.pair(0, 0), (BStr::new("GTAC"), BStr::new("IIFF")));

    // In-place reverse (COW-clones because the snapshot shares the Arc).
    let _reversed = source.reverse(None);
    assert_eq!(snap.pair(0, 0), (BStr::new("GTAC"), BStr::new("IIFF")));
}

#[test]
fn row_axis_keeps_snapshot_aligned() {
    let source = read_pod(&[
        ("AAAA", "1111"),
        ("BBBB", "2222"),
        ("CCCC", "3333"),
        ("DDDD", "4444"),
    ]);
    let mut snap = {
        let mut b = source.multi_location_alias_builder();
        for row in 0..4 {
            b.push_row(&[(0, 4)]);
            let _ = row;
        }
        b.finish()
    };
    assert_eq!(snap.row_count(), 4);

    snap.drain(1..3); // drop rows 1,2
    assert_eq!(snap.row_count(), 2);
    assert_eq!(snap.seq(0, 0), BStr::new("AAAA"));
    assert_eq!(snap.seq(1, 0), BStr::new("DDDD"));

    snap.retain_by_bools(&[false, true]); // keep only "DDDD"
    assert_eq!(snap.row_count(), 1);
    assert_eq!(snap.seq(0, 0), BStr::new("DDDD"));

    snap.truncate(0);
    assert!(snap.is_empty());
}

#[test]
fn aliases_correctly_through_divergent_first_byte() {
    // Build a source whose seq buffer starts 4 bytes after qual's (a dropped
    // leading entry), so seq_first_byte != qual_first_byte. The snapshot must
    // still read both buffers at the right offsets.
    fn fixed(stride: usize, entries: &[&str]) -> StringPod {
        let mut b = StringPodBuilder::with_capacity(stride, entries.len());
        for e in entries {
            b.push(e.as_bytes());
        }
        b.finish()
    }
    let mut seq = fixed(4, &["XXXX", "ACGT", "TTTT"]);
    seq.pop_front(1); // seq front_byte now 4; entries "ACGT","TTTT"
    let qual = fixed(4, &["IIII", "FFFF"]); // front_byte 0
    let source = DualStringPod::try_from_columns(seq, qual).expect("valid columns");
    assert_eq!(source.seq(0), BStr::new("ACGT"));
    assert_eq!(source.qual(0), BStr::new("IIII"));

    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(1, 2)]); // "CG" / "II"
        b.push_row(&[(0, 4)]); // "TTTT" / "FFFF"
        b.finish()
    };
    assert_eq!(snap.pair(0, 0), (BStr::new("CG"), BStr::new("II")));
    assert_eq!(snap.pair(1, 0), (BStr::new("TTTT"), BStr::new("FFFF")));
}

#[test]
fn make_exclusive_detaches_from_source() {
    let source = read_pod(&[("ACGT", "IIII")]);
    let mut snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(0, 4)]);
        b.finish()
    };
    snap.make_exclusive();
    drop(source);
    assert_eq!(snap.pair(0, 0), (BStr::new("ACGT"), BStr::new("IIII")));
}

#[test]
fn iter_row_lengths_sums_locations_with_and_without_sep() {
    // row 0: two hits of len 4 each → 8 without sep, 9 with sep
    // row 1: no hit                 → 0 in both cases
    // row 2: one hit of len 4       → 4 in both cases
    let source = read_pod(&[
        ("ACGTACGTACGT", "IIIIFFFFJJJJ"),
        ("TTTTTTTT", "########"),
        ("GGGGCCCC", "11112222"),
    ]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row(&[(0, 4), (8, 4)]);
        b.push_row(&[]);
        b.push_row(&[(4, 4)]);
        b.finish()
    };

    let no_sep: Vec<usize> = snap.iter_row_lengths(None).collect();
    assert_eq!(no_sep, vec![8, 0, 4]);

    let with_sep: Vec<usize> = snap.iter_row_lengths(Some(b'-')).collect();
    assert_eq!(with_sep, vec![9, 0, 4]);
}

#[test]
fn push_row_from_ranges_matches_tuple_form() {
    // The range front door must produce the exact same snapshot as the
    // `(start, len)` form: `0..4` ≡ `(0, 4)`, `8..12` ≡ `(8, 4)`.
    let source = read_pod(&[
        ("ACGTACGTACGT", "IIIIFFFFJJJJ"),
        ("TTTTTTTT", "########"),
        ("GGGGCCCC", "11112222"),
    ]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_row_from_ranges(&[0..4, 8..12]);
        b.push_row_from_ranges(&[]); // no hit
        b.push_row_from_ranges(&[4..8]);
        b.finish()
    };

    assert_eq!(snap.loc_region(0, 0), (0, 4));
    assert_eq!(snap.loc_region(0, 1), (8, 4));
    assert!(snap.row_is_empty(1));
    assert_eq!(snap.loc_region(2, 0), (4, 4));
    assert_eq!(snap.pair(0, 1), (BStr::new("ACGT"), BStr::new("JJJJ")));
    assert_eq!(snap.pair(2, 0), (BStr::new("CCCC"), BStr::new("2222")));
}

#[test]
#[should_panic(expected = "reversed range")]
fn push_row_from_ranges_rejects_reversed_range() {
    let source = read_pod(&[("ACGT", "IIII")]);
    let mut b = source.multi_location_alias_builder();
    b.push_row_from_ranges(&[3..1]);
}

#[test]
#[should_panic(expected = "exceeds entry length")]
fn push_row_rejects_out_of_bounds_location() {
    let source = read_pod(&[("ACGT", "IIII")]);
    let mut b = source.multi_location_alias_builder();
    b.push_row(&[(2, 5)]); // 2+5 > 4
}

#[test]
#[should_panic(expected = "already consumed")]
fn push_row_rejects_too_many_rows() {
    let source = read_pod(&[("ACGT", "IIII")]);
    let mut b = source.multi_location_alias_builder();
    b.push_row(&[(0, 4)]);
    b.push_row(&[(0, 4)]); // only one source entry
}
