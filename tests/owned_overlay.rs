//! Tests for the **owned overlay** on [`stringpod::DualStringPodMultiLocation`]:
//! rows whose content diverged from any single source slice (e.g. a regex
//! replacement) and now live in the pod's shared `owned` arena, carrying their
//! own seq+qual while still anchoring back to a read-relative span. Alias rows
//! (zero-copy views of the source) and owned rows must coexist in one column.

use bstr::{BStr, BString};
use stringpod::{DualStringPod, DualStringPodBuilder};

fn read_pod(entries: &[(&str, &str)]) -> DualStringPod {
    let mut bld = DualStringPodBuilder::with_capacity(0, entries.len());
    for (s, q) in entries {
        bld.push(s.as_bytes(), q.as_bytes());
    }
    bld.finish()
}

#[test]
fn owned_and_alias_rows_coexist() {
    // read 0: alias window; read 1: owned (conjured/divergent); read 2: no hit.
    let source = read_pod(&[
        ("AAAATTTT", "IIIIJJJJ"),
        ("NNNNNNNN", "########"),
        ("GGGGCCCC", "11112222"),
    ]);

    let snap = {
        let mut b = source.multi_location_alias_builder();
        // read 0 → live alias of the first 4 bases ("AAAA"/"IIII").
        b.push_row_from_ranges(&[0..4]);
        // read 1 → owned content "AGTC" with synthesized 'B' quality, anchored at
        // the empty span at the end of the read (grow-from-nothing case).
        b.push_owned_row(&[(4, 4)], b"AGTC", b"BBBB");
        // read 2 → no hit.
        b.push_row(&[]);
        b.finish()
    };

    assert_eq!(snap.row_count(), 3);

    // Alias row reads straight out of the source buffers.
    assert_eq!(snap.loc_count_in(0), 1);
    assert!(!snap.row_is_empty(0));
    assert_eq!(snap.joined_seq(0, None), BStr::new("AAAA"));
    assert_eq!(snap.joined_qual(0, None), BStr::new("IIII"));
    assert_eq!(snap.loc_region(0, 0), (0, 4));

    // Owned row reads out of the arena; its region is the anchor span, not the
    // content length (0 here — the content stands in for an empty span).
    assert_eq!(snap.loc_count_in(1), 1);
    assert!(!snap.row_is_empty(1));
    assert_eq!(snap.joined_seq(1, None), BStr::new("AGTC"));
    assert_eq!(snap.joined_qual(1, None), BStr::new("BBBB"));
    assert_eq!(snap.loc_region(1, 0), (4, 4));
    assert_eq!(snap.row_length(1, None), 4);
    assert_eq!(snap.joined_seq(1, None).as_ref(), BStr::new("AGTC"));
    assert_eq!(snap.joined_qual(1, None).as_ref(), BStr::new("BBBB"));
    // covered_positions reflects the (empty) anchor, not the content.
    assert_eq!(snap.covered_positions(0).collect::<Vec<_>>(), vec![0,1,2,3]);
    assert_eq!(snap.covered_positions(1).collect::<Vec<_>>(), vec![4,5,6,7]);

    // No-hit row stays empty.
    assert!(snap.row_is_empty(2));
    assert_eq!(snap.loc_count_in(2), 0);
}

#[test]
fn owned_row_with_real_anchor_and_doubled_content() {
    // The "$0$0" shape: owned content is the matched span's bytes doubled, with a
    // non-empty anchor (start 2, len 4) that write-back/liftover would target.
    let source = read_pod(&[("CTGTACGTAA", "0123456789")]);
    let snap = {
        let mut b = source.multi_location_alias_builder();
        // matched "GTAC" at 2..6, doubled → "GTACGTAC", quality doubled too.
        b.push_owned_row(&[(2, 4)], b"GTACGTAC", b"23452345");
        b.finish()
    };

    assert_eq!(snap.joined_seq(0, None), BStr::new("GTACGTAC"));
    assert_eq!(snap.joined_qual(0, None), BStr::new("23452345"));
    assert_eq!(snap.loc_region(0, 0), (2, 4));
    assert_eq!(snap.covered_positions(0).collect::<Vec<_>>(), vec![2, 3, 4, 5]);
    assert_eq!(snap.row_length(0, None), 4);
}

#[test]
fn owned_rows_share_one_arena_and_survive_make_exclusive() {
    let source = read_pod(&[("AAAA", "IIII"), ("CCCC", "JJJJ")]);
    let mut snap = {
        let mut b = source.multi_location_alias_builder();
        b.push_owned_row(&[(0, 4)], b"XX", b"BB");
        b.push_owned_row(&[(0, 4)], b"YYY", b"BBB");
        b.finish()
    };
    // make_exclusive only detaches the aliased source buffers; owned content is
    // already exclusive and must read back identically even after the source is
    // dropped.
    snap.make_exclusive();
    drop(source);
    assert_eq!(snap.joined_seq(0, None), BStr::new("XX"));
    assert_eq!(snap.joined_qual(0, None), BStr::new("BB"));

    let joined: Vec<BString> = (0..snap.row_count())
        .map(|r| snap.joined_seq(r, None).into_owned())
        .collect();
    assert_eq!(joined, vec![BString::from("XX"), BString::from("YYY")]);
}
