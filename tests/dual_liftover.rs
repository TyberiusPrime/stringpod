//! End-to-end liftover through the real `DualStringPod` mutators.
//!
//! The pods now record every coordinate edit into a `ColumnEdits` as a side
//! effect of their normal mutators (via the `Lifted` trait). A tag captures
//! `generation(row)` and its logical `[start, len)` when it is born, freezes its
//! bytes with the alias builder, and later lifts through `ops_since(born, row)`.
//! These tests check that the lifted span in the *live* pod still names the
//! frozen bytes — i.e. that the wiring records the right edits — across the
//! uniform pipeline, conditional/per-row edits, row drops, clipping, and a
//! single-read write-back.

use bstr::BStr;
use stringpod::{DualStringPod, DualStringPodBuilder, Lifted, RegionLift};

fn lift(
    pod: &DualStringPod,
    born: usize,
    row: usize,
    start: usize,
    len: usize,
    orig: usize,
) -> RegionLift {
    pod.ops_since(born, row)
        .expect("row & generation in range")
        .map_region(start, len, orig)
        .expect("region in range")
}

#[test]
fn lifts_a_tag_through_a_uniform_pipeline() {
    let mut bld = DualStringPodBuilder::with_capacity(0, 1);
    bld.push(b"ACGTACGTACGT", b"IIIIFFFFJJJJ"); // len 12
    let mut pod = bld.finish();

    // Tag born now: region [4, 8) of read 0 ("ACGT" / "FFFF").
    let born = pod.generation(0).expect("row 0 exists");
    assert_eq!(born, 0);
    let snapshot = {
        let mut ab = pod.alias_builder();
        ab.push_alias(4, 4);
        ab.finish()
    };
    assert_eq!(snapshot.seq(0), BStr::new("ACGT"));
    assert_eq!(snapshot.qual(0), BStr::new("FFFF"));
    let orig = 12;

    // A whole-segment pipeline: overlay cut, two rebuilds (Arc diverges).
    pod.cut_start(2, None);
    pod = pod.prefix(b"XX", b"##", None);
    pod = pod.reverse(None);

    match lift(&pod, born, 0, 4, 4, orig) {
        RegionLift::Kept { start, len } => {
            let mut rev_seq = b"ACGT".to_vec();
            rev_seq.reverse();
            let mut rev_qual = b"FFFF".to_vec();
            rev_qual.reverse();
            // One coordinate map serves both buffers.
            assert_eq!(&pod.seq(0)[start..start + len], BStr::new(&rev_seq));
            assert_eq!(&pod.qual(0)[start..start + len], BStr::new(&rev_qual));
        }
        RegionLift::Dropped => panic!("region should survive the pipeline"),
    }
    // The frozen snapshot never moved, and the generation advanced.
    assert_eq!(snapshot.seq(0), BStr::new("ACGT"));
    assert!(pod.generation(0).expect("row 0") > born);
}

fn assert_tag(pod: &DualStringPod, row: usize, expect_start: usize) {
    match lift(pod, 0, row, 4, 4, 8) {
        RegionLift::Kept { start, len } => {
            assert_eq!(start, expect_start, "row {row} lifted start");
            assert_eq!(&pod.seq(row)[start..start + len], BStr::new("CCCC"));
        }
        RegionLift::Dropped => panic!("row {row} tag should survive"),
    }
}

#[test]
fn conditional_edit_then_drain_keeps_rows_aligned() {
    let mut bld = DualStringPodBuilder::with_capacity(8, 3);
    for _ in 0..3 {
        bld.push(b"AAAACCCC", b"IIIIIIII"); // tag region [4, 8) = "CCCC"
    }
    let mut pod = bld.finish();
    assert_eq!(
        (0..3).map(|r| pod.generation(r)).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(0)],
    );

    // Conditional cut on rows 0 and 2 only — promotes the edit column per-entry.
    pod.cut_start(2, Some(&[true, false, true]));
    assert_eq!(pod.generation(0), Some(1));
    assert_eq!(pod.generation(1), Some(0)); // untouched row recorded nothing
    assert_eq!(pod.generation(2), Some(1));
    assert_tag(&pod, 0, 2); // shifted by the cut
    assert_tag(&pod, 1, 4); // untouched
    assert_tag(&pod, 2, 2);

    // Drop the middle (untouched) row; the survivors stay addressable and
    // their histories stay aligned.
    pod.drain(1..2);
    assert_eq!(pod.len(), 2);
    assert_tag(&pod, 0, 2); // old row 0
    assert_tag(&pod, 1, 2); // old row 2, now row 1
}

#[test]
fn max_len_drops_a_tag_beyond_the_cut() {
    let mut bld = DualStringPodBuilder::with_capacity(0, 1);
    bld.push(b"ACGTACGTACGT", b"IIIIFFFFJJJJ"); // len 12
    let mut pod = bld.finish();

    pod.max_len(6, None); // keep first 6 bytes

    // A tag in the kept prefix survives; one in the clipped tail is dropped.
    assert_eq!(
        lift(&pod, 0, 0, 0, 4, 12),
        RegionLift::Kept { start: 0, len: 4 }
    );
    assert_eq!(lift(&pod, 0, 0, 8, 4, 12), RegionLift::Dropped);
}

#[test]
fn max_len_lifts_through_a_slice_with_offset() {
    // Mirror the demultiplex pipeline that exposed the first-byte bug: a tag is
    // born, the pod is sliced to a sub-range (a non-zero first byte / shifted
    // rows), a splice makes the lengths uneven (so `max_len` takes the Variable
    // index-only path), then Truncate -> `max_len` clips. The tag must still
    // lift onto the right bytes across all of it.
    let mut bld = DualStringPodBuilder::with_capacity(0, 3);
    bld.push(b"AAAACCCCGG", b"IIIIIIIIII"); // row 0, len 10
    bld.push(b"TTTTGGGGAA", b"FFFFFFFFFF"); // row 1, len 10
    bld.push(b"CCCCAAAATT", b"JJJJJJJJJJ"); // row 2, len 10
    let pod = bld.finish();

    // Tag born on row 1: region [4, 8) ("GGGG"). Capture its generation now.
    let born = pod.generation(1).expect("row 1 exists");

    // Slice to rows 1..3 (row 1 -> sliced row 0). The edit history rides along.
    let mut sliced = pod.slice(1..3);
    assert_eq!(sliced.seq(0), BStr::new("TTTTGGGGAA"));

    // Make the lengths uneven so `max_len` takes the rebuild path (not the
    // FixedLength overlay), then clip to 8 bytes — past the [4, 8) tag's end.
    sliced.splice_entries(&[Some((9, 1, b"X".to_vec(), b"K".to_vec())), None]);
    sliced.max_len(8, None);

    // The tag's bytes ("GGGG" at [4, 8)) are entirely within the kept prefix, so
    // it lifts to the same offset and still names "GGGG".
    match lift(&sliced, born, 0, 4, 4, 10) {
        RegionLift::Kept { start, len } => {
            assert_eq!(&sliced.seq(0)[start..start + len], BStr::new("GGGG"));
        }
        RegionLift::Dropped => panic!("expected Kept, got Dropped"),
    }
}

#[test]
fn write_back_splice_is_isolated_to_its_read() {
    let mut bld = DualStringPodBuilder::with_capacity(0, 2);
    bld.push(b"ACGTACGT", b"IIIIIIII"); // len 8
    bld.push(b"ACGTACGT", b"IIIIIIII");
    let mut pod = bld.finish();

    // Write a longer tag back into read 0: at offset 2, replace 2 bytes with 5.
    pod.record_splice(0, 2, 2, 5);

    // A downstream tag at [6, 8) on read 0 shifts by +3; read 1 is untouched.
    assert_eq!(
        lift(&pod, 0, 0, 6, 2, 8),
        RegionLift::Kept { start: 9, len: 2 }
    );
    assert_eq!(
        lift(&pod, 0, 1, 6, 2, 8),
        RegionLift::Kept { start: 6, len: 2 }
    );
    assert_eq!(pod.generation(0), Some(1));
    assert_eq!(pod.generation(1), Some(0));
}

#[test]
fn splice_entries_rewrites_bytes_and_carries_history() {
    let mut bld = DualStringPodBuilder::with_capacity(0, 2);
    bld.push(b"AACCGGTT", b"IIIIIIII"); // len 8
    bld.push(b"AACCGGTT", b"IIIIIIII");
    let mut pod = bld.finish();

    // A tag born now at [6, 8) ("TT") of read 0, before any edit.
    let born = pod.generation(0).expect("row 0 exists");
    assert_eq!(born, 0);

    // An earlier whole-column edit: drop the first 2 bytes. The tag's birth-frame
    // [6, 8) now lives at [4, 6).
    pod.cut_start(2, None);
    assert_eq!(pod.seq(0), BStr::new(b"CCGGTT"));
    assert_eq!(
        lift(&pod, born, 0, 6, 2, 8),
        RegionLift::Kept { start: 4, len: 2 }
    );

    // Splice read 0 only: at current offset 0, replace 2 bytes ("CC") with 3 ("xyz").
    pod.splice_entries(&[Some((0, 2, b"xyz".to_vec(), b"###".to_vec())), None]);

    // Bytes are rebuilt; read 1 is untouched.
    assert_eq!(pod.seq(0), BStr::new(b"xyzGGTT"));
    assert_eq!(pod.qual(0), BStr::new(b"###IIII"));
    assert_eq!(pod.seq(1), BStr::new(b"CCGGTT"));

    // History survives the rebuild: the tag born at gen 0 lifts through *both* the
    // cut and the splice. Its [6, 8) sat at [4, 6) pre-splice; the +1 net insert
    // before it shifts it to [5, 7).
    assert_eq!(
        lift(&pod, born, 0, 6, 2, 8),
        RegionLift::Kept { start: 5, len: 2 }
    );
    // Read 1 saw only the cut.
    assert_eq!(
        lift(&pod, born, 1, 6, 2, 8),
        RegionLift::Kept { start: 4, len: 2 }
    );
}
