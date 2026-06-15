//! Liftover test suite for [`stringpod::EditLog`].
//!
//! Every claim the `EditLog` makes is cross-checked against an independent
//! brute-force model: apply the same edits to a vector of *cells* (`Some(i)` =
//! original byte `i`, `None` = inserted byte) and read off where the bytes
//! actually land. This mirrors how genomics liftover is validated against an
//! explicit alignment. Lives as an integration test so it exercises only the
//! public API.

use bstr::BStr;
use stringpod::{
    DualStringPodBuilder, EditLog, EditLogError, OffsetLift, RegionLift, StringPodBuilder,
};

// ── thin wrappers: the query API returns `Result`; in these tests the
//    coordinates are always in range, so unwrap the contract with `.expect`. ──

fn pos(log: &EditLog, position: usize, orig_len: usize) -> OffsetLift {
    log.map_position(position, orig_len)
        .expect("position in bounds")
}

fn region(log: &EditLog, start: usize, len: usize, orig_len: usize) -> RegionLift {
    log.map_region(start, len, orig_len)
        .expect("region in bounds")
}

// ── the edit vocabulary, mirroring EditLog's recording methods ──────────────

#[derive(Clone, Copy, Debug)]
enum Edit {
    CutStart(usize),
    CutEnd(usize),
    Prefix(usize),
    Postfix(usize),
    Splice { at: usize, del: usize, ins: usize },
    Reflect,
}

fn record(edits: &[Edit]) -> EditLog {
    let mut log = EditLog::new();
    for edit in edits {
        match *edit {
            Edit::CutStart(n) => log.cut_start(n),
            Edit::CutEnd(n) => log.cut_end(n),
            Edit::Prefix(k) => log.prefix(k),
            Edit::Postfix(k) => log.postfix(k),
            Edit::Splice { at, del, ins } => log.splice(at, del, ins),
            Edit::Reflect => log.reflect(),
        }
    }
    log
}

// ── independent brute-force model ───────────────────────────────────────────

fn model_apply(orig_len: usize, edits: &[Edit]) -> Vec<Option<usize>> {
    let mut cells: Vec<Option<usize>> = (0..orig_len).map(Some).collect();
    for edit in edits {
        let cur = cells.len();
        match *edit {
            Edit::CutStart(n) => {
                cells.drain(0..n.min(cur));
            }
            Edit::CutEnd(n) => {
                cells.truncate(cur - n.min(cur));
            }
            Edit::Prefix(k) => {
                cells.splice(0..0, vec![None; k]);
            }
            Edit::Postfix(k) => {
                cells.splice(cur..cur, vec![None; k]);
            }
            Edit::Splice { at, del, ins } => {
                let a = at.min(cur);
                let d = del.min(cur - a);
                cells.splice(a..a + d, vec![None; ins]);
            }
            Edit::Reflect => cells.reverse(),
        }
    }
    cells
}

fn model_position(cells: &[Option<usize>], original: usize) -> OffsetLift {
    match cells.iter().position(|c| *c == Some(original)) {
        Some(p) => OffsetLift::At(p),
        None => OffsetLift::Deleted,
    }
}

fn model_region(cells: &[Option<usize>], start: usize, len: usize) -> RegionLift {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for original in start..start + len {
        match model_position(cells, original) {
            OffsetLift::Deleted => return RegionLift::Dropped,
            OffsetLift::At(p) => {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    if (hi - lo) + 1 == len {
        RegionLift::Kept { start: lo, len }
    } else {
        RegionLift::Dropped
    }
}

// ── tiny deterministic RNG, so there are no dev-dependencies ────────────────

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        let high = self.next_u64() >> 33; // < 2^31, always fits a usize
        usize::try_from(high).expect("31-bit value fits usize") % bound
    }
}

fn random_edit(rng: &mut Lcg, hint: usize) -> Edit {
    match rng.below(6) {
        0 => Edit::CutStart(rng.below(hint + 2)),
        1 => Edit::CutEnd(rng.below(hint + 2)),
        2 => Edit::Prefix(rng.below(4)),
        3 => Edit::Postfix(rng.below(4)),
        4 => Edit::Splice {
            at: rng.below(hint + 2),
            del: rng.below(hint + 2),
            ins: rng.below(4),
        },
        _ => Edit::Reflect,
    }
}

fn check_against_model(orig_len: usize, edits: &[Edit], rng: &mut Lcg, region_samples: usize) {
    let cells = model_apply(orig_len, edits);
    let log = record(edits);

    assert_eq!(
        log.current_len(orig_len),
        cells.len(),
        "current_len mismatch: orig_len={orig_len}, edits={edits:?}"
    );

    for cell in 0..orig_len {
        assert_eq!(
            pos(&log, cell, orig_len),
            model_position(&cells, cell),
            "position {cell}: orig_len={orig_len}, edits={edits:?}"
        );
    }

    if region_samples == 0 {
        // exhaustive over all non-empty regions
        for start in 0..orig_len {
            for len in 1..=orig_len - start {
                assert_eq!(
                    region(&log, start, len, orig_len),
                    model_region(&cells, start, len),
                    "region [{start},{len}): orig_len={orig_len}, edits={edits:?}"
                );
            }
        }
    } else {
        for _ in 0..region_samples {
            let start = rng.below(orig_len);
            let len = 1 + rng.below(orig_len - start);
            assert_eq!(
                region(&log, start, len, orig_len),
                model_region(&cells, start, len),
                "region [{start},{len}): orig_len={orig_len}, edits={edits:?}"
            );
        }
    }
}

// ── property tests ──────────────────────────────────────────────────────────

#[test]
fn liftover_matches_model_exhaustively_small() {
    let mut rng = Lcg(0x1234_5678_9abc_def1);
    for _ in 0..50_000 {
        let orig_len = rng.below(9); // 0..=8
        let count = rng.below(7); // 0..=6 stacked edits
        let edits: Vec<Edit> = (0..count)
            .map(|_| random_edit(&mut rng, orig_len))
            .collect();
        check_against_model(orig_len, &edits, &mut rng, 0);
    }
}

#[test]
fn liftover_matches_model_larger_frames() {
    let mut rng = Lcg(0xfeed_face_dead_beef);
    for _ in 0..5_000 {
        let orig_len = 8 + rng.below(40);
        let count = rng.below(12);
        let edits: Vec<Edit> = (0..count)
            .map(|_| random_edit(&mut rng, orig_len))
            .collect();
        check_against_model(orig_len, &edits, &mut rng, 40);
    }
}

// ── explicit unit tests for each op and the tricky edges ────────────────────

#[test]
fn identity_log() {
    let log = EditLog::new();
    assert!(log.is_empty());
    assert_eq!(log.op_count(), 0);
    assert_eq!(log.current_len(10), 10);
    assert_eq!(pos(&log, 3, 10), OffsetLift::At(3));
    assert_eq!(
        region(&log, 2, 4, 10),
        RegionLift::Kept { start: 2, len: 4 }
    );
}

#[test]
fn cut_start_shifts_and_deletes() {
    let mut log = EditLog::new();
    log.cut_start(3);
    assert_eq!(log.current_len(10), 7);
    assert_eq!(pos(&log, 2, 10), OffsetLift::Deleted); // inside the cut
    assert_eq!(pos(&log, 3, 10), OffsetLift::At(0)); // first survivor
    assert_eq!(pos(&log, 9, 10), OffsetLift::At(6));
    assert_eq!(region(&log, 1, 4, 10), RegionLift::Dropped); // straddles the cut
    assert_eq!(
        region(&log, 3, 4, 10),
        RegionLift::Kept { start: 0, len: 4 }
    );
}

#[test]
fn cut_end_clips_tail() {
    let mut log = EditLog::new();
    log.cut_end(3);
    assert_eq!(log.current_len(10), 7);
    assert_eq!(pos(&log, 6, 10), OffsetLift::At(6));
    assert_eq!(pos(&log, 7, 10), OffsetLift::Deleted);
    assert_eq!(
        region(&log, 5, 2, 10),
        RegionLift::Kept { start: 5, len: 2 }
    );
    assert_eq!(region(&log, 6, 2, 10), RegionLift::Dropped);
}

#[test]
fn prefix_postfix_shift() {
    let mut log = EditLog::new();
    log.prefix(2);
    log.postfix(5);
    assert_eq!(log.current_len(4), 11);
    assert_eq!(pos(&log, 0, 4), OffsetLift::At(2));
    assert_eq!(pos(&log, 3, 4), OffsetLift::At(5));
    assert_eq!(region(&log, 0, 4, 4), RegionLift::Kept { start: 2, len: 4 });
}

#[test]
fn splice_grow_and_shrink() {
    // delete 2 bytes at offset 4, insert 5 — net +3.
    let mut grow = EditLog::new();
    grow.splice(4, 2, 5);
    assert_eq!(grow.current_len(10), 13);
    assert_eq!(pos(&grow, 3, 10), OffsetLift::At(3)); // before splice
    assert_eq!(pos(&grow, 4, 10), OffsetLift::Deleted); // inside deletion
    assert_eq!(pos(&grow, 6, 10), OffsetLift::At(9)); // after, +3
    assert_eq!(
        region(&grow, 0, 4, 10),
        RegionLift::Kept { start: 0, len: 4 }
    );
    assert_eq!(region(&grow, 3, 3, 10), RegionLift::Dropped); // overlaps deletion
}

#[test]
fn pure_insertion_boundary_vs_interior() {
    // insert 3 bytes at offset 5 (del = 0).
    let mut log = EditLog::new();
    log.splice(5, 0, 3);
    // region ending at the insertion's right edge keeps (insert is outside).
    assert_eq!(
        region(&log, 2, 3, 10),
        RegionLift::Kept { start: 2, len: 3 }
    );
    // region starting at the insertion's left edge keeps, shifted past it.
    assert_eq!(
        region(&log, 5, 3, 10),
        RegionLift::Kept { start: 8, len: 3 }
    );
    // region straddling the insertion point is split → dropped.
    assert_eq!(region(&log, 4, 3, 10), RegionLift::Dropped);
}

#[test]
fn reflect_mirrors_region() {
    let mut log = EditLog::new();
    log.reflect();
    assert_eq!(log.current_len(10), 10);
    assert_eq!(pos(&log, 0, 10), OffsetLift::At(9));
    assert_eq!(pos(&log, 9, 10), OffsetLift::At(0));
    // region [2,5) → mirror span [10-5, 10-2) = [5,8)
    assert_eq!(
        region(&log, 2, 3, 10),
        RegionLift::Kept { start: 5, len: 3 }
    );
}

#[test]
fn stacked_cut_prefix_reflect() {
    // The canonical "Arc and storage both moved" chain.
    let mut log = EditLog::new();
    log.cut_start(2);
    log.prefix(2);
    log.reflect();
    assert_eq!(
        region(&log, 4, 4, 12),
        RegionLift::Kept { start: 4, len: 4 }
    );
}

// ── error reporting (formerly panics) ───────────────────────────────────────

#[test]
fn map_region_rejects_empty() {
    assert_eq!(
        EditLog::new().map_region(0, 0, 10),
        Err(EditLogError::EmptyRegion)
    );
}

#[test]
fn map_region_rejects_out_of_bounds() {
    assert_eq!(
        EditLog::new().map_region(8, 4, 10),
        Err(EditLogError::RegionOutOfBounds {
            start: 8,
            len: 4,
            orig_len: 10,
        })
    );
}

#[test]
fn map_position_rejects_oob() {
    assert_eq!(
        EditLog::new().map_position(10, 10),
        Err(EditLogError::PositionOutOfBounds {
            position: 10,
            orig_len: 10,
        })
    );
}

// ── view_from: replay only the edits since a generation ─────────────────────

#[test]
fn view_from_end_is_identity() {
    let mut log = EditLog::new();
    log.cut_start(2);
    let view = log
        .view_from(log.op_count())
        .expect("end is a valid generation");
    assert!(view.is_empty());
    assert_eq!(view.current_len(10), 10);
    assert_eq!(view.map_position(0, 10), Ok(OffsetLift::At(0)));
    assert_eq!(
        view.map_region(0, 4, 10),
        Ok(RegionLift::Kept { start: 0, len: 4 })
    );
}

#[test]
fn view_from_rejects_future_generation() {
    let mut log = EditLog::new();
    log.cut_start(1);
    assert_eq!(
        log.view_from(5).map(|_| ()),
        Err(EditLogError::GenerationOutOfRange {
            generation: 5,
            recorded: 1,
        })
    );
}

#[test]
fn view_from_replays_only_the_suffix() {
    // A tag born after the first edit lifts its *birth-frame* coordinate; the
    // full log lifts the *original-frame* coordinate. They must agree once the
    // birth-frame query is the image of the original-frame one.
    let orig_len = 12usize;
    let mut log = EditLog::new();
    log.cut_start(2); // original [2,6) → birth [0,4)
    let born = log.op_count();
    let born_len = log.current_len(orig_len); // 10
    log.prefix(3);
    log.reflect();

    let view = log.view_from(born).expect("valid generation");
    let full = region(&log, 2, 4, orig_len); // from the original frame
    let via_view = view
        .map_region(0, 4, born_len)
        .expect("birth-frame region in bounds"); // from the birth frame
    assert_eq!(full, via_view);
}

// ── integration with the real alias builders ────────────────────────────────

#[test]
fn single_alias_lifts_through_edits() {
    let mut bld = StringPodBuilder::new();
    bld.push(b"ACGTACGTACGT"); // one entry, len 12
    let mut source = bld.finish();

    // Freeze a snapshot of the middle [4,8) = "ACGT" via the alias builder.
    let snapshot = {
        let mut ab = source.alias_builder();
        ab.push_alias(4, 4);
        ab.finish()
    };
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));

    let orig_len = 12usize;
    let mut log = EditLog::new();

    // Overlay edit: storage moves, Arc stays shared with the snapshot.
    source.cut_start(2, None);
    log.cut_start(2);
    match region(&log, 4, 4, orig_len) {
        RegionLift::Kept { start, len } => {
            assert_eq!(&source.get(0)[start..start + len], BStr::new("ACGT"));
        }
        RegionLift::Dropped => panic!("region should survive cut_start"),
    }

    // Rebuild edit: Arc diverges; the snapshot keeps the old buffer.
    source = source.prefix(b"XX");
    log.prefix(2);
    match region(&log, 4, 4, orig_len) {
        RegionLift::Kept { start, len } => {
            assert_eq!(&source.get(0)[start..start + len], BStr::new("ACGT"));
        }
        RegionLift::Dropped => panic!("region should survive prefix"),
    }
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));

    // Reflect: coordinates mirror, the live bytes are reversed.
    source = source.reverse(None);
    log.reflect();
    match region(&log, 4, 4, orig_len) {
        RegionLift::Kept { start, len } => {
            let mut reversed = b"ACGT".to_vec();
            reversed.reverse();
            assert_eq!(&source.get(0)[start..start + len], BStr::new(&reversed));
        }
        RegionLift::Dropped => panic!("region should survive reflect"),
    }
    // The frozen snapshot never moved.
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));
}

#[test]
fn alias_born_midway_lifts_via_view_from() {
    // The Option-B mechanism end to end: a tag created *after* some pipeline
    // edits records its birth generation, then lifts through only the edits
    // that follow — exactly what a pod's `ops_since(generation)` will forward.
    let mut bld = StringPodBuilder::new();
    bld.push(b"ACGTACGTACGT"); // len 12
    let mut source = bld.finish();

    let mut log = EditLog::new();

    // Pipeline edit BEFORE the tag exists.
    source.cut_start(2, None); // visible "GTACGTACGT", len 10
    log.cut_start(2);

    // Tag is created now: snapshot [2,6) of the current read = "ACGT".
    let born = log.op_count();
    let born_len = log.current_len(12); // 10
    let snapshot = {
        let mut ab = source.alias_builder();
        ab.push_alias(2, 4);
        ab.finish()
    };
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));

    // More pipeline edits AFTER creation.
    source = source.prefix(b"XX");
    log.prefix(2);
    source = source.reverse(None);
    log.reflect();

    // Lift the tag's birth-frame region through just the post-birth edits.
    let view = log.view_from(born).expect("valid generation");
    match view
        .map_region(2, 4, born_len)
        .expect("birth region in bounds")
    {
        RegionLift::Kept { start, len } => {
            let mut reversed = b"ACGT".to_vec();
            reversed.reverse();
            assert_eq!(&source.get(0)[start..start + len], BStr::new(&reversed));
        }
        RegionLift::Dropped => panic!("region should survive prefix + reflect"),
    }
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));
}

#[test]
fn dual_alias_lifts_seq_and_qual_together() {
    let mut bld = DualStringPodBuilder::with_capacity(0, 1);
    bld.push(b"ACGTACGT", b"IIIIFFFF"); // len 8
    let mut source = bld.finish();

    let snapshot = {
        let mut ab = source.alias_builder();
        ab.push_alias(2, 4); // "GTAC" / "IIFF"
        ab.finish()
    };
    assert_eq!(snapshot.seq(0), BStr::new("GTAC"));
    assert_eq!(snapshot.qual(0), BStr::new("IIFF"));

    let orig_len = 8usize;
    let mut log = EditLog::new();

    source.cut_start(1, None);
    log.cut_start(1);
    source = source.postfix(b"NN", b"##", None);
    log.postfix(2);

    // One coordinate map serves both buffers of the dual pod.
    match region(&log, 2, 4, orig_len) {
        RegionLift::Kept { start, len } => {
            assert_eq!(&source.seq(0)[start..start + len], BStr::new("GTAC"));
            assert_eq!(&source.qual(0)[start..start + len], BStr::new("IIFF"));
        }
        RegionLift::Dropped => panic!("region should survive cut_start + postfix"),
    }
}

#[test]
fn alias_region_dropped_when_overwritten() {
    let mut bld = StringPodBuilder::new();
    bld.push(b"ACGTACGTACGT");
    let source = bld.finish();
    let snapshot = {
        let mut ab = source.alias_builder();
        ab.push_alias(4, 4);
        ab.finish()
    };
    // The snapshot still holds the bytes...
    assert_eq!(snapshot.get(0), BStr::new("ACGT"));

    // ...but a splice through the aliased region means it can no longer be
    // lifted back into the read's current frame.
    let mut log = EditLog::new();
    log.splice(5, 2, 2); // overwrites bytes 5..7, strictly inside [4,8)
    assert_eq!(region(&log, 4, 4, 12), RegionLift::Dropped);
}
