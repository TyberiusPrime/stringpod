//! Tests for [`stringpod::ColumnEdits`].
//!
//! The contract: a `ColumnEdits` must behave exactly as if each live entry
//! owned an independent [`EditLog`](stringpod::EditLog) that received precisely
//! the edits which touched it — regardless of whether it is internally still in
//! the shared `Uniform` state or has promoted to per-entry. Whole-row ops
//! (`drain`/`pop_front`/`truncate`/`retain`) must keep those per-row histories aligned
//! with the live entries. `EditLog` itself is already cross-checked against a
//! brute-force cell model in `editlog_liftover.rs`, so here we validate the
//! aggregation against a ground-truth `Vec<Vec<Edit>>` of per-row edit lists.

use stringpod::{ColumnEdits, EditLog, EditLogError};

const L0: usize = 32; // every row's birth-frame length in the model

#[derive(Clone, Copy, Debug)]
enum Edit {
    CutStart(usize),
    CutEnd(usize),
    Prefix(usize),
    Postfix(usize),
    Splice { at: usize, del: usize, ins: usize },
    Reflect,
}

fn dispatch(log: &mut EditLog, e: Edit) {
    match e {
        Edit::CutStart(n) => log.cut_start(n),
        Edit::CutEnd(n) => log.cut_end(n),
        Edit::Prefix(k) => log.prefix(k),
        Edit::Postfix(k) => log.postfix(k),
        Edit::Splice { at, del, ins } => log.splice(at, del, ins),
        Edit::Reflect => log.reflect(),
    }
}

fn record(edits: &[Edit]) -> EditLog {
    let mut log = EditLog::new();
    for e in edits {
        dispatch(&mut log, *e);
    }
    log
}

// ── the model: one ground-truth edit list per live row ──────────────────────

struct Model {
    rows: Vec<Vec<Edit>>,
}

impl Model {
    fn new(len: usize) -> Self {
        Self {
            rows: vec![Vec::new(); len],
        }
    }

    fn apply_all(&mut self, col: &mut ColumnEdits, e: Edit) {
        for row in &mut self.rows {
            row.push(e);
        }
        col.apply_all(|log| dispatch(log, e));
    }

    fn apply_entries(&mut self, col: &mut ColumnEdits, mask: &[bool], e: Edit) {
        for (row, m) in self.rows.iter_mut().zip(mask) {
            if *m {
                row.push(e);
            }
        }
        col.apply_entries(|i, log| {
            if mask[i] {
                dispatch(log, e);
            }
        });
    }

    fn apply_entry(&mut self, col: &mut ColumnEdits, row: usize, e: Edit) {
        self.rows[row].push(e);
        col.apply_entry(row, |log| dispatch(log, e));
    }

    fn drain(&mut self, col: &mut ColumnEdits, start: usize, end: usize) {
        self.rows.drain(start..end);
        col.drain(start..end);
    }

    fn pop_front(&mut self, col: &mut ColumnEdits, n: usize) {
        self.rows.drain(0..n.min(self.rows.len()));
        col.pop_front(n);
    }

    fn truncate(&mut self, col: &mut ColumnEdits, len: usize) {
        self.rows.truncate(len);
        col.truncate(len);
    }

    fn retain(&mut self, col: &mut ColumnEdits, keep: &[bool]) {
        let mut it = keep.iter();
        self.rows.retain(|_| it.next().copied().unwrap_or(false));
        col.retain(keep);
    }

    /// Every claim `ColumnEdits` makes must match the per-row ground truth.
    fn check(&self, col: &ColumnEdits) {
        assert_eq!(col.len(), self.rows.len(), "len mismatch");
        for (row, edits) in self.rows.iter().enumerate() {
            assert_eq!(
                col.generation(row),
                Some(edits.len()),
                "generation mismatch at row {row}"
            );
            // The suffix from every generation must lift identically to an
            // independent EditLog built from that suffix.
            for born in 0..=edits.len() {
                let view = col.view_from(row, born).expect("row & born in range");
                let suffix = record(&edits[born..]);
                let born_len = record(&edits[..born]).current_len(L0);
                assert_eq!(
                    view.current_len(born_len),
                    suffix.current_len(born_len),
                    "current_len mismatch row {row} born {born}"
                );
                if born_len > 0 {
                    // a couple of representative regions inside the birth frame
                    for &(s, l) in &[(0usize, 1usize), (0, born_len), (born_len / 2, 1)] {
                        if l >= 1 && s + l <= born_len {
                            assert_eq!(
                                view.map_region(s, l, born_len),
                                suffix.map_region(s, l, born_len),
                                "region [{s},{l}) mismatch row {row} born {born}"
                            );
                        }
                    }
                }
            }
        }
    }
}

// ── tiny deterministic RNG (no dev-dependencies) ────────────────────────────

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
        let high = self.next_u64() >> 33;
        usize::try_from(high).expect("31-bit value fits usize") % bound
    }
}

fn random_edit(rng: &mut Lcg) -> Edit {
    match rng.below(6) {
        0 => Edit::CutStart(rng.below(5)),
        1 => Edit::CutEnd(rng.below(5)),
        2 => Edit::Prefix(rng.below(4)),
        3 => Edit::Postfix(rng.below(4)),
        4 => Edit::Splice {
            at: rng.below(10),
            del: rng.below(5),
            ins: rng.below(4),
        },
        _ => Edit::Reflect,
    }
}

fn random_mask(rng: &mut Lcg, len: usize) -> Vec<bool> {
    (0..len).map(|_| rng.below(2) == 1).collect()
}

// ── property test ───────────────────────────────────────────────────────────

#[test]
fn column_matches_per_row_model() {
    let mut rng = Lcg(0x0bad_c0de_1234_5678);
    for _ in 0..2_000 {
        let mut len = 1 + rng.below(6);
        let mut model = Model::new(len);
        let mut col = ColumnEdits::new(len);

        for _ in 0..12 {
            if len == 0 {
                break;
            }
            match rng.below(8) {
                0 => model.apply_all(&mut col, random_edit(&mut rng)),
                1 | 2 => {
                    let mask = random_mask(&mut rng, len);
                    model.apply_entries(&mut col, &mask, random_edit(&mut rng));
                }
                3 => {
                    let row = rng.below(len);
                    model.apply_entry(&mut col, row, random_edit(&mut rng));
                }
                4 => {
                    let start = rng.below(len);
                    let end = start + 1 + rng.below(len - start);
                    model.drain(&mut col, start, end);
                    len = model.rows.len();
                }
                5 => {
                    let n = rng.below(len + 1);
                    model.pop_front(&mut col, n);
                    len = model.rows.len();
                }
                6 => {
                    let keep = rng.below(len + 1);
                    model.truncate(&mut col, keep);
                    len = model.rows.len();
                }
                _ => {
                    let mask = random_mask(&mut rng, len);
                    model.retain(&mut col, &mask);
                    len = model.rows.len();
                }
            }
            model.check(&col);
        }
    }
}

// ── focused unit tests ──────────────────────────────────────────────────────

#[test]
fn starts_uniform_and_shares_generation() {
    let mut col = ColumnEdits::new(3);
    assert!(col.is_uniform());
    col.apply_all(|log| log.cut_start(2));
    col.apply_all(|log| log.prefix(1));
    assert!(col.is_uniform(), "whole-column edits stay uniform");
    assert_eq!(col.generation(0), Some(2));
    assert_eq!(col.generation(2), Some(2));
    assert_eq!(col.generation(3), None);
}

#[test]
fn conditional_edit_promotes_and_diverges() {
    let mut col = ColumnEdits::new(3);
    col.apply_all(|log| log.cut_start(1)); // gen 1 everywhere
    col.apply_entries(|row, log| {
        if row == 1 {
            log.cut_end(2);
        }
    });
    assert!(!col.is_uniform());
    assert_eq!(col.generation(0), Some(1));
    assert_eq!(col.generation(1), Some(2)); // row 1 saw the extra edit
    assert_eq!(col.generation(2), Some(1));
}

#[test]
fn apply_entry_touches_only_that_row() {
    let mut col = ColumnEdits::new(4);
    col.apply_entry(2, |log| log.splice(3, 1, 4));
    assert!(!col.is_uniform());
    assert_eq!(col.generation(0), Some(0));
    assert_eq!(col.generation(2), Some(1));
    assert_eq!(col.generation(3), Some(0));
}

#[test]
fn drain_in_uniform_state_keeps_shared_log() {
    let mut col = ColumnEdits::new(5);
    col.apply_all(|log| log.cut_start(3));
    col.drain(1..3);
    assert!(col.is_uniform(), "row-axis ops don't force promotion");
    assert_eq!(col.len(), 3);
    // every surviving row still carries the one shared edit
    for row in 0..3 {
        assert_eq!(col.generation(row), Some(1));
    }
}

#[test]
fn drain_in_per_entry_state_removes_the_right_logs() {
    let mut col = ColumnEdits::new(4);
    // give each row a distinct number of edits so we can identify them
    for row in 0..4 {
        for _ in 0..row {
            col.apply_entry(row, |log| log.cut_start(1));
        }
    }
    assert_eq!(
        (0..4).map(|r| col.generation(r)).collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2), Some(3)],
    );
    col.drain(1..3); // drop the rows with 1 and 2 edits
    assert_eq!(col.len(), 2);
    assert_eq!(col.generation(0), Some(0));
    assert_eq!(col.generation(1), Some(3));
}

#[test]
fn pop_front_and_truncate_shrink_lockstep() {
    let mut col = ColumnEdits::new(5);
    for row in 0..5 {
        for _ in 0..row {
            col.apply_entry(row, crate::EditLog::reflect);
        }
    }
    col.pop_front(2); // drop rows with 0 and 1 edits
    assert_eq!(col.len(), 3);
    assert_eq!(col.generation(0), Some(2));
    col.truncate(2); // drop the last (4-edit) row
    assert_eq!(col.len(), 2);
    assert_eq!(col.generation(0), Some(2));
    assert_eq!(col.generation(1), Some(3));
}

#[test]
fn retain_keeps_marked_rows() {
    let mut col = ColumnEdits::new(4);
    for row in 0..4 {
        for _ in 0..row {
            col.apply_entry(row, |log| log.cut_end(1));
        }
    }
    col.retain(&[true, false, true, false]);
    assert_eq!(col.len(), 2);
    assert_eq!(col.generation(0), Some(0));
    assert_eq!(col.generation(1), Some(2));
}

#[test]
fn view_from_errors_are_typed() {
    let mut col = ColumnEdits::new(2);
    col.apply_all(|log| log.cut_start(1));
    assert_eq!(
        col.view_from(5, 0).map(|_| ()),
        Err(EditLogError::RowOutOfBounds { row: 5, len: 2 })
    );
    assert_eq!(
        col.view_from(0, 9).map(|_| ()),
        Err(EditLogError::GenerationOutOfRange {
            generation: 9,
            recorded: 1,
        })
    );
}
