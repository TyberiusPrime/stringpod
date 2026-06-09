//! Shared coordinate-liftover wiring for the pod types.
//!
//! Both [`StringPod`](crate::StringPod) and [`DualStringPod`](crate::DualStringPod)
//! carry a [`ColumnEdits`] and must record the *same* coordinate edit whenever
//! they mutate read bytes — the only difference between them is how many byte
//! buffers the mutation touches, which the liftover doesn't care about. [`Lifted`]
//! captures everything that is buffer-count-agnostic: the dispatch from a pod
//! operation to the right `ColumnEdits` call (whole-column vs per-entry vs
//! single-entry) and the public query surface a tag uses to lift its frozen
//! coordinate.
//!
//! A pod implements [`Lifted`] by exposing its `ColumnEdits` through `edits` /
//! `edits_mut`; every other method is provided. The pod's own mutators do their
//! buffer work and then call the matching `record_*` helper — e.g.
//! [`DualStringPod::cut_start`](crate::DualStringPod) applies the storage cut and
//! then calls [`record_cut_start`](Lifted::record_cut_start). Rebuild ops
//! (`prefix`/`postfix`/`reverse`) construct a fresh pod, so they move the old
//! `ColumnEdits` into it and record the op there — the history must survive the
//! `Arc` divergence, which is exactly when liftover matters.

use crate::column::ColumnEdits;
use crate::editlog::{EditLog, EditLogError, EditLogView};
use std::ops::Range;

/// A column that tracks coordinate edits for liftover. See the
/// [module docs](crate::lifted).
pub trait Lifted {
    /// Shared read access to this column's edit history.
    fn edits(&self) -> &ColumnEdits;

    /// Shared mutable access to this column's edit history.
    fn edits_mut(&mut self) -> &mut ColumnEdits;

    // ── recording: called by the pod *after* it performs the byte/metadata op ──

    /// Record a `cut_start(n)`: whole-column when `cond` is `None`, otherwise
    /// only the entries whose mask bit is set.
    fn record_cut_start(&mut self, n: usize, cond: Option<&[bool]>) {
        match cond {
            None => self.edits_mut().apply_all(|log| log.cut_start(n)),
            Some(mask) => self.edits_mut().apply_entries(|i, log| {
                if mask[i] {
                    log.cut_start(n);
                }
            }),
        }
    }

    /// Record a `cut_end(n)`: whole-column or masked, as for
    /// [`record_cut_start`](Self::record_cut_start).
    fn record_cut_end(&mut self, n: usize, cond: Option<&[bool]>) {
        match cond {
            None => self.edits_mut().apply_all(|log| log.cut_end(n)),
            Some(mask) => self.edits_mut().apply_entries(|i, log| {
                if mask[i] {
                    log.cut_end(n);
                }
            }),
        }
    }

    /// Record a whole-column `prefix` of `k` bytes.
    fn record_prefix(&mut self, k: usize) {
        self.edits_mut().apply_all(|log| log.prefix(k));
    }

    /// Record a whole-column `postfix` of `k` bytes.
    fn record_postfix(&mut self, k: usize) {
        self.edits_mut().apply_all(|log| log.postfix(k));
    }

    /// Record a `reverse`: whole-column when `cond` is `None`, otherwise only
    /// the masked entries.
    fn record_reverse(&mut self, cond: Option<&[bool]>) {
        match cond {
            None => self.edits_mut().apply_all(EditLog::reflect),
            Some(mask) => self.edits_mut().apply_entries(|i, log| {
                if mask[i] {
                    log.reflect();
                }
            }),
        }
    }

    /// Record per-entry window narrowing for `resize` / `max_len`: entry `i`
    /// keeps `windows[i] = Some((start, len, cur_len))` of its current `cur_len`
    /// visible bytes (`None` leaves the entry unchanged). Expressed as the
    /// equivalent `cut_start(start)` + `cut_end(cur_len - start - len)`.
    ///
    /// # Panics
    /// If `windows.len()` is smaller than the number of live entries, or any
    /// window is not contained in its entry (`start + len > cur_len`).
    fn record_windows(&mut self, windows: &[Option<(usize, usize, usize)>]) {
        self.edits_mut().apply_entries(|i, log| {
            if let Some((start, len, cur_len)) = windows[i] {
                assert!(start + len <= cur_len, "resize window exceeds entry length");
                if start > 0 {
                    log.cut_start(start);
                }
                let tail = cur_len - start - len;
                if tail > 0 {
                    log.cut_end(tail);
                }
            }
        });
    }

    /// Record a length-changing write-back into a single read: delete `del`
    /// bytes at offset `at` and insert `ins`. A no-op if `row` is out of range.
    fn record_splice(&mut self, row: usize, at: usize, del: usize, ins: usize) {
        self.edits_mut().apply_entry(row, |log| log.splice(at, del, ins));
    }

    // ── row-axis: keep the per-entry logs aligned with the live entries ───────

    /// Drop the edit history for a contiguous range of live entries (`drain`).
    fn record_drain(&mut self, range: Range<usize>) {
        self.edits_mut().drain(range);
    }

    /// Drop the edit history for the first `n` live entries (`pop_front`).
    fn record_pop_front(&mut self, n: usize) {
        self.edits_mut().pop_front(n);
    }

    /// Keep only the first `len` entries' histories (`truncate`).
    fn record_truncate(&mut self, len: usize) {
        self.edits_mut().truncate(len);
    }

    /// Keep only the marked entries' histories (`retain_by_bools`).
    fn record_retain(&mut self, keep: &[bool]) {
        self.edits_mut().retain(keep);
    }

    // ── query: the public liftover surface a tag uses ────────────────────────

    /// The current generation of entry `row` — capture this when a tag's
    /// coordinate snapshot is taken. `None` if `row` is out of range.
    #[must_use]
    fn generation(&self, row: usize) -> Option<usize> {
        self.edits().generation(row)
    }

    /// A view over the edits applied to entry `row` since generation `born`,
    /// for lifting that entry's birth-frame coordinate into the current frame.
    ///
    /// # Errors
    /// [`EditLogError::RowOutOfBounds`] if `row` is out of range;
    /// [`EditLogError::GenerationOutOfRange`] if `born` exceeds the entry's
    /// recorded edits.
    fn ops_since(&self, born: usize, row: usize) -> Result<EditLogView<'_>, EditLogError> {
        self.edits().view_from(row, born)
    }
}
