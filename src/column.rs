//! Per-column edit tracking for coordinate liftover.
//!
//! [`ColumnEdits`] aggregates one [`EditLog`](crate::EditLog) per column entry,
//! but starts in a compact `Uniform` state — a single shared log — and only
//! *promotes* to one-log-per-entry the first time an edit touches a subset of
//! entries (a conditional cut, a per-read `resize`/`max_len`, a single-read tag
//! write-back). This mirrors how the pods' own storage starts `FixedLength` and
//! promotes to `Variable`: the common pipeline (whole-segment `cut_start`,
//! `prefix`, `reverse`, …) stays in the cheap shared state, and only genuinely
//! per-entry edits pay for the `Vec<EditLog>`.
//!
//! A tag captures its entry's [`generation`](ColumnEdits::generation) when it is
//! born, then later lifts its frozen coordinate through only the edits that
//! followed with [`view_from`](ColumnEdits::view_from). Whole-row operations
//! (`drain` / `pop_front` / `truncate` / `retain`) keep the per-entry logs
//! aligned with the live entries, so a row index stays valid across them.

use crate::editlog::{EditLog, EditLogError, EditLogView};
use std::ops::Range;

#[derive(Clone, Debug)]
enum Inner {
    /// One log shared by every entry: no per-entry edit has happened yet.
    Uniform(EditLog),
    /// One log per live entry, indexed in view order.
    PerEntry(Vec<EditLog>),
}

/// Column-wide coordinate-edit history, promoting from one shared
/// [`EditLog`](crate::EditLog) to one-per-entry on the first non-uniform edit.
/// See the [module docs](crate::column) for the model.
#[derive(Clone, Debug)]
pub struct ColumnEdits {
    inner: Inner,
    len: usize,
}

impl ColumnEdits {
    /// A column of `len` entries with no recorded edits (identity liftover).
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            inner: Inner::Uniform(EditLog::new()),
            len,
        }
    }

    /// Number of live entries tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if no entries are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` while still sharing one log across all entries — i.e. no
    /// per-entry edit has forced promotion yet.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        matches!(self.inner, Inner::Uniform(_))
    }

    /// Materialise one log per live entry (cloning the shared history into
    /// each). Idempotent once promoted.
    fn promote(&mut self) {
        if let Inner::Uniform(log) = &self.inner {
            self.inner = Inner::PerEntry(vec![log.clone(); self.len]);
        }
    }

    /// Record an edit that applies to **every** entry — a whole-column op such
    /// as an unconditional `cut_start`/`cut_end`/`prefix`/`postfix`/`reverse`.
    /// Stays in the cheap uniform state.
    pub fn apply_all(&mut self, mut record: impl FnMut(&mut EditLog)) {
        match &mut self.inner {
            Inner::Uniform(log) => record(log),
            Inner::PerEntry(logs) => logs.iter_mut().for_each(record),
        }
    }

    /// Record a per-entry edit: promotes to per-entry, then calls `record` once
    /// for every live entry, which decides what (if anything) that entry sees.
    /// This covers conditional cuts (record only where the mask is true),
    /// `max_len`, and `resize` (each entry gets its own window cut).
    pub fn apply_entries(&mut self, mut record: impl FnMut(usize, &mut EditLog)) {
        self.promote();
        if let Inner::PerEntry(logs) = &mut self.inner {
            for (row, log) in logs.iter_mut().enumerate() {
                record(row, log);
            }
        }
    }

    /// Grow the column by one fresh entry (mirrors `StringPod::push` appending
    /// to a finished pod). The newcomer has identity history, so if the shared
    /// uniform log already carries edits this promotes to per-entry first — the
    /// new entry didn't experience that history and must not inherit it.
    pub fn push_entry(&mut self) {
        let needs_own_log = match &self.inner {
            Inner::Uniform(log) => !log.is_empty(),
            Inner::PerEntry(_) => true,
        };
        if needs_own_log {
            self.promote();
            if let Inner::PerEntry(logs) = &mut self.inner {
                logs.push(EditLog::new());
            }
        }
        self.len += 1;
    }

    /// Record an edit touching a single entry — e.g. a tag write-back `splice`
    /// into one read. Promotes to per-entry; a no-op if `row` is out of range.
    pub fn apply_entry(&mut self, row: usize, record: impl FnOnce(&mut EditLog)) {
        if row >= self.len {
            return;
        }
        self.promote();
        if let Inner::PerEntry(logs) = &mut self.inner {
            record(&mut logs[row]);
        }
    }

    /// The current generation (count of recorded edits) of entry `row`, to be
    /// captured when a coordinate snapshot is taken. `None` if `row` is out of
    /// range. In the uniform state every row reports the same value.
    #[must_use]
    pub fn generation(&self, row: usize) -> Option<usize> {
        if row >= self.len {
            return None;
        }
        Some(match &self.inner {
            Inner::Uniform(log) => log.op_count(),
            Inner::PerEntry(logs) => logs[row].op_count(),
        })
    }

    /// A view over the edits applied to entry `row` at or after `born` (a
    /// generation previously read from [`generation`](Self::generation)).
    /// Replaying it lifts that entry's birth-frame coordinate into the current
    /// frame; pass the entry's length *at `born`* as `orig_len`.
    ///
    /// # Errors
    /// [`EditLogError::RowOutOfBounds`] if `row >= self.len()`;
    /// [`EditLogError::GenerationOutOfRange`] if `born` exceeds that entry's
    /// recorded edits.
    pub fn view_from(&self, row: usize, born: usize) -> Result<EditLogView<'_>, EditLogError> {
        if row >= self.len {
            return Err(EditLogError::RowOutOfBounds { row, len: self.len });
        }
        match &self.inner {
            Inner::Uniform(log) => log.view_from(born),
            Inner::PerEntry(logs) => logs[row].view_from(born),
        }
    }

    /// A copy of this column's edit history restricted to live entries `range`,
    /// in view order. Mirrors [`StringPod::slice`](crate::StringPod::slice): the
    /// sliced column carries exactly the edits of the entries it keeps. Stays in
    /// the cheap uniform state when the source is uniform.
    ///
    /// # Panics
    /// If `range.start > range.end` or `range.end > self.len()`.
    #[must_use]
    pub fn slice(&self, range: Range<usize>) -> ColumnEdits {
        assert!(range.start <= range.end, "slice range start > end");
        assert!(range.end <= self.len, "slice range past end of column");
        let len = range.end - range.start;
        let inner = match &self.inner {
            Inner::Uniform(log) => Inner::Uniform(log.clone()),
            Inner::PerEntry(logs) => Inner::PerEntry(logs[range].to_vec()),
        };
        ColumnEdits { inner, len }
    }

    // ── whole-row (entry-axis) operations ───────────────────────────────────
    //
    // These change *which* entries are live, not their coordinate frames, so
    // they don't record edits — they just keep the per-entry logs row-aligned
    // with the pod's live entries. Clamped like the pod's own storage.

    /// Drop the logs for a contiguous range of live entries.
    pub fn drain(&mut self, range: Range<usize>) {
        let start = range.start.min(self.len);
        let end = range.end.min(self.len);
        if start >= end {
            return;
        }
        if let Inner::PerEntry(logs) = &mut self.inner {
            logs.drain(start..end);
        }
        self.len -= end - start;
    }

    /// Drop the logs for the first `n` live entries.
    pub fn pop_front(&mut self, n: usize) {
        let n = n.min(self.len);
        if let Inner::PerEntry(logs) = &mut self.inner {
            logs.drain(0..n);
        }
        self.len -= n;
    }

    /// Keep only the first `len` live entries.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        if let Inner::PerEntry(logs) = &mut self.inner {
            logs.truncate(len);
        }
        self.len = len;
    }

    /// Keep only entries whose bool is `true` (parallel to the live entries).
    pub fn retain(&mut self, keep: &[bool]) {
        if let Inner::PerEntry(logs) = &mut self.inner {
            let mut it = keep.iter();
            logs.retain(|_| it.next().copied().unwrap_or(false));
        }
        self.len = keep.iter().filter(|b| **b).count();
    }
}
