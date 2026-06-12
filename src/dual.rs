use bstr::{BStr, ByteSlice as _};
use std::ops::Range;
use std::sync::Arc;

use crate::column::ColumnEdits;
use crate::lifted::Lifted;
use crate::single::StringPod;
use crate::storage::{Storage, VariableInfo};

/// Error returned by [`DualStringPod::try_from_columns`] when two columns
/// cannot be fused into one [`DualStringPod`] without copying — i.e. they are
/// not a valid sequence + quality pair sharing a single metadata layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnError {
    /// The two columns have a different number of entries.
    EqualCountViolated,
    /// Per-entry counts match, but some record's sequence and quality strings
    /// have different lengths.
    UnequalLengths,
    /// Per-entry lengths match, but the two byte layouts are not a constant
    /// translation of each other (entry starts differ by a non-constant
    /// offset), so they cannot share one metadata column. In practice this
    /// means the input isn't well-formed FASTQ.
    NotATranslation,
}

impl std::fmt::Display for ColumnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            ColumnError::EqualCountViolated => {
                "sequence and quality columns have a different number of entries"
            }
            ColumnError::UnequalLengths => {
                "sequence/quality entry length mismatch: a record's seq and qual differ in length"
            }
            ColumnError::NotATranslation => {
                "sequence and quality byte layouts are not a constant translation of each other"
            }
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ColumnError {}

/// Two parallel byte columns (e.g. sequence + quality) sharing a single
/// metadata layout. The shared metadata makes the per-entry length invariant
/// (`seq.len() == qual.len()` for every entry) structural rather than
/// runtime-checked. Each byte buffer is independently `Arc<[u8]>` so seq
/// can be aliased into a tag column without pinning qual, and vice versa.
#[derive(Clone)]
pub struct DualStringPod {
    // `Arc<Vec<u8>>` (not `Arc<[u8]>`) so `finish` wraps the builder Vecs as-is
    // rather than reallocating away any over-reserved capacity.
    pub(crate) seq: Arc<Vec<u8>>,
    pub(crate) qual: Arc<Vec<u8>>,
    pub(crate) storage: Storage,

    pub(crate) seq_first_byte: usize,
    pub(crate) qual_first_byte: usize,

    /// Per-entry coordinate-edit history for liftover (see [`Lifted`]).
    pub(crate) edits: ColumnEdits,
}

impl Lifted for DualStringPod {
    fn edits(&self) -> &ColumnEdits {
        &self.edits
    }
    fn edits_mut(&mut self) -> &mut ColumnEdits {
        &mut self.edits
    }
}

impl DualStringPod {
    #[must_use]
    pub fn empty() -> Self {
        DualStringPodBuilder::with_capacity(0, 0).finish()
    }

    /// Fuse two columns (e.g. FASTQ sequence + quality) into a single dual pod
    /// **without copying any bytes**: both byte buffers are moved in as-is and a
    /// single shared metadata column describes their *relative* per-entry
    /// layout. The two `*_first_byte` offsets record where each column's entry 0
    /// actually starts, so the buffers may live at a constant offset from each
    /// other (the "stray bytes at the start" from the 'pull into previous block'
    /// schema) and still share one metadata column.
    ///
    /// This is the zero-copy bridge from a columnar split that builds `seq` and
    /// `qual` as independent [`StringPod`]s to the structural
    /// `seq[i].len() == qual[i].len()` invariant a [`DualStringPod`] enforces.
    ///
    /// The two columns must have the same entry count and, for every entry, the
    /// same length. Their entry starts must also be a constant translation of
    /// each other — which, for lockstep-built columns, holds exactly when the
    /// input is well-formed FASTQ.
    ///
    /// # Errors
    ///
    /// - [`ColumnError::EqualCountViolated`] when `seq.len() != qual.len()`.
    /// - [`ColumnError::UnequalLengths`] when some record's seq and qual differ
    ///   in length.
    /// - [`ColumnError::NotATranslation`] when the per-entry starts are not a
    ///   constant translation of each other — at which point it's time to tell
    ///   the consumer that this ain't FASTQ.
    ///
    /// # Panics
    /// When exceeding u32 range
    pub fn try_from_columns(seq: StringPod, qual: StringPod) -> Result<Self, ColumnError> {
        if seq.len() != qual.len() {
            return Err(ColumnError::EqualCountViolated);
        }
        let count = seq.len();

        // Fast path: both columns are fixed-length with an identical *relative*
        // layout (same stride / cut overlay / count). Only their `front_byte`
        // may differ — that constant offset is captured by the two
        // `*_first_byte` fields, so we keep the strided metadata (true O(1),
        // no positions vec) and share it between both buffers.
        if let (
            Storage::FixedLength {
                stride: seq_stride,
                head_skip: seq_head_skip,
                visible_len: seq_visible_len,
                count: seq_count,
                front_byte: seq_front_byte,
            },
            Storage::FixedLength {
                stride: qual_stride,
                head_skip: qual_head_skip,
                visible_len: qual_visible_len,
                front_byte: qual_front_byte,
                ..
            },
        ) = (&seq.storage, &qual.storage)
        {
            if seq_stride == qual_stride
                && seq_head_skip == qual_head_skip
                && seq_visible_len == qual_visible_len
            {
                let storage = Storage::FixedLength {
                    stride: *seq_stride,
                    head_skip: *seq_head_skip,
                    visible_len: *seq_visible_len,
                    count: *seq_count,
                    // Entry 0 is now relative to each column's own first byte.
                    front_byte: 0,
                };
                return Ok(DualStringPod {
                    seq: seq.data,
                    qual: qual.data,
                    storage,
                    seq_first_byte: *seq_front_byte as usize,
                    qual_first_byte: *qual_front_byte as usize,
                    edits: ColumnEdits::new(count),
                });
            }
        }

        // General path: validate the per-entry length invariant and the
        // constant-translation invariant in a single pass, building a shared
        // metadata column whose positions are relative to each column's entry-0
        // start. Bytes are never touched — only this positions vec is built.
        let n = seq.len();
        let mut positions: Vec<(u32, u32)> = Vec::with_capacity(n);
        let mut base_seq = 0usize;
        let mut base_qual = 0usize;
        for i in 0..n {
            let sr = seq.storage.entry_range(i);
            let qr = qual.storage.entry_range(i);
            if (sr.end - sr.start) != (qr.end - qr.start) {
                return Err(ColumnError::UnequalLengths);
            }
            if i == 0 {
                base_seq = sr.start;
                base_qual = qr.start;
            } else if (sr.start - base_seq) != (qr.start - base_qual) {
                // Entry starts drift apart — not a constant translation.
                return Err(ColumnError::NotATranslation);
            }
            let rel_start =
                u32::try_from(sr.start - base_seq).expect("entry start exceeds u32::MAX");
            let rel_end = u32::try_from(sr.end - base_seq).expect("entry end exceeds u32::MAX");
            positions.push((rel_start, rel_end));
        }

        let storage = Storage::Variable(VariableInfo {
            positions,
            head_skip: 0,
            tail_skip: 0,
            front_skip: 0,
        });
        Ok(DualStringPod {
            seq: seq.data,
            qual: qual.data,
            storage,
            seq_first_byte: base_seq,
            qual_first_byte: base_qual,
            edits: ColumnEdits::new(count),
        })
    }

    /// Ensure this pod owns its **seq** buffer outright, cloning it (COW) only
    /// if it is currently shared with another pod. After this call, seq-mutating
    /// access such as [`seq_mut`](Self::seq_mut) / [`iter_seq_mut`](Self::iter_seq_mut)
    /// always succeeds. Leaves the qual buffer untouched (still shareable).
    pub fn make_seq_exclusive(&mut self) {
        // `Arc::make_mut` clones a buffer iff its strong count is > 1.
        let _ = Arc::make_mut(&mut self.seq);
    }

    /// Ensure this pod owns its **qual** buffer outright, cloning it (COW) only
    /// if it is currently shared with another pod. After this call, qual-mutating
    /// access such as [`qual_mut`](Self::qual_mut) / [`iter_qual_mut`](Self::iter_qual_mut)
    /// always succeeds. Leaves the seq buffer untouched (still shareable).
    pub fn make_qual_exclusive(&mut self) {
        let _ = Arc::make_mut(&mut self.qual);
    }

    /// Ensure this pod owns both byte buffers outright, cloning each (COW) only
    /// if it is currently shared with another pod. After this call, mutating
    /// accessors such as [`seq_mut`](Self::seq_mut) / [`iter_mut`](Self::iter_mut)
    /// always succeed. Equivalent to calling both
    /// [`make_seq_exclusive`](Self::make_seq_exclusive) and
    /// [`make_qual_exclusive`](Self::make_qual_exclusive).
    pub fn make_exclusive(&mut self) {
        self.make_seq_exclusive();
        self.make_qual_exclusive();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Sequence bytes of entry `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn seq(&self, i: usize) -> &BStr {
        let r = self.storage.entry_range(i);
        BStr::new(&self.seq[r.start + self.seq_first_byte..r.end + self.seq_first_byte])
    }

    /// Quality bytes of entry `i`. Shares the same range as `seq(i)`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn qual(&self, i: usize) -> &BStr {
        let r = self.storage.entry_range(i);
        BStr::new(&self.qual[r.start + self.qual_first_byte..r.end + self.qual_first_byte])
    }

    /// Both bytes of entry `i` at once.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn pair(&self, i: usize) -> (&BStr, &BStr) {
        let r = self.storage.entry_range(i);
        (
            BStr::new(&self.seq[r.start + self.seq_first_byte..r.end + self.seq_first_byte]),
            BStr::new(&self.qual[r.start + self.qual_first_byte..r.end + self.qual_first_byte]),
        )
    }

    /// Both bytes of entry `i` at once.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn pair_mut(&mut self, i: usize) -> Option<(&mut BStr, &mut BStr)> {
        let r = self.storage.entry_range(i);
        let seqs = Arc::get_mut(&mut self.seq)?;
        let quals = Arc::get_mut(&mut self.qual)?;
        Some((
            seqs[r.start + self.seq_first_byte..r.end + self.seq_first_byte].as_bstr_mut(),
            quals[r.start + self.qual_first_byte..r.end + self.qual_first_byte].as_bstr_mut(),
        ))
    }

    /// Mutable access to the sequence bytes of entry `i`, or `None` if the
    /// sequence buffer is shared (Arc strong count > 1). Drop or release other
    /// references before retrying, or rebuild into a fresh pod.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn seq_mut(&mut self, i: usize) -> Option<&mut BStr> {
        let data = Arc::get_mut(&mut self.seq)?;
        let r = self.storage.entry_range(i);
        let range = (r.start + self.seq_first_byte)..(r.end + self.seq_first_byte);
        Some(data[range].as_bstr_mut())
    }

    /// Mutable access to the quality bytes of entry `i`, or `None` if the
    /// quality buffer is shared (Arc strong count > 1). Drop or release other
    /// references before retrying, or rebuild into a fresh pod.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn qual_mut(&mut self, i: usize) -> Option<&mut BStr> {
        let data = Arc::get_mut(&mut self.qual)?;
        let r = self.storage.entry_range(i);
        let range = (r.start + self.qual_first_byte)..(r.end + self.qual_first_byte);
        Some(data[range].as_bstr_mut())
    }

    #[must_use]
    pub fn entry_len(&self, i: usize) -> usize {
        self.storage.entry_len(i)
    }

    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.storage.used_bytes()
    }

    /// Size of the seq buffer (qual is guaranteed identical in length).
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.seq.len()
    }

    #[must_use]
    pub fn is_fixed_length(&self) -> bool {
        self.storage.current_stride().is_some()
    }

    pub fn cut_start(&mut self, n: usize, conditional: Option<&[bool]>) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_start(n_u32, conditional);
        self.record_cut_start(n, conditional);
    }

    /// Cut `n` bytes off the end of every entry (both seq and qual). O(1).
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are affected; this promotes `FixedLength` → `Variable`.
    pub fn cut_end(&mut self, n: usize, conditional: Option<&[bool]>) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_end(n_u32, conditional);
        self.record_cut_end(n, conditional);
    }

    /// Drop the first `n` entries from the view. O(1): a byte offset on
    /// `FixedLength`, an entry-index skip on `Variable`. No bytes move.
    pub fn pop_front(&mut self, n: usize) {
        self.storage.pop_front(u32::try_from(n).unwrap_or(u32::MAX));
        self.record_pop_front(n);
    }

    /// Truncate the view to at most `len` entries (drops from the back). O(1).
    pub fn truncate(&mut self, len: usize) {
        self.storage.truncate(len);
        self.record_truncate(len);
    }

    /// # Panics
    /// If `range.end > self.len()` or `range.start > range.end`.
    pub fn drain(&mut self, range: Range<usize>) {
        assert!(range.start <= range.end, "drain range start > end");
        assert!(range.end <= self.len(), "drain range past end of pod");
        let (start, end) = (range.start, range.end);
        self.storage.drain(range);
        self.record_drain(start..end);
    }

    #[must_use]
    /// Iterate over (seq, qual) tuples
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            pod: self,
            front: 0,
            back: self.len(),
        }
    }

    /// Iterator yielding only the sequence column.
    #[must_use]
    pub fn iter_seq(&self) -> SeqIter<'_> {
        SeqIter {
            pod: self,
            front: 0,
            back: self.len(),
        }
    }

    /// Iterator yielding only the quality column.
    #[must_use]
    pub fn iter_qual(&self) -> QualIter<'_> {
        QualIter {
            pod: self,
            front: 0,
            back: self.len(),
        }
    }

    pub fn iter_seq_lens(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(move |i| self.storage.entry_len(i))
    }

    /// Prepend `seq_text` / `qual_text` to every entry in the respective
    /// columns, rebuilding both buffers. The two texts must have the same
    /// length (so the `seq.len() == qual.len()` invariant is preserved).
    ///
    /// # Panics
    /// If `seq_text.len() != qual_text.len()`.
    #[must_use]
    pub fn prefix(self, seq_text: &[u8], qual_text: &[u8]) -> Self {
        assert_eq!(
            seq_text.len(),
            qual_text.len(),
            "seq_text.len() {} != qual_text.len() {}",
            seq_text.len(),
            qual_text.len()
        );
        let n = self.len();
        let first_len = if n > 0 {
            self.entry_len(0) + seq_text.len()
        } else {
            seq_text.len()
        };
        let mut bld = DualStringPodBuilder::with_capacity(first_len, n);
        let mut seq_buf = Vec::with_capacity(first_len);
        let mut qual_buf = Vec::with_capacity(first_len);
        for i in 0..n {
            seq_buf.clear();
            seq_buf.extend_from_slice(seq_text);
            seq_buf.extend_from_slice(self.seq(i));
            qual_buf.clear();
            qual_buf.extend_from_slice(qual_text);
            qual_buf.extend_from_slice(self.qual(i));
            bld.push(&seq_buf, &qual_buf);
        }
        // Carry the edit history across the rebuild (the Arc diverges here, so
        // the snapshot bytes part company from the live ones) and append the op.
        let mut out = bld.finish();
        out.edits = self.edits;
        out.record_prefix(seq_text.len());
        out
    }

    /// Append `seq_text` / `qual_text` to every entry in the respective
    /// columns, rebuilding both buffers. The two texts must have the same
    /// length.
    ///
    /// # Panics
    /// If `seq_text.len() != qual_text.len()`.
    #[must_use]
    pub fn postfix(self, seq_text: &[u8], qual_text: &[u8]) -> Self {
        assert_eq!(
            seq_text.len(),
            qual_text.len(),
            "seq_text.len() {} != qual_text.len() {}",
            seq_text.len(),
            qual_text.len()
        );
        let n = self.len();
        let first_len = if n > 0 {
            self.entry_len(0) + seq_text.len()
        } else {
            seq_text.len()
        };
        let mut bld = DualStringPodBuilder::with_capacity(first_len, n);
        let mut seq_buf = Vec::with_capacity(first_len);
        let mut qual_buf = Vec::with_capacity(first_len);
        for i in 0..n {
            seq_buf.clear();
            seq_buf.extend_from_slice(self.seq(i));
            seq_buf.extend_from_slice(seq_text);
            qual_buf.clear();
            qual_buf.extend_from_slice(self.qual(i));
            qual_buf.extend_from_slice(qual_text);
            bld.push(&seq_buf, &qual_buf);
        }
        let mut out = bld.finish();
        out.edits = self.edits;
        out.record_postfix(seq_text.len());
        out
    }

    /// Apply a per-entry length-changing write-back (splice). For each entry `i`,
    /// `edits[i] = Some((at, del, ins_seq, ins_qual))` replaces the `del` bytes at
    /// offset `at` with `ins_seq` / `ins_qual` (equal length); `None` leaves the
    /// entry untouched. `at`/`del` are in the entry's *current* frame.
    ///
    /// Both buffers are rebuilt (a mid-entry length change can't live in an
    /// overlay). The edit history is carried across the rebuild and each splice is
    /// recorded, so any tag whose coordinates were captured earlier lifts through
    /// the change — the whole point of recording rather than rebuilding blind.
    ///
    /// # Panics
    /// - If `edits.len() != self.len()`.
    /// - If any `ins_seq.len() != ins_qual.len()`.
    /// - If any `at + del` exceeds the entry's current length.
    pub fn splice_entries(&mut self, edits: &[Option<(usize, usize, Vec<u8>, Vec<u8>)>]) {
        assert_eq!(
            edits.len(),
            self.len(),
            "splice edits length {} must match entry count {}",
            edits.len(),
            self.len(),
        );
        let n = self.len();
        let mut bld = DualStringPodBuilder::with_capacity(0, n);
        let mut seq_buf = Vec::new();
        let mut qual_buf = Vec::new();
        for i in 0..n {
            match &edits[i] {
                None => bld.push(self.seq(i), self.qual(i)),
                Some((at, del, ins_seq, ins_qual)) => {
                    assert_eq!(
                        ins_seq.len(),
                        ins_qual.len(),
                        "splice ins_seq.len() {} != ins_qual.len() {}",
                        ins_seq.len(),
                        ins_qual.len(),
                    );
                    let (seq, qual) = (self.seq(i), self.qual(i));
                    let (at, del) = (*at, *del);
                    assert!(
                        at + del <= seq.len(),
                        "splice {at}+{del} exceeds entry {i} length {}",
                        seq.len(),
                    );
                    seq_buf.clear();
                    seq_buf.extend_from_slice(&seq[..at]);
                    seq_buf.extend_from_slice(ins_seq);
                    seq_buf.extend_from_slice(&seq[at + del..]);
                    qual_buf.clear();
                    qual_buf.extend_from_slice(&qual[..at]);
                    qual_buf.extend_from_slice(ins_qual);
                    qual_buf.extend_from_slice(&qual[at + del..]);
                    bld.push(&seq_buf, &qual_buf);
                }
            }
        }
        let mut out = bld.finish();
        // Carry the edit history across the rebuild, then record each splice so
        // later liftover sees the length change.
        out.edits = std::mem::replace(&mut self.edits, ColumnEdits::new(0));
        for (i, e) in edits.iter().enumerate() {
            if let Some((at, del, ins_seq, _)) = e {
                out.record_splice(i, *at, *del, ins_seq.len());
            }
        }
        *self = out;
    }

    /// Truncate every entry to at most `n` bytes. O(1) for `FixedLength`
    /// pods; rebuilds both buffers for `Variable` pods.
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are clipped; this promotes `FixedLength` → `Variable`.
    pub fn max_len(&mut self, n: usize, conditional: Option<&[bool]>) {
        let count = self.len();
        // Capture each entry's keep-window from its *current* length, before any
        // mutation; "keep first min(len, n)". Recorded once at the end so the
        // FixedLength fast path and the Variable rebuild log identically.
        let windows: Vec<Option<(usize, usize, usize)>> = (0..count)
            .map(|i| {
                let affected = conditional.is_none_or(|c| c[i]);
                let cur = self.entry_len(i);
                (affected && cur > n).then_some((0, n, cur))
            })
            .collect();

        if let Some(cond) = conditional {
            let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
            self.storage.truncate_bytes_conditional(n_u32, cond);
        } else {
            if let Storage::FixedLength { visible_len, .. } = self.storage {
                let vl = visible_len as usize;
                if vl > n {
                    // storage-level cut: the coordinate edit is logged via
                    // `record_windows` below, so don't double-record here.
                    self.storage
                        .cut_end(u32::try_from(vl - n).unwrap_or(u32::MAX), None);
                }
            }
            let mut bld = DualStringPodBuilder::with_capacity(n, count);
            let mut seq_buf = Vec::with_capacity(n);
            let mut qual_buf = Vec::with_capacity(n);
            for i in 0..count {
                let s = self.seq(i);
                let q = self.qual(i);
                let len = s.len().min(n);
                seq_buf.clear();
                seq_buf.extend_from_slice(&s[..len]);
                qual_buf.clear();
                qual_buf.extend_from_slice(&q[..len]);
                bld.push(&seq_buf, &qual_buf);
            }
            let temp = bld.finish();
            self.seq = temp.seq;
            self.qual = temp.qual;
            self.storage = temp.storage;
        }
        self.record_windows(&windows);
    }

    /// Generalized per-entry resize. For each entry `i` (in order), invokes
    /// `f(i, seq, qual)` with the entry's current visible sequence and quality
    /// bytes. The closure returns:
    ///
    /// * `None` — leave the entry unchanged, or
    /// * `Some((start, len))` — narrow the entry to `[start..start + len]`. The
    ///   sub-range is relative to the entry's current visible bytes and **must
    ///   lie within them** (`start + len <= seq.len()`, equivalently
    ///   `qual.len()`); otherwise this panics.
    ///
    /// The single returned range applies to *both* columns, preserving the
    /// structural `seq.len() == qual.len()` invariant. No bytes are copied —
    /// only the shared metadata is rewritten — so the new region must fall
    /// inside the old one. Promotes `FixedLength` → `Variable`.
    pub fn resize<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &BStr, &BStr) -> Option<(usize, usize)>,
    {
        let seq = &self.seq;
        let qual = &self.qual;
        let sf = self.seq_first_byte;
        let qf = self.qual_first_byte;
        let mut windows: Vec<Option<(usize, usize, usize)>> = Vec::new();
        self.storage.resize_positions(|i, start, stop| {
            let (s, e) = (start as usize, stop as usize);
            let cur = e - s;
            let seq_bytes = BStr::new(&seq[s + sf..e + sf]);
            let qual_bytes = BStr::new(&qual[s + qf..e + qf]);
            let kept = f(i, seq_bytes, qual_bytes);
            windows.push(kept.map(|(st, ln)| (st, ln, cur)));
            kept
        });
        self.record_windows(&windows);
    }

    pub fn retain_by_bools(&mut self, keep: &[bool]) {
        self.storage.retain_by_bools(keep);
        self.record_retain(keep);
    }

    /// Reverse the bytes of every entry in both columns in-place. If either
    /// buffer is shared (Arc strong count > 1) it is cloned before reversing
    /// (COW).
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are reversed.
    #[must_use]
    pub fn reverse(mut self, conditional: Option<&[bool]>) -> Self {
        self.record_reverse(conditional);
        let seq_first_byte = self.seq_first_byte;
        let qual_first_byte = self.qual_first_byte;
        let n = self.storage.len();
        let seq = Arc::make_mut(&mut self.seq);
        let qual = Arc::make_mut(&mut self.qual);
        for i in 0..n {
            if conditional.is_none_or(|c| c[i]) {
                let r = self.storage.entry_range(i);
                seq[r.start + seq_first_byte..r.end + seq_first_byte].reverse();
                qual[r.start + qual_first_byte..r.end + qual_first_byte].reverse();
            }
        }
        self
    }

    /// Mutable iterator over seq+qual entry pairs. Always succeeds: it COWs
    /// either shared buffer first (the [`make_exclusive`](Self::make_exclusive)
    /// step), so sharing with another pod no longer blocks iteration. Also
    /// reachable as `(&mut pod).into_iter()`.
    #[must_use]
    pub fn iter_mut(&mut self) -> DualIterMut<'_> {
        let back = self.storage.len();
        let seq_first_byte = self.seq_first_byte;
        let qual_first_byte = self.qual_first_byte;
        // Three disjoint fields: seq + qual mutably, storage immutably. Skip
        // each buffer's first-byte prefix so both `remaining` slices start at
        // the same *relative* offset (storage's coordinate system), letting one
        // shared `consumed` drive both peels. `Arc::make_mut` is the COW: it
        // clones iff the buffer is shared, then hands back a unique `&mut`.
        let (_, seq_remaining) = Arc::make_mut(&mut self.seq)
            .as_mut_slice()
            .split_at_mut(seq_first_byte);
        let (_, qual_remaining) = Arc::make_mut(&mut self.qual)
            .as_mut_slice()
            .split_at_mut(qual_first_byte);
        DualIterMut {
            seq_remaining,
            qual_remaining,
            storage: &self.storage,
            front: 0,
            back,
            consumed: 0,
        }
    }

    /// Mutable iterator over just the **seq** entries, yielding `&mut BStr` per
    /// entry. Always succeeds: COWs only the seq buffer first (the
    /// [`make_seq_exclusive`](Self::make_seq_exclusive) step), leaving qual
    /// shareable.
    #[must_use]
    pub fn iter_seq_mut(&mut self) -> ColIterMut<'_> {
        let back = self.storage.len();
        let first = self.seq_first_byte;
        let (_, remaining) = Arc::make_mut(&mut self.seq)
            .as_mut_slice()
            .split_at_mut(first);
        ColIterMut {
            remaining,
            storage: &self.storage,
            front: 0,
            back,
            consumed: 0,
        }
    }

    /// Mutable iterator over just the **qual** entries, yielding `&mut BStr` per
    /// entry. Always succeeds: COWs only the qual buffer first (the
    /// [`make_qual_exclusive`](Self::make_qual_exclusive) step), leaving seq
    /// shareable.
    #[must_use]
    pub fn iter_qual_mut(&mut self) -> ColIterMut<'_> {
        let back = self.storage.len();
        let first = self.qual_first_byte;
        let (_, remaining) = Arc::make_mut(&mut self.qual)
            .as_mut_slice()
            .split_at_mut(first);
        ColIterMut {
            remaining,
            storage: &self.storage,
            front: 0,
            back,
            consumed: 0,
        }
    }

    /// Start an alias builder sharing both byte buffers.
    ///
    /// The builder borrows `self` until [`DualStringPodAliasBuilder::finish`]
    /// is called; the source pod cannot be mutated while the builder is live.
    #[must_use]
    pub fn alias_builder(&self) -> DualStringPodAliasBuilder<'_> {
        DualStringPodAliasBuilder {
            source: self,
            next: 0,
            positions: Vec::new(),
        }
    }
}

impl std::fmt::Debug for DualStringPod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DualStringPod")
            .field("len", &self.len())
            .field("fixed_length", &self.is_fixed_length())
            .field("buffer_bytes", &self.buffer_bytes())
            .field("used_bytes", &self.used_bytes())
            .finish()
    }
}

impl<'a> IntoIterator for &'a DualStringPod {
    type Item = DualEntry<'a>;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut DualStringPod {
    type Item = DualEntryMut<'a>;
    type IntoIter = DualIterMut<'a>;
    fn into_iter(self) -> DualIterMut<'a> {
        self.iter_mut()
    }
}

pub struct DualEntry<'a> {
    pub seq: &'a BStr,
    pub qual: &'a BStr,
}

pub struct Iter<'a> {
    pod: &'a DualStringPod,
    front: usize,
    back: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = DualEntry<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let (seq, qual) = self.pod.pair(self.front);
            self.front += 1;
            Some(DualEntry { seq, qual })
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.back - self.front;
        (r, Some(r))
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            let (seq, qual) = self.pod.pair(self.back);
            Some(DualEntry { seq, qual })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Iter<'_> {}

pub struct SeqIter<'a> {
    pod: &'a DualStringPod,
    front: usize,
    back: usize,
}

impl<'a> Iterator for SeqIter<'a> {
    type Item = &'a BStr;
    fn next(&mut self) -> Option<&'a BStr> {
        if self.front < self.back {
            let s = self.pod.seq(self.front);
            self.front += 1;
            Some(s)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.back - self.front;
        (r, Some(r))
    }
}

impl ExactSizeIterator for SeqIter<'_> {}

pub struct QualIter<'a> {
    pod: &'a DualStringPod,
    front: usize,
    back: usize,
}

impl<'a> Iterator for QualIter<'a> {
    type Item = &'a BStr;
    fn next(&mut self) -> Option<&'a BStr> {
        if self.front < self.back {
            let q = self.pod.qual(self.front);
            self.front += 1;
            Some(q)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.back - self.front;
        (r, Some(r))
    }
}

impl ExactSizeIterator for QualIter<'_> {}

pub struct DualEntryMut<'a> {
    pub seq: &'a mut BStr,
    pub qual: &'a mut BStr,
}

/// Mutable seq+qual entry iterator. Like [`single::IterMut`](crate::StringPodIterMut)
/// it peels entries off held buffer slices with [`split_at_mut`](slice::split_at_mut)
/// — no `unsafe`. Both buffers share one storage layout (offset by their
/// first-byte), so one relative `consumed` cursor drives both. Relies on the
/// ascending, non-overlapping entry order every builder path upholds.
pub struct DualIterMut<'a> {
    seq_remaining: &'a mut [u8],
    qual_remaining: &'a mut [u8],
    storage: &'a Storage,
    front: usize,
    back: usize,
    /// Relative offset (storage coordinates) of both `remaining` slices' starts.
    consumed: usize,
}

impl<'a> Iterator for DualIterMut<'a> {
    type Item = DualEntryMut<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let r = self.storage.entry_range(self.front);
        self.front += 1;
        debug_assert!(
            r.start >= self.consumed,
            "DualIterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
            r.start,
            self.consumed,
        );
        let gap = r.start - self.consumed;
        let len = r.end - r.start;
        let seq_rem = std::mem::take(&mut self.seq_remaining);
        let qual_rem = std::mem::take(&mut self.qual_remaining);
        let (_seq_gap, seq_rest) = seq_rem.split_at_mut(gap);
        let (seq, seq_tail) = seq_rest.split_at_mut(len);
        let (_qual_gap, qual_rest) = qual_rem.split_at_mut(gap);
        let (qual, qual_tail) = qual_rest.split_at_mut(len);
        self.seq_remaining = seq_tail;
        self.qual_remaining = qual_tail;
        self.consumed = r.end;
        Some(DualEntryMut {
            seq: seq.as_bstr_mut(),
            qual: qual.as_bstr_mut(),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.back - self.front;
        (r, Some(r))
    }
}

impl DoubleEndedIterator for DualIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let r = self.storage.entry_range(self.back);
        debug_assert!(
            r.start >= self.consumed,
            "DualIterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
            r.start,
            self.consumed,
        );
        let split = r.start - self.consumed;
        let len = r.end - r.start;
        let seq_rem = std::mem::take(&mut self.seq_remaining);
        let qual_rem = std::mem::take(&mut self.qual_remaining);
        let (seq_left, seq_right) = seq_rem.split_at_mut(split);
        let (seq, _seq_after) = seq_right.split_at_mut(len);
        let (qual_left, qual_right) = qual_rem.split_at_mut(split);
        let (qual, _qual_after) = qual_right.split_at_mut(len);
        self.seq_remaining = seq_left;
        self.qual_remaining = qual_left;
        Some(DualEntryMut {
            seq: seq.as_bstr_mut(),
            qual: qual.as_bstr_mut(),
        })
    }
}

impl ExactSizeIterator for DualIterMut<'_> {}

/// Mutable iterator over a single column's entries (`&mut BStr` apiece). Backs
/// [`DualStringPod::iter_seq_mut`] and [`iter_qual_mut`](DualStringPod::iter_qual_mut);
/// it peels entries off the held buffer slice with [`split_at_mut`](slice::split_at_mut)
/// — no `unsafe`. The `remaining` slice starts past the column's first-byte
/// prefix, so `consumed` runs in storage's relative coordinate system.
pub struct ColIterMut<'a> {
    remaining: &'a mut [u8],
    storage: &'a Storage,
    front: usize,
    back: usize,
    /// Relative offset (storage coordinates) of `remaining[0]`; advances as the
    /// front is peeled.
    consumed: usize,
}

impl<'a> Iterator for ColIterMut<'a> {
    type Item = &'a mut BStr;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let r = self.storage.entry_range(self.front);
        self.front += 1;
        debug_assert!(
            r.start >= self.consumed,
            "ColIterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
            r.start,
            self.consumed,
        );
        let rem = std::mem::take(&mut self.remaining);
        let (_gap, rest) = rem.split_at_mut(r.start - self.consumed);
        let (entry, tail) = rest.split_at_mut(r.end - r.start);
        self.remaining = tail;
        self.consumed = r.end;
        Some(entry.as_bstr_mut())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.back - self.front;
        (r, Some(r))
    }
}

impl DoubleEndedIterator for ColIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let r = self.storage.entry_range(self.back);
        debug_assert!(
            r.start >= self.consumed,
            "ColIterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
            r.start,
            self.consumed,
        );
        let rem = std::mem::take(&mut self.remaining);
        let (left, right) = rem.split_at_mut(r.start - self.consumed);
        let (entry, _after) = right.split_at_mut(r.end - r.start);
        self.remaining = left;
        Some(entry.as_bstr_mut())
    }
}

impl ExactSizeIterator for ColIterMut<'_> {}

// ── owning builder ───────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DualStringPodBuilder {
    seq: Vec<u8>,
    qual: Vec<u8>,
    storage: Storage,
}

impl DualStringPodBuilder {
    /// Create a builder reserving `entry_len * count` bytes in each buffer.
    ///
    /// # Panics
    /// If `entry_len` exceeds `u32::MAX`.
    #[must_use]
    pub fn with_capacity(entry_len: usize, count: usize) -> Self {
        let byte_cap = entry_len.checked_mul(count).unwrap_or(0);
        let seq = Vec::with_capacity(byte_cap);
        let qual = Vec::with_capacity(byte_cap);
        let storage = if entry_len == 0 {
            Storage::new_variable(count)
        } else {
            let stride = u32::try_from(entry_len).expect("entry_len exceeds u32");
            Storage::new_fixed(stride, count)
        };
        Self { seq, qual, storage }
    }

    /// Push one entry's seq and qual bytes. Storage promotes if length
    /// differs from the current stride.
    ///
    /// # Panics
    /// If `seq.len() != qual.len()` or the byte buffer would exceed `u32::MAX`.
    pub fn push(&mut self, seq: &[u8], qual: &[u8]) {
        assert!(
            seq.len() == qual.len(),
            "seq.len() {} != qual.len() {}",
            seq.len(),
            qual.len()
        );
        if let Some(stride) = self.storage.current_stride() {
            if seq.len() as u64 == u64::from(stride) {
                self.seq.extend_from_slice(seq);
                self.qual.extend_from_slice(qual);
                self.storage.builder_push_strided();
                return;
            }
        }
        let start_usize = self.seq.len();
        let start = u32::try_from(start_usize).expect("byte buffer exceeds u32::MAX");
        let stop = u32::try_from(start_usize + seq.len()).expect("byte buffer exceeds u32::MAX");
        self.seq.extend_from_slice(seq);
        self.qual.extend_from_slice(qual);
        self.storage.builder_push_position(start, stop);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.seq.len()
    }

    #[must_use]
    pub fn finish(self) -> DualStringPod {
        let count = self.storage.len();
        DualStringPod {
            seq: Arc::new(self.seq),
            qual: Arc::new(self.qual),
            storage: self.storage,
            seq_first_byte: 0,
            qual_first_byte: 0,
            edits: ColumnEdits::new(count),
        }
    }
}

// ── alias builder ────────────────────────────────────────────────────────

/// Builds a [`DualStringPod`] whose entries reference bytes in an existing
/// pod's `seq` and `qual` `Arc<[u8]>` buffers without copying.
///
/// Each [`push_alias`](DualStringPodAliasBuilder::push_alias) call consumes
/// the *next* source entry in order, so aliases are guaranteed to be
/// non-overlapping. At most `source.len()` aliases may be pushed.
///
/// The source pod is borrowed for the builder's lifetime and released on
/// [`finish`](DualStringPodAliasBuilder::finish). Both buffers are then
/// co-owned by the alias pod (snapshot semantics).
pub struct DualStringPodAliasBuilder<'a> {
    source: &'a DualStringPod,
    next: usize,
    positions: Vec<(u32, u32)>,
}

impl DualStringPodAliasBuilder<'_> {
    /// Alias the next source entry, taking `source_entry[offset..offset+len]`
    /// in both the seq and qual buffers.
    ///
    /// `offset` and `len` are relative to the visible start of the source
    /// entry (after any `cut_start` / `cut_end` overlays).
    ///
    /// # Panics
    /// - If all source entries have already been consumed.
    /// - If `offset + len > source.entry_len(next)`.
    /// - If any computed byte position would exceed `u32::MAX`.
    pub fn push_alias(&mut self, offset: usize, len: usize) {
        assert!(
            self.next < self.source.len(),
            "alias builder: all {} source entries already consumed",
            self.source.len(),
        );
        let r = self.source.storage.entry_range(self.next);
        let entry_len = r.end - r.start;
        let end = offset
            .checked_add(len)
            .expect("alias offset + len overflows usize");
        assert!(
            end <= entry_len,
            "alias offset {offset}+len {len}={end} exceeds entry length {entry_len}",
        );
        // Absolute position in both buffers (seq_first_byte == qual_first_byte
        // for pods built via DualStringPodBuilder, which is the expected source).
        let abs_start = u32::try_from(r.start + self.source.seq_first_byte + offset)
            .expect("alias start exceeds u32");
        let abs_end = u32::try_from(r.start + self.source.seq_first_byte + end)
            .expect("alias end exceeds u32");
        self.positions.push((abs_start, abs_end));
        self.next += 1;
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Finalise the alias builder, releasing the borrow of the source pod.
    #[must_use]
    pub fn finish(self) -> DualStringPod {
        let count = self.positions.len();
        DualStringPod {
            seq: Arc::clone(&self.source.seq),
            qual: Arc::clone(&self.source.qual),
            storage: Storage::Variable(VariableInfo {
                positions: self.positions,
                head_skip: 0,
                tail_skip: 0,
                front_skip: 0,
            }),
            seq_first_byte: 0,
            qual_first_byte: 0,
            edits: ColumnEdits::new(count),
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "it's tests")]
mod tests {
    use super::{ColumnError, DualStringPod, DualStringPodBuilder};
    use bstr::BStr;

    fn b(s: &str) -> &[u8] {
        s.as_bytes()
    }

    #[test]
    fn empty_dual_pod() {
        let p = DualStringPod::empty();
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
        assert_eq!(p.used_bytes(), 0);
    }

    #[test]
    fn fixed_length_dual_basic() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"), b("###"));
        bld.push(b("BBB"), b("FFF"));
        let p = bld.finish();
        assert!(p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("AAA"));
        assert_eq!(p.qual(0), BStr::new("###"));
        assert_eq!(p.seq(1), BStr::new("BBB"));
        assert_eq!(p.qual(1), BStr::new("FFF"));
        let (s, q) = p.pair(0);
        assert_eq!(s, BStr::new("AAA"));
        assert_eq!(q, BStr::new("###"));
    }

    #[test]
    fn dual_promotes_on_length_mismatch() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"), b("###"));
        bld.push(b("BB"), b("FF"));
        let p = bld.finish();
        assert!(!p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("AAA"));
        assert_eq!(p.seq(1), BStr::new("BB"));
        assert_eq!(p.qual(1), BStr::new("FF"));
    }

    #[test]
    #[should_panic(expected = "seq.len()")]
    fn push_unequal_lengths_panics() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 1);
        bld.push(b("AAA"), b("##")); // mismatch
    }

    #[test]
    fn cut_start_dual() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        let mut p = bld.finish();
        p.cut_start(2, None);
        assert_eq!(p.seq(0), BStr::new("LLO"));
        assert_eq!(p.qual(0), BStr::new("345"));
        assert_eq!(p.seq(1), BStr::new("RLD"));
        assert_eq!(p.qual(1), BStr::new("890"));
    }

    #[test]
    fn cut_end_dual() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 1);
        bld.push(b("HELLO"), b("12345"));
        let mut p = bld.finish();
        p.cut_end(2, None);
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
    }

    #[test]
    fn cuts_apply_identically_to_both() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ABCDEF"), b("uvwxyz"));
        bld.push(b("123"), b("XYZ"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None);
        assert_eq!(p.seq(0), BStr::new("BCDE"));
        assert_eq!(p.qual(0), BStr::new("vwxy"));
        assert_eq!(p.seq(1), BStr::new("2"));
        assert_eq!(p.qual(1), BStr::new("Y"));
    }

    #[test]
    fn dual_drain_promotes() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 4);
        bld.push(b("AA"), b("11"));
        bld.push(b("BB"), b("22"));
        bld.push(b("CC"), b("33"));
        bld.push(b("DD"), b("44"));
        let mut p = bld.finish();
        p.drain(1..3);
        assert!(!p.is_fixed_length());
        assert_eq!(p.len(), 2);
        assert_eq!(p.seq(0), BStr::new("AA"));
        assert_eq!(p.qual(0), BStr::new("11"));
        assert_eq!(p.seq(1), BStr::new("DD"));
        assert_eq!(p.qual(1), BStr::new("44"));
    }

    #[test]
    fn dual_iter_yields_pairs() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 2);
        bld.push(b("AB"), b("12"));
        bld.push(b("CD"), b("34"));
        let p = bld.finish();
        let pairs: Vec<_> = p.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].seq, BStr::new("AB"));
        assert_eq!(pairs[0].qual, BStr::new("12"));
        assert_eq!(pairs[1].seq, BStr::new("CD"));
        assert_eq!(pairs[1].qual, BStr::new("34"));
    }

    #[test]
    fn iter_seq_and_iter_qual_separate() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 2);
        bld.push(b("AB"), b("12"));
        bld.push(b("CD"), b("34"));
        let p = bld.finish();
        let seqs: Vec<&BStr> = p.iter_seq().collect();
        let quals: Vec<&BStr> = p.iter_qual().collect();
        assert_eq!(seqs, vec![BStr::new("AB"), BStr::new("CD")]);
        assert_eq!(quals, vec![BStr::new("12"), BStr::new("34")]);
    }

    #[test]
    fn dual_clone_shares_arcs() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 1);
        bld.push(b("AB"), b("12"));
        let p = bld.finish();
        let q = p.clone();
        assert!(std::ptr::eq(p.seq.as_ref(), q.seq.as_ref()));
        assert!(std::ptr::eq(p.qual.as_ref(), q.qual.as_ref()));
    }

    #[test]
    fn alias_builder_basic() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("HELLOWORLD"), b("0123456789"));
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(0, 5); // entry 0, offset 0, len 5 → "HELLO" / "01234"
        ab.push_alias(2, 4); // entry 1, offset 2, len 4 → "GTAC" / "IIII"
        let aliased = ab.finish();
        assert_eq!(aliased.len(), 2);
        assert_eq!(aliased.seq(0), BStr::new("HELLO"));
        assert_eq!(aliased.qual(0), BStr::new("01234"));
        assert_eq!(aliased.seq(1), BStr::new("GTAC"));
        assert_eq!(aliased.qual(1), BStr::new("IIII"));
    }

    #[test]
    fn alias_pod_survives_source_drain() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 3);
        bld.push(b("AAAA"), b("1111"));
        bld.push(b("BBBB"), b("2222"));
        bld.push(b("CCCC"), b("3333"));
        let mut source = bld.finish();
        let aliased = {
            let mut ab = source.alias_builder();
            ab.push_alias(0, 4); // entry 0 ("AAAA" / "1111"), full range
            ab.push_alias(0, 4); // entry 1 ("BBBB" / "2222"), full range
            ab.finish()
        };
        source.drain(1..2);
        assert_eq!(source.len(), 2);
        assert_eq!(aliased.seq(0), BStr::new("AAAA"));
        assert_eq!(aliased.qual(0), BStr::new("1111"));
        assert_eq!(aliased.seq(1), BStr::new("BBBB"));
        assert_eq!(aliased.qual(1), BStr::new("2222"));
    }

    #[test]
    fn alias_pod_pins_qual_independently() {
        // Even when only seq seems to matter, qual must remain accessible
        // for tag-column usages that later query qualities at hit ranges.
        let mut bld = DualStringPodBuilder::with_capacity(0, 1);
        bld.push(b("ACGTACGT"), b("!\"#$%&'("));
        let source = bld.finish();
        let aliased = {
            let mut ab = source.alias_builder();
            ab.push_alias(2, 4); // entry 0, offset 2, len 4 → "GTAC" / "#$%&"
            ab.finish()
        };
        drop(source);
        assert_eq!(aliased.seq(0), BStr::new("GTAC"));
        assert_eq!(aliased.qual(0), BStr::new("#$%&"));
    }

    #[test]
    #[should_panic(expected = "exceeds entry")]
    fn alias_out_of_bounds_panics() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 1);
        bld.push(b("hello"), b("xxxxx"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(3, 10); // offset 3 + len 10 = 13 > entry len 5
    }

    #[test]
    #[should_panic(expected = "already consumed")]
    fn alias_too_many_entries_panics() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 1);
        bld.push(b("AAAA"), b("1111"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(0, 4);
        ab.push_alias(0, 4); // second push on a 1-entry source — panics
    }

    #[test]
    fn seq_mut_exclusive_succeeds() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"), b("###"));
        let mut p = bld.finish();
        p.seq_mut(0).unwrap().copy_from_slice(b("ZZZ"));
        assert_eq!(p.seq(0), BStr::new("ZZZ"));
        assert_eq!(p.qual(0), BStr::new("###"));
    }

    #[test]
    fn qual_mut_exclusive_succeeds() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"), b("###"));
        let mut p = bld.finish();
        p.qual_mut(0).unwrap().copy_from_slice(b("@@@"));
        assert_eq!(p.seq(0), BStr::new("AAA"));
        assert_eq!(p.qual(0), BStr::new("@@@"));
    }

    #[test]
    fn seq_mut_shared_returns_none() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"), b("###"));
        let mut p = bld.finish();
        let _q = p.clone();
        assert!(p.seq_mut(0).is_none());
    }

    #[test]
    fn qual_mut_shared_returns_none() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"), b("###"));
        let mut p = bld.finish();
        let _q = p.clone();
        assert!(p.qual_mut(0).is_none());
    }

    #[test]
    fn dual_send_sync_check() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DualStringPod>();
        assert_send_sync::<super::DualStringPodBuilder>();
        assert_send_sync::<super::DualStringPodAliasBuilder<'static>>();
    }

    #[test]
    fn debug_format_does_not_panic() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 1);
        bld.push(b("AB"), b("12"));
        let p = bld.finish();
        let s = format!("{p:?}");
        assert!(s.contains("DualStringPod"));
    }

    // ── from_columns ──────────────────────────────────────────────────────

    use crate::single::{StringPod, StringPodBuilder};

    fn seq_pod(entries: &[&str]) -> StringPod {
        let mut bld = StringPodBuilder::with_capacity(0, entries.len());
        for e in entries {
            bld.push(b(e));
        }
        bld.finish()
    }

    /// Build a `FixedLength`-storage pod (stride = entry length).
    fn fixed_pod(stride: usize, entries: &[&str]) -> StringPod {
        let mut bld = StringPodBuilder::with_capacity(stride, entries.len());
        for e in entries {
            bld.push(b(e));
        }
        bld.finish()
    }

    #[test]
    fn from_columns_zero_copy() {
        let seq = seq_pod(&["ACGT", "TTTT", "GGGG"]);
        let qual = seq_pod(&["IIII", "FFFF", "####"]);
        // Buffers we expect to be reused (no copy) — capture the pointers.
        let seq_ptr = seq.data.as_ptr();
        let qual_ptr = qual.data.as_ptr();
        let dual = DualStringPod::try_from_columns(seq, qual).unwrap();
        assert_eq!(dual.len(), 3);
        assert_eq!(dual.seq(0), BStr::new("ACGT"));
        assert_eq!(dual.qual(0), BStr::new("IIII"));
        assert_eq!(dual.seq(2), BStr::new("GGGG"));
        assert_eq!(dual.qual(2), BStr::new("####"));
        // Zero-copy: the dual pod's buffers are the very same allocations.
        assert_eq!(dual.seq.as_ptr(), seq_ptr);
        assert_eq!(dual.qual.as_ptr(), qual_ptr);
    }

    #[test]
    fn from_columns_fixed_fast_path_keeps_strided() {
        // Both columns FixedLength with identical layout → strided metadata
        // is kept (no positions vec) and bytes are shared.
        let seq = fixed_pod(4, &["ACGT", "TTTT", "GGGG"]);
        let qual = fixed_pod(4, &["IIII", "FFFF", "####"]);
        let seq_ptr = seq.data.as_ptr();
        let qual_ptr = qual.data.as_ptr();
        let dual = DualStringPod::try_from_columns(seq, qual).unwrap();
        assert!(dual.is_fixed_length());
        assert_eq!(dual.seq(1), BStr::new("TTTT"));
        assert_eq!(dual.qual(1), BStr::new("FFFF"));
        assert_eq!(dual.seq.as_ptr(), seq_ptr);
        assert_eq!(dual.qual.as_ptr(), qual_ptr);
    }

    #[test]
    fn from_columns_constant_offset_no_copy() {
        // seq carries a stray leading entry (dropped via pop_front), so its
        // first byte sits 4 bytes past qual's. The shared metadata plus the two
        // per-column offsets must still index both correctly — and without
        // copying any bytes.
        let mut seq = fixed_pod(4, &["XXXX", "ACGT", "TTTT"]);
        seq.pop_front(1); // front_byte now 4; entries "ACGT","TTTT"
        let qual = fixed_pod(4, &["IIII", "FFFF"]); // front_byte 0
        let seq_ptr = seq.data.as_ptr();
        let qual_ptr = qual.data.as_ptr();
        let dual = DualStringPod::try_from_columns(seq, qual).unwrap();
        assert_eq!(dual.len(), 2);
        assert_eq!(dual.seq(0), BStr::new("ACGT"));
        assert_eq!(dual.qual(0), BStr::new("IIII"));
        assert_eq!(dual.seq(1), BStr::new("TTTT"));
        assert_eq!(dual.qual(1), BStr::new("FFFF"));
        // Still zero-copy despite the divergent offsets.
        assert_eq!(dual.seq.as_ptr(), seq_ptr);
        assert_eq!(dual.qual.as_ptr(), qual_ptr);
    }

    #[test]
    fn from_columns_variable_lengths() {
        let seq = seq_pod(&["A", "ACGTACGT", "ACG"]);
        let qual = seq_pod(&["I", "FFFFFFFF", "JJJ"]);
        let dual = DualStringPod::try_from_columns(seq, qual).unwrap();
        let pairs: Vec<_> = dual.iter().collect();
        assert_eq!(pairs[1].seq, BStr::new("ACGTACGT"));
        assert_eq!(pairs[1].qual, BStr::new("FFFFFFFF"));
    }

    #[test]
    fn from_columns_entry_length_mismatch_errors() {
        let seq = seq_pod(&["ACGT", "TT"]);
        let qual = seq_pod(&["IIII", "F"]); // entry 1: 2 != 1
        let err = DualStringPod::try_from_columns(seq, qual).unwrap_err();
        assert_eq!(err, ColumnError::UnequalLengths);
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn from_columns_count_mismatch_errors() {
        let seq = seq_pod(&["ACGT", "TTTT"]);
        let qual = seq_pod(&["IIII"]);
        let err = DualStringPod::try_from_columns(seq, qual).unwrap_err();
        assert_eq!(err, ColumnError::EqualCountViolated);
    }

    // ── prefix / postfix ──────────────────────────────────────────────────

    #[test]
    fn dual_prefix_prepends_both_columns() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"), b("###"));
        bld.push(b("BBB"), b("$$$"));
        let p = bld.finish().prefix(b("XX"), b("!!"));
        assert!(p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("XXAAA"));
        assert_eq!(p.qual(0), BStr::new("!!###"));
        assert_eq!(p.seq(1), BStr::new("XXBBB"));
        assert_eq!(p.qual(1), BStr::new("!!$$$"));
    }

    #[test]
    fn dual_postfix_appends_both_columns() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"), b("###"));
        bld.push(b("BBB"), b("$$$"));
        let p = bld.finish().postfix(b("ZZ"), b("++"));
        assert!(p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("AAAZZ"));
        assert_eq!(p.qual(0), BStr::new("###++"));
    }

    #[test]
    #[should_panic(expected = "seq_text.len()")]
    fn dual_prefix_length_mismatch_panics() {
        let p = DualStringPod::empty();
        let _ = p.prefix(b("AB"), b("X"));
    }

    // ── max_len ───────────────────────────────────────────────────────────

    #[test]
    fn dual_max_len_fixed_clips() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        let mut p = bld.finish();
        p.max_len(3, None);
        assert!(p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
    }

    #[test]
    fn dual_max_len_variable_clips_each() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("II"));
        let mut p = bld.finish();
        p.max_len(4, None);
        assert_eq!(p.seq(0), BStr::new("ACGT"));
        assert_eq!(p.qual(0), BStr::new("IIII"));
        assert_eq!(p.seq(1), BStr::new("AC"));
        assert_eq!(p.qual(1), BStr::new("II"));
    }

    // ── reverse ───────────────────────────────────────────────────────────

    #[test]
    fn dual_reverse_both_columns() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTTT"), b("####"));
        let p = bld.finish().reverse(None);
        assert_eq!(p.seq(0), BStr::new("TGCA"));
        assert_eq!(p.qual(0), BStr::new("IIII"));
        assert_eq!(p.seq(1), BStr::new("TTTT"));
        assert_eq!(p.qual(1), BStr::new("####"));
    }

    #[test]
    fn dual_reverse_cow_clones_shared_arc() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 1);
        bld.push(b("ACG"), b("III"));
        let p = bld.finish();
        let q = p.clone();
        let r = p.reverse(None);
        assert_eq!(q.seq(0), BStr::new("ACG"));
        assert_eq!(r.seq(0), BStr::new("GCA"));
    }

    // ── conditional operations ────────────────────────────────────────────

    #[test]
    fn dual_cut_start_conditional_fixed_promotes() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        bld.push(b("RUST!"), b("ABCDE"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        p.cut_start(2, Some(&[true, false, true]));
        assert!(!p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("LLO"));
        assert_eq!(p.qual(0), BStr::new("345"));
        assert_eq!(p.seq(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.qual(1), BStr::new("67890"));
        assert_eq!(p.seq(2), BStr::new("ST!"));
        assert_eq!(p.qual(2), BStr::new("CDE"));
    }

    #[test]
    fn dual_cut_end_conditional_fixed_promotes() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        bld.push(b("RUST!"), b("ABCDE"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        p.cut_end(2, Some(&[true, false, true]));
        assert!(!p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
        assert_eq!(p.seq(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.qual(1), BStr::new("67890"));
        assert_eq!(p.seq(2), BStr::new("RUS"));
        assert_eq!(p.qual(2), BStr::new("ABC"));
    }

    #[test]
    fn dual_cut_end_conditional_variable() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("JJ"));
        let mut p = bld.finish();
        p.cut_end(3, Some(&[true, false]));
        assert_eq!(p.seq(0), BStr::new("ACGTA"));
        assert_eq!(p.qual(0), BStr::new("IIIII"));
        assert_eq!(p.seq(1), BStr::new("AC")); // untouched
        assert_eq!(p.qual(1), BStr::new("JJ"));
    }

    #[test]
    fn dual_truncate_drops_entries() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"), b("11"));
        bld.push(b("BB"), b("22"));
        bld.push(b("CC"), b("33"));
        let mut p = bld.finish();
        p.truncate(2);
        assert_eq!(p.len(), 2);
        assert_eq!(p.seq(0), BStr::new("AA"));
        assert_eq!(p.seq(1), BStr::new("BB"));
    }

    #[test]
    fn dual_max_len_conditional_fixed_promotes() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        bld.push(b("RUST!"), b("ABCDE"));
        let mut p = bld.finish();
        p.max_len(3, Some(&[true, false, true]));
        assert!(!p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
        assert_eq!(p.seq(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.qual(1), BStr::new("67890"));
        assert_eq!(p.seq(2), BStr::new("RUS"));
        assert_eq!(p.qual(2), BStr::new("ABC"));
    }

    #[test]
    fn dual_max_len_conditional_variable() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("JJ"));
        let mut p = bld.finish();
        p.max_len(4, Some(&[true, false]));
        assert_eq!(p.seq(0), BStr::new("ACGT"));
        assert_eq!(p.qual(0), BStr::new("IIII"));
        assert_eq!(p.seq(1), BStr::new("AC")); // untouched
        assert_eq!(p.qual(1), BStr::new("JJ"));
    }

    #[test]
    fn dual_reverse_conditional_only_marked() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 3);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTTT"), b("####"));
        bld.push(b("GCCA"), b("FFFF"));
        let p = bld.finish().reverse(Some(&[true, false, true]));
        assert_eq!(p.seq(0), BStr::new("TGCA"));
        assert_eq!(p.qual(0), BStr::new("IIII")); // palindrome
        assert_eq!(p.seq(1), BStr::new("TTTT")); // untouched (palindrome anyway)
        assert_eq!(p.qual(1), BStr::new("####"));
        assert_eq!(p.seq(2), BStr::new("ACCG"));
        assert_eq!(p.qual(2), BStr::new("FFFF"));
    }

    #[test]
    fn dual_reverse_conditional_variable() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("JJ"));
        let p = bld.finish().reverse(Some(&[false, true]));
        assert_eq!(p.seq(0), BStr::new("ACGTACGT")); // untouched
        assert_eq!(p.qual(0), BStr::new("IIIIIIII"));
        assert_eq!(p.seq(1), BStr::new("CA"));
        assert_eq!(p.qual(1), BStr::new("JJ")); // palindrome
    }

    // ── iter_mut ──────────────────────────────────────────────────────────────

    #[test]
    fn dual_iter_mut_allows_in_place_mutation() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTTT"), b("####"));
        let mut p = bld.finish();
        for entry in p.iter_mut() {
            entry.seq.reverse();
            entry.qual.reverse();
        }
        assert_eq!(p.seq(0), BStr::new("TGCA"));
        assert_eq!(p.qual(0), BStr::new("IIII")); // palindrome
        assert_eq!(p.seq(1), BStr::new("TTTT")); // palindrome
        assert_eq!(p.qual(1), BStr::new("####")); // palindrome
    }

    #[test]
    fn dual_iter_mut_cows_when_shared() {
        // A shared buffer no longer blocks iteration: `iter_mut` calls
        // `make_exclusive`, COW-cloning so the mutation lands on this pod alone
        // and the clone is left untouched.
        let mut bld = DualStringPodBuilder::with_capacity(2, 1);
        bld.push(b("AB"), b("12"));
        let mut p = bld.finish();
        let q = p.clone();
        for e in p.iter_mut() {
            e.seq.make_ascii_lowercase();
            e.qual.reverse();
        }
        assert_eq!(p.seq(0), BStr::new("ab"));
        assert_eq!(p.qual(0), BStr::new("21"));
        // The clone still sees the originals.
        assert_eq!(q.seq(0), BStr::new("AB"));
        assert_eq!(q.qual(0), BStr::new("12"));
    }

    #[test]
    fn iter_seq_mut_and_iter_qual_mut_mutate_one_column() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTGA"), b("FFFF"));
        let mut p = bld.finish();

        for s in p.iter_seq_mut() {
            s.make_ascii_lowercase();
        }
        assert_eq!(p.seq(0), BStr::new("acgt"));
        assert_eq!(p.seq(1), BStr::new("ttga"));
        assert_eq!(p.qual(0), BStr::new("IIII")); // qual untouched
        assert_eq!(p.qual(1), BStr::new("FFFF"));

        for q in p.iter_qual_mut() {
            q.reverse();
        }
        assert_eq!(p.qual(0), BStr::new("IIII")); // palindrome
        assert_eq!(p.seq(0), BStr::new("acgt")); // seq untouched by qual pass
    }

    #[test]
    fn iter_seq_mut_cows_only_seq_buffer() {
        // Both columns shared with a clone. Mutating seq must COW seq alone and
        // leave the clone's seq untouched; the qual buffer can stay shared.
        let mut bld = DualStringPodBuilder::with_capacity(2, 1);
        bld.push(b("AB"), b("12"));
        let mut p = bld.finish();
        let q = p.clone();
        for s in p.iter_seq_mut() {
            s.make_ascii_lowercase();
        }
        assert_eq!(p.seq(0), BStr::new("ab"));
        assert_eq!(q.seq(0), BStr::new("AB")); // clone unaffected
        assert_eq!(q.qual(0), BStr::new("12"));
    }

    #[test]
    fn iter_seq_mut_double_ended_and_collectable() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 3);
        bld.push(b("ACG"), b("III"));
        bld.push(b("TTT"), b("###"));
        bld.push(b("GCA"), b("FFF"));
        let mut p = bld.finish();
        {
            let mut it = p.iter_seq_mut();
            it.next().unwrap().reverse(); // ACG → GCA
            it.next_back().unwrap().reverse(); // GCA → ACG
        }
        assert_eq!(p.seq(0), BStr::new("GCA"));
        assert_eq!(p.seq(1), BStr::new("TTT")); // untouched
        assert_eq!(p.seq(2), BStr::new("ACG"));

        let all: Vec<&mut BStr> = p.iter_seq_mut().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn dual_iter_mut_double_ended() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 3);
        bld.push(b("ACG"), b("III"));
        bld.push(b("TTT"), b("###"));
        bld.push(b("GCA"), b("FFF"));
        let mut p = bld.finish();
        {
            let mut it = p.iter_mut();
            it.next().unwrap().seq.reverse(); // ACG → GCA
            it.next_back().unwrap().seq.reverse(); // GCA → ACG
        }
        assert_eq!(p.seq(0), BStr::new("GCA"));
        assert_eq!(p.seq(1), BStr::new("TTT")); // untouched
        assert_eq!(p.seq(2), BStr::new("ACG"));
    }

    #[test]
    fn dual_iter_mut_skips_cut_gaps() {
        let mut bld = DualStringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"), b("12345"));
        bld.push(b("WORLD"), b("67890"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None); // visible seq: "ELL","ORL"; qual: "234","789"
        for e in p.iter_mut() {
            e.seq.reverse();
            e.qual.reverse();
        }
        assert_eq!(p.seq(0), BStr::new("LLE"));
        assert_eq!(p.qual(0), BStr::new("432"));
        assert_eq!(p.seq(1), BStr::new("LRO"));
        assert_eq!(p.qual(1), BStr::new("987"));
    }

    #[test]
    fn dual_iter_mut_variable_lengths() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("JJ"));
        let mut p = bld.finish();
        for e in p.iter_mut() {
            e.seq.make_ascii_lowercase();
        }
        assert_eq!(p.seq(0), BStr::new("acgtacgt"));
        assert_eq!(p.qual(0), BStr::new("IIIIIIII")); // untouched
        assert_eq!(p.seq(1), BStr::new("ac"));
    }

    #[test]
    fn dual_iter_mut_divergent_first_bytes() {
        // The key case: seq carries a stray leading entry (popped), so
        // seq_first_byte (4) != qual_first_byte (0). Both buffers must still be
        // peeled correctly off one shared relative cursor.
        let mut seq = fixed_pod(4, &["XXXX", "ACGT", "TTTT"]);
        seq.pop_front(1); // seq_first_byte becomes 4
        let qual = fixed_pod(4, &["IIII", "FFFF"]); // qual_first_byte 0
        let mut dual = DualStringPod::try_from_columns(seq, qual).unwrap();
        // Sources were moved in, so both Arcs are uniquely owned here.
        for e in dual.iter_mut() {
            e.seq.make_ascii_lowercase();
            e.qual.make_ascii_lowercase();
        }
        assert_eq!(dual.seq(0), BStr::new("acgt"));
        assert_eq!(dual.qual(0), BStr::new("iiii"));
        assert_eq!(dual.seq(1), BStr::new("tttt"));
        assert_eq!(dual.qual(1), BStr::new("ffff"));
    }

    #[test]
    fn dual_iter_mut_collect_all_then_mutate() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"), b("11"));
        bld.push(b("BB"), b("22"));
        bld.push(b("CC"), b("33"));
        let mut p = bld.finish();
        {
            let all: Vec<_> = p.iter_mut().collect();
            assert_eq!(all.len(), 3);
            for e in all {
                e.seq.make_ascii_lowercase();
            }
        }
        assert_eq!(p.seq(0), BStr::new("aa"));
        assert_eq!(p.seq(2), BStr::new("cc"));
    }

    #[test]
    fn from_columns_non_constant_offset_errors() {
        // Equal per-entry lengths, but qual has an orphaned gap (from a drain)
        // so the entry starts drift apart — not a constant translation.
        let seq = seq_pod(&["ACGT", "TTTT"]);
        let mut qual = seq_pod(&["IIII", "ZZZZ", "FFFF"]);
        qual.drain(1..2); // entries "IIII"(0..4), "FFFF"(8..12)
        let err = DualStringPod::try_from_columns(seq, qual).unwrap_err();
        assert_eq!(err, ColumnError::NotATranslation);
    }

    #[test]
    fn cut_start_conditional_after_pop_front_dual() {
        // Regression: conditional cuts must honor front_skip on both columns.
        let mut bld = DualStringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAAA"), b("vvvvv"));
        bld.push(b("BBB"), b("www")); // unequal length → Variable storage
        bld.push(b("CCCCC"), b("xxxxx"));
        let mut p = bld.finish();
        assert!(!p.is_fixed_length());
        p.pop_front(1); // view: ("BBB","www"), ("CCCCC","xxxxx")
        p.cut_start(2, Some(&[true, false]));
        assert_eq!(p.pair(0), (BStr::new("B"), BStr::new("w")));
        assert_eq!(p.pair(1), (BStr::new("CCCCC"), BStr::new("xxxxx")));
    }

    // ── resize ────────────────────────────────────────────────────────────

    #[test]
    fn resize_dual_per_entry_cut() {
        let mut bld = DualStringPodBuilder::with_capacity(6, 3);
        bld.push(b("HELLO!"), b("123456"));
        bld.push(b("WORLD!"), b("ABCDEF"));
        bld.push(b("RUST!!"), b("uvwxyz"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        // Cut each entry at a per-entry location: keep [i..i+2] in both columns.
        p.resize(|i, _, _| Some((i, 2)));
        assert!(!p.is_fixed_length()); // promoted
        assert_eq!(p.pair(0), (BStr::new("HE"), BStr::new("12")));
        assert_eq!(p.pair(1), (BStr::new("OR"), BStr::new("BC")));
        assert_eq!(p.pair(2), (BStr::new("ST"), BStr::new("wx")));
    }

    #[test]
    fn resize_dual_callback_sees_both_columns() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("FFFF####"));
        bld.push(b("TTTT"), b("!!!!"));
        let mut p = bld.finish();
        // Quality-style trim: cut at first '#' in the quality string.
        p.resize(|_, seq, qual| {
            assert_eq!(seq.len(), qual.len());
            let keep = qual.iter().position(|&c| c == b'#').unwrap_or(qual.len());
            Some((0, keep))
        });
        assert_eq!(p.pair(0), (BStr::new("ACGT"), BStr::new("FFFF")));
        assert_eq!(p.pair(1), (BStr::new("TTTT"), BStr::new("!!!!")));
    }

    #[test]
    fn resize_dual_none_and_offsets() {
        // A dual pod built from columns can carry non-zero first_byte offsets;
        // resize must honor them and leave None entries untouched.
        let mut bld = DualStringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAAA"), b("vvvvv"));
        bld.push(b("BBBBB"), b("wwwww"));
        bld.push(b("CCCCC"), b("xxxxx"));
        let mut p = bld.finish();
        p.cut_start(1, None); // visible: "AAAA"/"vvvv", etc. (head overlay)
        p.resize(|i, s, q| {
            assert_eq!(s.len(), 4);
            assert_eq!(q.len(), 4);
            if i == 1 { None } else { Some((1, 2)) }
        });
        assert_eq!(p.pair(0), (BStr::new("AA"), BStr::new("vv")));
        assert_eq!(p.pair(1), (BStr::new("BBBB"), BStr::new("wwww"))); // untouched
        assert_eq!(p.pair(2), (BStr::new("CC"), BStr::new("xx")));
    }

    #[test]
    #[should_panic(expected = "exceeds visible length")]
    fn resize_dual_out_of_range_panics() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 1);
        bld.push(b("ACGT"), b("FFFF"));
        let mut p = bld.finish();
        p.resize(|_, _, _| Some((1, 5))); // 1+5 > 4
    }
}
