use bstr::{BStr, ByteSlice as _};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use crate::single::StringPod;
use crate::storage::Storage;

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
    pub fn try_from_columns(seq: StringPod, qual: StringPod) -> Result<Self, ColumnError> {
        if seq.len() != qual.len() {
            return Err(ColumnError::EqualCountViolated);
        }

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

        let storage = Storage::Variable {
            positions,
            head_skip: 0,
            tail_skip: 0,
            front_skip: 0,
        };
        Ok(DualStringPod {
            seq: seq.data,
            qual: qual.data,
            storage,
            seq_first_byte: base_seq,
            qual_first_byte: base_qual,
        })
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

    pub fn cut_start(&mut self, n: usize) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_start(n_u32);
    }

    pub fn cut_end(&mut self, n: usize) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_end(n_u32);
    }

    /// # Panics
    /// If `range.end > self.len()` or `range.start > range.end`.
    pub fn drain(&mut self, range: Range<usize>) {
        assert!(range.start <= range.end, "drain range start > end");
        assert!(range.end <= self.len(), "drain range past end of pod");
        self.storage.drain(range);
    }

    #[must_use]
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
        let first_len = if n > 0 { self.entry_len(0) + seq_text.len() } else { seq_text.len() };
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
        bld.finish()
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
        let first_len = if n > 0 { self.entry_len(0) + seq_text.len() } else { seq_text.len() };
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
        bld.finish()
    }

    /// Truncate every entry to at most `n` bytes. O(1) for `FixedLength`
    /// pods; rebuilds both buffers for `Variable` pods.
    #[must_use]
    pub fn max_len(mut self, n: usize) -> Self {
        if let Storage::FixedLength { visible_len, .. } = self.storage {
            let vl = visible_len as usize;
            if vl > n {
                self.cut_end(vl - n);
            }
            return self;
        }
        let count = self.len();
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
        bld.finish()
    }

    /// Reverse the bytes of every entry in both columns in-place. If either
    /// buffer is shared (Arc strong count > 1) it is cloned before reversing
    /// (COW).
    #[must_use]
    pub fn reverse(mut self) -> Self {
        if Arc::get_mut(&mut self.seq).is_none() {
            self.seq = Arc::new((*self.seq).clone());
        }
        if Arc::get_mut(&mut self.qual).is_none() {
            self.qual = Arc::new((*self.qual).clone());
        }
        let seq_first_byte = self.seq_first_byte;
        let qual_first_byte = self.qual_first_byte;
        let n = self.storage.len();
        let seq = Arc::get_mut(&mut self.seq).expect("just ensured unique");
        let qual = Arc::get_mut(&mut self.qual).expect("just ensured unique");
        for i in 0..n {
            let r = self.storage.entry_range(i);
            seq[r.start + seq_first_byte..r.end + seq_first_byte].reverse();
            qual[r.start + qual_first_byte..r.end + qual_first_byte].reverse();
        }
        self
    }

    /// Returns a mutable iterator over seq+qual entry pairs, or `None` if
    /// either byte buffer is shared (Arc strong count > 1).
    pub fn iter_mut(&mut self) -> Option<DualIterMut<'_>> {
        let seq_ptr = Arc::get_mut(&mut self.seq)?.as_mut_ptr();
        let qual_ptr = Arc::get_mut(&mut self.qual)?.as_mut_ptr();
        Some(DualIterMut {
            seq_data: seq_ptr,
            qual_data: qual_ptr,
            storage: &self.storage,
            seq_first_byte: self.seq_first_byte,
            qual_first_byte: self.qual_first_byte,
            front: 0,
            back: self.storage.len(),
            _marker: PhantomData,
        })
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

pub struct DualIterMut<'a> {
    seq_data: *mut u8,
    qual_data: *mut u8,
    storage: &'a Storage,
    seq_first_byte: usize,
    qual_first_byte: usize,
    front: usize,
    back: usize,
    _marker: PhantomData<&'a mut u8>,
}

// Safety: `DualIterMut` holds exclusive access (via `Arc::get_mut`) to both
// underlying buffers, mirroring `slice::IterMut`.
unsafe impl Send for DualIterMut<'_> {}
unsafe impl Sync for DualIterMut<'_> {}

impl<'a> Iterator for DualIterMut<'a> {
    type Item = DualEntryMut<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let r = self.storage.entry_range(self.front);
        self.front += 1;
        let seq_start = r.start + self.seq_first_byte;
        let qual_start = r.start + self.qual_first_byte;
        let len = r.end - r.start;
        // Safety: each entry range is visited at most once (front is
        // monotone increasing). Both `seq_data` and `qual_data` were
        // obtained via `Arc::get_mut`, guaranteeing exclusive buffer access.
        Some(unsafe {
            DualEntryMut {
                seq: std::slice::from_raw_parts_mut(self.seq_data.add(seq_start), len)
                    .as_bstr_mut(),
                qual: std::slice::from_raw_parts_mut(self.qual_data.add(qual_start), len)
                    .as_bstr_mut(),
            }
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
        let seq_start = r.start + self.seq_first_byte;
        let qual_start = r.start + self.qual_first_byte;
        let len = r.end - r.start;
        Some(unsafe {
            DualEntryMut {
                seq: std::slice::from_raw_parts_mut(self.seq_data.add(seq_start), len)
                    .as_bstr_mut(),
                qual: std::slice::from_raw_parts_mut(self.qual_data.add(qual_start), len)
                    .as_bstr_mut(),
            }
        })
    }
}

impl ExactSizeIterator for DualIterMut<'_> {}

// ── owning builder ───────────────────────────────────────────────────────

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
        DualStringPod {
            seq: Arc::new(self.seq),
            qual: Arc::new(self.qual),
            storage: self.storage,
            seq_first_byte: 0,
            qual_first_byte: 0,
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

impl<'a> DualStringPodAliasBuilder<'a> {
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
        let abs_start =
            u32::try_from(r.start + self.source.seq_first_byte + offset)
                .expect("alias start exceeds u32");
        let abs_end =
            u32::try_from(r.start + self.source.seq_first_byte + end)
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
        DualStringPod {
            seq: Arc::clone(&self.source.seq),
            qual: Arc::clone(&self.source.qual),
            storage: Storage::Variable {
                positions: self.positions,
                head_skip: 0,
                tail_skip: 0,
                front_skip: 0,
            },
            seq_first_byte: 0,
            qual_first_byte: 0,
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
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
        p.cut_start(2);
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
        p.cut_end(2);
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
    }

    #[test]
    fn cuts_apply_identically_to_both() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ABCDEF"), b("uvwxyz"));
        bld.push(b("123"), b("XYZ"));
        let mut p = bld.finish();
        p.cut_start(1);
        p.cut_end(1);
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
        bld.push(b("ACGTACGT"),   b("IIIIIIII"));
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
        let p = bld.finish().max_len(3);
        assert!(p.is_fixed_length());
        assert_eq!(p.seq(0), BStr::new("HEL"));
        assert_eq!(p.qual(0), BStr::new("123"));
    }

    #[test]
    fn dual_max_len_variable_clips_each() {
        let mut bld = DualStringPodBuilder::with_capacity(0, 2);
        bld.push(b("ACGTACGT"), b("IIIIIIII"));
        bld.push(b("AC"), b("II"));
        let p = bld.finish().max_len(4);
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
        let p = bld.finish().reverse();
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
        let r = p.reverse();
        assert_eq!(q.seq(0), BStr::new("ACG"));
        assert_eq!(r.seq(0), BStr::new("GCA"));
    }

    // ── iter_mut ──────────────────────────────────────────────────────────

    #[test]
    fn dual_iter_mut_allows_in_place_mutation() {
        let mut bld = DualStringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTTT"), b("####"));
        let mut p = bld.finish();
        for entry in p.iter_mut().unwrap() {
            entry.seq.reverse();
            entry.qual.reverse();
        }
        assert_eq!(p.seq(0), BStr::new("TGCA"));
        assert_eq!(p.qual(0), BStr::new("IIII")); // palindrome
        assert_eq!(p.seq(1), BStr::new("TTTT")); // palindrome
        assert_eq!(p.qual(1), BStr::new("####")); // palindrome
    }

    #[test]
    fn dual_iter_mut_returns_none_when_shared() {
        let mut bld = DualStringPodBuilder::with_capacity(2, 1);
        bld.push(b("AB"), b("12"));
        let mut p = bld.finish();
        let _q = p.clone();
        assert!(p.iter_mut().is_none());
    }

    #[test]
    fn dual_iter_mut_double_ended() {
        let mut bld = DualStringPodBuilder::with_capacity(3, 3);
        bld.push(b("ACG"), b("III"));
        bld.push(b("TTT"), b("###"));
        bld.push(b("GCA"), b("FFF"));
        let mut p = bld.finish();
        {
            let mut it = p.iter_mut().unwrap();
            it.next().unwrap().seq.reverse();       // ACG → GCA
            it.next_back().unwrap().seq.reverse();  // GCA → ACG
        }
        assert_eq!(p.seq(0), BStr::new("GCA"));
        assert_eq!(p.seq(1), BStr::new("TTT")); // untouched
        assert_eq!(p.seq(2), BStr::new("ACG"));
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
}
