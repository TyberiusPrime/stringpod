use bstr::{BStr, ByteSlice as _};
use std::ops::Range;
use std::sync::Arc;

use crate::column::ColumnEdits;
use crate::lifted::Lifted;
use crate::storage::{Storage, VariableInfo};

/// A single column of byte strings backed by one shared `Arc<[u8]>` and a
/// columnar metadata layout. Once `finish()`ed by a builder, the byte buffer
/// is immutable; only the metadata can be mutated (cuts, drains).
#[derive(Clone)]
pub struct StringPod {
    // `Arc<Vec<u8>>` (not `Arc<[u8]>`) so `finish` can wrap the builder's Vec
    // as-is — converting to a boxed slice would realloc+copy the whole buffer
    // whenever the builder over-reserved.
    pub(crate) data: Arc<Vec<u8>>,
    pub(crate) storage: Storage,

    /// Per-entry coordinate-edit history for liftover (see [`Lifted`]).
    pub(crate) edits: ColumnEdits,
}

impl Lifted for StringPod {
    fn edits(&self) -> &ColumnEdits {
        &self.edits
    }
    fn edits_mut(&mut self) -> &mut ColumnEdits {
        &mut self.edits
    }
}

impl StringPod {
    #[must_use]
    pub fn new_all_empty(count: u32) -> Self {
        Self {
            data: Arc::new(Vec::new()),
            storage: Storage::FixedLength {
                stride: 0,
                head_skip: 0,
                visible_len: 0,
                count,
                front_byte: 0,
            },
            edits: ColumnEdits::new(count as usize),
        }
    }

    /// Ensure this pod owns its byte buffer outright, cloning it (COW) only if
    /// it is currently shared with another pod. After this call, mutating
    /// accessors such as [`get_mut`](Self::get_mut) always succeed.
    ///
    /// When the buffer is shared we have to clone it anyway, so we clone
    /// **compacted**: [`compact`](Self::compact) allocates an exact-size buffer
    /// and copies only the visible footprint, making the COW O(footprint) rather
    /// than O(full backing buffer). Building many pods over one shared buffer and
    /// mutating them therefore no longer amplifies to N × |buffer|. A
    /// uniquely-owned buffer is left untouched (no copy, no compaction).
    pub fn make_exclusive(&mut self) {
        if Arc::get_mut(&mut self.data).is_none() {
            self.compact();
        }
    }

    /// An empty pod with no entries and an empty buffer.
    #[must_use]
    pub fn empty() -> Self {
        StringPodBuilder::with_capacity(0, 0).finish()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get entry `i` as a `&BStr`.
    ///
    /// # Panics
    /// If `i >= self.len()` or the recorded byte range is out of bounds of
    /// the buffer (which would indicate corruption).
    #[must_use]
    pub fn get(&self, i: usize) -> &BStr {
        let range = self.storage.entry_range(i);
        BStr::new(&self.data[range])
    }

    /// Get entry `i` as `&mut BStr`, or `None` if the byte buffer is shared
    /// (Arc strong count > 1). When `None` is returned the buffer has multiple
    /// owners; drop or release the other references before retrying, or rebuild
    /// the pod into a fresh buffer.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut BStr> {
        let data = Arc::get_mut(&mut self.data)?;
        let range = self.storage.entry_range(i);
        Some(data[range].as_bstr_mut())
    }

    /// Visible length of entry `i` (after cuts).
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn entry_len(&self, i: usize) -> usize {
        self.storage.entry_len(i)
    }

    /// Sum of visible bytes across all entries (what a tight rebuild would need).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.storage.used_bytes()
    }

    /// Total size of the underlying Arc'd byte buffer, including any bytes
    /// orphaned by `drain` or hidden by `cut_*`.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.data.len()
    }

    /// `true` if storage is `FixedLength` (all entries share a stride).
    #[must_use]
    pub fn is_fixed_length(&self) -> bool {
        self.storage.current_stride().is_some()
    }

    /// Cut `n` bytes off the start of every entry. O(1).
    pub fn cut_start(&mut self, n: usize, conditional: Option<&[bool]>) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_start(n_u32, conditional);
        self.record_cut_start(n, conditional);
    }

    /// Cut `n` bytes off the end of every entry. O(1).
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are affected; this promotes `FixedLength` → `Variable`.
    pub fn cut_end(&mut self, n: usize, conditional: Option<&[bool]>) {
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        self.storage.cut_end(n_u32, conditional);
        self.record_cut_end(n, conditional);
    }

    /// Remove a contiguous range of entries. Promotes `FixedLength` → `Variable`.
    /// Bytes for removed entries remain in the buffer (orphaned).
    ///
    /// # Panics
    /// If `range.end > self.len()` or `range.start > range.end`.
    pub fn drain(&mut self, range: Range<usize>) {
        assert!(range.start <= range.end, "drain range start > end");
        assert!(range.end <= self.len(), "drain range past end of pod");
        let (start, end) = (range.start, range.end);
        self.storage.drain(range);
        self.record_drain(start..end);
    }

    /// Keep only entries where the boolean is true
    pub fn retain_by_bools(&mut self, keep: &[bool]) {
        self.storage.retain_by_bools(keep);
        self.record_retain(keep);
    }

    /// Append one entry to a *finished* pod, mirroring builder semantics
    /// (stride-fast-path with auto-promotion). The bytes are copied into the
    /// buffer's tail.
    ///
    /// Requires unique ownership of the byte buffer — call before the pod is
    /// shared (cloned / sent). Relies on reserved slack (see
    /// [`reserve_for_appends`](Self::reserve_for_appends)) to avoid
    /// reallocating a large buffer.
    ///
    /// # Panics
    /// If the buffer is shared (`Arc` strong count > 1).
    pub fn push(&mut self, bytes: &[u8]) {
        let data =
            Arc::get_mut(&mut self.data).expect("StringPod::push requires a uniquely-owned buffer");
        if let Some(stride) = self.storage.current_stride() {
            if bytes.len() as u64 == u64::from(stride) {
                data.extend_from_slice(bytes);
                self.storage.builder_push_strided();
                self.edits.push_entry();
                return;
            }
        }
        let start = u32::try_from(data.len()).expect("byte buffer exceeds u32::MAX");
        let stop = u32::try_from(data.len() + bytes.len()).expect("byte buffer exceeds u32::MAX");
        data.extend_from_slice(bytes);
        self.storage.builder_push_position(start, stop);
        self.edits.push_entry();
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

    /// Ensure the buffer has spare capacity for roughly `n` more average-sized
    /// entries, so subsequent [`push`](Self::push) calls land in place instead
    /// of reallocating a (possibly multi-MB) shared buffer. No-op if the buffer
    /// is already shared.
    pub fn reserve_for_appends(&mut self, n: usize) {
        let per = match self.storage.current_stride() {
            Some(stride) => stride as usize,
            None => self
                .data
                .len()
                .checked_div(self.len())
                .map_or(64, |per| per.max(1)),
        };
        if let Some(data) = Arc::get_mut(&mut self.data) {
            data.reserve(n.saturating_mul(per.max(1)));
        }
    }

    /// Prepend `text` to every entry, rebuilding the byte buffer.
    ///
    /// Preserves `FixedLength` storage when all input entries have equal
    /// length (since the prefix is uniform, the output stride is uniform too).
    #[must_use]
    pub fn prefix(self, text: &[u8]) -> Self {
        let n = self.len();
        let first_len = if n > 0 {
            self.entry_len(0) + text.len()
        } else {
            text.len()
        };
        let mut bld = StringPodBuilder::with_capacity(first_len, n);
        let mut buf = Vec::with_capacity(first_len);
        for i in 0..n {
            buf.clear();
            buf.extend_from_slice(text);
            buf.extend_from_slice(self.get(i));
            bld.push(&buf);
        }
        // Carry the edit history across the rebuild and append the prefix op.
        let mut out = bld.finish();
        out.edits = self.edits;
        out.record_prefix(text.len(), None);
        out
    }

    /// Append `text` to every entry, rebuilding the byte buffer.
    ///
    /// Preserves `FixedLength` storage when all input entries have equal
    /// length.
    #[must_use]
    pub fn postfix(self, text: &[u8]) -> Self {
        let n = self.len();
        let first_len = if n > 0 {
            self.entry_len(0) + text.len()
        } else {
            text.len()
        };
        let mut bld = StringPodBuilder::with_capacity(first_len, n);
        let mut buf = Vec::with_capacity(first_len);
        for i in 0..n {
            buf.clear();
            buf.extend_from_slice(self.get(i));
            buf.extend_from_slice(text);
            bld.push(&buf);
        }
        let mut out = bld.finish();
        out.edits = self.edits;
        out.record_postfix(text.len(), None);
        out
    }

    /// Truncate every entry to at most `n` bytes. Index-only — no bytes are
    /// copied and the buffer stays shared: O(1) for `FixedLength` (a uniform
    /// tail cut), an O(len) position rewrite for `Variable`. Truncated tails
    /// become unreferenced; reclaiming that space is a separate explicit step.
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are clipped; this promotes `FixedLength` → `Variable`.
    #[must_use]
    pub fn max_len(mut self, n: usize, conditional: Option<&[bool]>) -> Self {
        if let Some(cond) = conditional {
            let count = self.len();
            let windows: Vec<Option<(usize, usize, usize)>> = (0..count)
                .map(|i| {
                    let cur = self.entry_len(i);
                    (cond[i] && cur > n).then_some((0, n, cur))
                })
                .collect();
            let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
            self.storage.truncate_bytes_conditional(n_u32, cond);
            self.record_windows(&windows);
            return self;
        }
        if let Storage::FixedLength { visible_len, .. } = self.storage {
            let vl = visible_len as usize;
            if vl > n {
                // `cut_end` records the (uniform) coordinate edit itself.
                self.cut_end(vl - n, None);
            }
            return self;
        }
        // Variable + unconditional: narrow each entry to min(len, n) as an
        // index-only overlay. No bytes move and the buffer stays shared;
        // truncated tails become unreferenced (compaction is a separate step).
        let count = self.len();
        let windows: Vec<Option<(usize, usize, usize)>> = (0..count)
            .map(|i| {
                let cur = self.entry_len(i);
                (cur > n).then_some((0, n, cur))
            })
            .collect();
        self.storage.truncate_bytes(u32::try_from(n).unwrap_or(u32::MAX));
        self.record_windows(&windows);
        self
    }

    /// Reclaim space orphaned by index-only edits. The column is moved into a
    /// fresh, exactly-sized buffer (one allocation, no over-reserve) holding
    /// each entry's visible bytes contiguously in view order; any bytes hidden
    /// by cut / drop / slice / truncate overlays are dropped. After this
    /// `buffer_bytes() == used_bytes()` and the pod owns its buffer outright
    /// (the previous `Arc` is released, so a shared buffer is left untouched).
    ///
    /// This is the explicit, user-driven counterpart to the crate's
    /// index-only-by-default alterations — they never compact behind your back,
    /// so call this when, and only when, reclamation matters. The visible
    /// contents, entry count, fixed/variable layout and edit history are
    /// unchanged (no coordinate edit is recorded — the visible frame is
    /// identical, only its backing storage moves).
    ///
    /// # Panics
    /// If the compacted byte total or entry count exceeds `u32::MAX` (the same
    /// bound the builders enforce).
    pub fn compact(&mut self) {
        let count = self.len();
        let total: usize = (0..count).map(|i| self.entry_len(i)).sum();
        let was_fixed = self.storage.current_stride().is_some();
        let mut data = Vec::with_capacity(total);
        let mut positions: Vec<(u32, u32)> = if was_fixed {
            Vec::new()
        } else {
            Vec::with_capacity(count)
        };
        for i in 0..count {
            let start = u32::try_from(data.len()).expect("byte buffer exceeds u32::MAX");
            data.extend_from_slice(self.get(i));
            if !was_fixed {
                let stop = u32::try_from(data.len()).expect("byte buffer exceeds u32::MAX");
                positions.push((start, stop));
            }
        }
        let storage = if was_fixed {
            // FixedLength stays FixedLength: post-compaction stride == the (now
            // uniform, overlay-free) visible length.
            let stride = if count == 0 {
                self.storage.current_stride().unwrap_or(0)
            } else {
                u32::try_from(self.entry_len(0)).expect("entry length exceeds u32::MAX")
            };
            Storage::FixedLength {
                stride,
                head_skip: 0,
                visible_len: stride,
                count: u32::try_from(count).expect("entry count exceeds u32::MAX"),
                front_byte: 0,
            }
        } else {
            Storage::Variable(VariableInfo {
                positions,
                head_skip: 0,
                tail_skip: 0,
                front_skip: 0,
            })
        };
        self.data = Arc::new(data);
        self.storage = storage;
    }

    /// Generalized per-entry resize. For each entry `i` (in order), invokes
    /// `f(i, entry)` with the entry's current visible bytes. The closure
    /// returns:
    ///
    /// * `None` — leave the entry unchanged, or
    /// * `Some((start, len))` — narrow the entry to `entry[start..start + len]`.
    ///   The sub-range is relative to the entry's current visible bytes and
    ///   **must lie within them** (`start + len <= entry.len()`); otherwise this
    ///   panics.
    ///
    /// No bytes are copied — only metadata is rewritten — so the new region must
    /// fall inside the old one. This subsumes [`cut_start`](Self::cut_start),
    /// [`cut_end`](Self::cut_end), and [`max_len`](Self::max_len) for the case
    /// where each entry needs an individually-chosen cut (e.g. trimming every
    /// read at a per-read location). Promotes `FixedLength` → `Variable`.
    pub fn resize<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &BStr) -> Option<(usize, usize)>,
    {
        let mut windows: Vec<Option<(usize, usize, usize)>> = Vec::new();
        self.storage.resize_each(&self.data, |i, bytes| {
            let cur = bytes.len();
            let kept = f(i, bytes);
            windows.push(kept.map(|(st, ln)| (st, ln, cur)));
            kept
        });
        self.record_windows(&windows);
    }

    /// Reverse the bytes of every entry in-place. If the byte buffer is
    /// shared (Arc strong count > 1) it is cloned before reversing (COW).
    ///
    /// If `conditional` is `Some`, only entries where the boolean is `true`
    /// are reversed.
    #[must_use]
    pub fn reverse(mut self, conditional: Option<&[bool]>) -> Self {
        self.record_reverse(conditional);
        self.make_exclusive(); // compacting COW when shared; no-op when already exclusive
        let data = Arc::make_mut(&mut self.data);
        let n = self.storage.len();
        for i in 0..n {
            if conditional.is_none_or(|c| c[i]) {
                let r = self.storage.entry_range(i);
                data[r].reverse();
            }
        }
        self
    }

    /// Returns a mutable iterator over entries
    /// makes the buffer non-shared if shared
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.make_exclusive(); // compacting COW when shared; no-op when already exclusive
        let back = self.storage.len();
        // Disjoint fields: `data` borrowed mutably for the iterator, `storage`
        // immutably to drive the entry ranges.
        let buffer = Arc::make_mut(&mut self.data).as_mut_slice();
        IterMut {
            remaining: buffer,
            storage: &self.storage,
            front: 0,
            back,
            consumed: 0,
        }
    }

    /// Iterate visible entries as `&BStr` in order.
    #[must_use]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            pod: self,
            front: 0,
            back: self.len(),
        }
    }

    /// iterate across the lengths
    pub fn iter_lens(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(move |i| self.entry_len(i))
    }

    /// Start an alias builder that shares this pod's byte buffer. New entries
    /// pushed via the alias builder will reference bytes inside `self.data`
    /// without copying.
    ///
    /// The builder borrows `self` until [`StringPodAliasBuilder::finish`] is
    /// called; the source pod cannot be mutated while the builder is live.
    #[must_use]
    pub fn alias_builder(&self) -> StringPodAliasBuilder<'_> {
        StringPodAliasBuilder {
            source: self,
            next: 0,
            positions: Vec::new(),
        }
    }

    /// A pod over live entries `range`, sharing this pod's byte buffer — no bytes
    /// are copied. `FixedLength` stays `FixedLength` (O(1)); `Variable` copies
    /// only the range's positions (O(range)). The result reads byte-identical to
    /// `iter().skip(range.start).take(range.len())` and carries those entries'
    /// edit history.
    ///
    /// # Panics
    /// If `range.start > range.end` or `range.end > self.len()`.
    #[must_use]
    pub fn slice(&self, range: Range<usize>) -> StringPod {
        let edits = self.edits.slice(range.clone());
        StringPod {
            data: Arc::clone(&self.data),
            storage: self.storage.slice(range),
            edits,
        }
    }
}

impl<'a> IntoIterator for &'a mut StringPod {
    type Item = &'a mut bstr::BStr;
    type IntoIter = IterMut<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl std::fmt::Debug for StringPod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringPod")
            .field("len", &self.len())
            .field("fixed_length", &self.is_fixed_length())
            .field("buffer_bytes", &self.buffer_bytes())
            .field("used_bytes", &self.used_bytes())
            .finish()
    }
}

impl<'a> IntoIterator for &'a StringPod {
    type Item = &'a BStr;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

pub struct Iter<'a> {
    pod: &'a StringPod,
    front: usize,
    back: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a BStr;
    fn next(&mut self) -> Option<&'a BStr> {
        if self.front < self.back {
            let r = self.pod.get(self.front);
            self.front += 1;
            Some(r)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            Some(self.pod.get(self.back))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Iter<'_> {}

/// Mutable entry iterator. Holds the not-yet-yielded slice of the buffer and
/// peels each entry off with [`split_at_mut`](slice::split_at_mut) — no
/// `unsafe`, the same way `std`'s own `slice::IterMut` is shaped. This relies on
/// entries being ascending and non-overlapping in the buffer, which every
/// builder path upholds (the owning builder appends; the alias builder
/// sub-slices entries consumed in order).
pub struct IterMut<'a> {
    /// `buffer[consumed .. back_boundary]`: the bytes spanning live entries
    /// `front..back` plus the gaps (cuts, dropped fronts) between them.
    remaining: &'a mut [u8],
    storage: &'a Storage,
    front: usize,
    back: usize,
    /// Absolute buffer offset of `remaining[0]`; advances as the front is peeled.
    consumed: usize,
}

impl<'a> Iterator for IterMut<'a> {
    type Item = &'a mut BStr;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let r = self.storage.entry_range(self.front);
        self.front += 1;
        // The visible range sits at `r.start - consumed` into `remaining`: peel
        // the leading gap, then the entry, and keep the tail. `mem::take` hands
        // out the entry with the iterator's `'a`, not a borrow of `self`, so the
        // items outlive the iterator (collectable, like `slice::iter_mut`).
        debug_assert!(
            r.start >= self.consumed,
            "IterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
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

impl DoubleEndedIterator for IterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let r = self.storage.entry_range(self.back);
        // The back entry is the highest-offset live entry: keep everything
        // before it (`left`), hand out the entry, drop the trailing gap.
        debug_assert!(
            r.start >= self.consumed,
            "IterMut: entries must be ascending and non-overlapping (entry start {} < consumed {})",
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

impl ExactSizeIterator for IterMut<'_> {}

// ── owning builder ───────────────────────────────────────────────────────

/// Builds a [`StringPod`] by pushing owned byte strings into a fresh buffer.
/// Starts `FixedLength` with the supplied stride; auto-promotes to `Variable`
/// on the first length mismatch.
pub struct StringPodBuilder {
    data: Vec<u8>,
    storage: Storage,
}

impl StringPodBuilder {
    /// Create a builder, without preallocating anything.
    /// You probably want to use `with_capacity` if you have an idea
    /// of the target sizes
    #[must_use]
    #[expect(clippy::new_without_default, reason = "I don't want to promote this")]
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Create a builder reserving `entry_len * count` bytes. If `entry_len > 0`,
    /// storage starts `FixedLength` with stride `entry_len`; otherwise starts
    /// `Variable`.
    ///
    /// # Panics
    /// If `entry_len` exceeds `u32::MAX`.
    #[must_use]
    pub fn with_capacity(entry_len: usize, count: usize) -> Self {
        let byte_cap = entry_len.checked_mul(count).unwrap_or(0);
        let data = Vec::with_capacity(byte_cap);
        let storage = if entry_len == 0 {
            Storage::new_variable(count)
        } else {
            let stride = u32::try_from(entry_len).expect("entry_len exceeds u32");
            Storage::new_fixed(stride, count)
        };
        Self { data, storage }
    }

    /// Push one entry's bytes. If storage is `FixedLength` and `bytes.len()`
    /// doesn't match the stride, promotes to `Variable` first (one-time O(count)).
    ///
    /// # Panics
    /// If the byte buffer would exceed `u32::MAX` total bytes.
    pub fn push(&mut self, bytes: &[u8]) {
        if let Some(stride) = self.storage.current_stride() {
            if bytes.len() as u64 == u64::from(stride) {
                self.data.extend_from_slice(bytes);
                self.storage.builder_push_strided();
                return;
            }
        }
        // Variable (or promoted) path
        let start_usize = self.data.len();
        let start = u32::try_from(start_usize).expect("byte buffer exceeds u32::MAX");
        let stop = u32::try_from(start_usize + bytes.len()).expect("byte buffer exceeds u32::MAX");
        self.data.extend_from_slice(bytes);
        self.storage.builder_push_position(start, stop);
    }

    /// Append entries `range` from a finished [`StringPod`] *en bloc*.
    ///
    /// This is the bulk equivalent of calling [`push`](Self::push) once per
    /// entry, but instead of recomputing a range and doing a tiny `memcpy` per
    /// entry it copies the whole contiguous source span in one shot. When the
    /// source entries are contiguous in their buffer (the common
    /// freshly-built case) this collapses `n` small copies into one `memcpy`.
    ///
    /// A `FixedLength` source is appended as a single strided `memcpy` with no
    /// per-entry metadata at all, and the destination *stays* `FixedLength`
    /// when its layout matches — or, if the builder is still empty, it adopts
    /// the source's stride and cut overlay. Fixed-length reads are the common
    /// case, so this keeps the produced column fixed-length (and its downstream
    /// cuts O(1)) instead of needlessly promoting to `Variable`.
    ///
    /// Otherwise it falls back to copying the visible span and recording one
    /// position per entry (promoting to `Variable`). Any bytes hidden by the
    /// source's per-entry cut overlay (e.g. a stripped leading `@`) ride along
    /// in the copied span but stay hidden.
    ///
    /// # Panics
    /// If `range.end > src.len()` or the byte buffer would exceed `u32::MAX`.
    pub fn extend_from_pod(&mut self, src: &StringPod, range: Range<usize>) {
        assert!(range.end <= src.len(), "range past end of source pod");
        if range.start >= range.end {
            return;
        }
        let n = range.end - range.start;

        // Fast path: a FixedLength source appends as one strided memcpy and the
        // destination keeps a fixed layout (matching, or empty → adopt).
        if let Storage::FixedLength {
            stride,
            head_skip,
            visible_len,
            front_byte,
            ..
        } = src.storage
        {
            let dest_empty = self.storage.is_empty() && self.data.is_empty();
            let compatible = match &self.storage {
                Storage::FixedLength {
                    stride: ds,
                    head_skip: dh,
                    visible_len: dv,
                    ..
                } => *ds == stride && *dh == head_skip && *dv == visible_len,
                Storage::Variable(_) => dest_empty,
            };
            if compatible {
                let s = stride as usize;
                let raw_lo = front_byte as usize + range.start * s;
                let raw_hi = front_byte as usize + range.end * s;
                self.data.extend_from_slice(&src.data[raw_lo..raw_hi]);
                let added = u32::try_from(n).expect("entry count exceeds u32::MAX");
                match &mut self.storage {
                    Storage::FixedLength { count, .. } => {
                        *count = count
                            .checked_add(added)
                            .expect("StringPod count exceeded u32::MAX");
                    }
                    // Empty builder: adopt the source's stride + cut overlay.
                    Storage::Variable(_) => {
                        self.storage = Storage::FixedLength {
                            stride,
                            head_skip,
                            visible_len,
                            count: added,
                            front_byte: 0,
                        };
                    }
                }
                return;
            }
        }

        // General path: one memcpy of the visible span, one position per entry.
        // Entries are ascending and non-overlapping, so [first.start, last.end)
        // spans the whole run (plus any interior cut gaps, which stay hidden).
        let span_start = src.storage.entry_range(range.start).start;
        let span_end = src.storage.entry_range(range.end - 1).end;
        let dest_base = self.data.len();
        self.data.extend_from_slice(&src.data[span_start..span_end]);

        self.storage.promote_to_variable();
        let Storage::Variable(info) = &mut self.storage else {
            unreachable!("just promoted to Variable")
        };
        info.positions.reserve(n);
        for i in range {
            let r = src.storage.entry_range(i);
            // A source offset `p` was copied to `p - span_start + dest_base`.
            let start = u32::try_from(r.start - span_start + dest_base)
                .expect("byte buffer exceeds u32::MAX");
            let stop = u32::try_from(r.end - span_start + dest_base)
                .expect("byte buffer exceeds u32::MAX");
            info.positions.push((start, stop));
        }
    }

    /// Number of entries pushed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Bytes written to the buffer so far (including any wasted slack in
    /// `FixedLength` slots — currently none, but documented for symmetry).
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.data.len()
    }

    /// Finalise the builder. Wraps the byte buffer in `Arc<Vec<u8>>` without
    /// reallocating (any over-reserved capacity is retained, not copied away).
    #[must_use]
    pub fn finish(self) -> StringPod {
        let count = self.storage.len();
        StringPod {
            data: Arc::new(self.data),
            storage: self.storage,
            edits: ColumnEdits::new(count),
        }
    }
}

// ── alias builder ────────────────────────────────────────────────────────

/// Builds a [`StringPod`] whose entries reference bytes in an existing pod's
/// `Arc<[u8]>` without copying.
///
/// Each [`push_alias`](StringPodAliasBuilder::push_alias) call consumes the
/// *next* source entry in order, so aliases are guaranteed to be
/// non-overlapping (they are sub-ranges of distinct, non-overlapping source
/// entries). At most `source.len()` aliases may be pushed.
///
/// The source pod is borrowed for the builder's lifetime and released on
/// [`finish`](StringPodAliasBuilder::finish). The resulting pod co-owns the
/// source bytes; subsequent mutations of the source pod do not affect the
/// alias pod (snapshot semantics).
pub struct StringPodAliasBuilder<'a> {
    source: &'a StringPod,
    next: usize,
    positions: Vec<(u32, u32)>,
}

impl StringPodAliasBuilder<'_> {
    /// Alias the next source entry, taking `source_entry[offset..offset+len]`.
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
        let abs_start = u32::try_from(r.start + offset).expect("alias start exceeds u32");
        let abs_end = u32::try_from(r.start + end).expect("alias end exceeds u32");
        self.positions.push((abs_start, abs_end));
        self.next += 1;
    }

    /// Number of alias entries pushed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Finalise the alias builder, releasing the borrow of the source pod.
    /// The resulting pod always has `Variable` storage (entries reference
    /// non-contiguous regions of the shared buffer).
    #[must_use]
    pub fn finish(self) -> StringPod {
        let count = self.positions.len();
        StringPod {
            data: Arc::clone(&self.source.data),
            storage: Storage::Variable(VariableInfo {
                positions: self.positions,
                head_skip: 0,
                tail_skip: 0,
                front_skip: 0,
            }),
            edits: ColumnEdits::new(count),
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "it's tests")]
mod tests {
    use super::{StringPod, StringPodBuilder};
    use bstr::BStr;

    fn b(s: &str) -> &[u8] {
        s.as_bytes()
    }

    #[test]
    fn empty_pod() {
        let p = StringPod::empty();
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
        assert_eq!(p.used_bytes(), 0);
        assert_eq!(p.buffer_bytes(), 0);
        assert_eq!(p.iter().count(), 0);
    }

    #[test]
    fn fixed_length_basic() {
        let mut bld = StringPodBuilder::with_capacity(3, 4);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        bld.push(b("CCC"));
        let p = bld.finish();
        assert_eq!(p.len(), 3);
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("AAA"));
        assert_eq!(p.get(1), BStr::new("BBB"));
        assert_eq!(p.get(2), BStr::new("CCC"));
        assert_eq!(p.used_bytes(), 9);
        assert_eq!(p.buffer_bytes(), 9);
        let collected: Vec<&BStr> = p.iter().collect();
        assert_eq!(
            collected,
            vec![BStr::new("AAA"), BStr::new("BBB"), BStr::new("CCC")]
        );
    }

    #[test]
    fn fixed_length_promotes_on_mismatch() {
        let mut bld = StringPodBuilder::with_capacity(3, 4);
        bld.push(b("AAA"));
        bld.push(b("BB")); // promotes
        bld.push(b("CCC"));
        let p = bld.finish();
        assert!(!p.is_fixed_length());
        assert_eq!(p.len(), 3);
        assert_eq!(p.get(0), BStr::new("AAA"));
        assert_eq!(p.get(1), BStr::new("BB"));
        assert_eq!(p.get(2), BStr::new("CCC"));
        assert_eq!(p.used_bytes(), 8);
    }

    #[test]
    fn variable_from_start() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("hello"));
        bld.push(b("hi"));
        bld.push(b("foobar"));
        let p = bld.finish();
        assert!(!p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("hello"));
        assert_eq!(p.get(1), BStr::new("hi"));
        assert_eq!(p.get(2), BStr::new("foobar"));
        assert_eq!(p.used_bytes(), 13);
    }

    #[test]
    fn cut_start_fixed() {
        let mut bld = StringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        let mut p = bld.finish();
        p.cut_start(2, None);
        assert_eq!(p.get(0), BStr::new("LLO"));
        assert_eq!(p.get(1), BStr::new("RLD"));
        assert_eq!(p.entry_len(0), 3);
        assert_eq!(p.used_bytes(), 6);
        assert_eq!(p.buffer_bytes(), 10); // bytes still there
        // double cut
        p.cut_start(1, None);
        assert_eq!(p.get(0), BStr::new("LO"));
    }

    #[test]
    fn cut_end_fixed() {
        let mut bld = StringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        let mut p = bld.finish();
        p.cut_end(2, None);
        assert_eq!(p.get(0), BStr::new("HEL"));
        assert_eq!(p.get(1), BStr::new("WOR"));
    }

    #[test]
    fn cut_start_then_cut_end_fixed() {
        let mut bld = StringPodBuilder::with_capacity(6, 2);
        bld.push(b("ABCDEF"));
        bld.push(b("UVWXYZ"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None);
        assert_eq!(p.get(0), BStr::new("BCDE"));
        assert_eq!(p.get(1), BStr::new("VWXY"));
    }

    #[test]
    fn cut_oversaturates_fixed() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let mut p = bld.finish();
        p.cut_start(100, None);
        assert_eq!(p.get(0), BStr::new(""));
        assert_eq!(p.get(1), BStr::new(""));
        assert_eq!(p.entry_len(0), 0);
    }

    #[test]
    fn cut_oversaturates_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("AB"));
        bld.push(b("XYZWQ"));
        let mut p = bld.finish();
        p.cut_start(3, None);
        // entry 0 ("AB") is shorter than 3 — saturates to empty
        assert_eq!(p.get(0), BStr::new(""));
        // entry 1 ("XYZWQ") yields "WQ"
        assert_eq!(p.get(1), BStr::new("WQ"));
    }

    #[test]
    fn cut_end_oversaturates_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("AB"));
        bld.push(b("XYZWQ"));
        let mut p = bld.finish();
        p.cut_end(3, None);
        assert_eq!(p.get(0), BStr::new(""));
        assert_eq!(p.get(1), BStr::new("XY"));
    }

    #[test]
    fn cut_start_and_end_combined_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b("ABCDEFG"));
        let mut p = bld.finish();
        p.cut_start(2, None);
        p.cut_end(2, None);
        assert_eq!(p.get(0), BStr::new("CDE"));
    }

    #[test]
    fn cut_start_and_end_saturate_to_empty_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b("ABC"));
        let mut p = bld.finish();
        p.cut_start(2, None);
        p.cut_end(5, None); // would dip negative
        assert_eq!(p.get(0), BStr::new(""));
    }

    #[test]
    fn drain_fixed_promotes() {
        let mut bld = StringPodBuilder::with_capacity(2, 5);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        bld.push(b("DD"));
        bld.push(b("EE"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        p.drain(1..3);
        assert!(!p.is_fixed_length());
        assert_eq!(p.len(), 3);
        assert_eq!(p.get(0), BStr::new("AA"));
        assert_eq!(p.get(1), BStr::new("DD"));
        assert_eq!(p.get(2), BStr::new("EE"));
    }

    #[test]
    fn drain_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("alpha"));
        bld.push(b("beta"));
        bld.push(b("gamma"));
        let mut p = bld.finish();
        p.drain(0..1);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new("beta"));
        assert_eq!(p.get(1), BStr::new("gamma"));
    }

    #[test]
    fn drain_with_cut_preserves_cut() {
        let mut bld = StringPodBuilder::with_capacity(4, 3);
        bld.push(b("ABCD"));
        bld.push(b("EFGH"));
        bld.push(b("IJKL"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None);
        p.drain(1..2); // promotes, removes "FG"
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new("BC"));
        assert_eq!(p.get(1), BStr::new("JK"));
    }

    #[test]
    fn drain_empty_range_noop() {
        let mut bld = StringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        let mut p = bld.finish();
        p.drain(1..1);
        assert!(p.is_fixed_length());
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn iter_double_ended() {
        let mut bld = StringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        let p = bld.finish();
        let mut it = p.iter();
        assert_eq!(it.next().unwrap(), BStr::new("AA"));
        assert_eq!(it.next_back().unwrap(), BStr::new("CC"));
        assert_eq!(it.next().unwrap(), BStr::new("BB"));
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
    }

    #[test]
    fn iter_exact_size() {
        let mut bld = StringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        let p = bld.finish();
        let it = p.iter();
        assert_eq!(it.len(), 3);
        assert_eq!(it.size_hint(), (3, Some(3)));
    }

    #[test]
    fn into_iter_for_reference() {
        let mut bld = StringPodBuilder::with_capacity(2, 2);
        bld.push(b("AA"));
        bld.push(b("BB"));
        let p = bld.finish();
        let mut count = 0;
        for _ in &p {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn clone_is_cheap_arc_share() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let p = bld.finish();
        let q = p.clone();
        // Same underlying bytes (Arc pointer equality)
        assert!(std::ptr::eq(p.data.as_ref(), q.data.as_ref()));
        assert_eq!(p.get(0), q.get(0));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_out_of_bounds_panics_fixed() {
        let mut bld = StringPodBuilder::with_capacity(2, 2);
        bld.push(b("AA"));
        let p = bld.finish();
        let _ = p.get(5);
    }

    #[test]
    fn alias_builder_basic() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("FOOBAR"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(1, 3); // entry 0 "HELLO"  → "ELL"
        ab.push_alias(0, 5); // entry 1 "WORLD"  → "WORLD"
        ab.push_alias(2, 3); // entry 2 "FOOBAR" → "OBA"
        let aliased = ab.finish();
        assert_eq!(aliased.len(), 3);
        assert_eq!(aliased.get(0), BStr::new("ELL"));
        assert_eq!(aliased.get(1), BStr::new("WORLD"));
        assert_eq!(aliased.get(2), BStr::new("OBA"));
    }

    #[test]
    fn alias_builder_zero_length_entry() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b("hello"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(3, 0); // entry 0 "hello", offset 3, len 0 → ""
        let aliased = ab.finish();
        assert_eq!(aliased.get(0), BStr::new(""));
    }

    #[test]
    #[should_panic(expected = "exceeds entry")]
    fn alias_builder_out_of_bounds_panics() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b("hello"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(3, 10); // offset 3 + len 10 = 13 > entry len 5
    }

    #[test]
    #[should_panic(expected = "already consumed")]
    fn alias_builder_too_many_entries_panics() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("AA"));
        bld.push(b("BB"));
        let source = bld.finish();
        let mut ab = source.alias_builder();
        ab.push_alias(0, 2);
        ab.push_alias(0, 2);
        ab.push_alias(0, 1); // third push on a 2-entry source — panics
    }

    #[test]
    fn alias_pod_snapshot_survives_source_drain() {
        let mut bld = StringPodBuilder::with_capacity(3, 3);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        bld.push(b("CCC"));
        let mut source = bld.finish();
        let aliased = {
            let mut ab = source.alias_builder();
            ab.push_alias(0, 3); // entry 0 "AAA" → "AAA"
            ab.push_alias(0, 3); // entry 1 "BBB" → "BBB"
            ab.finish()
        };
        // Mutate source — drain promotes and drops entry 1
        source.drain(1..2);
        assert_eq!(source.len(), 2);
        // Alias still sees original bytes
        assert_eq!(aliased.get(0), BStr::new("AAA"));
        assert_eq!(aliased.get(1), BStr::new("BBB"));
    }

    #[test]
    fn alias_pod_snapshot_survives_source_cuts() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b("ABCDEFGH"));
        let mut source = bld.finish();
        let aliased = {
            let mut ab = source.alias_builder();
            ab.push_alias(0, 8); // entry 0, full range
            ab.finish()
        };
        source.cut_start(3, None);
        source.cut_end(3, None);
        // Source is now "DE" visible
        assert_eq!(source.get(0), BStr::new("DE"));
        // Alias still sees the full original
        assert_eq!(aliased.get(0), BStr::new("ABCDEFGH"));
    }

    // ── slice ───────────────────────────────────────────────────────────────

    #[test]
    fn slice_fixed_stays_fixed_and_shares_buffer() {
        let mut bld = StringPodBuilder::with_capacity(3, 4);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        bld.push(b("CCC"));
        bld.push(b("DDD"));
        let p = bld.finish();
        let s = p.slice(1..3);
        assert!(s.is_fixed_length());
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), BStr::new("BBB"));
        assert_eq!(s.get(1), BStr::new("CCC"));
        // No bytes copied: same underlying Arc buffer.
        assert!(std::ptr::eq(p.data.as_ref(), s.data.as_ref()));
        // Original untouched.
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn slice_variable_matches_skip_take() {
        let mut bld = StringPodBuilder::with_capacity(0, 4);
        bld.push(b("hello"));
        bld.push(b("hi"));
        bld.push(b("foobar"));
        bld.push(b("x"));
        let p = bld.finish();
        for start in 0..=p.len() {
            for end in start..=p.len() {
                let s = p.slice(start..end);
                let want: Vec<&BStr> = p.iter().skip(start).take(end - start).collect();
                let got: Vec<&BStr> = s.iter().collect();
                assert_eq!(got, want, "slice {start}..{end}");
                assert!(std::ptr::eq(p.data.as_ref(), s.data.as_ref()));
            }
        }
    }

    #[test]
    fn slice_preserves_cut_overlay_fixed() {
        let mut bld = StringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("RUSTY"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None);
        let s = p.slice(1..3);
        assert!(s.is_fixed_length());
        assert_eq!(s.get(0), BStr::new("ORL"));
        assert_eq!(s.get(1), BStr::new("UST"));
    }

    #[test]
    fn slice_under_pop_front_fixed() {
        let mut bld = StringPodBuilder::with_capacity(2, 4);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        bld.push(b("DD"));
        let mut p = bld.finish();
        p.pop_front(1); // live view: BB CC DD
        let s = p.slice(0..2);
        assert_eq!(s.get(0), BStr::new("BB"));
        assert_eq!(s.get(1), BStr::new("CC"));
    }

    #[test]
    fn slice_variable_under_overlays() {
        let mut bld = StringPodBuilder::with_capacity(0, 4);
        bld.push(b("alpha"));
        bld.push(b("beta"));
        bld.push(b("gamma"));
        bld.push(b("delta"));
        let mut p = bld.finish();
        p.cut_start(1, None); // promotes nothing; already variable
        p.pop_front(1); // live: eta amma elta (after cut_start)
        let s = p.slice(1..3);
        let want: Vec<&BStr> = p.iter().skip(1).take(2).collect();
        let got: Vec<&BStr> = s.iter().collect();
        assert_eq!(got, want);
    }

    #[test]
    fn slice_empty_range() {
        let mut bld = StringPodBuilder::with_capacity(2, 2);
        bld.push(b("AA"));
        bld.push(b("BB"));
        let p = bld.finish();
        let s = p.slice(1..1);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    #[should_panic(expected = "slice range past end")]
    fn slice_out_of_bounds_panics() {
        let mut bld = StringPodBuilder::with_capacity(2, 2);
        bld.push(b("AA"));
        bld.push(b("BB"));
        let p = bld.finish();
        let _ = p.slice(1..5);
    }

    #[test]
    fn used_bytes_vs_buffer_bytes_after_drain() {
        let mut bld = StringPodBuilder::with_capacity(3, 4);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        bld.push(b("CCC"));
        bld.push(b("DDD"));
        let mut p = bld.finish();
        assert_eq!(p.used_bytes(), 12);
        assert_eq!(p.buffer_bytes(), 12);
        p.drain(1..3);
        // 2 entries × 3 bytes = 6 visible; 12 in buffer (drain doesn't reclaim)
        assert_eq!(p.used_bytes(), 6);
        assert_eq!(p.buffer_bytes(), 12);
    }

    #[test]
    fn builder_len_tracks() {
        let mut bld = StringPodBuilder::with_capacity(2, 4);
        assert_eq!(bld.len(), 0);
        assert!(bld.is_empty());
        bld.push(b("AB"));
        assert_eq!(bld.len(), 1);
        bld.push(b("CD"));
        assert_eq!(bld.len(), 2);
    }

    #[test]
    fn rebuild_pattern_produces_fixed_again() {
        // Variable old pod, but transform yields uniform output → new pod is FixedLength.
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("hello"));
        bld.push(b("hi"));
        bld.push(b("yo"));
        let old = bld.finish();
        let mut nb = StringPodBuilder::with_capacity(3, old.len());
        for s in &old {
            let mut buf = vec![b'X'; 3];
            let take = s.len().min(3);
            buf[..take].copy_from_slice(&s[..take]);
            nb.push(&buf);
        }
        let new = nb.finish();
        assert!(new.is_fixed_length());
        assert_eq!(new.len(), 3);
        assert_eq!(new.get(0), BStr::new("hel"));
        assert_eq!(new.get(1), BStr::new("hiX"));
        assert_eq!(new.get(2), BStr::new("yoX"));
    }

    #[test]
    fn fixed_then_variable_zero_byte_entries() {
        // stride=0 means "all entries empty" until something different shows up.
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b(""));
        bld.push(b(""));
        let p = bld.finish();
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new(""));
        assert_eq!(p.get(1), BStr::new(""));
    }

    #[test]
    fn very_small_push_after_large_promotes_correctly() {
        let mut bld = StringPodBuilder::with_capacity(10, 3);
        bld.push(b("0123456789"));
        bld.push(b("abcdefghij"));
        bld.push(b("x")); // promotes
        let p = bld.finish();
        assert_eq!(p.get(0), BStr::new("0123456789"));
        assert_eq!(p.get(1), BStr::new("abcdefghij"));
        assert_eq!(p.get(2), BStr::new("x"));
    }

    #[test]
    fn debug_format_does_not_panic() {
        let mut bld = StringPodBuilder::with_capacity(2, 2);
        bld.push(b("AB"));
        bld.push(b("CD"));
        let p = bld.finish();
        let s = format!("{p:?}");
        assert!(s.contains("StringPod"));
    }

    // ── prefix / postfix ──────────────────────────────────────────────────

    #[test]
    fn prefix_fixed_prepends_and_stays_fixed() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let p = bld.finish().prefix(b("XX"));
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("XXAAA"));
        assert_eq!(p.get(1), BStr::new("XXBBB"));
    }

    #[test]
    fn prefix_variable_prepends() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("hello"));
        bld.push(b("hi"));
        let p = bld.finish().prefix(b("--"));
        assert_eq!(p.get(0), BStr::new("--hello"));
        assert_eq!(p.get(1), BStr::new("--hi"));
    }

    #[test]
    fn postfix_fixed_appends_and_stays_fixed() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let p = bld.finish().postfix(b("ZZ"));
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("AAAZZ"));
        assert_eq!(p.get(1), BStr::new("BBBZZ"));
    }

    #[test]
    fn postfix_variable_appends() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("hello"));
        bld.push(b("hi"));
        let p = bld.finish().postfix(b("!!"));
        assert_eq!(p.get(0), BStr::new("hello!!"));
        assert_eq!(p.get(1), BStr::new("hi!!"));
    }

    // ── max_len ───────────────────────────────────────────────────────────

    #[test]
    fn max_len_fixed_noop_when_under() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let p = bld.finish().max_len(5, None);
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("AAA"));
    }

    #[test]
    fn max_len_fixed_clips() {
        let mut bld = StringPodBuilder::with_capacity(5, 2);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        let p = bld.finish().max_len(3, None);
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("HEL"));
        assert_eq!(p.get(1), BStr::new("WOR"));
    }

    #[test]
    fn max_len_variable_clips_each_entry() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDE")); // 5 → 3
        bld.push(b("AB")); // 2, already ≤ 3
        bld.push(b("ABCDEFG")); // 7 → 3
        let p = bld.finish().max_len(3, None);
        assert_eq!(p.get(0), BStr::new("ABC"));
        assert_eq!(p.get(1), BStr::new("AB"));
        assert_eq!(p.get(2), BStr::new("ABC"));
    }

    // ── compact ───────────────────────────────────────────────────────────

    #[test]
    fn compact_fixed_reclaims_overlay_orphans_and_stays_fixed() {
        let mut bld = StringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("RUSTY"));
        let mut p = bld.finish();
        // Index-only edits leave orphaned bytes: cut both ends and drop a row.
        p.cut_start(1, None); // "ELLO" / "ORLD" / "USTY"
        p.cut_end(1, None); // "ELL"  / "ORL"  / "UST"
        p.pop_front(1); // drop row 0
        assert_eq!(p.len(), 2);
        assert!(p.buffer_bytes() > p.used_bytes()); // orphans present

        p.compact();

        assert!(p.is_fixed_length());
        assert_eq!(p.buffer_bytes(), p.used_bytes());
        assert_eq!(p.buffer_bytes(), 6); // 2 entries × 3 visible bytes
        assert_eq!(p.get(0), BStr::new("ORL"));
        assert_eq!(p.get(1), BStr::new("UST"));
    }

    #[test]
    fn compact_variable_reclaims_and_preserves_contents() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAA")); // 4
        bld.push(b("BBBBBB")); // 6
        bld.push(b("CC")); // 2
        let mut p = bld.finish().max_len(3, None); // "AAA" / "BBB" / "CC", orphans remain
        p.drain(0..1); // drop "AAA"
        assert_eq!(p.len(), 2);
        assert!(p.buffer_bytes() > p.used_bytes());

        p.compact();

        assert!(!p.is_fixed_length());
        assert_eq!(p.buffer_bytes(), p.used_bytes());
        assert_eq!(p.buffer_bytes(), 5); // "BBB" + "CC"
        assert_eq!(p.get(0), BStr::new("BBB"));
        assert_eq!(p.get(1), BStr::new("CC"));
    }

    #[test]
    fn compact_clones_a_shared_buffer() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let original = bld.finish();
        let mut view = original.slice(1..2); // shares the Arc, only "BBB" visible

        view.compact();

        // The shared original is untouched; the compacted view owns tight bytes.
        assert_eq!(original.len(), 2);
        assert_eq!(original.buffer_bytes(), 6);
        assert_eq!(view.buffer_bytes(), 3);
        assert_eq!(view.get(0), BStr::new("BBB"));
    }

    #[test]
    fn mutating_many_slices_of_one_buffer_does_not_amplify() {
        // Build one buffer, slice N descendants over it (sharing the Arc), then
        // mutate every descendant. Each must COW only its own footprint, so the
        // total stays one buffer's worth instead of N × |buffer|.
        let mut bld = StringPodBuilder::with_capacity(4, 8);
        for i in 0..8u8 {
            bld.push(&[b'A' + i, b'C', b'G', b'T']);
        }
        let original = bld.finish();
        let full = original.buffer_bytes();
        assert_eq!(full, 32);

        let mut views: Vec<_> = (0..8).map(|i| original.slice(i..i + 1)).collect();
        for v in &views {
            assert_eq!(v.buffer_bytes(), full); // zero-copy slices share the buffer
        }
        for v in &mut views {
            for e in v.iter_mut() {
                e.make_ascii_lowercase();
            }
        }

        let total: usize = views.iter().map(StringPod::buffer_bytes).sum();
        assert_eq!(total, full); // 8 × 4 == 32, not 8 × 32
        for (i, v) in views.iter().enumerate() {
            assert_eq!(v.buffer_bytes(), 4);
            assert_eq!(
                v.get(0),
                BStr::new(&[b'a' + u8::try_from(i).unwrap(), b'c', b'g', b't'])
            );
        }
        assert_eq!(original.buffer_bytes(), full); // shared source untouched
        assert_eq!(original.get(0), BStr::new("ACGT"));
    }

    #[test]
    fn make_exclusive_leaves_an_owned_pod_uncompacted() {
        // When the pod already owns its buffer, `make_exclusive` neither copies
        // nor compacts — orphans from prior index-only edits stay resident.
        let mut bld = StringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"));
        bld.push(b("TTGA"));
        let mut p = bld.finish().max_len(2, None); // index-only clip → orphans
        assert!(p.buffer_bytes() > p.used_bytes());
        let before = p.buffer_bytes();

        p.make_exclusive();

        assert_eq!(p.buffer_bytes(), before); // untouched: no compaction
        assert!(p.buffer_bytes() > p.used_bytes());
    }

    #[test]
    fn compact_empty_is_a_noop_sized_buffer() {
        let p_empty = StringPod::empty();
        let mut p = p_empty;
        p.compact();
        assert_eq!(p.len(), 0);
        assert_eq!(p.buffer_bytes(), 0);
    }

    // ── reverse ───────────────────────────────────────────────────────────

    #[test]
    fn reverse_fixed() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("ABC"));
        bld.push(b("XYZ"));
        let p = bld.finish().reverse(None);
        assert_eq!(p.get(0), BStr::new("CBA"));
        assert_eq!(p.get(1), BStr::new("ZYX"));
    }

    #[test]
    fn reverse_cow_clones_shared_arc() {
        let mut bld = StringPodBuilder::with_capacity(3, 1);
        bld.push(b("ABC"));
        let p = bld.finish();
        let q = p.clone(); // bump Arc refcount
        let r = p.reverse(None); // COW — must clone
        // Original clone still sees "ABC"
        assert_eq!(q.get(0), BStr::new("ABC"));
        assert_eq!(r.get(0), BStr::new("CBA"));
    }

    // ── conditional operations ────────────────────────────────────────────

    #[test]
    fn cut_start_conditional_fixed_promotes() {
        let mut bld = StringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("RUST!"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        // only trim entry 0 and 2
        p.cut_start(2, Some(&[true, false, true]));
        assert!(!p.is_fixed_length()); // promoted
        assert_eq!(p.get(0), BStr::new("LLO"));
        assert_eq!(p.get(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.get(2), BStr::new("ST!"));
    }

    #[test]
    fn cut_start_conditional_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDE"));
        bld.push(b("XY"));
        bld.push(b("FOOBAR"));
        let mut p = bld.finish();
        p.cut_start(2, Some(&[false, true, false]));
        assert_eq!(p.get(0), BStr::new("ABCDE"));
        assert_eq!(p.get(1), BStr::new("")); // "XY" len=2, cut 2 → empty
        assert_eq!(p.get(2), BStr::new("FOOBAR"));
    }

    #[test]
    fn cut_end_conditional_fixed_promotes() {
        let mut bld = StringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("RUST!"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        p.cut_end(2, Some(&[true, false, true]));
        assert!(!p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("HEL"));
        assert_eq!(p.get(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.get(2), BStr::new("RUS"));
    }

    #[test]
    fn cut_end_conditional_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDE"));
        bld.push(b("XY"));
        bld.push(b("FOOBAR"));
        let mut p = bld.finish();
        p.cut_end(3, Some(&[true, false, true]));
        assert_eq!(p.get(0), BStr::new("AB"));
        assert_eq!(p.get(1), BStr::new("XY")); // untouched
        assert_eq!(p.get(2), BStr::new("FOO"));
    }

    #[test]
    fn cut_end_conditional_saturates() {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("AB"));
        bld.push(b("XYZWQ"));
        let mut p = bld.finish();
        // cut 10 from end only for entry 0 (len=2): saturates to empty
        p.cut_end(10, Some(&[true, false]));
        assert_eq!(p.get(0), BStr::new(""));
        assert_eq!(p.get(1), BStr::new("XYZWQ"));
    }

    #[test]
    fn truncate_drops_entries() {
        let mut bld = StringPodBuilder::with_capacity(2, 4);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        bld.push(b("DD"));
        let mut p = bld.finish();
        p.truncate(2);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new("AA"));
        assert_eq!(p.get(1), BStr::new("BB"));
    }

    #[test]
    fn max_len_conditional_fixed_promotes() {
        let mut bld = StringPodBuilder::with_capacity(5, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("RUST!"));
        let p = bld.finish().max_len(3, Some(&[true, false, true]));
        assert!(!p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("HEL"));
        assert_eq!(p.get(1), BStr::new("WORLD")); // untouched
        assert_eq!(p.get(2), BStr::new("RUS"));
    }

    #[test]
    fn max_len_conditional_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDE")); // 5 → 3
        bld.push(b("AB")); // 2, untouched (cond=false)
        bld.push(b("FOOBAR")); // 6 → 3
        let p = bld.finish().max_len(3, Some(&[true, false, true]));
        assert_eq!(p.get(0), BStr::new("ABC"));
        assert_eq!(p.get(1), BStr::new("AB"));
        assert_eq!(p.get(2), BStr::new("FOO"));
    }

    #[test]
    fn reverse_conditional_only_marked() {
        let mut bld = StringPodBuilder::with_capacity(3, 3);
        bld.push(b("ABC"));
        bld.push(b("DEF"));
        bld.push(b("GHI"));
        let p = bld.finish().reverse(Some(&[true, false, true]));
        assert_eq!(p.get(0), BStr::new("CBA"));
        assert_eq!(p.get(1), BStr::new("DEF")); // untouched
        assert_eq!(p.get(2), BStr::new("IHG"));
    }

    #[test]
    fn reverse_conditional_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("hello"));
        bld.push(b("world"));
        bld.push(b("rust"));
        let p = bld.finish().reverse(Some(&[false, true, false]));
        assert_eq!(p.get(0), BStr::new("hello"));
        assert_eq!(p.get(1), BStr::new("dlrow"));
        assert_eq!(p.get(2), BStr::new("rust"));
    }

    // ── conditional ops compose with pop_front (front_skip) ─────────────────

    #[test]
    fn cut_start_conditional_after_pop_front() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAAA"));
        bld.push(b("BBB")); // unequal length → Variable storage
        bld.push(b("CCCCC"));
        let mut p = bld.finish();
        assert!(!p.is_fixed_length());
        p.pop_front(1); // view: "BBB", "CCCCC" (front_skip = 1)
        p.cut_start(2, Some(&[true, false]));
        assert_eq!(p.get(0), BStr::new("B")); // "BBB" minus 2 from front
        assert_eq!(p.get(1), BStr::new("CCCCC")); // untouched
    }

    #[test]
    fn cut_end_conditional_after_pop_front() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAAA"));
        bld.push(b("BBB"));
        bld.push(b("CCCCC"));
        let mut p = bld.finish();
        p.pop_front(1); // view: "BBB", "CCCCC"
        p.cut_end(2, Some(&[false, true]));
        assert_eq!(p.get(0), BStr::new("BBB")); // untouched
        assert_eq!(p.get(1), BStr::new("CCC")); // "CCCCC" minus 2 from end
    }

    #[test]
    fn max_len_conditional_after_pop_front() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAAA"));
        bld.push(b("BBB"));
        bld.push(b("CCCCC"));
        let mut p = bld.finish();
        p.pop_front(1); // view: "BBB", "CCCCC"
        let p = p.max_len(2, Some(&[true, true]));
        assert_eq!(p.get(0), BStr::new("BB"));
        assert_eq!(p.get(1), BStr::new("CC"));
    }

    // ── resize ────────────────────────────────────────────────────────────

    #[test]
    fn resize_per_entry_cut_fixed_promotes() {
        let mut bld = StringPodBuilder::with_capacity(6, 3);
        bld.push(b("HELLO!"));
        bld.push(b("WORLD!"));
        bld.push(b("RUST!!"));
        let mut p = bld.finish();
        assert!(p.is_fixed_length());
        // Cut each entry at a per-entry location: keep [i..i+2].
        p.resize(|i, _| Some((i, 2)));
        assert!(!p.is_fixed_length()); // promoted
        assert_eq!(p.get(0), BStr::new("HE"));
        assert_eq!(p.get(1), BStr::new("OR"));
        assert_eq!(p.get(2), BStr::new("ST"));
    }

    #[test]
    fn resize_none_leaves_entry_unchanged() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDE"));
        bld.push(b("FGHIJ"));
        bld.push(b("KLMNO"));
        let mut p = bld.finish();
        // Only narrow odd entries; return None otherwise.
        p.resize(|i, e| {
            if i % 2 == 1 {
                Some((1, e.len() - 1))
            } else {
                None
            }
        });
        assert_eq!(p.get(0), BStr::new("ABCDE")); // untouched
        assert_eq!(p.get(1), BStr::new("GHIJ"));
        assert_eq!(p.get(2), BStr::new("KLMNO")); // untouched
    }

    #[test]
    fn resize_max_len_per_entry() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("ABCDEFGH"));
        bld.push(b("XY"));
        bld.push(b("PQRST"));
        let mut p = bld.finish();
        // max_len each entry to 3 bytes (a per-entry-aware clip).
        p.resize(|_, e| Some((0, e.len().min(3))));
        assert_eq!(p.get(0), BStr::new("ABC"));
        assert_eq!(p.get(1), BStr::new("XY")); // already shorter
        assert_eq!(p.get(2), BStr::new("PQR"));
    }

    #[test]
    fn resize_composes_with_prior_cuts() {
        // cut_start/cut_end leave head/tail overlays the resize must see baked.
        let mut bld = StringPodBuilder::with_capacity(7, 2);
        bld.push(b("0123456"));
        bld.push(b("ABCDEFG"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None); // visible: "12345", "BCDEF"
        p.resize(|_, e| {
            assert_eq!(e.len(), 5); // callback sees the post-cut visible region
            Some((1, 2))
        });
        assert_eq!(p.get(0), BStr::new("23"));
        assert_eq!(p.get(1), BStr::new("CD"));
    }

    #[test]
    fn resize_composes_after_pop_front() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("AAAA"));
        bld.push(b("BBBB"));
        bld.push(b("CCCC"));
        let mut p = bld.finish();
        p.pop_front(1); // view: "BBBB", "CCCC"
        let mut seen = Vec::new();
        p.resize(|i, e| {
            seen.push((i, e.to_vec()));
            Some((0, 2))
        });
        assert_eq!(seen, vec![(0, b("BBBB").to_vec()), (1, b("CCCC").to_vec())]);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new("BB"));
        assert_eq!(p.get(1), BStr::new("CC"));
    }

    #[test]
    fn resize_to_empty() {
        let mut bld = StringPodBuilder::with_capacity(4, 1);
        bld.push(b("ABCD"));
        let mut p = bld.finish();
        p.resize(|_, _| Some((2, 0)));
        assert_eq!(p.get(0), BStr::new(""));
        assert_eq!(p.entry_len(0), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds visible length")]
    fn resize_out_of_range_panics() {
        let mut bld = StringPodBuilder::with_capacity(3, 1);
        bld.push(b("ABC"));
        let mut p = bld.finish();
        p.resize(|_, _| Some((2, 5))); // 2+5 > 3
    }

    // ── iter_mut ──────────────────────────────────────────────────────────

    #[test]
    fn iter_mut_allows_in_place_mutation() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("ABC"));
        bld.push(b("XYZ"));
        let mut p = bld.finish();
        for entry in &mut p {
            entry.reverse();
        }
        assert_eq!(p.get(0), BStr::new("CBA"));
        assert_eq!(p.get(1), BStr::new("ZYX"));
    }

    #[test]
    fn iter_mut_returns_even_when_shared() {
        let mut bld = StringPodBuilder::with_capacity(3, 1);
        bld.push(b("ABC"));
        let mut p = bld.finish();
        let _q = p.clone();
        assert!(p.iter_mut().next().is_some());
    }

    #[test]
    fn iter_mut_double_ended() {
        let mut bld = StringPodBuilder::with_capacity(3, 3);
        bld.push(b("ABC"));
        bld.push(b("DEF"));
        bld.push(b("GHI"));
        let mut p = bld.finish();
        {
            let mut it = p.iter_mut();
            it.next().unwrap().reverse(); // ABC → CBA
            it.next_back().unwrap().reverse(); // GHI → IHG
            // DEF untouched
        }
        assert_eq!(p.get(0), BStr::new("CBA"));
        assert_eq!(p.get(1), BStr::new("DEF"));
        assert_eq!(p.get(2), BStr::new("IHG"));
    }

    #[test]
    fn iter_mut_variable_lengths() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("hello"));
        bld.push(b("hi"));
        bld.push(b("foobar"));
        let mut p = bld.finish();
        for entry in &mut p {
            entry.make_ascii_uppercase();
        }
        assert_eq!(p.get(0), BStr::new("HELLO"));
        assert_eq!(p.get(1), BStr::new("HI"));
        assert_eq!(p.get(2), BStr::new("FOOBAR"));
    }

    #[test]
    fn iter_mut_skips_cut_gaps() {
        // cut_start / cut_end leave head/tail gaps the peel must step over.
        let mut bld = StringPodBuilder::with_capacity(6, 3);
        bld.push(b("ABCDEF"));
        bld.push(b("UVWXYZ"));
        bld.push(b("012345"));
        let mut p = bld.finish();
        p.cut_start(1, None);
        p.cut_end(1, None); // visible: "BCDE", "VWXY", "1234"
        for entry in &mut p {
            entry.reverse();
        }
        assert_eq!(p.get(0), BStr::new("EDCB"));
        assert_eq!(p.get(1), BStr::new("YXWV"));
        assert_eq!(p.get(2), BStr::new("4321"));
    }

    #[test]
    fn iter_mut_alias_pod_noncontiguous() {
        // Alias entries are non-contiguous sub-ranges with gaps between them.
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("HELLO"));
        bld.push(b("WORLD"));
        bld.push(b("FOOBAR"));
        let source = bld.finish();
        let mut aliased = {
            let mut ab = source.alias_builder();
            ab.push_alias(1, 3); // "ELL"
            ab.push_alias(0, 5); // "WORLD"
            ab.push_alias(2, 3); // "OBA"
            ab.finish()
        };
        drop(source); // make the shared buffer uniquely owned
        for entry in &mut aliased {
            entry.make_ascii_lowercase();
        }
        assert_eq!(aliased.get(0), BStr::new("ell"));
        assert_eq!(aliased.get(1), BStr::new("world"));
        assert_eq!(aliased.get(2), BStr::new("oba"));
    }

    #[test]
    fn iter_mut_double_ended_variable() {
        let mut bld = StringPodBuilder::with_capacity(0, 3);
        bld.push(b("alpha"));
        bld.push(b("be"));
        bld.push(b("gamma"));
        let mut p = bld.finish();
        {
            let mut it = p.iter_mut();
            it.next().unwrap().make_ascii_uppercase(); // alpha → ALPHA
            it.next_back().unwrap().make_ascii_uppercase(); // gamma → GAMMA
        }
        assert_eq!(p.get(0), BStr::new("ALPHA"));
        assert_eq!(p.get(1), BStr::new("be")); // untouched
        assert_eq!(p.get(2), BStr::new("GAMMA"));
    }

    #[test]
    fn iter_mut_collect_all_then_mutate() {
        // The items outlive the iterator (no borrow of it), so they collect.
        let mut bld = StringPodBuilder::with_capacity(2, 3);
        bld.push(b("AA"));
        bld.push(b("BB"));
        bld.push(b("CC"));
        let mut p = bld.finish();
        {
            let all: Vec<&mut BStr> = p.iter_mut().collect();
            assert_eq!(all.len(), 3);
            for entry in all {
                entry.make_ascii_lowercase();
            }
        }
        assert_eq!(p.get(0), BStr::new("aa"));
        assert_eq!(p.get(2), BStr::new("cc"));
    }

    #[test]
    fn get_mut_exclusive_succeeds() {
        let mut bld = StringPodBuilder::with_capacity(3, 2);
        bld.push(b("AAA"));
        bld.push(b("BBB"));
        let mut p = bld.finish();
        p.get_mut(0).unwrap().copy_from_slice(b("ZZZ"));
        assert_eq!(p.get(0), BStr::new("ZZZ"));
        assert_eq!(p.get(1), BStr::new("BBB"));
    }

    #[test]
    fn get_mut_shared_returns_none() {
        let mut bld = StringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"));
        let mut p = bld.finish();
        let _q = p.clone();
        assert!(p.get_mut(0).is_none());
    }

    #[test]
    fn get_mut_succeeds_after_shared_ref_dropped() {
        let mut bld = StringPodBuilder::with_capacity(3, 1);
        bld.push(b("AAA"));
        let mut p = bld.finish();
        {
            let _q = p.clone(); // bumps refcount to 2
        } // _q drops here — refcount back to 1
        p.get_mut(0).unwrap().copy_from_slice(b("ZZZ"));
        assert_eq!(p.get(0), BStr::new("ZZZ"));
    }

    // compile-time check: pods are Send+Sync
    #[test]
    fn send_sync_check() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StringPod>();
        assert_send_sync::<super::StringPodBuilder>();
        assert_send_sync::<super::StringPodAliasBuilder<'static>>();
    }

    #[test]
    fn test_empty() {
        let mut bld = StringPodBuilder::with_capacity(0, 1);
        bld.push(b"");
        bld.push(b"");
        bld.push(b"");
        bld.push(b"");
        let p = bld.finish();
        for ii in 0..4 {
            assert_eq!(p.get(ii), b"");
        }
        let p2 = StringPod::new_all_empty(10);
        assert_eq!(p2.len(), 10);
        for astring in &p2 {
            assert!(astring.is_empty());
        }
    }

    // ── extend_from_pod (en-bloc range append) ──────────────────────────────

    fn pod(entries: &[&str]) -> StringPod {
        let mut bld = StringPodBuilder::with_capacity(0, entries.len());
        for e in entries {
            bld.push(b(e));
        }
        bld.finish()
    }

    #[test]
    fn extend_from_pod_full_range_variable() {
        let src = pod(&["hello", "hi", "foobar"]);
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 0..src.len());
        let p = bld.finish();
        let got: Vec<&BStr> = p.iter().collect();
        assert_eq!(
            got,
            vec![BStr::new("hello"), BStr::new("hi"), BStr::new("foobar")]
        );
    }

    #[test]
    fn extend_from_pod_partial_range() {
        let src = pod(&["AA", "BB", "CC", "DD", "EE"]);
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 1..4); // BB CC DD
        let p = bld.finish();
        assert_eq!(p.len(), 3);
        assert_eq!(p.get(0), BStr::new("BB"));
        assert_eq!(p.get(1), BStr::new("CC"));
        assert_eq!(p.get(2), BStr::new("DD"));
    }

    #[test]
    fn extend_from_pod_concatenates_multiple_sources() {
        let a = pod(&["a1", "a2"]);
        let b_ = pod(&["b1", "b2", "b3"]);
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&a, 0..2);
        bld.extend_from_pod(&b_, 1..3); // b2 b3
        let p = bld.finish();
        let got: Vec<String> = p.iter().map(ToString::to_string).collect();
        assert_eq!(got, vec!["a1", "a2", "b2", "b3"]);
    }

    #[test]
    fn extend_from_pod_fixed_length_source_stays_fixed() {
        // Fixed-length source: span is one contiguous strided memcpy and the
        // empty destination adopts the source's stride, staying FixedLength.
        let mut sb = StringPodBuilder::with_capacity(3, 3);
        sb.push(b("AAA"));
        sb.push(b("BBB"));
        sb.push(b("CCC"));
        let src = sb.finish();
        assert!(src.is_fixed_length());
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 0..3);
        let p = bld.finish();
        assert!(p.is_fixed_length(), "destination should stay fixed-length");
        assert_eq!(p.get(0), BStr::new("AAA"));
        assert_eq!(p.get(2), BStr::new("CCC"));
    }

    #[test]
    fn extend_from_pod_two_matching_fixed_sources_stay_fixed() {
        let mut a = StringPodBuilder::with_capacity(2, 2);
        a.push(b("AA"));
        a.push(b("BB"));
        let a = a.finish();
        let mut b2 = StringPodBuilder::with_capacity(2, 2);
        b2.push(b("CC"));
        b2.push(b("DD"));
        let b2 = b2.finish();

        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&a, 0..2);
        bld.extend_from_pod(&b2, 0..2);
        let p = bld.finish();
        assert!(p.is_fixed_length());
        let got: Vec<String> = p.iter().map(ToString::to_string).collect();
        assert_eq!(got, vec!["AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn extend_from_pod_mismatched_stride_promotes_to_variable() {
        let mut a = StringPodBuilder::with_capacity(2, 2);
        a.push(b("AA"));
        a.push(b("BB"));
        let a = a.finish();
        let mut b3 = StringPodBuilder::with_capacity(3, 1);
        b3.push(b("CCC"));
        let b3 = b3.finish();

        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&a, 0..2); // adopts stride 2, fixed
        bld.extend_from_pod(&b3, 0..1); // stride mismatch → promote
        let p = bld.finish();
        assert!(!p.is_fixed_length());
        let got: Vec<String> = p.iter().map(ToString::to_string).collect();
        assert_eq!(got, vec!["AA", "BB", "CCC"]);
    }

    #[test]
    fn extend_from_pod_fixed_source_partial_range_with_cut() {
        // Fixed source carrying a cut overlay (like a stripped '@'): the fast
        // path must preserve the overlay through the strided copy.
        let mut sb = StringPodBuilder::with_capacity(4, 3);
        sb.push(b("@abc"));
        sb.push(b("@def"));
        sb.push(b("@ghi"));
        let mut src = sb.finish();
        src.cut_start(1, None); // visible: "abc", "def", "ghi"
        assert!(src.is_fixed_length());
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 1..3); // def ghi
        let p = bld.finish();
        assert!(p.is_fixed_length());
        assert_eq!(p.get(0), BStr::new("def"));
        assert_eq!(p.get(1), BStr::new("ghi"));
    }

    #[test]
    fn extend_from_pod_respects_cut_overlay() {
        // A leading-byte cut (like the '@' strip on FASTQ names): the hidden
        // byte rides along in the copied span but must stay hidden.
        let mut sb = StringPodBuilder::with_capacity(4, 2);
        sb.push(b("@abc"));
        sb.push(b("@xyz"));
        let mut src = sb.finish();
        src.cut_start(1, None); // visible: "abc", "xyz"
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 0..2);
        let p = bld.finish();
        assert_eq!(p.get(0), BStr::new("abc"));
        assert_eq!(p.get(1), BStr::new("xyz"));
    }

    #[test]
    fn extend_from_pod_after_pop_front() {
        let src = pod(&["AAAAA", "BBB", "CCCCC"]);
        let mut src = src;
        src.pop_front(1); // view: "BBB", "CCCCC"
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 0..src.len());
        let p = bld.finish();
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), BStr::new("BBB"));
        assert_eq!(p.get(1), BStr::new("CCCCC"));
    }

    #[test]
    fn extend_from_pod_empty_range_is_noop() {
        let src = pod(&["AA", "BB"]);
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 1..1);
        assert_eq!(bld.finish().len(), 0);
    }

    #[test]
    #[should_panic(expected = "range past end of source pod")]
    fn extend_from_pod_out_of_range_panics() {
        let src = pod(&["AA", "BB"]);
        let mut bld = StringPodBuilder::new();
        bld.extend_from_pod(&src, 0..3);
    }
}
