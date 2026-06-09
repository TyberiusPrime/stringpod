//! A dual byte-string pod whose every row holds a *list* of byte ranges.
//!
//! [`DualStringPod`](crate::DualStringPod) stores one contiguous range per row;
//! [`DualStringPodMultiLocation`] generalises that to a *list* of ranges per
//! row — the natural shape for a tag column where a single read can carry
//! several hits (and a read with none contributes an empty row). It is built by
//! aliasing the read pod in order (one row per source entry), so the frozen
//! seq + qual bytes of every hit are shared zero-copy with the read buffers.
//!
//! The snapshot is a frozen **Value**: its bytes never change. Copy-on-write
//! keeps it valid no matter what later happens to the read pod — overlay cuts
//! only touch the read's metadata, rebuilds (`prefix`/`reverse`) give the read a
//! fresh `Arc` and leave the snapshot on the old one, and in-place reversals go
//! through the read's own COW clone first. Coordinate liftover therefore lives
//! entirely on the *read* pod (see [`Lifted`](crate::Lifted)); this type does
//! not track edits.
//!
//! What it *does* keep is each location's **read-relative** `(start, len)` as
//! captured (see [`loc_region`](DualStringPodMultiLocation::loc_region)) — the
//! exact coordinate the read pod's log lifts — alongside the row's `base` offset
//! into the buffer. The single offset set therefore serves both jobs: `base +
//! start + first_byte` slices the frozen bytes, while `start` on its own is the
//! liftable read coordinate.

use bstr::BStr;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use crate::dual::DualStringPod;

/// One row's locations, stored **read-relative** as `(start, len)` pairs — i.e.
/// offsets into the source read's visible bytes at capture time, *not* absolute
/// buffer positions. The owning [`Row`]'s `base` plus the per-buffer first-byte
/// offset are added at access time to slice the frozen bytes; the read-relative
/// `(start, len)` is itself what the source read pod's edit log lifts. Single-
/// location rows — the common case — stay on the stack.
type Locs = SmallVec<[(u32, u32); 1]>;

/// One row of the snapshot: the source read entry's `base` offset in shared
/// metadata space at capture, plus its read-relative locations.
#[derive(Clone)]
struct Row {
    base: u32,
    locs: Locs,
}

/// A dual (seq + qual) pod whose every row is a list of byte ranges aliased
/// from a source [`DualStringPod`]. See the [module docs](crate::multiloc).
#[derive(Clone)]
pub struct DualStringPodMultiLocation {
    seq: Arc<Vec<u8>>,
    qual: Arc<Vec<u8>>,
    seq_first_byte: usize,
    qual_first_byte: usize,
    rows: Vec<Row>,
}

impl DualStringPodMultiLocation {
    /// Number of rows (one per source read).
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// `true` if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of locations in `row`.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn locs_in(&self, row: usize) -> usize {
        self.rows[row].locs.len()
    }

    /// `true` if `row` carries no locations (a read with no hit).
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn row_is_empty(&self, row: usize) -> bool {
        self.rows[row].locs.is_empty()
    }

    /// Sequence bytes of location `loc` in `row`.
    ///
    /// # Panics
    /// If `row` or `loc` is out of range.
    #[must_use]
    pub fn seq(&self, row: usize, loc: usize) -> &BStr {
        let r = &self.rows[row];
        let (rel, len) = r.locs[loc];
        let start = r.base as usize + rel as usize + self.seq_first_byte;
        BStr::new(&self.seq[start..start + len as usize])
    }

    /// Quality bytes of location `loc` in `row` (same range as [`seq`](Self::seq)).
    ///
    /// # Panics
    /// If `row` or `loc` is out of range.
    #[must_use]
    pub fn qual(&self, row: usize, loc: usize) -> &BStr {
        let r = &self.rows[row];
        let (rel, len) = r.locs[loc];
        let start = r.base as usize + rel as usize + self.qual_first_byte;
        BStr::new(&self.qual[start..start + len as usize])
    }

    /// Both bytes of location `loc` in `row` at once.
    ///
    /// # Panics
    /// If `row` or `loc` is out of range.
    #[must_use]
    pub fn pair(&self, row: usize, loc: usize) -> (&BStr, &BStr) {
        (self.seq(row, loc), self.qual(row, loc))
    }

    /// The read-relative `(start, len)` of location `loc` in `row`, as captured
    /// from the source read at build time. This is precisely the coordinate to
    /// feed the source read pod's edit-log liftover (`map_region`) to recover
    /// the location's *current* position after the read has been edited — the
    /// snapshot bytes themselves never move.
    ///
    /// # Panics
    /// If `row` or `loc` is out of range.
    #[must_use]
    pub fn loc_region(&self, row: usize, loc: usize) -> (usize, usize) {
        let (rel, len) = self.rows[row].locs[loc];
        (rel as usize, len as usize)
    }

    /// Iterate the `(seq, qual)` pairs of every location in `row`.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (&BStr, &BStr)> {
        (0..self.rows[row].locs.len()).map(move |loc| self.pair(row, loc))
    }

    /// Iterate the read-relative `(start, len)` of every location in `row`.
    /// See [`loc_region`](Self::loc_region).
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    pub fn row_regions(&self, row: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.rows[row]
            .locs
            .iter()
            .map(|&(rel, len)| (rel as usize, len as usize))
    }

    /// The `row`'s sequence locations joined (optionally with `sep` between
    /// them). Borrows for a single-location row; allocates otherwise.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn joined_seq(&self, row: usize, sep: Option<&[u8]>) -> Cow<'_, [u8]> {
        join(&self.seq, self.seq_first_byte, &self.rows[row], sep)
    }

    /// The `row`'s quality locations joined (optionally with `sep` between
    /// them). Borrows for a single-location row; allocates otherwise.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn joined_qual(&self, row: usize, sep: Option<&[u8]>) -> Cow<'_, [u8]> {
        join(&self.qual, self.qual_first_byte, &self.rows[row], sep)
    }

    // ── row-axis: keep the snapshot aligned with the live reads ──────────────

    /// Drop a contiguous range of rows (mirrors the read pod's `drain`).
    ///
    /// # Panics
    /// If the range is out of bounds.
    pub fn drain(&mut self, range: Range<usize>) {
        self.rows.drain(range);
    }

    /// Drop the first `n` rows (mirrors `pop_front`).
    pub fn pop_front(&mut self, n: usize) {
        self.rows.drain(0..n.min(self.rows.len()));
    }

    /// Keep only the first `len` rows.
    pub fn truncate(&mut self, len: usize) {
        self.rows.truncate(len);
    }

    /// Keep only rows whose bool is `true` (parallel to the rows).
    ///
    /// # Panics
    /// If `keep.len() != self.row_count()`.
    pub fn retain_by_bools(&mut self, keep: &[bool]) {
        assert_eq!(
            keep.len(),
            self.rows.len(),
            "retain_by_bools mask length must match row count"
        );
        let mut it = keep.iter();
        self.rows
            .retain(|_| *it.next().expect("mask length checked"));
    }

    /// Ensure this pod owns both byte buffers outright, cloning each (COW) only
    /// if it is currently shared (e.g. with the read pod it was aliased from).
    pub fn make_exclusive(&mut self) {
        let _ = Arc::make_mut(&mut self.seq);
        let _ = Arc::make_mut(&mut self.qual);
    }
}

impl std::fmt::Debug for DualStringPodMultiLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locs: usize = self.rows.iter().map(|r| r.locs.len()).sum();
        f.debug_struct("DualStringPodMultiLocation")
            .field("rows", &self.rows.len())
            .field("locations", &locs)
            .finish_non_exhaustive()
    }
}

/// Join a row's read-relative locations out of `buf` (offset by the row's `base`
/// plus the buffer's `first` byte), borrowing the single-location case.
fn join<'a>(buf: &'a [u8], first: usize, row: &Row, sep: Option<&[u8]>) -> Cow<'a, [u8]> {
    let off = row.base as usize + first;
    match row.locs.as_slice() {
        [] => Cow::Borrowed(&[]),
        [(rel, len)] => {
            let s = off + *rel as usize;
            Cow::Borrowed(&buf[s..s + *len as usize])
        }
        many => {
            let mut out = Vec::new();
            for (k, &(rel, len)) in many.iter().enumerate() {
                if k > 0 {
                    if let Some(sep) = sep {
                        out.extend_from_slice(sep);
                    }
                }
                let s = off + rel as usize;
                out.extend_from_slice(&buf[s..s + len as usize]);
            }
            Cow::Owned(out)
        }
    }
}

impl DualStringPod {
    /// Start a builder that aliases this pod's seq + qual buffers into a
    /// [`DualStringPodMultiLocation`] — one row per source entry, consumed in
    /// order, each row recording a list of sub-ranges (its hits).
    #[must_use]
    pub fn multi_location_alias_builder(&self) -> DualStringPodMultiLocationAliasBuilder<'_> {
        DualStringPodMultiLocationAliasBuilder {
            source: self,
            next: 0,
            rows: Vec::new(),
        }
    }
}

/// Builds a [`DualStringPodMultiLocation`] by aliasing the source pod's entries
/// in order — one [`push_row`](Self::push_row) per source entry.
pub struct DualStringPodMultiLocationAliasBuilder<'a> {
    source: &'a DualStringPod,
    next: usize,
    rows: Vec<Row>,
}

impl DualStringPodMultiLocationAliasBuilder<'_> {
    /// Alias the next source entry as one row, recording each `(offset, len)` in
    /// `locations` as a read-relative sub-range of that entry's visible bytes.
    /// Pass an empty slice for a read with no hit.
    ///
    /// # Panics
    /// - If all source entries have already been consumed.
    /// - If any `offset + len` exceeds the source entry's length.
    /// - If the entry's base offset or any `offset`/`len` would exceed `u32::MAX`.
    pub fn push_row(&mut self, locations: &[(usize, usize)]) {
        assert!(
            self.next < self.source.len(),
            "multi-location alias builder: all {} source entries already consumed",
            self.source.len(),
        );
        let r = self.source.storage.entry_range(self.next);
        let entry_len = r.end - r.start;
        let base = u32::try_from(r.start).expect("alias base exceeds u32");
        let mut locs: Locs = SmallVec::with_capacity(locations.len());
        for &(offset, len) in locations {
            let end = offset
                .checked_add(len)
                .expect("alias offset + len overflows usize");
            assert!(
                end <= entry_len,
                "alias offset {offset}+len {len}={end} exceeds entry length {entry_len}",
            );
            let rel_start = u32::try_from(offset).expect("alias offset exceeds u32");
            let len = u32::try_from(len).expect("alias len exceeds u32");
            locs.push((rel_start, len));
        }
        self.rows.push(Row { base, locs });
        self.next += 1;
    }

    /// Number of rows pushed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// `true` if no rows have been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Finalise the builder, releasing the borrow of the source pod. Both
    /// buffers are co-owned by the snapshot (COW snapshot semantics).
    #[must_use]
    pub fn finish(self) -> DualStringPodMultiLocation {
        DualStringPodMultiLocation {
            seq: Arc::clone(&self.source.seq),
            qual: Arc::clone(&self.source.qual),
            seq_first_byte: self.source.seq_first_byte,
            qual_first_byte: self.source.qual_first_byte,
            rows: self.rows,
        }
    }
}
