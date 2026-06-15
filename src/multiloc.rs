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

use bstr::{BStr, BString};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use crate::Lifted;
use crate::dual::DualStringPod;

/// One row's locations, stored **read-relative** as `(start, len)` pairs — i.e.
/// offsets into the source read's visible bytes at capture time, *not* absolute
/// buffer positions. The owning [`Row`]'s `base` plus the per-buffer first-byte
/// offset are added at access time to slice the frozen bytes; the read-relative
/// `(start, len)` is itself what the source read pod's edit log lifts. Single-
/// location rows — the common case — stay on the stack.
type Locs = SmallVec<[(u32, u32); 1]>;

#[derive(Clone, Debug)]
struct LocsAndBase {
    base: u32,
    locs: Locs,
}

/// One row of the snapshot. Either an **Alias** — a zero-copy view of the source
/// read, holding the read-relative `(start, len)` slices captured at build time —
/// or an **Owned** row whose content diverged from any single read slice (a regex
/// replacement that conjures or reorders bytes, or an in-place content edit such
/// as reverse-complement) and now lives in the pod's `owned` arena.
#[derive(Clone, Debug)]
enum Row {
    /// Zero-copy view of the source read: `base` offset in shared metadata space
    /// at capture, plus its read-relative locations.
    Alias(LocsAndBase),
    /// Divergent content held in the pod's [`owned`](DualStringPodMultiLocation::owned)
    /// arena: the sequence is `owned[owned_offset..owned_offset + len]` and the quality is the
    /// equal-length run right after it (`owned[off + len..off + 2 * len]`) — so a
    /// single `(off, len)` locates both. `anchor` is just the aliased positions
    /// that the content replaces.
    /// (precisely what liftover lifts and
    /// what write-back overwrites in the live read (the content length may differ).
    Owned {
        anchor: LocsAndBase,
        off: u32,
        len: u32,
    },
}

/// What a row's coordinates were captured against: the source entry's edit-log
/// generation at build time (`generation`) and its byte length in that frame
/// (`len`). Together they are everything [`EditLogView::map_region`](crate::EditLogView::map_region)
/// needs to lift a stored read-relative `(start, len)` forward into the source
/// pod's *current* frame: `pod.ops_since(generation, row).map_region(start, len, len_at_birth)`.
/// Stored per row because a column may mix tags born at different generations
/// (e.g. a later merge) and because a row's birth length differs from its
/// current length once the source read is edited.
#[derive(Clone, Copy, Debug)]
struct RowBorn {
    /// Source entry's edit-log generation at capture (`0` if no edit preceded it).
    generation: u32,
    /// Source entry's visible byte length in the birth frame — the `orig_len`
    /// the read pod's [`EditLogView`](crate::EditLogView) lifts from.
    len: u32,
}

/// A dual (seq + qual) pod whose every row is either a list of byte ranges aliased
/// from a source [`DualStringPod`] or divergent content owned in `owned`. See the
/// [module docs](crate::multiloc).
#[derive(Clone)]
pub struct DualStringPodMultiLocation {
    seq: Arc<Vec<u8>>,
    qual: Arc<Vec<u8>>,
    seq_first_byte: usize,
    qual_first_byte: usize,
    /// Arena backing every [`Row::Owned`]: each owns one `seq` run immediately
    /// followed by its equal-length `qual` run. One shared `Vec` (not a `BString`
    /// per row) so a 50k-row column is one allocation, not 50k.
    owned: Vec<u8>,
    rows: Vec<Row>,
    /// Per-row birth frame (parallel to `rows`), so forward liftover knows which
    /// generation and length to lift each row's coordinates from. Kept aligned
    /// with `rows` across every row-axis op; untouched by content edits
    /// ([`set_row_content`](Self::set_row_content)) since a tag's birth doesn't
    /// move when its bytes are rewritten.
    born: Vec<RowBorn>,
    /// Opaque, caller-defined identifier for the source this snapshot was
    /// aliased from. `stringpod` never interprets it — it is stamped by the
    /// builder (see
    /// [`with_source_id`](DualStringPodMultiLocationAliasBuilder::with_source_id),
    /// default `0`) and read back via [`source_id`](Self::source_id). Callers
    /// that build one snapshot per logical source (e.g. one per read segment)
    /// use it to recover which source a column came from when they later need to
    /// rebuild it against that same source.
    source_id: u32,
}

impl DualStringPodMultiLocation {
    /// Number of rows (one per source read).
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The opaque, caller-defined source identifier stamped on this snapshot at
    /// build time (default `0`). See the [`source_id`](Self::source_id) field
    /// docs and [`with_source_id`](DualStringPodMultiLocationAliasBuilder::with_source_id).
    #[must_use]
    pub fn source_id(&self) -> u32 {
        self.source_id
    }

    /// Re-stamp the opaque [`source_id`](Self::source_id). Used when the source
    /// this snapshot aliases is relabelled wholesale (e.g. two read segments are
    /// swapped as units, so a column captured against segment `a` now lifts and
    /// writes back against segment `b`). The rows, byte aliases, born frames and
    /// coordinates are unchanged — only the identifier of the segment they track.
    pub fn set_source_id(&mut self, source_id: u32) {
        self.source_id = source_id;
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
    pub fn loc_count_in(&self, row: usize) -> usize {
        match &self.rows[row] {
            Row::Alias(anchor) | Row::Owned { anchor, .. } => anchor.locs.len(),
        }
    }

    /// `true` if `row` carries no content — a read with no hit. An [`Owned`](Row::Owned)
    /// row is always present (even if its content is the empty string).
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn row_is_empty(&self, row: usize) -> bool {
        match &self.rows[row] {
            Row::Alias(anchor) | Row::Owned { anchor, .. } => anchor.locs.is_empty(),
        }
    }

    /// The length spanned by the regions.
    /// Assumes disjoint regions!
    pub fn row_length<'a, 'b>(&'a self, row: usize, sep: Option<&[u8]>) -> usize {
        match &self.rows[row] {
            Row::Alias(anchor) | Row::Owned { anchor, .. } => {
                let mut total = 0;
                for loc in &anchor.locs {
                    total += loc.1 as usize;
                }
                if let Some(sep) = sep {
                    total += anchor.locs.len().saturating_sub(sep.len());
                }
                total
            } // One contiguous owned run: separators never apply.
        }
    }

    // /// Sequence bytes of location `loc` in `row`.
    // ///
    // /// # Panics
    // /// If `row` or `loc` is out of range.
    // #[must_use]
    // pub fn seq(&self, row: usize, loc: usize) -> &BStr {
    //     match &self.rows[row] {
    //         Row::Alias ( anchor) => {
    //             let (rel, len) = anchor.locs[loc];
    //             let start = anchor.base as usize + rel as usize + self.seq_first_byte;
    //             BStr::new(&self.seq[start..start + len as usize])
    //         }
    //         Row::Owned { off, len, .. } => {
    //             assert_eq!(loc, 0, "owned row has a single location");
    //             let s = *off as usize;
    //             BStr::new(&self.owned[s..s + *len as usize])
    //         }
    //     }
    // }
    //
    // /// Quality bytes of location `loc` in `row` (same range as [`seq`](Self::seq)).
    // ///
    // /// # Panics
    // /// If `row` or `loc` is out of range.
    // #[must_use]
    // pub fn qual(&self, row: usize, loc: usize) -> &BStr {
    //     match &self.rows[row] {
    //         Row::Alias { base, locs } => {
    //             let (rel, len) = locs[loc];
    //             let start = *base as usize + rel as usize + self.qual_first_byte;
    //             BStr::new(&self.qual[start..start + len as usize])
    //         }
    //         Row::Owned { off, len, .. } => {
    //             assert_eq!(loc, 0, "owned row has a single location");
    //             let s = *off as usize + *len as usize;
    //             BStr::new(&self.owned[s..s + *len as usize])
    //         }
    //     }
    // }
    //
    // /// Both bytes of location `loc` in `row` at once.
    // ///
    // /// # Panics
    // /// If `row` or `loc` is out of range.
    // #[must_use]
    // pub fn pair(&self, row: usize, loc: usize) -> (&BStr, &BStr) {
    //     (self.seq(row, loc), self.qual(row, loc))
    // }

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
        match &self.rows[row] {
            Row::Alias(anchor) => {
                let (rel, len) = anchor.locs[loc];
                (rel as usize, len as usize)
            }
            Row::Owned { anchor, .. } => {
                let (rel, len) = anchor.locs[loc];
                (rel as usize, len as usize)
            }
        }
    }

    // /// Iterate the `(seq, qual)` pairs of every location in `row`.
    // ///
    // /// # Panics
    // /// If `row >= self.row_count()`.
    // pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (&BStr, &BStr)> {
    //     (0..self.loc_count_in(row)).map(move |loc| self.pair(row, loc))
    // }

    /// The `row`'s **birth frame**: the source entry's edit-log `generation` and
    /// byte `len` captured when this column was built. Feed these to the source
    /// pod's forward liftover to map a stored read-relative `(start, len)` (from
    /// [`row_regions`](Self::row_regions)) into the source's current frame:
    /// `pod.ops_since(generation, row).map_region(start, len, born_len)`.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn row_born(&self, row: usize) -> (usize, usize) {
        let b = self.born[row];
        (b.generation as usize, b.len as usize)
    }

    /// Iterate the read-relative `(start, len)` of every location in `row`.
    /// See [`loc_region`](Self::loc_region).
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    pub fn row_regions(&self, row: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.loc_count_in(row)).map(move |loc| {
            let (start, len) = self.loc_region(row, loc);
            (start as u32, len as u32)
        })
    }

    /// Iterate every row's read-relative locations — one `SmallVec<[(u32,u32);1]>`
    /// per row (empty for a no-hit row). For an [`Owned`](Row::Owned) row this is
    /// the single anchor span. See [`row_regions`](Self::row_regions). The vecs are
    /// freshly built (an owned row has no stored `Locs` to borrow).
    pub fn iter_row_regions(&self) -> impl Iterator<Item = SmallVec<[(u32, u32); 1]>> + '_ {
        (0..self.rows.len()).map(move |row| self.row_regions(row).collect())
    }

    /// Iterate the read-relative byte positions covered by `row`'s locations, in
    /// ascending order — the coordinate counterpart of [`joined_seq`](Self::joined_seq).
    /// Where that yields the covered *bytes*, this yields the covered *positions*
    /// in the source read's own coordinate space (the same `start`
    /// [`loc_region`](Self::loc_region) exposes and the read pod's
    /// [`map_position`](crate::EditLogView::map_position) lifts), so a caller can
    /// index straight back into the live read to modify exactly the covered bases.
    ///
    /// Locations are walked in stored order; like [`row_length`](Self::row_length)
    /// this assumes the row's regions are disjoint, so positions come out sorted
    /// and without repeats. A no-hit row yields nothing.
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    pub fn covered_positions(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.loc_count_in(row)).flat_map(move |loc| {
            let (start, len) = self.loc_region(row, loc);
            start..(start + len)
        })
    }

    /// The `row`'s sequence locations joined (optionally with `sep` between
    /// them). Borrows for a single-location row; allocates otherwise.
    /// For owned rows, returns the owned data instead
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn joined_seq(&self, row: usize, sep: Option<&[u8]>) -> Cow<'_, BStr> {
        match &self.rows[row] {
            Row::Alias(anchor) => join(
                &self.seq,
                self.seq_first_byte,
                anchor.base,
                &anchor.locs,
                sep,
            ),
            Row::Owned { off, len, .. } => {
                let s = *off as usize;
                Cow::Borrowed(BStr::new(&self.owned[s..s + *len as usize]))
            }
        }
    }

    /// The `row`'s quality locations joined (optionally with `sep` between
    /// them). Borrows for a single-location row; allocates otherwise.
    /// For owned rows, return the owned data instead
    ///
    /// # Panics
    /// If `row >= self.row_count()`.
    #[must_use]
    pub fn joined_qual(&self, row: usize, sep: Option<&[u8]>) -> Cow<'_, BStr> {
        match &self.rows[row] {
            Row::Alias(anchor) => join(
                &self.qual,
                self.qual_first_byte,
                anchor.base,
                &anchor.locs,
                sep,
            ),
            Row::Owned { off, len, .. } => {
                let s = *off as usize + *len as usize; //seq comes first, then qual
                Cow::Borrowed(BStr::new(&self.owned[s..s + *len as usize]))
            }
        }
    }

    #[must_use]
    pub fn joined_pair(&self, row: usize, sep: Option<&[u8]>) -> (Cow<'_, BStr>, Cow<'_, BStr>) {
        (self.joined_seq(row, sep), self.joined_qual(row, sep))
    }

    /// Iterate the length of every row (sum of location lengths, plus
    /// `sep.is_some() as usize * (locs - 1)` separators). Mirrors the
    /// per-row computation of [`row_length`](Self::row_length).
    pub fn iter_row_lengths<'a, 'b>(
        &'a self,
        sep: Option<&'a [u8]>,
    ) -> impl Iterator<Item = usize> + 'a {
        (0..self.rows.len()).map(move |row| self.row_length(row, sep))
    }

    /// Iterate every row's captured sequence — one item per row, the row's
    /// locations concatenated; a no-hit row (no stored locations) yields `None`.
    /// Borrows for single-location rows, allocates only for multi-location ones;
    /// use [`joined_seq`](Self::joined_seq) if you need a separator.
    pub fn iter_seq(&self) -> impl Iterator<Item = Cow<'_, BStr>> {
        (0..self.rows.len()).map(|row| self.joined_seq(row, None))
    }

    /// Iterate every row's captured sequence with `region_separator` placed
    /// between a row's locations — the separated counterpart of
    /// [`iter_seq`](Self::iter_seq). Single-location rows still borrow; only
    /// multi-location rows allocate (to splice in the separator). A no-hit row
    /// yields an empty `Cow`.
    pub fn iter_seq_joined<'a>(
        &'a self,
        region_separator: Option<&'a [u8]>,
    ) -> impl Iterator<Item = Cow<'a, BStr>> + 'a {
        (0..self.rows.len()).map(move |row| self.joined_seq(row, region_separator))
    }

    /// Iterate every row's captured quality — the quality counterpart of
    /// [`iter_seq`](Self::iter_seq); a no-hit row yields `None`.
    pub fn iter_qual(&self) -> impl Iterator<Item = Cow<'_, BStr>> {
        (0..self.rows.len()).map(|row| self.joined_qual(row, None))
    }
    ///
    /// Iterate every row's captured qualities with `region_separator` placed
    /// between a row's locations — the separated counterpart of
    /// [`iter_qual`](Self::iter_qual). Single-location rows still borrow; only
    /// multi-location rows allocate (to splice in the separator). A no-hit row
    /// yields an empty `Cow`.
    pub fn iter_qual_joined<'a>(
        &'a self,
        region_separator: Option<&'a [u8]>,
    ) -> impl Iterator<Item = Cow<'a, BStr>> + 'a {
        (0..self.rows.len()).map(move |row| self.joined_qual(row, region_separator))
    }

    /// Iterate every row's captured `(seq, qual)` pair — [`iter_seq`](Self::iter_seq)
    /// and [`iter_qual`](Self::iter_qual) walked in lockstep, one item per row
    /// (`None` for a no-hit row). Equivalent to `(&pod).into_iter()`.
    #[must_use]
    pub fn iter(&self) -> PairIter<'_> {
        PairIter { pod: self, row: 0 }
    }

    // ── row-axis: keep the snapshot aligned with the live reads ──────────────

    /// Drop a contiguous range of rows (mirrors the read pod's `drain`).
    ///
    /// # Panics
    /// If the range is out of bounds.
    pub fn drain(&mut self, range: Range<usize>) {
        self.born.drain(range.clone());
        self.rows.drain(range);
    }

    /// Poor man's retain, where the F doesn't get to look
    /// at the actual contents.
    /// Mostly use ful for filtering on a bool iter of the same length
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut() -> bool,
    {
        // Materialise the keep-mask once so `rows` and the parallel `born` are
        // filtered identically (the closure must only be called per row once).
        let keep: Vec<bool> = (0..self.rows.len()).map(|_| f()).collect();
        let mut it = keep.iter();
        self.rows
            .retain(|_| *it.next().expect("mask length matches rows"));
        let mut it = keep.iter();
        self.born
            .retain(|_| *it.next().expect("mask length matches born"));
    }

    /// Drop the first `n` rows (mirrors `pop_front`).
    pub fn pop_front(&mut self, n: usize) {
        let n = n.min(self.rows.len());
        self.born.drain(0..n);
        self.rows.drain(0..n);
    }

    /// Keep only the first `len` rows.
    pub fn truncate(&mut self, len: usize) {
        self.rows.truncate(len);
        self.born.truncate(len);
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
        let mut it = keep.iter();
        self.born
            .retain(|_| *it.next().expect("mask length checked"));
    }

    /// Ensure this pod owns both byte buffers outright, cloning each (COW) only
    /// if it is currently shared (e.g. with the read pod it was aliased from).
    pub fn make_exclusive(&mut self) {
        let _ = Arc::make_mut(&mut self.seq);
        let _ = Arc::make_mut(&mut self.qual);
    }

    /// The read-relative span a `row` occupies / stands in for — the natural
    /// write-back anchor.
    ///
    /// (`min(start) .. max(start+len)`); for a no-hit row, `(0, 0)`.
    #[must_use]
    pub fn row_span(&self, row: usize) -> (usize, usize) {
        match &self.rows[row] {
            Row::Owned { anchor, .. } | Row::Alias(anchor) => {
                if anchor.locs.is_empty() {
                    return (0, 0);
                }
                let start = anchor
                    .locs
                    .iter()
                    .map(|&(s, _)| s)
                    .min()
                    .expect("non-empty");
                let end = anchor
                    .locs
                    .iter()
                    .map(|&(s, l)| s + l)
                    .max()
                    .expect("non-empty");
                (start as usize, (end - start) as usize)
            }
        }
    }

    // /// If `row` holds owned (divergent) content, the read-relative `(start, len)`
    // /// span a write-back should overwrite in the live read. `None` for alias rows
    // /// (their content already *is* the read's own bytes — nothing to write back)
    // /// and for no-hit rows. The bytes to write are [`joined_seq`](Self::joined_seq)
    // /// / [`joined_qual`](Self::joined_qual); the content length may differ from the
    // /// span's, so the read grows or shrinks.
    // #[must_use]
    // pub fn owned_writeback_span(&self, row: usize) -> Option<(usize, usize)> {
    //     match &self.rows[row] {
    //         Row::Owned { anchor, .. } => Some((anchor.0 as usize, anchor.1 as usize)),
    //         Row::Alias { .. } => None,
    //     }
    // }

    /// Replace `row`'s content with owned `seq` + `qual` bytes, COW-detaching it
    /// from the source read (the read is *not* touched). Used by content edits on
    /// a tag — reverse-complement, case change — which change the tag's own bytes;
    /// the read only changes via an explicit write-back. `anchor` is the
    /// read-relative span this content stands in for (typically the row's current
    /// [`row_span`](Self::row_span)). The owned bytes go in the shared arena
    /// (`seq` then equal-length `qual`).
    ///
    /// # Panics
    /// - If `row >= self.row_count()`.
    /// - If `seq.len() != qual.len()`.
    /// - If the arena offset or content length would exceed `u32::MAX`.
    pub fn set_row_content(&mut self, row: usize, seq: &[u8], qual: &[u8]) {
        assert!(row < self.rows.len(), "row {row} out of range");
        assert_eq!(
            seq.len(),
            qual.len(),
            "owned row seq ({}) / qual ({}) length mismatch",
            seq.len(),
            qual.len(),
        );
        let anchor = match &self.rows[row] {
            Row::Alias(anchor) | Row::Owned { anchor, .. } => anchor,
        };
        let off = u32::try_from(self.owned.len()).expect("owned arena offset exceeds u32");
        let len = u32::try_from(seq.len()).expect("owned content len exceeds u32");
        self.owned.extend_from_slice(seq);
        self.owned.extend_from_slice(qual);
        self.rows[row] = Row::Owned {
            anchor: anchor.clone(),
            off,
            len,
        };
    }
}

impl std::fmt::Debug for DualStringPodMultiLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locs: usize = (0..self.rows.len()).map(|row| self.loc_count_in(row)).sum();
        f.debug_struct("DualStringPodMultiLocation")
            .field("rows", &self.rows.len())
            .field("locations", &locs)
            .finish_non_exhaustive()
    }
}

/// Iterator over a [`DualStringPodMultiLocation`]'s captured `(seq, qual)` pairs,
/// one item per row ("" for a no-hit row). Created by
/// [`DualStringPodMultiLocation::iter`] or by iterating `&pod`. Borrows single-
/// location rows, allocates only for multi-location ones.
pub struct PairIter<'a> {
    pod: &'a DualStringPodMultiLocation,
    row: usize,
}

impl<'a> Iterator for PairIter<'a> {
    type Item = (Cow<'a, BStr>, Cow<'a, BStr>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.pod.rows.len() {
            return None;
        }
        let row = self.row;
        self.row += 1;
        let pair = (
            self.pod.joined_seq(row, None),
            self.pod.joined_qual(row, None),
        );
        Some(pair)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.pod.rows.len() - self.row;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PairIter<'_> {}

impl<'a> IntoIterator for &'a DualStringPodMultiLocation {
    type Item = (Cow<'a, BStr>, Cow<'a, BStr>);
    type IntoIter = PairIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PairIter { pod: self, row: 0 }
    }
}

/// Join an alias row's read-relative `locs` out of `buf` (offset by the row's
/// `base` plus the buffer's `first` byte), borrowing the single-location case.
fn join<'a>(
    buf: &'a [u8],
    first: usize,
    base: u32,
    locs: &Locs,
    sep: Option<&[u8]>,
) -> Cow<'a, BStr> {
    let off = base as usize + first;
    match locs.as_slice() {
        [] => Cow::Borrowed(BStr::new(b"")),
        [(rel, len)] => {
            let s = off + *rel as usize;
            Cow::Borrowed(BStr::new(&buf[s..s + *len as usize]))
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
            Cow::Owned(BString::from(out))
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
            owned: Vec::new(),
            rows: Vec::new(),
            born: Vec::new(),
            source_id: 0,
        }
    }
}

/// Builds a [`DualStringPodMultiLocation`] by aliasing the source pod's entries
/// in order — one [`push_row`](Self::push_row) per source entry.
pub struct DualStringPodMultiLocationAliasBuilder<'a> {
    source: &'a DualStringPod,
    next: usize,
    owned: Vec<u8>,
    rows: Vec<Row>,
    born: Vec<RowBorn>,
    source_id: u32,
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
    pub fn push_row(&mut self, locations: &[(u32, u32)]) {
        assert!(
            self.next < self.source.len(),
            "multi-location alias builder: all {} source entries already consumed",
            self.source.len(),
        );
        let r = self.source.storage.entry_range(self.next);
        let entry_len: u32 = (r.end - r.start)
            .try_into()
            .expect("region longer than u32");
        let base = u32::try_from(r.start).expect("alias base exceeds u32");
        let mut locs: Locs = SmallVec::with_capacity(locations.len());
        for &(offset, len) in locations {
            let end = offset
                .checked_add(len)
                .expect("alias offset + len overflows u32");
            assert!(
                end <= entry_len,
                "alias offset {offset}+len {len}={end} exceeds entry length {entry_len}",
            );
            let rel_start = u32::try_from(offset).expect("alias offset exceeds u32");
            let len = u32::try_from(len).expect("alias len exceeds u32");
            locs.push((rel_start, len));
        }
        self.born.push(self.born_of(entry_len));
        self.rows.push(Row::Alias(LocsAndBase { base, locs }));
        self.next += 1;
    }

    /// Consume the next source entry as one **owned** row: `seq` (and the
    /// equal-length `qual` right after it) is copied into the shared arena,
    /// decoupled from the source bytes. `anchor` is the read-relative
    /// `(start, len)` span this content stands in for — what write-back overwrites
    /// in the live read and what liftover lifts — and may differ in length from
    /// the content. Use when a tag's content diverges from any single read slice
    /// (e.g. a regex replacement that conjures or reorders bytes).
    ///
    /// # Panics
    /// - If all source entries have already been consumed.
    /// - If `seq.len() != qual.len()`.
    /// - If `anchor.0 + anchor.1` exceeds the source entry's length.
    /// - If the arena offset, `anchor`, or content length would exceed `u32::MAX`.
    pub fn push_owned_row(&mut self, locations: &[(u32, u32)], seq: &[u8], qual: &[u8]) {
        assert!(
            self.next < self.source.len(),
            "multi-location alias builder: all {} source entries already consumed",
            self.source.len(),
        );
        assert_eq!(
            seq.len(),
            qual.len(),
            "owned row seq ({}) / qual ({}) length mismatch",
            seq.len(),
            qual.len(),
        );
        let r = self.source.storage.entry_range(self.next);
        let entry_len: u32 = (r.end - r.start)
            .try_into()
            .expect("region longer than u32");
        let base = u32::try_from(r.start).expect("alias base exceeds u32");
        let mut locs: Locs = SmallVec::with_capacity(locations.len());
        for &(offset, len) in locations {
            let end = offset
                .checked_add(len)
                .expect("alias offset + len overflows u32");
            assert!(
                end <= entry_len,
                "alias offset {offset}+len {len}={end} exceeds entry length {entry_len}",
            );
            let rel_start = u32::try_from(offset).expect("alias offset exceeds u32");
            let len = u32::try_from(len).expect("alias len exceeds u32");
            locs.push((rel_start, len));
        }
        let anchor = LocsAndBase { base, locs };
        let off = u32::try_from(self.owned.len()).expect("owned arena offset exceeds u32");
        let len = u32::try_from(seq.len()).expect("owned content len exceeds u32");
        self.owned.extend_from_slice(seq);
        self.owned.extend_from_slice(qual);
        self.born.push(self.born_of(entry_len));
        self.rows.push(Row::Owned { anchor, off, len });
        self.next += 1;
    }

    /// Capture the current source entry's birth frame: its edit-log generation
    /// and the `entry_len` already computed by the caller (the entry's visible
    /// length in that frame). Called once per pushed row, before `self.next`
    /// advances.
    fn born_of(&self, entry_len: u32) -> RowBorn {
        let generation = u32::try_from(
            self.source
                .generation(self.next)
                .expect("entry in range while building"),
        )
        .expect("generation exceeds u32");
        RowBorn {
            generation,
            len: entry_len,
        }
    }

    /// Like [`push_row`](Self::push_row), but takes each location as a half-open
    /// `start..end` [`Range`] into the source entry's visible bytes — the common
    /// shape when hits arrive as ranges — converting each to the `(start, len)`
    /// form `push_row` records and delegating to it for all bounds checking.
    ///
    /// # Panics
    /// - Every condition of [`push_row`](Self::push_row).
    /// - If any range is reversed (`end < start`).
    pub fn push_row_from_ranges(&mut self, locations: &[Range<u32>]) {
        let mut locs: Locs = SmallVec::with_capacity(locations.len());
        for r in locations {
            let len = r
                .end
                .checked_sub(r.start)
                .unwrap_or_else(|| panic!("reversed range {}..{}", r.start, r.end));
            locs.push((r.start, len));
        }
        self.push_row(&locs);
    }

    /// Stamp the snapshot's opaque [`source_id`](DualStringPodMultiLocation::source_id)
    /// (default `0`). `stringpod` never interprets it; callers use it to record
    /// which logical source (e.g. read segment) this snapshot was aliased from,
    /// so the column can later be rebuilt against that same source.
    #[must_use]
    pub fn with_source_id(mut self, source_id: u32) -> Self {
        self.source_id = source_id;
        self
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
            owned: self.owned,
            rows: self.rows,
            born: self.born,
            source_id: self.source_id,
        }
    }
}
