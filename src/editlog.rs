//! Coordinate liftover for edits applied to a single column entry.
//!
//! A [`StringPod`](crate::StringPod) / [`DualStringPod`](crate::DualStringPod)
//! built through an `*AliasBuilder` captures a *snapshot* of some source
//! entry's bytes: it co-owns the source `Arc<[u8]>` and records absolute byte
//! ranges, frozen at the moment the alias was taken. The snapshot stays valid
//! forever (COW guarantees the bytes it points at never change underneath it),
//! but it loses track of *where, in the source's current coordinate frame,*
//! those bytes now live once the source is edited.
//!
//! [`EditLog`] closes that gap. It records the sequence of edits applied to a
//! source entry — `cut_start`, `cut_end`, `prefix`, `postfix`, in-place length
//! changes (`splice`), and `reverse` (`reflect`) — and then *lifts* a position
//! or a region from the original frame into the current frame, reporting
//! [`Deleted`](OffsetLift::Deleted) / [`Dropped`](RegionLift::Dropped) when the
//! bytes no longer survive as a usable coordinate.
//!
//! This is the same problem as genomics coordinate liftover (a CIGAR string
//! over a single read): an insertion shifts everything after it, a deletion
//! removes a span, and a position that lands inside a deletion has no image.
//! The implementation is deliberately dependency-free and operates purely on
//! integer coordinates — it never touches the bytes, so one `EditLog` can be
//! shared across every entry of a column (each entry supplies its own
//! `orig_len`), and it applies identically to the seq and qual buffers of a
//! [`DualStringPod`](crate::DualStringPod).
//!
//! # Querying part of the history
//!
//! A coordinate snapshot taken *after* some edits already happened (a tag born
//! mid-pipeline) only needs the edits recorded since. Read the current edit
//! count with [`op_count`](EditLog::op_count) as a *generation* when the
//! snapshot is taken, then later replay just the tail with
//! [`view_from`](EditLog::view_from). The returned [`EditLogView`] lifts a
//! coordinate from the frame *as it stood at that generation*, so the caller
//! passes the entry's length **at that generation** as `orig_len`.
//!
//! # Semantics
//!
//! Coordinates are byte offsets. A region `[start, start + len)` is the set of
//! original bytes `start, start + 1, …, start + len - 1`. Lifting it succeeds
//! (`Kept`) only if every one of those bytes still exists *and they remain
//! contiguous*: a deletion touching the region, or an insertion landing
//! *strictly inside* it, makes it `Dropped`. Insertions exactly at a region
//! boundary stay *outside* the region (boundaries are exclusive), so a tag
//! keeps referring to its own original bytes rather than swallowing freshly
//! inserted ones. `reflect` reverses the frame: the bytes stay contiguous (in
//! reversed order), so a region survives a reflect and maps to its mirror span.
//!
//! Coordinates that don't describe a valid point/region in the frame being
//! lifted *from* are reported as an [`EditLogError`] (this crate carries no
//! `anyhow`), never a panic. `Deleted` / `Dropped` are ordinary outcomes, not
//! errors. The full liftover test suite lives in `tests/editlog_liftover.rs`,
//! where every claim is cross-checked against a brute-force cell-tracking model.

/// One recorded edit, in the coordinate frame *as it stood when the edit was
/// applied*. Length-relative edits (`CutEnd`, `Postfix`, `Reflect`) are
/// resolved against the running length at lift time, so a single log composes
/// over entries of differing lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Remove `n` bytes from the front (clamped to the current length).
    CutStart(usize),
    /// Remove `n` bytes from the back (clamped to the current length).
    CutEnd(usize),
    /// Insert `k` bytes at the front.
    Prefix(usize),
    /// Insert `k` bytes at the back.
    Postfix(usize),
    /// At offset `at`, delete `del` bytes then insert `ins` bytes. `at` and
    /// `del` are clamped to the current length at lift time.
    Splice { at: usize, del: usize, ins: usize },
    /// Reverse the whole coordinate frame.
    Reflect,
}

/// The image of a single original byte under an [`EditLog`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetLift {
    /// The byte survives and now sits at this offset in the current frame.
    At(usize),
    /// The byte was removed by some edit and has no image.
    Deleted,
}

/// The image of an original region under an [`EditLog`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionLift {
    /// The region survives intact and now spans `[start, start + len)` in the
    /// current frame. `len` always equals the queried length.
    Kept {
        /// First offset of the region in the current frame.
        start: usize,
        /// Length of the region (unchanged from the query).
        len: usize,
    },
    /// The region was partially deleted, or split by an interior insertion, and
    /// can no longer be addressed as one contiguous span.
    Dropped,
}

/// A liftover query that doesn't describe a valid point or region in the frame
/// being lifted *from*, or a generation past the recorded history. These are
/// caller contract violations surfaced as values rather than panics — the crate
/// carries no `anyhow`. Note that [`OffsetLift::Deleted`] / [`RegionLift::Dropped`]
/// are *not* errors: they're ordinary outcomes returned inside `Ok`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditLogError {
    /// [`map_position`](EditLog::map_position) was asked for a byte at or beyond
    /// the source length.
    PositionOutOfBounds {
        /// The position that was queried.
        position: usize,
        /// The original length it was queried against.
        orig_len: usize,
    },
    /// [`map_region`](EditLog::map_region) was given a zero-length region; a
    /// region must cover at least one byte to have a well-defined image.
    EmptyRegion,
    /// [`map_region`](EditLog::map_region)'s `[start, start + len)` doesn't fit
    /// within the source length (or overflows `usize`).
    RegionOutOfBounds {
        /// First offset of the queried region.
        start: usize,
        /// Length of the queried region.
        len: usize,
        /// The original length it was queried against.
        orig_len: usize,
    },
    /// [`view_from`](EditLog::view_from) was given a generation past the number
    /// of recorded edits — typically a coordinate snapshot taken against a
    /// *different* log.
    GenerationOutOfRange {
        /// The generation that was requested.
        generation: usize,
        /// The number of edits actually recorded.
        recorded: usize,
    },
    /// A [`ColumnEdits`](crate::ColumnEdits) query named an entry past the
    /// number of live entries.
    RowOutOfBounds {
        /// The row that was queried.
        row: usize,
        /// The number of live entries.
        len: usize,
    },
}

impl std::fmt::Display for EditLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            EditLogError::PositionOutOfBounds { position, orig_len } => {
                write!(
                    f,
                    "position {position} out of bounds for original length {orig_len}"
                )
            }
            EditLogError::EmptyRegion => f.write_str("a region must cover at least one byte"),
            EditLogError::RegionOutOfBounds {
                start,
                len,
                orig_len,
            } => write!(
                f,
                "region [{start}, {start} + {len}) exceeds original length {orig_len}"
            ),
            EditLogError::GenerationOutOfRange {
                generation,
                recorded,
            } => write!(
                f,
                "generation {generation} is past the {recorded} recorded edit(s)"
            ),
            EditLogError::RowOutOfBounds { row, len } => {
                write!(f, "row {row} out of bounds for {len} live entries")
            }
        }
    }
}

impl std::error::Error for EditLogError {}

/// Current length after replaying `ops` over an entry of length `orig_len`.
fn current_len_in(ops: &[Op], orig_len: usize) -> usize {
    let mut cur = orig_len;
    for op in ops {
        match *op {
            Op::CutStart(n) | Op::CutEnd(n) => cur -= n.min(cur),
            Op::Prefix(k) | Op::Postfix(k) => cur += k,
            Op::Splice { at, del, ins } => {
                let a = at.min(cur);
                let d = del.min(cur - a);
                cur = (cur - d) + ins;
            }
            Op::Reflect => {}
        }
    }
    cur
}

/// Forward-replay `position` through `ops` (shared by [`EditLog`] and [`EditLogView`]).
fn map_position_in(
    ops: &[Op],
    position: usize,
    orig_len: usize,
) -> Result<OffsetLift, EditLogError> {
    if position >= orig_len {
        return Err(EditLogError::PositionOutOfBounds { position, orig_len });
    }
    let mut pos = position;
    let mut cur_len = orig_len;
    for op in ops {
        match *op {
            Op::Reflect => pos = (cur_len - 1) - pos,
            Op::CutStart(n) => {
                let d = n.min(cur_len);
                if pos < d {
                    return Ok(OffsetLift::Deleted);
                }
                pos -= d;
                cur_len -= d;
            }
            Op::CutEnd(n) => {
                let d = n.min(cur_len);
                let keep = cur_len - d;
                if pos >= keep {
                    return Ok(OffsetLift::Deleted);
                }
                cur_len = keep;
            }
            Op::Prefix(k) => {
                pos += k;
                cur_len += k;
            }
            Op::Postfix(k) => cur_len += k,
            Op::Splice { at, del, ins } => {
                let a = at.min(cur_len);
                let d = del.min(cur_len - a);
                if pos >= a && pos < a + d {
                    return Ok(OffsetLift::Deleted);
                }
                if pos >= a + d {
                    pos = (pos + ins) - d;
                }
                cur_len = (cur_len + ins) - d;
            }
        }
    }
    Ok(OffsetLift::At(pos))
}

/// Lift a region cell-by-cell through `ops` (shared by [`EditLog`] and [`EditLogView`]).
fn map_region_in(
    ops: &[Op],
    start: usize,
    len: usize,
    orig_len: usize,
) -> Result<RegionLift, EditLogError> {
    if len == 0 {
        return Err(EditLogError::EmptyRegion);
    }
    match start.checked_add(len) {
        Some(end) if end <= orig_len => {}
        _ => {
            return Err(EditLogError::RegionOutOfBounds {
                start,
                len,
                orig_len,
            });
        }
    }
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for cell in start..start + len {
        match map_position_in(ops, cell, orig_len)? {
            OffsetLift::Deleted => return Ok(RegionLift::Dropped),
            OffsetLift::At(p) => {
                if p < lo {
                    lo = p;
                }
                if p > hi {
                    hi = p;
                }
            }
        }
    }
    // Survivors are contiguous iff they fill exactly `len` consecutive offsets.
    // A `reflect` reverses their order but keeps them packed; an interior
    // insertion opens a gap (range wider than `len`); an interior deletion is
    // already caught above as a missing byte.
    if (hi - lo) + 1 == len {
        Ok(RegionLift::Kept { start: lo, len })
    } else {
        Ok(RegionLift::Dropped)
    }
}

/// An ordered, dependency-free log of edits applied to one coordinate frame,
/// able to lift original positions and regions into the current frame.
///
/// Record edits in the same order you apply them to the source pod (the method
/// names mirror the pod mutators), then query with [`map_position`](Self::map_position)
/// and [`map_region`](Self::map_region). The log stores only integers, so
/// cloning is cheap and one log can drive a whole column.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditLog {
    ops: Vec<Op>,
}

impl EditLog {
    /// A log with no recorded edits (the identity transform).
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// `true` if no edits have been recorded (lifts are the identity).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of recorded edits. Doubles as a *generation* counter: capture it
    /// when a coordinate snapshot is taken, then replay the tail later with
    /// [`view_from`](Self::view_from).
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Drop the running length's first `n` bytes (mirrors `StringPod::cut_start`).
    pub fn cut_start(&mut self, n: usize) {
        self.ops.push(Op::CutStart(n));
    }

    /// Drop the running length's last `n` bytes (mirrors `StringPod::cut_end`).
    pub fn cut_end(&mut self, n: usize) {
        self.ops.push(Op::CutEnd(n));
    }

    /// Prepend `k` bytes (mirrors `StringPod::prefix`, where `k` is the prefix
    /// length).
    pub fn prefix(&mut self, k: usize) {
        self.ops.push(Op::Prefix(k));
    }

    /// Append `k` bytes (mirrors `StringPod::postfix`, where `k` is the suffix
    /// length).
    pub fn postfix(&mut self, k: usize) {
        self.ops.push(Op::Postfix(k));
    }

    /// At offset `at`, delete `del` bytes and insert `ins` bytes — the general
    /// in-place length change (e.g. writing a different-length tag back into a
    /// read). `cut_start`/`cut_end`/`prefix`/`postfix` are the common special
    /// cases; this covers the rest.
    pub fn splice(&mut self, at: usize, del: usize, ins: usize) {
        self.ops.push(Op::Splice { at, del, ins });
    }

    /// Reverse the coordinate frame (mirrors `StringPod::reverse`).
    pub fn reflect(&mut self) {
        self.ops.push(Op::Reflect);
    }

    /// A view over the edits recorded *at or after* `generation` (an op index
    /// previously read from [`op_count`](Self::op_count), e.g. when a coordinate
    /// snapshot was taken). Replaying the returned view lifts a coordinate from
    /// the frame *as it stood at `generation`* into the current frame: pass the
    /// entry's length **at that generation** as `orig_len`. `generation` equal
    /// to `op_count()` yields an empty (identity) view.
    ///
    /// # Errors
    /// [`EditLogError::GenerationOutOfRange`] if `generation > self.op_count()`.
    pub fn view_from(&self, generation: usize) -> Result<EditLogView<'_>, EditLogError> {
        match self.ops.get(generation..) {
            Some(ops) => Ok(EditLogView { ops }),
            None => Err(EditLogError::GenerationOutOfRange {
                generation,
                recorded: self.ops.len(),
            }),
        }
    }

    /// The current length of an entry whose original length was `orig_len`,
    /// after applying every recorded edit.
    #[must_use]
    pub fn current_len(&self, orig_len: usize) -> usize {
        current_len_in(&self.ops, orig_len)
    }

    /// Lift the original byte at index `position` into the current frame.
    ///
    /// Returns [`OffsetLift::Deleted`] if some edit removed that byte.
    ///
    /// # Errors
    /// [`EditLogError::PositionOutOfBounds`] if `position >= orig_len`.
    pub fn map_position(
        &self,
        position: usize,
        orig_len: usize,
    ) -> Result<OffsetLift, EditLogError> {
        map_position_in(&self.ops, position, orig_len)
    }

    /// Lift the original region `[start, start + len)` into the current frame.
    ///
    /// Returns [`RegionLift::Dropped`] if any byte of the region was deleted, or
    /// if an insertion landed strictly inside it (splitting it). A `reflect`
    /// keeps the region (mapped to its mirror span). See the module docs for
    /// the exact boundary rules.
    ///
    /// # Errors
    /// [`EditLogError::EmptyRegion`] if `len == 0`; [`EditLogError::RegionOutOfBounds`]
    /// if `start + len > orig_len`.
    pub fn map_region(
        &self,
        start: usize,
        len: usize,
        orig_len: usize,
    ) -> Result<RegionLift, EditLogError> {
        map_region_in(&self.ops, start, len, orig_len)
    }
}

/// A borrowed window over a *suffix* of an [`EditLog`]'s edits, produced by
/// [`EditLog::view_from`]. Lifts coordinates from the frame at the window's
/// start into the current frame, with the same semantics as [`EditLog`] —
/// callers pass the entry's length *at the window's start* as `orig_len`.
#[derive(Clone, Copy, Debug)]
pub struct EditLogView<'a> {
    ops: &'a [Op],
}

impl EditLogView<'_> {
    /// `true` if the window covers no edits (lifts are the identity).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of edits in the window.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// The current length of an entry whose length at the window's start was
    /// `orig_len`, after applying every edit in the window.
    #[must_use]
    pub fn current_len(&self, orig_len: usize) -> usize {
        current_len_in(self.ops, orig_len)
    }

    /// Lift a position from the window-start frame into the current frame.
    ///
    /// # Errors
    /// [`EditLogError::PositionOutOfBounds`] if `position >= orig_len`.
    pub fn map_position(
        &self,
        position: usize,
        orig_len: usize,
    ) -> Result<OffsetLift, EditLogError> {
        map_position_in(self.ops, position, orig_len)
    }

    /// Lift a region from the window-start frame into the current frame.
    ///
    /// # Errors
    /// [`EditLogError::EmptyRegion`] if `len == 0`; [`EditLogError::RegionOutOfBounds`]
    /// if `start + len > orig_len`.
    pub fn map_region(
        &self,
        start: usize,
        len: usize,
        orig_len: usize,
    ) -> Result<RegionLift, EditLogError> {
        map_region_in(self.ops, start, len, orig_len)
    }
}
