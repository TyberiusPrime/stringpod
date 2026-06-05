//! Cross-pod record locations: address one logical record's bytes as a list of
//! sub-slices ("sublocations") scattered across several [`StringPod`] /
//! [`DualStringPod`] columns at once.
//!
//! The motivating shape is a FASTQ chunk whose columns live in separate pods
//! (a name [`StringPod`], a sequence+quality [`DualStringPod`], a `+`-line
//! [`StringPod`]) but where you want to iterate *records* — `(name, seq, qual,
//! plus)` tuples — drawing one part from each column.
//!
//! ## Pieces
//!
//! * [`CrossPods`] — the adapter trait a user type implements. It exposes its
//!   pods in a fixed order ([`pods`](CrossPods::pods) /
//!   [`pods_mut`](CrossPods::pods_mut)) and knows how to assemble a borrowed
//!   *companion* struct from one record's resolved parts
//!   ([`to_companion`](CrossPods::to_companion)).
//! * [`PodRef`] / [`PodMut`] — borrowed handles to a single or dual pod, the
//!   element type of the slices the trait hands back. [`Pod`] is the owned
//!   counterpart for heterogeneous containers (e.g. a `BTreeMap<_, Pod>`).
//! * [`CrossPodLocations`] — the index itself: a `Vec<SmallVec<[Location; 4]>>`,
//!   one inner list of sublocations per record. It does **not** own the pods;
//!   resolution methods take the pods back so the same index can be applied to
//!   structurally-identical pods.
//!
//! ## Columns
//!
//! Pods are *flattened* into columns left-to-right: a [`StringPod`] is one
//! column, a [`DualStringPod`] is two (sequence then quality). A [`Location`]
//! addresses `column[entry][offset..offset+len]` — a column index, an entry
//! index, and a byte sub-range within that entry. Whole-entry parts (the common
//! case) just span the entry.
//!
//! ## Mutation
//!
//! [`try_iter_mut`](CrossPodLocations::try_iter_mut) yields each record assembled into a
//! [`CompanionMut`](CrossPods::CompanionMut) of `&mut BStr`, collectable just
//! like [`slice::iter_mut`] — and it contains no `unsafe`. Each column buffer is
//! split into its disjoint parts with [`split_at_mut`](slice::split_at_mut),
//! which the borrow checker accepts directly. Distinct pod entries never overlap
//! (non-overlapping by construction in the pod builders, by assertion in the
//! alias builders), so the common indices are trivially disjoint; the one way to
//! create overlap is aiming two [`part_sub`](CrossPodLocationsBuilder::part_sub)
//! windows at the *same* entry, which `try_iter_mut` rejects with a panic.
//! [`for_each_mut`](CrossPodLocations::for_each_mut) is the alternative for that
//! case: it hands out one part at a time, so it stays sound even with overlap.
//!
//! ## Snapshot semantics
//!
//! Locations are byte coordinates captured when the index is built. They stay
//! valid as long as the columns keep their entry layout — in-place byte edits
//! (same length) are fine, but structurally rebuilding a column (`drain`,
//! `prefix`, a fresh builder) invalidates the indices, exactly like the alias
//! builders elsewhere in this crate.

use bstr::{BStr, ByteSlice as _};
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::dual::DualStringPod;
use crate::single::{StringPod, StringPodBuilder};
use crate::storage::Storage;

/// A borrowed handle to one pod, single or dual. The element type of the slice
/// returned by [`CrossPods::pods`].
#[derive(Debug, Clone, Copy)]
pub enum PodRef<'a> {
    Single(&'a StringPod),
    Dual(&'a DualStringPod),
}

/// A mutably-borrowed handle to one pod. The element type of the slice returned
/// by [`CrossPods::pods_mut`].
#[derive(Debug)]
pub enum PodMut<'a> {
    Single(&'a mut StringPod),
    Dual(&'a mut DualStringPod),
}

/// An owned pod, single or dual. Useful when a container stores a heterogeneous
/// mix of pods behind one value type (e.g. `BTreeMap<String, Pod>`), so the
/// [`CrossPods`] impl can hand out [`PodRef`] / [`PodMut`] views in key order.
#[derive(Debug, Clone)]
pub enum Pod {
    Single(StringPod),
    Dual(DualStringPod),
}

impl Pod {
    /// Borrow as a [`PodRef`].
    #[must_use]
    pub fn as_pod_ref(&self) -> PodRef<'_> {
        match self {
            Pod::Single(pod) => PodRef::Single(pod),
            Pod::Dual(pod) => PodRef::Dual(pod),
        }
    }

    /// Borrow mutably as a [`PodMut`].
    #[must_use]
    pub fn as_pod_mut(&mut self) -> PodMut<'_> {
        match self {
            Pod::Single(pod) => PodMut::Single(pod),
            Pod::Dual(pod) => PodMut::Dual(pod),
        }
    }
}

/// Which buffer of a flattened column a location addresses. A [`StringPod`] is
/// `Single`; a [`DualStringPod`] splits into `Seq` then `Qual`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sub {
    Single,
    Seq,
    Qual,
}

/// One sublocation: a byte sub-range of a single entry of a single column.
///
/// `column` indexes the flattened column list (a [`StringPod`] contributes one
/// column, a [`DualStringPod`] two — seq then qual). The addressed bytes are
/// `column[entry][offset..offset + len]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    pub column: u32,
    pub entry: u32,
    pub offset: u32,
    pub len: u32,
}

/// The adapter trait a user type implements to be addressed by a
/// [`CrossPodLocations`] index.
///
/// Implementors expose their pods in a **fixed order** (the same order every
/// call — that order defines the column numbering and the part order passed to
/// [`to_companion`](CrossPods::to_companion)).
pub trait CrossPods {
    /// A borrowed view of one resolved record. Borrows the pods' bytes, not
    /// `Self`.
    type Companion<'a>;

    /// The mutable counterpart of [`Companion`](Self::Companion), yielded by
    /// [`CrossPodLocations::try_iter_mut`].
    type CompanionMut<'a>;

    /// The pods, in fixed order. A [`StringPod`] becomes one column, a
    /// [`DualStringPod`] two (seq then qual), flattened left-to-right.
    fn pods(&self) -> SmallVec<[PodRef<'_>; 4]>;

    /// The same pods in the same order, mutably. Required for
    /// [`CrossPodLocations::try_iter_mut`] / [`for_each_mut`](CrossPodLocations::for_each_mut).
    fn pods_mut(&mut self) -> SmallVec<[PodMut<'_>; 4]>;

    /// Assemble a companion from one record's resolved parts. `parts[k]` is the
    /// bytes addressed by the record's `k`-th [`Location`], in push order.
    fn to_companion<'a>(parts: &[&'a BStr]) -> Self::Companion<'a>;

    /// Assemble a mutable companion from one record's resolved parts, consuming
    /// them (each `&mut BStr` is moved into a field). Parts are in push order,
    /// the same order as [`to_companion`](Self::to_companion).
    fn to_companion_mut(parts: SmallVec<[&mut BStr; 4]>) -> Self::CompanionMut<'_>;
}

// ── column flattening helpers ──────────────────────────────────────────────

/// Map each flattened column to `(pod index, which buffer)`. Driven purely by
/// the single/dual shape, so it is identical whether derived from [`PodRef`]s
/// or [`PodMut`]s.
fn build_colmap<I: Iterator<Item = bool>>(is_dual: I) -> SmallVec<[(usize, Sub); 4]> {
    let mut map: SmallVec<[(usize, Sub); 4]> = SmallVec::new();
    for (pod_index, dual) in is_dual.enumerate() {
        if dual {
            map.push((pod_index, Sub::Seq));
            map.push((pod_index, Sub::Qual));
        } else {
            map.push((pod_index, Sub::Single));
        }
    }
    map
}

fn colmap_for(pods: &[PodRef<'_>]) -> SmallVec<[(usize, Sub); 4]> {
    build_colmap(pods.iter().map(|pod| matches!(pod, PodRef::Dual(_))))
}

/// Number of live entries in a flattened column.
fn col_len(pods: &[PodRef<'_>], colmap: &[(usize, Sub)], column: usize) -> usize {
    let (pod_index, sub) = colmap[column];
    match (pods[pod_index], sub) {
        (PodRef::Single(pod), Sub::Single) => pod.len(),
        (PodRef::Dual(pod), Sub::Seq | Sub::Qual) => pod.len(),
        _ => unreachable!("colmap/pods variant mismatch"),
    }
}

/// Visible length of `entry` in a flattened column.
fn col_entry_len(
    pods: &[PodRef<'_>],
    colmap: &[(usize, Sub)],
    column: usize,
    entry: usize,
) -> usize {
    let (pod_index, sub) = colmap[column];
    match (pods[pod_index], sub) {
        (PodRef::Single(pod), Sub::Single) => pod.entry_len(entry),
        (PodRef::Dual(pod), Sub::Seq | Sub::Qual) => pod.entry_len(entry),
        _ => unreachable!("colmap/pods variant mismatch"),
    }
}

/// Resolve a single location to its bytes.
fn part_bytes<'a>(pods: &[PodRef<'a>], colmap: &[(usize, Sub)], loc: Location) -> &'a BStr {
    let (pod_index, sub) = colmap[loc.column as usize];
    let whole: &'a BStr = match (pods[pod_index], sub) {
        (PodRef::Single(pod), Sub::Single) => pod.get(loc.entry as usize),
        (PodRef::Dual(pod), Sub::Seq) => pod.seq(loc.entry as usize),
        (PodRef::Dual(pod), Sub::Qual) => pod.qual(loc.entry as usize),
        _ => unreachable!("colmap/pods variant mismatch"),
    };
    let start = loc.offset as usize;
    let stop = start + loc.len as usize;
    BStr::new(&whole[start..stop])
}

/// Per-column mutable view: the whole buffer slice, the storage that maps
/// entries to relative ranges, and the column's first-byte offset into the
/// buffer (0 for a single pod and a dual `seq`; the recorded translation for a
/// dual `qual`). The buffer borrow and the storage borrow are disjoint fields
/// of the same pod, so both can carry the full `'a`.
struct ColView<'a> {
    buffer: &'a mut [u8],
    storage: &'a Storage,
    first_byte: usize,
}

/// Flatten the pods into per-column mutable views, or `None` if any pod's
/// buffer is shared (`Arc` strong count > 1). Consumes the `PodMut`s so each
/// `&mut [u8]` carries the pods' own `'a` rather than a borrow of a local.
fn column_views_mut<'a>(pods: SmallVec<[PodMut<'a>; 4]>) -> Option<SmallVec<[ColView<'a>; 4]>> {
    let mut out: SmallVec<[ColView<'a>; 4]> = SmallVec::new();
    for pod in pods {
        match pod {
            PodMut::Single(pod) => {
                out.push(ColView {
                    buffer: Arc::get_mut(&mut pod.data)?.as_mut_slice(),
                    storage: &pod.storage,
                    first_byte: 0,
                });
            }
            PodMut::Dual(pod) => {
                let seq = Arc::get_mut(&mut pod.seq)?.as_mut_slice();
                let qual = Arc::get_mut(&mut pod.qual)?.as_mut_slice();
                out.push(ColView {
                    buffer: seq,
                    storage: &pod.storage,
                    first_byte: pod.seq_first_byte,
                });
                out.push(ColView {
                    buffer: qual,
                    storage: &pod.storage,
                    first_byte: pod.qual_first_byte,
                });
            }
        }
    }
    Some(out)
}

/// Resolve a single location to its bytes mutably, or `None` if the backing
/// buffer is shared. The returned borrow re-borrows `pods` for its lifetime, so
/// only one part is ever live at a time — sound even if two locations overlap.
fn part_bytes_mut<'p>(
    pods: &'p mut [PodMut<'_>],
    colmap: &[(usize, Sub)],
    loc: Location,
) -> Option<&'p mut BStr> {
    let (pod_index, sub) = colmap[loc.column as usize];
    let whole: &'p mut BStr = match (&mut pods[pod_index], sub) {
        (PodMut::Single(pod), Sub::Single) => pod.get_mut(loc.entry as usize)?,
        (PodMut::Dual(pod), Sub::Seq) => pod.seq_mut(loc.entry as usize)?,
        (PodMut::Dual(pod), Sub::Qual) => pod.qual_mut(loc.entry as usize)?,
        _ => unreachable!("colmap/pods variant mismatch"),
    };
    let start = loc.offset as usize;
    let stop = start + loc.len as usize;
    Some(whole[start..stop].as_bstr_mut())
}

// ── the index ──────────────────────────────────────────────────────────────

/// A list of cross-pod record locations. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct CrossPodLocations {
    records: Vec<SmallVec<[Location; 4]>>,
    n_columns: u32,
}

impl CrossPodLocations {
    /// Build the "natural" index: one record per entry, pulling the whole
    /// entry `i` from every column (FASTQ-style row zipping). All columns must
    /// have the same number of entries.
    ///
    /// # Panics
    /// If the columns do not all have the same entry count.
    #[must_use]
    pub fn per_row<T: CrossPods>(pods: &T) -> Self {
        let podrefs = pods.pods();
        let colmap = colmap_for(&podrefs);

        let row_count = if colmap.is_empty() {
            0
        } else {
            let first = col_len(&podrefs, &colmap, 0);
            for column in 1..colmap.len() {
                assert_eq!(
                    col_len(&podrefs, &colmap, column),
                    first,
                    "per_row: column {column} has a different entry count than column 0",
                );
            }
            first
        };

        let mut records: Vec<SmallVec<[Location; 4]>> = Vec::with_capacity(row_count);
        for entry in 0..row_count {
            let mut record: SmallVec<[Location; 4]> = SmallVec::with_capacity(colmap.len());
            for column in 0..colmap.len() {
                let len = col_entry_len(&podrefs, &colmap, column, entry);
                record.push(Location {
                    column: u32::try_from(column).expect("too many columns"),
                    entry: u32::try_from(entry).expect("entry index exceeds u32"),
                    offset: 0,
                    len: u32::try_from(len).expect("entry length exceeds u32"),
                });
            }
            records.push(record);
        }

        Self {
            records,
            n_columns: u32::try_from(colmap.len()).expect("too many columns"),
        }
    }

    /// Start a builder for a hand-rolled index (records composed of arbitrary
    /// sub-slices across columns).
    #[must_use]
    pub fn builder<T: CrossPods>(pods: &T) -> CrossPodLocationsBuilder<'_> {
        CrossPodLocationsBuilder::new(pods)
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Number of flattened columns the index was built against.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.n_columns as usize
    }

    /// The raw sublocations of record `index`.
    ///
    /// # Panics
    /// If `index >= self.len()`.
    #[must_use]
    pub fn record(&self, index: usize) -> &[Location] {
        &self.records[index]
    }

    /// Resolve record `index` into a companion, or `None` if out of range.
    #[must_use]
    pub fn get<'a, T: CrossPods>(&'a self, pods: &'a T, index: usize) -> Option<T::Companion<'a>> {
        let record = self.records.get(index)?;
        let podrefs = pods.pods();
        let colmap = colmap_for(&podrefs);
        let mut parts: SmallVec<[&'a BStr; 4]> = SmallVec::with_capacity(record.len());
        for loc in record {
            parts.push(part_bytes(&podrefs, &colmap, *loc));
        }
        Some(T::to_companion(&parts))
    }

    /// Iterate records as companions.
    #[must_use]
    pub fn iter<'a, T: CrossPods>(&'a self, pods: &'a T) -> CrossPodRecords<'a, T> {
        let podrefs = pods.pods();
        let colmap = colmap_for(&podrefs);
        CrossPodRecords {
            records: &self.records,
            pods: podrefs,
            colmap,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Join each record's parts into one entry of a fresh [`StringPod`],
    /// optionally inserting `separator` between parts. Variable-length output,
    /// so the result is a `Variable`-storage pod.
    #[must_use]
    pub fn to_joined_string<T: CrossPods>(&self, pods: &T, separator: Option<&[u8]>) -> StringPod {
        let podrefs = pods.pods();
        let colmap = colmap_for(&podrefs);
        let mut builder = StringPodBuilder::with_capacity(0, self.records.len());
        let mut buf: Vec<u8> = Vec::new();
        for record in &self.records {
            buf.clear();
            for (part_index, loc) in record.iter().enumerate() {
                if let (true, Some(sep)) = (part_index > 0, separator) {
                    buf.extend_from_slice(sep);
                }
                let bytes: &[u8] = part_bytes(&podrefs, &colmap, *loc);
                buf.extend_from_slice(bytes);
            }
            builder.push(&buf);
        }
        builder.finish()
    }

    /// Iterate records, yielding each record's parts assembled into a
    /// [`CompanionMut`](CrossPods::CompanionMut) of `&mut BStr` — all of a
    /// record's parts at once, and (like [`slice::iter_mut`]) collectable across
    /// records. Contains no `unsafe`: each column buffer is split into its
    /// disjoint parts with [`split_at_mut`](slice::split_at_mut).
    ///
    /// Returns `None` (mutating nothing) if any pod's backing buffer is shared
    /// (`Arc` strong count > 1); release the other references and retry.
    ///
    /// # Panics
    /// If two locations address overlapping bytes of the same column — only
    /// reachable by pointing two [`part_sub`](CrossPodLocationsBuilder::part_sub)
    /// windows at one entry. Distinct entries never overlap, so indices built by
    /// [`per_row`](Self::per_row) / [`push_row`](CrossPodLocationsBuilder::push_row)
    /// or from whole entries never trip this. Use
    /// [`for_each_mut`](Self::for_each_mut) when you do need overlapping windows.
    #[must_use]
    pub fn try_iter_mut<'a, T: CrossPods>(&self, pods: &'a mut T) -> Option<RecordsMut<'a, T>> {
        let views = column_views_mut(pods.pods_mut())?;

        // For each column, gather the (start, len) of every part that lands in
        // it, tagged with the (record, part) slot it belongs to.
        let mut per_column: Vec<Vec<(usize, usize, usize, usize)>> = vec![Vec::new(); views.len()];
        for (record_index, record) in self.records.iter().enumerate() {
            for (part_index, loc) in record.iter().enumerate() {
                let view = &views[loc.column as usize];
                let entry = view.storage.entry_range(loc.entry as usize);
                let start = entry.start + view.first_byte + loc.offset as usize;
                per_column[loc.column as usize].push((
                    start,
                    loc.len as usize,
                    record_index,
                    part_index,
                ));
            }
        }

        // Empty slots to receive each record's parts as we peel them.
        let mut slots: Vec<SmallVec<[Option<&'a mut BStr>; 4]>> = self
            .records
            .iter()
            .map(|record| {
                let mut slot = SmallVec::new();
                slot.resize_with(record.len(), || None);
                slot
            })
            .collect();

        // Peel each column buffer into its parts. Sorting by start lets a single
        // forward sweep of split_at_mut hand out disjoint &mut sub-slices; a
        // non-positive gap means two parts overlap, which we reject.
        for (mut buffer, refs) in views
            .into_iter()
            .map(|view| view.buffer)
            .zip(per_column.iter_mut())
        {
            refs.sort_unstable_by_key(|&(start, ..)| start);
            let mut consumed = 0usize;
            for &(start, len, record_index, part_index) in refs.iter() {
                assert!(
                    start >= consumed,
                    "iter_mut: part at byte {start} overlaps an earlier part ending at \
                     {consumed} in the same column; use for_each_mut for overlapping windows",
                );
                let (_gap, after_gap) = buffer.split_at_mut(start - consumed);
                let (taken, tail) = after_gap.split_at_mut(len);
                slots[record_index][part_index] = Some(taken.as_bstr_mut());
                buffer = tail;
                consumed = start + len;
            }
        }

        // Drain each record's filled slots into a CompanionMut.
        let mut companions: Vec<T::CompanionMut<'a>> = Vec::with_capacity(slots.len());
        for slot in &mut slots {
            let mut parts: SmallVec<[&'a mut BStr; 4]> = SmallVec::with_capacity(slot.len());
            for part in slot.iter_mut() {
                parts.push(
                    part.take()
                        .expect("every part slot was filled while peeling"),
                );
            }
            companions.push(T::to_companion_mut(parts));
        }

        Some(RecordsMut {
            inner: companions.into_iter(),
        })
    }

    /// Visit every part of every record mutably, one part at a time, calling
    /// `visit(record_index, part_index, &mut bytes)`.
    ///
    /// Returns `false` (mutating nothing) if any pod's backing buffer is shared
    /// (`Arc` strong count > 1). Parts are handed out strictly one at a time, so
    /// this stays sound even when two locations address overlapping bytes —
    /// unlike [`iter_mut`](Self::iter_mut), which exposes a whole record's parts
    /// simultaneously and therefore requires them to be disjoint.
    pub fn for_each_mut<T, F>(&self, pods: &mut T, mut visit: F) -> bool
    where
        T: CrossPods,
        F: FnMut(usize, usize, &mut BStr),
    {
        let mut pod_muts = pods.pods_mut();

        // Pre-flight: bail before touching anything if any buffer is shared, so
        // a partial mutation can't happen.
        for pod in &mut pod_muts {
            let unique = match pod {
                PodMut::Single(pod) => Arc::get_mut(&mut pod.data).is_some(),
                PodMut::Dual(pod) => {
                    Arc::get_mut(&mut pod.seq).is_some() && Arc::get_mut(&mut pod.qual).is_some()
                }
            };
            if !unique {
                return false;
            }
        }

        let colmap = build_colmap(pod_muts.iter().map(|pod| matches!(pod, PodMut::Dual(_))));
        for (record_index, record) in self.records.iter().enumerate() {
            for (part_index, loc) in record.iter().enumerate() {
                let Some(part) = part_bytes_mut(&mut pod_muts, &colmap, *loc) else {
                    return false;
                };
                visit(record_index, part_index, part);
            }
        }
        true
    }
}

/// Iterator over [`CrossPodLocations`] records as companions. Created by
/// [`CrossPodLocations::iter`].
pub struct CrossPodRecords<'a, T: CrossPods> {
    records: &'a [SmallVec<[Location; 4]>],
    pods: SmallVec<[PodRef<'a>; 4]>,
    colmap: SmallVec<[(usize, Sub); 4]>,
    index: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: CrossPods> Iterator for CrossPodRecords<'a, T> {
    type Item = T::Companion<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.records.get(self.index)?;
        self.index += 1;
        let mut parts: SmallVec<[&'a BStr; 4]> = SmallVec::with_capacity(record.len());
        for loc in record {
            parts.push(part_bytes(&self.pods, &self.colmap, *loc));
        }
        Some(T::to_companion(&parts))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.records.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: CrossPods> ExactSizeIterator for CrossPodRecords<'_, T> {}

/// Iterator over [`CrossPodLocations`] records as mutable companions. Created
/// by [`CrossPodLocations::iter_mut`].
///
/// The companions own their `&mut BStr` parts (they don't borrow the iterator),
/// so the iterator is collectable — every part across the whole iteration is a
/// distinct, non-overlapping slice, carved out with safe `split_at_mut`.
pub struct RecordsMut<'a, T: CrossPods> {
    inner: std::vec::IntoIter<T::CompanionMut<'a>>,
}

impl<'a, T: CrossPods> Iterator for RecordsMut<'a, T> {
    type Item = T::CompanionMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: CrossPods> ExactSizeIterator for RecordsMut<'_, T> {}

// ── builder ──────────────────────────────────────────────────────────────

/// Builds a [`CrossPodLocations`] record by record. Parts are pushed into the
/// current record; [`commit`](Self::commit) closes it and starts the next.
pub struct CrossPodLocationsBuilder<'t> {
    pods: SmallVec<[PodRef<'t>; 4]>,
    colmap: SmallVec<[(usize, Sub); 4]>,
    current: SmallVec<[Location; 4]>,
    records: Vec<SmallVec<[Location; 4]>>,
}

impl<'t> CrossPodLocationsBuilder<'t> {
    fn new<T: CrossPods>(pods: &'t T) -> Self {
        let podrefs = pods.pods();
        let colmap = colmap_for(&podrefs);
        Self {
            pods: podrefs,
            colmap,
            current: SmallVec::new(),
            records: Vec::new(),
        }
    }

    /// Number of flattened columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.colmap.len()
    }

    /// Add a whole-entry part (the entire visible entry `entry` of `column`) to
    /// the current record.
    ///
    /// # Panics
    /// If `column` is out of range or `entry` is past the column's end.
    pub fn part_whole(&mut self, column: u32, entry: u32) {
        let column_idx = column as usize;
        assert!(
            column_idx < self.colmap.len(),
            "column {column} out of range ({} columns)",
            self.colmap.len(),
        );
        let len = col_entry_len(&self.pods, &self.colmap, column_idx, entry as usize);
        self.current.push(Location {
            column,
            entry,
            offset: 0,
            len: u32::try_from(len).expect("entry length exceeds u32"),
        });
    }

    /// Add a sub-slice part `column[entry][offset..offset + len]` to the
    /// current record.
    ///
    /// # Panics
    /// If `column` is out of range, `entry` is past the column's end, or
    /// `offset + len` exceeds the entry's visible length.
    pub fn part_sub(&mut self, column: u32, entry: u32, offset: u32, len: u32) {
        let column_idx = column as usize;
        assert!(
            column_idx < self.colmap.len(),
            "column {column} out of range ({} columns)",
            self.colmap.len(),
        );
        let entry_len = col_entry_len(&self.pods, &self.colmap, column_idx, entry as usize);
        let end = offset as usize + len as usize;
        assert!(
            end <= entry_len,
            "sub-slice offset {offset}+len {len}={end} exceeds entry length {entry_len}",
        );
        self.current.push(Location {
            column,
            entry,
            offset,
            len,
        });
    }

    /// Close the current record (which may be empty) and start a new one.
    pub fn commit(&mut self) {
        let record = std::mem::take(&mut self.current);
        self.records.push(record);
    }

    /// Convenience: push a whole-entry part for every column (in column order)
    /// and commit the record in one call.
    ///
    /// # Panics
    /// If `entry` is past the end of any column.
    pub fn push_row(&mut self, entry: u32) {
        let columns = self.colmap.len();
        for column in 0..columns {
            self.part_whole(u32::try_from(column).expect("too many columns"), entry);
        }
        self.commit();
    }

    /// Finalise the index. Any parts pushed but not yet committed are dropped
    /// (call [`commit`](Self::commit) first if you meant to keep them).
    ///
    /// # Panics
    /// If the number of columns exceeds `u32::MAX`.
    #[must_use]
    pub fn finish(self) -> CrossPodLocations {
        CrossPodLocations {
            records: self.records,
            n_columns: u32::try_from(self.colmap.len()).expect("too many columns"),
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason="it's tests")]
mod tests {
    use super::{CrossPodLocations, CrossPods, Pod, PodMut, PodRef};
    use crate::dual::{DualStringPod, DualStringPodBuilder};
    use crate::single::{StringPod, StringPodBuilder};
    use bstr::BStr;
    use smallvec::{SmallVec, smallvec};
    use std::collections::BTreeMap;

    fn b(s: &str) -> &[u8] {
        s.as_bytes()
    }

    // A FASTQ-shaped user type with named pods.
    struct FastQChunk {
        name: StringPod,
        seq_qual: DualStringPod,
        plus: StringPod,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct FastQRead<'a> {
        name: &'a BStr,
        seq: &'a BStr,
        qual: &'a BStr,
        plus: &'a BStr,
    }

    struct FastQReadMut<'a> {
        name: &'a mut BStr,
        seq: &'a mut BStr,
        qual: &'a mut BStr,
        plus: &'a mut BStr,
    }

    impl CrossPods for FastQChunk {
        type Companion<'a> = FastQRead<'a>;
        type CompanionMut<'a> = FastQReadMut<'a>;

        fn pods(&self) -> SmallVec<[PodRef<'_>; 4]> {
            smallvec![
                PodRef::Single(&self.name),
                PodRef::Dual(&self.seq_qual),
                PodRef::Single(&self.plus),
            ]
        }

        fn pods_mut(&mut self) -> SmallVec<[PodMut<'_>; 4]> {
            smallvec![
                PodMut::Single(&mut self.name),
                PodMut::Dual(&mut self.seq_qual),
                PodMut::Single(&mut self.plus),
            ]
        }

        fn to_companion<'a>(parts: &[&'a BStr]) -> FastQRead<'a> {
            FastQRead {
                name: parts[0],
                seq: parts[1],
                qual: parts[2],
                plus: parts[3],
            }
        }

        fn to_companion_mut(parts: SmallVec<[&mut BStr; 4]>) -> FastQReadMut<'_> {
            let mut it = parts.into_iter();
            FastQReadMut {
                name: it.next().expect("Length is known"),
                seq: it.next().expect("Length is known"),
                qual: it.next().expect("Length is known"),
                plus: it.next().expect("Length is known"),
            }
        }
    }

    fn names() -> StringPod {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("read1"));
        bld.push(b("read2"));
        bld.finish()
    }

    fn seq_qual() -> DualStringPod {
        let mut bld = DualStringPodBuilder::with_capacity(4, 2);
        bld.push(b("ACGT"), b("IIII"));
        bld.push(b("TTGG"), b("FF##"));
        bld.finish()
    }

    fn pluses() -> StringPod {
        let mut bld = StringPodBuilder::with_capacity(0, 2);
        bld.push(b("+"));
        bld.push(b("+"));
        bld.finish()
    }

    fn chunk() -> FastQChunk {
        FastQChunk {
            name: names(),
            seq_qual: seq_qual(),
            plus: pluses(),
        }
    }

    #[test]
    fn per_row_columns_and_len() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        // name(1) + seq/qual(2) + plus(1) = 4 columns.
        assert_eq!(locs.n_columns(), 4);
        assert_eq!(locs.len(), 2);
        assert!(!locs.is_empty());
        // Each record has one part per column.
        assert_eq!(locs.record(0).len(), 4);
    }

    #[test]
    fn get_resolves_named_companion() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let read = locs.get(&chunk, 1).expect("can't fail");
        assert_eq!(
            read,
            FastQRead {
                name: BStr::new("read2"),
                seq: BStr::new("TTGG"),
                qual: BStr::new("FF##"),
                plus: BStr::new("+"),
            }
        );
        assert!(locs.get(&chunk, 2).is_none());
    }

    #[test]
    fn iter_yields_every_record() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let reads: Vec<_> = locs.iter(&chunk).collect();
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].name, BStr::new("read1"));
        assert_eq!(reads[0].seq, BStr::new("ACGT"));
        assert_eq!(reads[0].qual, BStr::new("IIII"));
        assert_eq!(reads[1].seq, BStr::new("TTGG"));
    }

    #[test]
    fn iter_is_exact_size() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let it = locs.iter(&chunk);
        assert_eq!(it.len(), 2);
        assert_eq!(it.size_hint(), (2, Some(2)));
    }

    #[test]
    fn to_joined_string_with_separator() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let joined = locs.to_joined_string(&chunk, Some(b("\t")));
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.get(0), BStr::new("read1\tACGT\tIIII\t+"));
        assert_eq!(joined.get(1), BStr::new("read2\tTTGG\tFF##\t+"));
    }

    #[test]
    fn to_joined_string_no_separator() {
        let chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let joined = locs.to_joined_string(&chunk, None);
        assert_eq!(joined.get(0), BStr::new("read1ACGTIIII+"));
    }

    #[test]
    fn for_each_mut_rewrites_parts_in_place() {
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        // Uppercase nothing; instead reverse each part's bytes in place.
        let ok = locs.for_each_mut(&mut chunk, |_record, _part, bytes| {
            bytes.reverse();
        });
        assert!(ok);
        // Re-read through a fresh index against the mutated chunk.
        let locs2 = CrossPodLocations::per_row(&chunk);
        let read = locs2.get(&chunk, 0).unwrap();
        assert_eq!(read.name, BStr::new("1daer"));
        assert_eq!(read.seq, BStr::new("TGCA"));
        assert_eq!(read.qual, BStr::new("IIII"));
    }

    #[test]
    fn for_each_mut_can_target_record_part() {
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        // Only mutate the seq column (column index 1) of record 0.
        let ok = locs.for_each_mut(&mut chunk, |record, part, bytes| {
            if record == 0 && part == 1 {
                bytes.copy_from_slice(b("GGGG"));
            }
        });
        assert!(ok);
        assert_eq!(chunk.seq_qual.seq(0), BStr::new("GGGG"));
        assert_eq!(chunk.seq_qual.seq(1), BStr::new("TTGG")); // untouched
    }

    #[test]
    fn for_each_mut_returns_false_when_shared() {
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        // Share the name buffer by cloning the pod.
        let _shared = chunk.name.clone();
        let ok = locs.for_each_mut(&mut chunk, |_r, _p, bytes| bytes.reverse());
        assert!(!ok);
        // Nothing changed (pre-flight bailed before mutating).
        assert_eq!(chunk.name.get(0), BStr::new("read1"));
    }

    #[test]
    fn iter_mut_mutates_whole_records() {
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        {
            let mut it = locs.try_iter_mut(&mut chunk).unwrap();
            assert_eq!(it.len(), 2);
            for read in &mut it {
                // All four parts are available as named fields at once.
                read.name.make_ascii_uppercase();
                read.seq.reverse();
                read.qual.make_ascii_uppercase(); // already uppercase — no-op
                read.plus.make_ascii_uppercase(); // "+" — no-op
            }
        }
        let after = CrossPodLocations::per_row(&chunk);
        let read0 = after.get(&chunk, 0).unwrap();
        assert_eq!(read0.name, BStr::new("READ1"));
        assert_eq!(read0.seq, BStr::new("TGCA"));
        assert_eq!(read0.qual, BStr::new("IIII")); // untouched
        let read1 = after.get(&chunk, 1).unwrap();
        assert_eq!(read1.name, BStr::new("READ2"));
        assert_eq!(read1.seq, BStr::new("GGTT"));
    }

    #[test]
    fn iter_mut_companions_are_collectable_and_disjoint() {
        // Proves the companions don't borrow the iterator: collect every record
        // into one Vec, then mutate all of them after iteration is done.
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        {
            let reads: Vec<FastQReadMut<'_>> = locs.try_iter_mut(&mut chunk).unwrap().collect();
            assert_eq!(reads.len(), 2);
            for read in reads {
                read.name.make_ascii_lowercase();
                read.seq.make_ascii_lowercase();
            }
        }
        let after = CrossPodLocations::per_row(&chunk);
        assert_eq!(after.get(&chunk, 0).unwrap().name, BStr::new("read1"));
        assert_eq!(after.get(&chunk, 1).unwrap().seq, BStr::new("ttgg"));
    }

    #[test]
    fn iter_mut_returns_none_when_shared() {
        let mut chunk = chunk();
        let locs = CrossPodLocations::per_row(&chunk);
        let _shared = chunk.seq_qual.clone();
        assert!(locs.try_iter_mut(&mut chunk).is_none());
    }

    #[test]
    #[should_panic(expected = "overlaps an earlier part")]
    fn iter_mut_panics_on_overlapping_subslices() {
        let mut chunk = chunk();
        // Two windows into the same entry of column 0 ("read1"): [0,3) and [1,4).
        let locs = {
            let mut builder = CrossPodLocations::builder(&chunk);
            builder.part_sub(0, 0, 0, 3);
            builder.part_sub(0, 0, 1, 3);
            builder.commit();
            builder.finish()
        };
        let _ = locs.try_iter_mut(&mut chunk);
    }

    #[test]
    fn iter_mut_disjoint_subslices_same_buffer_ok() {
        // Two non-overlapping windows into the *same* seq entry — [0,2) and
        // [2,4) — i.e. two parts in one buffer, peeled apart by split_at_mut.
        // Uses the flexible Vec companion since the record has only two parts.
        let mut cols = map_columns();
        let locs = {
            let mut builder = CrossPodLocations::builder(&cols);
            builder.part_sub(1, 0, 0, 2); // column 1 = seq of b_seqqual
            builder.part_sub(1, 0, 2, 2);
            builder.commit();
            builder.finish()
        };
        {
            let mut it = locs.try_iter_mut(&mut cols).unwrap();
            let mut record = it.next().unwrap();
            record[0].copy_from_slice(b"xx");
            record[1].copy_from_slice(b"yy");
        }
        if let Pod::Dual(seq_qual) = &cols.pods["b_seqqual"] {
            assert_eq!(seq_qual.seq(0), BStr::new("xxyy"));
        } else {
            panic!("expected dual pod");
        }
    }

    #[test]
    fn builder_custom_subslices() {
        let chunk = chunk();
        let mut builder = CrossPodLocations::builder(&chunk);
        // Record 0: first two bases of seq (col 1), then the name (col 0).
        builder.part_sub(1, 0, 0, 2);
        builder.part_whole(0, 0);
        builder.commit();
        // Record 1: last base of seq of entry 1, then its qual at the same spot.
        builder.part_sub(1, 1, 3, 1);
        builder.part_sub(2, 1, 3, 1);
        builder.commit();
        let locs = builder.finish();

        assert_eq!(locs.len(), 2);
        let joined = locs.to_joined_string(&chunk, Some(b("|")));
        assert_eq!(joined.get(0), BStr::new("AC|read1"));
        assert_eq!(joined.get(1), BStr::new("G|#"));
    }

    #[test]
    fn builder_push_row_matches_per_row() {
        let chunk = chunk();
        let mut builder = CrossPodLocations::builder(&chunk);
        builder.push_row(0);
        builder.push_row(1);
        let locs = builder.finish();
        let via_per_row = CrossPodLocations::per_row(&chunk);
        assert_eq!(locs.record(0), via_per_row.record(0));
        assert_eq!(locs.record(1), via_per_row.record(1));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn builder_column_out_of_range_panics() {
        let chunk = chunk();
        let mut builder = CrossPodLocations::builder(&chunk);
        builder.part_whole(9, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds entry length")]
    fn builder_subslice_out_of_bounds_panics() {
        let chunk = chunk();
        let mut builder = CrossPodLocations::builder(&chunk);
        builder.part_sub(0, 0, 3, 10); // name "read1" is 5 bytes
    }

    #[test]
    #[should_panic(expected = "different entry count")]
    fn per_row_unequal_columns_panics() {
        let mut short = StringPodBuilder::with_capacity(0, 1);
        short.push(b("only-one"));
        let chunk = FastQChunk {
            name: short.finish(), // 1 entry
            seq_qual: seq_qual(), // 2 entries
            plus: pluses(),
        };
        let _ = CrossPodLocations::per_row(&chunk);
    }

    // ── BTreeMap-of-pods user type (fixed order via sorted keys) ────────────

    struct MapColumns {
        pods: BTreeMap<&'static str, Pod>,
    }

    impl CrossPods for MapColumns {
        // A dynamic companion: the parts in column order.
        type Companion<'a> = Vec<&'a BStr>;
        type CompanionMut<'a> = Vec<&'a mut BStr>;

        fn pods(&self) -> SmallVec<[PodRef<'_>; 4]> {
            self.pods.values().map(Pod::as_pod_ref).collect()
        }

        fn pods_mut(&mut self) -> SmallVec<[PodMut<'_>; 4]> {
            self.pods.values_mut().map(Pod::as_pod_mut).collect()
        }

        fn to_companion<'a>(parts: &[&'a BStr]) -> Vec<&'a BStr> {
            parts.to_vec()
        }

        fn to_companion_mut(parts: SmallVec<[&mut BStr; 4]>) -> Vec<&mut BStr> {
            parts.into_vec()
        }
    }

    fn map_columns() -> MapColumns {
        let mut map: BTreeMap<&'static str, Pod> = BTreeMap::new();
        // Insertion order is deliberately scrambled; BTreeMap iterates sorted.
        map.insert("c_plus", Pod::Single(pluses()));
        map.insert("a_name", Pod::Single(names()));
        map.insert("b_seqqual", Pod::Dual(seq_qual()));
        MapColumns { pods: map }
    }

    #[test]
    fn btreemap_pods_fixed_column_order() {
        let cols = map_columns();
        let locs = CrossPodLocations::per_row(&cols);
        // Sorted keys: a_name(1) + b_seqqual(2) + c_plus(1) = 4 columns.
        assert_eq!(locs.n_columns(), 4);
        let parts = locs.get(&cols, 0).unwrap();
        assert_eq!(
            parts,
            vec![
                BStr::new("read1"), // a_name
                BStr::new("ACGT"),  // b_seqqual / seq
                BStr::new("IIII"),  // b_seqqual / qual
                BStr::new("+"),     // c_plus
            ]
        );
    }

    #[test]
    fn btreemap_pods_join_and_mutate() {
        let mut cols = map_columns();
        let locs = CrossPodLocations::per_row(&cols);
        let joined = locs.to_joined_string(&cols, Some(b(":")));
        assert_eq!(joined.get(1), BStr::new("read2:TTGG:FF##:+"));

        let ok = locs.for_each_mut(&mut cols, |_r, _p, bytes| bytes.make_ascii_uppercase());
        assert!(ok);
        let after = CrossPodLocations::per_row(&cols);
        // Names uppercased too.
        assert_eq!(after.get(&cols, 0).unwrap()[0], BStr::new("READ1"));
    }

    #[test]
    fn send_sync_check() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CrossPodLocations>();
    }
}
