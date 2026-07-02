# stringpod

Cache-friendly columnar storage for many small byte strings, for example DNA sequences.

## Motivation

- Allocating and deallocating many smallish strings is slow.
- DNA sequencing data should represent sequences and their quality scores in a structure
that ensures their lengths match.
- Zero-copy, O(1) operations are nice. So is COW.

## Types

- **`StringPod`** — one column of byte strings backed by a single `Arc<Vec<u8>>`
  plus columnar metadata.
- **`DualStringPod`** — two parallel byte columns (e.g. FASTQ sequence + quality)
  that share one metadata layout, making the per-entry length invariant
  (`seq.len() == qual.len()`) a compile time invariant.

Each type is built via an owning `*Builder` (pushes into a fresh `Vec<u8>`)
or a `*AliasBuilder` (records sub-string ranges into a pod's buffer
without copying, enabling COW semantics).

## Guiding principle: index-only by default

Alterations rewrite **metadata, not bytes**, whenever that can express the
result. Narrowing, dropping, slicing, truncating, reordering — all are done by
adjusting the columnar index (positions / overlays / counts) over a buffer that
stays shared behind its `Arc`. The byte buffer is only cloned and written when
an operation genuinely changes byte *content* or *grows* an entry — `prefix`,
`postfix`, and the splice/write-back family — and even those clone the `Arc`
lazily (COW) so untouched data is never copied.

A direct consequence: index-only alterations never reclaim space. Truncated
tails, dropped entries and sliced-away regions stay resident in the buffer as
unreferenced bytes (`used_bytes()` shrinks; `buffer_bytes()` does not).
**Compaction is an explicit, separate, user-level step**, orthogonal to the
alterations themselves. *Index-only alterations* never compact behind your back,
so their cost stays predictable; call `compact()` when, and only when,
reclamation actually matters to you.

The one place compaction happens implicitly is the copy in copy-on-write. When
you mutate a pod whose buffer is **shared** with another pod, the buffer must be
cloned anyway — so it is cloned *compacted* (exact-size, footprint-only) rather
than duplicated whole. This costs no extra pass (you were already paying for the
clone) and it is what keeps the common "build N pods over one buffer, then mutate
all of them" pattern from amplifying to N × |buffer|: each descendant ends up
owning just its own bytes. A pod that already owns its buffer is never touched —
mutation of an exclusive pod stays in place, orphans and all.

`pod.compact()` moves every entry's visible bytes into a fresh, exactly-sized
buffer (one allocation per column — `seq` and `qual` for a `DualStringPod`),
drops the orphaned bytes, and leaves the pod owning its buffer outright (a
shared `Arc` is left untouched). Visible contents, entry count, fixed/variable
layout and edit history are all preserved — only the backing storage moves —
so it's safe to call at any point. Afterwards `buffer_bytes() == used_bytes()`.

## Storage strategy

Storage starts `FixedLength` (a stride + count, no positions array) for the
common case where every entry has the same length.  The first push with a
different length triggers a one-time O(n) promotion to `Variable` (a
`Vec<(u32, u32)>` of `(start, stop)` positions).

`cut_start` / `cut_end` apply a global head/tail overlay in O(1) — no bytes
move.  `max_len` truncates every entry index-only (O(1) on `FixedLength`, an
O(n) position rewrite on `Variable`).  `drain` removes entries but leaves their
bytes orphaned in the buffer.  None of these reclaim space — rebuild via a new
pod if reclamation matters (see the guiding principle above).

## Example

```rust
use stringpod::StringPodBuilder;

let mut bld = StringPodBuilder::with_capacity(150, 1024);
for read in reads {
    bld.push(read);
}
let pod = bld.finish();

for seq in &pod {
    println!("{}", seq);
}

//now remove the first 5 bytes
pod.cut_start(5);
for seq in &pod {
    println!("Shorter: {}", seq);
}


```

## Cross-pod record iteration

`CrossPodLocations` is a zero-copy index that addresses one logical record's
bytes as a list of sub-slices scattered across several pods at once.  The
motivating shape is a FASTQ chunk whose columns live in separate pods (a name
`StringPod`, a sequence+quality `DualStringPod`, a `+`-line `StringPod`) where
you want to iterate *records* — `(name, seq, qual, plus)` tuples — drawing one
part from each column.

```rust
// 1. Implement CrossPods for your column container.
impl CrossPods for FastQChunk {
    type Companion<'a> = FastQRead<'a>;       // borrowed view of one record
    type CompanionMut<'a> = FastQReadMut<'a>; // mutable counterpart

    fn pods(&self) -> SmallVec<[PodRef<'_>; 4]> { /* name, seq_qual, plus */ }
    fn pods_mut(&mut self) -> SmallVec<[PodMut<'_>; 4]> { /* … */ }
    fn to_companion<'a>(parts: &[&'a BStr]) -> FastQRead<'a> { /* … */ }
    fn to_companion_mut(parts: SmallVec<[&mut BStr; 4]>) -> FastQReadMut<'_> { /* … */ }
}

// 2. Build the index once (row-per-entry, or hand-rolled sub-slices).
let locs = CrossPodLocations::per_row(&chunk);

// 3. Iterate records as typed companions — zero-copy, borrowing the pod bytes.
for read in locs.iter(&chunk) {
    println!("{} {} {}", read.name, read.seq, read.qual);
}

// 4. Mutate in place — safe split_at_mut under the hood, collectable like iter_mut.
if let Some(it) = locs.try_iter_mut(&mut chunk) {
    for read in it {
        read.seq.make_ascii_lowercase();
    }
}
```

- **`CrossPodLocations::per_row`** — one record per entry, whole columns zipped.
- **`CrossPodLocations::builder`** — hand-roll records from arbitrary sub-slices
  across columns with `part_whole` / `part_sub`.
- **`iter`** / **`get`** — read-only record access.
- **`try_iter_mut`** — mutable access to a whole record's parts simultaneously,
  with no `unsafe`; returns `None` if any buffer is shared (`Arc` count > 1).
- **`for_each_mut`** — visits one part at a time; safe even with overlapping windows.
- **`to_joined_string`** — materialise each record into a fresh `StringPod` entry,
  with an optional separator.
- **`Pod`** / **`PodRef`** / **`PodMut`** — owned and borrowed handles for
  heterogeneous containers (e.g. `BTreeMap<String, Pod>`).

See the examples for full working code:

- [`examples/fastq_cross.rs`](examples/fastq_cross.rs) — named struct fields,
  typed companion, mutable iteration (`cargo run --example fastq_cross`)
- [`examples/btreemap_cross.rs`](examples/btreemap_cross.rs) — dynamic column
  order via `BTreeMap`, `Vec<&BStr>` companion
  (`cargo run --example btreemap_cross`)

## License

MIT 


## Alternatives

Apache arrow BinaryArray. Covers packed buffers + offset array, even with fixed strides.
Does not cover the DualStringPod case, nor the O(1) operations.



## Changelog
### 0.3.0
- clarify that there is no compaction unless you request it / unless you 
  make_mut()
- max_len() is now COW.
- introduce slicing on pods to get a subset of their contents.
- en-bloc extension from other pods.
