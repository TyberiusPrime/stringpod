# stringpod

Cache-friendly columnar storage for many small byte strings, for example DNA sequences.

## Types

- **`StringPod`** — one column of byte strings backed by a single `Arc<Vec<u8>>`
  plus columnar metadata.
- **`DualStringPod`** — two parallel byte columns (e.g. FASTQ sequence + quality)
  that share one metadata layout, making the per-entry length invariant
  (`seq.len() == qual.len()`) a compile time invariant.

Each type is built via an owning `*Builder` (pushes into a fresh `Vec<u8>`)
or a `*AliasBuilder` (records sub-string ranges into a pod's buffer
without copying, enabling COW semantics).

## Storage strategy

Storage starts `FixedLength` (a stride + count, no positions array) for the
common case where every entry has the same length.  The first push with a
different length triggers a one-time O(n) promotion to `Variable` (a
`Vec<(u32, u32)>` of `(start, stop)` positions).

`cut_start` / `cut_end` apply a global head/tail overlay in O(1) — no bytes
move.  `drain` removes entries but leaves their bytes orphaned in the buffer;
rebuild via a new pod if reclamation matters.

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
