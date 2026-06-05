# stringpod

Cache-friendly columnar storage for many small byte strings, for example DNA sequences.

## Types

- **`StringPod`** — one column of byte strings backed by a single `Arc<Vec<u8>>`
  plus columnar metadata.
- **`DualStringPod`** — two parallel byte columns (e.g. FASTQ sequence + quality)
  that share one metadata layout, making the per-entry length invariant
  (`seq.len() == qual.len()`) a compile time invariant.

Each type is built via an owning `*Builder` (pushes into a fresh `Vec<u8>`)
or a `*AliasBuilder` (records byte ranges into a  pod's buffer
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
```

## License

MIT OR Apache-2.0
