//! Iterate FASTQ-style records whose columns live in separate pods, using a
//! [`CrossPodLocations`] index over a user type with *named* pod fields.
//!
//! Run with: `cargo run --example fastq_cross`

use bstr::BStr;
use smallvec::{SmallVec, smallvec};
use stringpod::{
    CrossPodLocations, CrossPods, DualStringPod, DualStringPodBuilder, PodMut, PodRef, StringPod,
    StringPodBuilder,
};

/// One FASTQ chunk: a name column, a fused sequence+quality column, and a
/// `+`-line column. All three hold the same number of entries.
struct FastQChunk {
    name: StringPod,
    seq_qual: DualStringPod,
    plus: StringPod,
}

/// The companion: a borrowed view of one record's four parts.
#[derive(Debug)]
struct FastQRead<'a> {
    name: &'a BStr,
    seq: &'a BStr,
    qual: &'a BStr,
    plus: &'a BStr,
}

/// The mutable companion: the same four parts, each as `&mut BStr`.
struct FastQReadMut<'a> {
    name: &'a mut BStr,
    seq: &'a mut BStr,
    qual: &'a mut BStr,
    plus: &'a mut BStr,
}

impl CrossPods for FastQChunk {
    type Companion<'a> = FastQRead<'a>;
    type CompanionMut<'a> = FastQReadMut<'a>;

    // Fixed order: name (col 0), seq_qual (cols 1 & 2: seq then qual), plus (col 3).
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

    fn to_companion_mut(parts: SmallVec<[& mut BStr; 4]>) -> FastQReadMut<'_> {
        let mut it = parts.into_iter();
        FastQReadMut {
            name: it.next().expect("name part"),
            seq: it.next().expect("seq part"),
            qual: it.next().expect("qual part"),
            plus: it.next().expect("plus part"),
        }
    }
}

fn build_chunk() -> FastQChunk {
    let mut names = StringPodBuilder::with_capacity(0, 3);
    names.push(b"read_001");
    names.push(b"read_002");
    names.push(b"read_003");

    let mut seq_qual = DualStringPodBuilder::with_capacity(4, 3);
    seq_qual.push(b"ACGT", b"IIII");
    seq_qual.push(b"TTGG", b"FF##");
    seq_qual.push(b"GGCC", b"@@@@");

    let mut plus = StringPodBuilder::with_capacity(0, 3);
    plus.push(b"+");
    plus.push(b"+");
    plus.push(b"+");

    FastQChunk {
        name: names.finish(),
        seq_qual: seq_qual.finish(),
        plus: plus.finish(),
    }
}

fn main() {
    let mut chunk = build_chunk();

    // One record per row, drawing the whole entry `i` from every column.
    let locs = CrossPodLocations::per_row(&chunk);
    println!("{} records across {} columns", locs.len(), locs.n_columns());

    for read in locs.iter(&chunk) {
        println!(
            "  {} | seq={} qual={} {}",
            read.name, read.seq, read.qual, read.plus
        );
    }

    // Reconstruct tab-separated lines.
    println!("\njoined (tab-separated):");
    let joined = locs.to_joined_string(&chunk, Some(b"\t"));
    for line in &joined {
        println!("  {line}");
    }

    // In-place mutation: iter_mut yields a FastQReadMut per record, with every
    // part as a named &mut field at once — no unsafe, collectable like
    // slice::iter_mut.
    {
        let it = locs
            .try_iter_mut(&mut chunk)
            .expect("buffers were uniquely owned");
        for read in it {
            read.name.make_ascii_uppercase();
            read.seq.make_ascii_lowercase();
            read.qual.make_ascii_uppercase();
            read.plus.make_ascii_uppercase();
        }
    }

    println!("\nafter lowercasing sequences:");
    let locs = CrossPodLocations::per_row(&chunk);
    for read in locs.iter(&chunk) {
        println!("  {} | seq={}", read.name, read.seq);
    }
}
