//! Same cross-pod machinery, but the user type stores its pods in a
//! `BTreeMap<String, Pod>` rather than named fields. The map iterates in sorted
//! key order, which gives the *fixed* column order the index requires.
//!
//! Run with: `cargo run --example btreemap_cross`

use bstr::BStr;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use stringpod::{
    CrossPodLocations, CrossPods, DualStringPodBuilder, Pod, PodMut, PodRef, StringPodBuilder,
};

/// A column container keyed by name. The keys impose a deterministic order
/// (`BTreeMap` is sorted), so columns are numbered by sorted key.
struct Columns {
    pods: BTreeMap<String, Pod>,
}

impl CrossPods for Columns {
    // A dynamic companion: just the parts, in column order.
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

fn build_columns() -> Columns {
    let mut name = StringPodBuilder::with_capacity(0, 2);
    name.push(b"read_001");
    name.push(b"read_002");

    let mut seq_qual = DualStringPodBuilder::with_capacity(4, 2);
    seq_qual.push(b"ACGT", b"IIII");
    seq_qual.push(b"TTGG", b"FF##");

    let mut plus = StringPodBuilder::with_capacity(0, 2);
    plus.push(b"+");
    plus.push(b"+");

    // Insertion order scrambled on purpose; sorted iteration fixes the columns
    // to: name, seq_qual(seq, qual), plus.
    let mut pods: BTreeMap<String, Pod> = BTreeMap::new();
    pods.insert("3_plus".to_string(), Pod::Single(plus.finish()));
    pods.insert("1_name".to_string(), Pod::Single(name.finish()));
    pods.insert("2_seq_qual".to_string(), Pod::Dual(seq_qual.finish()));

    Columns { pods }
}

fn main() {
    let columns = build_columns();
    let locs = CrossPodLocations::per_row(&columns);

    println!(
        "{} records, {} columns (sorted keys: {:?})",
        locs.len(),
        locs.n_columns(),
        columns.pods.keys().collect::<Vec<_>>(),
    );

    for (i, parts) in locs.iter(&columns).enumerate() {
        let rendered: Vec<&BStr> = parts;
        println!("  record {i}: {rendered:?}");
    }

    println!("\njoined (pipe-separated):");
    let joined = locs.to_joined_string(&columns, Some(b" | "));
    for line in &joined {
        println!("  {line}");
    }
}
