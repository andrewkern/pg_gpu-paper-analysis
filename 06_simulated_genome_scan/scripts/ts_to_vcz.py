#!/usr/bin/env python
"""
Convert a tskit tree sequence to a VCZ-format zarr store, written directly
(no VCF intermediate, no text formatting per genotype). VCZ is the standard
biobank-scale per-sample genotype layout (bio2zarr / sgkit / pg_gpu's
``HaplotypeMatrix.from_zarr`` all read it), and chunked-compressed zarr is
roughly 30 GB-per-100k-haplotypes/chromosome here, vs the multi-TB dense
matrix or the ~150 GB / ~4-day bgzipped VCF.

The script streams ``ts.variants()`` in fixed-size variant batches, fills an
in-memory ``(batch, n_diploids, 2) int8`` buffer, and writes one zarr chunk at
a time so peak RAM stays at a few GB regardless of chromosome length.

Multiallelic sites (recurrent mutation -- ~0.15% under msprime's binary
mutation model) are kept in the variant order but written as genotype = -1
with ``call_genotype_mask`` = True at those rows, so the on-disk row count
matches ``ts.num_sites`` exactly and consumers can filter via the mask.

Usage
-----
    python 06_simulated_genome_scan/scripts/ts_to_vcz.py [--trees PATH]
                                                        [--out PATH]
                                                        [--variant-chunk N]
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import tskit
import zarr


def parse_args():
    here = Path(__file__).resolve().parents[1]
    default_trees = here / "data" / "ooa_2t12" / "chr15.trees"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees", default=str(default_trees),
                   help=f"input .trees file (default {default_trees})")
    p.add_argument("--out", default=None,
                   help="output VCZ store path (default: <trees>.vcz)")
    p.add_argument("--contig-id", default=None,
                   help="contig name (default: parsed from filename, "
                        "e.g. 'chr15.trees' -> '15')")
    p.add_argument("--variant-chunk", type=int, default=10_000,
                   help="variants per zarr chunk along the variant axis (default 10000)")
    p.add_argument("--progress-every", type=int, default=50,
                   help="print progress every N chunks (default 50)")
    p.add_argument("--pop-file", default=None,
                   help="output path for the sample->population TSV (default: "
                        "<out>.pops.tsv next to the store). Populations are "
                        "inferred from the tree sequence's per-sample population "
                        "metadata. VCZ itself has no population metadata, so a "
                        "companion file is the usual way to carry it.")
    return p.parse_args()


def main():
    args = parse_args()
    trees_path = Path(args.trees)
    out_path = Path(args.out) if args.out else trees_path.with_suffix(".vcz")
    if not trees_path.exists():
        raise SystemExit(f"input not found: {trees_path}")

    contig_id = args.contig_id
    if contig_id is None:
        stem = trees_path.stem  # 'chr15'
        contig_id = stem[3:] if stem.startswith("chr") else stem

    print(f"Loading {trees_path} ...", flush=True)
    ts = tskit.load(str(trees_path))
    n_var = ts.num_sites
    n_nodes = ts.num_samples
    if n_nodes % 2 != 0:
        raise SystemExit(f"expected even sample-node count for diploid layout, got {n_nodes}")
    n_dip = n_nodes // 2
    print(f"  {n_var:,} sites, {n_nodes:,} sample nodes (= {n_dip:,} diploids), "
          f"sequence_length {int(ts.sequence_length):,} bp", flush=True)

    if out_path.exists():
        print(f"  removing existing store {out_path}", flush=True)
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing VCZ store to {out_path}", flush=True)

    B = int(args.variant_chunk)
    sample_chunk = min(n_dip, 100_000)

    g = zarr.create_group(store=str(out_path), overwrite=True)

    # --- variant axis metadata (small, write up front) ---
    positions = ts.tables.sites.position.astype(np.int64)
    if positions.max() > np.iinfo(np.int32).max:
        raise SystemExit("variant_position exceeds int32 range; widen dtype")
    var_pos = g.create_array("variant_position", shape=(n_var,),
                             chunks=(min(n_var, 1_000_000),), dtype="int32")
    var_pos[:] = positions.astype(np.int32)

    var_contig = g.create_array("variant_contig", shape=(n_var,),
                                chunks=(min(n_var, 1_000_000),), dtype="int32")
    var_contig[:] = np.zeros(n_var, dtype=np.int32)

    contig_arr = g.create_array("contig_id", shape=(1,), chunks=(1,), dtype="<U16")
    contig_arr[:] = np.asarray([contig_id], dtype="<U16")

    contig_len_arr = g.create_array("contig_length", shape=(1,), chunks=(1,),
                                    dtype="int64")
    contig_len_arr[:] = np.asarray([int(ts.sequence_length)], dtype=np.int64)

    sample_ids = np.asarray([f"sim_{i}" for i in range(n_dip)], dtype="<U16")
    sa = g.create_array("sample_id", shape=(n_dip,),
                        chunks=(sample_chunk,), dtype="<U16")
    sa[:] = sample_ids

    # --- per-call arrays ---
    # call_genotype: (n_var, n_dip, ploidy=2) int8. -1 marks missing/multiallelic.
    cg = g.create_array("call_genotype",
                        shape=(n_var, n_dip, 2),
                        chunks=(B, sample_chunk, 2),
                        dtype="int8")
    cm = g.create_array("call_genotype_mask",
                        shape=(n_var, n_dip, 2),
                        chunks=(B, sample_chunk, 2),
                        dtype="bool")
    cph = g.create_array("call_genotype_phased",
                         shape=(n_var, n_dip),
                         chunks=(B, sample_chunk),
                         dtype="bool")
    va = g.create_array("variant_allele",
                        shape=(n_var, 2),
                        chunks=(min(n_var, 1_000_000), 2),
                        dtype="<U1")

    # --- stream ts.variants() in batches ---
    buf_gt = np.empty((B, n_dip, 2), dtype=np.int8)
    n_skipped = 0
    n_written = 0
    k_local = 0
    t0 = time.perf_counter()
    last_log = t0

    # all sites get the same placeholder REF/ALT alphabet (msprime binary
    # mutations); for multiallelic-masked rows it is also a placeholder.
    allele_pair = np.asarray(["A", "T"], dtype="<U1")

    def flush(k_local, n_written):
        start, end = n_written, n_written + k_local
        cg[start:end] = buf_gt[:k_local]
        cm[start:end] = buf_gt[:k_local] < 0
        cph[start:end] = np.ones((k_local, n_dip), dtype=bool)
        return end

    for k_global, var in enumerate(ts.variants()):
        gen = var.genotypes
        if gen.max() > 1:
            # multiallelic / recurrent mutation: write -1 + mask, keep position
            buf_gt[k_local] = -1
            n_skipped += 1
        else:
            buf_gt[k_local] = gen.reshape(n_dip, 2)
        k_local += 1
        if k_local == B:
            n_written = flush(k_local, n_written)
            k_local = 0
            chunks_done = n_written // B
            if chunks_done % args.progress_every == 0:
                now = time.perf_counter()
                rate = n_written / max(now - t0, 1e-6)
                eta_min = (n_var - n_written) / max(rate, 1) / 60.0
                print(f"  {n_written:,}/{n_var:,} variants "
                      f"({100 * n_written / n_var:.1f}%) | "
                      f"{rate:.0f} var/s | ETA {eta_min:.1f} min "
                      f"| {n_skipped:,} multiallelic so far",
                      flush=True)
                last_log = now
    if k_local > 0:
        n_written = flush(k_local, n_written)

    # variant_allele uses the same pair for every site -- write once after the loop
    va[:] = np.broadcast_to(allele_pair, (n_var, 2))

    dt = time.perf_counter() - t0
    total = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    print(f"Done: wrote {n_written:,} variants "
          f"({n_skipped:,} multiallelic kept as gt=-1 + mask) "
          f"in {dt/60:.1f} min", flush=True)
    print(f"  store size: {total/1e9:.2f} GB ({out_path})", flush=True)

    # ---- companion pops.tsv: sample_id<TAB>population ----
    # The tree sequence stores per-sample-node populations; map nodes -> their
    # diploid index (the sample_id we wrote) and emit one row per diploid.
    pop_file = Path(args.pop_file) if args.pop_file else out_path.with_suffix(".pops.tsv")
    name_by_id = {}
    for pop in ts.populations():
        md = pop.metadata or {}
        name_by_id[pop.id] = md.get("name", f"pop{pop.id}")
    node_pop = ts.nodes_population
    samples = ts.samples()
    # For diploid layout via ts_to_vcz: sample_id "sim_i" corresponds to the
    # i-th pair of sample nodes (haps 2*i and 2*i+1). The two nodes of a pair
    # are always in the same population for stdpopsim, so we take the first.
    print(f"Writing companion population map to {pop_file}", flush=True)
    with open(pop_file, "w") as f:
        f.write("sample_id\tpopulation\n")
        for i in range(n_dip):
            pop_name = name_by_id[node_pop[samples[2 * i]]]
            f.write(f"sim_{i}\t{pop_name}\n")
    print(f"  {n_dip:,} samples written", flush=True)


if __name__ == "__main__":
    main()
