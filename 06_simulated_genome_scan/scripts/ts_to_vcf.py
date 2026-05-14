#!/usr/bin/env python
"""
Convert a tskit tree sequence to a bgzip-compressed VCF.

For the OOA_2T12 chr15 example at 100,000 haplotypes/population (200,000 sample
nodes, 11.6 M sites) the uncompressed VCF is several TB and the bgzipped output
is hundreds of GB, so this is genuinely a long-running job. Output goes next to
the .trees file (which is gitignored).

Usage:
    python 06_simulated_genome_scan/scripts/ts_to_vcf.py [--trees PATH] [--out PATH]
                                                        [--threads N]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import tskit


def parse_args():
    here = Path(__file__).resolve().parents[1]
    default_trees = here / "data" / "ooa_2t12" / "chr15.trees"
    default_out = default_trees.with_suffix(".vcf.gz")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees", default=str(default_trees),
                   help=f"input .trees file (default {default_trees})")
    p.add_argument("--out", default=str(default_out),
                   help=f"output .vcf.gz path (default <trees>.vcf.gz)")
    p.add_argument("--contig-id", default=None,
                   help="VCF CHROM field (default: parse from input filename, e.g. 'chr15.trees' -> '15')")
    p.add_argument("--threads", type=int, default=8,
                   help="bgzip worker threads (default 8)")
    return p.parse_args()


def main():
    args = parse_args()
    trees_path = Path(args.trees)
    out_path = Path(args.out)
    if not trees_path.exists():
        raise SystemExit(f"input not found: {trees_path}")

    contig_id = args.contig_id
    if contig_id is None:
        stem = trees_path.stem  # 'chr15'
        contig_id = stem[3:] if stem.startswith("chr") else stem

    print(f"Loading {trees_path} ...", flush=True)
    ts = tskit.load(str(trees_path))
    print(f"  {ts.num_sites:,} sites, {ts.num_samples:,} sample nodes, "
          f"sequence_length {int(ts.sequence_length):,} bp", flush=True)
    print(f"Writing {out_path} via bgzip -@{args.threads} ...", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # bgzip in text mode through Popen; ts.write_vcf writes plain text.
    with open(out_path, "wb") as fout:
        with subprocess.Popen(
            ["bgzip", "-c", "-@", str(args.threads)],
            stdin=subprocess.PIPE, stdout=fout, text=True,
        ) as bgz:
            ts.write_vcf(bgz.stdin, contig_id=contig_id)
            bgz.stdin.close()
            rc = bgz.wait()
    if rc != 0:
        raise SystemExit(f"bgzip exited {rc}")
    dt = time.perf_counter() - t0
    size_gb = out_path.stat().st_size / 1e9
    print(f"Done: {out_path} ({size_gb:.2f} GB) in {dt/3600:.2f} h "
          f"({dt:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
