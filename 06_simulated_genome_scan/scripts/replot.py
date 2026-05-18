#!/usr/bin/env python
"""
Replot the chr15 figures from on-disk artifacts (CSVs + .npy) without
rerunning the windowed scan. Falls back to recomputing the joint SFS
and the pairwise-r^2 heatmap from the VCZ store when those .npy files
are absent (cheap -- ~minutes for chr15 at 100k haps -- versus the
full scan).

Reads (from --tables-dir, default ``06_simulated_genome_scan/tables``):
    windowed_stats_{10kb,100kb,1mb}.csv
    garud_h_10kb.csv
    ld_decay.csv
    chromosome_summary.json
    joint_sfs.npy           (optional; recomputed if absent)
    r2_heatmap.npy          (optional)
    r2_heatmap_pos.npy      (optional)

Writes (to --figures-dir):
    genome_scan_ooa.{pdf,png}
    multiscale_ooa.{pdf,png}
    ld_ooa.{pdf,png}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# Reuse plotting + constants from the scan script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import genome_scan_ooa as gs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=gs.DEFAULT_DATA_DIR,
                   help=f"directory holding chr*.vcz (default {gs.DEFAULT_DATA_DIR})")
    p.add_argument("--zarr", default=None,
                   help="explicit path to a single VCZ store (only used "
                        "when recomputing missing .npy caches)")
    p.add_argument("--pop-file", default=None,
                   help="sample_id<TAB>population TSV "
                        "(default: <store>.pops.tsv)")
    p.add_argument("--tables-dir", default=str(gs.TABLES_DIR))
    p.add_argument("--figures-dir", default=str(gs.FIGURES_DIR))
    p.add_argument("--chromosome", default=None,
                   help="contig name; defaults to "
                        "chromosome_summary.json's value")
    p.add_argument("--ld-region", default=None,
                   help="region 'start-end' bp for the r^2 heatmap when "
                        "recomputing (default: chromosome_summary.json)")
    p.add_argument("--chunk-bp", type=int, default=5_000_000,
                   help="streaming chunk size for the recompute path")
    return p.parse_args()


def main():
    args = parse_args()
    tdir = Path(args.tables_dir)
    fdir = Path(args.figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((tdir / "chromosome_summary.json").read_text())
    chrom = str(args.chromosome) if args.chromosome else str(summary["chromosome"])
    chrom_len = int(summary["chromosome_length"])
    n_haps_per_pop = int(summary["haplotypes_per_pop"])
    n_garud_sub = int(summary["garud_subsample"])
    n_joint_target = int(summary.get("joint_sfs_target",
                                      summary.get("joint_sfs_subsample", 200)))
    n_ld_sub = int(summary["ld_subsample"])
    region = tuple(summary["ld_heatmap_region"])
    n_sites = int(summary["n_sites"])
    subtitle_extra = f"{n_sites:,} variants"

    windows_by_scale = {label: pd.read_csv(tdir / f"windowed_stats_{label}.csv")
                        for label, _ in gs.WINDOW_SCALES}
    main_df = windows_by_scale[gs.MAIN_SCALE]
    garud_df = pd.read_csv(tdir / "garud_h_10kb.csv")

    decay_df = pd.read_csv(tdir / "ld_decay.csv")
    ld_r2 = {p: (decay_df.loc[decay_df["pop"] == p, "bin_mid_bp"].to_numpy(),
                 decay_df.loc[decay_df["pop"] == p, "mean_r2"].to_numpy())
             for p in gs.POPS}

    x_lo_mb = gs.SCAN_X_LO_MB

    joint_npy = tdir / "joint_sfs.npy"
    r2_npy = tdir / "r2_heatmap.npy"
    r2_pos_npy = tdir / "r2_heatmap_pos.npy"

    if joint_npy.exists() and r2_npy.exists() and r2_pos_npy.exists():
        print(f"Loading cached joint SFS / r^2 heatmap from {tdir}")
        joint = np.load(joint_npy)
        r2_mat = np.load(r2_npy)
        hm_pos = np.load(r2_pos_npy)
        n_hm_haps = min(gs.LD_HEATMAP_SUBSAMPLE, n_haps_per_pop)
    else:
        # Recompute paths reuse the same streaming entry points as the
        # main scan: open as streaming, materialize the subsample (for
        # the joint SFS) or the region (for the heatmap).
        from pg_gpu import HaplotypeMatrix, sfs
        zarr_path = (Path(args.zarr) if args.zarr
                     else gs.pick_zarr(args.data_dir, chrom))
        pop_file = Path(args.pop_file) if args.pop_file else (
            zarr_path.parent / f"{zarr_path.stem}.pops.tsv")
        print(f"Recomputing missing artifacts from {zarr_path} ...")
        stream = HaplotypeMatrix.from_zarr(
            str(zarr_path), streaming="always", chunk_bp=args.chunk_bp,
            pop_file=str(pop_file),
        )
        if args.ld_region:
            a, b = args.ld_region.split("-")
            region = (int(a), int(b))

        if not joint_npy.exists():
            t0 = time.perf_counter()
            joint = np.asarray(sfs.project_joint_sfs(
                stream, pop1=gs.POPS[0], pop2=gs.POPS[1],
                target_n1=n_joint_target, target_n2=n_joint_target))
            print(f"  joint SFS recompute (projected to "
                  f"{n_joint_target} haps/pop): "
                  f"{time.perf_counter()-t0:.1f}s")
            np.save(joint_npy, joint)
        else:
            joint = np.load(joint_npy)

        if not r2_npy.exists() or not r2_pos_npy.exists():
            t0 = time.perf_counter()
            r2_mat, hm_pos, n_hm_haps = gs.run_ld_heatmap(
                stream, gs.POPS, region, gs.LD_HEATMAP_SUBSAMPLE,
                gs.LD_HEATMAP_MIN_MAF, gs.LD_HEATMAP_MAX_SNPS,
            )
            print(f"  r^2 heatmap recompute: {time.perf_counter()-t0:.1f}s")
            np.save(r2_npy, r2_mat)
            np.save(r2_pos_npy, hm_pos)
        else:
            r2_mat = np.load(r2_npy)
            hm_pos = np.load(r2_pos_npy)
            n_hm_haps = min(gs.LD_HEATMAP_SUBSAMPLE, n_haps_per_pop)

    print("Plotting ...")
    gs.plot_composite(main_df, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                      chrom, x_lo_mb, chrom_len, n_haps_per_pop, gs.MAIN_SCALE,
                      region, n_garud_sub, n_joint_target, n_ld_sub, subtitle_extra,
                      str(fdir / "genome_scan_ooa"))
    gs.plot_multiscale(windows_by_scale, chrom, x_lo_mb, chrom_len,
                       str(fdir / "multiscale_ooa"))
    gs.plot_ld(ld_r2, r2_mat, hm_pos, n_hm_haps, chrom, region, n_ld_sub,
               str(fdir / "ld_ooa"))


if __name__ == "__main__":
    main()
