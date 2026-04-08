#!/usr/bin/env python
"""
Compare pg_gpu vs scikit-allel numerical accuracy on Ag1000G 3L.

Based on the validated comparison in debug/bench_3L_validate.py.

Produces:
  - tables/accuracy_vs_allel.csv
  - figures/accuracy_vs_allel.pdf
"""

import time
import numpy as np
import pandas as pd
import allel
import zarr
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import (
    HaplotypeMatrix, diversity, divergence, selection, sfs, admixture,
    windowed_analysis,
)
from pg_gpu.zarr_io import read_genotypes

OUT_DIR_FIG = "01_accuracy/figures"
OUT_DIR_TBL = "01_accuracy/tables"

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/ag1000g.unphased.3L.zarr"
REGION = "3L:1000000-5000000"
N_DIP = 100


def load():
    print(f"Loading {REGION}...", flush=True)
    t0 = time.time()
    hm = HaplotypeMatrix.from_zarr(ZARR_PATH, region=REGION)
    n_s = hm.num_haplotypes // 2

    # Correct diploid-to-haplotype mapping for unphased zarr layout
    def dip_to_hap(dip_indices):
        hap = []
        for i in dip_indices:
            hap.append(i)
            hap.append(i + n_s)
        return hap

    hm.sample_sets = {
        "pop1": dip_to_hap(range(0, N_DIP)),
        "pop2": dip_to_hap(range(N_DIP, 2 * N_DIP)),
    }

    data = read_genotypes(ZARR_PATH, region=REGION)
    gt, positions = data['gt'], data['positions']

    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()

    pct_missing = 100 * np.sum(hm.haplotypes < 0) / hm.haplotypes.size
    print(f"  {hm.num_haplotypes} haps x {hm.num_variants:,} variants "
          f"({pct_missing:.1f}% missing, {time.time()-t0:.0f}s)")
    return hm, gt, positions, n_s


def main():
    hm, gt, positions, n_samples = load()

    # Filter to biallelic sites so pg_gpu and allel operate on identical data
    print("Filtering to biallelic sites...", flush=True)
    hm = hm.apply_biallelic_filter()
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  {hm.num_variants:,} biallelic variants")

    pop1_dip = list(range(N_DIP))
    pop2_dip = list(range(N_DIP, 2 * N_DIP))

    # Build allel objects from the same biallelic sites
    print("Building allel objects...", flush=True)
    g_full = allel.GenotypeArray(gt)
    ac_full = g_full.count_alleles()
    is_bi = ac_full.is_biallelic_01()
    g = g_full.compress(is_bi, axis=0)
    positions = positions[is_bi]
    pos_allel = allel.SortedIndex(positions)
    ac1 = g.count_alleles(subpop=pop1_dip)
    ac2 = g.count_alleles(subpop=pop2_dip)
    del g_full, ac_full

    # Haplotype subsets for selection scans
    h_allel = g.to_haplotypes()
    pop1_hap = []
    for i in range(N_DIP):
        pop1_hap.extend([2*i, 2*i+1])
    h1_allel = h_allel.subset(sel1=pop1_hap)
    del h_allel

    span = positions[-1] - positions[0]
    rows = []

    def add_scalar(name, pg_val, al_val):
        pg_val, al_val = float(pg_val), float(al_val)
        if np.isnan(pg_val) and np.isnan(al_val):
            rows.append({"statistic": name, "pg_gpu": pg_val, "allel": al_val,
                         "abs_error": 0.0, "rel_error": 0.0, "match": "Y"})
            return
        abs_err = abs(pg_val - al_val)
        rel_err = abs_err / max(abs(al_val), 1e-15)
        match = "Y" if rel_err < 0.01 else "~" if rel_err < 0.15 else "N"
        rows.append({"statistic": name, "pg_gpu": pg_val, "allel": al_val,
                     "abs_error": abs_err, "rel_error": rel_err, "match": match})
        print(f"  {name:<30s} pg={pg_val:.6g}  al={al_val:.6g}  rel_err={rel_err:.2e}  {match}")

    def add_array(name, pg_arr, al_arr):
        mask = np.isfinite(pg_arr) & np.isfinite(al_arr) & (np.abs(al_arr) > 1e-12)
        n = int(np.sum(mask))
        if n == 0:
            rows.append({"statistic": name, "pg_gpu": np.nan, "allel": np.nan,
                         "abs_error": np.nan, "rel_error": np.nan, "match": "N"})
            return
        corr = float(np.corrcoef(pg_arr[mask], al_arr[mask])[0, 1]) if n > 1 else 1.0
        rel_err = float(np.median(np.abs(pg_arr[mask] - al_arr[mask]) / np.abs(al_arr[mask])))
        match = "Y" if corr > 0.999 else "~" if corr > 0.99 else "N"
        rows.append({"statistic": name, "pg_gpu": np.nan, "allel": np.nan,
                     "abs_error": np.nan, "rel_error": rel_err,
                     "match": match, "correlation": corr, "n_compared": n})
        print(f"  {name:<30s} corr={corr:.6f}  med_rel_err={rel_err:.2e}  n={n}  {match}")

    # --- Diversity ---
    print("\nDiversity:", flush=True)
    add_scalar("pi",
               diversity.pi(hm, population="pop1"),
               np.nansum(allel.mean_pairwise_difference(ac1)) / span)
    add_scalar("theta_w",
               diversity.theta_w(hm, population="pop1"),
               allel.watterson_theta(pos_allel, ac1))
    add_scalar("tajimas_d",
               diversity.tajimas_d(hm, population="pop1"),
               allel.tajima_d(ac1))

    # --- Divergence ---
    print("\nDivergence:", flush=True)
    num, den = allel.hudson_fst(ac1, ac2)
    add_scalar("fst_hudson",
               divergence.fst_hudson(hm, "pop1", "pop2"),
               np.nansum(num) / np.nansum(den))

    a, b, c = allel.weir_cockerham_fst(g, [pop1_dip, pop2_dip])
    add_scalar("fst_wc",
               divergence.fst_weir_cockerham(hm, "pop1", "pop2"),
               np.nansum(a) / np.nansum(a + b + c))

    add_scalar("dxy",
               divergence.dxy(hm, "pop1", "pop2"),
               allel.sequence_divergence(positions, ac1, ac2,
                                         start=int(positions[0]),
                                         stop=int(positions[-1])))

    # --- SFS ---
    print("\nSFS:", flush=True)
    pg_sfs = sfs.sfs(hm, population="pop1").astype(float)
    al_sfs = allel.sfs(ac1[:, 1]).astype(float)
    add_array("sfs", pg_sfs, al_sfs)

    pg_jsfs = sfs.joint_sfs(hm, pop1="pop1", pop2="pop2").flatten().astype(float)
    al_jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1]).flatten().astype(float)
    add_array("joint_sfs", pg_jsfs, al_jsfs)

    # --- Admixture (already biallelic) ---
    print("\nAdmixture:", flush=True)
    pg_f2 = admixture.patterson_f2(hm, "pop1", "pop2")
    al_f2 = allel.patterson_f2(ac1, ac2)
    add_scalar("patterson_f2 (mean)",
               np.nanmean(pg_f2),
               np.nanmean(al_f2))

    # --- Selection ---
    print("\nSelection:", flush=True)
    pg_h = selection.garud_h(hm, population="pop1")
    al_h = allel.garud_h(h1_allel)
    add_scalar("garud_h1", pg_h[0], al_h[0])
    add_scalar("garud_h12", pg_h[1], al_h[1])

    # --- Windowed ---
    print("\nWindowed:", flush=True)
    ws, we = int(positions[0]), int(positions[-1])
    pg_w = windowed_analysis(hm, window_size=50_000,
                             statistics=["pi", "theta_w", "tajimas_d"])
    ac_all = g.count_alleles()
    al_pi = allel.windowed_diversity(pos_allel, ac_all, size=50_000, start=ws, stop=we)[0]
    al_tw = allel.windowed_watterson_theta(pos_allel, ac_all, size=50_000, start=ws, stop=we)[0]
    add_array("windowed_pi", pg_w["pi"].values, al_pi)
    add_array("windowed_theta_w", pg_w["theta_w"].values, al_tw)

    pg_fst = windowed_analysis(hm, window_size=50_000,
                               statistics=["fst", "dxy"],
                               populations=["pop1", "pop2"])
    al_fst = allel.windowed_hudson_fst(pos_allel, ac1, ac2,
                                        size=50_000, start=ws, stop=we)[0]
    add_array("windowed_fst", pg_fst["fst"].values, al_fst)

    # --- Save ---
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR_TBL}/accuracy_vs_allel.csv", index=False)
    print(f"\nSaved to {OUT_DIR_TBL}/accuracy_vs_allel.csv")

    n_match = sum(1 for r in rows if r.get('match') == 'Y')
    n_close = sum(1 for r in rows if r.get('match') == '~')
    n_total = len(rows)
    print(f"Results: {n_match} exact, {n_close} close, "
          f"{n_total - n_match - n_close} mismatch out of {n_total}")

    # --- Figure ---
    scalar_rows = [r for r in rows if np.isfinite(r.get('rel_error', np.nan))
                   and r['rel_error'] > 0]
    if scalar_rows:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(scalar_rows) + 1.5))
        names = [r['statistic'] for r in scalar_rows]
        errs = [r['rel_error'] for r in scalar_rows]
        colors = ["#2ecc71" if r['match'] == 'Y' else "#f39c12" if r['match'] == '~'
                  else "#e74c3c" for r in scalar_rows]
        ax.barh(range(len(names)), errs, color=colors, edgecolor="0.3", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xscale('log')
        ax.set_xlabel('Relative error vs scikit-allel')
        ax.set_title(f'pg_gpu numerical accuracy\n(Ag1000G 3L, {hm.num_variants:,} variants, '
                     f'{hm.num_haplotypes} haplotypes)')
        ax.axvline(0.01, color='0.5', linestyle='--', linewidth=1, label='1% threshold')
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR_FIG}/accuracy_vs_allel.pdf", bbox_inches='tight')
        print(f"Figure saved to {OUT_DIR_FIG}/accuracy_vs_allel.pdf")


if __name__ == "__main__":
    main()
