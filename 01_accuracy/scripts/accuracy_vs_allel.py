#!/usr/bin/env python
"""
Compare pg_gpu vs scikit-allel numerical accuracy on Ag1000G 3L.

Produces:
  - tables/accuracy_vs_allel.csv
  - figures/accuracy_vs_allel.pdf
"""

import time
import numpy as np
import pandas as pd
import allel
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import (
    HaplotypeMatrix, diversity, divergence, selection, sfs, admixture,
)
from pg_gpu._memutil import dac_and_n
import cupy as cp

OUT_DIR_FIG = "01_accuracy/figures"
OUT_DIR_TBL = "01_accuracy/tables"

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/ag1000g.unphased.3L.zarr"
REGION = "3L:1000000-5000000"
N_DIP = 100  # diploid individuals per population


def load():
    print(f"Loading {REGION}...", flush=True)
    t0 = time.time()
    hm = HaplotypeMatrix.from_zarr(ZARR_PATH, region=REGION)
    n_hap = 2 * N_DIP
    hm.sample_sets = {
        "pop1": list(range(0, n_hap)),
        "pop2": list(range(n_hap, 2 * n_hap)),
    }
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  {hm.num_haplotypes} haps x {hm.num_variants:,} variants ({time.time()-t0:.0f}s)")
    return hm


def build_allel(hm):
    """Build allel objects for comparison."""
    hap_cpu = hm._haplotypes if hasattr(hm, '_haplotypes') else hm.haplotypes
    if hasattr(hap_cpu, 'get'):
        hap_cpu = hap_cpu.get()
    n_hap, n_var = hap_cpu.shape
    gt = np.stack([hap_cpu[0::2], hap_cpu[1::2]], axis=-1).transpose(1, 0, 2)
    g = allel.GenotypeArray(gt)
    pos = allel.SortedIndex(hm.positions if isinstance(hm.positions, np.ndarray)
                            else hm.positions.get())
    pop1_dip = list(range(N_DIP))
    pop2_dip = list(range(N_DIP, 2 * N_DIP))
    ac = g.count_alleles()
    ac1 = g.count_alleles(subpop=pop1_dip)
    ac2 = g.count_alleles(subpop=pop2_dip)

    # Biallelic filter for admixture stats (allel requires exactly 2 allele columns)
    is_biallelic = (ac.max_allele() == 1) & (ac.shape[1] >= 2)
    ac1_bi = ac1.compress(is_biallelic, axis=0)[:, :2]
    ac2_bi = ac2.compress(is_biallelic, axis=0)[:, :2]

    return g, pos, ac, ac1, ac2, ac1_bi, ac2_bi, pop1_dip, pop2_dip, is_biallelic


def compare_scalar(name, pg_val, al_val):
    """Compare two scalar values."""
    pg_val, al_val = float(pg_val), float(al_val)
    if np.isnan(pg_val) and np.isnan(al_val):
        return {"statistic": name, "pg_gpu": pg_val, "allel": al_val,
                "abs_error": 0.0, "rel_error": 0.0, "match": True}
    abs_err = abs(pg_val - al_val)
    rel_err = abs_err / max(abs(al_val), 1e-15)
    return {"statistic": name, "pg_gpu": pg_val, "allel": al_val,
            "abs_error": abs_err, "rel_error": rel_err,
            "match": rel_err < 1e-6}


def compare_array(name, pg_arr, al_arr):
    """Compare two arrays via correlation and max relative error."""
    mask = np.isfinite(pg_arr) & np.isfinite(al_arr)
    if mask.sum() == 0:
        return {"statistic": name, "n_compared": 0, "correlation": np.nan,
                "max_rel_error": np.nan, "match": False}
    p, a = pg_arr[mask], al_arr[mask]
    corr = np.corrcoef(p, a)[0, 1] if len(p) > 1 else 1.0
    denom = np.maximum(np.abs(a), 1e-15)
    max_rel = float(np.max(np.abs(p - a) / denom))
    return {"statistic": name, "n_compared": int(mask.sum()),
            "correlation": corr, "max_rel_error": max_rel,
            "match": corr > 0.9999 or max_rel < 1e-4}


def main():
    hm = load()
    g, pos, ac, ac1, ac2, ac1_bi, ac2_bi, pop1_dip, pop2_dip, is_bi = build_allel(hm)

    results = []

    # --- Scalar diversity ---
    span = hm.get_span()

    pg = diversity.pi(hm, population="pop1")
    al = float(np.sum(allel.mean_pairwise_difference(ac1)) / span)
    results.append(compare_scalar("pi", pg, al))

    pg = diversity.theta_w(hm, population="pop1")
    al = float(allel.watterson_theta(pos, ac1) / span)
    results.append(compare_scalar("theta_w", pg, al))

    pg = diversity.tajimas_d(hm, population="pop1")
    al = allel.tajima_d(ac1)
    results.append(compare_scalar("tajimas_d", pg, al))

    # --- Divergence ---
    pg = divergence.fst_hudson(hm, "pop1", "pop2")
    al_num, al_den = allel.hudson_fst(ac1, ac2)
    al = float(np.nansum(al_num) / np.nansum(al_den))
    results.append(compare_scalar("fst_hudson", pg, al))

    pg = divergence.fst_weir_cockerham(hm, "pop1", "pop2")
    al_a, al_b, al_c = allel.weir_cockerham_fst(g, [pop1_dip, pop2_dip])
    al = float(np.nansum(al_a) / (np.nansum(al_a) + np.nansum(al_b) + np.nansum(al_c)))
    results.append(compare_scalar("fst_wc", pg, al))

    pg = divergence.dxy(hm, "pop1", "pop2")
    al = float(allel.sequence_divergence(pos, ac1, ac2))
    results.append(compare_scalar("dxy", pg, al))

    # --- SFS ---
    pg = sfs.sfs(hm, population="pop1")
    al = allel.sfs(ac1[:, 1])
    # Pad to same length
    n = max(len(pg), len(al))
    pg_padded = np.zeros(n); pg_padded[:len(pg)] = pg
    al_padded = np.zeros(n); al_padded[:len(al)] = al
    results.append(compare_array("sfs", pg_padded, al_padded))

    pg = sfs.joint_sfs(hm, pop1="pop1", pop2="pop2")
    al_j = allel.joint_sfs(ac1[:, 1], ac2[:, 1])
    results.append(compare_array("joint_sfs", pg.ravel(), al_j.ravel()))

    # --- Admixture (biallelic only for allel) ---
    # pg_gpu returns per-variant F2 for all sites; allel only biallelic
    pg_f2 = admixture.patterson_f2(hm, "pop1", "pop2")
    al_f2 = allel.patterson_f2(ac1_bi, ac2_bi)
    # Compare on biallelic sites only
    pg_f2_bi = pg_f2[is_bi]
    results.append(compare_array("patterson_f2", pg_f2_bi, al_f2))

    # --- Selection (per-variant arrays) ---
    pg_h1, pg_h12, pg_h123, pg_h2h1 = selection.garud_h(hm, population="pop1")
    h1_hap = hm.haplotypes.get()[:2*N_DIP, :]
    al_h = allel.garud_h(allel.HaplotypeArray(h1_hap.T))
    results.append(compare_scalar("garud_h1", pg_h1, al_h[0]))
    results.append(compare_scalar("garud_h12", pg_h12, al_h[1]))

    # --- Build results table ---
    df = pd.DataFrame(results)
    df.to_csv(f"{OUT_DIR_TBL}/accuracy_vs_allel.csv", index=False)
    print(f"\nResults saved to {OUT_DIR_TBL}/accuracy_vs_allel.csv")
    print(df.to_string(index=False))

    # --- Figure ---
    scalar = df[df['rel_error'].notna()].copy()
    if len(scalar) > 0:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2ecc71" if m else "#e74c3c" for m in scalar['match']]
        bars = ax.barh(range(len(scalar)), scalar['rel_error'].values,
                       color=colors, edgecolor="0.3", linewidth=0.5)
        ax.set_yticks(range(len(scalar)))
        ax.set_yticklabels(scalar['statistic'].values)
        ax.set_xscale('log')
        ax.set_xlabel('Relative error vs scikit-allel')
        ax.set_title('pg_gpu numerical accuracy (Ag1000G 3L)')
        ax.axvline(1e-6, color='0.5', linestyle='--', linewidth=1, label='1e-6 threshold')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR_FIG}/accuracy_vs_allel.pdf", bbox_inches='tight')
        print(f"Figure saved to {OUT_DIR_FIG}/accuracy_vs_allel.pdf")


if __name__ == "__main__":
    main()
