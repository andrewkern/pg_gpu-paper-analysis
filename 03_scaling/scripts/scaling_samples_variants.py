#!/usr/bin/env python
"""
Runtime scaling with sample size and variant count.

Produces:
  - tables/scaling_samples.csv
  - tables/scaling_variants.csv
  - figures/scaling_samples.pdf
  - figures/scaling_variants.pdf
"""

import time
import numpy as np
import pandas as pd
import msprime
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import HaplotypeMatrix, diversity, divergence, windowed_analysis

OUT_DIR_FIG = "03_scaling/figures"
OUT_DIR_TBL = "03_scaling/tables"

N_POP = 10_000
MU = 1e-8
RECOMB = 1e-8


def bench(fn, n_warmup=1, n_iter=3):
    """Time a function, return median wall-clock."""
    for _ in range(n_warmup):
        fn()
        cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_iter):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def scaling_by_samples():
    """Fix variant count ~1M, vary sample size."""
    L = 5_000_000
    sample_sizes = [50, 100, 200, 500, 1000, 1500, 2000]
    rows = []

    for n_hap in sample_sizes:
        print(f"  n_hap={n_hap}...", end='', flush=True)
        ts = msprime.sim_ancestry(
            samples=n_hap // 2, sequence_length=L,
            recombination_rate=RECOMB, population_size=N_POP,
            random_seed=42, ploidy=2)
        ts = msprime.sim_mutations(ts, rate=MU, random_seed=42)
        hm = HaplotypeMatrix.from_ts(ts)
        n_half = n_hap // 2
        hm.sample_sets = {
            "pop1": list(range(n_half)),
            "pop2": list(range(n_half, n_hap)),
        }
        hm.transfer_to_gpu()
        cp.cuda.Stream.null.synchronize()

        n_var = hm.num_variants
        print(f" {n_var:,} variants", flush=True)

        stats = {
            'pi': lambda: diversity.pi(hm, population="pop1"),
            'tajimas_d': lambda: diversity.tajimas_d(hm, population="pop1"),
            'fst_hudson': lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
            'windowed_3': lambda: windowed_analysis(
                hm, window_size=50_000,
                statistics=['pi', 'theta_w', 'tajimas_d']),
        }

        for stat_name, fn in stats.items():
            t = bench(fn)
            rows.append({
                'n_haplotypes': n_hap,
                'n_variants': n_var,
                'statistic': stat_name,
                'time_s': t,
            })
            print(f"    {stat_name}: {t:.4f}s", flush=True)

    return pd.DataFrame(rows)


def scaling_by_variants():
    """Fix sample size n=200, vary variant count via sequence length."""
    n_hap = 200
    lengths = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]
    rows = []

    for L in lengths:
        print(f"  L={L:,}...", end='', flush=True)
        ts = msprime.sim_ancestry(
            samples=n_hap // 2, sequence_length=L,
            recombination_rate=RECOMB, population_size=N_POP,
            random_seed=42, ploidy=2)
        ts = msprime.sim_mutations(ts, rate=MU, random_seed=42)
        hm = HaplotypeMatrix.from_ts(ts)
        n_half = n_hap // 2
        hm.sample_sets = {
            "pop1": list(range(n_half)),
            "pop2": list(range(n_half, n_hap)),
        }
        hm.transfer_to_gpu()
        cp.cuda.Stream.null.synchronize()

        n_var = hm.num_variants
        print(f" {n_var:,} variants", flush=True)

        stats = {
            'pi': lambda: diversity.pi(hm, population="pop1"),
            'tajimas_d': lambda: diversity.tajimas_d(hm, population="pop1"),
            'fst_hudson': lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
            'windowed_3': lambda: windowed_analysis(
                hm, window_size=50_000,
                statistics=['pi', 'theta_w', 'tajimas_d']),
        }

        for stat_name, fn in stats.items():
            t = bench(fn)
            rows.append({
                'n_haplotypes': n_hap,
                'n_variants': n_var,
                'statistic': stat_name,
                'time_s': t,
            })
            print(f"    {stat_name}: {t:.4f}s", flush=True)

    return pd.DataFrame(rows)


def make_figure(df, x_col, x_label, outpath, title):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6, 4))

    for stat in df['statistic'].unique():
        sub = df[df['statistic'] == stat]
        ax.plot(sub[x_col], sub['time_s'], 'o-', label=stat, markersize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Figure saved to {outpath}")


def main():
    print("Scaling by sample size:")
    df_samples = scaling_by_samples()
    df_samples.to_csv(f"{OUT_DIR_TBL}/scaling_samples.csv", index=False)

    print("\nScaling by variant count:")
    df_variants = scaling_by_variants()
    df_variants.to_csv(f"{OUT_DIR_TBL}/scaling_variants.csv", index=False)

    make_figure(df_samples, 'n_haplotypes', 'Number of haplotypes',
                f"{OUT_DIR_FIG}/scaling_samples.pdf",
                'Runtime scaling with sample size')
    make_figure(df_variants, 'n_variants', 'Number of variants',
                f"{OUT_DIR_FIG}/scaling_variants.pdf",
                'Runtime scaling with variant count')


if __name__ == "__main__":
    main()
