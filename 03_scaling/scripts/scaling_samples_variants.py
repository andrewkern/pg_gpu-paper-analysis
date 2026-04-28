#!/usr/bin/env python
"""
Runtime scaling with sample size and variant count.

Sample-size scaling uses synthetic random haplotypes (fixed n_variants=100K)
to measure GPU throughput across 100 to 100K haplotypes.

Variant-count scaling uses msprime simulations with increasing sequence
length (fixed n_haplotypes=200) to get realistic LD structure.

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

from pg_gpu import HaplotypeMatrix, diversity, divergence, selection, ld_statistics, windowed_analysis

OUT_DIR_FIG = "03_scaling/figures"
OUT_DIR_TBL = "03_scaling/tables"


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
    """Fix variant count at 100K, vary sample size using synthetic data."""
    n_var = 100_000
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000]
    rows = []

    rng = np.random.default_rng(42)
    pos = np.arange(n_var, dtype=np.int32) * 100

    for n_hap in sample_sizes:
        mem_gb = n_hap * n_var / 1e9
        print(f"  n_hap={n_hap:>7,} ({mem_gb:.1f} GB)...", end='', flush=True)

        try:
            hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
            hm = HaplotypeMatrix(hap, pos.copy(), 0, n_var * 100)
            n_half = n_hap // 2
            hm.sample_sets = {
                "pop1": list(range(n_half)),
                "pop2": list(range(n_half, n_hap)),
            }
            del hap
            hm.transfer_to_gpu()
            cp.cuda.Stream.null.synchronize()

            # Use a smaller variant subset for ZnS (O(m^2))
            hm_ld = hm.get_subset(np.arange(min(5000, n_var)))

            stats = {
                'pi': lambda: diversity.pi(hm, population="pop1"),
                'tajimas_d': lambda: diversity.tajimas_d(hm, population="pop1"),
                'fst_hudson': lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
                'garud_h': lambda: selection.garud_h(hm),
                'zns': lambda: ld_statistics.zns(hm_ld),
            }

            stats['windowed_3'] = lambda: windowed_analysis(
                hm, window_size=500_000,
                statistics=['pi', 'theta_w', 'tajimas_d'])

            for stat_name, fn in stats.items():
                try:
                    t = bench(fn)
                    rows.append({
                        'n_haplotypes': n_hap,
                        'n_variants': n_var,
                        'statistic': stat_name,
                        'time_s': t,
                    })
                except Exception as e:
                    print(f" {stat_name}=OOM", end='', flush=True)
                    cp.get_default_memory_pool().free_all_blocks()

            timings = " ".join(f"{r['statistic']}={r['time_s']:.3f}s"
                               for r in rows if r['n_haplotypes'] == n_hap)
            print(f" {timings}", flush=True)

            del hm, hm_ld
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            cp.get_default_memory_pool().free_all_blocks()

    return pd.DataFrame(rows)


def scaling_by_variants():
    """Fix sample size at n=200, vary variant count via msprime sequence length."""
    n_hap = 200
    # Longer sequences = more variants under neutral model
    lengths = [100_000, 500_000, 2_000_000, 10_000_000, 50_000_000, 100_000_000]
    rows = []

    for L in lengths:
        print(f"  L={L:>12,}...", end='', flush=True)
        ts = msprime.sim_ancestry(
            samples=n_hap // 2, sequence_length=L,
            recombination_rate=1e-8, population_size=10_000,
            random_seed=42, ploidy=2)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
        hm = HaplotypeMatrix.from_ts(ts)
        n_half = n_hap // 2
        hm.sample_sets = {
            "pop1": list(range(n_half)),
            "pop2": list(range(n_half, n_hap)),
        }
        hm.transfer_to_gpu()
        cp.cuda.Stream.null.synchronize()

        n_var = hm.num_variants
        print(f" {n_var:>8,} variants", end='', flush=True)

        # ZnS subset (O(m^2), cap at 5K variants)
        hm_ld = hm.get_subset(np.arange(min(5000, hm.num_variants)))

        stats = {
            'pi': lambda: diversity.pi(hm, population="pop1"),
            'tajimas_d': lambda: diversity.tajimas_d(hm, population="pop1"),
            'fst_hudson': lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
            'garud_h': lambda: selection.garud_h(hm),
            'zns': lambda: ld_statistics.zns(hm_ld),
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

        timings = " ".join(f"{r['statistic']}={r['time_s']:.4f}s"
                           for r in rows if r['n_haplotypes'] == n_hap
                           and r['n_variants'] == n_var)
        print(f"  {timings}", flush=True)

        del hm
        cp.get_default_memory_pool().free_all_blocks()

    return pd.DataFrame(rows)


_MARKERS = {'pi': 'o', 'tajimas_d': 's', 'fst_hudson': '^', 'windowed_3': 'D',
             'garud_h': 'v', 'zns': 'P'}


def _draw_panel(ax, df, x_col, x_label, title, show_legend):
    for stat in df['statistic'].unique():
        sub = df[df['statistic'] == stat].sort_values(x_col)
        m = _MARKERS.get(stat, 'o')
        ax.plot(sub[x_col], sub['time_s'], f'-{m}', label=stat, markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=9)


def make_combined_figure(df_samples, df_variants, outpath):
    """Two-panel scaling figure: left=samples, right=variants."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(12, 4.5),
                                       sharey=True)
    _draw_panel(ax_s, df_samples, 'n_haplotypes',
                'Number of haplotypes',
                'Scaling with sample size (100K variants)',
                show_legend=False)
    _draw_panel(ax_v, df_variants, 'n_variants',
                'Number of variants',
                'Scaling with variant count (200 haplotypes)',
                show_legend=True)
    ax_s.set_ylabel('Wall-clock time (seconds)')
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Figure saved to {outpath}")


def main():
    import os
    samples_csv = f"{OUT_DIR_TBL}/scaling_samples.csv"
    variants_csv = f"{OUT_DIR_TBL}/scaling_variants.csv"

    if os.path.exists(samples_csv) and os.path.exists(variants_csv):
        print("Found existing CSVs; regenerating figure only.")
        df_samples = pd.read_csv(samples_csv)
        df_variants = pd.read_csv(variants_csv)
    else:
        print("Scaling by sample size (100K variants, synthetic data):")
        df_samples = scaling_by_samples()
        df_samples.to_csv(samples_csv, index=False)

        print("\nScaling by variant count (200 haplotypes, msprime):")
        df_variants = scaling_by_variants()
        df_variants.to_csv(variants_csv, index=False)

    make_combined_figure(df_samples, df_variants,
                         f"{OUT_DIR_FIG}/scaling_combined.pdf")


if __name__ == "__main__":
    main()
