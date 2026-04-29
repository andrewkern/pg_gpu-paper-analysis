#!/usr/bin/env python
"""
Performance benchmark: pg_gpu vs scikit-allel on Ag1000G 3R (full arm).

Wraps the existing stress_test_ag1000g.py and reformats output as
publication-ready table and figure.

Produces:
  - tables/benchmark_3R.csv
  - figures/benchmark_speedups.pdf
  - figures/benchmark_walltimes.pdf
"""

import os
import subprocess
import sys
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR_FIG = "02_performance/figures"
OUT_DIR_TBL = "02_performance/tables"
OUT_DIR_CACHE = "02_performance/cache"

STRESS_TEST = "/home/adkern/pg_gpu/debug/stress_test_ag1000g.py"

# LD subset benchmark configuration
ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
CHROM = "3R"
N_LD_SNPS = 10_000
N_DIP_PER_POP = 100  # mirrors stress_test_ag1000g.py
LD_CACHE = f"{OUT_DIR_CACHE}/ld_bench_3R.csv"


def bench_pairwise_ld():
    """Benchmark pg_gpu vs scikit-allel all-pairwise rogers_huff_r on
    the first N_LD_SNPS contiguous SNPs of CHROM, restricted to the
    first N_DIP_PER_POP diploids (mirrors the stress test's per-pop
    sizing). The full 3R x 2940-hap pairwise r matrix is intractable
    for scikit-allel, so this restricted setup gives a meaningful
    apples-to-apples timing point.

    Returns a row dict matching the parse_results schema.
    """
    import zarr
    import allel
    import cupy as cp
    from pg_gpu import HaplotypeMatrix
    from pg_gpu.ld_statistics import rogers_huff_r

    print(f"Loading first {N_LD_SNPS:,} SNPs of {CHROM} for LD bench...",
          flush=True)
    t0 = time.time()
    store = zarr.open(ZARR_PATH, mode='r')
    chrom_grp = store[CHROM]
    positions = np.array(chrom_grp['variants/POS'][:N_LD_SNPS])
    gt = np.array(chrom_grp['calldata/GT'][:N_LD_SNPS, :N_DIP_PER_POP, :])
    n_var, n_dip, ploidy = gt.shape
    assert ploidy == 2

    haplotypes = np.empty((n_var, 2 * n_dip), dtype=gt.dtype)
    haplotypes[:, :n_dip] = gt[:, :, 0]
    haplotypes[:, n_dip:] = gt[:, :, 1]
    haplotypes = haplotypes.T  # (n_hap, n_var)
    print(f"  {haplotypes.shape[0]} haplotypes x {haplotypes.shape[1]:,} "
          f"variants ({time.time()-t0:.0f}s)", flush=True)

    hm = HaplotypeMatrix(haplotypes, positions,
                          int(positions[0]), int(positions[-1]))
    hm.transfer_to_gpu()

    # scikit-allel format: (n_var, n_dip) int8 dosages in {0,1,2}
    gn = (haplotypes[0::2] + haplotypes[1::2]).T.astype(np.int8)

    print("Timing pg_gpu rogers_huff_r (1 warmup + 3 timed)...", flush=True)
    rogers_huff_r(hm); cp.cuda.Stream.null.synchronize()
    t_pg = []
    for _ in range(3):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        rogers_huff_r(hm)
        cp.cuda.Stream.null.synchronize()
        t_pg.append(time.perf_counter() - t)
    t_pg_med = float(np.median(t_pg))

    print("Timing scikit-allel rogers_huff_r (1 warmup + 3 timed)...",
          flush=True)
    allel.rogers_huff_r(gn)
    t_al = []
    for _ in range(3):
        t = time.perf_counter()
        allel.rogers_huff_r(gn)
        t_al.append(time.perf_counter() - t)
    t_al_med = float(np.median(t_al))

    speedup = t_al_med / t_pg_med if t_pg_med > 0 else float('nan')
    print(f"  pg_gpu={t_pg_med:.4f}s  allel={t_al_med:.4f}s  "
          f"speedup={speedup:.1f}x", flush=True)
    return {
        "statistic": f"ld.rogers_huff_r ({N_LD_SNPS//1000}k SNPs)",
        "pg_gpu_s": t_pg_med,
        "allel_s": t_al_med,
        "speedup": speedup,
    }


def parse_results(text):
    """Parse the stress test output into a DataFrame."""
    rows = []
    for line in text.split('\n'):
        # Match lines like: diversity.pi   0.142s   28.493s    201.2x
        m = re.match(r'^(\S+)\s+([\d.]+s|FAIL)\s+([\d.]+s|---)\s+([\d.]+x|---)\s*$', line.strip())
        if m:
            name = m.group(1)
            pg = float(m.group(2).rstrip('s')) if m.group(2) != 'FAIL' else np.nan
            al = float(m.group(3).rstrip('s')) if m.group(3) != '---' else np.nan
            sp = float(m.group(4).rstrip('x')) if m.group(4) != '---' else np.nan
            rows.append({"statistic": name, "pg_gpu_s": pg, "allel_s": al, "speedup": sp})
    return pd.DataFrame(rows)


def make_speedup_figure(df, outpath):
    """Horizontal bar chart of speedups."""
    compared = df.dropna(subset=['speedup']).sort_values('speedup')
    if len(compared) == 0:
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(compared) + 1.5))

    colors = ["#2ecc71" if s >= 10 else "#f39c12" if s >= 1 else "#e74c3c"
              for s in compared['speedup']]
    bars = ax.barh(range(len(compared)), compared['speedup'].values,
                   color=colors, edgecolor="0.3", linewidth=0.5)
    ax.set_yticks(range(len(compared)))
    ax.set_yticklabels(compared['statistic'].values, fontsize=9)
    ax.set_xscale('log')
    ax.axvline(1, color='0.4', linestyle='--', linewidth=1)
    ax.set_xlabel('Speedup (pg_gpu / scikit-allel)')
    ax.set_title('pg_gpu performance on Ag1000G 3R\n'
                 '(2940 haplotypes, 10.9M variants)')

    for bar, sp in zip(bars, compared['speedup'].values):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{sp:.0f}x", va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Figure saved to {outpath}")


def make_walltime_figure(df, outpath):
    """Grouped bar chart of absolute wall-clock times."""
    compared = df.dropna(subset=['speedup']).sort_values('speedup', ascending=False)
    if len(compared) == 0:
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(compared))
    w = 0.35

    ax.bar(x - w/2, compared['pg_gpu_s'], w, label='pg_gpu', color='#2ecc71',
           edgecolor='0.3', linewidth=0.5)
    ax.bar(x + w/2, compared['allel_s'], w, label='scikit-allel', color='#e74c3c',
           edgecolor='0.3', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(compared['statistic'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_title('Absolute wall-clock time on Ag1000G 3R')

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Figure saved to {outpath}")


def main():
    # Check for cached results
    cached = "/home/adkern/pg_gpu/debug/stress_test_3R_results.txt"
    try:
        with open(cached) as f:
            text = f.read()
        print(f"Using cached results from {cached}")
    except FileNotFoundError:
        print("Running stress test (this takes a few hours on full Ag1000G 3R)...")
        result = subprocess.run(
            ["pixi", "run", "python", STRESS_TEST],
            capture_output=True, text=True, cwd="/home/adkern/pg_gpu")
        text = result.stdout
        if result.returncode != 0:
            print(f"Warning: stress test exited with code {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "")
        else:
            with open(cached, 'w') as f:
                f.write(text)
            print(f"Stress test output cached at {cached}")

    df = parse_results(text)

    # LD subset benchmark (cached separately; pre-existing stress test
    # doesn't include all-pairwise LD because the full 3R x 2940-hap
    # matrix is intractable for scikit-allel).
    os.makedirs(OUT_DIR_CACHE, exist_ok=True)
    if os.path.exists(LD_CACHE):
        ld_df = pd.read_csv(LD_CACHE)
        print(f"\nUsing cached LD bench from {LD_CACHE}")
    else:
        print("\nRunning LD subset bench (not in stress test cache)...")
        ld_row = bench_pairwise_ld()
        ld_df = pd.DataFrame([ld_row])
        ld_df.to_csv(LD_CACHE, index=False)
        print(f"LD bench cached at {LD_CACHE}")
    df = pd.concat([df, ld_df], ignore_index=True)

    df.to_csv(f"{OUT_DIR_TBL}/benchmark_3R.csv", index=False)
    print(f"\n{len(df)} statistics benchmarked")
    print(df.to_string(index=False))

    compared = df.dropna(subset=['speedup'])
    if len(compared) > 0:
        print(f"\nMedian speedup: {compared['speedup'].median():.1f}x")
        print(f"Range: {compared['speedup'].min():.1f}x - {compared['speedup'].max():.1f}x")

    make_speedup_figure(df, f"{OUT_DIR_FIG}/benchmark_speedups.pdf")
    make_walltime_figure(df, f"{OUT_DIR_FIG}/benchmark_walltimes.pdf")


if __name__ == "__main__":
    main()
