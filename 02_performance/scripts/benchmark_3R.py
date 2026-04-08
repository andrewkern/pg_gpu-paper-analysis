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

import subprocess
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR_FIG = "02_performance/figures"
OUT_DIR_TBL = "02_performance/tables"

STRESS_TEST = "/home/adkern/pg_gpu/debug/stress_test_ag1000g.py"


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
        print("Running stress test (this takes ~30 minutes)...")
        result = subprocess.run(
            ["pixi", "run", "python", STRESS_TEST],
            capture_output=True, text=True, cwd="/home/adkern/pg_gpu")
        text = result.stdout
        if result.returncode != 0:
            print(f"Warning: stress test exited with code {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "")

    df = parse_results(text)
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
