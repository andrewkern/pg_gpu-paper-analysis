#!/usr/bin/env python
"""
Validate that theta estimators are unbiased under MCAR missing data.

Simulates under the standard neutral model with known theta, injects
missing data at rates from 0-60%, and compares E[theta_hat] to theta_true.

Produces:
  - tables/missing_data_bias.csv
  - figures/missing_data_bias.pdf
"""

import numpy as np
import msprime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import HaplotypeMatrix, diversity

OUT_DIR_FIG = "01_accuracy/figures"
OUT_DIR_TBL = "01_accuracy/tables"

N_POP = 10_000
MU = 1e-8
L = 100_000
N_HAP = 100
N_REPS = 200
MISSING_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
THETA_TRUE = 4 * N_POP * MU * L  # expected total theta


def simulate_one(seed):
    """Simulate one replicate, return HaplotypeMatrix."""
    ts = msprime.sim_ancestry(
        samples=N_HAP // 2, sequence_length=L,
        recombination_rate=1e-8, population_size=N_POP,
        random_seed=seed, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


def inject_missing(hm, rate, rng):
    """Inject MCAR missing data at the given rate."""
    hap = hm.haplotypes.copy()
    if hasattr(hap, 'get'):
        hap = hap.get()
    mask = rng.random(hap.shape) < rate
    hap[mask] = -1
    return HaplotypeMatrix(hap, hm.positions,
                           chrom_start=hm.chrom_start, chrom_end=hm.chrom_end)


def main():
    print(f"Simulating {N_REPS} replicates x {len(MISSING_RATES)} missing rates...")
    rows = []
    rng = np.random.default_rng(42)

    for seed in range(1, N_REPS + 1):
        if seed % 50 == 0:
            print(f"  {seed}/{N_REPS}", flush=True)

        hm = simulate_one(seed)
        if hm.num_variants < 5:
            continue

        for rate in MISSING_RATES:
            if rate > 0:
                hm_miss = inject_missing(hm, rate, rng)
            else:
                hm_miss = hm

            hm_miss.transfer_to_gpu()

            for stat_name, fn in [
                ('pi', lambda m: diversity.pi(m, span_normalize=False)),
                ('theta_w', lambda m: diversity.theta_w(m, span_normalize=False)),
                ('theta_h', lambda m: diversity.theta_h(m, span_normalize=False)),
                ('theta_l', lambda m: diversity.theta_l(m, span_normalize=False)),
                ('tajimas_d', lambda m: diversity.tajimas_d(m)),
            ]:
                val = fn(hm_miss)
                rows.append({
                    'seed': seed,
                    'missing_rate': rate,
                    'statistic': stat_name,
                    'value': val,
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR_TBL}/missing_data_bias.csv", index=False)

    # Summary: mean estimate / true value by missing rate
    summary = []
    for stat in ['pi', 'theta_w', 'theta_h', 'theta_l']:
        for rate in MISSING_RATES:
            vals = df[(df['statistic'] == stat) & (df['missing_rate'] == rate)]['value']
            summary.append({
                'statistic': stat,
                'missing_rate': rate,
                'mean_estimate': vals.mean(),
                'std_estimate': vals.std(),
                'bias_ratio': vals.mean() / THETA_TRUE,
            })
    for rate in MISSING_RATES:
        vals = df[(df['statistic'] == 'tajimas_d') & (df['missing_rate'] == rate)]['value']
        summary.append({
            'statistic': 'tajimas_d',
            'missing_rate': rate,
            'mean_estimate': vals.mean(),
            'std_estimate': vals.std(),
            'bias_ratio': np.nan,  # D has no "true" value to normalize by
        })

    sum_df = pd.DataFrame(summary)
    print(sum_df[sum_df['statistic'] == 'pi'].to_string(index=False))

    # Figure
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bias ratio for theta estimators
    ax = axes[0]
    for stat in ['pi', 'theta_w', 'theta_h', 'theta_l']:
        sub = sum_df[sum_df['statistic'] == stat]
        ax.plot(sub['missing_rate'], sub['bias_ratio'], 'o-', label=stat, markersize=4)
    ax.axhline(1.0, color='0.4', linestyle='--', linewidth=1)
    ax.set_xlabel('Missing data rate')
    ax.set_ylabel('E[estimate] / true theta')
    ax.set_title('Theta estimator bias under MCAR')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.5)

    # Right: Tajima's D mean
    ax = axes[1]
    sub = sum_df[sum_df['statistic'] == 'tajimas_d']
    ax.errorbar(sub['missing_rate'], sub['mean_estimate'],
                yerr=sub['std_estimate'] / np.sqrt(N_REPS),
                fmt='o-', color='#2ecc71', markersize=4)
    ax.axhline(0, color='0.4', linestyle='--', linewidth=1)
    ax.set_xlabel('Missing data rate')
    ax.set_ylabel("Mean Tajima's D")
    ax.set_title("Tajima's D under MCAR")

    fig.suptitle(f'Missing data robustness (n={N_HAP}, {N_REPS} reps, include mode)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR_FIG}/missing_data_bias.pdf", bbox_inches='tight')
    print(f"Figure saved to {OUT_DIR_FIG}/missing_data_bias.pdf")


if __name__ == "__main__":
    main()
