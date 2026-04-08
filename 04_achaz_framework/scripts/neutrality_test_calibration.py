#!/usr/bin/env python
"""
Validate neutrality test calibration under the standard neutral model.

For each test (Tajima's D, normalized H*, Zeng E), simulates n_reps
replicates under the neutral coalescent (no recombination) and checks
that the test statistic has mean ~ 0 and variance ~ 1.

Also compares the Achaz (2009) general variance against the classical
Zeng (2006) E variance formula to demonstrate the miscalibration.

Produces:
  - tables/neutrality_calibration.csv
  - figures/neutrality_calibration.pdf
"""

import numpy as np
import msprime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import HaplotypeMatrix
from pg_gpu._memutil import dac_and_n
from pg_gpu.diversity import (
    _achaz_variance_coefficients, _harmonic_a1_a2,
    _weights_pi, _weights_watterson, _weights_theta_h, _weights_theta_l,
    _achaz_alpha_beta,
)
import cupy as cp

OUT_DIR_FIG = "04_achaz_framework/figures"
OUT_DIR_TBL = "04_achaz_framework/tables"

N_POP = 10_000
MU = 1e-8
L = 100_000
N_HAP = 50  # haploid sample size (25 diploid individuals)
N_REPS = 1000


def classical_zeng_e_variance(n, S):
    """Zeng et al. (2006) Eq. 14 variance for E = theta_L - theta_W."""
    a_n = sum(1.0 / i for i in range(1, n))
    b_n = sum(1.0 / (i * i) for i in range(1, n))
    theta = S / a_n
    n_f = float(n)
    e1 = (n_f / (2 * (n_f - 1)) - 1.0 / a_n)
    e2_num = (b_n / (a_n * a_n)
              + 2 * (n_f / (n_f - 1)) ** 2 * b_n
              - 2 * (n_f * b_n - n_f + 1) / ((n_f - 1) * a_n)
              - (3 * n_f + 1) / (n_f - 1))
    e2 = e2_num / (a_n * a_n + b_n)
    return e1 * theta + e2 * theta * theta


def simulate_and_compute(seed):
    """Simulate one replicate and compute test statistics."""
    ts = msprime.sim_ancestry(
        samples=N_HAP // 2, sequence_length=L,
        recombination_rate=0,  # no recombination for calibration
        population_size=N_POP, random_seed=seed, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    hm = HaplotypeMatrix.from_ts(ts)
    n = hm.num_haplotypes
    if hm.num_variants < 5:
        return None

    hm.transfer_to_gpu()
    dac_arr, nv = dac_and_n(hm.haplotypes)
    dac_np = dac_arr.get()
    nv_np = nv.get()
    seg = (dac_np > 0) & (dac_np < nv_np)
    S = float(np.sum(seg))
    if S < 3:
        return None

    # Build SFS for theta computation
    sfs_arr = np.bincount(dac_np, minlength=n + 1).astype(float)
    sfs_arr[0] = sfs_arr[n] = 0.0

    w_pi = _weights_pi(n)
    w_tw = _weights_watterson(n)
    w_th = _weights_theta_h(n)
    w_tl = _weights_theta_l(n)

    pi_val = float(np.sum(w_pi[1:n] * sfs_arr[1:n]))
    tw_val = float(np.sum(w_tw[1:n] * sfs_arr[1:n]))
    th_val = float(np.sum(w_th[1:n] * sfs_arr[1:n]))
    tl_val = float(np.sum(w_tl[1:n] * sfs_arr[1:n]))

    a1, a2 = _harmonic_a1_a2(n)
    theta_est = S / a1
    theta_sq_est = S * (S - 1) / (a1 ** 2 + a2)

    row = {'S': S, 'n': n}

    # Tajima's D (Achaz)
    alpha, beta = _achaz_variance_coefficients('pi', 'watterson', n)
    var_d = alpha * theta_est + beta * theta_sq_est
    row['tajd_achaz'] = (pi_val - tw_val) / np.sqrt(var_d) if var_d > 0 else np.nan

    # Normalized H* (Achaz)
    alpha, beta = _achaz_variance_coefficients('pi', 'theta_h', n)
    var_h = alpha * theta_est + beta * theta_sq_est
    row['hstar_achaz'] = (pi_val - th_val) / np.sqrt(var_h) if var_h > 0 else np.nan

    # Zeng E (Achaz)
    alpha, beta = _achaz_variance_coefficients('theta_l', 'watterson', n)
    var_e = alpha * theta_est + beta * theta_sq_est
    row['zenge_achaz'] = (tl_val - tw_val) / np.sqrt(var_e) if var_e > 0 else np.nan

    # Zeng E (classical formula for comparison)
    var_e_classical = classical_zeng_e_variance(n, S)
    row['zenge_classical'] = ((tl_val - tw_val) / np.sqrt(var_e_classical)
                              if var_e_classical > 0 else np.nan)

    return row


def main():
    print(f"Simulating {N_REPS} neutral replicates (n={N_HAP}, L={L}, no recombination)...")
    rows = []
    for seed in range(1, N_REPS + 1):
        if seed % 100 == 0:
            print(f"  {seed}/{N_REPS}", flush=True)
        r = simulate_and_compute(seed)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} valid replicates")

    # Summary statistics
    summary = []
    for col, label in [('tajd_achaz', "Tajima's D (Achaz)"),
                        ('hstar_achaz', "H* (Achaz)"),
                        ('zenge_achaz', "Zeng E (Achaz)"),
                        ('zenge_classical', "Zeng E (classical)")]:
        vals = df[col].dropna()
        summary.append({
            'test': label,
            'mean': vals.mean(),
            'variance': vals.var(),
            'n_reps': len(vals),
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(f"{OUT_DIR_TBL}/neutrality_calibration.csv", index=False)
    print(f"\nCalibration results:")
    print(sum_df.to_string(index=False))

    # Figure: histograms
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

    tests = [
        ("tajd_achaz", "Tajima's D\n(Achaz variance)"),
        ("hstar_achaz", "Normalized H*\n(Achaz variance)"),
        ("zenge_achaz", "Zeng E\n(Achaz variance)"),
        ("zenge_classical", "Zeng E\n(classical variance)"),
    ]

    for ax, (col, title) in zip(axes, tests):
        vals = df[col].dropna()
        ax.hist(vals, bins=40, density=True, alpha=0.7, color='#2ecc71',
                edgecolor='0.3', linewidth=0.5)
        # Overlay standard normal
        x = np.linspace(-4, 4, 200)
        ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
                'k--', linewidth=1, label='N(0,1)')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Test statistic')
        m, v = vals.mean(), vals.var()
        ax.text(0.95, 0.95, f'mean={m:.2f}\nvar={v:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Neutrality test calibration (n={N_HAP}, {len(df)} reps, no recombination)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR_FIG}/neutrality_calibration.pdf", bbox_inches='tight')
    print(f"Figure saved to {OUT_DIR_FIG}/neutrality_calibration.pdf")


if __name__ == "__main__":
    main()
