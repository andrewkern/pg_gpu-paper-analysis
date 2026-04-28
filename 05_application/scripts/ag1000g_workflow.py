#!/usr/bin/env python
"""
End-to-end pg_gpu workflow on Ag1000G Phase 3 chromosome 3R.

Demonstrates the full catalog of windowed statistics on real data
with missing genotypes. Loads unphased data, assigns populations,
computes diversity, divergence, selection, LD, and SFS statistics
in windows, then produces a multi-panel genome scan figure.

Produces:
  - tables/ag1000g_workflow_timing.csv
  - figures/ag1000g_genome_scan.pdf
"""

import time
import numpy as np
import pandas as pd
import cupy as cp
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from pg_gpu import (
    HaplotypeMatrix, diversity, divergence, selection, sfs,
    windowed_analysis, ld_statistics, distance_stats,
)
from pg_gpu.accessible import AccessibleMask

OUT_DIR_FIG = "05_application/figures"
OUT_DIR_TBL = "05_application/tables"

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
# Canonical accessibility bitmask (boolean ndarray, 1-based offset).
# The sibling singer-test/*.mask.bed files are the inaccessible complement of
# this array and have inverted polarity -- do not use them as accessibility input.
MASK_NPZ = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"
CHROM = "3R"
N_DIP_PER_POP = 100
WINDOW_SIZE = 100_000
STEP_SIZE = 10_000


def load_data():
    """Load full chromosome arm from Ag1000G."""
    print(f"Loading {CHROM} from Ag1000G...", flush=True)
    t0 = time.time()

    store = zarr.open_group(ZARR_PATH, mode='r')
    chrom_grp = store[CHROM]
    positions = np.array(chrom_grp['variants/POS'])
    gt = np.array(chrom_grp['calldata/GT'])
    n_var, n_samp, _ = gt.shape

    haplotypes = np.empty((n_var, 2 * n_samp), dtype=gt.dtype)
    haplotypes[:, :n_samp] = gt[:, :, 0]
    haplotypes[:, n_samp:] = gt[:, :, 1]
    haplotypes = haplotypes.T
    del gt

    hm = HaplotypeMatrix(
        haplotypes, positions,
        chrom_start=int(positions[0]),
        chrom_end=int(positions[-1]))

    # Load biologically meaningful population assignments
    import json
    pop_path = "05_application/tables/population_assignments.json"
    with open(pop_path) as f:
        pops = json.load(f)
    hm.sample_sets = {
        "west_africa": pops["west_africa"],
        "east_africa": pops["east_africa"],
    }

    # Attach accessibility mask from the canonical bitmask npz (offset=1
    # because mask[0] represents 1-based position 1).
    acc_arr = np.load(MASK_NPZ)[f"access_{CHROM}"]
    hm.set_accessible_mask(AccessibleMask(acc_arr, offset=1))

    t_load = time.time() - t0
    print(f"  {hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants "
          f"({hm.n_total_sites:,} accessible bases, {t_load:.0f}s)")
    return hm, t_load


def timed(name, fn, timings):
    """Run fn, record timing, return result."""
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    result = fn()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    timings.append({"step": name, "time_s": elapsed})
    print(f"  {name}: {elapsed:.2f}s", flush=True)
    return result


def main():
    hm, t_load = load_data()
    timings = [{"step": "load_data", "time_s": t_load}]

    # Transfer to GPU
    print("\nTransfer to GPU...", flush=True)
    t0 = time.time()
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    t_xfer = time.time() - t0
    timings.append({"step": "gpu_transfer", "time_s": t_xfer})
    print(f"  {t_xfer:.1f}s", flush=True)

    # Warmup
    _ = diversity.pi(hm, population="west_africa")
    cp.cuda.Stream.null.synchronize()

    # =========================================================================
    # Windowed statistics: the full catalog
    # =========================================================================
    print(f"\nWindowed statistics ({WINDOW_SIZE // 1000}kb windows):", flush=True)

    # --- Single-population diversity ---
    df_div = timed("diversity (pi, theta_w, tajimas_d, theta_h, theta_l)",
        lambda: windowed_analysis(hm, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            statistics=['pi', 'theta_w', 'tajimas_d', 'theta_h', 'theta_l',
                        'segregating_sites', 'singletons', 'max_daf'],
            populations=['west_africa']),
        timings)

    # --- Neutrality tests ---
    df_neut = timed("neutrality tests (fay_wu_h, normalized_fay_wu_h, zeng_e)",
        lambda: windowed_analysis(hm, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            statistics=['fay_wu_h', 'normalized_fay_wu_h', 'zeng_e'],
            populations=['west_africa']),
        timings)

    # --- Two-population divergence ---
    df_div2 = timed("divergence (fst, fst_wc, dxy, da)",
        lambda: windowed_analysis(hm, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            statistics=['fst', 'fst_wc', 'dxy', 'da'],
            populations=['west_africa', 'east_africa']),
        timings)

    # --- Garud's H ---
    df_garud = timed("garud_h (h1, h12, h123, h2h1)",
        lambda: windowed_analysis(hm, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            statistics=['garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1'],
            populations=['west_africa']),
        timings)

    # --- nSL (per-population subset to avoid OOM at full-arm scale) ---
    hm_pop1 = hm.get_population_matrix("west_africa") if hasattr(hm, 'get_population_matrix') else None
    if hm_pop1 is None:
        from pg_gpu._utils import get_population_matrix
        hm_pop1 = get_population_matrix(hm, "west_africa")
    try:
        df_nsl = timed("mean_nsl",
            lambda: windowed_analysis(hm, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                statistics=['mean_nsl'],
                populations=['west_africa']),
            timings)
    except cp.cuda.memory.OutOfMemoryError:
        print("  mean_nsl: OOM at full-arm scale, skipping", flush=True)
        df_nsl = None

    # --- Scalar statistics ---
    print("\nScalar statistics:", flush=True)
    scalar = {}
    scalar['pi_pop1'] = timed("pi(pop1)",
        lambda: diversity.pi(hm, population="west_africa"), timings)
    scalar['pi_pop2'] = timed("pi(pop2)",
        lambda: diversity.pi(hm, population="east_africa"), timings)
    scalar['tajd_pop1'] = timed("tajimas_d(pop1)",
        lambda: diversity.tajimas_d(hm, population="west_africa"), timings)
    scalar['fst'] = timed("fst_hudson(pop1, pop2)",
        lambda: divergence.fst_hudson(hm, "west_africa", "east_africa"), timings)
    scalar['dxy'] = timed("dxy(pop1, pop2)",
        lambda: divergence.dxy(hm, "west_africa", "east_africa"), timings)

    # --- SFS ---
    print("\nSFS:", flush=True)
    sfs_pop1 = timed("sfs(pop1)",
        lambda: sfs.sfs(hm, population="west_africa"), timings)
    jsfs = timed("joint_sfs(pop1, pop2)",
        lambda: sfs.joint_sfs(hm, pop1="west_africa", pop2="east_africa"), timings)

    # --- Total ---
    total_compute = sum(t["time_s"] for t in timings
                        if t["step"] not in ("load_data", "gpu_transfer"))
    total_all = sum(t["time_s"] for t in timings)
    timings.append({"step": "TOTAL_COMPUTE", "time_s": total_compute})
    timings.append({"step": "TOTAL_WITH_IO", "time_s": total_all})

    print(f"\nTotal compute: {total_compute:.1f}s")
    print(f"Total with I/O: {total_all:.1f}s")

    # Save timings
    timing_df = pd.DataFrame(timings)
    timing_df.to_csv(f"{OUT_DIR_TBL}/ag1000g_workflow_timing.csv", index=False)

    # =========================================================================
    # Figure: multi-panel genome scan
    # =========================================================================
    print("\nGenerating figure...", flush=True)

    # Merge windowed results on position
    pos_mb = df_div['start'].values / 1e6

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(9, 2, figure=fig, hspace=0.4, wspace=0.3,
                  width_ratios=[3, 1])

    panel_idx = 0

    def add_panel(y, label, ylabel, color='#2c3e50', alpha=0.6):
        nonlocal panel_idx
        ax = fig.add_subplot(gs[panel_idx, 0])
        ax.plot(pos_mb, y, color=color, alpha=alpha, linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight='bold', loc='left')
        ax.tick_params(labelsize=7)
        if panel_idx < 8:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f'{CHROM} position (Mb)', fontsize=8)
        panel_idx += 1
        return ax

    # Row 0: pi
    add_panel(df_div['pi'].values, 'Nucleotide diversity', r'$\pi$',
              color='#2980b9')

    # Row 1: theta_w
    add_panel(df_div['theta_w'].values, "Watterson's theta", r'$\theta_W$',
              color='#27ae60')

    # Row 2: Tajima's D
    ax = add_panel(df_div['tajimas_d'].values, "Tajima's D", "D",
                   color='#8e44ad')
    ax.axhline(0, color='0.4', linewidth=0.5, linestyle='--')

    # Row 3: Fay & Wu's H (normalized)
    ax = add_panel(df_neut['normalized_fay_wu_h'].values,
                   "Fay & Wu's H*", "H*", color='#d35400')
    ax.axhline(0, color='0.4', linewidth=0.5, linestyle='--')

    # Row 4: Hudson FST
    add_panel(df_div2['fst'].values, r'Hudson $F_{ST}$ (pop1 vs pop2)',
              r'$F_{ST}$', color='#c0392b')

    # Row 5: Dxy
    add_panel(df_div2['dxy'].values, r'$D_{xy}$ (pop1 vs pop2)',
              r'$D_{xy}$', color='#16a085')

    # Row 6: Garud H12
    add_panel(df_garud['garud_h12'].values, "Garud's H12", 'H12',
              color='#e67e22')

    # Row 7: Mean nSL (may be None if OOM)
    if df_nsl is not None:
        ax = add_panel(df_nsl['mean_nsl'].values, 'Mean nSL', 'nSL',
                       color='#2c3e50')
        ax.axhline(0, color='0.4', linewidth=0.5, linestyle='--')
    else:
        ax = add_panel(np.full(len(pos_mb), np.nan),
                       'Mean nSL (skipped: OOM)', 'nSL', color='#bdc3c7')
        ax.text(0.5, 0.5, 'OOM at full-arm scale', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='0.5')

    # Row 8: Segregating sites
    add_panel(df_div['segregating_sites'].values, 'Segregating sites per window',
              'S', color='#7f8c8d')

    # Right column: SFS and joint SFS
    ax_sfs = fig.add_subplot(gs[0:3, 1])
    sfs_arr = sfs_pop1[1:-1]  # exclude 0 and n
    ax_sfs.bar(range(1, len(sfs_arr) + 1), sfs_arr, color='#2980b9',
               edgecolor='0.3', linewidth=0.3, width=1.0)
    ax_sfs.set_xlabel('Derived allele count', fontsize=8)
    ax_sfs.set_ylabel('Count', fontsize=8)
    ax_sfs.set_title('SFS (pop1)', fontsize=9, fontweight='bold')
    ax_sfs.set_xlim(0, min(50, len(sfs_arr)))
    ax_sfs.tick_params(labelsize=7)

    ax_jsfs = fig.add_subplot(gs[3:6, 1])
    # Log-scale joint SFS heatmap
    jsfs_plot = np.log10(jsfs + 1)
    n1 = min(50, jsfs_plot.shape[0])
    n2 = min(50, jsfs_plot.shape[1])
    im = ax_jsfs.imshow(jsfs_plot[:n1, :n2].T, origin='lower', aspect='auto',
                         cmap='viridis')
    ax_jsfs.set_xlabel('pop1 DAC', fontsize=8)
    ax_jsfs.set_ylabel('pop2 DAC', fontsize=8)
    ax_jsfs.set_title('Joint SFS (log10)', fontsize=9, fontweight='bold')
    ax_jsfs.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax_jsfs, fraction=0.046, pad=0.04)

    # Timing breakdown
    ax_time = fig.add_subplot(gs[6:9, 1])
    compute_timings = [(t['step'], t['time_s']) for t in timings
                       if t['step'] not in ('load_data', 'gpu_transfer',
                                             'TOTAL_COMPUTE', 'TOTAL_WITH_IO')]
    names = [t[0][:25] for t in compute_timings]
    times = [t[1] for t in compute_timings]
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax_time.barh(range(len(names)), times, color=colors_bar,
                 edgecolor='0.3', linewidth=0.5)
    ax_time.set_yticks(range(len(names)))
    ax_time.set_yticklabels(names, fontsize=7)
    ax_time.set_xlabel('Time (seconds)', fontsize=8)
    ax_time.set_title(f'Compute time ({total_compute:.1f}s total)',
                      fontsize=9, fontweight='bold')
    ax_time.tick_params(labelsize=7)
    ax_time.invert_yaxis()

    fig.suptitle(f'pg_gpu: Ag1000G {CHROM} genome scan\n'
                 f'{hm.num_haplotypes} haplotypes, {hm.num_variants:,} variants, '
                 f'{WINDOW_SIZE//1000}kb windows',
                 fontsize=12, fontweight='bold', y=0.995)

    fig.savefig(f"{OUT_DIR_FIG}/ag1000g_genome_scan.pdf",
                bbox_inches='tight', dpi=150)
    print(f"Figure saved to {OUT_DIR_FIG}/ag1000g_genome_scan.pdf")

    # Print scalar results
    print(f"\nScalar statistics:")
    for k, v in scalar.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
