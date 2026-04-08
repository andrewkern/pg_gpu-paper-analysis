#!/usr/bin/env python
"""
Generate the Ag1000G genome scan figure from cached or recomputed data.
Visualization-only script for rapid iteration.
"""

import os
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
    HaplotypeMatrix, diversity, divergence, sfs as sfs_mod,
    windowed_analysis, selection,
)

OUT_DIR = "05_application/figures"
CACHE_DIR = "05_application/tables"
ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
MASK_BED = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer-test/3R.mask.bed"
CHROM = "3R"
N_DIP = 100
WINDOW_SIZE = 100_000


def _cache_path(name):
    return os.path.join(CACHE_DIR, f"cached_{name}.csv")


def _cache_exists():
    """Check if all cached results are on disk."""
    needed = ['div', 'neut', 'div2', 'garud', 'sfs_pop1', 'jsfs']
    return all(os.path.exists(_cache_path(n)) for n in needed)


def _save_cache(df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs):
    """Save windowed results to CSV for fast iteration."""
    df_div.to_csv(_cache_path('div'), index=False)
    df_neut.to_csv(_cache_path('neut'), index=False)
    df_div2.to_csv(_cache_path('div2'), index=False)
    df_garud.to_csv(_cache_path('garud'), index=False)
    np.savetxt(_cache_path('sfs_pop1'), sfs_pop1)
    np.savetxt(_cache_path('jsfs'), jsfs)
    print(f"  Cached results to {CACHE_DIR}/", flush=True)


def _load_cache():
    """Load windowed results from CSV."""
    df_div = pd.read_csv(_cache_path('div'))
    df_neut = pd.read_csv(_cache_path('neut'))
    df_div2 = pd.read_csv(_cache_path('div2'))
    df_garud = pd.read_csv(_cache_path('garud'))
    sfs_pop1 = np.loadtxt(_cache_path('sfs_pop1'))
    jsfs = np.loadtxt(_cache_path('jsfs'))
    return df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs


def load_and_compute():
    """Load data and compute windowed statistics, or read from cache."""

    # Try cache first
    if _cache_exists():
        print("Loading cached results from disk...", flush=True)
        df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs = _load_cache()
        timing_df = pd.read_csv(os.path.join(CACHE_DIR, "ag1000g_workflow_timing.csv"))
        # Reconstruct n_hap and n_var from the data
        n_hap = 2940
        n_var = 10_939_888
        return n_hap, n_var, df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs, timing_df

    # Compute from scratch
    import time
    print("Loading data...", flush=True)
    store = zarr.open_group(ZARR_PATH, mode='r')
    grp = store[CHROM]
    positions = np.array(grp['variants/POS'])
    gt = np.array(grp['calldata/GT'])
    n_var, n_samp, _ = gt.shape
    haplotypes = np.empty((n_var, 2 * n_samp), dtype=gt.dtype)
    haplotypes[:, :n_samp] = gt[:, :, 0]
    haplotypes[:, n_samp:] = gt[:, :, 1]
    haplotypes = haplotypes.T
    del gt

    hm = HaplotypeMatrix(haplotypes, positions,
                          chrom_start=int(positions[0]),
                          chrom_end=int(positions[-1]))

    # Load biologically meaningful population assignments
    import json
    pop_path = os.path.join(CACHE_DIR, "population_assignments.json")
    with open(pop_path) as f:
        pops = json.load(f)
    hm.sample_sets = {
        "west_africa": pops["west_africa"],
        "east_africa": pops["east_africa"],
    }

    # Attach accessibility mask — filters variants to accessible regions
    # and uses accessible base count for per-base normalization
    hm.set_accessible_mask(MASK_BED, chrom=CHROM)
    n_accessible = hm.n_total_sites
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    n_hap = hm.num_haplotypes
    n_var = hm.num_variants
    print(f"  {n_hap} haps x {n_var:,} variants "
          f"({n_accessible:,} accessible bases)", flush=True)

    print("Computing windowed stats...", flush=True)
    t0 = time.perf_counter()
    df_div = windowed_analysis(hm, window_size=WINDOW_SIZE,
        statistics=['pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                    'singletons', 'max_daf'],
        populations=['west_africa'])
    df_neut = windowed_analysis(hm, window_size=WINDOW_SIZE,
        statistics=['fay_wu_h', 'normalized_fay_wu_h', 'zeng_e'],
        populations=['west_africa'])
    df_div2 = windowed_analysis(hm, window_size=WINDOW_SIZE,
        statistics=['fst', 'fst_wc', 'dxy', 'da'],
        populations=['west_africa', 'east_africa'])
    # Garud H: use SNP-count windows (matches Miles et al. 2017)
    h1, h12, h123, h2h1 = selection.moving_garud_h(
        hm, population='west_africa', size=1000, step=200)
    # Compute window center positions
    pos_cpu = hm.positions
    if hasattr(pos_cpu, 'get'):
        pos_cpu = pos_cpu.get()
    garud_centers = []
    for w_start in range(0, len(pos_cpu) - 1000 + 1, 200):
        garud_centers.append((pos_cpu[w_start] + pos_cpu[w_start + 999]) / 2)
    garud_centers = np.array(garud_centers)
    df_garud = pd.DataFrame({
        'center': garud_centers[:len(h12)],
        'garud_h1': h1, 'garud_h12': h12,
        'garud_h123': h123, 'garud_h2h1': h2h1,
    })
    print(f"  Windowed stats: {time.perf_counter()-t0:.1f}s", flush=True)

    sfs_pop1 = sfs_mod.sfs(hm, population="west_africa")
    jsfs = sfs_mod.joint_sfs(hm, pop1="west_africa", pop2="east_africa")

    _save_cache(df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs)

    timing_df = pd.read_csv(os.path.join(CACHE_DIR, "ag1000g_workflow_timing.csv"))

    return n_hap, n_var, df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs, timing_df


def make_figure(n_hap, n_var, df_div, df_neut, df_div2, df_garud, sfs_pop1, jsfs, timing_df):
    """Build the multi-panel genome scan figure."""

    pos_mb = df_div['start'].values / 1e6

    # --- Layout: 8 rows, left=scan panels (wide), right=SFS/timing ---
    n_rows = 8
    sns.set_theme(style="darkgrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(18, 20))

    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.25,
                  width_ratios=[3, 1],
                  left=0.06, right=0.96, top=0.95, bottom=0.04)

    # Gste cluster coordinates (glutathione S-transferase epsilon, 3R:28.48-28.60 Mb)
    GSTE_START_MB = 28.48
    GSTE_END_MB = 28.60
    scan_axes = []  # collect all scan panel axes for region shading

    panel_idx = 0

    def add_scan_panel(y, label, ylabel, color='#2c3e50', alpha=0.6, hline=None):
        nonlocal panel_idx
        ax = fig.add_subplot(gs[panel_idx, 0])
        ax.plot(pos_mb, y, color=color, alpha=alpha, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold', loc='left', pad=2)
        ax.tick_params(labelsize=7)
        ax.set_xlim(pos_mb[0], pos_mb[-1])
        if hline is not None:
            ax.axhline(hline, color='0.5', linewidth=0.5, linestyle='--')
        if panel_idx < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f'{CHROM} position (Mb)', fontsize=9)
        scan_axes.append(ax)
        panel_idx += 1
        return ax

    # Row 0: pi
    add_scan_panel(df_div['pi'].values, 'Nucleotide diversity', r'$\pi$',
                   color='#2980b9')

    # Row 1: theta_w
    add_scan_panel(df_div['theta_w'].values, "Watterson's theta", r'$\theta_W$',
                   color='#27ae60')

    # Row 2: Tajima's D
    add_scan_panel(df_div['tajimas_d'].values, "Tajima's D", "D",
                   color='#8e44ad', hline=0)

    # Row 3: Fay & Wu's H*
    add_scan_panel(df_neut['normalized_fay_wu_h'].values,
                   "Fay & Wu's H*", "H*", color='#d35400', hline=0)

    # Row 4: Zeng E
    add_scan_panel(df_neut['zeng_e'].values, "Zeng's E (West Africa)", 'E',
                   color='#2c3e50', hline=0)
    
    # Row 5: Hudson FST
    add_scan_panel(df_div2['fst'].values, r'Hudson $F_{ST}$ (West vs East Africa)',
                   r'$F_{ST}$', color='#c0392b')

    # Row 6: Dxy
    add_scan_panel(df_div2['dxy'].values, r'$D_{xy}$ (West vs East Africa)',
                   r'$D_{xy}$', color='#16a085')

    # Row 7: Garud H12 (SNP-count windows — different x-axis)
    ax = fig.add_subplot(gs[panel_idx, 0])
    garud_pos_mb = df_garud['center'].values / 1e6
    ax.plot(garud_pos_mb, df_garud['garud_h12'].values,
            color='#e67e22', alpha=0.6, linewidth=0.6)
    ax.set_ylabel('H12', fontsize=9)
    ax.set_title("Garud's H12 (West Africa, 1000-SNP windows)", fontsize=10,
                 fontweight='bold', loc='left', pad=2)
    ax.tick_params(labelsize=7)
    ax.set_xlim(pos_mb[0], pos_mb[-1])
    ax.set_xlabel(f'{CHROM} position (Mb)', fontsize=9)
    scan_axes.append(ax)
    panel_idx += 1

    # Shade Gste cluster region on all scan panels
    for sax in scan_axes:
        sax.axvspan(GSTE_START_MB, GSTE_END_MB, alpha=0.15, color='#e74c3c',
                    zorder=0)
    # Label on top panel only
    scan_axes[0].text((GSTE_START_MB + GSTE_END_MB) / 2,
                      scan_axes[0].get_ylim()[1],
                      'Gste', ha='center', va='bottom', fontsize=7,
                      fontstyle='italic', color='#c0392b')

    # --- Right column: SFS (rows 0-2) ---
    ax_sfs = fig.add_subplot(gs[0:3, 1])
    sfs_arr = sfs_pop1[1:-1]
    n_show = min(40, len(sfs_arr))
    ax_sfs.bar(range(1, n_show + 1), sfs_arr[:n_show], color='#2980b9',
               edgecolor='none', width=0.8)
    ax_sfs.set_xlabel('Derived allele count', fontsize=8)
    ax_sfs.set_ylabel('Count', fontsize=8)
    ax_sfs.set_title('SFS (West Africa)', fontsize=10, fontweight='bold', pad=6)
    ax_sfs.tick_params(labelsize=7)

    # --- Right column: Joint SFS (rows 3-5) ---
    ax_jsfs = fig.add_subplot(gs[3:6, 1])
    jsfs_plot = np.log10(np.maximum(jsfs, 1))
    n1 = min(40, jsfs_plot.shape[0])
    n2 = min(40, jsfs_plot.shape[1])
    im = ax_jsfs.imshow(jsfs_plot[:n1, :n2].T, origin='lower', aspect='equal',
                         cmap='viridis', interpolation='nearest')
    ax_jsfs.set_xlabel('West Africa DAC', fontsize=8)
    ax_jsfs.set_ylabel('East Africa DAC', fontsize=8)
    ax_jsfs.set_title('Joint SFS (log10)', fontsize=10, fontweight='bold', pad=6)
    ax_jsfs.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax_jsfs, fraction=0.046, pad=0.04, shrink=0.8)
    cb.ax.tick_params(labelsize=6)

    # --- Right column: Timing (rows 6-7) ---
    ax_time = fig.add_subplot(gs[6:8, 1])
    compute_rows = timing_df[~timing_df['step'].isin(
        ['load_data', 'gpu_transfer', 'TOTAL_COMPUTE', 'TOTAL_WITH_IO'])]
    label_map = {
        'diversity (pi, theta_w, tajimas_d, theta_h, theta_l)': 'diversity (5 stats)',
        'neutrality tests (fay_wu_h, normalized_fay_wu_h, zeng_e)': 'neutrality (3 tests)',
        'divergence (fst, fst_wc, dxy, da)': 'divergence (4 stats)',
        'garud_h (h1, h12, h123, h2h1)': 'Garud H (4 stats)',
    }
    names = [label_map.get(s, s)[:25] for s in compute_rows['step'].values]
    times = compute_rows['time_s'].values
    total = timing_df[timing_df['step'] == 'TOTAL_COMPUTE']['time_s'].values[0]
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax_time.barh(range(len(names)), times, color=colors_bar,
                 edgecolor='0.3', linewidth=0.5)
    ax_time.set_yticks(range(len(names)))
    ax_time.set_yticklabels(names, fontsize=7)
    ax_time.set_xscale('log')
    ax_time.set_xlabel('Time (seconds)', fontsize=8)
    ax_time.set_title(f'Compute time ({total:.1f}s total)', fontsize=10,
                      fontweight='bold', pad=6)
    ax_time.tick_params(labelsize=7)
    ax_time.invert_yaxis()

    # --- Title ---
    fig.suptitle(f'Ag1000G {CHROM} genome scan\n'
                 f'{n_hap} haplotypes, {n_var:,} variants, '
                 f'{WINDOW_SIZE//1000}kb windows',
                 fontsize=11, fontweight='bold')

    fig.savefig(f"{OUT_DIR}/ag1000g_genome_scan.pdf",
                bbox_inches='tight', dpi=150)
    fig.savefig(f"{OUT_DIR}/ag1000g_genome_scan.png",
                bbox_inches='tight', dpi=150)
    print(f"Figure saved to {OUT_DIR}/ag1000g_genome_scan.pdf")


if __name__ == "__main__":
    data = load_and_compute()
    make_figure(*data)
