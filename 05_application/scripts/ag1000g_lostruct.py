#!/usr/bin/env python
"""
Local PCA / lostruct on Ag1000G Phase 3 chromosome 3R, West Africa subset.

Runs the four-step Li & Ralph (2019) pipeline on the same West African
subset (200 phased haplotypes, 100 diploid individuals) used by the
companion genome-scan workflow in ag1000g_workflow.py: per-window local
PCA, Frobenius distance between window covariance representations,
classical MDS, and corner detection in MDS space. The MDS scatter is
plotted with a single neutral fill, with the windows that fall in
each detected corner circled per Li & Ralph (2019) Figure 2; a
companion windowed Garud's H12 track in physical (bp) windows
provides a familiar haplotype-frequency anchor.

Produces:
  - tables/ag1000g_lostruct_windows.csv
  - tables/ag1000g_lostruct_corners.csv
  - figures/ag1000g_lostruct.pdf
"""

import time
import numpy as np
import pandas as pd
import zarr
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from pg_gpu import HaplotypeMatrix, lostruct, windowed_analysis, selection
from pg_gpu.accessible import AccessibleMask

OUT_DIR_FIG = "05_application/figures"
OUT_DIR_TBL = "05_application/tables"

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
MASK_NPZ = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"
POP_JSON = "05_application/tables/population_assignments.json"
POPULATION = "west_africa"
CHROM = "3R"
# Canonical Li and Ralph (2019) lostruct uses fixed-SNP-count windows
# so each window has constant statistical power for the local PCA. The
# streaming engine added in pg_gpu (engine='streaming-dense' below)
# bounds peak GPU memory at the per-window working set, so 1000-SNP
# non-overlapping windows over 10.9M phased Ag1000G 3R variants
# (about 11k windows total) fit comfortably alongside the 32 GB
# haplotype buffer on a single 80 GB A100.
WINDOW_SIZE = 1_000     # SNPs per window
STEP_SIZE = 1_000       # SNPs per step (non-overlapping)
WINDOW_TYPE = 'snp'

# Companion Garud H12 track uses the *same* SNP-defined windows as
# the lostruct pass so the two panels share identical x-coordinates.
# Computing H12 in bp windows on a sweep region that's only ~0.3 Mb
# wide smears the signal across the wider 100 kb window and produces
# spurious offset between the lostruct corners and H12 peaks.
K_PCS = 2
N_CORNERS = 3
# Smaller corner_prop -> tighter / more visually distinct corner clusters.
CORNER_PROP = 0.01
RANDOM_STATE = 42


def load_data():
    """Load full 3R arm, attach accessibility mask, and tag sample sets."""
    import json
    print(f"Loading {CHROM} from Ag1000G...", flush=True)
    t0 = time.time()
    store = zarr.open_group(ZARR_PATH, mode='r')
    g = store[CHROM]
    positions = np.array(g['variants/POS'])
    gt = np.array(g['calldata/GT'])
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

    with open(POP_JSON) as f:
        pops = json.load(f)
    hm.sample_sets = {
        "west_africa": pops["west_africa"],
        "east_africa": pops["east_africa"],
    }

    acc_arr = np.load(MASK_NPZ)[f"access_{CHROM}"]
    hm.set_accessible_mask(AccessibleMask(acc_arr, offset=1))

    t_load = time.time() - t0
    n_pop = len(hm.sample_sets[POPULATION])
    print(f"  {hm.num_haplotypes} haplotypes total, {n_pop} in {POPULATION}, "
          f"{hm.num_variants:,} variants ({hm.n_total_sites:,} accessible "
          f"bases, {t_load:.0f}s)", flush=True)
    return hm


def main():
    hm = load_data()

    print(f"\nTransfer to GPU...", flush=True)
    t0 = time.time()
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  {time.time() - t0:.1f}s", flush=True)

    print(f"\nRunning lostruct (window={WINDOW_SIZE} {WINDOW_TYPE}, "
          f"step={STEP_SIZE}, k={K_PCS}, n_corners={N_CORNERS})...",
          flush=True)
    t0 = time.time()
    cp.cuda.Stream.null.synchronize()
    res = lostruct(hm,
                   population=POPULATION,
                   window_size=WINDOW_SIZE,
                   step_size=STEP_SIZE,
                   window_type=WINDOW_TYPE,
                   k=K_PCS,
                   corner_prop=CORNER_PROP,
                   n_corners=N_CORNERS,
                   random_state=RANDOM_STATE,
                   engine='streaming-dense')
    cp.cuda.Stream.null.synchronize()
    t_lostruct = time.time() - t0
    print(f"  n_windows={res.n_windows}  ({t_lostruct:.1f}s)", flush=True)
    print(f"  variance explained MDS1/MDS2: "
          f"{res.explained_variance_ratio[0]:.3f} / "
          f"{res.explained_variance_ratio[1]:.3f}", flush=True)

    mds = res.mds
    centers = res.windows['center'].to_numpy()
    starts = res.windows['start'].to_numpy()
    ends = res.windows['end'].to_numpy()
    corner_idx = res.corner_indices

    # Companion: Garud's H12 in the SAME 1000-SNP non-overlapping
    # windows as lostruct. selection.moving_garud_h precomputes a
    # global prefix-sum hash that needs three full float64 copies of
    # the haplotype matrix (~52 GB at 200 hap x 10.9M var) -- OOMs
    # on a single A100. Loop in Python instead and call the scalar
    # selection.garud_h per window: each window is just 200 x 1000
    # int8 (200 KB), so the total cost is dominated by Python overhead
    # not GPU work.
    print(f"\nCompanion Garud H12 scan ({WINDOW_SIZE} SNPs / step "
          f"{STEP_SIZE}, matching lostruct)...", flush=True)
    t0 = time.time()
    pos_cpu = hm.positions
    if hasattr(pos_cpu, 'get'):
        pos_cpu = pos_cpu.get()
    n_var_total = len(pos_cpu)
    h12_vals = []
    h12_centers = []
    for w_start in range(0, n_var_total - WINDOW_SIZE + 1, STEP_SIZE):
        w_end = w_start + WINDOW_SIZE
        hm_w = hm.get_subset(np.arange(w_start, w_end))
        _, h12, _, _ = selection.garud_h(hm_w, population=POPULATION)
        h12_vals.append(float(h12))
        h12_centers.append((pos_cpu[w_start] + pos_cpu[w_end - 1]) / 2)
    df_h12 = pd.DataFrame({
        'center': np.array(h12_centers),
        'garud_h12': np.array(h12_vals),
    })
    print(f"  {time.time() - t0:.1f}s, n_windows={len(df_h12)}", flush=True)
    df_h12.to_csv(f"{OUT_DIR_TBL}/ag1000g_lostruct_h12.csv", index=False)

    # ---------------- save tables ----------------
    win_df = pd.DataFrame({
        'start': starts,
        'end': ends,
        'center': centers,
        'mds1': mds[:, 0],
        'mds2': mds[:, 1],
    })
    win_df.to_csv(f"{OUT_DIR_TBL}/ag1000g_lostruct_windows.csv", index=False)

    corner_rows = []
    for ci in range(N_CORNERS):
        for wi in corner_idx[:, ci]:
            corner_rows.append({
                'corner': ci + 1,
                'start': int(starts[wi]),
                'end': int(ends[wi]),
                'center': int(centers[wi]),
                'mds1': float(mds[wi, 0]),
                'mds2': float(mds[wi, 1]),
            })
    pd.DataFrame(corner_rows).to_csv(
        f"{OUT_DIR_TBL}/ag1000g_lostruct_corners.csv", index=False)

    # ---------------- figure ----------------
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.5],
                   height_ratios=[1.0, 1.0],
                   hspace=0.18, wspace=0.22)
    ax_mds = fig.add_subplot(gs[:, 0])
    ax_mds1 = fig.add_subplot(gs[0, 1])
    ax_h12 = fig.add_subplot(gs[1, 1], sharex=ax_mds1)

    # left: MDS scatter, no regime colouring -- neutral grey fill for
    # all windows, with the corner windows circled per Li and Ralph
    # (2019) Figure 2. Base colour is a desaturated grey so the three
    # tab10 corner colours all read clearly against it.
    POINT_COLOR = '#9aa6b1'
    ax_mds.scatter(mds[:, 0], mds[:, 1],
                    c=POINT_COLOR, s=8,
                    edgecolors='white', linewidths=0.1, alpha=0.7)
    corner_edges = plt.get_cmap('tab10')(range(N_CORNERS))
    for ci in range(N_CORNERS):
        ax_mds.scatter(mds[corner_idx[:, ci], 0],
                        mds[corner_idx[:, ci], 1],
                        facecolors='none',
                        edgecolors=[corner_edges[ci]], s=120,
                        linewidths=1.4, label=f'corner {ci + 1}')
    ax_mds.set_xlabel('MDS 1')
    ax_mds.set_ylabel('MDS 2')
    n_pop_hap = len(hm.sample_sets[POPULATION])
    ax_mds.set_title(f"Local-PCA MDS along {CHROM}\n"
                      f"({POPULATION}: n_haplotypes={n_pop_hap}, "
                      f"n_windows={res.n_windows})", fontsize=10)
    ax_mds.legend(loc='best', fontsize=8)

    # top right: MDS1 along chromosome -- single neutral colour, with
    # the corner windows circled.
    pos_mb = centers / 1e6
    ax_mds1.scatter(pos_mb, mds[:, 0],
                     c=POINT_COLOR, s=8,
                     edgecolors='white', linewidths=0.1, alpha=0.7)
    for ci in range(N_CORNERS):
        ax_mds1.scatter(pos_mb[corner_idx[:, ci]],
                         mds[corner_idx[:, ci], 0],
                         facecolors='none',
                         edgecolors=[corner_edges[ci]], s=70,
                         linewidths=1.2)
    ax_mds1.set_ylabel('MDS 1')
    ax_mds1.set_title(f"{CHROM} ({WINDOW_SIZE}-SNP non-overlapping "
                       f"windows; lostruct + Garud $H_{{12}}$)", fontsize=10)
    plt.setp(ax_mds1.get_xticklabels(), visible=False)

    # bottom right: Garud H12 along the chromosome as a filled
    # area chart -- much cleaner than a 1064-segment line at this
    # zoom level on a single-population subset.
    h12_centers_mb = df_h12['center'].to_numpy() / 1e6
    h12_vals = df_h12['garud_h12'].to_numpy()
    ax_h12.fill_between(h12_centers_mb, 0, h12_vals,
                         color='steelblue', alpha=0.5, linewidth=0)
    ax_h12.plot(h12_centers_mb, h12_vals,
                 color='steelblue', lw=0.4, alpha=0.9)
    ax_h12.set_xlabel(f'{CHROM} position (Mb)')
    ax_h12.set_ylabel(r'Garud $H_{12}$')
    ax_h12.set_xlim(h12_centers_mb.min(), h12_centers_mb.max())
    ax_h12.set_ylim(0, max(h12_vals.max() * 1.05, 0.05))

    fig.suptitle(f'pg_gpu lostruct on Ag1000G 3R ({POPULATION})',
                  fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(f"{OUT_DIR_FIG}/ag1000g_lostruct.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT_DIR_FIG}/ag1000g_lostruct.png",
                bbox_inches='tight', dpi=150)
    print(f"\nFigure saved to {OUT_DIR_FIG}/ag1000g_lostruct.pdf", flush=True)
    print(f"Tables saved to {OUT_DIR_TBL}/ag1000g_lostruct_*.csv",
          flush=True)


if __name__ == "__main__":
    main()
