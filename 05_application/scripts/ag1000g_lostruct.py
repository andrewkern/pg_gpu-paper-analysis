#!/usr/bin/env python
"""
Local PCA / lostruct on Ag1000G Phase 3 chromosome 3R.

Runs the four-step Li & Ralph (2019) pipeline on the full 3R arm
across all 2,940 phased haplotypes: per-window local PCA, Frobenius
distance between window covariance representations, classical MDS,
and corner detection in MDS space. A 1D k-means partitions windows
into baseline / intermediate / outlier regimes by MDS1 distance from
the chromosome-wide median, identifying genomic intervals whose
local sample structure deviates most from the genome-wide pattern
(e.g. inversions, large segregating SVs, or recent sweeps).

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
from scipy.cluster.vq import kmeans2

from pg_gpu import HaplotypeMatrix, lostruct, windowed_analysis
from pg_gpu.accessible import AccessibleMask

OUT_DIR_FIG = "05_application/figures"
OUT_DIR_TBL = "05_application/tables"

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
MASK_NPZ = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"
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

# Companion Garud H12 track stays on bp windows so the figure has a
# stable Mb x-axis; lostruct windows of fixed SNP count have variable
# physical width, but their per-window center is reported in bp.
GARUD_BP_WINDOW = 100_000
GARUD_BP_STEP = 50_000
K_PCS = 2
N_CORNERS = 3
CORNER_PROP = 0.05
RANDOM_STATE = 42

REGIME_NAMES = ("baseline", "intermediate", "outlier")
REGIME_COLORS = {"baseline": "#4C9AFF",
                  "intermediate": "#F2B84B",
                  "outlier":     "#D94E4E"}


def load_data():
    """Load full 3R arm and attach the canonical accessibility mask."""
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

    acc_arr = np.load(MASK_NPZ)[f"access_{CHROM}"]
    hm.set_accessible_mask(AccessibleMask(acc_arr, offset=1))

    t_load = time.time() - t0
    print(f"  {hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants "
          f"({hm.n_total_sites:,} accessible bases, {t_load:.0f}s)",
          flush=True)
    return hm


def cluster_mds1(mds1, seed=RANDOM_STATE):
    """1D k-means (k=3) on MDS1, relabeled by distance from the median.

    Closest cluster -> 'baseline' (genome-wide PCA pattern), middle
    -> 'intermediate', farthest -> 'outlier' (windows where local
    structure deviates most strongly).
    """
    centroids, labels = kmeans2(mds1.astype(np.float64), k=3,
                                 minit='++', seed=seed)
    dist = np.abs(centroids - np.median(mds1))
    rank = np.argsort(dist)
    remap = np.empty(3, dtype=np.int64)
    for rank_idx, cluster_idx in enumerate(rank):
        remap[cluster_idx] = rank_idx
    regime = np.array([REGIME_NAMES[remap[l]] for l in labels])
    return regime, centroids[rank]


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

    print("\nClustering MDS1 -> baseline / intermediate / outlier (k=3)...",
          flush=True)
    regime, regime_centroids = cluster_mds1(mds[:, 0])
    for name, c in zip(REGIME_NAMES, regime_centroids):
        print(f"  {name:13s}  MDS1 centroid={c:+.3f}  "
              f"n_windows={(regime == name).sum()}", flush=True)

    # Companion: windowed Garud's H12 in physical (bp) windows so the
    # x-axis stays uniform in Mb across the figure even though the
    # lostruct windows are SNP-defined and therefore variable in width.
    print(f"\nCompanion Garud H12 scan ({GARUD_BP_WINDOW//1000}kb / "
          f"{GARUD_BP_STEP//1000}kb bp windows)...", flush=True)
    t0 = time.time()
    df_h12 = windowed_analysis(
        hm, window_size=GARUD_BP_WINDOW, step_size=GARUD_BP_STEP,
        statistics=['garud_h12'], window_type='bp')
    cp.cuda.Stream.null.synchronize()
    print(f"  {time.time() - t0:.1f}s, n_windows={len(df_h12)}", flush=True)

    # ---------------- save tables ----------------
    win_df = pd.DataFrame({
        'start': starts,
        'end': ends,
        'center': centers,
        'mds1': mds[:, 0],
        'mds2': mds[:, 1],
        'regime': regime,
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

    # left: MDS scatter, coloured by regime
    for name in REGIME_NAMES:
        m = regime == name
        ax_mds.scatter(mds[m, 0], mds[m, 1],
                        c=REGIME_COLORS[name], s=18,
                        edgecolors='white', linewidths=0.2,
                        label=f"{name} (n={m.sum()})")
    corner_edges = plt.get_cmap('tab10')(range(N_CORNERS))
    for ci in range(N_CORNERS):
        ax_mds.scatter(mds[corner_idx[:, ci], 0],
                        mds[corner_idx[:, ci], 1],
                        facecolors='none',
                        edgecolors=[corner_edges[ci]], s=120,
                        linewidths=1.2, label=f'corner {ci + 1}')
    ax_mds.set_xlabel('MDS 1')
    ax_mds.set_ylabel('MDS 2')
    ax_mds.set_title(f"Local-PCA MDS along {CHROM}\n"
                      f"(n_haplotypes={hm.num_haplotypes}, "
                      f"n_windows={res.n_windows})", fontsize=10)
    ax_mds.legend(loc='best', fontsize=8)

    # top right: MDS1 along chromosome
    pos_mb = centers / 1e6
    for name in REGIME_NAMES:
        m = regime == name
        ax_mds1.scatter(pos_mb[m], mds[m, 0],
                         c=REGIME_COLORS[name], s=12,
                         edgecolors='white', linewidths=0.15)
    for ci in range(N_CORNERS):
        ax_mds1.scatter(pos_mb[corner_idx[:, ci]],
                         mds[corner_idx[:, ci], 0],
                         facecolors='none',
                         edgecolors=[corner_edges[ci]], s=70,
                         linewidths=1.0)
    ax_mds1.set_ylabel('MDS 1')
    ax_mds1.set_title(f"{CHROM} (lostruct {WINDOW_SIZE} SNPs / "
                       f"{STEP_SIZE} step; H12 {GARUD_BP_WINDOW//1000}kb / "
                       f"{GARUD_BP_STEP//1000}kb bp)", fontsize=10)
    plt.setp(ax_mds1.get_xticklabels(), visible=False)

    # bottom right: Garud H12 in matched windows
    h12_centers_mb = df_h12['center'].to_numpy() / 1e6
    ax_h12.plot(h12_centers_mb, df_h12['garud_h12'].to_numpy(),
                color='steelblue', lw=0.7, alpha=0.9, label='Garud $H_{12}$')
    ax_h12.set_xlabel(f'{CHROM} position (Mb)')
    ax_h12.set_ylabel('Garud $H_{12}$')
    ax_h12.set_xlim(pos_mb.min(), pos_mb.max())
    ax_h12.legend(loc='best', fontsize=8)

    fig.suptitle('pg_gpu lostruct on Ag1000G 3R',
                  fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(f"{OUT_DIR_FIG}/ag1000g_lostruct.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT_DIR_FIG}/ag1000g_lostruct.png",
                bbox_inches='tight', dpi=150)
    print(f"\nFigure saved to {OUT_DIR_FIG}/ag1000g_lostruct.pdf", flush=True)
    print(f"Tables saved to {OUT_DIR_TBL}/ag1000g_lostruct_*.csv",
          flush=True)


if __name__ == "__main__":
    main()
