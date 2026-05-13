#!/usr/bin/env python
"""
Deep pg_gpu scan of a single simulated chromosome under the Tennessen et al.
(2012) two-population out-of-Africa model (``OutOfAfrica_2T12``, populations
``AFR`` and ``EUR``), as produced by ``simulate_ooa_genome.py``.

The chromosome is stored as a tree sequence, so a full sample-by-site genotype
matrix is never materialised at once: the chromosome is streamed in genomic
*chunks*, the variants in each chunk are pulled onto the GPU as a
HaplotypeMatrix, statistics are computed, GPU memory is released, and the next
chunk is read. GPU memory therefore scales with one chunk of variants, not the
chromosome length, so the haplotype count can run into the hundreds of thousands.

What it computes
----------------
* Windowed diversity (per population: ``pi``, ``theta_w``, ``tajimas_d``,
  ``fay_wu_h``, ``normalized_fay_wu_h``, ``segregating_sites``) and divergence
  (Hudson ``fst``, ``dxy``, ``da``) at three window scales (10 kb, 100 kb, 1 Mb)
  using the full haplotype set.
* Windowed Garud's H (``h1``, ``h12``, ``h123``, ``h2h1``) and distinct-haplotype
  count, per population, on a fixed haplotype subsample (pg_gpu's Garud kernel
  uses GPU shared memory for the per-window haplotype sort and caps at ~1024
  haplotypes, so this statistic is reported on a subsample, not all haplotypes).
* Genome-wide marginal SFS per population, and a joint SFS on a haplotype
  subsample (a full joint SFS at this sample size would be billions of cells).
* LD decay: mean r^2 over pairs of common SNPs (MAF >= LD_DECAY_MIN_MAF in the
  subsample), distance-binned and pooled over several large probe regions tiling
  the mappable chromosome, per population. Plus a pairwise-r^2 heatmap of one
  ~1 Mb sub-region (common SNPs, one population) with a zoomed-in inset on the
  densest LD block.

Run inside the pg_gpu pixi environment with a free GPU, from the repo root:

    cd /home/adkern/pg_gpu && pixi shell
    cd /home/adkern/pg_gpu-paper-analysis
    CUDA_VISIBLE_DEVICES=0 python 06_simulated_genome_scan/scripts/genome_scan_ooa.py

Outputs (under 06_simulated_genome_scan/)
-----------------------------------------
    tables/windowed_stats_{10kb,100kb,1mb}.csv   per-window diversity + divergence
    tables/garud_h_10kb.csv                      per-window Garud's H (subsample)
    tables/sfs_AFR.csv, sfs_EUR.csv              genome-wide marginal SFS
    tables/ld_decay.csv                          mean r^2 + n_pairs per distance bin, per pop
    tables/chromosome_summary.json               scalar summaries
    figures/genome_scan_ooa.pdf/.png             composite: left = scan panels (pi, theta_W,
                                                 Tajima's D, Fay-Wu H*, Hudson F_ST, D_xy,
                                                 Garud's H12); right = joint SFS, LD-decay
                                                 curve, pairwise-r^2 heatmap with zoom inset
    figures/multiscale_ooa.pdf/.png              pi & Tajima's D at 10 kb / 100 kb / 1 Mb
    figures/ld_ooa.pdf/.png                      standalone LD decay + r^2 heatmap (larger)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
import tskit
import cupy as cp
from scipy.ndimage import uniform_filter1d

from pg_gpu import HaplotypeMatrix, windowed_analysis, sfs


DEFAULT_DATA_DIR = "06_simulated_genome_scan/data/ooa_2t12"
TABLES_DIR = Path("06_simulated_genome_scan/tables")
FIGURES_DIR = Path("06_simulated_genome_scan/figures")

POPS = ("AFR", "EUR")
POP_COLORS = {"AFR": "#1f77b4", "EUR": "#d62728"}

DIVERSITY_STATS = ["pi", "theta_w", "tajimas_d", "fay_wu_h",
                   "normalized_fay_wu_h", "segregating_sites"]
DIVERGENCE_STATS = ["fst", "dxy", "da"]
GARUD_STATS = ["garud_h1", "garud_h12", "garud_h123", "garud_h2h1",
               "haplotype_count"]

# (label, bp) window scales for the diversity/divergence sweep.
WINDOW_SCALES = [("10kb", 10_000), ("100kb", 100_000), ("1mb", 1_000_000)]
MAIN_SCALE = "100kb"           # scale used for the headline scan figure
GARUD_SCALE_BP = 10_000        # Garud's H windowed at this scale

# Subsamples (haplotype columns per population) for statistics whose cost or
# kernel limits make the full sample impractical.
GARUD_SUBSAMPLE = 1000         # capped by the ~1024-hap Garud kernel
JOINT_SFS_SUBSAMPLE = 200

# LD analyses run on haplotype subsamples and on common variants only.
LD_SUBSAMPLE = 5000            # haplotypes per pop for the LD analyses
LD_DECAY_MIN_MAF = 0.15        # minor-allele-frequency cutoff for the decay curve
LD_DECAY_PROBE_BP = 5_000_000  # probe-region width for LD decay (large context)
LD_DECAY_N_PROBES = 16         # probe regions spread along the mappable chromosome
LD_DECAY_MAX_SNPS = 12_000     # common-SNP cap per probe (rarely reached at MAF 0.15)
LD_HEATMAP_REGION_BP = 1_000_000  # width of the r^2-heatmap probe region
LD_HEATMAP_MIN_MAF = 0.05      # minor-allele-frequency cutoff for the r^2 heatmap
LD_HEATMAP_SUBSAMPLE = 5000    # haplotypes (one pop) for the r^2 heatmap
LD_HEATMAP_MAX_SNPS = 2000     # common-SNP cap for the r^2-heatmap matrix


# ── tree-sequence streaming ──────────────────────────────────────────────────

def population_columns(ts):
    """Map population name -> column indices into the genotype matrix.

    Genotype-matrix columns are ordered as ``ts.samples()``; each sample node
    is one haplotype. The OOA_2T12 tree sequences carry population names in
    population metadata.
    """
    name_by_id = {}
    for pop in ts.populations():
        md = pop.metadata or {}
        name_by_id[pop.id] = md.get("name", f"pop{pop.id}")
    node_pop = ts.nodes_population
    cols = {name: [] for name in name_by_id.values()}
    for col, node in enumerate(ts.samples()):
        cols[name_by_id[node_pop[node]]].append(col)
    return {k: np.asarray(v, dtype=np.int64) for k, v in cols.items() if len(v)}


def site_index_range(site_pos, left, right):
    lo = int(np.searchsorted(site_pos, left, side="left"))
    hi = int(np.searchsorted(site_pos, right, side="left"))
    return lo, hi


def read_region_arrays(ts, left, right, site_pos):
    """Pull biallelic variants in [left, right) as (genotypes int8 (n_sites,
    n_nodes), positions float64 (n_sites,)). Drops multiallelic sites."""
    lo, hi = site_index_range(site_pos, left, right)
    n_max = hi - lo
    if n_max == 0:
        return np.empty((0, ts.num_samples), np.int8), np.empty(0)
    gm = np.empty((n_max, ts.num_samples), dtype=np.int8)
    pos = np.empty(n_max, dtype=np.float64)
    k = 0
    var_right = min(float(right), ts.sequence_length)
    for var in ts.variants(left=float(left), right=var_right):
        g = var.genotypes
        if g.max() > 1:  # recurrent mutation / multiallelic -> drop (HM is 0/1)
            continue
        gm[k] = g
        pos[k] = var.site.position
        k += 1
    return gm[:k], pos[:k]


def read_region_subsample(ts, left, right, site_pos, cols):
    """Variants in [left, right) for haplotype columns `cols` (indices into the
    genotype matrix / ts.samples() order) -- decodes just those sample nodes via
    ts.variants(samples=...), so it is much cheaper than reading all haplotypes.
    Returns (gm (n_sites, len(cols)) int8, pos)."""
    lo, hi = site_index_range(site_pos, left, right)
    n_max = hi - lo
    if n_max == 0:
        return np.empty((0, len(cols)), np.int8), np.empty(0)
    cols = np.asarray(cols, dtype=np.int64)
    node_ids = np.asarray(ts.samples())[cols]
    gm = np.empty((n_max, len(cols)), dtype=np.int8)
    pos = np.empty(n_max, dtype=np.float64)
    k = 0
    var_right = min(float(right), ts.sequence_length)
    for var in ts.variants(left=float(left), right=var_right, samples=node_ids):
        g = var.genotypes
        if g.max() > 1:  # recurrent mutation / multiallelic -> drop (HM is 0/1)
            continue
        gm[k] = g
        pos[k] = var.site.position
        k += 1
    return gm[:k], pos[:k]


def read_common_variants(ts, left, right, site_pos, cols, min_maf):
    """Variants in [left, right) restricted to haplotype columns `cols`,
    biallelic, with minor-allele frequency >= min_maf within `cols`.
    Returns (gm (n_common, len(cols)) int8, pos (n_common,) float64)."""
    gm, pos = read_region_subsample(ts, left, right, site_pos, cols)
    if gm.shape[0] == 0:
        return gm, pos
    af = gm.sum(axis=1) / gm.shape[1]
    keep = np.minimum(af, 1.0 - af) >= min_maf
    return np.ascontiguousarray(gm[keep]), pos[keep]


def build_hm(gm, pos, left, right, sample_sets):
    """Construct a GPU HaplotypeMatrix for a region of variants."""
    haps = cp.asarray(gm.T)           # (haplotypes, variants)
    positions = cp.asarray(pos)
    return HaplotypeMatrix(haps, positions,
                           chrom_start=int(left), chrom_end=int(right) - 1,
                           sample_sets={k: list(v) for k, v in sample_sets.items()})


def iter_chunks(seq_length, chunk_bp, align_bp):
    """Yield (start, stop) genomic intervals aligned to multiples of align_bp."""
    windows_per_chunk = max(1, chunk_bp // align_bp)
    step = windows_per_chunk * align_bp
    start, end = 0, int(seq_length)
    while start < end:
        yield start, min(start + step, end)
        start += step


def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ── per-chromosome windowed scan ─────────────────────────────────────────────

def windowed_scan(ts, chrom, site_pos, pop_cols, chunk_bp):
    """Stream the chromosome and return:

    windows_by_scale : {label: DataFrame}  diversity+divergence per scale
    garud_df         : DataFrame           Garud's H at GARUD_SCALE_BP (subsample)
    sfs_by_pop       : {pop: ndarray}      genome-wide marginal SFS
    joint_sfs        : ndarray             joint SFS on a subsample
    """
    # Named sample sets the chunk HMs will carry: full populations, a small
    # subsample per pop for Garud's H, and a smaller subsample for joint SFS.
    sub_g = {p: pop_cols[p][:min(GARUD_SUBSAMPLE, len(pop_cols[p]))] for p in POPS}
    sub_j = {p: pop_cols[p][:min(JOINT_SFS_SUBSAMPLE, len(pop_cols[p]))] for p in POPS}
    sample_sets = {}
    for p in POPS:
        sample_sets[p] = pop_cols[p]
        sample_sets[f"{p}_g"] = sub_g[p]

    align_bp = max(bp for _, bp in WINDOW_SCALES)
    chunks = list(iter_chunks(ts.sequence_length, chunk_bp, align_bp))
    print(f"chr{chrom}: {ts.num_sites:,} sites, {ts.num_samples:,} haplotypes, "
          f"{len(chunks)} chunk(s) of <= {chunk_bp/1e6:g} Mb")

    parts = {label: [] for label, _ in WINDOW_SCALES}
    garud_parts = []
    sfs_by_pop = {p: None for p in POPS}
    joint = None

    for ci, (left, right) in enumerate(chunks):
        cur = right - left
        t_chunk = time.perf_counter()
        # Process the whole chunk into chunk-local accumulators; only merge into
        # the global ones on success, so an OOM mid-chunk (which triggers a
        # restart at a smaller sub-chunk size) can't double-count windows.
        while True:
            loc = {label: [] for label, _ in WINDOW_SCALES}
            loc_garud, loc_sfs, loc_joint = [], {p: None for p in POPS}, None
            n_sites_chunk = 0
            try:
                sub_left = left
                while sub_left < right:
                    sub_right = min(sub_left + cur, right)
                    gm, pos = read_region_arrays(ts, sub_left, sub_right, site_pos)
                    if gm.shape[0] == 0:
                        sub_left = sub_right
                        continue
                    n_sites_chunk += gm.shape[0]
                    hm = build_hm(gm, pos, sub_left, sub_right, sample_sets)

                    for label, bp in WINDOW_SCALES:
                        per_pop = {p: windowed_analysis(
                            hm, window_size=bp, step_size=bp,
                            statistics=DIVERSITY_STATS, populations=[p])
                            for p in POPS}
                        df_div = windowed_analysis(
                            hm, window_size=bp, step_size=bp,
                            statistics=DIVERGENCE_STATS, populations=list(POPS))
                        m = per_pop[POPS[0]][["start", "end", "center"]].copy()
                        m.insert(0, "chrom", str(chrom))
                        for p in POPS:
                            for s in DIVERSITY_STATS + ["n_variants"]:
                                m[f"{s}_{p}"] = per_pop[p][s].values
                        for s in DIVERGENCE_STATS:
                            m[s] = df_div[s].values
                        m = m[m[f"n_variants_{POPS[0]}"].values > 0]
                        if not m.empty:
                            loc[label].append(m.reset_index(drop=True))

                    g = windowed_analysis(hm, window_size=GARUD_SCALE_BP,
                                          step_size=GARUD_SCALE_BP,
                                          statistics=GARUD_STATS,
                                          populations=[f"{POPS[0]}_g"])
                    gm_df = g[["start", "end", "center", "n_variants"]].copy()
                    gm_df.insert(0, "chrom", str(chrom))
                    for s in GARUD_STATS:
                        gm_df[f"{s}_{POPS[0]}"] = g[s].values
                    g2 = windowed_analysis(hm, window_size=GARUD_SCALE_BP,
                                           step_size=GARUD_SCALE_BP,
                                           statistics=GARUD_STATS,
                                           populations=[f"{POPS[1]}_g"])
                    for s in GARUD_STATS:
                        gm_df[f"{s}_{POPS[1]}"] = g2[s].values
                    gm_df = gm_df[gm_df["n_variants"].values > 0]
                    if not gm_df.empty:
                        loc_garud.append(gm_df.reset_index(drop=True))

                    for p in POPS:
                        s = np.asarray(sfs.sfs(hm, population=p))
                        loc_sfs[p] = s if loc_sfs[p] is None else loc_sfs[p] + s
                    j = np.asarray(sfs.joint_sfs(hm, pop1=list(sub_j[POPS[0]]),
                                                 pop2=list(sub_j[POPS[1]])))
                    loc_joint = j if loc_joint is None else loc_joint + j

                    del hm
                    free_gpu()
                    sub_left = sub_right
            except cp.cuda.memory.OutOfMemoryError:
                del loc, loc_garud, loc_sfs, loc_joint
                free_gpu()
                if cur <= align_bp:
                    raise
                cur = max(align_bp, cur // 2)
                print(f"  chunk {ci+1}/{len(chunks)}: OOM, retrying at "
                      f"{cur/1e6:g} Mb sub-chunks")
                continue
            # success: merge chunk-local results
            for label, _ in WINDOW_SCALES:
                parts[label].extend(loc[label])
            garud_parts.extend(loc_garud)
            for p in POPS:
                if loc_sfs[p] is not None:
                    sfs_by_pop[p] = loc_sfs[p] if sfs_by_pop[p] is None else sfs_by_pop[p] + loc_sfs[p]
            if loc_joint is not None:
                joint = loc_joint if joint is None else joint + loc_joint
            if n_sites_chunk == 0:
                print(f"  chunk {ci+1}/{len(chunks)} "
                      f"[{left/1e6:.1f}-{right/1e6:.1f} Mb]: no sites")
            else:
                print(f"  chunk {ci+1}/{len(chunks)} "
                      f"[{left/1e6:.1f}-{right/1e6:.1f} Mb]: {n_sites_chunk:,} sites, "
                      f"{time.perf_counter() - t_chunk:,.1f}s")
            break

    windows_by_scale = {label: (pd.concat(parts[label], ignore_index=True)
                                if parts[label] else pd.DataFrame())
                        for label, _ in WINDOW_SCALES}
    garud_df = pd.concat(garud_parts, ignore_index=True) if garud_parts else pd.DataFrame()
    return windows_by_scale, garud_df, sfs_by_pop, joint


# ── LD analyses ──────────────────────────────────────────────────────────────

# Distance bins for the LD-decay curve (bp).
LD_BP_BINS = [0, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
              100_000, 200_000, 500_000]


def ld_decay(ts, mappable_lo, mappable_hi, site_pos, pop_cols):
    """Distance-binned LD decay: mean r^2 over pairs of common variants
    (minor-allele frequency >= LD_DECAY_MIN_MAF in the subsample), pooled over
    LD_DECAY_N_PROBES large probe regions tiling the mappable chromosome. (The
    moments-LD sigma_d^2 estimator is calibrated for the full site-frequency
    spectrum, so an MAF-restricted version of it is not meaningful -- hence a
    direct mean r^2 here.) Returns (decay_df, r2_by_pop) with r2_by_pop[pop] =
    (bin_mid_bp array, mean_r2 array)."""
    sub = {p: pop_cols[p][:min(LD_SUBSAMPLE, len(pop_cols[p]))] for p in POPS}
    bins = np.asarray(LD_BP_BINS, dtype=float)
    mids = np.sqrt(np.maximum(bins[:-1], 1.0) * bins[1:])  # geometric bin centres
    mids[0] = bins[1] / 2.0
    max_d = float(bins[-1])
    sum_r2 = {p: np.zeros(len(bins) - 1) for p in POPS}
    n_pairs = {p: np.zeros(len(bins) - 1, dtype=np.int64) for p in POPS}

    span = max(LD_DECAY_PROBE_BP, (mappable_hi - mappable_lo) // LD_DECAY_N_PROBES)
    lefts = np.unique(np.linspace(mappable_lo, max(mappable_lo, mappable_hi - span),
                                  LD_DECAY_N_PROBES).astype(int))
    for pi, left in enumerate(lefts):
        right = min(int(left) + span, mappable_hi)
        for p in POPS:
            gm, pos = read_common_variants(ts, int(left), right, site_pos, sub[p],
                                           LD_DECAY_MIN_MAF)
            if gm.shape[0] < 3:
                continue
            if gm.shape[0] > LD_DECAY_MAX_SNPS:
                pick = np.linspace(0, gm.shape[0] - 1, LD_DECAY_MAX_SNPS).astype(int)
                gm, pos = np.ascontiguousarray(gm[pick]), pos[pick]
            if p == POPS[0]:
                print(f"  LD decay probe {pi+1}/{len(lefts)} "
                      f"[{int(left)/1e6:.1f}-{right/1e6:.1f} Mb]: ~{gm.shape[0]} "
                      f"common SNPs/pop (MAF>={LD_DECAY_MIN_MAF})")
            hm = HaplotypeMatrix(cp.asarray(gm.T), cp.asarray(pos),
                                 chrom_start=int(left), chrom_end=right - 1)
            r2 = hm.pairwise_r2()
            r2 = r2.get() if hasattr(r2, "get") else np.asarray(r2)
            del hm
            free_gpu()
            iu, ju = np.triu_indices(r2.shape[0], k=1)
            d = pos[ju] - pos[iu]            # positions are sorted, ju > iu -> >= 0
            v = r2[iu, ju].astype(np.float64)
            del r2, iu, ju
            keep = np.isfinite(v) & (d <= max_d)
            d, v = d[keep], v[keep]
            idx = np.digitize(d, bins) - 1
            for b in range(len(bins) - 1):
                m = idx == b
                if m.any():
                    sum_r2[p][b] += float(v[m].sum())
                    n_pairs[p][b] += int(m.sum())

    rows, r2by = [], {}
    for p in POPS:
        mean = np.where(n_pairs[p] > 0, sum_r2[p] / np.maximum(n_pairs[p], 1), np.nan)
        r2by[p] = (mids, mean)
        for i in range(len(bins) - 1):
            rows.append({"pop": p, "bin_lo_bp": int(bins[i]), "bin_hi_bp": int(bins[i + 1]),
                         "bin_mid_bp": float(mids[i]), "mean_r2": float(mean[i]),
                         "n_pairs": int(n_pairs[p][i])})
    return pd.DataFrame(rows), r2by


def ld_heatmap(ts, site_pos, pop_cols, region):
    """Pairwise r^2 heatmap for common variants (MAF >= LD_HEATMAP_MIN_MAF) in
    `region`, on one population's subsample. Returns (r2_matrix numpy,
    snp_positions, n_haps_used)."""
    left, right = region
    cols = pop_cols[POPS[0]][:min(LD_HEATMAP_SUBSAMPLE, len(pop_cols[POPS[0]]))]
    gm, pos = read_common_variants(ts, left, right, site_pos, cols, LD_HEATMAP_MIN_MAF)
    if gm.shape[0] < 10:
        return np.zeros((0, 0)), np.empty(0), len(cols)
    if gm.shape[0] > LD_HEATMAP_MAX_SNPS:
        pick = np.linspace(0, gm.shape[0] - 1, LD_HEATMAP_MAX_SNPS).astype(int)
        gm, pos = np.ascontiguousarray(gm[pick]), pos[pick]
    print(f"  LD heatmap: {POPS[0]} chr region {left/1e6:.2f}-{right/1e6:.2f} Mb, "
          f"{gm.shape[0]} common SNPs x {len(cols)} haplotypes")
    hm = HaplotypeMatrix(cp.asarray(gm.T), cp.asarray(pos),
                         chrom_start=int(left), chrom_end=int(right) - 1)
    r2 = hm.pairwise_r2()
    r2 = r2.get() if hasattr(r2, "get") else np.asarray(r2)
    del hm
    free_gpu()
    return r2, pos, len(cols)


# ── plotting ─────────────────────────────────────────────────────────────────

SMOOTH_WINDOWS = 5  # adjacent windows averaged for the bold smoothed trace


def _smooth(y):
    return uniform_filter1d(np.where(np.isfinite(y), y, 0.0),
                            size=SMOOTH_WINDOWS, mode="nearest")


def _scan_panel(ax, x_mb, series, ylabel, title, hline=None, legend=False):
    """One genome-scan panel: faded raw trace + bold smoothed line, per series.

    `series` is a list of (values, color, label) -- one entry for a single
    track, two for a two-population comparison."""
    for y, color, label in series:
        ax.plot(x_mb, y, color=color, alpha=0.15, lw=0.4)
        ax.plot(x_mb, _smooth(y), color=color, alpha=0.95, lw=1.0, label=label)
    if hline is not None:
        ax.axhline(hline, color="0.5", lw=0.5, ls="--")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=2)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)


def _draw_ld_decay(ax, r2_by_pop, n_ld_sub, title_size=10):
    for p in POPS:
        mids, mean_r2 = r2_by_pop[p]
        ax.plot(mids, mean_r2, "o-", color=POP_COLORS[p], lw=1.4, ms=4.5, label=p)
    ax.set_xscale("log")
    ax.set_xlabel("Distance between SNPs (bp)", fontsize=8)
    ax.set_ylabel(r"mean $r^2$", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, title="pop")
    ax.set_title(f"LD decay (mean $r^2$, common SNPs MAF $\\geq$ {LD_DECAY_MIN_MAF})\n"
                 f"{n_ld_sub:,}-hap subsample/pop, {LD_DECAY_N_PROBES} probe regions",
                 fontsize=title_size, fontweight="bold", loc="left")


def _densest_block(r2, frac=0.12):
    """Index range [i0, i1) of the contiguous SNP block (size ~frac*n) with the
    highest mean within-block r^2 -- used to pick what the LD zoom shows."""
    n = r2.shape[0]
    w = max(10, int(round(frac * n)))
    if w >= n:
        return 0, n
    r2f = np.nan_to_num(r2, nan=0.0)
    # sum of the w x w diagonal block starting at k, via 2D prefix sums
    cs = np.zeros((n + 1, n + 1))
    cs[1:, 1:] = np.cumsum(np.cumsum(r2f, axis=0), axis=1)
    best_k, best_v = 0, -1.0
    for k in range(0, n - w + 1):
        s = cs[k + w, k + w] - cs[k, k + w] - cs[k + w, k] + cs[k, k]
        if s > best_v:
            best_v, best_k = s, k
    return best_k, best_k + w


def _draw_r2_heatmap(ax, r2_mat, hm_pos, chrom, region, n_hm_haps, with_inset=True,
                     title_size=10):
    if not r2_mat.size:
        ax.set_title("Pairwise $r^2$: no common SNPs in region", fontsize=title_size)
        ax.set_xticks([]); ax.set_yticks([])
        return
    r2f = np.nan_to_num(r2_mat, nan=0.0)
    left_bp, right_bp = float(hm_pos[0]), float(hm_pos[-1])
    extent = [left_bp / 1e6, right_bp / 1e6, left_bp / 1e6, right_bp / 1e6]
    im = ax.imshow(r2f.T, cmap="magma", vmin=0, vmax=1, origin="lower",
                   interpolation="none", extent=extent, aspect="equal")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$r^2$")
    cb.ax.tick_params(labelsize=6)
    ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=8)
    ax.set_ylabel(f"chr{chrom} position (Mb)", fontsize=8)
    ax.tick_params(labelsize=6)
    if with_inset:
        i0, i1 = _densest_block(r2f, frac=0.14)
        z0, z1 = float(hm_pos[i0]) / 1e6, float(hm_pos[i1 - 1]) / 1e6
        axins = ax.inset_axes([0.58, 0.03, 0.40, 0.40])
        axins.imshow(r2f[i0:i1, i0:i1].T, cmap="magma", vmin=0, vmax=1,
                     origin="lower", interpolation="none",
                     extent=[z0, z1, z0, z1], aspect="equal")
        axins.set_xticks([z0, z1]); axins.set_yticks([z0, z1])
        axins.tick_params(labelsize=5)
        axins.set_title(f"zoom {z0:.3f}-{z1:.3f} Mb", fontsize=6, fontweight="bold")
        ax.indicate_inset_zoom(axins, edgecolor="white", lw=1.0, alpha=0.9)
    ax.set_title(f"Pairwise $r^2$ ({POPS[0]}, MAF $\\geq$ {LD_HEATMAP_MIN_MAF})\n"
                 f"chr{chrom}:{region[0]/1e6:.2f}-{region[1]/1e6:.2f} Mb, "
                 f"{r2_mat.shape[0]} SNPs x {n_hm_haps:,} haps",
                 fontsize=title_size, fontweight="bold", loc="left")


def plot_composite(df_main, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   chrom, x_lo_mb, chrom_len, n_haps_per_pop, scale_label, ld_region,
                   n_garud_sub, n_joint_sub, n_ld_sub, subtitle_extra, out_base):
    """The headline composite: wide left column of genome-scan panels (faded raw
    + bold smoothed traces, with the LD probe region shaded across all panels),
    narrow right column = joint SFS / LD decay curve / pairwise-r^2 heatmap+zoom."""
    afr, eur = POPS
    x = df_main["center"].values / 1e6
    gx = garud_df["center"].values / 1e6 if not garud_df.empty else None

    sns.set_theme(style="darkgrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(18, 19))
    gs = GridSpec(7, 2, figure=fig, hspace=0.32, wspace=0.20,
                  width_ratios=[2.4, 1], left=0.055, right=0.97, top=0.945, bottom=0.04)

    scan_axes = []

    def left(i):
        ax = fig.add_subplot(gs[i, 0])
        scan_axes.append(ax)
        return ax

    _scan_panel(left(0), x,
                [(df_main[f"pi_{afr}"].values, POP_COLORS[afr], afr),
                 (df_main[f"pi_{eur}"].values, POP_COLORS[eur], eur)],
                r"$\pi$ / bp", "Nucleotide diversity", legend=True)
    _scan_panel(left(1), x,
                [(df_main[f"theta_w_{afr}"].values, POP_COLORS[afr], afr),
                 (df_main[f"theta_w_{eur}"].values, POP_COLORS[eur], eur)],
                r"$\theta_W$ / bp", "Watterson's theta")
    _scan_panel(left(2), x,
                [(df_main[f"tajimas_d_{afr}"].values, POP_COLORS[afr], afr),
                 (df_main[f"tajimas_d_{eur}"].values, POP_COLORS[eur], eur)],
                "D", "Tajima's D", hline=0.0)
    _scan_panel(left(3), x,
                [(df_main[f"normalized_fay_wu_h_{afr}"].values, POP_COLORS[afr], afr),
                 (df_main[f"normalized_fay_wu_h_{eur}"].values, POP_COLORS[eur], eur)],
                "H*", "Fay & Wu's H*", hline=0.0)
    _scan_panel(left(4), x, [(df_main["fst"].values, "#c0392b", None)],
                r"$F_{ST}$", f"Hudson $F_{{ST}}$ ({afr} vs {eur})")
    _scan_panel(left(5), x, [(df_main["dxy"].values, "#16a085", None)],
                r"$D_{xy}$", f"$D_{{xy}}$ ({afr} vs {eur})")
    ax_g = left(6)
    if gx is not None:
        _scan_panel(ax_g, gx,
                    [(garud_df[f"garud_h12_{afr}"].values, POP_COLORS[afr], afr),
                     (garud_df[f"garud_h12_{eur}"].values, POP_COLORS[eur], eur)],
                    "H12", f"Garud's $H_{{12}}$ (10 kb windows, "
                           f"{n_garud_sub:,}-hap subsample/pop)", legend=True)

    for i, ax in enumerate(scan_axes):
        ax.set_xlim(x_lo_mb, chrom_len / 1e6)
        ax.axvspan(ld_region[0] / 1e6, ld_region[1] / 1e6, color="#e74c3c",
                   alpha=0.12, zorder=0)
        if i < len(scan_axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=9)
    scan_axes[0].text((ld_region[0] + ld_region[1]) / 2e6,
                      scan_axes[0].get_ylim()[1], "LD panel", ha="center",
                      va="bottom", fontsize=7, fontstyle="italic", color="#c0392b")

    # right column: joint SFS (rows 0-1), LD decay (rows 2-3), r^2 heatmap (rows 4-6)
    ax_j = fig.add_subplot(gs[0:2, 1])
    n1 = min(40, joint.shape[0]); n2 = min(40, joint.shape[1])
    im = ax_j.imshow(np.log10(joint[:n1, :n2].T + 1.0), origin="lower",
                     aspect="equal", cmap="viridis", interpolation="nearest")
    ax_j.set_xlabel(f"{afr} derived allele count", fontsize=7)
    ax_j.set_ylabel(f"{eur} derived allele count", fontsize=7)
    ax_j.set_title(f"Joint SFS, log10(sites+1)\n({n_joint_sub}-hap subsample/pop)",
                   fontsize=9, fontweight="bold", pad=4, loc="left")
    ax_j.tick_params(labelsize=6)
    cb = plt.colorbar(im, ax=ax_j, fraction=0.046, pad=0.04, shrink=0.85)
    cb.ax.tick_params(labelsize=6)

    _draw_ld_decay(fig.add_subplot(gs[2:4, 1]), ld_r2, n_ld_sub, title_size=9)
    _draw_r2_heatmap(fig.add_subplot(gs[4:7, 1]), r2_mat, hm_pos, chrom, ld_region,
                     n_hm_haps, with_inset=True, title_size=9)

    fig.suptitle(f"pg_gpu chromosome scan -- simulated OOA_2T12 (Tennessen 2012), "
                 f"chr{chrom}\n{n_haps_per_pop:,} haplotypes/population (AFR, EUR), "
                 f"{subtitle_extra}, {scale_label} windows",
                 fontsize=12, fontweight="bold", y=0.985)
    _save(fig, out_base)


def plot_multiscale(windows_by_scale, chrom, x_lo_mb, chrom_len, out_base):
    """Same statistic at three window scales, per population -- shows the
    resolution/variance tradeoff."""
    afr, eur = POPS
    rows = [(f"pi_{afr}", rf"$\pi$/bp ({afr})", None),
            (f"pi_{eur}", rf"$\pi$/bp ({eur})", None),
            (f"tajimas_d_{afr}", f"Tajima's D ({afr})", 0.0),
            (f"tajimas_d_{eur}", f"Tajima's D ({eur})", 0.0)]
    sns.set_theme(style="darkgrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(14, 2.2 * len(rows)))
    gs = GridSpec(len(rows), 1, figure=fig, hspace=0.28, left=0.07, right=0.97,
                  top=0.93, bottom=0.06)
    scale_style = {"10kb": ("#bdbdbd", 0.5, 0.7, 1),
                   "100kb": ("#fb8072", 1.0, 0.95, 2),
                   "1mb": ("#1f78b4", 1.8, 1.0, 3)}
    for r, (col, ylabel, hline) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        for label, _ in WINDOW_SCALES:
            d = windows_by_scale[label]
            if d.empty or col not in d:
                continue
            c, lw, a, z = scale_style[label]
            ax.plot(d["center"].values / 1e6, d[col].values, color=c, lw=lw,
                    alpha=a, zorder=z, label=label)
        if hline is not None:
            ax.axhline(hline, color="0.5", lw=0.5, ls="--")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(x_lo_mb, chrom_len / 1e6)
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3, title="window size")
        if r < len(rows) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=9)
    fig.suptitle(f"Window-scale comparison -- chr{chrom}, 10 kb / 100 kb / 1 Mb",
                 fontsize=12, fontweight="bold")
    _save(fig, out_base)


def plot_ld(r2_by_pop, r2_mat, hm_pos, n_hm_haps, chrom, region,
            n_ld_sub, out_base):
    """Standalone LD figure: mean-r^2 decay (per pop) + pairwise-r^2 heatmap of
    the probe region with a zoomed-in inset on the densest LD block."""
    sns.set_theme(style="white", context="paper", font_scale=0.95)
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.25], wspace=0.28,
                  left=0.07, right=0.965, top=0.86, bottom=0.12)
    _draw_ld_decay(fig.add_subplot(gs[0, 0]), r2_by_pop, n_ld_sub, title_size=11)
    _draw_r2_heatmap(fig.add_subplot(gs[0, 1]), r2_mat, hm_pos, chrom, region,
                     n_hm_haps, with_inset=True, title_size=11)
    fig.suptitle(f"Linkage disequilibrium -- simulated OOA_2T12, chr{chrom}",
                 fontsize=12, fontweight="bold", y=0.97)
    _save(fig, out_base)


def _save(fig, out_base):
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_base}.pdf / .png")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"directory holding chr*.trees + manifest.json (default {DEFAULT_DATA_DIR})")
    p.add_argument("--chromosome", default=None,
                   help="which chr{N}.trees to scan (default: the single one found, "
                        "or error if several are present)")
    p.add_argument("--chunk-bp", type=int, default=5_000_000,
                   help="genomic chunk size streamed to the GPU at a time, bp "
                        "(default 5,000,000; the in-GPU matrix is ~haplotypes x "
                        "variants-in-chunk bytes -- lower this for very large "
                        "samples); halved automatically on OOM")
    p.add_argument("--ld-region", default=None,
                   help="region 'start-end' in bp for the pairwise-r^2 heatmap "
                        f"(default: a {LD_HEATMAP_REGION_BP//1000} kb window near "
                        "the chromosome midpoint)")
    return p.parse_args()


def pick_chromosome(data_dir, requested):
    paths = sorted(Path(data_dir).glob("chr*.trees"))
    if not paths:
        raise SystemExit(f"no chr*.trees in {data_dir} -- run simulate_ooa_genome.py first")
    if requested:
        path = Path(data_dir) / f"chr{requested}.trees"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        return requested, path
    if len(paths) > 1:
        names = ", ".join(p.stem[3:] for p in paths)
        raise SystemExit(f"multiple chromosomes in {data_dir} ({names}); pass --chromosome")
    return paths[0].stem[3:], paths[0]


def main():
    args = parse_args()
    chrom, trees_path = pick_chromosome(args.data_dir, args.chromosome)
    manifest_path = Path(args.data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {trees_path} ...")
    ts = tskit.load(str(trees_path))
    site_pos = ts.tables.sites.position
    pop_cols = population_columns(ts)
    for p in POPS:
        if p not in pop_cols:
            raise SystemExit(f"population {p} not found in {trees_path}")
    n_haps_per_pop = int(len(pop_cols[POPS[0]]))
    chrom_len = int(ts.sequence_length)
    # mappable extent: the recombination map masks the acrocentric short arm /
    # centromere (NaN-rate interval) so no polymorphism is simulated there --
    # use the variant span as the plotted / probed region.
    mappable_lo = int(site_pos.min()) if ts.num_sites else 0
    mappable_hi = (int(site_pos.max()) + 1) if ts.num_sites else chrom_len
    x_lo_mb = float(np.floor(mappable_lo / 1e6))

    t0 = time.perf_counter()
    windows_by_scale, garud_df, sfs_by_pop, joint = windowed_scan(
        ts, chrom, site_pos, pop_cols, args.chunk_bp)
    print(f"windowed scan: {time.perf_counter() - t0:,.1f}s")

    for label, _ in WINDOW_SCALES:
        windows_by_scale[label].to_csv(TABLES_DIR / f"windowed_stats_{label}.csv", index=False)
    garud_df.to_csv(TABLES_DIR / "garud_h_10kb.csv", index=False)
    for p in POPS:
        pd.DataFrame({"derived_allele_count": np.arange(len(sfs_by_pop[p])),
                      "sites": sfs_by_pop[p]}).to_csv(TABLES_DIR / f"sfs_{p}.csv", index=False)

    print("LD decay ...")
    t0 = time.perf_counter()
    ld_decay_df, ld_r2 = ld_decay(ts, mappable_lo, mappable_hi, site_pos, pop_cols)
    ld_decay_df.to_csv(TABLES_DIR / "ld_decay.csv", index=False)
    if args.ld_region:
        a, b = args.ld_region.split("-")
        region = (int(a), int(b))
    else:
        mid = (mappable_lo + mappable_hi) // 2
        half = LD_HEATMAP_REGION_BP // 2
        region = (max(mappable_lo, mid - half), min(mappable_hi, mid + half))
    print("LD r^2 heatmap ...")
    r2_mat, hm_pos, n_hm_haps = ld_heatmap(ts, site_pos, pop_cols, region)
    print(f"LD analyses: {time.perf_counter() - t0:,.1f}s")

    # scalar summaries (window-width-weighted means over the 100 kb scale)
    main_df = windows_by_scale[MAIN_SCALE]
    w = (main_df["end"] - main_df["start"]).astype(float).values
    def wmean(col):
        v = main_df[col].astype(float).values
        msk = np.isfinite(v)
        return float(np.average(v[msk], weights=w[msk])) if msk.any() else float("nan")
    summary = {
        "model": manifest.get("model", "OutOfAfrica_2T12"),
        "genetic_map": manifest.get("genetic_map"),
        "chromosome": chrom,
        "chromosome_length": chrom_len,
        "haplotypes_per_pop": n_haps_per_pop,
        "n_sites": int(ts.num_sites),
        "garud_subsample": min(GARUD_SUBSAMPLE, n_haps_per_pop),
        "joint_sfs_subsample": min(JOINT_SFS_SUBSAMPLE, n_haps_per_pop),
        "ld_subsample": min(LD_SUBSAMPLE, n_haps_per_pop),
        "ld_heatmap_min_maf": LD_HEATMAP_MIN_MAF,
        "ld_heatmap_region": list(region),
        "ld_heatmap_n_snps": int(r2_mat.shape[0]),
        "windows": {label: int(len(windows_by_scale[label])) for label, _ in WINDOW_SCALES},
        "genomewide_100kb": {
            **{f"mean_pi_{p}": wmean(f"pi_{p}") for p in POPS},
            **{f"mean_theta_w_{p}": wmean(f"theta_w_{p}") for p in POPS},
            **{f"mean_tajimas_d_{p}": wmean(f"tajimas_d_{p}") for p in POPS},
            **{f"mean_normalized_fay_wu_h_{p}": wmean(f"normalized_fay_wu_h_{p}") for p in POPS},
            "mean_fst": wmean("fst"), "mean_dxy": wmean("dxy"), "mean_da": wmean("da"),
            **{f"total_segregating_sites_{p}": float(np.nansum(main_df[f"segregating_sites_{p}"])) for p in POPS},
        },
    }
    (TABLES_DIR / "chromosome_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary (100 kb windows):")
    for k, v in summary["genomewide_100kb"].items():
        print(f"  {k}: {v:.6g}")

    plot_composite(main_df, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   chrom, x_lo_mb, chrom_len, n_haps_per_pop, MAIN_SCALE, region,
                   min(GARUD_SUBSAMPLE, n_haps_per_pop),
                   min(JOINT_SFS_SUBSAMPLE, n_haps_per_pop),
                   min(LD_SUBSAMPLE, n_haps_per_pop),
                   f"{int(ts.num_sites):,} variants",
                   str(FIGURES_DIR / "genome_scan_ooa"))
    plot_multiscale(windows_by_scale, chrom, x_lo_mb, chrom_len,
                    str(FIGURES_DIR / "multiscale_ooa"))
    plot_ld(ld_r2, r2_mat, hm_pos, n_hm_haps, chrom, region,
            min(LD_SUBSAMPLE, n_haps_per_pop), str(FIGURES_DIR / "ld_ooa"))


if __name__ == "__main__":
    main()
