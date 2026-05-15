#!/usr/bin/env python
"""
Deep pg_gpu scan of one chromosome from a VCZ-format zarr store, biobank-scale.

The store is opened as a ``StreamingHaplotypeMatrix``: pg_gpu walks the
chromosome chunk-by-chunk, pulling each chunk onto the GPU, computing
its contribution to the per-window / SFS / LD statistics, and freeing
it before the next chunk. GPU memory scales with one chunk, not the
chromosome length, so haplotype counts in the hundreds of thousands
work on a single 80 GB A100.

Every statistic below is one call on the streaming matrix. The two
that need every variant simultaneously (Garud's H haplotype hashing,
and the pairwise-r^2 heatmap) call ``streaming.materialize(...)`` to
pull a subsample-only or region-only eager matrix for that step.

For empirical biobank data, replace the simulation + ``ts_to_vcz.py``
steps with any VCZ store (a VCF run through ``bio2zarr``, gnomAD
HGDP+1KG, or a scikit-allel store converted by
``pg_gpu.zarr_io.allel_zarr_to_vcz``); the rest is unchanged.

What it computes
----------------
* Windowed diversity (per-pop: ``pi``, ``theta_w``, ``tajimas_d``,
  ``fay_wu_h``, ``normalized_fay_wu_h``, ``segregating_sites``) and
  Hudson divergence (``fst``, ``dxy``, ``da``) at three scales
  (10 kb, 100 kb, 1 Mb) using the full haplotype set.
* Genome-wide marginal SFS per pop, joint SFS on a small subsample.
* Genome-wide LD decay (DD, Dz, pi2 -- the moments-LD pair-bin
  statistics -- per pop and between pops). r^2 proxy is sigma_d^2 = DD/pi2.
* Windowed Garud's H (``h1``, ``h12``, ``h123``, ``h2h1``) per pop on a
  1000-hap subsample (pg_gpu's Garud kernel caps near 1024).
* Pairwise r^2 heatmap of one ~1 Mb sub-region (common SNPs in one
  pop), with a zoomed-in inset on the densest LD block.

Run inside the pg_gpu pixi environment with a free GPU, from the repo root:

    cd /home/adkern/pg_gpu && pixi shell
    cd /home/adkern/pg_gpu-paper-analysis
    CUDA_VISIBLE_DEVICES=0 python 06_simulated_genome_scan/scripts/genome_scan_ooa.py

Outputs (under 06_simulated_genome_scan/)
-----------------------------------------
    tables/windowed_stats_{10kb,100kb,1mb}.csv   per-window diversity + divergence
    tables/garud_h_10kb.csv                      per-window Garud's H (subsample)
    tables/sfs_AFR.csv, sfs_EUR.csv              genome-wide marginal SFS
    tables/ld_decay.csv                          per-bin moments-LD stats per pop
    tables/joint_sfs.npy, r2_heatmap.npy,
    tables/r2_heatmap_pos.npy                    caches for replot.py
    tables/chromosome_summary.json               scalar summaries
    figures/genome_scan_ooa.{pdf,png}            composite scan + LD/SFS column
    figures/multiscale_ooa.{pdf,png}             pi & Tajima's D at 3 window scales
    figures/ld_ooa.{pdf,png}                     standalone LD decay + r^2 heatmap
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

# (label, window-size-bp) for the diversity / divergence sweep.
WINDOW_SCALES = [("10kb", 10_000), ("100kb", 100_000), ("1mb", 1_000_000)]
MAIN_SCALE = "100kb"           # scale used for the headline scan figure
GARUD_SCALE_BP = 10_000        # Garud's H windowed at this scale

# Subsamples for stats that can't take the full sample axis cheaply.
GARUD_SUBSAMPLE = 1000          # capped by the ~1024-hap Garud kernel
JOINT_SFS_SUBSAMPLE = 200       # full joint SFS would be n_hap^2 cells

# LD-decay haplotype subsample per pop. Each pair-count step reads
# n_hap entries for both endpoints, so cost scales linearly with this.
# At biobank-scale per-pop sizes (~100 k haps) a full-pop pass is
# bandwidth-bound and takes hours; 5 k haps brings it to minutes
# while still giving stable mean estimates.
LD_SUBSAMPLE = 5_000

# Per-probe SNP cap for LD decay. Pair count grows quadratically with
# the variant count, so a 5 Mb probe at biobank-scale variant density
# (~120k SNPs/Mb) would give 600k SNPs and ~6 billion pairs per probe;
# the cap downsamples (uniformly along position) to a tractable count
# whose pair-bin sums are still a sound moments-LD estimate.
LD_DECAY_MAX_SNPS = 12_000

# LD pair-bin breakpoints in bp. The last entry is the maximum pair
# distance the streaming compute walks; everything farther apart is
# skipped.
LD_BP_BINS = [0, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
              100_000, 200_000]

# Pairwise-r^2 heatmap of one sub-region (eager).
LD_HEATMAP_REGION_BP = 1_000_000  # width of the heatmap region
LD_HEATMAP_MIN_MAF = 0.05         # MAF cutoff for SNPs in the heatmap
LD_HEATMAP_SUBSAMPLE = 5000       # haplotypes drawn (one pop)
LD_HEATMAP_MAX_SNPS = 2000        # cap SNPs after MAF filter


# ── tiny helpers ────────────────────────────────────────────────────────────

def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _as_list_int(idx):
    """Convert any iterable of ints (numpy array, list, range) to a
    plain ``list[int]`` -- the eager HaplotypeMatrix's sample_sets
    setter rejects non-list values."""
    return [int(i) for i in idx]


def pick_zarr(data_dir, requested):
    paths = sorted(p for p in Path(data_dir).glob("chr*.vcz") if p.is_dir())
    if not paths:
        raise SystemExit(f"no chr*.vcz/ in {data_dir} -- run ts_to_vcz.py first")
    if requested:
        path = Path(data_dir) / f"chr{requested}.vcz"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        return path
    if len(paths) > 1:
        names = ", ".join(p.stem[3:] for p in paths)
        raise SystemExit(f"multiple chromosomes in {data_dir} ({names}); "
                         "pass --chromosome")
    return paths[0]


# ── per-statistic compute steps (one streaming call each) ───────────────────

def run_one_pass_scan(stream, populations):
    """Stream the chromosome once and accumulate every reduce-by-chunk
    statistic on each per-chunk eager matrix as it arrives:

      * windowed diversity + divergence at every scale in ``WINDOW_SCALES``,
      * per-pop marginal SFS,
      * joint SFS on the small ``JOINT_SFS_SUBSAMPLE`` per pop,
      * Garud's H per pop on the ``GARUD_SUBSAMPLE`` per pop (registered
        as a temporary pop ``{pop}_g`` on each chunk).

    Doing it as a single explicit pass over ``stream.iter_gpu_chunks()``
    -- instead of one dispatch per stat -- means chr15 is read once,
    not nine times.

    Returns ``(windowed_by_scale, garud_df, marginal_sfs_by_pop, joint_sfs)``.
    """
    # Subsamples for joint SFS and Garud's H. Convert to plain lists so
    # the per-chunk sample_sets setter (which only accepts list values)
    # accepts the streaming source's numpy-array pop indices too.
    full_pop_lists = {p: _as_list_int(stream.sample_sets[p]) for p in populations}
    sub_j = {p: full_pop_lists[p][:JOINT_SFS_SUBSAMPLE] for p in populations}
    sub_g = {p: full_pop_lists[p][:GARUD_SUBSAMPLE] for p in populations}

    windowed = {label: [] for label, _ in WINDOW_SCALES}
    garud_parts = []
    marginal_sfs = {p: None for p in populations}
    joint = None

    t_scan = time.perf_counter()
    n_chunks = len(stream._chunks)
    for ci, (left, right, chunk_hm) in enumerate(stream.iter_gpu_chunks()):
        t_c0 = time.perf_counter()
        # Register the Garud subsamples as named pops on this chunk so
        # windowed_analysis can pick them up alongside the full-pop scan.
        chunk_hm.sample_sets = {**full_pop_lists,
                                **{f"{p}_g": sub_g[p] for p in populations}}

        for label, bp in WINDOW_SCALES:
            # Per-pop diversity needs one call per pop -- when single +
            # two-pop stats are combined with len(populations)>=2,
            # windowed_analysis runs single-pop stats only on
            # populations[0], not all of them.
            per_pop = [windowed_analysis(chunk_hm, window_size=bp,
                                          step_size=bp,
                                          statistics=DIVERSITY_STATS,
                                          populations=[p])
                       for p in populations]
            div = windowed_analysis(chunk_hm, window_size=bp,
                                     step_size=bp,
                                     statistics=DIVERGENCE_STATS,
                                     populations=list(populations))
            base = per_pop[0][["chrom", "start", "end", "center"]].copy()
            for i, p in enumerate(populations):
                for s in DIVERSITY_STATS + ["n_variants"]:
                    base[f"{s}_{p}"] = per_pop[i][s].values
            for s in DIVERGENCE_STATS:
                base[s] = div[s].values
            base = base[base[f"n_variants_{populations[0]}"].values > 0]
            if not base.empty:
                windowed[label].append(base.reset_index(drop=True))

        garud_per_pop = []
        for p in populations:
            df_g = windowed_analysis(chunk_hm, window_size=GARUD_SCALE_BP,
                                      step_size=GARUD_SCALE_BP,
                                      statistics=GARUD_STATS,
                                      populations=[f"{p}_g"])
            df_g = df_g.rename(columns={**{s: f"{s}_{p}" for s in GARUD_STATS},
                                          "n_variants": f"n_variants_{p}"})
            garud_per_pop.append(df_g)
        gdf = garud_per_pop[0]
        for extra in garud_per_pop[1:]:
            gdf = gdf.merge(
                extra.drop(columns=["chrom"], errors="ignore"),
                on=["start", "end", "center"], how="inner",
                suffixes=("", "_dup"),
            )
            gdf = gdf.loc[:, ~gdf.columns.str.endswith("_dup")]
        any_var = sum(gdf[f"n_variants_{p}"].values > 0 for p in populations)
        gdf = gdf[any_var > 0]
        if not gdf.empty:
            garud_parts.append(gdf.reset_index(drop=True))

        for p in populations:
            s = np.asarray(sfs.sfs(chunk_hm, population=p))
            marginal_sfs[p] = s if marginal_sfs[p] is None else marginal_sfs[p] + s
        j = np.asarray(sfs.joint_sfs(chunk_hm, pop1=sub_j[populations[0]],
                                       pop2=sub_j[populations[1]]))
        joint = j if joint is None else joint + j

        del chunk_hm
        free_gpu()
        print(f"  chunk {ci+1}/{n_chunks} "
              f"[{left/1e6:.1f}-{right/1e6:.1f} Mb] in "
              f"{time.perf_counter()-t_c0:.1f}s", flush=True)

    print(f"  one-pass total wall: {time.perf_counter()-t_scan:,.1f}s")
    windowed_out = {label: (pd.concat(parts, ignore_index=True)
                             if parts else pd.DataFrame())
                    for label, parts in windowed.items()}
    garud_df = (pd.concat(garud_parts, ignore_index=True)
                if garud_parts else pd.DataFrame())
    return windowed_out, garud_df, marginal_sfs, joint


def run_ld_decay(stream, populations, bp_bins, *,
                  subsample=5_000, n_probes=16, probe_bp=5_000_000,
                  max_snps_per_probe=12_000):
    """Moments-LD pair-bin statistics (DD, Dz, pi2) for both pops and
    the between-pop case, sampled by tiling the chromosome with
    ``n_probes`` materialized probe regions. Each probe is loaded
    eagerly with only the per-pop ``subsample`` haplotypes -- so per
    probe ~ (subsample, probe_bp * density) int8 instead of the full
    biobank-scale matrix -- and the bin sums are accumulated as raw
    moments-LD numerators across probes. The final ratio sigma_d^2 =
    sum(DD) / sum(pi2) is computed once at the end.

    This is what the streaming pair-bin function would do natively, if
    the tail-buffer stitch were subsample-aware. Until then, probe-and-
    materialize keeps the GPU memory bounded by the subsample size.
    """
    afr, eur = populations
    afr_idx = _as_list_int(stream.sample_sets[afr])[:subsample]
    eur_idx = _as_list_int(stream.sample_sets[eur])[:subsample]
    n_afr = len(afr_idx)
    sample_subset = afr_idx + eur_idx

    bins = np.asarray(bp_bins, dtype=float)
    mids = np.sqrt(np.maximum(bins[:-1], 1.0) * bins[1:])
    mids[0] = bins[1] / 2.0
    n_bins = len(bins) - 1

    cats = (afr, eur, f"{afr}_{eur}")
    cat_stats = {afr: ("DD_0_0", "Dz_0_0_0", "pi2_0_0_0_0"),
                  eur: ("DD_1_1", "Dz_1_1_1", "pi2_1_1_1_1"),
                  f"{afr}_{eur}": ("DD_0_1", "Dz_0_0_1", "pi2_0_0_1_1")}
    accum = {c: {"DD": np.zeros(n_bins), "Dz": np.zeros(n_bins),
                  "pi2": np.zeros(n_bins)} for c in cats}

    # Use the variant-bearing range rather than the chunk grid; on a
    # chromosome with a large variant-free arm (e.g. chr15's
    # acrocentric region) probes pinned to the chunk grid would land
    # in empty space and materialize 0 variants.
    pos_arr = np.asarray(stream._source.site_pos)
    lo = int(pos_arr.min())
    hi = int(pos_arr.max()) + 1
    span = max(probe_bp, (hi - lo) // n_probes)
    lefts = np.unique(np.linspace(lo, max(lo, hi - span), n_probes).astype(int))
    for pi_, left in enumerate(lefts):
        right = min(int(left) + span, hi)
        # Skip empty intervals defensively -- biobank stores can have
        # masked / unmappable runs even within the variant-bearing
        # range.
        if not ((pos_arr >= left) & (pos_arr < right)).any():
            print(f"  probe {pi_+1}/{len(lefts)} "
                  f"[{int(left)/1e6:.1f}-{right/1e6:.1f} Mb] empty, skipped",
                  flush=True)
            continue
        t0 = time.perf_counter()
        eager = stream.materialize(region=(int(left), right),
                                    sample_subset=sample_subset)
        # The subsample arrives in the order we passed it: first n_afr
        # entries are AFR, the rest are EUR. ``materialize`` reshapes
        # the sample axis to (n_dip', 2) but the pair-count kernels
        # only care about per-row haplotype identity, so the diploid
        # mock-up is harmless.
        eager.sample_sets = {afr: list(range(n_afr)),
                              eur: list(range(n_afr, n_afr + len(eur_idx)))}
        # Cap variants per probe: pair count grows quadratically and
        # at biobank-scale variant density a 5 Mb probe has hundreds
        # of thousands of SNPs. Pick ~max_snps_per_probe along the
        # variant axis (uniform indices) before the pair iteration.
        if eager.num_variants > max_snps_per_probe:
            pick = cp.linspace(0, eager.num_variants - 1,
                                max_snps_per_probe).astype(cp.int64)
            eager = HaplotypeMatrix(
                cp.ascontiguousarray(eager.haplotypes[:, pick]),
                eager.positions[pick],
                chrom_start=int(left), chrom_end=right - 1,
                sample_sets=eager.sample_sets,
            )
        result = eager.compute_ld_statistics_gpu_two_pops(
            bp_bins, pop1=afr, pop2=eur, ac_filter=True, raw=True,
        )
        for i, key in enumerate(zip(bins[:-1], bins[1:])):
            stats = result[(float(key[0]), float(key[1]))]
            for c in cats:
                k_dd, k_dz, k_pi2 = cat_stats[c]
                accum[c]["DD"][i] += stats[k_dd]
                accum[c]["Dz"][i] += stats[k_dz]
                accum[c]["pi2"][i] += stats[k_pi2]
        del eager
        free_gpu()
        print(f"  probe {pi_+1}/{len(lefts)} "
              f"[{int(left)/1e6:.1f}-{right/1e6:.1f} Mb] "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)

    rows = []
    for c in cats:
        for i, (l_bp, h_bp) in enumerate(zip(bins[:-1], bins[1:])):
            dd, dz, p2 = accum[c]["DD"][i], accum[c]["Dz"][i], accum[c]["pi2"][i]
            rows.append({"pop": c, "bin_lo_bp": int(l_bp),
                         "bin_hi_bp": int(h_bp), "bin_mid_bp": float(mids[i]),
                         "DD": dd, "Dz": dz, "pi2": p2,
                         "sigma_d2": dd / p2 if p2 != 0 else float("nan")})
    return pd.DataFrame(rows)


    bins = np.asarray(bp_bins, dtype=float)
    mids = np.sqrt(np.maximum(bins[:-1], 1.0) * bins[1:])
    mids[0] = bins[1] / 2.0
    rows = []
    # DD / Dz / pi2 for: within-pop1, within-pop2, between-pop pair.
    label_to_stats = {afr: ("DD_0_0", "Dz_0_0_0", "pi2_0_0_0_0"),
                      eur: ("DD_1_1", "Dz_1_1_1", "pi2_1_1_1_1"),
                      f"{afr}_{eur}": ("DD_0_1", "Dz_0_0_1", "pi2_0_0_1_1")}
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        stats = result[(float(lo), float(hi))]
        for label, (k_dd, k_dz, k_pi2) in label_to_stats.items():
            dd, dz, p2 = stats[k_dd], stats[k_dz], stats[k_pi2]
            rows.append({"pop": label, "bin_lo_bp": int(lo),
                         "bin_hi_bp": int(hi), "bin_mid_bp": float(mids[i]),
                         "DD": dd, "Dz": dz, "pi2": p2,
                         "sigma_d2": dd / p2 if p2 != 0 else float("nan")})
    return pd.DataFrame(rows)


def run_ld_heatmap(stream, populations, region, subsample, min_maf, max_snps):
    """Materialize one sub-region restricted to ``subsample`` haplotypes,
    drop monomorphic / low-MAF sites, cap to ``max_snps``, return its
    (n_snps, n_snps) pairwise r^2 plus the SNP positions."""
    pop_cols = list(stream.sample_sets[populations[0]][:subsample])
    eager = stream.materialize(region=region, sample_subset=pop_cols)
    haps = eager.haplotypes
    pos = eager.positions
    af = (haps > 0).sum(axis=0).astype(cp.float64) / haps.shape[0]
    maf = cp.minimum(af, 1.0 - af)
    keep = maf >= min_maf
    haps = haps[:, keep]
    pos = pos[keep]
    n_var = int(haps.shape[1])
    if n_var > max_snps:
        pick = cp.linspace(0, n_var - 1, max_snps).astype(cp.int64)
        haps = haps[:, pick]
        pos = pos[pick]
    if haps.shape[1] < 10:
        del eager
        free_gpu()
        return np.zeros((0, 0)), np.empty(0), len(pop_cols)
    hm = HaplotypeMatrix(haps, pos, chrom_start=int(region[0]),
                          chrom_end=int(region[1]) - 1)
    r2 = hm.pairwise_r2()
    r2 = r2.get() if hasattr(r2, "get") else np.asarray(r2)
    pos_host = pos.get() if hasattr(pos, "get") else np.asarray(pos)
    del hm, eager
    free_gpu()
    return r2, pos_host, len(pop_cols)


# ── plotting ────────────────────────────────────────────────────────────────

SMOOTH_WINDOWS = 5  # adjacent windows averaged for the bold smoothed trace


def _smooth(y):
    return uniform_filter1d(np.where(np.isfinite(y), y, 0.0),
                            size=SMOOTH_WINDOWS, mode="nearest")


def _scan_panel(ax, x_mb, series, ylabel, title, hline=None, legend=False):
    """One genome-scan panel: faded raw trace + bold smoothed line, per
    series. ``series`` is a list of ``(values, color, label)``."""
    for y, color, label in series:
        ax.plot(x_mb, y, color=color, alpha=0.15, lw=0.4)
        ax.plot(x_mb, _smooth(y), color=color, alpha=0.95, lw=1.0, label=label)
    if hline is not None:
        ax.axhline(hline, color="0.5", lw=0.5, ls="--")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=2)
    ax.tick_params(labelsize=9)
    if legend:
        ax.legend(loc="upper right", fontsize=10, ncol=2, framealpha=0.9)


def _draw_ld_decay(ax, ld_r2, n_ld_sub, title_size=10, title=None):
    """Per-pop sigma_d^2 vs distance from the LD-decay table."""
    for p in POPS:
        mids, sig = ld_r2[p]
        ax.plot(mids, sig, "o-", color=POP_COLORS[p], lw=1.5, ms=5, label=p)
    ax.set_xscale("log")
    ax.set_xlabel("Distance between SNPs (bp)", fontsize=10)
    ax.set_ylabel(r"$\sigma_d^2 = DD / \pi^2$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, title="population", title_fontsize=9)
    if title is None:
        title = (f"LD decay (moments-LD $\\sigma_d^2$, common SNPs)\n"
                 f"{n_ld_sub:,} haps/pop")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left")


def _densest_block(r2, frac=0.12):
    """Index range ``[i0, i1)`` of the contiguous SNP block of size
    ``~frac * n`` with the highest mean within-block r^2."""
    n = r2.shape[0]
    w = max(10, int(round(frac * n)))
    if w >= n:
        return 0, n
    r2f = np.nan_to_num(r2, nan=0.0)
    cs = np.zeros((n + 1, n + 1))
    cs[1:, 1:] = np.cumsum(np.cumsum(r2f, axis=0), axis=1)
    best_k, best_v = 0, -1.0
    for k in range(0, n - w + 1):
        s = cs[k + w, k + w] - cs[k, k + w] - cs[k + w, k] + cs[k, k]
        if s > best_v:
            best_v, best_k = s, k
    return best_k, best_k + w


def _draw_r2_heatmap(ax, r2_mat, hm_pos, chrom, region, n_hm_haps,
                     with_inset=True, title_size=10, title=None):
    if not r2_mat.size:
        ax.set_title("Pairwise $r^2$: no common SNPs in region",
                     fontsize=title_size)
        ax.set_xticks([]); ax.set_yticks([])
        return
    r2f = np.nan_to_num(r2_mat, nan=0.0)
    left_bp, right_bp = float(hm_pos[0]), float(hm_pos[-1])
    extent = [left_bp / 1e6, right_bp / 1e6, left_bp / 1e6, right_bp / 1e6]
    im = ax.imshow(r2f.T, cmap="magma", vmin=0, vmax=1, origin="lower",
                   interpolation="none", extent=extent, aspect="equal")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$r^2$")
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=10)
    ax.set_ylabel(f"chr{chrom} position (Mb)", fontsize=10)
    ax.tick_params(labelsize=8)
    if with_inset:
        i0, i1 = _densest_block(r2f, frac=0.14)
        z0 = float(hm_pos[i0]) / 1e6
        z1 = float(hm_pos[i1 - 1]) / 1e6
        axins = ax.inset_axes([0.58, 0.03, 0.40, 0.40])
        axins.imshow(r2f[i0:i1, i0:i1].T, cmap="magma", vmin=0, vmax=1,
                     origin="lower", interpolation="none",
                     extent=[z0, z1, z0, z1], aspect="equal")
        axins.set_xticks([z0, z1]); axins.set_yticks([z0, z1])
        axins.tick_params(labelsize=7)
        axins.set_title(f"zoom {z0:.3f}-{z1:.3f} Mb", fontsize=8,
                        fontweight="bold")
        ax.indicate_inset_zoom(axins, edgecolor="white", lw=1.0, alpha=0.9)
    if title is None:
        title = (f"Pairwise $r^2$ ({POPS[0]}, MAF $\\geq$ {LD_HEATMAP_MIN_MAF})\n"
                 f"chr{chrom}:{region[0]/1e6:.2f}-{region[1]/1e6:.2f} Mb, "
                 f"{r2_mat.shape[0]} SNPs x {n_hm_haps:,} haps")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left")


def plot_composite(df_main, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   chrom, x_lo_mb, chrom_len, n_haps_per_pop, scale_label,
                   ld_region, n_garud_sub, n_joint_sub, n_ld_sub,
                   subtitle_extra, out_base):
    afr, eur = POPS
    x = df_main["center"].values / 1e6
    gx = garud_df["center"].values / 1e6 if not garud_df.empty else None
    div_color = "0.15"

    sns.set_theme(style="darkgrid", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(17, 12))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2.4, 1], wspace=0.16,
                     left=0.05, right=0.97, top=0.93, bottom=0.06)
    left_gs = outer[0, 0].subgridspec(7, 1, hspace=0.28)
    right_gs = outer[0, 1].subgridspec(3, 1, height_ratios=[2, 2, 3], hspace=0.20)

    scan_axes = []
    def left(i):
        ax = fig.add_subplot(left_gs[i, 0])
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
    ax_g = left(4)
    if gx is not None:
        _scan_panel(ax_g, gx,
                    [(garud_df[f"garud_h12_{afr}"].values, POP_COLORS[afr], afr),
                     (garud_df[f"garud_h12_{eur}"].values, POP_COLORS[eur], eur)],
                    "H12", f"Garud's $H_{{12}}$ (10 kb windows, "
                           f"{n_garud_sub:,}-hap subsample/pop)")
    _scan_panel(left(5), x, [(df_main["fst"].values, div_color, None)],
                r"$F_{ST}$", f"Hudson $F_{{ST}}$ ({afr} vs {eur})")
    _scan_panel(left(6), x, [(df_main["dxy"].values, div_color, None)],
                r"$D_{xy}$", f"$D_{{xy}}$ ({afr} vs {eur})")

    for i, ax in enumerate(scan_axes):
        ax.set_xlim(x_lo_mb, chrom_len / 1e6)
        ax.axvspan(ld_region[0] / 1e6, ld_region[1] / 1e6, color="#e74c3c",
                   alpha=0.12, zorder=0)
        if i < len(scan_axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=10)
    scan_axes[0].text((ld_region[0] + ld_region[1]) / 2e6,
                      scan_axes[0].get_ylim()[1], "LD panel", ha="center",
                      va="bottom", fontsize=10, fontweight="bold",
                      color="#c0392b",
                      bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                ec="#c0392b", lw=0.8, alpha=0.9))

    ax_j = fig.add_subplot(right_gs[0, 0])
    n1 = min(40, joint.shape[0]); n2 = min(40, joint.shape[1])
    im = ax_j.imshow(np.log10(joint[:n1, :n2].T + 1.0), origin="lower",
                     aspect="equal", cmap="viridis", interpolation="nearest")
    ax_j.set_xlabel(f"{afr} derived allele count", fontsize=8)
    ax_j.set_ylabel(f"{eur} derived allele count", fontsize=8)
    ax_j.set_title("Joint SFS", fontsize=11, fontweight="bold", pad=4, loc="left")
    ax_j.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax_j, fraction=0.046, pad=0.04, shrink=0.85)
    cb.ax.tick_params(labelsize=7)

    _draw_ld_decay(fig.add_subplot(right_gs[1, 0]), ld_r2, n_ld_sub,
                   title_size=11, title=r"LD decay ($\sigma_d^2$)")
    _draw_r2_heatmap(fig.add_subplot(right_gs[2, 0]), r2_mat, hm_pos, chrom,
                     ld_region, n_hm_haps, with_inset=True, title_size=11,
                     title="")

    fig.suptitle(f"pg_gpu chromosome scan -- simulated OOA_2T12 (Tennessen 2012), "
                 f"chr{chrom}: {n_haps_per_pop:,} haplotypes/population, "
                 f"{subtitle_extra}, {scale_label} windows",
                 fontsize=13, fontweight="bold", y=0.985)
    _save(fig, out_base)


def plot_multiscale(windows_by_scale, chrom, x_lo_mb, chrom_len, out_base):
    afr, eur = POPS
    rows = [(f"pi_{afr}", rf"$\pi$/bp ({afr})", None),
            (f"pi_{eur}", rf"$\pi$/bp ({eur})", None),
            (f"tajimas_d_{afr}", f"Tajima's D ({afr})", 0.0),
            (f"tajimas_d_{eur}", f"Tajima's D ({eur})", 0.0)]
    sns.set_theme(style="darkgrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(14, 2.2 * len(rows)))
    gs = GridSpec(len(rows), 1, figure=fig, hspace=0.28, left=0.07,
                  right=0.97, top=0.93, bottom=0.06)
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
            ax.legend(loc="upper right", fontsize=8, ncol=3,
                      title="window size")
        if r < len(rows) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=9)
    fig.suptitle(f"Window-scale comparison -- chr{chrom}, 10 kb / 100 kb / 1 Mb",
                 fontsize=12, fontweight="bold")
    _save(fig, out_base)


def plot_ld(ld_r2, r2_mat, hm_pos, n_hm_haps, chrom, region, n_ld_sub,
            out_base):
    sns.set_theme(style="white", context="paper", font_scale=0.95)
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.25], wspace=0.28,
                  left=0.07, right=0.965, top=0.86, bottom=0.12)
    _draw_ld_decay(fig.add_subplot(gs[0, 0]), ld_r2, n_ld_sub, title_size=11)
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


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"directory holding chr*.vcz + chr*.pops.tsv "
                        f"(default {DEFAULT_DATA_DIR})")
    p.add_argument("--zarr", default=None,
                   help="explicit path to a single VCZ store (overrides "
                        "--data-dir / --chromosome)")
    p.add_argument("--pop-file", default=None,
                   help="sample_id<TAB>population TSV (default: "
                        "<store>.pops.tsv next to the store)")
    p.add_argument("--chromosome", default=None,
                   help="contig name to scan (default: the single chr*.vcz "
                        "in --data-dir)")
    p.add_argument("--chunk-bp", type=int, default=5_000_000,
                   help="genomic chunk size streamed to the GPU at a time "
                        "(default 5,000,000)")
    p.add_argument("--prefetch", type=int, default=1,
                   help="zarr-read prefetch depth (default 1)")
    p.add_argument("--ld-region", default=None,
                   help="region 'start-end' bp for the r^2 heatmap "
                        f"(default: ~{LD_HEATMAP_REGION_BP//1000} kb near "
                        "chromosome midpoint)")
    p.add_argument("--tables-dir", default=str(TABLES_DIR))
    p.add_argument("--figures-dir", default=str(FIGURES_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = (Path(args.zarr) if args.zarr
                 else pick_zarr(args.data_dir, args.chromosome))
    pop_file = Path(args.pop_file) if args.pop_file else (
        zarr_path.parent / f"{zarr_path.stem}.pops.tsv")
    if not pop_file.exists():
        raise SystemExit(f"no pop file at {pop_file}; pass --pop-file")

    manifest_path = zarr_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tables_dir = Path(args.tables_dir); tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {zarr_path} as streaming (chunk_bp={args.chunk_bp:,})")
    stream = HaplotypeMatrix.from_zarr(
        str(zarr_path),
        streaming="always",
        chunk_bp=args.chunk_bp,
        prefetch=args.prefetch,
        pop_file=str(pop_file),
    )
    for p in POPS:
        if p not in stream.sample_sets:
            raise SystemExit(f"population {p} not in {pop_file} "
                             f"(have: {sorted(stream.sample_sets.keys())})")
    n_haps_per_pop = len(stream.sample_sets[POPS[0]])
    chrom_len = stream.chrom_end
    x_lo_mb = float(np.floor(stream.chrom_start / 1e6))
    print(f"  contig {stream.chrom}: {stream.num_variants:,} variants, "
          f"{n_haps_per_pop:,} haplotypes/pop, "
          f"mappable {stream.chrom_start/1e6:.1f}-{chrom_len/1e6:.1f} Mb")

    # Single chromosome pass: per-window diversity + divergence at every
    # scale, marginal + joint SFS, and Garud's H per pop. All these
    # stats reduce by chunk, so one walk through iter_gpu_chunks() does
    # them all.
    print("Single-pass scan: windowed + SFS + Garud ...")
    windows_by_scale, garud_df, sfs_by_pop, joint = run_one_pass_scan(
        stream, POPS)
    for label, _ in WINDOW_SCALES:
        windows_by_scale[label].to_csv(
            tables_dir / f"windowed_stats_{label}.csv", index=False)
    garud_df.to_csv(tables_dir / "garud_h_10kb.csv", index=False)
    for p in POPS:
        pd.DataFrame({"derived_allele_count": np.arange(len(sfs_by_pop[p])),
                      "sites": sfs_by_pop[p]}).to_csv(
            tables_dir / f"sfs_{p}.csv", index=False)
    np.save(tables_dir / "joint_sfs.npy", joint)

    # LD decay sampled by tiling the chromosome with probe regions,
    # each materialized at the per-pop subsample (the streaming
    # tail-buffer LD path is not subsample-aware at the data level,
    # so a probe-and-materialize pattern keeps GPU memory bounded).
    print(f"LD decay ({LD_SUBSAMPLE}-hap subsample/pop, probe regions) ...")
    t0 = time.perf_counter()
    ld_decay_df = run_ld_decay(stream, POPS, LD_BP_BINS,
                                subsample=LD_SUBSAMPLE,
                                max_snps_per_probe=LD_DECAY_MAX_SNPS)
    print(f"  {len(ld_decay_df)} bin x pop entries "
          f"in {time.perf_counter()-t0:.1f}s")
    ld_decay_df.to_csv(tables_dir / "ld_decay.csv", index=False)
    ld_r2 = {p: (ld_decay_df.loc[ld_decay_df["pop"] == p, "bin_mid_bp"].to_numpy(),
                 ld_decay_df.loc[ld_decay_df["pop"] == p, "sigma_d2"].to_numpy())
             for p in POPS}

    # Pairwise r^2 heatmap of a 1 Mb sub-region (materialized).
    print("LD r^2 heatmap ...")
    t0 = time.perf_counter()
    if args.ld_region:
        a, b = args.ld_region.split("-")
        region = (int(a), int(b))
    else:
        # Center on the variant-bearing range, not the chunk grid, so
        # a long variant-free arm doesn't put the heatmap in empty space.
        pos_arr = np.asarray(stream._source.site_pos)
        v_lo, v_hi = int(pos_arr.min()), int(pos_arr.max()) + 1
        mid = (v_lo + v_hi) // 2
        half = LD_HEATMAP_REGION_BP // 2
        region = (max(v_lo, mid - half), min(v_hi, mid + half))
    r2_mat, hm_pos, n_hm_haps = run_ld_heatmap(
        stream, POPS, region, LD_HEATMAP_SUBSAMPLE,
        LD_HEATMAP_MIN_MAF, LD_HEATMAP_MAX_SNPS,
    )
    np.save(tables_dir / "r2_heatmap.npy", r2_mat)
    np.save(tables_dir / "r2_heatmap_pos.npy", hm_pos)
    print(f"  {r2_mat.shape[0]} SNPs x {n_hm_haps:,} haps "
          f"in {time.perf_counter()-t0:.1f}s")

    # Headline summary.
    main_df = windows_by_scale[MAIN_SCALE]
    w = (main_df["end"] - main_df["start"]).astype(float).values
    def wmean(col):
        v = main_df[col].astype(float).values
        msk = np.isfinite(v)
        return float(np.average(v[msk], weights=w[msk])) if msk.any() else float("nan")
    summary = {
        "model": manifest.get("model", "OutOfAfrica_2T12"),
        "genetic_map": manifest.get("genetic_map"),
        "chromosome": stream.chrom,
        "chromosome_length": chrom_len,
        "haplotypes_per_pop": n_haps_per_pop,
        "n_sites": int(stream.num_variants),
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
            **{f"mean_normalized_fay_wu_h_{p}":
                   wmean(f"normalized_fay_wu_h_{p}") for p in POPS},
            "mean_fst": wmean("fst"), "mean_dxy": wmean("dxy"),
            "mean_da": wmean("da"),
            **{f"total_segregating_sites_{p}":
                   float(np.nansum(main_df[f"segregating_sites_{p}"])) for p in POPS},
        },
    }
    (tables_dir / "chromosome_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary (100 kb windows):")
    for k, v in summary["genomewide_100kb"].items():
        print(f"  {k}: {v:.6g}")

    n_ld_sub = min(LD_SUBSAMPLE, n_haps_per_pop)
    plot_composite(main_df, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   stream.chrom, x_lo_mb, chrom_len, n_haps_per_pop, MAIN_SCALE,
                   region, min(GARUD_SUBSAMPLE, n_haps_per_pop),
                   min(JOINT_SFS_SUBSAMPLE, n_haps_per_pop), n_ld_sub,
                   f"{stream.num_variants:,} variants",
                   str(figures_dir / "genome_scan_ooa"))
    plot_multiscale(windows_by_scale, stream.chrom, x_lo_mb, chrom_len,
                    str(figures_dir / "multiscale_ooa"))
    plot_ld(ld_r2, r2_mat, hm_pos, n_hm_haps, stream.chrom, region,
            n_ld_sub, str(figures_dir / "ld_ooa"))


if __name__ == "__main__":
    main()
