#!/usr/bin/env python
"""
Deep pg_gpu scan of a single chromosome from a VCZ-format zarr store -- the
standard biobank-scale per-sample genotype representation (sgkit / bio2zarr /
our companion ``ts_to_vcz.py`` output). Empirical biobank data lands here too:
a VCF run through bio2zarr or a gnomAD HGDP+1KG-style zarr release drops in
without changes; the only piece that's not in VCZ is the sample-to-population
map, which we read from a companion ``sample_id<TAB>population`` TSV.

The chromosome is streamed in genomic chunks: each chunk's variants are pulled
out of the zarr store as a dense ``(haplotypes x variants)`` matrix on the GPU,
the windowed / SFS / LD statistics for that chunk are computed, the GPU buffer
is released, and the next chunk is read. **GPU memory scales with one chunk,
not the chromosome length**, so haplotype counts in the hundreds of thousands
work on a single 80 GB GPU. For statistics that only need a haplotype subsample
(Garud's H, joint SFS, the LD analyses) the per-haplotype reads go through
zarr's ``oindex`` so host memory scales with the subsample, not the full
sample axis.

The OOA_2T12 simulation in this repo flows:

    simulate_ooa_genome.py  ->  data/<run>/chr15.trees
    ts_to_vcz.py            ->  data/<run>/chr15.vcz  +  chr15.pops.tsv
    genome_scan_ooa.py      ->  figures + tables

For real empirical data the first two steps are replaced by any VCZ store
(e.g. a VCF run through ``bio2zarr``) and a companion pop file -- the rest of
this script is unchanged.

What it computes
----------------
* Windowed diversity (per population: ``pi``, ``theta_w``, ``tajimas_d``,
  ``fay_wu_h``, ``normalized_fay_wu_h``, ``segregating_sites``) and divergence
  (Hudson ``fst``, ``dxy``, ``da``) at three window scales (10 kb, 100 kb, 1 Mb)
  using the full haplotype set.
* Windowed Garud's H (``h1``, ``h12``, ``h123``, ``h2h1``) and distinct-haplotype
  count, per population, on a 1000-hap subsample (pg_gpu's Garud kernel caps at
  ~1024 haplotypes).
* Genome-wide marginal SFS per population, and a joint SFS on a 200-hap
  subsample.
* LD decay: mean r^2 over pairs of common SNPs (MAF >= LD_DECAY_MIN_MAF in the
  subsample), distance-binned and pooled over several large probe regions
  tiling the mappable chromosome, per population. Plus a pairwise-r^2 heatmap
  of one ~1 Mb sub-region (common SNPs, one population) with a zoomed-in inset
  on the densest LD block.

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
    tables/joint_sfs.npy, r2_heatmap.npy,
    tables/r2_heatmap_pos.npy                    caches for replot.py
    tables/chromosome_summary.json               scalar summaries
    figures/genome_scan_ooa.pdf/.png             composite: left scan column + right
                                                 column = joint SFS / LD decay /
                                                 r^2 heatmap with zoom inset
    figures/multiscale_ooa.pdf/.png              pi & Tajima's D at 10 kb / 100 kb / 1 Mb
    figures/ld_ooa.pdf/.png                      standalone LD decay + r^2 heatmap
"""

import argparse
import json
import queue
import threading
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
import zarr
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


# ── zarr-backed chromosome source ───────────────────────────────────────────

class ZarrSource:
    """One contig of a VCZ-format zarr store, with sample-to-population
    bookkeeping.

    Wraps the on-disk array layout so the rest of the script speaks in terms
    of genomic regions and haplotype-column indices. The two read paths:

    * ``slice_region(left, right)`` reads every haplotype for variants in a
      bp range -- used by the full-sample windowed scan.
    * ``slice_subsample(left, right, hap_cols)`` reads only the requested
      haplotype columns via zarr's orthogonal indexing -- used by stats that
      only need a subsample (Garud's H, joint SFS, LD probes). The host array
      it returns is shaped ``(n_var, len(hap_cols))``, not the full sample
      axis, so RAM scales with the subsample.

    Haplotype-column layout matches pg_gpu's convention: hap ``0..n_dip-1`` is
    each diploid's first ploidy slot, hap ``n_dip..2*n_dip-1`` is the second.
    """

    def __init__(self, zarr_path, pop_file=None, contig_id=None):
        self.path = Path(zarr_path)
        self.store = zarr.open_group(str(self.path), mode="r")
        contigs = [str(c) for c in np.asarray(self.store["contig_id"])]
        if contig_id is None:
            if len(contigs) != 1:
                raise SystemExit(f"multiple contigs in {self.path} "
                                 f"({contigs}); pass --chromosome")
            contig_id = contigs[0]
        if contig_id not in contigs:
            raise SystemExit(f"contig {contig_id!r} not in {self.path} "
                             f"(available: {contigs})")
        self.chrom = contig_id
        self.contig_idx = contigs.index(contig_id)
        if "contig_length" in self.store:
            self.chrom_length = int(np.asarray(self.store["contig_length"])[self.contig_idx])
        else:
            self.chrom_length = int(np.asarray(self.store["variant_position"]).max()) + 1

        # Variant axis: restrict to this contig (here the VCZ is single-contig
        # so this is the identity, but the bookkeeping generalises).
        var_contig_all = np.asarray(self.store["variant_contig"])
        var_pos_all = np.asarray(self.store["variant_position"]).astype(np.float64)
        contig_mask = var_contig_all == self.contig_idx
        self._zarr_var_indices = np.where(contig_mask)[0]
        if self._zarr_var_indices.size == 0:
            raise SystemExit(f"no variants for contig {self.chrom} in {self.path}")
        self.site_pos = var_pos_all[contig_mask]
        # "Mappable" extent = the variant span -- real data is filtered by
        # the upstream callset / accessibility mask, simulated data is
        # filtered by the recombination map (NaN intervals get no mutations).
        self.mappable_lo = int(self.site_pos.min())
        self.mappable_hi = int(self.site_pos.max()) + 1

        cg = self.store["call_genotype"]
        if cg.ndim != 3 or cg.shape[2] != 2:
            raise SystemExit(f"expected diploid (n_var, n_samples, 2) call_genotype, "
                             f"got shape {cg.shape}")
        self.n_dip = int(cg.shape[1])
        self.num_haplotypes = 2 * self.n_dip

        self.sample_ids = list(np.asarray(self.store["sample_id"]))

        if pop_file is None:
            candidates = [
                self.path.with_suffix(".pops.tsv"),
                self.path.parent / (self.path.stem + ".pops.tsv"),
                self.path.parent / "pops.tsv",
            ]
            for c in candidates:
                if c.exists():
                    pop_file = c
                    break
            else:
                raise SystemExit(
                    f"no companion pop file found next to {self.path}; "
                    f"pass --pop-file. Tried: {[str(c) for c in candidates]}")
        self.pop_file = Path(pop_file)
        self.pop_cols = self._load_pop_file(self.pop_file)

    @property
    def num_variants(self):
        return int(self._zarr_var_indices.size)

    def _load_pop_file(self, path):
        """Parse a TSV of ``sample_id<TAB>population`` (header optional) and
        return ``{pop_name: haplotype-column indices}`` aligned with the
        store's sample order."""
        sample_to_pop = {}
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2 or parts[0].lower() == "sample_id":
                    continue
                sample_to_pop[parts[0]] = parts[1]
        pop_haps = {}
        for di, sid in enumerate(self.sample_ids):
            p = sample_to_pop.get(sid)
            if p is None:
                continue
            pop_haps.setdefault(p, []).append(di)
            pop_haps[p].append(di + self.n_dip)
        if not pop_haps:
            raise SystemExit(f"no samples from {path} matched the store's sample_ids "
                             f"(first store sample: {self.sample_ids[0]!r})")
        return {p: np.asarray(sorted(v), dtype=np.int64) for p, v in pop_haps.items()}

    def _site_index_range(self, left, right):
        """``[lo, hi)`` into ``self.site_pos`` (and the zarr variant axis,
        for this contig) such that ``left <= pos < right``."""
        lo = int(np.searchsorted(self.site_pos, left, side="left"))
        hi = int(np.searchsorted(self.site_pos, right, side="left"))
        return lo, hi

    def slice_region(self, left, right):
        """All-haplotype slice. Returns
        ``(gm (n_biallelic, num_haplotypes) int8, pos (n_biallelic,) float64)``.
        Multiallelic / masked rows (gt = -1) are dropped."""
        lo, hi = self._site_index_range(left, right)
        if hi <= lo:
            return np.empty((0, self.num_haplotypes), np.int8), np.empty(0)
        zlo = int(self._zarr_var_indices[lo])
        zhi = int(self._zarr_var_indices[hi - 1]) + 1
        gt = np.asarray(self.store["call_genotype"][zlo:zhi])   # (n, n_dip, 2) int8
        pos = self.site_pos[lo:hi].copy()
        biallelic = gt[:, 0, 0] >= 0
        gt = gt[biallelic]
        pos = pos[biallelic]
        n_var = gt.shape[0]
        if n_var == 0:
            return np.empty((0, self.num_haplotypes), np.int8), np.empty(0)
        # Lay out as (n_var, 2 * n_dip) -- ploidy 0 then ploidy 1.
        gm = np.empty((n_var, self.num_haplotypes), dtype=np.int8)
        gm[:, : self.n_dip] = gt[:, :, 0]
        gm[:, self.n_dip:] = gt[:, :, 1]
        return gm, pos

    def slice_subsample(self, left, right, hap_cols):
        """Variants in ``[left, right)`` restricted to the given haplotype
        columns (indices into the ``(2*n_dip,)`` haplotype axis).

        Uses zarr's ``oindex`` to pull only the requested diploid columns.
        With our VCZ chunked across the full sample axis the underlying
        decompression still touches the full sample chunk, but the returned
        numpy array is only ``(n_var, len(unique_dips), 2)``, so host RAM
        scales with the subsample size and not 2 * n_dip."""
        lo, hi = self._site_index_range(left, right)
        hap_cols = np.asarray(hap_cols, dtype=np.int64)
        if hi <= lo:
            return np.empty((0, len(hap_cols)), np.int8), np.empty(0)
        zlo = int(self._zarr_var_indices[lo])
        zhi = int(self._zarr_var_indices[hi - 1]) + 1
        is_p1 = hap_cols >= self.n_dip
        dip_idx = np.where(is_p1, hap_cols - self.n_dip, hap_cols)
        ploidy = is_p1.astype(np.int64)
        unique_dips, inv = np.unique(dip_idx, return_inverse=True)
        gt = np.asarray(
            self.store["call_genotype"].oindex[zlo:zhi, unique_dips, :]
        )                                                       # (n, len(unique_dips), 2)
        pos = self.site_pos[lo:hi].copy()
        biallelic = gt[:, 0, 0] >= 0
        gt = gt[biallelic]
        pos = pos[biallelic]
        n_var = gt.shape[0]
        if n_var == 0:
            return np.empty((0, len(hap_cols)), np.int8), np.empty(0)
        # Advanced indexing: pick (column-in-unique, ploidy-slot) per output column.
        gm = gt[np.arange(n_var)[:, None], inv[None, :], ploidy[None, :]]
        return gm, pos


# ── helpers reused below ────────────────────────────────────────────────────

def read_common_variants(source, left, right, hap_cols, min_maf):
    """Variants in ``[left, right)`` restricted to ``hap_cols``, biallelic,
    with minor-allele frequency >= ``min_maf`` within those columns."""
    gm, pos = source.slice_subsample(left, right, hap_cols)
    if gm.shape[0] == 0:
        return gm, pos
    af = gm.sum(axis=1) / gm.shape[1]
    keep = np.minimum(af, 1.0 - af) >= min_maf
    return np.ascontiguousarray(gm[keep]), pos[keep]


def build_hm(gm, pos, left, right, sample_sets):
    """Build a GPU HaplotypeMatrix from a chunk's ``(n_var, n_hap)`` array.

    The transpose to ``(n_hap, n_var)`` is done on the GPU rather than the
    host: numpy's strided int8 transpose of a tall-skinny chunk is single-
    threaded and cache-thrashes (measured ~80 MB/s here), whereas a PCIe
    Gen4 upload of the row-major chunk (~25 GB/s) followed by cupy's tiled
    transpose kernel is orders of magnitude faster."""
    gm_gpu = cp.asarray(gm)                       # (n_var, n_hap), host -> device
    haps = cp.ascontiguousarray(gm_gpu.T)         # (n_hap, n_var) on device
    del gm_gpu                                    # free the row-major staging copy
    positions = cp.asarray(pos)
    return HaplotypeMatrix(haps, positions,
                           chrom_start=int(left), chrom_end=int(right) - 1,
                           sample_sets={k: list(v) for k, v in sample_sets.items()})


def iter_chunks(seq_length, chunk_bp, align_bp, start=0):
    """Yield ``(left, right)`` genomic intervals aligned to multiples of
    ``align_bp`` (the largest window size, so windows can never straddle a
    chunk boundary)."""
    windows_per_chunk = max(1, chunk_bp // align_bp)
    step = windows_per_chunk * align_bp
    end = int(seq_length)
    while start < end:
        yield start, min(start + step, end)
        start += step


def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def chunk_iterator(source, chunks, prefetch):
    """Yield ``(ci, left, right, gm, pos, t_read_s)`` for each chunk.

    ``prefetch=0`` reads chunks serially in the main thread (each iteration
    blocks until the read completes, then hands off to compute).
    ``prefetch>=1`` launches a producer thread that reads the next chunk
    from the zarr while the main thread is computing on the current one, via
    a bounded queue. The dense host buffers (~28 GB / chunk at biobank scale)
    live in host RAM only, so peak host usage = (prefetch + 1) * chunk-bytes.

    Errors raised by the producer thread are forwarded to the consumer with
    their traceback preserved."""
    if prefetch <= 0:
        for ci, (left, right) in enumerate(chunks):
            t0 = time.perf_counter()
            gm, pos = source.slice_region(left, right)
            yield ci, left, right, gm, pos, time.perf_counter() - t0
        return

    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    _END = object()

    def producer():
        try:
            for ci, (left, right) in enumerate(chunks):
                if stop.is_set():
                    return
                t0 = time.perf_counter()
                gm, pos = source.slice_region(left, right)
                t_read = time.perf_counter() - t0
                if stop.is_set():
                    return
                q.put((ci, left, right, gm, pos, t_read))
        except BaseException as e:                   # producer-side failures
            q.put(("ERR", e))
            return
        q.put(_END)

    t = threading.Thread(target=producer, daemon=True, name="zarr-prefetch")
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                raise item[1]
            yield item
    finally:
        stop.set()
        # drain the queue so the producer can exit on its next q.put attempt
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=5)


# ── per-chromosome windowed scan ────────────────────────────────────────────

def windowed_scan(source, chunk_bp, prefetch=1):
    """Stream the chromosome chunk-by-chunk through the GPU.

    ``prefetch`` (default 1) runs the zarr region read on a worker thread so
    the next chunk's host buffer is ready by the time the GPU finishes the
    current chunk. With ``prefetch=0`` reads happen serially in the main
    thread (the GPU sits idle during decompression). Each per-chunk print
    line reports the read-time and compute-time separately so the overlap
    is visible.

    Returns ``(windows_by_scale, garud_df, sfs_by_pop, joint_sfs)``."""
    sub_g = {p: source.pop_cols[p][:min(GARUD_SUBSAMPLE, len(source.pop_cols[p]))]
             for p in POPS}
    sub_j = {p: source.pop_cols[p][:min(JOINT_SFS_SUBSAMPLE, len(source.pop_cols[p]))]
             for p in POPS}
    sample_sets = {}
    for p in POPS:
        sample_sets[p] = source.pop_cols[p]
        sample_sets[f"{p}_g"] = sub_g[p]

    align_bp = max(bp for _, bp in WINDOW_SCALES)
    chunks = list(iter_chunks(source.chrom_length, chunk_bp, align_bp))
    print(f"chr{source.chrom}: {source.num_variants:,} variants, "
          f"{source.num_haplotypes:,} haplotypes, "
          f"{len(chunks)} chunk(s) of <= {chunk_bp/1e6:g} Mb, "
          f"prefetch={prefetch}")

    parts = {label: [] for label, _ in WINDOW_SCALES}
    garud_parts = []
    sfs_by_pop = {p: None for p in POPS}
    joint = None

    t_scan = time.perf_counter()
    sum_read, sum_compute, sum_wait = 0.0, 0.0, 0.0
    for ci, left, right, gm, pos, t_read in chunk_iterator(source, chunks, prefetch):
        # Time from the moment we asked the queue for the next chunk until we
        # actually got it: this is how long the main thread was *blocked* on
        # I/O. With perfect prefetch this is ~0 (the next chunk is already
        # waiting in the queue); without prefetch it equals t_read.
        t_wait = t_read  # serial mode: we did the read ourselves, all of it counts
        if prefetch > 0:
            # In prefetch mode the producer already ran the read in the
            # background; the consumer's blocking time is hidden inside the
            # queue. We don't have a direct measurement of it without extra
            # plumbing, so report t_read for transparency (it's wall-clock
            # spent decompressing this chunk in the producer thread, which
            # bounds the *worst case* wait if compute is faster than read).
            pass
        sum_read += t_read

        n_sites = int(gm.shape[0])
        if n_sites == 0:
            print(f"  chunk {ci+1}/{len(chunks)} [{left/1e6:.1f}-{right/1e6:.1f} Mb]: "
                  f"no sites (read {t_read:.1f}s)")
            continue

        t_c0 = time.perf_counter()
        hm = build_hm(gm, pos, left, right, sample_sets)
        del gm, pos                                   # release host buffer ASAP

        for label, bp in WINDOW_SCALES:
            per_pop = {p: windowed_analysis(
                hm, window_size=bp, step_size=bp,
                statistics=DIVERSITY_STATS, populations=[p])
                for p in POPS}
            df_div = windowed_analysis(
                hm, window_size=bp, step_size=bp,
                statistics=DIVERGENCE_STATS, populations=list(POPS))
            m = per_pop[POPS[0]][["start", "end", "center"]].copy()
            m.insert(0, "chrom", str(source.chrom))
            for p in POPS:
                for s in DIVERSITY_STATS + ["n_variants"]:
                    m[f"{s}_{p}"] = per_pop[p][s].values
            for s in DIVERGENCE_STATS:
                m[s] = df_div[s].values
            m = m[m[f"n_variants_{POPS[0]}"].values > 0]
            if not m.empty:
                parts[label].append(m.reset_index(drop=True))

        g_afr = windowed_analysis(hm, window_size=GARUD_SCALE_BP,
                                  step_size=GARUD_SCALE_BP,
                                  statistics=GARUD_STATS,
                                  populations=[f"{POPS[0]}_g"])
        gm_df = g_afr[["start", "end", "center", "n_variants"]].copy()
        gm_df.insert(0, "chrom", str(source.chrom))
        for s in GARUD_STATS:
            gm_df[f"{s}_{POPS[0]}"] = g_afr[s].values
        g_eur = windowed_analysis(hm, window_size=GARUD_SCALE_BP,
                                  step_size=GARUD_SCALE_BP,
                                  statistics=GARUD_STATS,
                                  populations=[f"{POPS[1]}_g"])
        for s in GARUD_STATS:
            gm_df[f"{s}_{POPS[1]}"] = g_eur[s].values
        gm_df = gm_df[gm_df["n_variants"].values > 0]
        if not gm_df.empty:
            garud_parts.append(gm_df.reset_index(drop=True))

        for p in POPS:
            s = np.asarray(sfs.sfs(hm, population=p))
            sfs_by_pop[p] = s if sfs_by_pop[p] is None else sfs_by_pop[p] + s
        j = np.asarray(sfs.joint_sfs(hm, pop1=list(sub_j[POPS[0]]),
                                     pop2=list(sub_j[POPS[1]])))
        joint = j if joint is None else joint + j

        del hm
        free_gpu()
        t_compute = time.perf_counter() - t_c0
        sum_compute += t_compute

        print(f"  chunk {ci+1}/{len(chunks)} "
              f"[{left/1e6:.1f}-{right/1e6:.1f} Mb]: {n_sites:,} sites, "
              f"read {t_read:.1f}s + compute {t_compute:.1f}s")

    t_total = time.perf_counter() - t_scan
    overlap = sum_read + sum_compute - t_total
    print(f"windowed_scan totals: wall {t_total:,.1f}s | "
          f"sum-read {sum_read:,.1f}s | sum-compute {sum_compute:,.1f}s | "
          f"overlap saved {max(overlap, 0.0):,.1f}s ({100*max(overlap,0)/(sum_read+sum_compute+1e-9):.0f}%)")

    windows_by_scale = {label: (pd.concat(parts[label], ignore_index=True)
                                if parts[label] else pd.DataFrame())
                        for label, _ in WINDOW_SCALES}
    garud_df = pd.concat(garud_parts, ignore_index=True) if garud_parts else pd.DataFrame()
    return windows_by_scale, garud_df, sfs_by_pop, joint


# ── LD analyses ─────────────────────────────────────────────────────────────

LD_BP_BINS = [0, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
              100_000, 200_000, 500_000]


def ld_decay(source):
    """Mean r^2 over MAF-filtered common SNP pairs, distance-binned and pooled
    over large probe regions tiling the mappable chromosome."""
    sub = {p: source.pop_cols[p][:min(LD_SUBSAMPLE, len(source.pop_cols[p]))]
           for p in POPS}
    bins = np.asarray(LD_BP_BINS, dtype=float)
    mids = np.sqrt(np.maximum(bins[:-1], 1.0) * bins[1:])
    mids[0] = bins[1] / 2.0
    max_d = float(bins[-1])
    sum_r2 = {p: np.zeros(len(bins) - 1) for p in POPS}
    n_pairs = {p: np.zeros(len(bins) - 1, dtype=np.int64) for p in POPS}

    mappable_lo, mappable_hi = source.mappable_lo, source.mappable_hi
    span = max(LD_DECAY_PROBE_BP, (mappable_hi - mappable_lo) // LD_DECAY_N_PROBES)
    lefts = np.unique(np.linspace(mappable_lo, max(mappable_lo, mappable_hi - span),
                                  LD_DECAY_N_PROBES).astype(int))
    for pi, left in enumerate(lefts):
        right = min(int(left) + span, mappable_hi)
        for p in POPS:
            gm, pos = read_common_variants(source, int(left), right, sub[p],
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
            hm = HaplotypeMatrix(cp.asarray(np.ascontiguousarray(gm.T)),
                                 cp.asarray(pos),
                                 chrom_start=int(left), chrom_end=right - 1)
            r2 = hm.pairwise_r2()
            r2 = r2.get() if hasattr(r2, "get") else np.asarray(r2)
            del hm
            free_gpu()
            iu, ju = np.triu_indices(r2.shape[0], k=1)
            d = pos[ju] - pos[iu]
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
            rows.append({"pop": p, "bin_lo_bp": int(bins[i]),
                         "bin_hi_bp": int(bins[i + 1]),
                         "bin_mid_bp": float(mids[i]),
                         "mean_r2": float(mean[i]),
                         "n_pairs": int(n_pairs[p][i])})
    return pd.DataFrame(rows), r2by


def ld_heatmap(source, region):
    """Pairwise r^2 heatmap for common variants (MAF >= LD_HEATMAP_MIN_MAF)
    in one ~1 Mb sub-region, on a single population's haplotype subsample."""
    left, right = region
    cols = source.pop_cols[POPS[0]][:min(LD_HEATMAP_SUBSAMPLE,
                                          len(source.pop_cols[POPS[0]]))]
    gm, pos = read_common_variants(source, left, right, cols, LD_HEATMAP_MIN_MAF)
    if gm.shape[0] < 10:
        return np.zeros((0, 0)), np.empty(0), len(cols)
    if gm.shape[0] > LD_HEATMAP_MAX_SNPS:
        pick = np.linspace(0, gm.shape[0] - 1, LD_HEATMAP_MAX_SNPS).astype(int)
        gm, pos = np.ascontiguousarray(gm[pick]), pos[pick]
    print(f"  LD heatmap: {POPS[0]} chr region {left/1e6:.2f}-{right/1e6:.2f} Mb, "
          f"{gm.shape[0]} common SNPs x {len(cols)} haplotypes")
    hm = HaplotypeMatrix(cp.asarray(np.ascontiguousarray(gm.T)),
                         cp.asarray(pos),
                         chrom_start=int(left), chrom_end=int(right) - 1)
    r2 = hm.pairwise_r2()
    r2 = r2.get() if hasattr(r2, "get") else np.asarray(r2)
    del hm
    free_gpu()
    return r2, pos, len(cols)


# ── plotting ────────────────────────────────────────────────────────────────

SMOOTH_WINDOWS = 5  # adjacent windows averaged for the bold smoothed trace


def _smooth(y):
    return uniform_filter1d(np.where(np.isfinite(y), y, 0.0),
                            size=SMOOTH_WINDOWS, mode="nearest")


def _scan_panel(ax, x_mb, series, ylabel, title, hline=None, legend=False):
    """One genome-scan panel: faded raw trace + bold smoothed line, per series.
    ``series`` is a list of ``(values, color, label)`` -- one entry for a
    single track, two for a two-population comparison."""
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


def _draw_ld_decay(ax, r2_by_pop, n_ld_sub, title_size=10, title=None):
    """Plot mean-r^2 vs distance, per pop. ``title=None`` keeps the long
    default; pass an explicit string (or ``''`` for no title) to override."""
    for p in POPS:
        mids, mean_r2 = r2_by_pop[p]
        ax.plot(mids, mean_r2, "o-", color=POP_COLORS[p], lw=1.5, ms=5, label=p)
    ax.set_xscale("log")
    ax.set_xlabel("Distance between SNPs (bp)", fontsize=10)
    ax.set_ylabel(r"mean $r^2$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, title="population", title_fontsize=9)
    if title is None:
        title = (f"LD decay (mean $r^2$, common SNPs MAF $\\geq$ {LD_DECAY_MIN_MAF})\n"
                 f"{n_ld_sub:,}-hap subsample/pop, {LD_DECAY_N_PROBES} probe regions")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left")


def _densest_block(r2, frac=0.12):
    """Index range ``[i0, i1)`` of the contiguous SNP block of size
    ``~frac * n`` with the highest mean within-block r^2 -- used to pick what
    the LD zoom shows."""
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
    """``title=None`` keeps the long default; pass an explicit string (or
    ``''`` for no title) to override."""
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
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=10)
    ax.set_ylabel(f"chr{chrom} position (Mb)", fontsize=10)
    ax.tick_params(labelsize=8)
    if with_inset:
        i0, i1 = _densest_block(r2f, frac=0.14)
        z0, z1 = float(hm_pos[i0]) / 1e6, float(hm_pos[i1 - 1]) / 1e6
        axins = ax.inset_axes([0.58, 0.03, 0.40, 0.40])
        axins.imshow(r2f[i0:i1, i0:i1].T, cmap="magma", vmin=0, vmax=1,
                     origin="lower", interpolation="none",
                     extent=[z0, z1, z0, z1], aspect="equal")
        axins.set_xticks([z0, z1]); axins.set_yticks([z0, z1])
        axins.tick_params(labelsize=7)
        axins.set_title(f"zoom {z0:.3f}-{z1:.3f} Mb", fontsize=8, fontweight="bold")
        ax.indicate_inset_zoom(axins, edgecolor="white", lw=1.0, alpha=0.9)
    if title is None:
        title = (f"Pairwise $r^2$ ({POPS[0]}, MAF $\\geq$ {LD_HEATMAP_MIN_MAF})\n"
                 f"chr{chrom}:{region[0]/1e6:.2f}-{region[1]/1e6:.2f} Mb, "
                 f"{r2_mat.shape[0]} SNPs x {n_hm_haps:,} haps")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left")


def plot_composite(df_main, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   chrom, x_lo_mb, chrom_len, n_haps_per_pop, scale_label,
                   ld_region, n_garud_sub, n_joint_sub, n_ld_sub, subtitle_extra,
                   out_base):
    """The headline composite: wide left column of genome-scan panels (faded raw
    + bold smoothed traces, with the LD probe region shaded across all panels),
    narrow right column = joint SFS / LD decay curve / pairwise-r^2 heatmap+zoom."""
    afr, eur = POPS
    x = df_main["center"].values / 1e6
    gx = garud_df["center"].values / 1e6 if not garud_df.empty else None
    div_color = "0.15"  # F_ST / D_xy: single track in dark grey -- color always maps to pop

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
                      va="bottom", fontsize=10, fontweight="bold", color="#c0392b",
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

    _draw_ld_decay(fig.add_subplot(right_gs[1, 0]), ld_r2, n_ld_sub, title_size=11,
                   title=f"LD decay, MAF>{LD_DECAY_MIN_MAF}")
    _draw_r2_heatmap(fig.add_subplot(right_gs[2, 0]), r2_mat, hm_pos, chrom,
                     ld_region, n_hm_haps, with_inset=True, title_size=11, title="")

    fig.suptitle(f"pg_gpu chromosome scan -- simulated OOA_2T12 (Tennessen 2012), "
                 f"chr{chrom}: {n_haps_per_pop:,} haplotypes/population, "
                 f"{subtitle_extra}, {scale_label} windows",
                 fontsize=13, fontweight="bold", y=0.985)
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


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"directory holding chr*.vcz + chr*.pops.tsv "
                        f"(default {DEFAULT_DATA_DIR})")
    p.add_argument("--zarr", default=None,
                   help="explicit path to a single VCZ store "
                        "(overrides --data-dir / --chromosome)")
    p.add_argument("--pop-file", default=None,
                   help="sample_id<TAB>population TSV "
                        "(default: <store>.pops.tsv next to the store)")
    p.add_argument("--chromosome", default=None,
                   help="contig name to scan (default: the single chr*.vcz in "
                        "--data-dir, or the single contig in --zarr)")
    p.add_argument("--chunk-bp", type=int, default=5_000_000,
                   help="genomic chunk size streamed to the GPU at a time, bp "
                        "(default 5,000,000)")
    p.add_argument("--prefetch", type=int, default=1,
                   help="zarr-read prefetch depth (default 1: read next chunk "
                        "on a worker thread while GPU computes current; "
                        "0 disables and reads serially)")
    p.add_argument("--ld-region", default=None,
                   help="region 'start-end' in bp for the pairwise-r^2 heatmap "
                        f"(default: a {LD_HEATMAP_REGION_BP//1000} kb window near "
                        "the chromosome midpoint)")
    p.add_argument("--tables-dir", default=str(TABLES_DIR),
                   help=f"output directory for tables (default {TABLES_DIR})")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR),
                   help=f"output directory for figures (default {FIGURES_DIR})")
    return p.parse_args()


def pick_zarr(data_dir, requested):
    paths = sorted(p for p in Path(data_dir).glob("chr*.vcz") if p.is_dir())
    if not paths:
        raise SystemExit(f"no chr*.vcz/ in {data_dir} -- run "
                         f"ts_to_vcz.py first")
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


def main():
    args = parse_args()

    if args.zarr:
        zarr_path = Path(args.zarr)
    else:
        zarr_path = pick_zarr(args.data_dir, args.chromosome)

    manifest_path = zarr_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {zarr_path} ...")
    source = ZarrSource(zarr_path, pop_file=args.pop_file,
                        contig_id=args.chromosome)
    for p in POPS:
        if p not in source.pop_cols:
            raise SystemExit(f"population {p} not in {source.pop_file} "
                             f"(have: {sorted(source.pop_cols.keys())})")
    n_haps_per_pop = int(len(source.pop_cols[POPS[0]]))
    chrom_len = source.chrom_length
    x_lo_mb = float(np.floor(source.mappable_lo / 1e6))
    print(f"  contig {source.chrom}: {source.num_variants:,} variants, "
          f"{n_haps_per_pop:,} haplotypes/pop, "
          f"mappable {source.mappable_lo/1e6:.1f}-{source.mappable_hi/1e6:.1f} Mb")

    t0 = time.perf_counter()
    windows_by_scale, garud_df, sfs_by_pop, joint = windowed_scan(
        source, args.chunk_bp, prefetch=args.prefetch)
    print(f"windowed scan: {time.perf_counter() - t0:,.1f}s")

    for label, _ in WINDOW_SCALES:
        windows_by_scale[label].to_csv(tables_dir / f"windowed_stats_{label}.csv",
                                       index=False)
    garud_df.to_csv(tables_dir / "garud_h_10kb.csv", index=False)
    for p in POPS:
        pd.DataFrame({"derived_allele_count": np.arange(len(sfs_by_pop[p])),
                      "sites": sfs_by_pop[p]}).to_csv(
            tables_dir / f"sfs_{p}.csv", index=False)
    # Cache the joint SFS for cheap replots (it would otherwise be the slow
    # part to recompute -- needs another full chromosome pass).
    np.save(tables_dir / "joint_sfs.npy", joint)

    print("LD decay ...")
    t0 = time.perf_counter()
    ld_decay_df, ld_r2 = ld_decay(source)
    ld_decay_df.to_csv(tables_dir / "ld_decay.csv", index=False)
    if args.ld_region:
        a, b = args.ld_region.split("-")
        region = (int(a), int(b))
    else:
        mid = (source.mappable_lo + source.mappable_hi) // 2
        half = LD_HEATMAP_REGION_BP // 2
        region = (max(source.mappable_lo, mid - half),
                  min(source.mappable_hi, mid + half))
    print("LD r^2 heatmap ...")
    r2_mat, hm_pos, n_hm_haps = ld_heatmap(source, region)
    np.save(tables_dir / "r2_heatmap.npy", r2_mat)
    np.save(tables_dir / "r2_heatmap_pos.npy", hm_pos)
    print(f"LD analyses: {time.perf_counter() - t0:,.1f}s")

    main_df = windows_by_scale[MAIN_SCALE]
    w = (main_df["end"] - main_df["start"]).astype(float).values

    def wmean(col):
        v = main_df[col].astype(float).values
        msk = np.isfinite(v)
        return float(np.average(v[msk], weights=w[msk])) if msk.any() else float("nan")

    summary = {
        "model": manifest.get("model", "OutOfAfrica_2T12"),
        "genetic_map": manifest.get("genetic_map"),
        "chromosome": source.chrom,
        "chromosome_length": chrom_len,
        "haplotypes_per_pop": n_haps_per_pop,
        "n_sites": source.num_variants,
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
            **{f"total_segregating_sites_{p}":
                   float(np.nansum(main_df[f"segregating_sites_{p}"])) for p in POPS},
        },
    }
    (tables_dir / "chromosome_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary (100 kb windows):")
    for k, v in summary["genomewide_100kb"].items():
        print(f"  {k}: {v:.6g}")

    plot_composite(main_df, garud_df, joint, ld_r2, r2_mat, hm_pos, n_hm_haps,
                   source.chrom, x_lo_mb, chrom_len, n_haps_per_pop, MAIN_SCALE,
                   region, min(GARUD_SUBSAMPLE, n_haps_per_pop),
                   min(JOINT_SFS_SUBSAMPLE, n_haps_per_pop),
                   min(LD_SUBSAMPLE, n_haps_per_pop),
                   f"{source.num_variants:,} variants",
                   str(figures_dir / "genome_scan_ooa"))
    plot_multiscale(windows_by_scale, source.chrom, x_lo_mb, chrom_len,
                    str(figures_dir / "multiscale_ooa"))
    plot_ld(ld_r2, r2_mat, hm_pos, n_hm_haps, source.chrom, region,
            min(LD_SUBSAMPLE, n_haps_per_pop), str(figures_dir / "ld_ooa"))


if __name__ == "__main__":
    main()
