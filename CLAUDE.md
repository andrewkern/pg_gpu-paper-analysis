# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reproducibility scripts for the `pg_gpu` paper. Each top-level directory (`01_accuracy`, `02_performance`, `03_scaling`, `04_achaz_framework`, `05_application`) is one paper section; each holds `scripts/` (Python that produces results) and writes its outputs into sibling `figures/` (PDF/PNG) and `tables/` (CSV/JSON). Scripts are not a library and are not imported from each other — they are independent driver programs.

This repo has no Python package, no test suite, and no build system. The only "command" is running a script.

## Running scripts

Scripts depend on the `pg_gpu` package (sibling repo at `/home/adkern/pg_gpu`) and must run inside its pixi environment with a CUDA-capable GPU visible:

```bash
cd /home/adkern/pg_gpu && pixi shell -e gpu
cd /home/adkern/pg_gpu-paper-analysis
python 02_performance/scripts/benchmark_3R.py
```

Always invoke scripts **from the repo root** — output paths are hardcoded as relative strings (`"02_performance/figures"`, etc.), so running from inside `scripts/` writes to the wrong place.

GPU selection: there are 3 A100s on this machine. Scripts do not pin a device, so set `CUDA_VISIBLE_DEVICES=N` before launching, and verify the chosen GPU is idle first (`nvidia-smi`). Several scripts allocate >40 GB of GPU memory (full Ag1000G chromosome arms, 100K-haplotype scaling) and will OOM if the GPU is shared.

## External data dependencies

Several scripts read absolute paths under `/sietch_colab/data_share/Ag1000G/`. Notable ones:

- `Ag3.0/vcf/AgamP3.phased.zarr` — phased zarr, 3R chromosome arm (used by `02_performance/benchmark_3R.py`, `05_application/ag1000g_workflow.py`)
- `Ag3.0/ag1000g.unphased.3L.zarr` — unphased zarr (used by accuracy comparisons in `01_accuracy/`)
- `Ag3.0/args_trees/gamb.meta.tsinfer.csv` — sample metadata used by `05_application/build_populations.py`
- `Ag3.0/args_trees/singer-test/3R.mask.bed` — accessibility mask

If those paths change, update the constants at the top of each affected script.

## Cross-script artifacts

Most scripts are self-contained, but a few depend on outputs from others — produce them in this order:

- `05_application/build_populations.py` writes `05_application/tables/population_assignments.json`, which `ag1000g_workflow.py` reads. Run `build_populations.py` first.
- `02_performance/benchmark_3R.py` uses cached output from `/home/adkern/pg_gpu/debug/stress_test_3R_results.txt` if present (stress test takes ~30 min); delete the cache to force a rerun.
- `05_application/tables/cached_*.csv` are intermediate caches written by `ag1000g_workflow.py` to skip recomputation across reruns. Treat them as derived; do not edit by hand.

## Conventions worth preserving

- Every script's docstring lists the outputs it produces (paths under `tables/` and `figures/`). Keep this header accurate when adding/removing outputs — it's the closest thing to documentation this repo has.
- Figures: `matplotlib.use('Agg')` (no display), seaborn `paper` context, PDF as primary format. Don't switch to interactive backends.
- Timing pattern for GPU work: bracket the call with `cp.cuda.Stream.null.synchronize()` before reading `time.perf_counter()` — wall-clock without sync measures kernel launch, not execution.
- `bench()` in `03_scaling/scripts/scaling_samples_variants.py` is the canonical timing helper (warmup + median of N iters with explicit syncs). Reuse this pattern rather than inventing new timing code.
- Errors in a single statistic should not abort a whole sweep. The scaling and benchmark scripts catch per-statistic exceptions (especially `cp.cuda.memory.OutOfMemoryError`), record OOM/FAIL, free the memory pool, and continue. Preserve this when extending.

## Outputs are checked in

PDFs, PNGs, and CSVs under `*/figures/` and `*/tables/` are committed to git — the paper consumes them directly. Rerunning a script will produce a dirty working tree; commit the regenerated outputs along with any code change that produced them so the artifacts and the code that made them stay in sync.
