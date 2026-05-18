# Reproducibility map

One row per `\includegraphics{...}` entry in `paper/main.tex`,
mapping each figure to the exact script that produced it and the
command that regenerates it. All commands assume the working
directory is the repo root, the `pg_gpu` pixi environment is
active (or the moments feature where noted), and a free GPU is
selected via `CUDA_VISIBLE_DEVICES`. `make <section>/<target>`
runs the same command via the top-level [`Makefile`](Makefile).

Figures in `paper/figures/` are hand-mirrored from each section's
`figures/` directory — regenerate, inspect, then `cp` over.

## Main text + supplementary figures

| Paper figure | Section / script | Regenerate |
|---|---|---|
| `accuracy_vs_plink_pca.pdf` | `01_accuracy/scripts/accuracy_vs_plink.py` | `make 01_accuracy/accuracy_vs_plink` |
| `missing_data_bias.pdf` | `01_accuracy/scripts/missing_data_bias.py` | `make 01_accuracy/missing_data_bias` |
| `scikit_allel_comparison.pdf` | `01_accuracy/scripts/scikit_allel_comparison.py` (mirror of `pg_gpu/examples/scikit_allel_comparison.py`) | `make 01_accuracy/scikit_allel_comparison` |
| `moments_3pop_true_model.png` | `01_accuracy/scripts/moments_integration_demo.py` (mirror of `pg_gpu/examples/moments_integration_demo.py`) | `make 01_accuracy/moments_integration_demo` |
| `moments_3pop_fitted_vs_observed.png` | same as above (produced by the same run) | same |
| `benchmark_speedups.pdf` | `02_performance/scripts/benchmark_3R.py` | `make 02_performance/benchmark_3R` |
| `benchmark_walltimes.pdf` | same as above | same |
| `benchmark_simulated.pdf` | `02_performance/scripts/benchmark_simulated.py` | `make 02_performance/benchmark_simulated` |
| `scaling_combined.pdf` | `03_scaling/scripts/scaling_samples_variants.py` | `make 03_scaling` |
| `ag1000g_genome_scan.png` | `05_application/scripts/ag1000g_workflow.py` (depends on `build_populations.py` for `population_assignments.json`) | `make 05_application/ag1000g_workflow` |
| `ag1000g_lostruct.pdf` | `05_application/scripts/ag1000g_lostruct.py` | `make 05_application/ag1000g_lostruct` |
| `local_pca.png` | `05_application/scripts/local_pca.py` (mirror of `pg_gpu/examples/local_pca.py`) | `make 05_application/local_pca` |
| `genome_scan_ooa.pdf` | `06_simulated_genome_scan/scripts/genome_scan_ooa.py` (depends on `simulate_ooa_genome.py` and `ts_to_vcz.py`) | `make 06_simulated_genome_scan` |

## Figures with no in-repo producer

These five figures appear in the paper but no script in either
this repo or `pg_gpu/` currently produces them. They were
generated at some point and copied into `paper/figures/`; the
generator was not checked in. Before the preprint goes out we
either need to recover or rewrite the producer.

| Paper figure |
|---|
| `moments_ld_validation.pdf` |
| `moments_ld_2pop.png` |
| `moments_ld_3pop.png` |
| `moments_ld_4pop.png` |
| `moments_ld_decay_comparison.pdf` |

The `moments_integration_demo.py` script covers the demographic
inference figures (`moments_3pop_*`) but does not produce these
LD validation plots.

## Mirrored upstream scripts

Three scripts in this repo are mirrors of canonical examples in
`pg_gpu/examples/`. Each carries a header note pointing at the
upstream. When the upstream changes, sync the mirror — the
manuscript consumes the figures produced from this repo, not
from `pg_gpu/`.

| Mirror | Upstream |
|---|---|
| `01_accuracy/scripts/scikit_allel_comparison.py` | `pg_gpu/examples/scikit_allel_comparison.py` |
| `01_accuracy/scripts/moments_integration_demo.py` | `pg_gpu/examples/moments_integration_demo.py` |
| `05_application/scripts/local_pca.py` | `pg_gpu/examples/local_pca.py` |
