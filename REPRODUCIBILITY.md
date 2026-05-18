# how to remake every figure in the paper

The paper itself lives in
[`andrewkern/pg_gpu-paper`](https://github.com/andrewkern/pg_gpu-paper);
this repo is included there as a submodule under `analysis/`. Every
figure in the paper's `main.tex` is in the table below paired with
the script in this repo that made it. Run from this repo's root,
inside the `pg_gpu` pixi env (or the `moments` feature where noted),
with a free GPU picked via `CUDA_VISIBLE_DEVICES`. `make
<section>/<thing>` runs the same command if you'd rather use Make.

Each script writes its figure into its own section's `figures/` here.
The paper consumes from its own `figures/` directory at the root of
`pg_gpu-paper`. So: rerun the script, eyeball the new one, copy it up
into the parent repo's `figures/`.

## the figures

| figure | what makes it | command |
|---|---|---|
| `accuracy_vs_plink_pca.pdf` | `01_accuracy/scripts/accuracy_vs_plink.py` | `make 01_accuracy/accuracy_vs_plink` |
| `missing_data_bias.pdf` | `01_accuracy/scripts/missing_data_bias.py` | `make 01_accuracy/missing_data_bias` |
| `scikit_allel_comparison.pdf` | `01_accuracy/scripts/scikit_allel_comparison.py` | `make 01_accuracy/scikit_allel_comparison` |
| `moments_3pop_true_model.png` | `01_accuracy/scripts/moments_integration_demo.py` | `make 01_accuracy/moments_integration_demo` |
| `moments_3pop_fitted_vs_observed.png` | same script as the line above | same |
| `moments_ld_2pop.png` | `01_accuracy/scripts/multipop_ld_benchmark.py 2` | `make 01_accuracy/multipop_ld_benchmark_2pop` |
| `moments_ld_3pop.png` | `01_accuracy/scripts/multipop_ld_benchmark.py 3` | `make 01_accuracy/multipop_ld_benchmark_3pop` |
| `moments_ld_4pop.png` | `01_accuracy/scripts/multipop_ld_benchmark.py 4` | `make 01_accuracy/multipop_ld_benchmark_4pop` |
| `moments_ld_validation.pdf` | `01_accuracy/scripts/four_pop_ld_demo.py` | `make 01_accuracy/four_pop_ld_demo` |
| `moments_ld_decay_comparison.pdf` | `01_accuracy/scripts/moments_ld_decay_comparison.py` | `make 01_accuracy/moments_ld_decay_comparison` |
| `benchmark_speedups.pdf` | `02_performance/scripts/benchmark_3R.py` | `make 02_performance/benchmark_3R` |
| `benchmark_walltimes.pdf` | same | same |
| `benchmark_simulated.pdf` | `02_performance/scripts/benchmark_simulated.py` | `make 02_performance/benchmark_simulated` |
| `scaling_combined.pdf` | `03_scaling/scripts/scaling_samples_variants.py` | `make 03_scaling` |
| `ag1000g_genome_scan.png` | `05_application/scripts/ag1000g_workflow.py` (run `build_populations.py` first so the pop assignments are on disk) | `make 05_application/ag1000g_workflow` |
| `ag1000g_lostruct.pdf` | `05_application/scripts/ag1000g_lostruct.py` | `make 05_application/ag1000g_lostruct` |
| `local_pca.png` | `05_application/scripts/local_pca.py` | `make 05_application/local_pca` |
| `genome_scan_ooa.pdf` | `06_simulated_genome_scan/scripts/genome_scan_ooa.py` (after `simulate_ooa_genome.py` + `ts_to_vcz.py`) | `make 06_simulated_genome_scan` |

## scripts mirrored or recovered from pg_gpu

Several of the scripts above are not original to this repo. The copy
here is what makes the paper figure -- if you edit the original, sync
it back over. Each one has a note up top pointing at where it came
from.

| copy | source |
|---|---|
| `01_accuracy/scripts/scikit_allel_comparison.py` | `pg_gpu/examples/scikit_allel_comparison.py` (current) |
| `01_accuracy/scripts/moments_integration_demo.py` | `pg_gpu/examples/moments_integration_demo.py` (current) |
| `05_application/scripts/local_pca.py` | `pg_gpu/examples/local_pca.py` (current) |
| `01_accuracy/scripts/multipop_ld_benchmark.py` | `pg_gpu/examples/multipop_ld_benchmark.py` at d0a2cf8^ (removed by PR #61) |
| `01_accuracy/scripts/four_pop_ld_demo.py` | `pg_gpu/examples/four_pop_ld_demo.py` at d0a2cf8^ (removed by PR #61) |
| `01_accuracy/scripts/moments_ld_decay_comparison.py` | older `pg_gpu/examples/moments_integration_demo.py` at 1149612^ (the plot was dropped by PR #60 when the demo was rewritten for three populations) |
