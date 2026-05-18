# pg_gpu-paper-analysis

Reproducibility scripts and figures for the `pg_gpu` paper.

Each top-level directory is one paper section, with `scripts/` that
produce results and sibling `figures/` (PDF/PNG) and `tables/` (CSV/JSON)
that the manuscript consumes directly.

## Sections

| Directory | What it produces |
|---|---|
| `01_accuracy/` | Numerical accuracy of pg_gpu against scikit-allel, PLINK, and moments |
| `02_performance/` | Wall-clock benchmarks on real Ag1000G 3R data |
| `03_scaling/` | Sample-size and variant-count scaling sweeps |
| `05_application/` | End-to-end Ag1000G workflow (population assignments, genome scan, summary tables) |
| `06_simulated_genome_scan/` | Biobank-scale simulated chr15 (`stdpopsim` `OutOfAfrica_2T12`, 100k diploids) end-to-end scan using the pg_gpu streaming API |

## Requirements

Scripts depend on the [`pg_gpu`](https://github.com/kr-colab/pg_gpu)
package and run inside its pixi environment with a CUDA-capable GPU
visible (3× A100 80 GB on the development machine; one GPU is enough
for any single script):

```bash
cd /path/to/pg_gpu && pixi shell
cd /path/to/pg_gpu-paper-analysis
CUDA_VISIBLE_DEVICES=0 python 01_accuracy/scripts/accuracy_vs_allel.py
```

Always invoke scripts **from the repo root** — output paths are
hardcoded relative to it (e.g. `"02_performance/figures"`).

Several scripts allocate >40 GB of GPU memory (full Ag1000G chromosome
arms, 100k-haplotype scaling sweeps, the chr15 streaming scan); pick a
free GPU via `CUDA_VISIBLE_DEVICES` and verify with `nvidia-smi` before
launching.

### Section 06 has a three-step workflow

The simulated biobank-scale scan in `06_simulated_genome_scan/` is the
only section with a multi-stage pipeline. `stdpopsim` is not added to
the `pg_gpu` pixi environment; it lives in a local virtualenv at the
repo root (`python3 -m venv .venv && .venv/bin/pip install stdpopsim`,
gitignored). See `06_simulated_genome_scan/scripts/` for the
`simulate_ooa_genome.py` → `ts_to_vcz.py` → `genome_scan_ooa.py`
sequence and `CLAUDE.md` for the full step-by-step. The scan completes
in ~16 minutes on a single A100 80 GB for 100k diploids on chr15.

## External data

A handful of scripts read absolute paths under
`/sietch_colab/data_share/Ag1000G/`:

* `Ag3.0/vcf/AgamP3.phased.zarr` — phased zarr, used by
  `02_performance/benchmark_3R.py` and `05_application/ag1000g_workflow.py`.
* `Ag3.0/ag1000g.unphased.3L.zarr` — unphased zarr, used by accuracy
  comparisons in `01_accuracy/`.
* `Ag3.0/args_trees/gamb.meta.tsinfer.csv` — sample metadata read by
  `05_application/build_populations.py`.
* `Ag3.0/args_trees/singer-test/3R.mask.bed` — accessibility mask.

`01_accuracy/scripts/scikit_allel_comparison.py` additionally reads
phased zarr fixtures from `/home/adkern/pg_gpu/examples/data/`
(checked into the pg_gpu repo, not duplicated here).

If those paths are not available, update the constants at the top of
the affected scripts.

## Per-figure provenance

For an end-to-end map of every paper figure to the exact script that
produced it (including a flagged set of figures whose producer is
currently missing), see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Outputs are checked in

PDFs, PNGs, and CSVs under `*/figures/` and `*/tables/` are committed
to git — the manuscript consumes them directly. Re-running a script
will produce a dirty working tree; commit the regenerated outputs
alongside the code change that produced them so the artifacts and
the code that made them stay in sync.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff). The preprint DOI will be filled
in once the manuscript is on bioRxiv.
