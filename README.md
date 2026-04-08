# pg_gpu paper analysis

Reproducible scripts and figures for the pg_gpu paper.

## Structure

```
01_accuracy/     Numerical accuracy vs scikit-allel, PLINK, moments
02_performance/  Wall-clock benchmarks on Ag1000G 3R
03_scaling/      Sample size and variant count scaling
04_achaz_framework/  Neutrality test calibration (Achaz 2009)
05_application/  End-to-end Ag1000G workflow
```

Each directory contains:
- `scripts/` -- Python scripts that generate results
- `figures/` -- Output figures (PDF/PNG)
- `tables/` -- Output tables (CSV/TSV)

## Requirements

All scripts assume pg_gpu is installed in a pixi environment.
Run from within the pg_gpu pixi shell:

```bash
cd /path/to/pg_gpu
pixi shell
cd /path/to/pg_gpu-paper-analysis
python 01_accuracy/scripts/accuracy_vs_allel.py
```

## Data

Ag1000G Phase 3 zarr data is expected at:
- `/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr`
