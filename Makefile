# Re-run the paper-analysis pipeline end-to-end. Every script writes
# into its section's sibling figures/ and tables/, both of which are
# committed to git -- a `make all` should produce a clean diff if
# nothing in the pg_gpu library has drifted.
#
# Scripts must run from the repo root (their output paths are
# hardcoded relative); make uses $(CURDIR) by default, so just invoke
# `make` from this directory.
#
# Usage
# -----
#   make help              this message
#   make all               every section in dependency order (~1.5-2.5 h)
#   make <section>         one section, e.g. `make 02_performance`
#   make <section>/<name>  one script, e.g. `make 05_application/ag1000g_lostruct`
#   make 06_full           include the multi-hour stdpopsim regenerate of chr15
#   make estimates         per-section runtime estimates
#
# Knobs
# -----
#   CUDA=N                 GPU index for pg_gpu (default 0)
#   PIXI_MANIFEST=path     pg_gpu's pixi.toml (default /home/adkern/pg_gpu/pixi.toml)

SHELL := /bin/bash
CUDA ?= 0
PIXI_MANIFEST ?= /home/adkern/pg_gpu/pixi.toml

PIXI_PY = CUDA_VISIBLE_DEVICES=$(CUDA) pixi run --manifest-path $(PIXI_MANIFEST) python
VENV_PY = .venv/bin/python

# Wrap a command with a stamped wall-time line so a `make all` log
# shows section + per-script wall in real time. Avoids a hard
# dependency on /usr/bin/time (which not every host carries).
define timed
	@echo "[$(@F)] start"; \
	_t0=$$(date +%s); \
	$1; \
	echo "[$(@F)] done in $$(( $$(date +%s) - _t0 ))s"
endef

.DEFAULT_GOAL := help

# Phony declarations: every target below is a command, not a file
# (the section names happen to share names with the section
# directories, and the per-script targets use slashes; both would
# otherwise be treated as file targets and skipped if the path
# exists).
.PHONY: help all estimates \
        01_accuracy 01_accuracy/accuracy_vs_allel \
                    01_accuracy/accuracy_vs_plink \
                    01_accuracy/missing_data_bias \
        02_performance 02_performance/benchmark_3R \
                       02_performance/benchmark_simulated \
        03_scaling 03_scaling/scaling_samples_variants \
        05_application 05_application/build_populations \
                       05_application/ag1000g_workflow \
                       05_application/ag1000g_lostruct \
                       05_application/make_figure \
        06_simulated_genome_scan 06_full \
                       06_simulated_genome_scan/simulate \
                       06_simulated_genome_scan/ts_to_vcz \
                       06_simulated_genome_scan/genome_scan_ooa \
                       06_simulated_genome_scan/replot

# -----------------------------------------------------------------------
# Aggregate targets
# -----------------------------------------------------------------------

help:
	@echo "Usage:"
	@echo "  make all               rerun every section (~1.5-2.5 h)"
	@echo "  make <section>         e.g. make 02_performance"
	@echo "  make <section>/<name>  e.g. make 05_application/ag1000g_lostruct"
	@echo "  make 06_full           include the multi-hour stdpopsim simulate"
	@echo "  make estimates         show per-section wall estimates"
	@echo ""
	@echo "Knobs:  CUDA=N (default 0)  PIXI_MANIFEST=path"

estimates:
	@echo "Per-section estimates (caches missing, data on disk):"
	@echo "  01_accuracy                ~15-20 min"
	@echo "  02_performance             ~45-60 min  (~30 min for stress_test_3R if /home/adkern/pg_gpu/debug/stress_test_3R_results.txt is absent)"
	@echo "  03_scaling                 ~15-25 min"
	@echo "  05_application             ~10-20 min"
	@echo "  06_simulated_genome_scan   ~16 min   (assumes chr15.vcz on disk)"
	@echo "  total                      ~1.5-2.5 h serial"
	@echo ""
	@echo "  06_full adds ~3-4 h to regenerate chr15 from stdpopsim."

all: 01_accuracy 02_performance 03_scaling 05_application 06_simulated_genome_scan
	@echo "All sections completed."

# -----------------------------------------------------------------------
# 01_accuracy -- pg_gpu vs scikit-allel + PLINK on a 4 Mb Ag1000G 3L region
# -----------------------------------------------------------------------

01_accuracy: 01_accuracy/accuracy_vs_allel \
             01_accuracy/accuracy_vs_plink \
             01_accuracy/missing_data_bias

01_accuracy/accuracy_vs_allel:
	$(call timed,$(PIXI_PY) 01_accuracy/scripts/accuracy_vs_allel.py)

01_accuracy/accuracy_vs_plink:
	$(call timed,$(PIXI_PY) 01_accuracy/scripts/accuracy_vs_plink.py)

01_accuracy/missing_data_bias:
	$(call timed,$(PIXI_PY) 01_accuracy/scripts/missing_data_bias.py)

# -----------------------------------------------------------------------
# 02_performance -- wall-clock benchmarks on the full Ag1000G 3R arm
# benchmark_3R reads /home/adkern/pg_gpu/debug/stress_test_3R_results.txt
# as a ~30 min cache; if missing, the script regenerates it inline.
# -----------------------------------------------------------------------

02_performance: 02_performance/benchmark_3R 02_performance/benchmark_simulated

02_performance/benchmark_3R:
	$(call timed,$(PIXI_PY) 02_performance/scripts/benchmark_3R.py)

02_performance/benchmark_simulated:
	$(call timed,$(PIXI_PY) 02_performance/scripts/benchmark_simulated.py)

# -----------------------------------------------------------------------
# 03_scaling -- sample-size + variant-count sweeps
# -----------------------------------------------------------------------

03_scaling: 03_scaling/scaling_samples_variants

03_scaling/scaling_samples_variants:
	$(call timed,$(PIXI_PY) 03_scaling/scripts/scaling_samples_variants.py)

# -----------------------------------------------------------------------
# 05_application -- end-to-end Ag1000G workflow + lostruct
# build_populations writes 05_application/tables/population_assignments.json
# that ag1000g_workflow reads; make_figure consumes the workflow + lostruct
# tables.
# -----------------------------------------------------------------------

05_application: 05_application/build_populations \
                05_application/ag1000g_workflow \
                05_application/ag1000g_lostruct \
                05_application/make_figure

05_application/build_populations:
	$(call timed,$(PIXI_PY) 05_application/scripts/build_populations.py)

05_application/ag1000g_workflow: 05_application/build_populations
	$(call timed,$(PIXI_PY) 05_application/scripts/ag1000g_workflow.py)

05_application/ag1000g_lostruct:
	$(call timed,$(PIXI_PY) 05_application/scripts/ag1000g_lostruct.py)

05_application/make_figure: 05_application/ag1000g_workflow 05_application/ag1000g_lostruct
	$(call timed,$(PIXI_PY) 05_application/scripts/make_figure.py)

# -----------------------------------------------------------------------
# 06_simulated_genome_scan -- biobank-scale chr15 OOA scan
# simulate + ts_to_vcz run from a local .venv with stdpopsim (kept
# out of the pg_gpu pixi env on purpose); genome_scan_ooa + replot
# run in the pixi env. By default the scan reuses chr15.vcz on disk;
# `make 06_full` rebuilds it from the tree sequence.
# -----------------------------------------------------------------------

06_simulated_genome_scan: 06_simulated_genome_scan/genome_scan_ooa

06_full: 06_simulated_genome_scan/simulate \
         06_simulated_genome_scan/ts_to_vcz \
         06_simulated_genome_scan/genome_scan_ooa

06_simulated_genome_scan/simulate:
	$(call timed,$(VENV_PY) 06_simulated_genome_scan/scripts/simulate_ooa_genome.py --chromosomes 15 --num-samples 50000)

06_simulated_genome_scan/ts_to_vcz:
	$(call timed,$(VENV_PY) 06_simulated_genome_scan/scripts/ts_to_vcz.py --trees 06_simulated_genome_scan/data/ooa_2t12/chr15.trees)

06_simulated_genome_scan/genome_scan_ooa:
	$(call timed,$(PIXI_PY) 06_simulated_genome_scan/scripts/genome_scan_ooa.py)

06_simulated_genome_scan/replot:
	$(call timed,$(PIXI_PY) 06_simulated_genome_scan/scripts/replot.py)
