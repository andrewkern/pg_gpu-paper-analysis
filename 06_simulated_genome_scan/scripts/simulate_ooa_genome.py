#!/usr/bin/env python
"""
Simulate human chromosome(s) under the Tennessen et al. (2012) two-population
out-of-Africa model and store each as a tree sequence.

Uses stdpopsim's ``HomSap`` species, the ``OutOfAfrica_2T12`` demographic model
(populations ``AFR`` and ``EUR``), the ``HapMapII_GRCh38`` recombination map, and
the msprime engine. Tree sequences are written to ``<output-dir>/chr{N}.trees``;
because the genealogy is stored rather than a dense genotype matrix, the on-disk
size grows roughly linearly (not quadratically) with sample size, so sample
counts well into the biobank range remain practical to store.

The companion deep scan (``genome_scan_ooa.py``) is built around a single, large
chromosome -- the example in the paper repo is chr15 with 50,000 diploids per
population (100,000 haplotypes per population):

    .venv/bin/python 06_simulated_genome_scan/scripts/simulate_ooa_genome.py \
        --chromosomes 15 --num-samples 50000 --seed 42

Pass e.g. ``--chromosomes 1-22`` to simulate the whole genome instead.

This script only needs stdpopsim/msprime/tskit -- run it in this repo's
``.venv`` (it does not import ``pg_gpu`` or cupy).

Outputs
-------
    06_simulated_genome_scan/data/ooa_2t12/chr{N}.trees       (gitignored)
    06_simulated_genome_scan/data/ooa_2t12/manifest.json      simulation parameters
"""

import argparse
import json
import time
from pathlib import Path

import stdpopsim


SPECIES_ID = "HomSap"
MODEL_ID = "OutOfAfrica_2T12"
DEFAULT_GENETIC_MAP = "HapMapII_GRCh38"
DEFAULT_OUTPUT_DIR = "06_simulated_genome_scan/data/ooa_2t12"


def parse_chromosomes(spec, available):
    """Expand a chromosome spec like '1-22' or '1,5,15' into a list of ids."""
    available = [str(c) for c in available]
    out = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-")
            for i in range(int(lo), int(hi) + 1):
                out.append(str(i))
        else:
            out.append(token)
    missing = [c for c in out if c not in available]
    if missing:
        raise ValueError(f"chromosome(s) not in {SPECIES_ID} genome: {missing}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=50_000,
                   help="diploid individuals sampled PER population (AFR and EUR); "
                        "haplotypes per population = 2 x this (default 50000 -> "
                        "100k haplotypes/pop, 200k total)")
    p.add_argument("--chromosomes", default="15",
                   help="chromosomes to simulate, e.g. '15' or '1-22' or '1,15,22' "
                        "(default 15)")
    p.add_argument("--genetic-map", default=DEFAULT_GENETIC_MAP,
                   help=f"stdpopsim genetic map id (default {DEFAULT_GENETIC_MAP}); "
                        "pass 'none' for a uniform recombination rate")
    p.add_argument("--seed", type=int, default=42,
                   help="base random seed; chromosome i uses seed + i (default 42)")
    p.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"directory for chr{{N}}.trees (default {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--overwrite", action="store_true",
                   help="re-simulate chromosomes whose .trees file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    species = stdpopsim.get_species(SPECIES_ID)
    model = species.get_demographic_model(MODEL_ID)
    engine = stdpopsim.get_engine("msprime")

    pop_names = [p.name for p in model.populations if p.allow_samples]
    assert pop_names == ["AFR", "EUR"], pop_names
    samples = {name: args.num_samples for name in pop_names}

    genetic_map = None if args.genetic_map.lower() == "none" else args.genetic_map
    chroms = parse_chromosomes(args.chromosomes, [c.id for c in species.genome.chromosomes])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL_ID} ({model.description})")
    print(f"Populations: {pop_names}, {args.num_samples} diploids each "
          f"-> {2 * args.num_samples} haplotypes/pop, {4 * args.num_samples} total")
    print(f"Genetic map: {genetic_map or 'uniform'}")
    print(f"Chromosomes: {', '.join(chroms)}")
    print(f"Output: {out_dir}/\n")

    manifest = {
        "species": SPECIES_ID,
        "assembly": species.genome.assembly_name,
        "model": MODEL_ID,
        "populations": pop_names,
        "num_samples_per_pop": args.num_samples,
        "genetic_map": genetic_map,
        "mutation_rate": model.mutation_rate,
        "base_seed": args.seed,
        "chromosomes": {},
    }

    for chrom in chroms:
        out_path = out_dir / f"chr{chrom}.trees"
        if out_path.exists() and not args.overwrite:
            print(f"chr{chrom}: exists, skipping ({out_path})")
            ts = None
        else:
            # Use the demographic model's mutation rate (not the species
            # default) so simulated diversity matches what the model was
            # calibrated to; otherwise stdpopsim warns about a rate mismatch.
            contig = species.get_contig(chrom, genetic_map=genetic_map,
                                        mutation_rate=model.mutation_rate)
            seed = args.seed + int(chrom) if chrom.isdigit() else args.seed
            t0 = time.perf_counter()
            ts = engine.simulate(model, contig, samples, seed=seed)
            dt = time.perf_counter() - t0
            ts.dump(out_path)
            size_gb = out_path.stat().st_size / 1e9
            print(f"chr{chrom}: {ts.num_sites:,} sites, {ts.num_trees:,} trees, "
                  f"{ts.num_samples:,} sample nodes, {dt:,.1f}s, {size_gb:.2f} GB "
                  f"-> {out_path}")
        if ts is not None:
            manifest["chromosomes"][chrom] = {
                "num_sites": int(ts.num_sites),
                "num_trees": int(ts.num_trees),
                "num_samples": int(ts.num_samples),
                "sequence_length": int(ts.sequence_length),
            }

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
