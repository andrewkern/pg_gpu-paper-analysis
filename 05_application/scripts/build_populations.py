#!/usr/bin/env python
"""
Build West Africa vs East Africa population assignments for the Ag1000G
phased zarr, and save as a JSON file for use by other scripts.

Outputs:
  - tables/population_assignments.json
"""

import json
import numpy as np
import pandas as pd
import zarr

META_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
OUT_PATH = "05_application/tables/population_assignments.json"

WEST = ['Burkina Faso', 'Mali', 'Guinea', 'Ghana', 'Guinea-Bissau', 'Gambia']
EAST = ['Uganda', 'Tanzania', 'Kenya', 'Mozambique', 'Mayotte']
N_DIP_PER_POP = 100


def main():
    meta = pd.read_csv(META_PATH)
    store = zarr.open(zarr.storage.LocalStore(ZARR_PATH), mode='r')
    zarr_samples = [str(s) for s in store['3R/samples']]
    n_samples = len(zarr_samples)

    meta_dict = dict(zip(meta['sample_id'], meta['country']))

    # Collect diploid indices per region
    west_dip, east_dip = [], []
    for i, sid in enumerate(zarr_samples):
        country = meta_dict.get(sid, '')
        if country in WEST:
            west_dip.append(i)
        elif country in EAST:
            east_dip.append(i)

    print(f"West Africa diploids: {len(west_dip)}")
    print(f"East Africa diploids: {len(east_dip)}")

    # Balanced random subsample
    rng = np.random.default_rng(42)
    west_sub = sorted(rng.choice(west_dip, N_DIP_PER_POP, replace=False).tolist())
    east_sub = sorted(rng.choice(east_dip, N_DIP_PER_POP, replace=False).tolist())

    # Map diploid indices to haplotype indices
    # Phased zarr layout: haplotype i = allele 0 of diploid i,
    #                      haplotype i + n_samples = allele 1 of diploid i
    def dip_to_hap(dip_indices):
        hap = []
        for i in dip_indices:
            hap.append(i)
            hap.append(i + n_samples)
        return hap

    west_haps = dip_to_hap(west_sub)
    east_haps = dip_to_hap(east_sub)

    # Country breakdown
    west_countries = [meta_dict[zarr_samples[i]] for i in west_sub]
    east_countries = [meta_dict[zarr_samples[i]] for i in east_sub]

    result = {
        "west_africa": west_haps,
        "east_africa": east_haps,
        "n_diploid_per_pop": N_DIP_PER_POP,
        "n_haplotypes_per_pop": len(west_haps),
        "west_countries": {k: int(v) for k, v in pd.Series(west_countries).value_counts().items()},
        "east_countries": {k: int(v) for k, v in pd.Series(east_countries).value_counts().items()},
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")
    print(f"West ({N_DIP_PER_POP} diploid = {len(west_haps)} haplotypes):")
    for c, n in sorted(result['west_countries'].items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print(f"East ({N_DIP_PER_POP} diploid = {len(east_haps)} haplotypes):")
    for c, n in sorted(result['east_countries'].items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
