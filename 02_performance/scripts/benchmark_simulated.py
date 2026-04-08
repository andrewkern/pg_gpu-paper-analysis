#!/usr/bin/env python
"""
Performance comparison: pg_gpu vs scikit-allel vs PLINK2 on simulated data.

Uses msprime to simulate a two-population model, then benchmarks
shared statistics across all three tools on identical data.

Produces:
  - tables/benchmark_simulated.csv
  - figures/benchmark_simulated.pdf
"""

import os
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
import msprime
import allel
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import HaplotypeMatrix, diversity, divergence, decomposition, sfs

OUT_DIR_FIG = "02_performance/figures"
OUT_DIR_TBL = "02_performance/tables"

PLINK2 = os.path.expanduser("~/bin/plink2")
N_DIP = 500  # diploid individuals per population


def simulate():
    """Simulate two-population data."""
    print("Simulating data...", flush=True)
    demography = msprime.Demography()
    demography.add_population(name="pop1", initial_size=10_000)
    demography.add_population(name="pop2", initial_size=10_000)
    demography.add_population(name="anc", initial_size=10_000)
    demography.add_population_split(time=1000, derived=["pop1", "pop2"],
                                     ancestral="anc")
    ts = msprime.sim_ancestry(
        samples={"pop1": N_DIP, "pop2": N_DIP},
        sequence_length=10_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=42, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
    print(f"  {ts.num_samples} haplotypes, {ts.num_mutations:,} variants")
    return ts


def export_vcf_and_plink(hm, tmpdir):
    """Export HaplotypeMatrix to VCF and PLINK bed."""
    hap = hm.haplotypes
    n_hap, n_var = hap.shape
    n_ind = n_hap // 2
    pos = hm.positions

    vcf_path = os.path.join(tmpdir, "data.vcf")
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write("##contig=<ID=1>\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for i in range(n_ind):
            f.write(f"\tind{i}")
        f.write("\n")
        for j in range(n_var):
            f.write(f"1\t{int(pos[j])}\tsnp{j}\tA\tT\t.\tPASS\t.\tGT")
            for i in range(n_ind):
                f.write(f"\t{int(hap[2*i,j])}|{int(hap[2*i+1,j])}")
            f.write("\n")

    prefix = os.path.join(tmpdir, "data")
    subprocess.run([PLINK2, "--vcf", vcf_path, "--make-bed",
                    "--out", prefix, "--allow-extra-chr"],
                   capture_output=True, check=True)

    # Pop file for FST
    pop_path = os.path.join(tmpdir, "pops.txt")
    with open(pop_path, 'w') as f:
        f.write("#FID\tIID\tPOP\n")
        for i in range(n_ind):
            pop = "pop1" if i < N_DIP else "pop2"
            f.write(f"0\tind{i}\t{pop}\n")

    return prefix, pop_path


def bench(fn, n_warmup=1, n_iter=3, sync_gpu=False):
    """Time a function, return median."""
    for _ in range(n_warmup):
        fn()
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_iter):
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    ts = simulate()
    hm = HaplotypeMatrix.from_ts(ts)
    hm = hm.apply_biallelic_filter()
    n_hap = hm.num_haplotypes
    n_half = n_hap // 2
    hm.sample_sets = {
        "pop1": list(range(n_half)),
        "pop2": list(range(n_half, n_hap)),
    }
    print(f"  {n_hap} haplotypes x {hm.num_variants:,} biallelic variants")

    # Build allel objects
    hap_np = hm.haplotypes
    pos_np = hm.positions
    gt = np.stack([hap_np[0::2], hap_np[1::2]], axis=-1).transpose(1, 0, 2)
    g = allel.GenotypeArray(gt)
    pos_allel = allel.SortedIndex(pos_np)
    pop1_dip = list(range(N_DIP))
    pop2_dip = list(range(N_DIP, 2 * N_DIP))

    # Transfer pg_gpu to GPU
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()

    # Export for PLINK
    tmpdir = tempfile.mkdtemp()
    print("Exporting to PLINK format...", flush=True)
    t0 = time.time()
    prefix, pop_path = export_vcf_and_plink(hm, tmpdir)
    print(f"  Export: {time.time()-t0:.1f}s")

    rows = []

    def add(name, pg_fn, al_fn, plink_cmd=None):
        t_pg = bench(pg_fn, sync_gpu=True)
        t_al = bench(al_fn, sync_gpu=False)

        t_plink = None
        if plink_cmd:
            # Warmup
            subprocess.run(plink_cmd, capture_output=True)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                subprocess.run(plink_cmd, capture_output=True)
                times.append(time.perf_counter() - t0)
            t_plink = np.median(times)

        row = {"statistic": name, "pg_gpu_s": t_pg, "allel_s": t_al}
        if t_plink is not None:
            row["plink_s"] = t_plink
        rows.append(row)

        pg_str = f"{t_pg:.4f}s"
        al_str = f"{t_al:.4f}s"
        pk_str = f"{t_plink:.4f}s" if t_plink else "---"
        sp_al = f"{t_al/t_pg:.0f}x" if t_pg > 0 else "---"
        sp_pk = f"{t_plink/t_pg:.0f}x" if t_plink and t_pg > 0 else "---"
        print(f"  {name:<25s} pg={pg_str:>9s}  al={al_str:>9s}  pk={pk_str:>9s}  "
              f"vs_al={sp_al:>6s}  vs_pk={sp_pk:>6s}")

    print(f"\n{'Benchmarks':>25s} {'pg_gpu':>9s}  {'allel':>9s}  {'plink':>9s}  "
          f"{'vs_al':>6s}  {'vs_pk':>6s}")
    print("-" * 85)

    # --- Diversity ---
    add("pi",
        lambda: diversity.pi(hm, population="pop1"),
        lambda: np.nansum(allel.mean_pairwise_difference(
            g.count_alleles(subpop=pop1_dip))))

    add("theta_w",
        lambda: diversity.theta_w(hm, population="pop1"),
        lambda: allel.watterson_theta(pos_allel,
            g.count_alleles(subpop=pop1_dip)))

    add("tajimas_d",
        lambda: diversity.tajimas_d(hm, population="pop1"),
        lambda: allel.tajima_d(g.count_alleles(subpop=pop1_dip)))

    # --- Divergence ---
    fst_plink_cmd = [PLINK2, "--bfile", prefix, "--fst", "POP",
                     "--pheno", pop_path, "--out", os.path.join(tmpdir, "fst_bench"),
                     "--allow-extra-chr"]
    add("fst_hudson",
        lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
        lambda: allel.hudson_fst(
            g.count_alleles(subpop=pop1_dip),
            g.count_alleles(subpop=pop2_dip)),
        plink_cmd=fst_plink_cmd)

    add("fst_wc",
        lambda: divergence.fst_weir_cockerham(hm, "pop1", "pop2"),
        lambda: allel.weir_cockerham_fst(g, [pop1_dip, pop2_dip]))

    add("dxy",
        lambda: divergence.dxy(hm, "pop1", "pop2"),
        lambda: allel.sequence_divergence(pos_np,
            g.count_alleles(subpop=pop1_dip),
            g.count_alleles(subpop=pop2_dip)))

    # --- SFS ---
    add("sfs",
        lambda: sfs.sfs(hm, population="pop1"),
        lambda: allel.sfs(g.count_alleles(subpop=pop1_dip)[:, 1]))

    # --- PCA ---
    pca_plink_cmd = [PLINK2, "--bfile", prefix, "--pca", "10",
                     "--out", os.path.join(tmpdir, "pca_bench"),
                     "--allow-extra-chr"]
    add("pca (10 PCs)",
        lambda: decomposition.pca(hm, n_components=10, scaler='patterson'),
        lambda: allel.pca(g.to_n_alt(), n_components=10, scaler='patterson'),
        plink_cmd=pca_plink_cmd)

    # --- Heterozygosity ---
    add("het_expected",
        lambda: diversity.heterozygosity_expected(hm, population="pop1"),
        lambda: allel.heterozygosity_expected(
            g.count_alleles(subpop=pop1_dip).to_frequencies(), ploidy=2))

    add("het_observed",
        lambda: diversity.heterozygosity_observed(hm, population="pop1"),
        lambda: allel.heterozygosity_observed(g.subset(sel1=pop1_dip)))

    # --- Results ---
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR_TBL}/benchmark_simulated.csv", index=False)
    print(f"\nSaved to {OUT_DIR_TBL}/benchmark_simulated.csv")

    # Figure
    has_plink = 'plink_s' in df.columns and df['plink_s'].notna().any()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 2))

    x = np.arange(len(df))
    w = 0.25 if has_plink else 0.35
    ax.barh(x - w, df['pg_gpu_s'], w, label='pg_gpu', color='#2ecc71',
            edgecolor='0.3', linewidth=0.5)
    ax.barh(x, df['allel_s'], w, label='scikit-allel', color='#e74c3c',
            edgecolor='0.3', linewidth=0.5)
    if has_plink:
        plink_vals = df['plink_s'].fillna(0)
        ax.barh(x + w, plink_vals, w, label='PLINK2', color='#3498db',
                edgecolor='0.3', linewidth=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(df['statistic'])
    ax.set_xscale('log')
    ax.set_xlabel('Wall-clock time (seconds)')
    ax.set_title(f'Performance: pg_gpu vs allel vs PLINK2\n'
                 f'({n_hap} haplotypes, {hm.num_variants:,} variants, simulated)')
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR_FIG}/benchmark_simulated.pdf", bbox_inches='tight')
    print(f"Figure saved to {OUT_DIR_FIG}/benchmark_simulated.pdf")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
