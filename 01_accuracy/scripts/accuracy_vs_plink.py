#!/usr/bin/env python
"""
Compare pg_gpu vs PLINK2 numerical accuracy.

Simulates data with msprime, exports to PLINK format, runs PLINK2
for FST, heterozygosity, and PCA, then compares against pg_gpu.

Uses simulated data (no missing data) to ensure both tools
operate on identical input.

Produces:
  - tables/accuracy_vs_plink.csv
  - figures/accuracy_vs_plink_pca.pdf
"""

import os
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
import msprime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import HaplotypeMatrix, diversity, divergence, decomposition

OUT_DIR_FIG = "01_accuracy/figures"
OUT_DIR_TBL = "01_accuracy/tables"

PLINK2 = os.path.expanduser("~/bin/plink2")
N_DIP = 200  # diploid individuals per population
N_POP = 2


def simulate():
    """Simulate two-population data with msprime."""
    print("Simulating data...", flush=True)
    demography = msprime.Demography()
    demography.add_population(name="pop1", initial_size=10_000)
    demography.add_population(name="pop2", initial_size=10_000)
    demography.add_population(name="anc", initial_size=10_000)
    demography.add_population_split(time=1000, derived=["pop1", "pop2"],
                                     ancestral="anc")

    ts = msprime.sim_ancestry(
        samples={"pop1": N_DIP, "pop2": N_DIP},
        sequence_length=5_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=42, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
    print(f"  {ts.num_samples} haplotypes, {ts.num_mutations:,} variants")
    return ts


def export_plink(ts, tmpdir):
    """Export tree sequence to PLINK bed/bim/fam format."""
    hm = HaplotypeMatrix.from_ts(ts)
    hm = hm.apply_biallelic_filter()
    hap = hm.haplotypes  # (n_hap, n_var)
    n_hap, n_var = hap.shape
    n_ind = n_hap // 2
    pos = hm.positions
    print(f"  {n_var:,} biallelic variants after filtering")

    # Write VCF, then convert with plink2
    vcf_path = os.path.join(tmpdir, "data.vcf")
    with open(vcf_path, 'w') as f:
        # Header
        f.write("##fileformat=VCFv4.1\n")
        f.write("##contig=<ID=1>\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for i in range(n_ind):
            pop = "pop1" if i < N_DIP else "pop2"
            f.write(f"\t{pop}_ind{i}")
        f.write("\n")

        # Variants
        for j in range(n_var):
            f.write(f"1\t{int(pos[j])}\tsnp{j}\tA\tT\t.\tPASS\t.\tGT")
            for i in range(n_ind):
                a1 = int(hap[2 * i, j])
                a2 = int(hap[2 * i + 1, j])
                f.write(f"\t{a1}|{a2}")
            f.write("\n")

    # Convert to plink format
    prefix = os.path.join(tmpdir, "data")
    result = subprocess.run([PLINK2, "--vcf", vcf_path, "--make-bed",
                             "--out", prefix, "--allow-extra-chr"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PLINK make-bed failed (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "")
        print(result.stdout[-500:] if result.stdout else "")
        raise RuntimeError("PLINK make-bed failed")

    # Write population file for FST (FID=0 matches PLINK's default from VCF)
    pop_path = os.path.join(tmpdir, "pops.txt")
    with open(pop_path, 'w') as f:
        f.write("#FID\tIID\tPOP\n")
        for i in range(n_ind):
            pop = "pop1" if i < N_DIP else "pop2"
            iid = f"{pop}_ind{i}"
            f.write(f"0\t{iid}\t{pop}\n")

    return prefix, pop_path, hm


def run_plink_fst(prefix, pop_path, tmpdir):
    """Run PLINK2 FST."""
    out = os.path.join(tmpdir, "fst")
    result = subprocess.run(
        [PLINK2, "--bfile", prefix, "--fst", "POP",
         "--pheno", pop_path, "--out", out, "--allow-extra-chr"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  PLINK FST failed: {result.stderr[-300:]}")
        return np.nan
    # Parse .fst.summary (tab-delimited: #POP1 POP2 HUDSON_FST)
    summary_path = out + ".fst.summary"
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path, sep='\t')
        if 'HUDSON_FST' in df.columns and len(df) > 0:
            return float(df['HUDSON_FST'].iloc[0])
    print(f"  Could not parse FST from PLINK output")
    return np.nan


def run_plink_het(prefix, tmpdir):
    """Run PLINK2 heterozygosity."""
    out = os.path.join(tmpdir, "het")
    subprocess.run([PLINK2, "--bfile", prefix, "--het",
                    "--out", out, "--allow-extra-chr"],
                   capture_output=True, check=True)
    het_path = out + ".het"
    if os.path.exists(het_path):
        df = pd.read_csv(het_path, sep=r'\s+')
        # F = inbreeding coefficient
        return df['F'].values
    return None


def run_plink_pca(prefix, tmpdir, n_components=10):
    """Run PLINK2 PCA."""
    out = os.path.join(tmpdir, "pca")
    subprocess.run([PLINK2, "--bfile", prefix, "--pca", str(n_components),
                    "--out", out, "--allow-extra-chr"],
                   capture_output=True, check=True)
    eigenvec_path = out + ".eigenvec"
    eigenval_path = out + ".eigenval"
    if os.path.exists(eigenvec_path):
        vecs = pd.read_csv(eigenvec_path, sep=r'\s+')
        pc_cols = [c for c in vecs.columns if c.startswith('PC')]
        coords = vecs[pc_cols].values
        vals = np.loadtxt(eigenval_path) if os.path.exists(eigenval_path) else None
        return coords, vals
    return None, None


def procrustes_align(X, Y):
    """Align Y to X via Procrustes (handles sign/rotation ambiguity)."""
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # SVD of cross-covariance
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = (Vt.T @ U.T)
    Y_aligned = Y @ R
    return Y_aligned


def main():
    ts = simulate()
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix, pop_path, hm = export_plink(ts, tmpdir)
        n_hap = hm.num_haplotypes
        n_half = n_hap // 2
        hm.sample_sets = {
            "pop1": list(range(n_half)),
            "pop2": list(range(n_half, n_hap)),
        }
        hm.transfer_to_gpu()

        print(f"\n{hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants")

        # --- FST ---
        print("\nFST:", flush=True)
        pg_fst = divergence.fst_hudson(hm, "pop1", "pop2")
        plink_fst = run_plink_fst(prefix, pop_path, tmpdir)
        # PLINK uses Hudson's FST by default
        if not np.isnan(plink_fst):
            rel_err = abs(pg_fst - plink_fst) / max(abs(plink_fst), 1e-15)
            rows.append({"statistic": "fst_hudson", "pg_gpu": pg_fst,
                         "plink": plink_fst, "rel_error": rel_err})
            print(f"  pg_gpu={pg_fst:.6f}  plink={plink_fst:.6f}  rel_err={rel_err:.2e}")
        else:
            print(f"  pg_gpu={pg_fst:.6f}  plink=FAILED")

        # --- Heterozygosity / Inbreeding ---
        print("\nInbreeding coefficient (F):", flush=True)
        pg_f = diversity.inbreeding_coefficient(hm)
        plink_f = run_plink_het(prefix, tmpdir)
        if plink_f is not None:
            # PLINK reports per-individual F; pg_gpu reports per-variant F
            # These are different statistics -- compare means
            pg_mean_f = float(np.nanmean(pg_f))
            plink_mean_f = float(np.mean(plink_f))
            rows.append({"statistic": "mean_F", "pg_gpu": pg_mean_f,
                         "plink": plink_mean_f, "rel_error": np.nan,
                         "note": "per-variant vs per-individual"})
            print(f"  pg_gpu mean F (per-variant)={pg_mean_f:.6f}")
            print(f"  plink  mean F (per-individual)={plink_mean_f:.6f}")
            print(f"  (different statistics: per-variant vs per-individual)")

        # --- PCA ---
        print("\nPCA:", flush=True)
        n_pc = 10
        # pg_gpu PCA returns per-haplotype coords; average pairs for per-individual
        pg_hap_coords, pg_var = decomposition.pca(hm, n_components=n_pc,
                                                    scaler='patterson')
        pg_coords = (pg_hap_coords[0::2] + pg_hap_coords[1::2]) / 2.0
        plink_coords, plink_vals = run_plink_pca(prefix, tmpdir, n_pc)

        if plink_coords is not None:
            # Align via Procrustes (PCA has sign/rotation ambiguity)
            plink_aligned = procrustes_align(pg_coords, plink_coords)

            # Correlation per PC
            for pc in range(min(5, n_pc)):
                corr = abs(np.corrcoef(pg_coords[:, pc],
                                        plink_aligned[:, pc])[0, 1])
                rows.append({"statistic": f"PCA_PC{pc+1}_corr",
                             "pg_gpu": np.nan, "plink": np.nan,
                             "rel_error": 1.0 - corr})
                print(f"  PC{pc+1}: |corr| = {corr:.6f}")

            # Make PCA comparison figure
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

            n_ind = pg_coords.shape[0]
            pops = np.array(["pop1"] * (n_ind // 2) + ["pop2"] * (n_ind // 2))
            colors = {"pop1": "#2ecc71", "pop2": "#e74c3c"}

            for ax, coords, title in [(axes[0], pg_coords, "pg_gpu"),
                                       (axes[1], plink_aligned, "PLINK2")]:
                for pop in ["pop1", "pop2"]:
                    mask = pops == pop
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               c=colors[pop], label=pop, s=10, alpha=0.6)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(title)
                ax.legend(fontsize=9)

            fig.suptitle("PCA comparison: pg_gpu vs PLINK2", y=1.02)
            plt.tight_layout()
            fig.savefig(f"{OUT_DIR_FIG}/accuracy_vs_plink_pca.pdf",
                        bbox_inches='tight')
            print(f"  Figure saved to {OUT_DIR_FIG}/accuracy_vs_plink_pca.pdf")

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR_TBL}/accuracy_vs_plink.csv", index=False)
    print(f"\nResults saved to {OUT_DIR_TBL}/accuracy_vs_plink.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
