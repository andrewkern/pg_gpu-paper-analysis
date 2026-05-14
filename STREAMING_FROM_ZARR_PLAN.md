# Streaming `HaplotypeMatrix.from_zarr()` for biobank-scale stores

A plan for migrating the prefetch + GPU-side prep pipeline prototyped in
`06_simulated_genome_scan/scripts/genome_scan_ooa.py` into the `pg_gpu`
library, so users with a chromosome-sized VCZ store can call
`HaplotypeMatrix.from_zarr(path, ...)` and get the right thing
automatically.

## Motivation

A 100k-haplotype human chromosome is multi-TB as a dense int8 matrix and
cannot live on a single GPU (or even reasonably in a single host
allocation). Its on-disk VCZ representation is ~6 GB blosc-compressed. The
current `HaplotypeMatrix.from_zarr` materializes the whole thing eagerly,
which is fine at Ag1000G scale (~3k haps) and impossible at biobank
scale.

We demonstrated in the simulated chr15 scan that the right pattern is
chunk-streaming with prefetch: hold one ~1 Mb genomic chunk in GPU
memory, compute on it, free it, move on. The producer (zarr read +
upload) overlaps with the consumer (GPU compute), so steady-state wall
time is `max(read, compute)` per chunk. On chr15 at 100k haps/pop this
runs in 94 min for the windowed scan and ~3.5 h end-to-end (windowed +
LD probes + heatmap + plot), with `cpu/wall = 3-5x` peak parallelism
and no OOM.

The goal is to make this the default behavior of `from_zarr` whenever
the requested matrix would not fit in GPU memory, while keeping the
existing eager API exactly as it is for small stores.

## What was measured in the prototype

Numbers below are for the chr15 OOA_2T12 sim at 100k diploids/pop
(200k haps, 11.64M variants, 6.4 GB VCZ on NVMe; A100 80 GB, ~64 cores,
1 TB host RAM). Cite them in the design doc when justifying choices --
they are the empirical floor.

* **`np.ascontiguousarray(gm.T)` on a tall-skinny int8 chunk takes
  ~64 s on a 25k-site chunk**, single-threaded and cache-thrashing.
  Doing the same transpose on the GPU after a Gen4 x16 PCIe upload
  takes ~8 s on the same chunk. The library `build_hm` must do the
  transpose on the device, not the host.
* **`gt[gt[:,0,0]>=0]` and the ploidy reshape `gm[:,:n_dip]=gt[:,:,0];
  gm[:,n_dip:]=gt[:,:,1]` each cost ~30 s on a full 1 Mb chunk** because
  they allocate, page-fault, and memcpy 28 GB of host memory per chunk.
  Both must move to the GPU.
* **`gt_gpu[mask]` (cupy boolean indexing on a `(n_var, n_dip, 2)` array)
  attempts to allocate 227 GB of int64 gather indices on a 1 Mb / 200k-hap
  chunk and OOMs immediately**. Use `gt_gpu.compress(mask, axis=0)`
  instead -- single output allocation, no broadcast.
* **Stripe-parallel reads via `ThreadPoolExecutor` went slower** because
  zarr 3's async pipeline + blosc internal threading already achieve
  `cpu/wall ~ 3.5x` from a single `cg[lo:hi,:,:]` call. Adding
  ThreadPoolExecutor on top contends in the allocator and through the
  GIL. The single producer is the right default.
* **bio2zarr's default sample-axis chunking `(10000, 1000, 2)` is faster
  than our initial `(10000, n_dip, 2)`** on both access patterns: full
  reads at 100k-dip chunks were 4.29 s / 14k variants (cpu/wall=1.4x);
  the same read at 1k-dip chunks was 2.40 s / 14k variants
  (cpu/wall=4.7x), because more chunks means more parallel blosc decodes.
  Sample subsets win even more (5000-hap oindex was 2.93 s -> 1.38 s)
  because oindex only decompresses the sample chunks the requested haps
  live in. The library should not assume any particular chunking but
  should perform well across the bio2zarr-shaped defaults.
* **Cold and warm reads are essentially identical (~28-36 s / 1 Mb /
  200k-hap)** -- disk I/O is not the bottleneck at this scale on NVMe.
  CPU-side zstd/blosc decompression at ~1 GB/s of decoded output is.
* **prefetch=1 saved 42% of total scan wall** (`overlap saved 4,041 s
  of 9,723 s sequential = 42%`). prefetch>1 does not help when compute
  is the bottleneck.
* **GPU-side codec decode via `kvikio` + `nvidia-nvcomp` is fast,
  but only when the store is chunked the bio2zarr way.** Measured
  cold-cache on a 1 Mb / 200k-hap chunk after the rechunk to
  `(10000, 1000, 2)`:
  * Full-haplotype read: host 24.5 s @ 1.16 GB/s (cpu/wall=1.4x);
    kvikio + zarr GPU buffer **3.86 s @ 7.35 GB/s** (cpu/wall=4.9x).
    **6.34x speedup**, result already in GPU memory.
  * 5000-hap `oindex` (LD-probe pattern): host 30.3 s; kvikio
    **0.20 s @ 6.66x cpu/wall**. **152x speedup on the read alone**.
    Bytes byte-equal to the host path.
  Important caveat: kvikio with the **original** `(10000, n_dip, 2)`
  whole-sample-axis chunking shows **no speedup at all** on the full
  read (26.8 s vs 26.9 s host) and only ~4x on `oindex`. The win
  requires bio2zarr-style sample chunking; the two changes are
  complementary, not independent. zarr's host-buffer `oindex` does
  not appear to selectively decompress sample chunks even when the
  layout allows it (host oindex on the rechunked store was only ~1.4x
  faster than host oindex on the original); kvikio's GPU pipeline
  does the selective decode that the host pipeline doesn't, which is
  where the 152x asymmetry on oindex comes from.

  The win is **entirely in the GPU codec**, not in GPU Direct Storage:
  `compat_mode=ON` (kvikio bypasses cuFile, just does posix reads
  into host bounce buffers and hands bytes to the GPU codec) is
  actually *faster* than `compat_mode=AUTO` on systems without
  `/etc/cufile.json` configured, because AUTO pays a failed-handshake
  cost. The kvikio backend therefore has no filesystem dependency --
  just `kvikio + nvidia-nvcomp + a zarr store with a GPU-decodable
  codec`. Zstd is the zarr-3 default and is GPU-decodable; blosc, lz4,
  and deflate are also supported by nvCOMP.
* **End-to-end scan wall is compute-bound on both backends**, so the
  per-chunk read speedup translates into a smaller scan-wall win than
  the read numbers suggest. Projected impact on the chr15 example,
  rechunked store: windowed scan 94 min -> ~85 min (compute-bound;
  kvikio just makes the prefetched read trivially small), LD-probe
  phase 58 min -> ~35 min (pairwise r^2 compute still ~66 s per
  probe-pop pair regardless of how fast the read is), end-to-end
  3:28 -> ~2:30. The qualitatively important win is *not* wall: it's
  the host-memory budget. The kvikio path uses **0 host bytes per
  chunk in flight** vs 28 GB for the host path, which is what removes
  the practical ceiling on `prefetch` depth and lets the streaming
  class scale to chromosomes and sample counts where the decompressed
  matrix exceeds host RAM entirely.

## Out of scope for the first version

* Sharded / cloud-backed zarr stores. The store opener should just take
  whatever zarr accepts, but the streaming heuristics here are sized for
  local NVMe. Cloud reads dominated by latency would want a different
  prefetch depth.
* Multi-GPU. The streaming abstraction has nothing to do with multi-GPU;
  that's a separate axis.
* Compact representations (packed-bit, allele-count). The streaming path
  produces a `(n_hap, n_var)` int8 view because that is what every
  existing pg_gpu kernel takes.
* A native GPU VCF parser. See "Sizing guidance and the VCF warning"
  below -- the right answer for biobank-scale VCF is "convert once to
  VCZ"; pg_gpu surfaces this advice from the loader rather than trying
  to make `from_vcf` fast at that scale.

## API design

### Eager (unchanged for small stores)

```python
hm = HaplotypeMatrix.from_zarr("ag3.0.zarr", region="3R:1000000-2000000",
                               sample_sets={"BFM": [...], "AOM": [...]})
# returns a normal HaplotypeMatrix with all data on the GPU.
# windowed_analysis(hm, ...) works exactly as today.
```

This must keep working bit-for-bit for stores small enough to fit on
device. The trigger for streaming should be invisible at small scale.

### Streaming (new, for large stores)

```python
hm = HaplotypeMatrix.from_zarr("chr15.vcz",
                               sample_sets={"AFR": [...], "EUR": [...]},
                               # explicit when caller knows they want
                               # streaming, e.g. for a huge store on a
                               # smaller GPU
                               streaming="auto",         # auto | always | never
                               chunk_bp=1_500_000,       # genomic chunk width
                               prefetch=1)
# returns a StreamingHaplotypeMatrix, which has the same public surface
# as HaplotypeMatrix but materializes one chunk at a time.
```

`streaming="auto"` (default) chooses based on the heuristic in
"Detection" below.

### Compatibility shim

Every kernel that consumes a `HaplotypeMatrix` -- `windowed_analysis`,
`sfs.sfs`, `sfs.joint_sfs`, `pairwise_r2`, `compute_ld_statistics_gpu_*`,
`HaplotypeMatrix.pairwise_r2()`, the Garud kernel -- has to learn to
take a `StreamingHaplotypeMatrix` too.

For the **window-style kernels** the contract is straightforward: a
window never straddles a chunk boundary (we already align chunks to the
largest window size in the prototype), so the kernel can iterate chunks,
run itself on each, and concatenate results. The result is identical to
running on the materialized matrix.

For **global aggregate kernels** (`sfs.sfs`, `sfs.joint_sfs`, sum-style
LD estimators) the chunk results are added: `sfs_total = sum_c sfs(c)`.

For **pairwise kernels** (`pairwise_r2`, the r2 heatmap) we cannot
stream -- a pairwise computation needs both rows simultaneously. The
streaming path either rejects these or auto-falls-back to subsampling
into a normal in-memory `HaplotypeMatrix`, which is what the prototype
does for the heatmap. The library should expose
`StreamingHaplotypeMatrix.materialize(sample_subset=..., region=...)` as
the documented way to drop down to an eager matrix on a slice.

For **Garud's H and joint SFS subsample paths**, the kernel runs once
per chunk on a fixed `sample_subset` and the per-chunk results are
concatenated (Garud) or summed (joint SFS).

### Population metadata

VCZ has no canonical population field. The prototype reads a companion
`<store>.pops.tsv` (sample_id, population). The library should:

1. Accept an explicit `sample_sets={"pop": [hap_index, ...]}` argument
   (current API).
2. Accept `pop_file=PATH` pointing at a TSV with one
   `sample_id<TAB>population` row per diploid; resolve sample_id ->
   diploid index via the store's `sample_id` array and expand to
   `[2*i, 2*i + n_dip]` haplotype indices.
3. Look for `<store>.pops.tsv` next to the store as a default, with a
   one-line warning if used (so the user knows where the metadata came
   from).

This belongs on `HaplotypeMatrix.from_zarr` regardless of streaming
mode; the eager path benefits too.

## Internal architecture

```
HaplotypeMatrix.from_zarr(path, ..., streaming="auto")
    |
    +-- ZarrGenotypeSource(path, pop_file, region)        [new]
    |     .num_variants, .num_haplotypes, .site_pos
    |     .pop_cols : {pop_name -> hap-axis indices}
    |     .iter_chunks(chunk_bp, align_bp) -> [(left, right), ...]
    |     .slice_region(left, right)         -> (gt (n,n_dip,2) int8, pos float)
    |     .slice_subsample(left, right, hap_cols) -> (gm subset, pos)
    |
    +-- estimate_gpu_footprint()                           [new]
    |
    +-- if eager:
    |       eagerly read full region, return HaplotypeMatrix
    |    else:
    |       return StreamingHaplotypeMatrix(source, chunk_bp, prefetch,
    |                                       sample_sets)

StreamingHaplotypeMatrix
    .source, .chunks, .prefetch, .sample_sets
    .num_variants, .num_haplotypes (delegate to source)
    ._iter_gpu_chunks() -> yields (left, right, hm_chunk)
                          where hm_chunk is an in-memory HaplotypeMatrix
                          spanning [left, right]; the chunk is built on
                          a worker / producer model.
```

`_iter_gpu_chunks` is the core of the streaming abstraction. It owns:

* A `queue.Queue(maxsize=prefetch)` of `(left, right, gt, pos)` tuples
  filled by a producer thread calling `source.slice_region`.
* The host-to-device upload + biallelic filter + ploidy interleave +
  transpose, done on the GPU via `cp.asarray(gt) -> compress(mask,
  axis=0) -> transpose(2,1,0).reshape(2*n_dip, -1) ->
  ascontiguousarray`. This is the `build_hm` function from the
  prototype.
* Releasing the chunk's GPU memory at the end of the consumer step.

The kernels (`windowed_analysis(hm, ...)`) detect a
`StreamingHaplotypeMatrix` and loop:

```python
def windowed_analysis(hm, **kw):
    if isinstance(hm, StreamingHaplotypeMatrix):
        parts = []
        for left, right, chunk_hm in hm._iter_gpu_chunks():
            parts.append(_windowed_analysis_one(chunk_hm, **kw))
        return _concat(parts)
    return _windowed_analysis_one(hm, **kw)
```

The "one" function is the existing code path.

### Detection (streaming auto)

`HaplotypeMatrix.from_zarr(..., streaming="auto")` chooses streaming
when the eager footprint would not fit in free GPU memory with margin.

```
eager_bytes = n_variants * n_haplotypes * dtype_bytes   # int8 = 1
free_gpu    = cp.cuda.Device().mem_info[0]
if eager_bytes > 0.5 * free_gpu:    # half the device to leave room for kernels
    stream
else:
    eager
```

`streaming="always"` and `streaming="never"` are escape hatches and
have to coexist with the device-memory check (`always` on a tiny store
is fine; `never` on a huge store should raise).

### Chunk sizing

The default genomic chunk width is the one the user passes via
`chunk_bp` (default 1.5 Mb in the prototype, rounded down to a multiple
of the largest analysis window size so windows never straddle a chunk
boundary). The streaming object stores the alignment requirement and
asserts that all `window_size` arguments divide `chunk_bp`. The kernel
loop computes the chunk list lazily from `chunk_bp` and the source's
mappable range.

### Memory budget

Steady-state memory cost per chunk at our scale (1 Mb, 200k haps, 1.5%
multiallelic):

* Host: `(prefetch + 1) * chunk_bytes` for the queue + the chunk
  currently being uploaded. At 28 GB / chunk and `prefetch=1`, that is
  56 GB host. Should be a public attribute so the user can plan.
* Device: peak `~2 * chunk_bytes` during `build_hm` (the `(n_var,n_dip,2)`
  upload + the contiguous `(n_hap,n_var)` after compress + transpose),
  then `~1 * chunk_bytes` for the rest of the per-chunk compute. On A100
  80 GB this comfortably handles 1.5 Mb chunks at 200k haps; on A40
  48 GB the same workload requires 1 Mb chunks.

The library should also expose a `HaplotypeMatrix.estimate_chunk_bp(
gpu_free_bytes, n_haplotypes)` helper that returns a safe default. The
formula is `gpu_free_bytes / (2 * n_haplotypes * variants_per_bp * 1)`
with a 2x peak factor for the dual buffer, where `variants_per_bp` can
be read from the store (`n_variants / sequence_length`).

### Pluggable read backend: numpy-on-host vs kvikio-on-GPU

The streaming class fetches chunks through a pluggable
`ChunkFetcher` abstraction. Two implementations ship:

* `HostChunkFetcher` (default, no extra deps): the prototype's path --
  `zarr.storage.LocalStore` + numpy host buffers + `cp.asarray(gt)` to
  upload. Works everywhere zarr does, costs `chunk_bytes` of host RAM
  per in-flight chunk.
* `KvikioChunkFetcher` (opt-in, requires `kvikio` and
  `nvidia-nvcomp`): `kvikio.zarr.GDSStore` + `zarr.config.enable_gpu()`
  buffer prototype. Compressed bytes are read from disk and handed to
  the nvCOMP GPU codec; the decompressed chunk lands directly in GPU
  memory. ~3-4x faster per chunk (measured) and **zero host RAM per
  chunk in flight**, which is the qualitatively important property
  -- it removes the practical ceiling on `prefetch` depth and lets
  the streaming class run on hosts with far less RAM than the
  decompressed data would require.

Selection at construction:

```python
hm = HaplotypeMatrix.from_zarr(path, streaming="auto",
                               backend="auto")   # auto | host | kvikio
```

`backend="auto"` picks `kvikio` when the optional deps import cleanly
and the store's `call_genotype` codec is in the nvCOMP-supported list
(zstd, blosc, lz4, deflate). Falls back to `host` otherwise without
warning -- the host path is the correct baseline.

Two important defaults for the kvikio backend:

* **Force `kvikio.defaults.set({"compat_mode": kvikio.CompatMode.ON})`
  unless the user explicitly opts into GDS** via
  `backend="kvikio-gds"`. Measured: on this host (driver 535.288,
  CUDA 12.8, XFS, nvidia-fs loaded but no `/etc/cufile.json`), AUTO
  mode pays a failed-handshake cost and runs at ~20 s/chunk while
  COMPAT runs at ~7 s/chunk. AUTO can also be slow on systems that
  *do* have GDS configured if the filesystem path doesn't fully
  support it. COMPAT is reliably fast everywhere; opt into GDS only
  if the user is on a known-good storage path.
* `kvikio.defaults.set({"num_threads": 8})`. With 8 reader threads
  cpu/wall hit 5.3x; with fewer it scaled linearly down. Make this
  user-tunable via a kwarg on the fetcher.

The streaming class itself does not change between backends. Both
fetchers yield `(gt_or_gt_on_gpu, pos)` tuples; the `build_hm` step
is a no-op transfer when the chunk is already on the GPU (a
`cp.asarray` of a cupy array is a passthrough) and a real upload
when it's on the host.

A small caveat on what kvikio does *not* speed up: the per-chunk
`build_hm` GPU work (filter + reshape + transpose, ~27 s on a full
chunk) does not change. Stats kernels do not change. With the
current architecture, the windowed scan is compute-bound at ~65 s
per chunk; kvikio cuts the read from ~30 s to ~10 s but the wall
remains compute-bound. The major wall-time savings are:

* **LD-probe phase** (a `slice_subsample` over 5000 contiguous dips
  per probe): kvikio's 3x speedup compounds with the bio2zarr
  sample-chunk reduction (~50x), so probes should go from 4 min each
  to a few seconds each.
* **Host memory cost**: 0 GB/chunk in flight rather than 28 GB/chunk
  in flight. Lets `prefetch` go higher without hitting host limits.
* **Future compute pipelining**: if a later phase overlaps `build_hm`
  of chunk N+1 with stats of chunk N on a second cupy stream, the
  kvikio backend is the prerequisite that makes the GPU memory
  budget tractable (no extra host staging).

## Sizing guidance and the VCF warning

Users at biobank scale tend to arrive with one of three input shapes:
a VCF (or `.vcf.gz` / `.vcf.bgz`), a VCZ, or a tree sequence.
`pg_gpu` should make the right move easy, and emit a warning when
the wrong one is being attempted.

### Decision rule

* **Small stores** (any format, anything that fits eagerly in GPU
  memory): the existing eager `from_zarr` / `from_vcf` paths are
  fine. No streaming, no kvikio, no warning.
* **Mid scale** (~10k haps × whole chromosome, or 100k haps × small
  region): VCZ + eager `from_zarr` works. The streaming auto path
  also works and is transparent. VCF in this band still loads but
  prefer VCZ for repeated access.
* **Biobank scale** (≥10k samples × whole chromosome, or any
  combination producing > ~50 GB raw genotype matrix): VCZ +
  streaming + kvikio backend. `from_vcf` at this scale is the wrong
  tool -- text parsing is single-threaded in `htslib`/`cyvcf2` and
  will dominate the wall regardless of how fast `pg_gpu`'s downstream
  kernels are.

### The VCF biobank-scale warning

`HaplotypeMatrix.from_vcf(path, ...)` should detect inputs that are
"too big to load this way" and emit a `BiobankScaleWarning` with an
actionable message before it starts parsing. The check is cheap:

* `os.path.getsize(path)` for the on-disk byte count.
* If the file is `.gz` / `.bgz` the uncompressed size is at most
  ~50x larger; conservatively use the on-disk size as a proxy.
* Number of samples is in the `#CHROM` header line; reading and
  splitting that line is a few hundred microseconds.

Threshold heuristic (tunable, conservative): warn if
`(file_size > 10 GiB) or (n_samples > 5000)`. The intent is to catch
"this load will take hours, and you'll want it converted once" while
not bothering users of 1000 Genomes-scale VCFs (~1 GB compressed,
~2500 samples).

The warning text should be specific and copy-pastable, and should
point at pg_gpu's own `vcf_to_zarr` wrapper rather than the bare
`vcf2zarr` CLI -- bio2zarr is already a declared dependency and
`HaplotypeMatrix.vcf_to_zarr(...)` is the in-library entry point that
calls it.

```
BiobankScaleWarning: /path/to/chr15.vcf.gz is 47.3 GB with 200,000
samples. Loading this VCF will take hours because VCF text parsing
is single-threaded in htslib. For biobank-scale repeated analysis,
convert once to VCZ and use HaplotypeMatrix.from_zarr() instead --
subsequent reads finish in seconds rather than hours, and the
optional kvikio backend can speed up sample-subset reads (LD probes,
joint SFS subsamples) by 100x+.

One-time conversion (uses bio2zarr under the hood):
    from pg_gpu import HaplotypeMatrix
    HaplotypeMatrix.vcf_to_zarr("/path/to/chr15.vcf.gz",
                                "/path/to/chr15.vcz",
                                worker_processes=16)

Then in pg_gpu:
    hm = HaplotypeMatrix.from_zarr("/path/to/chr15.vcz",
                                   pop_file="/path/to/chr15.pops.tsv",
                                   backend="kvikio")  # or "auto"

To silence this warning:
    import warnings; warnings.filterwarnings("ignore",
        category=pg_gpu.BiobankScaleWarning)
```

`HaplotypeMatrix.vcf_to_zarr` already exists today and is the right
hook -- the only change is that the warning surfaces it at the moment
the user is about to do the slow thing.

The warning is just a warning -- the load proceeds. Users with a
genuine need to load big VCFs directly (e.g. for a one-off
sanity-check that doesn't merit a conversion) can silence it. The
filter category lives at the `pg_gpu` top level so the silence
incantation is one line.

### Why no kvikio path for VCF?

kvikio + nvCOMP accelerates **codec decompression**; VCF's bottleneck
is **text parsing**, which neither library addresses. nvCOMP does
support deflate / gzip and could in principle decompress bgzip blocks
on the GPU, but the next step (find tab positions per row, split
FORMAT vs samples, parse `0|1` -> int8) is CPU-bound regardless of
how the bytes got into memory. The fastest realistic VCF path stays
on CPU through `htslib`. There is no `cudf`-style polished VCF
parser available; building one is a separate research project, not
a quick win.

`pg_gpu` should not add a kvikio-VCF backend. The library already
exposes `HaplotypeMatrix.vcf_to_zarr(vcf_paths, zarr_path, ...)` (a
wrapper around bio2zarr) and `GenotypeMatrix.vcf_to_zarr`. The
streaming work should make the "convert VCF -> VCZ once" path as
discoverable as possible -- the warning above is the primary lever,
and the docs page below shows the conversion in three lines of
Python that don't require dropping out to a CLI.

### Docs

A new docs page `Loading biobank-scale data` covers:

* The three-way decision tree above (small / mid / biobank), with
  rough thresholds and a worked example for each.
* The one-time conversion recipe via the existing
  `HaplotypeMatrix.vcf_to_zarr(...)` / `GenotypeMatrix.vcf_to_zarr(...)`
  wrappers around bio2zarr, with approximate wall times per chromosome
  at common sample counts. (No CLI step; bio2zarr is already a
  declared dependency.)
* The kvikio backend setup (pip install kvikio nvidia-nvcomp; one
  paragraph on `compat_mode` and why we default it to ON; one
  paragraph on the GPU codec compatibility matrix -- zstd is the
  zarr-3 default and is always supported).
* The `BiobankScaleWarning` class, what triggers it, and how to
  silence it.
* A table of measured wall times at each scale on the chr15 sim, so
  users have a calibration point before they start. Numbers from the
  prototype:

  | sample count | input | path | wall time |
  |--------------|-------|------|-----------|
  | 2500 (1kG)   | VCF.gz| from_vcf eager | ~ 30 s |
  | 10k (mid)    | VCZ   | from_zarr eager | ~ 5 min |
  | 100k (biobank)| VCZ + bio2zarr chunking | from_zarr streaming + kvikio | ~ 2 h 30 min end-to-end |
  | 100k (biobank)| VCF.gz | from_vcf | days; **WARN** |

## Implementation phases

Each phase is independently shippable and testable; the eager API never
breaks.

### 1. Extract `ZarrGenotypeSource` into pg_gpu

Lift `ZarrSource` from the prototype into
`pg_gpu/io/zarr_source.py`. Keep its `slice_region` /
`slice_subsample` API, the auto-detection of single-contig stores, and
the companion `<store>.pops.tsv` resolution. Add a docstring covering
the chunked-zarr access contract and what types it returns. This step
is pure code movement plus an import smoke test.

Unit tests: round-trip a tiny in-memory zarr (built via
`zarr.create_group`) and verify shapes, dtypes, biallelic-filter
behavior on multiallelic rows, and `oindex` on a non-contiguous sample
subset.

### 2. GPU-side `build_hm` as a public helper

Promote `build_hm(gt, pos, ...)` from the prototype to a public
constructor: `HaplotypeMatrix.from_raw_genotypes(gt, pos, ...)`. It
takes a `(n_var, n_dip, 2)` int8 array (host or device), does the
biallelic filter via `compress(axis=0)`, does the ploidy interleave +
transpose on the GPU, and returns a `HaplotypeMatrix`. This makes the
existing eager `from_zarr` cheaper too: the eager path drops its own
host-side filter / reshape and goes through the same GPU prep code.

Tests:
- Identity round-trip: build_hm of a synthetic gt vs the manual
  `(haps as np.empty + assign)` reference; assert byte-equal.
- All-multiallelic chunk -> empty matrix without error.
- 0-variant chunk -> empty matrix without error.

### 3. `StreamingHaplotypeMatrix` shell + iterator

Add the streaming class with `_iter_gpu_chunks()`, the producer
thread, the bounded queue, and the per-chunk GPU prep. No public kernel
support yet; this phase just makes the iterator usable from tests.

Tests:
- Stream a small synthetic store of 200k variants at 1000 haps in
  10k-variant chunks; verify the yielded chunks reassemble to the
  source bit-for-bit.
- prefetch=0 vs prefetch=1: same output, prefetch faster.
- Producer-side error (force a corrupt chunk) is raised on the
  consumer side with its traceback preserved.

### 4. Streaming-aware `windowed_analysis`

Wire `windowed_analysis` to detect a `StreamingHaplotypeMatrix`, loop
chunks, and concatenate. Assert that `window_size` divides the chunk
alignment.

Tests:
- `windowed_analysis(streaming_hm, ...) == windowed_analysis(eager_hm,
  ...)` for a small store on every supported statistic and on
  multi-population requests.
- A window size that does not divide chunk alignment raises a clear
  error.

### 5. Streaming-aware aggregate kernels

`sfs.sfs`, `sfs.joint_sfs`, `compute_ld_statistics_gpu_single_pop`
(sum-only path), Garud's H over fixed subsamples. Loop and aggregate;
share an `_aggregate_over_chunks(kernel, reducer)` utility.

Tests: same equivalence assertion vs the eager path.

### 6. Pairwise kernels via materialize()

`pairwise_r2` and the r^2 heatmap path raise on a
`StreamingHaplotypeMatrix` directly, but the caller can call
`.materialize(region=(lo, hi), sample_subset=cols)` to get an eager
`HaplotypeMatrix` over a slice. Document the pattern in the docstring;
update the example script and the chr15 scan to use it explicitly.

### 7. `from_zarr` auto-detection

Add the `streaming="auto"` heuristic and the device-memory check.
Update the eager docstring to mention the new mode. The
`streaming="never"` path on a too-large store raises a `MemoryError`
with an actionable message that includes the projected eager bytes and
the free GPU memory.

Tests:
- A small synthetic store -> eager path; a large synthetic store ->
  streaming path; `streaming="always"` on a small store works.

### 8. `ChunkFetcher` abstraction + `KvikioChunkFetcher`

Introduce the `ChunkFetcher` ABC with `iter_chunks() -> (gt, pos)` and
add two implementations: `HostChunkFetcher` wrapping the current path
and `KvikioChunkFetcher` using `kvikio.zarr.GDSStore` +
`zarr.config.enable_gpu()`. The streaming class takes a fetcher in its
constructor; nothing else changes.

The kvikio fetcher must:

* Default `kvikio.defaults.set({"compat_mode": kvikio.CompatMode.ON,
  "num_threads": 8})`. Both are user-overridable.
* Probe the store's `call_genotype` codec at construction and refuse
  (with a clear `ValueError`) if it isn't one nvCOMP supports.
* Reset zarr's buffer prototype on `__exit__` / `close()` so the
  global config is not left in GPU mode for the rest of the program.

Tests:
- Build a synthetic zstd-coded zarr; `KvikioChunkFetcher` produces
  byte-equal chunks to `HostChunkFetcher`.
- Pass a synthetic store with an unsupported codec; constructor raises.
- The fetcher's import-side dependencies are guarded so the host path
  works on a machine without kvikio installed.

### 9. `from_zarr` backend auto-detection

Add `backend="auto" | "host" | "kvikio" | "kvikio-gds"`. `auto` picks
`kvikio` when **all** of the following hold:

* `kvikio` and `nvidia.nvcomp` import cleanly.
* `call_genotype`'s codec is in the nvCOMP-supported list (zstd,
  blosc, lz4, deflate).
* `call_genotype.chunks` has a sample-axis chunk size of less than
  about `n_dip / 4` (a heuristic for "bio2zarr-style" chunking).
  Whole-sample-axis chunking (`chunks[1] == n_dip`) gets no kvikio
  speedup on the full-read path (measured: 26.8 s vs 26.9 s host on
  our `chr15.vcz.oldchunk` store), only on `oindex`. To avoid the
  cost of swapping backends mid-process, auto picks `host` when the
  chunks aren't bio2zarr-shaped. A one-line `BadlyChunkedWarning`
  fires with a pointer to `HaplotypeMatrix.vcf_to_zarr(...)` /
  `bio2zarr` so the user knows the store is leaving performance on
  the table.

`kvikio-gds` is the explicit opt-in that sets `compat_mode=AUTO` and
skips the COMPAT default. `backend="kvikio"` is the explicit opt-in
that skips the chunk-shape heuristic (for users who know what they're
doing).

Tests:
- Import-time fallback to `host` is silent on a kvikio-less machine.
- `backend="kvikio"` on an unsupported-codec store raises with the
  same message as phase 8's codec-probe error.
- `backend="auto"` on a store with `chunks[1] == n_dip` picks `host`
  and emits one `BadlyChunkedWarning`.
- Same auto on a bio2zarr-shaped store picks `kvikio`.

### 10. `BiobankScaleWarning` on `from_vcf`

Add `pg_gpu.BiobankScaleWarning(UserWarning)` and have both
`HaplotypeMatrix.from_vcf` and `GenotypeMatrix.from_vcf` warn before
parse when:

```
(file_size > BIOBANK_VCF_WARN_BYTES) and
(region is None or _region_span_bp(region) > BIOBANK_VCF_WARN_REGION_BP)
and not _warned_for_path(path)
```

or when `n_samples_in_header > BIOBANK_VCF_WARN_SAMPLES`. The
AND-on-region clause is the "wrong tool vs too big" distinction from
"Risks and open questions": a user pulling
`chr15:30_000_000-30_100_000` from a 47 GB VCF is doing the right
thing via tabix and should not be warned.

Constants live at the module top:

```python
BIOBANK_VCF_WARN_BYTES   = 10 * 1024**3   # 10 GiB on-disk
BIOBANK_VCF_WARN_SAMPLES = 5_000
BIOBANK_VCF_WARN_REGION_BP = 5_000_000    # 5 Mb
```

Header parse is cheap (read until the first non-`##` line, split
tabs, count from column 10 onward); cache the result on the path.
The warning text is the copy-pastable block in "The VCF biobank-scale
warning", and it explicitly mentions `HaplotypeMatrix.vcf_to_zarr(...)`
since that's already in the public API.

Tests:
- Small VCF (1000 Genomes-scale, ~1 GB, 2500 samples): no warning.
- Synthetic VCF with 10k samples in header: warning fires once;
  parsing still proceeds.
- 47 GB VCF with `region="chr15:30000000-30100000"`: no warning
  (region is small).
- 47 GB VCF with no region: warning fires.
- `warnings.filterwarnings("ignore", category=BiobankScaleWarning)`
  silences it.
- Second call on the same path within a process: no duplicate
  warning.

### 11. Documentation + the chr15 example

* New docs page `Loading biobank-scale data` with the three-way
  decision tree, the one-time `bio2zarr` recipe, the kvikio backend
  setup paragraph, and the wall-time calibration table from "Sizing
  guidance".
* Replace the `genome_scan_ooa.py` prototype's private `ZarrSource` /
  `chunk_iterator` / `build_hm` / `windowed_scan` glue with two-line
  calls into the new library API. The example becomes the canonical
  streaming + kvikio demo.

## Performance acceptance criteria

Re-run the chr15 scan against the migrated library and verify:

* Same `chromosome_summary.json` numbers (within float tolerance) for
  every backend.
* `host` backend wall within 10% of the prototype's 3.5 h
  (post-rechunk: ~2 h).
* `kvikio` backend: per-chunk read time at most half the `host`
  backend's. (We measured ~3.4x faster reads; 2x is the conservative
  acceptance bar to absorb noise.)
* `kvikio` backend: per-chunk *host* RAM allocation stays at a small
  fixed bookkeeping cost (target < 1 GB resident), no growth with
  prefetch depth.
* Memory pool peak stays below 80 GB on GPU 0 on either backend at
  1.5 Mb chunks / 200k haps.
* `cpu/wall` stays above 3x on the producer side for `host` and
  above 5x for `kvikio`.
* `from_vcf` on a small VCF (~1 GB / 2500 samples): no warning.
* `from_vcf` on a synthetic 10k-sample header: warning fires, parse
  proceeds.

If any criterion regresses, root-cause before merging.

## Risks and open questions

* **Statistics that need cross-chunk state.** `tajimas_d` and
  `fay_wu_h` are per-window and self-contained, so they stream cleanly.
  But anything we add later that uses a chromosome-wide reference
  spectrum (e.g. a normalized iHS) needs a two-pass design: chunk-pass
  one builds the reference, chunk-pass two does the per-window calc.
  This should be designed in once we add such a stat; for now, document
  the constraint.
* **Determinism.** `cp.compress` and the GPU transpose are deterministic
  but kernel order across chunks may not be. Window-aligned chunking
  keeps results bit-for-bit deterministic per window even under
  different chunk schedules; the test in phase 4 checks this.
* **What chunking does the user have?** Our test only covered
  `(10000, 1000, 2)` (bio2zarr default) and `(10000, 100000, 2)` (our
  prototype's chunking). Stores with very different shapes (smaller
  variant chunks, larger sample chunks, sharded) may interact with the
  read pipeline differently. The library should treat the source's
  chunk shape as opaque and let zarr handle it, but a one-shot read
  benchmark at construction (read the first variant chunk, time it)
  could warn if `cpu/wall < 1.5` and recommend a rechunk.
* **Sample subset reads (LD probes).** `slice_subsample(...)` via
  `oindex` is fast under bio2zarr-style sample chunking but pathological
  under whole-sample-axis chunks (decompresses the whole axis and
  throws away the rest). The library should be honest: in the
  `oindex`-pessimal case it falls back to a full read + host-side mask,
  with a warning.
* **kvikio codec coverage.** nvCOMP currently has GPU decoders for
  zstd, blosc, lz4, and deflate. Stores using other codecs (e.g. the
  legacy `numcodecs.LZMA`, or a custom codec) cannot use the kvikio
  path. The fetcher must probe and fall back cleanly. zarr-3's default
  codec is zstd, so any store written by recent `bio2zarr` or our
  `ts_to_vcz.py` is compatible; older blosc-encoded VCZ stores are
  also compatible because nvCOMP supports the blosc meta-codec.
* **`zarr.config.enable_gpu()` is process-global state.** The kvikio
  fetcher must restore the CPU buffer prototype when it's done so the
  user can mix streaming and eager calls in the same process. Use a
  context manager scoped to the fetcher's lifetime; explicitly reset
  on `close()` and `__del__`.
* **Distinguish "too big to load" from "wrong tool".** The
  `BiobankScaleWarning` triggers on file size and sample count, but
  the *right* answer depends on access pattern. A user who wants one
  region from a huge VCF (e.g. `chr15:30000000-30100000`) is fine
  using tabix-region reads via `from_vcf(region=...)`. The warning
  text should mention this case so users with one-off region queries
  aren't pushed into a 4-hour conversion they don't need. Threshold
  the warning on `(file_size > 10 GiB) AND
  (region is None or region_size > 5 Mb)`.
* **Should this live in `pg_gpu` at all or in a sibling package?** The
  streaming logic is largely IO and orchestration; only `build_hm` and
  the per-kernel chunk loops are pg_gpu-specific. Splitting into a
  `pg_gpu_io` extension is also defensible. Decide once the prototype
  is lifted in phase 1 and the line is visible.
