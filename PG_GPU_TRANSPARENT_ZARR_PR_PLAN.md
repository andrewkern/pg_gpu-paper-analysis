# Transparent biobank-scale VCZ in `pg_gpu`: PR plan

A concrete, implementation-grade plan for a series of PRs against the
`pg_gpu` repository (`/home/adkern/pg_gpu`, branch
`feature/transparent-biobank-zarr` once opened) that make biobank-scale
VCZ stores "just work" behind `HaplotypeMatrix.from_zarr` and
`GenotypeMatrix.from_zarr`. The user types one line; the library picks
eager, streaming, host, or kvikio under the hood and never asks them
to know which.

Numbered design choices below are anchored in the prototype measured
in `STREAMING_FROM_ZARR_PLAN.md` (sibling file in this paper-analysis
repo); read it first if you need the empirical backing.

## Goal

After this work is merged:

```python
from pg_gpu import HaplotypeMatrix, windowed_analysis

# 5 GB VCZ at 10k samples
hm = HaplotypeMatrix.from_zarr("ag3.0_3R.vcz")          # eager, as today

# 6 GB VCZ at 100k samples (chr15 sim)
hm = HaplotypeMatrix.from_zarr("chr15.vcz",
                               pop_file="chr15.pops.tsv")
# auto-detected: streaming on, kvikio if available, bio2zarr-chunked
# store -> sample-subset reads at 152x speedup, end-to-end scan ~2h30
# instead of OOM or many hours

df = windowed_analysis(hm, window_size=100_000, statistics=["pi", "tajimas_d"])
# same call works for both. The streaming dispatch lives inside the
# kernel.
```

User-facing API stays identical to today. The internal change is that
`from_zarr` may return a `StreamingHaplotypeMatrix` instead of a
`HaplotypeMatrix`, and every statistic-level entry point dispatches on
the type.

## Non-goals

* No new public class the user has to construct. `from_zarr` /
  `from_vcf` return the right thing.
* No required CLI step. The user does not have to know about kvikio,
  nvCOMP, `enable_gpu()`, `compat_mode`, prefetch depth, or chunk
  alignment. All have sensible defaults.
* No regression on the small-store path. Existing tests pass byte-for-byte.
* No GPU Direct Storage requirement. The kvikio backend defaults to
  `compat_mode=ON`; GDS is an opt-in for users who know their storage
  path supports it.
* No new VCF parser. Big VCFs trigger a one-time warning steering the
  user at `HaplotypeMatrix.vcf_to_zarr(...)`, which already exists.

## Existing surface to preserve

The following public API stays identical (signatures, return values,
side effects, error messages where reasonable):

* `HaplotypeMatrix.__init__(genotypes, positions, ...)`
  -- `pg_gpu/haplotype_matrix.py:34`
* `HaplotypeMatrix.from_zarr(path, region=None, accessible_bed=None)`
  -- `pg_gpu/haplotype_matrix.py:361`
* `HaplotypeMatrix.from_vcf(path, region=None, ...)`
  -- `pg_gpu/haplotype_matrix.py:293`
* `HaplotypeMatrix.vcf_to_zarr(vcf_paths, zarr_path, ...)`
  -- `pg_gpu/haplotype_matrix.py:437`
* `HaplotypeMatrix.haplotypes`, `.positions`, `.samples`, etc.
* `GenotypeMatrix.from_zarr`, `.from_vcf`, `.vcf_to_zarr`
  -- `pg_gpu/genotype_matrix.py:343`, `:393`
* Every public top-level statistic: `windowed_analysis`, `sfs.sfs`,
  `sfs.joint_sfs`, `compute_ld_statistics_gpu_single_pop`, `pairwise_r2`,
  Garud `H1/H12/H123/H2H1`. See `pg_gpu/__init__.py`'s module-level
  exports (probed: `'GenotypeMatrix', 'HaplotypeMatrix', 'sfs',
  'windowed_analysis', 'ld_statistics', ...`).

## High-level architecture

```
HaplotypeMatrix.from_zarr(path, ...)
    |
    +---> ZarrGenotypeSource(path, region, pop_file)        [new in pg_gpu/zarr_source.py]
    |       .num_variants, .num_haplotypes, .site_pos
    |       .iter_chunks(chunk_bp, align_bp) -> [(left, right), ...]
    |       .slice_region(left, right)                       (full-hap)
    |       .slice_subsample(left, right, hap_cols)          (oindex)
    |
    +---> decide_layout(source, requested_streaming, requested_backend)
    |       returns one of:
    |         ("eager",     None)
    |         ("streaming", "host")
    |         ("streaming", "kvikio")
    |
    +---> eager:
    |       read full region via existing zarr_io path
    |       call HaplotypeMatrix.from_raw_genotypes(gt, pos, ...)   [new]
    |
    +---> streaming:
    |       fetcher = HostChunkFetcher(source) | KvikioChunkFetcher(source)
    |       return StreamingHaplotypeMatrix(source, fetcher,
    |                                       chunk_bp, prefetch, ...)


Kernel dispatch:
    windowed_analysis(hm, ...)
        if isinstance(hm, StreamingHaplotypeMatrix):
            return _stream_windowed(hm, ...)
        return _eager_windowed(hm, ...)        # current implementation
```

`HaplotypeMatrix` is unchanged structurally. `StreamingHaplotypeMatrix`
is a sibling class with the same *public* surface for kernels to read
(num_variants, num_haplotypes, sample_sets, accessible_mask, etc.) but
no `.haplotypes` array; instead an internal `_iter_gpu_chunks()`
iterator that kernels consume.

## File-level plan

### New files

`pg_gpu/zarr_source.py` (~250 LoC)
  - `ZarrGenotypeSource` class. Inherits nothing from existing
    `zarr_io.read_genotypes_*`; those keep working for the eager path
    and small stores. The new class is for the streaming path.
  - Auto-detects VCZ vs scikit-allel layouts via the existing
    `detect_zarr_layout()`. Only VCZ + grouped scikit-allel are
    supported for streaming; flat scikit-allel raises a clear "this
    layout cannot be streamed; convert to VCZ" message.
  - Handles `region="chrom:start-end"` for single-chrom subsetting.
  - Companion `pop_file` resolution: explicit kwarg, falls back to
    `<store>.pops.tsv` next to the store with a single-line warning
    when used.

`pg_gpu/streaming_matrix.py` (~400 LoC)
  - `StreamingHaplotypeMatrix` class (and `StreamingGenotypeMatrix`).
  - `ChunkFetcher` ABC, `HostChunkFetcher`, `KvikioChunkFetcher`.
  - Producer thread + bounded queue + cleanup on exit.
  - `.materialize(region=..., sample_subset=...) -> HaplotypeMatrix`
    for the pairwise-kernel escape hatch.

`pg_gpu/_gpu_genotype_prep.py` (~100 LoC)
  - `build_haplotype_matrix(gt, pos, ...) -> HaplotypeMatrix` -- the
    GPU-side biallelic filter + ploidy interleave + transpose path
    from the prototype's `build_hm`. Used by both eager and streaming
    paths so they share the post-decode pipeline.

`tests/test_zarr_source.py` (~120 LoC), `tests/test_streaming.py`
(~250 LoC), `tests/test_kvikio_backend.py` (~120 LoC),
`tests/test_biobank_warning.py` (~80 LoC).

`docs/source/biobank_scale.rst` (~150 LoC) -- new docs page covering
the three-way decision tree, the one-time `vcf_to_zarr` recipe, the
kvikio setup, and the wall-time calibration table.

`examples/biobank_streaming_demo.py` (~80 LoC) -- two-paragraph
example showing `from_zarr` on a large store; mirrors the
`06_simulated_genome_scan/scripts/genome_scan_ooa.py` prototype but
in 80 lines instead of 1000.

### Modified files

`pg_gpu/haplotype_matrix.py`
  - `from_zarr`: add `streaming="auto"`, `backend="auto"`,
    `pop_file=None`, `chunk_bp=None`, `prefetch=1` kwargs. Behavior
    on the eager path is unchanged. Small-store auto-detection picks
    eager and runs the current code path (no churn).
  - `from_vcf`: insert `_maybe_biobank_warn(path, region)` at the top.
  - No changes to `__init__`, `haplotypes`, `positions`, `samples`,
    or anything kernel-facing.

`pg_gpu/genotype_matrix.py`
  - Symmetric edits to `from_zarr` and `from_vcf`.

`pg_gpu/zarr_io.py`
  - No behavior change. Add an optional `as_int8=False` kwarg on
    `read_genotypes_vcz` and `read_genotypes` so the eager path can
    avoid an int8 widening when it's about to call `build_haplotype_matrix`.

`pg_gpu/windowed_analysis.py` (or wherever it lives)
  - Top of function: dispatch on `isinstance(hm, StreamingHaplotypeMatrix)`.

`pg_gpu/sfs.py`, `pg_gpu/ld_statistics.py`, `pg_gpu/distance_stats.py`,
`pg_gpu/selection.py` (the Garud module if it has its own file)
  - Same dispatch pattern, one entry point per public function.

`pg_gpu/__init__.py`
  - Export `StreamingHaplotypeMatrix`, `StreamingGenotypeMatrix`,
    `BiobankScaleWarning`, `BadlyChunkedWarning`.

`pyproject.toml` / `pixi.toml`
  - Add `kvikio` and `nvidia-nvcomp` as **optional** dependencies in
    a `[project.optional-dependencies] kvikio` extra (pip) and a
    `[feature.kvikio]` table (pixi). The library imports them
    conditionally; absence falls back to host path silently.

## Detection logic

The `decide_layout(source, requested_streaming, requested_backend)`
function in `pg_gpu/zarr_source.py` picks the path. Inputs are
deterministic and cheap to compute; no benchmarking.

```python
def decide_layout(source, streaming="auto", backend="auto"):
    """Return (mode, backend_name) where mode in {'eager','streaming'}."""

    # 1. Streaming yes/no?
    eager_bytes = source.num_variants * source.num_haplotypes  # int8
    free_gpu    = cp.cuda.Device().mem_info[0]
    if streaming == "always":
        mode = "streaming"
    elif streaming == "never":
        if eager_bytes > 0.5 * free_gpu:
            raise MemoryError(
                f"streaming='never' but eager footprint "
                f"{eager_bytes/1e9:.1f} GB > 0.5 * free GPU "
                f"({free_gpu/1e9:.1f} GB)."
            )
        mode = "eager"
    else:  # "auto"
        mode = "streaming" if eager_bytes > 0.5 * free_gpu else "eager"

    if mode == "eager":
        return ("eager", None)

    # 2. Which backend?
    if backend == "host":
        return ("streaming", "host")
    if backend in ("kvikio", "kvikio-gds"):
        _require_kvikio_or_raise()
        _require_supported_codec_or_raise(source)
        return ("streaming", backend)

    # backend == "auto"
    if not _have_kvikio():
        return ("streaming", "host")
    if not _codec_is_gpu_decodable(source):
        return ("streaming", "host")
    if not _chunking_is_subset_friendly(source):
        # whole-sample-axis chunks: kvikio gives no speedup on full reads
        # (measured 26.8s vs 26.9s host on chr15.vcz.oldchunk). Emit one
        # BadlyChunkedWarning and stay on host.
        warnings.warn(
            f"{source.path}: chunks[1] = {source.chunks[1]} spans the "
            f"full sample axis ({source.n_dip} diploids). kvikio's GPU "
            f"codec path gives no speedup at this chunking; staying on "
            f"the host backend. To unlock 6-150x reads, re-encode with "
            f"bio2zarr-style chunking (sample_chunk=1000) via "
            f"HaplotypeMatrix.vcf_to_zarr(...).",
            BadlyChunkedWarning, stacklevel=4,
        )
        return ("streaming", "host")
    return ("streaming", "kvikio")
```

Helpers:

* `_have_kvikio()`: `try: import kvikio, kvikio.zarr, nvidia.nvcomp;
  return True except ImportError: return False`. Cached.
* `_codec_is_gpu_decodable(source)`: read the
  `call_genotype/zarr.json` codec spec and check it's in
  `{"zstd", "blosc", "lz4", "deflate"}`.
* `_chunking_is_subset_friendly(source)`: `chunks[1] < n_dip / 4`.
  Heuristic, conservative.

## The user-facing API after the PR

The new kwargs on `from_zarr` (and `from_vcf` where applicable):

```python
HaplotypeMatrix.from_zarr(
    path,
    region=None,
    accessible_bed=None,
    # all new, all optional with sensible auto-defaults
    streaming="auto",      # auto | always | never
    backend="auto",        # auto | host | kvikio | kvikio-gds
    chunk_bp=None,         # default: pick from source via estimate_chunk_bp()
    prefetch=1,
    pop_file=None,         # path; auto-resolves to <store>.pops.tsv
)
```

99% of users use the default kwarg values and get the right behavior.
`streaming="always"` / `backend="host"` exist for explicit testing /
benchmarking and reproducibility, not for production use.

## PR sequencing

Breaking this into reviewable PRs is essential -- one mega-PR would
be unreviewable and would block all the other work for weeks. Each
PR below is independently shippable, ships its own tests, and leaves
the library in a consistent state.

### PR 1 -- `ZarrGenotypeSource` + `build_haplotype_matrix` (foundation)

Adds `pg_gpu/zarr_source.py` and `pg_gpu/_gpu_genotype_prep.py`. No
behavior change in any existing entry point. The new helpers are not
called from anywhere yet -- this PR is pure plumbing. Reviewable in
under 500 LoC of new code + tests.

Tests:
- `test_zarr_source.py`: open a synthetic in-memory zarr, verify
  shapes / dtypes / position arrays match expectations. Empty region,
  multiallelic-only chunk, pop_file resolution.
- `test_gpu_genotype_prep.py`: `build_haplotype_matrix` on a synthetic
  `(n, n_dip, 2)` gt; assert byte-equal to the reference numpy path.
  Empty chunk, all-multiallelic chunk.

Acceptance: existing test suite passes unchanged.

### PR 2 -- eager `from_zarr` rewritten to use the new helpers

`HaplotypeMatrix.from_zarr` and `GenotypeMatrix.from_zarr` now do
their host-side allocate + reshape via `build_haplotype_matrix` on
the GPU. **No streaming yet.** This PR is the "GPU prep is faster"
win for the eager path. Adds the `streaming`, `backend`, `chunk_bp`,
`prefetch`, `pop_file` kwargs to the signatures with auto-routing
that always picks eager.

Tests:
- Existing `from_zarr` test cases pass byte-for-byte.
- Eager path on a 1 Mb / 200k-hap synthetic store: confirm it now
  uses the GPU prep path (peak host RAM stays below ~2 GB; previously
  it would be ~28 GB).

Acceptance: existing test suite passes; new microbench shows the
eager path's host RAM cost drops.

### PR 3 -- `StreamingHaplotypeMatrix` and host backend

Adds `pg_gpu/streaming_matrix.py` with `StreamingHaplotypeMatrix`,
`ChunkFetcher` ABC, `HostChunkFetcher`. `from_zarr(streaming="always")`
now returns a `StreamingHaplotypeMatrix`. **Kernels do not yet
dispatch on type**; they would raise on a `StreamingHaplotypeMatrix`.
That's intentional: this PR lands the data structure and the iterator,
the next PR wires kernels.

Tests:
- `test_streaming.py`: build a small synthetic store, stream through
  `iter_gpu_chunks()`, reassemble, verify byte-equal to the eager
  read.
- prefetch=0 vs prefetch=1 give same data.
- Producer-side exception is raised on the consumer side with
  traceback preserved.
- `.materialize(region=(lo, hi))` returns a normal
  `HaplotypeMatrix` over the slice.

Acceptance: kernels still work eagerly; streaming class exists but
isn't wired in.

### PR 4 -- streaming-aware kernels

`windowed_analysis`, `sfs.sfs`, `sfs.joint_sfs`,
`compute_ld_statistics_gpu_single_pop` (sum path), `selection.garud_h*`
gain top-of-function dispatch:

```python
def windowed_analysis(hm, ...):
    if isinstance(hm, StreamingHaplotypeMatrix):
        return _stream_windowed(hm, ...)
    return _eager_windowed(hm, ...)
```

`_stream_windowed` iterates `hm._iter_gpu_chunks()`, calls
`_eager_windowed` on each chunk, concatenates DataFrames. Same shape
for `sfs.sfs` (sum the chunk SFSs).

Pairwise kernels (`pairwise_r2`, the r^2 heatmap path) raise a clear
`StreamingNotSupported` on `StreamingHaplotypeMatrix` and document
the `.materialize(region=...)` escape hatch.

Tests:
- `test_streaming_kernels.py`: for each public kernel, assert
  `kernel(streaming_hm, ...) == kernel(eager_hm, ...)` for a small
  synthetic store. Use `pytest.approx` for floats.
- Window size that does not divide chunk alignment raises.
- `pairwise_r2(streaming_hm)` raises with the materialize hint.

Acceptance: a real run of `genome_scan_ooa.py`-equivalent statistics
on the chr15 store completes via the library API only (no private
helpers from the prototype). Identical results to the prototype
within float tolerance.

### PR 5 -- `from_zarr` auto-detection (streaming flips on)

Wires `streaming="auto"` to actually do the GPU free-memory check
and return a `StreamingHaplotypeMatrix` for big stores. Updates
docstrings, examples, and `from_vcf` to call `from_zarr` on a
freshly converted VCZ when the user asked for VCF but the file is
small enough to fit eagerly.

Tests:
- Small synthetic store: `from_zarr(path)` returns `HaplotypeMatrix`.
- Large synthetic store: `from_zarr(path)` returns
  `StreamingHaplotypeMatrix`. `windowed_analysis(hm)` works.
- `streaming="never"` on a too-big store raises a `MemoryError` with
  the projected eager bytes in the message.

Acceptance: the prototype's `genome_scan_ooa.py` works *unchanged*
when the private helpers are replaced with calls into
`HaplotypeMatrix.from_zarr` / public kernels. Wall time matches the
prototype within 10%.

### PR 6 -- `KvikioChunkFetcher` + backend auto-detection

Adds `KvikioChunkFetcher`. Guarded imports of `kvikio`,
`kvikio.zarr`, `nvidia.nvcomp`; absence falls back to host silently.
Forces `kvikio.defaults.set({"compat_mode": kvikio.CompatMode.ON,
"num_threads": 8})` unless the user explicitly opts into `kvikio-gds`.
Probes store codec and chunk shape, emits `BadlyChunkedWarning` when
the chunks would defeat kvikio's win.

Tests:
- `test_kvikio_backend.py`: skipped if `kvikio` not importable.
- Build two synthetic stores: one with `(10000, 1000, 2)` chunks, one
  with `(10000, n_dip, 2)`. `backend="auto"` picks `kvikio` on the
  first and `host` (with `BadlyChunkedWarning`) on the second.
- `backend="kvikio"` on an unsupported-codec store raises with a
  message that includes the codec name and the supported list.
- Byte-equality check: `KvikioChunkFetcher` chunks match
  `HostChunkFetcher` chunks for the same region on the same store.
- Sample-subset oindex returns byte-equal output to host path.

Acceptance: on the chr15 paper-analysis VCZ, end-to-end scan with
`backend="auto"` finishes in ~2 h 30 min wall (vs ~3 h 28 min on the
host backend), with identical `chromosome_summary.json` numbers.

### PR 7 -- `BiobankScaleWarning` on `from_vcf`

Adds `pg_gpu.BiobankScaleWarning` (UserWarning subclass) and the
header-parse-on-open size check. Constants live at the
`pg_gpu/haplotype_matrix.py` module top:

```python
BIOBANK_VCF_WARN_BYTES     = 10 * 1024**3   # 10 GiB on-disk
BIOBANK_VCF_WARN_SAMPLES   = 5_000
BIOBANK_VCF_WARN_REGION_BP = 5_000_000      # 5 Mb
```

The check fires once per path per process (cache the result). The
warning text is the copy-pastable one in
`STREAMING_FROM_ZARR_PLAN.md`'s "Sizing guidance" section and points
at `HaplotypeMatrix.vcf_to_zarr(...)`.

Tests:
- `test_biobank_warning.py`: ~1 GB VCF, no warning.
- Synthetic VCF with 10k samples in `#CHROM` header, no `region=`
  argument: warning fires once, parse proceeds.
- 47 GB VCF with `region="chr15:30M-30.1M"`: no warning (region is
  small).
- 47 GB VCF with no region: warning fires.
- `warnings.filterwarnings("ignore", category=BiobankScaleWarning)`
  silences it.
- Second call on the same path in the same process: no duplicate
  warning.

### PR 8 -- docs + chr15 example

* New `docs/source/biobank_scale.rst` covering the decision tree,
  conversion recipe via `HaplotypeMatrix.vcf_to_zarr`, kvikio setup
  with one paragraph on `compat_mode`, the GPU-codec compatibility
  matrix, and the wall-time calibration table:

  | sample count | input | path | wall time |
  |--------------|-------|------|-----------|
  | 2500 (1kG)   | VCF.gz| from_vcf eager | ~30 s |
  | 10k (mid)    | VCZ   | from_zarr eager | ~5 min |
  | 100k (biobank)| VCZ + bio2zarr chunking | from_zarr streaming + kvikio | ~2 h 30 min |
  | 100k (biobank)| VCF.gz | from_vcf | days; **WARN** |

* `examples/biobank_streaming_demo.py` -- 80-line script that
  reproduces the chr15 scan via the library API only.
* Update docstrings on `from_zarr`, `from_vcf`, `vcf_to_zarr` to
  mention streaming and kvikio.

## Test plan summary

| Test file | Coverage | LoC |
|---|---|---|
| `test_zarr_source.py` | source construction, region subsetting, pop_file | ~120 |
| `test_gpu_genotype_prep.py` | build_haplotype_matrix correctness | ~100 |
| `test_streaming.py` | iterator, prefetch, error propagation, materialize | ~250 |
| `test_streaming_kernels.py` | all kernels stream=eager equivalence | ~200 |
| `test_kvikio_backend.py` | codec probe, chunk-shape probe, byte-equality | ~120 |
| `test_biobank_warning.py` | warning triggers + silencing | ~80 |

Every comparison test uses a synthetic in-memory zarr store built
in-process via `zarr.create_group(store=zarr.storage.MemoryStore())`
so tests stay fast and have no on-disk dependency.

## Performance acceptance

A new `tests/perf/test_streaming_perf.py` (manual / nightly) checks:

* Eager `from_zarr` on a 1 Mb region: peak host RSS < 2 GB.
* Streaming + host: per-chunk wall < 100 s on a 1 Mb / 200k-hap
  chunk; `cpu/wall > 3x` on the producer.
* Streaming + kvikio (when kvikio importable, store zstd-coded,
  bio2zarr-chunked): per-chunk read wall < 5 s, sample-subset oindex
  wall < 1 s. `cpu/wall > 5x`.
* End-to-end chr15 windowed scan completes within 110 min wall on
  the host backend, within 90 min wall on the kvikio backend, on
  an A100 80 GB / 64-core / NVMe host. (Numbers calibrated to the
  prototype.)

## Open questions to resolve during review

* **`StreamingHaplotypeMatrix` as subclass vs sibling?** The plan
  treats it as a sibling. The alternative is making
  `HaplotypeMatrix` itself lazy via an internal `_source` attribute,
  with `.haplotypes` materializing one chunk on demand. The sibling
  design is more honest (no chance of "I touched
  `hm.haplotypes` and OOM'd the host"); it requires kernels to
  dispatch but only at the public entry points. Confirm with
  reviewers in PR 3.

* **Should kernels ever fall through to "materialize the whole
  thing on demand"?** Probably not -- if a user wrote
  `np.asarray(streaming_hm.haplotypes)` and we silently materialized
  4.6 TB of data, that's worse than raising. The pairwise kernels
  raise; everything else streams. Document the contract.

* **What about `GenotypeMatrix` parity?** The plan calls for
  symmetric edits to `GenotypeMatrix.from_zarr` / `.from_vcf`. Most
  of the kernels are on `HaplotypeMatrix`, so we need to confirm
  which `GenotypeMatrix`-consuming entry points exist and whether
  streaming them is in scope for the initial work.

* **`zarr` 2 vs zarr 3.** `pg_gpu`'s `pixi.toml` allows `zarr >=
  2.16`. The kvikio path requires zarr 3.x (`zarr.config.enable_gpu()`
  is a zarr-3 API). PR 6 should either bump the lower bound to
  zarr 3 unconditionally, or guard the kvikio path on a zarr-3
  detection. Recommend the former: zarr 3 has been out long enough
  and our `zarr_io.py` is already written for the modern API.

* **`scikit-allel` zarr layouts and streaming.** The existing
  `read_genotypes_allel*` functions support the legacy scikit-allel
  layout. Should `ZarrGenotypeSource` support streaming for those
  too? PR 1 should at least make this an explicit yes/no decision
  rather than a silent omission. Recommend: VCZ only for the streaming
  path in this PR series; flat / grouped scikit-allel layouts raise a
  clear "convert to VCZ via HaplotypeMatrix.vcf_to_zarr" message.

* **Naming.** `StreamingHaplotypeMatrix` is descriptive but
  long-winded. `LazyHaplotypeMatrix` is shorter but suggests "loads
  on first .haplotypes access" which isn't quite what we do.
  `ChunkedHaplotypeMatrix` is technically what it is. Decide during
  PR 3 review.

* **The `pop_file` resolution.** Auto-loading `<store>.pops.tsv` is
  convenient but slightly magical. Consider whether to require it
  to be explicit on the streaming path (where pop assignments are
  fixed at construction) while keeping the auto-load on eager.
  Recommend: auto-load on both, single-line `print` to stderr
  showing what was loaded so it's not invisible.

## What's not in this plan

* The two-stream compute pipelining mentioned at the end of
  `STREAMING_FROM_ZARR_PLAN.md` (overlap chunk N+1's build_hm with
  chunk N's stats compute on a second cupy stream) is a follow-up
  optimization, not part of this PR series. The current series
  already gets to compute-bound on the windowed scan, so further
  wins require this refactor; do it after PR 8 lands and we have a
  baseline.

* The `BadlyChunkedWarning` rechunk recipe could in principle be
  automated (the library could re-encode silently on first access),
  but rechunking 100k-hap chr15 is a 2-hour job that should not
  happen without the user's knowledge. The warning is the right level
  of automation.

* A `pgen` / `bgen` backend. UK Biobank distributes BGEN natively;
  supporting it via kvikio + nvCOMP is a natural follow-up but is
  out of scope here. The streaming abstraction is designed to make
  this addition straightforward later (just add a `BgenGenotypeSource`
  alongside `ZarrGenotypeSource`).
