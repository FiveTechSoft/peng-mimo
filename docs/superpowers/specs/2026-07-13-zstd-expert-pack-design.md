# zstd expert pack — lossless compressed container (design)

**Date:** 2026-07-13
**Status:** approved (design discussion in session; measurements in `findings.md` §41)
**Goal:** cut cold-token disk bytes ~25% losslessly by storing every tensor in the
int4 container as a zstd-1 frame, decompressed on load. Bit-exact: the decompressed
bytes are identical to today's container, so the computation graph and all results
are unchanged.

## Motivation (measured, §41)

- Real int4 expert data compresses to **74.7%** with zstd-1 (order-0 entropy floor
  5.89 bits/byte; zstd-3 gains nothing — symbol skew, not repeats).
- Decompress: 0.95 GB/s single thread, 3.4+ GB/s on 8 threads — above the
  2.75 GB/s NVMe line rate, and it runs in loader threads that are IO-blocked today.
- Cold-token math: 4.7 GB → ~3.5 GB reads; estimated **0.60 → ~0.72–0.79 tok/s**,
  plus disk 165 → ~123 GB.
- Origin: DFloat11-style BF16 exponent compression (arXiv 2504.11651) is **not**
  applicable (we are int4 already; 11.2 bits/weight would be ~2.8× larger); only
  the lossless-compression question transfers, and it pays.

## Format: zstd frames inside safetensors

Keep the existing container layout (sharded `out-NNNNN.safetensors`, JSON header,
tensor names, shapes, dtypes). Change only the data regions:

- Every tensor's data region is **one zstd-1 frame** (standard frame, written with
  `ZSTD_compress` level 1).
- `data_offsets` in the header point to the **compressed** frame (physical file
  range, as today).
- Each tensor entry gains one extra JSON key `"nb": <uncompressed_nbytes>`.
  Our parser (`json.h`) ignores unknown keys today; `st_init` will read it.
- File-level `__metadata__` gains `"peng_zstd": "1"` — the presence flag that
  switches the engine to the decompress path. Absent flag = legacy container,
  loader unchanged (backward compatible).
- Scope: **whole container** — routed experts, their `.qs` scales, and the dense
  tensors (dense decompress happens once at startup; ~8 GB extra disk saved).
- The repack tool writes the 3 weight tensors of each expert **adjacent** in the
  file, in the same relative order as today, so `expert_load`'s coalesced
  single-pread path still applies (contiguity check works on compressed offsets).

Trade-off accepted: the container remains peng-specific (it already is — packed U8
int4 with `.qs` twins). External safetensors tools can still parse the header; they
just can't interpret the frame bytes.

## Components

### 1. Repack tool — `c/tools/repack_zstd.py`

- Input: existing container dir (`SNAP`). Output: new dir with same file names,
  `config.json`, `generation_config.json`, `_meta/` copied through.
- Per shard: parse header, read each tensor, `zstd.compress(data, 1)`, write new
  header (same names/dtypes/shapes + `"nb"` key + metadata flag) and frames.
  Preserve tensor order within each shard (keeps expert w1/w2/w3 adjacency).
- Atomic: write `*.tmp`, fsync, rename — same convention as `convert_fp8_to_int4.py`.
- `--verify`: after writing, re-read every frame, decompress, byte-compare against
  the source container. Exit nonzero on any mismatch.
- Runtime estimate: 152 GB read + ~113 GB write at NVMe speed plus zstd-1
  compress (~500 MB/s/thread, parallelizable) — order 1–2 h.

### 2. Engine: `c/st.h`

- `st_tensor` gains `int64_t znbytes` (compressed size; 0 ⇒ uncompressed entry).
  `nbytes` keeps meaning **uncompressed payload size** everywhere, so all existing
  size math (slab sizing, `fmt` detection, `out_cap` checks, `numel*esz`
  validation) is untouched.
- `st_init`: if `__metadata__.peng_zstd == "1"`, read `"nb"` per tensor;
  `znbytes = data_offsets[1]-data_offsets[0]`, `nbytes = nb`. Validation: for
  standard dtypes require `nb == numel*esz` (same rule as today, applied to `nb`).
- New helper `st_read_frame(S, t, dst, dst_cap)`: pread `znbytes` from `t->off`
  into a per-thread grow-only scratch buffer, `ZSTD_decompress` into `dst`.
  Returns bytes written; hard-exit on frame error (corrupt container).
- `st_read_raw`, `st_read_f32`, `st_read_slice_f32`: route through the frame
  helper when `t->znbytes > 0`. (`st_read_slice_f32` decompresses the whole
  tensor into scratch, then copies the slice — only used for GLM fused experts,
  not on the MiMo hot path.)
- Prefetch (`st_prefetch`, `st_prefetch_phys`) and `DONTNEED` use
  `off/znbytes` — the physical range. Less data to warm: strictly better.

### 3. Engine: `c/mimo.c` `expert_load`

Compressed path mirrors today's structure:

- Coalesced case: 1 pread (O_DIRECT twin with the existing 4K round-down) of the
  3 adjacent **compressed** frames into per-thread compressed scratch, then 3×
  `ZSTD_decompress` into `slab` at `pos[k]` computed from **uncompressed** sizes.
- Non-contiguous fallback: 3 preads into scratch + 3 decompresses.
- Scales `.qs`: pread frame into scratch, decompress into `fslab` (stored F32,
  as today).
- Per-thread compressed scratch: grow-only, same pattern as the existing
  thread-local quant scratch (§26); sized `max(znbytes)+8K`, ~15 MB/loader thread.
- Decompress executes inside the existing async loader pthread pool — this is
  where read/decompress overlap comes from. Loader pool size may need a bump to
  keep aggregate decompress ≥ line rate (measure; 8 is the target).
- `qt_from_disk` fallback path (unquantized oracle models) goes through
  `st_read_f32`, which handles frames — no extra work.

### 4. Build

- `Makefile`: link `-lzstd`; require `libzstd-dev` (document in README). No
  ifdef fallback — the flagless legacy path never calls zstd, but the binary
  links it unconditionally for simplicity.

## Data flow (cold expert, compressed)

```
router → miss → loader thread:
  pread(dfd, scratch, znbytes_3frames, off0&~4095)     ~14 MB, O_DIRECT
  ZSTD_decompress(slab+pos[k], nbytes[k], scratch+…)   3 frames → ~19 MB
  pread + decompress .qs → fslab
  QT views into slab (zero-copy, unchanged)
compute thread: consumes slot exactly as today
```

## Error handling

- Frame decompress error or output-size mismatch → `fprintf` tensor name +
  offsets, `exit(1)` (same policy as short pread today: container corruption is
  fatal).
- Mixed containers (flag present but a tensor missing `"nb"`) → fatal at
  `st_init` with the tensor name.
- Legacy container (no flag): `znbytes = 0` everywhere, code paths identical to
  today — regression risk isolated to one branch per read site.

## Testing / gates

1. **Unit:** repack `mimo_tiny_i4` (and `mimo_tiny_i8`) → `--verify` passes.
2. **Bit-exact gate:** tiny oracle TF + greedy (`validate.sh` machinery) on the
   repacked tiny container must equal current results exactly (same logits).
3. **Legacy regression:** same oracle on the *un*-repacked container with the new
   binary — unchanged.
4. **Full container:** repack `/root/mimo25_i4` → `/root/mimo25_i4z` (790 GB
   free, fits), `--verify`, then §37 speed protocol (TAO=1 stack, Rome NGEN=24,
   warm run of 2). Success = no quality change (bit-exact) and tok/s ≥ 0.60;
   target 0.72+.
5. SCORE ppl matrix spot-check (BASE top-8) equals §40 values — belt and braces.

## Non-goals

- Disk-order expert re-layout (separate roadmap item; composes later).
- zstd level tuning / dictionaries (level 1 already at the order-0 entropy floor).
- GPU decompress (nvcomp) — CPU keeps up with NVMe on this box.
- Compressing the published HF container (decide after speed numbers are in).

## Risks

- **O_DIRECT alignment:** frame starts are unaligned; round-down + slack handled
  in scratch (read side), decompress consumes at `off0-base` offset. Reuses the
  existing alignment idiom.
- **Loader pool sizing:** single-frame decompress ~20 ms/expert; need ≥8 in
  flight to hold line rate. Measure, bump pool size if needed.
- **Page-cache interplay under WSL2:** compressed reads shrink cache pressure
  (fewer bytes) — expected to help, but §18-style WSL I/O tax may eat some win.
  The speed gate (test 4) is the arbiter.
