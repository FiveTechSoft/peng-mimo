# Design: `mimo.c` — colibrì engine for MiMo-V2.5 (text-only)

Date: 2026-07-10
Status: approved by user

## Goal

Adapt the colibrì streaming-MoE technology to run Xiaomi MiMo-V2.5 (311B total / 15B active
parameters) on consumer hardware, following the same philosophy as `glm.c`: pure C, zero
runtime dependencies, experts streamed from NVMe, validated token-exact against the
`transformers` reference implementation.

Target machine (dev box): 8-core Xeon W-2140B (AVX2/AVX-512), 32 GB RAM, NVMe measured at
2.75 GB/s random 19 MB reads (O_DIRECT, 8 threads), 222 GB free disk, WSL2.

Expected performance: ~4.7 GB of expert reads per cold token → ~0.6 tok/s ceiling on this
disk, higher with warm LRU/learning cache. Model on disk at int4: ~165 GB (fits).

## Scope

- Text-only chat and generation. Vision and audio encoders are out of scope.
- Normal context (8–32k tokens). The 1M-token native context is out of scope.
- MTP speculative decoding out of scope for v1 (checkpoint ships a separate
  `model_mtp.safetensors`, 1.19 GB — the door stays open for v2).
- Success criterion: teacher-forcing 32/32 and greedy 20/20 token-exact vs a tiny-random
  oracle built with the official `modeling_mimo_v2.py` (`trust_remote_code`), including
  sequences longer than 128 tokens so the sliding-window boundary is exercised.

## MiMo-V2.5 architecture facts (from official `config.json`)

| field | value |
|---|---|
| architectures | `MiMoV2ForCausalLM` (`model_type: mimo_v2`, trust_remote_code) |
| hidden_size | 4096 |
| num_hidden_layers | 48 (1 dense + 47 MoE) |
| attention | GQA: 64 Q heads, 4 KV heads, head_dim 192, v_head_dim 128 |
| hybrid pattern | full attention at layer indices 0, 5, 11, 17, 23, 29, 35, 41, 47; sliding-window 128 elsewhere |
| RoPE | partial_rotary_factor 0.334 (≈64 dims); theta 10,000,000 (full layers) / 10,000 (SWA layers) |
| MoE | 256 routed experts, top-8, moe_intermediate_size 2048, no shared expert |
| router | sigmoid scoring, `noaux_tc`, norm_topk_prob=true — identical to GLM-5.2/DeepSeek-V3 style |
| dense MLP | intermediate_size 16384 (first layer) |
| vocab | 152,576; byte-level BPE (vocab.json + merges.txt, GPT-2 style) |
| checkpoint | FP8 e4m3, block scales 128×128 — same container as GLM-5.2-FP8 |
| shards | 16 files `model_pp0_ep{0..7}-*.safetensors`, 316 GB total, largest 34.4 GB |

## Approach (chosen: sibling engine)

New `c/mimo.c` cloned from `glm.c`, then transformed. Precedent: `c/olmoe.c`. The repo
philosophy is one flat readable file per model; shared primitives live in headers.

Reused untouched (proven identical by config):
- expert streaming: pread + per-layer LRU + learning cache (`.coli_usage`) + RAM auto-cap
- int8/int4/int2 quantization kernels (AVX2), integer-dot matmuls
- sigmoid router with noaux_tc, correction bias, routed scaling, top-8 selection
- sampling (temperature + nucleus), safetensors reader (`st.h`), JSON (`json.h`)
- tokenizer machinery (`tok.h`) — byte-BPE, pending validation against HF

Removed relative to `glm.c`: MLA (q/kv LoRA, absorption), DSA indexer, shared expert,
MTP head, KV persistence (v2).

New code:
1. **GQA attention** with per-head-group KV sharing (64 Q / 4 KV), asymmetric head dims
   (K 192, V 128).
2. **Partial RoPE**: rotate first ~64 dims of each 192-dim head (factor 0.334), dual theta
   selected per layer type.
3. **Hybrid KV-cache**: circular 128-slot window buffer for the 39 SWA layers; linear
   append for the 9 full-attention layers. Causal mask respects the window.
4. **Config loader** for `config.json` of `mimo_v2` (layer pattern derived from
   `full_attention_interval` / explicit index list).

## Converter

Extend `c/tools/convert_fp8_to_int4.py` with `--arch mimo`:
- same FP8 e4m3 dequant with 128×128 block scales
- new tensor-name map (MiMoV2 naming; experts sharded expert-parallel across
  `model_pp0_ep{0..7}` — the shard-to-expert mapping is read from the safetensors index)
- dense tensors → int4 resident container; experts → per-expert int4 streaming container;
  norms/router/bias stay f32 (same policy as GLM)
- resumable; deletes each source shard after conversion. Peak disk ≈ final 165 GB +
  largest shard 34.4 GB ≈ 200 GB of the 222 GB free — acceptable but leaves no slack for
  anything else during conversion.

## Validation plan

1. `tools/make_mimo_oracle.py` — tiny-random MiMoV2 via official remote code
   (hidden 128, 5 layers with the real hybrid pattern scaled down: full at 0 and 4,
   SWA-8 elsewhere so the window boundary is crossed by a 32-token sequence; 8 experts
   top-2; partial RoPE + dual theta preserved). Emits `mimo_tiny/` + `ref_mimo.json`
   (prompt, greedy continuation, teacher-forcing argmax) exactly like `make_glm_oracle.py`.
2. `mimo.c` runs the oracle: TF 32/32 positions, greedy 20/20 tokens, f32 and then with
   quantized cache slots (int8 container must stay bit-identical to f32 packing tests).
3. Tokenizer: encode/decode a unicode-heavy corpus, diff vs `AutoTokenizer` output.
4. Perf fixture: `tools/make_mimo_bench_model.py` (~300M params, real shapes) for
   streaming/cache behavior before any big download.
5. Only after 1–4 pass: user decides on the 316 GB download + conversion.

## Testing

- Existing `make test-c` (json/st/tier) must keep passing.
- New `make mimo` target mirrors `glm`.
- Oracle runs are the acceptance gate for every phase; a failed position count is a stop.
- SWA boundary test: TF over a sequence at least 4× the scaled-down window length.

## Phases

1. Oracle + tokenizer validation (Python side, no C yet)
2. `mimo.c` dense forward on tiny oracle (f32, no streaming) → TF exact
3. Streaming + quantized experts + LRU wired in → TF/greedy exact at int8/int4 cache
4. Full validation suite + perf fixture
5. Converter `--arch mimo` (validated on tiny + fixture)
6. User decision: real 316 GB download → conversion → first real chat

## Risks

- `modeling_mimo_v2.py` details not visible in config (QK-norm, attention bias, MoE
  routing details) — mitigated by reading the modeling file in phase 1 and by the oracle
  being the ground truth.
- Tokenizer may deviate from GPT-2 byte-BPE in pretokenization regex — mitigated by the
  diff suite; worst case is a new regex table in `tok_unicode.h`.
- Disk headroom during conversion is thin (~20 GB slack); conversion order deletes shards
  as it goes and is resumable.
- Expert-parallel shard layout may split one expert's tensors across shards; the index
  json resolves placement, but the converter must handle out-of-order availability.
