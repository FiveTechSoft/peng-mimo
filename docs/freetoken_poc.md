# FreeToken PoC README

This PoC branch implements a minimal in-memory KV cache and a small
Python tool to build a semantic ANN index for prompt embeddings.

Files added in the PoC branch:
- c/kv_cache.h, c/kv_cache.c  -- minimal in-memory KV cache API (single-process)
- tools/kv_index_build.py     -- build/query HNSW index for prompt embeddings

How to use (PoC)
1. Build the `glm` engine as usual (`make -C c glm`): it now compiles and
   links `kv_cache.c`, and `glm.c` carries the PoC save/restore helpers
   (`try_save_kv_to_cache` / `try_restore_kv_from_cache`). `mimo` is unchanged.

2. The prefill path is wired to the cache in both entry points:
   - `run_text` (`FREETOKEN=1`) — on an exact token-prefix HIT, restore Lc/Rc
     and forward only the last prompt token; on MISS, full prefill then save.
     Fallback is always a full recompute. Exact token match only (no
     semantics): lossless by construction. Disabled automatically when DSA is
     active (the indexer cache `Ic` is not saved). The MTP row is excluded
     (its KV is born in decode). `FREETOKEN_CACHE_GB` sets the RAM cap
     (default 8).
   - `run_serve` (`FREETOKEN=1`) — covers the fresh-conversation case
     (`len==0`: first turn, after `\x02RESET`, or context-overflow reset).
     Turns that share a prefix with the live history are already served by
     the persistent KV and do not need the cache. Measured with the tiny
     fixture (1618-token turn, RESET, identical turn): turn throughput
     0.31 -> 30.69 tok/s (~100x on the repeated turn).
   - `FREETOKEN_BENCH=1` — in `run_text`, after a MISS, re-run the prefill
     from the cache on fresh KV buffers and print hit/miss times, speedup,
     max |logit diff| and argmax agreement.

3. To build an ANN index for prompts, precompute embeddings (numpy .npy)
   using your preferred encoder and run:

   python3 tools/kv_index_build.py build --embeddings prompt_embs.npy --output index.bin

4. The runtime should, in a later step, compute an embedding for the current
   prompt, query the index, and on a conservative hit load the corresponding
   cached K/V (using the API in c/kv_cache.h). This semantic layer is NOT
   wired yet — the runtime hook uses exact token hashes only.

Measured (2026-08-22, WSL CPU, tiny random-weight GLM-MoE fixture:
8 layers, hidden 1024, 32 experts, vocab 152576, prompt of 2001 tokens,
`DSA=0 FREETOKEN_BENCH=1`):
- prefill MISS (full): 12.5–14.1 s
- prefill HIT (cache restore + 1-token forward): 0.02–0.16 s
- speedup: ~90–600x on the prefill step; greedy argmax identical
  (max |logit diff| ~0.45 over 152576 entries comes from the S=2001 vs S=1
  kernel batch shapes, not from the cache: restore is a byte-exact memcpy of
  what the full prefill itself wrote).
- Note: the tiny fixture has no disk-streaming cost per expert; on the real
  379 GB model the absolute times differ but the mechanism (skip np-1 tokens
  of prefill) is the same. The paper's semantic ANN + CUDA streaming numbers
  are NOT reproduced here — this PoC measures exact-prefix reuse only.

Next steps (planned)
- Semantic lookup (ANN over embeddings) on top of the exact-hash cache.
- On-disk serialization for warm reboots (model-hash + prompt hash metadata).
- Robust LRU and memory caps, plus tests and benchmarks.
- DSA support (save/restore the indexer cache `Ic` too).

Notes
- This is a minimal PoC. Do not treat it as production-ready. It is
  intentionally conservative and unoptimized to be safe for early validation.
