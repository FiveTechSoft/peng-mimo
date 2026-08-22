# FreeToken PoC README

This PoC branch implements a minimal in-memory KV cache and a small
Python tool to build a semantic ANN index for prompt embeddings.

Files added in the PoC branch:
- c/kv_cache.h, c/kv_cache.c  -- minimal in-memory KV cache API (single-process)
- tools/kv_index_build.py     -- build/query HNSW index for prompt embeddings

How to use (PoC)
1. Build the project as usual. The kv_cache files are standalone and do not
   change existing build targets. You can link them into the runtime or use
   the demo code in c/ to integrate later.

2. To build an ANN index for prompts, precompute embeddings (numpy .npy)
   using your preferred encoder and run:

   python3 tools/kv_index_build.py build --embeddings prompt_embs.npy --output index.bin

3. The runtime should, in an integration step, compute an embedding for the
   current prompt, query the index, and on a conservative hit load the
   corresponding cached K/V (using the API in c/kv_cache.h).

Next steps (planned)
- Integrate kv_cache lookups into the prefill path (mimo.c) with a safe
  fallback to full prefill when cache miss or mismatch occurs.
- Add serialization for KV metadata (model-hash, prompt hash) and an on-disk
  cache for warm reboots.
- Add robust LRU and memory caps, plus tests and benchmarks.

Notes
- This is a minimal PoC. Do not treat it as production-ready. It is
  intentionally conservative and unoptimized to be safe for early validation.
