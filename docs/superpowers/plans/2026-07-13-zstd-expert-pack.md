# zstd Expert Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store every tensor in the int4 container as a zstd-1 frame and decompress on load, cutting cold-token disk bytes ~25% losslessly (spec: `docs/superpowers/specs/2026-07-13-zstd-expert-pack-design.md`, measurements: `findings.md` §41).

**Architecture:** Container keeps safetensors structure; each tensor's data region becomes one zstd frame, `data_offsets` point to compressed bytes, per-tensor JSON key `"nb"` holds the uncompressed size, `__metadata__.peng_zstd="1"` flags the format. `st.h` learns to decompress on read (`znbytes` field + thread-local scratch); `expert_load` in `mimo.c` decompresses frames into the existing slab so QT zero-copy views are unchanged. Decompression runs in the existing OVERLAP loader pool. Legacy containers (`znbytes==0`) take the exact current code path.

**Tech Stack:** C (gcc, `-lzstd`), Python 3.12 + `zstandard` for the repack tool. All run/test commands execute in WSL (container lives at `/root/mimo25_i4`; repo mounted at `/mnt/c/peng-mimo`).

---

### Task 1: Dependencies

**Files:** none (environment + `c/Makefile`)

- [ ] **Step 1: Install libzstd-dev and python zstandard in WSL**

```bash
wsl -e bash -c "apt-get install -y libzstd-dev && /mnt/c/peng-mimo/venv/bin/pip install zstandard"
```

Expected: both install without error. Verify: `wsl -e bash -c "echo '#include <zstd.h>' | gcc -E - >/dev/null && echo OK"` prints `OK`.

- [ ] **Step 2: Add `-lzstd` to both LDFLAGS branches in `c/Makefile`**

```makefile
# macOS branch (line ~19):
LDFLAGS = -lm $(OMPL) -pthread -lzstd
# Linux branch (line ~28):
LDFLAGS = -lm -fopenmp -pthread -rdynamic -lzstd
```

- [ ] **Step 3: Verify build still passes**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && make mimo 2>&1 | tail -3"`
Expected: builds clean (no zstd symbols used yet; link flag proven).

- [ ] **Step 4: Commit**

```bash
git add c/Makefile
git commit -m "build: link libzstd (zstd expert pack, findings §41)"
```

---

### Task 2: Repack tool `c/tools/repack_zstd.py`

**Files:**
- Create: `c/tools/repack_zstd.py`
- Test: `c/tests/test_repack_zstd.py`

- [ ] **Step 1: Write the failing test**

`c/tests/test_repack_zstd.py`:

```python
import json, os, struct, subprocess, sys, tempfile, unittest

import zstandard

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")


def write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, raw_bytes) in desired file order."""
    header, off = {}, 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for _, _, _, raw in tensors:
            f.write(raw)


class TestRepackZstd(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.mkdtemp(prefix="peng_src_")
        self.dst = tempfile.mkdtemp(prefix="peng_dst_")
        # skewed bytes so zstd actually shrinks them (like int4 weights do)
        w = bytes([i % 16 for i in range(64 * 32)])          # U8 packed int4-ish
        s = struct.pack("<64f", *([0.5] * 64))               # F32 scales
        write_safetensors(os.path.join(self.src, "out-00000.safetensors"),
                          [("w.weight", "U8", [64, 64], w),
                           ("w.weight.qs", "F32", [64], s)])
        with open(os.path.join(self.src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

    def run_tool(self, *extra):
        return subprocess.run(
            [sys.executable, os.path.join(TOOLS, "repack_zstd.py"),
             "--indir", self.src, "--out", self.dst, *extra],
            capture_output=True, text=True)

    def test_roundtrip_and_flag(self):
        r = self.run_tool("--verify")
        self.assertEqual(r.returncode, 0, r.stderr)
        p = os.path.join(self.dst, "out-00000.safetensors")
        with open(p, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            data = f.read()
        self.assertEqual(hdr["__metadata__"]["peng_zstd"], "1")
        for name in ("w.weight", "w.weight.qs"):
            a, b = hdr[name]["data_offsets"]
            raw = zstandard.ZstdDecompressor().decompress(
                data[a:b], max_output_size=hdr[name]["nb"])
            self.assertEqual(len(raw), hdr[name]["nb"])
        # aux files copied through
        self.assertTrue(os.path.exists(os.path.join(self.dst, "config.json")))
        # frames contiguous: preserved order, no gaps
        self.assertEqual(hdr["w.weight"]["data_offsets"][1],
                         hdr["w.weight.qs"]["data_offsets"][0])

    def test_verify_catches_corruption(self):
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        p = os.path.join(self.dst, "out-00000.safetensors")
        with open(p, "r+b") as f:
            f.seek(-1, 2)
            f.write(b"\xff")
        r = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "repack_zstd.py"),
             "--indir", self.src, "--out", self.dst, "--verify-only"],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && ../venv/bin/python -m unittest tests.test_repack_zstd -v"`
Expected: FAIL — `repack_zstd.py` does not exist (subprocess returncode 2).

- [ ] **Step 3: Write the tool**

`c/tools/repack_zstd.py`:

```python
#!/usr/bin/env python3
"""Repack a peng safetensors container with per-tensor zstd-1 frames.

Output keeps names/dtypes/shapes and tensor order; data_offsets point to the
compressed frame; each tensor gains "nb" = uncompressed nbytes; __metadata__
gains peng_zstd="1". Engine support: c/st.h (findings §41, spec 2026-07-13).
"""
import argparse, json, os, shutil, struct, sys

import zstandard

LEVEL = 1  # measured at the order-0 entropy floor; higher levels buy nothing (§41)
AUX = ("config.json", "generation_config.json")


def read_header(f):
    hlen = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(hlen)), 8 + hlen


def tensor_items(hdr):
    """Tensor entries in file order (by data_offsets start)."""
    items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
    items.sort(key=lambda kv: kv[1]["data_offsets"][0])
    return items


def repack_shard(src, dst):
    cctx = zstandard.ZstdCompressor(level=LEVEL)
    with open(src, "rb") as f:
        hdr, data_start = read_header(f)
        new_hdr, frames, off = {}, [], 0
        for name, t in tensor_items(hdr):
            a, b = t["data_offsets"]
            f.seek(data_start + a)
            frame = cctx.compress(f.read(b - a))
            new_hdr[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "data_offsets": [off, off + len(frame)],
                             "nb": b - a}
            frames.append(frame)
            off += len(frame)
    meta = dict(hdr.get("__metadata__", {}))
    meta["peng_zstd"] = "1"
    new_hdr["__metadata__"] = meta
    hj = json.dumps(new_hdr).encode()
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for frame in frames:
            f.write(frame)
        f.flush(); os.fsync(f.fileno())
    os.rename(tmp, dst)
    return off


def verify_shard(src, dst):
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fs, open(dst, "rb") as fd:
        sh, ss = read_header(fs)
        dh, ds = read_header(fd)
        for name, t in tensor_items(sh):
            a, b = t["data_offsets"]
            fs.seek(ss + a)
            want = fs.read(b - a)
            za, zb = dh[name]["data_offsets"]
            fd.seek(ds + za)
            got = dctx.decompress(fd.read(zb - za), max_output_size=dh[name]["nb"])
            if got != want:
                print(f"VERIFY FAIL: {name} in {os.path.basename(src)}",
                      file=sys.stderr)
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    shards = sorted(x for x in os.listdir(args.indir)
                    if x.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {args.indir}")
    os.makedirs(args.out, exist_ok=True)
    if not args.verify_only:
        tin = tout = 0
        for i, sh in enumerate(shards):
            src = os.path.join(args.indir, sh)
            dst = os.path.join(args.out, sh)
            out = repack_shard(src, dst)
            tin += os.path.getsize(src); tout += os.path.getsize(dst)
            print(f"[{i+1}/{len(shards)}] {sh}: "
                  f"{out/2**30:.2f} GiB ({100*os.path.getsize(dst)/os.path.getsize(src):.1f}%)",
                  flush=True)
        for aux in AUX:
            p = os.path.join(args.indir, aux)
            if os.path.exists(p):
                shutil.copy2(p, args.out)
        meta = os.path.join(args.indir, "_meta")
        if os.path.isdir(meta):
            shutil.copytree(meta, os.path.join(args.out, "_meta"),
                            dirs_exist_ok=True)
        print(f"total: {tin/2**30:.1f} -> {tout/2**30:.1f} GiB "
              f"({100*tout/tin:.1f}%)")
    if args.verify or args.verify_only:
        for sh in shards:
            if not verify_shard(os.path.join(args.indir, sh),
                                os.path.join(args.out, sh)):
                sys.exit(1)
        print("verify: all frames byte-exact")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && ../venv/bin/python -m unittest tests.test_repack_zstd -v"`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add c/tools/repack_zstd.py c/tests/test_repack_zstd.py
git commit -m "feat(tools): repack_zstd.py — per-tensor zstd-1 frames + verify"
```

---

### Task 3: `st.h` — compressed container support

**Files:**
- Modify: `c/st.h` (st_tensor struct, `st_init`, read helpers)
- Test: `c/tests/test_st.c` (append cases), `c/Makefile:77` (add `-lzstd` — already global from Task 1)

- [ ] **Step 1: Write the failing test** — append to `c/tests/test_st.c` `main()` before `return 0;` (and add `#include <zstd.h>` at top):

```c
    /* §41: peng_zstd container — st_init parses znbytes/nb, reads decompress */
    {
        char dir[] = "/tmp/peng_st_zstdXXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        char path[512];
        snprintf(path, sizeof(path), "%s/z.safetensors", dir);
        float vals[8] = {1, -2, 3.5f, 0, 4, -0.5f, 6, 7};
        uint8_t packed[16];
        for (int i = 0; i < 16; i++) packed[i] = (uint8_t)(i % 16);
        char zf[64], zw[64];
        size_t zfn = ZSTD_compress(zf, sizeof(zf), vals, sizeof(vals), 1);
        size_t zwn = ZSTD_compress(zw, sizeof(zw), packed, sizeof(packed), 1);
        CHECK(!ZSTD_isError(zfn) && !ZSTD_isError(zwn));
        char hdr[512];
        int hn = snprintf(hdr, sizeof(hdr),
            "{\"__metadata__\":{\"peng_zstd\":\"1\"},"
            "\"a.weight\":{\"dtype\":\"F32\",\"shape\":[8],"
            "\"data_offsets\":[0,%zu],\"nb\":32},"
            "\"b.weight\":{\"dtype\":\"U8\",\"shape\":[4,8],"
            "\"data_offsets\":[%zu,%zu],\"nb\":16}}",
            zfn, zfn, zfn + zwn);
        uint8_t blob[1024];
        uint64_t hlen = (uint64_t)hn;
        memcpy(blob, &hlen, 8);
        memcpy(blob + 8, hdr, hn);
        memcpy(blob + 8 + hn, zf, zfn);
        memcpy(blob + 8 + hn + zfn, zw, zwn);
        CHECK(write_file(path, blob, 8 + hn + zfn + zwn) == 0);
        shards S;
        st_init(&S, dir);
        st_tensor *ta = st_find(&S, "a.weight");
        st_tensor *tb = st_find(&S, "b.weight");
        CHECK(ta && ta->nbytes == 32 && ta->znbytes == (int64_t)zfn);
        CHECK(tb && tb->nbytes == 16 && tb->znbytes == (int64_t)zwn);
        float out[8];
        CHECK(st_read_f32(&S, "a.weight", out, 8, 0) == 8);
        CHECK(out[0] == 1.0f && out[2] == 3.5f && out[7] == 7.0f);
        uint8_t rawb[16];
        st_read_raw(&S, "b.weight", rawb, 16, 0);
        for (int i = 0; i < 16; i++) CHECK(rawb[i] == (uint8_t)(i % 16));
        float slice[4];
        st_read_slice_f32(&S, "a.weight", 2, 4, slice, 4, 0);
        CHECK(slice[0] == 3.5f && slice[3] == -0.5f);
        unlink(path);
        rmdir(dir);
    }

    /* §41: flagged container with a tensor missing "nb" must be fatal */
    {
        char dir[] = "/tmp/peng_st_zbadXXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        char path[512];
        snprintf(path, sizeof(path), "%s/zbad.safetensors", dir);
        const char *h =
            "{\"__metadata__\":{\"peng_zstd\":\"1\"},"
            "\"a.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        uint8_t blob[512];
        uint64_t hlen = strlen(h);
        memcpy(blob, &hlen, 8);
        memcpy(blob + 8, h, hlen);
        memset(blob + 8 + hlen, 0, 8);
        CHECK(write_file(path, blob, 8 + hlen + 8) == 0);
        CHECK(expect_st_init_fail(dir) == 1);
        unlink(path);
        rmdir(dir);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && make tests/test_st && ./tests/test_st"`
Expected: compile FAIL — `st_tensor` has no member `znbytes`.

- [ ] **Step 3: Implement in `c/st.h`**

3a. Include + struct field:

```c
#include <zstd.h>            /* after the other includes, before json.h */
```

```c
typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;    /* SEMPRE dimensione non compressa (payload logico) */
    int64_t znbytes;   /* >0: frame zstd di znbytes byte a off (peng_zstd) */
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;
```

3b. In `st_init`, after `json_parse` succeeds and before the tensor loop, detect the flag (two-pass: metadata first, tensors after):

```c
        int f_zstd = 0;
        for (int i = 0; i < root->len; i++) {
            if (root->keys[i] && !strcmp(root->keys[i], "__metadata__")) {
                jval *md = root->kids[i];
                jval *z = (md && md->t == J_OBJ) ? json_get(md, "peng_zstd") : NULL;
                if (z && z->t == J_STR && z->str && !strcmp(z->str, "1")) f_zstd = 1;
            }
        }
```

3c. Inside the tensor loop, replace the size bookkeeping: `nbytes` from `"nb"` when flagged, `znbytes` from data_offsets. After the existing `int64_t nbytes = b0 - a0;` line:

```c
            int64_t znbytes = 0;
            if (f_zstd) {
                jval *nb = json_get(m, "nb");
                if (!nb || nb->t != J_NUM || nb->num < 0 ||
                    nb->num != (double)(int64_t)nb->num) {
                    fprintf(stderr, "peng_zstd tensor senza \"nb\": %s in %s\n",
                            name, files[fi]);
                    exit(1);
                }
                znbytes = nbytes;                 /* frame compresso su disco */
                nbytes  = (int64_t)nb->num;       /* payload logico */
            }
```

The existing `nbytes == numel*esz` validation (dtype != 3) now runs on the logical size — unchanged code, correct meaning. At the struct fill add `t->znbytes = znbytes;`.

3d. Frame reader + thread-local scratch, placed right after `pread_full`:

```c
/* §41 peng_zstd: pread del frame compresso in uno scratch per-thread (grow-only),
 * ZSTD_decompress verso dst. dst_cap in byte. Errore frame = container corrotto: fatale. */
static __thread char   *st_zbuf;
static __thread int64_t st_zbuf_cap;
static void st_zread(const st_tensor *t, void *dst, int64_t dst_cap) {
    if (dst_cap < t->nbytes) {
        fprintf(stderr, "st_zread dst too small: %s need %lld cap %lld\n",
                t->name, (long long)t->nbytes, (long long)dst_cap);
        exit(1);
    }
    if (!st_zbuf || st_zbuf_cap < t->znbytes) {
        free(st_zbuf);
        st_zbuf_cap = t->znbytes + 4096;
        st_zbuf = malloc((size_t)st_zbuf_cap);
        if (!st_zbuf) { fprintf(stderr, "OOM st_zbuf\n"); exit(1); }
    }
    if (pread_full(t->fd, st_zbuf, t->znbytes, t->off) != t->znbytes) {
        perror("pread zstd frame"); exit(1);
    }
    size_t r = ZSTD_decompress(dst, (size_t)dst_cap, st_zbuf, (size_t)t->znbytes);
    if (ZSTD_isError(r) || (int64_t)r != t->nbytes) {
        fprintf(stderr, "zstd frame corrotto: %s off=%lld z=%lld -> %s\n",
                t->name, (long long)t->off, (long long)t->znbytes,
                ZSTD_isError(r) ? ZSTD_getErrorName(r) : "size mismatch");
        exit(1);
    }
}
```

3e. Route the read helpers. In `st_read_raw`, replace the single `pread_full` call:

```c
    if (t->znbytes > 0) st_zread(t, out, out_cap_bytes);
    else if (pread_full(t->fd, out, t->nbytes, t->off) != t->nbytes) { perror("pread raw"); exit(1); }
    if (drop) posix_fadvise(t->fd, t->off, t->znbytes > 0 ? t->znbytes : t->nbytes, POSIX_FADV_DONTNEED);
```

In `st_read_f32`, replace the `pread_full(t->fd, raw, ...)` block the same way (`st_zread(t, raw, t->nbytes)` when `t->znbytes > 0`), and use `znbytes` in the final `drop` fadvise as above.

In `st_read_slice_f32`, when `t->znbytes > 0` decompress the whole tensor then slice (cold path, GLM fused experts only):

```c
    if (t->znbytes > 0) {
        char *full = malloc((size_t)t->nbytes);
        if (!full) { fprintf(stderr, "OOM slice zstd\n"); exit(1); }
        st_zread(t, full, t->nbytes);
        memcpy(raw, full + elem_off * esz, (size_t)nb);
        free(full);
    } else if (pread_full(t->fd, raw, nb, boff) != nb) { perror("pread slice"); exit(1); }
```

(keep the malloc of `raw` above; the `drop` fadvise at the end switches to `t->off/t->znbytes` when compressed).

In `st_prefetch` and `st_prefetch_phys`, WILLNEED length becomes `t->znbytes > 0 ? t->znbytes : t->nbytes` (the physical range; add a small inline `st_disk_len(t)` helper used by both and by the `drop` sites above).

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && make test-c"`
Expected: `test_json`, `test_st` (incl. 2 new cases), `test_tier` all PASS.

- [ ] **Step 5: Commit**

```bash
git add c/st.h c/tests/test_st.c
git commit -m "feat(st): peng_zstd containers — znbytes + frame decompress on read"
```

---

### Task 4: `expert_load` compressed path in `c/mimo.c`

**Files:**
- Modify: `c/mimo.c:1179-1261` (`expert_load`)

No new unit test file: the behavior gate is the tiny-oracle run in Task 5 (bit-exact greedy + TF on a repacked container), which exercises this exact code through the real engine including the OVERLAP loader pool.

- [ ] **Step 1: Add per-thread compressed scratch next to `expert_load`** (above it, after `embed_row`):

```c
/* §41: scratch per-thread per i frame zstd degli expert (3 frame coalescenti,
 * ~14 MB compressi). Allineato 4K per il pread O_DIRECT. Grow-only. */
static __thread char   *ez_buf;
static __thread int64_t ez_cap;
static char *ez_scratch(int64_t need) {
    if (!ez_buf || ez_cap < need) {
        free(ez_buf);
        ez_cap = need + 8192;
        if (posix_memalign((void**)&ez_buf, 4096, (size_t)ez_cap)) {
            fprintf(stderr, "OOM ez_scratch\n"); exit(1);
        }
    }
    return ez_buf;
}
static void ez_decompress(const st_tensor *t, const char *src, char *dst) {
    size_t r = ZSTD_decompress(dst, (size_t)t->nbytes, src, (size_t)t->znbytes);
    if (ZSTD_isError(r) || (int64_t)r != t->nbytes) {
        fprintf(stderr, "zstd expert frame corrotto: %s -> %s\n", t->name,
                ZSTD_isError(r) ? ZSTD_getErrorName(r) : "size mismatch");
        exit(1);
    }
}
```

- [ ] **Step 2: Insert the compressed branch in `expert_load`**, right after the `tw`/`tq` lookup and the slab/fslab (re)allocation (which stay as-is — `wtot`/`ftot` are already computed from the **uncompressed** `nbytes`, so slab sizing is untouched). Replace the block from `int ord[3]={0,1,2};` through the `.qs` pread loop with:

```c
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int64_t pos[3]; int done=0;
    if(tw[0]->znbytes>0){                        /* §41: container peng_zstd */
        int64_t zw[3], zpos[3], zo=0;
        for(int a=0;a<3;a++){ int k=ord[a]; zw[a]=tw[k]->znbytes; zpos[a]=zo; zo+=zw[a]; }
        int zcontig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
                   && tw[ord[0]]->off+tw[ord[0]]->znbytes==tw[ord[1]]->off
                   && tw[ord[1]]->off+tw[ord[1]]->znbytes==tw[ord[2]]->off;
        char *zb=ez_scratch(zo+8192); int64_t zskew=0; int zdone=0;
        if(zcontig){
            int64_t off0=tw[ord[0]]->off;
            int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
            if(dfd>=0){                          /* O_DIRECT: offset/len allineati a 4K */
                int64_t base=off0 & ~4095LL, need=(off0-base)+zo;
                int64_t len=(need+4095)&~4095LL;
                ssize_t r=pread(dfd, zb, len, base);
                if(r>=need){ zskew=off0-base; zdone=1; }
            }
            if(!zdone){
                if(pread(tw[ord[0]]->fd, zb, zo, off0)!=zo){ perror("pread zexpert"); exit(1); }
                zskew=0; zdone=1;
            }
        } else {
            for(int a=0;a<3;a++){ int k=ord[a];
                if(pread(tw[k]->fd, zb+zpos[a], zw[a], tw[k]->off)!=zw[a]){ perror("pread zexpert"); exit(1); } }
            zskew=0;
        }
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            ez_decompress(tw[k], zb+zskew+zpos[a], (char*)s->slab+o);
            pos[k]=o; o+=tw[k]->nbytes; }
        done=1;
        float *fpz[3]; int64_t foz=0;            /* scale: frame piccoli, decompress diretto */
        for(int k=0;k<3;k++){
            st_zread(tq[k], (char*)(s->fslab+foz), tq[k]->nbytes);
            fpz[k]=s->fslab+foz; foz+=tq[k]->nbytes/4; }
        if(g_drop){
            posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, zo, POSIX_FADV_DONTNEED);
            for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->znbytes, POSIX_FADV_DONTNEED);
        }
        QT *qtz[3]={&s->g,&s->u,&s->d}; int OOz[3]={I,I,D}, IIz[3]={D,D,I};
        for(int k=0;k<3;k++){
            int64_t nb=tw[k]->nbytes;
            int fmt = (nb==(int64_t)OOz[k]*IIz[k])?1 : (nb==(int64_t)OOz[k]*((IIz[k]+1)/2))?2 : 3;
            qtz[k]->fmt=fmt; qtz[k]->O=OOz[k]; qtz[k]->I=IIz[k]; qtz[k]->qf=NULL;
            qtz[k]->q8=(int8_t*)(s->slab+pos[k]); qtz[k]->q4=s->slab+pos[k]; qtz[k]->s=fpz[k];
        }
        s->eid=eid; return;
    }
```

Everything below (the legacy contiguous/O_DIRECT/fallback path, the `.qs` preads, the QT view fill) stays byte-for-byte as today — the compressed branch returns early.

Note the deliberate reuse: `pos[k]` cumsum uses **uncompressed** `nbytes` (slab layout identical to today), `zpos[a]`/`zo` use `znbytes` (disk layout). `fmt` detection reads `tw[k]->nbytes` — the logical size — so int4/int8/int2 detection is unchanged.

- [ ] **Step 3: Build**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && make mimo 2>&1 | tail -3"`
Expected: clean build.

- [ ] **Step 4: Legacy regression — tiny oracle on the UNCOMPRESSED container**

Run: `wsl -e bash -c "cd /mnt/c/peng-mimo/c && SNAP=mimo_tiny_i4 OMP_NUM_THREADS=4 ./mimo 64 8 8 2>&1 | tail -4 && SNAP=mimo_tiny_i4 OMP_NUM_THREADS=4 TF=1 ./mimo 64 8 8 2>&1 | tail -4"`
Expected: greedy and TF match current baseline (greedy 15/20, TF 31/32 per §39 — same numbers as before this change).

- [ ] **Step 5: Commit**

```bash
git add c/mimo.c
git commit -m "feat(mimo): expert_load reads peng_zstd containers (coalesced frame pread + decompress into slab)"
```

---

### Task 5: Bit-exact gate on the tiny container

**Files:** none (validation)

- [ ] **Step 1: Repack the tiny container**

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && ../venv/bin/python tools/repack_zstd.py --indir mimo_tiny_i4 --out mimo_tiny_i4z --verify"
```

Expected: `verify: all frames byte-exact`.

- [ ] **Step 2: Oracle on the compressed container — must equal Step 4 of Task 4 exactly**

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && SNAP=mimo_tiny_i4z OMP_NUM_THREADS=4 ./mimo 64 8 8 2>&1 | tail -4 && SNAP=mimo_tiny_i4z OMP_NUM_THREADS=4 TF=1 ./mimo 64 8 8 2>&1 | tail -4"
```

Expected: identical greedy/TF scores to the uncompressed run (bit-exact: same logits, same tokens). Any deviation = bug in Task 3/4 — stop and fix before touching the full container.

- [ ] **Step 3: Also gate with OVERLAP forced on/off (loader-pool path both ways)**

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && OVERLAP=0 SNAP=mimo_tiny_i4z OMP_NUM_THREADS=4 TF=1 ./mimo 64 8 8 2>&1 | tail -2 && OVERLAP=1 SNAP=mimo_tiny_i4z OMP_NUM_THREADS=4 TF=1 ./mimo 64 8 8 2>&1 | tail -2"
```

Expected: same TF score both ways.

- [ ] **Step 4: Commit checkpoint (docs note only if anything was fixed).**

---

### Task 6: Full container repack + verify

**Files:** none (data; ~1–2 h wall clock — run in background)

- [ ] **Step 1: Repack `/root/mimo25_i4` → `/root/mimo25_i4z`** (790 GB free, needs ~113 GB)

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && nohup ../venv/bin/python tools/repack_zstd.py --indir /root/mimo25_i4 --out /root/mimo25_i4z --verify > /root/repack_zstd.log 2>&1 &"
```

Monitor: `wsl -e bash -c "tail -3 /root/repack_zstd.log"`. Expected final lines: total ratio ~74–75% and `verify: all frames byte-exact`.

- [ ] **Step 2: Sanity — engine boots and generates on the compressed container**

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && SNAP=/root/mimo25_i4z COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TEMP=0 NGEN=8 PROMPT='Write one short sentence about Rome.' ./mimo 64 4 8 2>&1 | tail -6"
```

Expected: coherent 8-token output, no frame errors. With TEMP=0, tokens must equal the same command against `SNAP=/root/mimo25_i4` (spot bit-exactness at full scale).

---

### Task 7: Speed measurement (§37 protocol) + docs

**Files:**
- Modify: `findings.md` (§41 — add measured results), `roadmap.md` (A1 checkboxes), `README.md` (if record moves)

- [ ] **Step 1: Warm-pair benchmark, compressed vs uncompressed, same §37 stack**

```bash
# run each TWICE (warm run of 2 counts), record PROFILE lines
wsl -e bash -c "cd /mnt/c/peng-mimo/c && for s in /root/mimo25_i4 /root/mimo25_i4z; do for i in 1 2; do TAO=1 SPEED=1 PILOT=0 SNAP=\$s PROMPT='Write one short sentence about Rome.' NGEN=24 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 PROFILE=1 ./mimo 64 4 8 2>&1 | grep -E 'tok/s|PROFILE'; done; done"
```

Record: tok/s, disk seconds, hit-rate for all four runs. Success gate: compressed ≥ uncompressed; target ~0.72+ tok/s (estimate §41). If decompress serializes (disk drops but matmul/other inflates), try `OVERLAP_T=8`:

```bash
wsl -e bash -c "cd /mnt/c/peng-mimo/c && OVERLAP_T=8 TAO=1 SPEED=1 PILOT=0 SNAP=/root/mimo25_i4z PROMPT='Write one short sentence about Rome.' NGEN=24 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 PROFILE=1 ./mimo 64 4 8 2>&1 | grep -E 'tok/s|PROFILE'"
```

(`OVERLAP_T` is read at `c/mimo.c:3704` area next to `OVERLAP`; if only `g_overlap_t=4` constant exists, add the env read `g_overlap_t = getenv("OVERLAP_T")?atoi(getenv("OVERLAP_T")):4;` beside `g_overlap` and commit that as part of this task.)

- [ ] **Step 2: Update `findings.md` §41** — append a "Measured (landed)" block with the real tok/s table (4+ runs), final container size, and whether OVERLAP_T mattered. Update `roadmap.md` A1: tick repack/loader/measure boxes, note the new record if any. Update README speed line if the record moves.

- [ ] **Step 3: Commit**

```bash
git add findings.md roadmap.md README.md
git commit -m "docs(findings): §41 zstd expert pack measured — <X> tok/s vs 0.60 baseline"
```

---

## Self-Review (done at write time)

- **Spec coverage:** format flag + `"nb"` (T2/T3), repack tool + verify (T2), `st_init`/read helpers (T3), `expert_load` coalesced + O_DIRECT + scratch (T4), prefetch/DONTNEED on compressed ranges (T3 3e, T4 step 2), `-lzstd` (T1), bit-exact gates tiny+full (T5, T6 step 2), legacy regression (T4 step 4), speed protocol + docs (T7). Loader-pool decompress is implicit: `expert_load` runs on `ov_worker` threads (mimo.c:1418) — covered by T5 step 3 OVERLAP gate.
- **Placeholder scan:** none — all code inline.
- **Type consistency:** `znbytes` (st_tensor), `st_zread(t, dst, cap)`, `ez_scratch/ez_decompress` names used consistently across T3/T4. `st_zread` is defined in st.h (T3) and reused by mimo.c (T4) — st.h is included by mimo.c, and `__thread` statics are per-TU-safe here because st.h is a single-header included once.
