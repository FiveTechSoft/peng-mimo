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

    def test_unpack_roundtrip(self):
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        back = tempfile.mkdtemp(prefix="peng_back_")
        r = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "repack_zstd.py"),
             "--indir", self.dst, "--out", back, "--unpack"],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        orig = os.path.join(self.src, "out-00000.safetensors")
        got = os.path.join(back, "out-00000.safetensors")
        with open(orig, "rb") as a, open(got, "rb") as b:
            self.assertEqual(a.read(), b.read())

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
