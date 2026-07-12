"""F-05/F-06 converter helpers: atomic write, validation, revision URLs."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np
from tools.convert_fp8_to_int4 import (
    atomic_save_file,
    file_sha256,
    hf_resolve_url,
    safetensors_is_valid,
    write_convert_manifest,
)

try:
    from safetensors.numpy import save_file
    HAVE_ST = True
except ImportError:
    HAVE_ST = False


class TestHfResolve(unittest.TestCase):
    def test_pinned_sha(self):
        u = hf_resolve_url("org/model", "model.safetensors.index.json", "abc123def")
        self.assertEqual(
            u,
            "https://huggingface.co/org/model/resolve/abc123def/model.safetensors.index.json",
        )

    def test_default_main(self):
        u = hf_resolve_url("org/model", "x.safetensors")
        self.assertIn("/resolve/main/", u)


@unittest.skipUnless(HAVE_ST, "safetensors not installed")
class TestAtomicSave(unittest.TestCase):
    def test_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            outp = os.path.join(td, "out-00000.safetensors")
            tensors = {"w": np.arange(8, dtype=np.float32)}
            atomic_save_file(tensors, outp)
            self.assertTrue(safetensors_is_valid(outp))
            self.assertFalse(os.path.exists(outp + ".tmp"))
            h = file_sha256(outp)
            self.assertEqual(len(h), 64)

    def test_invalid_truncated_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.safetensors")
            with open(bad, "wb") as f:
                f.write(b"\x00" * 32)  # not a valid header
            self.assertFalse(safetensors_is_valid(bad))

    def test_resume_skips_valid_only(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "out-00000.safetensors")
            atomic_save_file({"a": np.ones(4, np.float32)}, good)
            trunc = os.path.join(td, "out-00001.safetensors")
            with open(trunc, "wb") as f:
                f.write(b"half")
            self.assertTrue(safetensors_is_valid(good))
            self.assertFalse(safetensors_is_valid(trunc))

    def test_manifest_written(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_convert_manifest(
                td,
                repo="XiaomiMiMo/MiMo-V2.5-FP8",
                revision="deadbeef",
                arch="mimo",
                ebits=8,
                io_bits=16,
                xbits=4,
                shards_meta=[{"name": "out-00000.safetensors", "size": 1, "sha256": "00"}],
                source_index_sha="aa",
            )
            self.assertTrue(os.path.isfile(p))
            man = json.loads(Path(p).read_text(encoding="utf-8"))
            self.assertEqual(man["peng_format_version"], 1)
            self.assertEqual(man["revision"], "deadbeef")
            self.assertEqual(man["arch"], "mimo")


if __name__ == "__main__":
    unittest.main()
