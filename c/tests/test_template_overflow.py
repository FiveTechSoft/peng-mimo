"""F-02: mimo_turn_render must not overflow a fixed buffer (TEMPLATE_DUMP path).

Does not require transformers. Needs a built ./mimo binary.
"""
import os
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MIMO = HERE / "mimo"


@unittest.skipUnless(MIMO.exists(), "mimo binary not built (make mimo)")
class TestTemplateOverflow(unittest.TestCase):
    def _dump(self, user_msg: str):
        env = dict(os.environ, TEMPLATE_DUMP="1")
        env.pop("THINK", None)
        env.pop("SYSTEM", None)
        return subprocess.run(
            [str(MIMO)],
            input=f"U {user_msg}\n".encode(),
            env=env,
            capture_output=True,
        )

    def test_small_fits(self):
        r = self._dump("Hello")
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        out = r.stdout.decode()
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)

    def test_70000_chars_no_asan_crash(self):
        # Previously: heap-buffer-overflow, exit 134 under ASan.
        huge = "x" * 70000
        r = self._dump(huge)
        self.assertEqual(r.returncode, 2, msg=f"stderr={r.stderr.decode()!r}")
        self.assertIn(b"does not fit", r.stderr)
        # Must not write a huge corrupted stdout blob as if success
        self.assertLess(len(r.stdout), 1000)

    def test_cap_boundary_short_ok(self):
        # A few hundred chars still fit in 64 KiB template buffer.
        r = self._dump("y" * 500)
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        self.assertIn("y" * 500, r.stdout.decode())


if __name__ == "__main__":
    unittest.main()
