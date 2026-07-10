"""Compare C tokenizer (tok.h, via the test binary) against HF AutoTokenizer."""
import json, subprocess, unittest
from pathlib import Path
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent   # c/
CASES = [
    "Hello, world!",
    "café mañana übermäßig 東京タワー 🐦🔥",
    "def f(x):\n    return x**2  # comment",
    "   leading spaces\tand\ttabs\n\n",
    "número 3.14159, 100%, C++/C#, e=mc^2",
    "الْعَرَبِيَّة русский 한국어 ελληνικά",
]

class TestMimoTok(unittest.TestCase):
    def test_roundtrip_vs_hf(self):
        hf = AutoTokenizer.from_pretrained(HERE / "mimo_tok")
        for s in CASES:
            want = hf.encode(s, add_special_tokens=False)
            got = json.loads(subprocess.check_output(
                [str(HERE / "tok_mimo_cli"), str(HERE / "mimo_tok")],
                input=s.encode()).decode())
            self.assertEqual(want, got, f"mismatch on {s!r}")

if __name__ == "__main__":
    unittest.main()
