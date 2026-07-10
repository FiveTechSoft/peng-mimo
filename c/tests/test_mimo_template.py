"""MiMo-V2.5 chat template: C builder (mimo.c mimo_turn_render) vs HF apply_chat_template.

Executable comparison: TEMPLATE_DUMP=1 makes the mimo binary render turns from stdin
and print the exact prompt string it would tokenize, without loading any model.
Protocol (one line per message):
    "U <msg>"  -> user turn, rendered through the chat template builder
    "A <text>" -> what the model WOULD have generated (kept verbatim: in serve mode the
                  generated reply stays in the KV cache without its <|im_end|> stop token,
                  which spec_decode never emits; the NEXT user turn opens by closing it)
HF renders an assistant message {'content': 'A1'} as '<think></think>A1', so the
simulated generation on the C side must include the empty think block explicitly.
Kept in sync with mimo_turn_render() in c/mimo.c.
"""
import subprocess, unittest
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent   # c/
MIMO = HERE / "mimo"


def c_render(lines, env_extra=None):
    import os
    env = dict(os.environ, TEMPLATE_DUMP="1")
    env.pop("THINK", None)
    env.pop("SYSTEM", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.check_output(
        [str(MIMO)], input="".join(l + "\n" for l in lines).encode(),
        env=env).decode()


@unittest.skipUnless(MIMO.exists(), "binario mimo non compilato (make mimo)")
class TestMimoTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf = AutoTokenizer.from_pretrained(HERE / "mimo_tok")

    def hf_render(self, msgs, **kw):
        return self.hf.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kw)

    def test_single_user_turn(self):
        want = self.hf_render([{"role": "user", "content": "Hello"}])
        self.assertEqual(want, c_render(["U Hello"]))

    def test_multi_turn(self):
        want = self.hf_render([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        got = c_render(["U Q1", "A <think></think>A1", "U Q2"])
        self.assertEqual(want, got)

    def test_multi_turn_with_reasoning(self):
        # keep_all_reasoning defaults to true: the generated <think> block stays in KV
        want = self.hf_render([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>R1</think>A1"},
            {"role": "user", "content": "Q2"},
        ])
        got = c_render(["U Q1", "A <think>R1</think>A1", "U Q2"])
        self.assertEqual(want, got)

    def test_system_message(self):
        want = self.hf_render([
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Q"},
        ])
        self.assertEqual(want, c_render(["U Q"], {"SYSTEM": "Sys prompt"}))

    def test_thinking_disabled(self):
        # THINK=0 == enable_thinking=False: generation prompt pre-fills <think></think>
        want = self.hf_render([{"role": "user", "content": "Hello"}],
                              enable_thinking=False)
        self.assertEqual(want, c_render(["U Hello"], {"THINK": "0"}))


if __name__ == "__main__":
    unittest.main()
