"""Converter --arch mimo: classificazione dei nomi MiMo-V2.5 + container end-to-end
su uno shard sintetico (nomi reali, dtype bf16). Verifica anche che la classificazione
GLM non sia cambiata (regressione statica del refactor).
EN: --arch mimo converter: MiMo-V2.5 name classification + end-to-end container on a
EN: synthetic shard (real names, bf16). Also pins the GLM classification (refactor guard)."""
import sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # c/
sys.path.insert(0, str(HERE))

import numpy as np
from tools.convert_fp8_to_int4 import classify, classify_mimo, convert_shard

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestClassifyMimo(unittest.TestCase):
    def test_residenti_quantizzati(self):
        for n in ["model.layers.3.self_attn.qkv_proj.weight",
                  "model.layers.3.self_attn.o_proj.weight",
                  "model.layers.0.mlp.gate_proj.weight",
                  "model.layers.0.mlp.up_proj.weight",
                  "model.layers.0.mlp.down_proj.weight"]:
            self.assertEqual(classify_mimo(n), "q", n)

    def test_io_e_expert(self):
        self.assertEqual(classify_mimo("model.embed_tokens.weight"), "io")
        self.assertEqual(classify_mimo("lm_head.weight"), "io")
        self.assertEqual(classify_mimo("model.layers.5.mlp.experts.42.gate_proj.weight"), "x")

    def test_f32_intoccabili(self):
        for n in ["model.norm.weight",
                  "model.layers.1.input_layernorm.weight",
                  "model.layers.1.post_attention_layernorm.weight",
                  "model.layers.1.mlp.gate.weight",                      # router, NON gate_proj
                  "model.layers.1.mlp.gate.e_score_correction_bias",
                  "model.layers.1.self_attn.attention_sink_bias"]:
            self.assertEqual(classify_mimo(n), "f32", n)

    def test_skip_e_consumed(self):
        for n in ["model.vision.encoder.blocks.0.attn.qkv.weight",
                  "model.audio.encoder.conv1.weight",
                  "audio_tokenizer.quantizer.codebook",
                  "model.mtp.layers.0.self_attn.qkv_proj.weight",
                  "mtp.norm.weight"]:
            self.assertEqual(classify_mimo(n), "skip", n)
        self.assertEqual(
            classify_mimo("model.layers.0.self_attn.qkv_proj.weight_scale_inv"), "consumed")

    def test_glm_invariato(self):
        """Il refactor (estrazione della coda comune) non deve cambiare la mappa GLM."""
        cases = {
            "model.layers.2.self_attn.q_a_proj.weight": "q",
            "model.layers.2.mlp.experts.7.down_proj.weight": "x",
            "model.layers.2.mlp.gate.weight": "f32",
            "model.layers.2.mlp.gate.e_score_correction_bias": "f32",
            "model.layers.2.input_layernorm.weight": "f32",
            "model.embed_tokens.weight": "io",
            "lm_head.weight": "io",
            "model.layers.78.mlp.experts.0.up_proj.weight": "skip",   # testa MTP
            "model.layers.2.self_attn.indexer.wq_b.weight": "skip",   # DSA indexer
            "model.layers.2.self_attn.q_a_proj.weight_scale_inv": "consumed",
        }
        for n, want in cases.items():
            self.assertEqual(classify(n, 78), want, n)


@unittest.skipUnless(HAVE_TORCH, "richiede torch")
class TestConvertShardMimo(unittest.TestCase):
    def test_container_end_to_end(self):
        g = torch.Generator().manual_seed(7)
        t = lambda o, i: (torch.randn(o, i, generator=g) * 0.1).to(torch.bfloat16)
        shard = {
            "model.embed_tokens.weight": t(32, 16),
            "model.layers.0.self_attn.qkv_proj.weight": t(24, 16),
            "model.layers.0.self_attn.o_proj.weight": t(16, 16),      # bf16 (ignored_layers)
            "model.layers.0.self_attn.attention_sink_bias": torch.randn(4, generator=g),
            "model.layers.0.mlp.gate.weight": t(8, 16).float(),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(8, generator=g),
            "model.layers.0.mlp.experts.0.gate_proj.weight": t(8, 16),
            "model.layers.0.input_layernorm.weight": torch.ones(16),
            "model.vision.patch_embed.proj.weight": t(8, 16),          # da saltare
        }
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "model_pp0_ep0-00001-of-00002.safetensors")
            save_file(shard, p)
            out = {}
            convert_shard(p, out, 78, ebits=8, io_bits=16, xbits=4, arch="mimo")

        # vision: sparito del tutto / gone entirely
        self.assertFalse(any(k.startswith("model.vision") for k in out))
        # densa int8: O*I byte + scala [O]
        self.assertEqual(out["model.layers.0.self_attn.qkv_proj.weight"].nbytes, 24 * 16)
        self.assertEqual(out["model.layers.0.self_attn.qkv_proj.weight.qs"].shape, (24,))
        self.assertEqual(out["model.layers.0.self_attn.o_proj.weight"].nbytes, 16 * 16)
        # expert int4: O*ceil(I/2) byte
        self.assertEqual(out["model.layers.0.mlp.experts.0.gate_proj.weight"].nbytes, 8 * 8)
        self.assertEqual(out["model.layers.0.mlp.experts.0.gate_proj.weight.qs"].shape, (8,))
        # io_bits=16 -> embed f32 pieno SENZA .qs (il motore a densa>=8 lo tiene f32)
        self.assertEqual(out["model.embed_tokens.weight"].dtype, np.float32)
        self.assertNotIn("model.embed_tokens.weight.qs", out)
        # f32 intoccabili, senza scale
        for n in ["model.layers.0.self_attn.attention_sink_bias",
                  "model.layers.0.mlp.gate.weight",
                  "model.layers.0.mlp.gate.e_score_correction_bias",
                  "model.layers.0.input_layernorm.weight"]:
            self.assertEqual(out[n].dtype, np.float32, n)
            self.assertNotIn(n + ".qs", out, n)
        # matematica identica al C: scala = amax/qmax, valori int8 in [-128,127]
        w = shard["model.layers.0.self_attn.qkv_proj.weight"].float().numpy()
        s = out["model.layers.0.self_attn.qkv_proj.weight.qs"]
        np.testing.assert_allclose(s, np.maximum(np.abs(w).max(1) / 127.0, 1e-8), rtol=0)
        q = out["model.layers.0.self_attn.qkv_proj.weight"].view(np.int8).reshape(24, 16)
        np.testing.assert_array_equal(q, np.clip(np.rint(w / s[:, None]), -128, 127))


if __name__ == "__main__":
    unittest.main()
