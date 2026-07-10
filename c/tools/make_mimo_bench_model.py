"""Build a deterministic, medium-size MiMoV2 fixture for backend benchmarks.

This is not a useful language model. It preserves the real mimo_v2 data flow
(hybrid full/SWA attention, fused QKV, noaux_tc MoE routing, attention sink
bias) while remaining small enough to generate locally and run repeated
CPU streaming benchmarks without downloading a real checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimo_ref.configuration_mimo_v2 import MiMoV2Config
from mimo_ref.modeling_mimo_v2 import MiMoV2ForCausalLM


def build_config() -> MiMoV2Config:
    return MiMoV2Config(
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=64,
        v_head_dim=64,
        partial_rotary_factor=0.5,   # rope_dim 32
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        sliding_window=128,
        hybrid_layer_pattern=[0, 1, 1, 1, 0, 1, 1, 1],
        moe_layer_freq=[0, 1, 1, 1, 1, 1, 1, 1],
        n_routed_experts=32,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        attention_projection_layout="fused_qkv",
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        attention_value_scale=0.707,
        swa_num_attention_heads=16,
        swa_num_key_value_heads=4,
        attention_bias=False,
        layernorm_epsilon=1e-5,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mimo_bench_medium")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    model = MiMoV2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.1, 0.1, cfg.n_routed_experts)
                )
            if getattr(layer.self_attn, "attention_sink_bias", None) is not None:
                layer.self_attn.attention_sink_bias.copy_(
                    torch.linspace(-0.5, 0.5, layer.self_attn.num_attention_heads)
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")

    model.to(args.device)
    prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        full = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0]
        logits = model(full.unsqueeze(0), use_cache=False).logits[0]

    ref = {
        "prompt_ids": prompt,
        "full_ids": full.cpu().tolist(),
        "tf_pred": logits.argmax(-1).cpu().tolist(),
    }
    (output / "ref_mimo.json").write_text(json.dumps(ref))
    manifest = {
        "seed": args.seed,
        "parameters": params,
        "parameters_billions": round(params / 1e9, 4),
        "purpose": "backend benchmark fixture; random weights, not a language model",
    }
    (output / "bench_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
