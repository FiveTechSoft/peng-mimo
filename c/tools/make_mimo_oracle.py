"""Tiny random MiMoV2 oracle: real architecture, minuscule dims.
Saves weights+config to c/mimo_tiny/ and greedy/TF reference to c/ref_mimo.json."""
import json, sys, torch
sys.path.insert(0, "tools")
from mimo_ref.configuration_mimo_v2 import MiMoV2Config
from mimo_ref.modeling_mimo_v2 import MiMoV2ForCausalLM

torch.manual_seed(1234)

cfg = MiMoV2Config(
    vocab_size=256, hidden_size=128, intermediate_size=64,
    num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2,
    head_dim=24, v_head_dim=16, partial_rotary_factor=0.334,   # rope_dim = 8, even
    rope_theta=10000000.0, swa_rope_theta=10000.0,
    sliding_window=8,
    hybrid_layer_pattern=[0,1,1,0,1,1],       # full at 0 and 3, SWA elsewhere
    moe_layer_freq=[0,1,1,1,1,1],             # layer 0 dense
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=None,
    attention_projection_layout="fused_qkv",
    add_full_attention_sink_bias=False, add_swa_attention_sink_bias=True,
    attention_value_scale=0.707,
    swa_num_attention_heads=8, swa_num_key_value_heads=4,
    attention_bias=False, layernorm_epsilon=1e-5,
    max_position_embeddings=4096, tie_word_embeddings=False,
)
cfg._attn_implementation = "eager"
model = MiMoV2ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.1, 0.1, cfg.n_routed_experts))
        if getattr(layer.self_attn, "attention_sink_bias", None) is not None:
            layer.self_attn.attention_sink_bias.copy_(
                torch.linspace(-0.5, 0.5, layer.self_attn.num_attention_heads))

print("=== state_dict tensors (names for the C loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("full  :", full)
print("tf_pred:", tf_pred)

model.save_pretrained("mimo_tiny", safe_serialization=True)
json.dump(cfg.to_dict(), open("mimo_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
          open("ref_mimo.json", "w"))
print("saved: mimo_tiny/ + ref_mimo.json")
