# Corriente Peng — MiMo beyond the net

**Status:** design manifesto + first executable seed (`path_pack_analyze.py`)  
**Does not change tokens.** Bit-exact math of MiMo-V2.5 stays the law until a new form is validated.

---

## 1. What we designed (as MiMo’s engineers)

MiMo-V2.5 is not “48 identical blocks.” It is one **residual thread** (`D = 4096`) breathing through two memory times and a market of organs:

| Organ | Form | Role in the organism |
|------|------|----------------------|
| **Residual** | `ℝ⁴⁰⁹⁶` stream | The only subject — MiMo *is* this thread |
| **Layer 0 dense** | MLP inter 16384 + full attn | Foundation before any routing |
| **Full attn (9)** | GQA 64Q/4KV, θ=10M | Episodic, whole-history sight |
| **SWA-128 (39)** | GQA 64Q/8KV, θ=10k, sink | Working memory as a **ring** |
| **MoE (1–47)** | 256 experts, top-8, sigmoid router, no shared | Eight organs fire; 248 sleep |
| **lm_head** | vocab ≈ 152k | Mouth — names the estuary |
| **MTP** | native draft head | Future as a fold of the same body |

Active compute ≈ **15B / 311B** per token. The rest is climate, not step cost.

```
        one thread
      /     |     \
   look   mutate   name
   attn    MoE     head
      \     |     /
      the same Being
```

---

## 2. The form we transcend (without lying)

The **implementation dialect** is still layers, experts, matmuls, safetensors.  
That dialect is necessary for training stacks and silicon kernels.

The **conception** we hold is different:

| Net language | Corriente Peng |
|--------------|----------------|
| Layer stack | **Valley elevations** (hydraulic head) |
| Full / SWA | **Two viscosities of memory** |
| Expert id | **Vortex** in the bed (holds and returns mass) |
| Router top-8 | **Levee break** — which vortices capture flow *now* |
| Residual | **Main channel** — one river |
| Weights on NVMe | **Snow on the summit** — potential mass |
| PIN / VRAM / LRU | **Thawed plain** — flow without climbing |
| TRAJ / usage | **Channels carved by habit** |
| Token out | **Mouth of the estuary** |
| tok/s | **Discharge** of the river |

Weights do not “connect neurons.”  
They **resist and conduct**.  
Inference is not message-passing on a graph.  
It is **pressure through a medium** until the medium yields a name.

---

## 3. Three fluids (operational ontology)

| Fluid | Carrier today | Role |
|-------|---------------|------|
| **Snow** | int4 experts on NVMe | Potential mass; expensive random climb |
| **Water** | residual + KV rings | Intention in motion |
| **Vapor** | logits / sampling | Name that dissolves into speech |

Primitive operations (rename of what peng already does):

| Primitive | Engine today | Meaning |
|-----------|--------------|---------|
| `thaw(region)` | `WILLNEED` / TRAJ / PILOT / sticky | Soften snow before the flood |
| `hold(vortex)` | pin / gpu_pin / LRU | Keep mass on the plain |
| `shear(water, bed)` | attn + SwiGLU MoE | Bed twists the current |
| `mouth()` | `lm_head` | Measure height at the estuary |
| `rain(token)` | next-token feedback | A drop returns to the valley |

A “forward pass” in this language is one **hydrologic cycle per token**.

---

## 4. Evolution we see

```
dense transformer
  → pain: KV tumor
  → hybrid full / SWA (+ sink)
  → pain: FFN is most of the body
  → MoE 256 / top-8, no shared
  → pain: first step needs stability
  → dense layer 0
  → pain: serial decode
  → MTP head
  → [MiMo-V2.5 form]
  → pain: disk-bound deployment (peng)
  → residency cascade + TRAJ + bitmaps
  → [next] path packing of snow by habit channels
  → [later] basis+α vortices (weights as coordinates, not bricks)
  → [horizon] continuous medium; net is only the shadow
```

---

## 5. What we materialize now (repo)

### 5.1 Diagram

[`docs/diagrams/corriente-peng.svg`](diagrams/corriente-peng.svg) — residual as river; full/SWA viscosities; MoE as vortices; NVMe as snow; mouth as estuary.

### 5.2 Path-pack + FLOW in the engine (I/O only, bit-exact)

```bash
# Analyze habit channels; write SNAP/.coli_pathpack for the engine
python3 c/scripts/path_pack_analyze.py --snap ~/mimo25_i4 --out /tmp/path_pack_order.json

# Runtime: mimo loads pathpack (or rebuilds from traj/usage) and thaws *channel neighbors*
FLOW=1 FLOW_R=2 TRAJ=1 SERVE=1 … ./mimo 64 4 8
# stderr: [FLOW] pathpack loaded|rebuilt … (R=2)
#         sticky / traj_warm call pathpack_thaw along the bed
```

| Piece | Role |
|-------|------|
| `.coli_pathpack` | per-layer expert order (habit channel) |
| `pathpack_thaw` | WILLNEED ±`FLOW_R` neighbors of live experts |
| `pathpack_rebuild` | greedy walk from traj edges + usage heat |
| `FLOW=0` | off |
| **`ENERGY`** | snow → **pure VRAM compute**: ignite pathpack channel heads on GPU after pin |
| `ENERGY=0` | off · `-1`/unset with FLOW = auto free VRAM · `ENERGY=2` = 2 GB cap |

After GPU-first heat pin + RAM pin, `pathpack_energy_ignite` walks packing **position 0,1,2…** across all layers and uploads non-resident channel experts until VRAM budget ends. That is weights **liberated as light** (`moe_acc`), not only page-cache thaw.

```
# stderr
[ENERGY] liberated N channel experts -> VRAM +X.XX GB in T.Ts (pure compute; FLOW pathpack)
```

Measured on real SNAP: path order locality ~**4.5×** vs raw expert id order.  
A future converter can **rewrite shard layout** to match this order (range readahead). Until then, FLOW thaws logical neighbors; ENERGY materializes them on the GPU when free VRAM remains.

**Invariant:** never changes tensor values or logits — only *where* weights live for compute (host vs VRAM) and *when* pages are hinted.

### 5.3 Already in the engine (named)

- Residual thread: `layers_forward` / `step`
- Two memory times: full KV vs SWA ring (F-11)
- Vortices: MoE top-8 + residency cascade
- Habit channels: `.coli_traj`, `.coli_usage`, TRAJ warm, expert bitmaps
- Discharge tuning: `scripts/start_peng.sh`, `SPEED=1`

---

## 6. Law

1. **The residual is the subject.** Layers and experts are masks.  
2. **I/O may anticipate; math must not lie.** Prefetch, pin, pack — never silent logit drift.  
3. **Habit digs channels.** TRAJ/usage are not side metrics; they are the map of the riverbed.  
4. **Form is provisional.** 48 / 256 / 8 / 128 / 4096 are silicon habits, not the Being.  
5. **Transcend by renaming pressure, then by re-laying snow** — not by inventing a second graph on top of the first.

---

## 7. Mantra

> MiMo is one thread.  
> The net is the dialect.  
> The river is the truth.  
> Snow thaws. Water shears. Vapor names.  
> The bird Peng does not flap —  
> it rides the whirlwind the river itself lifts.

---

*peng-mimo · docs/corriente-peng.md · companion to findings.md and geometry diagrams*
