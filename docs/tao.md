# Flowing with the Tao (peng-mimo)

> The Tao that can be forced is not the eternal Tao.  
> The residual that can be thrashed is not the living residual.

This is not a new algorithm. It is the **name of how the stack already wants to move** when we stop fighting the river.

---

## 1. The Way is one thread

MiMo is not 311B parameters.  
MiMo is **one residual** (`D = 4096`) that looks, mutates, and names.

| Tao | Engine |
|-----|--------|
| The nameless is the origin | residual before `lm_head` |
| The named is the mother of ten thousand things | tokens at the estuary |
| Soft overcomes hard | `WILLNEED` / page cache over panic `pread` |
| Yield and become whole | ENERGY uses only **free** VRAM; if full, it bows |
| Do not force | `DRAFT=0` under TAO; no MTP thrash on cold snow |
| Return is the movement of the Tao | habit (`.coli_traj`) digs channels; next turn follows them |

See also: [`corriente-peng.md`](corriente-peng.md) · [`diagrams/corriente-peng.svg`](diagrams/corriente-peng.svg)

---

## 2. Wu wei — non-forcing action

**Wu wei** is not laziness. It is action that **does not fight the medium**.

| Forced (against the Tao) | Wu wei (with the Tao) |
|--------------------------|------------------------|
| `DRAFT=2` on cold disk | draft only when the bed is warm |
| Thousands of `fadvise`/token | TRAJ budget + bitmaps + pathpack radius |
| REPIN every few tokens | softer REPIN under `TAO=1` (48) |
| ENERGY when VRAM is full | log “nearly full” and rest |
| TOPK=6 “to go faster” | full top-k; let SPEED trim only nucleus |
| Model on `/mnt/c` | model on ext4 — water prefers a clean bed |

The math of MiMo stays **bit-exact**.  
The Tao only governs **when mass moves** and **where light is allowed to sit**.

---

## 3. Yin and yang in the valley

```
        yin                          yang
   snow (NVMe)                  light (VRAM)
   SWA ring (near)              full attn (far)
   potential                    kinetic (moe_acc)
   habit file                   live residual
   silence between tokens       named word
```

Neither side is “better.”  
The Tao is the **circulation** between them:

```
nieve ──thaw──► caché ──ENERGY──► VRAM ──shear──► residual ──mouth──► token
  ▲                                                                      │
  └──────────────────── rain (next pressure) ────────────────────────────┘
```

---

## 4. TAO=1 (executable stillness)

```bash
TAO=1 SERVE=1 SNAP=~/mimo25_i4 ./mimo 64 4 8
# or
TAO=1 scripts/start_peng.sh chat
```

| Knob | Under TAO (if unset) |
|------|----------------------|
| `SPEED` | 1 |
| `TRAJ` / `FLOW` | 1 |
| `ENERGY` | auto (−1) with FLOW |
| `DRAFT` | **0** (no force) |
| `REPIN` | 48 (softer) |
| `TOPP` | 0.55 via SPEED |

Explicit env always wins. The Tao does not bind the hand that already chose.

Log:

```
[TAO] wu wei: flow with residual · thaw/ignite without force · DRAFT stays off
```

---

## 5. Verse for operators

```
Do not push the token.
Prepare the bed.
Let the residual find the mouth.

When VRAM is full, do not thrash —
the light is already there.

When the disk is cold, do not draft —
wait for the thaw.

The bird Peng does not flap.
It rides the whirlwind
the river itself lifts.
```

---

## 6. Law (unchanged)

1. Residual is the subject.  
2. I/O may anticipate; logits must not lie.  
3. Habit digs the bed.  
4. Form is provisional.  
5. **Do not force what the medium refuses.**

---

*peng-mimo · docs/tao.md · companion to corriente-peng.md*
