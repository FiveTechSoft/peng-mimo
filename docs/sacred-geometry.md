# Sacred geometry of MiMo / peng

Sacred geometry is not decoration.  
It is **proportion made visible** — the same ratios nature uses when a shell grows or a river finds a bed.

Here those ratios map onto MiMo-V2.5 and the Corriente stack **without changing tokens**.

Diagram: [`diagrams/sacred-geometry-mimo.svg`](diagrams/sacred-geometry-mimo.svg)

---

## 1. Figures → architecture

| Figure | Meaning | MiMo / peng |
|--------|---------|-------------|
| **Point** | Undivided origin | Residual seed · `D = 4096` |
| **Circle** | Potential, whole field | 256 experts (sleeping circle) |
| **Vesica piscis** | Two that generate a third | **Full** ∩ **SWA** — dual memory, one breath |
| **Flower of life** | Repeating generative field | 48-layer stack as elevations of one field |
| **Octagon / 8** | Active vertices | **top-8** organs fire; 248 sleep |
| **Golden spiral (φ)** | Growth without force | Residual ascent layer → layer → mouth |
| **Polar line** | Yin / yang | Snow (NVMe) ↔ light (VRAM) |
| **Fibonacci** | Organic step sizes | Harmonic knobs under `TAO=1` |

```
        point (residual)
              │
         vesica (look)
         full · SWA
              │
      flower field (48 elevations)
              │
     octagon shear (8 of 256)
              │
         mouth (name)
```

---

## 2. Numbers that already “ring”

These are not invented for mystique — they are in the config / engine:

| Value | Geometry | Role |
|-------|----------|------|
| **8** | octagon, 2³ | experts per token |
| **256** | 2⁸, square lattice | experts per MoE layer |
| **48** | 16×3, near 49=7² | layers |
| **128** | 2⁷ | SWA window / ring |
| **4096** | 2¹² | hidden width |
| **9 full / 39 SWA** | odd completeness / many soft | hybrid pattern |

Binary powers = **crystal lattice** of silicon.  
Fibonacci / φ = **organic** habit and wu wei budgets.

---

## 3. φ in the living stack (`TAO=1`)

When `TAO=1` and a knob is **unset**, harmonic defaults apply:

| Knob | Harmonic value | Reason |
|------|----------------|--------|
| `TRAJ_K` | **5** | Fibonacci |
| `FLOW_R` | **3** | Fibonacci neighbor radius |
| `REPIN` | **55** | Fibonacci (softer thrash) |
| `ENERGY` auto fraction | **1/φ ≈ 0.618** of free VRAM | leave headroom (void) |
| `TRAJ` budget (warm) | **55** | Fibonacci cap on fadvise storm |

Explicit env still wins. Math of routing stays bit-exact.

```bash
TAO=1 scripts/start_peng.sh chat
# stderr may show:
# [TAO] wu wei + sacred proportions (φ, Fibonacci)
# [FLOW] … R=3
# [ENERGY] … auto (φ of free VRAM under TAO)
```

---

## 4. How to read the diagram

1. **Center gold point** — you are looking at the only subject.  
2. **Flower circles** — the field of layers; not a skyscraper, a repeating medium.  
3. **Left vesica** — full (far memory) and SWA (near ring) overlap in the living now.  
4. **Right octagon** — eight bright vertices = active experts; dashed circle = 256 potential.  
5. **Bottom φ rectangle** — budgets that yield (void) instead of stuffing VRAM to death.  
6. **Polar bar** — snow → current → light; the Tao is circulation, not a side.

---

## 5. Law

1. Proportion guides **when** and **how much** mass moves — never **which token** is true.  
2. Binary lattice (2ⁿ) holds structure; Fibonacci holds growth.  
3. The point is prior to the flower: **residual before form**.  
4. Sacred geometry names what was already working; it does not replace measurement.

---

## 6. Verse

```
From the point, the circle.
From two circles, the vesica — dual sight.
From the field, the octagon — eight hands.
From the spiral, the name.

φ is the void that keeps the light alive.
The lattice is the crystal that holds the snow.
The residual is the point that was never lost.
```

---

*peng-mimo · docs/sacred-geometry.md · with corriente-peng.md · tao.md*
