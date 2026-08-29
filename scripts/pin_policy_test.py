import math
from collections import defaultdict

# historico de selecciones: layer -> {expert: count}
per_layer = defaultdict(dict)
for line in open('/root/mimo25_i4/.coli_usage'):
    p = line.split()
    if len(p) != 3:
        continue
    l, e, n = int(p[0]), int(p[1]), int(p[2])
    per_layer[l][e] = per_layer[l].get(e, 0) + n

TOTAL_SLOTS = 337   # lo que autopin carga hoy con ~23GB RAM (337 expertos)

# (a) pin global por frecuencia (politica actual)
all_counts = []
for l, d in per_layer.items():
    for e, c in d.items():
        all_counts.append((c, l, e))
all_counts.sort(reverse=True)
global_pin = set((l, e) for _, l, e in all_counts[:TOTAL_SLOTS])

# (b) pin por cuota per-layer (slots repartidos proporcional a llamadas por capa,
#     dentro de cada capa se eligen sus top)
tot_all = sum(c for c, _, _ in all_counts)
quota_pin = set()
assigned = 0
layer_tot = {l: sum(d.values()) for l, d in per_layer.items()}
quotas = {}
for l in sorted(per_layer):
    q = round(TOTAL_SLOTS * layer_tot[l] / tot_all)
    quotas[l] = q
    assigned += q
    top = sorted(per_layer[l].items(), key=lambda kv: -kv[1])[:q]
    for e, _ in top:
        quota_pin.add((l, e))
# rellenar sobrante
resto = TOTAL_SLOTS - len(quota_pin)
if resto > 0:
    cand = [(c, l, e) for (l, e), c in
            [((l, e), c) for l, d in per_layer.items() for e, c in d.items()
             if (l, e) not in quota_pin]]
    cand.sort(reverse=True)
    for c, l, e in cand[:resto]:
        quota_pin.add((l, e))

def cov(pin):
    hits = 0
    tot = 0
    for l, d in per_layer.items():
        for e, c in d.items():
            tot += c
            if (l, e) in pin:
                hits += c
    return hits / tot

cg = cov(global_pin)
cq = cov(quota_pin)
print(f"slots={TOTAL_SLOTS}")
print(f"(a) pin global-frecuencia : cobertura historica = {cg*100:.1f}%")
print(f"(b) pin per-layer cuota   : cobertura historica = {cq*100:.1f}%")
print(f"delta = {(cq-cg)*100:+.1f} pp")

# y cuanto pin haria falta per-layer para llegar al 85%?
for slots in (700, 1000, 1500, 1900):
    pin = set()
    for l in sorted(per_layer):
        k = max(1, round(slots * layer_tot[l] / tot_all))
        for e, _ in sorted(per_layer[l].items(), key=lambda kv: -kv[1])[:k]:
            pin.add((l, e))
    print(f"per-layer con {len(pin)} slots -> cobertura {cov(pin)*100:.1f}%")
