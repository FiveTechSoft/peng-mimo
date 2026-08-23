import math
from collections import defaultdict

per_layer = defaultdict(dict)
tot_by_layer = defaultdict(int)
for line in open('/root/mimo25_i4/.coli_usage'):
    p = line.split()
    if len(p) != 3:
        continue
    l, e, n = int(p[0]), int(p[1]), int(p[2])
    per_layer[l][e] = per_layer[l].get(e, 0) + n
    tot_by_layer[l] += n

print("layer | total | top40%cov | top16%cov | Hnorm | gini")
summary = []
for l in sorted(per_layer):
    d = per_layer[l]
    tot = tot_by_layer[l]
    counts = sorted(d.values(), reverse=True)
    E = len(counts)
    k40 = max(1, round(E * 0.4))
    k16 = max(1, round(E * 0.16))
    cov40 = sum(counts[:k40]) / tot
    cov16 = sum(counts[:k16]) / tot
    H = -sum((c / tot) * math.log(c / tot) for c in counts)
    Hn = H / math.log(max(2, E)) if E > 1 else 0.0
    n_ = len(counts)
    cum = sum((2 * i - n_ - 1) * c for i, c in enumerate(counts, 1))
    gini = cum / (n_ * sum(counts)) if sum(counts) else 0
    summary.append((l, tot, cov40, cov16, Hn, gini))
    print(f"{l:5d} | {tot:5d} | {cov40:9.3f} | {cov16:9.3f} | {Hn:5.3f} | {gini:5.3f}")

early = [r for r in summary if 1 <= r[0] <= 8]
mid = [r for r in summary if 20 <= r[0] <= 27]
late = [r for r in summary if r[0] >= 40]
def avg(rs, i): return sum(r[i] for r in rs) / len(rs) if rs else float('nan')
print()
print(f"EARLY (L1-8)   : top40cov={avg(early,2):.3f} top16cov={avg(early,3):.3f} Hn={avg(early,4):.3f}")
print(f"MID   (L20-27) : top40cov={avg(mid,2):.3f} top16cov={mid and avg(mid,3):.3f} Hn={avg(mid,4):.3f}")
print(f"LATE  (L40-47) : top40cov={avg(late,2):.3f} top16cov={avg(late,3):.3f} Hn={avg(late,4):.3f}")
