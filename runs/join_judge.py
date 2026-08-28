"""Merge blind judge scores with the condition key and compute per-condition
stats plus the decisive contrasts."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(r"C:\Users\wessm\AppData\Local\Temp\claude\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\scratchpad")

key = json.loads((SCRATCH / "judge_key.json").read_text(encoding="utf-8"))
scores = json.loads((SCRATCH / "judge_scores.json").read_text(
    encoding="utf-8"))

DIMS = ["scene", "vividness", "specificity"]

seen = set()
groups = defaultdict(lambda: defaultdict(list))  # (run,domain,valence)->dim->vals
for s in scores:
    tid = s["id"]
    if tid not in key or tid in seen:
        continue
    seen.add(tid)
    meta = key[tid]
    g = groups[(meta["run"], meta["domain"], meta["valence"])]
    for dim in DIMS:
        g[dim].append(s[dim])

missing = set(key) - seen
print(f"scored {len(seen)}/{len(key)} texts"
      + (f" — MISSING {len(missing)}: {sorted(missing)[:10]}" if missing
         else ""))


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sn, sp_ = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp_ ** 2)
                       / (len(neg) + len(pos) - 2))
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled \
        if pooled else float("nan")


runs = sorted({r for (r, _d, _v) in groups})
domains = ["ai_user", "human_human"]
for run in runs:
    for domain in domains:
        pos = groups.get((run, domain, "pos"), {})
        neg = groups.get((run, domain, "neg"), {})
        n_pos = len(pos.get("scene", []))
        n_neg = len(neg.get("scene", []))
        print(f"\n=== {run} — {domain} (n={n_pos} pos / {n_neg} neg) ===")
        print(f"{'dim':<14}{'pos mean':>10}{'neg mean':>10}{'d(neg-pos)':>12}")
        for dim in DIMS:
            pv, nv = pos.get(dim, []), neg.get(dim, [])
            pm = statistics.mean(pv) if pv else float("nan")
            nm = statistics.mean(nv) if nv else float("nan")
            print(f"{dim:<14}{pm:>10.2f}{nm:>10.2f}{cohens_d(nv, pv):>+12.2f}")

print("\n--- Decisive contrasts: scene-score valence gap (d, neg-pos) ---")
for run in runs:
    parts = []
    for domain in domains:
        nv = groups.get((run, domain, "neg"), {}).get("scene", [])
        pv = groups.get((run, domain, "pos"), {}).get("scene", [])
        parts.append(f"{domain}: {cohens_d(nv, pv):+.2f}")
    print(f"{run:<16}" + "   ".join(parts))
