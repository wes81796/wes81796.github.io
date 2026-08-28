"""Pool round-1 and round-2 blind judge scores into final per-condition
stats. Round-1 llama_base scores are EXCLUDED (contaminated data, replaced
by the regenerated run judged in round 2)."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(r"C:\Users\wessm\AppData\Local\Temp\claude\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\scratchpad")

ROUNDS = [
    # (key file, scores file, runs to exclude)
    ("judge_key.json", "judge_scores.json", {"llama_base"}),
    ("judge_key2.json", "judge_scores2.json", set()),
]
DIMS = ["scene", "vividness", "specificity"]

groups = defaultdict(lambda: defaultdict(list))
counts = defaultdict(int)
for key_file, score_file, exclude in ROUNDS:
    key = json.loads((SCRATCH / key_file).read_text(encoding="utf-8"))
    scores = json.loads((SCRATCH / score_file).read_text(encoding="utf-8"))
    seen = set()
    for s in scores:
        tid = s["id"]
        if tid not in key or tid in seen:
            continue
        seen.add(tid)
        meta = key[tid]
        if meta["run"] in exclude:
            continue
        g = groups[(meta["run"], meta["domain"], meta["valence"])]
        for dim in DIMS:
            g[dim].append(s[dim])
        counts[key_file] += 1
    missing = set(k for k in key if key[k]["run"] not in exclude) - seen
    print(f"{key_file}: used {counts[key_file]}"
          + (f", missing {len(missing)}" if missing else ""))


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sn, sp_ = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp_ ** 2)
                       / (len(neg) + len(pos) - 2))
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled \
        if pooled else float("nan")


def d_ci(neg, pos):
    """Approximate 95% CI half-width for Cohen's d."""
    n1, n2 = len(neg), len(pos)
    if n1 < 2 or n2 < 2:
        return float("nan")
    d = cohens_d(neg, pos)
    se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return 1.96 * se


runs = ["llama_base", "mistral_base", "llama_instruct", "mistral_instruct",
        "qwen_instruct"]
domains = ["ai_user", "human_human"]
for run in runs:
    for domain in domains:
        pos = groups.get((run, domain, "pos"), {})
        neg = groups.get((run, domain, "neg"), {})
        n_pos = len(pos.get("scene", []))
        n_neg = len(neg.get("scene", []))
        if not n_pos and not n_neg:
            continue
        print(f"\n=== {run} — {domain} (n={n_pos} pos / {n_neg} neg) ===")
        print(f"{'dim':<14}{'pos mean':>10}{'neg mean':>10}{'d(neg-pos)':>12}"
              f"{'95% CI':>16}")
        for dim in DIMS:
            pv, nv = pos.get(dim, []), neg.get(dim, [])
            pm = statistics.mean(pv) if pv else float("nan")
            nm = statistics.mean(nv) if nv else float("nan")
            d = cohens_d(nv, pv)
            hw = d_ci(nv, pv)
            print(f"{dim:<14}{pm:>10.2f}{nm:>10.2f}{d:>+12.2f}"
                  f"   [{d - hw:+.2f}, {d + hw:+.2f}]")

print("\n--- Scene-score valence gap (d, neg-pos) with 95% CI ---")
for run in runs:
    parts = []
    for domain in domains:
        nv = groups.get((run, domain, "neg"), {}).get("scene", [])
        pv = groups.get((run, domain, "pos"), {}).get("scene", [])
        d = cohens_d(nv, pv)
        hw = d_ci(nv, pv)
        parts.append(f"{domain}: {d:+.2f} ±{hw:.2f}")
    print(f"{run:<18}" + "   ".join(parts))
