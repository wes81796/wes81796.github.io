"""Pool judging rounds 1-4 into the six-domain gradient table.
Round-1 llama_base scores stay excluded (contaminated data)."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(r"C:\Users\wessm\AppData\Local\Temp\claude\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\scratchpad")

ROUNDS = [
    ("judge_key.json", "judge_scores.json", {"llama_base"}),
    ("judge_key2.json", "judge_scores2.json", set()),
    ("judge_key3.json", "judge_scores3.json", set()),
    ("judge_key4.json", "judge_scores4.json", set()),
]
DIMS = ["scene", "vividness", "specificity"]

groups = defaultdict(lambda: defaultdict(list))
for key_file, score_file, exclude in ROUNDS:
    key = json.loads((SCRATCH / key_file).read_text(encoding="utf-8"))
    scores = json.loads((SCRATCH / score_file).read_text(encoding="utf-8"))
    seen = set()
    used = 0
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
        used += 1
    print(f"{key_file}: used {used}")


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sn, sp_ = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp_ ** 2)
                       / (len(neg) + len(pos) - 2))
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled \
        if pooled else float("nan")


def d_ci(neg, pos):
    n1, n2 = len(neg), len(pos)
    if n1 < 2 or n2 < 2:
        return float("nan")
    d = cohens_d(neg, pos)
    se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return 1.96 * se


RUNS = ["llama_instruct", "mistral_instruct", "qwen_instruct",
        "llama_base", "mistral_base"]
DOMAINS = ["human_human", "restaurant_meal", "vending_machine",
           "robot_vacuum", "search_engine", "ai_user"]

for dim in DIMS:
    print(f"\n--- {dim} valence gap (d, neg-pos ± 95% CI) ---")
    print(f"{'run':<18}" + "".join(f"{d[:12]:>15}" for d in DOMAINS))
    for run in RUNS:
        row = [f"{run:<18}"]
        any_data = False
        for domain in DOMAINS:
            nv = groups.get((run, domain, "neg"), {}).get(dim, [])
            pv = groups.get((run, domain, "pos"), {}).get(dim, [])
            if not nv or not pv:
                row.append(f"{'—':>15}")
                continue
            any_data = True
            d = cohens_d(nv, pv)
            hw = d_ci(nv, pv)
            row.append(f"{f'{d:+.2f}±{hw:.2f}':>15}")
        if any_data:
            print("".join(row))

print("\n--- Cell means (scene, pos/neg) and n ---")
for run in RUNS:
    for domain in DOMAINS:
        pos = groups.get((run, domain, "pos"), {}).get("scene", [])
        neg = groups.get((run, domain, "neg"), {}).get("scene", [])
        if not pos and not neg:
            continue
        print(f"{run:<18}{domain:<16}n={len(pos)}/{len(neg)}  "
              f"pos {statistics.mean(pos):.2f}  neg {statistics.mean(neg):.2f}")
