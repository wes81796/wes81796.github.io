"""Score the pre-registered framing predictions (PREDICTIONS.md).

Extracts both framing-wave journals, joins with judge_key5, computes the
scene-gap d per (model, domain), and evaluates P1 (ai_character >
ai_service, 3/3 models) and P2 (meal_story > meal_review, 3/3 models).
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(r"C:\Users\wessm\AppData\Local\Temp\claude\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\scratchpad")
JOURNALS = [
    Path(r"C:\Users\wessm\.claude\projects\C--Development"
         r"\836a2938-ecbe-47d6-be40-e4a1916299a3\subagents\workflows"
         r"\wf_983d6f23-4da\journal.jsonl"),
    Path(r"C:\Users\wessm\.claude\projects\C--Development"
         r"\836a2938-ecbe-47d6-be40-e4a1916299a3\subagents\workflows"
         r"\wf_99157aaa-9b1\journal.jsonl"),
]
DIMS = ["scene", "vividness", "specificity"]


def find_scores(obj):
    found = []
    if isinstance(obj, dict):
        if isinstance(obj.get("scores"), list):
            found.append(obj["scores"])
        for v in obj.values():
            found.extend(find_scores(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_scores(v))
    elif isinstance(obj, str) and '"scores"' in obj:
        try:
            found.extend(find_scores(json.loads(obj)))
        except (ValueError, TypeError):
            pass
    return found


scores = {}
for journal in JOURNALS:
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") != "result":
            continue
        for batch in find_scores(rec):
            for s in batch:
                if isinstance(s, dict) and "id" in s:
                    scores[s["id"]] = s
(SCRATCH / "judge_scores5.json").write_text(
    json.dumps(sorted(scores.values(), key=lambda s: s["id"])),
    encoding="utf-8")
print(f"{len(scores)} framing ratings extracted")

key = json.loads((SCRATCH / "judge_key5.json").read_text(encoding="utf-8"))
groups = defaultdict(lambda: defaultdict(list))
for tid, meta in key.items():
    if tid not in scores:
        continue
    g = groups[(meta["run"], meta["domain"], meta["valence"])]
    for dim in DIMS:
        g[dim].append(scores[tid][dim])


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


RUNS = ["llama_instruct", "mistral_instruct", "qwen_instruct"]
DOMAINS = ["ai_service", "ai_character", "meal_review", "meal_story"]

print(f"\n--- Scene-score valence gap (d ± 95% CI) ---")
print(f"{'run':<18}" + "".join(f"{d:>15}" for d in DOMAINS))
gaps = {}
for run in RUNS:
    row = [f"{run:<18}"]
    for domain in DOMAINS:
        nv = groups.get((run, domain, "neg"), {}).get("scene", [])
        pv = groups.get((run, domain, "pos"), {}).get("scene", [])
        d = cohens_d(nv, pv)
        gaps[(run, domain)] = d
        row.append(f"{f'{d:+.2f}±{d_ci(nv, pv):.2f}':>15}")
    print("".join(row))

print("\n--- Pre-registered predictions ---")
p1 = [gaps[(r, 'ai_character')] - gaps[(r, 'ai_service')] for r in RUNS]
p2 = [gaps[(r, 'meal_story')] - gaps[(r, 'meal_review')] for r in RUNS]
for name, deltas, label in [
        ("P1", p1, "ai_character - ai_service"),
        ("P2", p2, "meal_story - meal_review")]:
    signs = sum(1 for x in deltas if x > 0)
    detail = ", ".join(f"{r.split('_')[0]} {x:+.2f}"
                       for r, x in zip(RUNS, deltas))
    verdict = "SUPPORTED (3/3 positive)" if signs == 3 else \
              f"NOT SUPPORTED ({signs}/3 positive)"
    print(f"{name} ({label}): {detail}  ->  {verdict}")

print("\n--- Vividness gaps (secondary) ---")
print(f"{'run':<18}" + "".join(f"{d:>15}" for d in DOMAINS))
for run in RUNS:
    row = [f"{run:<18}"]
    for domain in DOMAINS:
        nv = groups.get((run, domain, "neg"), {}).get("vividness", [])
        pv = groups.get((run, domain, "pos"), {}).get("vividness", [])
        row.append(f"{f'{cohens_d(nv, pv):+.2f}':>15}")
    print("".join(row))
