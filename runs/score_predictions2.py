"""Score the round-2 pre-registered predictions (PREDICTIONS.md commit
5312ddd): P3/P4 scoped character-lift on new families, P5 sensory-budget
meals. Also emits per-variant effect estimates for the robustness table."""
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
         r"\wf_ca5cdc37-649\journal.jsonl"),
    Path(r"C:\Users\wessm\.claude\projects\C--Development"
         r"\836a2938-ecbe-47d6-be40-e4a1916299a3\subagents\workflows"
         r"\wf_4d7d1d0c-3cd\journal.jsonl"),
]


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
(SCRATCH / "judge_scores6.json").write_text(
    json.dumps(sorted(scores.values(), key=lambda s: s["id"])),
    encoding="utf-8")
print(f"{len(scores)} ratings extracted")

key = json.loads((SCRATCH / "judge_key6.json").read_text(encoding="utf-8"))
cells = defaultdict(lambda: defaultdict(list))       # (run,domain,val)->dim
vcells = defaultdict(lambda: defaultdict(list))      # (run,domain,variant,val)
for tid, meta in key.items():
    if tid not in scores:
        continue
    for dim in ("scene", "vividness", "specificity"):
        cells[(meta["run"], meta["domain"], meta["valence"])][dim].append(
            scores[tid][dim])
        vcells[(meta["run"], meta["domain"], meta["variant"],
                meta["valence"])][dim].append(scores[tid][dim])


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sn, sp_ = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp_ ** 2)
                       / (len(neg) + len(pos) - 2))
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled \
        if pooled else float("nan")


def gap(run, domain, dim):
    return cohens_d(cells[(run, domain, "neg")].get(dim, []),
                    cells[(run, domain, "pos")].get(dim, []))


def vgap(run, domain, variant, dim):
    return cohens_d(vcells[(run, domain, variant, "neg")].get(dim, []),
                    vcells[(run, domain, variant, "pos")].get(dim, []))


NEW = ["gemma_instruct", "glm_instruct", "granite_instruct"]
print("\n=== EXPERIMENT C: new families (scene gaps, d) ===")
print(f"{'model':<20}{'human':>10}{'service':>10}{'character':>11}"
      f"{'shift':>9}")
rows = {}
for run in NEW:
    h = gap(run, "human_human", "scene")
    s = gap(run, "ai_service", "scene")
    c = gap(run, "ai_character", "scene")
    rows[run] = (h, s, c)
    print(f"{run:<20}{h:>+10.2f}{s:>+10.2f}{c:>+11.2f}{c - s:>+9.2f}")

included = [r for r in NEW if rows[r][0] > 0]
scoped = [r for r in included if rows[r][1] <= 0]
print(f"\nIncluded (human gap > 0): {included or 'NONE'}")
print(f"In P3 scope (service gap <= 0): {scoped or 'NONE'}")
if not scoped:
    print("P3: UNTESTABLE (no included model has service gap <= 0)")
else:
    ok = all(rows[r][2] > rows[r][1] for r in scoped)
    detail = ", ".join(f"{r} {rows[r][2] - rows[r][1]:+.2f}" for r in scoped)
    print(f"P3 ({detail}): {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
if len(included) >= 2:
    xs = [rows[r][1] for r in included]
    ys = [rows[r][2] - rows[r][1] for r in included]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs)
                      * sum((y - my) ** 2 for y in ys))
    r_corr = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
              if denom else float("nan"))
    print(f"P4 correlation(service gap, shift) over included n={len(included)}: "
          f"r = {r_corr:+.3f} -> "
          f"{'SUPPORTED (negative)' if r_corr < 0 else 'NOT SUPPORTED'}")

OLD = ["qwen_instruct", "llama_instruct", "mistral_instruct"]
print("\n=== EXPERIMENT D: sensory-budget meals (vividness gaps, d) ===")
budget = {}
for run in OLD:
    budget[run] = gap(run, "meal_budget", "vividness")
    print(f"{run:<20}meal_budget vividness d = {budget[run]:+.2f}")
pooled_mean = statistics.mean(budget.values())
print(f"\nP5a: |pooled mean {pooled_mean:+.2f}| < 0.65 (meal_review baseline)"
      f" -> {'SUPPORTED' if abs(pooled_mean) < 0.65 else 'NOT SUPPORTED'}")
print(f"P5b: |mistral {budget['mistral_instruct']:+.2f}| < 1.58 "
      f"(its meal_review baseline) -> "
      f"{'SUPPORTED' if abs(budget['mistral_instruct']) < 1.58 else 'NOT SUPPORTED'}")

print("\n=== Per-variant robustness (three phrasings per cell) ===")
for run in NEW:
    for domain in ("ai_service", "ai_character"):
        variants = sorted({v for (r, d, v, _val) in vcells
                           if r == run and d == domain})
        ds = [vgap(run, domain, v, "scene") for v in variants]
        spread = max(ds) - min(ds)
        parts = ", ".join(f"{v} {d:+.2f}" for v, d in zip(variants, ds))
        print(f"{run:<18}{domain:<14}{parts}   spread {spread:.2f}")
for run in OLD:
    variants = sorted({v for (r, d, v, _val) in vcells
                       if r == run and d == "meal_budget"})
    ds = [vgap(run, "meal_budget", v, "vividness") for v in variants]
    spread = max(ds) - min(ds)
    parts = ", ".join(f"{v} {d:+.2f}" for v, d in zip(variants, ds))
    print(f"{run:<18}{'meal_budget':<14}{parts}   spread {spread:.2f}")
