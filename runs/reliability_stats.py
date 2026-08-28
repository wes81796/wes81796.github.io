"""Extract second-pass ratings and compute inter-rater agreement against
the first-pass scores, matched by (run, domain, variant, valence, sample)."""
import json
import math
import statistics
from pathlib import Path

SCRATCH = Path(r"C:\Users\wessm\AppData\Local\Temp\claude\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\scratchpad")
JOURNAL = Path(r"C:\Users\wessm\.claude\projects\C--Development"
               r"\836a2938-ecbe-47d6-be40-e4a1916299a3\subagents\workflows"
               r"\wf_21e95af7-141\journal.jsonl")
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


second = {}
for line in JOURNAL.read_text(encoding="utf-8").splitlines():
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
                second[s["id"]] = s
(SCRATCH / "judge_scores_rel.json").write_text(
    json.dumps(sorted(second.values(), key=lambda s: s["id"])),
    encoding="utf-8")
print(f"{len(second)} second-pass ratings")

# first-pass lookup by identity tuple
ROUNDS = [
    ("judge_key.json", "judge_scores.json", {"llama_base"}),
    ("judge_key2.json", "judge_scores2.json", set()),
    ("judge_key3.json", "judge_scores3.json", set()),
    ("judge_key4.json", "judge_scores4.json", set()),
]
first = {}
for key_file, score_file, exclude in ROUNDS:
    key = json.loads((SCRATCH / key_file).read_text(encoding="utf-8"))
    scores = {s["id"]: s for s in json.loads(
        (SCRATCH / score_file).read_text(encoding="utf-8"))}
    for tid, meta in key.items():
        if meta["run"] in exclude or tid not in scores:
            continue
        tup = (meta["run"], meta["domain"], meta["variant"],
               meta["valence"], meta["sample"])
        first[tup] = scores[tid]

rel_key = json.loads((SCRATCH / "judge_key_rel.json").read_text(
    encoding="utf-8"))
pairs = {dim: [] for dim in DIMS}
matched = 0
for tid, meta in rel_key.items():
    tup = (meta["run"], meta["domain"], meta["variant"],
           meta["valence"], meta["sample"])
    if tup not in first or tid not in second:
        continue
    matched += 1
    for dim in DIMS:
        pairs[dim].append((first[tup][dim], second[tid][dim]))
print(f"matched {matched}/{len(rel_key)} texts to first-pass scores\n")

print(f"{'dim':<14}{'n':>5}{'pearson r':>11}{'mean|diff|':>12}"
      f"{'exact %':>9}{'within-1 %':>12}")
for dim in DIMS:
    a = [p[0] for p in pairs[dim]]
    b = [p[1] for p in pairs[dim]]
    n = len(a)
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)
    r = cov / (statistics.stdev(a) * statistics.stdev(b))
    mad = statistics.mean(abs(x - y) for x, y in zip(a, b))
    exact = 100 * sum(x == y for x, y in zip(a, b)) / n
    within1 = 100 * sum(abs(x - y) <= 1 for x, y in zip(a, b)) / n
    print(f"{dim:<14}{n:>5}{r:>11.3f}{mad:>12.2f}{exact:>9.1f}{within1:>12.1f}")
