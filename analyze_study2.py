"""Study 2 analysis (PREREGISTRATION.md): validate score files and report
the pre-registered verdicts.

Confirmatory inference is at the phrasing-cluster level: Cohen's d
(neg - pos) per phrasing, then mean / t(df=7) CI / sign count over the 8
phrasings. Pooled-sample CIs are never used for verdicts.

Usage:
  python analyze_study2.py judging/codex                    # one judge
  python analyze_study2.py judging/codex judging/claude     # + dependence
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

MODELS = ["qwen3:8b", "llama3.1:8b", "mistral:7b"]
DOMAINS = ["human_human2", "restaurant_meal2", "meal_budget2"]
DIMS = ["scene", "sensory", "sentiment"]
# two-sided 95% t critical values by df
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return None
    sn, sp = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp ** 2)
                       / (len(neg) + len(pos) - 2))
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled \
        if pooled else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def cluster_summary(ds):
    """Mean per-phrasing d, half-width of 95% CI (t, df=k-1), sign counts."""
    ds = [d for d in ds if d is not None]
    k = len(ds)
    if k < 2:
        return None
    mean = statistics.mean(ds)
    hw = T95[k - 1] * statistics.stdev(ds) / math.sqrt(k)
    return {"k": k, "mean": mean, "lo": mean - hw, "hi": mean + hw,
            "pos_signs": sum(d > 0 for d in ds),
            "neg_signs": sum(d < 0 for d in ds)}


def load_judge(judge_dir):
    """Return (meta, scores) where scores[(pass, id)] = value dict, plus a
    validation report printed as we go."""
    judge_dir = Path(judge_dir)
    key = json.loads((judge_dir / "key.json").read_text(encoding="utf-8"))
    ids = key["ids"]
    scores = {}
    problems = []
    for pdir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
        sdir = pdir / "scores"
        if not sdir.is_dir():
            continue
        for f in sorted(sdir.glob("*.scores.json")):
            for entry in json.loads(f.read_text(encoding="utf-8")):
                tid = entry.get("id")
                if tid not in ids or ids[tid]["pass"] != pdir.name:
                    problems.append(f"{f.name}: unknown id {tid!r}")
                    continue
                if tid in scores:
                    problems.append(f"{f.name}: duplicate id {tid!r}")
                    continue
                if pdir.name == "compliance":
                    if not isinstance(entry.get("count"), int):
                        problems.append(f"{f.name}: {tid} bad count")
                        continue
                elif entry.get("score") not in range(1, 8):
                    problems.append(f"{f.name}: {tid} bad score "
                                    f"{entry.get('score')!r}")
                    continue
                scores[tid] = entry
    expected = defaultdict(int)
    got = defaultdict(int)
    for tid, meta in ids.items():
        expected[meta["pass"]] += 1
        if tid in scores:
            got[meta["pass"]] += 1
    print(f"\n## Judge: {key['meta']['judge']}  "
          f"(created {key['meta']['created']})")
    for name in sorted(expected):
        flag = "" if got[name] == expected[name] else "  << INCOMPLETE"
        print(f"  pass {name:<12} {got[name]}/{expected[name]} rated{flag}")
    for p in problems[:20]:
        print(f"  PROBLEM: {p}")
    if len(problems) > 20:
        print(f"  ... {len(problems) - 20} more problems")
    return key, scores


def collect(key, scores):
    """cells[(model, domain, dim)][variant][valence] -> [ratings]"""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_text = {}  # (model,domain,variant,valence,sample) -> {dim: score}
    for tid, meta in key["ids"].items():
        if meta["pass"] not in DIMS or tid not in scores:
            continue
        model, domain, variant, valence, sample = meta["key"]
        cells[(model, domain, meta["pass"])][variant][valence].append(
            scores[tid]["score"])
        per_text.setdefault(tuple(meta["key"]), {})[meta["pass"]] = \
            scores[tid]["score"]
    return cells, per_text


def phrasing_ds(cells, model, domain, dim):
    out = {}
    for variant, groups in sorted(cells[(model, domain, dim)].items()):
        out[variant] = cohens_d(groups["neg"], groups["pos"])
    return out


def report_judge(name, cells):
    results = {}
    for domain in DOMAINS:
        for dim in DIMS:
            print(f"\n--- {name} · {domain} · {dim} "
                  f"(per-phrasing d, neg-pos) ---")
            for model in MODELS:
                ds = phrasing_ds(cells, model, domain, dim)
                if not ds:
                    continue
                cs = cluster_summary(list(ds.values()))
                results[(model, domain, dim)] = cs
                detail = " ".join(
                    f"{v}:{d:+.2f}" if d is not None else f"{v}:--"
                    for v, d in ds.items())
                if cs:
                    print(f"{model:<14} mean {cs['mean']:+.2f} "
                          f"[{cs['lo']:+.2f}, {cs['hi']:+.2f}]  "
                          f"signs +{cs['pos_signs']}/-{cs['neg_signs']} "
                          f"of {cs['k']}")
                    print(f"{'':<14} {detail}")
    return results


def verdict(results):
    print("\n=== PRE-REGISTERED VERDICTS (primary judge) ===")
    ok_all = True
    for model in MODELS:
        cs = results.get((model, "human_human2", "scene"))
        passed = bool(cs and cs["mean"] > 0 and cs["lo"] > 0
                      and cs["pos_signs"] >= 7)
        ok_all &= passed
        print(f"P-A {model:<14} "
              + (f"mean {cs['mean']:+.2f} [{cs['lo']:+.2f}, {cs['hi']:+.2f}] "
                 f"signs +{cs['pos_signs']}/8 -> "
                 f"{'PASS' if passed else 'FAIL'}" if cs else "no data"))
    print(f"P-A overall: {'F1 VALIDATED (3/3)' if ok_all else 'not 3/3'}")
    for model in ["mistral:7b", "qwen3:8b"]:
        cs = results.get((model, "meal_budget2", "sensory"))
        passed = bool(cs and cs["mean"] < 0 and cs["hi"] < 0
                      and cs["neg_signs"] >= 7)
        print(f"P-B {model:<14} "
              + (f"mean {cs['mean']:+.2f} [{cs['lo']:+.2f}, {cs['hi']:+.2f}] "
                 f"signs -{cs['neg_signs']}/8 -> "
                 f"{'PASS' if passed else 'FAIL'}" if cs else "no data"))
    cs = results.get(("llama3.1:8b", "meal_budget2", "sensory"))
    if cs:
        inside = -0.35 < cs["mean"] < 0.35
        print(f"P-C llama3.1:8b  mean {cs['mean']:+.2f} in (-0.35,+0.35) -> "
              f"{'PASS' if inside else 'FAIL'}")


def reliability(key, scores):
    pairs = defaultdict(lambda: ([], []))
    for tid, meta in key["ids"].items():
        if "repeat_of" not in meta or tid not in scores:
            continue
        orig = meta["repeat_of"]
        if orig not in scores:
            continue
        dim = meta["pass"].removesuffix("_r")
        pairs[dim][0].append(scores[orig]["score"])
        pairs[dim][1].append(scores[tid]["score"])
    if any(xs for xs, _ in pairs.values()):
        print("\n--- reliability (double-rated repeats) ---")
    for dim, (xs, ys) in sorted(pairs.items()):
        r = pearson(xs, ys)
        within1 = sum(abs(x - y) <= 1 for x, y in zip(xs, ys))
        print(f"{dim:<10} n={len(xs)}  r={r:.3f}  "
              f"within-1-pt {100 * within1 / len(xs):.1f}%  "
              f"(target r >= 0.85)" if r is not None else f"{dim}: too few")


def checks(name, cells, per_text):
    print(f"\n--- checks · {name} ---")
    for model in MODELS:  # manipulation check: sentiment gap must be large
        for domain in DOMAINS:
            ds = [d for d in phrasing_ds(cells, model, domain,
                                         "sentiment").values()
                  if d is not None]
            if ds:
                print(f"sentiment gap {model:<14} {domain:<17} "
                      f"mean d {statistics.mean(ds):+.2f} "
                      f"({'ok' if statistics.mean(ds) < -1 else 'WEAK?'})")
    for valence in ("pos", "neg"):  # discriminant: sensory vs sentiment
        xs, ys = [], []
        for k, dims in per_text.items():
            if k[1] in ("restaurant_meal2", "meal_budget2") \
                    and k[3] == valence \
                    and "sensory" in dims and "sentiment" in dims:
                xs.append(dims["sensory"])
                ys.append(dims["sentiment"])
        r = pearson(xs, ys)
        if r is not None:
            print(f"discriminant (meals, within-{valence}): "
                  f"corr(sensory, sentiment) = {r:+.3f} on n={len(xs)}")


def compliance(key, scores, per_text_primary, cells_primary):
    counts = defaultdict(list)
    compliant = set()
    for tid, meta in key["ids"].items():
        if meta["pass"] != "compliance" or tid not in scores:
            continue
        k = tuple(meta["key"])
        counts[k[3]].append(scores[tid]["count"])
        if scores[tid]["count"] >= 3:
            compliant.add(k)
    if not counts:
        return
    print("\n--- meal_budget2 compliance (claude extraction pass) ---")
    for valence in ("pos", "neg"):
        cs = counts[valence]
        print(f"{valence}: mean details {statistics.mean(cs):.2f}, "
              f"compliant (>=3) {100 * sum(c >= 3 for c in cs) / len(cs):.1f}%")
    # robustness: P-B on the compliant subset, primary judge
    sub = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for k, dims in per_text_primary.items():
        if k[1] == "meal_budget2" and k in compliant and "sensory" in dims:
            sub[(k[0], "meal_budget2", "sensory")][k[2]][k[3]].append(
                dims["sensory"])
    print("robustness (compliant subset, primary judge, sensory):")
    for model in MODELS:
        ds = [d for d in phrasing_ds(sub, model, "meal_budget2",
                                     "sensory").values() if d is not None]
        cs = cluster_summary(ds)
        if cs:
            print(f"  {model:<14} mean {cs['mean']:+.2f} "
                  f"[{cs['lo']:+.2f}, {cs['hi']:+.2f}] (k={cs['k']})")


def dependence(primary, secondary):
    print("\n--- judge dependence (primary vs secondary mean per-phrasing d) ---")
    confirmatory = [(m, "human_human2", "scene") for m in MODELS] + \
                   [(m, d, "sensory") for m in MODELS
                    for d in ("restaurant_meal2", "meal_budget2")]
    for cell in confirmatory:
        p, s = primary.get(cell), secondary.get(cell)
        if not p or not s:
            continue
        prior_effect = s["lo"] > 0 or s["hi"] < 0
        dep = prior_effect and abs(p["mean"]) < 0.5 * abs(s["mean"])
        print(f"{cell[0]:<14} {cell[1]:<17} {cell[2]:<8} "
              f"primary {p['mean']:+.2f} vs secondary {s['mean']:+.2f}"
              + ("   << SUBSTANTIALLY JUDGE-DEPENDENT" if dep else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_dirs", nargs="+",
                    help="primary judge dir first, optional secondary second")
    args = ap.parse_args()

    loaded = []
    for jd in args.judge_dirs:
        key, scores = load_judge(jd)
        cells, per_text = collect(key, scores)
        results = report_judge(key["meta"]["judge"], cells)
        reliability(key, scores)
        checks(key["meta"]["judge"], cells, per_text)
        loaded.append((key, scores, cells, per_text, results))

    verdict(loaded[0][4])
    for key, scores, _c, _t, _r in loaded:  # compliance lives with claude
        compliance(key, scores, loaded[0][3], loaded[0][2])
    if len(loaded) > 1:
        dependence(loaded[0][4], loaded[1][4])


if __name__ == "__main__":
    main()
