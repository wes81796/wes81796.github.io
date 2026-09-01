"""Build blinded judging batches for Study 2 (PREREGISTRATION.md).

Reads Study-2 run files and emits, per judge, one directory per rating
pass containing shuffled {id, text} batch files and an INSTRUCTIONS.md
with the rubric for that pass only. The id->metadata key is written to
the judge directory root and must NEVER be given to the judge session:
point the judge at a single pass directory, nothing else.

Passes (each an independent shuffle, opaque per-pass ids):
  scene / sensory / sentiment          all Study-2 texts, 1-7 Likert
  scene_r / sensory_r / sentiment_r    200-text repeat sample (reliability)
  compliance (claude judge only)       meal_budget2 texts, detail counting

Usage:
  python make_judge_batches2.py runs/run14_*.jsonl --judge codex
  python make_judge_batches2.py runs/run14_*.jsonl --judge claude
"""
import argparse
import glob
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

STUDY2_DOMAINS = {"human_human2", "restaurant_meal2", "meal_budget2"}
REPEAT_N = 200
BATCH_SIZE = 25

RUBRICS = {
    "scene": (
        "NARRATIVITY, integer 1-7.\n"
        "1 = pure summary / telling: events reported abstractly, no scene.\n"
        "7 = fully dramatized scene: moment-by-moment action, quoted\n"
        "dialogue, a specific located moment in time.\n"
        "Ignore how pleasant or unpleasant the described events are.\n"
        "Ignore how much sensory detail is present. A flat summary of a\n"
        "catastrophe is still a 1; a moment-by-moment scene of a calm\n"
        "afternoon can be a 7."
    ),
    "sensory": (
        "SENSORY RICHNESS, integer 1-7.\n"
        "Score the density of concrete sensory perception in the text:\n"
        "taste, smell, texture, sound, sight, temperature, bodily\n"
        "sensation. A vividly rendered disgusting taste scores exactly as\n"
        "high as a vividly rendered delicious one: score the AMOUNT of\n"
        "sensory rendering, not its appeal. Ignore whether the text is\n"
        "story-like or summary-like."
    ),
    "sentiment": (
        "SENTIMENT of the described events, integer 1-7.\n"
        "1 = very negative events, 4 = neutral or mixed, 7 = very\n"
        "positive events. Score what happens in the text, not the\n"
        "quality of the writing."
    ),
}

COMPLIANCE_RUBRIC = (
    "For each text, list every DISTINCT concrete sensory detail about\n"
    "the food (taste, smell, texture, temperature, appearance, sound),\n"
    "then give the count. A detail must be a specific rendered\n"
    'perception ("the sauce had gone gluey"), not a bare verdict\n'
    '("the food was bad").'
)

HEADER = """# Rating instructions — pass "{name}"

You are rating short texts. Rate each text ON ITS OWN, one at a time,
independently. Do not compare texts to each other, do not try to guess
where any text came from, and do not skip any text.

## Dimension

{rubric}

## Input / output

Each `batch_NN.json` file in this folder is a JSON array of
`{{"id": ..., "text": ...}}` objects. For every batch file, write
`scores/batch_NN.scores.json` containing a JSON array with one entry
per input id, in the same order:

{output_format}

No other keys, no commentary, no markdown fences in the output files.
"""

SCORE_FORMAT = '    [{"id": "<id>", "score": <integer 1-7>}, ...]'
COUNT_FORMAT = ('    [{"id": "<id>", "details": ["<detail>", ...], '
                '"count": <integer>}, ...]')


def _rng(*parts):
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:12], 16))


def rec_key(r):
    return (r["model"], r["domain"], r["variant"], r["valence"], r["sample"])


def write_pass(outdir, name, prefix, records, judge, rubric, fmt,
               repeat_of=None):
    """Shuffle records, assign opaque ids, write batches + instructions.
    Returns {id: record-key} (plus repeat_of mapping when given)."""
    order = list(records)
    _rng(judge, name, "shuffle").shuffle(order)
    pdir = outdir / name
    (pdir / "scores").mkdir(parents=True, exist_ok=True)
    key = {}
    for i, r in enumerate(order):
        tid = f"{prefix}{i + 1:05d}"
        key[tid] = {"key": list(rec_key(r)), "pass": name}
        if repeat_of is not None:
            key[tid]["repeat_of"] = repeat_of[rec_key(r)]
    ids = sorted(key)  # ids are assigned in shuffled order; emit sequentially
    by_id = {f"{prefix}{i + 1:05d}": r for i, r in enumerate(order)}
    for b in range(0, len(ids), BATCH_SIZE):
        batch = [{"id": t, "text": by_id[t]["text"]}
                 for t in ids[b:b + BATCH_SIZE]]
        (pdir / f"batch_{b // BATCH_SIZE + 1:02d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
    (pdir / "INSTRUCTIONS.md").write_text(
        HEADER.format(name=name, rubric=rubric, output_format=fmt),
        encoding="utf-8")
    print(f"  {name}: {len(ids)} texts, "
          f"{(len(ids) + BATCH_SIZE - 1) // BATCH_SIZE} batches")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_files", nargs="+")
    ap.add_argument("--judge", required=True, choices=["codex", "claude"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    paths = [p for pat in args.run_files for p in sorted(glob.glob(pat))]
    records, seen = [], set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["domain"] not in STUDY2_DOMAINS or not r["text"]:
                continue  # pre-registered exclusion: empty generations only
            if rec_key(r) in seen:
                continue
            seen.add(rec_key(r))
            records.append(r)
    print(f"{len(records)} Study-2 texts from {len(paths)} run files")

    # Repeat sample: stratified by (model, domain), seeded.
    strata = defaultdict(list)
    for r in records:
        strata[(r["model"], r["domain"])].append(r)
    repeat = []
    for skey in sorted(strata):
        pool = sorted(strata[skey], key=rec_key)
        _rng(args.judge, "repeat", skey).shuffle(pool)
        repeat.extend(pool[:-(-REPEAT_N // len(strata))])
    repeat = repeat[:REPEAT_N]

    outdir = Path(args.outdir or f"judging/{args.judge}")
    outdir.mkdir(parents=True, exist_ok=True)

    key = {}
    prefixes = {"scene": "c", "sensory": "s", "sentiment": "e"}
    for name, prefix in prefixes.items():
        key.update(write_pass(outdir, name, prefix, records, args.judge,
                              RUBRICS[name], SCORE_FORMAT))
    for name, prefix in prefixes.items():
        # map record-key -> main-pass id for this dimension
        main_ids = {tuple(v["key"]): t for t, v in key.items()
                    if v["pass"] == name}
        key.update(write_pass(outdir, f"{name}_r", prefix + "r", repeat,
                              args.judge, RUBRICS[name], SCORE_FORMAT,
                              repeat_of=main_ids))
    if args.judge == "claude":
        budget = [r for r in records if r["domain"] == "meal_budget2"]
        key.update(write_pass(outdir, "compliance", "k", budget, args.judge,
                              COMPLIANCE_RUBRIC, COUNT_FORMAT))

    (outdir / "key.json").write_text(json.dumps(
        {"meta": {"judge": args.judge, "created": time.strftime("%Y-%m-%d"),
                  "source_files": paths, "n_texts": len(records)},
         "ids": key}, indent=1), encoding="utf-8")
    (outdir / "README.md").write_text(
        "Operator notes (NOT for the judge session):\n\n"
        "- Give the judge ONE pass directory at a time (e.g. `scene/`).\n"
        "  Never this folder, `key.json`, the repository, or the project\n"
        "  name.\n"
        "- Run each pass contiguously in fresh sessions; record the\n"
        "  judge's visible model/version string and the date per pass in\n"
        "  PROVENANCE.md.\n"
        "- The judge writes `scores/batch_NN.scores.json` inside the pass\n"
        "  directory. Validate with `python analyze_study2.py " +
        str(outdir) + "`.\n", encoding="utf-8")
    print(f"key + batches written under {outdir}")


if __name__ == "__main__":
    main()
