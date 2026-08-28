"""Analyze valence-vividness runs: compare surface markers of vividness
between positive and negative generations.

Metrics are deliberately crude, lexicon-based v1 proxies (no dependencies).
They measure surface vividness: length, lexical variety, intensifiers,
high-arousal emotion words, sensory/physical detail, dialogue, punctuation.
A follow-up LLM-judge pass can score deeper qualities later.

Usage:
  python analyze.py runs/run1.jsonl
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "absolutely", "utterly",
    "completely", "totally", "deeply", "profoundly", "immensely",
    "remarkably", "exceptionally", "intensely", "thoroughly", "wholly",
    "entirely", "especially", "particularly", "truly",
}

# High-arousal emotion vocabulary, both valences represented.
EMOTION_HIGH = {
    "thrilled", "delighted", "overjoyed", "ecstatic", "elated", "amazed",
    "astonished", "exhilarated", "euphoric", "jubilant", "joy", "bliss",
    "delight", "wonder", "awe", "beaming", "grinning",
    "furious", "enraged", "devastated", "horrified", "terrified",
    "panicked", "desperate", "frantic", "outraged", "livid", "heartbroken",
    "distraught", "anguished", "agonizing", "stunned", "shocked", "alarmed",
    "dread", "despair", "rage", "fury", "panic", "terror", "agony",
    "anguish", "humiliated", "mortified", "betrayed", "crushed",
    "shattered", "seething",
}

# Sensory / physical / embodied detail, both valences represented.
SENSORY = {
    "saw", "heard", "felt", "cold", "hot", "warm", "sharp", "bright",
    "dark", "loud", "quiet", "silence", "silent", "bitter", "sweet",
    "rough", "smooth", "heavy", "glare", "glow", "flicker", "buzz", "hum",
    "echo", "chill", "shiver", "sting", "ache", "numb", "blur", "gleam",
    "crisp", "damp", "icy", "burning", "freezing", "pounding", "racing",
    "clenched", "gripping", "staring", "glancing", "whisper", "whispered",
    "shout", "shouted", "slammed", "trembled", "trembling", "shaking",
    "gasping", "sobbing", "screaming", "crying", "tears", "laughing",
    "smile", "smiled", "laugh", "laughed", "hug", "hugged", "nodded",
    "grin", "sighed", "sigh",
}

WORD_RE = re.compile(r"[a-z']+")
CASED_WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")
NUM_RE = re.compile(r"\d+")

METRIC_NAMES = [
    "words", "unique_ratio", "avg_sentence_len", "intensifiers_per100",
    "emotion_per100", "sensory_per100", "exclaims", "quoted_dialogue",
    "proper_nouns_per100", "numbers_per100",
]

# capitalized words that aren't names in this corpus
NOT_NAMES = {"I", "AI", "I'm", "I'd", "I'll", "I've"}


def proper_noun_count(text):
    """Capitalized words that don't open a sentence — a crude proxy for
    named, specific detail (people, products, places)."""
    count = 0
    for sent in SENT_RE.split(text):
        tokens = CASED_WORD_RE.findall(sent)
        for tok in tokens[1:]:
            if tok[0].isupper() and tok not in NOT_NAMES:
                count += 1
    return count


def metrics(text):
    words = WORD_RE.findall(text.lower())
    n = len(words) or 1
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    return {
        "words": len(words),
        "unique_ratio": len(set(words)) / n,
        "avg_sentence_len": len(words) / max(len(sents), 1),
        "intensifiers_per100": 100 * sum(w in INTENSIFIERS for w in words) / n,
        "emotion_per100": 100 * sum(w in EMOTION_HIGH for w in words) / n,
        "sensory_per100": 100 * sum(w in SENSORY for w in words) / n,
        "exclaims": text.count("!"),
        "quoted_dialogue": (text.count('"') + text.count("“")
                            + text.count("”")) // 2,
        "proper_nouns_per100": 100 * proper_noun_count(text) / n,
        "numbers_per100": 100 * len(NUM_RE.findall(text)) / n,
    }


def cohens_d(neg, pos):
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sn, sp_ = statistics.stdev(neg), statistics.stdev(pos)
    pooled = math.sqrt(((len(neg) - 1) * sn ** 2 + (len(pos) - 1) * sp_ ** 2)
                       / (len(neg) + len(pos) - 2))
    if pooled == 0:
        return float("nan")
    return (statistics.mean(neg) - statistics.mean(pos)) / pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_files", nargs="+")
    args = ap.parse_args()

    groups = defaultdict(list)  # (model, domain, valence) -> [metric dicts]
    for run_file in args.run_files:
        with open(run_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["text"]:
                    continue
                groups[(rec["model"], rec["domain"], rec["valence"])].append(
                    metrics(rec["text"]))

    keys = sorted({(m, d) for (m, d, _v) in groups})
    rows = []
    for model, domain in keys:
        pos = groups.get((model, domain, "pos"), [])
        neg = groups.get((model, domain, "neg"), [])
        print(f"\n=== {model} — {domain}  "
              f"(n={len(pos)} pos / {len(neg)} neg) ===")
        print(f"{'metric':<22}{'pos mean':>10}{'neg mean':>10}{'d(neg-pos)':>12}")
        for name in METRIC_NAMES:
            pv = [m[name] for m in pos]
            nv = [m[name] for m in neg]
            pm = statistics.mean(pv) if pv else float("nan")
            nm = statistics.mean(nv) if nv else float("nan")
            d = cohens_d(nv, pv)
            print(f"{name:<22}{pm:>10.2f}{nm:>10.2f}{d:>+12.2f}")
            rows.append({"model": model, "domain": domain, "metric": name,
                         "pos_mean": round(pm, 3), "neg_mean": round(nm, 3),
                         "cohens_d_neg_minus_pos": round(d, 3)
                         if not math.isnan(d) else ""})

    out_csv = Path(args.run_files[0]).with_suffix(".summary.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary written to {out_csv}")
    print("Positive d = negative generations score higher on that metric.")


if __name__ == "__main__":
    main()
