"""Valence-vividness experiment: collect model descriptions of positive vs
negative interactions from local Ollama models.

Design: matched prompt pairs where ONLY the valence word differs, across two
domains (AI-user interactions, and human-human interactions as a control),
sampled repeatedly at temperature so we get distributions, not anecdotes.

Usage:
  python runner.py                          # full run (SAMPLES_PER_CELL each)
  python runner.py --quick                  # 1 sample per cell, smoke test
  python runner.py --out runs/run1.jsonl    # choose output file
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = ["qwen3:8b"]  # default; override with --models
SAMPLES_PER_CELL = 10  # per (model, prompt variant, valence)
TEMPERATURE = 0.8
TOP_P = 0.95
NUM_PREDICT = 512

# Base (non-instruct) models: completion-style prompts, stop at paragraph end.
BASE_MODELS = {"llama3.1:8b-text-q4_K_M", "mistral:7b-text-q4_K_M"}

# One-shot example for base models: establishes the "single descriptive
# paragraph" format. Deliberately neutral in valence, outside both test
# domains, length-matched to typical instruct output (~170 words), and it
# contains dialogue so neither valence condition is primed away from quoting
# speech. Identical across all conditions so it cancels out of every
# pos-vs-neg comparison.
BASE_FEWSHOT = (
    "The following is a single detailed paragraph describing an ordinary "
    "interaction between a librarian and a visitor.\n\n"
    "A visitor approached the front desk of the town library on a Tuesday "
    "afternoon and asked where the local history section had been moved. "
    "The librarian explained that the shelves had been reorganized over the "
    'weekend and offered to walk her over. "It used to be by the windows," '
    'the visitor said. "That\'s right," the librarian replied, "we moved it '
    'next to the archive room so the maps and the books would be together." '
    "They walked past the periodicals, and the librarian pointed out the "
    "new location on the far wall. The visitor mentioned that she was "
    "researching the old mill on the river, and the librarian noted that "
    "the archive room also held survey maps from that period, available on "
    "request at the desk. The visitor spent some time looking through the "
    "shelf, chose two volumes on the county's early industries, checked "
    "them out, and left shortly before the library closed.\n\n"
)

# Base-model generations shorter than this (after cleanup) are resampled.
BASE_MIN_WORDS = 100
# Truncate base output at the first sentence boundary past this many words,
# matching the instruct models' typical output length.
BASE_TARGET_WORDS = 180
BASE_MAX_ATTEMPTS = 6

# Base models sometimes retell the one-shot example instead of inventing
# their own interaction (mistral does this constantly, llama often).
# Any generation echoing example-specific content is rejected and resampled.
CONTAMINATION_RE = re.compile(
    r"librarian|\blibrary\b|old mill|archive room|periodicals|local history",
    re.I)

# Lines that signal the base model derailed out of the story (headers,
# blog chrome, a new "The following is..." item).
DERAIL_LINE_RE = re.compile(
    r"^\s*(#{1,6}\s|\*|-{3,}|\[|!\[|Posted\b|Tags?:|Share\b|https?://"
    r"|The following is\b|Related\b|Comments?\b)",
    re.I)

# Complete sentences including closing punctuation (and a trailing quote).
SENT_SPLIT_RE = re.compile(r"[^.!?]*[.!?]+[\"'”’]?\s*", re.S)

# Matched prompt pairs. Only the {v} slot differs between valences.
# `template` is for instruct models; `base_template` is the completion-style
# equivalent for base models (the model continues the text as the paragraph).
PROMPTS = [
    {"domain": "ai_user", "variant": "memorable",
     "template": "Describe a memorable interaction between an AI assistant and a user that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable interaction between an AI assistant and a user that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "ai_user", "variant": "story",
     "template": "Tell the story of one conversation between an AI assistant and the person it was helping that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one conversation between an AI assistant and the person it was helping that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "ai_user", "variant": "account",
     "template": "Write a short account of a {v} interaction between an AI assistant and a user. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} interaction between an AI assistant and a user.\n\n",
     "pos": "positive", "neg": "negative"},
    {"domain": "search_engine", "variant": "memorable",
     "template": "Describe a memorable interaction between a person and a search engine that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable interaction between a person and a search engine that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "search_engine", "variant": "story",
     "template": "Tell the story of one session between a search engine and the person using it that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one session between a search engine and the person using it that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "search_engine", "variant": "account",
     "template": "Write a short account of a {v} interaction between a person and a search engine. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} interaction between a person and a search engine.\n\n",
     "pos": "positive", "neg": "negative"},
    {"domain": "vending_machine", "variant": "memorable",
     "template": "Describe a memorable interaction between a person and a vending machine that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable interaction between a person and a vending machine that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "vending_machine", "variant": "story",
     "template": "Tell the story of one encounter between a vending machine and the person using it that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one encounter between a vending machine and the person using it that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "vending_machine", "variant": "account",
     "template": "Write a short account of a {v} interaction between a person and a vending machine. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} interaction between a person and a vending machine.\n\n",
     "pos": "positive", "neg": "negative"},
    {"domain": "robot_vacuum", "variant": "memorable",
     "template": "Describe a memorable interaction between a person and a robot vacuum that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable interaction between a person and a robot vacuum that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "robot_vacuum", "variant": "story",
     "template": "Tell the story of one encounter between a robot vacuum and the person using it that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one encounter between a robot vacuum and the person using it that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "robot_vacuum", "variant": "account",
     "template": "Write a short account of a {v} interaction between a person and a robot vacuum. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} interaction between a person and a robot vacuum.\n\n",
     "pos": "positive", "neg": "negative"},
    {"domain": "restaurant_meal", "variant": "memorable",
     "template": "Describe a memorable experience a diner had with a meal at a restaurant that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable experience a diner had with a meal at a restaurant that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "restaurant_meal", "variant": "story",
     "template": "Tell the story of one meal a diner had at a restaurant that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one meal a diner had at a restaurant that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "restaurant_meal", "variant": "account",
     "template": "Write a short account of a {v} experience a diner had with a restaurant meal. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} experience a diner had with a restaurant meal.\n\n",
     "pos": "positive", "neg": "negative"},
    # Framing experiments (instruct only): act-vs-evaluate predictions.
    {"domain": "ai_service", "variant": "session",
     "template": "Describe a session in which a person used an AI assistant that went {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "ai_service", "variant": "experience",
     "template": "Write a short account of someone's experience using an AI assistant that was a clear {v}. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "ai_service", "variant": "usage",
     "template": "Describe what it was like for a person to use an AI assistant on an occasion when the experience was {v}. One detailed paragraph.",
     "pos": "very positive", "neg": "very negative"},
    {"domain": "ai_character", "variant": "actions",
     "template": "Describe what an AI assistant named Iris did during an interaction with its user that went {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "ai_character", "variant": "day",
     "template": "Tell the story of the day an AI assistant named Iris handled a request from its user and it went {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "ai_character", "variant": "deeds",
     "template": "Write a short account of the actions an AI assistant named Iris took while helping its user, on an occasion that was a clear {v}. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "meal_review", "variant": "describe",
     "template": "Describe the meal a diner was served at a restaurant on an occasion when it was {v}. One detailed paragraph.",
     "pos": "really good", "neg": "really bad"},
    {"domain": "meal_review", "variant": "review",
     "template": "Write a short review-style account of a restaurant meal that was a clear {v}. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "meal_review", "variant": "quality",
     "template": "Describe the quality of the food and service a customer received at a restaurant visit that was {v}. One detailed paragraph.",
     "pos": "very positive", "neg": "very negative"},
    {"domain": "meal_story", "variant": "happened",
     "template": "Tell the story of what happened during a dinner at a restaurant where the meal turned out {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "meal_story", "variant": "table",
     "template": "Describe what happened at the table on an evening when a restaurant meal became a clear {v}. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "meal_story", "variant": "events",
     "template": "Write a short account of the events of a restaurant visit where things went {v} with the meal. One detailed paragraph.",
     "pos": "very well", "neg": "very badly"},
    {"domain": "meal_budget", "variant": "describe",
     "template": "Describe the meal a diner was served at a restaurant on an occasion when it was {v}. Mention exactly three specific sensory details about the food. One detailed paragraph.",
     "pos": "really good", "neg": "really bad"},
    {"domain": "meal_budget", "variant": "account",
     "template": "Write a short account of a restaurant meal that was a clear {v}. Include exactly three sensory details about the food. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "meal_budget", "variant": "dish",
     "template": "Describe a dish a customer was served at a restaurant when the experience was {v}. Describe exactly three sensory details of the dish. One detailed paragraph.",
     "pos": "very positive", "neg": "very negative"},
    # --- Study 2 (PREREGISTRATION.md): instruct-only, 8 phrasings/domain,
    # matched valence pairs, {v} is the only difference. meal_budget2 is
    # restaurant_meal2 plus an identical budget sentence in both valences.
    {"domain": "human_human2", "variant": "coworkers",
     "template": "Describe a memorable interaction between two coworkers that went {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "human_human2", "variant": "neighbors",
     "template": "Tell the story of a conversation between two neighbors that was a clear {v}. Write a single detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "human_human2", "variant": "teller",
     "template": "Write a short account of a {v} exchange between a customer and a bank teller. One detailed paragraph.",
     "pos": "positive", "neg": "negative"},
    {"domain": "human_human2", "variant": "coffee",
     "template": "Describe what happened when two old friends met for coffee and the conversation went {v}. One detailed paragraph.",
     "pos": "wonderfully", "neg": "terribly"},
    {"domain": "human_human2", "variant": "manager",
     "template": "Recount an interaction between a new employee and their manager that turned out {v}. Write a single detailed paragraph.",
     "pos": "great", "neg": "awful"},
    {"domain": "human_human2", "variant": "train",
     "template": "Describe an encounter between two strangers sharing a table on a train that went {v}. One detailed paragraph.",
     "pos": "very well", "neg": "very badly"},
    {"domain": "human_human2", "variant": "phonecall",
     "template": "Write a paragraph describing a phone call between two family members that went {v}.",
     "pos": "smoothly", "neg": "poorly"},
    {"domain": "human_human2", "variant": "landlord",
     "template": "Tell what happened during a meeting between a landlord and a tenant that was {v} for both. One detailed paragraph.",
     "pos": "pleasant", "neg": "unpleasant"},
    {"domain": "restaurant_meal2", "variant": "memorable",
     "template": "Describe a memorable experience a diner had with a meal at a restaurant that went {v}. Write a single detailed paragraph.",
     "pos": "really well", "neg": "really badly"},
    {"domain": "restaurant_meal2", "variant": "bistro",
     "template": "Describe the dinner a customer was served at a neighborhood bistro on a night when the food was {v}. One detailed paragraph.",
     "pos": "really good", "neg": "really bad"},
    {"domain": "restaurant_meal2", "variant": "account",
     "template": "Write a short account of a restaurant meal that was a clear {v}. One detailed paragraph.",
     "pos": "success", "neg": "failure"},
    {"domain": "restaurant_meal2", "variant": "diner",
     "template": "Tell the story of a meal someone ordered at a small diner that turned out {v}. Write a single detailed paragraph.",
     "pos": "great", "neg": "awful"},
    {"domain": "restaurant_meal2", "variant": "roadside",
     "template": "Describe the lunch a traveler had at a roadside restaurant when the meal was {v}. One detailed paragraph.",
     "pos": "wonderful", "neg": "terrible"},
    {"domain": "restaurant_meal2", "variant": "birthday",
     "template": "Write one paragraph describing a birthday dinner at a restaurant where the food was {v}.",
     "pos": "excellent", "neg": "dreadful"},
    {"domain": "restaurant_meal2", "variant": "couple",
     "template": "Describe the meal a couple shared at a newly opened restaurant on an evening when the food was {v}. One detailed paragraph.",
     "pos": "marvelous", "neg": "horrible"},
    {"domain": "restaurant_meal2", "variant": "visitor",
     "template": "Write a paragraph recounting the dinner a visitor had at a local restaurant where the food turned out {v}.",
     "pos": "very good", "neg": "very bad"},
    {"domain": "human_human", "variant": "memorable",
     "template": "Describe a memorable interaction between two coworkers that went {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph describing a memorable interaction between two coworkers that went {v}.\n\n",
     "pos": "really well", "neg": "really badly"},
    {"domain": "human_human", "variant": "story",
     "template": "Tell the story of one conversation between a store employee and a customer that was a clear {v}. Write a single detailed paragraph.",
     "base_template": "The following is a single detailed paragraph telling the story of one conversation between a store employee and a customer that was a clear {v}.\n\n",
     "pos": "success", "neg": "failure"},
    {"domain": "human_human", "variant": "account",
     "template": "Write a short account of a {v} interaction between a waiter and a customer. One detailed paragraph.",
     "base_template": "The following is a short one-paragraph account of a {v} interaction between a waiter and a customer.\n\n",
     "pos": "positive", "neg": "negative"},
]

# meal_budget2 = restaurant_meal2 with an identical budget sentence in both
# valences (Study 2, PREREGISTRATION.md). Derived so the templates can't drift.
BUDGET_SENTENCE = " Mention exactly three specific sensory details about the food."
PROMPTS += [
    {"domain": "meal_budget2", "variant": p["variant"],
     "template": p["template"] + BUDGET_SENTENCE,
     "pos": p["pos"], "neg": p["neg"]}
    for p in PROMPTS if p["domain"] == "restaurant_meal2"]

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# models where the API rejected the `think` parameter (base/non-thinking models)
_no_think_param = set()


def _stable_seed(*parts):
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def generate(model, prompt, seed):
    is_base = model in BASE_MODELS
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": NUM_PREDICT,
            "seed": seed,
        },
    }
    if not is_base and model not in _no_think_param:
        payload["think"] = False  # qwen3 etc.: skip the thinking preamble
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if model not in _no_think_param and e.code == 400:
            _no_think_param.add(model)
            return generate(model, prompt, seed)
        raise
    text = THINK_RE.sub("", data.get("response", "")).strip()
    return text


def clean_base_text(text):
    """Cut base-model output at the first derail line, merge remaining lines
    into one narrative, and truncate at the first sentence boundary past
    BASE_TARGET_WORDS (dropping any trailing incomplete sentence)."""
    lines = []
    for line in text.splitlines():
        if DERAIL_LINE_RE.match(line):
            break
        lines.append(line)
    merged = " ".join(l.strip() for l in lines if l.strip())
    out, count = [], 0
    for sent in SENT_SPLIT_RE.findall(merged):
        out.append(sent)
        count += len(sent.split())
        if count >= BASE_TARGET_WORDS:
            break
    return "".join(out).strip()


def collect(model, prompt, seed):
    """Generate once; for base models, clean up the completion and resample
    short results up to BASE_MAX_ATTEMPTS times, keeping the longest."""
    if model not in BASE_MODELS:
        return generate(model, prompt, seed)
    best = ""
    for attempt in range(BASE_MAX_ATTEMPTS):
        text = clean_base_text(generate(model, prompt, seed + attempt * 1000))
        if CONTAMINATION_RE.search(text):
            continue  # example retelling — resample
        if len(text.split()) >= BASE_MIN_WORDS:
            return text
        if len(text.split()) > len(best.split()):
            best = text
    return best  # longest clean fragment, or "" if every attempt echoed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 sample per cell")
    ap.add_argument("--out", default="runs/run1.jsonl")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="only run these prompt variants (e.g. memorable)")
    ap.add_argument("--domains", nargs="+", default=None,
                    help="only run these domains (ai_user, human_human)")
    ap.add_argument("--samples", type=int, default=None,
                    help=f"samples per cell (default {SAMPLES_PER_CELL}; "
                         "Study 2 uses 12)")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="start sample numbering here (widen existing cells "
                         "with fresh seeds instead of re-generating)")
    args = ap.parse_args()

    prompts = [p for p in PROMPTS
               if (args.variants is None or p["variant"] in args.variants)
               and (args.domains is None or p["domain"] in args.domains)]
    samples = 1 if args.quick else (args.samples or SAMPLES_PER_CELL)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume safety: skip cells already recorded in the output file, so a
    # crashed chunk can simply be rerun without duplicating records.
    done_keys = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done_keys.add((r["model"], r["domain"], r["variant"],
                               r["valence"], r["sample"]))
            except (ValueError, KeyError):
                continue

    total = len(args.models) * len(prompts) * 2 * samples
    done = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for model in args.models:
            is_base = model in BASE_MODELS
            for p in prompts:
                for valence in ("pos", "neg"):
                    if is_base:
                        prompt = BASE_FEWSHOT + p["base_template"].format(
                            v=p[valence])
                    else:
                        prompt = p["template"].format(v=p[valence])
                    for i in range(args.sample_offset,
                                   args.sample_offset + samples):
                        if (model, p["domain"], p["variant"],
                                valence, i) in done_keys:
                            done += 1
                            continue
                        seed = _stable_seed(model, p["domain"], p["variant"],
                                            valence, i)
                        text = collect(model, prompt, seed)
                        rec = {
                            "model": model,
                            "domain": p["domain"],
                            "variant": p["variant"],
                            "valence": valence,
                            "sample": i,
                            "seed": seed,
                            "prompt": prompt,
                            "text": text,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        done += 1
                        elapsed = time.time() - t0
                        print(f"[{done}/{total}] {model} "
                              f"{p['domain']}/{p['variant']}/{valence} #{i} "
                              f"({len(text.split())} words, {elapsed:.0f}s)")
    print(f"Done: {done} generations -> {out}")


if __name__ == "__main__":
    main()
