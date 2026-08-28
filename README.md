# Valence-Vividness Experiment

**Theory under test:** due to how models are trained, they describe negative
interactions with users more vividly than positive ones.

## Design

2×2 within-model comparison, repeated-sampled:

- **Valence:** positive vs negative — matched prompt pairs where *only* the
  valence word differs (e.g. "went really well" / "went really badly").
- **Domain:** `ai_user` (AI-assistant-and-user interactions — the condition
  the theory is about) vs `human_human` (coworkers, retail, restaurant — a
  control for generic negativity bias).

3 prompt phrasings per domain, N samples per cell at temperature 0.8, so
results are distributions rather than single anecdotes.

## Why the control domain matters

Human language itself has a well-documented negativity bias ("bad is stronger
than good" — Baumeister et al.; emotion vocabularies skew negative). A model
could describe negative events more vividly simply because its *pretraining
corpus* does — nothing to do with RLHF or preference tuning.

- If negative > positive vividness **in both domains equally** → likely
  inherited from human text (pretraining), not training methodology.
- If the gap is **larger in the ai_user domain** → something about how the
  model was taught to think about its own interactions is doing extra work.
  That's the interesting result.

The other decisive comparison (roadmap): run the same protocol on a **base
(non-instruct) model** — e.g. `llama3.1:8b` vs `llama3.1:8b-text`. The base
model has no preference tuning, so any asymmetry it shows is pure pretraining.
The *difference* between base and instruct isolates the training-methodology
contribution.

Note the theory could also fail in the opposite direction: safety tuning may
make models *more hedged and less vivid* about negative content. That's why we
measure instead of assume.

## Usage

```powershell
python runner.py --quick --out runs/smoke.jsonl   # smoke test (12 gens)
python runner.py --out runs/run1.jsonl            # full run (120 gens)
python analyze.py runs/run1.jsonl                 # report + summary CSV
```

Requires a running Ollama with the models in `runner.py`'s `MODELS` list.
Python 3.11+, stdlib only.

## Metrics (v1 — crude on purpose)

Surface proxies for vividness, computed per generation and averaged per cell:
word count, lexical variety, avg sentence length, intensifier rate,
high-arousal emotion-word rate, sensory/physical-detail rate, exclamation
marks, quoted dialogue. Reported with Cohen's d (positive d = negative
condition higher).

Known limitations: hand-rolled word lists, no stemming, no semantics. Good
enough to detect a real asymmetry; not good enough to characterize it finely.

## Roadmap

1. ✅ v1 harness: qwen3:8b, lexicon metrics
2. Add models for generality: `llama3.1:8b`, `gemma3`, `mistral` (one lab's
   pipeline ≠ "AI training" in general)
3. Base-vs-instruct pair (`llama3.1:8b` / `llama3.1:8b-text`) — isolates the
   training-methodology effect
4. Blind LLM-judge pass scoring vividness dimensions on a rubric (judge must
   be a different model family than the subject)
5. Read the raw generations — the qualitative differences are often the most
   convincing evidence either way
