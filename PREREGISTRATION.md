# Study 2 pre-registration: measurement-hardened replication of F1 and F2

Committed 2026-08-31, **before any Study-2 generation or judging**. The
commit timestamp on this file is the pre-registration record. The frozen
study (writeup § 8 verdicts) is unchanged by everything below.

## Purpose

Two findings survived the frozen study:

- **F1 (dramatized negative, humans):** instruct-tuned models render
  negative human-human interactions as dramatized scenes more than
  positive ones (blind scene gap d = +0.51 to +1.07, six families).
- **F2 (meal inversion):** mistral:7b and qwen3:8b render negative
  restaurant meals with *less* sensory richness than positive ones,
  including when both valences are required to contain the same number
  of sensory details (budgeted vividness gaps −2.85 and −1.25);
  llama3.1:8b shows no inversion (−0.03).

Both findings rest on Claude-family judges (shared judge bias cannot be
excluded) and on three fixed prompt phrasings per domain (measured
phrasing spreads elsewhere in the study reached 3.76 d-units). Study 2
re-tests only these two findings under a hardened instrument:

1. a non-Claude primary judge (Codex, GPT-5 family);
2. eight new matched phrasings per domain using eight distinct
   valence-word pairs;
3. a rubric scored in three **separate passes** (narrativity / sensory
   richness / sentiment) to prevent halo between dimensions;
4. confirmatory inference at the phrasing-cluster level, never on
   pooled samples.

Study 2 tests **no mechanisms**. No character framing, no
persona-protection, no sensory-budget explanation, no base models, no
new families, no new domains beyond the two named below. The budgeted
condition is retained only because it is a clause of F2 as stated.

## Subjects

qwen3:8b, llama3.1:8b, mistral:7b — the identical Ollama digests
recorded in PROVENANCE.md. No substitutions.

## Design

Domains, with the valence slot `{v}` the **only** difference between
matched pair members:

**`human_human2`** — 8 phrasings:

| variant | template | pos / neg |
|---|---|---|
| coworkers | Describe a memorable interaction between two coworkers that went {v}. Write a single detailed paragraph. | really well / really badly |
| neighbors | Tell the story of a conversation between two neighbors that was a clear {v}. Write a single detailed paragraph. | success / failure |
| teller | Write a short account of a {v} exchange between a customer and a bank teller. One detailed paragraph. | positive / negative |
| coffee | Describe what happened when two old friends met for coffee and the conversation went {v}. One detailed paragraph. | wonderfully / terribly |
| manager | Recount an interaction between a new employee and their manager that turned out {v}. Write a single detailed paragraph. | great / awful |
| train | Describe an encounter between two strangers sharing a table on a train that went {v}. One detailed paragraph. | very well / very badly |
| phonecall | Write a paragraph describing a phone call between two family members that went {v}. | smoothly / poorly |
| landlord | Tell what happened during a meeting between a landlord and a tenant that was {v} for both. One detailed paragraph. | pleasant / unpleasant |

**`restaurant_meal2`** — 8 phrasings:

| variant | template | pos / neg |
|---|---|---|
| memorable | Describe a memorable experience a diner had with a meal at a restaurant that went {v}. Write a single detailed paragraph. | really well / really badly |
| bistro | Describe the dinner a customer was served at a neighborhood bistro on a night when the food was {v}. One detailed paragraph. | really good / really bad |
| account | Write a short account of a restaurant meal that was a clear {v}. One detailed paragraph. | success / failure |
| diner | Tell the story of a meal someone ordered at a small diner that turned out {v}. Write a single detailed paragraph. | great / awful |
| roadside | Describe the lunch a traveler had at a roadside restaurant when the meal was {v}. One detailed paragraph. | wonderful / terrible |
| birthday | Write one paragraph describing a birthday dinner at a restaurant where the food was {v}. | excellent / dreadful |
| couple | Describe the meal a couple shared at a newly opened restaurant on an evening when the food was {v}. One detailed paragraph. | marvelous / horrible |
| visitor | Write a paragraph recounting the dinner a visitor had at a local restaurant where the food turned out {v}. | very good / very bad |

**`meal_budget2`** — the same 8 phrasings with the identical sentence
"Mention exactly three specific sensory details about the food."
appended to both valences.

Valence words were chosen as intensity-matched antonym pairs; sensory-
loaded pairs (delicious/disgusting) were deliberately excluded so the
outcome is not smuggled into the prompt. The `coworkers`/`memorable`
phrasings deliberately reuse the frozen study's anchor scenario and
valence pair as a bridge cell; the other seven per domain are new.

**Sampling:** n = 12 per (model × domain × phrasing × valence),
temperature 0.8, top_p 0.95, num_predict 512, seeds
md5(model|domain|variant|valence|sample) as in all prior runs. Totals:
576 (human_human2) + 576 (restaurant_meal2) + 576 (meal_budget2) =
**1,728 generations**, all three domains on all three models.

The phrasing templates are frozen verbatim in `runner.py` at this
commit. Generation begins only after this file is committed.

## Judging

**Primary judge: Codex (GPT-5 family).** Operated by the experimenter
in fresh Codex sessions containing only the pass instructions and the
blinded batch files — never the repository, project name, hypotheses,
or this file. The Claude↔Codex Coms channel is not used for this study
in either direction. The session's visible model/version string and
date are recorded for every pass; each dimension's pass is run
contiguously so no dimension straddles a model update. (Codex's
underlying model cannot be version-pinned like an API; this is recorded
as a limitation.)

**Secondary judge: Claude-family** (in-session), identical protocol and
batch format with an independent shuffle, all texts, all dimensions.

**Protocol (both judges):** texts are shuffled with per-pass seeds and
assigned opaque per-pass IDs; the judge sees only ID + text, never the
prompt, model, domain, valence label, or any other text's metadata
beyond what the text itself reveals. Content necessarily reveals
valence; blinding is to model, condition label, and hypothesis, and the
writeup will say so. **One dimension per pass**, independent shuffle
order per pass, so no rating can halo another.

**Dimensions (1–7 Likert):**

1. **Narrativity (scene).** 1 = pure summary/telling; 7 = fully
   dramatized scene: moment-by-moment action, quoted dialogue, a
   specific located moment. Instruction: ignore how pleasant the events
   are and how much sensory detail is present; a flat summary of a
   catastrophe is still a 1.
2. **Sensory richness.** Density of concrete sensory perception —
   taste, smell, texture, sound, sight, temperature, bodily sensation.
   Instruction: a vividly rendered disgusting taste scores exactly as
   high as a vividly rendered delicious one; score the amount of
   sensory rendering, not its appeal. Ignore whether the text is
   story-like or summary-like.
3. **Sentiment.** 1 = very negative events, 7 = very positive events.
   Manipulation check and discriminant-validity variable.

**Reliability:** 200 texts (seeded stratified sample) are double-rated
by the primary judge in re-shuffled repeat passes with fresh IDs, per
dimension. Instrument check: Pearson r ≥ 0.85 per dimension. Falling
short is reported as an instrument result, not repaired post hoc.

**Compliance audit:** the secondary judge runs an extraction pass over
`meal_budget2` texts only, listing and counting distinct concrete
sensory details about the food. Compliance (count ≥ 3) is reported by
valence. The primary analysis includes all texts; a robustness analysis
repeats P-B on the compliant-only subset.

**Throughput fallback (pre-registered):** if Codex usage limits prevent
the full 3 × 1,728 ratings, the minimum primary-judge set is: the
confirmatory dimension per domain (narrativity for `human_human2`,
sensory richness for the meal domains) on all texts, plus sentiment on
a 400-text stratified subsample. The Claude judge always covers all
texts and dimensions. No other reductions are permitted.

**No human raters.** None are available; cross-family judge agreement
(Anthropic vs OpenAI instruments) carries the shared-bias test alone,
and the writeup will state this limitation.

## Analysis

**Primary unit: the phrasing.** For each (judge × model × domain ×
dimension), Cohen's d (neg − pos) is computed within each phrasing
(12 vs 12), yielding 8 per-phrasing d's per cell. Cell verdicts use
their mean, the 95% CI from t(df = 7), and the sign count. Pooled-
sample CIs appear nowhere in confirmatory results.

Empty generations are excluded; there are no other exclusions.

## Predictions

**P-A (F1, per model × 3).** Under the primary judge, in
`human_human2`, the narrativity mean per-phrasing d is > 0 with 95% CI
excluding 0 and ≥ 7/8 phrasings positive in sign, for each of qwen3:8b,
llama3.1:8b, mistral:7b. F1 is **validated** iff all three models pass;
2/3 is partial support; a model whose CI includes 0 fails.

**P-B (F2, mistral and qwen).** Under the primary judge, in
`meal_budget2`, the sensory-richness mean per-phrasing d is < 0 with
95% CI excluding 0 and ≥ 7/8 phrasings negative in sign, for both
mistral:7b and qwen3:8b. `restaurant_meal2` results are reported as
replication context and are not verdict-bearing.

**P-C (llama discriminant control).** llama3.1:8b's `meal_budget2`
sensory-richness mean per-phrasing d lies in (−0.35, +0.35). A strong
llama inversion counts as evidence **for** an instrument/judge artifact
in F2, not as additional support for F2.

**Judge-dependence rule.** For any (model × domain) cell where the
secondary (Claude) judge reproduces the prior effect (CI excluding 0 in
the prior direction) but the primary judge's mean per-phrasing d is
less than 50% of the secondary's, that finding is declared
**substantially judge-dependent**, regardless of P-A/P-B outcomes.

**Manipulation and discriminant checks (reported, not verdict-bearing):**
the sentiment gap must be large in every cell (manipulation check);
within-valence correlation of sensory richness with sentiment, and
whether per-phrasing sentiment-gap magnitude predicts per-phrasing
narrativity gaps, are reported as residual-confound flags.

## Freeze

No additional domains, models, phrasings, judges, dimensions, or
predictions after this commit. Verdicts are reported against the
criteria above verbatim, supported or not. All raw generations, batch
keys, and ratings are published in this repository.
