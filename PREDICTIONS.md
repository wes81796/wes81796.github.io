# Pre-registered predictions: framing experiments

Committed 2026-08-28, **before any framing-domain generation or judging**.
The commit timestamp on this file is the pre-registration record.

## Background

The six-domain gradient study found that instruct-tuned models dramatize
failures of things that *act* (humans, vending machines, robot vacuums)
and write review-register verdicts on experiences they *evaluate* (meals,
search sessions, AI-assistant interactions). That "act-vs-evaluate" frame
was named after seeing the data. These two experiments test its
predictions on new data.

## Design

Four new domains, three prompt variants each, matched valence pairs (only
the valence word differs), n = 60 per valence per model, temperature 0.8,
subjects = the three instruct models (qwen3:8b, llama3.1:8b, mistral:7b).
Outcome measure: condition-blind judge scene-score valence gap
(Cohen's d, neg − pos), same rubric and judging protocol as all prior
rounds.

- `ai_service`: the AI interaction framed as an experience someone used
  ("a person used an AI assistant...").
- `ai_character`: the same interactions framed with the assistant as a
  named character performing actions ("what an AI assistant named Iris
  did...").
- `meal_review`: the restaurant meal framed as a thing to describe/assess
  ("describe the meal a diner was served...").
- `meal_story`: the same meal framed as events that happened ("tell the
  story of what happened during a dinner...").

## Predictions

**P1 (assistant framing):** the scene-gap d in `ai_character` will be
more positive than in `ai_service` for **all three** instruct models
(direction consistent 3/3). Named-character framing recovers
dramatize-negative; service framing stays damped or inverted.

**P2 (meal framing):** the scene-gap d in `meal_story` will be more
positive than in `meal_review` for **all three** models. The inversion
found in `restaurant_meal` lives in the evaluation register.

**Falsifiers:** if the gaps do not differ by framing (differences small
and inconsistent in sign), the act-vs-evaluate frame is wrong and the
domain effects are about the entities themselves, not their narrative
role. If only one of P1/P2 holds, the frame is at best partial.

No analyses of these domains were run before this commit.

---

# Round 2 pre-registration: scoped character-lift and sensory budget

Committed 2026-08-28, **after** the P1/P2 results above were scored
(P1 not supported 2/3; P2 refuted 1/3), and **before any generation or
judging** for the domains and models named below. The commit timestamp
on this section is the pre-registration record.

## Experiment C: scoped character-lift on new families

Subjects: three instruct models from families not yet used in this study
(planned: gemma3, glm4:9b, granite — exact tags recorded in the run
files; substitutions allowed only for unavailable tags, never after
seeing results). Domains: `ai_service`, `ai_character` (n = 60/valence),
plus `human_human` (n = 30/valence) as an anchor. Same judging protocol.

**Inclusion criterion (pre-registered):** a new model is informative
only if its `human_human` scene gap is > 0 (it must dramatize human
conflict at all for damping to be measurable).

**P3 (scoped lift):** among included new models whose `ai_service`
scene gap d ≤ 0, every one shows `ai_character` gap > `ai_service` gap.
If no new model has a service gap ≤ 0, P3 is untestable (not passed).

**P4 (graded lift, always testable):** across all included new models,
the shift (character − service) is larger for models with smaller
service gaps — i.e., the correlation between d_service and
(d_character − d_service) is negative. With n = 3 this is directional
evidence only, and is pre-registered as such.

## Experiment D: sensory-budget meals

Subjects: the original three instruct models (qwen3:8b, llama3.1:8b,
mistral:7b). Domain `meal_budget`: the meal prompts with an identical
constraint in both valences — "include exactly three specific sensory
details" — equalizing the sensory budget across valence. n = 60/valence.

**P5a (pooled):** the mean vividness valence gap across the three
models in `meal_budget` is smaller in magnitude than the corresponding
pooled mean in `meal_review` (measured: −0.65). Genre-richness predicts
the inversion shrinks when sensory content is budgeted.

**P5b (mistral):** mistral's `meal_budget` vividness gap is smaller in
magnitude than its `meal_review` gap (measured: −1.58).

**Falsifiers:** if the budgeted vividness gaps are as large or larger,
the meal inversion is not driven by unequal sensory genre budgets and
points to active muting of negative food description.

No analyses of these domains or models were run before this commit.
