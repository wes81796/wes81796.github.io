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
