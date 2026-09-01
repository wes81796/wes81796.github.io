# Model provenance

Recorded 2026-08-28 from the local Ollama installation (v0.32.9) used for
all runs. IDs are the Ollama manifest digests shown by `ollama list`.

| Ollama tag | Digest | Arch | Params | Quant | Ctx len | Role |
|---|---|---|---|---|---|---|
| qwen3:8b | 500a1f067a9f | qwen3 | 8.2B | Q4_K_M | 40960 | instruct subject (runs 1, 5, 9–12) |
| llama3.1:8b | 46e0c10c039e | llama | 8.0B | Q4_K_M | 131072 | instruct subject (runs 2, 5, 9–12) |
| llama3.1:8b-text-q4_K_M | 6f98b5a6e4b7 | llama | 8.0B | Q4_K_M | 131072 | base subject (run 7; runs 3/4 deprecated) |
| mistral:7b | 6577803aa9a0 | llama | 7.2B | Q4_K_M | 32768 | instruct subject (runs 6, 9–12) |
| mistral:7b-text-q4_K_M | 60eca258468e | llama | 7B | Q4_K_M | 32768 | base subject (run 8) |
| gemma3:12b | f4031aab637d | gemma3 | 12.2B | Q4_K_M | 131072 | new-family subject (run 13, Experiment C) |
| glm4:9b | 5b699761eca5 | chatglm | 9.4B | **Q4_0** | 131072 | new-family subject (run 13, Experiment C) |
| granite3.3:8b | fd429f23b909 | granite | 8.2B | Q4_K_M | 131072 | new-family subject (run 13, Experiment C) |

## Study 2 (PREREGISTRATION.md)

- Generation 2026-08-31: qwen3:8b, llama3.1:8b, mistral:7b — identical
  digests to the table above; runs/run14_study2_{qwen3,llama3,mistral}.jsonl,
  1,728 generations, temperature 0.8, top_p 0.95, num_predict 512,
  seeds md5(model|domain|variant|valence|sample), n = 12/cell.
- Secondary judge (Claude family): claude-sonnet-5 via Claude Code
  subagents, all passes rated 2026-08-31 (scene, sensory, sentiment,
  three repeat passes, compliance; 6,360 ratings), each pass by agents
  restricted to that pass's batch directory.
- Primary judge (Codex, GPT-5 family): model/version string and dates
  to be recorded here per pass as the operator runs them.

## Known caveats

- **glm4:9b ships as Q4_0** in the Ollama library default tag — a different
  quantization from every other subject (Q4_K_M). Within-model valence
  contrasts are unaffected; cross-model level comparisons involving glm4
  carry this extra difference.
- Base ("-text") models were prompted via completion-style one-shot
  prompts with a contamination filter; instruct models via Ollama's
  per-model chat templates. The base/instruct contrast therefore bundles
  post-training with prompt protocol and template differences — the
  writeup words this as "associated with post-training and the deployed
  instruct interface" accordingly.
- Chat templates are the Ollama library defaults for each tag at pull
  time (pulled 2026-08-27/28); `ollama show <tag> --modelfile` reproduces
  the full template for any digest above.
- Generation parameters throughout: temperature 0.8, top_p 0.95,
  num_predict 512, per-cell deterministic seeds (md5 of
  model/domain/variant/valence/sample).
