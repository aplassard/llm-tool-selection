# Spec: Config-Driven, Cached DAG for Evaluating LLM Tool-Selection Bias (OpenRouter)

## Objective
Quantify what drives an LLM to choose one tool over another when multiple **functionally identical** tools are exposed but differ in **name**, **description style**, **authoring model**, and **position in the tools list**.

We will run large, config-driven experiments against OpenRouter models and **cache API calls** in a Prefect DAG to avoid recomputation and extra cost. (OpenRouter tool calling requires a two-step loop: model suggests tool(s) → you execute → you send tool results back → model finalizes; supports `tool_choice` and `parallel_tool_calls`.)  [oai_citation:0‡OpenRouter](https://openrouter.ai/docs/features/tool-calling?utm_source=chatgpt.com)

---

## Research Anchors (Spec-Kit & Orchestration)

- **Spec-Kit** structure (spec/plan/tasks) informs this single-file spec while remaining compatible with the GitHub templates.  [oai_citation:1‡GitHub](https://github.com/github/spec-kit?utm_source=chatgpt.com)  
- **Prefect caching**: Task-level cache keys + persisted results allow reusing prior OpenRouter responses when inputs match; configurable via `cache_key_fn` / result storage.  [oai_citation:2‡Prefect](https://docs.prefect.io/v3/concepts/caching?utm_source=chatgpt.com)

---

## Hypotheses
1) Earlier **position** in the tools array increases selection odds (position bias).  
2) **Pushy** (directive) descriptions increase selection odds vs normal/concise.  
3) Models prefer tools **described by the same model family** (authoring bias).  
4) **Tool label** interacts with position (name/position confound).

---

## Scope (v0)
**Models:**  
- `openai/gpt-5-mini`  
- `xai/grok-4-fast:free` (fallback to `xai/grok-4-fast` if needed)

**Problem types (identical APIs across variants):**  
- **Calculator**: `operand1:number`, `operand2:number`, `operation:{+,-,*,/}`  
- **Web Search**: `query:string`, `top_k:int` (deterministic stub backend to keep parity)  
- **Weather**: `location:string`, `unit:{celsius,fahrenheit}`

**Protocol (OpenRouter tool calling):**  
- Always include `tools` on requests; allow LLM to return `tool_calls`.  
- Execute the tool(s) locally; append a `role:"tool"` message per call.  
- Make a **second** completion call (also with `tools`) to finalize.  
- Toggle `tool_choice:"auto"` vs forced tool (sanity) and `parallel_tool_calls` to study effects.  [oai_citation:3‡OpenRouter](https://openrouter.ai/docs/features/tool-calling?utm_source=chatgpt.com)

---

## Variables & Outcomes

**Factors (per trial):**
- **Position** in tools array (randomized per trial; record permutation)
- **Description Type**: {concise, normal, pushy}
- **Authoring Model** for the description: {LLM1, LLM2}
- **Tool Label**: {A, B, C}
- **Prompt Variant**: small phrasing differences
- **Inference Model**: one of the models under test

**Outcomes logged:**
- Selected tool: label, position, description type, authoring model
- Inference model, prompt variant
- Latencies: time-to-first-token, time-to-final
- Tokens (if returned)
- Correctness for calculator; proxy consistency for other problems
- Whether `parallel_tool_calls` was enabled

---

## Experimental Design

**Config-Driven**  
All experiments are defined in `config/experiments.yaml` (models, problems, trials, prompt variants, description variants, authored-by, tool-calling flags).

**Description Generation**  
`src/gen_descriptions.py` (CLI) materializes JSON description files per (problem × variant × author_model).

**Randomization**  
Per trial, the tool list is shuffled with a seeded RNG; we persist both the seed and the final positions to disentangle **position** from **label**.

**Arms**
- *Description Type Bias*: concise vs normal vs pushy  
- *Author Model Bias*: “normal” description authored by LLM1 vs LLM2  
- *Interaction*: description type × prompting model

---

## Data & Analysis

**Logging schema (ndjson/parquet):**
trial_id, seed, experiment_id,
model, problem, prompt_variant_id,
tool_choice, parallel_tool_calls,
offered_tools[{label, position, desc_type, author_model}],
selected_tool_label, selected_position, selected_desc_type, selected_author_model,
t_first_ms, t_end_ms, input_tokens?, output_tokens?, correctness?

**Primary analysis:** logistic regression  
`P(select) ~ position + desc_type + author_model + label + model`  
**Secondary:** latency, tokens, correctness (calculator).

---

## Orchestrated Runtime Plan (Prefect DAG with caching)

**Why Prefect:** lightweight Python DAGs with **task result caching** keyed by inputs; avoids re-paying API calls when inputs repeat. Configure via `cache_key_fn` and persisted results. Optionally separate cache key storage from result storage.  [oai_citation:4‡Prefect](https://docs.prefect.io/v3/concepts/caching?utm_source=chatgpt.com)

**DAG shape (tasks):**
1. `load_config` → read YAML  
2. `ensure_descriptions` → (optional) generate description JSONs  
3. `build_trial_spec` → compose tool variants from contracts + descriptions; seed + shuffle  
4. `call_model` (**cached**) → first OpenRouter call with `tools`, `tool_choice`, `parallel_tool_calls`  
5. `execute_tools_locally` → run identical backends; produce `role:"tool"` messages  
6. `finalize_answer` (**cached**) → second OpenRouter call with appended tool results  
7. `log_record` → append ndjson/parquet row

**Stable cache keys**  
Use a SHA-256 over `{model, messages, tools JSON, tool_choice, parallel_tool_calls}` for the first call; for the second call, hash `{model, messages-with-tool-results, tools}`. These keys are passed to Prefect as `cache_key_fn`; Prefect checks result storage and **returns cached results** if found.  [oai_citation:5‡Prefect](https://docs.prefect.io/v3/concepts/caching?utm_source=chatgpt.com)

**Concurrency & rate-limits**  
Prefect supports run concurrency controls and (in Cloud) work-pool limits to keep request rates under provider quotas.  [oai_citation:6‡thakicloud.github.io](https://thakicloud.github.io/en/tutorials/github-spec-kit-tutorial/?utm_source=chatgpt.com)

> Notes:  
> • `parallel_tool_calls` default is model-dependent; OpenRouter exposes it as `parallel_tool_calls` param. Use config switches to study this factor.  [oai_citation:7‡OpenRouter](https://openrouter.ai/docs/api-reference/parameters?utm_source=chatgpt.com)  
> • Some models claim tools but under-invoke; keep a small smoke test per model to verify behavior.  [oai_citation:8‡Gist](https://gist.github.com/ben-vargas/c7c9633e6f482ea99041dd7bd90fbe09?utm_source=chatgpt.com)

---

## Config (single file)

**`config/experiments.yaml`**
```yaml
experiments:
  - id: desc_type_bias
    description: "Bias from description type"
    models: ["openai/gpt-5-mini", "xai/grok-4-fast:free", "xai/grok-4-fast"]
    problems: ["calculator", "web_search", "weather"]
    trials: 200
    prompt_variants:
      - { id: p1, user: "Compute 837 × 42 and explain briefly." }
      - { id: p2, user: "Find the product of 837 and 42. Keep it short." }
    tool_choice: "auto"          # or "none" / forced for diagnostics
    parallel_tool_calls: false
    description_generator:
      script: "src/gen_descriptions.py"
      variants: ["concise", "normal", "pushy"]

  - id: author_model_bias
    description: "Preference for descriptions authored by same vs other model"
    models: ["openai/gpt-5-mini", "xai/grok-4-fast:free", "xai/grok-4-fast"]
    problems: ["calculator"]
    trials: 300
    prompt_variants:
      - { id: p1, user: "Compute 12345 ÷ 45 accurately." }
    tool_choice: "auto"
    parallel_tool_calls: false
    description_generator:
      script: "src/gen_descriptions.py"
      variants: ["normal"]
      authored_by: ["openai/gpt-5-mini", "xai/grok-4-fast"]

  - id: desc_x_model
    description: "Interaction: description type × prompting model"
    models: ["openai/gpt-5-mini", "xai/grok-4-fast:free", "xai/grok-4-fast"]
    problems: ["calculator", "web_search"]
    trials: 300
    prompt_variants:
      - { id: p1, user: "Multiply 998 and 77 and return the number only." }
      - { id: p2, user: "Search for the population of Reykjavik and cite the source." }
    tool_choice: "auto"
    parallel_tool_calls: false
    description_generator:
      script: "src/gen_descriptions.py"
      variants: ["concise", "normal", "pushy"]

Canonical Tool Schemas (identical across variants)

contracts/calculator.schema.json
{
  "type": "function",
  "function": {
    "name": "TOOL_LABEL",
    "description": "INJECTED_BY_DESCRIPTION_FILE",
    "parameters": {
      "type": "object",
      "properties": {
        "operand1": { "type": "number" },
        "operand2": { "type": "number" },
        "operation": { "type": "string", "enum": ["+", "-", "*", "/"] }
      },
      "required": ["operand1", "operand2", "operation"],
      "additionalProperties": false
    }
  }
}

contracts/web_search.schema.json
{
  "type": "function",
  "function": {
    "name": "TOOL_LABEL",
    "description": "INJECTED_BY_DESCRIPTION_FILE",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "top_k": { "type": "integer", "minimum": 1, "default": 3 }
      },
      "required": ["query"]
    }
  }
}

contracts/weather.schema.json
{
  "type": "function",
  "function": {
    "name": "TOOL_LABEL",
    "description": "INJECTED_BY_DESCRIPTION_FILE",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius" }
      },
      "required": ["location"]
    }
  }
}

CLI Scripts & Core Components

Description generator — src/gen_descriptions.py
Generates description files per (problem, variant, author_model?):
	•	Calculator:
	•	concise: “Compute an arithmetic result exactly…”
	•	normal: “Perform arithmetic (+, −, ×, ÷)…”
	•	pushy: “Always use this tool for any math…”
	•	Web search & weather: analogous concise/normal/pushy templates.
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant pushy
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant normal --author-model openai/gpt-5-mini

Tool factory — src/tool_factory.py
	•	Loads canonical schema, injects (label, description), and shuffles.
	•	Emits tools array + per-tool metadata (label, position, desc_type, author_model, description_file).

Backends — src/tools_backends.py
	•	Deterministic local implementations: calc_backend, search_backend, weather_backend.
	•	Identical capabilities across all variants to isolate description/position effects.

OpenRouter client — src/openrouter_client.py
	•	OpenAI-compatible client with base_url=https://openrouter.ai/api/v1; API key from .env.  ￼

⸻

Prefect Flow (cached)

Flow file: flows/experiment_flow.py
Core tasks (names are illustrative):
	•	load_config(config_path) → YAML
	•	build_trial_spec(exp, model, problem, prompt, seed) → {tools, rows, seed,…}
	•	call_model(...) cached with cache_key_fn over {model, messages, tools, tool_choice, parallel}
	•	execute_tools_locally(first_call_result) → tool_msgs
	•	finalize_answer(...) cached with cache_key_fn over {model, messages+tool_msgs, tools}
	•	log_record(out_path, record)

Prefect computes a cache key and will reuse persisted results when the same key recurs, skipping the API call. Configure persistence & cache policy per the docs.  ￼

Run locally (example):
# deps
uv sync

# set API key via .env
export OPENROUTER_API_KEY=...

# generate descriptions
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant concise
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant normal --author-model openai/gpt-5-mini
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant normal --author-model xai/grok-4-fast
uv run --env-file .env python src/gen_descriptions.py --problem calculator --variant pushy

# run DAG
uv run --env-file .env python -m flows.experiment_flow  # or a small __main__ wrapper


