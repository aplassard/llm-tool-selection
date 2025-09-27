# Tool-Calling Evaluation

This project measures how large language models choose between functionally identical tools when their descriptions, authorship, labels, and list positions vary. We orchestrate experiments against OpenRouter using Prefect, collect detailed selection logs, and analyze biases with statistics.

## Experiments

All experiments are defined in `config/experiments.yaml` and executed via the Prefect flow `flows/experiment_flow.py`. Key arms:

- **desc_type_bias** – tests concise vs normal vs pushy copy for calculator, web search, and weather problems.
- **author_model_bias** – compares descriptions authored by different upstream models.
- **desc_x_model** – crosses description style with inference model to measure interaction effects.

Experiments currently run across four inference models:

- `openai/gpt-5-mini`
- `x-ai/grok-4-fast:free`
- `google/gemini-2.5-flash`
- `deepseek/deepseek-chat-v3.1`

Each tool description is generated ahead of time with `src/gen_descriptions.py` from templates, with authored variants for OpenAI, Grok, and Gemini. DeepSeek is evaluated as an inference model but does not author descriptions to keep the tool list within limits.

## Running the Flow

1. Install dependencies (uses [uv](https://github.com/astral-sh/uv)):
   ```bash
   UV_CACHE_DIR=./.uv-cache uv sync
   ```
2. Populate `.env` with your OpenRouter API key (and Prefect API settings if using the UI).
3. Run an experiment (example, 10 trials of desc_x_model):
   ```bash
   PREFECT_API_URL=http://127.0.0.1:4200/api \
   UV_CACHE_DIR=./.uv-cache uv run --env-file .env \
     python -m flows.experiment_flow --experiment desc_x_model --trial-limit 10
   ```
   Results land in `results/intermediate/<experiment>.ndjson`.

## Analysis Tools

- `analysis/run_logistic.py` fits a statsmodels logistic regression on tool-level records. Default formula:
  ```
  selected ~ C(desc_type) + C(author_model) + position + C(label) + C(model) + author_match
  ```
  Use it via:
  ```bash
  UV_CACHE_DIR=./.uv-cache uv run --env-file .env \
    python analysis/run_logistic.py --experiment desc_x_model \
    --inputs results/intermediate/desc_x_model.ndjson
  ```
- `analysis/descriptive_tables.py` prints counts of selections by requested model for position, label, description type, and author model:
  ```bash
  UV_CACHE_DIR=./.uv-cache uv run --env-file .env \
    python analysis/descriptive_tables.py --experiment desc_x_model \
    --inputs results/intermediate/desc_x_model.ndjson
  ```

## Recent Findings (desc_x_model sample)

Using 2,400 trials (9 tools per trial) collected across the four models, key observations:

- **Copy matters**: Pushy descriptions are ~5× more likely to be selected than concise ones; “normal” copy delivers ~1.4× the odds.
- **Author bias**: OpenAI-authored descriptions are favored (≈1.7× Gemini), while Grok-authored text underperforms (≈0.6×). Matching author to the inference model triples selection odds.
- **Position bias**: Every extra slot lower in the list reduces odds by ~16%, and label “A” is disproportionately preferred.
- **Model behavior**: DeepSeek selects tools more readily than the others, even after controlling for copy/position.
- **Descriptive counts**: Tables show, for example, DeepSeek and Gemini skew heavily toward early positions, while GPT-5 mini spreads into mid positions; author preference tables confirm each model over-indexes on its own prose.

These metrics provide a baseline for future mitigations (e.g., reordering, label normalization, author randomization) and can be extended with additional covariates or visualization notebooks.

## Repository Layout

- `flows/` – Prefect orchestration code.
- `config/experiments.yaml` – experiment definitions.
- `contracts/` – canonical tool schemas.
- `descriptions/` – generated tool descriptions per problem/variant/author.
- `src/` – tool factory, description generator, OpenRouter client, backends.
- `analysis/` – statistical scripts.
- `results/` – NDJSON logs (not tracked in git).

## Next Steps

- Parallelize trial execution (ConcurrentTaskRunner or async OpenAI client).
- Add notebook dashboards for mix effects, calibration, and visualization.
- Experiment with randomizing labels and positions more aggressively to isolate residual biases.

