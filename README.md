# llm-tool-selection

What impacts the tool selection for a model?

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

1. Create the virtual environment and install dependencies:

   ```bash
   uv sync
   ```

2. Provide your OpenAI-compatible API key in the environment. You can create a `.env` file or set the variables directly:

   ```bash
   export OPENAI_API_KEY=your_key_here
   export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
   export MODEL_NAME="openai/gpt-4o-mini"  # optional
   ```

3. Run the example LLM script:

   ```bash
   uv run examples/llm.py
   ```
