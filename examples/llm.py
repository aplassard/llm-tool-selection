from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/llm-tool-selection",
            "X-Title": "llm-tool-selection",
        },
    )
    model = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello, world"}],
        max_tokens=20,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
