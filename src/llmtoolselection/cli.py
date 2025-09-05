from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from llmtoolselection.agents.tool_calling import select_weather_tool


def run_experiment(
    iterations: int,
    model: str = "openai/gpt-5-mini",
    threads: int = 1,
) -> List[dict]:
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(select_weather_tool, model=model) for _ in range(iterations)]
        return [f.result() for f in futures]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tool selection experiment")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    results = run_experiment(args.iterations, model=args.model, threads=args.threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(out_file)


if __name__ == "__main__":
    main()
