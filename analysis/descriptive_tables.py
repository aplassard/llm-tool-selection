"""Produce descriptive selection tables grouped by requested model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

DEFAULT_INPUT_GLOB = "results/intermediate/*.ndjson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate descriptive selection statistics by model.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[DEFAULT_INPUT_GLOB],
        help="Paths or glob patterns pointing to NDJSON result files (default: results/intermediate/*.ndjson).",
    )
    parser.add_argument(
        "--experiment",
        help="Optional experiment_id filter (e.g., desc_x_model).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write CSV summary (per table appended to one file).",
    )
    return parser.parse_args()


def read_selected_rows(patterns: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for path in _iter_files(patterns):
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                selected = record.get("selected_tool_label")
                if not selected:
                    continue
                rows.append(
                    {
                        "experiment_id": record.get("experiment_id"),
                        "requested_model": record.get("requested_model") or record.get("model"),
                        "resolved_model": record.get("resolved_model"),
                        "selected_label": selected,
                        "selected_position": record.get("selected_position"),
                        "selected_desc_type": record.get("selected_desc_type"),
                        "selected_author_model": record.get("selected_author_model"),
                    }
                )
    if not rows:
        raise ValueError("No selected tools found in provided files.")
    return pd.DataFrame(rows)


def _iter_files(patterns: Sequence[str]) -> Iterable[str]:
    for pattern in patterns:
        path_obj = Path(pattern)
        if path_obj.exists():
            yield str(path_obj)
            continue
        matches = sorted(Path().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        for match in matches:
            yield str(match)


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    table = (
        df.groupby(["requested_model", value_col])
        .size()
        .reset_index(name="count")
        .pivot(index="requested_model", columns=value_col, values="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    return table


def main() -> None:
    args = parse_args()
    df = read_selected_rows(args.inputs)
    if args.experiment:
        df = df[df["experiment_id"] == args.experiment]
        if df.empty:
            raise ValueError(f"No rows match experiment_id={args.experiment}")

    tables = {
        "position": summarize(df, "selected_position"),
        "label": summarize(df, "selected_label"),
        "description_type": summarize(df, "selected_desc_type"),
        "author_model": summarize(df, "selected_author_model"),
    }

    for name, table in tables.items():
        print(f"\n=== {name.replace('_', ' ').title()} ===")
        print(table.to_string())

    if args.output:
        output = Path(args.output)
        with output.open("w", encoding="utf-8") as fp:
            for name, table in tables.items():
                fp.write(f"## {name}\n")
                table.to_csv(fp)
                fp.write("\n")
        print(f"Saved tables to {output}")


if __name__ == "__main__":
    main()
