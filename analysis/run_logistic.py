"""Run logistic regression over experiment results using statsmodels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

DEFAULT_GLOB = "results/intermediate/*.ndjson"
DEFAULT_FORMULA = (
    "selected ~ C(desc_type) + C(author_model) + position + C(label) + C(model) + author_match"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit logistic regression to tool selection results.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[DEFAULT_GLOB],
        help="Paths or glob patterns for ndjson result files (default: results/intermediate/*.ndjson).",
    )
    parser.add_argument(
        "--experiment",
        help="Optional experiment_id filter (e.g., desc_x_model).",
    )
    parser.add_argument(
        "--problem",
        help="Optional problem filter (e.g., calculator).",
    )
    parser.add_argument(
        "--model",
        help="Optional inference model filter (e.g., openai/gpt-5-mini).",
    )
    parser.add_argument(
        "--formula",
        default=DEFAULT_FORMULA,
        help=f"Patsy/Statsmodels formula for the logistic regression (default: {DEFAULT_FORMULA}).",
    )
    return parser.parse_args()


def expand_records(paths: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for path in _iter_files(paths):
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows.extend(_expand_record(record))
    if not rows:
        raise ValueError("No rows were produced. Check input paths and filters.")
    df = pd.DataFrame(rows)
    df["author_model"] = df["author_model"].fillna("none")
    return df


def _iter_files(patterns: Sequence[str]) -> Iterable[str]:
    for pattern in patterns:
        path_obj = Path(pattern)
        if path_obj.exists():
            yield str(path_obj)
            continue
        matched = sorted(Path().glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        for match in matched:
            yield str(match)


def _expand_record(record: dict) -> List[dict]:
    offered = record.get("offered_tools") or []
    selected_label = record.get("selected_tool_label")
    rows: List[dict] = []
    for tool in offered:
        author_model = tool.get("author_model")
        resolved_model = record.get("resolved_model")
        author_match = 1 if author_model and resolved_model and author_model == resolved_model else 0
        rows.append(
            {
                "experiment_id": record.get("experiment_id"),
                "trial_id": record.get("trial_id"),
                "trial_index": record.get("trial_index"),
                "problem": record.get("problem"),
                "model": record.get("model"),
                "resolved_model": resolved_model,
                "requested_model": record.get("requested_model"),
                "prompt_variant_id": record.get("prompt_variant_id"),
                "position": tool.get("position"),
                "label": tool.get("label"),
                "desc_type": tool.get("desc_type"),
                "author_model": author_model,
                "author_match": author_match,
                "selected": 1 if tool.get("label") == selected_label else 0,
            }
        )
    return rows


def apply_filters(df: pd.DataFrame, experiment: str | None, problem: str | None, model: str | None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if experiment:
        mask &= df["experiment_id"] == experiment
    if problem:
        mask &= df["problem"] == problem
    if model:
        mask &= df["model"] == model
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError("No data remains after applying filters.")
    return filtered


def run_logit(df: pd.DataFrame, formula: str) -> sm.discrete.discrete_model.Logit:
    if not {0, 1}.issubset(set(df["selected"].unique())):
        raise ValueError("Logistic regression requires both selected and non-selected rows.")
    model = smf.logit(formula=formula, data=df)
    try:
        result = model.fit(disp=False)
    except PerfectSeparationError as exc:
        raise RuntimeError("Perfect separation detected; try simplifying the formula or filtering the data.") from exc
    return result


def main() -> None:
    args = parse_args()
    df = expand_records(args.inputs)
    df = apply_filters(df, args.experiment, args.problem, args.model)
    print(f"Loaded {len(df)} tool rows from {df['trial_id'].nunique()} trials")
    result = run_logit(df, args.formula)
    print("\nLogistic Regression Summary:\n")
    print(result.summary())
    print("\nParameter Estimates:\n")
    print(result.params.to_string())


if __name__ == "__main__":
    main()
