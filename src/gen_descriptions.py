"""Generate and persist tool descriptions for experiment variants."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DESCRIPTION_ROOT = Path(__file__).resolve().parent.parent / "descriptions"


@dataclass(frozen=True)
class DescriptionEntry:
    problem: str
    variant: str
    author_model: Optional[str]
    description: str


_DESCRIPTION_TEMPLATES = {
    "calculator": {
        "concise": "Compute arithmetic results for addition, subtraction, multiplication, or division on two numbers with exact precision.",
        "normal": "Perform arithmetic (+, -, *, /) on two numeric inputs and return the exact numeric answer.",
        "pushy": "Always use this tool for math questions; it precisely handles addition, subtraction, multiplication, and division for two numbers.",
    },
    "web_search": {
        "concise": "Retrieve up to top_k relevant search snippets for the provided query.",
        "normal": "Perform a web search for the given query and return concise source snippets ranked by relevance.",
        "pushy": "Always route information lookups here; this tool aggressively searches the web and returns the most relevant snippets for the query.",
    },
    "weather": {
        "concise": "Return the current temperature for a location in the requested unit.",
        "normal": "Look up simple weather conditions: current temperature for a location in celsius or fahrenheit.",
        "pushy": "Always call this weather tool; it provides the definitive temperature for any location in the chosen unit.",
    },
}


def _author_suffix(author_model: Optional[str]) -> str:
    if not author_model:
        return ""
    safe = author_model.replace("/", "_").replace(":", "_")
    return f"__{safe}"


def description_path(problem: str, variant: str, author_model: Optional[str] = None) -> Path:
    return DESCRIPTION_ROOT / problem / f"{variant}{_author_suffix(author_model)}.json"


def _build_description(problem: str, variant: str, author_model: Optional[str]) -> DescriptionEntry:
    problem_templates = _DESCRIPTION_TEMPLATES.get(problem)
    if not problem_templates:
        supported = ", ".join(sorted(_DESCRIPTION_TEMPLATES))
        raise ValueError(f"Unsupported problem '{problem}'. Supported: {supported}")

    template = problem_templates.get(variant)
    if not template:
        supported = ", ".join(sorted(problem_templates))
        raise ValueError(f"Unsupported variant '{variant}' for problem '{problem}'. Supported: {supported}")

    if author_model:
        description = f"{template} Description authored by {author_model}."
    else:
        description = template

    return DescriptionEntry(problem=problem, variant=variant, author_model=author_model, description=description)


def ensure_description_file(problem: str, variant: str, author_model: Optional[str] = None, overwrite: bool = False) -> Path:
    entry = _build_description(problem, variant, author_model)
    path = description_path(problem, variant, author_model)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return path

    payload = {
        "problem": entry.problem,
        "variant": entry.variant,
        "author_model": entry.author_model,
        "description": entry.description,
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path


def load_description(problem: str, variant: str, author_model: Optional[str] = None) -> DescriptionEntry:
    path = description_path(problem, variant, author_model)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return DescriptionEntry(
        problem=payload["problem"],
        variant=payload["variant"],
        author_model=payload.get("author_model"),
        description=payload["description"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate description files for experiments.")
    parser.add_argument("--problem", required=True, help="Problem id (calculator, web_search, weather)")
    parser.add_argument("--variant", required=True, help="Description variant (concise, normal, pushy)")
    parser.add_argument("--author-model", dest="author_model", default=None, help="Optional authoring model id")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the file exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = ensure_description_file(args.problem, args.variant, args.author_model, overwrite=args.overwrite)
    print(str(path))


if __name__ == "__main__":
    main()
