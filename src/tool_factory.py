"""Utilities to assemble tool payloads for experiments."""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .gen_descriptions import DescriptionEntry, load_description, description_path

CONTRACTS_ROOT = Path(__file__).resolve().parent.parent / "contracts"
_TOOL_LABELS = ["A", "B", "C", "D", "E"]


@dataclass(frozen=True)
class VariantSpec:
    problem: str
    variant: str
    author_model: Optional[str]


@dataclass
class OfferedTool:
    label: str
    position: int
    desc_type: str
    author_model: Optional[str]
    description_path: str
    function_name: str
    tool_payload: Dict[str, Any]


def _load_contract(problem: str) -> Dict[str, Any]:
    path = CONTRACTS_ROOT / f"{problem}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Contract not found for problem '{problem}' at {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _assign_labels(count: int, rng: random.Random) -> Sequence[str]:
    if count > len(_TOOL_LABELS):
        raise ValueError(f"Cannot assign labels for {count} tools; only {len(_TOOL_LABELS)} labels available")
    labels = _TOOL_LABELS[:count]
    labels = list(labels)
    rng.shuffle(labels)
    return labels


def build_tools(problem: str, variant_specs: Iterable[VariantSpec], seed: int) -> Tuple[List[Dict[str, Any]], List[OfferedTool]]:
    specs = list(variant_specs)
    if not specs:
        raise ValueError("variant_specs must contain at least one entry")

    rng = random.Random(seed)
    contract_template = _load_contract(problem)

    labels = _assign_labels(len(specs), rng)
    working: List[OfferedTool] = []

    for spec, label in zip(specs, labels):
        description_entry: DescriptionEntry = load_description(problem, spec.variant, spec.author_model)
        tool_payload = copy.deepcopy(contract_template)
        function_block = tool_payload["function"]
        function_block["name"] = f"{problem}_tool_{label.lower()}"
        function_block["description"] = description_entry.description
        offered = OfferedTool(
            label=label,
            position=0,
            desc_type=spec.variant,
            author_model=description_entry.author_model,
            description_path=str(description_path(problem, spec.variant, spec.author_model)),
            function_name=function_block["name"],
            tool_payload=tool_payload,
        )
        working.append(offered)

    rng.shuffle(working)
    for index, offered in enumerate(working, start=1):
        offered.position = index

    tools_payload = [item.tool_payload for item in working]
    return tools_payload, working
