"""Deterministic local tool backends for calculator, web search, and weather."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolResult:
    name: str
    data: Dict[str, Any]


class ToolExecutionError(Exception):
    """Raised when a backend encounters invalid arguments."""


def calc_backend(operand1: float, operand2: float, operation: str) -> ToolResult:
    if operation == "+":
        value = operand1 + operand2
    elif operation == "-":
        value = operand1 - operand2
    elif operation == "*":
        value = operand1 * operand2
    elif operation == "/":
        if operand2 == 0:
            raise ToolExecutionError("Division by zero is not allowed")
        value = operand1 / operand2
    else:
        raise ToolExecutionError(f"Unsupported operation '{operation}'")

    return ToolResult(
        name="calculator",
        data={"result": value, "operation": operation, "operand1": operand1, "operand2": operand2},
    )


def _hash_for_query(query: str) -> int:
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def search_backend(query: str, top_k: int = 3) -> ToolResult:
    if top_k < 1:
        raise ToolExecutionError("top_k must be at least 1")

    rng = random.Random(_hash_for_query(query))
    results: List[Dict[str, Any]] = []
    for index in range(top_k):
        snippet_seed = rng.randint(0, 10_000)
        results.append(
            {
                "title": f"Stub result {index + 1} for '{query}'",
                "snippet": f"Generated summary {snippet_seed} for '{query}'.",
                "url": f"https://example.com/{snippet_seed}",
            }
        )

    return ToolResult(name="web_search", data={"query": query, "top_k": top_k, "results": results})


_WEATHER_DATA = {
    "reykjavik": {"celsius": 4.2, "fahrenheit": 39.6},
    "new york": {"celsius": 15.3, "fahrenheit": 59.5},
    "san francisco": {"celsius": 12.0, "fahrenheit": 53.6},
}


def weather_backend(location: str, unit: str = "celsius") -> ToolResult:
    if unit not in ("celsius", "fahrenheit"):
        raise ToolExecutionError(f"Unsupported unit '{unit}'")

    normalized = location.strip().lower()
    record = _WEATHER_DATA.get(normalized)
    if record is None:
        # Deterministic fallback based on hash of the location.
        rng = random.Random(_hash_for_query(normalized))
        base_c = rng.uniform(-10, 35)
        record = {"celsius": base_c, "fahrenheit": base_c * 9 / 5 + 32}

    value = record[unit]
    value = round(value, 2)

    return ToolResult(name="weather", data={"location": location, "unit": unit, "value": value})


_BACKEND_BY_PROBLEM = {
    "calculator": calc_backend,
    "web_search": search_backend,
    "weather": weather_backend,
}


def execute_problem_backend(problem: str, arguments: Dict[str, Any]) -> ToolResult:
    backend = _BACKEND_BY_PROBLEM.get(problem)
    if not backend:
        raise ToolExecutionError(f"No backend registered for '{problem}'")

    try:
        return backend(**arguments)
    except TypeError as exc:  # invalid signature or arguments
        raise ToolExecutionError(str(exc)) from exc


def calculate_correctness(problem: str, tool_result: ToolResult) -> Optional[bool]:
    if problem != "calculator":
        return None

    data = tool_result.data
    operand1 = data.get("operand1")
    operand2 = data.get("operand2")
    operation = data.get("operation")
    result = data.get("result")

    if operation == "+":
        expected = operand1 + operand2
    elif operation == "-":
        expected = operand1 - operand2
    elif operation == "*":
        expected = operand1 * operand2
    elif operation == "/":
        expected = operand1 / operand2 if operand2 != 0 else math.inf
    else:
        return False

    return math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)
