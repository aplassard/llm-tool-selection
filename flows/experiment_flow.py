"""Prefect flow to run config-driven tool selection experiments."""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from prefect import flow, get_run_logger, task

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.gen_descriptions import ensure_description_file
from src.openrouter_client import OpenRouterClient
from src.tool_factory import VariantSpec, build_tools
from src.tools_backends import ToolExecutionError, calculate_correctness, execute_problem_backend

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "experiments.yaml"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "intermediate"
DEFAULT_CACHE_DIR = DEFAULT_RESULTS_DIR / "cache"  # retained for CLI compatibility


@task
def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


@task
def ensure_descriptions(problem: str, description_generator: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = description_generator.get("variants", [])
    authors = description_generator.get("authored_by")
    combos: List[Dict[str, Any]] = []

    for variant in variants:
        if authors:
            for author in authors:
                ensure_description_file(problem, variant, author)
                combos.append({"variant": variant, "author_model": author})
        else:
            ensure_description_file(problem, variant, None)
            combos.append({"variant": variant, "author_model": None})
    return combos


@task
def build_trial_spec(
    experiment_id: str,
    model: str,
    problem: str,
    prompt_variant: Dict[str, Any],
    variant_specs: Iterable[Dict[str, Any]],
    trial_index: int,
    seed: int,
) -> Dict[str, Any]:
    variants = [VariantSpec(problem=problem, variant=item["variant"], author_model=item.get("author_model")) for item in variant_specs]
    tools_payload, offered = build_tools(problem, variants, seed=seed)

    offered_tools: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in offered:
        entry = {
            "label": item.label,
            "position": item.position,
            "desc_type": item.desc_type,
            "author_model": item.author_model,
            "description_path": item.description_path,
            "function_name": item.function_name,
        }
        offered_tools.append(entry)
        lookup[item.function_name] = entry

    trial_id = f"{experiment_id}-{model.replace('/', '_')}-{problem}-{trial_index:05d}"
    messages = [{"role": "user", "content": prompt_variant["user"]}]

    return {
        "trial_id": trial_id,
        "trial_index": trial_index,
        "seed": seed,
        "messages": messages,
        "tools": tools_payload,
        "offered_tools": offered_tools,
        "tool_lookup": lookup,
        "prompt_variant": prompt_variant,
        "model": model,
        "problem": problem,
        "experiment_id": experiment_id,
    }
@task
def call_model(
    model: str,
    messages: Any,
    tools: Any,
    tool_choice: Optional[Any],
    parallel_tool_calls: Optional[bool],
    cache_dir: str,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        with OpenRouterClient() as client:
            response = client.chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
    except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger = get_run_logger()
        status_code = getattr(exc, "status_code", None)
        body = exc.response.text if isinstance(exc, APIStatusError) and exc.response is not None else None
        request_id = getattr(exc, "request_id", None)
        logger.error(
            "call_model failure (status=%s request_id=%s): %s",
            status_code,
            request_id,
            body or str(exc),
        )
        record = {
            "error": {
                "type": exc.__class__.__name__,
                "status_code": status_code,
                "body": body,
                "message": str(exc),
                "request_id": request_id,
            },
            "latency_ms": latency_ms,
            "cache_hit": False,
        }
        return record

    latency_ms = (time.perf_counter() - start) * 1000
    meta = response.pop("_openrouter", None)
    record = {"response": response, "latency_ms": latency_ms, "cache_hit": False}
    if meta:
        record["openrouter_meta"] = meta
    return record


@task
def finalize_answer(
    model: str,
    messages: Any,
    tools: Any,
    tool_choice: Optional[Any],
    cache_dir: str,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        with OpenRouterClient() as client:
            response = client.chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
    except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger = get_run_logger()
        status_code = getattr(exc, "status_code", None)
        body = exc.response.text if isinstance(exc, APIStatusError) and exc.response is not None else None
        request_id = getattr(exc, "request_id", None)
        logger.error(
            "finalize_answer failure (status=%s request_id=%s): %s",
            status_code,
            request_id,
            body or str(exc),
        )
        record = {
            "error": {
                "type": exc.__class__.__name__,
                "status_code": status_code,
                "body": body,
                "message": str(exc),
                "request_id": request_id,
            },
            "latency_ms": latency_ms,
            "cache_hit": False,
        }
        return record

    latency_ms = (time.perf_counter() - start) * 1000
    meta = response.pop("_openrouter", None)
    record = {"response": response, "latency_ms": latency_ms, "cache_hit": False}
    if meta:
        record["openrouter_meta"] = meta
    return record


@task
def execute_tools(
    problem: str,
    first_call: Dict[str, Any],
    tool_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    if "response" not in first_call:
        return {"messages": [], "executions": [], "error": first_call.get("error")}
    response = first_call["response"]
    choices = response.get("choices", [])
    if not choices:
        return {"messages": [], "executions": []}
    assistant_message = choices[0].get("message", {})
    tool_calls = assistant_message.get("tool_calls", []) or []

    executions: List[Dict[str, Any]] = []
    tool_messages: List[Dict[str, Any]] = []

    for call in tool_calls:
        function_data = call.get("function", {})
        function_name = function_data.get("name")
        arguments_blob = function_data.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_blob)
        except json.JSONDecodeError as exc:
            arguments = {}
            error = f"Failed to parse arguments: {exc}"
            executions.append(
                {
                    "tool_call_id": call.get("id"),
                    "function_name": function_name,
                    "arguments": arguments_blob,
                    "result": None,
                    "error": error,
                    "metadata": tool_lookup.get(function_name),
                    "correct": False,
                }
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": function_name,
                    "content": json.dumps({"error": error}),
                }
            )
            continue

        try:
            backend_result = execute_problem_backend(problem, arguments)
            error_text = None
        except ToolExecutionError as exc:
            backend_result = None
            error_text = str(exc)

        if backend_result is not None:
            result_payload = backend_result.data
            correct = calculate_correctness(problem, backend_result)
        else:
            result_payload = {"error": error_text}
            correct = False

        executions.append(
            {
                "tool_call_id": call.get("id"),
                "function_name": function_name,
                "arguments": arguments,
                "result": result_payload,
                "error": error_text,
                "metadata": tool_lookup.get(function_name),
                "correct": correct,
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": function_name,
                "content": json.dumps(result_payload, ensure_ascii=False),
            }
        )

    return {"messages": tool_messages, "executions": executions}


@task
def log_record(results_dir: str, experiment_id: str, record: Dict[str, Any]) -> str:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    file_path = results_path / f"{experiment_id}.ndjson"
    with file_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(file_path)


def _select_prompts_for_problem(prompt_variants: List[Dict[str, Any]], problem: str) -> List[Dict[str, Any]]:
    candidates = [variant for variant in prompt_variants if variant.get("problem") in (None, problem)]
    if not candidates:
        raise ValueError(f"No prompt variants available for problem '{problem}'")
    return candidates


def _extract_assistant_message(first_call: Dict[str, Any]) -> Dict[str, Any]:
    response = first_call.get("response")
    if not response:
        return {"role": "assistant", "content": ""}
    choices = response.get("choices", [])
    if not choices:
        return {"role": "assistant", "content": ""}
    message = choices[0].get("message", {})
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }


@flow(name="experiment-flow")
def run_experiments(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    results_dir: str = str(DEFAULT_RESULTS_DIR),
    cache_dir: str = str(DEFAULT_CACHE_DIR),
    trial_limit: Optional[int] = None,
    experiments: Optional[List[str]] = None,
) -> None:
    logger = get_run_logger()
    config = load_config(config_path)
    selected = config.get("experiments", [])
    if experiments:
        selected = [exp for exp in selected if exp["id"] in set(experiments)]

    if not selected:
        logger.warning("No experiments selected")
        return

    for experiment in selected:
        experiment_id = experiment["id"]
        description_generator = experiment.get("description_generator", {})
        logger.info("Starting experiment %s", experiment_id)

        variant_specs_by_problem: Dict[str, Any] = {}
        for problem in experiment.get("problems", []):
            future = ensure_descriptions.submit(problem, description_generator)
            variant_specs_by_problem[problem] = future.result()

        for model in experiment.get("models", []):
            for problem in experiment.get("problems", []):
                prompt_variants = _select_prompts_for_problem(experiment.get("prompt_variants", []), problem)
                total_trials = experiment.get("trials", 0)
                if trial_limit is not None:
                    total_trials = min(total_trials, trial_limit)

                for trial_index in range(total_trials):
                    prompt_variant = prompt_variants[trial_index % len(prompt_variants)]
                    seed = secrets.randbits(32)

                    spec = build_trial_spec.submit(
                        experiment_id=experiment_id,
                        model=model,
                        problem=problem,
                        prompt_variant=prompt_variant,
                        variant_specs=variant_specs_by_problem[problem],
                        trial_index=trial_index,
                        seed=seed,
                    ).result()

                    call_result = call_model.submit(
                        model=model,
                        messages=spec["messages"],
                        tools=spec["tools"],
                        tool_choice=experiment.get("tool_choice"),
                        parallel_tool_calls=experiment.get("parallel_tool_calls"),
                        cache_dir=cache_dir,
                    ).result()

                    executions = execute_tools.submit(
                        problem=problem,
                        first_call=call_result,
                        tool_lookup=spec["tool_lookup"],
                    ).result()

                    if "response" in call_result:
                        assistant_message = _extract_assistant_message(call_result)
                        combined_messages = list(spec["messages"])
                        combined_messages.append(assistant_message)
                        combined_messages.extend(executions.get("messages", []))

                        final_result = finalize_answer.submit(
                            model=model,
                            messages=combined_messages,
                            tools=spec["tools"],
                            tool_choice=experiment.get("tool_choice"),
                            cache_dir=cache_dir,
                        ).result()
                    else:
                        final_result = {
                            "error": {
                                "type": "SkippedFinalize",
                                "message": "Skipped finalize due to first call error.",
                                "upstream_error": call_result.get("error"),
                            },
                            "latency_ms": None,
                            "cache_hit": False,
                        }

                    execution_list = executions.get("executions", []) if executions else []
                    first_execution = execution_list[0] if execution_list else None
                    metadata = first_execution.get("metadata") if first_execution else None
                    first_meta = call_result.get("openrouter_meta") if call_result else {}
                    final_meta = final_result.get("openrouter_meta") if final_result else {}
                    resolved_model = (
                        (first_meta or {}).get("resolved_model")
                        or (final_meta or {}).get("resolved_model")
                        or (call_result.get("response", {}).get("model") if call_result else None)
                        or spec["model"]
                    )
                    requested_model = (
                        (first_meta or {}).get("requested_model")
                        or (final_meta or {}).get("requested_model")
                        or spec["model"]
                    )
                    record = {
                        "trial_id": spec["trial_id"],
                        "trial_index": spec.get("trial_index"),
                        "seed": spec["seed"],
                        "experiment_id": spec["experiment_id"],
                        "model": spec["model"],
                        "resolved_model": resolved_model,
                        "requested_model": requested_model,
                        "problem": spec["problem"],
                        "prompt_variant_id": spec["prompt_variant"]["id"],
                        "tool_choice": experiment.get("tool_choice"),
                        "parallel_tool_calls": experiment.get("parallel_tool_calls"),
                        "offered_tools": spec["offered_tools"],
                        "selected_tool_label": metadata.get("label") if metadata else None,
                        "selected_position": metadata.get("position") if metadata else None,
                        "selected_desc_type": metadata.get("desc_type") if metadata else None,
                        "selected_author_model": metadata.get("author_model") if metadata else None,
                        "t_first_ms": call_result.get("latency_ms") if call_result else None,
                        "t_end_ms": final_result.get("latency_ms") if final_result else None,
                        "input_tokens": call_result.get("response", {}).get("usage", {}).get("prompt_tokens") if call_result else None,
                        "output_tokens": final_result.get("response", {}).get("usage", {}).get("completion_tokens") if final_result else None,
                        "correctness": first_execution.get("correct") if first_execution else None,
                        "first_call_cache_hit": call_result.get("cache_hit") if call_result else None,
                        "final_call_cache_hit": final_result.get("cache_hit") if final_result else None,
                        "first_call_error": call_result.get("error") if call_result else None,
                        "final_call_error": final_result.get("error") if final_result else None,
                        "first_openrouter_meta": first_meta or None,
                        "final_openrouter_meta": final_meta or None,
                    }

                    log_record.submit(results_dir, experiment_id, record).result()

    logger.info("Experiments finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experiment Prefect flow")
    parser.add_argument("--config", dest="config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--results", dest="results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--cache", dest="cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--trial-limit", dest="trial_limit", type=int, default=None)
    parser.add_argument("--experiment", dest="experiments", action="append", help="Filter experiments by id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        config_path=args.config_path,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        trial_limit=args.trial_limit,
        experiments=args.experiments,
    )


if __name__ == "__main__":
    main()
