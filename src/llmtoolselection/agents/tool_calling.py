from __future__ import annotations

import json
import logging
import os
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(location: str) -> str:
    """Return the weather for a given location."""
    return f"The weather in {location} is sunny."


@dataclass
class ToolConfig:
    function: Callable[[str], str]
    tags: dict[str, Sequence[str]]


def load_tool_configs(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_weather_tool(name: str, description: str) -> Tool:
    def _inner(location: str) -> str:
        return get_weather(location)

    return Tool(name=name, func=_inner, description=description)


def build_agent(
    model: str | None = None,
    tools: Sequence[Callable] | None = None,
    return_intermediate_steps: bool = False,
) -> AgentExecutor:
    """Build an agent capable of calling the weather tool."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("Missing required environment variable: OPENAI_BASE_URL")

    llm = ChatOpenAI(
        model=model or os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/llm-tool-selection",
            "X-Title": "llm-tool-selection",
        },
    )
    tools = list(tools or [get_weather])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful weather assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, return_intermediate_steps=return_intermediate_steps
    )


def select_weather_tool(
    model: str | None = None,
    location: str = "Boston",
    config_path: Path | None = None,
) -> dict[str, object]:
    """Run the agent and return information about the selected tool."""
    config_path = config_path or Path(__file__).resolve().parent.parent / "tools" / "get_weather.yaml"
    configs = load_tool_configs(config_path)

    base_name = configs[0]["tool_name"] if configs else "get_weather"
    suffixes = list(string.ascii_uppercase)
    if len(configs) > len(suffixes):
        raise ValueError("Too many tool variants")
    random.shuffle(suffixes)

    tool_entries: list[ToolConfig] = []
    for cfg, suffix in zip(configs, suffixes):
        name = f"{base_name}_{suffix}"
        description = cfg["description"]
        tags = cfg.get("tags", {})
        func = _make_weather_tool(name, description)
        tool_entries.append(ToolConfig(function=func, tags=tags))

    random.shuffle(tool_entries)
    tools = [entry.function for entry in tool_entries]
    agent = build_agent(model=model, tools=tools, return_intermediate_steps=True)

    result = agent.invoke(
        {"input": f"What's the weather in {location}?"}, return_intermediate_steps=True
    )
    step = result["intermediate_steps"][0][0]
    name = step.tool
    position = next(i for i, entry in enumerate(tool_entries) if entry.function.name == name)
    tags = tool_entries[position].tags
    logging.info("Selected tool %s at position %s with tags %s", name, position, tags)
    return {"position": position, "name": name, "tags": tags}


def main() -> None:
    info = select_weather_tool()
    print(json.dumps(info))


if __name__ == "__main__":
    main()
