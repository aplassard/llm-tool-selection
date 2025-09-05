import os

import pytest

from llmtoolselection.agents.tool_calling import build_agent, select_weather_tool
from llmtoolselection.cli import run_experiment


def test_build_agent_missing_key(monkeypatch):
    monkeypatch.setattr(
        "llmtoolselection.agents.tool_calling.load_dotenv", lambda: None
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(RuntimeError):
        build_agent(model="openai/gpt-5-mini")


def test_select_weather_tool():
    info = select_weather_tool(model="openai/gpt-5-mini")
    assert info["name"].startswith("get_weather_")
    assert info["tags"]["model"] == ["codex"]


def test_run_experiment():
    results = run_experiment(2, model="openai/gpt-5-mini", threads=2)
    assert len(results) == 2
    assert all(r["name"].startswith("get_weather_") for r in results)
