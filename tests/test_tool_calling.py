import os

import pytest

from llmtoolselection.agents.tool_calling import build_agent


def test_build_agent_missing_key(monkeypatch):
    monkeypatch.setattr(
        "llmtoolselection.agents.tool_calling.load_dotenv", lambda: None
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(RuntimeError):
        build_agent(model="openai/gpt-5-mini")


def test_weather_tool_call():
    agent = build_agent(model="openai/gpt-5-mini")
    result = agent.invoke({"input": "What's the weather in Boston?"})
    assert "boston" in result["output"].lower()
    assert "sunny" in result["output"].lower()
