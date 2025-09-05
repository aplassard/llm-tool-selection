from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(location: str) -> str:
    """Return the weather for a given location."""
    return f"The weather in {location} is sunny."


def build_agent(model: str | None = None) -> AgentExecutor:
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
    tools = [get_weather]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful weather assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def main() -> None:
    agent = build_agent()
    result = agent.invoke({"input": "What's the weather in Boston?"})
    print(result["output"])


if __name__ == "__main__":
    main()
