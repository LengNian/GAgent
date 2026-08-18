"""Factory for the minimal LangGraph agent."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from app.settings import Settings, get_settings


def _build_model(settings: Settings) -> BaseChatModel:
    """Construct the configured chat model without exposing configuration to callers."""

    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url

    return ChatOpenAI(**model_kwargs)


def create_agent(
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
):
    """Create a minimal one-step LangGraph agent.

    The optional model argument allows callers and tests to inject a chat model.
    Memory, tools, persistence, and HTTP transport are intentionally outside this
    first framework boundary.
    """

    chat_model = model or _build_model(settings or get_settings())

    def call_model(state: MessagesState) -> dict[str, list[Any]]:
        response = chat_model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    
    return graph.compile()
