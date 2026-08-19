"""最小 LangGraph Agent 的构造逻辑。"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from app.settings import Settings, get_settings


def _build_model(settings: Settings) -> BaseChatModel:
    """根据已校验配置创建聊天模型。

    逻辑规划：
    1. 从 Settings 读取模型地址、模型名、密钥、温度和超时配置。
    2. 将敏感密钥只传给模型客户端，不向调用方返回或暴露配置内容。
    3. 仅在配置了自定义地址时添加 base_url，最后创建并返回 ChatOpenAI 实例。
    """

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
    """创建只执行一次模型调用的最小 LangGraph Agent。

    Args:
        model: 可选的注入模型；传入后不再根据环境配置创建模型。
        settings: 可选的已校验配置；未传入模型时用于创建默认模型。
    Returns:
        接收 ``MessagesState`` 并返回模型消息的已编译图。
    Raises:
        pydantic.ValidationError: 配置缺失或格式不正确时抛出。
        Exception: 模型客户端创建失败时抛出。
    逻辑规划：
    1. 优先使用调用方注入的模型，便于测试和替换模型实现。
    2. 未注入模型时加载缓存配置，并通过 _build_model 创建模型。
    3. 构建“开始 -> 模型调用 -> 结束”的单节点图，不在此处处理记忆、工具和持久化。
    """
    chat_model = model or _build_model(settings or get_settings())

    async def call_model(state: MessagesState) -> dict[str, list[Any]]:
        """调用模型一次，并将模型回复追加到图状态。

        逻辑规划：
        1. 从状态中读取完整消息列表。
        2. 异步调用模型，避免阻塞 FastAPI 的流式响应。
        3. 将单条模型回复包装成 messages 增量返回给 LangGraph。
        """

        response = await chat_model.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    
    return graph.compile()
