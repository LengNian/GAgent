"""按需维护会话滚动摘要并组装模型上下文。"""

# ─────────────────────────────────────────────
# 一、模块导入
# ─────────────────────────────────────────────
import json                          # 把消息打包成 JSON 字符串发给模型
import logging                       # 记日志
from dataclasses import dataclass    # 定义轻量数据结构
from collections.abc import Sequence # 可迭代序列的类型标记
from typing import Any               # 任意类型
from uuid import UUID                # 会话唯一 ID 的类型

from anyio import to_thread         # 把阻塞的数据库操作丢到后台线程
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI  # 调用大模型

from app import database
from app.context import ContextCompiler, TokenCounter  # 按预算裁剪窗口
from app.database import StoredMessage, ThreadSummary   # 数据库里的消息/摘要结构
from app.prompt_loader import get_thread_summary_prompt # 读取摘要提示词
from app.settings import Settings


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 二、返回结果的数据结构
#   整个函数最后返回的就是它：最终要发给模型的消息，
#   外加一个标记表示本轮摘要有没有更新成功。
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class CompiledThreadContext:
    """最终模型上下文及本轮摘要是否成功推进。"""

    messages: list[BaseMessage]
    # 本轮是否真的/推进了摘要，用于读写数据库的判断
    summary_updated: bool


# ─────────────────────────────────────────────
# 三、带前缀的 token 计数器
#   核心目的：算 token 时把「摘要」也一并算进去，
#   这样裁剪窗口才不会超出总预算。
# ─────────────────────────────────────────────
class _PrefixedTokenCounter(TokenCounter):
    """将固定上下文一并纳入原文窗口的聊天模板计数。"""

    def __init__(self, token_counter: TokenCounter, prefixes: Sequence[SystemMessage]) -> None:
        self._token_counter = token_counter  # 真正的计数器，实际算 token 靠它
        self._prefixes = prefixes            # 要算进去的固定前缀（比如摘要）

    def count_text(self, text: str) -> int:
        """复用当前模型的正文 token 计算方式。"""
        return self._token_counter.count_text(text)

    def count_message(self, message: BaseMessage) -> int:
        """复用当前模型的单条正文 token 计算方式。"""
        return self._token_counter.count_message(message)

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        """计算固定摘要与候选原文共同使用的聊天模板 token。"""
        # 关键：把前缀拼到最前面一起算，于是裁剪窗口时已扣掉摘要占的空间
        return self._token_counter.count_messages([*self._prefixes, *messages])


# ─────────────────────────────────────────────
# 四、数据库消息 → 模型消息
# ─────────────────────────────────────────────
def _to_message(stored_message: StoredMessage) -> BaseMessage:
    """将持久化业务消息转换为 LangChain 消息。"""

    if stored_message.role == "user":
        return HumanMessage(content=stored_message.content)
    if stored_message.role == "assistant":
        return AIMessage(content=stored_message.content)
    # 不认识的角色直接报错，不信任外部输入
    raise ValueError(f"Unsupported stored message role: {stored_message.role}")


# ─────────────────────────────────────────────
# 五、构造「专门做摘要」的模型
#   温度设 0（要稳定可复现），且不绑定任何工具。
# ─────────────────────────────────────────────
# 创建进行摘要总结的模型
def _build_summary_model(settings: Settings) -> ChatOpenAI:
    """创建不绑定工具的摘要模型。"""

    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),  # 敏感字段要这样取
        "model": settings.llm_model,
        "temperature": 0,                                      # 摘要要稳定
        "timeout": settings.llm_timeout_seconds,
        "max_tokens": settings.context_summary_max_tokens,     # 摘要最大长度
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**model_kwargs)


# ─────────────────────────────────────────────
# 六、统一的窗口裁剪封装
#   把「按预算挑最近窗口」这件事统一起来。
#   token_counter 可以传普通版，也可以传带前缀的版本（见第三节）。
# ─────────────────────────────────────────────
def _compile_window(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    settings: Settings,
    token_counter: TokenCounter,
):
    """使用统一规则选择最近原文窗口。"""

    return ContextCompiler(
        max_tokens=max_tokens,
        max_message_tokens=settings.context_max_message_tokens,  # 单条上限
        max_messages=settings.context_max_messages,              # 条数上限
        token_counter=token_counter,                             # 关键开关
    ).compile(messages)


# ─────────────────────────────────────────────
# 七、真正调模型生成摘要
#   失败一律返回 None（摘要挂了也不能卡住用户对话）。
# ─────────────────────────────────────────────
async def _generate_summary(
    existing_summary: ThreadSummary | None,
    messages: list[StoredMessage],
    *,
    settings: Settings,
    model: ChatOpenAI | None = None,
) -> str | None:
    """根据旧摘要和新增旧消息生成新的完整摘要。"""

    # 把旧摘要 + 这批要摘要的消息打包成数据包
    payload = {
        "existing_summary": existing_summary.summary if existing_summary else "",
        "messages": [{"role": message.role, "content": message.content} for message in messages],
    }
    try:
        # 有外部模型就用外部的，否则现场造一个；异步调用
        response = await (model or _build_summary_model(settings)).ainvoke(
            [
                SystemMessage(content=get_thread_summary_prompt()),  # ① 摘要提示词
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),  # ② 数据包
            ]
        )
    except Exception:
        logger.exception("Thread summary generation failed")
        return None
    content = getattr(response, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None

    print("\n*********************Summary********************************")
    print(content.strip())
    print("************************************************************\n")

    return content.strip()


# ─────────────────────────────────────────────
# 八、把摘要包装成系统消息 / 造一条占位预算消息
# ─────────────────────────────────────────────
# 将已生成的摘要包装成一条系统信息
def _summary_context_message(summary: ThreadSummary) -> SystemMessage:
    """将已验证摘要标注为上下文，而不是 Agent 系统 Prompt。"""

    return SystemMessage(
        content=f"会话历史摘要：\n{summary.summary}",
        # 打标记：这是上下文而不是指令，防止摘要里混入指令影响模型
        additional_kwargs={"context_kind": "thread_summary"},
    )


def _summary_budget_message(settings: Settings, token_counter: TokenCounter) -> SystemMessage:
    """构造摘要最大输出预算的占位消息，用于预留完整模板空间。"""

    # 用「中」字重复若干次（每汉字=1 token，约等于占满摘要预算）做占位
    content = "中" * settings.context_summary_max_tokens
    while token_counter.count_text(content) < settings.context_summary_max_tokens:
        content += content
    return SystemMessage(content=f"会话历史摘要：\n{content}")


# ─────────────────────────────────────────────
# 九、主入口：compile_thread_context
#   调用方就是来调它的。流程：
#   1) 把历史分成「已摘要 / 未摘要」
#   2) 试着把未摘要原文塞进预算，放得下就直接返回、不调模型
#   3) 放不下 → 把最旧被挤出的那批拿去生成摘要、写库、推进覆盖序号
#   4) 最后把「真实摘要 + 最近原文」拼好返回
# ─────────────────────────────────────────────
async def compile_thread_context(
    thread_id: UUID,
    user_id: str,
    stored_messages: list[StoredMessage],
    summary: ThreadSummary | None,
    *,
    settings: Settings,
    token_counter: TokenCounter,
) -> CompiledThreadContext:
    """必要时更新摘要，并返回符合总预算的最终业务上下文。"""

    # covered_to_seq：摘要已覆盖到的最大消息序号（没有摘要就当 0）
    covered_to_seq = summary.covered_to_seq if summary else 0
    # 所有序号比它还大的消息 = 还没被摘要过的
    unsummarized = [message for message in stored_messages if message.seq > covered_to_seq]
    # 转成模型能用的消息对象
    raw_messages = [_to_message(message) for message in unsummarized]
    # 如果有旧摘要，就把它包成一条前缀；没有就空列表
    context_prefixes = [_summary_context_message(summary)] if summary is not None else []
    # 前缀 + 原文 拼在一起，代表完整上下文长什么样
    context_messages = [*context_prefixes, *raw_messages]
    # 算一下当前这套上下文一共占多少 token
    context_tokens = token_counter.count_messages(context_messages)

    # 临时查看摘要触发余量时，只需注释或取消注释这一处输出。
    print(
        "=== context budget ===\n"
        f"tokens: {context_tokens}/{settings.context_max_tokens}, "
        f"remaining: {settings.context_max_tokens - context_tokens}\n"
        f"unsummarized messages: {len(raw_messages)}/{settings.context_max_messages}, "
        f"remaining: {settings.context_max_messages - len(raw_messages)}\n"
        "======================",
        flush=True,
    )

    # 有前缀就用带前缀的计数器（把摘要占的空间算进裁剪），否则用普通计数器
    preliminary_counter = _PrefixedTokenCounter(token_counter, context_prefixes) if context_prefixes else token_counter

    # 第一次裁剪：用未摘要的原文
    preliminary = _compile_window(
        raw_messages,
        max_tokens=settings.context_max_tokens,
        settings=settings,
        token_counter=preliminary_counter,
    )

    # 关键判断：这次裁剪「丢消息了」→ 说明放不下，必须触发摘要
    needs_summary = preliminary.dropped_message_count > 0
    active_summary = summary        # 先假设沿用旧摘要
    summary_updated = False         # 先假设本轮没更新

    if needs_summary:
        # 造一条「占满摘要预算」的假前缀，用来算原文还能留几条
        summary_budget_prefixes = [_summary_budget_message(settings, token_counter)]
        summary_budget_counter = _PrefixedTokenCounter(
            token_counter,
            summary_budget_prefixes,
        )

        # 第二次裁剪：给摘要留满预算后，原文还能留几条
        reduced_window = _compile_window(
            raw_messages,
            max_tokens=settings.context_max_tokens,
            settings=settings,
            token_counter=summary_budget_counter,
        )

        # 未摘要数 - reduced_window 条数
        summarized_count = len(unsummarized) - len(reduced_window.messages)
        # 这批最旧、即将被挤掉的消息 = 真正要拿去摘要的
        messages_to_summarize = unsummarized[:summarized_count]

        if messages_to_summarize:
            generated_summary = await _generate_summary(
                summary,
                messages_to_summarize,
                settings=settings,
            )
            if generated_summary is not None:
                # 覆盖序号推进到这批消息的最后一条
                covered_to_seq = messages_to_summarize[-1].seq
                summary_token_count = token_counter.count_text(generated_summary)
                # 把数据库写入丢到后台线程，不阻塞异步主流程
                persisted = await to_thread.run_sync(
                    database.upsert_thread_summary,
                    thread_id,
                    user_id,
                    generated_summary,
                    covered_to_seq,
                    summary_token_count,
                )
                if persisted:
                    active_summary = ThreadSummary(
                        generated_summary,
                        covered_to_seq,
                        (summary.summary_version + 1) if summary else 1,  # 版本号 +1
                        summary_token_count,
                    )
                    summary_updated = True
        # 本轮模型用的原文窗口 = 给摘要留满预算后的窗口
        raw_window = reduced_window
    else:
        # 容量够、不用摘要，直接用初步窗口
        raw_window = preliminary

    if active_summary is not None:
        if summary_updated:
            # 本轮刚更新了摘要，用新覆盖序号重新算未摘要消息（窗口起点前移）
            raw_messages = [
                _to_message(message)
                for message in stored_messages
                if message.seq > active_summary.covered_to_seq
            ]
        # 用「真实摘要」做前缀，再裁一次窗口（这次用真实摘要长度，确保不超预算）
        final_prefixes = [_summary_context_message(active_summary)]
        context_counter = _PrefixedTokenCounter(token_counter, final_prefixes)
        raw_window = _compile_window(
            raw_messages,
            max_tokens=settings.context_max_tokens,
            settings=settings,
            token_counter=context_counter,
        )
        # 最终 = 真实摘要前缀 + 最近原文窗口
        return CompiledThreadContext(
            [*final_prefixes, *raw_window.messages],
            summary_updated,
        )
    # 全程没有摘要（比如首轮对话），直接返回 前缀(空) + 原文窗口
    return CompiledThreadContext([*context_prefixes, *raw_window.messages], summary_updated)
