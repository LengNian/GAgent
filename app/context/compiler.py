"""按 token 预算编译发送给模型的短期上下文。"""

from dataclasses import dataclass
from math import ceil
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class TokenCounter:
    """为上下文预算提供保守的文本 token 估算。

    当前项目使用 OpenAI 兼容接口，但目标模型的 tokenizer 未随 SDK 暴露。
    因此这里对非 ASCII 字符按一个 token、ASCII 连续文本按每四个字符一个 token
    估算，并通过向上取整避免低估请求大小。后续接入供应商 tokenizer 时可替换此类。
    """

    def count_text(self, text: str) -> int:
        """估算文本 token 数；空文本返回 0。"""

        ascii_count = 0
        non_ascii_count = 0
        for character in text:
            if ord(character) < 128:
                ascii_count += 1
            else:
                non_ascii_count += 1
        return non_ascii_count + ceil(ascii_count / 4)

    def count_message(self, message: BaseMessage) -> int:
        """估算单条 LangChain 消息的 token 数。"""

        content = message.content
        if isinstance(content, str):
            return self.count_text(content)
        if isinstance(content, list):
            return self.count_text(
                "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
            )
        return self.count_text(str(content))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        """估算消息列表的 token 数。"""

        return sum(self.count_message(message) for message in messages)


@dataclass(frozen=True)
class ContextCompilation:
    """上下文编译结果及其统计信息。"""

    # 最终要送给模型的消息列表，按原始时间顺序
    messages: list[BaseMessage]
    # 这批消息送进模型后真实占用的 token 数
    estimated_tokens: int
    # 原始信息数量
    original_message_count: int
    # 丢弃的信息刷零
    dropped_message_count: int
    # 被截断的信息数量
    truncated_message_count: int
    # 被丢弃的用户消息数量
    dropped_user_message_count: int = 0
    # 被丢弃的助手消息数量
    dropped_assistant_message_count: int = 0


class ContextCompiler:
    """从完整历史中选择符合预算的最近对话上下文。"""

    def __init__(
        self,
        *,
        max_tokens: int,
        max_message_tokens: int,
        max_messages: int,
        token_counter: TokenCounter | None = None,
    ) -> None:

        # max_tokens: 总预算
        # max_message_tokens: 单条上限
        # max_messages: 条数上限

        if max_tokens < 1 or max_message_tokens < 1 or max_messages < 1:
            raise ValueError("context limits must be positive")

        self.max_tokens = max_tokens
        self.max_message_tokens = max_message_tokens
        self.max_messages = max_messages
        self.token_counter = token_counter or TokenCounter()

    def compile(self, messages: Sequence[BaseMessage]) -> ContextCompilation:
        """选择最近消息并限制总 token 数。

        逻辑规划：
        1. 校验输入消息，空列表直接返回空上下文。
        2. 从最新消息向前选择；user 是独立锚点，紧邻的 user/assistant 作为完整轮次。
        3. 候选用户消息或完整轮次放不下时立即停止，不回看更早消息；孤立 assistant 才允许跳过。
        4. 对最终选中的超长消息进行安全截断，再按原始顺序返回。
        5. 记录丢弃和截断数量，供日志与测试使用，不修改传入列表。
        """

        original_count = len(messages)

        if not messages:
            return ContextCompilation([], 0, 0, 0, 0)


        # 候选按最新到最旧保存；每项同时保留原始索引，便于最终统计丢弃数量。
        # (消息对象, 在原始列表中的下标, 是否被截断过)
        selected_entries: list[tuple[BaseMessage, int, bool]] = []
        # 累计选择的token
        estimated_tokens = 0
        remaining_messages = list(messages)
        # index 从最后一条向0递减
        index = len(remaining_messages) - 1

        while index >= 0:
            latest_message = remaining_messages[index]
            if isinstance(latest_message, AIMessage) and index > 0 and isinstance(
                remaining_messages[index - 1], HumanMessage
            ):
                candidate_messages = [remaining_messages[index - 1], latest_message]
                candidate_indices = [index - 1, index]
            elif isinstance(latest_message, HumanMessage):
                candidate_messages = [latest_message]
                candidate_indices = [index]
            else:
                # 孤立 assistant 或未知消息不会遮挡更早的用户意图。
                index -= 1
                continue

            if index == len(remaining_messages) - 1 and isinstance(latest_message, HumanMessage):
                # 当前问题一次满足单条和完整 chat template 预算，确保不会被历史挤掉。
                prepared_current, _ = self._fit_current_message(latest_message)
                prepared_messages = [prepared_current]
            else:
                prepared_messages = []
                for candidate in candidate_messages:
                    prepared, _ = self._fit_message(candidate)
                    prepared_messages.append(prepared)
            

            candidate_text_tokens = sum(
                self.token_counter.count_message(message) for message in prepared_messages
            )
            if (
                len(selected_entries) + len(prepared_messages) > self.max_messages
                or estimated_tokens + candidate_text_tokens > self.max_tokens
            ):
                break

            for prepared, original_index in reversed(list(zip(prepared_messages, candidate_indices))):
                selected_entries.append(
                    (prepared, original_index, prepared != remaining_messages[original_index])
                )
            estimated_tokens += candidate_text_tokens
            index -= len(candidate_indices)

        selected = [entry[0] for entry in reversed(selected_entries)]
        # 对快速筛选结果做完整模板校验；若超限，二分定位可保留的最近窗口。
        if self.token_counter.count_messages(selected) > self.max_tokens:
            lower_bound = 1
            upper_bound = len(selected_entries)
            retained_count = 1
            while lower_bound <= upper_bound:
                midpoint = (lower_bound + upper_bound) // 2
                candidate = [entry[0] for entry in reversed(selected_entries[:midpoint])]
                if self.token_counter.count_messages(candidate) <= self.max_tokens:
                    retained_count = midpoint
                    lower_bound = midpoint + 1
                else:
                    upper_bound = midpoint - 1

            # 从最新向最旧保存时，AIMessage 后面必须紧邻其对应 HumanMessage。
            if retained_count > 1 and isinstance(selected_entries[retained_count - 1][0], AIMessage):
                retained_count -= 1
            selected_entries = selected_entries[:retained_count]
            selected = [entry[0] for entry in reversed(selected_entries)]
        selected_indices = {original_index for _, original_index, _ in selected_entries}
        dropped_user_count = sum(
            isinstance(message, HumanMessage)
            for item, message in enumerate(remaining_messages)
            if item not in selected_indices
        )
        dropped_assistant_count = sum(
            isinstance(message, AIMessage)
            for item, message in enumerate(remaining_messages)
            if item not in selected_indices
        )
        selected_tokens = self.token_counter.count_messages(selected) if selected else 0


        # print("\n###########################################################")
        # print("tokens:", selected_tokens)
        # print(selected)
        # print("###########################################################\n")


        return ContextCompilation(
            messages=selected,
            estimated_tokens=selected_tokens,
            original_message_count=original_count,
            dropped_message_count=original_count - len(selected),
            truncated_message_count=sum(was_truncated for _, _, was_truncated in selected_entries),
            dropped_user_message_count=dropped_user_count,
            dropped_assistant_message_count=dropped_assistant_count,
        )



    # _fit_message和_fit_current_message两个截断方法
    def _fit_message(self, message: BaseMessage) -> tuple[BaseMessage, bool]:
        """
            将单条文本消息限制在消息预算内，保持消息类型和元数据。
            约束：正文 ≤ min(max_message_tokens, max_tokens)
        """

        if isinstance(message.content, str):
            content_text = message.content
        elif isinstance(message.content, list):
            content_text = "".join(
                block.get("text", "")
                for block in message.content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        else:
            content_text = str(message.content)

        # 预算上限
        message_limit = min(self.max_message_tokens, self.max_tokens)

        if self.token_counter.count_text(content_text) <= message_limit:
            return message, False

        marker = "[内容已截断]"
        if self.token_counter.count_text(marker) > message_limit:
            truncated = "".join(list(content_text)[:message_limit])
            return message.model_copy(update={"content": truncated}), True

        # token_counter 可能需要扫描完整前缀，因此用二分查找避免逐字符重复拼接。
        lower_bound = 0
        upper_bound = len(content_text)
        while lower_bound < upper_bound:
            midpoint = (lower_bound + upper_bound + 1) // 2
            candidate = content_text[:midpoint] + marker
            if self.token_counter.count_text(candidate) <= message_limit:
                lower_bound = midpoint
            else:
                upper_bound = midpoint - 1
        truncated = content_text[:lower_bound] + marker
        return message.model_copy(update={"content": truncated}), True


    def _fit_current_message(self, message: HumanMessage) -> tuple[HumanMessage, bool]:
        """
        一次满足当前问题的单条正文和完整聊天模板预算。
        约束：正文 ≤ message_limit 且 完整模板 ≤ max_tokens
        """

        if not isinstance(message.content, str):
            fitted, was_truncated = self._fit_message(message)
            if self.token_counter.count_messages([fitted]) > self.max_tokens:
                raise ValueError("max_tokens is too small for the current message template")
            return fitted, was_truncated  # type: ignore[return-value]

        message_limit = min(self.max_message_tokens, self.max_tokens)
        if (
            self.token_counter.count_text(message.content) <= message_limit
            and self.token_counter.count_messages([message]) <= self.max_tokens
        ):
            return message, False

        marker = "[内容已截断]"
        lower_bound = 0
        upper_bound = len(message.content)
        while lower_bound < upper_bound:
            midpoint = (lower_bound + upper_bound + 1) // 2
            candidate = message.model_copy(update={"content": message.content[:midpoint] + marker})
            if (
                self.token_counter.count_text(candidate.content) <= message_limit
                and self.token_counter.count_messages([candidate]) <= self.max_tokens
            ):
                lower_bound = midpoint
            else:
                upper_bound = midpoint - 1
        fitted = message.model_copy(update={"content": message.content[:lower_bound] + marker})
        if (
            self.token_counter.count_text(fitted.content) > message_limit
            or self.token_counter.count_messages([fitted]) > self.max_tokens
        ):
            raise ValueError("max_tokens is too small for the current message template")
        return fitted, True
