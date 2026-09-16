"""长期记忆的本地 Embedding 生成。"""

from functools import lru_cache
import math
from typing import Any

from app.settings import Settings


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str, device: str) -> Any:
    """按模型和运行设备缓存本地 Sentence Transformers 模型。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 延迟导入运行时依赖，应用启动和未启用长期记忆时不加载 PyTorch。
    # 2. 首次调用时下载或加载本地模型，后续复用进程内模型实例。
    # 3. 导入或加载失败向调用方抛出，由长期记忆后台任务隔离故障。
    # =========================================================================
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required for long-term memory embeddings"
        ) from error
    return SentenceTransformer(model_name, device=device)


def embed_long_term_memory(content: str, settings: Settings) -> list[float]:
    """将规范化后的长期记忆正文编码为可写入 pgvector 的向量。

    Args:
        content: 已通过长期记忆服务校验的非空正文。
        settings: 包含模型、设备和预期维度的运行配置。
    Returns:
        与数据库向量维度一致的有限浮点数列表。
    Raises:
        ValueError: 正文为空、返回维度不匹配或含非有限值。
        RuntimeError: 本地 Embedding 依赖或模型不可用。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 拒绝空正文，避免为无业务含义的字符串创建语义索引。
    # 2. 复用缓存模型并只编码记忆正文，避免类型、证据等元数据干扰语义相似度。
    # 3. 严格校验维度和数值，防止错误模型或异常推理结果写入 VECTOR(768)。
    # =========================================================================
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("Long-term memory content cannot be empty")

    model = _load_embedding_model(
        settings.long_term_memory_embedding_model,
        settings.long_term_memory_embedding_device,
    )
    encoded = model.encode(normalized_content, normalize_embeddings=True)
    vector = [float(value) for value in encoded.tolist()]
    if len(vector) != settings.long_term_memory_embedding_dimensions:
        raise ValueError(
            "Embedding dimension does not match LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains non-finite values")
    return vector
