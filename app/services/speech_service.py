"""StepFun 语音识别和语音合成服务。"""

from __future__ import annotations

import re
import base64
import json

import httpx

from app.settings import Settings


class SpeechServiceError(Exception):
    """语音上游服务不可用或返回无效响应。"""


def _stepfun_headers(settings: Settings) -> dict[str, str]:
    """创建 StepFun 鉴权请求头。"""

    if settings.stepfun_api_key is None:
        raise SpeechServiceError("语音服务尚未配置")
    return {"Authorization": f"Bearer {settings.stepfun_api_key.get_secret_value()}"}


# 语音 转 文字，调用stepaudio的api
async def transcribe_wav(audio: bytes, settings: Settings) -> str:
    """调用 StepAudio 2.5 ASR，将 WAV 音频转写为文本。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 拒绝空音频和超过本地上限的请求，避免无效内容消耗外部配额。
    # 2. 使用 StepFun multipart 契约上传 WAV，并请求 JSON 格式以稳定读取 text。
    # 3. 上游失败或空转写统一转换为受控异常，调用方不暴露第三方响应细节。
    # =========================================================================

    if not audio:
        raise ValueError("录音文件不能为空")
    if len(audio) > settings.stepfun_asr_max_bytes:
        raise ValueError("录音文件超过允许大小")

    try:
        async with httpx.AsyncClient(timeout=settings.stepfun_audio_timeout_seconds) as client:
            response = await client.post(
                f"{settings.stepfun_base_url.rstrip('/')}/audio/transcriptions",
                headers=_stepfun_headers(settings),
                data={"model": settings.stepfun_asr_model, "response_format": "json"},
                files={"file": ("recording.wav", audio, "audio/wav")},
            )
            response.raise_for_status()
    except httpx.TimeoutException as error:
        raise SpeechServiceError("语音识别超时，请重试") from error
    except httpx.HTTPError as error:
        raise SpeechServiceError("语音识别服务暂时不可用") from error
    try:
        text = response.json().get("text")
    except ValueError as error:
        raise SpeechServiceError("语音识别服务返回无效结果") from error
    if not isinstance(text, str) or not text.strip():
        raise SpeechServiceError("未识别到有效语音")
    return text.strip()


# 以下部分为 文字 转 语音 的内容
def strip_markdown(text: str) -> str:
    """移除适合阅读但不适合播报的常见 Markdown 标记。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 先将链接保留为可读文本，避免把 URL 一起读出。
    # 2. 删除标题、列表、引用和强调等排版标记，保留原句与换行。
    # 3. 压缩多余空白，空文本由调用方拒绝。
    # =========================================================================
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|>|[-*+] |\d+\. )\s*", "", text)
    text = re.sub(r"[*_~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


## 把多个子句拼在一起，如果子句长度超过limit，就切断，剩余的保存到下一部分里
def split_speech_text(text: str, limit: int = 1000) -> list[str]:
    """按句子边界把播报文本分割为 StepFun 支持的长度。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 先按中文和英文句末符号切分，尽量保持自然停顿。
    # 2. 单句超长时按硬上限再切分，确保每个请求都符合 API 约束。
    # 3. 空白片段被丢弃，调用方据此判断是否可合成。
    # =========================================================================
    segments: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[。！？!?；;])", text):
        if not sentence:
            continue
        if len(current) + len(sentence) <= limit:
            current += sentence
            continue
        if current:
            segments.append(current)
        while len(sentence) > limit:
            segments.append(sentence[:limit])
            sentence = sentence[limit:]
        current = sentence
    if current:
        segments.append(current)
    return [segment for segment in segments if segment.strip()]


async def stream_speech(text: str, settings: Settings):
    """以 StepFun SSE 方式转发 PCM 音频分片，首片到达即可播放。"""

    clean_text = strip_markdown(text)
    segments = split_speech_text(clean_text)
    if not segments:
        raise ValueError("没有可播报的文本")
    headers = {**_stepfun_headers(settings), "Content-Type": "application/json", "Accept": "text/event-stream"}
    async with httpx.AsyncClient(timeout=settings.stepfun_audio_timeout_seconds) as client:
        for segment in segments:
            payload = {
                "model": settings.stepfun_tts_model,
                "voice": settings.stepfun_tts_voice,
                "input": segment,
                "response_format": "pcm",
                "stream_format": "sse",
                "sample_rate": 24000,
                "text_normalization": "enhanced",
            }
            if settings.stepfun_tts_language:
                payload["voice_label"] = {"language": settings.stepfun_tts_language}
            elif settings.stepfun_tts_emotion:
                payload["voice_label"] = {"emotion": settings.stepfun_tts_emotion}
            try:
                async with client.stream(
                    "POST",
                    f"{settings.stepfun_base_url.rstrip('/')}/audio/speech",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line[6:] == "[DONE]":
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "speech.audio.delta" and isinstance(event.get("audio"), str):
                            # Normalize and validate the upstream Base64 before forwarding.
                            base64.b64decode(event["audio"], validate=True)
                            yield event["audio"]
            except httpx.TimeoutException as error:
                raise SpeechServiceError("语音合成超时，请重试") from error
            except httpx.HTTPError as error:
                raise SpeechServiceError("语音合成服务暂时不可用") from error