"""语音转写和播报接口。"""

import json
from uuid import UUID

from anyio import to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.agent import _DEFAULT_AUTH_DATA, _user_id_from_auth_data
from app.db.repositories import thread_repository
from app.services.speech_service import SpeechServiceError, stream_speech, transcribe_wav
from app.settings import get_settings


router = APIRouter(prefix="/api/threads", tags=["speech"])
_WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}


@router.post("/{thread_id}/transcription")
async def transcribe_thread_audio(thread_id: UUID, file: UploadFile = File(...)) -> dict[str, str]:
    """转写当前用户在线程内录制的 WAV 音频，但不创建聊天消息。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 检查线程归属，避免向未授权会话使用语音服务。
    # 2. 仅接受浏览器生成的 WAV，避免引入服务端转码和不受支持的上游格式。
    # 3. 读取受限大小的内容，调用 ASR 后仅返回文本，由前端等待用户确认发送。
    # =========================================================================
    user_id = _user_id_from_auth_data(_DEFAULT_AUTH_DATA)
    if file.content_type not in _WAV_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 WAV 录音文件")
    try:
        exists = await to_thread.run_sync(thread_repository.thread_exists_for_user, thread_id, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not exists:
        raise HTTPException(status_code=404, detail="Thread not found")
    settings = get_settings()
    audio = await file.read(settings.stepfun_asr_max_bytes + 1)
    if len(audio) > settings.stepfun_asr_max_bytes:
        raise HTTPException(status_code=413, detail="录音文件超过允许大小")
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="录音文件不是有效的 WAV 格式")
    try:
        return {"text": await transcribe_wav(audio, settings)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except SpeechServiceError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/{thread_id}/messages/{sequence}/speech/stream")
async def stream_message_speech(thread_id: UUID, sequence: int) -> StreamingResponse:
    """流式转发已持久化助手消息的 PCM 播报。"""

    if sequence < 1:
        raise HTTPException(status_code=404, detail="Message not found")
    user_id = _user_id_from_auth_data(_DEFAULT_AUTH_DATA)
    try:
        message = await to_thread.run_sync(thread_repository.load_message, thread_id, user_id, sequence)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if message is None or message.role != "assistant":
        raise HTTPException(status_code=404, detail="Assistant message not found")

    async def events():
        try:
            async for audio in stream_speech(message.content, get_settings()):
                yield f"event: audio\ndata: {audio}\n\n"
            yield "event: done\ndata: {}\n\n"
        except (SpeechServiceError, ValueError) as error:
            yield f"event: error\ndata: {{\"message\": {json.dumps(str(error), ensure_ascii=False)}}}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
