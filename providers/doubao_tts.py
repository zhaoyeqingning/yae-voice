"""豆包语音合成 2.0（火山引擎双向流式 WebSocket）。

一次 synthesize 走完一条连接：建连 → 建会话 → 发文本 → 收音频 → 收尾，
音频在 AudioOnlyServer 的 payload 里；帧格式见 providers/doubao_protocol.py。
"""

from __future__ import annotations

import asyncio
import json
import uuid

import websockets

from providers.doubao_protocol import (
    EventType,
    Message,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
)

URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_SAMPLE_RATE = 24000
# 文档：speech_rate 取值 [-50, 100]，100 = 2.0 倍速，-50 = 0.5 倍速
MIN_SPEECH_RATE = -50
MAX_SPEECH_RATE = 100


class DoubaoTtsError(RuntimeError):
    """豆包返回的可读错误。"""


def speech_rate_from_ratio(ratio: float) -> int:
    """把 0.5~2.0 的倍速换成豆包的 [-50, 100]。"""
    clamped = min(2.0, max(0.5, ratio))
    return int(round((clamped - 1.0) * 100))


class DoubaoTts:
    def __init__(
        self,
        *,
        api_key: str,
        speaker: str,
        resource_id: str = DEFAULT_RESOURCE_ID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speech_rate: int = 0,
        loudness_rate: int = 0,
        context_texts: list[str] | None = None,
        url: str = URL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.speaker = speaker
        self.resource_id = resource_id
        self.sample_rate = sample_rate
        self.speech_rate = min(MAX_SPEECH_RATE, max(MIN_SPEECH_RATE, speech_rate))
        self.loudness_rate = min(100, max(-50, loudness_rate))
        self.context_texts = context_texts or []
        self.url = url
        self.timeout = timeout

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        session_id = str(uuid.uuid4())
        audio = bytearray()

        async with websockets.connect(
            self.url, additional_headers=headers, max_size=16 * 1024 * 1024
        ) as ws:
            await start_connection(ws)
            await self._wait(ws, {EventType.ConnectionStarted})

            req_params: dict[str, object] = {
                "speaker": self.speaker,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "speech_rate": self.speech_rate,
                    "loudness_rate": self.loudness_rate,
                },
            }
            if self.context_texts:
                req_params["context_texts"] = self.context_texts

            await start_session(
                ws,
                json.dumps(
                    {"req_params": req_params, "event": int(EventType.StartSession)}
                ).encode("utf-8"),
                session_id,
            )
            await self._wait(ws, {EventType.SessionStarted})

            await task_request(
                ws,
                json.dumps(
                    {
                        "req_params": {"text": text},
                        "event": int(EventType.TaskRequest),
                    }
                ).encode("utf-8"),
                session_id,
            )
            # 文本一次给完，直接收尾；等 SessionFinished 才知道音频发完了
            await finish_session(ws, session_id)

            while True:
                msg = await self._receive(ws)
                if msg.type == MsgType.AudioOnlyServer:
                    audio.extend(msg.payload)
                    continue
                if msg.type == MsgType.Error:
                    raise DoubaoTtsError(
                        f"豆包错误码 {msg.error_code}: "
                        f"{msg.payload.decode('utf-8', 'ignore')}"
                    )
                if msg.type == MsgType.FullServerResponse:
                    if msg.event in (EventType.SessionFailed, EventType.ConnectionFailed):
                        raise DoubaoTtsError(
                            f"豆包失败 {msg.event}: "
                            f"{msg.payload.decode('utf-8', 'ignore')}"
                        )
                    if msg.event == EventType.SessionFinished:
                        break

            await finish_connection(ws)
            await self._wait(ws, {EventType.ConnectionFinished})

        # 帧边界不保证 16bit 对齐，多出的单字节会让整段错位
        return bytes(audio[:-1] if len(audio) % 2 else audio)

    async def _receive(self, ws) -> Message:
        try:
            return await asyncio.wait_for(receive_message(ws), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise DoubaoTtsError("豆包合成超时") from exc
        except ValueError as exc:
            # 协议模块对异常帧会抛 ValueError（例如读写侧的字段顺序不一致）
            raise DoubaoTtsError(f"豆包消息解析失败: {exc}") from exc

    async def _wait(self, ws, events: set[int]) -> None:
        """容错版等待：允许中间夹 UsageResponse 之类的通知事件。"""
        while True:
            msg = await self._receive(ws)
            if msg.type == MsgType.Error:
                raise DoubaoTtsError(
                    f"豆包错误码 {msg.error_code}: "
                    f"{msg.payload.decode('utf-8', 'ignore')}"
                )
            if msg.event in events:
                return
            if msg.event in (EventType.SessionFailed, EventType.ConnectionFailed):
                raise DoubaoTtsError(
                    f"豆包失败 {msg.event}: "
                    f"{msg.payload.decode('utf-8', 'ignore')}"
                )

