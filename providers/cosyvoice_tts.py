"""阿里百炼 CosyVoice 流式合成（DashScope WebSocket）。

输出 24kHz 16bit 单声道 PCM，契约同 OpenluxTts。
音频可能藏在 JSON 的 base64 里，也可能直接是二进制帧，两种都处理。
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any

import websockets

DEFAULT_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
DEFAULT_MODEL = "cosyvoice-v2"
DEFAULT_VOICE = "longxiaochun_v2"
DEFAULT_SAMPLE_RATE = 24000
# cosyvoice-v2 支持的档位；默认不是 24k，写错会让播放端音调不对
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)


class CosyVoiceError(RuntimeError):
    """CosyVoice 返回的可读错误。"""


def build_run_task(
    *,
    task_id: str,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    text_type: str = "PlainText",
    volume: int = 50,
    rate: float = 1.0,
    pitch: float = 1.0,
) -> dict[str, Any]:
    """建合成任务。streaming 必须是 duplex，否则拿不到增量音频。"""
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": model,
            "parameters": {
                "text_type": text_type,
                "voice": voice,
                "format": "pcm",
                "sample_rate": sample_rate,
                "volume": volume,
                "rate": rate,
                "pitch": pitch,
            },
            "input": {},
        },
    }


def build_continue_task(task_id: str, text: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "continue-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": text}},
    }


def build_finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def parse_event(raw: str | bytes) -> dict[str, Any]:
    """解析一帧文本消息，返回 {event, audio, error}。

    error 非空时调用方应当直接抛出，避免把半截音频播出去。
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CosyVoiceError(f"无法解析服务端消息: {raw[:80]!r}") from exc
    if not isinstance(payload, dict):
        raise CosyVoiceError(f"服务端消息不是对象: {payload!r}")

    header = payload.get("header") or {}
    output = (payload.get("payload") or {}).get("output") or {}
    event = str(header.get("event") or "")

    audio = b""
    data = (output.get("audio") or {}).get("data")
    if data:
        audio = base64.b64decode(_pad(str(data)))

    code = header.get("error_code") or output.get("code")
    message = header.get("error_message") or output.get("message")
    error = None
    if event == "task-failed" or code:
        error = f"{code or 'unknown'}: {message or '未知错误'}"
    return {"event": event, "audio": audio, "error": error, "raw": payload}


class CosyVoiceTts:
    """一次调用一条 WebSocket：建任务、发文本、收音频、断开。

    每片都重新握手（几十毫秒）是这一版的简化；按轮复用连接留到后面。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rate: float = 1.0,
        volume: int = 50,
        pitch: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"CosyVoice 不支持 {sample_rate} Hz，可用 {SUPPORTED_SAMPLE_RATES}"
            )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.timeout = timeout
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        task_id = uuid.uuid4().hex
        chunks: list[bytes] = []
        async with websockets.connect(
            self.base_url,
            additional_headers={"Authorization": f"bearer {self.api_key}"},
            max_size=8 * 1024 * 1024,
        ) as ws:
            await ws.send(
                json.dumps(
                    build_run_task(
                        task_id=task_id,
                        model=self.model,
                        voice=self.voice,
                        sample_rate=self.sample_rate,
                        volume=self.volume,
                        rate=self.rate,
                        pitch=self.pitch,
                    ),
                    ensure_ascii=False,
                )
            )
            sent_text = False
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise CosyVoiceError("CosyVoice 合成超时")
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise CosyVoiceError("CosyVoice 合成超时") from exc
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    # 真机把 PCM 直接放在二进制帧里
                    chunks.append(bytes(raw))
                    continue
                event = parse_event(raw)
                if event["error"]:
                    raise CosyVoiceError(f"CosyVoice 失败 {event['error']}")
                if event["audio"]:
                    chunks.append(event["audio"])
                if event["event"] == "task-started" and not sent_text:
                    # 任务建好才喂文本；一次把这一片发完并收尾
                    await ws.send(
                        json.dumps(build_continue_task(task_id, text), ensure_ascii=False)
                    )
                    await ws.send(
                        json.dumps(build_finish_task(task_id), ensure_ascii=False)
                    )
                    sent_text = True
                elif event["event"] == "task-finished":
                    break

        pcm = b"".join(chunks)
        # 帧边界不保证 16bit 对齐，多出的单字节要丢掉，否则整段错位
        return pcm[:-1] if len(pcm) % 2 else pcm


def _pad(data: str) -> str:
    return data + "=" * (-len(data) % 4)

