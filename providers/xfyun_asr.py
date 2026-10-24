"""讯飞语音听写（流式 WebAPI）。输入 16kHz 16bit 单声道 PCM。"""

from __future__ import annotations

import asyncio
import base64
import json

import websockets

from providers.xfyun_auth import build_auth_url

HOST = "iat-api.xfyun.cn"
PATH = "/v2/iat"
FRAME_BYTES = 1280  # 40ms @ 16kHz 16bit mono
FRAME_SLEEP_S = 0.04  # 官方要求按近实时节奏送帧，过快易空结果


class XfyunAsr:
    def __init__(self, *, app_id: str, api_key: str, api_secret: str) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    async def transcribe(self, audio: bytes) -> str:
        if not audio:
            return ""
        url = build_auth_url(
            host=HOST, path=PATH, api_key=self.api_key, api_secret=self.api_secret
        )
        parts: list[str] = []
        done = asyncio.Event()
        error: list[Exception] = []

        async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:

            async def recv() -> None:
                try:
                    async for message in ws:
                        body = json.loads(message)
                        if body.get("code", 0) != 0:
                            raise RuntimeError(
                                f"讯飞 ASR 失败: {body.get('code')} {body.get('message')}"
                            )
                        data = body.get("data") or {}
                        result = data.get("result") or {}
                        ws_list = result.get("ws") or []
                        chunk = "".join(
                            cw.get("w", "")
                            for item in ws_list
                            for cw in item.get("cw") or []
                        )
                        if chunk:
                            parts.append(chunk)
                        if data.get("status") == 2:
                            break
                except Exception as exc:
                    error.append(exc)
                finally:
                    done.set()

            recv_task = asyncio.create_task(recv())
            try:
                frames = [
                    audio[i : i + FRAME_BYTES] for i in range(0, len(audio), FRAME_BYTES)
                ] or [b""]
                for index, frame in enumerate(frames):
                    status = 0 if index == 0 else 1
                    payload: dict = {
                        "data": {
                            "status": status,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": base64.b64encode(frame).decode("utf-8"),
                        }
                    }
                    if index == 0:
                        payload = {
                            "common": {"app_id": self.app_id},
                            "business": {
                                "language": "zh_cn",
                                "domain": "iat",
                                "accent": "mandarin",
                                "vad_eos": 2000,
                            },
                            "data": payload["data"],
                        }
                    await ws.send(json.dumps(payload))
                    if index + 1 < len(frames):
                        await asyncio.sleep(FRAME_SLEEP_S)
                    if error:
                        break
                # 结束帧
                await ws.send(
                    json.dumps(
                        {
                            "data": {
                                "status": 2,
                                "format": "audio/L16;rate=16000",
                                "encoding": "raw",
                                "audio": "",
                            }
                        }
                    )
                )
                await asyncio.wait_for(done.wait(), timeout=30.0)
            finally:
                if not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass
        if error:
            raise error[0]
        return "".join(parts).strip()
