"""讯飞在线语音合成（流式 WebAPI）。输出 16kHz 16bit 单声道 PCM。"""

from __future__ import annotations

import base64
import json

import websockets

from providers.xfyun_auth import build_auth_url

HOST = "tts-api.xfyun.cn"
PATH = "/v2/tts"


class XfyunTts:
    def __init__(
        self,
        *,
        app_id: str,
        api_key: str,
        api_secret: str,
        vcn: str = "x4_xiaoyan",
    ) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.vcn = vcn
        self.sample_rate = 16000

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        url = build_auth_url(
            host=HOST, path=PATH, api_key=self.api_key, api_secret=self.api_secret
        )
        chunks: list[bytes] = []
        async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
            await ws.send(
                json.dumps(
                    {
                        "common": {"app_id": self.app_id},
                        "business": {
                            "aue": "raw",
                            "auf": "audio/L16;rate=16000",
                            "vcn": self.vcn,
                            "tte": "UTF8",
                            "speed": 50,
                            "volume": 50,
                            "pitch": 50,
                        },
                        "data": {
                            "status": 2,
                            "text": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
                        },
                    }
                )
            )
            async for message in ws:
                body = json.loads(message)
                if body.get("code", 0) != 0:
                    raise RuntimeError(f"讯飞 TTS 失败: {body.get('code')} {body.get('message')}")
                data = body.get("data") or {}
                audio = data.get("audio")
                if audio:
                    chunks.append(base64.b64decode(audio))
                if data.get("status") == 2:
                    break
        return b"".join(chunks)
