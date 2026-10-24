"""OpenLux TTS：默认走 OpenAI 兼容 /audio/speech（如 gpt-4o-mini-tts）。

Gemini 系 *-tts 仍走 generateContent。输出 24kHz 16bit 单声道 PCM。
"""

from __future__ import annotations

import base64
import io
import wave

import httpx


class OpenluxTts:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openlux.ai/v1",
        model: str = "tts-1",
        voice: str = "shimmer",
        speed: float = 1.0,
        instructions: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.speed = speed
        self.instructions = instructions
        self.base_url = base_url.rstrip("/")
        self.sample_rate = 24000

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        if _is_gemini_tts(self.model):
            return await self._gemini_synthesize(text)
        return await self._speech_synthesize(text)

    async def _speech_synthesize(self, text: str) -> bytes:
        body = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "speed": self.speed,
        }
        # instructions 只对 gpt-4o-mini-tts 生效，tts-1 会拒
        if self.instructions and "tts-1" not in self.model:
            body["instructions"] = self.instructions
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(
                f"{self.base_url}/audio/speech",
                headers=headers,
                json=body,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenLux TTS 失败 HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            raw = response.content
        return _to_pcm16(raw, self)

    async def _gemini_synthesize(self, text: str) -> bytes:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[:-3]
        url = f"{root}/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.voice}
                    }
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Gemini TTS 失败 HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            payload = response.json()
        try:
            part = payload["candidates"][0]["content"]["parts"][0]
            inline = part.get("inlineData") or part.get("inline_data") or {}
            raw = base64.b64decode(inline["data"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini TTS 响应异常: {payload!r}") from exc
        return _to_pcm16(raw, self)


# 兼容旧 import 名
OpenluxGeminiTts = OpenluxTts


def _is_gemini_tts(model: str) -> bool:
    lower = model.lower()
    return "gemini" in lower and "tts" in lower


def _to_pcm16(raw: bytes, tts: OpenluxTts) -> bytes:
    if raw[:4] != b"RIFF":
        # OpenAI pcm：24kHz 16bit 单声道裸 PCM
        tts.sample_rate = 24000
        return raw
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise RuntimeError(
                f"暂不支持的 WAV 格式: ch={handle.getnchannels()} "
                f"sw={handle.getsampwidth()}"
            )
        tts.sample_rate = handle.getframerate()
        return handle.readframes(handle.getnframes())
