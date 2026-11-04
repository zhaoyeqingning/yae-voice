"""OpenAI 兼容的多模态音频输入 LLM（gpt-4o-audio-preview 等）。

直接把 PCM 音频 base64 编码后放进 messages，跳过 ASR，端到端一步到位。
需要模型支持 input_audio 输入格式（如 gpt-4o-audio-preview、gpt-4o-mini-audio）。
"""

from __future__ import annotations

import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OpenAiAudioLlm:
    """支持音频输入的 OpenAI 兼容 LLM。

    既支持纯文字 stream()，也支持 stream_audio() 直接吃 PCM。
    这样可以无缝替换原来的 OpenluxLlm，自动获得多模态能力。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-audio-preview",
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.enable_thinking = enable_thinking

    async def stream(
        self,
        history: list[dict[str, str]],
        user_text: str,
        *,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """纯文字流式对话（和 OpenluxLlm 接口一致）。"""
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            {
                "role": item["role"],
                "content": item.get("text") or item.get("content", ""),
            }
            for item in history
        )
        messages.append({"role": "user", "content": user_text})
        async for delta in self._stream_chat(messages):
            yield delta

    async def stream_audio(
        self,
        history: list[dict[str, str]],
        audio: bytes,
        *,
        audio_sample_rate: int = 16000,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """直接吃 PCM 音频，流式输出文字回复。

        音频会被打包成 WAV 再 base64 编码（OpenAI 要求带 WAV 头）。
        """
        wav_bytes = _pcm_to_wav(audio, audio_sample_rate)
        b64 = base64.b64encode(wav_bytes).decode("ascii")

        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        # 历史记录里的音频轮次暂时降级为文字（因为没存原始音频）
        for item in history:
            text = item.get("text") or item.get("content", "")
            if text:
                messages.append({"role": item["role"], "content": text})
        # 当前轮：音频 + 文字提示
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        )
        async for delta in self._stream_chat(messages):
            yield delta

    async def _stream_chat(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        """底层 SSE 流式请求。"""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            ) as response:
                if response.status_code >= 400:
                    detail = await response.aread()
                    raise RuntimeError(
                        f"Audio LLM 失败 HTTP {response.status_code}: "
                        f"{detail.decode('utf-8', 'ignore')[:400]}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    payload = json.loads(data)
                    delta = (payload.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """把裸 PCM 打包成带 WAV 头的字节流。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
