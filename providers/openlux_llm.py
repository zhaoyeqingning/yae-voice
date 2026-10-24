"""OpenLux OpenAI 兼容流式对话；默认关闭思考。"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


class OpenluxLlm:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
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
        messages: list[dict[str, str]] = []
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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        # 标准 Chat Completions（如 gpt-4o-mini）不认思考相关字段
        if self.enable_thinking:
            body["enable_thinking"] = True
            body["chat_template_kwargs"] = {
                "enable_thinking": True,
                "preserve_thinking": False,
            }
        elif _supports_thinking_knobs(self.model):
            body["enable_thinking"] = False
            body["chat_template_kwargs"] = {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
            body["reasoning_effort"] = "none"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        buffer = ""
        # trust_env=False：避开系统代理导致的 SSL EOF
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
                        f"LLM 失败 HTTP {response.status_code}: {detail.decode('utf-8', 'ignore')}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    payload = json.loads(data)
                    delta = (payload.get("choices") or [{}])[0].get("delta") or {}
                    # 只取最终内容，忽略 reasoning / thinking 字段
                    piece = delta.get("content") or ""
                    if not piece:
                        continue
                    buffer += piece
                    cleaned, buffer = _drain_visible(buffer)
                    if cleaned:
                        yield cleaned
        leftover = _THINK_RE.sub("", buffer).strip()
        if leftover:
            yield leftover


def _supports_thinking_knobs(model: str) -> bool:
    lower = model.lower()
    if lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3") or lower.startswith("o4"):
        return False
    if "claude" in lower:
        return False
    # 聚合池里的 Qwen / DeepSeek / Gemini 等常带思考开关
    return any(token in lower for token in ("qwen", "deepseek", "gemini", "glm", "doubao", "kimi"))


def _drain_visible(buffer: str) -> tuple[str, str]:
    """去掉已闭合的 think 块；未闭合时先攒着，避免把思考内容吐给 TTS。"""
    lower = buffer.lower()
    open_tag = lower.find("<think>")
    if open_tag < 0:
        return buffer, ""
    before = buffer[:open_tag]
    close_tag = lower.find("</think>", open_tag)
    if close_tag < 0:
        return before, buffer[open_tag:]
    after = buffer[close_tag + len("</think>") :]
    return before + _THINK_RE.sub("", after), ""
