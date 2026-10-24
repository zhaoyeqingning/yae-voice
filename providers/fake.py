"""没有密钥时使用的转写、模型和合成。"""

from __future__ import annotations

import asyncio

from ports import silence_pcm


class FakeAsr:
    def __init__(self, text: str = "今天天气不错。") -> None:
        self.text = text

    async def transcribe(self, audio: bytes) -> str:
        await asyncio.sleep(0)
        return self.text


class FakeLlm:
    def __init__(self, text: str = "你好。我在，刚才那句已经听到了。", delay: float = 0) -> None:
        self.text = text
        self.delay = delay
        self.done = False
        self.seen_user: str | None = None
        self.seen_history: list[dict[str, str]] | None = None
        self.seen_system: str | None = None

    async def stream(
        self,
        history: list[dict[str, str]],
        user_text: str,
        *,
        system_prompt: str = "",
    ):
        self.seen_user = user_text
        self.seen_history = list(history)
        self.seen_system = system_prompt
        try:
            for char in self.text:
                await asyncio.sleep(self.delay)
                yield char
        finally:
            self.done = True


class FakeTts:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.sample_rate = 24000

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        await asyncio.sleep(0)
        return silence_pcm()
