"""把一轮对话的字流 / 音频推到 SSE 订阅者。"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Callable

from ports import AudioBuffer


class TurnRelay:
    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def publish(self, event: dict[str, Any]) -> None:
        event = {**event, "generation": self.generation}
        await self.queue.put(event)

    async def close(self) -> None:
        await self.queue.put(None)


class Hub:
    def __init__(self) -> None:
        self._active: TurnRelay | None = None
        self._lock = asyncio.Lock()

    async def attach(self, generation: int) -> TurnRelay:
        async with self._lock:
            self._active = TurnRelay(generation)
            return self._active

    def on_user_text(self, text: str) -> None:
        active = self._active
        if active is None:
            return
        active.queue.put_nowait(
            {
                "type": "user.transcript",
                "generation": active.generation,
                "text": text,
            }
        )

    def on_text_delta(self, delta: str) -> None:
        active = self._active
        if active is None or not delta:
            return
        active.queue.put_nowait(
            {"type": "text.delta", "generation": active.generation, "delta": delta}
        )

    async def publish_active(self, event: dict[str, Any]) -> None:
        active = self._active
        if active is None:
            return
        if event.get("generation") not in (None, active.generation):
            return
        await active.publish(event)


class HubPlayer:
    """把 PCM 推给当前 turn 的 SSE，而不是本机声卡。"""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.stop_count = 0

    async def play(self, audio: AudioBuffer, on_started: Callable[[], None]) -> None:
        on_started()
        active = self.hub._active
        if active is None or active.generation != audio.generation:
            return
        await active.publish(
            {
                "type": "audio",
                "seq": audio.seq,
                "sample_rate": audio.sample_rate,
                "text": audio.text,
                "pcm_b64": base64.b64encode(audio.pcm).decode("ascii"),
            }
        )

    async def stop(self) -> None:
        self.stop_count += 1
