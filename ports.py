"""流水线各段之间传递的音频和接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable, Protocol


SAMPLE_RATE = 24000


@dataclass(frozen=True)
class AudioBuffer:
    pcm: bytes
    sample_rate: int
    generation: int
    seq: int
    text: str


def silence_pcm(duration_ms: int = 10) -> bytes:
    frames = SAMPLE_RATE * duration_ms // 1000
    return b"\x00\x00" * frames


class Asr(Protocol):
    async def transcribe(self, audio: bytes) -> str:
        """整段录音进去，转写文字出来。"""


class Llm(Protocol):
    def stream(
        self,
        history: list[dict[str, str]],
        user_text: str,
        *,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """按增量吐出回复的字。system_prompt 由 Session 初始化时带入。"""


class Tts(Protocol):
    async def synthesize(self, text: str) -> bytes:
        """一小段文字进去，16-bit 单声道 PCM 出来。"""


class Player(Protocol):
    async def play(self, audio: AudioBuffer, on_started: Callable[[], None]) -> None:
        """开始播放时调用 on_started。"""

    async def stop(self) -> None:
        """立刻停掉当前声音。"""
