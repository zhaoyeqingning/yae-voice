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


class MultimodalLlm(Llm, Protocol):
    """支持直接吃音频输入的多模态 LLM（跳过 ASR）。

    实现此接口后，pipeline 会在输入是音频时自动跳过 ASR 步骤，
    把 PCM 直接送给模型，降低一步延迟。
    """

    def stream_audio(
        self,
        history: list[dict[str, str]],
        audio: bytes,
        *,
        audio_sample_rate: int = 16000,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """直接吃音频 PCM，按增量吐出回复的字。"""


class Tts(Protocol):
    async def synthesize(self, text: str) -> bytes:
        """一小段文字进去，16-bit 单声道 PCM 出来。"""


class Player(Protocol):
    async def play(self, audio: AudioBuffer, on_started: Callable[[], None]) -> None:
        """开始播放时调用 on_started。"""

    async def stop(self) -> None:
        """立刻停掉当前声音。"""
