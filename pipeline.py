"""一轮对话：转写、切段、合成和播放重叠进行。"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from chunker import Chunker
from clock import Timing
from ports import SAMPLE_RATE, Asr, AudioBuffer, Llm, Player, Tts

# 字流回调：前端流式字幕用。不要在这里做慢活，否则会拖住读 LLM。
TextDeltaHandler = Callable[[str], None | Awaitable[None]]


@dataclass
class TurnRequest:
    generation: int
    history: list[dict[str, str]]
    user_text: str | None = None
    audio: bytes | None = None
    system_prompt: str = ""


@dataclass
class TurnOutcome:
    input: str  # "text" | "audio"
    user_text: str
    spoken_text: str
    assistant_text: str
    timing: Timing
    stale_audio_played: int = 0
    played_generations: list[int] = field(default_factory=list)
    tts_pieces: list[str] = field(default_factory=list)


async def run_turn(
    request: TurnRequest,
    *,
    asr: Asr,
    llm: Llm,
    tts: Tts,
    player: Player,
    is_current: Callable[[], bool],
    timing: Timing | None = None,
    on_text_delta: TextDeltaHandler | None = None,
    on_user_text: TextDeltaHandler | None = None,
) -> TurnOutcome:
    timing = timing or Timing(started_at=time.perf_counter())
    spoken: list[str] = []
    assistant_parts: list[str] = []
    played_generations: list[int] = []
    tts_pieces: list[str] = []
    user_text = request.user_text or ""
    input_kind = "audio" if request.audio is not None else "text"
    stale_audio_played = 0
    # speak_q：待合成的句子。无界，避免 TTS 慢时堵住继续读 LLM / 推字幕。
    speak_q: asyncio.Queue[str | None] = asyncio.Queue()
    # audio_q：已合成的 PCM。有界，限制未播音频占用的内存。
    audio_q: asyncio.Queue[AudioBuffer | None] = asyncio.Queue(maxsize=2)

    async def emit_text(delta: str) -> None:
        if not delta or on_text_delta is None:
            return
        result = on_text_delta(delta)
        if inspect.isawaitable(result):
            await result

    async def emit_user(text: str) -> None:
        if on_user_text is None:
            return
        result = on_user_text(text)
        if inspect.isawaitable(result):
            await result

    async def produce() -> None:
        nonlocal user_text
        try:
            if request.audio is not None:
                user_text = await asr.transcribe(request.audio)
                await emit_user(user_text)
            timing.asr_done_at = time.perf_counter()
            if not is_current():
                return
            if not (user_text or "").strip():
                print("输入为空，跳过本轮", flush=True)
                return
            chunker = Chunker()
            async for delta in llm.stream(
                list(request.history),
                user_text,
                system_prompt=request.system_prompt,
            ):
                if not is_current():
                    return
                assistant_parts.append(delta)
                await emit_text(delta)
                for piece in chunker.push(delta):
                    if timing.first_chunk_at is None:
                        # 第一个可送 TTS 的句子
                        timing.first_chunk_at = time.perf_counter()
                    await speak_q.put(piece)
            if is_current():
                for piece in chunker.finish():
                    if timing.first_chunk_at is None:
                        timing.first_chunk_at = time.perf_counter()
                    await speak_q.put(piece)
        finally:
            await speak_q.put(None)

    async def synthesize() -> None:
        seq = 0
        try:
            while True:
                piece = await speak_q.get()
                if piece is None or not is_current():
                    return
                tts_pieces.append(piece)
                pcm = await tts.synthesize(piece)
                if not is_current():
                    return
                if timing.audio_ready_at is None:
                    timing.audio_ready_at = time.perf_counter()
                rate = getattr(tts, "sample_rate", SAMPLE_RATE)
                await audio_q.put(
                    AudioBuffer(pcm, rate, request.generation, seq, piece)
                )
                seq += 1
        finally:
            await audio_q.put(None)

    async def play() -> None:
        nonlocal stale_audio_played
        while True:
            audio = await audio_q.get()
            if audio is None:
                return
            if audio.generation != request.generation or not is_current():
                continue

            def started() -> None:
                if timing.playback_started_at is None:
                    timing.playback_started_at = time.perf_counter()

            await player.play(audio, started)
            if audio.generation != request.generation or not is_current():
                stale_audio_played += 1
                continue
            spoken.append(audio.text)
            played_generations.append(audio.generation)

    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        group.create_task(synthesize())
        group.create_task(play())

    return TurnOutcome(
        input=input_kind,
        user_text=user_text,
        spoken_text="".join(spoken),
        assistant_text="".join(assistant_parts),
        timing=timing,
        stale_audio_played=stale_audio_played,
        played_generations=played_generations,
        tts_pieces=list(tts_pieces),
    )
