"""一轮会话。新的一轮会取消上一轮还没播完的内容。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from clock import Timing
from pipeline import TextDeltaHandler, TurnOutcome, TurnRequest, run_turn
from ports import Asr, Llm, Player, Tts


class Session:
    def __init__(
        self,
        asr: Asr,
        llm: Llm,
        tts: Tts,
        player: Player,
        log_path: Path | None = None,
        *,
        system_prompt: str = "",
        on_text_delta: TextDeltaHandler | None = None,
        on_user_text: TextDeltaHandler | None = None,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.player = player
        self.log_path = log_path
        # 人设只在初始化时设一次；history 策略后续再定
        self.system_prompt = system_prompt
        self.on_text_delta = on_text_delta
        self.on_user_text = on_user_text
        self.generation = 0
        self.history: list[dict[str, str]] = []
        self._task: asyncio.Task[None] | None = None
        self.last_error: str | None = None

    async def begin_text(self, text: str) -> int:
        """正式文字入口：启动一轮并立刻返回 generation；再发会 cancel 上一轮。"""
        return await self._begin(user_text=text)

    async def begin_audio(self, audio: bytes) -> int:
        return await self._begin(audio=audio)

    async def submit_text(self, text: str) -> int:
        """调试/CLI：begin 后等到本轮结束（或被更新一轮取消）。"""
        generation = await self.begin_text(text)
        await self._await_generation(generation)
        return generation

    async def submit_audio(self, audio: bytes) -> int:
        generation = await self.begin_audio(audio)
        await self._await_generation(generation)
        return generation

    async def _begin(
        self, user_text: str | None = None, audio: bytes | None = None
    ) -> int:
        self.generation += 1
        generation = self.generation
        self.last_error = None
        previous = self._task
        if previous is not None:
            previous.cancel()
            await _wait(previous)
        await self.player.stop()
        timing = Timing(started_at=time.perf_counter())
        self._task = asyncio.create_task(
            self._run(generation, user_text, audio, timing)
        )
        return generation

    async def _await_generation(self, generation: int) -> None:
        task = self._task
        if task is None or self.generation != generation:
            return
        try:
            await task
        except asyncio.CancelledError:
            if self.generation != generation:
                return
            raise

    async def _run(
        self,
        generation: int,
        user_text: str | None,
        audio: bytes | None,
        timing: Timing,
    ) -> None:
        outcome: TurnOutcome | None = None
        cancelled = False
        try:
            outcome = await run_turn(
                TurnRequest(
                    generation,
                    list(self.history),
                    user_text,
                    audio,
                    system_prompt=self.system_prompt,
                ),
                asr=self.asr,
                llm=self.llm,
                tts=self.tts,
                player=self.player,
                is_current=lambda: self.generation == generation,
                timing=timing,
                on_text_delta=self.on_text_delta,
                on_user_text=self.on_user_text,
            )
            if outcome.spoken_text and self.generation == generation:
                self.history.append({"role": "user", "text": outcome.user_text})
                self.history.append({"role": "assistant", "text": outcome.spoken_text})
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            # 网络/模型错误不炸死会话；网页侧靠 SSE error 提示
            print(f"turn {generation} failed: {type(exc).__name__}: {exc}", flush=True)
            self.last_error = str(exc)
        finally:
            self._append_log(generation, cancelled, user_text, audio, outcome, timing)

    def _append_log(
        self,
        generation: int,
        cancelled: bool,
        user_text: str | None,
        audio: bytes | None,
        outcome: TurnOutcome | None,
        timing: Timing,
    ) -> None:
        if self.log_path is None:
            return
        input_kind = (
            outcome.input
            if outcome is not None
            else ("audio" if audio is not None else "text")
        )
        record = {
            "generation": generation,
            "input": input_kind,
            "cancelled": cancelled,
            "user_text": outcome.user_text if outcome is not None else (user_text or ""),
            "spoken_text": outcome.spoken_text if outcome is not None else "",
            "assistant_text": outcome.assistant_text if outcome is not None else "",
            "tts_pieces": outcome.tts_pieces if outcome is not None else [],
            "stale_audio_played": outcome.stale_audio_played if outcome is not None else 0,
            **timing.snapshot(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _wait(task: asyncio.Task[None]) -> None:
    try:
        await task
    except (asyncio.CancelledError, Exception):
        # 上一轮失败（含 TaskGroup/网络错误）不能挡住新一轮
        return
