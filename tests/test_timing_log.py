import asyncio
import json

from player import FakePlayer
from ports import silence_pcm
from providers.fake import FakeAsr, FakeLlm, FakeTts
from session import Session


def test_each_turn_appends_a_timing_line(tmp_path):
    async def scenario() -> None:
        log_path = tmp_path / "turns.jsonl"
        session = Session(
            FakeAsr(),
            FakeLlm("你好。我在。"),
            FakeTts(),
            FakePlayer(),
            log_path=log_path,
        )
        await session.submit_text("在吗")
        record = json.loads(log_path.read_text(encoding="utf-8"))
        assert record["cancelled"] is False
        assert record["stale_audio_played"] == 0
        assert record["input"] == "text"
        assert record["generation"] == 1
        assert record["user_text"] == "在吗"
        assert record["spoken_text"] == "你好。我在。"
        assert record["assistant_text"] == "你好。我在。"
        assert record["tts_pieces"] == ["你好。", "我在。"]
        assert record["listen_ms"] is not None
        assert record["think_ms"] is not None
        assert record["speak_ms"] is not None
        assert record["total_ms"] is not None

    asyncio.run(scenario())


class _GatedTts:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return silence_pcm()


def test_cancelled_turn_is_logged_without_stale_playback(tmp_path):
    async def scenario() -> None:
        log_path = tmp_path / "turns.jsonl"
        tts = _GatedTts()
        session = Session(
            FakeAsr(),
            FakeLlm("你好。后面还有一句。"),
            tts,
            FakePlayer(),
            log_path=log_path,
        )
        first = asyncio.create_task(session.submit_text("第一轮"))
        await asyncio.wait_for(tts.started.wait(), timeout=2)
        await asyncio.wait_for(session.submit_text("第二轮"), timeout=2)
        await asyncio.wait_for(first, timeout=2)
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        cancelled = [item for item in records if item["cancelled"]]
        finished = [item for item in records if not item["cancelled"]]
        assert len(cancelled) == 1
        assert cancelled[0]["stale_audio_played"] == 0
        assert len(finished) == 1
        assert finished[0]["user_text"] == "第二轮"

    asyncio.run(scenario())
