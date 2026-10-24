import asyncio

from player import FakePlayer
from ports import silence_pcm
from providers.fake import FakeAsr, FakeLlm, FakeTts
from session import Session


class GatedTts:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        return silence_pcm()


def test_new_turn_drops_audio_from_the_previous_turn():
    async def scenario() -> None:
        tts = GatedTts()
        player = FakePlayer()
        session = Session(FakeAsr(), FakeLlm("你好。后面还有一句。"), tts, player)
        first = asyncio.create_task(session.submit_text("第一轮"))
        await asyncio.wait_for(tts.started.wait(), timeout=2)
        await asyncio.wait_for(session.submit_text("第二轮"), timeout=2)
        await asyncio.wait_for(first, timeout=2)
        assert player.played
        assert all(item.generation != 1 for item in player.played)
        assert player.stop_count >= 1
        assert all(item["text"] != "第一轮" for item in session.history)
        assert session.history[0]["text"] == "第二轮"

    asyncio.run(scenario())


def test_finished_turn_keeps_the_reply_that_started_playback():
    async def scenario() -> None:
        session = Session(FakeAsr("语音内容。"), FakeLlm("好。"), FakeTts(), FakePlayer())
        await session.submit_audio(b"\x00\x00")
        assert [item["text"] for item in session.history] == ["语音内容。", "好。"]

    asyncio.run(scenario())


def test_begin_text_returns_generation_and_new_text_cancels():
    async def scenario() -> None:
        tts = GatedTts()
        session = Session(
            FakeAsr(),
            FakeLlm("你好。后面还有一句。"),
            tts,
            FakePlayer(),
            system_prompt="你是助手。",
        )
        first_gen = await session.begin_text("第一轮")
        assert first_gen == 1
        await asyncio.wait_for(tts.started.wait(), timeout=2)
        second_gen = await session.begin_text("第二轮")
        assert second_gen == 2
        tts.release.set()
        await asyncio.wait_for(session._await_generation(2), timeout=2)

    asyncio.run(scenario())
