import asyncio

from pipeline import TurnRequest, run_turn
from player import FakePlayer
from providers.fake import FakeAsr, FakeLlm, FakeTts


def test_first_piece_is_synthesized_before_model_finishes():
    async def scenario() -> None:
        llm = FakeLlm("你好。这是还没说完的后半段。")
        tts = FakeTts()
        seen: dict[str, object] = {}

        async def synthesize(text: str) -> bytes:
            if "done" not in seen:
                seen["done"] = llm.done
                seen["text"] = text
            return await FakeTts.synthesize(tts, text)

        tts.synthesize = synthesize  # type: ignore[method-assign]
        outcome = await run_turn(
            TurnRequest(generation=1, history=[], user_text="在吗"),
            asr=FakeAsr(),
            llm=llm,
            tts=tts,
            player=FakePlayer(),
            is_current=lambda: True,
        )
        assert seen["text"] == "你好。"
        assert seen["done"] is False
        assert outcome.spoken_text == "你好。这是还没说完的后半段。"
        assert outcome.stale_audio_played == 0

    asyncio.run(scenario())


def test_audio_input_uses_the_transcript():
    async def scenario() -> None:
        llm = FakeLlm("好。")
        outcome = await run_turn(
            TurnRequest(generation=1, history=[], audio=b"\x00\x00"),
            asr=FakeAsr("语音内容。"),
            llm=llm,
            tts=FakeTts(),
            player=FakePlayer(),
            is_current=lambda: True,
        )
        assert llm.seen_user == "语音内容。"
        assert outcome.input == "audio"

    asyncio.run(scenario())


def test_text_deltas_keep_flowing_when_tts_is_slow():
    async def scenario() -> None:
        full = "一，二，三，四。"
        llm = FakeLlm(full)
        deltas: list[str] = []
        saw_full_during_first_tts = asyncio.Event()

        class SlowTts:
            sample_rate = 24000
            calls = 0

            async def synthesize(self, text: str) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    # TTS 卡住时，字流仍应读完 LLM（speak_q 无界）
                    await asyncio.sleep(0.15)
                    if "".join(deltas) == full:
                        saw_full_during_first_tts.set()
                return b"\x00\x00"

        outcome = await run_turn(
            TurnRequest(generation=1, history=[], user_text="说"),
            asr=FakeAsr(),
            llm=llm,
            tts=SlowTts(),
            player=FakePlayer(),
            is_current=lambda: True,
            on_text_delta=deltas.append,
        )
        assert saw_full_during_first_tts.is_set()
        assert outcome.assistant_text == full
        assert outcome.input == "text"

    asyncio.run(scenario())


def test_audio_transcript_is_reported():
    async def scenario() -> None:
        seen: list[str] = []
        await run_turn(
            TurnRequest(generation=1, history=[], audio=b"\x00\x00"),
            asr=FakeAsr("语音内容。"),
            llm=FakeLlm("好。"),
            tts=FakeTts(),
            player=FakePlayer(),
            is_current=lambda: True,
            on_user_text=seen.append,
        )
        assert seen == ["语音内容。"]

    asyncio.run(scenario())


def test_system_prompt_is_passed_to_llm():
    async def scenario() -> None:
        llm = FakeLlm("好。")
        await run_turn(
            TurnRequest(
                generation=1,
                history=[],
                user_text="在吗",
                system_prompt="你是测试角色。",
            ),
            asr=FakeAsr(),
            llm=llm,
            tts=FakeTts(),
            player=FakePlayer(),
            is_current=lambda: True,
        )
        assert llm.seen_system == "你是测试角色。"

    asyncio.run(scenario())
