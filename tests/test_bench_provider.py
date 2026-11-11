import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_provider import measure


def test_tts_duration_uses_provider_sample_rate():
    class Tts:
        sample_rate = 16000

        async def synthesize(self, text):
            return b"\x00\x00" * 8000

    records = asyncio.run(measure(Tts(), "tts", "你好", 2))
    assert len(records) == 2
    assert records[0]["audio_seconds"] == 0.5
    assert records[0]["rtf"] >= 0


def test_asr_records_actual_transcript():
    class Asr:
        async def transcribe(self, audio):
            assert audio == b"pcm"
            return "你好"

    assert asyncio.run(measure(Asr(), "asr", b"pcm", 1))[0]["transcript"] == "你好"


def test_empty_tts_is_rejected():
    class Tts:
        sample_rate = 24000

        async def synthesize(self, text):
            return b""

    with pytest.raises(ValueError):
        asyncio.run(measure(Tts(), "tts", "你好", 1))
