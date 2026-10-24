import io
import wave

import pytest

from web_audio import pcm16_from_wav


def _wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buf.getvalue()


def test_pcm16_from_wav_roundtrip():
    pcm = b"\x01\x00" * 4000
    assert pcm16_from_wav(_wav(pcm)) == pcm


def test_short_wav_is_rejected():
    with pytest.raises(ValueError):
        pcm16_from_wav(_wav(b"\x00\x00" * 10))
