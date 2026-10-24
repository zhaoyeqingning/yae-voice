"""浏览器上传的 WAV 转成讯飞要的 16kHz 16bit 单声道 PCM。"""

from __future__ import annotations

import io
import wave


def pcm16_from_wav(raw: bytes) -> bytes:
    if not raw:
        raise ValueError("空音频")
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("需要 16bit 单声道 WAV")
        if handle.getframerate() != 16000:
            raise ValueError("采样率需为 16000")
        pcm = handle.readframes(handle.getnframes())
    if len(pcm) < 3200:
        raise ValueError("录音太短")
    return pcm
