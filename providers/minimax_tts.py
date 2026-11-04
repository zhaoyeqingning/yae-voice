"""MiniMax TTS 适配器。"""
from __future__ import annotations

import base64
import io
import wave

import httpx


class MinimaxTts:
    def __init__(
        self,
        *,
        api_key: str,
        group_id: str,
        voice_id: str = "female-tianmei",
        speed: float = 1.0,
        pitch: int = 0,
        emotion: str = "neutral",
        model: str = "speech-2.8-hd",
        sample_rate: int = 24000,
    ) -> None:
        self.api_key = api_key
        self.group_id = group_id
        self.voice_id = voice_id
        self.speed = speed
        self.pitch = pitch
        self.emotion = emotion
        self.model = model
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={self.group_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": 1.0,
                "pitch": self.pitch,
                "emotion": self.emotion,
            },
            "text_type": "dialogue",
            "language_boost": "Chinese,Yue",
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "wav",
                "channel": 1,
            },
        }

        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"MiniMax TTS 失败 HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            payload = response.json()

        if payload.get("base_resp", {}).get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax TTS 业务错误: {payload.get('base_resp')}")

        audio = payload.get("data", {}).get("audio", "")
        if not audio:
            raise RuntimeError(f"MiniMax TTS 响应无音频数据: {payload!r}")

        raw = _decode_audio(audio)
        fallback_rate = (payload.get("extra_info") or {}).get("audio_sample_rate")
        return _to_pcm16(raw, self, fallback_rate=fallback_rate)


def _decode_audio(audio: str) -> bytes:
    audio = audio.strip()
    if len(audio) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in audio):
        return bytes.fromhex(audio)
    return base64.b64decode(audio)


def _to_pcm16(raw: bytes, tts: MinimaxTts, fallback_rate: int | None = None) -> bytes:
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as handle:
            nch = handle.getnchannels()
            sw = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        tts.sample_rate = rate
        if sw == 2 and nch == 1:
            return frames
        import numpy as np
        if sw == 1:
            arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
        elif sw == 2:
            arr = np.frombuffer(frames, dtype=np.int16)
        elif sw == 3:
            b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            arr = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) ^ 0x800000) - 0x800000
            arr = (arr >> 8).astype(np.int16)
        else:
            raise RuntimeError(f"MiniMax WAV 位深不支持: {sw * 8}bit")
        if nch > 1:
            arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
        return arr.tobytes()

    if fallback_rate:
        tts.sample_rate = int(fallback_rate)
        return raw

    raise RuntimeError(
        f"MiniMax 返回了无法识别的音频数据（前4字节 {raw[:8]!r}）"
    )
