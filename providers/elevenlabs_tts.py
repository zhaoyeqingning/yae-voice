"""ElevenLabs TTS 适配器。

ElevenLabs 是目前最自然的英文/多语种 TTS 服务，音色克隆能力极强。
官网：https://elevenlabs.io/
"""

from __future__ import annotations

import httpx


# 常用预设音色 ID（可在 ElevenLabs 后台查看更多）
ELEVENLABS_VOICES = {
    "rachel": "Rachel（温柔女声，美式英语）",
    "bella": "Bella（亲切女声）",
    "antoni": "Antoni（沉稳男声）",
    "elli": "Elli（年轻女声）",
    "josh": "Josh（自信男声）",
    "arnold": "Arnold（机械感男声）",
    "adam": "Adam（低沉旁白男声）",
    "sam": "Sam（清晰播报男声）",
}


class ElevenLabsTts:
    """ElevenLabs 语音合成。

    用法：
        tts = ElevenLabsTts(
            api_key="your-elevenlabs-api-key",
            voice_id="rachel",
        )
        pcm = await tts.synthesize("Hello! How are you?")
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        sample_rate: int = 24000,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        params = {
            "output_format": "pcm_" + str(self.sample_rate),
        }
        body = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        }

        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                json=body,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"ElevenLabs TTS 失败 HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            # ElevenLabs 的 pcm_24000 直接返回 16bit 单声道裸 PCM
            return response.content
