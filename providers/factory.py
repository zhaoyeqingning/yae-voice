"""按 TTS_PROVIDER 选引擎；网页、live_smoke、live_mic 都从这里拿 TTS。"""

from __future__ import annotations

from providers.config import Settings
from ports import Tts

COSYVOICE_NAMES = {"cosyvoice", "cosy"}
DOUBAO_NAMES = {"doubao", "volc"}
OPENLUX_NAMES = {"openlux", "openai"}


def build_tts(settings: Settings) -> Tts:
    provider = (settings.tts_provider or "openlux").strip().lower()
    if provider in COSYVOICE_NAMES:
        return _build_cosyvoice(settings)
    if provider in DOUBAO_NAMES:
        return _build_doubao(settings)
    if provider in OPENLUX_NAMES:
        return _build_openlux(settings)
    raise RuntimeError(
        f"未知的 TTS_PROVIDER: {settings.tts_provider}"
        "（可用 openlux / cosyvoice / doubao）"
    )


def tts_label(settings: Settings) -> str:
    """启动日志用的一行摘要。"""
    provider = (settings.tts_provider or "openlux").strip().lower()
    if provider in COSYVOICE_NAMES:
        return (
            f"provider=cosyvoice model={settings.cosyvoice_model} "
            f"voice={settings.cosyvoice_voice} "
            f"sample_rate={settings.cosyvoice_sample_rate}"
        )
    if provider in DOUBAO_NAMES:
        return (
            f"provider=doubao speaker={settings.doubao_speaker} "
            f"resource={settings.doubao_resource_id} "
            f"sample_rate={settings.doubao_sample_rate}"
        )
    return (
        f"provider={provider} model={settings.tts_model} "
        f"voice={settings.tts_voice} speed={settings.tts_speed}"
    )


def _build_cosyvoice(settings: Settings) -> Tts:
    from providers.cosyvoice_tts import CosyVoiceTts

    if not settings.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY：CosyVoice 合成需要它")
    rate = settings.cosyvoice_rate
    if rate is None:
        rate = settings.tts_speed
    return CosyVoiceTts(
        api_key=settings.dashscope_api_key,
        model=settings.cosyvoice_model,
        voice=settings.cosyvoice_voice,
        sample_rate=settings.cosyvoice_sample_rate,
        rate=_clamp(rate, 0.5, 2.0),
        volume=settings.cosyvoice_volume,
    )


def _build_doubao(settings: Settings) -> Tts:
    from providers.doubao_tts import DoubaoTts, speech_rate_from_ratio

    if not settings.doubao_api_key:
        raise RuntimeError("缺少 DOUBAO_API_KEY：豆包合成需要它")
    if not settings.doubao_speaker:
        raise RuntimeError("缺少 DOUBAO_SPEAKER：要指定豆包 2.0 音色")
    return DoubaoTts(
        api_key=settings.doubao_api_key,
        speaker=settings.doubao_speaker,
        resource_id=settings.doubao_resource_id,
        sample_rate=settings.doubao_sample_rate,
        speech_rate=speech_rate_from_ratio(settings.tts_speed),
        context_texts=[settings.doubao_context] if settings.doubao_context else None,
    )


def _build_openlux(settings: Settings) -> Tts:
    from providers.openlux_tts import OpenluxTts

    return OpenluxTts(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.tts_model,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        instructions=settings.tts_instructions,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
