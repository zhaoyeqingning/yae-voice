"""根据配置创建 LLM / TTS / ASR 实例的工厂。"""

from __future__ import annotations

from .config import Settings
from .elevenlabs_tts import ElevenLabsTts
from .minimax_tts import MinimaxTts
from .openlux_llm import OpenluxLlm
from .openlux_tts import OpenluxTts
from .openai_audio_llm import OpenAiAudioLlm
from .xfyun_asr import XfyunAsr


def create_llm(settings: Settings) -> object:
    """根据配置创建 LLM 实例。

    如果 llm_use_audio=True，返回多模态音频 LLM（跳过 ASR）。
    否则返回普通文字 LLM。
    """
    if settings.llm_use_audio:
        return OpenAiAudioLlm(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            enable_thinking=settings.llm_enable_thinking,
        )
    return OpenluxLlm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        enable_thinking=settings.llm_enable_thinking,
    )


def create_tts(settings: Settings) -> object:
    """根据 tts_provider 创建对应的 TTS 实例。"""
    provider = settings.tts_provider

    if provider == "minimax":
        return MinimaxTts(
            api_key=settings.minimax_api_key,
            group_id=settings.minimax_group_id,
            voice_id=settings.minimax_voice_id,
            speed=settings.tts_speed,
        )
    elif provider == "elevenlabs":
        return ElevenLabsTts(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            sample_rate=24000,
        )
    else:
        # 默认 openlux / OpenAI 兼容
        return OpenluxTts(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.tts_model,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            instructions=settings.tts_instructions,
        )


def create_asr(settings: Settings) -> object:
    """创建 ASR 实例。多模态模式下可以返回 None。"""
    if settings.llm_use_audio:
        return None  # 多模态模式不需要 ASR
    return XfyunAsr(
        app_id=settings.xfyun_app_id,
        api_key=settings.xfyun_api_key,
        api_secret=settings.xfyun_api_secret,
    )
