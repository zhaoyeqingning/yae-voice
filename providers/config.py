"""从环境变量 / .env 读取密钥。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# 官方短模板（Calm support）：一行一个维度，不叠音高/气声/否定句。
# shimmer 是偏柔的女声；语速交给说明，speed 保持 1。
DEFAULT_TTS_INSTRUCTIONS = (
    "Voice Affect: Calm and reassuring.\n"
    "Tone: Sincere, warm, and gentle.\n"
    "Pacing: Steady, natural conversation.\n"
    "Emotion: Quiet care.\n"
    "Pronunciation: Standard Mandarin, clear and unforced."
)


@dataclass(frozen=True)
class Settings:
    # 讯飞 ASR
    xfyun_app_id: str
    xfyun_api_key: str
    xfyun_api_secret: str

    # LLM
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_enable_thinking: bool
    llm_use_audio: bool  # 是否使用多模态音频模型（跳过 ASR）

    # TTS 通用
    tts_provider: str  # openlux / minimax / elevenlabs
    tts_speed: float

    # OpenLux / OpenAI TTS
    tts_model: str
    tts_voice: str
    tts_instructions: str

    # MiniMax TTS
    minimax_api_key: str
    minimax_group_id: str
    minimax_voice_id: str

    # ElevenLabs TTS
    elevenlabs_api_key: str
    elevenlabs_voice_id: str


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env")
    model = os.getenv("LLM_MODEL", "").strip()
    thinking = os.getenv("LLM_ENABLE_THINKING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    llm_use_audio = os.getenv("LLM_USE_AUDIO", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tts_provider = os.getenv("TTS_PROVIDER", "openlux").strip().lower() or "openlux"

    settings = Settings(
        # 讯飞 ASR
        xfyun_app_id=os.getenv("XFYUN_APPID", "").strip(),
        xfyun_api_key=os.getenv("XFYUN_API_KEY", "").strip(),
        xfyun_api_secret=os.getenv("XFYUN_API_SECRET", "").strip(),
        # LLM
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openlux.ai/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=model or "gpt-4o-mini",
        llm_enable_thinking=thinking,
        llm_use_audio=llm_use_audio,
        # TTS 通用
        tts_provider=tts_provider,
        tts_speed=_tts_speed(os.getenv("TTS_SPEED", "1.15")),
        # OpenLux TTS
        tts_model=os.getenv("TTS_MODEL", "tts-1").strip() or "tts-1",
        tts_voice=os.getenv("TTS_VOICE", "shimmer").strip() or "shimmer",
        tts_instructions=_tts_instructions(os.getenv("TTS_INSTRUCTIONS")),
        # MiniMax
        minimax_api_key=os.getenv("MINIMAX_API_KEY", "").strip(),
        minimax_group_id=os.getenv("MINIMAX_GROUP_ID", "").strip(),
        minimax_voice_id=os.getenv("MINIMAX_VOICE_ID", "female-tianmei").strip(),
        # ElevenLabs
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "rachel").strip(),
    )

    # 根据 TTS provider 检查必需的密钥
    missing: list[str] = []
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY")

    # 不走多模态时才需要讯飞 ASR
    if not settings.llm_use_audio:
        if not settings.xfyun_app_id:
            missing.append("XFYUN_APPID")
        if not settings.xfyun_api_key:
            missing.append("XFYUN_API_KEY")
        if not settings.xfyun_api_secret:
            missing.append("XFYUN_API_SECRET")

    if settings.tts_provider == "minimax":
        if not settings.minimax_api_key:
            missing.append("MINIMAX_API_KEY")
        if not settings.minimax_group_id:
            missing.append("MINIMAX_GROUP_ID")
    elif settings.tts_provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
    # openlux 用 LLM_API_KEY 即可

    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")
    return settings


def _tts_speed(raw: str | None) -> float:
    try:
        speed = float((raw or "0.92").strip())
    except ValueError:
        speed = 1.0
    return min(4.0, max(0.25, speed))


def _tts_instructions(raw: str | None) -> str:
    if raw is None:
        return DEFAULT_TTS_INSTRUCTIONS
    return raw.strip()
