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
    xfyun_app_id: str
    xfyun_api_key: str
    xfyun_api_secret: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_enable_thinking: bool
    tts_model: str
    tts_voice: str
    tts_speed: float
    tts_instructions: str


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env")
    model = os.getenv("LLM_MODEL", "").strip()
    thinking = os.getenv("LLM_ENABLE_THINKING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    settings = Settings(
        xfyun_app_id=os.getenv("XFYUN_APPID", "").strip(),
        xfyun_api_key=os.getenv("XFYUN_API_KEY", "").strip(),
        xfyun_api_secret=os.getenv("XFYUN_API_SECRET", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openlux.ai/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=model or "gpt-4o-mini",
        llm_enable_thinking=thinking,
        tts_model=os.getenv("TTS_MODEL", "tts-1").strip() or "tts-1",
        tts_voice=os.getenv("TTS_VOICE", "shimmer").strip() or "shimmer",
        tts_speed=_tts_speed(os.getenv("TTS_SPEED", "1.15")),
        tts_instructions=_tts_instructions(os.getenv("TTS_INSTRUCTIONS")),
    )
    missing = [
        name
        for name, value in [
            ("XFYUN_APPID", settings.xfyun_app_id),
            ("XFYUN_API_KEY", settings.xfyun_api_key),
            ("XFYUN_API_SECRET", settings.xfyun_api_secret),
            ("LLM_API_KEY", settings.llm_api_key),
        ]
        if not value
    ]
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
