"""用真实 ASR/TTS/LLM 冒烟：默认文本进 LLM -> TTS；可选本地 PCM 测 ASR。"""

from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

import httpx

from player import FakePlayer
from providers.config import load_settings
from providers.openlux_llm import OpenluxLlm
from providers.openlux_tts import OpenluxGeminiTts
from providers.xfyun_asr import XfyunAsr
from session import Session


async def resolve_model(base_url: str, api_key: str, preferred: str) -> str:
    if preferred:
        return preferred
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        ids = [item["id"] for item in data.get("data") or [] if item.get("id")]
    if not ids:
        raise RuntimeError("OpenLux /models 没有返回可用模型，请在 .env 设置 LLM_MODEL")
    for item in ids:
        lower = item.lower()
        if "embed" in lower or "tts" in lower or "whisper" in lower:
            continue
        if "think" in lower:
            continue
        return item
    return ids[0]


def read_pcm16(path: Path) -> bytes:
    if path.suffix.lower() == ".pcm":
        return path.read_bytes()
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError("WAV 需为 16bit 单声道")
        if handle.getframerate() != 16000:
            raise RuntimeError("WAV 采样率需为 16000")
        return handle.readframes(handle.getnframes())


async def run(text: str, audio_path: Path | None, out_pcm: Path | None) -> None:
    settings = load_settings()
    model = await resolve_model(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    print(f"LLM model={model} thinking={settings.llm_enable_thinking}")

    asr = XfyunAsr(
        app_id=settings.xfyun_app_id,
        api_key=settings.xfyun_api_key,
        api_secret=settings.xfyun_api_secret,
    )
    llm = OpenluxLlm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        enable_thinking=settings.llm_enable_thinking,
    )
    tts = OpenluxGeminiTts(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.tts_model,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        instructions=settings.tts_instructions,
    )
    print(f"TTS model={settings.tts_model} voice={settings.tts_voice} speed={settings.tts_speed}")
    player = FakePlayer()
    session = Session(asr, llm, tts, player, log_path=Path("runs/live_turns.jsonl"))

    if audio_path is not None:
        pcm = read_pcm16(audio_path)
        print(f"ASR 输入: {audio_path} ({len(pcm)} bytes)")
        await session.submit_audio(pcm)
    else:
        print(f"用户文本: {text}")
        await session.submit_text(text)

    spoken = "".join(item.text for item in player.played)
    print(f"助手文本: {spoken}")
    print(f"TTS 片段数: {len(player.played)}")
    if out_pcm is not None and player.played:
        out_pcm.parent.mkdir(parents=True, exist_ok=True)
        out_pcm.write_bytes(b"".join(item.pcm for item in player.played))
        print(f"已写出 PCM: {out_pcm}")


def main() -> None:
    parser = argparse.ArgumentParser(description="真实接口冒烟")
    parser.add_argument("--text", default="用一句话介绍你自己")
    parser.add_argument("--audio", type=Path, default=None, help="16k 单声道 wav/pcm，测 ASR")
    parser.add_argument("--out", type=Path, default=Path("runs/live_reply.pcm"))
    args = parser.parse_args()
    asyncio.run(run(args.text, args.audio, args.out))


if __name__ == "__main__":
    main()
