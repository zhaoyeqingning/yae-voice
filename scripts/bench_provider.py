"""测量现有 provider 的整段请求耗时。工厂用 module:function 指定。"""
import argparse
import asyncio
import importlib
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def measure(provider, kind, value, repeats):
    records = []
    for index in range(repeats):
        started = time.perf_counter()
        result = await (provider.transcribe(value) if kind == "asr" else provider.synthesize(value))
        elapsed = (time.perf_counter() - started) * 1000
        record = {"iteration": index, "kind": kind, "request_ms": elapsed}
        if kind == "asr":
            record["transcript"] = result
        else:
            rate = provider.sample_rate
            if not result or len(result) % 2 or rate <= 0:
                raise ValueError("TTS 必须返回非空 PCM16，且采样率为正")
            duration = len(result) / (2 * rate)
            record.update(audio_seconds=duration, rtf=elapsed / 1000 / duration)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="无参工厂 module:function，返回已配置 provider")
    parser.add_argument("--kind", choices=["asr", "tts"], required=True)
    parser.add_argument("--input", required=True, help="ASR 为 WAV 路径，TTS 为文字")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats 必须为正")
    value = args.input
    if args.kind == "asr":
        with wave.open(value, "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, "NONE"):
                parser.error("ASR 输入须为 16kHz 单声道 PCM16 WAV")
            value = audio.readframes(audio.getnframes())
    module, name = args.factory.split(":", 1)
    provider = getattr(importlib.import_module(module), name)()
    records = asyncio.run(measure(provider, args.kind, value, args.repeats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


if __name__ == "__main__":
    main()
