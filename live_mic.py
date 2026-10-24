"""按住空格录音，松开后 ASR -> LLM -> TTS。按住期间不播放回复。"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import signal
import sys
import traceback
from pathlib import Path

import sounddevice as sd

from audio.gain import peak_normalize
from audio.mic import (
    Microphone,
    describe_device,
    list_input_devices,
    pick_input_rate,
    prefer_headset_input,
    rms_int16,
)
from audio.vad import SAMPLE_RATE
from audio.playback import SoundDevicePlayer
from live_smoke import resolve_model
from providers.config import load_settings
from providers.openlux_llm import OpenluxLlm
from providers.openlux_tts import OpenluxGeminiTts
from providers.xfyun_asr import XfyunAsr
from session import Session

# 16kHz * 16bit * 0.2s，短于这个的按键忽略
MIN_PCM_BYTES = 6400
VK_SPACE = 0x20


def _print_last_timing(path: Path) -> None:
    if not path.exists():
        return
    line = path.read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    print(
        "计时 "
        f"听 {record.get('listen_ms')} ms / "
        f"想 {record.get('think_ms')} ms / "
        f"说 {record.get('speak_ms')} ms / "
        f"合计 {record.get('total_ms')} ms"
    )
    pieces = record.get("tts_pieces") or []
    if pieces:
        print(f"TTS 切段 ({len(pieces)}):")
        for index, piece in enumerate(pieces, start=1):
            print(f"  [{index}] {piece}")


def space_down() -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000)


def probe_mic_open(device: int | None, capture_hz: int) -> None:
    """启动前试开阻塞读；WDM-KS 等会在这里失败。"""
    block = max(1, int(round(512 * capture_hz / SAMPLE_RATE)))
    stream = sd.RawInputStream(
        samplerate=capture_hz,
        blocksize=block,
        device=device,
        channels=1,
        dtype="int16",
    )
    try:
        stream.start()
        stream.read(block)
    finally:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass


async def run_loop(*, turns: int, device: int | None, list_devices: bool) -> None:
    if list_devices:
        for line in list_input_devices():
            print(line)
        return

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
    player = SoundDevicePlayer()
    session = Session(asr, llm, tts, player, log_path=Path("runs/live_mic_turns.jsonl"))
    selected = prefer_headset_input(device)
    mic = Microphone(512, device=selected)
    capture_hz = pick_input_rate(selected)
    print(f"输入设备: {describe_device(selected)}")
    print(f"采集 {capture_hz} Hz -> 重采样 {SAMPLE_RATE} Hz 送 ASR")
    if selected is None:
        print("提示: 若用耳机说话却没反应，先 --list-devices，再加 --device <耳机麦编号>")
    try:
        probe_mic_open(selected, capture_hz)
    except Exception as exc:
        print(f"无法以阻塞方式打开该输入设备: {exc}")
        print("请改用 MME/DirectSound 设备，例如: python -u live_mic.py --device 2")
        return

    print("按住空格说话，松开后识别并回复。按住期间不播放。Ctrl+C 结束。")
    done = 0
    stopping = False
    recording = False
    chunks: list[bytes] = []
    turn_task: asyncio.Task[None] | None = None

    def request_stop(*_args) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\n正在退出...", flush=True)
        player.stop_sync()
        mic.stop()

    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            signal.signal(sig, request_stop)

    try:
        while not stopping:
            held = space_down()
            if held and not recording:
                if turn_task is not None and not turn_task.done():
                    turn_task.cancel()
                turn_task = None
                player.stop_sync()
                chunks = []
                try:
                    mic.start()
                except Exception as exc:
                    recording = False
                    print(f"\n开麦失败: {exc}", flush=True)
                    print("请换 --device（不要用 WDM-KS 的 capture 设备）", flush=True)
                    await asyncio.sleep(0.3)
                    continue
                recording = True
                print("\n录音中，松开空格结束", flush=True)
                continue

            if recording:
                if not held:
                    recording = False
                    mic.stop()
                    pcm = b"".join(chunks)
                    chunks = []
                    if len(pcm) < MIN_PCM_BYTES:
                        print("太短，已忽略", flush=True)
                        continue
                    level = rms_int16(pcm)
                    pcm, gain = peak_normalize(pcm)
                    print(
                        f"\n松开，{len(pcm)} bytes rms={level:.0f} gain×{gain:.1f} -> 识别中...",
                        flush=True,
                    )
                    turn_task = asyncio.create_task(session.submit_audio(pcm))
                    continue
                frame = await mic.read_frame()
                if frame:
                    chunks.append(frame)
                    print(f"\r录音 rms={rms_int16(frame):.0f}   ", end="", flush=True)
                continue

            if turn_task is not None and turn_task.done():
                exc = turn_task.exception()
                turn_task = None
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    traceback.print_exception(exc)
                elif session.history:
                    done += 1
                    print(f"用户: {session.history[-2]['text']}")
                    print(f"助手: {session.history[-1]['text']}")
                    _print_last_timing(Path("runs/live_mic_turns.jsonl"))
                    if turns > 0 and done >= turns:
                        break
            await asyncio.sleep(0.02)
    except Exception:
        traceback.print_exc()
    finally:
        stopping = True
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except (asyncio.CancelledError, Exception):
                pass
        await player.stop()
        mic.stop()
        print("已停止", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="按住空格说话")
    parser.add_argument("--turns", type=int, default=0, help="完成几轮后退出；0 表示一直聊")
    parser.add_argument("--device", type=int, default=None, help="sounddevice 输入设备编号")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(
            run_loop(turns=args.turns, device=args.device, list_devices=args.list_devices)
        )
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    except Exception as exc:
        traceback.print_exc()
        print(f"异常退出: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
