"""分步定位麦克风：设备列表 -> 实时电平 -> VAD 是否判到说话。"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio.mic import (
    describe_device,
    list_input_devices,
    pick_input_rate,
    prefer_headset_input,
    resample_pcm16,
    rms_int16,
)
from audio.vad import SAMPLE_RATE, SileroVad, choose_vad

MODEL = ROOT / "models" / "silero_vad.onnx"


def bar(rms: float, width: int = 40) -> str:
    # int16 满幅约 32768，说话常见几百~几千
    level = min(1.0, rms / 3000.0)
    n = int(level * width)
    return "[" + ("#" * n).ljust(width) + f"] rms={rms:7.1f}"


def open_stream(device: int | None, frame_samples: int):
    capture_rate = pick_input_rate(device)
    native_block = max(1, int(round(frame_samples * capture_rate / SAMPLE_RATE)))
    stream = sd.RawInputStream(
        samplerate=capture_rate,
        blocksize=native_block,
        device=device,
        channels=1,
        dtype="int16",
    )
    stream._voice_capture_rate = capture_rate  # type: ignore[attr-defined]
    stream._voice_frame_samples = frame_samples  # type: ignore[attr-defined]
    return stream


def read_pcm16(stream, frame_samples: int) -> tuple[bytes, bool]:
    capture_rate = getattr(stream, "_voice_capture_rate", SAMPLE_RATE)
    native_block = max(1, int(round(frame_samples * capture_rate / SAMPLE_RATE)))
    data, overflowed = stream.read(native_block)
    # RawInputStream 的 buffer 已是 int16 LE；asarray(..., dtype=int16) 会按字节拆样
    pcm = bytes(data)
    pcm = resample_pcm16(pcm, capture_rate, SAMPLE_RATE)
    need = frame_samples * 2
    if len(pcm) < need:
        pcm = pcm + b"\x00" * (need - len(pcm))
    elif len(pcm) > need:
        pcm = pcm[:need]
    return pcm, bool(overflowed)


def step_list() -> None:
    print("=== 步骤1：输入设备列表 ===")
    print(f"系统默认 device={sd.default.device}")
    auto = prefer_headset_input()
    print(f"自动优选耳机麦: {describe_device(auto) if auto is not None else '无'}")
    print()
    for line in list_input_devices():
        mark = " <== 建议" if auto is not None and line.startswith(f"{auto}:") else ""
        print(line + mark)


def step_level(device: int | None, seconds: float) -> Path:
    print("\n=== 步骤2：实时电平（对着耳机麦持续说话）===")
    print(f"设备: {describe_device(device)}")
    print("若 rms 一直 < 100，多半没采到这路麦；> 300 且说话时明显升高才正常。")
    frame_samples = 512
    out = ROOT / "runs" / "mic_probe.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    peak = 0.0
    with open_stream(device, frame_samples) as stream:
        stream.start()
        print(f"采集 {getattr(stream, '_voice_capture_rate', SAMPLE_RATE)} Hz -> {SAMPLE_RATE} Hz")
        deadline = time.time() + seconds
        while time.time() < deadline:
            pcm, overflowed = read_pcm16(stream, frame_samples)
            chunks.append(pcm)
            value = rms_int16(pcm)
            peak = max(peak, value)
            flag = "OVERFLOW" if overflowed else ""
            print(f"\r{bar(value)} {flag}", end="", flush=True)
        stream.stop()
    print()
    audio = b"".join(chunks)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(audio)
    print(f"峰值 rms={peak:.1f}；已保存 {out}（可用播放器听）")
    if peak < 100:
        print("结论: 电平过低 -> 换 --device，或检查 Windows 是否把 Jabra 设为默认通讯设备/隐私麦克风权限。")
    elif peak < 300:
        print("结论: 有一点信号但偏弱，可贴耳说话再试，或提高系统麦克风音量。")
    else:
        print("结论: 采集正常，继续步骤3看 VAD。")
    return out


def step_vad(device: int | None, seconds: float) -> None:
    print("\n=== 步骤3：VAD 是否判到说话 ===")
    if not MODEL.exists():
        raise SystemExit(f"缺少模型 {MODEL}")
    chosen, _, _ = choose_vad(MODEL, budget_ms=100.0)
    print(f"使用 VAD: {chosen.name}, 设备: {describe_device(device)}")
    print("请说话约几秒，观察 speech / prob")
    speech_frames = 0
    total = 0
    max_prob = 0.0
    with open_stream(device, chosen.frame_samples) as stream:
        stream.start()
        deadline = time.time() + seconds
        while time.time() < deadline:
            pcm, _overflowed = read_pcm16(stream, chosen.frame_samples)
            if hasattr(chosen, "probability"):
                prob = float(chosen.probability(pcm))  # type: ignore[attr-defined]
                talking = prob >= getattr(chosen, "threshold", 0.5)
            else:
                talking = chosen.is_speech(pcm)
                prob = 1.0 if talking else 0.0
            total += 1
            speech_frames += int(talking)
            max_prob = max(max_prob, prob)
            print(
                f"\r{bar(rms_int16(pcm))} speech={str(talking):5} "
                f"prob={prob:.2f} 命中={speech_frames}/{total}",
                end="",
                flush=True,
            )
        stream.stop()
    print()
    ratio = speech_frames / max(1, total)
    print(f"最大 prob={max_prob:.3f}")
    if ratio < 0.05:
        print("结论: 几乎没有 speech=True。看 prob 是否始终 <0.5；若 rms 高而 prob 低，是 VAD 问题。")
    else:
        print(f"结论: VAD 能工作（说话帧占比 {ratio:.0%}）。可继续 live_mic.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="麦克风分步探测")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--step",
        choices=("all", "list", "level", "vad"),
        default="all",
    )
    args = parser.parse_args()
    device = prefer_headset_input(args.device)
    if args.step in ("all", "list"):
        step_list()
    if args.step in ("all", "level"):
        if args.step == "level" or args.step == "all":
            step_level(device, args.seconds)
    if args.step in ("all", "vad"):
        step_vad(device, args.seconds)


if __name__ == "__main__":
    main()
