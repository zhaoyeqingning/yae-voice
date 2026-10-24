"""本地 VAD：Silero（ONNX）与 WebRTC。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
# 当前 models/silero_vad.onnx 在 16k 下需 256 点窗口；512 点会导致概率恒接近 0
SILERO_WINDOW = 256
WEBRTC_FRAME_MS = 30


class FrameVad(Protocol):
    name: str
    frame_samples: int

    def is_speech(self, pcm16: bytes) -> bool:
        """一帧 16bit 小端 PCM，长度须为 frame_samples*2。"""

    def reset(self) -> None:
        """清空内部状态（Silero LSTM）。"""


def _ensure_pkg_resources() -> None:
    """webrtcvad 依赖已移除的 pkg_resources；缺省时塞一个最小桩。"""
    import sys
    import types

    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        stub = types.ModuleType("pkg_resources")

        class _Dist:
            version = "2.0.10"

        stub.get_distribution = lambda _name: _Dist()  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = stub


class WebRtcVad:
    name = "webrtc"

    def __init__(self, aggressiveness: int = 2) -> None:
        _ensure_pkg_resources()
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        self.frame_samples = SAMPLE_RATE * WEBRTC_FRAME_MS // 1000

    def is_speech(self, pcm16: bytes) -> bool:
        if len(pcm16) != self.frame_samples * 2:
            raise ValueError("WebRTC 帧长不匹配")
        return self._vad.is_speech(pcm16, SAMPLE_RATE)

    def reset(self) -> None:
        return


class SileroVad:
    name = "silero"

    def __init__(self, model_path: Path, threshold: float = 0.4) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self.frame_samples = SILERO_WINDOW
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def probability(self, pcm16: bytes) -> float:
        if len(pcm16) != self.frame_samples * 2:
            raise ValueError("Silero 帧长不匹配")
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.ascontiguousarray(audio.reshape(1, -1))
        output, state_n = self._session.run(
            None,
            {"input": audio, "state": self._state, "sr": self._sr},
        )
        self._state = state_n
        return float(np.asarray(output).reshape(-1)[0])

    def is_speech(self, pcm16: bytes) -> bool:
        return self.probability(pcm16) >= self.threshold

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)


@dataclass(frozen=True)
class VadBench:
    name: str
    avg_ms: float
    p95_ms: float


def bench_vad(vad: FrameVad, frames: int = 200) -> VadBench:
    import time

    # 混合能量帧，避免编译器把纯零路径优化掉观感
    rng = np.random.default_rng(0)
    pcm = (rng.integers(-800, 800, size=vad.frame_samples, dtype=np.int16)).tobytes()
    vad.reset()
    # warmup
    for _ in range(20):
        vad.is_speech(pcm)
    times: list[float] = []
    for _ in range(frames):
        t0 = time.perf_counter()
        vad.is_speech(pcm)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    avg = sum(times) / len(times)
    p95 = times[int(len(times) * 0.95)]
    return VadBench(vad.name, avg, p95)


def choose_vad(model_path: Path, budget_ms: float = 100.0) -> tuple[FrameVad, VadBench, VadBench]:
    """差值不超过 budget_ms 用 Silero，否则用 WebRTC。"""
    webrtc = WebRtcVad()
    silero = SileroVad(model_path)
    w = bench_vad(webrtc)
    s = bench_vad(silero)
    # 基准测试会污染 LSTM 状态，正式使用前重置；并新开实例避免残留
    if s.avg_ms - w.avg_ms <= budget_ms:
        chosen: FrameVad = SileroVad(model_path, threshold=silero.threshold)
    else:
        chosen = WebRtcVad()
    return chosen, w, s
