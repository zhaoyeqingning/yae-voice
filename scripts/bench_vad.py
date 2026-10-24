"""对比 Silero / WebRTC 单帧推理延迟，并按 100ms 阈值选型。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio.vad import SileroVad, WebRtcVad, bench_vad, choose_vad

MODEL = ROOT / "models" / "silero_vad.onnx"


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"缺少模型: {MODEL}")
    chosen, webrtc, silero = choose_vad(MODEL, budget_ms=100.0)
    print(f"WebRTC  avg={webrtc.avg_ms:.3f}ms  p95={webrtc.p95_ms:.3f}ms")
    print(f"Silero  avg={silero.avg_ms:.3f}ms  p95={silero.p95_ms:.3f}ms")
    delta = silero.avg_ms - webrtc.avg_ms
    print(f"差值 Silero-WebRTC = {delta:.3f}ms （阈值 100ms）")
    print(f"选用: {chosen.name}")


if __name__ == "__main__":
    main()
