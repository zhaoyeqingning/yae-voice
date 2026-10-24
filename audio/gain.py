"""整段 PCM 增益：PTT 松开后、送 ASR 前做峰值归一化。"""

from __future__ import annotations

import numpy as np


def peak_normalize(
    pcm16: bytes,
    *,
    target_peak: float = 0.6,
    max_gain: float = 20.0,
) -> tuple[bytes, float]:
    """把峰值拉到 target_peak * 32767，增益上限 max_gain。返回 (pcm, gain)。"""
    if not pcm16:
        return pcm16, 1.0
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 1.0:
        return pcm16, 1.0
    desired = target_peak * 32767.0
    gain = min(max_gain, desired / peak)
    if gain <= 1.01:
        return pcm16, 1.0
    y = np.clip(x * gain, -32768, 32767).astype(np.int16)
    return y.tobytes(), float(gain)
