"""PCM16 重采样：只依赖 numpy，CI 与无声卡环境也能跑。"""

from __future__ import annotations

import numpy as np

from audio.vad import SAMPLE_RATE


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE) -> bytes:
    if not pcm or src_rate == dst_rate:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(x) < 2:
        return pcm if src_rate == dst_rate else b""
    n_out = max(1, int(round(len(x) * dst_rate / src_rate)))
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    y = np.interp(t_dst, t_src, x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()
