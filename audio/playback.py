"""用独立输出流播放 16bit PCM，避免和正在打开的麦克风抢同一条全局播放流。"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from ports import AudioBuffer


class SoundDevicePlayer:
    def __init__(self) -> None:
        self._stopped = False
        self._stream: sd.OutputStream | None = None

    async def play(self, audio: AudioBuffer, on_started: Callable[[], None]) -> None:
        self._stopped = False
        if not audio.pcm:
            on_started()
            return
        data = np.frombuffer(audio.pcm, dtype=np.int16)
        on_started()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._blocking_play, data, audio.sample_rate)

    def _blocking_play(self, data: np.ndarray, sample_rate: int) -> None:
        if self._stopped or len(data) == 0:
            return
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        self._stream = stream
        try:
            stream.start()
            chunk = max(1, int(sample_rate * 0.05))
            offset = 0
            while offset < len(data):
                if self._stopped:
                    return
                end = min(len(data), offset + chunk)
                stream.write(data[offset:end])
                offset = end
            # 把缓冲播完
            deadline = time.perf_counter() + 0.3
            while time.perf_counter() < deadline and not self._stopped:
                time.sleep(0.02)
        finally:
            self._stream = None
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def stop_sync(self) -> None:
        self._stopped = True
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    async def stop(self) -> None:
        self.stop_sync()
