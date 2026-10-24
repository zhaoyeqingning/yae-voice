"""麦克风采集：按设备可用采样率开流，对外输出 16kHz 单声道 int16。"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from audio.vad import SAMPLE_RATE

_STOP = object()


def pick_input_rate(device: int | str | None) -> int:
    """选设备能打开的采样率；优先默认速率，再试常见速率。"""
    try:
        info = sd.query_devices(device, "input")
        default = int(info.get("default_samplerate") or SAMPLE_RATE)
    except Exception:
        default = SAMPLE_RATE
    candidates: list[int] = []
    for rate in (default, 48000, 44100, 32000, 16000):
        if rate > 0 and rate not in candidates:
            candidates.append(rate)
    for rate in candidates:
        try:
            sd.check_input_settings(
                device=device, channels=1, dtype="int16", samplerate=rate
            )
            return rate
        except Exception:
            continue
    return default


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


class Microphone:
    def __init__(self, frame_samples: int, device: int | str | None = None) -> None:
        self.frame_samples = frame_samples  # 对外 16kHz 帧长（样本数）
        self.device = device
        self.capture_rate = SAMPLE_RATE
        self._q: queue.Queue[bytes | object] = queue.Queue(maxsize=64)
        self._stream: sd.RawInputStream | None = None
        self._thread: threading.Thread | None = None
        self.paused = False
        self._closed = False
        self._native_block = frame_samples

    @staticmethod
    def _pcm_from_read(data) -> bytes:
        # RawInputStream 返回 cffi buffer：已是 int16 小端字节。
        # 不可 np.asarray(..., dtype=int16)，那会把每个字节当成一个样本（峰值恒≤255）。
        return bytes(data)

    def _enqueue(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pcm)
            except queue.Full:
                pass

    def _reader(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            while not self._closed:
                if self.paused:
                    time.sleep(0.01)
                    continue
                try:
                    data, _overflowed = stream.read(self._native_block)
                except Exception as exc:
                    if not self._closed:
                        print(f"\n麦克风读取中断: {exc}", flush=True)
                    break
                native = self._pcm_from_read(data)
                pcm = resample_pcm16(native, self.capture_rate, SAMPLE_RATE)
                self._enqueue(pcm)
        finally:
            # 只能在读麦的线程里关，主线程关会在 Windows 上把进程打崩
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            if self._stream is stream:
                self._stream = None

    def start(self) -> None:
        if self._stream is not None or (self._thread is not None and self._thread.is_alive()):
            return
        self._closed = False
        self.paused = False
        self.capture_rate = pick_input_rate(self.device)
        self._native_block = max(
            1, int(round(self.frame_samples * self.capture_rate / SAMPLE_RATE))
        )
        self._stream = sd.RawInputStream(
            samplerate=self.capture_rate,
            blocksize=self._native_block,
            device=self.device,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._reader, name="mic-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._closed = True
        self.paused = True
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass

    def pause(self) -> None:
        self.paused = True
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def resume(self) -> None:
        if self._closed:
            return
        self.paused = False

    async def read_frame(self) -> bytes | None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                item = await loop.run_in_executor(None, lambda: self._q.get(timeout=0.1))
            except queue.Empty:
                continue
            if item is _STOP:
                return None
            return item  # type: ignore[return-value]
        return None


def list_input_devices() -> list[str]:
    devices = sd.query_devices()
    lines: list[str] = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            lines.append(
                f"{index}: {device['name']} ({int(device['default_samplerate'])} Hz)"
            )
    return lines


def describe_device(device: int | str | None = None) -> str:
    info = sd.query_devices(device, "input")
    index = info.get("index", device)
    return f"{index}: {info['name']}"


def prefer_headset_input(device: int | None = None) -> int | None:
    """未指定设备时，若默认输出是耳机，则尽量选同名、且支持阻塞读的输入。"""
    if device is not None:
        return device
    _default_in, default_out = sd.default.device
    try:
        out_name = str(sd.query_devices(default_out)["name"])
    except Exception:
        return None
    hint = out_name
    for token in ("(", ")", "扬声器", "Speaker", "Headphones", "头戴式耳机", "耳机"):
        hint = hint.replace(token, " ")
    keywords = [part for part in hint.split() if len(part) >= 3]
    if not keywords:
        return None

    best_index: int | None = None
    best_score = -1
    for index, item in enumerate(sd.query_devices()):
        if item["max_input_channels"] <= 0:
            continue
        name = str(item["name"])
        hits = sum(1 for k in keywords if k.lower() in name.lower())
        if hits == 0:
            continue
        api_name = str(sd.query_hostapis(item["hostapi"])["name"])
        score = hits * 10
        # 阻塞 RawInputStream：MME/DirectSound 稳；WDM-KS 不可用
        if "WDM-KS" in api_name:
            score -= 50
        if "MME" in api_name or "DirectSound" in api_name:
            score += 6
        if "WASAPI" in api_name:
            score += 1
        if "Hands-Free" in name or "Hands Free" in name:
            score -= 20
        if "麦克风" in name or "mic" in name.lower():
            score += 1
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def rms_int16(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    data = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(data))))
