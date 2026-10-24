"""VAD 状态机：起说 / 尾静音切句。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from audio.vad import FrameVad, SAMPLE_RATE


@dataclass
class UtteranceConfig:
    start_ms: int = 200
    end_ms: int = 700
    min_ms: int = 300
    max_ms: int = 20000
    pad_ms: int = 200


class UtteranceCollector:
    def __init__(self, vad: FrameVad, config: UtteranceConfig | None = None) -> None:
        self.vad = vad
        self.config = config or UtteranceConfig()
        frame_ms = max(1, self.vad.frame_samples * 1000 // SAMPLE_RATE)
        self.frame_ms = frame_ms
        self._start_frames = max(1, self.config.start_ms // frame_ms)
        self._end_frames = max(1, self.config.end_ms // frame_ms)
        self._min_frames = max(1, self.config.min_ms // frame_ms)
        self._max_frames = max(1, self.config.max_ms // frame_ms)
        self._pad_frames = max(0, self.config.pad_ms // frame_ms)
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._frames: list[bytes] = []
        self._pre_roll: list[bytes] = []
        self._recent = deque(maxlen=3)
        self.last_prob = 0.0
        self.last_speech = False

    def reset(self) -> None:
        self.vad.reset()
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._frames.clear()
        self._pre_roll.clear()
        self._recent.clear()
        self.last_prob = 0.0
        self.last_speech = False

    def _frame_is_speech(self, pcm16: bytes) -> bool:
        if hasattr(self.vad, "probability"):
            prob = float(self.vad.probability(pcm16))  # type: ignore[attr-defined]
            self.last_prob = prob
            threshold = float(getattr(self.vad, "threshold", 0.4))
            # 已在说话中时降低阈值，减少闪烁导致一直切不出去/进不去
            if self._in_speech:
                hit = prob >= threshold * 0.6
            else:
                hit = prob >= threshold
        else:
            hit = self.vad.is_speech(pcm16)
            self.last_prob = 1.0 if hit else 0.0
        self._recent.append(hit)
        # 起说：近 3 帧里至少 2 帧为语音，避免偶发漏检
        if not self._in_speech and len(self._recent) >= 2:
            hit = sum(self._recent) >= 2
        self.last_speech = hit
        return hit

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, pcm16: bytes) -> bytes | None:
        """喂入一帧；若切出一句则返回整段 PCM（含前后 padding）。"""
        speech = self._frame_is_speech(pcm16)
        if not self._in_speech:
            self._pre_roll.append(pcm16)
            if len(self._pre_roll) > self._pad_frames + self._start_frames:
                self._pre_roll.pop(0)
            if speech:
                self._speech_run += 1
            else:
                self._speech_run = 0
            if self._speech_run >= self._start_frames:
                self._in_speech = True
                self._silence_run = 0
                self._frames = list(self._pre_roll)
                self._pre_roll.clear()
            return None

        self._frames.append(pcm16)
        if speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        too_long = len(self._frames) >= self._max_frames
        ended = self._silence_run >= self._end_frames
        if not (ended or too_long):
            return None

        drop = max(0, self._silence_run - self._pad_frames)
        keep = len(self._frames) - drop
        utterance = self._frames[:keep]
        self._frames.clear()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._pre_roll.clear()
        self._recent.clear()
        self.vad.reset()
        if len(utterance) < self._min_frames and not too_long:
            return None
        return b"".join(utterance)
