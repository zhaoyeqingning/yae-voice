from audio.utterance import UtteranceCollector, UtteranceConfig


class FakeVad:
    name = "fake"
    frame_samples = 480  # 30ms @ 16k

    def __init__(self, flags: list[bool]) -> None:
        self.flags = list(flags)
        self.i = 0

    def is_speech(self, pcm16: bytes) -> bool:
        del pcm16
        flag = self.flags[min(self.i, len(self.flags) - 1)]
        self.i += 1
        return flag

    def reset(self) -> None:
        return


def test_utterance_cuts_after_trailing_silence():
    # 30ms 帧：start 90ms(~3), end 90ms(~3), min 60ms(~2)
    flags = [False, True, True, True, True, True, False, False, False]
    vad = FakeVad(flags)
    collector = UtteranceCollector(
        vad,
        UtteranceConfig(start_ms=90, end_ms=90, min_ms=60, max_ms=5000, pad_ms=30),
    )
    frame = b"\x00\x00" * vad.frame_samples
    out = None
    for _ in flags:
        out = collector.push(frame) or out
    assert out is not None
    # 起说前 pad + 语音帧 - 尾静音裁切后仍保留 pad
    assert len(out) >= vad.frame_samples * 2 * 2
