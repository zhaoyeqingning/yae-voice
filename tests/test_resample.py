from audio.resample import resample_pcm16
from audio.vad import SAMPLE_RATE


def test_resample_48k_to_16k_length():
    # 48kHz 下 480 点 ≈ 10ms；到 16k 应为 160 点
    src = (b"\x00\x10" * 480)
    out = resample_pcm16(src, 48000, SAMPLE_RATE)
    assert len(out) == 160 * 2


def test_resample_same_rate_noop():
    src = b"\x01\x00\x02\x00\x03\x00"
    assert resample_pcm16(src, SAMPLE_RATE, SAMPLE_RATE) is src or resample_pcm16(
        src, SAMPLE_RATE, SAMPLE_RATE
    ) == src
