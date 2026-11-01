"""factory 的选择逻辑：TTS_PROVIDER 决定返回哪个引擎。"""

from dataclasses import replace

import pytest

from providers.config import Settings
from providers.cosyvoice_tts import CosyVoiceTts
from providers.doubao_tts import DoubaoTts
from providers.factory import build_tts, tts_label
from providers.openlux_tts import OpenluxTts


def settings(**overrides) -> Settings:
    base = Settings(
        xfyun_app_id="app",
        xfyun_api_key="key",
        xfyun_api_secret="secret",
        llm_base_url="https://example.invalid/v1",
        llm_api_key="llm-key",
        llm_model="gpt-4o-mini",
        llm_enable_thinking=False,
        tts_model="tts-1",
        tts_voice="shimmer",
        tts_speed=1.15,
        tts_instructions="",
    )
    return replace(base, **overrides)


def test_defaults_to_openlux():
    tts = build_tts(settings())
    assert isinstance(tts, OpenluxTts)
    assert tts.voice == "shimmer"


def test_builds_cosyvoice():
    tts = build_tts(
        settings(
            tts_provider="cosyvoice",
            dashscope_api_key="sk-test",
            cosyvoice_voice="longwan_v2",
        )
    )
    assert isinstance(tts, CosyVoiceTts)
    assert tts.voice == "longwan_v2"
    assert tts.sample_rate == 24000


def test_cosyvoice_rate_prefers_its_own_value_and_clamps():
    tts = build_tts(
        settings(tts_provider="cosyvoice", dashscope_api_key="sk", cosyvoice_rate=3.0)
    )
    assert tts.rate == 2.0


def test_cosyvoice_rate_falls_back_to_tts_speed():
    tts = build_tts(
        settings(tts_provider="cosyvoice", dashscope_api_key="sk", tts_speed=1.25)
    )
    assert tts.rate == 1.25


def test_builds_doubao_and_maps_speed():
    tts = build_tts(
        settings(
            tts_provider="doubao",
            doubao_api_key="key",
            doubao_speaker="zh_female_test_uranus_bigtts",
            tts_speed=1.5,
        )
    )
    assert isinstance(tts, DoubaoTts)
    assert tts.speaker == "zh_female_test_uranus_bigtts"
    # 1.5 倍速 → 豆包的 speech_rate=50
    assert tts.speech_rate == 50


def test_doubao_context_texts_are_passed():
    tts = build_tts(
        settings(tts_provider="doubao", doubao_api_key="k", doubao_context="温柔一点")
    )
    assert tts.context_texts == ["温柔一点"]


@pytest.mark.parametrize("alias", ["cosy", "cosyvoice"])
def test_cosyvoice_alias(alias):
    tts = build_tts(settings(tts_provider=alias, dashscope_api_key="sk"))
    assert isinstance(tts, CosyVoiceTts)


@pytest.mark.parametrize("alias", ["volc", "doubao"])
def test_doubao_alias(alias):
    tts = build_tts(settings(tts_provider=alias, doubao_api_key="k"))
    assert isinstance(tts, DoubaoTts)


def test_unknown_provider_is_rejected():
    with pytest.raises(RuntimeError) as excinfo:
        build_tts(settings(tts_provider="nope"))
    assert "TTS_PROVIDER" in str(excinfo.value)


def test_missing_keys_name_the_variable():
    with pytest.raises(RuntimeError) as cosy:
        build_tts(settings(tts_provider="cosyvoice"))
    assert "DASHSCOPE_API_KEY" in str(cosy.value)

    with pytest.raises(RuntimeError) as doubao:
        build_tts(settings(tts_provider="doubao"))
    assert "DOUBAO_API_KEY" in str(doubao.value)


def test_label_mentions_the_provider():
    assert "provider=cosyvoice" in tts_label(settings(tts_provider="cosyvoice"))
    assert "provider=doubao" in tts_label(settings(tts_provider="doubao"))
    assert "provider=openlux" in tts_label(settings(tts_provider="openlux"))
