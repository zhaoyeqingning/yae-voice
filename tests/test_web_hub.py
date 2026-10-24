import asyncio
import base64

from ports import AudioBuffer, silence_pcm
from web_hub import Hub, HubPlayer


def test_hub_player_publishes_audio_for_active_generation():
    async def scenario() -> None:
        hub = Hub()
        player = HubPlayer(hub)
        relay = await hub.attach(3)
        await player.play(
            AudioBuffer(silence_pcm(), 24000, 3, 0, "你好。"),
            lambda: None,
        )
        event = await asyncio.wait_for(relay.queue.get(), timeout=1)
        assert event["type"] == "audio"
        assert event["generation"] == 3
        assert event["seq"] == 0
        assert base64.b64decode(event["pcm_b64"])

    asyncio.run(scenario())


def test_text_delta_goes_to_active_relay():
    async def scenario() -> None:
        hub = Hub()
        relay = await hub.attach(1)
        hub.on_text_delta("嘿")
        event = await asyncio.wait_for(relay.queue.get(), timeout=1)
        assert event == {"type": "text.delta", "generation": 1, "delta": "嘿"}

    asyncio.run(scenario())
