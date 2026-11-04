import asyncio
from providers.config import load_settings
from providers.minimax_tts import MinimaxTts

async def main():
    s = load_settings()
    print("voice_id:", s.minimax_voice_id)
    tts = MinimaxTts(
        api_key=s.minimax_api_key,
        group_id=s.minimax_group_id,
        voice_id=s.minimax_voice_id,
        speed=s.tts_speed,
    )
    pcm = await tts.synthesize("你好，呢度係一個測試。")
    print("成功! PCM bytes:", len(pcm), "采样率:", tts.sample_rate)

asyncio.run(main())
