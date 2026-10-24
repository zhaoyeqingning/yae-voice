# Yae voice agent

仓库名 `yae-voice`。实时语音对话：一轮从转写开始，模型按字返回，切成小段后合成并播放。新的一轮会取消还在播放的旧回复。

第 1 步用假的转写、假的模型和假的播放器把这条链路跑通：第一小段在模型说完之前就开始合成，新的一轮会丢掉上一轮还没播出的音频，每一轮把听、想、说和合计时间追加到日志。还没有麦克风，也没有真实接口。

规范见 [docs/提交规范.md](docs/提交规范.md)。

```shell
python -m pip install -r requirements-dev.txt
python main.py
pytest -q
```

`python main.py` 会跑一轮假对话，在终端打印计时，并把同一条记录追加到 `runs/turns.jsonl`。这个目录不提交。

真实接口（讯飞 ASR/TTS + OpenLux LLM）把密钥放进本地 `.env`（参考 `.env.example`），然后：

```shell
python live_smoke.py --text "用一句话介绍你自己"
```

LLM 默认 `LLM_ENABLE_THINKING=false`。可选 `--audio path.wav`（16kHz 单声道）测 ASR。

麦克风 + 本地 VAD（Silero / WebRTC 按本机延迟自动选型，差值 ≤100ms 用 Silero）：

```shell
python scripts/bench_vad.py
python live_mic.py --turns 1
python live_mic.py --list-devices
```

文字正式入口（近黑陪伴界面：空状态、流式状态、停止、独立通话层）：

```shell
python web_server.py
```

浏览器打开 http://127.0.0.1:8787 。有 `.env` 密钥时走真实 LLM/TTS；否则或设 `YAE_WEB_FAKE=1` 用假链路。

推送到 GitHub 时，「提交检查」会把这次推送中的每一条提交单独检出并运行测试。
