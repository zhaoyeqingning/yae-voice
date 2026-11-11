# ASR 与 TTS 选型及评测交付

核对日期为 2026-11-11。面向中文陪伴语聊，优先考察响应延迟、识别准确度、合成自然度与接入成本。

## 当前结论

建议先以仓库已有的讯飞 ASR 与 OpenLux TTS 为基线，优先试验 Qwen3-ASR 与 CosyVoice，SenseVoice 作为整句识别候选。最终选型需要在同一机器、同一组录音和文字上比较。

| 候选 | 官方资料提供的能力 | 当前接入判断 | 验证状态 |
| --- | --- | --- | --- |
| 讯飞 ASR | 仓库已有 WebSocket 实现 | 输入整句录音，分帧发送，等最终文本 | 已读实现，待真实测量 |
| Qwen3-ASR | 0.6B 和 1.7B，提供流式推理工具 | 先适配整句接口，流式需进一步修改输入协议 | 已核对官方仓库，待部署 |
| SenseVoice Small | 多语言识别、情绪及音频事件识别 | 可考察短句识别速度，具体流式能力需按所用推理后端验证 | 已核对官方仓库，待部署 |
| OpenLux TTS | 仓库支持 speech 与 Gemini 分支 | 当前实现等待整段音频响应后返回 | 已读实现，待真实测量 |
| CosyVoice | 提供语音生成、推理及部署工具 | 先适配 PCM 输出，发挥流式输出能力需要扩展 Tts 协议 | 已核对官方仓库，待部署 |

官方资料分别见 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[SenseVoice](https://github.com/QwenAudio/SenseVoice)、[CosyVoice](https://github.com/QwenAudio/CosyVoice)。模型版本、权重许可和部署依赖应在实际部署时固定到对应版本。

## 指标与实验方法

ASR 记录音频时长、请求发起到最终文字返回的耗时、人工参考文本和识别文本。计算中文字符错误率时固定标点、空格、数字的归一化规则，并保留原始文本供复核。流式实验另外记录首个结果时间、最后一帧输入时间及最终结果时间。

TTS 记录整段合成耗时、输出时长和实时率 RTF。流式实验额外测量第一段可解码播放音频到达时间。当前 Tts.synthesize 接口返回完整 PCM，本工具能测得整段合成耗时。

每个方案使用相同输入，先记录一次冷启动，再至少测量 20 次热启动，分别报告中位数、P95、失败次数。记录 CPU、GPU、内存、模型版本、网络、是否代理和运行参数。当前工具按 iteration 保留每次结果，首轮可单独用于分析冷启动。

主观听感用相同文本、匿名编号、固定音量试听，建议至少三名听者按 1 至 5 分评价自然度、停顿、读音和情绪贴合度。保留每位听者的原始打分。说话人相似度衡量声音是否像参考说话人，自然度仍需单独评价。

推荐测试句包括“你好，今天过得怎么样？”、“请在下午三点半提醒我参加会议。”、“这个 API 返回 HTTP 404，应该怎么处理？”和含长句、数字、多音字的内容。ASR 录音覆盖安静、背景噪声、语速变化与中英混合。

## 可复现工具

新增 scripts/bench_provider.py，复用现有 Asr.transcribe 和 Tts.synthesize 接口。使用 module:function 指定无参工厂，工厂返回完成密钥或本地模型配置的 provider。工厂模块放在项目可导入路径，本地凭据通过环境变量提供。

```python
# 本地 bench_factory.py 的示意实现
import os
from providers.openlux_tts import OpenluxTts

def tts():
    return OpenluxTts(api_key=os.environ["LLM_API_KEY"])
```

```shell
python scripts/bench_provider.py --factory bench_factory:tts --kind tts --input "你好，今天过得怎么样？" --repeats 21 --output runs/tts-benchmark.jsonl
python scripts/bench_provider.py --factory bench_factory:asr --kind asr --input sample.wav --repeats 21 --output runs/asr-benchmark.jsonl
pytest tests/test_bench_provider.py -q
```

ASR 工厂同样返回已配置的 Asr 实例。输入要求 16kHz、单声道、16-bit PCM WAV。TTS 输出使用 provider.sample_rate 计算音频时长。输出 JSONL 可供后续统计，runs 目录遵循现有忽略规则。真实接口调用可能产生服务费用。

## 本次结果与后续决策

基线在 Python 3.12 的已有虚拟环境中通过 33 项测试。Windows 临时目录存在用户权限差异，验证时指定独立 basetemp 并关闭 pytest 缓存后通过。

本次尚无真实 ASR/TTS 请求结果。指定目录中检查的旧项目路径未发现 .env，尚缺统一录音集、实际服务配置或本地模型部署。候选推荐依据是官方功能资料和当前接口的适配成本。

组长可据此决定优先分配接口实测还是本地模型部署。建议首轮比较现有基线与 Qwen3-ASR、CosyVoice，完成同输入测量和试听后再决定替换。若优先追求首音延迟，还需评估 TTS 音频流接口以及麦克风采集与 ASR 的边采边传。
