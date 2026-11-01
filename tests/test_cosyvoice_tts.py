"""CosyVoice 的离线测试：协议字段、音频帧、错误处理（本地假服务端，不联网）。"""

import asyncio
import base64
import json

import websockets

from providers.cosyvoice_tts import (
    CosyVoiceError,
    CosyVoiceTts,
    build_continue_task,
    build_finish_task,
    build_run_task,
    parse_event,
)


async def _with_server(handler, callback):
    """起一个本地假 DashScope 服务端，把 url 交给回调。"""
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        return await callback(f"ws://127.0.0.1:{port}")


def _task_id(raw: str) -> str:
    return json.loads(raw)["header"]["task_id"]


def _event(name: str, task_id: str, **payload) -> str:
    return json.dumps({"header": {"task_id": task_id, "event": name}, "payload": payload})


def test_run_task_declares_24k_pcm_duplex():
    task = build_run_task(task_id="t1")
    assert task["header"] == {
        "action": "run-task",
        "task_id": "t1",
        "streaming": "duplex",
    }
    payload = task["payload"]
    assert payload["model"] == "cosyvoice-v2"
    assert payload["function"] == "SpeechSynthesizer"
    # 24k 必须显式写：CosyVoice 默认不是 24k，写错播放端音调会偏低
    assert payload["parameters"]["sample_rate"] == 24000
    assert payload["parameters"]["format"] == "pcm"
    assert payload["parameters"]["voice"] == "longxiaochun_v2"


def test_continue_and_finish_reuse_the_task_id():
    cont = build_continue_task("t1", "你好。")
    assert cont["header"]["action"] == "continue-task"
    assert cont["header"]["task_id"] == "t1"
    assert cont["payload"]["input"]["text"] == "你好。"

    finish = build_finish_task("t1")
    assert finish["header"]["action"] == "finish-task"
    assert finish["header"]["task_id"] == "t1"


def test_result_generated_decodes_audio():
    raw = json.dumps(
        {
            "header": {"event": "result-generated", "task_id": "t1"},
            "payload": {
                "output": {"audio": {"data": base64.b64encode(b"\x01\x02").decode()}}
            },
        }
    )
    event = parse_event(raw)
    assert event["event"] == "result-generated"
    assert event["audio"] == b"\x01\x02"
    assert event["error"] is None


def test_unpadded_base64_frame_is_accepted():
    raw = json.dumps(
        {
            "header": {"event": "result-generated"},
            "payload": {"output": {"audio": {"data": "AQI"}}},
        }
    )
    assert parse_event(raw)["audio"] == b"\x01\x02"


def test_task_failed_is_reported_with_code():
    raw = json.dumps(
        {
            "header": {
                "event": "task-failed",
                "error_code": "InvalidApiKey",
                "error_message": "invalid api key",
            },
            "payload": {},
        }
    )
    event = parse_event(raw)
    assert event["error"] is not None
    assert "InvalidApiKey" in event["error"]


def test_synthesize_collects_json_audio_chunks():
    async def scenario():
        received = {}

        async def handler(ws):
            run_task = json.loads(await ws.recv())
            received["run"] = run_task
            request = getattr(ws, "request", None)
            if request is not None:
                received["auth"] = request.headers.get("Authorization")
            task_id = run_task["header"]["task_id"]
            await ws.send(_event("task-started", task_id, output={}))
            received["continue"] = json.loads(await ws.recv())
            received["finish"] = json.loads(await ws.recv())
            for chunk in (b"\x01\x02", b"\x03\x04"):
                await ws.send(
                    json.dumps(
                        {
                            "header": {"task_id": task_id, "event": "result-generated"},
                            "payload": {
                                "output": {
                                    "audio": {
                                        "data": base64.b64encode(chunk).decode("ascii")
                                    }
                                }
                            },
                        }
                    )
                )
            await ws.send(_event("task-finished", task_id, output={}))
            await ws.wait_closed()

        async def call(url):
            tts = CosyVoiceTts(api_key="test-key", base_url=url)
            return await tts.synthesize("你好。")

        pcm = await _with_server(handler, call)
        assert pcm == b"\x01\x02\x03\x04"
        assert received["auth"] == "bearer test-key"
        assert received["run"]["payload"]["parameters"]["sample_rate"] == 24000
        assert received["continue"]["header"]["action"] == "continue-task"
        assert received["continue"]["payload"]["input"]["text"] == "你好。"
        assert received["finish"]["header"]["action"] == "finish-task"

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_synthesize_accepts_binary_audio_frames():
    """真机返回的是二进制 PCM 帧，不是 JSON base64 —— 不能拿 json.loads 去解。"""

    async def scenario():
        async def handler(ws):
            task_id = _task_id(await ws.recv())
            await ws.send(_event("task-started", task_id, output={}))
            await ws.recv()
            await ws.recv()
            await ws.send(b"\x01\x02\x03\x04")
            await ws.send(b"\x05\x06")
            await ws.send(_event("task-finished", task_id, output={}))
            await ws.wait_closed()

        async def call(url):
            tts = CosyVoiceTts(api_key="k", base_url=url)
            return await tts.synthesize("你好。")

        return await _with_server(handler, call)

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=10)) == (
        b"\x01\x02\x03\x04\x05\x06"
    )


def test_synthesize_drops_half_of_a_16bit_frame():
    async def scenario():
        async def handler(ws):
            task_id = _task_id(await ws.recv())
            await ws.send(_event("task-started", task_id, output={}))
            await ws.recv()
            await ws.recv()
            await ws.send(b"\x01\x02\x03")
            await ws.send(_event("task-finished", task_id, output={}))
            await ws.wait_closed()

        async def call(url):
            tts = CosyVoiceTts(api_key="k", base_url=url)
            return await tts.synthesize("嗯。")

        return await _with_server(handler, call)

    # 3 字节里最后半个样本会让整段错位，必须丢掉
    assert asyncio.run(asyncio.wait_for(scenario(), timeout=10)) == b"\x01\x02"


def test_synthesize_raises_on_task_failed():
    async def scenario():
        async def handler(ws):
            await ws.recv()
            await ws.send(
                json.dumps(
                    {
                        "header": {
                            "event": "task-failed",
                            "error_code": "Throttling",
                            "error_message": "too many requests",
                        },
                        "payload": {},
                    }
                )
            )
            await ws.wait_closed()

        async def call(url):
            tts = CosyVoiceTts(api_key="k", base_url=url)
            try:
                await tts.synthesize("你好。")
            except CosyVoiceError as exc:
                return str(exc)
            raise AssertionError("task-failed 应当抛出 CosyVoiceError")

        return await _with_server(handler, call)

    assert "Throttling" in asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_rejects_unsupported_sample_rate():
    try:
        CosyVoiceTts(api_key="k", sample_rate=12345)
    except ValueError as exc:
        assert "12345" in str(exc)
    else:
        raise AssertionError("不支持的采样率应当报错")
