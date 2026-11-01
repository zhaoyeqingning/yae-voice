"""豆包 TTS 的离线测试：帧编解码 + 本地假服务端上的完整会话。"""

import asyncio
import json
import struct

import websockets

from providers.doubao_protocol import (
    EventType,
    Message,
    MsgType,
    MsgTypeFlagBits,
    finish_session,
    start_connection,
)
from providers.doubao_tts import DoubaoTts, DoubaoTtsError, speech_rate_from_ratio

SERVER_FULL = int(MsgType.FullServerResponse)
SERVER_AUDIO = int(MsgType.AudioOnlyServer)
WITH_EVENT = int(MsgTypeFlagBits.WithEvent)
CONNECTION_EVENTS = {
    int(EventType.ConnectionStarted),
    int(EventType.ConnectionFailed),
    int(EventType.ConnectionFinished),
}


def server_frame(
    msg_type: int,
    event: int,
    *,
    session_id: str = "",
    connect_id: str = "",
    payload: bytes = b"",
) -> bytes:
    """按 protocols.py 的读侧规则拼一帧服务端消息。

    connection 级事件带 connect_id，其余带 session_id；两者都是 uint32 长度 + bytes。
    """
    frame = bytearray(
        [
            (1 << 4) | 1,               # version 1 / header size 4
            (msg_type << 4) | WITH_EVENT,
            (1 << 4) | 0,               # JSON / 不压缩
            0,
        ]
    )
    frame += struct.pack(">i", event)
    if event in CONNECTION_EVENTS:
        raw = connect_id.encode()
        frame += struct.pack(">I", len(raw)) + raw
    else:
        raw = session_id.encode()
        frame += struct.pack(">I", len(raw)) + raw
    frame += struct.pack(">I", len(payload)) + payload
    return bytes(frame)


def test_protocol_roundtrip():
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.StartSession
    msg.session_id = "s-1"
    msg.payload = b'{"req_params":{}}'
    parsed = Message.from_bytes(msg.marshal())
    assert parsed.event == EventType.StartSession
    assert parsed.session_id == "s-1"
    assert parsed.payload == b'{"req_params":{}}'


def test_speech_rate_mapping():
    assert speech_rate_from_ratio(1.0) == 0
    assert speech_rate_from_ratio(1.5) == 50
    assert speech_rate_from_ratio(2.0) == 100
    assert speech_rate_from_ratio(0.5) == -50
    # 超出范围要夹住
    assert speech_rate_from_ratio(3.0) == 100
    assert speech_rate_from_ratio(0.1) == -50


def test_synthesize_collects_audio_and_sends_expected_params():
    async def scenario():
        received: dict[str, object] = {}

        async def handler(ws):
            start = Message.from_bytes(await ws.recv())
            assert start.event == EventType.StartConnection
            await ws.send(
                server_frame(SERVER_FULL, int(EventType.ConnectionStarted), connect_id="c-1")
            )

            session = Message.from_bytes(await ws.recv())
            assert session.event == EventType.StartSession
            received["session"] = json.loads(session.payload)
            received["session_id"] = session.session_id
            await ws.send(
                server_frame(
                    SERVER_FULL,
                    int(EventType.SessionStarted),
                    session_id=session.session_id,
                )
            )

            task = Message.from_bytes(await ws.recv())
            assert task.event == EventType.TaskRequest
            received["text"] = json.loads(task.payload)["req_params"]["text"]

            # 中间插一个 UsageResponse，适配器应当忽略它继续等
            await ws.send(
                server_frame(
                    SERVER_FULL,
                    int(EventType.UsageResponse),
                    session_id=session.session_id,
                    payload=b'{"usage":{"text_words":6}}',
                )
            )
            for chunk in (b"\x01\x02", b"\x03\x04"):
                await ws.send(
                    server_frame(
                        SERVER_AUDIO,
                        int(EventType.TTSResponse),
                        session_id=session.session_id,
                        payload=chunk,
                    )
                )

            finish = Message.from_bytes(await ws.recv())
            assert finish.event == EventType.FinishSession
            await ws.send(
                server_frame(
                    SERVER_FULL,
                    int(EventType.SessionFinished),
                    session_id=session.session_id,
                    payload=b'{"usage":{"text_words":6}}',
                )
            )

            end = Message.from_bytes(await ws.recv())
            assert end.event == EventType.FinishConnection
            await ws.send(
                server_frame(SERVER_FULL, int(EventType.ConnectionFinished), connect_id="c-1")
            )
            await ws.wait_closed()

        async def call(url):
            tts = DoubaoTts(api_key="k", speaker="zh_female_test_uranus_bigtts", url=url)
            return await tts.synthesize("你好。")

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            pcm = await call(f"ws://127.0.0.1:{port}")
        return pcm, received

    pcm, received = asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    assert pcm == b"\x01\x02\x03\x04"
    assert received["text"] == "你好。"
    params = received["session"]["req_params"]
    assert params["speaker"] == "zh_female_test_uranus_bigtts"
    assert params["audio_params"]["format"] == "pcm"
    assert params["audio_params"]["sample_rate"] == 24000


def test_error_frame_is_raised_as_readable_error():
    async def scenario():
        async def handler(ws):
            await ws.recv()
            # 注意：protocols.py 里 Error 帧的"读侧"顺序是 error_code → event → session → payload
            frame = bytearray(
                [0x11, (int(MsgType.Error) << 4) | WITH_EVENT, 0x10, 0]
            )
            frame += struct.pack(">I", 401)        # error code
            frame += struct.pack(">i", 0)          # event
            frame += struct.pack(">I", 0)          # session_id 长度 0
            body = "invalid api key".encode()
            frame += struct.pack(">I", len(body)) + body
            await ws.send(bytes(frame))
            await ws.wait_closed()

        async def call(url):
            tts = DoubaoTts(api_key="bad", speaker="spk", url=url)
            try:
                await tts.synthesize("你好。")
            except DoubaoTtsError as exc:
                return str(exc)
            raise AssertionError("错误帧应当抛 DoubaoTtsError")

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await call(f"ws://127.0.0.1:{port}")

    message = asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    assert "401" in message
    assert "invalid api key" in message


def test_empty_text_returns_without_connecting():
    async def scenario():
        tts = DoubaoTts(api_key="k", speaker="spk", url="ws://127.0.0.1:1")
        return await tts.synthesize("   ")

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=5)) == b""


def test_client_helpers_send_parseable_frames():
    async def scenario():
        async def handler(ws):
            first = Message.from_bytes(await ws.recv())
            assert first.event == EventType.StartConnection
            await finish_session(ws, "s-1")
            await ws.send(b"")
            await ws.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await start_connection(ws)
                await asyncio.sleep(0.05)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))
