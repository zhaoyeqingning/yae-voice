"""文字与通话入口：近黑陪伴前端 + POST/SSE。"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from session import Session
from web_audio import pcm16_from_wav
from web_hub import Hub, HubPlayer

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


class TurnBody(BaseModel):
    text: str = Field(min_length=1)


class AppState:
    def __init__(self) -> None:
        self.hub = Hub()
        self.session: Session | None = None
        self.mode = "fake"
        self.turn_lock = asyncio.Lock()


state = AppState()


async def _build_live_session(player: HubPlayer, system_prompt: str) -> Session:
    from live_smoke import resolve_model
    from providers.config import load_settings
    from providers.openlux_llm import OpenluxLlm
    from providers.openlux_tts import OpenluxTts
    from providers.xfyun_asr import XfyunAsr

    settings = load_settings()
    model = await resolve_model(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model
    )
    print(
        f"LLM model={model} TTS={settings.tts_model} voice={settings.tts_voice} speed={settings.tts_speed}",
        flush=True,
    )
    return Session(
        XfyunAsr(
            app_id=settings.xfyun_app_id,
            api_key=settings.xfyun_api_key,
            api_secret=settings.xfyun_api_secret,
        ),
        OpenluxLlm(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=model,
            enable_thinking=settings.llm_enable_thinking,
        ),
        OpenluxTts(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.tts_model,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            instructions=settings.tts_instructions,
        ),
        player,
        log_path=ROOT / "runs" / "web_turns.jsonl",
        system_prompt=system_prompt,
        on_text_delta=state.hub.on_text_delta,
        on_user_text=state.hub.on_user_text,
    )


def _build_fake_session(player: HubPlayer, system_prompt: str) -> Session:
    from providers.fake import FakeAsr, FakeLlm, FakeTts

    return Session(
        FakeAsr(),
        FakeLlm("嗯。我在听。有什么想说的吗？"),
        FakeTts(),
        player,
        log_path=ROOT / "runs" / "web_turns.jsonl",
        system_prompt=system_prompt,
        on_text_delta=state.hub.on_text_delta,
        on_user_text=state.hub.on_user_text,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "你是 Evanescia，语气轻松自然，用简体中文短句回复，像面对面聊天。",
    )
    player = HubPlayer(state.hub)
    use_fake = os.getenv("YAE_WEB_FAKE", "").strip().lower() in {"1", "true", "yes"}
    if use_fake:
        state.session = _build_fake_session(player, system_prompt)
        state.mode = "fake"
    else:
        try:
            state.session = await _build_live_session(player, system_prompt)
            state.mode = "live"
        except Exception as exc:
            print(f"真实接口不可用，改用假链路: {exc}", flush=True)
            state.session = _build_fake_session(player, system_prompt)
            state.mode = "fake"
    print(f"web server mode={state.mode}", flush=True)
    yield


app = FastAPI(title="Yae voice web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "mode": state.mode}


@app.post("/api/session/reset")
async def reset_session() -> dict[str, bool]:
    session = state.session
    if session is not None:
        task = session._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        session.history.clear()
    return {"ok": True}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_turn(
    request: Request,
    *,
    text: str | None = None,
    audio: bytes | None = None,
    input_kind: str,
) -> AsyncIterator[str]:
    session = state.session
    if session is None:
        yield _sse({"type": "error", "message": "session not ready"})
        return

    async with state.turn_lock:
        predicted = session.generation + 1
        relay = await state.hub.attach(predicted)
        if audio is not None:
            generation = await session.begin_audio(audio)
        else:
            generation = await session.begin_text(text or "")
        if generation != predicted:
            relay = await state.hub.attach(generation)

    async def watch() -> None:
        cancelled = session.generation != generation
        try:
            await session._await_generation(generation)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            await relay.publish(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
            await relay.close()
            return
        if session.generation != generation:
            cancelled = True
        if session.last_error and not cancelled:
            await relay.publish({"type": "error", "message": session.last_error})
        else:
            await relay.publish({"type": "cancelled" if cancelled else "done"})
        await relay.close()

    watcher = asyncio.create_task(watch())
    yield _sse({"type": "start", "generation": generation, "input": input_kind})
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(relay.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                break
            yield _sse(item)
            if item.get("type") in {"done", "cancelled", "error"}:
                break
    finally:
        task = session._task
        if session.generation == generation and task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            await session.player.stop()
        if not watcher.done():
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


@app.post("/api/turns")
async def create_turn(body: TurnBody, request: Request) -> StreamingResponse:
    text = body.text.strip()
    if not text:
        return _sse_response(iter([_sse({"type": "error", "message": "empty text"})]))
    return _sse_response(_stream_turn(request, text=text, input_kind="text"))


@app.post("/api/turns/audio")
async def create_audio_turn(request: Request) -> StreamingResponse:
    raw = await request.body()
    if len(raw) > 8_000_000:
        return _sse_response(iter([_sse({"type": "error", "message": "录音太大"})]))
    try:
        pcm = pcm16_from_wav(raw)
    except ValueError as exc:
        return _sse_response(iter([_sse({"type": "error", "message": str(exc)})]))
    return _sse_response(_stream_turn(request, audio=pcm, input_kind="audio"))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "web_server:app",
        host=os.getenv("YAE_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("YAE_WEB_PORT", "8787")),
        reload=False,
    )


if __name__ == "__main__":
    main()
