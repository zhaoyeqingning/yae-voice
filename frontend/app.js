const CONTACT = "Evanescia";

const stage = document.getElementById("stage");
const emptyEl = document.getElementById("empty");
const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("btn-send");
const statusEl = document.getElementById("status");
const jumpBtn = document.getElementById("btn-jump");
const drawer = document.getElementById("drawer");
const drawerMask = document.getElementById("drawer-mask");
const callEl = document.getElementById("call");
const callStatus = document.getElementById("call-status");
const caption = document.getElementById("caption");
const holdBtn = document.getElementById("btn-hold");

const messages = [];
let activeGeneration = 0;
let abortController = null;
let phase = "idle";
let stick = true;
let audioCtx = null;
let playChain = Promise.resolve();
let playGeneration = 0;
let pendingText = "";
let capture = null;

function setPhase(next, label) {
  phase = next;
  statusEl.textContent = label;
  const streaming = next === "thinking" || next === "streaming";
  sendBtn.classList.toggle("is-stop", streaming);
  sendBtn.querySelector(".icon-send").hidden = streaming;
  sendBtn.querySelector(".icon-stop").hidden = !streaming;
  sendBtn.disabled = !streaming && !input.value.trim();
  sendBtn.setAttribute("aria-label", streaming ? "停止" : "发送");
  if (callEl.hidden) return;
  if (next === "thinking") setCall("想着…", false);
  else if (next === "streaming") setCall("在说", true);
  else if (next === "idle") setCall("按住说话", false);
}

function setCall(text, talking) {
  callStatus.textContent = text;
  callEl.classList.toggle("is-talking", talking);
  callEl.classList.toggle("is-live", !callEl.hidden && (talking || text === "在听"));
}

function showThread(on) {
  emptyEl.hidden = on;
  thread.hidden = !on;
}

function isNearBottom() {
  return stage.scrollHeight - stage.scrollTop - stage.clientHeight < 96;
}

function follow() {
  if (stick) stage.scrollTop = stage.scrollHeight;
  jumpBtn.hidden = stick || thread.hidden;
}

stage.addEventListener("scroll", () => {
  stick = isNearBottom();
  jumpBtn.hidden = stick || thread.hidden;
});

jumpBtn.addEventListener("click", () => {
  stick = true;
  follow();
});

function render() {
  thread.replaceChildren();
  for (const msg of messages) {
    const row = document.createElement("article");
    row.className = `msg ${msg.role === "user" ? "user" : "ai"}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (msg.status === "thinking") {
      bubble.innerHTML = '<span class="dots" aria-label="想着"><i></i><i></i><i></i></span>';
    } else {
      bubble.textContent = msg.text;
      if (msg.status === "streaming") bubble.classList.add("caret");
    }
    const wrap = document.createElement("div");
    wrap.appendChild(bubble);
    if (msg.status === "cancelled" && msg.text) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = "已中断";
      wrap.appendChild(meta);
    }
    if (msg.status === "error") {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "retry";
      retry.textContent = "重试";
      retry.addEventListener("click", () => {
        if (pendingText) sendText(pendingText);
      });
      wrap.appendChild(retry);
    }
    row.appendChild(wrap);
    thread.appendChild(row);
  }
  showThread(messages.length > 0);
  follow();
  const last = [...messages].reverse().find((item) => item.role === "assistant");
  const mine = [...messages].reverse().find((item) => item.role === "user");
  if (!callEl.hidden) {
    const parts = [];
    if (mine) parts.push(mine.text);
    if (last && last.text) parts.push(last.text);
    caption.textContent = parts.filter(Boolean).join("\n");
  }
}

function growInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  if (phase !== "thinking" && phase !== "streaming") {
    sendBtn.disabled = !input.value.trim();
  }
}

input.addEventListener("input", () => {
  growInput();
  if (phase === "idle") setPhase("idle", "在线");
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function ensureAudio() {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function decodePcm16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const samples = new Int16Array(bytes.buffer);
  const f32 = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) f32[i] = samples[i] / 32768;
  return f32;
}

function stopAudio(generation) {
  playGeneration = generation;
  playChain = Promise.resolve();
}

function enqueueAudio(generation, sampleRate, pcmB64) {
  if (generation !== activeGeneration) return;
  ensureAudio();
  const samples = decodePcm16(pcmB64);
  playChain = playChain.then(async () => {
    if (generation !== activeGeneration || generation !== playGeneration) return;
    const ctx = ensureAudio();
    const buffer = ctx.createBuffer(1, samples.length || 1, sampleRate);
    if (samples.length) buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    if (!callEl.hidden) setCall("在说", true);
    await new Promise((resolve) => {
      src.onended = resolve;
      src.start();
    });
    if (!callEl.hidden && phase === "idle") setCall("按住说话", false);
  });
}

function currentAssistant() {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "assistant" && messages[i].generation === activeGeneration) {
      return messages[i];
    }
  }
  return null;
}

function handleEvent(ev) {
  if (ev.type === "start") {
    activeGeneration = ev.generation;
    stopAudio(ev.generation);
    for (const msg of messages) {
      if (msg.generation == null) msg.generation = ev.generation;
    }
    return;
  }
  if (ev.generation != null && ev.generation !== activeGeneration) return;
  if (ev.type === "user.transcript") {
    const mine = [...messages].reverse().find((item) => item.role === "user" && item.generation === ev.generation);
    if (mine) mine.text = ev.text || "（没听清）";
    if (!ev.text) {
      const ai = currentAssistant();
      if (ai && ai.status === "thinking") messages.splice(messages.indexOf(ai), 1);
      setPhase("idle", "没听清");
    }
    render();
    return;
  }
  if (ev.type === "text.delta") {
    const ai = currentAssistant();
    if (!ai) return;
    ai.status = "streaming";
    ai.text += ev.delta;
    setPhase("streaming", "在说");
    render();
    return;
  }
  if (ev.type === "audio") {
    enqueueAudio(ev.generation, ev.sample_rate || 24000, ev.pcm_b64);
    return;
  }
  if (ev.type === "done") {
    const ai = currentAssistant();
    if (ai) {
      if (!ai.text) messages.splice(messages.indexOf(ai), 1);
      else ai.status = "done";
    }
    setPhase("idle", "在线");
    render();
    return;
  }
  if (ev.type === "cancelled") {
    const ai = currentAssistant();
    if (ai) {
      if (!ai.text) messages.splice(messages.indexOf(ai), 1);
      else ai.status = "cancelled";
    }
    setPhase("idle", "已打断");
    render();
    return;
  }
  if (ev.type === "error") {
    const ai = currentAssistant();
    if (ai) {
      ai.status = "error";
      ai.text = ai.text || `出错了：${ev.message || "未知错误"}`;
    }
    if (pendingText && callEl.hidden) input.value = pendingText;
    growInput();
    setPhase("idle", "出错了");
    render();
  }
}

async function consume(response) {
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").map((item) => item.trim()).find((item) => item.startsWith("data:"));
      if (!line) continue;
      handleEvent(JSON.parse(line.slice(5).trim()));
    }
  }
}

function markStopped() {
  const ai = currentAssistant() || [...messages].reverse().find((item) => item.role === "assistant");
  if (ai && (ai.status === "thinking" || ai.status === "streaming")) {
    if (!ai.text) messages.splice(messages.indexOf(ai), 1);
    else ai.status = "cancelled";
  }
  setPhase("idle", "已打断");
  render();
}

function beginTurn(userText) {
  if (abortController) abortController.abort();
  abortController = new AbortController();
  messages.push({ role: "user", text: userText, status: "done", generation: null });
  messages.push({ role: "assistant", text: "", status: "thinking", generation: null });
  setPhase("thinking", "想着");
  render();
  return abortController;
}

async function sendText(text) {
  const clean = text.trim();
  if (!clean) return;
  pendingText = clean;
  input.value = "";
  growInput();
  ensureAudio();
  const controller = beginTurn(clean);
  try {
    const response = await fetch("/api/turns", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ text: clean }),
      signal: controller.signal,
    });
    const user = [...messages].reverse().find((item) => item.role === "user" && item.generation == null);
    await consume(response);
    if (user && user.generation == null) user.generation = activeGeneration;
    const ai = [...messages].reverse().find((item) => item.role === "assistant" && item.generation == null);
    if (ai) ai.generation = activeGeneration;
  } catch (err) {
    if (err.name === "AbortError") {
      markStopped();
      return;
    }
    const ai = currentAssistant() || [...messages].reverse().find((item) => item.role === "assistant");
    if (ai) {
      ai.status = "error";
      ai.text = ai.text || "没有连上";
    }
    input.value = pendingText;
    growInput();
    setPhase("idle", "出错了");
    render();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (phase === "thinking" || phase === "streaming") {
    if (abortController) abortController.abort();
    return;
  }
  sendText(input.value);
});

document.getElementById("starters").addEventListener("click", (event) => {
  const prompt = event.target?.dataset?.prompt;
  if (prompt) sendText(prompt);
});

function setDrawer(open) {
  drawer.hidden = !open;
  drawerMask.hidden = !open;
}

document.getElementById("btn-drawer").addEventListener("click", () => setDrawer(drawer.hidden));
drawerMask.addEventListener("click", () => setDrawer(false));

document.getElementById("btn-clear").addEventListener("click", async () => {
  if (abortController) abortController.abort();
  messages.length = 0;
  pendingText = "";
  setPhase("idle", "在线");
  render();
  showThread(false);
  setDrawer(false);
  await fetch("/api/session/reset", { method: "POST" }).catch(() => {});
});

document.getElementById("btn-call").addEventListener("click", () => {
  callEl.hidden = false;
  setCall("按住说话", false);
  caption.textContent = "";
});

document.getElementById("btn-hangup").addEventListener("click", () => {
  if (capture) capture.cancel();
  if (abortController && (phase === "thinking" || phase === "streaming")) abortController.abort();
  callEl.hidden = true;
  callEl.classList.remove("is-live", "is-talking");
});

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function downsample(chunks, fromRate, toRate) {
  const merged = new Float32Array(chunks.reduce((sum, item) => sum + item.length, 0));
  let pos = 0;
  for (const chunk of chunks) {
    merged.set(chunk, pos);
    pos += chunk.length;
  }
  if (fromRate === toRate) return merged;
  const ratio = fromRate / toRate;
  const length = Math.floor(merged.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(merged.length, Math.floor((i + 1) * ratio));
    let acc = 0;
    for (let j = start; j < end; j += 1) acc += merged[j];
    out[i] = acc / Math.max(1, end - start);
  }
  return out;
}

async function startCapture() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);
  let stopped = false;
  return {
    cancel() {
      stopped = true;
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      ctx.close();
    },
    async finish() {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      const rate = ctx.sampleRate;
      await ctx.close();
      if (stopped) return null;
      return encodeWav(downsample(chunks, rate, 16000), 16000);
    },
  };
}

async function sendAudio(blob) {
  ensureAudio();
  const controller = beginTurn("…");
  const mine = messages[messages.length - 2];
  const ai = messages[messages.length - 1];
  try {
    const response = await fetch("/api/turns/audio", {
      method: "POST",
      headers: { "Content-Type": "audio/wav", Accept: "text/event-stream" },
      body: blob,
      signal: controller.signal,
    });
    await consume(response);
    if (mine) mine.generation = activeGeneration;
    if (ai) ai.generation = activeGeneration;
  } catch (err) {
    if (err.name === "AbortError") {
      markStopped();
      return;
    }
    if (ai) {
      ai.status = "error";
      ai.text = "通话没接上";
    }
    setPhase("idle", "出错了");
    if (!callEl.hidden) setCall("没接上，再按住试试", false);
    render();
  }
}

async function pressHold() {
  if (capture || phase === "thinking" || phase === "streaming") {
    if (abortController) abortController.abort();
  }
  try {
    capture = await startCapture();
  } catch {
    setCall("麦克风打不开", false);
    return;
  }
  holdBtn.classList.add("is-down");
  setCall("在听", false);
  callEl.classList.add("is-live");
}

async function releaseHold() {
  holdBtn.classList.remove("is-down");
  const current = capture;
  capture = null;
  if (!current) return;
  const blob = await current.finish();
  if (!blob || blob.size < 4000) {
    setCall("太短了，按住再说", false);
    callEl.classList.remove("is-live");
    return;
  }
  setCall("想着…", false);
  await sendAudio(blob);
}

holdBtn.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  holdBtn.setPointerCapture(event.pointerId);
  pressHold();
});
holdBtn.addEventListener("pointerup", releaseHold);
holdBtn.addEventListener("pointercancel", releaseHold);

fetch("/api/health")
  .then((response) => response.json())
  .then((data) => setPhase("idle", data.mode === "live" ? "在线" : "在线 · 演示"))
  .catch(() => setPhase("idle", "没连上"));
