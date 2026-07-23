"""Streaming (real-time) STT adapters for the accuracy + latency benchmark.

The cloud adapters in cloud_adapters.py send a whole clip to a batch endpoint and
return one transcript. These adapters instead open a provider's live streaming
socket, feed the same clip at real-time pace (1x wall-clock, ~100ms per chunk), and
capture every partial and final result with a timestamp. From that we get two things
batch mode cannot show:

  - the FINALIZED transcript (all is_final segments joined), written in the same
    {"text": ...} shape the batch runner uses, so score_ood.py scores streaming and
    batch on identical clips with the same WER/CER and confidence intervals; and
  - real-time latency: time-to-first-partial, time-to-first-final, and finalization
    lag (how long after the audio ends the transcript keeps changing).

Each adapter is `fn(wav_path, cfg) -> StreamResult`, registered in PROVIDERS_STREAM
with the same fields cloud_adapters.PROVIDERS uses (engine label carries a `-stream`
suffix; model is env-overridable). Auth comes from the same env vars; nothing here
logs a key.

Smoke test one adapter on 20 LibriSpeech clips (needs data_librispeech prepped and
the provider key in .env.providers):

    ./.venv/bin/python src/streaming_adapters.py deepgram --n 20
"""
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import requests
import soundfile as sf
import websockets

from cloud_adapters import (
    LANG_AMAZON,
    LANG_ASSEMBLYAI,
    LANG_BARE,
    LANG_DEEPGRAM,
    LANG_SPEECHMATICS,
    TranscribeError,
    _env,
    provider_language,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_MS = 100  # audio pushed per send; real-time pacing keeps send rate at 1x


@dataclass
class StreamResult:
    """One clip through one streaming engine.

    text        finalized transcript (is_final segments joined, in order)
    events      (t_s, is_final, text) for every result, t_s from first chunk sent
    audio_end_s wall time when the last audio chunk was sent (~ the clip duration)
    wall_s      wall time when the socket closed (first chunk sent -> stream done)
    """

    text: str
    events: list = field(default_factory=list)
    audio_end_s: float = 0.0
    wall_s: float = 0.0


class _Clock:
    """Monotonic wall clock zeroed at the first audio chunk, in seconds."""

    def __init__(self, loop):
        self._loop = loop
        self._start = loop.time()

    def now(self):
        return self._loop.time() - self._start


def _read_pcm(wav):
    """16-bit mono PCM bytes plus sample rate (first channel if stereo)."""
    data, sr = sf.read(wav, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    return data.tobytes(), int(sr)


async def _feed_paced(send, pcm, sr, clock, chunk_ms=CHUNK_MS):
    """Push PCM through `send` (an async one-arg callable) at ~1x real time. Returns
    the audio-end wall time. `send` is a socket send for WebSocket engines, or an SDK
    push call for engines like Amazon."""
    step = int(sr * chunk_ms / 1000) * 2  # int16 => 2 bytes per frame
    period = chunk_ms / 1000.0
    sent = 0
    for i in range(0, len(pcm), step):
        await send(pcm[i:i + step])
        sent += 1
        delay = sent * period - clock.now()  # keep sends aligned to real time
        if delay > 0:
            await asyncio.sleep(delay)
    return round(clock.now(), 3)


def compute_metrics(result):
    """Derive the real-time latency signals from a StreamResult's event log."""
    events = result.events
    nonempty = [e for e in events if e[2].strip()]
    ttfp = nonempty[0][0] if nonempty else None  # first non-empty result of any kind
    ttff = next((t for (t, is_final, _tx) in nonempty if is_final), None)
    last_change = events[-1][0] if events else None
    lag = (
        round(last_change - result.audio_end_s, 3)
        if last_change is not None
        else None
    )
    partials = [tx for (_t, is_final, tx) in events if not is_final]
    revisions, prev = 0, None
    for tx in partials:
        if tx != prev:
            revisions += 1
        prev = tx
    return {
        "ttfp_s": ttfp,
        "ttff_s": ttff,
        "finalization_lag_s": lag,
        "n_partials": len(partials),
        "n_revisions": revisions,
        "stream_wall_s": round(result.wall_s, 3),
    }


# --- Shared WebSocket runner ----------------------------------------------------

async def _ws_stream(url, headers, pcm, sr, handle, pre_send=None, close_msg=None):
    """Open a WebSocket, optionally send a config frame, feed PCM at real-time pace,
    and hand every received frame to `handle(msg, t_s)`. The clock is zeroed at the
    first audio chunk, so `handle` timestamps are relative to audio start. Returns
    (audio_end_s, wall_s)."""
    audio_end = 0.0
    loop = asyncio.get_running_loop()
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        if pre_send is not None:
            await ws.send(pre_send)
        clock = _Clock(loop)

        async def feed():
            nonlocal audio_end
            audio_end = await _feed_paced(ws.send, pcm, sr, clock)
            if close_msg is not None:
                await ws.send(close_msg)

        async def recv():
            async for msg in ws:
                handle(msg, round(clock.now(), 3))

        await asyncio.gather(feed(), recv())
        wall = round(clock.now(), 3)
    return audio_end, wall


def _run(url, headers, pcm, sr, handle, build_text, pre_send=None, close_msg=None,
         label="stream"):
    """Drive one clip through _ws_stream and package a StreamResult. `handle` appends
    (t, is_final, text) tuples to the shared `events` list; `build_text(events)`
    returns the finalized transcript."""
    events = []
    try:
        audio_end, wall = asyncio.run(
            _ws_stream(url, headers, pcm, sr,
                       lambda msg, t: handle(msg, t, events),
                       pre_send=pre_send, close_msg=close_msg))
    except (OSError, websockets.WebSocketException) as e:
        raise TranscribeError(f"{label} failed: {type(e).__name__}: {e}") from e
    return StreamResult(text=build_text(events).strip(), events=events,
                        audio_end_s=audio_end, wall_s=wall)


# --- Deepgram -------------------------------------------------------------------

def _deepgram_handle(msg, t, events):
    if isinstance(msg, bytes):
        return
    data = json.loads(msg)
    if data.get("type") != "Results":
        return
    alts = data.get("channel", {}).get("alternatives", [])
    events.append((t, bool(data.get("is_final")), alts[0]["transcript"] if alts else ""))


def deepgram_stream(wav, cfg):
    """Deepgram live streaming (wss /v1/listen). Same model + language flags as the
    batch adapter (punctuate on, smart_format off) so the transcripts are comparable;
    interim_results on so we see partials."""
    key = _env("DEEPGRAM_API_KEY")
    model = _env("DEEPGRAM_MODEL", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    params = {
        "model": model,
        "language": provider_language(cfg),
        "encoding": "linear16",
        "sample_rate": sr,
        "channels": 1,
        "interim_results": "true",
        "punctuate": "true",
        "smart_format": "false",
    }
    url = "wss://api.deepgram.com/v1/listen?" + urlencode(params)
    return _run(url, {"Authorization": f"Token {key}"}, pcm, sr,
                _deepgram_handle,
                lambda ev: " ".join(tx for _t, f, tx in ev if f and tx.strip()),
                close_msg=json.dumps({"type": "CloseStream"}), label="deepgram stream")


# --- Soniox ---------------------------------------------------------------------
# Auth is in the first JSON config frame (no header). Tokens carry is_final and
# already include their own spacing, so the transcript is the concatenation of final
# token texts. `<end>` is an endpoint marker token, not words. The stream ends when we
# send an empty TEXT frame (an empty BINARY frame is NOT accepted and the server 408s);
# the server then finalizes, replies "finished", and closes.

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


def _soniox_handle(msg, t, events, state):
    data = json.loads(msg)
    if data.get("error_code") is not None:
        raise TranscribeError(f"soniox {data['error_code']}: {data.get('error_message')}")
    tokens = data.get("tokens", [])
    got_final = False
    pending = ""
    for tok in tokens:
        txt = tok.get("text", "")
        if not txt or txt == "<end>":
            continue
        if tok.get("is_final"):
            state["finals"].append(txt)
            got_final = True
        else:
            pending += txt
    running = "".join(state["finals"]) + pending
    if running:
        state["last"] = running
    if tokens:
        events.append((t, got_final, running.strip()))


def soniox_stream(wav, cfg):
    key = _env("SONIOX_API_KEY")
    model = _env("SONIOX_MODEL", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    config = json.dumps({
        "api_key": key,
        "model": model,
        "audio_format": "pcm_s16le",
        "sample_rate": sr,
        "num_channels": 1,
        "language_hints": [provider_language(cfg)],
        "enable_endpoint_detection": True,
    })
    state = {"finals": [], "last": ""}
    return _run(SONIOX_WS_URL, None, pcm, sr,
                lambda msg, t, ev: _soniox_handle(msg, t, ev, state),
                lambda _ev: "".join(state["finals"]) or state["last"],
                pre_send=config, close_msg="", label="soniox stream")


# --- AssemblyAI (Universal-Streaming v3) ----------------------------------------
# Config is query params; auth is the raw key in the Authorization header. Turns are
# emitted repeatedly; a turn is final when end_of_turn AND turn_is_formatted are true
# (format_turns=true gives a punctuated final). Keep the latest final per turn_order
# and join them in order. Terminate ends the session.

def _assemblyai_handle(msg, t, events, finals):
    data = json.loads(msg)
    if data.get("type") == "Error" or data.get("error"):
        raise TranscribeError(f"assemblyai: {data.get('error') or data}")
    if data.get("type") != "Turn":
        return
    transcript = data.get("transcript", "") or ""
    is_final = bool(data.get("end_of_turn")) and bool(data.get("turn_is_formatted"))
    if is_final:
        finals[data.get("turn_order", len(finals))] = transcript
    events.append((t, is_final, transcript))


def assemblyai_stream(wav, cfg):
    key = _env("ASSEMBLYAI_API_KEY")
    model = _env("ASSEMBLYAI_STREAM_MODEL", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    params = {"sample_rate": sr, "speech_model": model, "format_turns": "true"}
    url = "wss://streaming.assemblyai.com/v3/ws?" + urlencode(params)
    finals = {}
    return _run(url, {"Authorization": key}, pcm, sr,
                lambda msg, t, ev: _assemblyai_handle(msg, t, ev, finals),
                lambda _ev: " ".join(finals[k] for k in sorted(finals)),
                close_msg=json.dumps({"type": "Terminate"}), label="assemblyai stream")


# --- Amazon Transcribe ----------------------------------------------------------
# Amazon has no batch REST endpoint, so cloud_adapters already streamed it, but it
# discarded partials and fed the clip as fast as possible. Here it uses the shared
# paced feeder and keeps every partial, so it produces the same latency signals as
# the WebSocket engines.

async def _amazon_run(pcm, sr, language, region):
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler

    events = []
    audio_end = 0.0
    clock = _Clock(asyncio.get_running_loop())
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language, media_sample_rate_hz=sr, media_encoding="pcm",
    )

    class Handler(TranscriptResultStreamHandler):
        async def handle_transcript_event(self, event):
            for res in event.transcript.results:
                if res.alternatives:
                    events.append((round(clock.now(), 3), not res.is_partial,
                                   res.alternatives[0].transcript))

    async def feed():
        nonlocal audio_end

        async def send(chunk):
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        audio_end = await _feed_paced(send, pcm, sr, clock)
        await stream.input_stream.end_stream()

    await asyncio.gather(feed(), Handler(stream.output_stream).handle_events())
    wall = clock.now()
    finals = [tx for (_t, is_final, tx) in events if is_final and tx.strip()]
    return StreamResult(text=" ".join(finals).strip(), events=events,
                        audio_end_s=audio_end, wall_s=round(wall, 3))


def amazon_stream(wav, cfg):
    region = _env("AWS_REGION")
    language = provider_language(cfg)
    pcm, sr = _read_pcm(wav)
    try:
        return asyncio.run(_amazon_run(pcm, sr, language, region))
    except (OSError, RuntimeError) as e:
        raise TranscribeError(f"amazon stream failed: {type(e).__name__}: {e}") from e


# --- Gladia (live v2) -----------------------------------------------------------
# Two-step: POST /v2/live with the config returns a tokenized wss URL; connect there
# (no auth header, token is in the URL). Audio is raw PCM binary frames. Transcript
# messages carry data.is_final and data.utterance.text. `stop_recording` ends it.

def _gladia_handle(msg, t, events, finals):
    data = json.loads(msg)
    if data.get("type") == "error" or data.get("error"):
        raise TranscribeError(f"gladia: {data.get('error') or data}")
    if data.get("type") != "transcript":
        return
    d = data.get("data", {})
    text = (d.get("utterance", {}) or {}).get("text", "") or ""
    is_final = bool(d.get("is_final"))
    if is_final:
        finals.append(text)
    events.append((t, is_final, text))


def gladia_stream(wav, cfg):
    key = _env("GLADIA_API_KEY")
    model = _env("GLADIA_MODEL", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    init = requests.post(
        "https://api.gladia.io/v2/live",
        headers={"x-gladia-key": key},
        json={"encoding": "wav/pcm", "sample_rate": sr, "bit_depth": 16, "channels": 1,
              "model": model,
              "language_config": {"languages": [provider_language(cfg)],
                                  "code_switching": False},
              "messages_config": {"receive_partial_transcripts": True}},
        timeout=30)
    try:
        init.raise_for_status()
    except requests.HTTPError as e:
        raise TranscribeError(f"gladia init failed: {e}") from e
    url = init.json()["url"]
    finals = []
    return _run(url, None, pcm, sr,
                lambda msg, t, ev: _gladia_handle(msg, t, ev, finals),
                lambda _ev: " ".join(finals),
                close_msg=json.dumps({"type": "stop_recording"}), label="gladia stream")


# --- Cartesia (Ink-Whisper, manual endpoint) ------------------------------------
# /stt/websocket streams transcript deltas that are additive within a turn, so the
# fullest text is the last non-empty message. `finalize` flushes, `done` ends. Only
# English is offered on the streaming endpoint.

def _cartesia_handle(msg, t, events, state):
    data = json.loads(msg)
    typ = data.get("type")
    if typ == "error":
        raise TranscribeError(f"cartesia: {data.get('message') or data}")
    text = data.get("text", "") or ""
    is_final = typ in ("turn.end", "final") or bool(data.get("is_final"))
    if text:
        state["last"] = text
    if typ in ("transcript", "turn.update", "turn.end", "final") or text:
        events.append((t, is_final, text))


def cartesia_stream(wav, cfg):
    key = _env("CARTESIA_API_KEY")
    model = _env("CARTESIA_MODEL", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    params = {"model": model, "encoding": "pcm_s16le", "sample_rate": sr,
              "cartesia_version": "2026-03-01", "language": provider_language(cfg)}
    url = "wss://api.cartesia.ai/stt/websocket?" + urlencode(params)
    state = {"last": ""}
    return _run(url, {"Authorization": f"Bearer {key}", "Cartesia-Version": "2026-03-01"},
                pcm, sr,
                lambda msg, t, ev: _cartesia_handle(msg, t, ev, state),
                lambda _ev: state["last"],
                close_msg=json.dumps({"type": "finalize"}), label="cartesia stream")


# --- Speechmatics (realtime v2) -------------------------------------------------
# Bespoke handshake: send StartRecognition, WAIT for RecognitionStarted before any
# audio, then feed. EndOfStream must carry last_seq_no = number of audio chunks sent.
# AddPartialTranscript = partials, AddTranscript = finalized segments (concatenate).

SPEECHMATICS_RT_URL = "wss://eu2.rt.speechmatics.com/v2"


async def _speechmatics_run(pcm, sr, language, op, key):
    events, finals = [], []
    audio_end = 0.0
    loop = asyncio.get_running_loop()
    clock = None
    async with websockets.connect(SPEECHMATICS_RT_URL,
                                  additional_headers={"Authorization": f"Bearer {key}"},
                                  max_size=None) as ws:
        await ws.send(json.dumps({
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": sr},
            "transcription_config": {"language": language, "operating_point": op,
                                     "enable_partials": True, "max_delay": 0.7}}))
        ready = asyncio.Event()

        async def recv():
            nonlocal clock
            async for msg in ws:
                d = json.loads(msg)
                m = d.get("message")
                if m == "RecognitionStarted":
                    ready.set()
                    continue
                if m == "Error":
                    raise TranscribeError(f"speechmatics {d.get('type')}: {d.get('reason')}")
                if m == "EndOfTranscript":
                    break
                if clock is None:
                    continue
                t = round(clock.now(), 3)
                if m == "AddPartialTranscript":
                    events.append((t, False, d.get("metadata", {}).get("transcript", "")))
                elif m == "AddTranscript":
                    finals.append(d.get("metadata", {}).get("transcript", ""))
                    events.append((t, True, " ".join(finals)))

        async def feed():
            nonlocal clock, audio_end
            await ready.wait()
            clock = _Clock(loop)
            step = int(sr * CHUNK_MS / 1000) * 2
            seq = 0
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                seq += 1
                delay = seq * (CHUNK_MS / 1000.0) - clock.now()
                if delay > 0:
                    await asyncio.sleep(delay)
            audio_end = round(clock.now(), 3)
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq}))

        await asyncio.gather(recv(), feed())
        wall = round(clock.now(), 3) if clock else 0.0
    return StreamResult(text=" ".join(finals).strip(), events=events,
                        audio_end_s=audio_end, wall_s=wall)


def speechmatics_stream(wav, cfg):
    key = _env("SPEECHMATICS_API_KEY")
    op = _env("SPEECHMATICS_OPERATING_POINT", cfg["model"], required=False)
    pcm, sr = _read_pcm(wav)
    try:
        return asyncio.run(_speechmatics_run(pcm, sr, provider_language(cfg), op, key))
    except (OSError, websockets.WebSocketException) as e:
        raise TranscribeError(f"speechmatics stream failed: {type(e).__name__}: {e}") from e


# --- Registry -------------------------------------------------------------------
# Same shape as cloud_adapters.PROVIDERS, minus `workers`: streaming runs one clip at
# a time so real-time pacing is honest, so there is no per-provider concurrency knob.
PROVIDERS_STREAM = {
    "deepgram": {"engine": "deepgram-nova-3-stream", "model": "nova-3",
                 "fn": deepgram_stream, "languages": LANG_DEEPGRAM},
    "amazon": {"engine": "amazon-transcribe-stream", "model": "",
               "fn": amazon_stream, "languages": LANG_AMAZON},
    "soniox": {"engine": "soniox-stream", "model": "stt-rt-v5",
               "fn": soniox_stream, "languages": LANG_BARE},
    "assemblyai": {"engine": "assemblyai-universal-stream", "model": "universal-3-5-pro",
                   "fn": assemblyai_stream, "languages": LANG_ASSEMBLYAI},
    "gladia": {"engine": "gladia-solaria-stream", "model": "solaria-1",
               "fn": gladia_stream, "languages": LANG_BARE},
    "cartesia": {"engine": "cartesia-ink-whisper-stream", "model": "ink-whisper",
                 "fn": cartesia_stream, "languages": {"en": "en"}},
    "speechmatics": {"engine": "speechmatics-enhanced-stream", "model": "enhanced",
                     "fn": speechmatics_stream, "languages": LANG_SPEECHMATICS},
}


# --- Smoke test -----------------------------------------------------------------

def _smoke():
    from run_cloud_engines import load_env

    load_env()
    args = sys.argv[1:]
    n = 20
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    provider = args[0] if args else "deepgram"
    if provider not in PROVIDERS_STREAM:
        sys.exit(f"unknown streaming provider {provider!r}; "
                 f"have: {', '.join(PROVIDERS_STREAM)}")
    cfg = {**PROVIDERS_STREAM[provider], "lang": "en"}

    audio_dir = os.path.join(ROOT, "data_librispeech", "audio")
    refs = json.load(open(os.path.join(ROOT, "data_librispeech", "refs.json")))
    if not os.path.isdir(audio_dir):
        sys.exit("prep data_librispeech first (src/prep_librispeech.py)")
    import numpy as np
    uids = list(refs)
    sample = [uids[j] for j in np.linspace(0, len(uids) - 1, min(n, len(uids))).astype(int)]

    print(f"{provider}: {cfg['engine']}  clips={len(sample)}\n")
    lags, ttfps = [], []
    for uid in sample:
        wav = os.path.join(audio_dir, f"{uid}.wav")
        t0 = time.time()
        try:
            res = cfg["fn"](wav, cfg)
        except TranscribeError as e:
            print(f"  {uid}: ERROR {e}")
            continue
        m = compute_metrics(res)
        if m["ttfp_s"] is not None:
            ttfps.append(m["ttfp_s"])
        if m["finalization_lag_s"] is not None:
            lags.append(m["finalization_lag_s"])
        print(f"  {uid}  ttfp={m['ttfp_s']}s ttff={m['ttff_s']}s "
              f"lag={m['finalization_lag_s']}s partials={m['n_partials']} "
              f"revs={m['n_revisions']} wall={m['stream_wall_s']}s ({time.time() - t0:.1f}s)")
        print(f"    ref:  {refs[uid][:90]}")
        print(f"    hyp:  {res.text[:90]}")
    if ttfps:
        print(f"\nmedian ttfp={sorted(ttfps)[len(ttfps) // 2]}s  "
              f"median lag={sorted(lags)[len(lags) // 2] if lags else None}s  "
              f"n={len(ttfps)}")


if __name__ == "__main__":
    _smoke()
