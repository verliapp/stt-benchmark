"""Cloud STT adapters for the 12-provider accuracy benchmark.

Each adapter is `transcribe(wav_path, cfg) -> str`: it sends one utterance wav to a
provider's pre-recorded/batch API and returns the plain transcript. Scoring,
normalization, and confidence intervals are the harness's job (score_ood.py), so
adapters only produce raw text, exactly like the on-device engines do.

Auth comes from environment variables (load them from .env.providers via
run_cloud_engines.py). Nothing here prints or logs a key.

Model ids are pinned to the flagship model each provider was credited with in part
one of the blog, and every one is overridable with an env var (e.g. DEEPGRAM_MODEL)
so we can adjust without editing code. A handful are worth confirming on the first
live call because vendors rename them (noted per provider):
  - ELEVENLABS_MODEL: "scribe_v1" default; set "scribe_v2" if your account has it.
  - ASSEMBLYAI_MODEL: "universal" (Universal family); "slam-1" is the other option.
  - AZURE_MODEL:      "" = fast-transcription default; set "mai-transcribe" for MAI.
  - GOOGLE_MODEL:     "chirp_2" (global). "chirp_3" needs a regional GCP_LOCATION.

REST providers use `requests` only. AWS/Azure/Google use their SDKs, imported
lazily so the REST providers never need them installed.
"""
import collections
import io
import json
import os
import threading
import time

import requests

TIMEOUT = 300  # per HTTP call
POLL_EVERY = 1.5  # seconds between status polls for async providers
POLL_MAX = 600  # give up on one clip after this many seconds

# Groq's free on_demand tier caps at 20 requests/min per model. We pace calls
# globally to stay just under that (many workers share one window), and also honor
# a 429's Retry-After inside the adapter so a burst can't blow the whole budget.
GROQ_RPM = 18
_groq_lock = threading.Lock()
_groq_calls = collections.deque()


def _groq_throttle():
    with _groq_lock:
        now = time.time()
        while _groq_calls and now - _groq_calls[0] > 60:
            _groq_calls.popleft()
        if len(_groq_calls) >= GROQ_RPM:
            time.sleep(max(0.0, 60 - (now - _groq_calls[0]) + 0.2))
            now = time.time()
            while _groq_calls and now - _groq_calls[0] > 60:
                _groq_calls.popleft()
        _groq_calls.append(time.time())


class TranscribeError(RuntimeError):
    """Raised when a provider fails a clip; the driver counts it as missing."""


def _env(name, default=None, required=True):
    val = os.environ.get(name, default)
    if required and not val:
        raise TranscribeError(f"missing env var {name}")
    return val


def _poll(get_status, is_done, is_error):
    """Poll get_status() until is_done() or is_error(); returns the final payload."""
    waited = 0.0
    while True:
        payload = get_status()
        if is_error(payload):
            raise TranscribeError(f"provider job failed: {json.dumps(payload)[:300]}")
        if is_done(payload):
            return payload
        time.sleep(POLL_EVERY)
        waited += POLL_EVERY
        if waited > POLL_MAX:
            raise TranscribeError("provider job timed out")


# --- Synchronous REST providers -------------------------------------------------

def deepgram(wav, cfg):
    model = _env("DEEPGRAM_MODEL", cfg["model"], required=False)
    key = _env("DEEPGRAM_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": model, "punctuate": "true", "smart_format": "false"},
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            data=fh.read(), timeout=TIMEOUT,
        )
    r.raise_for_status()
    alts = r.json()["results"]["channels"][0]["alternatives"]
    return alts[0]["transcript"] if alts else ""


def openai(wav, cfg):
    model = _env("OPENAI_MODEL", cfg["model"], required=False)
    key = _env("OPENAI_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (os.path.basename(wav), fh, "audio/wav")},
            data={"model": model, "response_format": "json"}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def groq(wav, cfg):
    model = _env("GROQ_MODEL", cfg["model"], required=False)
    key = _env("GROQ_API_KEY")
    for _ in range(6):
        _groq_throttle()
        with open(wav, "rb") as fh:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (os.path.basename(wav), fh, "audio/wav")},
                data={"model": model, "response_format": "json"}, timeout=TIMEOUT,
            )
        if r.status_code == 429:
            time.sleep(float(r.headers.get("retry-after", 5)) + 0.5)
            continue
        r.raise_for_status()
        return r.json().get("text", "")
    raise TranscribeError("groq: still rate-limited after retries")


def elevenlabs(wav, cfg):
    model = _env("ELEVENLABS_MODEL", cfg["model"], required=False)
    key = _env("ELEVENLABS_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": key},
            files={"file": (os.path.basename(wav), fh, "audio/wav")},
            data={"model_id": model, "language_code": "en",
                  "tag_audio_events": "false", "diarize": "false"}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def gemini(wav, cfg):
    # LLM-based transcription (not a dedicated STT engine). We prompt for a verbatim
    # transcript and disable "thinking" so it does not spend reasoning tokens. LLMs
    # tend to clean up disfluencies, so this is labeled as LLM transcription in the
    # results, not compared like-for-like with the dedicated STT APIs.
    import base64
    model = _env("GEMINI_MODEL", cfg["model"], required=False)
    key = _env("GEMINI_API_KEY")
    with open(wav, "rb") as fh:
        audio_b64 = base64.b64encode(fh.read()).decode()
    prompt = ("Transcribe the spoken audio verbatim. Output only the exact words "
              "spoken, in order, with nothing added. Do not translate, summarize, "
              "correct grammar, or add commentary. If nothing is said, output nothing.")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [{"text": prompt},
                                      {"inline_data": {"mime_type": "audio/wav",
                                                       "data": audio_b64}}]}],
              "generationConfig": {"temperature": 0,
                                   "thinkingConfig": {"thinkingBudget": 0}}},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    cands = r.json().get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    return " ".join(p.get("text", "") for p in parts).strip()


def lemonfox(wav, cfg):
    # OpenAI-compatible, single Whisper-based model (no model param). Language is
    # given as a full name ("english"), not an ISO code, per Lemonfox's API.
    key = _env("LEMONFOX_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.lemonfox.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (os.path.basename(wav), fh, "audio/wav")},
            data={"language": "english", "response_format": "json"}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def grok(wav, cfg):
    # xAI Grok Speech-to-Text (dedicated STT API, model grok-stt). format defaults
    # off so we get verbatim words, not inverse-text-normalized ("$100") output.
    # The file field must be last in the multipart form; requests orders data
    # fields before files, so passing language via data satisfies that.
    key = _env("XAI_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.x.ai/v1/stt",
            headers={"Authorization": f"Bearer {key}"},
            data={"language": "en"},
            files={"file": (os.path.basename(wav), fh, "audio/wav")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def fish(wav, cfg):
    # Fish Audio ASR (dedicated STT, single model transcribe-1; the /v1/asr endpoint
    # takes no model param). multipart/form-data. We skip timestamps since we only
    # score transcript text, and that also lowers latency on short clips.
    key = _env("FISH_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.fish.audio/v1/asr",
            headers={"Authorization": f"Bearer {key}"},
            files={"audio": (os.path.basename(wav), fh, "audio/wav")},
            data={"language": "en", "ignore_timestamps": "true"}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def cartesia(wav, cfg):
    # Cartesia Ink STT. Batch endpoint only serves the ink-whisper family.
    # multipart/form-data; requires the dated Cartesia-Version header.
    model = _env("CARTESIA_MODEL", cfg["model"], required=False)
    key = _env("CARTESIA_API_KEY")
    with open(wav, "rb") as fh:
        r = requests.post(
            "https://api.cartesia.ai/stt",
            headers={"Authorization": f"Bearer {key}",
                     "Cartesia-Version": "2026-03-01"},
            files={"file": (os.path.basename(wav), fh, "audio/wav")},
            data={"model": model, "language": "en"}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json().get("text", "")


def inworld(wav, cfg):
    # Inworld Realtime STT-1 (sync path). Audio is base64 in the JSON body; the
    # API key is used directly as the HTTP Basic credential. AUTO_DETECT lets the
    # WAV header carry the sample rate.
    import base64
    model = _env("INWORLD_MODEL", cfg["model"], required=False)
    key = _env("INWORLD_API_KEY")
    with open(wav, "rb") as fh:
        audio_b64 = base64.b64encode(fh.read()).decode()
    r = requests.post(
        "https://api.inworld.ai/stt/v1/transcribe",
        headers={"Authorization": f"Basic {key}", "Content-Type": "application/json"},
        json={"transcribeConfig": {"modelId": model, "language": "en",
                                   "audioEncoding": "AUTO_DETECT"},
              "audioData": {"content": audio_b64}}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("transcription", {}).get("transcript", "") or ""


# --- Upload + poll REST providers -----------------------------------------------

def assemblyai(wav, cfg):
    model = _env("ASSEMBLYAI_MODEL", cfg["model"], required=False)
    key = _env("ASSEMBLYAI_API_KEY")
    base, headers = "https://api.assemblyai.com/v2", {"authorization": key}
    with open(wav, "rb") as fh:
        up = requests.post(f"{base}/upload", headers=headers, data=fh.read(), timeout=TIMEOUT)
    up.raise_for_status()
    audio_url = up.json()["upload_url"]
    sub = requests.post(f"{base}/transcript", headers=headers,
                        json={"audio_url": audio_url, "speech_models": [model],
                              "language_code": "en", "punctuate": True}, timeout=TIMEOUT)
    sub.raise_for_status()
    tid = sub.json()["id"]

    def status():
        g = requests.get(f"{base}/transcript/{tid}", headers=headers, timeout=TIMEOUT)
        g.raise_for_status()
        return g.json()

    done = _poll(status, lambda p: p["status"] == "completed",
                 lambda p: p["status"] == "error")
    return done.get("text") or ""


def gladia(wav, cfg):
    key = _env("GLADIA_API_KEY")
    base, headers = "https://api.gladia.io/v2", {"x-gladia-key": key}
    with open(wav, "rb") as fh:
        up = requests.post(f"{base}/upload", headers=headers,
                           files={"audio": (os.path.basename(wav), fh, "audio/wav")},
                           timeout=TIMEOUT)
    up.raise_for_status()
    audio_url = up.json()["audio_url"]
    sub = requests.post(f"{base}/pre-recorded", headers=headers,
                        json={"audio_url": audio_url, "language": "en",
                              "diarization": False}, timeout=TIMEOUT)
    sub.raise_for_status()
    result_url = sub.json()["result_url"]

    def status():
        g = requests.get(result_url, headers=headers, timeout=TIMEOUT)
        g.raise_for_status()
        return g.json()

    done = _poll(status, lambda p: p["status"] == "done",
                 lambda p: p["status"] == "error")
    return done["result"]["transcription"]["full_transcript"]


def soniox(wav, cfg):
    model = _env("SONIOX_MODEL", cfg["model"], required=False)
    key = _env("SONIOX_API_KEY")
    base, headers = "https://api.soniox.com/v1", {"Authorization": f"Bearer {key}"}
    file_id = tid = None
    try:
        with open(wav, "rb") as fh:
            up = requests.post(f"{base}/files", headers=headers,
                               files={"file": (os.path.basename(wav), fh, "audio/wav")},
                               timeout=TIMEOUT)
        up.raise_for_status()
        file_id = up.json()["id"]
        sub = requests.post(f"{base}/transcriptions", headers=headers,
                            json={"file_id": file_id, "model": model,
                                  "language_hints": ["en"]}, timeout=TIMEOUT)
        sub.raise_for_status()
        tid = sub.json()["id"]

        def status():
            g = requests.get(f"{base}/transcriptions/{tid}", headers=headers, timeout=TIMEOUT)
            g.raise_for_status()
            return g.json()

        _poll(status, lambda p: p["status"] == "completed",
              lambda p: p["status"] == "error")
        tr = requests.get(f"{base}/transcriptions/{tid}/transcript", headers=headers, timeout=TIMEOUT)
        tr.raise_for_status()
        return tr.json().get("text", "")
    finally:
        # Soniox caps stored files at 1000 and transcriptions at 2000. Delete both
        # after each clip so a full run does not fill the account and start
        # refusing uploads (which surfaces as 429s and dropped TLS connections).
        # Deletes share the same rate-limited file API, so retry on 429; a delete
        # that silently gave up would leak a file and slowly refill the cap.
        for url in ((f"{base}/transcriptions/{tid}" if tid else None),
                    (f"{base}/files/{file_id}" if file_id else None)):
            if not url:
                continue
            for _ in range(5):
                try:
                    d = requests.delete(url, headers=headers, timeout=30)
                    if d.status_code == 429:
                        time.sleep(float(d.headers.get("retry-after", 2)) + 0.5)
                        continue
                    break
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    time.sleep(1.0)


def speechmatics(wav, cfg):
    op = _env("SPEECHMATICS_OPERATING_POINT", cfg["model"], required=False)
    key = _env("SPEECHMATICS_API_KEY")
    base, headers = "https://asr.api.speechmatics.com/v2", {"Authorization": f"Bearer {key}"}
    config = {"type": "transcription",
              "transcription_config": {"language": "en", "operating_point": op}}
    with open(wav, "rb") as fh:
        sub = requests.post(f"{base}/jobs", headers=headers,
                            data={"config": json.dumps(config)},
                            files={"data_file": (os.path.basename(wav), fh, "audio/wav")},
                            timeout=TIMEOUT)
    sub.raise_for_status()
    jid = sub.json()["id"]

    def status():
        g = requests.get(f"{base}/jobs/{jid}", headers=headers, timeout=TIMEOUT)
        g.raise_for_status()
        return g.json()["job"]

    _poll(status, lambda p: p["status"] == "done",
          lambda p: p["status"] in ("rejected", "deleted"))
    tr = requests.get(f"{base}/jobs/{jid}/transcript", headers=headers,
                      params={"format": "txt"}, timeout=TIMEOUT)
    tr.raise_for_status()
    return tr.text.strip()


def rev(wav, cfg):
    model = _env("REV_MODEL", cfg["model"], required=False)
    key = _env("REV_API_KEY")
    base, headers = "https://api.rev.ai/speechtotext/v1", {"Authorization": f"Bearer {key}"}
    options = {"transcriber": model, "language": "en"}
    with open(wav, "rb") as fh:
        sub = requests.post(f"{base}/jobs", headers=headers,
                            data={"options": json.dumps(options)},
                            files={"media": (os.path.basename(wav), fh, "audio/wav")},
                            timeout=TIMEOUT)
    sub.raise_for_status()
    jid = sub.json()["id"]

    def status():
        g = requests.get(f"{base}/jobs/{jid}", headers=headers, timeout=TIMEOUT)
        g.raise_for_status()
        return g.json()

    _poll(status, lambda p: p["status"] == "transcribed",
          lambda p: p["status"] == "failed")
    tr = requests.get(f"{base}/jobs/{jid}/transcript",
                      headers={**headers,
                               "Accept": "application/vnd.rev.transcript.v1.0+json"},
                      timeout=TIMEOUT)
    tr.raise_for_status()
    # Join element values across monologues. The text/plain format prepends
    # "Speaker 0   00:00:00   " to each turn, which would be scored as errors;
    # the JSON elements carry only the words and punctuation.
    parts = []
    for mono in tr.json().get("monologues", []):
        for el in mono.get("elements", []):
            parts.append(el.get("value", ""))
    return "".join(parts).strip()


def resemble(wav, cfg):
    # Resemble STT (async job): create with a file upload, then poll by uuid.
    key = _env("RESEMBLE_API_KEY")
    base, headers = "https://app.resemble.ai/api/v2", {"Authorization": f"Bearer {key}"}
    with open(wav, "rb") as fh:
        sub = requests.post(f"{base}/speech-to-text", headers=headers,
                            files={"file": (os.path.basename(wav), fh, "audio/wav")},
                            timeout=TIMEOUT)
    sub.raise_for_status()
    uuid = sub.json()["item"]["uuid"]

    def status():
        g = requests.get(f"{base}/speech-to-text/{uuid}", headers=headers, timeout=TIMEOUT)
        g.raise_for_status()
        return g.json()["item"]

    done = _poll(status, lambda p: p.get("status") == "completed",
                 lambda p: p.get("status") == "failed")
    return done.get("text") or ""


# --- Cloud giants (lazy SDK imports) --------------------------------------------

def azure(wav, cfg):
    region = _env("AZURE_SPEECH_REGION")
    key = _env("AZURE_SPEECH_KEY")
    model = _env("AZURE_MODEL", "", required=False)
    url = (f"https://{region}.api.cognitive.microsoft.com"
           f"/speechtotext/transcriptions:transcribe?api-version=2024-11-15")
    definition = {"locales": ["en-US"]}
    if model:
        definition["model"] = model
    with open(wav, "rb") as fh:
        r = requests.post(
            url, headers={"Ocp-Apim-Subscription-Key": key},
            files={"audio": (os.path.basename(wav), fh, "audio/wav")},
            data={"definition": json.dumps(definition)}, timeout=TIMEOUT,
        )
    r.raise_for_status()
    phrases = r.json().get("combinedPhrases", [])
    return " ".join(p.get("text", "") for p in phrases).strip()


def google(wav, cfg):
    from google.cloud import speech_v2
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions

    project = _env("GCP_PROJECT")
    location = _env("GCP_LOCATION", "global", required=False)
    model = _env("GOOGLE_MODEL", cfg["model"], required=False)
    opts = None
    if location != "global":
        opts = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    client = speech_v2.SpeechClient(client_options=opts)
    with open(wav, "rb") as fh:
        content = fh.read()
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"], model=model,
    )
    req = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project}/locations/{location}/recognizers/_",
        config=config, content=content,
    )
    resp = client.recognize(request=req)
    return " ".join(r.alternatives[0].transcript
                    for r in resp.results if r.alternatives).strip()


def amazon(wav, cfg):
    """Amazon Transcribe via the streaming API, one clip at a time.

    Transcribe has no synchronous pre-recorded REST endpoint; batch jobs read from
    S3 and are one-job-per-file (thousands of jobs). Feeding each short clip through
    the streaming API is the practical per-utterance path and returns the same
    engine output. Needs `amazon-transcribe` and `soundfile`; creds from the env.
    """
    import asyncio

    import soundfile as sf
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent

    region = _env("AWS_REGION")
    data, sr = sf.read(wav, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    pcm = data.tobytes()

    finals = []

    class Handler(TranscriptResultStreamHandler):
        async def handle_transcript_event(self, event: TranscriptEvent):
            for res in event.transcript.results:
                if not res.is_partial and res.alternatives:
                    finals.append(res.alternatives[0].transcript)

    async def run():
        client = TranscribeStreamingClient(region=region)
        stream = await client.start_stream_transcription(
            language_code="en-US", media_sample_rate_hz=sr,
            media_encoding="pcm",
        )
        chunk = 16 * 1024
        async def feed():
            for i in range(0, len(pcm), chunk):
                await stream.input_stream.send_audio_event(audio_chunk=pcm[i:i + chunk])
            await stream.input_stream.end_stream()
        handler = Handler(stream.output_stream)
        await asyncio.gather(feed(), handler.handle_events())

    asyncio.run(run())
    return " ".join(finals).strip()


# --- Registry -------------------------------------------------------------------
# engine = output dir label (results/reports/<config>/<engine>/); model = default
# model id (env-overridable); workers = safe concurrency for that provider's limits.

PROVIDERS = {
    "deepgram":     {"engine": "deepgram-nova-3",           "model": "nova-3",                 "fn": deepgram,     "workers": 12},
    "assemblyai":   {"engine": "assemblyai-universal",      "model": "universal-3-5-pro",      "fn": assemblyai,   "workers": 12},
    "soniox":       {"engine": "soniox",                    "model": "stt-async-preview",      "fn": soniox,       "workers": 2},
    "speechmatics": {"engine": "speechmatics-enhanced",     "model": "enhanced",               "fn": speechmatics, "workers": 6},
    "gladia":       {"engine": "gladia-solaria",            "model": "solaria-1",              "fn": gladia,       "workers": 10},
    "elevenlabs":   {"engine": "elevenlabs-scribe",         "model": "scribe_v1",              "fn": elevenlabs,   "workers": 8},
    "openai":       {"engine": "openai-gpt-4o-transcribe",  "model": "gpt-4o-transcribe",      "fn": openai,       "workers": 8},
    "groq":         {"engine": "groq-whisper-v3-turbo",     "model": "whisper-large-v3-turbo", "fn": groq,         "workers": 2},
    "lemonfox":     {"engine": "lemonfox-whisper",          "model": "whisper-large-v3",       "fn": lemonfox,     "workers": 8},
    "gemini":       {"engine": "gemini-2.5-flash",          "model": "gemini-2.5-flash",       "fn": gemini,       "workers": 6},
    "grok":         {"engine": "grok-stt",                  "model": "grok-stt",               "fn": grok,         "workers": 8},
    "fish":         {"engine": "fish-audio-asr",            "model": "transcribe-1",           "fn": fish,         "workers": 5},
    "cartesia":     {"engine": "cartesia-ink-whisper",      "model": "ink-whisper",            "fn": cartesia,     "workers": 6},
    "inworld":      {"engine": "inworld-stt-1",             "model": "inworld/inworld-stt-1",  "fn": inworld,      "workers": 6},
    "resemble":     {"engine": "resemble",                  "model": "",                       "fn": resemble,     "workers": 4},
    "rev":          {"engine": "rev",                       "model": "reverb",                 "fn": rev,          "workers": 6},
    "amazon":       {"engine": "amazon-transcribe",         "model": "",                       "fn": amazon,       "workers": 4},
    "azure":        {"engine": "azure",                     "model": "",                       "fn": azure,        "workers": 2},
    "google":       {"engine": "google-chirp",              "model": "chirp_2",                "fn": google,       "workers": 6},
}
