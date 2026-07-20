"""Run the commercial cloud STT APIs on a prepared test set and save raw transcripts.

Same contract as run_ood_engines.py: writes results/reports/<config>/<engine>/<uid>.json
({"text": ...}) plus a per-engine _meta.json, then score_ood.py does WER + CIs. Only
produces transcripts, so scoring/CIs can be re-run without re-hitting the APIs.

    python run_cloud_engines.py librispeech                       # all 19 providers
    python run_cloud_engines.py earnings22 deepgram openai        # a subset
    python run_cloud_engines.py librispeech --limit 50 groq       # smoke test
    python run_cloud_engines.py fleurs_ja_jp --lang ja deepgram

Reads keys from .env.providers in this directory (KEY=VALUE lines) if present, else
from the environment. Resumable: a clip whose <uid>.json already exists and is
non-empty is skipped, so re-running only fills gaps and retries failures.
"""
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

from cloud_adapters import PROVIDERS, TranscribeError, normalize_language, supports_language

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIES = 3
FLEURS_LANGUAGES = {
    "es_419": "es",
    "fr_fr": "fr",
    "de_de": "de",
    "cmn_hans_cn": "zh",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "ar_eg": "ar",
    "hi_in": "hi",
    "ru_ru": "ru",
    "vi_vn": "vi",
    "th_th": "th",
}


def load_env():
    path = os.path.join(ROOT, ".env.providers")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def audio_seconds(wavs):
    import soundfile as sf
    return sum(sf.info(p).frames / sf.info(p).samplerate for p in wavs)


def transcribe_one(fn, cfg, wav, out_path):
    last = None
    for attempt in range(RETRIES):
        try:
            text = fn(wav, cfg)
            with open(out_path, "w") as fh:
                json.dump({"text": text}, fh)
            return True, None
        except Exception as e:  # noqa: BLE001 - record and let the driver count it
            last = e
            time.sleep(1.5 * (attempt + 1))
    return False, f"{type(last).__name__}: {str(last)[:200]}"


def choose_language(cfg_name, requested):
    if requested is not None:
        return normalize_language(requested)
    if not cfg_name.startswith("fleurs_"):
        return "en"
    suffix = cfg_name.removeprefix("fleurs_")
    try:
        return FLEURS_LANGUAGES[suffix]
    except KeyError as e:
        raise ValueError(
            f"cannot infer language for {cfg_name!r}; pass --lang CODE"
        ) from e


def validate_resume_language(cfg, reports):
    rdir = os.path.join(reports, cfg["engine"])
    existing = []
    if os.path.isdir(rdir):
        existing = [name for name in os.listdir(rdir)
                    if name.endswith(".json") and name != "_meta.json"
                    and os.path.getsize(os.path.join(rdir, name)) > 0]
    if not existing:
        return

    meta_path = os.path.join(rdir, "_meta.json")
    recorded = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                recorded = json.load(fh).get("language")
        except (OSError, ValueError, AttributeError):
            recorded = None
    if recorded != cfg["lang"]:
        shown = repr(recorded) if recorded is not None else "missing or unreadable"
        raise RuntimeError(
            f"refusing to reuse {len(existing)} existing result(s) for "
            f"{cfg['engine']}: _meta.json language is {shown}, but this run uses "
            f"{cfg['lang']!r}. Use a matching language or separate the old results."
        )


def run_provider(key, cfg, refs, audio_dir, reports, total_audio):
    engine = cfg["engine"]
    rdir = os.path.join(reports, engine)
    os.makedirs(rdir, exist_ok=True)
    meta_path = os.path.join(rdir, "_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    meta["language"] = cfg["lang"]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    todo = []
    for uid in refs:
        out = os.path.join(rdir, f"{uid}.json")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        todo.append((uid, out))
    if not todo:
        print(f"[{key}] all {len(refs)} clips already done", flush=True)

    done = len(refs) - len(todo)
    errors = []
    lock = threading.Lock()
    t0 = time.time()

    def work(item):
        uid, out = item
        wav = os.path.join(audio_dir, f"{uid}.wav")
        ok, err = transcribe_one(cfg["fn"], cfg, wav, out)
        with lock:
            nonlocal done
            done += 1
            if not ok:
                errors.append((uid, err))
            if done % 100 == 0 or done == len(refs):
                print(f"[{key}] {done}/{len(refs)}  errors={len(errors)}", flush=True)
        return ok

    if todo:
        with cf.ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            list(ex.map(work, todo))

    wall = time.time() - t0
    missing = sum(1 for uid in refs
                  if not os.path.exists(os.path.join(rdir, f"{uid}.json"))
                  or os.path.getsize(os.path.join(rdir, f"{uid}.json")) == 0)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"audioSeconds": round(total_audio), "wallSeconds": round(wall),
                   "missing": missing, "language": cfg["lang"]}, fh, indent=2)
    print(f"[{key}] finished: {len(refs) - missing}/{len(refs)} transcribed, "
          f"missing={missing}, wall={round(wall)}s", flush=True)
    if errors:
        print(f"[{key}] first errors: {errors[:3]}", flush=True)


def main():
    load_env()
    if len(sys.argv) < 2:
        sys.exit(f"usage: python run_cloud_engines.py <config> [--limit N] [--lang CODE] "
                 f"[providers...]\nproviders: {', '.join(PROVIDERS)}")
    cfg_name = sys.argv[1]
    rest = sys.argv[2:]
    limit = None
    requested_lang = None
    selected = []
    i = 0
    while i < len(rest):
        if rest[i] == "--limit":
            if i + 1 == len(rest):
                sys.exit("--limit requires a value")
            limit = int(rest[i + 1])
            i += 2
        elif rest[i] == "--lang":
            if i + 1 == len(rest):
                sys.exit("--lang requires a value")
            requested_lang = rest[i + 1]
            i += 2
        elif rest[i].startswith("--"):
            sys.exit(f"unknown option: {rest[i]}")
        else:
            selected.append(rest[i])
            i += 1
    selected = selected or list(PROVIDERS)
    unknown = [p for p in selected if p not in PROVIDERS]
    if unknown:
        sys.exit(f"unknown providers: {unknown}. known: {', '.join(PROVIDERS)}")
    try:
        lang = choose_language(cfg_name, requested_lang)
    except ValueError as e:
        sys.exit(str(e))

    runnable = []
    for key in selected:
        provider_cfg = PROVIDERS[key]
        if not supports_language(provider_cfg, lang):
            print(f"[{key}] skip: engine={provider_cfg['engine']} does not support "
                  f"language={lang}", flush=True)
            continue
        runnable.append((key, {**provider_cfg, "lang": lang}))
    if not runnable:
        print(f"no selected providers support language={lang}; nothing to run", flush=True)
        return

    audio_dir = os.path.join(ROOT, f"data_{cfg_name}", "audio")
    refs_path = os.path.join(ROOT, f"data_{cfg_name}", "refs.json")
    assert os.path.isdir(audio_dir), f"prep data_{cfg_name} first"
    refs = json.load(open(refs_path))
    uids = list(refs)
    if limit:
        import numpy as np
        keep = {uids[j] for j in np.linspace(0, len(uids) - 1, limit).astype(int)}
        refs = {u: refs[u] for u in uids if u in keep}
    wavs = [os.path.join(audio_dir, f"{u}.wav") for u in refs]
    total_audio = audio_seconds(wavs)
    reports = os.path.join(ROOT, "results", "reports", cfg_name)
    os.makedirs(reports, exist_ok=True)

    try:
        for _key, provider_cfg in runnable:
            validate_resume_language(provider_cfg, reports)
    except RuntimeError as e:
        sys.exit(str(e))

    print(f"config={cfg_name}  clips={len(refs)}  audio={total_audio/60:.1f}min  "
          f"language={lang}  providers={[key for key, _cfg in runnable]}", flush=True)
    for key, provider_cfg in runnable:
        run_provider(key, provider_cfg, refs, audio_dir, reports, total_audio)

    print(f"\nALL DONE. Now: ./.venv/bin/python src/score_ood.py {cfg_name}", flush=True)


if __name__ == "__main__":
    main()
