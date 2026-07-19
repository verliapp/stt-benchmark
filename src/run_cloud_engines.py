"""Run the commercial cloud STT APIs on a prepared test set and save raw transcripts.

Same contract as run_ood_engines.py: writes results/reports/<config>/<engine>/<uid>.json
({"text": ...}) plus a per-engine _meta.json, then score_ood.py does WER + CIs. Only
produces transcripts, so scoring/CIs can be re-run without re-hitting the APIs.

    python run_cloud_engines.py librispeech                       # all 12 providers
    python run_cloud_engines.py earnings22 deepgram openai        # a subset
    python run_cloud_engines.py librispeech --limit 50 groq       # smoke test

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

from cloud_adapters import PROVIDERS, TranscribeError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIES = 3


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


def run_provider(key, cfg, refs, audio_dir, reports, total_audio):
    engine = cfg["engine"]
    rdir = os.path.join(reports, engine)
    os.makedirs(rdir, exist_ok=True)

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
    json.dump({"audioSeconds": round(total_audio), "wallSeconds": round(wall),
               "missing": missing},
              open(os.path.join(rdir, "_meta.json"), "w"), indent=2)
    print(f"[{key}] finished: {len(refs) - missing}/{len(refs)} transcribed, "
          f"missing={missing}, wall={round(wall)}s", flush=True)
    if errors:
        print(f"[{key}] first errors: {errors[:3]}", flush=True)


def main():
    load_env()
    if len(sys.argv) < 2:
        sys.exit(f"usage: python run_cloud_engines.py <config> [--limit N] "
                 f"[providers...]\nproviders: {', '.join(PROVIDERS)}")
    cfg_name = sys.argv[1]
    rest = sys.argv[2:]
    limit = None
    if "--limit" in rest:
        i = rest.index("--limit")
        limit = int(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    selected = rest or list(PROVIDERS)
    unknown = [p for p in selected if p not in PROVIDERS]
    if unknown:
        sys.exit(f"unknown providers: {unknown}. known: {', '.join(PROVIDERS)}")

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

    print(f"config={cfg_name}  clips={len(refs)}  audio={total_audio/60:.1f}min  "
          f"providers={selected}", flush=True)
    for key in selected:
        run_provider(key, PROVIDERS[key], refs, audio_dir, reports, total_audio)

    print(f"\nALL DONE. Now: ./.venv/bin/python score_ood.py {cfg_name}", flush=True)


if __name__ == "__main__":
    main()
