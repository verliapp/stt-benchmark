"""Run the streaming (real-time) STT adapters on a prepared test set.

Same idea as run_cloud_engines.py, two differences:

  1. clips run ONE AT A TIME per engine (no worker pool). Real-time pacing means each
     clip already takes ~its own duration in wall time; parallelism would distort the
     time-to-first-partial and finalization-lag numbers this run exists to capture,
     the same reason measure_latency.py runs serially.
  2. each clip writes two files: the finalized transcript to
     results/reports/<config>/<engine>/<uid>.json ({"text": ...}, the exact shape
     score_ood.py scores, so streaming and batch are scored identically), and the
     per-clip latency to results/streaming/<config>/<engine>/<uid>.json.

Engine labels carry a `-stream` suffix (deepgram-nova-3-stream), so streaming sits
next to the batch engine (deepgram-nova-3) under the same config and score_ood.py
scores both in one pass.

    python src/run_streaming_engines.py librispeech
    python src/run_streaming_engines.py librispeech --limit 100 deepgram
    python src/run_streaming_engines.py fleurs_ja_jp --lang ja deepgram

Resumable: a clip whose transcript json already exists and is non-empty is skipped.
Then: python src/score_ood.py <config>
"""
import json
import os
import sys
import time

from cloud_adapters import supports_language
from run_cloud_engines import choose_language, load_env
from streaming_adapters import PROVIDERS_STREAM, compute_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIES = 3


def transcribe_one(cfg, wav):
    """Run one clip; return (StreamResult, error_str). Retries transient failures."""
    last = None
    for attempt in range(RETRIES):
        try:
            return cfg["fn"](wav, cfg), None
        except Exception as e:  # noqa: BLE001 - record and let the driver count it
            last = e
            time.sleep(1.5 * (attempt + 1))
    return None, f"{type(last).__name__}: {str(last)[:200]}"


def run_provider(key, cfg, refs, audio_dir, reports, timing):
    engine = cfg["engine"]
    rdir = os.path.join(reports, engine)
    tdir = os.path.join(timing, engine)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)

    todo = [uid for uid in refs
            if not (os.path.exists(os.path.join(rdir, f"{uid}.json"))
                    and os.path.getsize(os.path.join(rdir, f"{uid}.json")) > 0)]
    done = len(refs) - len(todo)
    if not todo:
        print(f"[{key}] all {len(refs)} clips already done", flush=True)

    errors = []
    t0 = time.time()
    for uid in todo:
        wav = os.path.join(audio_dir, f"{uid}.wav")
        res, err = transcribe_one(cfg, wav)
        done += 1
        if err:
            errors.append((uid, err))
        else:
            with open(os.path.join(rdir, f"{uid}.json"), "w") as fh:
                json.dump({"text": res.text}, fh)
            with open(os.path.join(tdir, f"{uid}.json"), "w") as fh:
                json.dump(compute_metrics(res), fh)
        if done % 25 == 0 or done == len(refs):
            print(f"[{key}] {done}/{len(refs)}  errors={len(errors)}", flush=True)

    wall = time.time() - t0
    missing = sum(1 for uid in refs
                  if not os.path.exists(os.path.join(rdir, f"{uid}.json"))
                  or os.path.getsize(os.path.join(rdir, f"{uid}.json")) == 0)
    with open(os.path.join(rdir, "_meta.json"), "w") as fh:
        json.dump({"wallSeconds": round(wall), "missing": missing,
                   "language": cfg["lang"], "streaming": True}, fh, indent=2)
    print(f"[{key}] finished: {len(refs) - missing}/{len(refs)} transcribed, "
          f"missing={missing}, wall={round(wall)}s", flush=True)
    if errors:
        print(f"[{key}] first errors: {errors[:3]}", flush=True)


def main():
    load_env()
    if len(sys.argv) < 2:
        sys.exit(f"usage: python run_streaming_engines.py <config> [--limit N] "
                 f"[--lang CODE] [providers...]\nproviders: {', '.join(PROVIDERS_STREAM)}")
    cfg_name = sys.argv[1]
    rest = sys.argv[2:]
    limit = None
    requested_lang = None
    selected = []
    i = 0
    while i < len(rest):
        if rest[i] == "--limit":
            limit = int(rest[i + 1]); i += 2
        elif rest[i] == "--lang":
            requested_lang = rest[i + 1]; i += 2
        elif rest[i].startswith("--"):
            sys.exit(f"unknown option: {rest[i]}")
        else:
            selected.append(rest[i]); i += 1
    selected = selected or list(PROVIDERS_STREAM)
    unknown = [p for p in selected if p not in PROVIDERS_STREAM]
    if unknown:
        sys.exit(f"unknown providers: {unknown}. known: {', '.join(PROVIDERS_STREAM)}")
    try:
        lang = choose_language(cfg_name, requested_lang)
    except ValueError as e:
        sys.exit(str(e))

    runnable = []
    for key in selected:
        pcfg = PROVIDERS_STREAM[key]
        if not supports_language(pcfg, lang):
            print(f"[{key}] skip: {pcfg['engine']} does not support language={lang}",
                  flush=True)
            continue
        runnable.append((key, {**pcfg, "lang": lang}))
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
    reports = os.path.join(ROOT, "results", "reports", cfg_name)
    timing = os.path.join(ROOT, "results", "streaming", cfg_name)
    os.makedirs(reports, exist_ok=True)
    os.makedirs(timing, exist_ok=True)

    print(f"config={cfg_name}  clips={len(refs)}  language={lang}  "
          f"providers={[k for k, _c in runnable]}  (serial, real-time paced)", flush=True)
    for key, pcfg in runnable:
        run_provider(key, pcfg, refs, audio_dir, reports, timing)

    print(f"\nALL DONE. Now: ./.venv/bin/python src/score_ood.py {cfg_name}", flush=True)


if __name__ == "__main__":
    main()
