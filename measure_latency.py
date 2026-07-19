"""Measure per-service latency in isolation (not under the bulk-run concurrency).

For each provider, sends a fixed even sample of clips ONE AT A TIME (no concurrency)
and records end-to-end wall time per call: submit -> transcript in hand. That is the
honest per-request latency; the bulk accuracy runs use many workers, which distorts
per-call timing, so latency is measured separately here.

    python measure_latency.py                      # all providers with a key present, 40 clips
    python measure_latency.py --n 25 deepgram openai grok
    python measure_latency.py --config earnings22 --n 40

Writes latency.json: per provider, median / mean / p90 / min / max seconds, sample
size, and sync-vs-async label (async = submit + poll, so its latency includes job
queue turnaround and our poll cadence, and is not comparable to a sync request).
"""
import json
import os
import sys
import time

from run_cloud_engines import load_env
from cloud_adapters import PROVIDERS

ROOT = os.path.dirname(os.path.abspath(__file__))
# providers that submit a job then poll for the result (latency includes queue + poll)
ASYNC = {"assemblyai", "soniox", "speechmatics", "gladia", "rev", "resemble"}


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def has_key(key):
    # a provider is runnable if its required env var(s) are set; cheap check via a
    # dry call would cost money, so we just look for the obvious key env var.
    env_by_provider = {
        "deepgram": "DEEPGRAM_API_KEY", "assemblyai": "ASSEMBLYAI_API_KEY",
        "soniox": "SONIOX_API_KEY", "speechmatics": "SPEECHMATICS_API_KEY",
        "gladia": "GLADIA_API_KEY", "elevenlabs": "ELEVENLABS_API_KEY",
        "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY",
        "lemonfox": "LEMONFOX_API_KEY", "gemini": "GEMINI_API_KEY",
        "grok": "XAI_API_KEY", "fish": "FISH_API_KEY", "rev": "REV_API_KEY",
        "cartesia": "CARTESIA_API_KEY", "inworld": "INWORLD_API_KEY",
        "resemble": "RESEMBLE_API_KEY",
        "azure": "AZURE_SPEECH_KEY", "google": "GCP_PROJECT",
        "amazon": "AWS_ACCESS_KEY_ID",
    }
    return bool(os.environ.get(env_by_provider.get(key, ""), ""))


def main():
    load_env()
    args = sys.argv[1:]
    n = 40
    cfg = "librispeech"
    if "--n" in args:
        i = args.index("--n"); n = int(args[i + 1]); args = args[:i] + args[i + 2:]
    if "--config" in args:
        i = args.index("--config"); cfg = args[i + 1]; args = args[:i] + args[i + 2:]
    selected = args or [k for k in PROVIDERS if has_key(k)]

    audio_dir = os.path.join(ROOT, f"data_{cfg}", "audio")
    refs = json.load(open(os.path.join(ROOT, f"data_{cfg}", "refs.json")))
    uids = list(refs)
    import numpy as np
    sample = [uids[j] for j in np.linspace(0, len(uids) - 1, min(n, len(uids))).astype(int)]

    out = {}
    existing = {}
    lp = os.path.join(ROOT, "latency.json")
    if os.path.exists(lp):
        existing = json.load(open(lp))

    for key in selected:
        cfgp = PROVIDERS[key]
        times = []
        errs = 0
        for uid in sample:
            wav = os.path.join(audio_dir, f"{uid}.wav")
            t0 = time.time()
            try:
                cfgp["fn"](wav, cfgp)
                times.append(time.time() - t0)
            except Exception:  # noqa: BLE001
                errs += 1
        if times:
            out[cfgp["engine"]] = {
                "provider": key, "mode": "async" if key in ASYNC else "sync",
                "n": len(times), "errors": errs,
                "median_s": pct(times, 50), "mean_s": round(sum(times) / len(times), 3),
                "p90_s": pct(times, 90), "min_s": round(min(times), 3),
                "max_s": round(max(times), 3), "config": cfg,
            }
            print(f"{cfgp['engine']:26} median={out[cfgp['engine']]['median_s']}s "
                  f"p90={out[cfgp['engine']]['p90_s']}s ({out[cfgp['engine']]['mode']}, "
                  f"n={len(times)}, errors={errs})", flush=True)
        else:
            print(f"{cfgp['engine']:26} no successful calls (errors={errs})", flush=True)

    existing.update(out)
    json.dump(existing, open(lp, "w"), indent=2)
    print(f"\nwrote {lp}")


if __name__ == "__main__":
    main()
