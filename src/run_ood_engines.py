"""Run every engine on a prepared out-of-domain test set and save raw transcripts.

This only produces transcripts (the raw artifact). Scoring, confidence intervals,
and the summary table are done separately by score_ood.py, so you can re-score or
add confidence intervals without re-running the engines.

Expects data_<config>/audio/*.wav + data_<config>/refs.json (see prep_ood_dataset.py),
the WhisperKit CLI built under WhisperKit/, and the SpeechAnalyzer harness built under
SpeechAnalyzerCLI/. Parakeet runs via parakeet-mlx.

    python run_ood_engines.py earnings22
    python run_ood_engines.py earnings22 apple whisper-large-v3-v20240930   # subset

Writes results/reports/<config>/<engine>/<uid>.json (each {"text": ...}) plus a
_meta.json per engine with wall time and missing count. Then run score_ood.py.
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WHISPER_MODELS = ["whisper-small", "whisper-large-v3-v20240930"]  # small + turbo
PARAKEET = [("parakeet-v2", "mlx-community/parakeet-tdt-0.6b-v2"),
            ("parakeet-v3", "mlx-community/parakeet-tdt-0.6b-v3")]


def audio_seconds(wavs) -> float:
    import soundfile as sf
    total = 0.0
    for p in wavs:
        info = sf.info(p)
        total += info.frames / info.samplerate
    return total


def main():
    cfg = sys.argv[1]
    only = set(sys.argv[2:])  # optional engine allowlist
    audio = f"{ROOT}/data_{cfg}"
    assert os.path.isdir(f"{audio}/audio"), f"run prep_ood_dataset.py {cfg} first"
    refs = json.load(open(f"{audio}/refs.json"))
    wavs = [f"{audio}/audio/{uid}.wav" for uid in refs]
    total_audio = audio_seconds(wavs)
    reports = f"{ROOT}/results/reports/{cfg}"

    def want(name):
        return not only or name in only

    def meta(engine, compute_s, wall_s, missing):
        d = f"{reports}/{engine}"
        os.makedirs(d, exist_ok=True)
        json.dump({"audioSeconds": round(total_audio), "computeSeconds": round(compute_s),
                   "wallSeconds": round(wall_s), "missing": missing},
                  open(f"{d}/_meta.json", "w"), indent=2)
        print(f"{engine}: {len(refs) - missing}/{len(refs)} transcribed, "
              f"wall={round(wall_s)}s", flush=True)

    # Apple SpeechAnalyzer
    sacli = f"{ROOT}/SpeechAnalyzerCLI/.build/release/sacli"
    if want("apple") and os.path.exists(sacli):
        ad = f"{reports}/apple-speechanalyzer"
        os.makedirs(ad, exist_ok=True)
        t0 = time.time()
        p = subprocess.run([sacli, "--audio-folder", f"{audio}/audio", "--out", ad],
                           stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            print(f"!!! apple FAILED: {p.stderr[-300:]}", flush=True)
        else:
            miss = sum(0 if os.path.exists(f"{ad}/{u}.json") else 1 for u in refs)
            meta("apple-speechanalyzer", wall, wall, miss)
    elif want("apple"):
        print("skip apple: build SpeechAnalyzerCLI first (macOS 26+)", flush=True)

    # WhisperKit (report writes <uid>.json with text + timings)
    for model in WHISPER_MODELS:
        if not want(model):
            continue
        rdir = f"{reports}/{model}"
        os.makedirs(rdir, exist_ok=True)
        t0 = time.time()
        p = subprocess.run(
            ["swift", "run", "-c", "release", "whisperkit-cli", "transcribe",
             "--audio-folder", f"{audio}/audio", "--model", model, "--language", "en",
             "--chunking-strategy", "none", "--report", "--report-path", rdir],
            cwd=f"{ROOT}/WhisperKit", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            print(f"!!! {model} FAILED: {p.stderr[-300:]}", flush=True)
            continue
        compute = 0.0
        miss = 0
        for u in refs:
            rf = f"{rdir}/{u}.json"
            if os.path.exists(rf):
                compute += json.load(open(rf)).get("timings", {}).get("fullPipeline", 0.0)
            else:
                miss += 1
        meta(model, compute, wall, miss)

    # Parakeet (write per-utterance {"text": ...} ourselves)
    if any(want(n) for n, _ in PARAKEET):
        from parakeet_mlx import from_pretrained
        for name, repo in PARAKEET:
            if not want(name):
                continue
            rdir = f"{reports}/{name}"
            os.makedirs(rdir, exist_ok=True)
            m = from_pretrained(repo)
            compute = 0.0
            for uid in refs:
                t = time.time()
                text = m.transcribe(f"{audio}/audio/{uid}.wav").text
                compute += time.time() - t
                json.dump({"text": text}, open(f"{rdir}/{uid}.json", "w"))
            meta(name, compute, compute, 0)

    print("ALL DONE. Now: python score_ood.py", cfg, flush=True)


if __name__ == "__main__":
    main()
