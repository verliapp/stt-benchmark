"""Run Apple's on-device SpeechAnalyzer over LibriSpeech test-clean and score it.

This actually runs the engine on the audio (unlike rescore_published.py, which only
re-scores Inscribe's published transcripts), so it independently verifies Apple's
number end-to-end. Requires macOS 26+ and the built `sacli` harness:

    (cd SpeechAnalyzerCLI && swift build -c release)
    python run_apple.py

sacli is single-stream, so its wall time / audio seconds is already a clean RTF.
"""
import glob
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRI = os.path.join(ROOT, "data", "LibriSpeech", "test-clean")
FLAT = os.path.join(ROOT, "data", "audio_flat")
SACLI = os.path.join(ROOT, "SpeechAnalyzerCLI", ".build", "release", "sacli")
OUT = os.path.join(ROOT, "results", "reports", "apple-speechanalyzer")
RESULTS = os.path.join(ROOT, "results", "apple.json")


def load_references() -> dict[str, str]:
    refs = {}
    for tf in glob.glob(os.path.join(LIBRI, "**", "*.trans.txt"), recursive=True):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            refs[uid] = text
    return refs


def audio_seconds() -> float:
    import soundfile as sf
    total = 0.0
    for p in glob.glob(os.path.join(LIBRI, "**", "*.flac"), recursive=True):
        info = sf.info(p)
        total += info.frames / info.samplerate
    return total


def main():
    import gzip
    from normalize_wer import score_pairs, bootstrap_ci

    assert os.path.exists(SACLI), f"build the harness first: (cd SpeechAnalyzerCLI && swift build -c release)"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    refs = load_references()

    t0 = time.time()
    subprocess.run([SACLI, "--audio-folder", FLAT, "--out", OUT], check=True)
    wall = time.time() - t0

    hyps = {}
    missing = 0
    for uid in refs:
        rf = os.path.join(OUT, f"{uid}.json")
        if os.path.exists(rf):
            hyps[uid] = json.load(open(rf)).get("text", "")
        else:
            hyps[uid] = ""
            missing += 1

    wer, errors, ref_words, counts = score_pairs([(refs[uid], hyps[uid]) for uid in refs])
    lo, hi = bootstrap_ci(counts)
    # sacli is single-stream, so wall time / measured audio seconds is a clean RTF.
    audio_s = audio_seconds()
    res = {
        "engine": "apple-speechanalyzer", "split": "test-clean",
        "werPercent": round(wer, 2), "ci95": [lo, hi],
        "missing": missing, "utterances": len(refs),
        "audioSeconds": round(audio_s), "wallSeconds": round(wall),
        "realTimeFactor": round(wall / audio_s, 4) if audio_s else None,
    }
    json.dump(res, open(RESULTS, "w"), indent=2)
    tdir = os.path.join(ROOT, "results", "transcripts")
    os.makedirs(tdir, exist_ok=True)
    with gzip.open(os.path.join(tdir, "librispeech-apple-speechanalyzer.json.gz"), "wt") as fh:
        json.dump(hyps, fh)
    print(f"Apple SpeechAnalyzer test-clean: WER {res['werPercent']}% (95% CI {lo}-{hi})  "
          f"RTF {res['realTimeFactor']}  missing={missing}")


if __name__ == "__main__":
    main()
