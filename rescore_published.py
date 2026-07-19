"""Tier B: re-score Inscribe's PUBLISHED Apple transcripts. No audio needed.

Inscribe released the per-utterance transcripts for both Apple engines
(SpeechAnalyzer and the legacy SFSpeechRecognizer). Each record carries the
LibriSpeech `reference` and the engine `hypothesis`, so we can recompute WER
from scratch with our own normalizer and check it lands on their summary.json.

This verifies their SCORING is honest. It does NOT verify the transcripts are
genuine engine output (for that, run the audio yourself: see run_whisperkit.py).
"""
import gzip
import json
import urllib.request

BASE = "https://get-inscribe.com/data/speech-benchmark"
FILES = {
    "summary": "summary.json",
    "apple": "raw-transcripts-apple.json.gz",
    "legacy": "raw-transcripts-legacy.json.gz",
}


def fetch(name: str) -> bytes:
    url = f"{BASE}/{FILES[name]}"
    req = urllib.request.Request(url, headers={"User-Agent": "asr-repro"})
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    from normalize_wer import corpus_wer

    summary = json.loads(fetch("summary"))
    reported = {(r["engine"], r["split"]): r["werPercent"] for r in summary["results"]}

    print(f"{'engine':38}{'split':11}{'ours':>8}{'theirs':>8}{'delta':>8}")
    for key in ("apple", "legacy"):
        blocks = json.loads(gzip.decompress(fetch(key)))
        for blk in blocks:
            eng, split = blk["engine"], blk["split"]
            pairs = ((u["reference"], u.get("hypothesis") or "") for u in blk["transcripts"])
            wer, _, _ = corpus_wer(pairs)
            their = reported.get((eng, split))
            delta = f"{wer - their:+.2f}" if their is not None else "n/a"
            print(f"{eng:38}{split:11}{wer:>8.2f}{their:>8.2f}{delta:>8}")

    print(
        "\nNote: with OpenAI's stock EnglishTextNormalizer our numbers run a few "
        "tenths BELOW Inscribe's, because their normalizer is slightly stricter "
        "(they say so). The normalizer applies equally to every engine, so the "
        "ranking is unchanged; only the absolute WER shifts."
    )


if __name__ == "__main__":
    main()
