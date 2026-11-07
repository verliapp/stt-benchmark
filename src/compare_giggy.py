"""Reproduce the Giggy article comparisons offline from packed transcripts.

Requires the prepared reference files under data_<dataset>/refs.json. Use
--references-root to select an existing prepared dataset directory. No API calls.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from normalize_wer import bootstrap_ci, edit_distance, tokens, tokens_no_disfluency

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "giggy-free-stt-v1"
COMPARATORS = {
    "librispeech": ["openai-gpt-4o-transcribe", "assemblyai-universal", "fish-audio-asr",
                   "inworld-stt-1", "grok-stt", "parakeet-tdt-0.6b-v2"],
    "earnings22": ["grok-stt", "fish-audio-asr", "inworld-stt-1", "assemblyai-universal"],
    "spgispeech": ["assemblyai-universal", "parakeet-v2", "google-chirp", "azure",
                   "fish-audio-asr", "deepgram-nova-3", "speechmatics-enhanced",
                   "lemonfox-whisper", "groq-whisper-v3-turbo", "elevenlabs-scribe"],
}
METHOD = (
    "Corpus WER with EnglishTextNormalizer; paired bootstrap over common returned "
    "clips only, 1000 resamples seed 0, unadjusted 95% intervals. Differences = "
    "Giggy minus comparator; positive means Giggy higher error. Providers were "
    "measured on different dates. Empty returned text is scored; absent hypotheses "
    "excluded. Reference insertion order as score_ood.py."
)


def counts(refs, hypotheses, normalize=tokens):
    result = {}
    for uid, reference in refs.items():
        if uid in hypotheses:
            words = normalize(reference)
            result[uid] = (edit_distance(words, normalize(hypotheses[uid])), len(words))
    return result


def summarize(values):
    array = np.array(list(values.values()))
    return {"n": len(values), "errors": int(array[:, 0].sum()),
            "reference_words": int(array[:, 1].sum()),
            "wer": float(100 * array[:, 0].sum() / array[:, 1].sum()),
            "ci95": bootstrap_ci(list(values.values()))}


def paired(refs, first, second):
    common = [uid for uid in refs if uid in first and uid in second]
    a = np.array([first[uid][0] for uid in common])
    b = np.array([second[uid][0] for uid in common])
    words = np.array([first[uid][1] for uid in common])
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(1000):
        idx = rng.integers(0, len(common), len(common))
        samples.append(100 * (a[idx].sum() - b[idx].sum()) / words[idx].sum())
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"n_common": len(common), "giggy_common_wer": float(100 * a.sum() / words.sum()),
            "other_common_wer": float(100 * b.sum() / words.sum()),
            "difference_pp": float(100 * (a.sum() - b.sum()) / words.sum()),
            "ci95": [float(low), float(high)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/giggy_comparisons.json")
    args = parser.parse_args()
    output = {"method": METHOD, "sources": {}, "datasets": {}}

    def load(path, label):
        output["sources"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                return json.load(stream)
        return json.loads(path.read_text())

    for dataset, engines in COMPARATORS.items():
        name = f"data_{dataset}/refs.json"
        refs = load(args.references_root / name, name)
        name = f"results/transcripts/{dataset}-{ENGINE}.json.gz"
        hypotheses = load(ROOT / name, name)
        giggy = counts(refs, hypotheses)
        row = {"total_clips": len(refs), "giggy": summarize(giggy), "pairs": []}
        if dataset == "earnings22":
            row["giggy_normalized"] = summarize(counts(refs, hypotheses, tokens_no_disfluency))
        for engine in engines:
            name = f"results/transcripts/{dataset}-{engine}.json.gz"
            other = counts(refs, load(ROOT / name, name))
            row["pairs"].append({"engine": engine, **paired(refs, giggy, other),
                                 "other_full": summarize(other)})
        output["datasets"][dataset] = row
        print(f"Scored {dataset}: {len(row['pairs'])} paired comparisons")
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
