"""Paired bootstrap: is engine A's WER really different from engine B's?

Comparing two engines by whether their marginal confidence intervals overlap is a
weak test. A paired bootstrap resamples the same utterances and takes the WER
difference on each resample, which cancels the shared per-utterance difficulty and is
the correct significance test. If the difference interval excludes 0, the gap is real.

Reads committed transcripts under results/transcripts/ and the references (LibriSpeech
from setup.sh, out-of-domain from prep_ood_dataset.py). Deterministic (seed 0).

    python paired_test.py
"""
import glob
import gzip
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_refs(ds):
    if ds == "librispeech":
        refs = {}
        for tf in glob.glob(f"{ROOT}/data/LibriSpeech/test-clean/**/*.trans.txt", recursive=True):
            for line in open(tf):
                uid, text = line.strip().split(" ", 1)
                refs[uid] = text
        return refs
    return json.load(open(f"{ROOT}/data_{ds}/refs.json"))


def load_hyps(ds, engine):
    return json.load(gzip.open(f"{ROOT}/results/transcripts/{ds}-{engine}.json.gz", "rt"))


def paired(ds, a, b, n=1000, seed=0):
    from normalize_wer import tokens, edit_distance

    refs = load_refs(ds)
    ha, hb = load_hyps(ds, a), load_hyps(ds, b)
    ea, eb, w = [], [], []
    for u in refs:
        r = tokens(refs[u])
        ea.append(edit_distance(r, tokens(ha.get(u, ""))))
        eb.append(edit_distance(r, tokens(hb.get(u, ""))))
        w.append(len(r))
    ea, eb, w = np.array(ea), np.array(eb), np.array(w)
    rng = np.random.default_rng(seed)
    N = len(w)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = 100 * (ea[idx].sum() - eb[idx].sum()) / w[idx].sum()
    pt = 100 * (ea.sum() - eb.sum()) / w.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    sig = "significant" if (lo > 0 or hi < 0) else "TIE (interval spans 0)"
    print(f"{ds:12} {a} - {b}: {pt:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]  {sig}")


if __name__ == "__main__":
    paired("earnings22", "apple-speechanalyzer", "parakeet-v2")
    paired("earnings22", "apple-speechanalyzer", "parakeet-v3")
    paired("earnings22", "apple-speechanalyzer", "whisper-large-v3-v20240930")
    paired("librispeech", "parakeet-tdt-0.6b-v2", "apple-speechanalyzer")
    paired("librispeech", "apple-speechanalyzer", "whisper-large-v3")
    paired("librispeech", "apple-speechanalyzer", "whisper-large-v3-v20240930")
