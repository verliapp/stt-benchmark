"""Per-language paired-bootstrap winner test for the FLEURS runs.

For each language, rank engines by score (WER, or CER for no-whitespace
languages), then run a paired bootstrap on the top pairs. The paired test
resamples the same clips and takes the score difference, which is far more
sensitive than comparing marginal confidence intervals. A language has a
"clear winner" only if #1 significantly beats #2 (difference interval excludes 0).

Compares only clips BOTH engines returned, so a coverage gap does not distort
the comparison. Deterministic (seed 0).

    ./.venv/bin/python src/paired_langs.py
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from normalize_wer import edit_distance, mode_tokens  # noqa: E402

CFG = ["es_419", "fr_fr", "de_de", "cmn_hans_cn", "ja_jp",
       "ko_kr", "ar_eg", "hi_in", "ru_ru", "vi_vn"]
CER = {"cmn_hans_cn", "ja_jp", "th_th"}
NAME = {"openai-gpt-4o-transcribe": "OpenAI", "assemblyai-universal": "AssemblyAI",
        "soniox": "Soniox", "speechmatics-enhanced": "Speechmatics",
        "gladia-solaria": "Gladia", "elevenlabs-scribe": "ElevenLabs",
        "groq-whisper-v3-turbo": "Groq", "lemonfox-whisper": "Lemonfox",
        "gemini-2.5-flash": "Gemini", "grok-stt": "Grok", "fish-audio-asr": "Fish",
        "inworld-stt-1": "Inworld", "amazon-transcribe": "Amazon", "azure": "Azure",
        "google-chirp": "Google", "deepgram-nova-3": "Deepgram"}


def hyps(cfg, engine):
    p = f"{ROOT}/results/transcripts/fleurs_{cfg}-{engine}.json.gz"
    return json.load(gzip.open(p, "rt")) if os.path.exists(p) else {}


def corpus(refs, h, uids, mode):
    e = w = 0
    for u in uids:
        r = mode_tokens(refs[u], mode)
        e += edit_distance(r, mode_tokens(h[u], mode)); w += len(r)
    return 100.0 * e / w if w else 0.0


def paired(refs, ha, hb, uids, mode, n=1000, seed=0):
    ea, eb, w = [], [], []
    for u in uids:
        r = mode_tokens(refs[u], mode)
        ea.append(edit_distance(r, mode_tokens(ha[u], mode)))
        eb.append(edit_distance(r, mode_tokens(hb[u], mode)))
        w.append(len(r))
    ea, eb, w = np.array(ea), np.array(eb), np.array(w)
    rng = np.random.default_rng(seed)
    N = len(w); diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        d = w[idx].sum()
        diffs[i] = 100.0 * (ea[idx].sum() - eb[idx].sum()) / d if d else 0.0
    pt = 100.0 * (ea.sum() - eb.sum()) / w.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return pt, lo, hi


for cfg in CFG:
    mode = "cer" if cfg in CER else "word_basic"
    metric = "CER" if cfg in CER else "WER"
    refs = json.load(open(f"{ROOT}/data_fleurs_{cfg}/refs.json"))
    # Empty returned transcripts stay in the matched set and score as deletions.
    # Only absent UIDs are coverage gaps.
    cov = {}
    for e in NAME:
        h = hyps(cfg, e)
        got = {u: h[u] for u in refs if u in h}
        if len(got) >= 0.85 * len(refs):
            cov[e] = got
    if len(cov) < 2:
        print(f"{cfg:12} ({metric})  <2 engines with >=85% coverage; skipping", flush=True)
        continue
    matched = [u for u in refs if all(u in h for h in cov.values())]
    if len(matched) < 0.85 * len(refs):
        coverage = 100.0 * len(matched) / len(refs) if refs else 0.0
        print(f"{cfg:12} ({metric})  matched coverage {coverage:.1f}% is below 85%; "
              "skipping", flush=True)
        continue
    ranked = sorted(cov, key=lambda e: corpus(refs, cov[e], matched, mode))
    top = ranked[:3]
    a, b = top[0], top[1]
    pt, lo, hi = paired(refs, cov[a], cov[b], matched, mode)
    sig = ("CLEAR WINNER" if hi < 0 else
           f"REVERSED: {NAME[b]} wins" if lo > 0 else "tie")
    sa = corpus(refs, cov[a], matched, mode)
    sb = corpus(refs, cov[b], matched, mode)
    print(f"{cfg:12} ({metric})  {NAME[a]} {sa:.2f} vs {NAME[b]} {sb:.2f}  "
          f"paired {pt:+.2f} [{lo:+.2f},{hi:+.2f}]  -> {sig}")
    # if top-2 tie, test #1 vs #3 to size the leading cluster
    if sig == "tie" and len(top) >= 3:
        c = top[2]
        pt3, lo3, hi3 = paired(refs, cov[a], cov[c], matched, mode)
        sig3 = ("beats #3" if hi3 < 0 else
                "#3 wins" if lo3 > 0 else "also ties #3")
        print(f"{'':12}          vs #3 {NAME[c]} {corpus(refs, cov[c], matched, mode):.2f}  "
              f"paired {pt3:+.2f} [{lo3:+.2f},{hi3:+.2f}]  -> {sig3}")
