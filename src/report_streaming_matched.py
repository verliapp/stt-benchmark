"""Matched-clip streaming-vs-batch WER delta with a paired bootstrap.

report_streaming.py pairs streaming WER (on the streamed sample) against batch WER
from results_<config>.json (the full set), so small deltas sit inside sampling
noise. This tool removes that: it scores the batch and streaming transcripts on the
exact same clips (the clips the engine streamed) and puts a 95% paired-bootstrap
interval on the delta, the same test part two used for its tie claims.

    python src/report_streaming_matched.py librispeech
    python src/report_streaming_matched.py earnings22

Reads the committed packed transcripts under results/transcripts/:
  <config>-<engine>-stream.json.gz   streaming hypotheses
  <config>-<engine>.json.gz          batch hypotheses (same engine, no suffix)
and data_<config>/refs.json. Writes results/streaming_delta_<config>.json and
prints the table. A positive delta means streaming scored worse than batch.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from normalize_wer import score_pairs  # noqa: E402


def load_gz(path):
    return json.load(gzip.open(path, "rt")) if os.path.exists(path) else None


def paired_delta_ci(stream_counts, batch_counts, resamples=1000, seed=0):
    """95% interval for (stream corpus WER - batch corpus WER), resampling the
    shared clip set with replacement. Both engines are resampled on the same drawn
    indices each iteration, so the interval is on the paired difference. Returns
    (delta_point, lo, hi)."""
    se = np.array([c[0] for c in stream_counts], dtype=float)
    sw = np.array([c[1] for c in stream_counts], dtype=float)
    be = np.array([c[0] for c in batch_counts], dtype=float)
    bw = np.array([c[1] for c in batch_counts], dtype=float)
    point = (100.0 * se.sum() / sw.sum()) - (100.0 * be.sum() / bw.sum())
    n = len(stream_counts)
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        s = 100.0 * se[idx].sum() / sw[idx].sum()
        b = 100.0 * be[idx].sum() / bw[idx].sum()
        deltas[i] = s - b
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return round(float(point), 2), round(float(lo), 2), round(float(hi), 2)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python report_streaming_matched.py <config>")
    cfg = sys.argv[1]
    refs = json.load(open(f"{ROOT}/data_{cfg}/refs.json"))
    tdir = f"{ROOT}/results/transcripts"

    rows = []
    for sp in sorted(glob.glob(f"{tdir}/{cfg}-*-stream.json.gz")):
        engine = os.path.basename(sp)[len(cfg) + 1:-len("-stream.json.gz")]
        stream = load_gz(sp)
        batch = load_gz(f"{tdir}/{cfg}-{engine}.json.gz")
        if batch is None:
            rows.append({"engine": engine, "n": len(stream), "streamWer": None})
            continue
        uids = sorted(set(stream) & set(batch) & set(refs))
        s_pairs = [(refs[u], stream[u]) for u in uids]
        b_pairs = [(refs[u], batch[u]) for u in uids]
        s_wer, _, _, s_counts = score_pairs(s_pairs)
        b_wer, _, _, b_counts = score_pairs(b_pairs)
        point, lo, hi = paired_delta_ci(s_counts, b_counts)
        rows.append({
            "engine": engine, "n": len(uids),
            "streamWer": round(s_wer, 2), "batchWer": round(b_wer, 2),
            "delta": point, "deltaCi95": [lo, hi],
            "tie": lo <= 0.0 <= hi,
        })

    rows.sort(key=lambda r: (r.get("streamWer") is None, r.get("streamWer") or 0))
    outp = f"{ROOT}/results/streaming_delta_{cfg}.json"
    json.dump({"config": cfg, "rows": rows}, open(outp, "w"), indent=2)

    print(f"\n{cfg}  matched-clip streaming vs batch (+ = streaming worse)\n")
    hdr = f"{'engine':30}{'n':>5}{'stream':>8}{'batch':>8}{'delta':>8}{'95% CI':>16}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("streamWer") is None:
            print(f"{r['engine']:30}{r['n']:>5}{'-':>8}{'no batch':>8}")
            continue
        ci = f"{r['deltaCi95'][0]:+.2f},{r['deltaCi95'][1]:+.2f}"
        verdict = "tie (within noise)" if r["tie"] else (
            "streaming worse" if r["delta"] > 0 else "streaming better")
        print(f"{r['engine']:30}{r['n']:>5}{r['streamWer']:>8.2f}{r['batchWer']:>8.2f}"
              f"{r['delta']:>+8.2f}{ci:>16}  {verdict}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
