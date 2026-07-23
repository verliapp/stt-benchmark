"""Summarize the streaming run: real-time latency per engine, and the batch-vs-stream
WER delta on the same config.

Reads the per-clip timing under results/streaming/<config>/<engine>/ and, if present,
results_<config>.json (written by score_ood.py) to pair each streaming engine with its
batch counterpart (same name minus the `-stream` suffix).

    python src/report_streaming.py librispeech

Writes results/streaming_latency_<config>.json and prints the delta table.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def median(vals):
    return pct(vals, 50)


def collect(config):
    troot = os.path.join(ROOT, "results", "streaming", config)
    out = {}
    for edir in sorted(glob.glob(os.path.join(troot, "*"))):
        if not os.path.isdir(edir):
            continue
        engine = os.path.basename(edir)
        ttfp, ttff, lag, revs = [], [], [], []
        n = 0
        for fp in glob.glob(os.path.join(edir, "*.json")):
            if os.path.basename(fp) == "_meta.json":
                continue
            m = json.load(open(fp))
            n += 1
            if m.get("ttfp_s") is not None:
                ttfp.append(m["ttfp_s"])
            if m.get("ttff_s") is not None:
                ttff.append(m["ttff_s"])
            if m.get("finalization_lag_s") is not None:
                lag.append(m["finalization_lag_s"])
            if m.get("n_revisions") is not None:
                revs.append(m["n_revisions"])
        if n:
            out[engine] = {
                "n": n,
                "ttfp_median_s": median(ttfp), "ttfp_p90_s": pct(ttfp, 90),
                "ttff_median_s": median(ttff), "ttff_p90_s": pct(ttff, 90),
                "finalization_lag_median_s": median(lag),
                "finalization_lag_p90_s": pct(lag, 90),
                "mean_revisions": round(sum(revs) / len(revs), 1) if revs else None,
            }
    return out


def wer_by_engine(config):
    """{engine: werPercent} from score_ood's results/results_<config>.json (the
    werPercent key holds WER or CER depending on the config's mode)."""
    path = os.path.join(ROOT, "results", f"results_{config}.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    return {r["engine"]: r["werPercent"] for r in data.get("results", [])
            if "engine" in r and "werPercent" in r}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python report_streaming.py <config>")
    config = sys.argv[1]
    latency = collect(config)
    if not latency:
        sys.exit(f"no streaming timing under results/streaming/{config}; "
                 f"run run_streaming_engines.py {config} first")
    wer = wer_by_engine(config)

    outp = os.path.join(ROOT, f"results/streaming_latency_{config}.json")
    json.dump(latency, open(outp, "w"), indent=2)

    print(f"config={config}\n")
    header = (f"{'engine':30} {'n':>4} {'ttfp med':>9} {'ttff med':>9} "
              f"{'lag med':>8} {'stream WER':>11} {'batch WER':>10} {'Δ':>7}")
    print(header)
    print("-" * len(header))
    for engine in sorted(latency):
        m = latency[engine]
        s_wer = wer.get(engine)
        b_wer = wer.get(engine.removesuffix("-stream")) if engine.endswith("-stream") else None
        delta = round(s_wer - b_wer, 2) if (s_wer is not None and b_wer is not None) else None
        print(f"{engine:30} {m['n']:>4} {str(m['ttfp_median_s']):>9} "
              f"{str(m['ttff_median_s']):>9} {str(m['finalization_lag_median_s']):>8} "
              f"{str(s_wer):>11} {str(b_wer):>10} {str(delta):>7}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
