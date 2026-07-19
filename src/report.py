"""Merge WER + cost + latency + throughput into one master table per dataset.

Pulls together everything recorded elsewhere:
  - accuracy:   results_<cfg>.json           (WER% + 95% CI + missing, from score_ood.py)
  - cost:       _meta.json per engine        (audioSeconds) x pricing.json
  - latency:    latency.json                 (from measure_latency.py)
  - throughput: _meta.json                   (audioSeconds / wallSeconds, under bulk load)

Cost is an estimate: it cannot be read back from most provider APIs, so we compute it
from the audio duration each engine processed times its published rate (pricing.json),
applying a per-job minimum where one exists, and pricing Gemini on tokens. Every rate
is dated in pricing.json.

    python report.py librispeech
    python report.py earnings22

Writes report_<cfg>.json and prints the table. Missing pieces (WER not scored yet,
latency not measured yet) show as "-" rather than failing.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def transcript_words(rdir):
    total = 0
    for f in glob.glob(f"{rdir}/*.json"):
        if f.endswith("_meta.json"):
            continue
        try:
            total += len((json.load(open(f)).get("text") or "").split())
        except Exception:  # noqa: BLE001
            pass
    return total


def clip_count(rdir):
    return sum(1 for f in glob.glob(f"{rdir}/*.json") if not f.endswith("_meta.json"))


def cost_for(engine, meta, pricing, rdir):
    p = pricing["providers"].get(engine)
    if not p or not meta:
        return None
    audio_s = meta.get("audioSeconds", 0)
    if p["unit"] == "per_audio_hour":
        billed_s = audio_s
        if p.get("min_seconds_per_job"):
            billed_s = max(audio_s, clip_count(rdir) * p["min_seconds_per_job"])
        return round(billed_s / 3600.0 * p["rate_usd"], 4)
    if p["unit"] == "per_token":
        in_tok = audio_s * p.get("audio_tokens_per_second", 32)
        out_tok = transcript_words(rdir) * 1.3  # rough words->tokens
        return round(in_tok / 1e6 * p["input_usd_per_mtok"]
                     + out_tok / 1e6 * p["output_usd_per_mtok"], 4)
    return None


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "librispeech"
    pricing = load_json(os.path.join(ROOT, "pricing.json"))
    latency = load_json(os.path.join(ROOT, "results", "latency.json"), {})
    scored = load_json(os.path.join(ROOT, "results", f"results_{cfg}.json"), {})
    wer_by = {r["engine"]: r for r in scored.get("results", [])}
    reports = os.path.join(ROOT, "results", "reports", cfg)

    rows = []
    for rdir in sorted(glob.glob(f"{reports}/*/")):
        engine = os.path.basename(os.path.dirname(rdir))
        meta = load_json(os.path.join(rdir, "_meta.json"))
        w = wer_by.get(engine, {})
        lat = latency.get(engine, {})
        cost = cost_for(engine, meta, pricing, rdir)
        audio_s = (meta or {}).get("audioSeconds")
        wall = (meta or {}).get("wallSeconds")
        rows.append({
            "engine": engine,
            "werPercent": w.get("werPercent"),
            "ci95": w.get("ci95"),
            "missing": w.get("missing"),
            "cost_usd": cost,
            "cost_per_audio_hour": round(cost / (audio_s / 3600.0), 4)
                if (cost is not None and audio_s) else None,
            "latency_median_s": lat.get("median_s"),
            "latency_p90_s": lat.get("p90_s"),
            "latency_mode": lat.get("mode"),
            "throughput_xrt_bulk": round(audio_s / wall, 1)
                if (audio_s and wall) else None,
        })

    rows.sort(key=lambda r: (r["werPercent"] is None, r["werPercent"] or 0))
    out = {"config": cfg, "note": "cost is an estimate from audio duration x published "
           "rate (pricing.json); latency is isolated (measure_latency.py); throughput "
           "is under bulk concurrency, not a clean per-call number.", "rows": rows}
    json.dump(out, open(os.path.join(ROOT, "results", f"report_{cfg}.json"), "w"), indent=2)

    print(f"\n{cfg}\n")
    hdr = f"{'engine':26}{'WER%':>7}{'95% CI':>15}{'cost$':>9}{'$/hr':>7}{'lat med':>9}{'xRT':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        wer = f"{r['werPercent']:.2f}" if r["werPercent"] is not None else "-"
        ci = f"{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}" if r.get("ci95") else "-"
        cost = f"{r['cost_usd']:.3f}" if r["cost_usd"] is not None else "-"
        cph = f"{r['cost_per_audio_hour']:.3f}" if r["cost_per_audio_hour"] is not None else "-"
        lat = (f"{r['latency_median_s']:.2f}{'a' if r['latency_mode']=='async' else ''}"
               if r["latency_median_s"] is not None else "-")
        xrt = f"{r['throughput_xrt_bulk']:.0f}x" if r["throughput_xrt_bulk"] else "-"
        print(f"{r['engine']:26}{wer:>7}{ci:>15}{cost:>9}{cph:>7}{lat:>9}{xrt:>7}")
    print(f"\nwrote results/report_{cfg}.json  (lat 'a' = async: includes job queue + poll)")


if __name__ == "__main__":
    main()
