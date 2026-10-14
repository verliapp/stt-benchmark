"""Collect Giggy results and provenance without rerunning any provider calls."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "giggy-free-stt-v1"


def main():
    rows = []
    pricing = json.loads((ROOT / "pricing.json").read_text())["providers"][ENGINE]
    for summary in sorted((ROOT / "results").glob("results_*.json")):
        scores = json.loads(summary.read_text())
        result = next((r for r in scores["results"] if r["engine"] == ENGINE), None)
        if result is None:
            continue
        cfg = scores["config"]
        rdir = ROOT / "results/reports" / cfg / ENGINE
        meta = json.loads((rdir / "_meta.json").read_text())
        errors_path = rdir / ".errors.json"
        errors = json.loads(errors_path.read_text()) if errors_path.exists() else {}
        reasons = {}
        for error in errors.values():
            reasons[error] = reasons.get(error, 0) + 1
        refs = ROOT / f"data_{cfg}/refs.json"
        rows.append({
            "config": cfg, "metric": scores.get("metric", "WER"),
            "normalization": scores.get("mode", "word_en"), **result,
            "provenance": meta,
            "reference_sha256": hashlib.sha256(refs.read_bytes()).hexdigest(),
            "missing_reasons": reasons,
            "retail_equivalent_usd": round(meta.get("creditsDeducted", 0)
                                            * pricing["usd_per_credit"], 4),
        })
    latency_path = ROOT / "results/latency.json"
    latency = json.loads(latency_path.read_text()).get(ENGINE) if latency_path.exists() else None
    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": ENGINE, "mode": "fast", "advertised_model": "Giggy Free STT v1",
        "model_note": "Vendor email described an own model; canonical transcript artifacts identify Qwen/Qwen3-ASR-1.7B. Record actual revisions per dataset.",
        "sources": ["https://giggy.ai/docs/speech-api", "https://giggy.ai/v1/openapi.json",
                    "https://giggy.ai/pricing"],
        "methodology": {
            "bootstrap_resamples": 1000, "bootstrap_seed": 0,
            "coverage": "Missing results excluded from accuracy and counted separately. Empty returned transcripts count as deletions. Clips below 0.5 seconds rejected locally, never padded.",
            "preprocessing": "Mono PCM16 at 16 kHz, polyphase resampling only when required; original dataset audio preserved.",
            "comparison": "Other providers' results are historical snapshots. Compare on mutually returned clips before claiming a statistically significant win.",
            "cost": "Vendor-granted evaluation credits. USD is a retail-equivalent estimate at $10/million credits, including actual per-request credit rounding; not an invoice.",
            "packages": {p: importlib.metadata.version(p) for p in
                         ("openai-whisper", "numpy", "scipy", "soundfile", "requests")},
        },
        "datasets": rows, "isolated_stt_latency": latency,
        "totals": {"scored_clips": sum(r["scored"] for r in rows),
                   "missing_clips": sum(r["missing"] for r in rows),
                   "credits_deducted": sum(r["provenance"].get("creditsDeducted", 0) for r in rows),
                   "retail_equivalent_usd": round(sum(r["provenance"].get("creditsDeducted", 0)
                                                       for r in rows) * pricing["usd_per_credit"], 4)},
    }
    path = ROOT / "results/giggy_evaluation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
