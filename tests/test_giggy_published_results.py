"""Offline consistency checks for published Giggy results and immutable archive."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "archive/giggy/2026-10-13"
ENGINE = "giggy-free-stt-v1"


def read(path):
    return json.loads(path.read_text())


def test_current_transcript_counts_and_scores_agree():
    evaluation = read(RESULTS / "giggy_evaluation.json")
    assert len(evaluation["datasets"]) == 14
    assert sum(r["scored"] for r in evaluation["datasets"]) == 18348
    assert sum(r["missing"] for r in evaluation["datasets"]) == 2755
    assert evaluation["coverage"]["local_short_exclusions"] == 2752
    assert evaluation["coverage"]["service_or_admission_gaps"] == 3
    assert evaluation["run_status"] == "needs_review"
    for row in evaluation["datasets"]:
        cfg = row["config"]
        with gzip.open(RESULTS / f"transcripts/{cfg}-{ENGINE}.json.gz", "rt") as f:
            transcripts = json.load(f)
        assert len(transcripts) == row["scored"]
        assert all(isinstance(v, str) for v in transcripts.values())
        scores = read(RESULTS / f"results_{cfg}.json")
        score = next(r for r in scores["results"] if r["engine"] == ENGINE)
        for key in ("werPercent", "ci95", "missing", "scored", "coveragePercent"):
            assert score[key] == row[key]
        report = read(RESULTS / f"report_{cfg}.json")
        current = next(r for r in report["rows"] if r["engine"] == ENGINE)
        for key in ("werPercent", "ci95", "missing"):
            assert current[key] == row[key]
        assert current["latency_median_s"] == 4.535
        assert current["latency_p90_s"] == 5.166
        assert row["provenance"]["models"] == ["GSTT"]
    metrics = {r["config"]: r["metric"] for r in evaluation["datasets"]}
    assert metrics["fleurs_cmn_hans_cn"] == metrics["fleurs_ja_jp"] == "CER"


def test_archive_hashes_and_other_providers_unchanged():
    for name, expected in read(ARCHIVE / "manifest.json")["sha256"].items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == expected
        if name.startswith(("results_", "report_")):
            key = "results" if name.startswith("results_") else "rows"
            old = read(ARCHIVE / name)
            current = read(RESULTS / name)
            assert [r for r in old[key] if r["engine"] != ENGINE] == [
                r for r in current[key] if r["engine"] != ENGINE]
            assert {k: v for k, v in old.items() if k != key} == {
                k: v for k, v in current.items() if k != key}
    old = read(ARCHIVE / "latency.json")
    current = read(RESULTS / "latency.json")
    assert {k: v for k, v in old.items() if k != ENGINE} == {
        k: v for k, v in current.items() if k != ENGINE}
    assert read(ARCHIVE / "giggy_evaluation.json")["totals"]["scored_clips"] == 18351


def test_latency_and_accounting_caveats_retained():
    report = read(RESULTS / "giggy_evaluation.json")
    timing = report["isolated_stt_latency"]
    assert timing == read(RESULTS / "latency.json")[ENGINE]
    assert (timing["n"], timing["median_s"], timing["p90_s"]) == (40, 4.535, 5.166)
    assert timing["transport_cohorts"]["system_address_selection"]["n"] == 2
    assert timing["mean_s"] == 19.263 and timing["max_s"] == 305.089
    assert report["usage"]["confirmed_net_credits_all_attempts"] == 124964
    assert report["usage"]["uncertain_reserved_credits"] == 12
    assert sum(r["provenance"]["creditsDeducted"] for r in report["datasets"]) == 124549
    assert len(report["coverage"]["missing_eligible_clips"]) == 3
    readme = (ROOT / "README.md").read_text()
    assert "| Giggy Free STT v1 (Fast)  | 1.69 | 10.06 |" in readme
    assert "2,752 clips" in readme and "305-second IPv6" in readme
    assert not any("giggy" in line.lower() for line in readme.splitlines() if line.startswith("#"))
    assert "Qwen" not in readme


def test_article_comparisons_use_matched_coverage():
    comparison = read(RESULTS / "giggy_comparisons.json")
    earnings = comparison["datasets"]["earnings22"]
    assert round(earnings["giggy_normalized"]["wer"], 2) == 9.48
    assert earnings["giggy_normalized"]["n"] == 2652
    assert all(pair["n_common"] == 2652 for pair in earnings["pairs"])
    inworld = next(pair for pair in earnings["pairs"] if pair["engine"] == "inworld-stt-1")
    assert inworld["ci95"][1] < 0
    for name, expected in comparison["sources"].items():
        if name.startswith("results/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert "/Users/" not in json.dumps(comparison)


def test_paired_helper_excludes_missing_but_scores_returned_empty():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from compare_giggy import counts, paired
    refs = {"a": "hello", "b": "world", "missing": "no transcript"}
    first = counts(refs, {"a": "hello", "b": ""})
    second = counts(refs, {"a": "hello", "b": "world", "missing": "wrong"})
    assert first["b"] == (1, 1)
    assert "missing" not in first
    result = paired(refs, first, second)
    assert result["n_common"] == 2
    assert result["difference_pp"] == 50
