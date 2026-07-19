"""Score the saved out-of-domain transcripts: corpus WER + 95% bootstrap CI.

Reads data_<config>/refs.json and results/reports/<config>/<engine>/<uid>.json for
every engine, scores each with the shared OpenAI normalizer and corpus WER, puts a
95% confidence interval on each number by resampling utterances, and writes:

  results_<config>.json                          summary (committed)
  results/transcripts/<config>-<engine>.json.gz  raw hypotheses (committed)

    python score_ood.py earnings22

Confidence intervals matter here: out-of-domain the top engines are close, so any
ranking whose intervals overlap is a tie, not a real ordering.
"""
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from normalize_wer import score_pairs, bootstrap_ci

    cfg = sys.argv[1]
    refs = json.load(open(f"{ROOT}/data_{cfg}/refs.json"))
    reports = f"{ROOT}/results/reports/{cfg}"
    tdir = f"{ROOT}/results/transcripts"
    os.makedirs(tdir, exist_ok=True)

    engines = sorted(os.path.basename(os.path.dirname(p))
                     for p in glob.glob(f"{reports}/*/"))
    if not engines:
        sys.exit(f"no engine reports under {reports}; run run_ood_engines.py {cfg} first")

    rows = []
    for engine in engines:
        rdir = f"{reports}/{engine}"
        # Score each engine only on the clips it actually returned. A clip the
        # engine never produced (file absent) is a coverage gap, not a wrong
        # transcript, so it is excluded from WER rather than counted as a full
        # error. Coverage is reported alongside so a partial engine (e.g. Rev AI,
        # which rejects clips under ~2s) is visibly not scored on the full set.
        hyps = {}
        for uid in refs:
            rf = f"{rdir}/{uid}.json"
            if os.path.exists(rf):
                hyps[uid] = json.load(open(rf)).get("text", "") or ""
        scored_uids = [uid for uid in refs if uid in hyps]
        missing = len(refs) - len(scored_uids)
        coverage = round(100.0 * len(scored_uids) / len(refs), 1) if refs else 0.0
        pairs = [(refs[uid], hyps[uid]) for uid in scored_uids]
        wer, errors, ref_words, counts = score_pairs(pairs)
        lo, hi = bootstrap_ci(counts)
        meta = {}
        mp = f"{rdir}/_meta.json"
        if os.path.exists(mp):
            meta = json.load(open(mp))
        rows.append({"engine": engine, "werPercent": round(wer, 2),
                     "ci95": [lo, hi], "missing": missing,
                     "scored": len(scored_uids), "coveragePercent": coverage,
                     "utterances": len(refs), "refWords": ref_words,
                     "wallSeconds": meta.get("wallSeconds")})
        # pack raw hypotheses for committing (mirrors how Inscribe published theirs)
        with gzip.open(f"{tdir}/{cfg}-{engine}.json.gz", "wt") as fh:
            json.dump(hyps, fh)

    rows.sort(key=lambda r: r["werPercent"])
    json.dump({"config": cfg, "utterances": len(refs), "results": rows},
              open(f"{ROOT}/results/results_{cfg}.json", "w"), indent=2)

    print(f"\n{cfg}  ({len(refs)} utterances)\n")
    print(f"{'engine':30}{'WER%':>8}{'95% CI':>16}{'scored':>8}{'cov%':>7}")
    for r in rows:
        ci = f"{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}"
        print(f"{r['engine']:30}{r['werPercent']:>8.2f}{ci:>16}"
              f"{r['scored']:>8}{r['coveragePercent']:>7.1f}")


if __name__ == "__main__":
    main()
