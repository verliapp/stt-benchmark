"""Run the full Giggy accuracy evaluation, score it, and retain progress.

    ./.venv/bin/python -u src/evaluate_giggy.py

Run isolated STT and TTS latency before this bulk run. This controller runs only
Giggy, merges only its result rows, and checks that all expected clips either
returned a transcript or have an explicit failure reason.
"""
import datetime
import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "giggy-free-stt-v1"
DATASETS = ["librispeech", "earnings22", "spgispeech", "ami"] + [
    "fleurs_" + lang for lang in
    ("ar_eg", "cmn_hans_cn", "de_de", "es_419", "fr_fr", "hi_in",
     "ja_jp", "ko_kr", "ru_ru", "vi_vn")
]
STATE = ROOT / "results/reports/giggy-evaluation-state.json"


def save(state):
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def run(script, *args):
    command = [sys.executable, str(ROOT / "src" / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def verified_complete(dataset):
    """Only skip a checkpoint whose packed results and failure counts reconcile."""
    try:
        refs = json.loads((ROOT / f"data_{dataset}/refs.json").read_text())
        scores = json.loads((ROOT / "results" / f"results_{dataset}.json").read_text())
        row = next(r for r in scores["results"] if r["engine"] == ENGINE)
        with gzip.open(ROOT / "results/transcripts" / f"{dataset}-{ENGINE}.json.gz", "rt") as fh:
            hyps = json.load(fh)
        rdir = ROOT / "results/reports" / dataset / ENGINE
        errors = json.loads((rdir / ".errors.json").read_text())
        meta = json.loads((rdir / "_meta.json").read_text())
        return (set(hyps).issubset(refs) and set(errors) == set(refs) - set(hyps)
                and row["scored"] == len(hyps) and row["missing"] == len(errors)
                and row["utterances"] == len(refs) and meta["missing"] == len(errors)
                and all("0.5-second minimum" in e for e in errors.values()))
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return False


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state:
        state.setdefault("resumed_at", []).append(now)
    state.update({"status": "running", "started_at": state.get("started_at", now),
                  "datasets": DATASETS,
                  "completed_datasets": state.get("completed_datasets", [])})
    state.pop("error", None)
    try:
        for dataset in DATASETS:
            if dataset in state["completed_datasets"] and verified_complete(dataset):
                state.get("unexpected_failures", {}).pop(dataset, None)
                print("RESUME: keeping completed dataset", dataset, flush=True)
                continue
            state["completed_datasets"] = [d for d in state["completed_datasets"] if d != dataset]
            state.get("unexpected_failures", {}).pop(dataset, None)
            state["current_dataset"] = dataset
            save(state)
            print("START", dataset, flush=True)
            run("run_cloud_engines.py", dataset, "giggy")
            run("score_ood.py", dataset, "--engine", ENGINE)
            run("report.py", dataset, "--engine", ENGINE)
            scores = json.loads((ROOT / "results" / f"results_{dataset}.json").read_text())
            row = next(r for r in scores["results"] if r["engine"] == ENGINE)
            rdir = ROOT / "results/reports" / dataset / ENGINE
            errors = json.loads((rdir / ".errors.json").read_text())
            if row["scored"] + row["missing"] != row["utterances"]:
                raise RuntimeError(f"{dataset}: coverage does not reconcile")
            if row["missing"] != len(errors):
                raise RuntimeError(f"{dataset}: missing transcripts do not reconcile with errors")
            unexpected = {u: e for u, e in errors.items() if "0.5-second minimum" not in e}
            if unexpected:
                state.setdefault("unexpected_failures", {})[dataset] = unexpected
            state["completed_datasets"].append(dataset)
            save(state)
            run("report_giggy.py")
            print("SCORED", dataset, row, flush=True)
        state["status"] = "needs_review" if state.get("unexpected_failures") else "complete"
        state["current_dataset"] = None
        save(state)
        print(state["status"].upper(), flush=True)
    except Exception as e:
        state["status"] = "failed"
        state["error"] = f"{type(e).__name__}: {e}"
        save(state)
        raise


if __name__ == "__main__":
    main()
