"""Giggy recording contract, credential isolation, and resumable admission."""
import io
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cloud_adapters as api
import run_cloud_engines as runner
import score_ood
import evaluate_giggy as controller

GID = "450a9402-3568-4a96-b9db-5e183b2f93ef"


class Response:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status
        self.headers = {}

    def json(self):
        return self.body


def audio(tmp_path, seconds=1, sr=16000, channels=1):
    path = tmp_path / "clip.wav"
    samples = np.zeros((int(sr * seconds), channels))
    sf.write(path, samples, sr, subtype="PCM_16")
    return path


def cfg(tmp_path):
    return {**api.PROVIDERS["giggy"], "lang": "en", "job_dir": str(tmp_path / "jobs")}


def fake_api(monkeypatch, *, artifact=None, post_error=None):
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            calls.append(("post", url, kwargs))
            if post_error:
                raise post_error
            return Response({"generation": {"generation_uuid": GID}}, 202)

        def get(self, url, **kwargs):
            calls.append(("status", url, kwargs))
            return Response({"generation": {
                "status": "completed", "credits_deducted": 2,
                "result": {"url": "https://artifacts.example/final?secret=signed"},
            }})

    def get(url, **kwargs):
        calls.append(("artifact", url, kwargs))
        return artifact or Response({"text": "hello", "model": "Qwen/Qwen3-ASR-1.7B",
                                     "model_revision": "revision", "source_commit": "commit"})

    monkeypatch.setenv("GIGGY_API_KEY", "test-only-key")
    monkeypatch.setattr(api.requests, "Session", Session)
    monkeypatch.setattr(api.requests, "get", get)
    return calls


def test_final_text_provenance_and_credential_isolation(tmp_path, monkeypatch):
    calls = fake_api(monkeypatch)
    out = tmp_path / "output.json"
    ok, error = runner.transcribe_one(api.giggy, cfg(tmp_path), audio(tmp_path), out)
    assert ok and error is None
    result = json.loads(out.read_text())
    assert result["text"] == "hello"
    assert result["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert result["mode"] == "fast"
    assert calls[0][2]["params"]["language"] == "en"
    assert calls[0][2]["params"]["mode"] == "fast"
    assert calls[-1][0] == "artifact" and "headers" not in calls[-1][2]
    assert "signed" not in (tmp_path / "jobs/clip.json").read_text()


def test_resume_reuses_admitted_job(tmp_path, monkeypatch):
    calls = fake_api(monkeypatch, artifact=Response({}, 503))
    wav = audio(tmp_path)
    with pytest.raises(api.TranscribeError, match="transcript download HTTP 503"):
        api.giggy(wav, cfg(tmp_path))
    resumed_calls = fake_api(monkeypatch)
    assert api.giggy(wav, cfg(tmp_path)) == "hello"
    assert not any(c[0] == "post" for c in resumed_calls)
    assert sum(c[0] == "post" for c in calls) == 1
    state = json.loads((tmp_path / "jobs/clip.json").read_text())
    assert state["generation_uuid"] == GID


def test_ambiguous_upload_is_not_replayed(tmp_path, monkeypatch):
    calls = fake_api(monkeypatch, post_error=requests.Timeout("upload timed out"))
    wav = audio(tmp_path)
    with pytest.raises(requests.Timeout):
        api.giggy(wav, cfg(tmp_path))
    with pytest.raises(api.TranscribeError, match="ambiguous admission"):
        api.giggy(wav, cfg(tmp_path))
    assert sum(c[0] == "post" for c in calls) == 1


def test_short_clips_are_not_padded_or_submitted(tmp_path, monkeypatch):
    calls = fake_api(monkeypatch)
    with pytest.raises(api.TranscribeError, match="0.5-second minimum"):
        api.giggy(audio(tmp_path, seconds=0.2), cfg(tmp_path))
    assert calls == []


def test_artifact_network_error_does_not_expose_signed_url(tmp_path, monkeypatch):
    fake_api(monkeypatch)

    def fail(url, **kwargs):
        raise requests.ConnectionError("failed at " + url)

    monkeypatch.setattr(api.requests, "get", fail)
    with pytest.raises(api.TranscribeError) as caught:
        api.giggy(audio(tmp_path), cfg(tmp_path))
    assert "secret" not in str(caught.value)
    assert "ConnectionError" in str(caught.value)


def test_local_conversion_preserves_duration_and_source(tmp_path):
    wav = audio(tmp_path, sr=22050, channels=2)
    original = wav.read_bytes()
    data = api._giggy_audio(wav)
    info = sf.info(io.BytesIO(data))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert info.duration == pytest.approx(1, abs=1 / 16000)
    assert wav.read_bytes() == original


def test_selected_scoring_preserves_other_engines(tmp_path, monkeypatch):
    root = tmp_path
    (root / "data_test").mkdir()
    (root / "data_test/refs.json").write_text(json.dumps({"a": "hello"}))
    reports = root / "results/reports/test/giggy-free-stt-v1"
    reports.mkdir(parents=True)
    (reports / "a.json").write_text(json.dumps({"text": "hello"}))
    # Older AMI summaries predate scored/coveragePercent. Preserve them verbatim
    # and do not require the new fields merely to print the selected engine.
    old = {"engine": "other", "werPercent": 42, "ci95": [40, 44],
           "missing": 0, "utterances": 1}
    summary = root / "results/results_test.json"
    summary.write_text(json.dumps({"config": "test", "utterances": 1,
                                   "mode": "word_en", "results": [old]}))
    monkeypatch.setattr(score_ood, "ROOT", str(root))
    monkeypatch.setattr(sys, "argv", ["score_ood.py", "test", "--engine", "giggy-free-stt-v1"])
    score_ood.main()
    rows = json.loads(summary.read_text())["results"]
    assert next(r for r in rows if r["engine"] == "other") == old
    assert next(r for r in rows if r["engine"] == "giggy-free-stt-v1")["werPercent"] == 0


def test_controller_resume_preserves_completed_dataset(tmp_path, monkeypatch):
    (tmp_path / "data_done").mkdir()
    (tmp_path / "data_done/refs.json").write_text(json.dumps({"a": "hello"}))
    rdir = tmp_path / "results/reports/done" / controller.ENGINE
    rdir.mkdir(parents=True)
    (rdir / ".errors.json").write_text("{}")
    original_meta = json.dumps({"missing": 0, "wallSeconds": 5843})
    (rdir / "_meta.json").write_text(original_meta)
    (tmp_path / "results/results_done.json").write_text(json.dumps({
        "results": [{"engine": controller.ENGINE, "scored": 1, "missing": 0, "utterances": 1}]}))
    packed = tmp_path / "results/transcripts"
    packed.mkdir()
    with gzip.open(packed / f"done-{controller.ENGINE}.json.gz", "wt") as fh:
        json.dump({"a": "hello"}, fh)
    state_path = tmp_path / "results/reports/giggy-evaluation-state.json"
    state_path.write_text(json.dumps({"started_at": "original start", "completed_datasets": ["done"],
                                      "unexpected_failures": {"done": {"a": "previous timeout"}}}))
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "STATE", state_path)
    monkeypatch.setattr(controller, "DATASETS", ["done"])
    calls = []
    monkeypatch.setattr(controller, "run", lambda *args: calls.append(args))
    assert controller.verified_complete("done")
    controller.main()
    assert calls == []
    assert (rdir / "_meta.json").read_text() == original_meta
    state = json.loads(state_path.read_text())
    assert state["started_at"] == "original start" and state["status"] == "complete"
    assert not state.get("unexpected_failures")
    (rdir / ".errors.json").write_text(json.dumps({"b": "unexpected missing"}))
    assert not controller.verified_complete("done")
