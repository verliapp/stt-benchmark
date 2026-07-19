"""Optional: cross-implementation check with mlx-whisper (Apple MLX) instead of
WhisperKit. Useful to see how much of the WER is implementation/quantization
rather than the model itself. Numbers here will differ from the WhisperKit run.

Usage:
    python run_mlx.py tiny base small large-v3 large-v3-turbo
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRI = os.path.join(ROOT, "data", "LibriSpeech", "test-clean")
RESULTS = os.path.join(ROOT, "results", "mlx.json")


def load_references() -> dict[str, str]:
    refs = {}
    for tf in glob.glob(os.path.join(LIBRI, "**", "*.trans.txt"), recursive=True):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            refs[uid] = text
    return refs


def main():
    import mlx_whisper
    from normalize_wer import corpus_wer

    models = sys.argv[1:] or ["tiny", "base", "small"]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    refs = load_references()
    paths = {os.path.basename(p)[:-5]: p
             for p in glob.glob(os.path.join(LIBRI, "**", "*.flac"), recursive=True)}

    for m in models:
        repo = f"mlx-community/whisper-{m}"
        t0 = time.time()
        hyps = {}
        for i, uid in enumerate(refs, 1):
            r = mlx_whisper.transcribe(paths[uid], path_or_hf_repo=repo,
                                       language="en", fp16=True)
            hyps[uid] = r["text"]
            if i % 300 == 0:
                print(f"  [{m}] {i}/{len(refs)}  {time.time()-t0:.0f}s", flush=True)
        wer, _, _ = corpus_wer((refs[u], hyps[u]) for u in refs)
        res = {"engine": f"mlx-whisper-{m}", "split": "test-clean",
               "werPercent": round(wer, 2), "computeSeconds": round(time.time() - t0)}
        allr = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
        allr = [x for x in allr if x["engine"] != res["engine"]] + [res]
        json.dump(allr, open(RESULTS, "w"), indent=2)
        print(f"DONE {res['engine']}: WER {res['werPercent']}%", flush=True)


if __name__ == "__main__":
    main()
