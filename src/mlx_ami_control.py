"""Control: run reference-style Whisper (mlx-whisper, which applies Whisper's
temperature-fallback and compression-ratio / log-probability suppression) on the same
AMI clips WhisperKit ran, to test whether the AMI hallucination is WhisperKit-specific.

mlx-whisper mirrors openai-whisper's decoding, unlike the WhisperKit CLI at defaults.
If mlx-whisper lands near the leaderboard (~15%) while WhisperKit is ~21%, the inflation
is a WhisperKit CLI artifact, not a property of the model or our segmentation.

    python mlx_ami_control.py [N]     # N = sample size, default 1500 evenly spaced
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "mlx-community/whisper-large-v3-turbo"


def main():
    import mlx_whisper
    from normalize_wer import score_pairs, bootstrap_ci

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    refs = json.load(open(f"{ROOT}/data_ami/refs.json"))
    uids = sorted(refs)
    idx = sorted(set(np.linspace(0, len(uids) - 1, min(n, len(uids))).astype(int)))
    sample = [uids[i] for i in idx]

    wk = json.load(gzip.open(f"{ROOT}/results/transcripts/ami-whisper-large-v3-v20240930.json.gz", "rt"))

    mlx_hyp = {}
    for j, u in enumerate(sample):
        r = mlx_whisper.transcribe(f"{ROOT}/data_ami/audio/{u}.wav",
                                   path_or_hf_repo=MODEL, language="en", verbose=False)
        mlx_hyp[u] = r["text"]
        if (j + 1) % 200 == 0:
            print(f"  mlx {j + 1}/{len(sample)}", flush=True)

    out = {"model": MODEL, "sample": len(sample), "engines": {}}
    for name, hyps in [("mlx-whisper-large-v3-turbo", mlx_hyp),
                       ("whisperkit-large-v3-turbo", {u: wk.get(u, "") for u in sample})]:
        wer, _, _, counts = score_pairs([(refs[u], hyps[u]) for u in sample])
        lo, hi = bootstrap_ci(counts)
        out["engines"][name] = {"werPercent": round(wer, 2), "ci95": [lo, hi]}
        print(f"{name}: {wer:.2f}%  (95% CI {lo}-{hi})", flush=True)
    json.dump(out, open(f"{ROOT}/results/ami_whisper_control.json", "w"), indent=2)
    print("wrote results/ami_whisper_control.json", flush=True)


if __name__ == "__main__":
    main()
