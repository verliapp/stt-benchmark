"""Build a FULL out-of-domain test set from the Open ASR Leaderboard bundle.

Downloads every parquet shard of a config in hf-audio/esb-datasets-test-only-sorted,
decodes all audio bytes with soundfile (no torchcodec needed), writes 16-bit wavs,
and saves references. Running the whole test set (not a sample) is what makes our
numbers directly comparable to the published Open ASR Leaderboard figures.

    python prep_ood_dataset.py earnings22        # full test set
    python prep_ood_dataset.py ami
    python prep_ood_dataset.py earnings22 --limit 300   # smoke test only

The shards in this bundle are sorted by duration, so a single shard is one
duration band, not a representative sample. That is exactly why we decode all of
them. `--limit` samples evenly across the whole set and is for quick checks only.

Set HF_TOKEN in the environment for faster/authenticated downloads (optional).
Writes to data_<config>/audio/*.wav and data_<config>/refs.json.
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "hf-audio/esb-datasets-test-only-sorted"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python prep_ood_dataset.py <config> [--limit N] [text_col]")
    cfg = sys.argv[1]
    limit = None
    min_dur = 0.0  # AMI uses --min-dur 0.15 (WhisperKit CLI can't handle sub-0.15s clips)
    text_col = "text"
    rest = sys.argv[2:]
    i = 0
    while i < len(rest):
        if rest[i] == "--limit":
            limit = int(rest[i + 1])
            i += 2
        elif rest[i] == "--min-dur":
            min_dur = float(rest[i + 1])
            i += 2
        else:
            text_col = rest[i]
            i += 1

    token = os.environ.get("HF_TOKEN")
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset", token=token)
                    if f.startswith(f"{cfg}/") and f.endswith(".parquet"))
    if not shards:
        sys.exit(f"no parquet shards found for config {cfg!r} in {REPO}")

    out = f"data_{cfg}"
    audio_dir = f"{out}/audio"
    os.makedirs(audio_dir, exist_ok=True)
    print(f"{cfg}: {len(shards)} shard(s)", flush=True)

    refs = {}
    audio_s = 0.0
    gi = -1
    for shard in shards:
        print(f"  downloading {shard} ...", flush=True)
        path = hf_hub_download(REPO, shard, repo_type="dataset", token=token)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["audio", text_col]):
            d = batch.to_pydict()
            for au, tx in zip(d["audio"], d[text_col]):
                gi += 1
                raw = au["bytes"] if isinstance(au, dict) else au
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                if min_dur and len(arr) / sr < min_dur:
                    continue  # too short for the WhisperKit CLI; dropped for all engines
                uid = f"{cfg}_{gi:06d}"
                sf.write(f"{audio_dir}/{uid}.wav", arr, sr, subtype="PCM_16")
                refs[uid] = tx
                audio_s += len(arr) / sr

    if limit is not None and limit < len(refs):
        keep = {list(refs)[j] for j in np.linspace(0, len(refs) - 1, limit).astype(int)}
        for uid in list(refs):
            if uid not in keep:
                os.remove(f"{audio_dir}/{uid}.wav")
                del refs[uid]
        print(f"  --limit {limit}: kept an even sample of {len(refs)}", flush=True)

    json.dump(refs, open(f"{out}/refs.json", "w"))
    print(f"wrote {len(refs)} wavs, ~{audio_s / 60:.1f} min audio -> {out}/refs.json",
          flush=True)


if __name__ == "__main__":
    main()
