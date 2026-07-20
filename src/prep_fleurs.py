"""Build a FLEURS test set from the google/fleurs Parquet files.

Reads audio bytes directly with pyarrow and soundfile, writes 16-bit wavs, and
uses the dataset's raw_transcription column as the reference text.

    python prep_fleurs.py ja_jp
    python prep_fleurs.py ja_jp --limit 20

Set HF_TOKEN in the environment for faster/authenticated downloads (optional).
Writes to data_fleurs_<lang>/audio/*.wav and data_fleurs_<lang>/refs.json.
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "google/fleurs"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python prep_fleurs.py <fleurs_lang_config> [--limit N]")
    lang = sys.argv[1]
    limit = None
    rest = sys.argv[2:]
    if rest:
        if len(rest) != 2 or rest[0] != "--limit":
            sys.exit("usage: python prep_fleurs.py <fleurs_lang_config> [--limit N]")
        limit = int(rest[1])
        if limit < 1:
            sys.exit("--limit must be at least 1")

    token = os.environ.get("HF_TOKEN")
    prefix = f"parquet-data/{lang}/test-"
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset", token=token)
                    if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        sys.exit(f"no test parquet shards found for config {lang!r} in {REPO}")

    paths = []
    for shard in shards:
        print(f"  downloading {shard} ...", flush=True)
        paths.append(hf_hub_download(REPO, shard, repo_type="dataset", token=token))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    keep = None
    if limit is not None and limit < rows:
        keep = set(np.linspace(0, rows - 1, limit).astype(int))

    out = os.path.join(ROOT, f"data_fleurs_{lang}")
    audio_dir = os.path.join(out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"{lang}: {len(shards)} test shard(s), {rows} rows", flush=True)

    refs = {}
    audio_s = 0.0
    gi = -1
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64,
                                     columns=["audio", "raw_transcription"]):
            d = batch.to_pydict()
            for au, tx in zip(d["audio"], d["raw_transcription"]):
                gi += 1
                if keep is not None and gi not in keep:
                    continue
                raw = au["bytes"] if isinstance(au, dict) else au
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                uid = f"fleurs_{lang}_{gi:06d}"
                sf.write(os.path.join(audio_dir, f"{uid}.wav"), arr, sr,
                         subtype="PCM_16")
                refs[uid] = tx
                audio_s += len(arr) / sr

    if keep is not None:
        print(f"  --limit {limit}: kept an even sample of {len(refs)}", flush=True)
    with open(os.path.join(out, "refs.json"), "w", encoding="utf-8") as fh:
        json.dump(refs, fh, ensure_ascii=False)
    print(f"wrote {len(refs)} wavs, {audio_s / 60:.1f} min audio -> "
          f"data_fleurs_{lang}/refs.json", flush=True)


if __name__ == "__main__":
    main()
