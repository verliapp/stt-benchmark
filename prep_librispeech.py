"""Prepare LibriSpeech test-clean into the same OOD harness format as the other sets.

Extracts ../speechbench/test-clean.tar.gz (if data_librispeech is not already built),
converts every .flac utterance to a 16-bit wav, and writes references from the
.trans.txt files. Output mirrors prep_ood_dataset.py:

    data_librispeech/audio/<uid>.wav      # uid is LibriSpeech's own utterance id
    data_librispeech/refs.json            # {uid: reference_text}

    python prep_librispeech.py

Same downstream flow as the OOD sets: run engines, then score_ood.py librispeech.
"""
import json
import os
import tarfile

import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
TAR = os.path.join(ROOT, "..", "speechbench", "test-clean.tar.gz")
OUT = os.path.join(ROOT, "data_librispeech")
AUDIO = os.path.join(OUT, "audio")
EXTRACT = os.path.join(OUT, "_src")


def main():
    os.makedirs(AUDIO, exist_ok=True)
    root = os.path.join(EXTRACT, "LibriSpeech", "test-clean")
    if not os.path.isdir(root):
        assert os.path.exists(TAR), f"missing {TAR}"
        print(f"extracting {TAR} ...", flush=True)
        with tarfile.open(TAR) as tf:
            tf.extractall(EXTRACT)

    refs = {}
    audio_s = 0.0
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            if not os.path.isdir(chap_dir):
                continue
            trans = os.path.join(chap_dir, f"{spk}-{chap}.trans.txt")
            with open(trans) as fh:
                for line in fh:
                    uid, text = line.strip().split(" ", 1)
                    flac = os.path.join(chap_dir, f"{uid}.flac")
                    arr, sr = sf.read(flac, dtype="float32")
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    sf.write(f"{AUDIO}/{uid}.wav", arr, sr, subtype="PCM_16")
                    refs[uid] = text
                    audio_s += len(arr) / sr

    json.dump(refs, open(f"{OUT}/refs.json", "w"))
    print(f"wrote {len(refs)} wavs, ~{audio_s / 60:.1f} min audio -> {OUT}/refs.json",
          flush=True)


if __name__ == "__main__":
    main()
