"""Run a WhisperKit model over the full AMI set in chunks.

The WhisperKit CLI segfaults on the full 12k-file AMI folder (a scale bug in the
on-device tool; Apple/Parakeet handle the same audio fine). Processing in chunks
of 1500 into a shared report dir avoids the crash and, if one chunk still fails,
retries it in smaller sub-chunks to isolate and skip the offending clip.

    python chunk_whisper_ami.py whisper-small
    python chunk_whisper_ami.py whisper-large-v3-v20240930
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNK = 1500


def run_folder(cdir, model, rdir):
    return subprocess.run(
        ["swift", "run", "-c", "release", "whisperkit-cli", "transcribe",
         "--audio-folder", cdir, "--model", model, "--language", "en",
         "--chunking-strategy", "none", "--report", "--report-path", rdir],
        cwd=f"{ROOT}/WhisperKit", stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode


def link_dir(wavs, cdir):
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir)
    for w in wavs:
        os.symlink(os.path.abspath(w), f"{cdir}/{os.path.basename(w)}")


def process(wavs, model, rdir, depth=0):
    """Run a list of wavs; on failure, split and recurse to isolate bad clips."""
    cdir = f"/tmp/ami_chunk_{depth}_{os.getpid()}"
    link_dir(wavs, cdir)
    code = run_folder(cdir, model, rdir)
    if code == 0:
        return []
    if len(wavs) == 1:
        print(f"  BAD CLIP skipped: {os.path.basename(wavs[0])} (exit {code})", flush=True)
        return wavs
    mid = len(wavs) // 2
    return process(wavs[:mid], model, rdir, depth + 1) + \
        process(wavs[mid:], model, rdir, depth + 1)


def main():
    model = sys.argv[1]
    wavs = sorted(glob.glob(f"{ROOT}/data_ami/audio/*.wav"))
    rdir = f"{ROOT}/results/reports/ami/{model}"
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir)
    bad = []
    for i in range(0, len(wavs), CHUNK):
        chunk = wavs[i:i + CHUNK]
        bad += process(chunk, model, rdir)
        got = len([f for f in os.listdir(rdir) if f.endswith(".json")])
        print(f"chunk {i}-{i + len(chunk)}: total_reports={got}, bad_clips={len(bad)}",
              flush=True)
    print(f"DONE {model}: {len(wavs) - len(bad)}/{len(wavs)} transcribed, "
          f"{len(bad)} bad clips skipped", flush=True)


if __name__ == "__main__":
    main()
