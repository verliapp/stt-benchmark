#!/usr/bin/env bash
# One-time setup: Python deps, LibriSpeech test-clean, and the WhisperKit CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# WhisperKit is pinned so a clone next month builds the same CLI we benchmarked.
WHISPERKIT_REF="${WHISPERKIT_REF:-v0.18.0}"
# LibriSpeech test-clean checksum (published by OpenSLR).
TEST_CLEAN_MD5="32fa31d27d2e1cad72775fee3f4849a9"

echo "==> Python venv + deps (installs torch via openai-whisper, ~2 GB)"
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
  || { echo "need Python 3.10+, found $PYV"; exit 1; }
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

echo "==> LibriSpeech test-clean (~346 MB)"
mkdir -p data && cd data
if [ ! -d LibriSpeech/test-clean ]; then
  curl -L -o test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
  got="$(md5 -q test-clean.tar.gz 2>/dev/null || md5sum test-clean.tar.gz | awk '{print $1}')"
  [ "$got" = "$TEST_CLEAN_MD5" ] || { echo "checksum mismatch: $got != $TEST_CLEAN_MD5"; exit 1; }
  tar xzf test-clean.tar.gz && rm test-clean.tar.gz
fi

echo "==> flatten audio into one folder (absolute symlinks)"
mkdir -p audio_flat
find "$ROOT/data/LibriSpeech/test-clean" -name '*.flac' | while read -r f; do
  ln -sf "$f" "audio_flat/$(basename "$f")"
done
echo "    $(ls audio_flat | wc -l | tr -d ' ') files linked"
cd "$ROOT"

echo "==> WhisperKit CLI ($WHISPERKIT_REF; only for the audio reproduction, needs Xcode/Swift)"
if command -v swift >/dev/null 2>&1; then
  if [ ! -d WhisperKit ]; then
    git clone --depth 1 --branch "$WHISPERKIT_REF" https://github.com/argmaxinc/WhisperKit.git
  fi
  ( cd WhisperKit && swift build -c release --product whisperkit-cli )
else
  echo "    swift not found; skip. rescore_published.py still works without it."
fi

echo "==> done. Try:  ./.venv/bin/python rescore_published.py"
