#!/usr/bin/env bash
# Runs Apple SpeechAnalyzer (sacli) over the FLEURS sets whose language Apple supports.
# Usage: bash src/run_fleurs_apple.sh [fleurs_config ...]   (then python src/score_ood.py <config>)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SACLI="$ROOT/SpeechAnalyzerCLI/.build/release/sacli"
declare -A LOCALES=(
  [fleurs_de_de]=de-DE [fleurs_es_419]=es-MX [fleurs_fr_fr]=fr-FR
  [fleurs_ja_jp]=ja-JP [fleurs_ko_kr]=ko-KR [fleurs_cmn_hans_cn]=zh-CN
)
CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then CONFIGS=("${!LOCALES[@]}"); fi
for cfg in "${CONFIGS[@]}"; do
  out="$ROOT/results/reports/$cfg/apple-speechanalyzer"
  mkdir -p "$out"
  start=$(date +%s)
  SACLI_LOCALE="${LOCALES[$cfg]}" "$SACLI" --audio-folder "$ROOT/data_$cfg/audio" --out "$out" > /dev/null
  wall=$(( $(date +%s) - start ))
  audio=$("$ROOT/.venv/bin/python" -c "import glob,soundfile as sf,sys;print(round(sum(sf.info(p).frames/sf.info(p).samplerate for p in glob.glob(sys.argv[1]+'/*.wav'))))" "$ROOT/data_$cfg/audio")
  printf '{"audioSeconds": %s, "wallSeconds": %s, "missing": 0}\n' "$audio" "$wall" > "$out/_meta.json"
  echo "$cfg (${LOCALES[$cfg]}): ${audio}s audio, ${wall}s wall"
done
