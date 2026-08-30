"""Run an on-device engine over the FLEURS sets and write reports for score_ood.py.

Parakeet TDT 0.6B v3 (multilingual, via parakeet-mlx) on the FLEURS languages it supports.
Apple SpeechAnalyzer is run separately through the sacli harness with --locale.

    python run_fleurs_ondevice.py parakeet-v3 fleurs_de_de fleurs_es_419
    python run_fleurs_ondevice.py parakeet-v3            # all supported configs

Writes results/reports/<config>/parakeet-v3/<uid>.json and _meta.json, then run
score_ood.py <config> to score.
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FLEURS configs whose language Parakeet v3 covers (25 European languages).
PARAKEET_V3_CONFIGS = ["fleurs_de_de", "fleurs_es_419", "fleurs_fr_fr", "fleurs_ru_ru"]
REPO = "mlx-community/parakeet-tdt-0.6b-v3"


def main():
    import soundfile as sf
    from parakeet_mlx import from_pretrained

    args = sys.argv[1:]
    if not args or args[0] != "parakeet-v3":
        sys.exit("usage: python run_fleurs_ondevice.py parakeet-v3 [fleurs_config ...]")
    configs = args[1:] or PARAKEET_V3_CONFIGS
    model = from_pretrained(REPO)

    for cfg in configs:
        refs = json.load(open(f"{ROOT}/data_{cfg}/refs.json"))
        out = f"{ROOT}/results/reports/{cfg}/parakeet-v3"
        os.makedirs(out, exist_ok=True)
        audio_seconds = 0.0
        wall = 0.0
        for uid in refs:
            path = f"{ROOT}/data_{cfg}/audio/{uid}.wav"
            info = sf.info(path)
            audio_seconds += info.frames / info.samplerate
            t = time.time()
            text = model.transcribe(path).text
            wall += time.time() - t
            json.dump({"text": text}, open(f"{out}/{uid}.json", "w"), ensure_ascii=False)
        json.dump({"audioSeconds": round(audio_seconds), "wallSeconds": round(wall),
                   "missing": 0, "realTimeFactor": round(wall / audio_seconds, 4)},
                  open(f"{out}/_meta.json", "w"), indent=2)
        print(f"{cfg}: {len(refs)} clips, {audio_seconds:.0f}s audio, {wall:.0f}s compute")


if __name__ == "__main__":
    main()
