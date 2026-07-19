# Speech-to-text accuracy benchmark

An independent word-error-rate benchmark of the major speech-to-text engines, commercial
and open, run on the same audio with the same scorer and 95% confidence intervals. It
covers eighteen cloud APIs and the open and on-device models, across four datasets that
range from clean read speech to hard conference-call audio, and adds cost and latency next
to accuracy so the numbers can be compared on more than one axis.

Every number here is reproducible with the scripts in this repo, and the raw per-utterance
transcripts our runs produced are committed under `results/transcripts/`, so anyone can
re-score them without re-running a single API call.

Two writeups build on this data:

- [Choosing a speech-to-text API](https://verli.app/blog/choosing-a-speech-to-text-api):
  the buying-decision guide (pricing, features, rate limits, deployment).
- [Benchmarking speech-to-text accuracy](https://verli.app/blog/speech-to-text-accuracy-benchmark):
  the accuracy results, with the full WER, cost, and latency tables.

It started as a reproduction of one on-device benchmark and grew from there. That origin,
Apple SpeechAnalyzer versus Whisper on LibriSpeech, is still here in full under
[Where this started](#where-this-started-the-on-device-reproduction); it is now one dataset
and a handful of engines inside a much larger comparison.

## What's in here

- **Engines.** Eighteen commercial cloud APIs (Deepgram, AssemblyAI, OpenAI, Google,
  Azure, Amazon, ElevenLabs, Speechmatics, Gladia, Soniox, Rev, Groq, xAI Grok, Fish,
  Cartesia, Inworld, Lemonfox, and Gemini), plus the open and on-device engines (the
  Whisper family through WhisperKit, NVIDIA Parakeet v2 and v3, and Apple SpeechAnalyzer).
  A nineteenth adapter, Resemble, exists in the code but was dropped from the results on
  cost.
- **Datasets.** LibriSpeech test-clean (clean read speech), Earnings-22 and SPGISpeech
  (two kinds of financial earnings-call audio), and AMI (meeting audio). Every engine gets
  the same files.
- **Metric.** Corpus WER (total edits over total reference words) with a 95% bootstrap
  confidence interval, plus a paired bootstrap (`paired_test.py`) for the "is A really
  better than B" question when two engines land close.
- **Cost and latency.** `pricing.json` plus `report.py` estimate dollars per audio-hour
  from the duration each engine processed; `measure_latency.py` times per-call latency in
  isolation.
- **Reproducible transcripts.** Every engine's output is committed gzipped under
  `results/transcripts/`, so the scoring can be re-run offline.

Accuracy depends on the set, and no single engine wins all of them: the engine tied for the
lowest error on clean read speech is the worst finished engine on hard earnings-call audio,
and the engine that wins the hard set is only mid-pack on the clean one. The two blog posts
above carry the full tables; this repo is the code and data behind them.

## Repository layout

All Python lives in `src/` and is run from the repo root (`./.venv/bin/python src/<script>.py`).

- `src/` — cloud adapters, the on-device runners, scoring, the paired bootstrap, cost and
  latency, and dataset prep.
- `results/` — committed outputs: per-engine transcripts under `transcripts/`, plus the
  WER, cost, and latency summaries (`results_*.json`, `report_*.json`, `latency.json`). The
  large per-clip `reports/` tree is regenerable and gitignored.
- `SpeechAnalyzerCLI/` — the macOS 26 Swift harness for Apple SpeechAnalyzer.
- `assets/` — the on-device charts.
- `pricing.json`, `requirements*.txt`, `setup.sh`, and `.env.providers.example` sit at the
  root.

## The cloud APIs

Each provider is one adapter in `cloud_adapters.py`, pinned to a flagship model
(env-overridable), driven by `run_cloud_engines.py` through the same
`results/reports/<config>/<engine>/` layout the on-device runners use, and scored by the
same `score_ood.py`. So a cloud API's WER is directly comparable to an open model's.

```bash
./.venv/bin/pip install -r requirements-cloud.txt   # only AWS + Google need SDKs
./.venv/bin/python src/prep_librispeech.py              # LibriSpeech test-clean into the OOD format

cp .env.providers.example .env.providers            # then fill in keys (gitignored)

# every provider with a key present, on a dataset (resumable; re-run to fill gaps / retry)
./.venv/bin/python src/run_cloud_engines.py librispeech
./.venv/bin/python src/run_cloud_engines.py earnings22
./.venv/bin/python src/score_ood.py librispeech
./.venv/bin/python src/score_ood.py earnings22

# cost + latency
./.venv/bin/python src/report.py earnings22             # merges WER with pricing.json into $/hr
./.venv/bin/python src/measure_latency.py --n 40        # per-call latency, one clip at a time

# subset / smoke test
./.venv/bin/python src/run_cloud_engines.py earnings22 deepgram openai --limit 50
```

The plain-key providers hit synchronous or upload-and-poll REST endpoints. The three cloud
giants need accounts: Amazon runs per-clip through the streaming API (Transcribe has no sync
pre-recorded REST), Google uses Speech-to-Text V2 `recognize`, Azure uses the
fast-transcription API. A few model ids are worth confirming on the first live call because
vendors rename them (ElevenLabs Scribe v1/v2, AssemblyAI universal/slam-1, Azure
fast/MAI-transcribe, Google chirp_2/chirp_3); each is an env override documented in
`.env.providers.example`.

Failures are counted, not hidden. `score_ood.py` scores each engine only on the clips it
returned and reports coverage, so an engine that refuses a file shows up as a coverage gap
rather than a silent accuracy hit.

## Where this started: the on-device reproduction

The repo began as an independent reproduction of Inscribe's benchmark
["Apple's New Speech API vs Whisper"](https://get-inscribe.com/blog/apple-speech-api-benchmark.html)
(2026-07-13), which measured Apple's new on-device `SpeechAnalyzer` API against the legacy
`SFSpeechRecognizer` and several Whisper models on LibriSpeech. The original is a good,
unusually transparent benchmark: it publishes its raw per-utterance transcripts. This repo
re-scores those transcripts and re-runs every engine on the real audio, including Apple
SpeechAnalyzer through a small macOS 26 Swift harness. It adds the engine the original left
out (NVIDIA Parakeet), puts 95% confidence intervals on every number, and extends the test
to two out-of-domain sets (Earnings-22 and AMI).

### TL;DR

Numbers are WER (word error rate): the share of words the engine got wrong, so lower is
better. "pp" means percentage points.

- **On LibriSpeech, Inscribe's headline holds only within noise.** Apple SpeechAnalyzer
  (1.82%) and Whisper large-v3 (1.82%) tie, and Parakeet v2 (1.69%) edges both by just
  0.01 to 0.25pp on a paired test. "Most accurate on-device engine" is inside the noise,
  not a clean win.
- **The original benchmark's big omission is Parakeet.** NVIDIA Parakeet TDT (untested by
  the original) ties for best on LibriSpeech and is the strongest on-device engine on the
  fair out-of-domain set (Earnings-22, ~11.2%). It belongs in any on-device comparison.
- **Out of domain, Apple is competitive but not ahead.** On Earnings-22, Apple (12.03%)
  lands 0.5 to 1.2pp behind both Parakeet models and level with on-device Whisper turbo
  (12.35%). Reference Whisper (the original PyTorch build) scores ~11.1%, on par with
  Parakeet, so Apple trails the leaders.
- **Two caveats.** LibriSpeech and AMI are both in Parakeet's training data; Earnings-22
  is the only set no engine trained on, so it is the fairest test. And Whisper's AMI
  numbers are a decoding artifact of running it clip-by-clip on very short segments, not
  its real accuracy (see "The AMI Whisper numbers").

Every WER below is corpus WER (total word errors / total reference words) with a 95%
bootstrap confidence interval. When a ranking is close, overlapping intervals are a weak
signal, so we also run a paired bootstrap (`paired_test.py`): it compares two engines on
the same clips and is the right test for "is A really better than B."

### LibriSpeech test-clean (2620 utterances, our independent runs)

| Engine | WER% | 95% CI | LibriSpeech in training? |
|--------|-----:|:------:|--------------------------|
| Parakeet TDT 0.6b v2 (English)      | 1.69 | 1.54-1.84 | **yes (in-domain)** |
| Apple SpeechAnalyzer                | 1.82 | 1.66-1.99 | undisclosed |
| Whisper large-v3                    | 1.82 | 1.67-1.98 | zero-shot |
| Whisper large-v3-turbo              | 1.93 | 1.77-2.09 | zero-shot |
| Parakeet TDT 0.6b v3 (multilingual) | 2.15 | 1.92-2.40 | **yes (in-domain)** |
| Whisper medium                      | 2.55 | 2.36-2.76 | zero-shot |
| Whisper small                       | 3.32 | 3.09-3.56 | zero-shot |
| Whisper base                        | 5.01 | 4.75-5.29 | zero-shot |
| Whisper tiny                        | 7.47 | 7.12-7.81 | zero-shot |

The top four (Parakeet v2, Apple, Whisper large-v3, turbo) sit inside one another's
marginal CIs. A paired bootstrap agrees Apple ties both Whisper large-v3 and turbo, and
separates Parakeet v2 from Apple by only 0.01 to 0.25pp, the sliver of an edge you would
expect from a model trained on LibriSpeech. Treat the top as a tie, not a ranking.

![LibriSpeech test-clean WER by engine, with 95% confidence intervals; the top four
engines overlap. Colored by training exposure.](assets/librispeech.png)

### Out of domain: does the ranking hold?

Two harder sets from the Open ASR Leaderboard bundle, run end to end with the same
normalizer and corpus WER. **Earnings-22 is in no engine's training data (the fairest
test; Apple discloses nothing). AMI is in Parakeet's training set**, so Parakeet has
home-field there.

| Engine | LibriSpeech | Earnings-22 (out-of-domain, fair) | AMI (Parakeet in-domain) |
|--------|------------:|:-------------------------------:|:------------------------:|
| Parakeet v2            | 1.69 | 11.23 (10.73-11.73) | 11.58 (11.23-11.93) |
| Parakeet v3            | 2.15 | 11.20 (10.71-11.74) | 11.43 (11.08-11.79) |
| Apple SpeechAnalyzer   | 1.82 | 12.03 (11.52-12.56) | 14.32 (13.96-14.69) |
| Whisper large-v3-turbo | 1.93 | 12.35 (11.87-12.86) | 21.68 (21.22-22.15) † |
| Whisper small          | 3.32 | 14.92 (14.27-15.57) | 22.88 (22.40-23.35) † |

† These AMI Whisper numbers are a clip-level Whisper decoding artifact (both WhisperKit
and reference mlx-whisper hallucinate on AMI's short clips), not a fair reading of Whisper.
See "The AMI Whisper numbers" below; the leaderboard's long-form figures are ~15.2 (turbo)
and ~14.9 (large-v3).

![Out-of-domain WER by engine with 95% confidence intervals: Earnings-22 (fair,
out-of-domain for all) and AMI (in Parakeet's training). Parakeet leads both; Apple is
mid-pack; the AMI Whisper bars are hatched to mark the clip-level Whisper decoding artifact.](assets/out-of-domain.png)

How to read this:

- **On Earnings-22, the fair set, Parakeet is the best on-device engine.** A paired test
  over the same clips puts Apple 0.5 to 1.2pp behind both Parakeet models, and level with
  on-device Whisper turbo. Reference Whisper scores ~11.1% (turbo), on par with Parakeet.
  So Apple sits behind Parakeet and reference Whisper here, not ahead. Parakeet leads even
  though Earnings-22 is not in its training data, so the edge is real, and it is exactly
  the engine the original benchmark missed. (Comparing the marginal CIs alone would
  miscall Apple vs Parakeet a tie; the paired test separates them.)
- **On AMI, Parakeet leads** (~11.5% vs Apple's 14.3%), but AMI is in Parakeet's training
  set, so part of that gap is home advantage, not pure skill. Ignore the raw AMI Whisper
  cells (see below).

### The AMI Whisper numbers (a clip-level decoding artifact, not the model's accuracy)

Our raw AMI WER for Whisper (21.7% turbo, 22.9% small) is far worse than the Open ASR
Leaderboard's reference Whisper (~15.2% turbo, ~14.9% large-v3), and we do not treat it
as Whisper's true AMI error. AMI is heavily segmented (median clip 1.52s, 38% under 1s),
and Whisper decoded clip-by-clip hallucinates on short segments: it transcribes the words
correctly and then invents more. One example (WhisperKit):

```
reference:  Say nice machine, it goes
WhisperKit: It's a nice machine that goes in. Yeah. I spent too much time. All right.
Apple:      It's a nice machine that goes.
```

This is not specific to WhisperKit. We ran the check (`mlx_ami_control.py`): reference-style
`mlx-whisper` (which applies Whisper's temperature-fallback and compression-ratio /
no-speech thresholds) on the same clips is, if anything, more erratic. It loops on some of
them (reference "Something else." came back as "get get get get get..."), so its sample WER
swings to ~73% with a very wide interval. Apple and Parakeet do not hallucinate on these
clips at all. So the AMI Whisper gap comes from decoding Whisper on very short isolated
segments, not from a WhisperKit bug and not from Whisper's underlying accuracy; the
leaderboard's ~15% uses a long-form decoding pipeline we do not replicate. We keep the raw
numbers in the table for transparency, treat the leaderboard figure as the fair
Whisper-on-AMI reading, and draw no Apple-vs-Whisper conclusion on AMI. On Earnings-22 the
effect is small, because its clips are longer: our WhisperKit turbo 12.35 sits just above
the leaderboard's 11.07.

### Training-data overlap (read before ranking)

LibriSpeech is not cleanly held out across these models, so LibriSpeech-only rankings
are not apples-to-apples:

- **Parakeet v2 and v3: trained on LibriSpeech and AMI.** Both NVIDIA model cards list
  LibriSpeech (960 hours) and AMI among their training datasets. So LibriSpeech and AMI
  are in-distribution for Parakeet; Earnings-22 is not listed on either card and is
  out-of-domain for it.
- **Whisper: zero-shot on all three.** OpenAI reports LibriSpeech as zero-shot with
  transcript-level dedup, and AMI/Earnings-22 are not in its training either. (680k h of
  web audio means perfect exclusion is unverifiable, but this is OpenAI's stated
  diligence.)
- **Apple SpeechAnalyzer: undisclosed.** Apple publishes nothing about training data,
  so whether LibriSpeech or AMI was seen is unknown.

In short: LibriSpeech and AMI favor the LibriSpeech/AMI-trained models (Parakeet).
Earnings-22 is the only set here not listed in any engine's training data, so it is the
fairest read of general capability (with the caveat that Apple discloses nothing).

### Sanity checks against published numbers

Our self-measured numbers line up with independently published figures, which is the main
check that the harness and scoring are sound.

| Engine / set | This repo | Published | Source |
|--------------|----------:|----------:|--------|
| Parakeet v2, LibriSpeech clean | 1.69 | 1.69 | NVIDIA model card |
| Parakeet v2, Earnings-22       | 11.23 | 11.15 | NVIDIA model card |
| Parakeet v2, AMI               | 11.58 | 11.16 | NVIDIA model card |
| Parakeet v3, Earnings-22       | 11.20 | 11.42 | NVIDIA model card |

Parakeet v2 on LibriSpeech reproduces NVIDIA's card to the hundredth (1.69 vs 1.69), and
Earnings-22 lands within the confidence interval (11.23 vs 11.15). AMI runs a bit higher
than the card (11.58 vs 11.16); the card scores AMI on the leaderboard's "cleaned"
references and we use the bundle's standard ones, which explains why ours runs slightly
higher.

Our small Whisper models also match OpenAI's published zero-shot LibriSpeech figures,
which checks the WhisperKit harness end to end:

| Model | This repo | OpenAI published |
|-------|----------:|-----------------:|
| Whisper tiny  | 7.47 | 7.6 |
| Whisper base  | 5.01 | 5.0 |
| Whisper small | 3.32 | 3.4 |

One caveat for every Whisper number here: we run Whisper through **WhisperKit CoreML** (the
on-device path Inscribe used), not the reference PyTorch build the Open ASR Leaderboard
scores, so our absolute Whisper numbers differ from the leaderboard's by implementation and
quantization. For context, the leaderboard (read 2026-07-15) reports Whisper
large-v3-turbo at 11.07 (Earnings-22) and 13.87 cleaned / 15.16 original (AMI), and
large-v3 at 11.59 and 13.63 / 14.86.

### Re-scoring Inscribe's published transcripts

Inscribe released every per-utterance transcript for both Apple engines. Recomputing
WER from those with OpenAI's normalizer (no audio needed, `rescore_published.py`):

| Engine | Split | This repo | Inscribe | Delta |
|--------|-------|----------:|---------:|------:|
| Apple SpeechAnalyzer         | test-clean | 1.83 | 2.12 | -0.29 |
| Apple SpeechAnalyzer         | test-other | 4.24 | 4.56 | -0.32 |
| SFSpeechRecognizer (legacy)  | test-clean | 8.65 | 9.02 | -0.37 |
| SFSpeechRecognizer (legacy)  | test-other | 15.80 | 16.25 | -0.45 |

The consistent negative delta is the normalizer, not a scoring error: Inscribe's
normalizer is slightly stricter than OpenAI's stock one (they disclose this), so every
engine shifts by a similar amount (0.29-0.45pp) and the ranking is unchanged. Better still,
our own independent Apple run (1.82% on test-clean above) matches this re-score of their
published transcripts (1.83%), which confirms their published transcripts are real engine
output, not only that their scoring was sound.

## Environment

- Apple M4 Pro, 64 GB RAM, macOS 26.5.1 (build 25F80).
- Xcode 26.6 / Swift 6.3.3 (for the WhisperKit CLI and the SpeechAnalyzer harness).
- WhisperKit pinned to v0.18.0; `parakeet-mlx` 0.5.2; `openai-whisper` 20250625 (used
  only for its `EnglishTextNormalizer`); `huggingface_hub` 1.23.0.
- Apple's speech assets are server-provided and cannot be version-pinned, so the Apple
  numbers are tied to whatever asset shipped as of 2026-07-15 and may drift.
- On-device runs dated 2026-07-15; cloud runs 2026-07-16 to 2026-07-18.

## Reproducing the on-device runs

```bash
./setup.sh                              # venv, LibriSpeech test-clean, WhisperKit CLI (pinned)

# 1. Re-score Inscribe's published Apple transcripts (no audio, ~1 min)
./.venv/bin/python src/rescore_published.py

# 2. Independently run Whisper on the real audio (WhisperKit CoreML)
./.venv/bin/python src/run_whisperkit.py tiny base small large-v3-v20240930 large-v3

# 3. Parakeet, the engine the original left out (on-device via MLX)
./.venv/bin/python src/run_parakeet.py v2 v3

# 4. Apple SpeechAnalyzer on the real audio (macOS 26+, builds a small Swift harness)
(cd SpeechAnalyzerCLI && swift build -c release)
./.venv/bin/python src/run_apple.py

# 5. Out of domain: full test sets, every engine, then score with CIs
export HF_TOKEN=...                     # optional, for faster Hugging Face downloads
./.venv/bin/python src/prep_ood_dataset.py earnings22
./.venv/bin/python src/run_ood_engines.py earnings22
./.venv/bin/python src/score_ood.py earnings22

# AMI is the same, but drop sub-0.15s clips (--min-dur) and run Whisper in chunks,
# because the WhisperKit CLI segfaults on the full folder. Apple and Parakeet are normal.
./.venv/bin/python src/prep_ood_dataset.py ami --min-dur 0.15
./.venv/bin/python src/run_ood_engines.py ami apple parakeet-v2 parakeet-v3
./.venv/bin/python src/chunk_whisper_ami.py whisper-small
./.venv/bin/python src/chunk_whisper_ami.py whisper-large-v3-v20240930
./.venv/bin/python src/score_ood.py ami

# significance of close rankings, and the reference-Whisper control on AMI:
./.venv/bin/python src/paired_test.py
./.venv/bin/python src/mlx_ami_control.py
```

Step 2 accepts `medium` too. `run_mlx.py` is an optional cross-implementation check of
Whisper via `mlx-whisper`. `make_charts.py` regenerates the charts above (needs matplotlib).

## Methodology

- **Corpus:** LibriSpeech `test-clean` (2620 utterances, OpenSLR). Out-of-domain sets
  are the full Earnings-22 and AMI test splits from the Open ASR Leaderboard bundle
  `hf-audio/esb-datasets-test-only-sorted`, plus a 1200-clip sample of SPGISpeech.
- **Metric:** corpus WER (total edits / total reference words), not the mean of
  per-utterance WERs. Empty or missing output is handled as a coverage gap, reported per
  engine, rather than silently counted as accuracy. Each number carries a 95% bootstrap
  CI over utterances (1000 resamples, fixed seed).
- **Engine (on-device):** WhisperKit CoreML for Whisper, same on-device path as the
  original. Parakeet via `parakeet-mlx`. Apple SpeechAnalyzer via `SpeechAnalyzerCLI/`
  (macOS 26 Speech framework, fully on-device).
- **Normalizer:** OpenAI's `EnglishTextNormalizer`, applied to every engine equally, so
  it shifts all absolute numbers together and leaves rankings unchanged. We use it
  because OpenAI published Whisper's LibriSpeech WER with it.
- **Decoding:** `--language en`, greedy, no VAD chunking (utterances are short clips).
  Note the WhisperKit CLI at these defaults does not apply the temperature-fallback and
  compression-ratio / no-speech thresholds that reference `openai-whisper` uses to
  suppress hallucination, which is the mechanism behind the AMI Whisper artifact above.
- **AMI note:** the WhisperKit CLI segfaults on the full 12k-file AMI folder (a scale
  limit in the tool; Apple and Parakeet process the same folder fine), so AMI Whisper is
  run in chunks of 1500 via `chunk_whisper_ami.py`. Separately, we drop 101 clips shorter
  than 0.15s (backchannels like "Yeah.", single letters; 105 reference words, 0.117% of
  the total) for every engine so all engines are scored on the identical set.
- **Cost:** estimated from the audio duration each engine processed times its published
  per-hour rate in `pricing.json`, so it is a modeled figure, not an invoice.
- **Speed:** Apple and Parakeet run single-stream at roughly 0.02 real-time factor
  (about 50x faster than audio) on the M4 Pro. Cloud latency is measured separately by
  `measure_latency.py`, one call at a time, since the bulk runs use many workers and that
  distorts per-call timing. We do not rank on speed across machines.

## Requirements

- macOS on Apple Silicon for the on-device engines (WhisperKit CoreML, Parakeet MLX).
  The cloud adapters, `rescore_published.py`, and scoring run anywhere with Python.
- **macOS 26+** for `run_apple.py` (the SpeechAnalyzer API is macOS 26 only).
- Xcode / Swift toolchain for the on-device Whisper and Apple harnesses.
- Python 3.10+. Note `openai-whisper` pulls in torch (~2 GB); it is used only for the
  text normalizer.
- ~4 GB disk for LibriSpeech plus the CoreML models, more if you cache the OOD audio.

## Credits

- Original on-device benchmark and published transcripts:
  [Inscribe](https://get-inscribe.com/blog/apple-speech-api-benchmark.html), which
  inspired this repo.
- [LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015).
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) by Argmax.
- [Whisper](https://github.com/openai/whisper) and its English normalizer by OpenAI.
- [Parakeet](https://huggingface.co/nvidia) by NVIDIA, run via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).
- Out-of-domain sets from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) ESB bundle (reference WER figures read 2026-07-15).

MIT licensed. Not affiliated with Inscribe, Apple, Argmax, NVIDIA, OpenAI, or any of the
cloud providers benchmarked here; all product names belong to their owners.
