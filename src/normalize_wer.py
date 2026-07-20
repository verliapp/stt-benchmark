"""Shared scoring helpers: OpenAI's English normalizer + corpus WER.

We deliberately use OpenAI's own EnglishTextNormalizer (from the openai-whisper
package) because OpenAI published Whisper's LibriSpeech WER with it, which makes
our Whisper numbers directly comparable to theirs. Corpus WER = total word edits
across all utterances / total reference words (not the mean of per-utterance WERs),
matching the standard LibriSpeech convention.
"""
from whisper.normalizers import EnglishTextNormalizer

_norm = EnglishTextNormalizer()


def tokens(text: str) -> list[str]:
    return _norm(text or "").split()


# Residual hesitation fillers (the OpenAI normalizer already drops most "uh"/"um",
# this catches the rest). Multi-word discourse markers like "you know" are NOT
# removed: those are real words, so the normalized number still keeps some of the
# penalty a cleaner earns for dropping them.
_FILLERS = {"uh", "um", "uhm", "er", "erm", "ah", "eh", "hmm", "hm", "mm",
            "mhm", "mmm", "huh", "uhhuh"}


def tokens_no_disfluency(text: str) -> list[str]:
    """Tokens with residual fillers removed and immediate word repetitions
    collapsed (a stutter or false start like "in in" or "I I I" becomes one).
    Applied identically to reference and hypothesis, so it measures content
    accuracy while not penalizing an engine that cleans disfluencies."""
    out = []
    for t in tokens(text):
        if t in _FILLERS:
            continue
        if out and out[-1] == t:
            continue
        out.append(t)
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def corpus_wer(pairs) -> tuple[float, int, int]:
    """pairs: iterable of (reference_text, hypothesis_text).

    Returns (wer_percent, total_errors, total_reference_words).
    An empty hypothesis scores as all-deletions (100% WER for that utterance),
    matching the benchmark's "failures counted, not hidden" rule.
    """
    _wer, errors, ref_words, _counts = score_pairs(pairs)
    return (_wer, errors, ref_words)


def score_pairs(pairs):
    """Like corpus_wer but also returns per-utterance (errors, ref_words) counts,
    which the bootstrap uses to put a confidence interval on the corpus WER.

    Returns (wer_percent, total_errors, total_reference_words, counts).
    """
    errors = ref_words = 0
    counts = []
    for ref, hyp in pairs:
        r, h = tokens(ref), tokens(hyp)
        e = edit_distance(r, h)
        counts.append((e, len(r)))
        errors += e
        ref_words += len(r)
    wer = 100.0 * errors / ref_words if ref_words else 0.0
    return (wer, errors, ref_words, counts)


def bootstrap_ci(counts, resamples: int = 1000, seed: int = 0):
    """95% confidence interval for corpus WER by resampling utterances with
    replacement. `counts` is the per-utterance (errors, ref_words) list from
    score_pairs. Deterministic given the seed. Returns (low, high) in percent.

    The interval says how much the corpus WER would wobble if we had drawn a
    different sample of utterances of the same size, so ranks whose intervals
    overlap should be read as ties rather than a real ordering.
    """
    import numpy as np

    if not counts:
        return (0.0, 0.0)
    e = np.array([c[0] for c in counts], dtype=float)
    w = np.array([c[1] for c in counts], dtype=float)
    n = len(counts)
    rng = np.random.default_rng(seed)
    wers = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        denom = w[idx].sum()
        wers[i] = 100.0 * e[idx].sum() / denom if denom else 0.0
    lo, hi = np.percentile(wers, [2.5, 97.5])
    return (round(float(lo), 2), round(float(hi), 2))


# Multilingual scoring is kept separate so the original English path above stays
# identical. Existing callers of tokens(), score_pairs(), and corpus_wer() retain
# the exact English normalization and scoring behavior.
from whisper.normalizers import BasicTextNormalizer

_basic_norm = BasicTextNormalizer()
_MODES = {"word_en", "word_basic", "cer"}


def mode_tokens(text: str, mode: str) -> list[str]:
    if mode == "word_en":
        return tokens(text)
    if mode == "word_basic":
        return _basic_norm(text or "").split()
    if mode == "cer":
        return list("".join(_basic_norm(text or "").split()))
    raise ValueError(f"unknown scoring mode {mode!r}; choose from {sorted(_MODES)}")


def score_pairs_mode(pairs, mode: str):
    if mode == "word_en":
        return score_pairs(pairs)

    errors = ref_words = 0
    counts = []
    for ref, hyp in pairs:
        r, h = mode_tokens(ref, mode), mode_tokens(hyp, mode)
        e = edit_distance(r, h)
        counts.append((e, len(r)))
        errors += e
        ref_words += len(r)
    wer = 100.0 * errors / ref_words if ref_words else 0.0
    return (wer, errors, ref_words, counts)
