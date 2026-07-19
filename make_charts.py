"""Render the benchmark charts embedded in the README (needs matplotlib).

Reads the committed result JSONs and writes PNGs to assets/. Colors encode
training exposure (the contamination caveat): in-domain, zero-shot, undisclosed.

    ./.venv/bin/pip install matplotlib
    ./.venv/bin/python make_charts.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

INK = "#1f2328"
COLOR = {"in-domain": "#e08a3c", "zero-shot": "#3f8f5b", "undisclosed": "#8a8a94"}
LABEL = {"in-domain": "in Parakeet training (in-domain)",
         "zero-shot": "zero-shot / held out",
         "undisclosed": "training data undisclosed (Apple)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#b8bcc4", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e7e9ee", linewidth=1)
    ax.yaxis.grid(False)


def bars(ax, names, wers, cis, cats):
    y = range(len(names))
    lo = [w - c[0] for w, c in zip(wers, cis)]
    hi = [c[1] - w for w, c in zip(wers, cis)]
    ax.barh(y, wers, color=[COLOR[c] for c in cats], height=0.68,
            xerr=[lo, hi], error_kw=dict(ecolor="#5b5f66", elinewidth=1.1, capsize=3))
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    for i, (w, c) in enumerate(zip(wers, cis)):
        ax.text(c[1] + max(wers) * 0.015, i, f"{w:.2f}", va="center", ha="left",
                fontsize=10, color=INK)
    style(ax)


def librispeech():
    wk = {r["engine"]: r for r in json.load(open(f"{ROOT}/results/whisperkit.json"))}
    pk = {r["engine"]: r for r in json.load(open(f"{ROOT}/results/parakeet.json"))}
    ap = json.load(open(f"{ROOT}/results/apple.json"))
    rows = [
        ("Parakeet v2", pk["parakeet-tdt-0.6b-v2"], "in-domain"),
        ("Apple SpeechAnalyzer", ap, "undisclosed"),
        ("Whisper large-v3", wk["whisper-large-v3"], "zero-shot"),
        ("Whisper large-v3-turbo", wk["whisper-large-v3-v20240930"], "zero-shot"),
        ("Parakeet v3", pk["parakeet-tdt-0.6b-v3"], "in-domain"),
        ("Whisper medium", wk["whisper-medium"], "zero-shot"),
        ("Whisper small", wk["whisper-small"], "zero-shot"),
        ("Whisper base", wk["whisper-base"], "zero-shot"),
        ("Whisper tiny", wk["whisper-tiny"], "zero-shot"),
    ]
    names = [r[0] for r in rows]
    wers = [r[1]["werPercent"] for r in rows]
    cis = [r[1]["ci95"] for r in rows]
    cats = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    bars(ax, names, wers, cis, cats)
    ax.set_xlim(0, max(wers) * 1.12)
    ax.set_xlabel("Word error rate (%), lower is better. Bars show 95% CI.")
    ax.set_title("LibriSpeech test-clean: top four are a statistical tie",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    seen = list(dict.fromkeys(cats))
    ax.legend(handles=[Patch(color=COLOR[c], label=LABEL[c]) for c in
                       ["in-domain", "zero-shot", "undisclosed"] if c in seen],
              loc="upper right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/librispeech.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def out_of_domain():
    e = {r["engine"]: r for r in json.load(open(f"{ROOT}/results_earnings22.json"))["results"]}
    a = {r["engine"]: r for r in json.load(open(f"{ROOT}/results_ami.json"))["results"]}
    # order by the fair set (Earnings-22)
    order = [("Parakeet v2", "parakeet-v2", "in-domain"),
             ("Parakeet v3", "parakeet-v3", "in-domain"),
             ("Apple SpeechAnalyzer", "apple-speechanalyzer", "undisclosed"),
             ("Whisper large-v3-turbo", "whisper-large-v3-v20240930", "zero-shot"),
             ("Whisper small", "whisper-small", "zero-shot")]
    # AMI is in-domain for Parakeet; Whisper AMI is a WhisperKit artifact (hatched).
    ami_artifact = {"whisper-large-v3-v20240930", "whisper-small"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, data, title, note in [
        (ax1, e, "Earnings-22: out-of-domain for all (fair)", None),
        (ax2, a, "AMI: in Parakeet's training set", "hatched = clip-level Whisper artifact"),
    ]:
        names = [o[0] for o in order]
        wers = [data[o[1]]["werPercent"] for o in order]
        cis = [data[o[1]]["ci95"] for o in order]
        cats = [o[2] for o in order]
        y = range(len(names))
        lo = [w - c[0] for w, c in zip(wers, cis)]
        hi = [c[1] - w for w, c in zip(wers, cis)]
        hatches = [("///" if (ax is ax2 and o[1] in ami_artifact) else None) for o in order]
        b = ax.barh(y, wers, color=[COLOR[c] for c in cats], height=0.66,
                    xerr=[lo, hi], error_kw=dict(ecolor="#5b5f66", elinewidth=1.1, capsize=3))
        for bar, h in zip(b, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("white")
        ax.set_yticks(list(y))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        for i, (w, c, o) in enumerate(zip(wers, cis, order)):
            tag = " †" if (ax is ax2 and o[1] in ami_artifact) else ""
            ax.text(c[1] + 0.5, i, f"{w:.1f}{tag}", va="center", ha="left",
                    fontsize=9.5, color=INK)
        ax.set_xlim(0, 26)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="center", pad=8)
        ax.set_xlabel("WER (%), lower is better. Bars show 95% CI.")
        style(ax)
        if note:
            ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, color="#7a7a84", style="italic")

    fig.suptitle("Out of domain: Parakeet leads the on-device engines; Apple mid-pack",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02,
             "† Clip-level Whisper decoding hallucinates on AMI's short segments (both "
             "WhisperKit and mlx-whisper); leaderboard long-form ~15.2 (turbo) / ~14.9 "
             "(large-v3). See README.",
             fontsize=8.5, color="#7a7a84")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(f"{ASSETS}/out-of-domain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    librispeech()
    out_of_domain()
    print("wrote assets/librispeech.png and assets/out-of-domain.png")
