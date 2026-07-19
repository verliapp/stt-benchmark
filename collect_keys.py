"""Scan ~/dev/**/.env* for the provider keys and fill .env.providers.

Privacy: this NEVER prints a key value. For each canonical key it prints only the
source file, the source variable name it matched, and a short non-reversible
sha256 fingerprint (so you can confirm the right key landed without exposing it).
Values are read and written internally only.

    python collect_keys.py            # scan + fill .env.providers
    python collect_keys.py --dry-run  # report matches only, write nothing
"""
import hashlib
import os
import re
import sys

HOME = os.path.expanduser("~")
DEV = os.path.join(HOME, "dev")
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, ".env.providers.example")
TARGET = os.path.join(HERE, ".env.providers")

SKIP_DIRS = {"node_modules", ".venv", "venv", ".git", ".next", "dist", "build",
             ".pnpm-store", "coverage", "__pycache__", ".turbo", ".cache"}

# canonical name -> source variable names to match (case-insensitive)
ALIASES = {
    "DEEPGRAM_API_KEY": ["DEEPGRAM_API_KEY", "DEEPGRAM_KEY", "DEEPGRAM_TOKEN", "DG_API_KEY"],
    "ASSEMBLYAI_API_KEY": ["ASSEMBLYAI_API_KEY", "ASSEMBLY_AI_API_KEY", "ASSEMBLYAI_KEY", "AAI_API_KEY"],
    "SONIOX_API_KEY": ["SONIOX_API_KEY", "SONIOX_KEY"],
    "SPEECHMATICS_API_KEY": ["SPEECHMATICS_API_KEY", "SPEECHMATICS_KEY", "SM_API_KEY"],
    "GLADIA_API_KEY": ["GLADIA_API_KEY", "GLADIA_KEY"],
    "ELEVENLABS_API_KEY": ["ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "ELEVENLABS_KEY", "ELEVEN_LABS_API_KEY", "XI_API_KEY"],
    "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_SECRET_KEY", "OPEN_AI_API_KEY"],
    "GROQ_API_KEY": ["GROQ_API_KEY", "GROQ_KEY"],
    "LEMONFOX_API_KEY": ["LEMONFOX_API_KEY", "LEMONFOX_KEY", "LEMON_FOX_API_KEY"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_GENERATIVEAI_API_KEY"],
    "XAI_API_KEY": ["XAI_API_KEY", "GROK_API_KEY", "X_AI_API_KEY"],
    "FISH_API_KEY": ["FISH_API_KEY", "FISH_AUDIO_API_KEY"],
    "CARTESIA_API_KEY": ["CARTESIA_API_KEY", "CARTESIA_KEY"],
    "RESEMBLE_API_KEY": ["RESEMBLE_API_KEY", "RESEMBLE_API_TOKEN", "RESEMBLE_TOKEN"],
    "INWORLD_API_KEY": ["INWORLD_API_KEY", "INWORLD_KEY"],
    "REV_API_KEY": ["REV_API_KEY", "REVAI_API_KEY", "REV_AI_API_KEY", "REV_ACCESS_TOKEN"],
    "AWS_ACCESS_KEY_ID": ["AWS_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": ["AWS_SECRET_ACCESS_KEY"],
    "AWS_REGION": ["AWS_REGION", "AWS_DEFAULT_REGION"],
    "AZURE_SPEECH_KEY": ["AZURE_SPEECH_KEY", "AZURE_SPEECH_SUBSCRIPTION_KEY", "SPEECH_KEY", "AZURE_SPEECH_API_KEY"],
    "AZURE_SPEECH_REGION": ["AZURE_SPEECH_REGION", "SPEECH_REGION", "AZURE_SPEECH_LOCATION"],
    "GCP_PROJECT": ["GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT_ID"],
    "GOOGLE_APPLICATION_CREDENTIALS": ["GOOGLE_APPLICATION_CREDENTIALS"],
}
# names whose template default should NOT be overwritten by a found value
KEEP_DEFAULT = {"AWS_REGION", "AZURE_SPEECH_REGION"}

PLACEHOLDER = re.compile(r"(your|xxxx|changeme|placeholder|example|<|>|\.\.\.|sk-\.\.\.)", re.I)


def looks_real(val):
    v = val.strip().strip('"').strip("'")
    return bool(v) and not PLACEHOLDER.search(v)


def fingerprint(val):
    v = val.strip().strip('"').strip("'")
    return f"sha256:{hashlib.sha256(v.encode()).hexdigest()[:8]} len={len(v)}"


def clean(val):
    return val.strip().strip('"').strip("'")


def parse_env(path):
    out = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                line = line[len("export "):] if line.startswith("export ") else line
                k, v = line.split("=", 1)
                out[k.strip().upper()] = v.strip()
    except OSError:
        pass
    return out


def find_env_files():
    files = []
    for root, dirs, names in os.walk(DEV):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".venv"))
        # don't scan our own harness dir (target file, example, copied env)
        if os.path.abspath(root).startswith(HERE):
            continue
        for n in sorted(names):
            if n.startswith(".env") and not n.endswith((".example", ".sample", ".template")):
                files.append(os.path.join(root, n))
            elif n.endswith(".env"):
                files.append(os.path.join(root, n))
    return files


def main():
    dry = "--dry-run" in sys.argv
    if not os.path.isdir(DEV):
        sys.exit(f"no {DEV}")

    files = find_env_files()
    print(f"scanned {len(files)} .env* file(s) under {DEV}\n")

    # canonical -> list of (source_file, source_name, value)
    found = {c: [] for c in ALIASES}
    for path in files:
        env = parse_env(path)
        for canon, names in ALIASES.items():
            for name in names:
                if name.upper() in env and looks_real(env[name.upper()]):
                    found[canon].append((path, name, clean(env[name.upper()])))

    # base document: existing .env.providers if present, else the template
    base = TARGET if os.path.exists(TARGET) else TEMPLATE
    with open(base) as fh:
        lines = fh.readlines()

    chosen = {}  # canonical -> value to write
    print(f"{'KEY':32} RESULT")
    print("-" * 78)
    for canon in ALIASES:
        hits = found[canon]
        if not hits:
            print(f"{canon:32} not found")
            continue
        # distinct values across sources (compared by hash, never shown)
        by_val = {}
        for path, name, val in hits:
            by_val.setdefault(val, []).append((path, name))
        path, name, val = hits[0]
        chosen[canon] = val
        rel = os.path.relpath(path, HOME)
        note = ""
        if len(by_val) > 1:
            note = f"  (!) {len(by_val)} DIFFERENT values across sources; using first"
        extra = "" if len(hits) == 1 else f"  [+{len(hits)-1} more source(s)]"
        print(f"{canon:32} found in ~/{rel} as {name}  {fingerprint(val)}{extra}{note}")

    if dry:
        print("\n--dry-run: nothing written")
        return

    # fill the base document: only replace empty slots (keep region/location defaults)
    written = []
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.rstrip("\n"))
        if not m:
            continue
        key, cur = m.group(1), m.group(2).strip()
        if key in chosen and key not in KEEP_DEFAULT and not cur:
            v = chosen[key]
            v = f'"{v}"' if (" " in v) else v
            lines[i] = f"{key}={v}\n"
            written.append(key)

    with open(TARGET, "w") as fh:
        fh.writelines(lines)
    os.chmod(TARGET, 0o600)
    print(f"\nwrote {len(written)} key(s) into {TARGET} (chmod 600): {', '.join(written)}")
    missing = [c for c in ALIASES if c not in chosen]
    if missing:
        print(f"still missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
