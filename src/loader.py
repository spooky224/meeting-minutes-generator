import re
from pathlib import Path


def _strip_vtt(raw: str) -> str:
    """Remove WebVTT timestamps and metadata, keep only spoken text."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip WEBVTT header, cue identifiers, timestamps, and blank lines
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if re.match(r"^\d+$", line):          # cue number
            continue
        if re.match(r"[\d:,\. ]+-->", line):  # timestamp line
            continue
        lines.append(line)
    return "\n".join(lines)


def load_transcript(path: str) -> str:
    """
    Load a .txt or .vtt transcript file and return plain text.
    Raises FileNotFoundError if the path does not exist.
    Raises ValueError for unsupported file types.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    suffix = p.suffix.lower()
    raw = p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".txt":
        return raw.strip()
    elif suffix == ".vtt":
        return _strip_vtt(raw).strip()
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .txt or .vtt")