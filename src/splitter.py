from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

SHORT_THRESHOLD  = 8_000    # chars — fits in one call comfortably
MEDIUM_CHUNK     = 6_000    # chars per chunk for medium transcripts
LONG_CHUNK_TOKS  = 1_200    # tokens per chunk for long transcripts (safe for free-tier)
OVERLAP_TOKS     = 80       # token overlap between consecutive chunks


def split_transcript(text: str) -> List[str]:
    """
    Adaptively split a transcript into chunks sized for the LLM.

    Returns a list with a single element for short transcripts,
    or multiple chunks for longer ones.
    """
    length = len(text)

    # ── SHORT: send as-is ────────────────────────────────────────────────────
    if length <= SHORT_THRESHOLD:
        print(f"   [splitter] SHORT transcript ({length} chars) — single chunk")
        return [text]

    # ── MEDIUM: paragraph-aware character splitting ───────────────────────────
    if length <= 40_000:
        print(f"   [splitter] MEDIUM transcript ({length} chars) — paragraph split")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=MEDIUM_CHUNK,
            chunk_overlap=300,
        )
        chunks = splitter.split_text(text)
        print(f"   [splitter] → {len(chunks)} chunks")
        return chunks

    # ── LONG: strict token budget for large transcripts ───────────────────────
    print(f"   [splitter] LONG transcript ({length} chars) — token split")
    splitter = TokenTextSplitter(
        chunk_size=LONG_CHUNK_TOKS,
        chunk_overlap=OVERLAP_TOKS,
        encoding_name="gpt2",
    )
    chunks = splitter.split_text(text)
    print(f"   [splitter] → {len(chunks)} chunks of ~{LONG_CHUNK_TOKS} tokens each")
    return chunks