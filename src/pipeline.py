import time
from typing import List

from src.schema import MeetingMinutes
from src.splitter import split_transcript

def _invoke_with_retry(agent, message: str, delay: float = 3.0, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            return agent.invoke({
                "messages": [{"role": "user", "content": message}]
            })
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate_limit" in err or "Too Many Requests" in err
            is_large = "413" in err or "Request too large" in err

            if is_large:
                raise RuntimeError(
                    f"Chunk is too large for the model even after splitting. "
                    f"Reduce LONG_CHUNK_TOKS in splitter.py.\nOriginal: {err}"
                ) from e

            if is_rate and attempt < retries - 1:
                wait = delay * (2 ** attempt)   # exponential backoff
                print(f"      ⏳ Rate limit (attempt {attempt+1}) — waiting {wait:.0f}s…")
                time.sleep(wait)
                continue

            raise   # non-recoverable or out of retries


# ── MAP phase ─────────────────────────────────────────────────────────────────

def run_map(text: str, map_agent, delay_between_chunks: float = 2.0) -> List[MeetingMinutes]:

    chunks = split_transcript(text)
    total = len(chunks)
    results: List[MeetingMinutes] = []

    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(delay_between_chunks)   # polite pacing

        print(f"   ➡️  Chunk {i+1}/{total}  ({len(chunk)} chars)")
        try:
            result = _invoke_with_retry(
                map_agent,
                f"Transcript segment ({len(chunk)} chars):\n\n{chunk}\n\n"
                "Please extract structured meeting minutes from this segment."
            )
            # Doc pattern: result["structured_response"]
            minutes: MeetingMinutes = result["structured_response"]
            results.append(minutes)
            print(f"      ✅  {len(minutes.decisions)} decisions, "
                  f"{len(minutes.action_items)} action items, "
                  f"sentiment={minutes.sentiment}")
        except Exception as e:
            print(f"      ⚠️  Chunk {i+1} failed: {str(e)[:200]}")

    return results


# ── REDUCE phase ──────────────────────────────────────────────────────────────

def run_reduce(chunk_results: List[MeetingMinutes], reduce_agent) -> MeetingMinutes:

    if not chunk_results:
        return MeetingMinutes(
            summary="No content was processed successfully.",
            decisions=[],
            action_items=[],
            sentiment="neutral",
        )

    if len(chunk_results) == 1:
        return chunk_results[0]

    # Format the combined input for the reduce agent
    combined = "\n\n".join([
        f"=== SEGMENT {i+1} ===\n"
        f"Summary: {r.summary}\n"
        f"Sentiment: {r.sentiment}\n"
        f"Decisions:\n" + "\n".join(f"  • {d}" for d in r.decisions) + "\n"
        f"Action Items:\n" + "\n".join(
            f"  • {a.owner}: {a.task} (deadline: {a.deadline})"
            for a in r.action_items
        )
        for i, r in enumerate(chunk_results)
    ])

    print(f"   🔀 Merging {len(chunk_results)} segment summaries…")
    result = _invoke_with_retry(
        reduce_agent,
        f"Merge these meeting segment summaries into one final structured output:\n\n{combined}",
    )
    # Doc pattern: result["structured_response"]
    return result["structured_response"]


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(text: str, map_agent, reduce_agent) -> MeetingMinutes:

    print("\n── MAP phase ─────────────────────────────────────────────")
    chunk_results = run_map(text, map_agent)

    succeeded = len(chunk_results)
    total = len(split_transcript(text))
    print(f"\n   MAP complete: {succeeded}/{total} chunks succeeded")

    if not chunk_results:
        raise RuntimeError(
            "All chunks failed. Check your API key and model availability."
        )

    print("\n── REDUCE phase ───────────────────────────────────────────")
    final = run_reduce(chunk_results, reduce_agent)
    print("   ✅ Reduce complete")

    return final