import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
 
from src.loader   import load_transcript
from src.agents   import create_map_agent, create_reduce_agent
from src.pipeline import run_pipeline
from src.qa       import print_minutes, run_qa_loop


def main():
    # ── Resolve transcript path ───────────────────────────────────────────────
    if len(sys.argv) > 1:
        transcript_path = sys.argv[1]
    else:
        transcript_path = "data/sample.txt"

    print(f"\n📋  MEETING MINUTES GENERATOR")
    print(f"    Transcript : {transcript_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        text = load_transcript(transcript_path)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        print(f"    Place your .txt or .vtt transcript in the data/ folder.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    print(f"    Length     : {len(text):,} characters\n")

    # ── Build agents ──────────────────────────────────────────────────────────
    # Doc pattern (agents page):
    print("🔧  Initialising agents…")
    map_agent    = create_map_agent()
    reduce_agent = create_reduce_agent()
    print("    Map agent    : ready")
    print("    Reduce agent : ready")

    # ── Run Map-Reduce pipeline ───────────────────────────────────────────────
    minutes = run_pipeline(text, map_agent, reduce_agent)

    # ── Display structured output ─────────────────────────────────────────────
    print_minutes(minutes)

    # ── Q&A loop (Phase 4) ────────────────────────────────────────────────────
    print("\nWould you like to ask follow-up questions about this meeting?")
    answer = input("Enter 'yes' to start Q&A, or press Enter to exit: ").strip().lower()
    if answer in {"yes", "y"}:
        run_qa_loop(text, minutes)


if __name__ == "__main__":
    main()