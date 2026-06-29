
from src.agents import create_qa_agent
from src.schema import MeetingMinutes
from src.rag import build_faiss_index, query_faiss
from src.agents import create_qa_agent, search_meeting_content


def print_minutes(minutes: MeetingMinutes) -> None:
    """Pretty-print the structured meeting minutes to the terminal."""
    print("\n" + "=" * 65)
    print("  MEETING MINUTES")
    print("=" * 65)

    print(f"\n📝  SUMMARY\n    {minutes.summary}")

    print(f"\n🎭  SENTIMENT:  {minutes.sentiment.upper()}")

    print(f"\n📌  DECISIONS ({len(minutes.decisions)}):")
    if minutes.decisions:
        for d in minutes.decisions:
            print(f"    • {d}")
    else:
        print("    (none recorded)")

    print(f"\n✅  ACTION ITEMS ({len(minutes.action_items)}):")
    if minutes.action_items:
        for a in minutes.action_items:
            print(f"    • [{a.deadline}]  {a.owner}: {a.task}")
    else:
        print("    (none recorded)")

    print("\n" + "=" * 65)


def run_qa_loop(transcript: str, minutes: MeetingMinutes) -> None:
    print("\n" + "─" * 65)
    print("  Q&A MODE  — Ask questions about this meeting.")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("─" * 65)

    # Build FAISS index once for the transcript
    vectorstore = build_faiss_index(transcript)

    qa_agent = create_qa_agent()
    thread_config = {"configurable": {"thread_id": "meeting-qa-session"}}

    while True:
        try:
            question = input("\n❓  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting Q&A. Goodbye!")
            break

        if question.lower() in {"quit", "exit", "q", "bye"}:
            print("Exiting Q&A. Goodbye!")
            break
        if not question:
            continue

        try:
            # Step 1: semantic search → top-k chunks
            semantic_chunks = query_faiss(vectorstore, question, k=5)

            # 🔎 Debug log: show retrieved chunks
            print("\n[DEBUG] FAISS retrieved chunks:")
            for i, chunk in enumerate(semantic_chunks, 1):
                preview = chunk[:200].replace("\n", " ")
                print(f"   {i}. {preview}...")

            # Step 2: keyword search inside those chunks
            refined_matches = []
            for chunk in semantic_chunks:
                match = search_meeting_content.func(query=question, transcript=chunk)
                if match and "No directly matching" not in match:
                    refined_matches.append(match)

            # 🔎 Debug log: show keyword matches
            print("\n[DEBUG] Keyword matches after refinement:")
            if refined_matches:
                for i, match in enumerate(refined_matches, 1):
                    preview = match[:200].replace("\n", " ")
                    print(f"   {i}. {preview}...")
            else:
                print("   (none found)")

            # Step 3: agent sees question + refined context
            context = "\n\n".join(refined_matches) if refined_matches else "No exact matches found."
            result = qa_agent.invoke(
                {"messages": [{"role": "user", "content": f"Q: {question}\n\nContext:\n{context}"}]},
                thread_config,
            )
            answer = result["messages"][-1].content
            print(f"\n💬  {answer}")

        except Exception as e:
            print(f"\n⚠️  Error: {str(e)[:300]}")
