from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ModelRetryMiddleware, SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import os
from langchain_groq import ChatGroq
from src.schema import MeetingMinutes
from src.model_pool import ModelPool



MODEL_POOL = ModelPool()
MODEL = MODEL_POOL.model
#MODEL = ChatGroq(
#            model="llama-3.3-70b-versatile",
#            api_key=os.getenv("MY_KEY")
#        )

print(f"🛠️ Map agent using model: {MODEL.model}")



@tool
def analyze_transcript_chunk(chunk: str) -> str:
    """
    Analyze a segment of a meeting transcript.
    Extract the main topics discussed, decisions made, and tasks assigned.
    Return a detailed plain-text analysis of this segment.
    """
    # The tool's return value feeds back into the agent's reasoning loop.
    # The agent uses it to compose the final structured MeetingMinutes response.
    return (
        f"Transcript segment ({len(chunk)} chars) ready for analysis:\n\n{chunk}"
    )


@tool
def search_meeting_content(query: str, transcript: str) -> str:
    """
    Search the meeting transcript for information related to a query.
    Returns relevant excerpts from the transcript that address the query.
    Use this tool when answering follow-up questions about the meeting.
    """
    query_lower = query.lower()
    lines = transcript.splitlines()
    matches = [
        line for line in lines
        if any(word in line.lower() for word in query_lower.split())
    ]
    if not matches:
        return "No directly matching content found in transcript."
    # Return up to 20 most relevant lines
    return "\n".join(matches[:20])



def create_map_agent():
    print(f"🛠️ Map agent using model: {MODEL.model}")
    return create_agent(
        model=MODEL,
        tools=[analyze_transcript_chunk],   # ← tools doc @tool pattern
        system_prompt="""
        You are a professional meeting analyst. You will receive a segment of a meeting transcript.

        Steps:
        1. Call analyze_transcript_chunk with the transcript text you receive.
        2. Based on the analysis, produce structured meeting minutes with:
        - summary: one concise paragraph covering what was discussed
        - decisions: list of key decisions (no duplicates, meaningful only)
        - action_items: list of tasks with owner, task description, and deadline
        - sentiment: overall tone (positive / neutral / negative)

        Only include information explicitly stated in the transcript.
        Write "not specified" for missing deadlines or owners.
        """,
        response_format=MeetingMinutes,     # ← structured-output doc pattern
        middleware=[
            ModelRetryMiddleware(max_retries=3),   # ← middleware doc pattern
        ],
        name="map_agent",                   # ← agents doc: name parameter
    )


def create_reduce_agent():
    print(f"🛠️ Reduce agent using model: {MODEL.model}")
    return create_agent(
        model=MODEL,
        tools=[],                           # ← pure reasoning, no tools needed
        system_prompt="""
You are merging structured meeting summaries from multiple transcript chunks
into one single, coherent set of meeting minutes.

Rules:
- Write one concise global summary paragraph covering the whole meeting.
- Preserve ALL decisions across chunks; merge near-duplicates into one clear statement.
- Keep every unique action item; remove only exact duplicates.
- Determine the overall sentiment for the whole meeting.
- Do not add any information not present in the input summaries.
""",
        response_format=MeetingMinutes,     # ← structured-output doc pattern
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            SummarizationMiddleware(        # ← middleware doc pattern
                model=MODEL,
                trigger=("tokens", 8000),
                keep=("messages", 10),
            ),
        ],
        name="reduce_agent",
    )


def create_qa_agent():
    print(f"🛠️ QA agent using model: {MODEL.model}")
    """
    QA agent that consumes refined context (semantic + keyword search done outside).
    It no longer defines its own search tool, avoiding duplicate keyword calls.
    """
    return create_agent(
        model=MODEL,
        tools=[],   # no keyword tool here
        system_prompt="""
You are a meeting assistant. You will receive a user question and a refined context.

For every user question:
1. Read the provided context carefully.
2. Answer the question using ONLY information from the context.
3. If the context does not contain an answer, say so clearly.
4. Never guess or hallucinate — ground every answer in the context.
""",
        checkpointer=InMemorySaver(),       # short-term memory across turns
        middleware=[
            ModelRetryMiddleware(max_retries=3),
        ],
        name="qa_agent",
    )
