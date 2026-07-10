"""
src/rag_judge.py
=================
LLM-as-judge scoring for benchmark results. Fills in context_relevance,
faithfulness, and answer_relevance on each src.rag_pipeline.QueryResult.

Uses the SAME model pool as the rest of the app (src.model_pool.ModelPool) —
no new model dependency, no new API key. One structured-output call per
query scores all three metrics at once, rather than three separate calls,
to keep judging cost roughly 1x the number of probe queries, not 3x.

Deliberately decoupled from RAGPipeline: pipeline.run() produces
QueryResults with the judge fields left as None, and the functions here
fill them in afterward as a separate step. That means you can re-score an
existing benchmark run with a different/better judge model later without
re-running retrieval or generation.
"""

from typing import List

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from src.model_pool import ModelPool
from src.rag_pipeline import QueryResult, BenchmarkResult

_MODEL_POOL = ModelPool()
_MODEL = _MODEL_POOL.model


class JudgeScore(BaseModel):
    """Structured output for one query's judge evaluation."""
    context_relevance: float = Field(
        ge=0, le=1,
        description="How relevant the retrieved chunks are to the query. 0=irrelevant, 1=perfectly relevant.",
    )
    faithfulness: float = Field(
        ge=0, le=1,
        description="Whether the answer is fully grounded in the retrieved context with no unsupported claims. 0=hallucinated, 1=fully grounded.",
    )
    answer_relevance: float = Field(
        ge=0, le=1,
        description="Whether the answer actually addresses what the query asked. 0=off-topic, 1=directly and completely answers.",
    )
    reasoning: str = Field(description="One brief sentence explaining the scores.")


_judge_agent = None


def _get_judge_agent():
    """Created once and reused — same lazy-singleton pattern as the rest of
    the app's agents, so repeated scoring calls don't rebuild the agent."""
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = create_agent(
            model=_MODEL,
            tools=[],
            system_prompt="""
You are a strict, impartial evaluator of a RAG (retrieval-augmented generation)
system's output for ONE query. You will be given the query, the chunks of text
that were retrieved, and the answer that was generated from them.

Score three things, each from 0.0 to 1.0:

1. context_relevance — Do the retrieved chunks actually contain information
   relevant to answering the query? Score low if the chunks are off-topic or
   only tangentially related, even if the final answer happens to be correct.

2. faithfulness — Is every claim in the answer actually supported by the
   retrieved chunks? Score low if the answer includes any fact, name, date,
   or number that does NOT appear in the retrieved context, even if that
   fact happens to be true in general. Faithfulness measures groundedness
   in the given context, not real-world correctness.

3. answer_relevance — Does the answer actually address what the query asked,
   directly and completely? Score low if it's evasive, off-topic, or only
   partially answers the question.

Be strict. A score of 1.0 should be rare and reserved for a genuinely
excellent case. Base every score only on the query, chunks, and answer
provided — never use outside knowledge to fill in what the "correct" answer
should be.
""",
            response_format=JudgeScore,
            middleware=[ModelRetryMiddleware(max_retries=3)],
            name="rag_judge_agent",
        )
    return _judge_agent


def score_query_result(qr: QueryResult) -> QueryResult:
    """
    Score a single QueryResult. Returns a NEW QueryResult with the judge
    fields filled in (QueryResult is a pydantic model, so this is a copy,
    not a mutation).

    If the judge call fails for any reason, the original result is returned
    unchanged (scores stay None) rather than crashing the whole benchmark —
    a single flaky judge call shouldn't invalidate an entire run.
    """
    agent = _get_judge_agent()
    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(qr.retrieved_chunks)
    ) or "(no chunks were retrieved)"

    prompt = (
        f"QUERY:\n{qr.query}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"GENERATED ANSWER:\n{qr.answer}\n\n"
        "Score this RAG output as instructed."
    )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        score: JudgeScore = result["structured_response"]
    except Exception as e:
        print(f"      ⚠️  Judge scoring failed for query {qr.query!r}: {str(e)[:200]}")
        return qr

    return qr.model_copy(update={
        "context_relevance": score.context_relevance,
        "faithfulness":       score.faithfulness,
        "answer_relevance":   score.answer_relevance,
    })


def score_results(query_results: List[QueryResult]) -> List[QueryResult]:
    """Score every QueryResult in a benchmark run. Returns a new list —
    does not mutate the input."""
    scored = []
    for i, qr in enumerate(query_results):
        print(f"      🧑‍⚖️  Judging query {i + 1}/{len(query_results)}…")
        scored.append(score_query_result(qr))
    return scored


def judge_benchmark_result(result: BenchmarkResult) -> BenchmarkResult:
    """Convenience wrapper: score every query in a BenchmarkResult and
    return a new BenchmarkResult with the scored query_results."""
    scored = score_results(result.query_results)
    return result.model_copy(update={"query_results": scored})


def aggregate_scores(query_results: List[QueryResult]) -> dict:
    """
    Compute averages across a (scored) list of QueryResults, plus the
    heuristic timing stats that were already present on each result. Call
    this AFTER score_results()/judge_benchmark_result(). Any result missing
    scores (e.g. a judge call failed) is excluded from that metric's
    average rather than crashing the whole aggregation.
    """
    def _avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "avg_context_relevance":     _avg(qr.context_relevance for qr in query_results),
        "avg_faithfulness":          _avg(qr.faithfulness for qr in query_results),
        "avg_answer_relevance":      _avg(qr.answer_relevance for qr in query_results),
        "avg_retrieval_latency_ms":  _avg(qr.retrieval_latency_ms for qr in query_results),
        "avg_generation_latency_ms": _avg(qr.generation_latency_ms for qr in query_results),
        "num_queries_scored": sum(1 for qr in query_results if qr.context_relevance is not None),
        "num_queries_total":  len(query_results),
    }