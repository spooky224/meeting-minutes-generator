from pathlib import Path
from src.rag_pipeline import RAGConfig, RAGPipeline
from src.agents import create_qa_agent   # real QA agent
from src.rag_judge import judge_benchmark_result, aggregate_scores

# Load a transcript from your data folder
transcript_path = Path("data/aws_meeting.txt")
with open(transcript_path, "r", encoding="utf-8") as f:
    transcript = f.read()

# Define probe queries
queries = [
    "What was the main topic of the meeting?",
    "Who mentioned deadlines?",
    "What decisions were made?"
]

# Use the same QA agent as production
qa_agent = create_qa_agent()

# Configure pipeline (example config)
config = RAGConfig(
    chunking_method="token",
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="all-mpnet-base-v2",
    use_reranker=True,
    top_k=3,
)

# Run benchmark
pipeline = RAGPipeline(config)
result = pipeline.run(transcript, queries, qa_agent)

# Judge the results
scored_result = judge_benchmark_result(result)
scores_summary = aggregate_scores(scored_result.query_results)

# Print results
print("Config:", scored_result.config.label())
print("Chunks:", scored_result.chunk_count, "Avg size:", scored_result.avg_chunk_size)
print("Indexing latency (ms):", scored_result.indexing_latency_ms)

for qr in scored_result.query_results:
    print("\nQuery:", qr.query)
    print("Retrieved chunks:", qr.retrieved_chunks)
    print("Answer:", qr.answer)
    print("Retrieval latency (ms):", qr.retrieval_latency_ms)
    print("Generation latency (ms):", qr.generation_latency_ms)
    print("Context relevance:", qr.context_relevance)
    print("Faithfulness:", qr.faithfulness)
    print("Answer relevance:", qr.answer_relevance)

# Print aggregate scores
print("\n=== Aggregate Scores ===")
for k, v in scores_summary.items():
    print(f"{k}: {v}")
