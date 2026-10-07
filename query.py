#!/usr/bin/env python3
"""
Query the hybrid RAG system with citation grounding.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker
from generation.grounded import GroundedGenerator


def main():
    parser = argparse.ArgumentParser(description="Query hybrid RAG system")
    parser.add_argument("--index", required=True, help="Index directory")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="LLM model for generation")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()

    index_path = Path(args.index)

    # Load retrievers
    print("Loading indexes...")
    dense = DenseRetriever(model_name=args.model, device=args.device)
    dense.load(str(index_path / "dense.index"), str(index_path / "dense_meta.pkl"))

    sparse = SparseRetriever()
    sparse.load(str(index_path / "sparse.index"), str(index_path / "sparse_meta.pkl"))

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    # Search
    print(f"Searching for: {args.query}")
    results = hybrid.search(args.query, top_k=args.top_k * 2)

    # Rerank if requested
    if args.rerank:
        print("Reranking...")
        reranker = CrossEncoderReranker(device=args.device)
        results = reranker.rerank(args.query, results, top_k=args.top_k)
    else:
        results = results[:args.top_k]

    # Print retrieval results
    print(f"\nRetrieved {len(results)} documents:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['source']} (score: {r['score']:.4f}, retrievers: {r.get('retrievers', r.get('retriever'))})")
        print(f"      {r['text'][:150]}...")

    # Generate answer
    print("\nGenerating answer...")
    generator = GroundedGenerator(model_name=args.llm_model, api_key=args.api_key)
    result = generator.generate(args.query, results)

    print(f"\nAnswer: {result['answer']}")
    print(f"Refusal: {result['refusal']}")
    print(f"Citations: {len(result['citations'])}")
    for c in result['citations']:
        print(f"  - [{c['index']+1}] {c['source']}: {c['text'][:100]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())