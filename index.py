#!/usr/bin/env python3
"""
Index documents for hybrid RAG retrieval.
Builds dense (FAISS) and sparse (BM25) indexes.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.dense import DenseRetriever, load_documents as load_dense_docs
from retrieval.sparse import SparseRetriever, load_documents as load_sparse_docs


def main():
    parser = argparse.ArgumentParser(description="Build hybrid RAG indexes")
    parser.add_argument("--corpus", required=True, help="Corpus directory with .txt files")
    parser.add_argument("--output", required=True, help="Output directory for indexes")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load documents
    print(f"Loading documents from {corpus_path}...")
    documents = load_dense_docs(str(corpus_path))
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents found!")
        return 1

    # Build dense index
    print("Building dense index...")
    dense_retriever = DenseRetriever(model_name=args.model, device=args.device)
    dense_retriever.build(documents)
    dense_retriever.save(str(output_path / "dense.index"), str(output_path / "dense_meta.pkl"))
    print(f"Dense index saved to {output_path}")

    # Build sparse index
    print("Building sparse index...")
    sparse_retriever = SparseRetriever()
    sparse_retriever.build(documents)
    sparse_retriever.save(str(output_path / "sparse.index"), str(output_path / "sparse_meta.pkl"))
    print(f"Sparse index saved to {output_path}")

    print("Indexing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())