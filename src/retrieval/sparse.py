import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class SparseRetriever:
    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def build(self, documents: List[Dict[str, Any]]):
        self.metadata = documents
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, index_path: str, metadata_path: str):
        if self.bm25 is None:
            raise ValueError("No index to save. Call build() first.")
        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.metadata]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load() or build() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                result["retriever"] = "sparse"
                results.append(result)
        return results


import numpy as np


def load_documents(corpus_dir: str) -> List[Dict[str, Any]]:
    documents = []
    corpus_path = Path(corpus_dir)
    for file_path in corpus_path.rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = file_path.stem
        documents.append({"id": doc_id, "text": text, "source": str(file_path)})
    return documents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build sparse index")
    parser.add_argument("--corpus", required=True, help="Corpus directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    docs = load_documents(args.corpus)
    retriever = SparseRetriever()
    retriever.build(docs)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_path / "sparse.index"), str(output_path / "sparse_meta.pkl"))
    print(f"Built sparse index with {len(docs)} documents")