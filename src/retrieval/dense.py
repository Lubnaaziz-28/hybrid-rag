import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = self.model.get_sentence_embedding_dimension()

        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def build(self, documents: List[Dict[str, Any]], batch_size: int = 32):
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = documents

    def save(self, index_path: str, metadata_path: str):
        if self.index is None:
            raise ValueError("No index to save. Call build() first.")
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build() first.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                result["retriever"] = "dense"
                results.append(result)
        return results


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
    parser = argparse.ArgumentParser(description="Build dense index")
    parser.add_argument("--corpus", required=True, help="Corpus directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    docs = load_documents(args.corpus)
    retriever = DenseRetriever()
    retriever.build(docs)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_path / "dense.index"), str(output_path / "dense_meta.pkl"))
    print(f"Built dense index with {len(docs)} documents")