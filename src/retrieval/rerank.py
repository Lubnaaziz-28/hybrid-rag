from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not results:
            return []

        pairs = [[query, result["text"]] for result in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    query = "What are the side effects?"
    results = [
        {"id": "doc1", "text": "The medication causes nausea and headache.", "score": 0.9},
        {"id": "doc2", "text": "Common side effects include dizziness.", "score": 0.8},
    ]
    reranked = reranker.rerank(query, results, top_k=2)
    for r in reranked:
        print(f"Score: {r['rerank_score']:.4f} - {r['text'][:60]}")