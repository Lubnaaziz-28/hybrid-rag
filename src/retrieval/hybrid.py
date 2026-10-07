from typing import List, Dict, Any, Optional
from .dense import DenseRetriever
from .sparse import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, dense_k: int = 50, sparse_k: int = 50) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, top_k=dense_k) if self.dense else []
        sparse_results = self.sparse.search(query, top_k=sparse_k) if self.sparse else []

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        return fused

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            if doc_id not in doc_data:
                doc_data[doc_id] = result.copy()
                doc_data[doc_id]["retrievers"] = ["dense"]
            else:
                doc_data[doc_id]["retrievers"].append("dense")

        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            if doc_id not in doc_data:
                doc_data[doc_id] = result.copy()
                doc_data[doc_id]["retrievers"] = ["sparse"]
            else:
                doc_data[doc_id]["retrievers"].append("sparse")

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            result = doc_data[doc_id].copy()
            result["score"] = score
            result["retriever"] = "hybrid"
            results.append(result)

        return results