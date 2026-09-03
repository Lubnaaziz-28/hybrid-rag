<div align="center">

# Hybrid RAG

### Citation-Grounded Retrieval-Augmented Generation

[![Tech](https://img.shields.io/badge/Tech-Dense_%2B_Sparse-9B59B6)]()
[![Feature](https://img.shields.io/badge/Feature-Citation_Grounding-3498DB)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white)]()
[![FAISS](https://img.shields.io/badge/FAISS-Facebook-0076D6)]()

*Production RAG that hallucinates less. Dense + sparse retrieval with verifiable citations.*

</div>

---

## The Problem

Standard RAG systems retrieve relevant documents but produce answers that are **hard to verify**. Dense retrieval misses rare exact terms. BM25 misses paraphrases. Neither provides source attribution.

## The Solution

**Hybrid retrieval** (dense + sparse) with **reciprocal rank fusion**, **cross-encoder re-ranking**, and **citation grounding** that traces every answer back to its source.

```
Query
  │
  ├──────────────────┬──────────────────┐
  ▼                  ▼                  ▼
┌─────────┐    ┌──────────┐    ┌──────────────┐
│  FAISS  │    │  BM25    │    │   Cross-     │
│  Dense  │    │  Sparse  │    │   Encoder    │
│  Retrieval│  │  Retrieval│   │   Reranker   │
└────┬────┘    └────┬─────┘    └──────┬───────┘
     │              │                  │
     └──────┬───────┘                  │
            ▼                          │
   ┌────────────────┐                  │
   │ Reciprocal Rank│                  │
   │ Fusion (RRF)   │                  │
   └────────┬───────┘                  │
            │                          │
            └──────────┬───────────────┘
                       ▼
              ┌────────────────┐
              │  LLM Grounded  │
              │  Generation    │
              │  + Citations   │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │  Answer with   │
              │  Source Spans  │
              └────────────────┘
```

## Results

| Method | Recall@5 | Faithfulness |
|---|---|---|
| Dense only | TBD | TBD |
| BM25 only | TBD | TBD |
| **Hybrid + Rerank** | **TBD** | **TBD** |

## Quickstart

```bash
pip install -r requirements.txt

# Index documents
python index.py --docs ./documents/ --output ./index/

# Query with citations
python query.py --index ./index/ --query "What are the side effects?" --top-k 5
```

## Citation

If you use this in research, cite:
```bibtex
@software{aziz2024hybridrag,
  title={Hybrid RAG with Citation Grounding},
  author={Aziz, Lubna},
  year={2024}
}
```

## Contact

Dr. Lubna Aziz — engr.lubnaaziz@gmail.com — [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
