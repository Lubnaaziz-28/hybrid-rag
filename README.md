# hybrid-rag

Retrieval-Augmented Generation over institutional document collections, with **hybrid dense+sparse retrieval** and answers grounded in verifiable citations.

## Why hybrid
Dense retrieval misses rare exact terms; BM25 misses paraphrases. This repo fuses both with reciprocal rank fusion, then re-ranks, and attaches every claim in the answer to a source span.

## Architecture
```
docs -> chunker -> [FAISS dense | BM25 sparse] -> RRF fusion -> cross-encoder rerank -> LLM (grounded prompt) -> answer + citations
```

## Evaluation
| Method | Recall@5 | Faithfulness |
|---|---|---|
| Dense only | TBD | TBD |
| BM25 only | TBD | TBD |
| Hybrid + rerank | **TBD** | **TBD** |

## Quickstart
```bash
pip install -r requirements.txt
python src/ingest.py --path data/sample_docs/
uvicorn src.api:app --reload
```

## Contact
Dr. Lubna Aziz | engr.lubnaaziz@gmail.com | [Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
