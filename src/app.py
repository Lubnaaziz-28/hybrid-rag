from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from verifier import verify_answer

app = FastAPI(
    title="Hybrid RAG Demo",
    description="Citation-grounded retrieval-augmented generation",
)


class ChunkIn(BaseModel):
    text: str
    source: str = "unknown"


class AskRequest(BaseModel):
    query: str
    answer: str | None = None
    retrieved_chunks: list[ChunkIn] | None = None
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[ChunkIn]
    verified: bool
    verification: dict[str, Any] | None = None


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>Hybrid RAG Demo</title></head>
    <body>
    <h1>Hybrid RAG Demo</h1>
    <p>Citation-grounded retrieval-augmented generation with hybrid dense+sparse retrieval.</p>
    <h2>Features</h2>
    <ul>
    <li>Dense retrieval (FAISS) + sparse retrieval (BM25)</li>
    <li>Reciprocal Rank Fusion</li>
    <li>Cross-encoder re-ranking</li>
    <li>Citation grounding</li>
    <li>HallucinationGuard truthfulness scoring</li>
    </ul>
    <h2>API</h2>
    <p><a href="/docs">Interactive API Docs (Swagger)</a></p>
    <p><a href="/redoc">ReDoc Documentation</a></p>
    <p>Verify an answer:
    <code>POST /ask?verify=true</code> with
    <code>{"query": "...", "answer": "...",
    "retrieved_chunks": [{"text": "...", "source": "..."}]}</code></p>
    <h2>Query Example</h2>
    <pre>POST /query
{
    "query": "What are the side effects?",
    "top_k": 5
}</pre>
    <p>Ask + verify:
    <pre>POST /ask?verify=true
{
    "query": "What is the governing law?",
    "answer": "The governing law is New York.",
    "retrieved_chunks": [{"text": "..."}]
}</pre>
    </body>
    </html>
    """


@app.post("/query")
async def query(query: str, top_k: int = 5):
    return {
        "query": query,
        "results": [
            {"text": "Sample result with citation", "source": "doc1.pdf", "score": 0.95},
            {"text": "Another result", "source": "doc2.pdf", "score": 0.87},
        ],
        "citations": ["doc1.pdf", "doc2.pdf"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _stub_answer(chunks: list[ChunkIn]) -> str:
    if not chunks:
        return "Unable to answer the query from the provided context."
    return " ".join(chunk.text for chunk in chunks[:3])[:600]


def _verify(payload: AskRequest, do_verify: bool):
    answer = payload.answer
    retrieved = payload.retrieved_chunks or []
    if answer is None:
        answer = _stub_answer(retrieved)
    if not do_verify:
        return AskResponse(
            query=payload.query,
            answer=answer,
            retrieved_chunks=retrieved,
            verified=False,
            verification=None,
        )
    chunk_dicts = [{"text": c.text, "source": c.source} for c in retrieved]
    result = verify_answer(answer, chunk_dicts)
    verification = {
        "truthfulness_score": result.truthfulness_score,
        "hallucination_risk": result.hallucination_risk,
        "strategy": result.strategy,
        "totals": result.totals,
        "claims": [
            {
                "text": claim.text,
                "verdict": claim.verdict.value,
                "confidence": claim.confidence,
                "evidence": claim.evidence,
            }
            for claim in result.claims
        ],
    }
    return AskResponse(
        query=payload.query,
        answer=answer,
        retrieved_chunks=retrieved,
        verified=True,
        verification=verification,
    )


@app.get("/ask", response_model=AskResponse)
async def ask_get(
    query: str = Query(default=""),
    verify: bool = Query(default=True),
):
    payload = AskRequest(query=query)
    return _verify(payload, verify)


@app.post("/ask", response_model=AskResponse)
async def ask_post(
    payload: AskRequest,
    verify: bool = Query(default=True),
):
    return _verify(payload, verify)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
