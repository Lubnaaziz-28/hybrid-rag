from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Hybrid RAG Demo", description="Citation-grounded retrieval-augmented generation")

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
    </ul>
    <h2>API</h2>
    <p><a href="/docs">Interactive API Docs (Swagger)</a></p>
    <p><a href="/redoc">ReDoc Documentation</a></p>
    <h2>Query Example</h2>
    <pre>POST /query
{
    "query": "What are the side effects?",
    "top_k": 5
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
            {"text": "Another result", "source": "doc2.pdf", "score": 0.87}
        ],
        "citations": ["doc1.pdf", "doc2.pdf"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
