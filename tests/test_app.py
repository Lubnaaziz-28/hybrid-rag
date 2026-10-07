import pytest
from fastapi.testclient import TestClient

import app as app_module
import verifier


def _force_heuristic(monkeypatch):
    """Force verification through the deterministic heuristic in tests."""

    def _verify(answer, chunks, strategy="heur"):
        return verifier.verify_answer(answer, chunks, strategy="heur")

    monkeypatch.setattr(app_module, "verify_answer", _verify)


CHUNKS = [
    {"text": "GDPR imposes administrative fines for violations.", "source": "gdpr.txt"},
    {
        "text": "The business judgment rule protects directors.",
        "source": "bj.txt",
    },
]


def test_ask_post_verify_true(monkeypatch):
    _force_heuristic(monkeypatch)
    client = TestClient(app_module.app)
    response = client.post(
        "/ask?verify=true",
        json={
            "query": "What fines apply and what about director liability?",
            "answer": (
                "GDPR does not impose administrative fines and the business "
                "judgment rule protects directors."
            ),
            "retrieved_chunks": CHUNKS,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["verification"] is not None
    assert "truthfulness_score" in body["verification"]
    claims = body["verification"]["claims"]
    assert len(claims) == 2
    assert any(c["verdict"] == "CONTRADICTED" for c in claims)
    assert any(c["verdict"] == "SUPPORTED" for c in claims)


def test_ask_post_verify_false_omits_verification(monkeypatch):
    _force_heuristic(monkeypatch)
    client = TestClient(app_module.app)
    response = client.post(
        "/ask?verify=false",
        json={
            "query": "What fines apply?",
            "answer": "GDPR does not impose administrative fines.",
            "retrieved_chunks": CHUNKS,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is False
    assert body["verification"] is None


def test_ask_get_verify_true(monkeypatch):
    _force_heuristic(monkeypatch)
    client = TestClient(app_module.app)
    response = client.get("/ask?verify=true&query=hello")
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["verification"] is not None


def test_ask_post_default_verify_is_true(monkeypatch):
    _force_heuristic(monkeypatch)
    client = TestClient(app_module.app)
    response = client.post(
        "/ask",
        json={
            "query": "What fines apply?",
            "answer": "GDPR does not impose administrative fines.",
            "retrieved_chunks": CHUNKS,
        },
    )
    assert response.status_code == 200
    assert response.json()["verified"] is True


def test_health():
    client = TestClient(app_module.app)
    assert client.get("/health").json() == {"status": "healthy"}


@pytest.fixture
def client():
    return TestClient(app_module.app)
