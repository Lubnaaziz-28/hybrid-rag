from verifier import (
    Claim,
    Verdict,
    normalize_chunks,
    split_claims,
    verify_answer,
)

CHUNK = (
    "GDPR imposes administrative fines for violations of up to 20 million "
    "EUR or 4% of global annual turnover. The business judgment rule "
    "protects directors from personal liability."
)


def _verdicts(result):
    return [c.verdict for c in result.claims]


def test_split_claims_sentences_and_lists():
    answer = "First claim. Second claim! Third claim? 1) Fourth. 2) Fifth."
    claims = split_claims(answer)
    assert claims == [
        "First claim.",
        "Second claim!",
        "Third claim?",
        "Fourth.",
        "Fifth.",
    ]


def test_split_claims_empty():
    assert split_claims("") == []
    assert split_claims("   ") == []


def test_normalize_chunks_accepts_str_dict_and_obj():
    from verifier import Chunk

    out = normalize_chunks(["a text", {"text": "b", "source": "s"}])
    assert out[0] == Chunk(text="a text", source="chunk")
    assert out[1] == Chunk(text="b", source="s")
    assert normalize_chunks([]) == []


def test_supported_claim():
    answer = (
        "The GDPR imposes administrative fines for violations of up to "
        "20 million EUR."
    )
    result = verify_answer(answer, [CHUNK], strategy="heur")
    assert result.totals["SUPPORTED"] == 1
    assert result.truthfulness_score == 1.0
    assert result.hallucination_risk == 0.0
    assert result.strategy == "heur"
    assert _verdicts(result)[0] == Verdict.SUPPORTED
    assert result.claims[0].evidence


def test_contradicted_claim():
    answer = "The GDPR imposes no administrative fines for violations."
    result = verify_answer(answer, [CHUNK], strategy="heur")
    assert result.totals["CONTRADICTED"] == 1
    assert result.truthfulness_score == 0.0
    assert _verdicts(result)[0] == Verdict.CONTRADICTED
    assert result.claims[0].evidence


def test_not_mentioned_claim():
    answer = "The refund policy allows returns within 90 days."
    result = verify_answer(answer, [CHUNK], strategy="heur")
    assert result.totals["NOT_MENTIONED"] == 1
    assert _verdicts(result)[0] == Verdict.NOT_MENTIONED
    assert result.claims[0].evidence == []


def test_truthfulness_score_mixed():
    answer = (
        "The GDPR imposes administrative fines for violations of up to "
        "20 million EUR. The GDPR imposes no administrative fines for "
        "violations. The refund policy allows returns within 90 days."
    )
    result = verify_answer(answer, [CHUNK], strategy="heur")
    assert result.totals["total"] == 3
    assert result.totals["SUPPORTED"] == 1
    assert result.totals["CONTRADICTED"] == 1
    assert result.totals["NOT_MENTIONED"] == 1
    assert abs(result.truthfulness_score - round(1 / 3, 4)) < 1e-9


def test_empty_answer_returns_safe_result():
    result = verify_answer("", [CHUNK], strategy="heur")
    assert result.claims == []
    assert result.truthfulness_score == 0.0
    assert result.hallucination_risk == 1.0
    assert result.strategy == "empty"


def test_auto_strategy_falls_back_to_heuristic_without_deps(monkeypatch):
    # Force the "no neural model" code path so the test stays offline and
    # deterministic regardless of the host environment.
    monkeypatch.setattr("verifier._nli_available", lambda: False)
    answer = (
        "The GDPR imposes administrative fines for violations of up to "
        "20 million EUR."
    )
    result = verify_answer(answer, [CHUNK], strategy="auto")
    assert result.strategy == "heur"
    assert result.totals["total"] == 1
    assert _verdicts(result)[0] == Verdict.SUPPORTED


def test_auto_strategy_uses_nli_when_available(monkeypatch):
    # When the neural backend is available, "auto" must delegate to the
    # NLI strategy without falling back to the heuristic.
    monkeypatch.setattr(verifier, "_nli_available", lambda: True)
    captured = {}

    def _fake_nli(claims, chunks):
        captured["called"] = True
        return [Claim(c, Verdict.SUPPORTED, 0.9, []) for c in claims]

    monkeypatch.setattr("verifier._nli", _fake_nli)
    answer = "Serious violations can reach 20 million EUR."
    result = verify_answer(answer, [CHUNK], strategy="auto")
    assert captured.get("called") is True
    assert result.strategy == "nli"
    assert result.totals["SUPPORTED"] == 1

