"""HallucinationGuard demo: vanilla RAG vs guarded answers on legal docs.

Compares a "vanilla" RAG answer (no trust signal) against the same answer
after running :func:`verifier.verify_answer`, showing the per-claim
verdicts and the overall truthfulness score.

Evidence is drawn from ``data/corpus/legal/*``. If the corpus is not
present, an embedded fallback is used so the demo always runs.

Run:
    python demo/hallucination_guard_demo.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)

CORPUS_DIR = os.path.join(HERE, "..", "data", "corpus", "legal")

_FALLBACKS = {
    "gdpr_fines.txt": (
        "The General Data Protection Regulation (GDPR) imposes administrative "
        "fines for violations. For less severe infringements, the maximum fine "
        "is EUR 10 million or 2% of global annual turnover, whichever is "
        "higher. For more serious violations, the fine can reach EUR 20 "
        "million or 4% of global turnover."
    ),
    "business_judgment.txt": (
        "The business judgment rule is a fundamental principle of corporate "
        "law that protects directors from personal liability for business "
        "decisions made in good faith. The rule creates a presumption that "
        "directors act on an informed basis and protects directors from all "
        "liability for their decisions, even if those decisions turn out to be "
        "wrong."
    ),
    "will_requirements.txt": (
        "A valid will in California must be in writing and signed by the "
        "testator. Holographic wills (entirely handwritten) are valid if "
        "signed, even without witnesses."
    ),
    "insider_trading.txt": (
        "Under the Securities Exchange Act of 1934, insider trading violations "
        "carry significant penalties. Criminal penalties include up to 10 years "
        "imprisonment. Civil penalties can reach $1 million."
    ),
}


def _load(name: str) -> str:
    path = os.path.join(CORPUS_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return _FALLBACKS.get(name, "")


# Two legal Q&As; each answer mixes true claims with deliberate lies.
CASES: list[dict] = [
    {
        "question": (
            "What GDPR fines apply and what does the business judgment rule "
            "say about director liability?"
        ),
        "evidence_names": ["gdpr_fines.txt", "business_judgment.txt"],
        "answer": (
            "Serious GDPR violations can reach a fine of 20 million EUR or "
            "4% of global turnover. GDPR does not impose administrative fines "
            "for violations. The business judgment rule does not protect "
            "directors from personal liability."
        ),
    },
    {
        "question": "What are the requirements for a valid California will?",
        "evidence_names": ["will_requirements.txt", "insider_trading.txt"],
        "answer": (
            "A valid will in California must be in writing and signed by the "
            "testator. A holographic will in California is invalid without "
            "witness signatures. Insider trading violations do not carry any "
            "penalties."
        ),
    },
]


def _print_vanilla(case: dict) -> None:
    print("=" * 72)
    print("VANILLA RAG (no truthfulness check)")
    print("=" * 72)
    print(f"Q: {case['question']}\n")
    print("A:")
    print(f"  {case['answer']}\n")
    print("  Trust signal: unknown\n")


def _print_guarded(case: dict, result) -> None:
    contradicted = result.totals.get("CONTRADICTED", 0)
    not_mentioned = result.totals.get("NOT_MENTIONED", 0)
    flagged = contradicted + not_mentioned
    print("=" * 72)
    print("HALLUCINATIONGUARD (truthfulness verified)")
    print("=" * 72)
    print(f"Q: {case['question']}\n")
    print(f"Strategy: {result.strategy}")
    print(
        f"Truthfulness score: {result.truthfulness_score:.2f} "
        f"(hallucination risk {result.hallucination_risk:.2f})\n"
    )
    print("Per-claim verdicts:")
    for i, claim in enumerate(result.claims, 1):
        print(
            f"  [{i}] {claim.verdict.value} "
            f"(conf {claim.confidence:.2f}) - {claim.text}"
        )
        if claim.evidence:
            print(f"      evidence: {claim.evidence[0][:110]}")
    print(
        f"\nClaims flagged as unsupported/contradicted: "
        f"{flagged} / {result.totals['total']}\n"
    )


def main() -> int:
    from verifier import verify_answer

    print("\nHallucinationGuard - vanilla RAG vs guarded comparison\n")
    total_claims = 0
    total_flagged = 0
    for case in CASES:
        evidence = [_load(name) for name in case["evidence_names"]]
        evidence = [text for text in evidence if text]
        _print_vanilla(case)
        result = verify_answer(case["answer"], evidence)
        contradicted = result.totals.get("CONTRADICTED", 0)
        not_mentioned = result.totals.get("NOT_MENTIONED", 0)
        total_claims += result.totals["total"]
        total_flagged += contradicted + not_mentioned
        _print_guarded(case, result)
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"Vanilla RAG emitted {total_claims} claims with no verification. "
        f"HallucinationGuard flagged {total_flagged} of them as unsupported "
        "or contradicted."
    )
    print(
        "With the cross-encoder NLI model installed "
        "(pip install -r requirements.txt), numeric hallucinations such as "
        "'5 years' vs '2 years' are additionally caught."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
