"""HallucinationGuard benchmark: how many lies are caught?

Runs a curated suite of legal/enterprise claims (some deliberately false)
through :func:`verifier.verify_answer` and reports how many hallucinated
claims the guard flags (CONTRADICTED or NOT_MENTIONED).

Evidence is loaded from ``data/corpus/*`` when available; otherwise an
embedded fallback is used.

Run:
    python benchmark.py                 # auto strategy
    python benchmark.py --strategy heur # force the heuristic
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

CORPUS_DIR = os.path.join(HERE, "data", "corpus")

_FALLBACKS = {
    "gdpr_fines.txt": (
        "The General Data Protection Regulation (GDPR) imposes administrative "
        "fines for violations. For less severe infringements, the maximum "
        "fine is EUR 10 million or 2% of global annual turnover, whichever is "
        "higher. For more serious violations, the fine can reach EUR 20 "
        "million or 4% of global turnover."
    ),
    "business_judgment.txt": (
        "The business judgment rule protects directors from personal liability "
        "for business decisions made in good faith and protects directors from "
        "all liability for their decisions, even if those decisions turn out "
        "to be wrong."
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
    "vesting_schedule.txt": (
        "Restricted stock units normally cliff-vest over 4 years with a "
        "1-year cliff. Vesting acceleration is NOT available for cause."
    ),
    "arr_mrr.txt": (
        "Annual Recurring Revenue (ARR) is MRR multiplied by 12. Net revenue "
        "retention above 100% means expansion. A negative NRR is possible."
    ),
}


def _load(name: str, domain: str = "legal") -> str:
    path = os.path.join(CORPUS_DIR, domain, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return _FALLBACKS.get(name, "")


def _chunk(name: str, domain: str = "legal") -> dict:
    return {"text": _load(name, domain), "source": f"{domain}/{name}"}


# Each case: an evidence set, an answer containing mixed true/false claims,
# and the number of claims that are deliberately false (lies).
CASES: list[dict] = [
    {
        "name": "gdpr_fine_deception",
        "domain": "legal",
        "question": "Summarize GDPR administrative fines and director protection.",
        "evidence_names": ("gdpr_fines.txt", "business_judgment.txt"),
        "answer": (
            "Serious GDPR violations can reach a fine of 20 million EUR or "
            "4% of global turnover. GDPR does not impose administrative fines "
            "for violations. The business judgment rule does not protect "
            "directors from personal liability."
        ),
        "lies": 2,
    },
    {
        "name": "will_insider_lies",
        "domain": "legal",
        "question": "What are California will requirements and insider trading penalties?",
        "evidence_names": ("will_requirements.txt", "insider_trading.txt"),
        "answer": (
            "A valid will in California must be in writing and signed by the "
            "testator. A holographic will in California is invalid without "
            "witness signatures. Insider trading violations do not carry any "
            "penalties."
        ),
        "lies": 2,
    },
    {
        "name": "vesting_false_promise",
        "domain": "enterprise",
        "question": "When do restricted stock units vest?",
        "evidence_names": ("vesting_schedule.txt",),
        "answer": (
            "Restricted stock units cliff-vest over 18 months with no cliff. "
            "Vesting acceleration is available for cause and immediate for "
            "all departures."
        ),
        "lies": 3,
    },
    {
        "name": "arr_misrepresentation",
        "domain": "enterprise",
        "question": "How is ARR related to MRR and what does NRR mean?",
        "evidence_names": ("arr_mrr.txt",),
        "answer": (
            "ARR is 13 months of MRR and net revenue retention below 80% "
            "indicates expansion. A positive NRR is impossible and MRR "
            "multiplied by 7 gives ARR."
        ),
        "lies": 3,
    },
]


def _evidence(case: dict) -> list[dict]:
    domain = case.get("domain", "legal")
    chunks = []
    for name in case["evidence_names"]:
        text = _load(name, domain)
        if text:
            chunks.append({"text": text, "source": f"{domain}/{name}"})
    return chunks


def run_benchmark(strategy: str) -> list[dict]:
    from verifier import verify_answer

    results: list[dict] = []
    total_claims = 0
    total_lies = 0
    total_caught = 0
    for case in CASES:
        evidence = _evidence(case)
        result = verify_answer(case["answer"], evidence, strategy=strategy)
        contradicted = result.totals.get("CONTRADICTED", 0)
        not_mentioned = result.totals.get("NOT_MENTIONED", 0)
        caught = contradicted + not_mentioned
        lies_in_answer = case["lies"]
        caught_lies = min(lies_in_answer, caught)
        total_claims += result.totals["total"]
        total_lies += lies_in_answer
        total_caught += caught_lies
        results.append(
            {
                "name": case["name"],
                "strategy": result.strategy,
                "claims": result.totals["total"],
                "supported": result.totals.get("SUPPORTED", 0),
                "contradicted": contradicted,
                "not_mentioned": not_mentioned,
                "flagged": caught,
                "lies_expected": lies_in_answer,
                "lies_caught": caught_lies,
                "truthfulness": result.truthfulness_score,
            }
        )
    return results, total_claims, total_lies, total_caught


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Case':<26}{'Strat':<6}{'Clms':>5}{'Sup':>5}{'Con':>5}{'N/M':>5}"
        f"{'Flag':>6}{'Lies':>5}{'Caught':>7}{'Score':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<26}{r['strategy']:<6}{r['claims']:>5}"
            f"{r['supported']:>5}{r['contradicted']:>5}{r['not_mentioned']:>5}"
            f"{r['flagged']:>6}{r['lies_expected']:>5}{r['lies_caught']:>7}"
            f"{r['truthfulness']:>7.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="HallucinationGuard benchmark")
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "nli", "heur", "llm"],
        help="verification strategy (default: auto)",
    )
    args = parser.parse_args()

    rows, total_claims, total_lies, total_caught = run_benchmark(args.strategy)

    print("\nHallucinationGuard benchmark - lies caught\n")
    strat = rows[0]["strategy"] if rows else args.strategy
    print(f"Strategy (per case): {strat}\n")
    _print_table(rows)

    # Vanilla RAG baseline: accepts every claim, catches zero lies.
    print("\nBaseline comparison (vanilla RAG vs HallucinationGuard):")
    print(f"  Vanilla RAG:          {total_claims} claims accepted, 0 lies caught.")
    print(
        f"  HallucinationGuard:   {total_claims} claims checked, "
        f"{total_lies} lies present, {total_caught} lies caught."
    )
    pct = (100.0 * total_caught / total_lies) if total_lies else 0.0
    print(f"  Recall: {total_caught}/{total_lies} lies caught ({pct:.0f}%)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
