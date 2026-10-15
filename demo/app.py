"""HallucinationGuard Gradio demo for Hugging Face Spaces.

Compares vanilla RAG answers vs HallucinationGuard-verified answers on legal docs.
Shows per-claim verdicts and truthfulness scores.

Run: python demo/app.py
"""

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

CASES = [
    {
        "id": "gdpr_business",
        "question": "What GDPR fines apply and what does the business judgment rule say about director liability?",
        "evidence_names": ["gdpr_fines.txt", "business_judgment.txt"],
        "answer": (
            "Serious GDPR violations can reach a fine of 20 million EUR or "
            "4% of global turnover. GDPR does not impose administrative fines "
            "for violations. The business judgment rule does not protect "
            "directors from personal liability."
        ),
    },
    {
        "id": "will_insider",
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


def _load(name: str) -> str:
    path = os.path.join(CORPUS_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return _FALLBACKS.get(name, "")


import gradio as gr
from verifier import verify_answer

SPACE_URL = "https://huggingface.co/spaces/Lubnaaziz-28/hybrid-rag-hallucination-guard"


def run_case(case_id: str):
    case = next(c for c in CASES if c["id"] == case_id)
    evidence = [_load(name) for name in case["evidence_names"]]
    evidence = [t for t in evidence if t]
    result = verify_answer(case["answer"], evidence)

    # Build vanilla answer display
    vanilla_md = f"""## Vanilla RAG (no truthfulness check)
**Q:** {case['question']}

**A:** {case['answer']}

**Trust signal:** unknown (no verification)"""

    # Build guarded display
    verdict_icons = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "NOT_MENTIONED": "⚠️"}
    claims_md = []
    for i, claim in enumerate(result.claims, 1):
        icon = verdict_icons.get(claim.verdict.value, "❓")
        evidence_text = claim.evidence[0][:110] if claim.evidence else "none"
        claims_md.append(
            f"{icon} **Claim {i}** [{claim.verdict.value}] (conf {claim.confidence:.2f}): {claim.text}\n"
            f"   Evidence: {evidence_text}"
        )

    totals = result.totals
    flagged = totals.get("CONTRADICTED", 0) + totals.get("NOT_MENTIONED", 0)
    guarded_md = f"""## HallucinationGuard (truthfulness verified)
**Truthfulness score:** {result.truthfulness_score:.2f} (hallucination risk {result.hallucination_risk:.2f})
**Strategy:** {result.strategy}
**Claims flagged:** {flagged} / {totals['total']}

### Per-claim verdicts:
{chr(10).join(claims_md)}"""

    # Summary
    summary = f"""
### Summary
- Vanilla RAG: {totals['total']} claims, 0 verified
- HallucinationGuard: {totals['total']} claims, {flagged} flagged ({totals.get('CONTRADICTED', 0)} contradicted, {totals.get('NOT_MENTIONED', 0)} not mentioned)
- Truthfulness score: {result.truthfulness_score:.2f}
"""

    return vanilla_md, guarded_md, summary


with gr.Blocks(title="HallucinationGuard — RAG Truthfulness Verifier", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        "# HallucinationGuard — RAG Truthfulness Verifier\n"
        "**Compare vanilla RAG answers vs HallucinationGuard-verified answers on legal documents.**\n"
        "Each case contains a deliberately hallucinated RAG answer. HallucinationGuard runs "
        "cross-encoder NLI verification against source documents and shows per-claim verdicts."
    )

    with gr.Row():
        with gr.Column():
            case_selector = gr.Radio(
                choices=[c["id"] for c in CASES],
                label="Select test case",
                value=CASES[0]["id"],
            )
            run_btn = gr.Button("Run Verification", variant="primary")

    with gr.Row():
        with gr.Column():
            vanilla_out = gr.Markdown(label="Vanilla RAG")
        with gr.Column():
            guarded_out = gr.Markdown(label="HallucinationGuard")

    summary_out = gr.Markdown(label="Summary")

    run_btn.click(
        fn=run_case,
        inputs=[case_selector],
        outputs=[vanilla_out, guarded_out, summary_out],
    )

    gr.Markdown(
        "---\n"
        "**How it works:** HallucinationGuard runs a cross-encoder NLI model "
        "(DeBERTa-v3-base) against retrieved source documents to classify each "
        "claim in the RAG answer as **SUPPORTED**, **CONTRADICTED**, or **NOT_MENTIONED**. "
        "The truthfulness score = supported / total claims.\n\n"
        "This is a **verification layer** for any RAG pipeline. Plug in your retriever + LLM, "
        "then call `verify_answer(answer, evidence)` before returning to users.\n\n"
        "Code: [github.com/Lubnaaziz-28/hybrid-rag](https://github.com/Lubnaaziz-28/hybrid-rag)"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="purple"),
    )