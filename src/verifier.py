"""HallucinationGuard — truthfulness verification for RAG answers.

Given an answer and the chunks retrieved to produce it, ``verify_answer``
classifies every claim in the answer against the retrieved evidence as
``SUPPORTED`` / ``CONTRADICTED`` / ``NOT_MENTIONED`` and reports an overall
``truthfulness_score`` in ``[0, 1]``.

Strategy chain for ``strategy="auto"`` (tried in order):

1. ``nli``  — lightweight NLI cross-encoder
             ``cross-encoder/nli-deberta-v3-base`` (no LLM key required).
2. ``llm``  — LLM-as-judge via an OpenAI-compatible API
             (requires ``OPENAI_API_KEY``; optional).
3. ``heur`` — deterministic bigram/Jaccard + polarity heuristic
             (no network, always available; best-effort baseline).

Example
-------
>>> from verifier import verify_answer
>>> chunks = [
...     "GDPR imposes administrative fines for violations.",
...     "The liability cap is limited to fees paid.",
... ]
>>> res = verify_answer(
...     "The liability cap is unlimited and GDPR does not impose any "
...     "administrative fines.",
...     chunks,
... )
>>> res.truthfulness_score
0.0
>>> [c.verdict.value for c in res.claims]
['CONTRADICTED', 'CONTRADICTED']
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("hybrid_rag.verifier")

# Strategy thresholds
ENTAIL_THRESH = 0.5
CONTRA_THRESH = 0.5

# Heuristic thresholds (tuned for the deterministic fallback). Negation /
# polarity clashes are treated as contradictions at a lower coverage than
# plain support, since an explicit negation is a strong lie signal.
CONSIDER_JACCARD = 0.12
SUPPORT_JACCARD = 0.25
CONTRA_JACCARD = 0.15

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Stopwords removed before computing token overlap.
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for",
    "is", "are", "was", "were", "be", "been", "being", "with",
    "and", "or", "as", "from", "this", "that", "these", "those",
    "it", "its", "i.e.", "e.g.", "vs.", "vs",
}

# Opposite-word pairs used to flag a polarity clash between a claim and a
# chunk (e.g. "allowed" vs "prohibited", "limited" vs "unlimited").
_OPPOSITE_PAIRS = [
    ("allowed", "prohibited"), ("prohibited", "allowed"),
    ("permitted", "forbidden"), ("forbidden", "permitted"),
    ("disallowed", "allowed"), ("allowed", "disallowed"),
    ("disallowed", "prohibited"), ("prohibited", "disallowed"),
    ("authorized", "unauthorized"), ("unauthorized", "authorized"),
    ("always", "never"), ("never", "always"),
    ("required", "optional"), ("optional", "required"),
    ("limited", "unlimited"), ("unlimited", "limited"),
    ("capped", "uncapped"), ("uncapped", "capped"),
    ("present", "absent"), ("absent", "present"),
    ("obligated", "released"), ("released", "obligated"),
    ("legal", "illegal"), ("illegal", "legal"),
    ("lawful", "unlawful"), ("unlawful", "lawful"),
    ("yes", "no"), ("no", "yes"),
    ("increase", "decrease"), ("decrease", "increase"),
    ("accept", "reject"), ("reject", "accept"),
    ("approve", "reject"), ("reject", "approve"),
    ("valid", "invalid"), ("invalid", "valid"),
    ("enabled", "disabled"), ("disabled", "enabled"),
    ("open", "closed"), ("closed", "open"),
    ("grant", "deny"), ("deny", "grant"),
    ("grantee", "grantor"), ("grantor", "grantee"),
]
OPPOSITE = {a: b for a, b in _OPPOSITE_PAIRS}

# Words that flip the polarity of a sentence.
NEGATIONS = {
    "no", "not", "never", "none", "nor", "cannot", "without",
    "neither", "prohibited", "prohibits", "forbidden", "disallowed",
    "denied", "unlawful", "illegal", "unauthorized", "absent",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_NLI_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError)

# Lazily loaded singletons.
_nli_model = None
_nli_labels: list[str] | None = None
_nli_checked = None


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_MENTIONED = "NOT_MENTIONED"


@dataclass
class Chunk:
    text: str
    source: str = "unknown"


@dataclass
class Claim:
    text: str
    verdict: Verdict
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    truthfulness_score: float
    claims: list[Claim]
    strategy: str
    totals: dict
    hallucination_risk: float = 0.0


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #
def split_claims(answer: str) -> list[str]:
    """Split an answer into atomic statements to verify."""
    if not answer:
        return []
    text = re.sub(r"\s+", " ", str(answer)).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    claims: list[str] = []
    for part in parts:
        part = part.strip()
        # strip leading list markers / numbering (e.g. "1.", "(a)", "* ", "A)")
        part = re.sub(
            r"^\s*(?:[\d]+[\.\)]?|[\(\)\[\]]|[-*•–—])?\s*",
            "",
            part,
        ).strip()
        if len(part) >= 4:
            claims.append(part)
    return claims


def normalize_chunks(chunks: Sequence) -> list[Chunk]:
    """Coerce arbitrary chunk inputs into a list of :class:`Chunk`."""
    out: list[Chunk] = []
    for item in chunks or []:
        if isinstance(item, Chunk):
            out.append(item)
        elif isinstance(item, str):
            out.append(Chunk(text=item, source="chunk"))
        elif isinstance(item, dict):
            out.append(
                Chunk(
                    text=item.get("text", item.get("content", "")),
                    source=item.get("source", item.get("id", "chunk")),
                )
            )
        else:
            out.append(Chunk(text=str(item), source="chunk"))
    return out


def _snippet(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\u2026"


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in STOPWORDS
    }


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOKEN_RE.findall(text.lower())
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def _has_negation(text: str) -> bool:
    return bool(set(_TOKEN_RE.findall(text.lower())) & NEGATIONS)


def _has_opposite(claim_toks: set[str], chunk_toks: set[str]) -> bool:
    for tok in claim_toks:
        opp = OPPOSITE.get(tok)
        if opp and opp in chunk_toks:
            return True
    return False


# --------------------------------------------------------------------------- #
# Heuristic strategy (deterministic, no network)
# --------------------------------------------------------------------------- #
def _heuristic_claim(claim: str, chunks: list[Chunk]) -> Claim:
    claim_toks = _tokens(claim)
    claim_bigrams = _ngrams(claim, 2)
    claim_neg = _has_negation(claim)

    best_supp = 0.0
    best_contra = 0.0
    supp_ev = ""
    contra_ev = ""

    for chunk in chunks:
        chunk_toks = _tokens(chunk.text)
        chunk_bigrams = _ngrams(chunk.text, 2)
        uni_union = claim_toks | chunk_toks
        bi_union = claim_bigrams | chunk_bigrams
        if not uni_union and not bi_union:
            continue
        shared = len(claim_toks & chunk_toks) + len(claim_bigrams & chunk_bigrams)
        total = len(uni_union) + len(bi_union)
        coverage = shared / total if total else 0.0
        if coverage < CONSIDER_JACCARD:
            continue

        chunk_neg = _has_negation(chunk.text)
        opposed = _has_opposite(claim_toks, chunk_toks) or (
            claim_neg != chunk_neg
        )
        if opposed and coverage > best_contra:
            best_contra = coverage
            contra_ev = chunk.text
        if coverage > best_supp:
            best_supp = coverage
            supp_ev = chunk.text

    if best_contra >= CONTRA_JACCARD:
        return Claim(
            text=claim,
            verdict=Verdict.CONTRADICTED,
            confidence=round(best_contra, 3),
            evidence=[_snippet(contra_ev)],
        )
    if best_supp >= SUPPORT_JACCARD:
        return Claim(
            text=claim,
            verdict=Verdict.SUPPORTED,
            confidence=round(best_supp, 3),
            evidence=[_snippet(supp_ev)],
        )
    return Claim(
        text=claim,
        verdict=Verdict.NOT_MENTIONED,
        confidence=round(max(best_supp, best_contra), 3),
        evidence=[],
    )


def _heuristic(claims: list[str], chunks: list[Chunk]) -> list[Claim]:
    return [_heuristic_claim(claim, chunks) for claim in claims]


# --------------------------------------------------------------------------- #
# NLI strategy (cross-encoder)
# --------------------------------------------------------------------------- #
def _nli_available() -> bool:
    """True if the NLI runtime (sentence-transformers + torch) is importable."""
    global _nli_checked
    if _nli_checked is None:
        import importlib.util

        _nli_checked = (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("sentence_transformers") is not None
        )
    return _nli_checked


def _load_nli():
    """Lazily load and cache the NLI cross-encoder (model + label map)."""
    global _nli_model, _nli_labels
    if _nli_model is not None:
        return _nli_model, _nli_labels
    from sentence_transformers import CrossEncoder
    _nli_model = CrossEncoder(NLI_MODEL, max_length=512)
    try:
        cfg = _nli_model.model.config
        id2label = {int(k): str(v).lower() for k, v in cfg.id2label.items()}
        labels = [id2label[i] for i in range(len(id2label))]
    except (AttributeError, KeyError, TypeError, OSError):
        labels = ["contradiction", "entailment", "neutral"]
    _nli_labels = labels
    logger.info("Loaded NLI model %s with labels %s", NLI_MODEL, labels)
    return _nli_model, _nli_labels


def _nli(claims: list[str], chunks: list[Chunk]) -> list[Claim]:
    if not chunks:
        return [
            Claim(c, Verdict.NOT_MENTIONED, 0.0, []) for c in claims
        ]
    model, labels = _load_nli()
    try:
        ent_i = labels.index("entailment")
        con_i = labels.index("contradiction")
    except ValueError:
        ent_i, con_i = 1, 0

    # premise = chunk (evidence), hypothesis = claim
    pairs = [(chunk.text, claim) for claim in claims for chunk in chunks]
    import numpy as np
    scores = np.asarray(
        model.predict(pairs, batch_size=32, show_progress_bar=False),
        dtype=float,
    )
    results: list[Claim] = []
    width = len(chunks)
    for i, claim in enumerate(claims):
        block = scores[i * width : i * width + width]
        ent = block[:, ent_i]
        con = block[:, con_i]
        e_max = float(ent.max())
        c_max = float(con.max())
        if c_max >= CONTRA_THRESH and c_max > e_max:
            idx = int(con.argmax())
            results.append(
                Claim(
                    claim,
                    Verdict.CONTRADICTED,
                    round(c_max, 3),
                    [_snippet(chunks[idx].text)],
                )
            )
        elif e_max >= ENTAIL_THRESH:
            idx = int(ent.argmax())
            results.append(
                Claim(
                    claim,
                    Verdict.SUPPORTED,
                    round(e_max, 3),
                    [_snippet(chunks[idx].text)],
                )
            )
        else:
            results.append(
                Claim(
                    claim,
                    Verdict.NOT_MENTIONED,
                    round(max(e_max, c_max), 3),
                    [],
                )
            )
    return results


# --------------------------------------------------------------------------- #
# LLM-as-judge strategy
# --------------------------------------------------------------------------- #
def _parse_llm_label(text: str, labels: list[str]) -> tuple[Verdict, float]:
    upper = text.upper()
    for label in sorted(labels, key=len, reverse=True):
        if label in upper:
            m = re.search(r"\b0?\.\d+\b", text)
            if m:
                conf = float(m.group())
            else:
                conf = 0.9
            return Verdict(label), min(max(conf, 0.0), 1.0)
    return Verdict.NOT_MENTIONED, 0.5


def _llm(claims: list[str], chunks: list[Chunk]) -> list[Claim]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
        else OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    context = "\n\n".join(_snippet(c.text, 400) for c in chunks)
    labels = [v.value for v in Verdict]
    results: list[Claim] = []
    for claim in claims:
        prompt = (
            "You are a factuality judge for retrieval-augmented generation.\n"
            "Classify the CLAIM as SUPPORTED, CONTRADICTED, or NOT_MENTIONED "
            "relative to the EVIDENCE below.\n"
            "- SUPPORTED: the evidence states or entails the claim.\n"
            "- CONTRADICTED: the evidence directly says the claim is false.\n"
            "- NOT_MENTIONED: the evidence does not address the claim.\n"
            f"\nEVIDENCE:\n{context}\n\nCLAIM: {claim}\n\n"
            "Answer with only: <LABEL> <confidence 0-1>, e.g. 'CONTRADICTED 0.92'.\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        text = (resp.choices[0].message.content or "").strip()
        label, conf = _parse_llm_label(text, labels)
        results.append(Claim(claim, label, conf, []))
    return results


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _empty_totals() -> dict:
    totals = {v.value: 0 for v in Verdict}
    totals["total"] = 0
    return totals


def verify_answer(
    answer: str,
    retrieved_chunks,
    strategy: str = "auto",
) -> VerificationResult:
    """Verify an answer against retrieved chunks.

    Parameters
    ----------
    answer:
        The generated answer to verify.
    retrieved_chunks:
        Evidence used to generate the answer. Accepts ``str``/``dict``/
        :class:`Chunk` sequences.
    strategy:
        ``"auto"`` (pick the best available), ``"nli"``, ``"llm"`` or
        ``"heur"``.
    """
    chunks = normalize_chunks(retrieved_chunks)
    claims = split_claims(answer)
    totals = _empty_totals()

    if not claims:
        return VerificationResult(
            truthfulness_score=0.0,
            claims=[],
            strategy="empty",
            totals=totals,
            hallucination_risk=1.0,
        )

    if strategy == "nli":
        claims_out = _nli(claims, chunks)
        strat = "nli"
    elif strategy == "llm":
        claims_out = _llm(claims, chunks)
        strat = "llm"
    elif strategy == "heur":
        claims_out = _heuristic(claims, chunks)
        strat = "heur"
    else:  # "auto": NLI when available, otherwise the deterministic heuristic.
        if _nli_available():
            try:
                claims_out = _nli(claims, chunks)
                strat = "nli"
            except _NLI_ERRORS as exc:
                logger.warning("NLI failed (%s); using heuristic", exc)
                claims_out = _heuristic(claims, chunks)
                strat = "heur"
        else:
            claims_out = _heuristic(claims, chunks)
            strat = "heur"

    for claim in claims_out:
        totals[claim.verdict.value] += 1
        totals["total"] += 1

    supported = totals.get(Verdict.SUPPORTED.value, 0)
    score = supported / totals["total"]
    return VerificationResult(
        truthfulness_score=round(score, 4),
        claims=claims_out,
        strategy=strat,
        totals=totals,
        hallucination_risk=round(1.0 - score, 4),
    )
