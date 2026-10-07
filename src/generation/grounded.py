from typing import List, Dict, Any, Optional
import re


class GroundedGenerator:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                pass

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, ctx in enumerate(contexts):
            source = ctx.get("source", ctx.get("id", f"doc_{i}"))
            text = ctx["text"]
            context_blocks.append(f"[Source {i+1}: {source}]\n{text}")

        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are a precise assistant that answers questions using ONLY the provided sources. 
Each source is labeled with a citation marker like [Source 1: filename]. 
You MUST cite your sources inline using these exact markers (e.g., [Source 1], [Source 2]).
If the answer is not in the sources, say "I cannot answer based on the provided sources."

Question: {query}

Sources:
{context_str}

Answer:"""
        return prompt

    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self._build_prompt(query, contexts)

        if self._client:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"[Error calling LLM: {e}]"
        else:
            answer = self._mock_generate(query, contexts)

        citations = self._extract_citations(answer, contexts)
        refusal = self._is_refusal(answer)

        return {
            "answer": answer,
            "citations": citations,
            "refusal": refusal,
            "prompt": prompt,
        }

    def _mock_generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "I cannot answer based on the provided sources."

        first_ctx = contexts[0]["text"][:200]
        return f"Based on the provided sources [Source 1], the answer relates to: {first_ctx}..."

    def _extract_citations(self, answer: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citation_pattern = r"\[Source (\d+)\]"
        matches = re.findall(citation_pattern, answer)
        cited_indices = set(int(m) - 1 for m in matches if 1 <= int(m) <= len(contexts))

        citations = []
        for idx in cited_indices:
            ctx = contexts[idx]
            citations.append({
                "source": ctx.get("source", ctx.get("id", f"doc_{idx}")),
                "text": ctx["text"][:200],
                "index": idx,
            })
        return citations

    def _is_refusal(self, answer: str) -> bool:
        refusal_phrases = [
            "cannot answer",
            "not in the sources",
            "not provided",
            "insufficient information",
            "unable to answer",
            "no information",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)


if __name__ == "__main__":
    gen = GroundedGenerator()
    query = "What are the side effects?"
    contexts = [
        {"id": "doc1", "source": "med1.txt", "text": "The medication causes nausea and headache."},
        {"id": "doc2", "source": "med2.txt", "text": "Common side effects include dizziness."},
    ]
    result = gen.generate(query, contexts)
    print(f"Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")
    print(f"Refusal: {result['refusal']}")