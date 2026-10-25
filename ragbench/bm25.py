"""Reusable BM25 lexical index and source-preserving evidence extraction."""

from collections import Counter
from math import isfinite, log

from .core import TOKEN_RE


def validate_docs(docs: list[dict]) -> None:
    seen = set()
    for doc in docs:
        if not isinstance(doc, dict) or any(
            not isinstance(doc.get(k), str) or not doc[k].strip()
            for k in ("id", "text")
        ):
            raise ValueError("documents require nonempty string id and text")
        if doc["id"] in seen:
            raise ValueError("document IDs must be unique")
        seen.add(doc["id"])


class BM25Index:
    """Snapshot index; exact ties resolve by document ID. No external models."""

    def __init__(self, docs: list[dict], *, k1: float = 1.5, b: float = 0.75):
        validate_docs(docs)
        if not isfinite(k1) or k1 <= 0 or not isfinite(b) or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b between zero and one")
        self._docs = [{k: d[k] for k in ("id", "text")} for d in docs]
        self._counts = [Counter(TOKEN_RE.findall(d["text"].casefold())) for d in docs]
        self._lengths = [sum(count.values()) for count in self._counts]
        self._avg = sum(self._lengths) / len(docs) if docs else 0
        self._df = Counter(term for count in self._counts for term in count)
        self.k1, self.b = k1, b

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        terms = set(TOKEN_RE.findall(question.casefold()))
        results = []
        if not self._avg:
            return results
        for doc, counts, length in zip(self._docs, self._counts, self._lengths):
            score = 0.0
            for term in sorted(terms):
                frequency = counts[term]
                if not frequency:
                    continue
                idf = log(
                    1
                    + (len(self._docs) - self._df[term] + 0.5) / (self._df[term] + 0.5)
                )
                norm = self.k1 * (1 - self.b + self.b * length / self._avg)
                score += idf * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0:
                results.append({**doc, "score": score})
        return sorted(results, key=lambda r: (-r["score"], r["id"]))[:k]

    def evidence(self, question: str, k: int = 3) -> dict:
        """Return verbatim passages, never a synthesized or verified answer."""
        hits = self.retrieve(question, k)
        return {
            "mode": "extractive_evidence",
            "status": "evidence_found" if hits else "no_evidence",
            "citations": [
                {"source_id": h["id"], "quote": h["text"], "score": h["score"]}
                for h in hits
            ],
        }
