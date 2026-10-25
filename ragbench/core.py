"""Token-overlap retrieval with explicit metrics and validation."""
from __future__ import annotations
import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))

def read_jsonl(path: str | Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            data.append(item)
    return data

def retrieve(question: str, docs: list[dict], k: int = 3) -> list[dict]:
    if k < 1:
        raise ValueError("k must be >= 1")
    query = tokens(question)
    ranks = []
    for doc in docs:
        score = len(query & tokens(doc["text"]))
        if score > 0:
            ranks.append((score, doc["id"]))
    ranks.sort(key=lambda row: (-row[0], row[1]))
    doc_map = {d["id"]: d for d in docs}
    return [{"id": doc_id, "score": score, "text": doc_map[doc_id]["text"]}
            for score, doc_id in ranks[:k]]

def evaluate(docs: list[dict], queries: list[dict], k: int = 3) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    ids = [d.get("id") for d in docs]
    if any(not isinstance(d.get("id"), str) or not isinstance(d.get("text"), str) for d in docs):
        raise ValueError("documents must have string id and text")
    if len(set(ids)) != len(ids):
        raise ValueError("document IDs must be unique")
    if not queries:
        raise ValueError("at least one query is required")
    records, total_recall, total_mrr, total_hit = [], 0.0, 0.0, 0.0
    for q in queries:
        if not isinstance(q.get("question"), str) or not isinstance(q.get("relevant_ids"), list):
            raise ValueError("queries require question and relevant_ids")
        relevant = set(q["relevant_ids"])
        if not relevant or not all(isinstance(x, str) for x in relevant):
            raise ValueError("relevant_ids must be nonempty string IDs")
        if not relevant.issubset(set(ids)):
            raise ValueError("relevant_ids refer to unknown documents")
        ranking = retrieve(q["question"], docs, k)
        ranked_ids = [r["id"] for r in ranking]
        hits = [i + 1 for i, doc_id in enumerate(ranked_ids) if doc_id in relevant]
        recall = len(hits) / len(relevant)
        mrr = 1 / hits[0] if hits else 0.0
        hit = float(bool(hits))
        total_recall += recall
        total_mrr += mrr
        total_hit += hit
        records.append({"id": q.get("id"), "retrieved_ids": ranked_ids,
                        "recall_at_k": recall, "reciprocal_rank": mrr, "hit_at_k": bool(hits)})
    count = len(queries)
    return {"k": k, "questions": count, "recall_at_k": round(total_recall / count, 4),
            "mrr_at_k": round(total_mrr / count, 4), "hit_at_k": round(total_hit / count, 4),
            "results": records}
