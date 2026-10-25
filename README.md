# RAG Bench — Grounded Answer Evaluation (local-first)

A tiny, reproducible **retrieval + evaluation** workbench for junior AI Engineer / LLM Engineer portfolios. It reads a JSONL corpus, retrieves top-k passages with transparent token-overlap ranking, and measures recall@k, MRR, and answer citation coverage using a labeled question set.

This baseline does **not** call an LLM or claim to produce generated answers. It exists to provide measurable evidence before introducing embeddings, vector databases, rerankers, or an answer generation model.

## Quick start
Python 3.10+, standard library only.

```bash
python -m ragbench.cli --corpus examples/docs.jsonl --queries examples/questions.jsonl --k 2
python -m unittest discover -s tests -v
```

## Data format

Corpus: one JSON object per line: `{"id":"doc-1","text":"..."}`.
Questions: `{"id":"q1","question":"...","relevant_ids":["doc-1"]}`.
All included data is synthetic and requires no secrets.

## Metrics

- **Recall@k:** fraction of relevant documents retrieved within top k, averaged over questions.
- **MRR@k:** reciprocal rank of the first relevant document, averaged over questions.
- **Hit@k:** fraction of questions with at least one relevant document among the top k.
- **Citation coverage:** fraction of retrieved results with a resolvable source id (100% by construction on valid local data); **not** a measure of factual correctness.

Zero-relevance questions are rejected rather than silently skewing evaluation. Ties break lexicographically by document id. This isn't a semantic retriever: synonym matches may fail.

## Roadmap

1. Add embedding retrieval (with fixed model/version and reproducible evaluation).
2. Add a vector-store backend and ingestion/chunk metadata.
3. Add grounded answer generation with source citations.
4. Human-review groundedness and hallucination error cases.
5. Trace latency, token costs, and retrieval drift under CI.

## Privacy

Run on public or consented documents. Do not upload proprietary material to hosted models without approval.

MIT licensed; see LICENSE.
