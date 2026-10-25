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


## BM25 comparison and evidence demo

```bash
python -m ragbench.cli --corpus examples/docs.jsonl --queries examples/questions.jsonl --k 2 --method bm25
python -m unittest discover -s tests -v
```

`--method overlap` keeps the original baseline. BM25 adds corpus term rarity,
term-frequency saturation and document-length normalization. It builds one index
per evaluation and resolves ties by source ID. Defaults: k1=1.5, b=0.75. The same
labeled queries and metrics can be used for either method. The toy corpus is a
smoke test, not evidence of real-world retrieval quality.

```python
from ragbench.bm25 import BM25Index
index = BM25Index([{"id": "refund-policy", "text": "Refunds are available within 30 days."}])
print(index.evidence("refunds"))
print(index.evidence("astronomy"))  # no_evidence, empty citations
```

Evidence contains exact source passages, IDs and ranking scores. It is not an LLM
answer and does not follow instructions inside documents. A positive score means
lexical overlap, not factual verification or confidence. There is no semantic
matching, chunking, generated answer, or groundedness metric. English/ASCII token
rules are inherited from the baseline. Large corpora require an inverted index;
this version scans documents and is intended for small local evaluations.

A regression fixture demonstrates length normalization correcting an overlap tie.
The test suite also covers no-match abstention, source preservation, deterministic
ties, malformed relevance labels and mutation isolation. Before claiming an
improvement for a business use case, build a larger held-out labeled dataset and
inspect misses. No API keys, model downloads or network access are required.

### CI cost control

This repository is private. The workflow is manual-only (`workflow_dispatch`)
to avoid consuming paid Actions minutes without budget approval. It is not
automatically triggered by pushes or pull requests. Local checks run the same
commands. Enable automatic PR triggers after confirming free quota or a free runner.
