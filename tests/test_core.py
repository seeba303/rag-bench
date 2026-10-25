import tempfile
import unittest
from pathlib import Path
from ragbench.core import evaluate, read_jsonl, retrieve

DOCS = [{"id": "a", "text": "python retrieval search"},
        {"id": "b", "text": "vector database"},
        {"id": "c", "text": "sales leads"}]

class TestRagBench(unittest.TestCase):
    def test_retrieve(self):
        self.assertEqual(retrieve("vector", DOCS, 1)[0]["id"], "b")

    def test_metrics(self):
        result = evaluate(DOCS, [{"id": "q", "question": "vector", "relevant_ids": ["b"]}], 1)
        self.assertEqual((result["recall_at_k"], result["mrr_at_k"]), (1.0, 1.0))

    def test_bad_relevance(self):
        with self.assertRaises(ValueError):
            evaluate(DOCS, [{"question": "vector", "relevant_ids": ["missing"]}])

    def test_no_queries(self):
        with self.assertRaises(ValueError):
            evaluate(DOCS, [])

    def test_read_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"id":"a","text":"demo"}\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path)[0]["id"], "a")

if __name__ == "__main__":
    unittest.main()
