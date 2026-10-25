import unittest

from ragbench.bm25 import BM25Index
from ragbench.core import evaluate


class TestBM25(unittest.TestCase):
    def test_length_normalization_beats_overlap_tie(self):
        docs = [
            {"id": "a-long", "text": "refund " + "unrelated " * 100},
            {"id": "z-short", "text": "refund policy"},
        ]
        queries = [{"question": "refund", "relevant_ids": ["z-short"]}]
        self.assertEqual(evaluate(docs, queries, 1)["mrr_at_k"], 0)
        self.assertEqual(evaluate(docs, queries, 1, method="bm25")["mrr_at_k"], 1)

    def test_ties_and_repeatability(self):
        index = BM25Index(
            [{"id": "b", "text": "refund"}, {"id": "a", "text": "refund"}]
        )
        self.assertEqual([x["id"] for x in index.retrieve("refund")], ["a", "b"])
        self.assertEqual(index.retrieve("refund refund"), index.retrieve("refund"))

    def test_source_quotes_and_abstention(self):
        doc = {"id": "policy", "text": "Refund within 30 days."}
        index = BM25Index([doc])
        evidence = index.evidence("refund")["citations"][0]
        self.assertEqual(
            (evidence["source_id"], evidence["quote"]), ("policy", doc["text"])
        )
        self.assertEqual(index.evidence("astronomy")["status"], "no_evidence")
        self.assertEqual(index.evidence("astronomy")["citations"], [])

    def test_snapshot_ignores_caller_mutation(self):
        docs = [{"id": "a", "text": "refund"}]
        index = BM25Index(docs)
        docs[0]["text"] = "mutated"
        result = index.retrieve("refund")
        result[0]["text"] = "also mutated"
        self.assertEqual(index.retrieve("refund")[0]["text"], "refund")

    def test_empty_and_punctuation_corpora(self):
        for docs in ([], [{"id": "a", "text": "!!!"}]):
            self.assertEqual(BM25Index(docs).retrieve("refund"), [])

    def test_invalid_docs(self):
        for docs in (
            [{"id": "a", "text": ""}],
            [{"id": [], "text": "x"}],
            [{"id": "a", "text": "x"}] * 2,
            [None],
        ):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                BM25Index(docs)

    def test_invalid_parameters(self):
        for kwargs in ({"b": -1}, {"b": 2}, {"k1": 0}, {"k1": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BM25Index([], **kwargs)
        index = BM25Index([])
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                index.retrieve("refund", k)
        with self.assertRaises(ValueError):
            index.retrieve(" ")

    def test_invalid_relevance_is_value_error(self):
        with self.assertRaises(ValueError):
            evaluate(
                [{"id": "a", "text": "x"}], [{"question": "x", "relevant_ids": [[]]}]
            )

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            evaluate(
                [{"id": "a", "text": "x"}],
                [{"question": "x", "relevant_ids": ["a"]}],
                method="unknown",
            )
