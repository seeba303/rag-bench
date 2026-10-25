"""CLI entry point for baseline retrieval metrics."""
import argparse
import json
from .core import evaluate, read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(evaluate(read_jsonl(args.corpus), read_jsonl(args.queries), args.k), indent=2))

if __name__ == "__main__":
    main()
