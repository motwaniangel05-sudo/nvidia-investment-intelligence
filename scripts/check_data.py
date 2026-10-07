"""Tells you which data files are missing and which command creates each one.
Run from the project folder:  python3 scripts/check_data.py"""
import glob
import os
import sys

STEPS = [
    ("1. Filing documents (10-K / 10-Q)", "data/documents/*.htm",
     "python -m data_acquisition.document_downloader"),
    ("2. Filing text chunks", "data/processed/chunks_NVDA.csv",
     "python -m rag.chunker"),
    ("3. Knowledge database", "data/knowledge.db",
     "python -m core.memory"),
    ("4. Search index", "models/embeddings/tfidf_store.pkl",
     "python -m rag.vector_store"),
]


def check(root="."):
    missing = []
    for name, pattern, command in STEPS:
        found = glob.glob(os.path.join(root, pattern))
        print(f"{'OK     ' if found else 'MISSING'}  {name}")
        if not found:
            missing.append(command)
    return missing


if __name__ == "__main__":
    todo = check()
    if todo:
        print("\nRun these commands, in this order:")
        for command in todo:
            print("  " + command)
        sys.exit(1)
    print("\nAll data files are present.")
