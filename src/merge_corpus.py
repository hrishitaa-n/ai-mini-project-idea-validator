"""
Merges all per-source CSVs (github_corpus.csv, arxiv_corpus.csv) into the
single data/corpus.csv that build_database.py loads.

- Assigns clean sequential integer IDs (GitHub and arXiv use different ID formats)
- Drops duplicate titles across sources
- Prints a per-source breakdown (good numbers to quote to your jury)

Usage: python src/merge_corpus.py
"""

import csv
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SOURCE_FILES = ["github_corpus.csv", "arxiv_corpus.csv"]
OUTPUT_PATH = os.path.join(DATA_DIR, "corpus.csv")
FIELDS = ["id", "title", "abstract", "source", "url", "tech_stack", "topics", "stars"]


def main():
    merged, seen_titles = [], set()

    for fname in SOURCE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  (skipping {fname}: not found)")
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept = 0
        for row in rows:
            key = " ".join(row["title"].lower().split())
            if key in seen_titles:
                continue
            seen_titles.add(key)
            merged.append(row)
            kept += 1
        print(f"  {fname}: {len(rows)} rows read, {kept} kept after dedup")

    if not merged:
        raise SystemExit("No source CSVs found. Run the fetch scripts first.")

    for new_id, row in enumerate(merged, start=1):
        row["id"] = new_id

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in FIELDS} for row in merged)

    print(f"\nWrote {len(merged)} documents to {OUTPUT_PATH}")
    for source, n in Counter(r["source"] for r in merged).items():
        print(f"  {source}: {n}")


if __name__ == "__main__":
    main()
