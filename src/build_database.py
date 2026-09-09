"""
Loads the corpus CSV into a SQLite database.

Run after fetch_github_corpus.py has produced data/corpus.csv:
    python src/build_database.py
"""

import csv
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT,
    tech_stack TEXT,
    topics TEXT,
    stars INTEGER DEFAULT 0
);
"""


def build_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No corpus CSV found at {CSV_PATH}. Run fetch_github_corpus.py first."
        )

    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute("DELETE FROM documents")  # rebuild fresh each run

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                int(row["id"]),
                row["title"],
                row["abstract"],
                row["source"],
                row["url"],
                row["tech_stack"],
                row["topics"],
                int(row["stars"] or 0),
            )
            for row in reader
        ]

    conn.executemany(
        "INSERT OR REPLACE INTO documents "
        "(id, title, abstract, source, url, tech_stack, topics, stars) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"Loaded {count} documents into {DB_PATH}")
    conn.close()


def get_all_documents():
    """Helper used by other scripts to fetch all corpus docs as dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents").fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    build_database()
