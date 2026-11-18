"""
Builds the "popular apps & websites" knowledge base.

  data/products.csv  ->  `products` table in data/corpus.db (SQLite)
                     ->  data/products.index + data/product_ids.npy (FAISS)

Run once after build_database.py, and again whenever you edit products.csv:
    python src/build_products.py
"""

import csv
import os
import sqlite3

import faiss
import numpy as np

from build_database import DB_PATH
from retrieve import get_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "products.csv")
INDEX_PATH = os.path.join(DATA_DIR, "products.index")
IDS_PATH = os.path.join(DATA_DIR, "product_ids.npy")

SCHEMA = """
DROP TABLE IF EXISTS products;
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    category TEXT,
    description TEXT NOT NULL,
    related_ideas TEXT,
    url TEXT
);
"""


def embedding_text(row: dict) -> str:
    # Including "related project ideas" makes product text resemble how
    # students phrase their ideas, which improves matching.
    return (
        f"{row['name']} ({row['category']}). {row['description']} "
        f"Related project ideas: {row['related_ideas']}."
    )


def build_products():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}")

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, start=1):
        row["id"] = i

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO products (id, name, kind, category, description, related_ideas, url) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r["id"], r["name"], r["kind"], r["category"], r["description"],
          r["related_ideas"], r["url"]) for r in rows],
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(rows)} products into SQLite table 'products'")

    model = get_model()
    embeddings = model.encode(
        [embedding_text(r) for r in rows],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array([r["id"] for r in rows], dtype=np.int64))
    print(f"Saved products FAISS index ({index.ntotal} vectors) to {INDEX_PATH}")


if __name__ == "__main__":
    build_products()
