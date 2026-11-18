"""
Finds popular real-world apps and websites related to a project idea.

Same pattern as the prior-work search: embed the idea, find nearest vectors
in FAISS, then fetch the full records from SQLite by ID.

If the products index hasn't been built yet, the functions return empty
results instead of crashing, so the rest of the app keeps working.
"""

import os
import sqlite3

import faiss
import numpy as np

from build_database import DB_PATH
from retrieve import get_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "products.index")
IDS_PATH = os.path.join(DATA_DIR, "product_ids.npy")

# Tunable constants. MIN_SIMILARITY hides weak matches so unrelated products
# aren't shown; raise it if too many irrelevant results appear, lower it if
# good matches are being hidden. (Heuristic value, not calibrated.)
TOP_K = 5
MIN_SIMILARITY = 0.30


def _get_products_by_ids(ids: list[int]) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM products WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()
    return {r["id"]: dict(r) for r in rows}


def retrieve_products(
    idea_text: str, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY
) -> list[dict]:
    """Return up to top_k related products, each with a 'similarity' score."""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH)):
        return []

    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH)

    query = get_model().encode(
        [idea_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    scores, positions = index.search(query, min(top_k, index.ntotal))
    hits = [
        (int(ids[pos]), float(score))
        for score, pos in zip(scores[0], positions[0])
        if pos != -1 and score >= min_similarity
    ]
    if not hits:
        return []

    by_id = _get_products_by_ids([pid for pid, _ in hits])
    return [{**by_id[pid], "similarity": score} for pid, score in hits if pid in by_id]


def get_product_count() -> int:
    """Number of products in the catalog (0 if the table doesn't exist yet)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        return n
    except sqlite3.Error:
        return 0
