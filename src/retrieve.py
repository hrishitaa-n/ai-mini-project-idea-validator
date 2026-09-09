"""
Given a new project idea, retrieves the most similar documents from the
corpus and computes a novelty score.

This is the core module the Streamlit app calls. Can also be run
directly for a quick CLI test:
    python src/retrieve.py "your idea text here"
"""

import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from build_database import DB_PATH
import sqlite3

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
IDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "doc_ids.npy")

_model = None  # lazy-loaded singleton so Streamlit doesn't reload it per interaction


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_documents_by_ids(doc_ids: list[int]) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT * FROM documents WHERE id IN ({placeholders})", doc_ids
    ).fetchall()
    conn.close()
    return {row["id"]: dict(row) for row in rows}


def retrieve_similar(idea_text: str, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k most similar corpus documents to the given idea,
    each with a similarity score (cosine similarity, 0-1 range since
    embeddings are normalized).
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. Run build_index.py first."
        )

    model = get_model()
    index = faiss.read_index(INDEX_PATH)
    doc_ids = np.load(IDS_PATH)

    query_embedding = model.encode(
        [idea_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)
    matched_ids = [int(doc_ids[i]) for i in indices[0]]
    docs_by_id = get_documents_by_ids(matched_ids)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc_id = int(doc_ids[idx])
        doc = docs_by_id.get(doc_id)
        if doc:
            results.append({**doc, "similarity": float(score)})
    return results


def compute_novelty_score(idea_text: str, top_k: int = 5) -> dict:
    """
    Heuristic novelty score: 1 - max_similarity_to_corpus.

    This is intentionally simple (see project README for why) rather
    than a formally calibrated score against human labels — that's
    flagged as future work, not silently glossed over.
    """
    matches = retrieve_similar(idea_text, top_k=top_k)
    if not matches:
        return {"novelty_score": 1.0, "matches": [], "interpretation": "No corpus matches found."}

    max_similarity = max(m["similarity"] for m in matches)
    novelty_score = round(1 - max_similarity, 3)

    if novelty_score >= 0.6:
        interpretation = "Highly novel — little overlap found with existing corpus."
    elif novelty_score >= 0.4:
        interpretation = "Moderately novel — some conceptual overlap exists."
    else:
        interpretation = "Low novelty — significant overlap with existing work found."

    return {
        "novelty_score": novelty_score,
        "matches": matches,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    idea = " ".join(sys.argv[1:]) or "A machine learning model to detect spam emails using NLP"
    print(f"Idea: {idea}\n")
    result = compute_novelty_score(idea)
    print(f"Novelty score: {result['novelty_score']} — {result['interpretation']}\n")
    print("Top similar prior work:")
    for m in result["matches"]:
        print(f"  [{m['similarity']:.3f}] {m['title']} ({m['source']})")
        print(f"      {m['abstract'][:100]}...")
