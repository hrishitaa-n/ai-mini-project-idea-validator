"""
Embeds every document in the corpus database using a sentence-transformer
and builds a FAISS index for fast similarity search.

Run after build_database.py:
    python src/build_index.py

Produces:
    data/faiss.index   - the FAISS index (vectors only)
    data/doc_ids.npy   - maps FAISS index positions -> document IDs in SQLite
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from build_database import get_all_documents

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this use case
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
IDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "doc_ids.npy")


def build_index():
    docs = get_all_documents()
    if not docs:
        raise RuntimeError("No documents found. Run build_database.py first.")

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # Embed title + abstract together for richer semantic signal
    texts = [f"{d['title']}. {d['abstract']}" for d in docs]
    doc_ids = np.array([d["id"] for d in docs], dtype=np.int64)

    print(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    # normalize_embeddings=True means we can use inner product as cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, doc_ids)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {INDEX_PATH}")
    print(f"Saved doc ID mapping to {IDS_PATH}")


if __name__ == "__main__":
    build_index()
