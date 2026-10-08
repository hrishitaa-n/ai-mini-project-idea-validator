"""
Fetches real research-paper abstracts from arXiv using the official `arxiv`
Python package (the sanctioned way to query arXiv programmatically; it
automatically respects arXiv's rate limit of 1 request / 3 seconds).

Usage:
    pip install arxiv
    python src/fetch_arxiv_corpus.py              # full run (~3-5 min)
    python src/fetch_arxiv_corpus.py --limit 3    # quick test

Output: data/arxiv_corpus.csv
"""

import argparse
import csv
import os
import re

import arxiv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "arxiv_corpus.csv")

RESULTS_PER_TOPIC = 60
MIN_ABSTRACT_LEN = 150

# Applied topics that match typical student mini-projects, so the paper
# side of the corpus is relevant rather than dominated by frontier-LLM papers.
TOPICS = [
    "spam detection", "fake news detection", "sentiment analysis", "text classification",
    "named entity recognition", "text summarization", "question answering",
    "chatbot dialogue system", "machine translation", "plagiarism detection",
    "semantic search", "retrieval augmented generation", "knowledge graph",
    "image classification", "object detection", "face recognition",
    "facial emotion recognition", "handwritten character recognition",
    "optical character recognition", "image segmentation", "license plate recognition",
    "gesture recognition", "human pose estimation", "image captioning",
    "image similarity retrieval", "deepfake detection", "medical image classification",
    "plant disease detection", "traffic sign recognition", "driver drowsiness detection",
    "face mask detection", "speech recognition", "speech emotion recognition",
    "music genre classification", "voice assistant",
    "recommendation system", "stock price prediction", "time series forecasting",
    "fraud detection", "credit risk prediction", "customer churn prediction",
    "house price prediction", "disease prediction", "student performance prediction",
    "crop yield prediction", "anomaly detection", "air quality prediction",
    "predictive maintenance", "reinforcement learning game", "generative adversarial network",
    "intrusion detection", "malware detection", "job recommendation",
]


def clean(text: str) -> str:
    return " ".join((text or "").split())


def base_id(entry_id: str) -> str:
    """http://arxiv.org/abs/2101.00001v2 -> 2101.00001"""
    tail = entry_id.rsplit("/abs/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N topics (for quick testing)")
    args = parser.parse_args()

    topics = TOPICS[: args.limit] if args.limit else TOPICS
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    os.makedirs(DATA_DIR, exist_ok=True)
    seen, rows = set(), []

    for i, topic in enumerate(topics, 1):
        search = arxiv.Search(
            query=f'all:"{topic}"',
            max_results=RESULTS_PER_TOPIC,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        added = 0
        try:
            for result in client.results(search):
                pid = base_id(result.entry_id)
                abstract = clean(result.summary)
                if pid in seen or len(abstract) < MIN_ABSTRACT_LEN:
                    continue
                seen.add(pid)
                rows.append({
                    "id": pid,
                    "title": clean(result.title),
                    "abstract": abstract,
                    "source": "arxiv",
                    "url": result.entry_id,
                    "tech_stack": "Research paper",
                    "topics": ";".join(result.categories),
                    "stars": 0,
                })
                added += 1
        except Exception as e:
            print(f"[{i}/{len(topics)}] '{topic}' FAILED: {e}")
            continue
        print(f"[{i}/{len(topics)}] '{topic}': +{added} new (total {len(rows)})")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "abstract", "source", "url", "tech_stack", "topics", "stars"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} unique papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
