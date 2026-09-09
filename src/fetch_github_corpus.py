"""
Fetches real machine learning / AI project repositories from the GitHub
public search API and saves them as a corpus CSV for the Idea Validator.

No auth required for light use (rate-limited to 10 req/min unauthenticated,
which is enough for this one-time corpus build). If you hit rate limits,
wait a minute and re-run — the script skips repos already saved.

Usage: python src/fetch_github_corpus.py
"""

import csv
import os
import time
import requests

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.csv")

# Search queries chosen to cover the breadth of typical ML/AI mini-project
# topics a student might propose, so retrieval has real matches to find.
SEARCH_QUERIES = [
    "machine learning mini project",
    "deep learning project classifier",
    "image classification project pytorch",
    "nlp sentiment analysis project",
    "recommendation system project",
    "chatbot project python",
    "predictive model machine learning",
    "computer vision object detection project",
    "time series forecasting project",
    "fraud detection machine learning",
    "spam detection classifier project",
    "stock price prediction ml",
    "handwriting recognition cnn",
    "face recognition project python",
    "speech recognition project",
]

HEADERS = {"Accept": "application/vnd.github+json"}
API_URL = "https://api.github.com/search/repositories"


def fetch_for_query(query: str, per_page: int = 10) -> list[dict]:
    params = {
        "q": f"{query} language:python",
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    }
    resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 403:
        print(f"  Rate limited on '{query}', waiting 60s...")
        time.sleep(60)
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return items


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    seen_ids = set()
    rows = []

    for query in SEARCH_QUERIES:
        print(f"Fetching: {query}")
        try:
            items = fetch_for_query(query)
        except requests.RequestException as e:
            print(f"  Failed: {e}")
            continue

        for item in items:
            if item["id"] in seen_ids:
                continue
            if not item.get("description"):
                continue  # skip repos with no description, useless for embeddings
            seen_ids.add(item["id"])
            rows.append({
                "id": item["id"],
                "title": item["full_name"],
                "abstract": item["description"],
                "source": "github",
                "url": item["html_url"],
                "tech_stack": item.get("language") or "Unknown",
                "topics": ";".join(item.get("topics", [])),
                "stars": item.get("stargazers_count", 0),
            })

        time.sleep(2)  # be polite to the API, avoid hitting secondary rate limits

    print(f"\nCollected {len(rows)} unique repos with descriptions.")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "abstract", "source", "url", "tech_stack", "topics", "stars"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved corpus to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
