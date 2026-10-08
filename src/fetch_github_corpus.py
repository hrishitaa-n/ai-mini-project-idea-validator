"""
Fetches real ML/AI project repositories from the GitHub Search API.

Scaled-up version:
  - ~65 topic queries (vs. 15 before)
  - each query is run twice: once for Python repos and once for Jupyter
    Notebook repos (most student mini-projects are notebooks)
  - up to 100 results per request, filtered to repos with 2-1000 stars so
    results are "mini-project" scale, not giant frameworks like TensorFlow
  - optional GITHUB_TOKEN env var for 3x faster fetching (recommended)

Usage:
    python src/fetch_github_corpus.py              # full run
    python src/fetch_github_corpus.py --limit 3    # quick test, first 3 queries

Output: data/github_corpus.csv
"""

import argparse
import csv
import os
import time

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "github_corpus.csv")
API_URL = "https://api.github.com/search/repositories"

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

PER_PAGE = 100
DELAY = 2.5 if TOKEN else 7.0  # search API: 30 req/min with token, 10 without
MIN_DESC_LEN = 25
LANG_FILTERS = ["language:python", 'language:"Jupyter Notebook"']
STAR_RANGE = "stars:2..1000"

SEARCH_QUERIES = [
    # general
    "machine learning mini project", "deep learning project", "data science project",
    # NLP
    "spam detection", "fake news detection", "sentiment analysis", "text classification",
    "named entity recognition", "text summarization", "question answering system",
    "chatbot", "machine translation", "language detection", "resume parser",
    "plagiarism detection", "semantic search", "retrieval augmented generation",
    "knowledge graph",
    # computer vision
    "image classification", "object detection", "face recognition",
    "facial emotion recognition", "handwritten digit recognition",
    "optical character recognition", "image segmentation", "license plate recognition",
    "gesture recognition", "pose estimation", "image captioning",
    "image similarity search", "deepfake detection", "medical image classification",
    "plant disease detection", "traffic sign recognition", "drowsiness detection",
    "face mask detection", "face recognition attendance system",
    # speech / audio
    "speech recognition", "speech emotion recognition", "music genre classification",
    "voice assistant",
    # prediction / tabular / recsys
    "recommendation system", "movie recommendation", "job recommendation",
    "stock price prediction", "time series forecasting", "fraud detection",
    "credit risk prediction", "customer churn prediction", "house price prediction",
    "disease prediction", "heart disease prediction", "diabetes prediction",
    "student performance prediction", "crop yield prediction", "sales forecasting",
    "anomaly detection", "weather prediction", "air quality prediction",
    "predictive maintenance",
    # other
    "reinforcement learning game", "generative adversarial network",
    "intrusion detection", "malware detection",
]


def clean(text: str) -> str:
    return " ".join((text or "").split())


def fetch_page(query: str) -> list[dict]:
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": PER_PAGE,
        "page": 1,
    }
    for attempt in range(4):
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code in (403, 429):
            reset = resp.headers.get("x-ratelimit-reset")
            wait = 60
            if reset and reset.isdigit():
                wait = min(max(int(reset) - int(time.time()), 0) + 2, 120)
            print(f"    rate limited, waiting {wait}s (attempt {attempt + 1}/4)...")
            time.sleep(wait)
            continue
        if resp.status_code == 422:
            return []  # query GitHub couldn't parse; skip it
        resp.raise_for_status()
        return resp.json().get("items", [])
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N queries (for quick testing)")
    args = parser.parse_args()

    queries = SEARCH_QUERIES[: args.limit] if args.limit else SEARCH_QUERIES
    total_requests = len(queries) * len(LANG_FILTERS)
    print(f"Token: {'yes' if TOKEN else 'NO (slower; set GITHUB_TOKEN to speed up)'}")
    print(f"{len(queries)} queries x {len(LANG_FILTERS)} language filters = "
          f"{total_requests} requests (~{int(total_requests * DELAY / 60) + 1} min)\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    seen_ids, seen_desc, rows = set(), set(), []
    done = 0

    for query in queries:
        for lang in LANG_FILTERS:
            done += 1
            full_query = f"{query} {lang} {STAR_RANGE}"
            try:
                items = fetch_page(full_query)
            except requests.RequestException as e:
                print(f"[{done}/{total_requests}] FAILED '{query}': {e}")
                time.sleep(DELAY)
                continue

            added = 0
            for item in items:
                desc = clean(item.get("description"))
                if item["id"] in seen_ids or len(desc) < MIN_DESC_LEN:
                    continue
                if desc.lower() in seen_desc:  # drop forks / copy-paste duplicates
                    continue
                seen_ids.add(item["id"])
                seen_desc.add(desc.lower())

                language = item.get("language") or "Unknown"
                if language == "Jupyter Notebook":
                    language = "Python"  # notebooks are Python; keeps tech_stack consistent

                rows.append({
                    "id": item["id"],
                    "title": item["full_name"],
                    "abstract": desc,
                    "source": "github",
                    "url": item["html_url"],
                    "tech_stack": language,
                    "topics": ";".join(item.get("topics", [])),
                    "stars": item.get("stargazers_count", 0),
                })
                added += 1

            print(f"[{done}/{total_requests}] '{query}' ({lang.split(':')[1]}): "
                  f"+{added} new (total {len(rows)})")
            time.sleep(DELAY)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "abstract", "source", "url", "tech_stack", "topics", "stars"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} unique repos to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()