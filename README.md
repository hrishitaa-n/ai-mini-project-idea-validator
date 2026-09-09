# AI Mini Project Idea Validator

Checks a proposed mini-project idea against a real corpus of prior ML/AI
projects (pulled live from GitHub), scores novelty and feasibility, and
generates an explainable RAG report using Gemini.

## What's real vs. simplified (be upfront about this in your report)

- **Corpus**: 142 REAL GitHub repositories with real titles, descriptions,
  tech stacks, and star counts — fetched live from the GitHub API across
  15 ML/AI topic searches. Not synthetic data.
- **Retrieval**: Real dense retrieval using `sentence-transformers` +
  FAISS (cosine similarity via normalized inner product). Hybrid
  BM25+dense re-ranking was scoped out for time — flagged as future work.
- **Novelty score**: `1 - max_cosine_similarity_to_corpus`. This is an
  honest heuristic, not formally calibrated against human-labeled
  similarity judgments — flagged as future work in the report, not
  silently overclaimed.
- **Feasibility check**: Rule-based on form inputs (dataset availability,
  compute level, tech stack) rather than LLM-inferred — simpler, faster,
  and arguably more reliable.
- **RAG report**: Real Gemini API call grounded in the actual retrieved
  documents and scores computed above.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then set it:

```bash
export GEMINI_API_KEY="your-key-here"     # Mac/Linux
set GEMINI_API_KEY=your-key-here          # Windows cmd
```

## Build the corpus (run once, in order)

```bash
cd src
python fetch_github_corpus.py   # pulls ~150 real repos from GitHub API -> data/corpus.csv
python build_database.py        # loads CSV into SQLite -> data/corpus.db
python build_index.py           # embeds + builds FAISS index -> data/faiss.index
```

Note: `build_index.py` downloads the `all-MiniLM-L6-v2` model from Hugging
Face the first time it runs (~90MB), so it needs normal internet access.
This step was tested with fake embeddings in the development sandbox
(confirmed the FAISS/SQLite plumbing is correct) but needs to be run for
real on your machine to get actual semantic embeddings.

## Test retrieval from the command line (optional, before touching the UI)

```bash
python retrieve.py "A machine learning model to detect spam emails using NLP"
```

This should print a novelty score and the most similar real repos found.

## Run the app

```bash
cd ..
streamlit run src/app.py
```

## Project structure

```
idea-validator/
├── src/
│   ├── fetch_github_corpus.py   # pulls real data from GitHub API
│   ├── build_database.py        # CSV -> SQLite
│   ├── build_index.py           # embeddings + FAISS index
│   ├── retrieve.py              # similarity search + novelty scoring
│   ├── feasibility.py           # rule-based feasibility check
│   ├── generate_report.py       # Gemini RAG report generation
│   └── app.py                   # Streamlit UI
├── data/                        # corpus.csv, corpus.db, faiss.index (generated)
├── requirements.txt
└── README.md
```

## Status

- [x] Real corpus fetched from GitHub API (142 repos)
- [x] SQLite ingestion tested
- [x] FAISS/retrieval plumbing verified (with fake embeddings in dev sandbox)
- [x] Feasibility checker written
- [x] RAG report prompt tested (structure verified, not yet run against live API)
- [x] Streamlit app wired up, syntax-checked
- [ ] Run build_index.py for real (needs your own internet access)
- [ ] End-to-end test with real Gemini API key
- [ ] Optional: expand corpus, add more search queries for broader coverage
