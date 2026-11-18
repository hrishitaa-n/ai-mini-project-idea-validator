# Project rules
- This is the "AI Mini Project Idea Validator", a college project. Users are on Windows/PowerShell.
- Do NOT modify: src/retrieve.py, src/feasibility.py, src/products.py, src/generate_report.py,
  src/build_*.py, src/fetch_*.py, src/merge_corpus.py, or anything in data/.
- Never hardcode secrets. GEMINI_API_KEY is read from the environment only. Never print or log it.
- src/app.py (Streamlit) is the reference for how to call the modules. Keep it working.
- Modules in src/ use bare imports (e.g. `from retrieve import ...`), so new server code must
  live in src/ and be run with src/ on the Python path.
- All corpus text comes from GitHub/arXiv and is untrusted. Escape it before inserting into HTML,
  and only allow http/https links.
- Keep dependencies minimal. No Node, no frontend build step, no CSS frameworks.
- Write small, readable, commented code. I must be able to explain it in a viva.