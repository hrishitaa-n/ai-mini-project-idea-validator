"""
Generates a structured, explainable report using Gemini, grounded in the
retrieved similar documents, novelty score, and feasibility check.

Requires a free Gemini API key: https://aistudio.google.com/apikey
Set it as an environment variable before running:
    export GEMINI_API_KEY="your-key-here"      (Mac/Linux)
    $env:GEMINI_API_KEY="your-key-here"         (Windows PowerShell)
"""

import os
import time
from google import genai

MODEL_NAME = "gemini-3.6-flash"  # fast + free-tier friendly (gemini-2.5-flash was retired)

# Tried in order if a model is overloaded (503) or unavailable (404).
# Add/remove names here if Google changes availability again.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def build_prompt(idea_text: str, novelty_result: dict, feasibility_result: dict,
                 products: list | None = None) -> str:
    matches_text = "\n".join(
        f"- \"{m['title']}\" (similarity: {m['similarity']:.2f}): {m['abstract'][:200]}"
        for m in novelty_result["matches"][:3]
    )

    products_text = "\n".join(
        f"- {p['name']} ({p['category']}): {p['description']}"
        for p in (products or [])[:4]
    ) or "(none found in the catalog)"

    return f"""You are an academic project advisor helping a student evaluate their mini-project idea.

STUDENT'S PROPOSED IDEA:
{idea_text}

NOVELTY ANALYSIS (computed via semantic retrieval against a corpus of real prior projects):
Novelty score: {novelty_result['novelty_score']} / 1.0 ({novelty_result['interpretation']})

MOST SIMILAR PRIOR WORK FOUND:
{matches_text}

POPULAR REAL-WORLD APPS/WEBSITES RELATED TO THE IDEA (from a curated catalog):
{products_text}

FEASIBILITY CHECK:
Verdict: {feasibility_result['verdict']} (score: {feasibility_result['feasibility_score']}/100)
Flags: {'; '.join(feasibility_result['flags'])}

Write a structured report with these sections:
1. **Overlap Analysis** — explain specifically how the idea overlaps with the prior work found, citing the titles above. Be concrete, not generic.
2. **Feasibility Notes** — summarize practical concerns based on the feasibility check.
3. **Suggestions to Increase Originality** — give 2-3 concrete, specific ways to differentiate this idea from the prior work cited above (not generic advice).

4. **Real-World Context** — in 1-2 sentences, say how the idea relates to or differs from the popular apps/websites listed above. Mention ONLY products from that list; never invent others. If the list says "(none found in the catalog)", write one sentence saying no closely related popular product was found in the catalog and do not name any.

Keep the whole report under 350 words. Be direct and specific, not vague."""


def _is_retryable_error(e: Exception) -> bool:
    msg = str(e)
    return "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower()


def _is_model_unavailable_error(e: Exception) -> bool:
    msg = str(e)
    return "404" in msg or "NOT_FOUND" in msg


def generate_report(
    idea_text: str,
    novelty_result: dict,
    feasibility_result: dict,
    max_retries_per_model: int = 2,
    products: list | None = None,
) -> str:
    """
    Tries each model in FALLBACK_MODELS in order. For each model, retries
    briefly on 503 (overloaded) errors before moving to the next model.
    Raises the last error encountered if every model/attempt fails.
    """
    client = _get_client()
    prompt = build_prompt(idea_text, novelty_result, feasibility_result, products)

    last_error = None

    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries_per_model + 1):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e

                if _is_model_unavailable_error(e):
                    # This model doesn't exist / was retired - no point retrying it,
                    # move straight to the next model in the fallback list.
                    break

                if _is_retryable_error(e) and attempt < max_retries_per_model:
                    wait = 3 * (attempt + 1)  # 3s, then 6s
                    time.sleep(wait)
                    continue

                # Non-retryable error, or retries exhausted for this model -
                # move on to the next model in the fallback list.
                break

    raise RuntimeError(
        f"All models in FALLBACK_MODELS failed. Last error: {last_error}"
    )


if __name__ == "__main__":
    # Quick manual test with fake inputs (no retrieval needed for the prompt part)
    fake_novelty = {
        "novelty_score": 0.35,
        "interpretation": "Low novelty — significant overlap with existing work found.",
        "matches": [
            {"title": "spam-detector-nlp", "similarity": 0.72, "abstract": "A machine learning spam classifier using NLP and scikit-learn."},
        ],
    }
    fake_feasibility = {
        "verdict": "Feasible",
        "feasibility_score": 90,
        "flags": ["No major feasibility risks identified."],
    }
    print("--- PROMPT PREVIEW (no API call) ---")
    print(build_prompt("A spam email detector using machine learning", fake_novelty, fake_feasibility))