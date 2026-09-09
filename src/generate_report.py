"""
Generates a structured, explainable report using Gemini, grounded in the
retrieved similar documents, novelty score, and feasibility check.

Requires a free Gemini API key: https://aistudio.google.com/apikey
Set it as an environment variable before running:
    export GEMINI_API_KEY="your-key-here"      (Mac/Linux)
    set GEMINI_API_KEY=your-key-here            (Windows cmd)
"""

import os
from google import genai

MODEL_NAME = "gemini-3.6-flash"  # fast + free-tier friendly


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def build_prompt(idea_text: str, novelty_result: dict, feasibility_result: dict) -> str:
    matches_text = "\n".join(
        f"- \"{m['title']}\" (similarity: {m['similarity']:.2f}): {m['abstract'][:200]}"
        for m in novelty_result["matches"][:3]
    )

    return f"""You are an academic project advisor helping a student evaluate their mini-project idea.

STUDENT'S PROPOSED IDEA:
{idea_text}

NOVELTY ANALYSIS (computed via semantic retrieval against a corpus of real prior projects):
Novelty score: {novelty_result['novelty_score']} / 1.0 ({novelty_result['interpretation']})

MOST SIMILAR PRIOR WORK FOUND:
{matches_text}

FEASIBILITY CHECK:
Verdict: {feasibility_result['verdict']} (score: {feasibility_result['feasibility_score']}/100)
Flags: {'; '.join(feasibility_result['flags'])}

Write a structured report with these sections:
1. **Overlap Analysis** — explain specifically how the idea overlaps with the prior work found, citing the titles above. Be concrete, not generic.
2. **Feasibility Notes** — summarize practical concerns based on the feasibility check.
3. **Suggestions to Increase Originality** — give 2-3 concrete, specific ways to differentiate this idea from the prior work cited above (not generic advice).

Keep the whole report under 300 words. Be direct and specific, not vague."""


def generate_report(idea_text: str, novelty_result: dict, feasibility_result: dict) -> str:
    client = _get_client()
    prompt = build_prompt(idea_text, novelty_result, feasibility_result)
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt)
    return response.text


if __name__ == "__main__":
    # Quick manual test with fake inputs (no retrieval/API needed for the prompt part)
    fake_novelty = {
        "novelty_score": 0.35,
        "interpretation": "Low novelty — significant overlap with existing work found.",
        "matches": [
            {"title": "spam-detector-nlp", "similarity": 0.72,
                "abstract": "A machine learning spam classifier using NLP and scikit-learn."},
        ],
    }
    fake_feasibility = {
        "verdict": "Feasible",
        "feasibility_score": 90,
        "flags": ["No major feasibility risks identified."],
    }
    print("--- PROMPT PREVIEW (no API call) ---")
    print(build_prompt("A spam email detector using machine learning",
          fake_novelty, fake_feasibility))
