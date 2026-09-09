"""
Rule-based feasibility check. Deliberately simple (form inputs, not LLM
inference) so it's fast, transparent, and doesn't need extra API calls.
"""

COMMON_TECH_STACKS = [
    "Python", "TensorFlow", "PyTorch", "Scikit-learn", "Keras",
    "OpenCV", "NLTK", "Streamlit", "Flask", "SQL", "R", "Java",
]

COMPUTE_LEVELS = ["Laptop CPU only", "Free-tier GPU (Colab/Kaggle)", "Paid cloud GPU"]


def check_feasibility(
    proposed_tech: list[str],
    dataset_available: bool,
    compute_level: str,
    similar_docs: list[dict],
) -> dict:
    """
    Compares the proposed tech stack against what similar existing
    projects actually used, and flags basic feasibility risks.
    """
    flags = []
    score = 100  # start at fully feasible, deduct for each risk found

    if not dataset_available:
        flags.append("No confirmed dataset — this is the most common reason mini-projects stall. Identify a specific public dataset before starting.")
        score -= 30

    if compute_level == "Laptop CPU only" and any(
        t in proposed_tech for t in ["TensorFlow", "PyTorch", "Keras"]
    ):
        flags.append("Deep learning framework selected with CPU-only compute — training will be slow. Consider a free-tier GPU (Colab/Kaggle) or a lighter model.")
        score -= 15

    # Compare against what similar real projects actually used
    similar_tech = set()
    for doc in similar_docs:
        if doc.get("tech_stack"):
            similar_tech.add(doc["tech_stack"])

    if similar_tech and not (set(proposed_tech) & similar_tech):
        flags.append(
            f"Similar existing projects commonly used {', '.join(sorted(similar_tech)[:3])}, "
            f"which differs from your selected stack — worth double-checking compatibility."
        )
        score -= 10

    score = max(0, score)

    if score >= 80:
        verdict = "Feasible"
    elif score >= 50:
        verdict = "Feasible with caveats"
    else:
        verdict = "High risk — address flags before proceeding"

    return {
        "feasibility_score": score,
        "verdict": verdict,
        "flags": flags if flags else ["No major feasibility risks identified."],
    }
