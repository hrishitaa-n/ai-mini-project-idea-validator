"""
AI Mini Project Idea Validator — Streamlit app.

Run with: streamlit run src/app.py

Requires:
- data/corpus.db and data/faiss.index already built
  (run fetch_github_corpus.py -> build_database.py -> build_index.py first)
- GEMINI_API_KEY environment variable set for report generation
"""

import os
import streamlit as st

from retrieve import compute_novelty_score
from feasibility import check_feasibility, COMMON_TECH_STACKS, COMPUTE_LEVELS
from generate_report import generate_report

st.set_page_config(page_title="AI Mini Project Idea Validator", layout="centered")

st.title("AI Mini Project Idea Validator")
st.caption(
    "Checks your project idea against a real corpus of prior ML/AI projects, "
    "scores novelty and feasibility, and generates an explainable report."
)

with st.form("idea_form"):
    idea_text = st.text_area(
        "Describe your project idea",
        height=120,
        placeholder="e.g. A system that predicts student dropout risk using academic and attendance data...",
    )

    proposed_tech = st.multiselect(
        "Planned tech stack", options=COMMON_TECH_STACKS, default=["Python"]
    )

    dataset_available = st.radio(
        "Do you have a specific, confirmed dataset in mind?", ["Yes", "No"]
    ) == "Yes"

    compute_level = st.selectbox("Available compute", COMPUTE_LEVELS)

    submitted = st.form_submit_button("Validate Idea")

if submitted:
    if not idea_text.strip():
        st.error("Please describe your idea before submitting.")
        st.stop()

    with st.spinner("Retrieving similar prior work..."):
        try:
            novelty_result = compute_novelty_score(idea_text, top_k=5)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    with st.spinner("Checking feasibility..."):
        feasibility_result = check_feasibility(
            proposed_tech, dataset_available, compute_level, novelty_result["matches"]
        )

    # --- Results display ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Novelty Score", f"{novelty_result['novelty_score']:.2f} / 1.0")
        st.caption(novelty_result["interpretation"])
    with col2:
        st.metric("Feasibility", feasibility_result["verdict"])
        st.caption(f"Score: {feasibility_result['feasibility_score']}/100")

    st.subheader("Most Similar Prior Work")
    for m in novelty_result["matches"]:
        with st.expander(f"[{m['similarity']:.2f}] {m['title']}"):
            st.write(m["abstract"])
            st.caption(f"Source: {m['source']} | Tech: {m['tech_stack']} | [Link]({m['url']})")

    st.subheader("Feasibility Flags")
    for flag in feasibility_result["flags"]:
        st.write(f"- {flag}")

    st.subheader("Explainable Report")
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning(
            "GEMINI_API_KEY not set — skipping AI-generated report. "
            "Set the environment variable to enable this section."
        )
    else:
        with st.spinner("Generating report..."):
            try:
                report = generate_report(idea_text, novelty_result, feasibility_result)
                st.markdown(report)
            except Exception as e:
                st.error(f"Report generation failed: {e}")
