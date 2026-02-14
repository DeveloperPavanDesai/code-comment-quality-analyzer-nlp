"""
Streamlit app: Code Comment Semantic Quality Classifier.
Predicts Poor / Average / Excellent from comment text (same pipeline as notebook).
"""
import streamlit as st
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict

st.set_page_config(page_title="Comment Quality Classifier", page_icon="💬", layout="centered")

st.title("Code Comment Semantic Quality Classifier")
st.markdown(
    "Predict **semantic quality** of a code comment (Poor / Average / Excellent) using the same pipeline as the exploration notebook."
)

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a TF–IDF + classifier model trained on CodeSearchNet comments, "
    "with NLTK preprocessing and semantic-style labels."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("1. Paste or type a code comment.")
st.sidebar.markdown("2. Click **Predict** to get the quality label.")

user_input = st.text_area(
    "Enter a code comment:",
    height=120,
    placeholder="e.g. Computes the new parent id for the node being moved and returns the integer value.",
)

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    run = st.button("Predict", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

if run:
    text = (user_input or "").strip()
    if not text:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            try:
                result = predict(text)
                st.success(f"**Predicted quality:** {result}")
                if result == "Poor":
                    st.caption("Short or vague comments tend to be labeled Poor.")
                elif result == "Average":
                    st.caption("Moderate length and clarity.")
                else:
                    st.caption("Longer, more descriptive comments tend to be Excellent.")
            except FileNotFoundError as e:
                st.error(
                    "Model not found. Train first: `python -m src.train` (and ensure `data/dataset.csv` exists, e.g. by running `python -m src.data`)."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")