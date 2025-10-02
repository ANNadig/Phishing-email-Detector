# src/app.py
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"
VECT_PATH = ROOT / "models" / "vectorizer.pkl"

# Page config
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")

# CSS styling
st.markdown(
    """
    <style>
    .reportview-container { background-color: #f7fbff; }
    .stTextArea textarea { border: 2px solid #4a90e2; border-radius: 8px; font-size: 15px; }
    .result-box { padding: 14px; border-radius: 10px; font-size: 18px; text-align:center; font-weight:700; }
    .phishing { background: linear-gradient(90deg,#ffd6d6,#ffbcbc); color:#8b0000; border:1px solid #cc0000; }
    .ham { background: linear-gradient(90deg,#dfffe0,#b8f0be); color:#005500; border:1px solid #009933; }
    .small-muted { color:#666; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🛡️ Phishing Email Detector")
st.markdown("Paste the email (subject and body) below and click **Classify**. The model will show the prediction and confidence.")

# Load artifacts
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECT_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()
if model is None or vectorizer is None:
    st.error("Model or vectorizer not found. Run `python -m src.train` to create them.")
    st.stop()

# Input
email_text = st.text_area("✉️ Paste email text here", height=240, placeholder="Include subject, body, links...")

# Classify button and logic
if st.button("🔍 Classify", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste an email to classify.")
    else:
        # Clean / preprocess text
        import re, string
        def clean_text(text):
            text = text.lower()  # lowercase
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
            text = re.sub(r"\S+@\S+", "", text)  # remove emails
            text = re.sub(r"\d+", "", text)  # remove numbers
            text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
            text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
            return text

        cleaned = clean_text(email_text)
        X = vectorizer.transform([cleaned])

        # Predict
        pred = model.predict(X)[0]  # label as string 'phishing' or 'ham'
        pred_label = str(pred)

        # Get probabilities
        probs = None
        labels = list(model.classes_)
        try:
            probs = model.predict_proba(X)[0]
        except Exception:
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                try:
                    s = scores.ravel()
                    if s.shape[0] == 1:
                        s = np.array([-s[0], s[0]])
                    exp = np.exp(s - np.max(s))
                    probs = exp / exp.sum()
                except Exception:
                    probs = None
            else:
                probs = None

        # Show result
        confidence_text = f" ({(probs.max()*100):.2f}% confidence)" if probs is not None else ""
        if pred_label.lower() == "phishing":
            st.markdown(f"<div class='result-box phishing'>🚨 PHISHING DETECTED{confidence_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box ham'>✅ HAM (Legitimate){confidence_text}</div>", unsafe_allow_html=True)

        # Probabilities table
        if probs is not None:
            label_probs = list(zip(labels, probs))
            label_probs.sort(key=lambda x: x[1], reverse=True)
            st.markdown("**Probabilities:**")
            for lbl, p in label_probs:
                st.write(f"- **{lbl}**: {p:.3f}")

# Sidebar info
st.sidebar.title("ℹ️ Info & Tips")
st.sidebar.write(
    """
- PHISHING EMAILS can trick users into revealing sensitive information.
- Always check the sender's email address and be cautious of unexpected requests.
- Hover over links to see the actual URL before clicking.
- Use security software to help detect and block phishing attempts.
"""
)
