import os
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_NLTK_DATA = os.path.join(BASE_DIR, "nltk_data")

if os.path.isdir(LOCAL_NLTK_DATA):
    nltk.data.path.insert(0, LOCAL_NLTK_DATA)


def _ensure_nltk_resource(resource_path: str, package: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)


_ensure_nltk_resource("tokenizers/punkt", "punkt")
_ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
_ensure_nltk_resource("corpora/stopwords", "stopwords")

if "history" not in st.session_state:
    st.session_state.history = []

ps = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))
_PUNCT = set(string.punctuation)


def transformText(text):
    if text is None:
        return ""

    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in _STOPWORDS and t not in _PUNCT]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


model = pickle.load(open(os.path.join(BASE_DIR, "MultinomialNB.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "Vectorizer.pkl"), "rb"))

st.set_page_config(page_title="Spam Detector", layout="wide")
st.title("SPAAAAAAAM")
st.markdown("Check if any text is Spam or not(ham)")

with st.sidebar:
    st.header("How it works")
    st.write(
        """
    - type or paste a message
    - Click check
    - See predicted label and confidence
    """
    )

st.subheader("Enter a text to check")
text = st.text_area("Message", height=150, placeholder="write or paste to check for spam")

col1, col2 = st.columns([1, 3])

with col1:
    check_btn = st.button("check")
with col2:
    st.write("")


if check_btn:
    if not text or not text.strip():
        st.warning("Please enter some text to check.")
    else:
        text_cleaned = transformText(text)
        X = vectorizer.transform([text_cleaned])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba.max())

        if pred == 1:
            st.error("🚨 **SPAM** 🚨")
        else:
            st.success("✅ **Not Spam (Ham)**")
        st.write(f"Confidence: {confidence:.1%}")

        st.session_state.history.insert(
            0,
            {
                "text": text,
                "label": "Spam" if pred == 1 else "Not Spam",
                "confidence": confidence,
            },
        )
        st.session_state.history = st.session_state.history[:5]

if st.session_state.history:
    st.subheader("Recent Checks")
    for item in st.session_state.history:
        st.write(f"**{item['label']}** ({item['confidence']:.1%}) — {item['text']}")
