"""
Text preprocessing and semantic quality scoring (aligned with exploration notebook).
"""
import re
import ssl

# Avoid NLTK SSL errors on macOS when downloading/loading corpora
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    MAX_FEATURES,
    NGRAM_RANGE,
    SEMANTIC_POOR_THRESHOLD,
    SEMANTIC_EXCELLENT_THRESHOLD,
    GOOD_KEYWORDS,
    BAD_KEYWORDS,
)

# Use NLTK if available; otherwise fall back to simple preprocessing (no stopwords/lemmatization)
_USE_NLTK = None


def _ensure_nltk():
    """Download NLTK data if needed. Set _USE_NLTK to False if NLTK is unavailable."""
    global _USE_NLTK
    if _USE_NLTK is False:
        return
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stopwords.words("english")
        WordNetLemmatizer().lemmatize("test")
        _USE_NLTK = True
    except Exception:
        _USE_NLTK = False


def preprocess_text(text):
    """Lowercase, keep letters only, remove stopwords, lemmatize (matches notebook)."""
    _ensure_nltk()
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    if _USE_NLTK:
        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            tokens = [w for w in tokens if w not in stop_words]
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
        except Exception:
            pass
    return " ".join(tokens)


def semantic_score(text):
    """Score based on good/bad keywords, uniqueness, avg word length (notebook logic)."""
    words = str(text).split()
    word_count = len(words)
    if word_count == 0:
        return 0.0
    unique_ratio = len(set(words)) / word_count
    good_count = sum(1 for w in words if w in GOOD_KEYWORDS)
    bad_count = sum(1 for w in words if w in BAD_KEYWORDS)
    avg_word_len = np.mean([len(w) for w in words])
    score = (
        unique_ratio * 2
        + good_count * 1.5
        - bad_count * 2
        + avg_word_len * 0.1
    )
    return float(score)


def semantic_label(score):
    """Map semantic score to Poor / Average / Excellent (notebook thresholds)."""
    if score <= SEMANTIC_POOR_THRESHOLD:
        return "Poor"
    if score <= SEMANTIC_EXCELLENT_THRESHOLD:
        return "Average"
    return "Excellent"


def load_data(path):
    """Load training DataFrame from CSV."""
    return pd.read_csv(path)


def create_vectorizer():
    """TfidfVectorizer matching notebook (no extra stop_words; text is preprocessed)."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )


def transform_text(vectorizer, texts):
    """Transform list of strings with fitted vectorizer."""
    return vectorizer.transform(texts)