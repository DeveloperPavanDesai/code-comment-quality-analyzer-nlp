"""
Load trained model and vectorizer; preprocess input and return semantic quality label.
"""
import os
import joblib

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocessing import preprocess_text

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer
    if _model is None:
        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
            raise FileNotFoundError(
                "Model not found. Run training first: python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)


def predict(text):
    """Preprocess comment text and return predicted semantic quality (Poor / Average / Excellent)."""
    _load_artifacts()
    clean = preprocess_text(text)
    X = _vectorizer.transform([clean])
    return _model.predict(X)[0]