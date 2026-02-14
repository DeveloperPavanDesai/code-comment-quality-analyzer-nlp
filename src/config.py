import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Optional: limit rows for faster training (None = use full dataset)
SAMPLE_SIZE = 100_000

# Semantic quality score thresholds (from notebook percentiles)
SEMANTIC_POOR_THRESHOLD = 2.544444
SEMANTIC_EXCELLENT_THRESHOLD = 4.05

GOOD_KEYWORDS = [
    "return", "calculate", "compute", "validate", "check",
    "initialize", "convert", "parse", "generate",
    "create", "update", "extract", "handle", "process",
]
BAD_KEYWORDS = [
    "temp", "stuff", "thing", "helper", "test",
    "fix", "misc", "todo", "bug",
]