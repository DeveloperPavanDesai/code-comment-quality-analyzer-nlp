import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from .config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH, TEST_SIZE, RANDOM_STATE, MODEL_DIR
from .preprocessing import load_data, create_vectorizer

# Use data module to create dataset if CSV missing
try:
    from . import data
    _has_data = True
except Exception:
    _has_data = False

def train_model():
    if not os.path.isfile(DATA_PATH) and _has_data:
        print("Dataset not found. Downloading and preparing...")
        data.load_or_create_dataset(force_download=True)
    df = load_data(DATA_PATH)

    X_text = df["clean_comment"]
    y = df["semantic_quality"]

    vectorizer = create_vectorizer()
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB()
    }

    best_acc = 0
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nBest model saved with accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()