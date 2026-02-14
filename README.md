# Code Comment Quality Analyzer (NLP)

Classify **code comment semantic quality** (Poor / Average / Excellent) using the pipeline from the exploration notebook: CodeSearchNet data, NLTK preprocessing, semantic scoring, TF–IDF features, and a scikit-learn classifier.

## Project layout

- **`notebooks/exploration_and_modeling.ipynb`** – EDA, preprocessing, and model experiments (source of the pipeline).
- **`src/`** – Production code split from the notebook:
  - **`config.py`** – Paths, model and data constants, semantic thresholds and keyword lists.
  - **`data.py`** – Load CodeSearchNet from HuggingFace, optional sampling, add `clean_comment` and `semantic_quality`.
  - **`preprocessing.py`** – NLTK text preprocessing, semantic score/label, TF–IDF vectorizer.
  - **`train.py`** – Train/test split, train Naive Bayes / Logistic Regression / Linear SVM, save best model and vectorizer.
  - **`predict.py`** – Load model and vectorizer, preprocess input, return predicted label.
  - **`vectorizer.py`** – Re-exports vectorizer helpers.
  - **`utils.py`** – Re-exports preprocessing helpers.
- **`app.py`** – Streamlit UI to type a comment and get the predicted quality.
- **`data/`** – Generated `dataset.csv` (created by the data step).
- **`models/`** – Saved `best_model.pkl` and `tfidf_vectorizer.pkl` (created by training).

## Setup

```bash
cd code-comment-quality-analyzer-nlp
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare data (optional if you already have `data/dataset.csv`)

Download CodeSearchNet and build the training CSV with preprocessed text and semantic labels:

```bash
python -m src.data
```

By default this uses a sample size set in `src/config.py` (`SAMPLE_SIZE`). To change it or use the full dataset, edit `SAMPLE_SIZE` (or set to `None`) and run again. The script writes `data/dataset.csv`.

## 2. Train the model

If `data/dataset.csv` is missing, training will try to create it automatically (same as running `python -m src.data`). Then it trains and saves the best classifier and the TF–IDF vectorizer:

```bash
python -m src.train
```

Outputs:

- `models/best_model.pkl`
- `models/tfidf_vectorizer.pkl`

## 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal. Enter a code comment and click **Predict** to see the predicted semantic quality (Poor / Average / Excellent).

## Summary of the pipeline (from the notebook)

1. **Data**: CodeSearchNet “pair” split → `code` and `comment`; optional sampling.
2. **Preprocessing**: Lowercase, keep letters, NLTK stopwords + lemmatization → `clean_text` / `clean_comment`.
3. **Labels**: Semantic score (keyword + uniqueness + length) → thresholds → `semantic_quality` (Poor / Average / Excellent).
4. **Features**: TF–IDF (e.g. 5k features, 1–2 grams) on `clean_comment`.
5. **Model**: Train/test split, then Naive Bayes, Logistic Regression, Linear SVM; keep the best by accuracy and save it with the vectorizer.
6. **Predict**: Preprocess input text → same TF–IDF transform → trained model → predicted label.

## Requirements

- Python 3.8+
- See `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `joblib`, `nltk`, `datasets`.
