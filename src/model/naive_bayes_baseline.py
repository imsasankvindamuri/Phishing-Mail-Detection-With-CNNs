import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

METRICS_PATH = Path("analytics/results/metrics/naive_bayes.csv")
MODEL_DIR = Path("analytics/results/models")
MODEL_PATH = MODEL_DIR / "naive_bayes.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"


def ensure_paths():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_paths()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()

    start_train = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = model.predict(X_test_vec)
    test_time = time.time() - start_test

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_time_sec": train_time,
        "avg_inference_ms": (test_time / len(y_test)) * 1000,
    }

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Naive Bayes results:", metrics)


if __name__ == "__main__":
    main()

