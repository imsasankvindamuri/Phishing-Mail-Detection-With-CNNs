import argparse
import json
import joblib
import numpy as np
import re
import tensorflow as tf
from pathlib import Path

from utils.tokenizer import tokenize_text  # your existing tokenizer
from model.cnn_dqa_classifier import ZipfAttentionLayer
from src.utils.zipf_weightage import compute_zipf_weights

BASE_DIR = Path(__file__).resolve().parent.parent


# -------------------------
# Basic preprocessing
# -------------------------
def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Load models
# -------------------------
def load_nb_rf(model_name):
    model_path = BASE_DIR / "analytics/results/models"
    vectorizer = joblib.load(model_path / "tfidf_vectorizer.joblib")

    if model_name == "nb":
        model = joblib.load(model_path / "naive_bayes.joblib")
    else:
        model = joblib.load(model_path / "random_forest.joblib")

    return model, vectorizer


def load_cnn():
    model_path = BASE_DIR / "analytics/results/models/cnn_dqa_model.keras"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ZipfAttentionLayer": ZipfAttentionLayer}
    )
    return model

# -------------------------
# Prediction functions
# -------------------------
def predict_nb_rf(text, model_name):
    model, vectorizer = load_nb_rf(model_name)
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    return prob


def predict_cnn(text):
    model = load_cnn()

    # load vocab
    vocab_path = BASE_DIR / "analytics/results/vocabulary.json"
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data["word_to_idx"]

    # tokenize
    token_seq = tokenize_text(text, vocab)  # (500,)
    
    # compute zipf weights
    zipf_seq = compute_zipf_weights(token_seq)  # (500,)

    # add batch dimension
    token_seq = np.expand_dims(token_seq, axis=0)   # (1, 500)
    zipf_seq = np.expand_dims(zipf_seq, axis=0)     # (1, 500)

    # predict
    prob = model.predict([token_seq, zipf_seq], verbose=0)[0][0]
    return float(prob)

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Phishing Email Detection CLI")
    parser.add_argument("--model", choices=["nb", "rf", "cnn"], required=True)
    parser.add_argument("--text", help="Input email text")
    parser.add_argument("--file", help="Path to text file")

    args = parser.parse_args()

    if not args.text and not args.file:
        raise ValueError("Provide --text or --file")

    if args.file:
        with open(args.file) as f:
            raw_text = f.read()
    else:
        raw_text = args.text

    cleaned = basic_clean(raw_text)
    if len(cleaned) == 0:
        print("Input empty after cleaning.")
        return


    if args.model in ["nb", "rf"]:
        prob = predict_nb_rf(cleaned, args.model)
    else:
        prob = predict_cnn(cleaned)

    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"

    print("\n--- Prediction Result ---")
    print(f"Model:                   {args.model.upper()}")
    print(f"Probability of phishing: {prob:.4f}")
    print(f"Predicted label:         {label}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
