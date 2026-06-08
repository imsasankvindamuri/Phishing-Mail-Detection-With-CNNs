import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Local imports
from src.utils.tokenizer import tokenize_text
from src.utils.zipf_weightage import compute_zipf_weights
from src.model.cnn_dqa_classifier import ZipfAttentionLayer

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data/processed/test.csv"
MODEL_DIR = BASE_DIR / "analytics/results/models"
VOCAB_PATH = BASE_DIR / "analytics/results/vocabulary.json"
OUTPUT_DIR = BASE_DIR / "analytics/pictures"

def load_test_data():
    print(f"Loading test data from {TEST_PATH}...")
    df = pd.read_csv(TEST_PATH)
    return df['text'].astype(str).tolist(), df['label'].values

def evaluate_nb_rf(texts, labels, model_name):
    print(f"Evaluating {model_name.upper()}...")
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    model_path = MODEL_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return None
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    
    return y_pred

def evaluate_cnn(texts, labels):
    print("Evaluating CNN-DQA...")
    model_path = MODEL_DIR / "cnn_dqa_model.keras"
    
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return None
    
    # Load vocab
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)['word_to_idx']
    
    # Load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ZipfAttentionLayer": ZipfAttentionLayer}
    )
    
    # Preprocess
    n_samples = len(texts)
    max_length = 500
    tokens = np.zeros((n_samples, max_length), dtype=np.int32)
    zipf_weights = np.zeros((n_samples, max_length), dtype=np.float32)
    
    print(f"Preprocessing {n_samples} samples for CNN...")
    for i, text in enumerate(texts):
        token_seq = tokenize_text(text, vocab, max_length=max_length)
        tokens[i] = token_seq
        zipf_weights[i] = compute_zipf_weights(token_seq)
    
    # Predict
    y_pred_probs = model.predict([tokens, zipf_weights], batch_size=64, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    return y_pred

def plot_cm(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f'Confusion Matrix: {model_name.upper()}')
    
    save_path = OUTPUT_DIR / f'cm_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    texts, labels = load_test_data()
    
    models = {
        'naive_bayes': 'nb',
        'random_forest': 'rf'
    }
    
    for name, key in models.items():
        y_pred = evaluate_nb_rf(texts, labels, name)
        if y_pred is not None:
            plot_cm(labels, y_pred, key)
    
    # CNN evaluation
    y_pred_cnn = evaluate_cnn(texts, labels)
    if y_pred_cnn is not None:
        plot_cm(labels, y_pred_cnn, 'cnn')

if __name__ == "__main__":
    main()
