# src/model/cnn_dqa_classifier.py

"""CNN-DQA model for phishing email detection with Zipf attention"""

import time
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

from src.utils.tokenizer import tokenize_text
from src.utils.zipf_weightage import compute_zipf_weights

# Paths
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
VOCAB_PATH = "analytics/results/vocabulary.json"
MODEL_SAVE_PATH = "analytics/results/models/cnn_dqa_model.keras"
METRICS_SAVE_PATH = "analytics/results/metrics/cnn_dqa.csv"

# Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
MAX_LENGTH = 500
KERNEL_SIZES = [3, 5]
NUM_FILTERS = 64
DROPOUT_RATE = 0.5
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001


class ZipfAttentionLayer(layers.Layer):
    """Applies Zipf-based attention weights to embeddings"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        embeddings, zipf_weights = inputs
        # embeddings: (batch, 500, 128)
        # zipf_weights: (batch, 500)
        
        # Expand weights to match embedding dimension
        zipf_weights = tf.expand_dims(zipf_weights, -1)  # (batch, 500, 1)
        
        # Element-wise multiplication (broadcasting)
        weighted = embeddings * zipf_weights
        
        return weighted
    
    def get_config(self):
        return super().get_config()


def create_cnn_dqa_model(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    max_length=MAX_LENGTH,
    kernel_sizes=KERNEL_SIZES,
    num_filters=NUM_FILTERS,
    dropout_rate=DROPOUT_RATE
):
    """
    Build CNN-DQA model with Zipf attention.
    
    Architecture:
      1. Embedding layer (vocab → dense vectors)
      2. Zipf attention (weight embeddings by word rarity)
      3. Dual-kernel 1D CNN (kernel sizes 3 and 5)
      4. GlobalMaxPooling (extract strongest features)
      5. Concatenate both CNN branches
      6. Dense layer with dropout
      7. Binary classification head
    """
    
    # Input 1: Token sequences
    input_tokens = layers.Input(shape=(max_length,), dtype=tf.int32, name='token_input')
    
    # Input 2: Zipf weights
    input_zipf = layers.Input(shape=(max_length,), dtype=tf.float32, name='zipf_input')
    
    # Embedding layer
    embeddings = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        mask_zero=True,  # Mask padding tokens
        name='embedding'
    )(input_tokens)
    
    # Apply Zipf attention
    weighted_embeddings = ZipfAttentionLayer(name='zipf_attention')([embeddings, input_zipf])
    
    # Dual-kernel CNN branches
    conv_outputs = []
    for kernel_size in kernel_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_k{kernel_size}'
        )(weighted_embeddings)
        
        # Global max pooling
        pooled = layers.GlobalMaxPooling1D(name=f'maxpool_k{kernel_size}')(conv)
        conv_outputs.append(pooled)
    
    # Concatenate both CNN branches
    concatenated = layers.Concatenate(name='concat_features')(conv_outputs)
    
    # Dense layer with dropout
    dense = layers.Dense(128, activation='relu', name='dense_features')(concatenated)
    dropout = layers.Dropout(dropout_rate, name='dropout')(dense)
    
    # Binary classification head
    output = layers.Dense(1, activation='sigmoid', name='output')(dropout)
    
    # Build model
    model = keras.Model(
        inputs=[input_tokens, input_zipf],
        outputs=output,
        name='CNN_DQA'
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def load_and_preprocess_data(csv_path, vocab, max_samples=None):
    """
    Load CSV and preprocess into token arrays and Zipf weights.
    
    Args:
        csv_path: Path to CSV file
        vocab: Word-to-index dictionary
        max_samples: Optional limit for testing (use None for full dataset)
    
    Returns:
        tokens: np.array of shape (N, 500)
        zipf_weights: np.array of shape (N, 500)
        labels: np.array of shape (N,)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    n_samples = len(df)
    tokens = np.zeros((n_samples, MAX_LENGTH), dtype=np.int32)
    zipf_weights = np.zeros((n_samples, MAX_LENGTH), dtype=np.float32)
    labels = df['label'].values
    
    print(f"Tokenizing {n_samples} emails...")
    for i, text in enumerate(df['text']):
        if i % 10000 == 0:
            print(f"  Processed {i}/{n_samples}")
        
        # Tokenize
        token_seq = tokenize_text(str(text), vocab, max_length=MAX_LENGTH)
        tokens[i] = token_seq
        
        # Compute Zipf weights
        zipf_seq = compute_zipf_weights(token_seq)
        zipf_weights[i] = zipf_seq
    
    print(f"Preprocessing complete: {n_samples} samples")
    return tokens, zipf_weights, labels


def main():
    print("=" * 60)
    print("CNN-DQA Phishing Email Classifier")
    print("=" * 60)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    with open(VOCAB_PATH, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['word_to_idx']
    print(f"Vocabulary size: {len(vocab) + 2} (including PAD and UNK)")
    
    # Load and preprocess data
    print("\n" + "=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    X_train_tokens, X_train_zipf, y_train = load_and_preprocess_data(TRAIN_PATH, vocab)
    
    print("\n" + "=" * 60)
    print("LOADING VALIDATION DATA")
    print("=" * 60)
    X_val_tokens, X_val_zipf, y_val = load_and_preprocess_data(VAL_PATH, vocab)
    
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    X_test_tokens, X_test_zipf, y_test = load_and_preprocess_data(TEST_PATH, vocab)
    
    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    model = create_cnn_dqa_model()
    model.summary()
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    start_train = time.time()
    history = model.fit(
        x=[X_train_tokens, X_train_zipf],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val_tokens, X_val_zipf], y_val),
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_train
    
    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    start_test = time.time()
    y_pred_probs = model.predict([X_test_tokens, X_test_zipf], batch_size=BATCH_SIZE)
    test_time = time.time() - start_test
    
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_time_sec": train_time,
        "avg_inference_ms": (test_time / len(y_test)) * 1000
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL AND METRICS")
    print("=" * 60)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(METRICS_SAVE_PATH, index=False)
    print(f"Metrics saved to: {METRICS_SAVE_PATH}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
