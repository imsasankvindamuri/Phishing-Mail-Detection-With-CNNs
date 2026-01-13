"""Build vocabulary and analyze corpus statistics"""
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

PAD_IDX = 0
UNK_IDX = 1

def build_vocabulary(train_path, vocab_size=10000, output_dir='analytics/results'):
    """
    Build vocabulary from training data.

    Indexing scheme:
      PAD = 0
      UNK = 1
      Words = 2 .. vocab_size-1
    """

    print("Loading training data...")
    train_df = pd.read_csv(train_path)

    print("Counting words...")
    word_counts = Counter()
    email_lengths = []

    for text in train_df['text']:
        words = str(text).split()
        word_counts.update(words)
        email_lengths.append(len(words))

    # Reserve 2 slots: PAD and UNK
    num_words = vocab_size - 2
    most_common = word_counts.most_common(num_words)

    vocab_data = {
        'words': [word for word, _ in most_common],
        'word_to_idx': {
            word: idx + 2  # shift by 2 to reserve PAD=0, UNK=1
            for idx, (word, _) in enumerate(most_common)
        },
        'pad_idx': PAD_IDX,
        'unk_idx': UNK_IDX,
        'vocab_size': vocab_size,
        'total_unique_words': len(word_counts),
        'total_tokens': sum(word_counts.values()),
        'avg_email_length': np.mean(email_lengths),
        'max_email_length': max(email_lengths),
        'min_email_length': min(email_lengths)
    }

    output_path = Path(output_dir) / 'vocabulary.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print("\nVocabulary Statistics:")
    print(f"  Total unique words: {vocab_data['total_unique_words']:,}")
    print(f"  Vocabulary size (incl PAD+UNK): {vocab_data['vocab_size']:,}")
    print(f"  Total tokens: {vocab_data['total_tokens']:,}")
    print(f"  Avg email length: {vocab_data['avg_email_length']:.1f} words")
    print(f"  Max email length: {vocab_data['max_email_length']:,} words")

    plot_zipf(word_counts, output_dir='analytics/pictures')

    return vocab_data


def plot_zipf(word_counts, top_n=1000, output_dir='analytics/pictures'):
    """Plot Zipf's law distribution"""
    frequencies = [count for _, count in word_counts.most_common(top_n)]
    ranks = np.arange(1, len(frequencies) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, 'b.', alpha=0.6)
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title("Word Frequency Distribution - Zipf's Law")
    plt.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'zipf_distribution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved Zipf plot to {output_path}")
    plt.close()


if __name__ == '__main__':
    vocab = build_vocabulary('data/processed/train.csv')
