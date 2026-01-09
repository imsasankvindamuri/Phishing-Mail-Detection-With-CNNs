"""Build vocabulary and analyze corpus statistics"""
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

def build_vocabulary(train_path, vocab_size=10000, output_dir='analytics/results'):
    """Build vocabulary from training data"""
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    
    # Count words (CHANGED: text_combined → text)
    print("Counting words...")
    word_counts = Counter()
    email_lengths = []
    
    for text in train_df['text']:  # <-- FIXED
        words = str(text).lower().split()
        word_counts.update(words)
        email_lengths.append(len(words))
    
    # Get most common
    most_common = word_counts.most_common(vocab_size)
    
    # Create vocab dict
    vocab_data = {
        'words': [word for word, _ in most_common],
        'word_to_idx': {word: idx+1 for idx, (word, _) in enumerate(most_common)},
        'vocab_size': vocab_size,
        'total_unique_words': len(word_counts),
        'total_tokens': sum(word_counts.values()),
        'avg_email_length': np.mean(email_lengths),
        'max_email_length': max(email_lengths),
        'min_email_length': min(email_lengths)
    }
    
    # Save
    output_path = Path(output_dir) / 'vocabulary.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"\nVocabulary Statistics:")
    print(f"  Total unique words: {vocab_data['total_unique_words']:,}")
    print(f"  Vocabulary size: {vocab_data['vocab_size']:,}")
    print(f"  Total tokens: {vocab_data['total_tokens']:,}")
    print(f"  Avg email length: {vocab_data['avg_email_length']:.1f} words")
    print(f"  Max email length: {vocab_data['max_email_length']:,} words")
    
    # Plot Zipf distribution
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
    plt.title('Word Frequency Distribution - Zipf\'s Law')
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'zipf_distribution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved Zipf plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    vocab = build_vocabulary('data/processed/train.csv')
