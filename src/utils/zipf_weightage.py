"""Zipf's Law attention weighting for token sequences"""
import numpy as np
from numpy.typing import NDArray

PAD_IDX = 0
UNK_IDX = 1

def compute_zipf_weights(token_sequence: NDArray[np.int32]) -> NDArray[np.float32]:
    """
    Compute Zipf-based attention weights for a tokenized sequence.
    
    Formula: weight = ln(rank) + 1
    
    Args:
        token_sequence: Array of token indices, shape (seq_len,) or (batch, seq_len)
   
    Returns:
        Attention weights, same shape as input
    """
    weights = np.zeros_like(token_sequence, dtype=np.float32)
    
    # PAD tokens: weight = 0
    weights[token_sequence == PAD_IDX] = 0.0
    
    # UNK tokens: assign median weight
    weights[token_sequence == UNK_IDX] = np.log(5000) + 1
    
    # Vocabulary words: ln(index) + 1
    vocab_mask = token_sequence >= 2
    weights[vocab_mask] = np.log(token_sequence[vocab_mask].astype(np.float32)) + 1    
    return weights


if __name__ == "__main__":
    # Smoke test
    test_sequence = np.array([0, 1, 2, 100, 5000, 9999], dtype=np.int32)
    weights = compute_zipf_weights(test_sequence)
    
    print("Token indices:", test_sequence)
    print("Zipf weights:", weights)
