import numpy as np
from numpy.typing import NDArray

PAD_IDX = 0
UNK_IDX = 1

def tokenize_text(
    text: str,
    vocab: dict[str, int],
    max_length: int = 500
) -> NDArray[np.int32]:
    """
    Convert preprocessed email text into a fixed-length sequence of word indices.

    Indexing scheme:
      PAD = 0
      UNK = 1
      Vocabulary words start from index 2
    """

    tokens = text.strip().split()
    vector = np.array(
        [vocab.get(tok, UNK_IDX) for tok in tokens],
        dtype=np.int32
    )

    if len(vector) > max_length:
        vector = vector[:max_length]
    elif len(vector) < max_length:
        pad_len = max_length - len(vector)
        vector = np.pad(vector, (0, pad_len), constant_values=PAD_IDX)

    return vector


if __name__ == "__main__":
    pass
