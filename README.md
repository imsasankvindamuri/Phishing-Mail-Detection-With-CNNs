# Phishing Email Detection using CNN-DQA

Capstone project adapting Zhu et al.'s CNN-DQA hybrid approach from **phishing URLs** to **phishing emails**.

## Overview

This project implements a hybrid phishing detection system comparing three approaches:
- **Traditional ML baselines**: Naive Bayes and Random Forest classifiers with TF-IDF features
- **Deep learning**: CNN with Zipf-based Disorderly Quantized Attention (DQA) adapted for email text

**Key adaptation**: The original paper (Zhu et al., 2024) applied CNN-DQA to URL strings using character-level encoding and dynamic programming segmentation.
We adapt this architecture for natural language email text by using word-level tokenization and direct Zipf weighting, eliminating URL-specific preprocessing
while preserving the core attention mechanism.

**Research question**: Does a CNN architecture designed for phishing URLs transfer effectively to natural language emails?

## Project Structure

```
.
├── analytics/
│   ├── pictures/                         # Visualizations (Zipf distribution, model comparisons)
│   └── results/
│       ├── metrics/                      # Model performance CSVs (NB, RF, CNN-DQA)
│       ├── models/                       # Saved trained models (.joblib, .keras)
│       └── vocabulary.json               # 10k word vocabulary with corpus statistics
├── data/
│   ├── raw/                              # Original phishing_email.csv (82,487 emails)
│   └── processed/                        # train.csv (80%), val.csv (10%), test.csv (10%)
├── src/
│   ├── main.py                           # Main orchastration script
│   ├── preprocessing.py                  # Data preprocessing and stratified 80-10-10 split
│   ├── model/
│   │   ├── naive_bayes_baseline.py       # TF-IDF + Multinomial NB
│   │   ├── random_forest_classifier.py   # TF-IDF + Random Forest
│   │   └── cnn_dqa_classifier.py         # CNN with Zipf attention
│   └── utils/
│       ├── build_vocabulary.py           # Vocabulary construction + Zipf analysis
│       ├── tokenizer.py                  # Text → padded integer sequences
│       └── zipf_weightage.py             # Zipf-based attention weight computation
├── THEORY.md                             # Architectural adaptations and defense Q&A
└── README.md                             # This file
```

## Dataset

**Source**: Kaggle Phishing Email Dataset (82,487 emails)  
**License**: Creative Commons Attribution  
**Composition**: Compiled from Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin datasets  
**Preprocessing**: Already lowercased, stemmed, and cleaned (space-separated text, punctuation removed)  
**Class balance**: ~52% phishing, ~48% legitimate across all splits

**Citation**:  
Al-Subaiey et al., *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection*, arXiv:2405.11619 (2024).  
Available at: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

## Setup

### Requirements
- Python 3.11 or 3.12
- poetry (dependency management)
- tensorflow 2.20.0
- keras 3.13.1
- numpy 2.4.1
- scipy 1.17.0
- scikit-learn 1.8.0
- pandas 2.3.3
- matplotlib 3.10.8
- GPU recommended for CNN training (Google Colab T4 used in this project)

### Installation

```bash
# Clone repository
git clone https://github.com/imsasankvindamuri/Phishing-Mail-Detection-With-CNNs.git
cd Phishing-Mail-Detection-With-CNNs

# Install dependencies
poetry install

# Download dataset from Kaggle and place in data/raw/
# Expected file: data/raw/phishing_email.csv
```

## Usage

### 1. Data Preprocessing

```bash
poetry run python3 src/preprocessing.py
```
Splits raw data into train (80%), validation (10%), test (10%) with balanced class distribution.

### 2. Build Vocabulary

```bash
poetry run python3 src/utils/build_vocabulary.py
```
- Creates vocabulary of top 9,998 words (reserves indices 0=PAD, 1=UNK)
- Generates Zipf's law distribution plot (`analytics/pictures/zipf_distribution.png`)
- Saves vocabulary with corpus statistics to `analytics/results/vocabulary.json`

### 3. Train Models

```bash
# Naive Bayes (CPU, ~0.02s training time)
poetry run python3 src/model/naive_bayes_baseline.py

# Random Forest (CPU, ~32s training time)
poetry run python3 src/model/random_forest_classifier.py

# CNN-DQA (GPU recommended, ~55s on Colab T4)
poetry run python3 src/model/cnn_dqa_classifier.py
```

**Note**: CNN training requires GPU. Use Google Colab with T4 GPU for efficient training:
1. Upload project to GitHub
2. Clone in Colab, install dependencies
3. Runtime → Change runtime type → T4 GPU
4. Run training script

**While training on CPU is theoretically possible, it is not recommended.**

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference (ms/email) |
|-------|----------|-----------|--------|----------|---------------|----------------------|
| **Naive Bayes** | 97.56% | 98.62% | 96.67% | 97.63% | 0.023s | 0.00035 |
| **Random Forest** | 98.47% | 98.58% | 98.48% | 98.53% | 32.2s | 0.0225 |
| **CNN-DQA** | 98.97% | 98.70% | 99.32% | 99.01% | 54.6s | 0.143 |

### Key Findings

1. **CNN-DQA achieves highest accuracy** (98.97%), beating Random Forest by 0.5% and Naive Bayes by 1.4%

2. **CNN-DQA has highest recall** (99.32%), making it best for security applications where missing phishing emails is costly

3. **Computational tradeoffs**:
   - Naive Bayes: Fastest (2400x faster training than CNN), suitable for resource-constrained deployment
   - Random Forest: Balanced performance, but 1400x slower training than NB for only 1% accuracy gain
   - CNN-DQA: Best accuracy/recall, but 2.4x slower training and 6.4x slower inference than RF

4. **Domain transfer successful**: CNN-DQA architecture designed for URLs generalizes effectively to natural language emails

### Performance Analysis

**Why CNN-DQA wins on recall**:
- Zipf attention amplifies rare words ("verify", "urgent", "transfer") that are strong phishing indicators
- Dual-kernel CNN (sizes 3 & 5) captures both short and long phishing phrases
- Deep architecture learns complex patterns beyond TF-IDF bag-of-words

**Production deployment considerations**:
- **High-throughput systems**: Use Naive Bayes (fastest inference)
- **Balanced accuracy/speed**: Use Random Forest
- **Maximum accuracy needed**: Use CNN-DQA (accepts slower inference for better catch rate)

## Model Status and Future Work

### Current Implementation: Baseline Validation
This implementation uses **baseline hyperparameters** from Zhu et al. (2024) without tuning:
- Vocabulary size: 10,000 words
- Embedding dimension: 128
- CNN kernel sizes: [3, 5]
- Filters per kernel: 64
- Sequence length: 500 tokens

**Rationale:** Domain transfer experiments should validate architecture viability before optimization. 
Our results (98.97% accuracy, 99.32% recall) confirm CNN-DQA transfers effectively from URLs to emails.

### Planned Hyperparameter Optimization

- Grid search over vocabulary sizes: [5000, 10000, 15000, 20000]
- Test embedding dimensions: [64, 128, 256]
- Experiment with kernel combinations: [3,5], [2,4,6], [3,5,7]
- Optimize sequence length based on email length distribution
- Use Keras Tuner or Optuna for Bayesian optimization

**Expected outcome:** Further accuracy improvements while maintaining computational efficiency.

## Implementation Details

### CNN-DQA Architecture

- **Input**: Tokenized email sequences (max length 500)
- **Embedding**: 10,000 vocab → 128-dimensional dense vectors
- **Zipf Attention**: Weight embeddings by word rarity (`weight = ln(rank) + 1`)
- **Dual-kernel CNN**: Parallel 1D convolutions with kernel sizes 3 and 5 (64 filters each)
- **Pooling**: GlobalMaxPooling extracts strongest activations per filter
- **Classification**: Concatenate features → Dense(128) → Dropout(0.5) → Sigmoid output

### Zipf Weighting Formula

```
For word at vocabulary index i:
  - If i = 0 (PAD): weight = 0.0
  - If i = 1 (UNK): weight = ln(5000) + 1 ≈ 9.5 (median)
  - If i ≥ 2 (vocab): weight = ln(i) + 1

Result: Rare words get ~6x more weight than common words
```

### Key Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Vocabulary size | 10,000 | Covers 98%+ of corpus, manageable embedding size |
| Sequence length | 500 tokens | Covers mean (160) + 3σ of email lengths |
| Embedding dim | 128 | From paper (Table 9: p=128) |
| CNN kernels | [3, 5] | Paper Section 5.2 (q1=3, q2=5) |
| Filters per kernel | 64 | Paper Table 10 (K=64) |
| Batch size | 256 | Standard for NLP tasks |
| Learning rate | 0.001 | Adam optimizer default |

## Architectural Adaptations

See `THEORY.md` for comprehensive explanation of how we adapted Zhu et al.'s URL-based model to emails:

1. **Tokenization**: Character-level → Word-level (emails have natural word boundaries)
2. **Segmentation**: Skip dynamic programming (no ambiguous word boundaries in natural language)
3. **Attention**: Direct Zipf weighting on vocabulary ranks (preserve core insight, remove URL-specific machinery)
4. **Features**: Deep-only (no artificial URL/HTML features in preprocessed text dataset)

## Defense Panel Q&A

**Q: Why not use character-level encoding like the paper?**  
A: Character-level is for URLs without spaces. Emails use natural language with clear word boundaries, so word-level tokenization (standard in NLP) is appropriate.

**Q: Why skip dynamic programming segmentation?**  
A: DP solves URL word segmentation ("loginpage" → "login page"), which doesn't exist in emails with whitespace.

**Q: Is this still CNN-DQA without full DP machinery?**  
A: Yes. DQA's core contribution is Zipf-based attention weighting (Equation 14), which we preserve. DP is preprocessing specific to URLs, not the attention mechanism itself.

**Q: Why use CNN if Random Forest is faster?**  
A: Research question was domain transfer validation. CNN achieves 0.5% higher accuracy with significantly better recall (99.32% vs 98.48%)—important for security applications
where false negatives are costly.

## Limitations and Future Work

### Current Limitations
1. **Preprocessing dependency**: Requires pre-stemmed, cleaned text (dataset constraint)
2. **Computational cost**: 404x slower inference than Naive Bayes
3. **No artificial features**: Dataset lacks email metadata (sender domain, headers, attachments)
4. **Single language**: English-only (dataset constraint)

### Future Extensions
1. **Add metadata features**: Sender domain reputation, subject line patterns, attachment presence
2. **Transformer baseline**: Compare against BERT/RoBERTa embeddings
3. **Multi-modal approach**: Combine text CNN with header/metadata features
4. **Adversarial robustness**: Test against character substitution attacks
5. **Real-time preprocessing**: Build end-to-end pipeline accepting raw email text

## References

### Primary Paper
Zhu, E., Cheng, K., Zhang, Z., & Wang, H. (2024). PDHF: Effective phishing detection model combining optimal artificial and automatic deep features.
*Computers & Security*, 136, 103561.  
https://doi.org/10.1016/j.cose.2023.103561

### Dataset
Al-Subaiey et al. (2024). Phishing Email Dataset. Kaggle.  
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

### Related Work
- Xiao et al. (2020). CNN-MHSA: Multi-head self-attention for phishing detection. *Neural Networks*, 125, 303-312.
- Chai et al. (2022). Multi-modal hierarchical attention for phishing websites. *IEEE Trans. Dependable Secure Comput.*, 19(2), 790-803.

## License

This project is for academic use only. Dataset is licensed under Creative Commons Attribution.
