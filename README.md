# Phishing Email Detection using CNN-DQA

Capstone project adapting Zhu et al.'s PDHF hybrid CNN approach from **phishing URLs** to **phishing emails**.

## Overview

This project implements a hybrid phishing detection system that combines:
- **Traditional ML baselines**: Naive Bayes and Random Forest classifiers
- **Deep learning**: CNN with simplified Disorderly Quantized Attention (DQA) adapted for email text

**Key adaptation**: The original paper (Zhu et al., 2024) applied CNN-DQA to URL strings. We adapt this architecture for natural language email text
by using word-level tokenization instead of character-level encoding.

## Project Structure

```
.
├── analytics/
│   ├── pictures/          # Visualizations (Zipf distribution, performance graphs)
│   └── results/
│       ├── metrics/       # Model performance CSVs
│       ├── models/        # Saved model files (.joblib, .keras)
│       └── vocabulary.json # 10k word vocabulary with statistics
├── data/
│   ├── raw/              # Original phishing_email.csv (not in repo)
│   └── processed/        # train.csv, val.csv, test.csv splits
├── src/
│   ├── feature_engineering/
│   │   ├── artificial_features.py  # Feature extraction (not used in current approach)
│   │   └── deep_features/
│   │       └── cnn_dqa_model.py   # CNN-DQA architecture
│   ├── model/
│   │   ├── naive_bayes_baseline.py      # NB classifier
│   │   └── random_forest_classifier.py  # RF classifier
│   └── utils/
│       ├── build_vocabulary.py   # Vocabulary construction
│       └── tokenizer.py          # Text → padded sequences
├── THEORY.md             # Architectural decisions and paper adaptations
└── README.md             # This file
```

## Dataset

This project uses a publicly available phishing email dataset released under a Creative Commons Attribution license. The dataset contains approximately
82,500 emails (phishing and legitimate) compiled from multiple sources including Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin datasets.

**Citation:**  
Al-Subaiey et al., *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection*, arXiv:2405.11619 (2024).  
Available at: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

**Preprocessing:** The Kaggle dataset is already lowercased, stemmed, and cleaned. Text is space-separated with punctuation removed.

## Setup

### Requirements
- Python 3.11+
- Poetry (dependency management)

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
Splits raw data into train (80%), validation (10%), test (10%) sets.

### 2. Build Vocabulary

```bash
poetry run python3 src/utils/build_vocabulary.py
```
Creates vocabulary of 9,998 most frequent words (reserves index 0=PAD, 1=UNK).

### 3. Train Baselines

```bash
# Naive Bayes
poetry run python3 src/model/naive_bayes_baseline.py

# Random Forest
poetry run python3 src/model/random_forest_classifier.py
```

### 4. Train CNN-DQA

```bash
poetry run python3 src/train_cnn_dqa.py
```
*Note: Training requires GPU. Use Google Colab for free T4 GPU access.*

### 5. Evaluate Models

```bash
poetry run python3 src/evaluate.py
```
Generates comparison metrics and visualizations.

## Results (Preliminary)

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Naive Bayes | 97.56% | 98.62% | 96.67% | 97.63% | 0.023s |
| Random Forest | 98.47% | 98.58% | 98.48% | 98.53% | 32.2s |
| CNN-DQA | TBD | TBD | TBD | TBD | TBD |

**Key Finding:** Random Forest's 1% accuracy improvement over Naive Bayes comes at 1400x computational cost. CNN-DQA evaluation will determine if deep learning
justifies its complexity.

## Implementation Notes

- **Vocabulary size:** 10,000 words (including PAD and UNK tokens)
- **Sequence length:** 500 tokens (covers mean + 3σ of email lengths)
- **CNN architecture:** Dual-kernel (sizes 3 & 5), 64 filters each
- **Embedding dimension:** 128
- **Attention mechanism:** Simplified Zipf-based weighting (see THEORY.md)

## References

**Primary Paper:**  
Zhu, E., Cheng, K., Zhang, Z., & Wang, H. (2024). PDHF: Effective phishing detection model combining optimal artificial and automatic deep features.
*Computers & Security*, 136, 103561.

**Dataset:**  
Al-Subaiey et al. (2024). Phishing Email Dataset. Kaggle. https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
