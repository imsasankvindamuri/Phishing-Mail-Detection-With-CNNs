# Phishing-Mail-Detection-With-CNNs

Capstone project adapting Zhu et al.'s PDHF hybrid CNN approach for phishing **emails** instead of URLs.

## Structure
- **feature_engineering/** – Artificial + deep feature extraction
- **model/** – Random forest classifier for hybrid features
- **data/** – Raw + processed datasets

## Setup

```
poetry install
poetry run python src/main.py
```

## Dataset

This project uses a publicly available phishing email dataset released under a Creative Commons Attribution
license. The dataset contains approximately 82,500 emails (phishing and legitimate) compiled from multiple
sources including Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin datasets.

*Citation:*

**Al-Subaiey et al., Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection,
arXiv:2405.11619 (2024).**  
`https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset`
