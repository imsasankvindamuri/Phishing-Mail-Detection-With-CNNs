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
