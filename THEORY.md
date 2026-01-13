# Theoretical Framework: Adapting CNN-DQA from URLs to Emails

## Executive Summary

This document explains our architectural decisions in adapting Zhu et al.'s CNN-DQA model from **phishing URL detection** to **phishing email detection**.
The key insight: URLs and natural language emails require different preprocessing strategies, but the core deep learning architecture transfers effectively.

---

## Original Paper Context

**Source:** Zhu, E., Cheng, K., Zhang, Z., & Wang, H. (2024). PDHF: Effective phishing detection model combining optimal artificial and automatic deep features.
*Computers & Security*, 136, 103561.

**Original Domain:** Phishing URL detection  
**Input Format:** URL strings (e.g., `http://example-phishing.com/login?user=abc`)  
**Key Challenge:** URLs lack natural word boundaries—characters run together without spaces

**Original Approach:**
1. **Character-level encoding:** One-hot encode each character (97 possible characters)
2. **Dynamic programming word segmentation:** Use Zipf's law + DP to break URLs into meaningful "words"
3. **Disorderly Quantized Attention (DQA):** Weight URL segments by their disorder/entropy
4. **Hybrid features:** Combine CNN-learned features with hand-crafted artificial features

---

## Our Adaptation: URLs → Emails

### Domain Differences

| Aspect | URLs | Emails |
|--------|------|--------|
| **Structure** | No whitespace, mixed case, special chars | Natural language with spaces |
| **Word boundaries** | Ambiguous (e.g., `loginpage` vs `login-page`) | Clear (whitespace-delimited) |
| **Vocabulary** | Domain names, paths, parameters | Natural language corpus |
| **Length** | Short (typically <200 chars) | Variable (mean ~160 words, max 107k) |

### Key Architectural Changes

#### 1. **Tokenization: Character-level → Word-level**

**Original (Section 5.1):**

```
URL: "example.com/login"
→ One-hot encode each character
→ Matrix: 97 × L (97 char types, L = URL length)
```

**Our Adaptation:**

```
Email: "click here to verify your account"
→ Split on whitespace
→ Map words to indices via vocabulary
→ Sequence: [523, 1847, 92, 3341, 284, 1092] (word IDs)
```

**Justification:**
- Emails already have natural word boundaries (spaces)
- Word-level tokenization is standard in NLP (BERT, GPT, etc.)
- Character-level would lose semantic information (e.g., "verify" vs "ver ify")

---

#### 2. **Word Segmentation: Skip Dynamic Programming**

**Original (Section 5.3.1-5.3.2):**
- Uses DP algorithm to segment URLs into words based on corpus frequency
- Example: `"loginpage"` → `["login", "page"]` (find optimal split)
- Necessary because URLs lack spaces

**Our Adaptation:**
- Skip DP segmentation entirely
- Emails already have whitespace, so we split on spaces
- No ambiguity in word boundaries

**Justification:**
- DP segmentation (Equations 13-15 in paper) solves a problem emails don't have
- Natural language has explicit word boundaries
- Simplification reduces complexity without losing information

---

#### 3. **Attention Mechanism: Simplified Zipf Weighting**

**Original DQA (Section 5.3):**
1. Segment URL using DP
2. Assign cost to each segment based on Zipf's law (Equation 14)
3. Normalize costs via sigmoid (Equation 18)
4. Apply attention to CNN features (Equation 20)

**Our Simplified Approach:**

- Apply Zipf's Law directly
- Use Equation 14: $\text{Attention Weight} = \ln{\text{rank(word)}} + 1$

**Justification:**
- Core insight preserved: rare words are more informative than common words
- Skip URL-specific DP machinery
- Directly apply Zipf weighting to vocabulary words

**Trade-off:**
- Lose: DP-based optimal segmentation (not needed for natural language)
- Keep: Frequency-based attention weighting (key contribution of paper)

---

#### 4. **Hybrid Features: Deep-Only vs. Artificial + Deep**

**Original PDHF Model:**
- Artificial features: 18 optimal features from 48 URL/HTML features
- Deep features: CNN-DQA learned from URLs
- Hybrid: Combine both using Random Forest

**Our Implementation (Current):**
- **Phase 1:** Deep features only (CNN-DQA on email text)
- **Phase 2 (Future):** Add artificial features (email metadata, sender domain, etc.)

**Justification:**
- Dataset is already preprocessed (stemmed, lowercased, cleaned)
- Focus on validating core CNN-DQA architecture first
- Artificial features can be added later if deep learning underperforms

---

## CNN Architecture Details

### Input Pipeline
```
Raw Email Text (string)
    ↓
Tokenization (word-level split)
    ↓
Vocabulary Lookup (word → index)
    ↓
Padding/Truncation (fixed length = 500)
    ↓
Embedding Layer (10k vocab → 128-dim vectors)
```

### Network Layers
```
Input: [batch_size, 500] (padded word indices)
    ↓
Embedding: [batch_size, 500, 128]
    ↓
┌─────────────────────┬─────────────────────┐
│ Conv1D (kernel=3)   │ Conv1D (kernel=5)   │
│ 64 filters          │ 64 filters          │
│ ReLU activation     │ ReLU activation     │
└──────────┬──────────┴──────────┬──────────┘
           ↓                     ↓
   GlobalMaxPool1D       GlobalMaxPool1D
           ↓                     ↓
           └──────── Concatenate ────────┘
                       ↓
                Dense(128, ReLU)
                       ↓
                  Dropout(0.5)
                       ↓
                Dense(1, Sigmoid)
                       ↓
              Output: [0, 1] (phishing score)
```

### Hyperparameters (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Vocabulary size | 10,000 | Paper uses 95 chars; we use 10k words |
| Embedding dim | 128 | Table 9 (paper uses p=128) |
| CNN kernels | 3, 5 | Section 5.2 (q1=3, Convolution q2=5) |
| Filters per kernel | 64 | Table 10 (paper uses K=64) |
| Sequence length | 500 | Covers mean (160) + 3σ |
| Batch size | 256 | Standard for NLP tasks |

---

## Justification for Defense Panel

### Q: Why not use character-level encoding like the paper?

**A:** Character-level encoding is appropriate for URLs, which lack spaces and have ambiguous word boundaries (e.g., `"loginpage"` could be `"log in page"` or `"login page"`).
Emails are natural language with clear whitespace, so word-level tokenization is standard practice (used in BERT, GPT, Word2Vec, etc.).

### Q: Why skip the dynamic programming segmentation?

**A:** The DP algorithm (Section 5.3.1-5.3.2) solves URL word segmentation—a problem that doesn't exist in natural language. Emails already have explicit word boundaries.
Skipping DP reduces complexity without losing information.

### Q: Is this still CNN-DQA if you simplified the attention?

**A:** Yes. The core contribution of DQA is **Zipf-based attention weighting** (Equation 14), which we preserve. The DP segmentation is a preprocessing step specific to URLs,
not the attention mechanism itself. We apply the same Zipf weighting directly to email vocabulary words.

### Q: How do you know this adaptation is valid?

**A:** This is a **domain transfer experiment**. The research question is: *"Does a method designed for URL phishing transfer to email phishing?"* Our adaptations are justified
by domain differences (URLs vs. natural language), and our evaluation will show whether the approach generalizes.

### Q: Why not include artificial features like the paper?

**A:** The paper's artificial features (Section 4) are URL/HTML-specific (e.g., `NumDots`, `SubdomainLevel`, `IpAddress`). Our dataset is preprocessed text without URLs or HTML.
We focus on validating the deep learning component first. Future work could add email-specific artificial features (sender domain, attachment presence, etc.).

---

## Expected Outcomes

### Hypothesis
CNN-DQA will achieve competitive accuracy with traditional ML (Naive Bayes, Random Forest) but at higher computational cost.

### Success Criteria
1. **Implementation validity:** Model trains without errors, converges on validation set
2. **Baseline comparison:** CNN-DQA accuracy ≥ Random Forest (98.47%)
3. **Computational cost:** Training time and inference speed documented

### Research Contribution
- **Domain transfer validation:** Tests whether URL-based methods generalize to emails
- **Simplification study:** Shows which paper components are URL-specific vs. transferable
- **Benchmark dataset:** Provides CNN results on Kaggle phishing email dataset

---

## Limitations and Future Work

### Current Limitations
1. **No artificial features:** Dataset lacks URL/HTML metadata
2. **Simplified attention:** Full DQA machinery not implemented
3. **Single language:** English emails only (dataset constraint)

### Future Extensions
1. **Add email metadata features:** Sender domain, subject line length, attachment count
2. **Experiment with BERT embeddings:** Replace word2vec-style embeddings with pre-trained transformers
3. **Multi-modal approach:** Combine text CNN with header/metadata features
4. **Adversarial robustness:** Test against perturbation attacks (character substitution, etc.)

---

## References

### Primary Source
Zhu, E., Cheng, K., Zhang, Z., & Wang, H. (2024). PDHF: Effective phishing detection model combining optimal artificial and automatic deep features.
*Computers & Security*, 136, 103561.  
https://doi.org/10.1016/j.cose.2023.103561

### Related Work on Attention Mechanisms
- Xiao et al. (2020). CNN-MHSA: Multi-head self-attention for phishing detection. *Neural Networks*, 125, 303-312.
- Chai et al. (2022). Multi-modal hierarchical attention for phishing websites. *IEEE Trans. Dependable Secure Comput.*, 19(2), 790-803.

### NLP Background
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
- Piantadosi, S. T. (2014). Zipf's word frequency law in natural language: A critical review. *Psychonomic Bulletin & Review*, 21, 1112-1130.

---

## Appendix: Equation Mapping

| Paper Reference | Our Implementation |
|-----------------|-------------------|
| Section 5.1 (One-hot encoding) | Word-level tokenization via vocabulary lookup |
| Equation 8 (Embedding layer) | Keras `Embedding(vocab_size=10000, output_dim=128)` |
| Equations 10-11 (Dual CNN) | Two `Conv1D` branches, kernel sizes 3 & 5 |
| Equation 12 (Concatenation) | `Concatenate([conv1_output, conv2_output])` |
| Equation 14 (Zipf cost) | `attention_weight = log(rank(word)) + 1` |
| Equation 18 (Sigmoid normalization) | `sigmoid(λ × attention_weight)` where λ < 1 |
| Equation 20 (Attention application) | Element-wise multiply attention × CNN features |
| Equation 21 (Classification) | `Dense(1, activation='sigmoid')` |

---

**Document Version:** 1.0  
**Last Updated:** January 13, 2026
