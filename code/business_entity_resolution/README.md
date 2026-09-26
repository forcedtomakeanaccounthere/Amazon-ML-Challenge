# Business Entity Resolution

## Overview

This project implements a machine learning solution for the Amazon ML Challenge - Business Entity Resolution. The system identifies which records from multiple independent data sources (Source 2 and Source 3) refer to the same real-world business entities as deduplicated reference records in Source 1.

## Problem Description

- **Goal**: Match business entity records across multiple noisy data sources
- **Evaluation Metric**: Macro-averaged F_0.5 score (precision-heavy)
- **Challenge**: Handle incomplete, noisy, and differently formatted business names and addresses
- **Constraints**: 
  - Model must be MIT/Apache 2.0 licensed
  - Maximum 8 billion parameters
  - No external data lookup allowed

## Project Structure

```
business_entity_resolution/
├── src/
│   ├── cleaning.py           # Data cleaning and normalization
│   ├── blocking.py            # Candidate generation (blocking)
│   ├── features.py            # Feature engineering for pairs
│   ├── model.py              # Model training and evaluation
│   ├── train_pipeline.py     # Complete training pipeline
│   └── inference.py          # Inference pipeline for test data
├── notebooks/
│   └── 00_eda.ipynb          # Exploratory data analysis
├── output/
│   ├── candidate_pairs.tsv   # Generated candidate pairs
│   └── matching_results.tsv  # Final matching predictions
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis

Run the EDA notebook to understand the data:
```bash
jupyter notebook notebooks/00_eda.ipynb
```

### 2. Train the Model

Run the complete training pipeline:
```bash
cd src
python train_pipeline.py
```

This will:
- Load and clean training data
- Split into train/validation sets (80/20)
- Generate candidate pairs using multi-strategy blocking
- Extract pairwise features
- Train a LightGBM classifier
- Calibrate probabilities
- Tune decision threshold for F_0.5
- Save the model to `output/entity_matcher_model.pkl`

### 3. Generate Test Predictions

Run inference on test data:
```bash
cd src
python inference.py
```

This will:
- Load and clean test data
- Generate candidate pairs
- Extract features
- Load trained model and predict matches
- Generate output files:
  - `output/candidate_pairs.tsv`
  - `output/matching_results.tsv`

### 4. Validate Submission

Before submitting, validate the output files:
```bash
cd ../../student_resource
python utils/validate_submission.py \
  --matching ../code/business_entity_resolution/output/matching_results.tsv \
  --candidate ../code/business_entity_resolution/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Methodology

### 1. Data Cleaning (`cleaning.py`)

- **Unicode normalization**: NFKC normalization and casefolding
- **Legal suffix extraction**: Extract and normalize company suffixes (Inc, Corp, Pvt, Ltd, SARL, etc.)
- **Punctuation normalization**: Handle "&" vs "and", remove special characters
- **Address parsing**: Extract building numbers, postal codes, landmark flags
- **Phonetic encoding**: Generate Soundex-like codes for name matching

### 2. Blocking / Candidate Generation (`blocking.py`)

Multi-strategy blocking to achieve high recall:
- **Sorted token blocking**: Match on alphabetically sorted name tokens
- **Phonetic blocking**: Match on phonetic codes
- **Address number blocking**: Match on building numbers and postal codes
- **Country blocking**: Match within same country
- **First letter blocking**: Match on first letters of tokens
- **Exact name blocking**: Match on normalized full names

**Union approach**: Combine all blocking strategies to maximize recall while keeping candidate set manageable.

### 3. Feature Engineering (`features.py`)

Comprehensive pairwise features:

**Name Features**:
- Exact match (with and without suffix)
- Token Jaccard similarity
- Token sort ratio
- Levenshtein distance
- Jaro-Winkler similarity
- Monge-Elkan similarity (token-level best match)
- IDF-weighted Jaccard
- Phonetic match
- First token match

**Address Features**:
- Building number exact match
- Postal code exact and partial match
- Token Jaccard similarity
- Levenshtein and Jaro-Winkler on full address
- Landmark presence flags

**Country Features**:
- Country match
- Country indicators (US, India, Other)

### 4. Matching Model (`model.py`)

- **Algorithm**: LightGBM classifier (MIT licensed, < 8B parameters)
- **Class imbalance handling**: Scale positive weight based on class distribution
- **Probability calibration**: Isotonic regression for better probability estimates
- **Threshold tuning**: Grid search to maximize F_0.5 on validation set
- **Early stopping**: Prevent overfitting with validation monitoring

### 5. Evaluation

**Metrics tracked**:
- F_0.5 score (primary metric - precision-heavy)
- Precision and Recall
- Candidate recall (blocking upper bound)
- Singleton accuracy

**Validation strategy**:
- 80/20 train-validation split on Source 1 entities
- Stratified by country when possible
- Threshold tuned specifically for F_0.5

## Key Design Decisions

1. **Open-set country handling**: No hard-coding of countries, generic fallback for unseen countries (e.g., France in test)

2. **High recall blocking**: Multiple blocking strategies with union to ensure true matches aren't filtered out

3. **Precision-focused matching**: Given F_0.5 metric, threshold tuned to favor precision over recall

4. **Singleton support**: Model explicitly handles entities with no matches (empty predictions)

5. **Feature diversity**: Combine exact matches, fuzzy string matching, and token-based similarities

6. **No external data**: All matching based solely on provided training data

## Performance Considerations

- **Candidate generation**: Reduces O(n²) comparisons to manageable subset
- **Feature caching**: IDF weights computed once and reused
- **Parallel processing**: LightGBM uses all CPU cores
- **Memory efficient**: Streaming feature extraction for large datasets

## Output Format

### candidate_pairs.tsv
```
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003             
```

### matching_results.tsv
```
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S3-00812
S1-00002             S3-00004
S1-00003             
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas, numpy: Data manipulation
- scikit-learn: Model evaluation and calibration
- lightgbm: Gradient boosting classifier
- rapidfuzz: Fast string similarity
- faiss-cpu: Approximate nearest neighbor (optional)

## License

This implementation uses only MIT and Apache 2.0 licensed libraries to comply with challenge requirements.

## Authors

Amazon ML Challenge 2024 Submission
