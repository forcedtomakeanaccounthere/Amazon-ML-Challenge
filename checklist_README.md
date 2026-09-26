# Business Entity Resolution — Build Checklist

Module-by-module task list. Suggested format per module (not a hard rule):
- **Notebooks (`.ipynb`)** for exploratory / visual / iterative work — EDA, cleaning trials, feature inspection, threshold tuning, error analysis.
- **Scripts (`.py`)** for deterministic, reusable, must-run-identically-on-train-and-test logic — blocking, feature extraction at scale, final inference, validation.

Suggested repo layout:
```
code/business_entity_resolution/
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_cleaning.ipynb
│   ├── 04_feature_inspection.ipynb
│   ├── 06_threshold_tuning.ipynb
│   └── 08_error_analysis.ipynb
├── src/
│   ├── cleaning.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── inference.py
│   └── validate_local.py
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── README.md
└── requirements.txt
```

---

## 0. Setup & EDA
`notebooks/00_eda.ipynb`

- [ ] Load all 6 files with `sep="\t"`; confirm no silent single-column parsing
- [ ] Row counts per source, per country (confirm France appears only in test)
- [ ] Null / empty-field audit per column per source
- [ ] Ground-truth stats: % singletons, match-count distribution (0 / 1 / many), duplicate check
- [ ] Sample 20–30 true-positive pairs side by side — catalogue the actual noise patterns you observe (don't assume the PDF's list is exhaustive)
- [ ] Sample 20–30 hard negatives (same block, different business) to see what a precision failure looks like
- [ ] Note any country-specific formatting quirks (US ZIP patterns, India PIN/landmark usage) to inform normalization rules later

---

## 1. Cleaning & Normalization
`notebooks/01_cleaning.ipynb` → promote finalized logic into `src/cleaning.py`

- [ ] Unicode NFKC normalization, casefold
- [ ] Legal-suffix dictionary per country (derive from US/India training data; hand-build a French one — SARL, SAS, EURL, etc. — since no training examples exist)
- [ ] Keep both a raw and a suffix-stripped name field (don't discard the suffix — its presence/absence is itself a signal)
- [ ] Punctuation normalization, `&` vs `and` unification
- [ ] Address: regex-extract leading house/building number, a normalized street-type token (Rd/Road/St/Street/Rue), trailing numeric block (PIN/ZIP/postal code)
- [ ] Landmark-phrase stripping ("Near X", "Opp Y") into a separate flag rather than silent deletion
- [ ] Phonetic encoding (Soundex/NYSIIS/Metaphone) on name tokens
- [ ] Sanity check: re-run EDA true-positive samples through the cleaner and visually confirm they look closer to each other post-cleaning
- [ ] Freeze `src/cleaning.py`; add a few unit tests on hand-picked noisy examples (typos, transliteration, missing components)
- [ ] Confirm cleaning functions run identically on train and test data (no train-only branches)

---

## 2. Blocking / Candidate Generation
`src/blocking.py` — must run identically on train and test

- [ ] Build multiple blocking keys: sorted name tokens, phonetic code, numeric address token, city/state token
- [ ] **Union** blocks (not intersect) — recall is the priority here
- [ ] Add a TF-IDF character-n-gram + approximate nearest neighbor (FAISS/Annoy) layer as a second, independent retrieval path — catches typos and word-order transpositions token blocking misses
- [ ] Cap candidates per S1 entity (top-K by a cheap similarity score) to keep pair count tractable
- [ ] Build blocking **per-country where possible, with a generic fallback path** — explicitly test that the fallback fires correctly and doesn't silently return zero candidates on an unseen country label
- [ ] Compute and log **candidate recall** on the training split: fraction of true matches present in the candidate set — this is your recall ceiling, track it before touching the matcher
- [ ] Compute reduction ratio (candidate pairs vs. brute-force pairs) to confirm blocking is actually reducing search space
- [ ] Save the **final** candidate set (post all blocking/filtering stages) — this exact set is what `candidate_pairs.tsv` must contain
- [ ] Stress test: hold out one training country's normalization dictionary (e.g. train only on US suffixes) and check blocking recall on India, as a proxy for how it'll behave on the unseen France test data

---

## 3. Feature Engineering
`notebooks/04_feature_inspection.ipynb` (build/inspect) → `src/features.py` (production)

- [ ] Name features: exact-normalized match, Jaccard, token-sort ratio, Levenshtein/Jaro-Winkler ratio, TF-IDF cosine, Monge-Elkan, acronym-expansion match
- [ ] Address features: token Jaccard, numeric-component exact match, street-type-normalized similarity, postal-code exact match (when present both sides), city/state match
- [ ] Rarity weighting: down-weight common tokens ("inc", "the", "restaurant"), up-weight rare token overlaps (per-token IDF-style weighting)
- [ ] Competitive/relative features per S1 entity: gap between top-1 and top-2 candidate scores; reciprocal-nearest-neighbor flag (is this candidate also *this side's* best match)
- [ ] Country as categorical feature with a safe "unseen" fallback bucket — never filter or hard-encode to {US, India}
- [ ] Optional: cosine similarity from a small open-source multilingual embedding model (MIT/Apache-licensed, well under 8B params) as one additional feature — confirm via the challenge query form that this doesn't cross into the "external lookup" rule before relying on it
- [ ] Plot feature distributions for true matches vs. non-matches (in the notebook) to sanity-check discriminative power before training
- [ ] Check for feature leakage (e.g., no feature that only exists because of how blocking was constructed)

---

## 4. Matching Model
`src/model.py` (train/save) + `notebooks/06_threshold_tuning.ipynb` (calibration & threshold search)

- [ ] Train a pairwise binary classifier (LightGBM/CatBoost/XGBoost) on engineered features
- [ ] Confirm model choice satisfies the MIT/Apache-2.0 + ≤8B-parameter constraint (tree ensembles trivially qualify)
- [ ] Hold out a validation split from training data (stratified by country) before touching test data
- [ ] Calibrate output probabilities (isotonic regression or Platt scaling)
- [ ] Grid-search the decision threshold **directly against validation F₀.₅**, not F1/accuracy — precision is weighted 2x here
- [ ] Consider per-country threshold tuning (US/India noise profiles differ; leave a documented default for unseen countries like France)
- [ ] Explicitly validate singleton handling: confirm entities with best-candidate score below threshold correctly predict empty and score 1.0 in your local F₀.₅ calc
- [ ] Log feature importances; sanity-check they align with intuition (address numeric match and name similarity should dominate)

---

## 5. Local Evaluation
`src/validate_local.py` + `notebooks/08_error_analysis.ipynb`

- [ ] Implement the exact macro-averaged F₀.₅ formula locally, matching the official one
- [ ] Score: overall F₀.₅, precision, recall, candidate recall, reduction ratio, singleton accuracy
- [ ] Error analysis: pull worst-scoring S1 entities, inspect false merges vs. missed matches separately
- [ ] Check performance breakdown by country (US vs. India) to catch any imbalance
- [ ] Simulate the France scenario (hold-out-country test described in Module 2) and re-check F₀.₅ degradation as a generalization proxy
- [ ] Iterate: cleaning → blocking → features → model → threshold, re-measuring F₀.₅ each pass; keep a simple experiment log (what changed, what score resulted)

---

## 6. Inference & Output Generation
`src/inference.py`

- [ ] Run full pipeline (cleaning → blocking → features → model → threshold) end-to-end on the test set
- [ ] Write `output/candidate_pairs.tsv`: one row per test S1 entity, `source1_entity_id`, `candidate_entity_ids` (comma-separated, no duplicates, empty when none) — must be the **final** candidate set actually scored by the model
- [ ] Write `output/matching_results.tsv`: one row per test S1 entity, `matched_entity_ids` (comma-separated, no duplicates, empty when singleton)
- [ ] Confirm every match ID also appears in the corresponding candidate list (subset invariant)
- [ ] Confirm every test S1 entity has exactly one row in both files, no duplicates, no S1 self-references, no nonexistent IDs
- [ ] Confirm tab-separation on output files

---

## 7. Validation & Submission
- [ ] Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- [ ] Fix any reported issues; re-run until `PASS` / exit 0
- [ ] Upload `matching_results.tsv` to the leaderboard portal; confirm `SCORED` status with an F₀.₅ value
- [ ] Track submission versions (max 5/day) — keep a changelog of what changed between submissions
- [ ] Assemble final ZIP: `output/`, `code/business_entity_resolution/` (src, README, requirements.txt), completed `Documentation_template.md`
- [ ] README.md inside the code folder: exact reproduction instructions, data → blocking → matching → output
- [ ] requirements.txt: pinned dependency versions
- [ ] Methodology doc covers: methodology, blocking strategy, model architecture, feature engineering, other relevant notes
- [ ] Final self-check against the fair-play rules: no external DB/API/geocoding lookups anywhere in the pipeline, model license and parameter count compliant

---

## Common failure modes to double-check before submitting
- [ ] Reading `.tsv` without `sep="\t"`
- [ ] Hard-coding country to `{US, India}` anywhere (filters, one-hot encoders, dictionaries without a fallback)
- [ ] `candidate_pairs.tsv` reflecting an early blocking stage instead of the final set fed to the model
- [ ] Any Source 1 self-matches or out-of-test-set IDs in output
- [ ] Duplicate IDs within a list, or duplicate `source1_entity_id` rows
- [ ] Missing Source 1 test entities in the output
- [ ] Writing `"None"`/`"NULL"` instead of leaving singleton fields empty
