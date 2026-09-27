# Quick Start Guide

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.9 or higher
- [ ] The actual TSV data files (not just `._*.tsv` metadata files)
- [ ] At least 4GB RAM available
- [ ] ~30-60 minutes for complete training + inference

## Step-by-Step Instructions

### 1. Verify Data Files

```bash
cd code/business_entity_resolution
python verify_data.py
```

**Expected output**: All files should show ✓ OK

**If you see ❌ or ⚠️**: See `DATA_SETUP.md` for troubleshooting

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Run Complete Pipeline

```bash
python run_complete_pipeline.py
```

This single command will:
- ✓ Load and clean training data
- ✓ Generate candidate pairs using blocking
- ✓ Extract features
- ✓ Train LightGBM model
- ✓ Tune threshold for F_0.5
- ✓ Run inference on test data
- ✓ Generate output files

**Output files created**:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`
- `output/entity_matcher_model.pkl`

### 4. Validate Submission

```bash
cd ../../student_resource
python utils/validate_submission.py ^
  --matching ../code/business_entity_resolution/output/matching_results.tsv ^
  --candidate ../code/business_entity_resolution/output/candidate_pairs.tsv ^
  --test-dir dataset/test
```

**Expected**: `PASS` with exit code 0

### 5. Submit to Leaderboard

Upload `output/matching_results.tsv` to the competition portal.

## Alternative: Step-by-Step Execution

If you want to run stages separately:

### Train Model Only
```bash
cd src
python train_pipeline.py
```

### Run Inference Only
```bash
cd src
python inference.py
```

### Explore Data
```bash
jupyter notebook notebooks/00_eda.ipynb
```

## Troubleshooting

### Issue: "Data files not found"
- **Solution**: Check `DATA_SETUP.md` - you might only have macOS metadata files

### Issue: "Module not found"
- **Solution**: Activate virtual environment and reinstall requirements
  ```bash
  venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Issue: "Memory error"
- **Solution**: Reduce `max_candidates` in `run_complete_pipeline.py` (line with `max_candidates=100`)

### Issue: Validation fails
- **Solution**: Check the specific error message from `validate_submission.py`
- Common fixes are in the validation output

## Expected Timings

On a typical development machine:

| Stage | Time |
|-------|------|
| Data cleaning | ~1-2 min |
| Candidate generation | ~5-10 min |
| Feature extraction | ~10-20 min |
| Model training | ~5-10 min |
| Inference | ~10-20 min |
| **Total** | **~30-60 min** |

## What Each File Does

| File | Purpose |
|------|---------|
| `verify_data.py` | Check if data files exist and are valid |
| `run_complete_pipeline.py` | Run everything (training + inference) |
| `src/train_pipeline.py` | Train model only |
| `src/inference.py` | Generate predictions only |
| `src/cleaning.py` | Text normalization functions |
| `src/blocking.py` | Candidate generation logic |
| `src/features.py` | Feature engineering |
| `src/model.py` | Model training and evaluation |

## Need Help?

1. Check `README.md` for detailed documentation
2. Check `DATA_SETUP.md` if data files are missing
3. Check `Documentation_template.md` for methodology details
4. Review error messages carefully - they usually indicate the exact issue

## Success Checklist

- [x] Code structure created
- [x] All modules implemented
- [ ] Data files present and validated
- [ ] Dependencies installed
- [ ] Pipeline executed successfully
- [ ] Validation passes
- [ ] Submission uploaded

Good luck! 🚀
