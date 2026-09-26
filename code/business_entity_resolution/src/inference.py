"""
Inference module for generating predictions on test data.
Runs complete pipeline: cleaning → blocking → features → matching → output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict
from tqdm import tqdm

from cleaning import BusinessDataCleaner, clean_dataframe
from blocking import MultiBlocker, calculate_candidate_recall
from features import FeatureExtractor
from model import EntityMatcher


def run_inference_pipeline(test_dir: Path, model_path: str, output_dir: Path,
                          max_candidates: int = 100) -> Dict[str, pd.DataFrame]:
    """
    Run complete inference pipeline on test data.
    
    Args:
        test_dir: Path to test data directory
        model_path: Path to trained model file
        output_dir: Path to output directory
        max_candidates: Maximum candidates per entity
        
    Returns:
        Dictionary with candidate_pairs and matching_results DataFrames
    """
    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION - INFERENCE PIPELINE")
    print("=" * 80)
    
    # Step 1: Load test data
    print("\\n[1/7] Loading test data...")
    test_s1 = pd.read_csv(test_dir / 'test_source1.tsv', sep='\\t')
    test_s2 = pd.read_csv(test_dir / 'test_source2.tsv', sep='\\t')
    test_s3 = pd.read_csv(test_dir / 'test_source3.tsv', sep='\\t')
    
    print(f"  Source 1: {len(test_s1):,} entities")
    print(f"  Source 2: {len(test_s2):,} entities")
    print(f"  Source 3: {len(test_s3):,} entities")
    
    # Step 2: Clean data
    print("\\n[2/7] Cleaning and normalizing data...")
    cleaner = BusinessDataCleaner()
    
    test_s1_clean = clean_dataframe(test_s1, cleaner)
    test_s2_clean = clean_dataframe(test_s2, cleaner)
    test_s3_clean = clean_dataframe(test_s3, cleaner)
    
    # Combine candidates
    test_candidates_clean = pd.concat([test_s2_clean, test_s3_clean], ignore_index=True)
    print(f"  ✓ Cleaned {len(test_candidates_clean):,} candidate entities")
    
    # Step 3: Generate candidate pairs (blocking)
    print("\\n[3/7] Generating candidate pairs (blocking)...")
    blocker = MultiBlocker(max_candidates_per_entity=max_candidates)
    blocker.build_blocking_index(test_candidates_clean)
    
    candidate_pairs = blocker.generate_candidate_pairs(test_s1_clean)
    
    # Step 4: Extract features
    print("\\n[4/7] Extracting features for candidate pairs...")
    feature_extractor = FeatureExtractor()
    
    # Compute IDF weights from all test data
    test_all = pd.concat([test_s1_clean, test_s2_clean, test_s3_clean], ignore_index=True)
    idf_weights = feature_extractor.compute_idf_weights(test_all)
    print(f"  ✓ Computed IDF weights for {len(idf_weights):,} tokens")
    
    features_df = feature_extractor.create_feature_matrix(
        candidate_pairs,
        test_s1_clean,
        test_candidates_clean,
        idf_weights
    )
    
    # Step 5: Load model and predict
    print("\\n[5/7] Loading trained model and predicting...")
    matcher = EntityMatcher()
    matcher.load(model_path)
    
    if len(features_df) > 0:
        features_df['match_proba'] = matcher.predict_proba(features_df)
        features_df['is_match'] = matcher.predict(features_df)
    else:
        print("  ⚠️  No candidate pairs to predict!")
        features_df['match_proba'] = []
        features_df['is_match'] = []
    
    # Step 6: Generate matching results
    print("\\n[6/7] Generating matching results...")
    matching_results = []
    
    for s1_id in test_s1_clean['entity_id']:
        # Get predicted matches for this S1 entity
        s1_features = features_df[features_df['source1_entity_id'] == s1_id]
        
        if len(s1_features) == 0:
            # No candidates → singleton
            matched_ids = []
        else:
            # Get candidates predicted as matches
            matches = s1_features[s1_features['is_match'] == 1]
            matched_ids = matches['candidate_entity_id'].tolist()
        
        matching_results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(matched_ids) if matched_ids else ''
        })
    
    matching_results_df = pd.DataFrame(matching_results)
    
    # Statistics
    num_singletons = (matching_results_df['matched_entity_ids'] == '').sum()
    num_with_matches = (matching_results_df['matched_entity_ids'] != '').sum()
    total_matches = features_df['is_match'].sum() if len(features_df) > 0 else 0
    
    print(f"\\n  Results:")
    print(f"    Total S1 entities: {len(matching_results_df):,}")
    print(f"    Singletons (no matches): {num_singletons:,} ({100*num_singletons/len(matching_results_df):.1f}%)")
    print(f"    With matches: {num_with_matches:,} ({100*num_with_matches/len(matching_results_df):.1f}%)")
    print(f"    Total predicted matches: {total_matches:,}")
    
    # Step 7: Save outputs
    print("\\n[7/7] Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidate_pairs_path = output_dir / 'candidate_pairs.tsv'
    matching_results_path = output_dir / 'matching_results.tsv'
    
    candidate_pairs.to_csv(candidate_pairs_path, sep='\\t', index=False)
    matching_results_df.to_csv(matching_results_path, sep='\\t', index=False)
    
    print(f"  ✓ Saved candidate_pairs.tsv ({len(candidate_pairs):,} rows)")
    print(f"  ✓ Saved matching_results.tsv ({len(matching_results_df):,} rows)")
    
    print("\\n" + "=" * 80)
    print("✓ INFERENCE PIPELINE COMPLETE")
    print("=" * 80)
    
    return {
        'candidate_pairs': candidate_pairs,
        'matching_results': matching_results_df,
        'features': features_df
    }


if __name__ == '__main__':
    # Default paths
    test_dir = Path('../../student_resource/dataset/test')
    model_path = '../output/entity_matcher_model.pkl'
    output_dir = Path('../output')
    
    # Run inference
    results = run_inference_pipeline(
        test_dir=test_dir,
        model_path=model_path,
        output_dir=output_dir,
        max_candidates=100
    )
    
    print("\\nInference complete. Output files ready for submission.")
