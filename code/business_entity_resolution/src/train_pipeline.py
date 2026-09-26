"""
Complete training pipeline for Business Entity Resolution.
Trains model on training data with validation split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from cleaning import BusinessDataCleaner, clean_dataframe
from blocking import MultiBlocker, calculate_candidate_recall
from features import FeatureExtractor
from model import EntityMatcher, calculate_f_beta_score


def run_training_pipeline(train_dir: Path, output_dir: Path, 
                         max_candidates: int = 100,
                         val_split: float = 0.2,
                         random_state: int = 42):
    """
    Run complete training pipeline.
    
    Args:
        train_dir: Path to training data directory
        output_dir: Path to output directory
        max_candidates: Maximum candidates per entity
        val_split: Validation split fraction
        random_state: Random seed
    """
    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION - TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load training data
    print("\\n[1/9] Loading training data...")
    train_s1 = pd.read_csv(train_dir / 'train_source1.tsv', sep='\\t')
    train_s2 = pd.read_csv(train_dir / 'train_source2.tsv', sep='\\t')
    train_s3 = pd.read_csv(train_dir / 'train_source3.tsv', sep='\\t')
    train_gt = pd.read_csv(train_dir / 'train_ground_truth.tsv', sep='\\t')
    
    print(f"  Source 1: {len(train_s1):,} entities")
    print(f"  Source 2: {len(train_s2):,} entities")
    print(f"  Source 3: {len(train_s3):,} entities")
    print(f"  Ground truth: {len(train_gt):,} entities")
    
    # Step 2: Split into train and validation sets
    print(f"\\n[2/9] Splitting into train ({100*(1-val_split):.0f}%) and validation ({100*val_split:.0f}%)...")
    
    s1_ids = train_s1['entity_id'].values
    train_ids, val_ids = train_test_split(
        s1_ids, test_size=val_split, random_state=random_state
    )
    
    train_s1_split = train_s1[train_s1['entity_id'].isin(train_ids)]
    val_s1_split = train_s1[train_s1['entity_id'].isin(val_ids)]
    
    train_gt_split = train_gt[train_gt['source1_entity_id'].isin(train_ids)]
    val_gt_split = train_gt[train_gt['source1_entity_id'].isin(val_ids)]
    
    print(f"  Training: {len(train_s1_split):,} S1 entities")
    print(f"  Validation: {len(val_s1_split):,} S1 entities")
    
    # Step 3: Clean data
    print("\\n[3/9] Cleaning and normalizing data...")
    cleaner = BusinessDataCleaner()
    
    train_s1_clean = clean_dataframe(train_s1_split, cleaner)
    val_s1_clean = clean_dataframe(val_s1_split, cleaner)
    train_s2_clean = clean_dataframe(train_s2, cleaner)
    train_s3_clean = clean_dataframe(train_s3, cleaner)
    
    # Combine candidates (same for both train and val)
    train_candidates_clean = pd.concat([train_s2_clean, train_s3_clean], ignore_index=True)
    print(f"  ✓ Cleaned {len(train_candidates_clean):,} candidate entities")
    
    # Step 4: Build blocking index
    print("\\n[4/9] Building blocking index...")
    blocker = MultiBlocker(max_candidates_per_entity=max_candidates)
    blocker.build_blocking_index(train_candidates_clean)
    
    # Step 5: Generate candidate pairs for training
    print("\\n[5/9] Generating training candidate pairs...")
    train_candidate_pairs = blocker.generate_candidate_pairs(train_s1_clean)
    
    # Calculate candidate recall on training
    print("\\nCalculating candidate recall on training set...")
    train_recall_metrics = calculate_candidate_recall(train_candidate_pairs, train_gt_split)
    print(f"  Pair Recall: {train_recall_metrics['pair_recall']:.4f}")
    print(f"  Entity Recall: {train_recall_metrics['entity_recall']:.4f}")
    print(f"  Found {train_recall_metrics['found_matches']:,} / {train_recall_metrics['total_true_matches']:,} true matches")
    
    # Step 6: Generate candidate pairs for validation
    print("\\n[6/9] Generating validation candidate pairs...")
    val_candidate_pairs = blocker.generate_candidate_pairs(val_s1_clean)
    
    # Calculate candidate recall on validation
    print("\\nCalculating candidate recall on validation set...")
    val_recall_metrics = calculate_candidate_recall(val_candidate_pairs, val_gt_split)
    print(f"  Pair Recall: {val_recall_metrics['pair_recall']:.4f}")
    print(f"  Entity Recall: {val_recall_metrics['entity_recall']:.4f}")
    print(f"  Found {val_recall_metrics['found_matches']:,} / {val_recall_metrics['total_true_matches']:,} true matches")
    
    # Step 7: Extract features
    print("\\n[7/9] Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Compute IDF weights from all training data
    train_all = pd.concat([train_s1_clean, train_s2_clean, train_s3_clean], ignore_index=True)
    idf_weights = feature_extractor.compute_idf_weights(train_all)
    print(f"  ✓ Computed IDF weights for {len(idf_weights):,} tokens")
    
    # Extract training features
    print("\\nExtracting training features...")
    train_features = feature_extractor.create_feature_matrix(
        train_candidate_pairs,
        train_s1_clean,
        train_candidates_clean,
        idf_weights
    )
    
    # Extract validation features
    print("\\nExtracting validation features...")
    val_features = feature_extractor.create_feature_matrix(
        val_candidate_pairs,
        val_s1_clean,
        train_candidates_clean,
        idf_weights
    )
    
    # Step 8: Train model
    print("\\n[8/9] Training entity matching model...")
    matcher = EntityMatcher(random_state=random_state)
    
    # Prepare training data with labels
    train_features_labeled, train_labels = matcher.prepare_training_data(
        train_features, train_gt_split
    )
    
    # Train model
    train_metrics = matcher.train(train_features_labeled, train_labels, val_size=0.0)
    
    # Step 9: Tune threshold on validation set
    print("\\n[9/9] Tuning threshold on validation set...")
    val_features_labeled, val_labels = matcher.prepare_training_data(
        val_features, val_gt_split
    )
    
    best_threshold, threshold_metrics = matcher.tune_threshold(
        val_features_labeled, val_labels, beta=0.5
    )
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'entity_matcher_model.pkl'
    matcher.save(str(model_path))
    
    # Final validation metrics
    print("\\n" + "=" * 80)
    print("TRAINING COMPLETE - VALIDATION METRICS")
    print("=" * 80)
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"F_0.5 Score: {threshold_metrics['f_beta']:.4f}")
    print(f"Precision: {threshold_metrics['precision']:.4f}")
    print(f"Recall: {threshold_metrics['recall']:.4f}")
    print(f"\\nCandidate Recall (upper bound): {val_recall_metrics['pair_recall']:.4f}")
    print("=" * 80)
    
    return {
        'matcher': matcher,
        'train_metrics': train_metrics,
        'threshold_metrics': threshold_metrics,
        'candidate_recall': val_recall_metrics
    }


if __name__ == '__main__':
    # Default paths
    train_dir = Path('../../student_resource/dataset/train')
    output_dir = Path('../output')
    
    # Run training pipeline
    results = run_training_pipeline(
        train_dir=train_dir,
        output_dir=output_dir,
        max_candidates=100,
        val_split=0.2,
        random_state=42
    )
    
    print("\\nTraining complete. Model saved to output/entity_matcher_model.pkl")
