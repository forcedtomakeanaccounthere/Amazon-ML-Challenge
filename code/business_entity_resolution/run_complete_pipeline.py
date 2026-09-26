"""
Complete end-to-end pipeline script.
Runs training and inference in sequence.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.train_pipeline import run_training_pipeline
from src.inference import run_inference_pipeline


def main():
    """Run complete pipeline: training + inference."""
    
    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION - COMPLETE PIPELINE")
    print("=" * 80)
    
    # Define paths
    base_dir = Path(__file__).parent
    data_root = base_dir.parent.parent / 'student_resource' / 'dataset'
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'
    output_dir = base_dir / 'output'
    
    # Verify data exists
    if not train_dir.exists() or not (train_dir / 'train_source1.tsv').exists():
        print("\\n❌ ERROR: Training data not found!")
        print(f"   Expected location: {train_dir}")
        print("\\n   Please ensure the following files exist:")
        print("   - train_source1.tsv")
        print("   - train_source2.tsv")
        print("   - train_source3.tsv")
        print("   - train_ground_truth.tsv")
        return False
    
    if not test_dir.exists() or not (test_dir / 'test_source1.tsv').exists():
        print("\\n❌ ERROR: Test data not found!")
        print(f"   Expected location: {test_dir}")
        print("\\n   Please ensure the following files exist:")
        print("   - test_source1.tsv")
        print("   - test_source2.tsv")
        print("   - test_source3.tsv")
        return False
    
    # Step 1: Training
    print("\\n" + "=" * 80)
    print("STEP 1: TRAINING")
    print("=" * 80)
    
    try:
        train_results = run_training_pipeline(
            train_dir=train_dir,
            output_dir=output_dir,
            max_candidates=100,
            val_split=0.2,
            random_state=42
        )
        print("\\n✓ Training completed successfully")
    except Exception as e:
        print(f"\\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Inference
    print("\\n" + "=" * 80)
    print("STEP 2: INFERENCE")
    print("=" * 80)
    
    try:
        model_path = output_dir / 'entity_matcher_model.pkl'
        
        inference_results = run_inference_pipeline(
            test_dir=test_dir,
            model_path=str(model_path),
            output_dir=output_dir,
            max_candidates=100
        )
        print("\\n✓ Inference completed successfully")
    except Exception as e:
        print(f"\\n❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\\nOutput files generated:")
    print(f"  - {output_dir / 'candidate_pairs.tsv'}")
    print(f"  - {output_dir / 'matching_results.tsv'}")
    print(f"  - {output_dir / 'entity_matcher_model.pkl'}")
    
    print("\\nNext steps:")
    print("  1. Validate submission:")
    print(f"     cd ../../student_resource")
    print(f"     python utils/validate_submission.py \\\\")
    print(f"       --matching ../code/business_entity_resolution/output/matching_results.tsv \\\\")
    print(f"       --candidate ../code/business_entity_resolution/output/candidate_pairs.tsv \\\\")
    print(f"       --test-dir dataset/test")
    print("\\n  2. If validation passes, submit matching_results.tsv to the leaderboard")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
