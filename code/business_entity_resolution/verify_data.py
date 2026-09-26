"""
Data verification script.
Checks if all required data files exist and are properly formatted.
"""

import sys
from pathlib import Path


def verify_file(filepath: Path, file_type: str) -> bool:
    """Verify a single file exists and looks valid."""
    if not filepath.exists():
        print(f"  ❌ MISSING: {filepath.name}")
        return False
    
    # Check file size (should be more than just metadata)
    size = filepath.stat().st_size
    if size < 1000:  # Less than 1KB is suspicious
        print(f"  ⚠️  SUSPICIOUS: {filepath.name} (only {size} bytes - might be metadata file)")
        return False
    
    # Try to read first few lines
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [f.readline() for _ in range(3)]
        
        # Check for tab separation
        if '\\t' not in lines[0]:
            print(f"  ⚠️  WARNING: {filepath.name} doesn't appear to be tab-separated")
            print(f"      First line: {lines[0][:100]}")
            return False
        
        # Check expected columns
        if file_type == 'source':
            expected_cols = ['entity_id', 'business_name', 'business_address', 'country']
        elif file_type == 'ground_truth':
            expected_cols = ['source1_entity_id', 'matched_entity_ids']
        else:
            expected_cols = []
        
        if expected_cols:
            header = lines[0].strip().split('\\t')
            if header != expected_cols:
                print(f"  ⚠️  WARNING: {filepath.name} has unexpected columns")
                print(f"      Expected: {expected_cols}")
                print(f"      Found: {header}")
                return False
        
        print(f"  ✓ OK: {filepath.name} ({size:,} bytes, {len(lines)-1} sample lines)")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR reading {filepath.name}: {str(e)}")
        return False


def main():
    """Verify all required data files."""
    
    print("=" * 80)
    print("DATA VERIFICATION")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    data_root = base_dir.parent.parent / 'student_resource' / 'dataset'
    
    all_ok = True
    
    # Check training data
    print("\\n[Training Data]")
    train_dir = data_root / 'train'
    
    if not train_dir.exists():
        print(f"  ❌ Training directory not found: {train_dir}")
        all_ok = False
    else:
        print(f"  Directory: {train_dir}")
        all_ok &= verify_file(train_dir / 'train_source1.tsv', 'source')
        all_ok &= verify_file(train_dir / 'train_source2.tsv', 'source')
        all_ok &= verify_file(train_dir / 'train_source3.tsv', 'source')
        all_ok &= verify_file(train_dir / 'train_ground_truth.tsv', 'ground_truth')
    
    # Check test data
    print("\\n[Test Data]")
    test_dir = data_root / 'test'
    
    if not test_dir.exists():
        print(f"  ❌ Test directory not found: {test_dir}")
        all_ok = False
    else:
        print(f"  Directory: {test_dir}")
        all_ok &= verify_file(test_dir / 'test_source1.tsv', 'source')
        all_ok &= verify_file(test_dir / 'test_source2.tsv', 'source')
        all_ok &= verify_file(test_dir / 'test_source3.tsv', 'source')
    
    # Check utils
    print("\\n[Utilities]")
    utils_dir = base_dir.parent.parent / 'student_resource' / 'utils'
    if not utils_dir.exists():
        print(f"  ⚠️  Utils directory not found: {utils_dir}")
    else:
        print(f"  Directory: {utils_dir}")
        validator = utils_dir / 'validate_submission.py'
        if validator.exists():
            print(f"  ✓ OK: validate_submission.py")
        else:
            print(f"  ❌ MISSING: validate_submission.py")
            all_ok = False
    
    # Summary
    print("\\n" + "=" * 80)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print("\\nYou can now run the pipeline:")
        print("  python run_complete_pipeline.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 80)
        print("\\nPlease fix the issues above before running the pipeline.")
        print("\\nIf you only see ._*.tsv files (macOS metadata), see DATA_SETUP.md")
        print("for instructions on extracting the actual data files.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
