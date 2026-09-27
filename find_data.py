import os
from pathlib import Path

# Search for TSV files
root = Path(r"c:\Users\Abhishek\Downloads\amazon_ml_student_resource")
print(f"Searching for TSV files in: {root}")
print("=" * 80)

tsv_files = []
for file in root.rglob("*.tsv"):
    # Skip macOS metadata files
    if not file.name.startswith("._"):
        size = file.stat().st_size
        if size > 1000:  # Larger than 1KB
            tsv_files.append((file, size))
            print(f"Found: {file}")
            print(f"  Size: {size:,} bytes")

print("=" * 80)
print(f"Total TSV files found: {len(tsv_files)}")

if len(tsv_files) == 0:
    print("\n❌ No actual TSV data files found!")
    print("Only macOS metadata files (._*.tsv) are present.")
    print("\nPlease provide the actual data files to continue.")
