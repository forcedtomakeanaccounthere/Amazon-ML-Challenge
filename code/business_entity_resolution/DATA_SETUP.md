# Data Setup Guide

## Issue: Missing TSV Files

If you extracted the student resources from a macOS-created archive on Windows, you might only see `._*.tsv` files (macOS metadata files) instead of the actual data files.

## Solution

### Option 1: Extract on macOS or Linux
If possible, extract the `student_resource` archive on a macOS or Linux system, then copy the files to Windows.

### Option 2: Find the Original Data Files
The actual TSV files should be provided separately by the challenge organizers. Check:
- Challenge download page
- Email from organizers
- Competition forum/discussion board

### Option 3: Clean Extraction on Windows
Try using 7-Zip or another archive tool that properly handles macOS archives:
1. Download and install 7-Zip (https://www.7-zip.org/)
2. Right-click the archive → 7-Zip → Extract Here
3. Look for files WITHOUT the `._` prefix

## Expected File Structure

Once you have the correct data files, your directory should look like this:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv          ← Need this (not ._train_source1.tsv)
│   │   ├── train_source2.tsv          ← Need this
│   │   ├── train_source3.tsv          ← Need this
│   │   └── train_ground_truth.tsv     ← Need this
│   └── test/
│       ├── test_source1.tsv           ← Need this
│       ├── test_source2.tsv           ← Need this
│       └── test_source3.tsv           ← Need this
└── utils/
    └── validate_submission.py
```

## Verify Data Files

Run this command to check if you have the correct files:

### Windows Command Prompt:
```cmd
dir /b student_resource\dataset\train\*.tsv | findstr /v "._"
```

### Windows PowerShell:
```powershell
Get-ChildItem student_resource\dataset\train\*.tsv | Where-Object {$_.Name -notlike "._*"} | Select-Object Name
```

You should see:
- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

## Quick Verification

To verify the TSV files are correctly formatted, try opening one in a text editor or Excel:

```cmd
head student_resource\dataset\train\train_source1.tsv
```

You should see tab-separated columns like:
```
entity_id	business_name	business_address	country
S1-00001	Example Corp	123 Main St	US
...
```

## Contact Support

If you still cannot find the actual data files, contact the challenge organizers through the official competition platform.
