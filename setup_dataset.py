#!/usr/bin/env python3
"""
setup_dataset.py — Extract ISIC-2017 ZIP files into the expected folder structure.

Expected input  (Dataset/):
  ISIC-2017_Training_Data.zip
  ISIC-2017_Training_Part1_GroundTruth.zip
  ISIC-2017_Validation_Data.zip
  ISIC-2017_Validation_Part1_GroundTruth.zip
  ISIC-2017_Test_v2_Data.zip
  ISIC-2017_Test_v2_Part1_GroundTruth.zip   ← test segmentation masks

Output (Dataset/extracted/):
  trainx/       — 2,000 dermoscopy images
  trainy/       — 2,000 segmentation masks  (*_segmentation.png)
  validationx/  — 150 images
  validationy/  — 150 segmentation masks
  testx/        — 600 test images
  testy/        — 600 test segmentation masks (*_segmentation.png)

Run once before training:
  python setup_dataset.py
"""

import zipfile
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATASET    = BASE_DIR / "Dataset"
EXTRACTED  = DATASET / "extracted"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Map: ZIP filename → destination subfolder
# Ordering matters for the summary printout — images before their masks
ZIP_MAP = {
    "ISIC-2017_Training_Data.zip":                ("trainx",      False),
    "ISIC-2017_Training_Part1_GroundTruth.zip":   ("trainy",      True),
    "ISIC-2017_Validation_Data.zip":              ("validationx", False),
    "ISIC-2017_Validation_Part1_GroundTruth.zip": ("validationy", True),
    "ISIC-2017_Test_v2_Data.zip":                 ("testx",       False),
    "ISIC-2017_Test_v2_Part1_GroundTruth.zip":    ("testy",       True),   # ← test GT
}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract all image files from zip into dest (flat — no sub-dirs)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist()
                   if Path(m).suffix.lower() in IMAGE_EXTS
                   and not m.startswith("__MACOSX")]
        print(f"  Extracting {len(members)} files → {dest.name}/")
        for member in members:
            filename = Path(member).name        # flatten nested dirs
            (dest / filename).write_bytes(zf.read(member))
    return count_images(dest)


def verify_pairs(img_dir: Path, mask_dir: Path, label: str):
    """Print image ↔ mask pairing stats for a split."""
    if not img_dir.exists() or not mask_dir.exists():
        print(f"  [SKIP] {label}: folder(s) missing")
        return
    imgs  = {f.stem for f in img_dir.iterdir()  if f.suffix.lower() in IMAGE_EXTS}
    masks = {f.stem.replace("_segmentation", "")
             for f in mask_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    paired = imgs & masks
    unmatched_img  = imgs  - masks
    unmatched_mask = masks - imgs
    status = "✓" if not unmatched_img else "⚠"
    print(f"  {status} {label:20s}  paired={len(paired)}"
          + (f"  img_no_mask={len(unmatched_img)}" if unmatched_img  else "")
          + (f"  mask_no_img={len(unmatched_mask)}" if unmatched_mask else ""))


def main():
    print("=" * 65)
    print("  ISIC-2017 Dataset Setup")
    print("=" * 65)

    # Check for missing ZIPs
    missing = [z for z in ZIP_MAP if not (DATASET / z).exists()]
    if missing:
        print("\n[WARNING] Missing ZIP files in Dataset/:")
        for m in missing:
            print(f"  ✗ {m}")
        print("  Download from: https://challenge.isic-archive.com/data/#2017")
    else:
        print("\n  All 6 ZIP files found ✓")

    # Extract each ZIP
    for zip_name, (subfolder, _is_mask) in ZIP_MAP.items():
        zp   = DATASET / zip_name
        dest = EXTRACTED / subfolder

        if not zp.exists():
            print(f"\n[SKIP] {zip_name}  (not found)")
            continue

        existing = count_images(dest)
        if existing > 0:
            print(f"\n[SKIP] {dest.name:12s} already has {existing:4d} files"
                  "  (delete folder to re-extract)")
            continue

        print(f"\n[EXTRACT] {zip_name}")
        n = extract_zip(zp, dest)
        print(f"  ✓ {n} files extracted to {dest.name}/")

    # Summary table
    print("\n" + "=" * 65)
    print(f"  {'Folder':<15}  {'Files':>6}  Role")
    print("  " + "-" * 50)
    roles = {
        "trainx":      "training images",
        "trainy":      "training masks   (*_segmentation.png)",
        "validationx": "validation images",
        "validationy": "validation masks  (*_segmentation.png)",
        "testx":       "test images",
        "testy":       "test masks        (*_segmentation.png)",
    }
    for subfolder, role in roles.items():
        n = count_images(EXTRACTED / subfolder)
        flag = "⚠  missing" if n == 0 else ""
        print(f"  {subfolder:<15}  {n:6d}  {role}  {flag}")

    # Pairing verification
    print()
    verify_pairs(EXTRACTED / "trainx",      EXTRACTED / "trainy",      "Training")
    verify_pairs(EXTRACTED / "validationx", EXTRACTED / "validationy", "Validation")
    verify_pairs(EXTRACTED / "testx",       EXTRACTED / "testy",       "Test")

    print("\n" + "=" * 65)
    print("  [DONE]  Run training with:")
    print("    python clipsam.py --batch-size 16 --num-workers 8")
    print("=" * 65)


if __name__ == "__main__":
    main()
