"""
Preprocess BraTS 2020 NIfTI volumes into 96³ .npz patches
matching the BraTS 2023 patch format used by our training pipeline.

BraTS 2020 structure (from Kaggle awsaf49/brats20-dataset-training-validation):
  BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
    BraTS20_Training_001/
      BraTS20_Training_001_t1.nii
      BraTS20_Training_001_t1ce.nii
      BraTS20_Training_001_t2.nii
      BraTS20_Training_001_flair.nii
      BraTS20_Training_001_seg.nii

  (BraTS 2020 validation set has NO seg masks — unusable for supervised training.)

Our target patch format:
  images: (4, 96, 96, 96) float32  — channels: [T1, T1ce, T2, FLAIR]
  label:  (1, 96, 96, 96) int64    — classes:  {0=BG, 1=NCR, 2=Edema, 3=ET}

Output structure:
  E:\\data\\patches\\brats2020\\train\\*.npz
  E:\\data\\patches\\brats2020\\val\\*.npz

Usage:
  python scripts/preprocess_brats2020.py
  python scripts/preprocess_brats2020.py --raw-dir <path> --output-dir <path>
  python scripts/preprocess_brats2020.py --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 96
STRIDE = 64
MIN_TUMOR_RATIO_DEFAULT = 0.005
VAL_RATIO_DEFAULT = 0.2
SEED_DEFAULT = 42

MODALITY_SUFFIXES = ["_t1.nii", "_t1ce.nii", "_t2.nii", "_flair.nii"]
SEG_SUFFIX = "_seg.nii"


def normalize_volume(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Per-channel percentile clip + z-score, matching BraTS 2023 preprocessing."""
    out = np.empty_like(vol, dtype=np.float32)
    for c in range(vol.shape[0]):
        channel = vol[c]
        lo, hi = np.percentile(channel, [p_low, p_high])
        if hi <= lo:
            out[c] = 0.0
            continue
        clipped = np.clip(channel, lo, hi).astype(np.float32)
        mean = clipped.mean()
        std = clipped.std()
        if std < 1e-8:
            out[c] = 0.0
        else:
            out[c] = (clipped - mean) / std
    return out


def load_brats2020_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load all 4 modalities + seg for one BraTS 2020 case.
    Returns (images, label) with shapes (4, D, H, W), (D, H, W).
    Returns None if any file is missing.
    """
    case_name = case_dir.name
    modality_files = [case_dir / f"{case_name}{suf}" for suf in MODALITY_SUFFIXES]
    seg_file = case_dir / f"{case_name}{SEG_SUFFIX}"

    for f in modality_files + [seg_file]:
        if not f.exists():
            gz = Path(str(f) + ".gz")
            if not gz.exists():
                print(f"  [SKIP] Missing file: {f}")
                return None

    channels = []
    for f in modality_files:
        p = f if f.exists() else Path(str(f) + ".gz")
        img = nib.load(str(p))
        data = np.asarray(img.dataobj, dtype=np.float32)
        channels.append(data)

    images = np.stack(channels, axis=0)

    seg_path = seg_file if seg_file.exists() else Path(str(seg_file) + ".gz")
    seg = np.asarray(nib.load(str(seg_path)).dataobj)

    # BraTS 2020 labels: {0, 1, 2, 4} — remap 4 (ET) -> 3 to match our pipeline
    seg = seg.astype(np.int64)
    seg[seg == 4] = 3

    return images, seg


def extract_patches(
    images: np.ndarray,
    label: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    min_tumor_ratio: float = MIN_TUMOR_RATIO_DEFAULT,
    skip_empty: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract overlapping 96³ patches from a full volume."""
    _, D, H, W = images.shape
    patches = []
    voxels_per_patch = patch_size ** 3

    for d in range(0, D - patch_size + 1, stride):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                img_patch = images[:, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                lbl_patch = label[d:d+patch_size, h:h+patch_size, w:w+patch_size]

                if skip_empty:
                    tumor_voxels = np.count_nonzero(lbl_patch)
                    if tumor_voxels / voxels_per_patch < min_tumor_ratio:
                        continue

                patches.append((img_patch, lbl_patch))

    return patches


def save_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    case_name: str,
    output_dir: Path,
) -> int:
    """Save a list of (image, label) patches as .npz files. Returns count saved."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped_name = case_name.replace("_", "-")

    for i, (img_patch, lbl_patch) in enumerate(patches):
        fname = f"{mapped_name}_patch_{i:04d}.npz"
        np.savez_compressed(
            output_dir / fname,
            images=img_patch.astype(np.float32),
            label=lbl_patch[np.newaxis].astype(np.int64),
        )
    return len(patches)


def process_dataset(
    raw_dir: Path,
    output_base: Path,
    patch_size: int,
    stride: int,
    min_tumor_ratio: float,
    skip_empty: bool,
    val_ratio: float,
    seed: int,
) -> None:
    """Process all BraTS 2020 cases into train/val .npz patches."""
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS20")
    ])

    if not case_dirs:
        print(f"ERROR: No BraTS20* case directories found in {raw_dir}")
        print(f"Contents: {[x.name for x in raw_dir.iterdir()][:20]}")
        sys.exit(1)

    # Shuffle and split at case level
    rng = random.Random(seed)
    shuffled = list(case_dirs)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    val_cases = set(d.name for d in shuffled[:n_val])
    train_cases = set(d.name for d in shuffled[n_val:])

    print(f"Found {len(case_dirs)} cases in {raw_dir}")
    print(f"Split: {len(train_cases)} train / {n_val} val  (seed={seed}, val_ratio={val_ratio})")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Min tumor ratio: {min_tumor_ratio}, Skip empty: {skip_empty}")
    print(f"Output train: {train_dir}")
    print(f"Output val:   {val_dir}")
    print()

    total_train = 0
    total_val = 0
    skipped = 0

    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        result = load_brats2020_case(case_dir)
        if result is None:
            skipped += 1
            continue

        images, label = result
        images = normalize_volume(images)

        patches = extract_patches(
            images, label,
            patch_size=patch_size,
            stride=stride,
            min_tumor_ratio=min_tumor_ratio,
            skip_empty=skip_empty,
        )

        if case_dir.name in val_cases:
            total_val += save_patches(patches, case_dir.name, val_dir)
        else:
            total_train += save_patches(patches, case_dir.name, train_dir)

    print()
    print("=" * 60)
    print("Done!")
    print(f"  Cases processed: {len(case_dirs) - skipped}")
    print(f"  Cases skipped:   {skipped}")
    print(f"  Train patches:   {total_train}  ({train_dir})")
    print(f"  Val patches:     {total_val}  ({val_dir})")
    print(f"  Total patches:   {total_train + total_val}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2020 into 96³ .npz patches (train/val split)")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Path to BraTS2020 training data (auto-detected if not set)")
    parser.add_argument("--output-dir", type=str, default=r"E:\data\patches\brats2020",
                        help="Base output directory (train/ and val/ created inside)")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-tumor-ratio", type=float, default=MIN_TUMOR_RATIO_DEFAULT)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO_DEFAULT,
                        help="Fraction of cases for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--skip-empty", action="store_true", default=True)
    parser.add_argument("--keep-empty", action="store_true", default=False)
    args = parser.parse_args()

    skip_empty = not args.keep_empty

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        import kagglehub
        print("Locating BraTS 2020 dataset via kagglehub...")
        dataset_path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
        print(f"Dataset path: {dataset_path}")

        raw_dir = Path(dataset_path)
        candidates = [
            raw_dir / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
            raw_dir / "BraTS2020_TrainingData",
            raw_dir / "MICCAI_BraTS2020_TrainingData",
            raw_dir,
        ]
        for c in candidates:
            if c.exists() and any(d.name.startswith("BraTS20") for d in c.iterdir() if d.is_dir()):
                raw_dir = c
                break

    print(f"Raw data directory: {raw_dir}")

    process_dataset(
        raw_dir=raw_dir,
        output_base=Path(args.output_dir),
        patch_size=args.patch_size,
        stride=args.stride,
        min_tumor_ratio=args.min_tumor_ratio,
        skip_empty=skip_empty,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
