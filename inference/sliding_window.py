# sliding_window.py — Full-volume sliding window inference for BraTS
#
# Loads NIfTI (or 4D numpy), extracts overlapping 96³ patches, runs ensemble
# with Gaussian-weighted stitching, optionally post-processes, saves NIfTI.
#
# Usage:
#   python inference/sliding_window.py volume.nii.gz --output seg.nii.gz
#   python inference/sliding_window.py volume.nii.gz --device cuda --stride 48

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

OUTPUTS_DIR = Path(r"E:\data\outputs")
PATCH_SIZE = 96
DEFAULT_STRIDE = 32  # 67% overlap — better boundary stitching


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI volume. Expects 4D (D, H, W, 4) or 4 separate files.
    Returns (images, affine) with images shape (4, D, H, W) float32.
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()

    if data.ndim == 3:
        raise ValueError("Single 3D NIfTI given. BraTS needs 4 modalities (T1, T1ce, T2, FLAIR). Use a 4D file or provide 4 paths.")
    if data.ndim == 4:
        # Assume (D, H, W, C) or (C, D, H, W)
        if data.shape[-1] == 4:
            data = np.moveaxis(data, -1, 0)  # (D,H,W,4) -> (4,D,H,W)
        elif data.shape[0] == 4:
            data = np.ascontiguousarray(data)
        else:
            raise ValueError(f"4D volume must have 4 modalities; got shape {data.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI; got shape {data.shape}")

    data = data.astype(np.float32)
    return data, affine


def normalize_volume(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Per-channel percentile-based normalization to match preprocessed patches.
    Clips to [p_low, p_high] percentile then z-score per channel.
    """
    out = np.empty_like(vol)
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


def make_gaussian_weights(size: int, sigma: float | None = None) -> np.ndarray:
    """3D Gaussian weight array (center=1, edges~0). sigma default size/4."""
    if sigma is None:
        sigma = size / 4.0
    co = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g1d = np.exp(-0.5 * (co / sigma) ** 2)
    g3d = np.einsum("i,j,k->ijk", g1d, g1d, g1d).astype(np.float32)
    g3d /= g3d.max()
    return g3d


def sliding_window_inference(
    ensemble,
    volume: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = 1,
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
    return_case: bool = False,
):
    """
    Run ensemble on overlapping patches and stitch with Gaussian weighting.
    volume: (4, D, H, W) float32 normalized.
    progress_callback: optional (step, detail, progress_pct).
    return_case: if True and ensemble has case cls head, also average case logits over patches.
    Returns: (D, H, W) int64 mask, or (mask, case_meta_dict) when return_case and meta available.
    """
    _, D, H, W = volume.shape
    # Pad so we have full coverage
    def pad_to_cover(L, patch_len, s):
        n = max(1, int(np.ceil((L - patch_len) / s)) + 1)
        need = (n - 1) * s + patch_len
        return need if need > L else L, 0 if need <= L else need - L

    Dp, pd = pad_to_cover(D, patch_size, stride)
    Hp, ph = pad_to_cover(H, patch_size, stride)
    Wp, pw = pad_to_cover(W, patch_size, stride)

    pad_width = ((0, 0), (0, pd), (0, ph), (0, pw))
    vol_padded = np.pad(volume, pad_width, mode="edge").astype(np.float32)

    gaussian = make_gaussian_weights(patch_size)
    num_classes = 4
    out_soft = np.zeros((num_classes, Dp, Hp, Wp), dtype=np.float32)
    out_weight = np.zeros((Dp, Hp, Wp), dtype=np.float32)

    positions = []
    for d in range(0, Dp - patch_size + 1, stride):
        for h in range(0, Hp - patch_size + 1, stride):
            for w in range(0, Wp - patch_size + 1, stride):
                positions.append((d, h, w))

    n_pos = len(positions)
    num_cls = int(getattr(ensemble, "num_cls", 0) or 0)
    case_logit_sum = (
        np.zeros((num_cls,), dtype=np.float32) if return_case and num_cls > 0 else None
    )
    n_case_patches = 0

    for i in range(0, n_pos, batch_size):
        if progress_callback is not None and n_pos > 0:
            # Report progress 20-70% over patches
            pct = 20 + int((i + min(batch_size, n_pos - i)) / n_pos * 50)
            progress_callback("sliding_window", f"Patch {min(i + batch_size, n_pos)}/{n_pos}", pct)
        batch_pos = positions[i : i + batch_size]
        for (d, h, w) in batch_pos:
            patch = vol_padded[:, d : d + patch_size, h : h + patch_size, w : w + patch_size]
            probs = ensemble.predict_soft(patch)  # (4, 96, 96, 96)
            if case_logit_sum is not None:
                cl = ensemble.predict_case_logits(patch)
                if cl is not None:
                    case_logit_sum += cl
                    n_case_patches += 1
            wg = gaussian
            out_soft[:, d : d + patch_size, h : h + patch_size, w : w + patch_size] += probs * wg
            out_weight[d : d + patch_size, h : h + patch_size, w : w + patch_size] += wg

    out_weight[out_weight < 1e-8] = 1.0
    for c in range(num_classes):
        out_soft[c] /= out_weight
    mask = out_soft.argmax(axis=0).astype(np.int64)
    # Crop to original size
    mask = mask[:D, :H, :W]

    if case_logit_sum is not None and n_case_patches > 0:
        mean_logits = case_logit_sum / float(n_case_patches)
        m = float(mean_logits.max())
        ex = np.exp(mean_logits - m)
        case_probs = (ex / ex.sum()).astype(np.float32)
        case_meta = {
            "case_logits_mean": mean_logits,
            "case_probs": case_probs,
            "case_pred": int(np.argmax(mean_logits)),
            "n_patches_averaged": n_case_patches,
        }
        return mask, case_meta
    if return_case:
        return mask, {}
    return mask


def save_nifti(mask: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.int16), affine)
    nib.save(img, str(path))


def run_from_nifti(
    nifti_path: str | Path,
    ensemble=None,
    output_path: str | Path | None = None,
    device: str = "cpu",
    stride: int = DEFAULT_STRIDE,
    model_dir: str | Path | None = None,
    post_process: bool = True,
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
    return_case_meta: bool = False,
    case_json_path: str | Path | None = None,
):
    """
    Load NIfTI, normalize, run sliding-window ensemble, optionally post-process, return mask (D,H,W).
    If output_path is set, also save as NIfTI.
    return_case_meta: if True, returns (mask, case_meta_dict) when model has cls head.
    case_json_path: if set, write case_meta JSON here (even if return_case_meta is False).
    progress_callback: optional (step, detail, progress_pct).
    """
    import json

    if ensemble is None:
        from predict import BraTSEnsemble
        ensemble = BraTSEnsemble(model_dir=model_dir, device=device)

    volume, affine = load_nifti(nifti_path)
    volume = normalize_volume(volume)
    want_case = return_case_meta or case_json_path is not None
    sw_out = sliding_window_inference(
        ensemble, volume, stride=stride, progress_callback=progress_callback,
        return_case=want_case,
    )
    case_meta = None
    if isinstance(sw_out, tuple) and len(sw_out) == 2:
        mask, case_meta = sw_out
    else:
        mask = sw_out
    if post_process:
        from post_process import post_process_brats
        mask = post_process_brats(mask)
    if output_path:
        save_nifti(mask, affine, output_path)
    if case_json_path and case_meta:
        Path(case_json_path).parent.mkdir(parents=True, exist_ok=True)
        # JSON-serializable (numpy -> list)
        serial = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in case_meta.items()}
        Path(case_json_path).write_text(json.dumps(serial, indent=2), encoding="utf-8")
    if return_case_meta:
        return mask, case_meta or {}
    return mask


def main():
    import argparse
    from predict import BraTSEnsemble

    p = argparse.ArgumentParser(description="Full-volume sliding window inference")
    p.add_argument("input", help="Path to 4D NIfTI (.nii or .nii.gz)")
    p.add_argument("--output", "-o", default=None, help="Output NIfTI path; default: outputs/<stem>_seg.nii.gz")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding window stride (default 48)")
    p.add_argument("--models", default=None, help="Directory with run_1_best.pt, run_2_best.pt, run_3_best.pt")
    p.add_argument("--no-post-process", action="store_true", help="Skip BraTS post-processing")
    p.add_argument(
        "--case-json", default=None,
        help="If set, write case-level classification JSON (mean logits/probs over patches).",
    )
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else OUTPUTS_DIR / f"{inp.stem.replace('.nii', '')}_seg.nii.gz"

    print("Loading ensemble...")
    ensemble = BraTSEnsemble(model_dir=args.models, device=args.device)
    print("Loading and normalizing volume...")
    cj = Path(args.case_json) if args.case_json and getattr(ensemble, "num_cls", 0) > 0 else None
    mask = run_from_nifti(
        inp, ensemble=ensemble, output_path=out, device=args.device, stride=args.stride,
        post_process=not args.no_post_process,
        case_json_path=cj,
    )
    print(f"Segmentation shape: {mask.shape}")
    print(f"Saved: {out}")
    if cj and cj.exists():
        print(f"Case classification: {cj}")


if __name__ == "__main__":
    main()
