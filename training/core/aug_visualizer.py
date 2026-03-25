"""
aug_visualizer.py

Augmentation visualization + logging for BraTS 2023 cyclic training.
Called every 5th epoch from trial_cyclic.py (main process, after epoch ends).

For a randomly selected training patch, applies augmentation step-by-step and saves:
  - PNG  : 6-row x 5-col figure (T1ce + FLAIR x 3 planes x 5 transform steps)
           with tumor contour overlay and per-panel slice index annotation
  - JSON : full metadata per event (transforms, per-channel stats, label dist,
           slice indices, display percentiles)
  - CSV  : one row appended to aug_logs/aug_summary.csv (shared across all runs)

Output layout:
  New Trial/aug_logs/
      run_1_epoch_005_aug_viz.png
      run_1_epoch_005_aug_meta.json
      run_1_epoch_010_aug_viz.png
      run_1_epoch_010_aug_meta.json
      ...
      aug_summary.csv
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — must be set before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np

from core.augment import random_flip, random_rotation_90, gaussian_noise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
_VIZ_CHANNELS  = [(1, "T1ce"), (3, "FLAIR")]          # (index, name) to display
_CLASS_COLORS  = {1: "yellow", 2: "cyan", 3: "red"}   # tumor class contour colours
_CLASS_LABELS  = {1: "NCR/NET", 2: "ED", 3: "ET"}
_PLANES        = ["axial", "coronal", "sagittal"]

# Intensity augmentation ranges (subset of apply_v2_augmentation)
_INTENSITY_SHIFT_RANGE = 0.05
_INTENSITY_SCALE_RANGE = (0.95, 1.05)
_NOISE_PROB            = 0.3
_NOISE_SIGMA_RANGE     = (0.01, 0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tumor_slice(label: np.ndarray, axis: int) -> int:
    """Index along `axis` whose 2-D cross-section has the most tumor voxels.
    Falls back to the middle slice if no tumor is present."""
    counts = (label > 0).sum(axis=tuple(i for i in range(label.ndim) if i != axis))
    best = int(np.argmax(counts))
    return best if int(counts[best]) > 0 else label.shape[axis] // 2


def _get_slice(images: np.ndarray, label: np.ndarray, plane: str, idx: int):
    """Return (img_2d, lbl_2d) for the given plane / slice index.

    images : (C, D, H, W)   label : (D, H, W)
    Returns img_2d (C, *, *) and lbl_2d (*, *)
    """
    if plane == "axial":
        return images[:, idx, :, :], label[idx, :, :]
    elif plane == "coronal":
        return images[:, :, idx, :], label[:, idx, :]
    else:                               # sagittal
        return images[:, :, :, idx], label[:, :, idx]


def _norm_display(channel_2d: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Clip to [p_low, p_high] and scale to [0, 1] for display."""
    arr  = np.clip(channel_2d, p_low, p_high)
    span = p_high - p_low
    return (arr - p_low) / (span + 1e-8)


def _channel_stats(images: np.ndarray) -> dict:
    """Per-channel mean / std / min / max for a (C, D, H, W) array."""
    return {
        name: {
            "mean": round(float(np.mean(images[i])), 6),
            "std":  round(float(np.std(images[i])),  6),
            "min":  round(float(np.min(images[i])),  6),
            "max":  round(float(np.max(images[i])),  6),
        }
        for i, name in enumerate(_CHANNEL_NAMES)
    }


def _label_dist(label: np.ndarray) -> dict:
    """Voxel count per BraTS label class."""
    return {
        "background":  int((label == 0).sum()),
        "NCR_NET_c1":  int((label == 1).sum()),
        "ED_c2":       int((label == 2).sum()),
        "ET_c3":       int((label == 3).sum()),
        "total_tumor": int((label  > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Step-by-step augmentation  (core transforms from V2, captures state)
# ---------------------------------------------------------------------------

def _augment_verbose(images: np.ndarray, label: np.ndarray, rng: np.random.Generator):
    """Apply augmentation step-by-step, recording every intermediate result.

    Returns
    -------
    steps : list of (step_name: str, images_np, label_np)  — 5 entries
    meta  : dict — all transform parameters actually used
    """
    img, msk = images.copy(), label.copy()
    steps = [("Original", img.copy(), msk.copy())]

    # ── Step 1 : Random flip ────────────────────────────────────────────────
    flipped_axes = []
    for a_img, a_msk, a_name in [(1, 0, "H"), (2, 1, "W"), (3, 2, "D")]:
        if rng.random() > 0.5:
            img = np.flip(img, axis=a_img).copy()
            msk = np.flip(msk, axis=a_msk).copy()
            flipped_axes.append(a_name)
    steps.append(("+Flip", img.copy(), msk.copy()))

    # ── Step 2 : Random 90° rotation (axial plane H×W) ─────────────────────
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        msk = np.rot90(msk, k=k, axes=(0, 1)).copy()
    steps.append(("+Rotation", img.copy(), msk.copy()))

    # ── Step 3 : Gaussian noise  (probabilistic p=0.3) ─────────────────────
    noise_applied = bool(rng.random() < _NOISE_PROB)
    sigma = None
    if noise_applied:
        sigma = float(rng.uniform(*_NOISE_SIGMA_RANGE))
        img   = (img + rng.normal(0, sigma, img.shape).astype(np.float32)).astype(np.float32)
    steps.append(("+Noise", img.copy(), msk.copy()))

    # ── Step 4 : Per-channel intensity shift / scale ────────────────────────
    n_ch   = img.shape[0]
    shifts = rng.uniform(-_INTENSITY_SHIFT_RANGE, _INTENSITY_SHIFT_RANGE, size=n_ch)
    scales = rng.uniform(*_INTENSITY_SCALE_RANGE, size=n_ch)
    for c in range(n_ch):
        img[c] = img[c] * scales[c] + shifts[c]
    img = img.astype(np.float32)
    steps.append(("+Intensity", img.copy(), msk.copy()))

    meta = {
        "flip": {
            "axes_flipped": flipped_axes,
            "no_flip":      len(flipped_axes) == 0,
        },
        "rotation": {
            "k":        k,
            "degrees":  k * 90,
            "plane":    "axial H×W",
            "no_rotation": k == 0,
        },
        "noise": {
            "applied":           noise_applied,
            "sigma":             round(sigma, 6) if sigma is not None else None,
            "p_threshold":       _NOISE_PROB,
            "sigma_range":       list(_NOISE_SIGMA_RANGE),
        },
        "intensity_per_channel": {
            _CHANNEL_NAMES[c]: {
                "shift": round(float(shifts[c]), 6),
                "scale": round(float(scales[c]), 6),
            }
            for c in range(n_ch)
        },
    }
    return steps, meta


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _render_figure(
    steps:       list,
    viz_slices:  dict,
    percentiles: dict,
    run_id:      int,
    epoch:       int,
    patch_id:    str,
    meta:        dict,
) -> plt.Figure:
    """
    Render the 6-row × 5-col augmentation figure.

    Row layout  (top → bottom):
        0 : Axial    – T1ce
        1 : Axial    – FLAIR
        2 : Coronal  – T1ce
        3 : Coronal  – FLAIR
        4 : Sagittal – T1ce
        5 : Sagittal – FLAIR

    Col layout (left → right):
        0 : Original
        1 : +Flip
        2 : +Rotation
        3 : +Noise   (greyed out if not applied)
        4 : +Intensity  (final)
    """
    noise_applied = meta["noise"]["applied"]

    n_rows, n_cols = 6, 5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.3 + 2.2),
        squeeze=False,
        dpi=100,
    )
    fig.subplots_adjust(hspace=0.14, wspace=0.04, top=0.90, bottom=0.05, left=0.10, right=0.99)

    # ── Suptitle ──────────────────────────────────────────────────────────
    fm        = meta
    flip_str  = ", ".join(fm["flip"]["axes_flipped"]) or "none"
    rot_str   = f'{fm["rotation"]["degrees"]}°'
    noise_str = (
        f'σ={fm["noise"]["sigma"]:.4f}'
        if noise_applied else
        f'skipped (p={_NOISE_PROB})'
    )
    intensity_summary = "  ".join(
        f'{ch}: shift={v["shift"]:+.4f} scale={v["scale"]:.4f}'
        for ch, v in fm["intensity_per_channel"].items()
    )
    suptitle = (
        f"Augmentation Visualisation  |  Run {run_id}  |  Epoch {epoch:03d}\n"
        f"Patch: {patch_id}\n"
        f"Flip [{flip_str}]   Rotation {rot_str}   Noise {noise_str}\n"
        f"Intensity (per-channel): {intensity_summary}"
    )
    fig.suptitle(suptitle, fontsize=8, y=0.995, va="top", family="monospace",
                 linespacing=1.6)

    # ── Row definitions ───────────────────────────────────────────────────
    rows_def = [
        ("axial",    1, "T1ce"),
        ("axial",    3, "FLAIR"),
        ("coronal",  1, "T1ce"),
        ("coronal",  3, "FLAIR"),
        ("sagittal", 1, "T1ce"),
        ("sagittal", 3, "FLAIR"),
    ]

    for row_i, (plane, ch_idx, ch_name) in enumerate(rows_def):
        slice_idx = viz_slices[plane]
        p_low     = percentiles[ch_name]["p1"]
        p_high    = percentiles[ch_name]["p99"]

        for col_i, (step_name, step_img, step_lbl) in enumerate(steps):
            ax = axes[row_i][col_i]
            img_2d, lbl_2d = _get_slice(step_img, step_lbl, plane, slice_idx)
            disp = _norm_display(img_2d[ch_idx], p_low, p_high)

            # ── Image ────────────────────────────────────────────────────
            if step_name == "+Noise" and not noise_applied:
                ax.imshow(disp, cmap="gray", aspect="auto",
                          interpolation="nearest", alpha=0.30)
                ax.text(
                    0.5, 0.5, "Noise\nskipped\n(p=0.3)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="white",
                    bbox=dict(facecolor="#333", alpha=0.85,
                              boxstyle="round,pad=0.35"),
                )
            else:
                ax.imshow(disp, cmap="gray", aspect="auto",
                          interpolation="nearest")

            # ── Tumor contour overlay ─────────────────────────────────────
            for cls_id, color in _CLASS_COLORS.items():
                mask = (lbl_2d == cls_id).astype(float)
                if mask.sum() > 3:
                    try:
                        ax.contour(mask, levels=[0.5], colors=[color],
                                   linewidths=0.8, alpha=0.9)
                    except Exception:
                        pass

            # ── Slice index annotation (bottom-right) ────────────────────
            ax.text(0.98, 0.02, f"s={slice_idx}",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=6, color="lightgreen", alpha=0.85)

            # ── Column header (first row only) ───────────────────────────
            if row_i == 0:
                ax.set_title(step_name, fontsize=9, pad=4, fontweight="bold")

            # ── Row label (first column only) ────────────────────────────
            if col_i == 0:
                ax.set_ylabel(
                    f"{plane.capitalize()}\n{ch_name}",
                    fontsize=8, labelpad=4,
                )

            ax.set_xticks([])
            ax.set_yticks([])

        # ── Heavier bottom border to separate plane sections ─────────────
        if row_i in (1, 3):
            for col_i in range(n_cols):
                for spine in axes[row_i][col_i].spines.values():
                    spine.set_linewidth(0.5)
                axes[row_i][col_i].spines["bottom"].set_linewidth(2.5)
                axes[row_i][col_i].spines["bottom"].set_color("#777777")

    # ── Contour legend ────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=color, linewidth=2,
                   label=f"{_CLASS_LABELS[cls_id]} (c{cls_id})")
        for cls_id, color in _CLASS_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )

    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_augmentation_viz(
    patch_path,
    run_id:      int,
    epoch:       int,
    aug_logs_dir: Path,
) -> tuple:
    """
    Generate and save the augmentation visualisation for one epoch.

    Parameters
    ----------
    patch_path   : path-like — an original (non-_aug) .npz patch to use
    run_id       : int — current run index (1 / 2 / 3)
    epoch        : int — current epoch number
    aug_logs_dir : Path — output root (New Trial/aug_logs/)

    Returns
    -------
    (png_path, json_path)  — strings, for logging
    """
    aug_logs_dir = Path(aug_logs_dir)
    aug_logs_dir.mkdir(parents=True, exist_ok=True)

    patch_path = Path(patch_path)
    patch_id   = patch_path.stem
    stem       = f"run_{run_id}_epoch_{epoch:03d}"
    ts         = datetime.now().isoformat(timespec="seconds")

    # ── Load patch ────────────────────────────────────────────────────────
    with np.load(patch_path, mmap_mode="r") as data:
        images = np.array(data["images"], dtype=np.float32)
        label  = data["label"]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
        label = np.array(label, dtype=np.int32)

    # ── Step-by-step augmentation ─────────────────────────────────────────
    rng              = np.random.default_rng()
    steps, transform_meta = _augment_verbose(images, label, rng)

    # ── Tumour-rich slices (from original label) ──────────────────────────
    viz_slices = {
        "axial":    _tumor_slice(label, axis=0),
        "coronal":  _tumor_slice(label, axis=1),
        "sagittal": _tumor_slice(label, axis=2),
    }

    # ── Display percentiles (from original, applied consistently to all steps)
    percentiles = {}
    for ch_idx, ch_name in _VIZ_CHANNELS:
        ch = images[ch_idx]
        percentiles[ch_name] = {
            "p1":  float(np.percentile(ch, 1)),
            "p99": float(np.percentile(ch, 99)),
        }

    # ── Before / after statistics ─────────────────────────────────────────
    stats_before      = _channel_stats(images)
    stats_after       = _channel_stats(steps[-1][1])
    label_dist_before = _label_dist(label)
    label_dist_after  = _label_dist(steps[-1][2])

    # ── Render and save PNG ───────────────────────────────────────────────
    png_path = aug_logs_dir / f"{stem}_aug_viz.png"
    fig = _render_figure(
        steps, viz_slices, percentiles,
        run_id, epoch, patch_id, transform_meta,
    )
    fig.savefig(png_path, bbox_inches="tight", dpi=100)
    plt.close(fig)

    # ── Build full metadata dict ──────────────────────────────────────────
    json_path = aug_logs_dir / f"{stem}_aug_meta.json"
    meta_dict = {
        "run_id":                    run_id,
        "epoch":                     epoch,
        "timestamp":                 ts,
        "patch_id":                  patch_id,
        "patch_path":                str(patch_path),
        "transforms":                transform_meta,
        "stats_before":              stats_before,
        "stats_after":               stats_after,
        "label_distribution_before": label_dist_before,
        "label_distribution_after":  label_dist_after,
        "viz_slices":                viz_slices,
        "display_percentiles":       {
            ch: {"p1": round(v["p1"], 5), "p99": round(v["p99"], 5)}
            for ch, v in percentiles.items()
        },
        "figure_path":               str(png_path),
        "json_path":                 str(json_path),
    }

    # ── Save JSON ─────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # ── Append row to shared CSV ──────────────────────────────────────────
    csv_path  = aug_logs_dir / "aug_summary.csv"
    fm        = transform_meta
    csv_row   = {
        "run_id":              run_id,
        "epoch":               epoch,
        "timestamp":           ts,
        "patch_id":            patch_id,
        # flip
        "flip_H":              "H" in fm["flip"]["axes_flipped"],
        "flip_W":              "W" in fm["flip"]["axes_flipped"],
        "flip_D":              "D" in fm["flip"]["axes_flipped"],
        # rotation
        "rotation_k":          fm["rotation"]["k"],
        "rotation_deg":        fm["rotation"]["degrees"],
        # noise
        "noise_applied":       fm["noise"]["applied"],
        "noise_sigma":         fm["noise"]["sigma"] if fm["noise"]["applied"] else "",
        # per-channel intensity
        "T1_shift":            fm["intensity_per_channel"]["T1"]["shift"],
        "T1_scale":            fm["intensity_per_channel"]["T1"]["scale"],
        "T1ce_shift":          fm["intensity_per_channel"]["T1ce"]["shift"],
        "T1ce_scale":          fm["intensity_per_channel"]["T1ce"]["scale"],
        "T2_shift":            fm["intensity_per_channel"]["T2"]["shift"],
        "T2_scale":            fm["intensity_per_channel"]["T2"]["scale"],
        "FLAIR_shift":         fm["intensity_per_channel"]["FLAIR"]["shift"],
        "FLAIR_scale":         fm["intensity_per_channel"]["FLAIR"]["scale"],
        # label
        "tumor_voxels_before": label_dist_before["total_tumor"],
        "tumor_voxels_after":  label_dist_after["total_tumor"],
        # channel stats (T1ce + FLAIR focus)
        "T1ce_mean_before":    stats_before["T1ce"]["mean"],
        "T1ce_std_before":     stats_before["T1ce"]["std"],
        "T1ce_mean_after":     stats_after["T1ce"]["mean"],
        "T1ce_std_after":      stats_after["T1ce"]["std"],
        "FLAIR_mean_before":   stats_before["FLAIR"]["mean"],
        "FLAIR_std_before":    stats_before["FLAIR"]["std"],
        "FLAIR_mean_after":    stats_after["FLAIR"]["mean"],
        "FLAIR_std_after":     stats_after["FLAIR"]["std"],
        # visualization slice indices used
        "axial_slice":         viz_slices["axial"],
        "coronal_slice":       viz_slices["coronal"],
        "sagittal_slice":      viz_slices["sagittal"],
        # output files
        "figure_path":         str(png_path),
        "json_path":           str(json_path),
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    return str(png_path), str(json_path)
