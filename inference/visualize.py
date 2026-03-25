# visualize.py — View prediction mask (and optionally overlay on input patch)
#
# Usage:
#   python inference/visualize.py outputs/my_result.npz
#   python inference/visualize.py outputs/my_result.npz --input patches/val/BraTS-GLI-00006-000_patch_0000.npz
#   python inference/visualize.py outputs/my_result.npz --save outputs/my_viz.png

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from project_config import OUTPUTS_DIR as PROJECT_OUTPUTS_DIR

OUTPUTS_DIR = PROJECT_OUTPUTS_DIR

# BraTS class colors (R, G, B) for overlay — match monitor.py
CLASS_COLORS = {
    1: [1.0, 0.2, 0.2],   # NCR/NET — red
    2: [0.2, 1.0, 0.2],   # Edema — green
    3: [0.2, 0.4, 1.0],   # ET — blue
}
CLASS_NAMES = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "ET"}


def mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    """Convert 2D mask (H,W) with values 0,1,2,3 to RGB (H,W,3)."""
    out = np.zeros((*mask_2d.shape, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        out[mask_2d == cls_id] = color
    return out


def normalize_slice(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize 2D slice for display."""
    a, b = np.percentile(img, [p_low, p_high])
    if b <= a:
        return np.zeros_like(img)
    return np.clip((img.astype(np.float32) - a) / (b - a), 0, 1)


def main():
    p = argparse.ArgumentParser(description="Visualize BraTS prediction mask")
    p.add_argument("prediction", help="Path to prediction .npz (contains 'mask')")
    p.add_argument("--input", default=None, help="Path to original patch .npz for background (images + optional label)")
    p.add_argument("--save", default=None, help="Path to save PNG; default: outputs/<pred_stem>_viz.png")
    args = p.parse_args()

    pred_path = Path(args.prediction)
    if not pred_path.exists():
        raise FileNotFoundError(f"Not found: {pred_path}")

    with np.load(pred_path) as f:
        if "mask" not in f:
            raise ValueError(f"No 'mask' key in {pred_path}. Keys: {list(f.keys())}")
        mask = np.array(f["mask"])

    if mask.ndim != 3 or mask.shape[0] != 96:
        raise ValueError(f"Expected mask shape (96,96,96), got {mask.shape}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.input:
        # Overlay on input: load patch and show T1ce + prediction (and GT if present)
        inp_path = Path(args.input)
        if not inp_path.exists():
            raise FileNotFoundError(f"Not found: {inp_path}")
        with np.load(inp_path) as d:
            images = np.array(d["images"])  # (4, 96, 96, 96)
            gt = None
            if "label" in d:
                gt = np.array(d["label"])
                if gt.ndim == 4:
                    gt = gt[0]
        # Pick slice with most tumor in prediction
        depth = mask.shape[0]
        mid = depth // 2
        for s in range(depth):
            if (mask[s] > 0).sum() > (mask[mid] > 0).sum():
                mid = s
        t1ce = images[1, mid]  # channel 1 = T1ce
        t1ce_norm = normalize_slice(t1ce)
        pred_rgb = mask_to_rgb(mask[mid])

        n_cols = 3 if gt is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.patch.set_facecolor("#1a1a2e")
        axes[0].imshow(t1ce_norm, cmap="gray")
        axes[0].set_title("T1ce", color="white", fontsize=12)
        axes[0].axis("off")
        axes[0].set_facecolor("#1a1a2e")

        if gt is not None:
            gt_2d = gt[mid] if gt.ndim == 3 else gt[0, mid]
            gt_rgb = mask_to_rgb(gt_2d)
            axes[1].imshow(t1ce_norm, cmap="gray")
            axes[1].imshow(gt_rgb, alpha=0.5)
            axes[1].set_title("Ground truth", color="white", fontsize=12)
            axes[1].axis("off")
            axes[1].set_facecolor("#1a1a2e")
            ax_pred = axes[2]
        else:
            ax_pred = axes[1]
        ax_pred.imshow(t1ce_norm, cmap="gray")
        ax_pred.imshow(pred_rgb, alpha=0.5)
        ax_pred.set_title("Prediction", color="white", fontsize=12)
        ax_pred.axis("off")
        ax_pred.set_facecolor("#1a1a2e")

        # Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=CLASS_COLORS[k], linewidth=3, label=CLASS_NAMES[k])
                   for k in (1, 2, 3)]
        fig.legend(handles=handles, loc="lower center", ncol=3, facecolor="#1a1a2e",
                   labelcolor="white", fontsize=10)
    else:
        # No input: show three orthogonal slices of the mask only
        D, H, W = mask.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
        axial = mask[mid_d]   # (H, W)
        coronal = mask[:, mid_h, :]  # (D, W)
        sagittal = mask[:, :, mid_w]  # (D, H)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")
        for ax, slc, title in [
            (axes[0], axial, "Axial"),
            (axes[1], coronal, "Coronal"),
            (axes[2], sagittal, "Sagittal"),
        ]:
            ax.imshow(mask_to_rgb(slc))
            ax.set_title(title, color="white", fontsize=12)
            ax.axis("off")
            ax.set_facecolor("#1a1a2e")
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=CLASS_COLORS[k], linewidth=3, label=CLASS_NAMES[k])
                   for k in (1, 2, 3)]
        fig.legend(handles=handles, loc="lower center", ncol=3, facecolor="#1a1a2e",
                   labelcolor="white", fontsize=10)

    plt.tight_layout()
    out_path = Path(args.save) if args.save else OUTPUTS_DIR / f"{pred_path.stem}_viz.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor="#1a1a2e", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
