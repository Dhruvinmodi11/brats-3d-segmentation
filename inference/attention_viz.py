# attention_viz.py — Extract and visualize attention maps from UNet3DAttn AttentionGate modules
#
# Uses forward hooks to capture the sigmoid attention 'a' from attn1, attn2, attn3.
# Multi-scale: attn3 (coarse, ~12³), attn2 (~24³), attn1 (fine, 96³).
#
# Usage:
#   from attention_viz import get_attention_maps
#   maps = get_attention_maps(model, tensor_input)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "training"))
from project_config import OUTPUTS_DIR as PROJECT_OUTPUTS_DIR

# Store last captured attention per module (for hook)
_attention_store = {}


def _make_hook(name: str):
    def hook(module, input, output):
        # output = x * a, with a shape (B, 1, D, H, W). Recover a = output / (x + eps)
        x = input[1]
        a = output[:, :1] / (x[:, :1].clamp(min=1e-8) + 1e-8)
        _attention_store[name] = a.detach()
    return hook


def get_attention_maps(model: torch.nn.Module, x: torch.Tensor) -> dict[str, np.ndarray]:
    """
    Run one forward pass and capture attention from attn1, attn2, attn3.
    x: (1, 4, D, H, W) tensor.
    Returns dict with keys "attn1", "attn2", "attn3"; values (D, H, W) float32 numpy.
    """
    global _attention_store
    _attention_store = {}
    hooks = []
    for name in ("attn1", "attn2", "attn3", "attn4"):
        sub = getattr(model, name, None)
        if sub is not None:
            h = sub.register_forward_hook(_make_hook(name))
            hooks.append(h)
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for name in ("attn1", "attn2", "attn3", "attn4"):
        if name in _attention_store:
            a = _attention_store[name].squeeze().cpu().numpy()
            if a.ndim == 3:
                out[name] = a.astype(np.float32)
            else:
                out[name] = a[0].astype(np.float32)
    return out


def attention_maps_to_figure(attention_maps: dict[str, np.ndarray], slice_idx: int | None = None) -> "matplotlib.figure.Figure":
    """Produce a 1x3 figure of attn1, attn2, attn3 mid-slice (or slice_idx)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    keys = ["attn1", "attn2", "attn3"]
    for ax, key in zip(axes, keys):
        if key not in attention_maps:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        arr = attention_maps[key]
        # Each map has different spatial size (attn1=96, attn2=48, attn3=24); use valid index
        depth = arr.shape[0] if arr.ndim == 3 else 1
        idx = (depth // 2) if slice_idx is None else min(slice_idx, depth - 1)
        slc = arr[idx] if arr.ndim == 3 else arr
        im = ax.imshow(slc, cmap="hot", vmin=0, vmax=1)
        ax.set_title(key, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def main():
    import argparse
    import matplotlib.pyplot as plt
    from predict import BraTSEnsemble

    p = argparse.ArgumentParser(description="Extract and save attention maps for a patch")
    p.add_argument("input", help="Path to .npz patch")
    p.add_argument("--output", default=None, help="Output PNG path")
    p.add_argument("--model-index", type=int, default=0, help="Which ensemble model (0, 1, 2)")
    p.add_argument("--slice", type=int, default=None, help="Slice index; default mid")
    args = p.parse_args()

    with np.load(args.input) as d:
        images = np.array(d["images"], dtype=np.float32)
    x = torch.from_numpy(images).unsqueeze(0)  # (1, 4, 96, 96, 96)

    ensemble = BraTSEnsemble(device="cpu")
    model = ensemble.models[args.model_index]
    maps = get_attention_maps(model, x)
    fig = attention_maps_to_figure(maps, slice_idx=args.slice)
    out_path = Path(args.output) if args.output else PROJECT_OUTPUTS_DIR / "attention_maps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
