# post_process.py — BraTS post-processing pipeline
#
# 1. Connected component filtering: remove small isolated components per class
# 2. BraTS hierarchy enforcement: ET ⊆ TC ⊆ WT (largest WT component)
# 3. Hole filling: fill small holes inside tumor regions
#
# Usage:
#   from post_process import post_process_brats
#   mask_clean = post_process_brats(mask, min_component_size=50)

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes


# BraTS labels: 0=Background, 1=NCR/NET, 2=Edema, 3=ET
# WT = Whole Tumor = 1+2+3,  TC = Tumor Core = 1+3,  ET = Enhancing = 3
LABEL_BG = 0
LABEL_NCR = 1
LABEL_EDEMA = 2
LABEL_ET = 3


def remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    For each foreground class (1, 2, 3), keep only the largest connected component
    and remove components with fewer than min_size voxels. Background (0) unchanged.
    """
    out = np.zeros_like(mask)
    out[mask == LABEL_BG] = LABEL_BG

    for label_id in (LABEL_ET, LABEL_EDEMA, LABEL_NCR):  # process ET first so we don't merge
        binary = (mask == label_id)
        if not binary.any():
            continue
        labeled, num = label(binary)
        if num == 0:
            continue
        sizes = ndimage.sum(binary, labeled, range(1, num + 1))
        keep = 1 + np.where(sizes >= min_size)[0]
        if len(keep) == 0:
            # keep largest even if below min_size
            keep = [1 + np.argmax(sizes)]
        for k in keep:
            out[labeled == k] = label_id
    return out


def enforce_brats_hierarchy(mask: np.ndarray) -> np.ndarray:
    """
    Enforce ET ⊆ TC ⊆ WT:
    - WT = union of 1,2,3; take largest connected component of WT
    - Zero out any tumor (1,2,3) outside this WT component
    - TC and ET remain as-is inside WT
    """
    wt_binary = (mask >= 1) & (mask <= 3)
    if not wt_binary.any():
        return mask
    labeled, num = label(wt_binary)
    if num == 0:
        return mask
    sizes = ndimage.sum(wt_binary, labeled, range(1, num + 1))
    largest_id = 1 + np.argmax(sizes)
    wt_keep = (labeled == largest_id)
    out = np.zeros_like(mask)
    out[mask == LABEL_BG] = LABEL_BG
    out[wt_keep & (mask == LABEL_NCR)] = LABEL_NCR
    out[wt_keep & (mask == LABEL_EDEMA)] = LABEL_EDEMA
    out[wt_keep & (mask == LABEL_ET)] = LABEL_ET
    return out


def fill_holes(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Fill small holes inside each foreground class using binary_fill_holes per 2D slice
    along the given axis (faster and less memory than 3D).
    """
    out = mask.copy()
    for label_id in (LABEL_NCR, LABEL_EDEMA, LABEL_ET):
        binary = (mask == label_id)
        if not binary.any():
            continue
        filled = np.zeros_like(binary)
        for idx in range(binary.shape[axis]):
            if axis == 0:
                slice_2d = binary[idx]
                filled_slice = binary_fill_holes(slice_2d)
                filled[idx] = filled_slice
            elif axis == 1:
                slice_2d = binary[:, idx, :]
                filled_slice = binary_fill_holes(slice_2d)
                filled[:, idx, :] = filled_slice
            else:
                slice_2d = binary[:, :, idx]
                filled_slice = binary_fill_holes(slice_2d)
                filled[:, :, idx] = filled_slice
        out[filled & (out == LABEL_BG)] = label_id
    return out


def post_process_brats(
    mask: np.ndarray,
    min_component_size: int = 50,
    enforce_hierarchy: bool = True,
    fill_holes_flag: bool = True,
) -> np.ndarray:
    """
    Full BraTS post-processing pipeline.

    Args:
        mask: (D, H, W) int array with values in {0, 1, 2, 3}
        min_component_size: remove components smaller than this (voxels)
        enforce_hierarchy: keep only largest WT component
        fill_holes_flag: fill holes inside tumor regions

    Returns:
        Post-processed mask, same shape and dtype
    """
    out = np.asarray(mask, dtype=np.int64)
    out = remove_small_components(out, min_size=min_component_size)
    if enforce_hierarchy:
        out = enforce_brats_hierarchy(out)
    if fill_holes_flag:
        out = fill_holes(out)
    return out
