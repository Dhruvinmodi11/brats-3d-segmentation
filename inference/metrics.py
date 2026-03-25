# metrics.py — Official BraTS-style metrics: Dice, HD95, Sensitivity, Specificity
#
# Per-class (NCR, Edema, ET) and for composite regions (WT, TC) as used in BraTS.
# spacing_mm: (d, h, w) voxel spacing in mm for HD95; default (1,1,1).

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_erosion

CLASS_NAMES = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "ET"}
# BraTS composite: WT = 1+2+3, TC = 1+3, ET = 3
COMPOSITE = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3)}


def _binary_surface(binary: np.ndarray) -> np.ndarray:
    """Voxels on the boundary of the binary mask (erosion removes interior)."""
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    eroded = binary_erosion(binary)
    return binary & ~eroded


def hausdorff_95(pred_binary: np.ndarray, gt_binary: np.ndarray, spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile of symmetric Hausdorff distance (in mm).
    If either mask is empty, returns 0 if both empty else a large penalty (e.g. 373 mm).
    """
    if not gt_binary.any() and not pred_binary.any():
        return 0.0
    if not gt_binary.any() or not pred_binary.any():
        return 373.0  # BraTS convention for missing lesion

    spacing = np.array(spacing_mm, dtype=np.float64)
    # Distance from every voxel to nearest gt voxel (in voxels; we scale by spacing later)
    dist_gt = distance_transform_edt(~gt_binary, sampling=spacing)
    dist_pred = distance_transform_edt(~pred_binary, sampling=spacing)

    pred_surf = _binary_surface(pred_binary)
    gt_surf = _binary_surface(gt_binary)

    if not pred_surf.any():
        d_pred_to_gt = np.array([0.0])
    else:
        d_pred_to_gt = dist_gt[pred_surf]
    if not gt_surf.any():
        d_gt_to_pred = np.array([0.0])
    else:
        d_gt_to_pred = dist_pred[gt_surf]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if len(all_d) == 0:
        return 0.0
    return float(np.percentile(all_d, 95))


def dice_per_class(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Dice for a single class (0-3). Returns 1.0 if both empty."""
    pred_c = (pred == class_id)
    gt_c = (gt == class_id)
    inter = (pred_c & gt_c).sum()
    union = pred_c.sum() + gt_c.sum()
    if union == 0:
        return 1.0
    return float(2.0 * inter / union)


def sensitivity_specificity(pred: np.ndarray, gt: np.ndarray, class_id: int) -> tuple[float, float]:
    """Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP) for class_id. Positive = class_id."""
    pred_c = (pred == class_id)
    gt_c = (gt == class_id)
    tp = (pred_c & gt_c).sum()
    fn = (~pred_c & gt_c).sum()
    fp = (pred_c & ~gt_c).sum()
    tn = (~pred_c & ~gt_c).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    return float(sens), float(spec)


def iou_per_class(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Jaccard / IoU for a single class. Returns 1.0 if both empty."""
    pred_c = (pred == class_id)
    gt_c = (gt == class_id)
    inter = (pred_c & gt_c).sum()
    union = pred_c.sum() + gt_c.sum() - inter
    if union == 0:
        return 1.0
    return float(inter / union)


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray, class_id: int) -> tuple[float, float, float]:
    """Precision, Recall, F1 for a single class."""
    pred_c = (pred == class_id)
    gt_c = (gt == class_id)
    tp = (pred_c & gt_c).sum()
    fp = (pred_c & ~gt_c).sum()
    fn = (~pred_c & gt_c).sum()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def average_surface_distance(pred_binary: np.ndarray, gt_binary: np.ndarray,
                             spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Average Surface Distance (ASD): mean distance from pred surface to GT."""
    if not gt_binary.any() and not pred_binary.any():
        return 0.0
    if not gt_binary.any() or not pred_binary.any():
        return 373.0
    pred_surf = _binary_surface(pred_binary)
    if not pred_surf.any():
        return 0.0
    dist_gt = distance_transform_edt(~gt_binary, sampling=spacing_mm)
    return float(dist_gt[pred_surf].mean())


def average_symmetric_surface_distance(pred_binary: np.ndarray, gt_binary: np.ndarray,
                                       spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """ASSD: average of ASD(pred->gt) and ASD(gt->pred)."""
    asd_fwd = average_surface_distance(pred_binary, gt_binary, spacing_mm)
    asd_bwd = average_surface_distance(gt_binary, pred_binary, spacing_mm)
    return (asd_fwd + asd_bwd) / 2.0


def normalized_surface_distance(pred_binary: np.ndarray, gt_binary: np.ndarray,
                                spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
                                tau_mm: float = 2.0) -> float:
    """NSD: fraction of surface points within tolerance tau_mm."""
    if not gt_binary.any() and not pred_binary.any():
        return 1.0
    if not gt_binary.any() or not pred_binary.any():
        return 0.0
    pred_surf = _binary_surface(pred_binary)
    gt_surf = _binary_surface(gt_binary)
    if not pred_surf.any() or not gt_surf.any():
        return 0.0
    dist_gt = distance_transform_edt(~gt_binary, sampling=spacing_mm)
    dist_pred = distance_transform_edt(~pred_binary, sampling=spacing_mm)
    n_pred = pred_surf.sum()
    n_gt = gt_surf.sum()
    within_pred = (dist_gt[pred_surf] <= tau_mm).sum()
    within_gt = (dist_pred[gt_surf] <= tau_mm).sum()
    return float((within_pred + within_gt) / (n_pred + n_gt))


def volume_similarity(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Volume Similarity: 1 - |Vp - Vg| / (Vp + Vg). Range [0, 1], higher is better."""
    vp = pred_binary.sum()
    vg = gt_binary.sum()
    if (vp + vg) == 0:
        return 1.0
    return float(1.0 - abs(vp - vg) / (vp + vg))


def relative_volume_difference(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Relative Volume Difference: (Vp - Vg) / Vg. 0 = perfect, positive = over-seg."""
    vg = gt_binary.sum()
    vp = pred_binary.sum()
    if vg == 0:
        return 0.0 if vp == 0 else float("inf")
    return float((vp - vg) / vg)


def _connected_components_3d(mask: np.ndarray):
    """Return (labeled_array, num_features) using scipy."""
    return ndimage.label(mask)


def lesion_wise_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Lesion-wise Dice: compute Dice per connected GT component, then average.
    Penalises missed lesions heavily."""
    gt_c = (gt == class_id)
    pred_c = (pred == class_id)
    if not gt_c.any():
        return 1.0 if not pred_c.any() else 0.0

    labeled_gt, n_gt = _connected_components_3d(gt_c)
    dices = []
    for i in range(1, n_gt + 1):
        lesion_mask = (labeled_gt == i)
        inter = (pred_c & lesion_mask).sum()
        union_sum = pred_c.sum() + lesion_mask.sum()
        if union_sum == 0:
            dices.append(1.0)
        else:
            dices.append(float(2 * inter / union_sum))
    return float(np.mean(dices)) if dices else 0.0


def compute_brats_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    per_class: bool = True,
    composites: bool = True,
    extended: bool = False,
) -> dict:
    """
    Compute metrics for BraTS evaluation.

    Base metrics (always): Dice, HD95, Sensitivity, Specificity.
    Extended metrics (extended=True): IoU, Precision, Recall, F1,
        ASD, ASSD, NSD, Volume Similarity, RVD, Lesion-wise Dice.

    pred, gt: (D,H,W) int with values in {0,1,2,3}.
    """
    out = {}
    classes = [1, 2, 3] if per_class else []
    for c in classes:
        pred_c = (pred == c)
        gt_c = (gt == c)
        out[f"Dice_c{c}"] = dice_per_class(pred, gt, c)
        out[f"HD95_c{c}"] = hausdorff_95(pred_c, gt_c, spacing_mm)
        sens, spec = sensitivity_specificity(pred, gt, c)
        out[f"Sensitivity_c{c}"] = sens
        out[f"Specificity_c{c}"] = spec

        if extended:
            out[f"IoU_c{c}"] = iou_per_class(pred, gt, c)
            prec, rec, f1 = precision_recall_f1(pred, gt, c)
            out[f"Precision_c{c}"] = prec
            out[f"Recall_c{c}"] = rec
            out[f"F1_c{c}"] = f1
            out[f"ASD_c{c}"] = average_surface_distance(pred_c, gt_c, spacing_mm)
            out[f"ASSD_c{c}"] = average_symmetric_surface_distance(pred_c, gt_c, spacing_mm)
            out[f"NSD_c{c}"] = normalized_surface_distance(pred_c, gt_c, spacing_mm)
            out[f"VolSim_c{c}"] = volume_similarity(pred_c, gt_c)
            out[f"RVD_c{c}"] = relative_volume_difference(pred_c, gt_c)
            out[f"LesionDice_c{c}"] = lesion_wise_dice(pred, gt, c)

    if composites:
        for name, labels in COMPOSITE.items():
            pred_bin = np.isin(pred, labels)
            gt_bin = np.isin(gt, labels)
            inter = (pred_bin & gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum()
            out[f"Dice_{name}"] = float(2.0 * inter / union) if union > 0 else 1.0
            out[f"HD95_{name}"] = hausdorff_95(pred_bin, gt_bin, spacing_mm)
            pred_bin_b = pred_bin.astype(bool)
            gt_bin_b = gt_bin.astype(bool)
            tp = (pred_bin_b & gt_bin_b).sum()
            fn = (~pred_bin_b & gt_bin_b).sum()
            fp = (pred_bin_b & ~gt_bin_b).sum()
            tn = (~pred_bin_b & ~gt_bin_b).sum()
            out[f"Sensitivity_{name}"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            out[f"Specificity_{name}"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

            if extended:
                union_iou = pred_bin.sum() + gt_bin.sum() - inter
                out[f"IoU_{name}"] = float(inter / union_iou) if union_iou > 0 else 1.0
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
                out[f"Precision_{name}"] = prec
                out[f"Recall_{name}"] = rec
                out[f"F1_{name}"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                out[f"ASD_{name}"] = average_surface_distance(pred_bin_b, gt_bin_b, spacing_mm)
                out[f"ASSD_{name}"] = average_symmetric_surface_distance(pred_bin_b, gt_bin_b, spacing_mm)
                out[f"NSD_{name}"] = normalized_surface_distance(pred_bin_b, gt_bin_b, spacing_mm)
                out[f"VolSim_{name}"] = volume_similarity(pred_bin_b, gt_bin_b)
                out[f"RVD_{name}"] = relative_volume_difference(pred_bin_b, gt_bin_b)

    if per_class:
        out["Dice_mean_fg"] = np.mean([out[f"Dice_c{c}"] for c in (1, 2, 3)])
        if extended:
            out["IoU_mean_fg"] = np.mean([out[f"IoU_c{c}"] for c in (1, 2, 3)])
            out["ASSD_mean_fg"] = np.mean([out[f"ASSD_c{c}"] for c in (1, 2, 3)])
            out["NSD_mean_fg"] = np.mean([out[f"NSD_c{c}"] for c in (1, 2, 3)])
    return out


def dice_per_class_simple(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Alias for compatibility."""
    return dice_per_class(pred, gt, class_id)
