import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

from core.parameters import EPS


def compute_dice_loss_from_logits(logits, target, exclude_background=True):
    logits = logits.float()

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits before softmax.")

    probabilities = torch.softmax(logits, dim=1)

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities after softmax.")

    first_class = 1 if exclude_background else 0
    B, C = probabilities.shape[:2]

    dice_losses = []
    for class_id in range(first_class, C):
        pred_prob = probabilities[:, class_id].reshape(B, -1)
        true_mask = (target == class_id).float().reshape(B, -1)

        intersection = (pred_prob * true_mask).sum(dim=1)
        union = pred_prob.sum(dim=1) + true_mask.sum(dim=1)

        dice = (2.0 * intersection + EPS) / (union + EPS)
        dice = torch.where(union > 0, dice, torch.ones_like(dice))
        dice_losses.append(1.0 - dice)

    if not dice_losses:
        return torch.zeros((), device=logits.device)

    return torch.stack(dice_losses, dim=0).mean()


def precompute_distance_maps(label_np: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Precompute signed distance maps for boundary loss (runs on dataloader workers).

    Args:
        label_np: (D, H, W) int array with class labels.
        num_classes: number of classes (foreground classes are 1..num_classes-1).

    Returns:
        dist_maps: (num_classes-1, D, H, W) float32 array. One signed distance
        map per foreground class. Negative inside GT, positive outside.
    """
    spatial = label_np.shape
    n_fg = num_classes - 1
    dist_maps = np.zeros((n_fg, *spatial), dtype=np.float32)

    for i, cls in enumerate(range(1, num_classes)):
        gt_bin = (label_np == cls)
        if not gt_bin.any():
            continue
        pos_dist = distance_transform_edt(~gt_bin).astype(np.float32)
        neg_dist = distance_transform_edt(gt_bin).astype(np.float32)
        dist_maps[i] = pos_dist - neg_dist

    return dist_maps


def compute_boundary_loss_from_precomputed(logits, dist_maps_tensor):
    """Boundary loss using precomputed distance maps (no CPU EDT on main thread).

    Args:
        logits: (B, C, D, H, W) raw model output.
        dist_maps_tensor: (B, num_fg_classes, D, H, W) precomputed signed distance maps.

    Returns:
        Scalar boundary loss.
    """
    probabilities = torch.softmax(logits.float(), dim=1)
    boundary_loss = torch.tensor(0.0, device=logits.device)
    n_fg = dist_maps_tensor.shape[1]

    for i in range(n_fg):
        pred_prob = probabilities[:, i + 1]  # class i+1 (skip background)
        boundary_loss = boundary_loss + (pred_prob * dist_maps_tensor[:, i]).mean()

    return boundary_loss / max(n_fg, 1)


def compute_combined_segmentation_loss(logits, target, class_weights,
                                       boundary_weight=0.0,
                                       dist_maps=None):
    """Dice + Weighted CE + optional boundary loss.

    boundary_weight: ramp-in externally (0 early in training, 0.5 later).
    dist_maps: precomputed (B, num_fg, D, H, W) tensor from dataloader.
               If None and boundary_weight > 0, boundary loss is skipped.
    """
    logits32 = logits.float()
    weights32 = class_weights.float() if class_weights is not None else None
    dice = compute_dice_loss_from_logits(logits32, target, exclude_background=True)
    ce = F.cross_entropy(logits32, target, weight=weights32)
    total = dice + ce

    bdl = torch.tensor(0.0, device=logits.device)
    if boundary_weight > 0 and dist_maps is not None:
        bdl = compute_boundary_loss_from_precomputed(logits32, dist_maps)
        total = total + boundary_weight * bdl

    return total, dice, ce, bdl
