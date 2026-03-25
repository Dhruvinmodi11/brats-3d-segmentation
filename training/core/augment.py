"""
Augmentation module for BraTS 3D patches.

V1 strategy: flip + rotation + noise + intensity
V2 additions: elastic deformation, gamma augmentation, random scaling
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter, map_coordinates

NOISE_SIGMA = 0.02
INTENSITY_SHIFT = 0.05
INTENSITY_SCALE = (0.95, 1.05)


def random_flip(
    images: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Random flip on axes 1, 2, 3. Same transform for images and mask."""
    flip_img, flip_msk = [], []
    for axis_img, axis_mask in [(1, 0), (2, 1), (3, 2)]:
        if rng.random() > 0.5:
            flip_img.append(axis_img)
            flip_msk.append(axis_mask)
    if not flip_img:
        return images.copy(), mask.copy()
    return np.flip(images, axis=flip_img).copy(), np.flip(mask, axis=flip_msk).copy()


def random_rotation_90(
    images: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Random 90-degree rotation in axial plane (axes 1, 2)."""
    k = int(rng.integers(0, 4))
    if k == 0:
        return images, mask
    img = np.rot90(images, k=k, axes=(1, 2)).copy()
    msk = np.rot90(mask, k=k, axes=(0, 1)).copy()
    return img, msk


def gaussian_noise(
    images: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise to images only."""
    if sigma <= 0:
        return images.copy()
    noise = rng.normal(0, sigma, images.shape).astype(np.float32)
    return (images + noise).astype(np.float32)


def intensity_shift_scale(
    images: np.ndarray,
    shift_range: float,
    scale_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent per-channel shift and scale to images."""
    if shift_range <= 0 and scale_range[0] == 1.0 and scale_range[1] == 1.0:
        return images.copy()
    n_ch   = images.shape[0]
    shifts = rng.uniform(-shift_range, shift_range,
                         size=(n_ch, 1, 1, 1)).astype(np.float32)
    scales = rng.uniform(scale_range[0], scale_range[1],
                         size=(n_ch, 1, 1, 1)).astype(np.float32)
    return (images * scales + shifts).astype(np.float32)


# ---------------------------------------------------------------------------
# V2 augmentations
# ---------------------------------------------------------------------------

def elastic_deformation(
    images: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 7.0,
    sigma: float = 3.0,
    ctrl_grid: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random elastic deformation on a 3D patch (fast version).

    Generates a low-resolution displacement field on a ctrl_grid^3 grid,
    smooths it, then upsamples to full resolution. This is ~10-20x faster
    than computing gaussian_filter on the full 96^3 volume.

    alpha: magnitude of displacement (voxels)
    sigma: smoothness of displacement field (Gaussian sigma, on ctrl_grid)
    ctrl_grid: size of the low-resolution control grid
    """
    from scipy.ndimage import zoom as scipy_zoom

    shape = mask.shape  # (D, H, W)
    ctrl = (ctrl_grid, ctrl_grid, ctrl_grid)

    dz = gaussian_filter(rng.standard_normal(ctrl).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal(ctrl).astype(np.float32), sigma) * alpha
    dx = gaussian_filter(rng.standard_normal(ctrl).astype(np.float32), sigma) * alpha

    zoom_f = [s / c for s, c in zip(shape, ctrl)]
    dz = scipy_zoom(dz, zoom_f, order=1).astype(np.float32)
    dy = scipy_zoom(dy, zoom_f, order=1).astype(np.float32)
    dx = scipy_zoom(dx, zoom_f, order=1).astype(np.float32)

    z, y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = [z + dz, y + dy, x + dx]

    out_images = np.empty_like(images)
    for c in range(images.shape[0]):
        out_images[c] = map_coordinates(images[c], coords, order=1, mode="reflect").astype(np.float32)
    out_mask = map_coordinates(mask.astype(np.float32), coords, order=0, mode="reflect").astype(mask.dtype)

    return out_images, out_mask


def gamma_augmentation(
    images: np.ndarray,
    rng: np.random.Generator,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """Per-channel gamma (power-law) transform.

    Shifts intensities to [0,1], applies x^gamma, shifts back.
    Each channel gets its own random gamma for independent augmentation.
    """
    out = np.empty_like(images)
    for c in range(images.shape[0]):
        ch = images[c]
        cmin, cmax = ch.min(), ch.max()
        if cmax - cmin < 1e-8:
            out[c] = ch
            continue
        normalized = (ch - cmin) / (cmax - cmin)
        gamma = float(rng.uniform(gamma_range[0], gamma_range[1]))
        out[c] = (np.power(normalized, gamma) * (cmax - cmin) + cmin).astype(np.float32)
    return out


def random_scale(
    images: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float] = (0.85, 1.25),
) -> Tuple[np.ndarray, np.ndarray]:
    """Random isotropic scaling with center-crop/pad back to original size.

    Uses affine_transform instead of zoom for better performance — single
    call handles both the scaling and the center-crop/pad in one pass.
    """
    from scipy.ndimage import affine_transform

    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    if abs(scale - 1.0) < 0.02:
        return images, mask

    D, H, W = mask.shape
    inv_scale = 1.0 / scale
    center = np.array([D / 2.0, H / 2.0, W / 2.0])
    offset = center - center * inv_scale

    out_images = np.empty_like(images)
    for c in range(images.shape[0]):
        affine_transform(
            images[c], matrix=np.eye(3) * inv_scale, offset=offset,
            output=out_images[c], order=1, mode="constant", cval=0.0,
        )
    out_mask = affine_transform(
        mask.astype(np.float32), matrix=np.eye(3) * inv_scale, offset=offset,
        order=0, mode="constant", cval=0.0,
    ).astype(mask.dtype)

    return out_images, out_mask


# ---------------------------------------------------------------------------
# Apply strategy (V2)
# ---------------------------------------------------------------------------

def apply_v2_augmentation(
    images: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    noise_prob: float = 0.2,
    elastic_prob: float = 0.2,
    gamma_prob: float = 0.3,
    scale_prob: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """V2 augmentation: all V1 transforms + elastic deformation, gamma, scaling.

    Probabilities kept moderate so augmentation doesn't dominate training time.
    Elastic and scale are the most expensive (scipy-based), so their probs are lower.
    """
    img, msk = random_flip(images, mask, rng)
    img, msk = random_rotation_90(img, msk, rng)

    if rng.random() < elastic_prob:
        img, msk = elastic_deformation(img, msk, rng)

    if rng.random() < scale_prob:
        img, msk = random_scale(img, msk, rng)

    if rng.random() < noise_prob:
        sigma = float(rng.uniform(0.01, 0.1))
        img = gaussian_noise(img, sigma, rng)

    img = intensity_shift_scale(img, INTENSITY_SHIFT, INTENSITY_SCALE, rng)

    if rng.random() < gamma_prob:
        img = gamma_augmentation(img, rng)

    return img, msk
