# runs/v2_5fold.py
#
# V2 model training: UNet3DAttnV2 with 5-fold cross-validation.
# Mixed BraTS 2020+2023, V2 augmentation, deep supervision, cosine annealing.
#
# Run from e:\data\training\:
#   python runs/v2_5fold.py --fold 1
#   python runs/v2_5fold.py --fold 1 --resume

import argparse
import hashlib
import random
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset

from core.model import UNet3DAttnV2
from core.augment import apply_v2_augmentation
from core.aug_visualizer import log_augmentation_viz
from core.parameters import DEVICE, CROSS_ENTROPY_CLASS_WEIGHTS, SEED
from core.helpers import SimpleLogger, seed_everything, make_dataloaders, _set_rng_state
from core.metrics import precompute_distance_maps
from core.case_classification import case_id_from_patch_stem, load_case_class_csv
from engine.train_engine import run_training_loop

DATA_ROOT = Path(r"E:\data")

TRAIN_DIRS = [
    DATA_ROOT / "patches" / "train",
    DATA_ROOT / "patches" / "brats2020" / "train",
]
VAL_DIRS = [
    DATA_ROOT / "patches" / "val",
    DATA_ROOT / "patches" / "brats2020" / "val",
]

NUM_FOLDS       = 5
EPOCHS_PER_FOLD = 200
EARLY_STOP_PAT  = 25
BASE_LR         = 5e-4
MIN_LR          = 1e-6
WEIGHT_DECAY    = 1e-4
MODEL_BASE      = 24   # smaller = faster; 32 was ~22.5M params, 24 ~10–12M
V2_BATCH_SIZE   = 3   # ~5.5 GB peak, no thrashing on 8 GB GPU
V2_ACCUM_STEPS  = 3   # effective batch = 3 * 3 = 9
V2_VAL_BATCH    = 4   # safe val VRAM on 8 GB GPU
V2_NUM_WORKERS  = 6
USE_GRAD_CKPT   = False
USE_DSD         = False  # too much VRAM for 8 GB; enable on >=16 GB GPU
USE_BOUNDARY    = True   # GPU EDT in training loop (CuPy); no dataloader slowdown

MODELS_DIR   = DATA_ROOT / "models" / "v2"
LOGS_DIR     = DATA_ROOT / "logs" / "v2"
AUG_LOGS_DIR = DATA_ROOT / "logs" / "v2" / "aug"

DATASET_TAG = "V2 Mixed BraTS 2020+2023"


class CosineSchedulerWrapper:
    """Wraps CosineAnnealingLR so .step(metric) works in train_engine."""

    def __init__(self, scheduler):
        self._sched = scheduler

    def step(self, *args, **kwargs):
        self._sched.step()

    def state_dict(self):
        return self._sched.state_dict()

    def load_state_dict(self, sd):
        self._sched.load_state_dict(sd)


def parse_args():
    p = argparse.ArgumentParser(description="V2 5-fold training")
    p.add_argument("--fold", type=int, choices=list(range(1, NUM_FOLDS + 1)), required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--num-cls", type=int, default=0,
        help="Number of case-level classification classes (e.g. 2 for binary). 0 = segmentation only.",
    )
    p.add_argument(
        "--cls-csv", type=str, default=None,
        help="CSV with case_id,class_id columns (required if --num-cls > 0).",
    )
    p.add_argument("--lambda-cls", type=float, default=0.5, help="Weight for case classification loss.")
    p.add_argument("--cls-id-col", type=str, default="case_id", help="CSV column for case ID.")
    p.add_argument("--cls-label-col", type=str, default="class_id", help="CSV column for class index.")
    return p.parse_args()


def collect_all_patches(*dirs):
    """Gather all .npz file paths from multiple directories."""
    files = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")
        files.extend(sorted(d.glob("*.npz")))
    return files


def assign_fold(stem: str, num_folds: int = NUM_FOLDS) -> int:
    """Deterministically assign a patch to a fold using hash."""
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % num_folds


class _V2Dataset(Dataset):
    """Loads .npz patches with V2 augmentation and precomputed boundary distance maps."""

    def __init__(self, files: list[Path], augment: bool = False,
                 compute_dist: bool = False,
                 case_to_cls: dict[str, int] | None = None):
        self.files = list(files)
        self.augment = augment
        self.compute_dist = compute_dist
        self.case_to_cls = case_to_cls

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], mmap_mode="r") as data:
            images_np = np.array(data["images"], dtype=np.float32)
            label_np = data["label"]
            if label_np.ndim == 4 and label_np.shape[0] == 1:
                label_np = label_np[0]
            label_np = np.array(label_np)

        if self.augment:
            rng = np.random.default_rng()
            images_np, label_np = apply_v2_augmentation(images_np, label_np, rng)

        if label_np.dtype != np.int64:
            label_np = label_np.astype(np.int64)

        cls_t = None
        if self.case_to_cls is not None:
            cid = case_id_from_patch_stem(self.files[idx].stem)
            lab = int(self.case_to_cls.get(cid, -1))
            cls_t = torch.tensor(lab, dtype=torch.long)

        if self.compute_dist:
            dist_np = precompute_distance_maps(label_np)
            out = (torch.from_numpy(images_np),
                   torch.from_numpy(label_np),
                   torch.from_numpy(dist_np))
            if cls_t is not None:
                return out + (cls_t,)
            return out

        if cls_t is not None:
            return torch.from_numpy(images_np), torch.from_numpy(label_np), cls_t
        return torch.from_numpy(images_np), torch.from_numpy(label_np)


def main():
    args = parse_args()
    fold_id = args.fold

    num_cls = int(args.num_cls)
    lambda_cls = float(args.lambda_cls)
    case_to_cls = None
    cls_csv_path_str = None

    if num_cls > 0:
        if not args.cls_csv:
            raise SystemExit("When --num-cls > 0, provide --cls-csv with case_id,class_id.")
        cls_path = Path(args.cls_csv)
        case_to_cls = load_case_class_csv(
            cls_path,
            id_column=args.cls_id_col,
            label_column=args.cls_label_col,
        )
        cls_csv_path_str = str(cls_path.resolve())
        for cid, v in case_to_cls.items():
            if v < -1 or v >= num_cls:
                raise SystemExit(
                    f"Invalid class_id {v} for case {cid!r}: must be in [-1, {num_cls - 1}] "
                    f"(-1 = ignore in loss)."
                )
        if case_to_cls and all(int(v) < 0 for v in case_to_cls.values()):
            print(
                "WARNING: Every class_id in CSV is < 0; classification loss will always be zero. "
                "Add rows with class_id in [0, num_cls-1].",
                flush=True,
            )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(SEED)

    # ── Collect all patches ────────────────────────────────────────────────
    all_train = collect_all_patches(*TRAIN_DIRS)
    all_val   = collect_all_patches(*VAL_DIRS)

    # 5-fold split: fold N is held out for validation from the train pool,
    # remaining 4/5 are used for training. The original val set is always
    # included in validation for consistent comparison across folds.
    fold_train_files = []
    fold_extra_val   = []
    for f in all_train:
        if assign_fold(f.stem) == (fold_id - 1):
            fold_extra_val.append(f)
        else:
            fold_train_files.append(f)

    fold_val_files = all_val + fold_extra_val

    brats2023_train = sum(1 for f in fold_train_files if "BraTS-GLI" in f.stem)
    brats2020_train = sum(1 for f in fold_train_files if "BraTS20" in f.stem)

    # ── Output paths ───────────────────────────────────────────────────────
    ckpt_dir  = MODELS_DIR / f"fold_{fold_id}"
    log_dir   = LOGS_DIR   / f"fold_{fold_id}"
    best_ckpt = MODELS_DIR / f"fold_{fold_id}_best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    for d in (ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    AUG_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Logger ─────────────────────────────────────────────────────────────
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimpleLogger(log_dir / f"v2_fold_{fold_id}_{EPOCHS_PER_FOLD}ep_{ts}.log")

    model_base_eff = MODEL_BASE
    # ── Resume: align num_cls / base / CSV with checkpoint ─────────────────
    if args.resume and last_ckpt.exists():
        pre = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        num_cls = int(pre.get("num_cls", num_cls))
        lambda_cls = float(pre.get("lambda_cls", lambda_cls))
        model_base_eff = int(pre.get("model_base", model_base_eff))
        if num_cls > 0 and pre.get("cls_csv_path"):
            cls_csv_path_str = pre["cls_csv_path"]
            case_to_cls = load_case_class_csv(
                Path(cls_csv_path_str),
                id_column=args.cls_id_col,
                label_column=args.cls_label_col,
            )

    # ── Dataset ────────────────────────────────────────────────────────────
    train_ds = _V2Dataset(
        fold_train_files, augment=True, compute_dist=False,
        case_to_cls=case_to_cls,
    )
    val_ds = _V2Dataset(
        fold_val_files, augment=False, compute_dist=False,
        case_to_cls=case_to_cls,
    )
    train_loader, val_loader = make_dataloaders(
        train_ds, val_ds,
        batch_size=V2_BATCH_SIZE, val_batch_size=V2_VAL_BATCH,
        num_workers=V2_NUM_WORKERS,
    )

    # ── Banner ─────────────────────────────────────────────────────────────
    dev = f"{DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "")
    param_count = sum(
        p.numel() for p in UNet3DAttnV2(
            base=model_base_eff, use_dsd=USE_DSD, num_cls=num_cls,
        ).parameters()
    ) / 1e6
    cls_banner = (
        f"  Case classification: {num_cls} classes, λ_cls={lambda_cls}  CSV={cls_csv_path_str or '—'}"
        if num_cls > 0
        else "  Case classification: OFF"
    )
    banner = [
        "",
        "=" * 72,
        f"  {DATASET_TAG} — V2 Fold {fold_id}/{NUM_FOLDS}",
        "=" * 72,
        f"  Model: UNet3DAttnV2 (base={model_base_eff}, ~{param_count:.1f}M params)",
        cls_banner,
        f"  Epochs: {EPOCHS_PER_FOLD}  Batch: {V2_BATCH_SIZE} (accum {V2_ACCUM_STEPS}, eff={V2_BATCH_SIZE*V2_ACCUM_STEPS})  ValBatch: {V2_VAL_BATCH}  Device: {dev}",
        f"  LR: Cosine Annealing (T_max={EPOCHS_PER_FOLD}, eta_min={MIN_LR})",
        f"  Deep supervision: ON (weights 1.0, 0.5, 0.25)",
        f"  Gradient checkpointing: {'ON' if USE_GRAD_CKPT else 'OFF'}",
        f"  Boundary loss: {'ON (ramp-in 20 epochs → 0.5)' if USE_BOUNDARY else 'OFF'}",
        f"  Dual self-distillation: {'ON' if USE_DSD else 'OFF'}",
        f"  Augmentation: V2 (flip, rot, elastic, scale, noise, intensity, gamma)",
        f"  Early stop patience: {EARLY_STOP_PAT}",
        f"  Train: {len(train_ds)} patches  (BraTS2023: {brats2023_train}  BraTS2020: {brats2020_train})",
        f"  Val: {len(val_ds)} patches  (original val + held-out fold)",
        f"  Best model: {best_ckpt}",
        "=" * 72,
    ]
    for line in banner:
        print(line, flush=True); logger._fp.write(line + "\n")
    logger._fp.flush()
    logger.log(f"Dataset: {DATASET_TAG}")
    for td in TRAIN_DIRS:
        logger.log(f"  Train src: {td}")
    for vd in VAL_DIRS:
        logger.log(f"  Val src:   {vd}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = UNet3DAttnV2(
        in_ch=4, num_classes=4, base=model_base_eff,
        use_grad_ckpt=USE_GRAD_CKPT, use_dsd=USE_DSD, num_cls=num_cls,
    ).to(DEVICE)
    class_weights = CROSS_ENTROPY_CLASS_WEIGHTS.to(DEVICE)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PER_FOLD, eta_min=MIN_LR,
    )
    lr_scheduler = CosineSchedulerWrapper(cosine_sched)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logger.log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume ─────────────────────────────────────────────────────────────
    best_mean_fg = -1.0
    global_step  = 0
    start_epoch  = 1
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            lr_scheduler.load_state_dict(ckpt["scheduler"])
        best_mean_fg = float(ckpt.get("best_mean_fg", -1.0))
        global_step  = int(ckpt.get("global_step", 0))
        start_epoch  = int(ckpt.get("epoch", 0)) + 1
        logger.log(f"Resumed from {last_ckpt} | epoch={start_epoch} best_mean_fg={best_mean_fg:.4f}")

    logger.log("TF32 on, cuDNN benchmark on")

    # ── Aug viz hook ───────────────────────────────────────────────────────
    def aug_viz(epoch):
        if epoch % 10 == 0 and train_ds.files:
            try:
                png, jsn = log_augmentation_viz(
                    patch_path=random.choice(train_ds.files),
                    run_id=fold_id, epoch=epoch,
                    aug_logs_dir=AUG_LOGS_DIR,
                )
                logger.log(f"Aug viz: {png}")
            except Exception as exc:
                logger.log(f"[WARNING] Aug viz failed: {exc}")

    # ── Train ──────────────────────────────────────────────────────────────
    wall_s, best_mean_fg = run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        class_weights=class_weights, logger=logger,
        ckpt_dir=ckpt_dir, last_ckpt=last_ckpt, best_ckpt=best_ckpt,
        epochs=EPOCHS_PER_FOLD, start_epoch=start_epoch,
        best_mean_fg=best_mean_fg, global_step=global_step,
        run_id=fold_id, on_epoch_end=aug_viz,
        early_stop_patience=EARLY_STOP_PAT,
        accum_steps=V2_ACCUM_STEPS,
        use_boundary_loss=USE_BOUNDARY,
        use_dsd=USE_DSD,
        lambda_cls=lambda_cls,
        num_cls=num_cls,
        cls_csv_path=cls_csv_path_str,
    )

    # ── Final banner ───────────────────────────────────────────────────────
    final = [
        "",
        "=" * 72,
        f"  {DATASET_TAG} — V2 FOLD {fold_id}/{NUM_FOLDS} COMPLETE",
        "=" * 72,
        f"  Epochs: {EPOCHS_PER_FOLD}  Wall time: {wall_s/60:.1f} min ({wall_s/3600:.1f} hr)",
        f"  Best mean_fg Dice: {best_mean_fg:.4f}",
        f"  Best model: {best_ckpt}",
        "=" * 72,
        "",
    ]
    for line in final:
        print(line, flush=True); logger._fp.write(line + "\n")
    logger.log(f"Log saved to: {logger.log_path}")
    logger.close()


if __name__ == "__main__":
    main()
