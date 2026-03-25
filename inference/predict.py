# predict.py — Ensemble inference for BraTS brain tumor segmentation
#
# Supports both V1 (UNet3DAttn, 3 cyclic runs) and V2 (UNet3DAttnV2, 5 folds).
#
# Usage:
#   from predict import BraTSEnsemble
#   ensemble = BraTSEnsemble(model_dir="E:/data/models/v2", device="cuda")
#   mask = ensemble.predict(images)

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from core.model import UNet3DAttn, UNet3DAttnV2

V1_CONFIG = dict(in_ch=4, num_classes=4, base=24)
V2_CONFIG = dict(in_ch=4, num_classes=4, base=32)

CLASS_NAMES = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "ET"}


def _infer_v2_base_from_state_dict(sd: dict) -> int:
    """Recover width base from enc1 first conv (out channels = base)."""
    w = sd.get("enc1.conv1.weight")
    if w is None:
        return V2_CONFIG["base"]
    return int(w.shape[0])


def _forward_v2(model, tensor, device: torch.device):
    """Returns (seg_logits, case_logits_or_None)."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tensor)
    else:
        out = model(tensor)
    if isinstance(out, tuple):
        seg = out[0].float()
        case = out[1].float() if len(out) > 1 else None
        return seg, case
    return out.float(), None

V2_MODELS_DIR   = Path(os.environ.get("V2_MODELS_DIR", r"E:\data\models\v2"))
MIXED_MODELS_DIR = Path(r"E:\data\models\mixed")
OUTPUTS_DIR     = Path(os.environ.get("OUTPUTS_DIR", r"E:\data\outputs"))


def _detect_model_version(model_dir: Path) -> str:
    """Detect whether a model directory contains V1 or V2 checkpoints."""
    sample = None
    for pattern in ["fold_1_best.pt", "run_1_best.pt"]:
        p = model_dir / pattern
        if p.exists():
            sample = p
            break
    if sample is None:
        return "v1"
    ckpt = torch.load(sample, map_location="cpu", weights_only=False)
    keys = set(ckpt.get("model", {}).keys())
    if any(k.startswith("enc5.") for k in keys):
        return "v2"
    return "v1"


def _find_model_paths(model_dir: Path) -> list[Path]:
    """Find all best checkpoint files in a directory."""
    # V2 folds
    fold_paths = sorted(model_dir.glob("fold_*_best.pt"))
    if fold_paths:
        return fold_paths
    # V1 cyclic runs
    run_paths = sorted(model_dir.glob("run_*_best.pt"))
    if run_paths:
        return run_paths
    raise FileNotFoundError(f"No model checkpoints found in {model_dir}")


class BraTSEnsemble:
    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)

        if model_dir is None:
            if (V2_MODELS_DIR / "fold_1_best.pt").exists():
                model_dir = V2_MODELS_DIR
            elif (MIXED_MODELS_DIR / "run_1_best.pt").exists():
                model_dir = MIXED_MODELS_DIR
            else:
                raise FileNotFoundError(
                    f"No models found. Place checkpoints in {V2_MODELS_DIR} or {MIXED_MODELS_DIR}"
                )
        model_dir = Path(model_dir)

        self.model_paths = _find_model_paths(model_dir)
        self.version = _detect_model_version(model_dir)

        if self.version == "v2":
            ModelClass = UNet3DAttnV2
        else:
            ModelClass = UNet3DAttn

        self.models = []
        self.num_cls = 0
        self.model_base = V2_CONFIG["base"]

        for i, p in enumerate(self.model_paths):
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            sd = ckpt["model"]
            extra_kw = {}
            if self.version == "v2":
                has_dsd = any(k.startswith("dsd_") for k in sd)
                num_cls = int(ckpt.get("num_cls", 0))
                if num_cls <= 0 and any(k.startswith("cls_head.") for k in sd):
                    # Infer num classes from cls_head final linear (Sequential: pool, flatten, linear)
                    for k, v in sd.items():
                        if k.endswith("cls_head.2.weight"):
                            num_cls = int(v.shape[0])
                            break
                mb = int(ckpt.get("model_base", 0) or 0)
                if mb <= 0:
                    mb = _infer_v2_base_from_state_dict(sd)
                if i == 0:
                    self.num_cls = num_cls
                    self.model_base = mb
                extra_kw = {"use_dsd": has_dsd, "num_cls": self.num_cls}
                config = {**V2_CONFIG, "base": self.model_base}
            else:
                config = V1_CONFIG

            model = ModelClass(**config, **extra_kw)
            model.load_state_dict(sd)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict(
        self,
        images: np.ndarray,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> np.ndarray:
        if images.ndim != 4 or images.shape[0] != 4:
            raise ValueError(f"Expected shape (4, D, H, W), got {images.shape}")

        tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(self.device)
        logits_sum = None
        n = len(self.models)

        with torch.no_grad():
            for i, model in enumerate(self.models):
                if progress_callback is not None:
                    progress_callback("running_ensemble", f"Model {i + 1}/{n}")
                if self.version == "v2":
                    logits, _ = _forward_v2(model, tensor, self.device)
                else:
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(tensor).float()
                    else:
                        logits = model(tensor).float()

                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

        avg_logits = logits_sum / len(self.models)
        mask = avg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        return mask

    def predict_with_tta(
        self,
        images: np.ndarray,
        use_tta: bool = True,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> np.ndarray:
        if not use_tta:
            return self.predict(images, progress_callback=progress_callback)
        probs_sum = None
        flip_idx = 0
        for flip_d in (0, 1):
            for flip_h in (0, 1):
                for flip_w in (0, 1):
                    if progress_callback is not None:
                        progress_callback("tta", f"TTA flip {flip_idx + 1}/8")
                    flip_idx += 1
                    aug = np.copy(images)
                    if flip_d:
                        aug = np.flip(aug, axis=1).copy()
                    if flip_h:
                        aug = np.flip(aug, axis=2).copy()
                    if flip_w:
                        aug = np.flip(aug, axis=3).copy()
                    p = self.predict_soft(aug)
                    if flip_d:
                        p = np.flip(p, axis=1).copy()
                    if flip_h:
                        p = np.flip(p, axis=2).copy()
                    if flip_w:
                        p = np.flip(p, axis=3).copy()
                    if probs_sum is None:
                        probs_sum = p
                    else:
                        probs_sum += p
        probs_sum /= 8.0
        mask = probs_sum.argmax(axis=0).astype(np.int64)
        return mask

    def predict_soft(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4 or images.shape[0] != 4:
            raise ValueError(f"Expected shape (4, D, H, W), got {images.shape}")

        tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(self.device)
        logits_sum = None

        with torch.no_grad():
            for model in self.models:
                if self.version == "v2":
                    logits, _ = _forward_v2(model, tensor, self.device)
                else:
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(tensor).float()
                    else:
                        logits = model(tensor).float()

                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

        avg_logits = logits_sum / len(self.models)
        probs = torch.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return probs

    def predict_case_logits(self, images: np.ndarray) -> np.ndarray | None:
        """Mean case-classification logits (num_cls,) over ensemble. None if heads absent."""
        if self.version != "v2" or self.num_cls <= 0:
            return None
        if images.ndim != 4 or images.shape[0] != 4:
            raise ValueError(f"Expected shape (4, D, H, W), got {images.shape}")
        tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(self.device)
        acc = None
        with torch.no_grad():
            for model in self.models:
                _, case = _forward_v2(model, tensor, self.device)
                if case is None:
                    continue
                v = case.squeeze(0).float().cpu().numpy()
                acc = v if acc is None else acc + v
        if acc is None:
            return None
        return (acc / len(self.models)).astype(np.float32)

    def predict_case_probs(self, images: np.ndarray) -> np.ndarray | None:
        """Softmax of mean case logits."""
        lg = self.predict_case_logits(images)
        if lg is None:
            return None
        m = float(lg.max())
        ex = np.exp(lg - m)
        return (ex / ex.sum()).astype(np.float32)

    def predict_with_uncertainty(self, images: np.ndarray) -> dict:
        if images.ndim != 4 or images.shape[0] != 4:
            raise ValueError(f"Expected shape (4, D, H, W), got {images.shape}")

        tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(self.device)
        all_logits = []
        with torch.no_grad():
            for model in self.models:
                if self.version == "v2":
                    logits, _ = _forward_v2(model, tensor, self.device)
                else:
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(tensor).float()
                    else:
                        logits = model(tensor).float()
                all_logits.append(logits)
        stack = torch.stack(all_logits, dim=0)
        avg_logits = stack.mean(dim=0)
        p_avg = torch.softmax(avg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        p_per_model = torch.softmax(stack, dim=2).cpu().numpy().astype(np.float32)

        mask = p_avg.argmax(axis=0).astype(np.int64)

        eps = 1e-8
        entropy = -np.sum(p_avg * np.log(p_avg + eps), axis=0).astype(np.float32)
        ent_per = -np.sum(p_per_model * np.log(p_per_model + eps), axis=2).squeeze(1)
        expected_entropy = ent_per.mean(axis=0).astype(np.float32)
        mutual_info = (entropy - expected_entropy).astype(np.float32)
        max_probs = p_per_model.max(axis=2).squeeze(2)
        variance_max = np.var(max_probs, axis=0).astype(np.float32)

        return {
            "mask": mask,
            "entropy": entropy,
            "mutual_info": mutual_info,
            "variance_max": variance_max,
            "p_avg": p_avg,
        }

    def predict_from_file(self, npz_path: str | Path, with_uncertainty: bool = False) -> dict:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")

        with np.load(npz_path) as data:
            images = np.array(data["images"], dtype=np.float32)
            gt = None
            if "label" in data:
                gt = np.array(data["label"])
                if gt.ndim == 4 and gt.shape[0] == 1:
                    gt = gt[0]
                gt = gt.astype(np.int64)

        if with_uncertainty:
            out = self.predict_with_uncertainty(images)
            mask = out["mask"]
            result = {
                "mask": mask,
                "images": images,
                "ground_truth": gt,
                "entropy": out["entropy"],
                "mutual_info": out["mutual_info"],
                "variance_max": out["variance_max"],
            }
        else:
            mask = self.predict(images)
            result = {"mask": mask, "images": images, "ground_truth": gt}

        if self.num_cls > 0:
            cl = self.predict_case_logits(images)
            if cl is not None:
                result["case_logits"] = cl
                cp = self.predict_case_probs(images)
                if cp is not None:
                    result["case_probs"] = cp
                    result["case_pred"] = int(np.argmax(cp))

        class_counts = {}
        for cls_id, cls_name in CLASS_NAMES.items():
            class_counts[cls_name] = int((mask == cls_id).sum())
        result["class_counts"] = class_counts
        return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="BraTS ensemble inference")
    p.add_argument("input", help="Path to .npz patch file")
    p.add_argument("--output", default=None, help="Path to save predicted mask (.npz)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--models", default=None, help="Directory with model checkpoints")
    p.add_argument("--post-process", action="store_true")
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.models) if args.models else None
    print(f"Loading ensemble on {args.device}...")
    ensemble = BraTSEnsemble(model_dir=model_dir, device=args.device)
    print(f"Loaded {len(ensemble.models)} models (version: {ensemble.version})")

    print(f"Running inference on {args.input}...")
    if getattr(args, "tta", False):
        with np.load(args.input) as d:
            images = np.array(d["images"], dtype=np.float32)
            gt = np.array(d["label"]) if "label" in d else None
            if gt is not None and gt.ndim == 4:
                gt = gt[0]
            if gt is not None:
                gt = gt.astype(np.int64)
        mask = ensemble.predict_with_tta(images, use_tta=True)
        class_counts = {CLASS_NAMES[c]: int((mask == c).sum()) for c in (0, 1, 2, 3)}
        result = {"mask": mask, "images": images, "ground_truth": gt, "class_counts": class_counts}
    else:
        result = ensemble.predict_from_file(args.input)

    mask = result["mask"]
    print(f"Prediction shape: {mask.shape}")
    for name, count in result["class_counts"].items():
        pct = count / mask.size * 100
        print(f"  {name}: {count:,} voxels ({pct:.2f}%)")

    if result["ground_truth"] is not None:
        gt = result["ground_truth"]
        for cls_id in [1, 2, 3]:
            pred_c = (mask == cls_id)
            true_c = (gt == cls_id)
            inter = (pred_c & true_c).sum()
            union = pred_c.sum() + true_c.sum()
            dice = (2.0 * inter + 1e-6) / (union + 1e-6) if union > 0 else 1.0
            print(f"  Dice c{cls_id} ({CLASS_NAMES[cls_id]}): {dice:.4f}")

    if getattr(args, "post_process", False):
        from post_process import post_process_brats
        mask = post_process_brats(mask)
        print("Applied post-processing")

    out_path = Path(args.output) if args.output else OUTPUTS_DIR / f"{Path(args.input).stem}_mask.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"mask": mask}
    for k in ("case_logits", "case_probs", "case_pred"):
        if k in result:
            save_kw[k] = result[k]
    np.savez_compressed(out_path, **save_kw)
    print(f"Saved: {out_path}")
    if "case_pred" in result:
        print(f"  case_pred={result['case_pred']}  (num_cls={getattr(ensemble, 'num_cls', 0)})")
