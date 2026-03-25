import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from core.parameters import DEVICE, BATCH_SIZE, NUM_WORKERS


# -----------------------------
# RNG state
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state():
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    t = state["torch"]
    if isinstance(t, torch.Tensor):
        t = t.detach().to("cpu")
    torch.random.set_rng_state(t)

    if torch.cuda.is_available() and "cuda_all" in state:
        cuda_states = state["cuda_all"]
        if isinstance(cuda_states, (list, tuple)):
            fixed = []
            for s in cuda_states:
                if isinstance(s, torch.Tensor):
                    s = s.detach().to("cpu")
                fixed.append(s)
            torch.cuda.set_rng_state_all(fixed)


# -----------------------------
# DataLoaders
# -----------------------------
def make_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = BATCH_SIZE,
    val_batch_size: int | None = None,
    num_workers: int = NUM_WORKERS,
):
    if val_batch_size is None:
        val_batch_size = batch_size
    pin = DEVICE.type == "cuda"
    mp = num_workers > 0
    pf = 2 if mp else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=mp, prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=mp, prefetch_factor=pf,
    )
    return train_loader, val_loader


# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on Windows


# -----------------------------
# Logger
# -----------------------------
class SimpleLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.log_path, "a", encoding="utf-8")

    def log(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


# -----------------------------
# GPU memory
# -----------------------------
def get_gpu_memory_summary_string():
    if DEVICE.type != "cuda":
        return "cpu"
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb  = torch.cuda.memory_reserved()   / (1024**3)
    peak_gb      = torch.cuda.max_memory_allocated() / (1024**3)
    return f"mem alloc={allocated_gb:.2f}G res={reserved_gb:.2f}G peak={peak_gb:.2f}G"


# -----------------------------
# Display formatting
# -----------------------------
def _format_eta_minutes(remaining_seconds: float) -> str:
    if remaining_seconds < 0:
        remaining_seconds = 0.0
    return f"{(remaining_seconds / 60.0):.1f}m"


def _progress_bar(current: int, total: int, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    if total <= 0:
        return empty * width
    pct = current / total
    filled = int(width * pct)
    return fill * filled + empty * (width - filled)


def _epoch_summary_table(epoch, total_epochs, train_loss, train_dice,
                         val_loss, d1, d2, d3, mean_fg,
                         train_min, val_min, lr, best_fg) -> str:
    bar = _progress_bar(epoch, total_epochs, width=35)
    new_best = " ★ NEW BEST ★" if mean_fg > best_fg and best_fg >= 0 else ""
    lines = [
        "",
        "┌──────────────────────────────────────────────────────────────┐",
        f"│  EPOCH {epoch}/{total_epochs}  [{bar}]  │",
        "├──────────────────────────────────────────────────────────────┤",
        f"│  TRAIN   loss={train_loss:.4f}  dice={train_dice:.4f}  ({train_min:.1f}m)          │",
        f"│  VAL     loss={val_loss:.4f}  c1={d1:.3f} c2={d2:.3f} c3={d3:.3f}  ({val_min:.1f}m)    │",
        f"│  DICE    mean_fg={mean_fg:.4f}{new_best}  lr={lr:.2e}   │",
        "└──────────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)
