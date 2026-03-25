## Project: BraTS 3D Glioma Segmentation Pipeline (3D U-Net + Attention, Deep Supervision, Ensemble Inference)

### One-line pitch
I built an end-to-end **3D brain tumor segmentation** pipeline (BraTS-style labels) covering **training (5-fold), inference (ensemble + sliding window), and evaluation (Dice/HD95 + HTML report)** with VRAM-safe engineering for consumer GPUs.

### What this project is
This repository trains and evaluates a **3D medical image segmentation model** to label glioma subregions in MRI volumes using the common BraTS label scheme:

- **0**: Background  
- **1**: NCR/NET (necrotic / non-enhancing tumor core)  
- **2**: Edema  
- **3**: Enhancing tumor (ET)  

The current active model line is **V2**: `UNet3DAttnV2` (V1 exists only for legacy experiments and isn’t used in current runs).

### Why I built it (motivation)
Medical segmentation is a real-world ML problem where success depends on much more than “a model”: you need **data handling**, **training stability**, **evaluation correctness**, and **inference engineering** (ensembles, sliding windows, post-processing) — especially when operating under tight VRAM constraints. This project is my “full-stack ML” portfolio piece for that end-to-end lifecycle.

---

## Key outcomes (what I achieved)

### Training system (robust + VRAM-aware)
- **AMP (bf16)** on CUDA for speed/memory; logits cast to float32 for stable loss.
- **Gradient accumulation** to simulate larger batches on limited VRAM.
- **Deep supervision** heads with weighted losses (`[1.0, 0.5, 0.25]`).
- **Early stopping** on validation mean foreground Dice.
- **Stability guards** (finite checks) + crash checkpointing.
- Optional **boundary loss** and optional DSD/grad checkpointing (kept off by default on 8GB GPUs).

### Loss design (segmentation-focused)
Combined segmentation loss includes:
- **Foreground Dice loss** (classes 1–3)
- **Weighted cross-entropy** to fight class imbalance
- Optional **boundary loss** with signed distance maps (ramped up over epochs)

### GPU-accelerated boundary loss (performance engineering)
To avoid a CPU bottleneck from distance transforms, boundary distance maps are computed **on GPU via CuPy** inside the training step when enabled. This keeps dataloader workers light and pushes heavy work to CUDA.

### Inference and evaluation (realistic deployment-style)
- **Ensemble inference** over fold checkpoints.
- **Sliding-window** full-volume prediction with Gaussian overlap stitching.
- Optional **8-flip test-time augmentation** and **uncertainty estimates** across folds.
- Evaluation metrics include **per-class Dice**, composites (**WT/TC/ET**), and **HD95**, plus an **HTML report** option.

---

## Repository map (where things live)

- **Training**
  - `training/core/model.py`: V2 architecture (`UNet3DAttnV2`) + legacy V1
  - `training/core/metrics.py`: combined loss (Dice + weighted CE + optional boundary)
  - `training/core/augment.py`: augmentation pipeline
  - `training/engine/train_engine.py`: reusable training + validation loops
  - `training/runs/v2_5fold.py`: main 5-fold training entrypoint

- **Inference / Evaluation**
  - `inference/predict.py`: fold ensemble + optional TTA/uncertainty
  - `inference/sliding_window.py`: full-volume sliding-window inference
  - `inference/evaluate.py`: evaluation runner + optional HTML report
  - `inference/metrics.py`: Dice, composites, HD95, and extended metrics
  - `inference/post_process.py`: post-processing utilities

- **Data / Scripts**
  - `scripts/preprocess_brats2020.py`: BraTS 2020 NIfTI → `96³` patch `.npz`
  - `patches/`: train/val patch datasets (`.npz`)

---

## System architecture (how it works end-to-end)

### Data format (patch-based)
Training uses `.npz` patches sized **\(96 \times 96 \times 96\)**:
- `images`: shape `(4, D, H, W)` float32 (4 MRI modalities)
- `label`: shape `(D, H, W)` int64 with values `{0,1,2,3}`

Patch-based training is a deliberate choice to fit on consumer GPUs and to increase minibatch diversity.

### Model (V2: 3D U-Net with attention + deep supervision)
`UNet3DAttnV2` is a 3D U-Net variant with:
- **Residual conv blocks**, InstanceNorm, LeakyReLU
- **Attention gates** on skip connections
- **Deep supervision** auxiliary heads (`ds4`, `ds3`, `ds2`) during training
- Evaluation mode returns only the main output logits

**Important operational detail:** model width is controlled by a `base` channel parameter. Training and inference must use the same `base` setting as the checkpoints you load.

### Training flow (5-fold style)
Training uses a deterministic patch-stem hashing approach:
- `assign_fold(stem) = md5(stem) % 5`
- For fold `k`, patches assigned to that fold are held out as extra validation, in addition to fixed validation dirs.

This approach is intentionally simple and reproducible (portfolio-appropriate), while a stricter case-level CV split is documented as a possible future improvement.

### Inference flow (ensemble + sliding window)
For patch inference, logits are averaged across fold models and converted to a final class mask via `argmax`.

For full-volume inference, the pipeline:
- splits a volume into overlapping `96³` windows,
- normalizes channels (percentile-based),
- predicts each window,
- stitches outputs using **Gaussian-weighted blending** to reduce seams.

---

## How to run it (quick-start)

### Environment
Dependencies are in `requirements.txt`. Notable ones:
- **PyTorch** (CUDA)
- **CuPy** (`cupy-cuda12x`) for GPU distance transform when boundary loss is enabled

### Training (V2, 5-fold)
Run from the `training/` directory (see `COMMANDS.md` for the exact commands used in this repo):
- Train a fold with `training/runs/v2_5fold.py`
- Checkpoints written under `models/v2/` (e.g. `fold_N_best.pt`, plus `last.pt`)
- Logs under `logs/v2/fold_N/`

### Validation & evaluation
- Validation runs each epoch inside `training/engine/train_engine.py`.
- Offline evaluation + optional HTML report are available via `inference/evaluate.py`.

### Inference
- Ensemble prediction: `inference/predict.py`
- Full-volume sliding window: `inference/sliding_window.py`

---

## What’s “portfolio-grade” here (engineering highlights)

### I treated this like a real ML product, not a notebook
- Clear **training/inference separation**
- Reusable **engine** code (`train_engine.py`)
- Checkpointing with resume support and best-model tracking
- Dedicated evaluation tooling and report generation

### Performance and resource constraints were a first-class design concern
- Patch-based training for VRAM feasibility
- AMP + accumulation to stabilize throughput
- GPU boundary loss computation to avoid CPU dataloader bottlenecks
- Validation loop optimized to avoid excessive GPU↔CPU synchronization

### Evaluation is aligned with the domain
- Foreground class Dice (classes 1–3)
- BraTS composites: **WT**, **TC**, **ET**
- Distance metric support (**HD95**)

---

## Lessons learned (the “soul”)

### Data and compute realities dominate design
In 3D medical imaging, you can’t ignore memory limits — you architect around them (patching, accumulation, mixed precision), or you don’t ship anything reliable.

### “Correct + fast” beats “correct but unusable”
I repeatedly optimized bottlenecks that appear only at scale (distance transforms, validation synchronization). The end result is a pipeline that’s not only accurate, but also **practical to iterate on**.

### Operational consistency matters
Some settings (especially model width `base`) must match across training and inference. This repo documents those constraints and highlights where configuration drift can break results.

---

## Known issues / technical debt (honest notes)
- **Hardcoded path assumptions:** several scripts assume `E:\\data` as the root.
- **Train/infer config drift risk:** `MODEL_BASE` (training) and inference `base` must match checkpoints.
- **Fold split is patch-level hash-based:** acceptable for portfolio work; case-level CV is a future improvement.
- **Val loss intentionally differs from train loss** when boundary term is enabled (keeps validation simpler).

---

## Roadmap ideas (future upgrades)
- Unify train + infer config in a single YAML/TOML to eliminate `base` drift.
- Add optional case-level k-fold split logic for stricter methodology.
- Expand evaluation export to CSV/JSON and broaden report metrics.
- Add a deployment layer (FastAPI is already listed as a future dependency direction).

---

## If you’re reviewing this on my profile
This project demonstrates:
- **Deep learning engineering** (3D architectures, loss design, augmentation, mixed precision)
- **ML systems thinking** (training/inference pipelines, checkpoints, metrics, reports)
- **Performance debugging** (identifying and eliminating GPU↔CPU sync bottlenecks)
- **Practical constraints** (VRAM-aware design, reproducibility trade-offs, honest tech debt)

