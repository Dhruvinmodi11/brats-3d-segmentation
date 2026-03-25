# PROJECT SOUL — BraTS 3D Glioma Segmentation Pipeline

**Checkpoint date:** 2026-03-12 (update this line when you change the project materially)

This document is the **single source of truth** for *why* the repo is built the way it is, *what* works, *what* was tried, *what* failed, *what* we did instead, and *what* is left. It is meant for future you (and collaborators) so the project flow stays safe when adding features (e.g. classification, new metrics, new data years).

**Maintainer context (answered 2026-03-12):** This is a **personal / portfolio project** (e.g. **resume**), not a publication or regulatory submission. **V1** and **`models/mixed`** are **no longer used** — active work is **V2 only**. **FastAPI** in `requirements.txt` is for a **future deployment** after a **single finalized model** is chosen. **`MODEL_BASE` / `base` is tuned per experiment** and may keep changing — inference must always use the **same `base` as the checkpoints** you load (see [§11](#11-known-issues--technical-debt)).

> **Honesty note:** Parts labeled *inferred* come from code + conversation context. Anything you know from memory that contradicts this file should win — **please edit this file** when you remember details we did not have in the repo.

---

## Table of contents

1. [Mission & scope](#1-mission--scope)
2. [What we achieved](#2-what-we-achieved)
3. [Repository layout](#3-repository-layout)
4. [Architecture](#4-architecture)
5. [Data: sources, format, splits](#5-data-sources-format-splits)
6. [Training flow](#6-training-flow)
7. [Inference flow](#7-inference-flow)
8. [Evaluation & reporting](#8-evaluation--reporting)
9. [Design decisions (why things are as they are)](#9-design-decisions-why-things-are-as-they-are)
10. [What we tried, what failed, mitigations](#10-what-we-tried-what-failed-mitigations)
11. [Known issues & technical debt](#11-known-issues--technical-debt)
12. [Roadmap / not implemented](#12-roadmap--not-implemented)
13. [Reproducibility & operations](#13-reproducibility--operations)
14. [Maintainer notes (answered)](#14-maintainer-notes-answered)

---

## 1. Mission & scope

**Goal:** Train and deploy a **3D brain tumor segmentation** model for **BraTS-style** glioma labels:

| Label | Meaning (BraTS) |
|------|------------------|
| 0 | Background |
| 1 | NCR/NET (necrotic / non-enhancing tumor core) |
| 2 | Peritumoral edema |
| 3 | Enhancing tumor (ET) |

**In scope today:**

- Patch-based training on **mixed BraTS 2023 + BraTS 2020** preprocessed to `96³` `.npz` files.
- **V2** model: `UNet3DAttnV2` with deep supervision, optional boundary loss, optional DSD / grad checkpointing (usually off on 8 GB GPUs).
- **5-fold** cross-validation style split: hash patch stem → fold; one fold held from train pool; fixed val dirs always in validation.
- Inference: **ensemble** over fold checkpoints; single-patch and **sliding-window** full-volume with Gaussian stitching.
- Evaluation: BraTS-style **Dice**, composites (**WT, TC, ET**), **HD95** (and more with `--extended`).

**Explicitly out of scope (today):**

- End-to-end **classification** (e.g. grade, MGMT, survival) — *planned*, not implemented.
- BraTS 2021 or other years as first-class citizens in preprocessing (only 2020 script in-repo; 2023 patches assumed pre-existing under `patches/`).
- A FastAPI service in this tree **today** — planned **after** one production model is frozen (deps already listed for that future step).

---

## 2. What we achieved

- **V1 + V2 models** in `training/core/model.py`: **V2 is the only active line**; V1 remains in code for historical checkpoints, but **V1 and `mixed` model dirs are not used anymore** in day-to-day work.
- **Robust training loop** (`training/engine/train_engine.py`): AMP (bfloat16 on CUDA), gradient accumulation, deep supervision loss, optional DSD loss, early stopping on **val mean foreground Dice**, NaN guard + crash checkpoint.
- **Combined segmentation loss** (`training/core/metrics.py`): Dice (foreground) + **weighted CE** + optional **boundary loss** using precomputed signed distance maps.
- **Boundary loss without dataloader CPU bottleneck:** when `USE_BOUNDARY=True`, distance maps are computed **on GPU via CuPy** inside the training step (`_compute_dist_maps_gpu`), so workers are not stuck on SciPy EDT per sample.
- **V2 augmentation** (`training/core/augment.py`): flips, 90° rot, noise, intensity, elastic, gamma, scaling — documented in code.
- **Augmentation visualization hook** (`training/core/aug_visualizer.py`): periodic PNG + JSON logs under `logs/v2/aug/`.
- **5-fold script** (`training/runs/v2_5fold.py`): single entrypoint per fold, resume, cosine LR, configurable VRAM-safe batch/accum.
- **Inference ensemble** (`inference/predict.py`): auto-detect V1 vs V2 from checkpoint keys (V1 path legacy); load all `fold_*_best.pt` or `run_*_best.pt`; optional **8-flip TTA** and **uncertainty** (entropy / mutual information across folds).
- **Sliding window** (`inference/sliding_window.py`): NIfTI load, percentile normalization aligned with training philosophy, Gaussian-weighted overlap stitching.
- **Rich eval metrics** (`inference/metrics.py`): per-class + composites; extended set behind `--extended` in `evaluate.py`.
- **HTML report** (`inference/evaluate.py --report`): tables + optional Dice box plot.
- **Dependencies** documented in `requirements.txt`, including **`cupy-cuda12x`** for GPU boundary loss.

---

## 3. Repository layout

```
E:\data\
├── PROJECT_SOUL.md          ← this file (project checkpoint)
├── README.md
├── COMMANDS.md
├── requirements.txt
├── training/
│   ├── core/                # model, augment, metrics, parameters, helpers, aug_visualizer
│   ├── engine/              # train_engine.py
│   └── runs/                # v2_5fold.py
├── inference/               # predict, sliding_window, evaluate, metrics, visualize, post_process, export_onnx, attention_viz
├── scripts/                 # preprocess_brats2020.py
├── patches/                 # train/val .npz (+ brats2020/train|val)
├── models/                  # v2/fold_N_best.pt, last.pt (mixed/ legacy — unused)
├── logs/                    # v2/fold_N/*.log, aug/
└── outputs/                 # predictions, eval_report.html
```

**Path assumption:** Many scripts hardcode `E:\data`. Moving the project requires search-replace or env-based paths.

---

## 4. Architecture

### 4.1 V1 — `UNet3DAttn` (legacy, unused in current workflow)

- Classic 3D U-Net + **attention gates** on skip connections.
- 4 encoder levels, single full-res output head.
- **Status:** **Not used anymore** (including `models/mixed`). Kept in `model.py` only if old artifacts need to be opened; all new training is V2.

### 4.2 V2 — `UNet3DAttnV2` (primary)

- **Residual** blocks (`ResConv3dBlock`), **LeakyReLU**, **InstanceNorm**.
- **5 encoder levels** (`enc1` … `enc5`) + bottleneck.
- **Decoder** with attention gates at 4 levels; deep supervision heads **`ds4`, `ds3`, `ds2`** on intermediate decoder feature maps.
- **Output:** `main_out` shape `(B, 4, D, H, W)` logits.
- **Training-only return:** `(main_out, [ds4, ds3, ds2])` or with **DSD** a third dict of encoder/decoder softmax “teachers/students”.
- **Eval / `torch.no_grad()` inference:** returns **only** `main_out` (single tensor).

**Channel width:** controlled by `base` (class default 32 in docstring; **`v2_5fold.py` sets `MODEL_BASE` per run** — often lowered for VRAM, **changed across experiments**). Whatever you train with **must** match inference (`predict.py` `V2_CONFIG["base"]` or future shared config).

### 4.3 Loss stack (training)

1. **Dice loss** (mean over foreground classes 1–3).
2. **Cross-entropy** with fixed **class weights** (`CROSS_ENTROPY_CLASS_WEIGHTS` in `parameters.py`) to counter imbalance.
3. **Deep supervision:** same combined loss (without boundary in DS path — see `compute_combined_segmentation_loss` usage in `_compute_ds_loss`) weighted by `[1.0, 0.5, 0.25]`.
4. **Boundary loss (optional):** `boundary_weight * mean_f(pred_prob * signed_dist_map)` per foreground class; maps either from GPU (CuPy) or precomputed in dataset (unused path when `compute_dist=False`).
5. **DSD (optional):** KL between softmax pairs at different depths; **off by default** (VRAM).

---

## 5. Data: sources, format, splits

### 5.1 Patch file format (`.npz`)

- **`images`:** `(4, D, H, W)` float32 — modalities aligned with BraTS (order must match training preprocessing).
- **`label`:** `(D, H, W)` or `(1, D, H, W)` int64 — values in `{0,1,2,3}`.

**No** case-level classification label in the file today.

### 5.2 Directories (`v2_5fold.py`)

- Train: `patches/train`, `patches/brats2020/train`
- Val: `patches/val`, `patches/brats2020/val`

### 5.3 5-fold assignment

- `assign_fold(stem) = md5(stem) % 5`
- For fold `k`, patches with `assign_fold == k-1` move from train pool to **extra val**; original val always included.
- **Property:** deterministic given stem; **not** guaranteed balanced by case (patch-level hashing).

### 5.4 BraTS 2020 preprocessing script

- `scripts/preprocess_brats2020.py`: NIfTI → `96³` patches, stride/overlap configurable, train/val split at **case** level with seed.
- Output naming: `{CaseName}_patch_{i:04d}.npz` with underscores normalized.

---

## 6. Training flow

1. **Entry:** `python runs/v2_5fold.py --fold N` from `training/` (see `COMMANDS.md`).
2. **Dataset** `_V2Dataset`: load `.npz`, optional V2 augment (train only), optional `precompute_distance_maps` if `compute_dist=True` (currently **False** when using GPU EDT).
3. **Loader:** `make_dataloaders` in `helpers.py` — batch size, workers from `v2_5fold.py` constants.
4. **Epoch loop** (`run_training_loop`):
   - Boundary weight **linear ramp** 0 → `0.5` over **20** epochs (`BOUNDARY_RAMP_EPOCHS`, `BOUNDARY_MAX_WEIGHT`).
   - Training step: autocast → forward → seg (+ DS + DSD) loss → backward / accum / clip / step.
   - Validation: no deep supervision in loss path; running mean **Dice per class 1,2,3** and **mean_fg**.
   - **Checkpoint:** every epoch `last.pt`; **best** by `mean_fg` → `fold_N_best.pt` at `models/v2/`.
   - **Early stop:** no improvement in `mean_fg` for `EARLY_STOP_PAT` epochs (25 in `v2_5fold.py`).
5. **Resume:** `--resume` loads `last.pt` (model, optimizer, scheduler, epoch, global_step, best_mean_fg).

### 6.1 Key hyperparameters (as in code — adjust per run)

| Constant | Typical role |
|----------|----------------|
| `EPOCHS_PER_FOLD` | 200 |
| `BASE_LR` / `MIN_LR` | 5e-4 → cosine to 1e-6 |
| `WEIGHT_DECAY` | 1e-4 |
| `V2_BATCH_SIZE` / `V2_ACCUM_STEPS` | effective batch = product |
| `MODEL_BASE` | **Tuned per experiment** (VRAM vs capacity); keep in sync with inference |
| `USE_BOUNDARY` | True → needs CuPy on CUDA |
| `USE_DSD` / `USE_GRAD_CKPT` | False on 8 GB class GPUs |

---

## 7. Inference flow

### 7.1 `BraTSEnsemble` (`inference/predict.py`)

- Discovers checkpoints; detects V2 via keys like `enc5.*`.
- **V2 `base`:** set in `V2_CONFIG` in `predict.py`. **`base` is not fixed forever** — it changes when you change `MODEL_BASE` in training. **Rule:** for every checkpoint set, **train and infer must use the same `base`** or state_dict load/silent mismatch will be wrong (see [§11](#11-known-issues--technical-debt)).
- For each model: forward → sum logits → **average** → `argmax` → `(D,H,W)` int64 mask.
- **TTA:** 8 corner flips on probabilities via `predict_soft`.
- **Uncertainty:** disagreement across folds (entropy, mutual information, etc.).

### 7.2 Sliding window (`inference/sliding_window.py`)

- Patch size **96**, default stride **32** (67% overlap); `COMMANDS.md` suggests `--stride 24` for heavier overlap.
- Normalize volume per-channel with percentiles (match training intent).
- Gaussian weighting for stitch.

### 7.3 Post-processing

- `inference/post_process.py` exists for refinement (e.g. morphological / connected components — consult file for exact behavior when using it).

---

## 8. Evaluation & reporting

- **`evaluate.py`:** loads val patches, runs ensemble, `compute_brats_metrics`.
- **Default report columns:** Dice mean fg, WT, TC, ET, **HD95_WT** (not all HD95 variants in HTML table).
- **`--extended`:** more metrics (IoU, ASD, etc. — see `inference/metrics.py`).
- **Per-case aggregation:** mean metrics over patches sharing `case_id` from filename regex.

**Gaps (by design / backlog):** no TTA in default eval path; no automatic metric export to CSV/JSON; report does not show every metric the library can compute.

---

## 9. Design decisions (why things are as they are)

| Decision | Rationale |
|----------|-----------|
| **Patches, not full volumes** | Fits consumer GPU VRAM; more diverse minibatches; standard BraTS practice. |
| **Mixed 2020 + 2023** | More data → better generalization; single model handles both naming schemes in the pool. |
| **Hash-based 5-fold on patch stem** | Simple, reproducible without external split files; patch-level hashing is **fine for a personal / resume project**; not optimized for publication-grade case-level CV. |
| **Dice + weighted CE** | Standard strong baseline; weights address class imbalance. |
| **Deep supervision** | Helps gradients in deeper network; low-res heads regularize. |
| **Boundary loss** | Sharpens boundaries; signed distance formulation ties probability mass to GT contour. |
| **GPU EDT (CuPy) vs CPU in dataloader** | CPU EDT per batch was a **throughput killer**; GPU path keeps workers light and moves heavy work to CUDA. |
| **bf16 autocast on CUDA** | Speed + memory; logits cast to float32 for stable loss. |
| **Gradient accumulation** | Simulates larger batch on limited VRAM. |
| **DSD / grad checkpointing off** | **VRAM**; enable when hardware allows. |
| **Keep V1 in same `model.py`** | One import path; legacy code path if ever needed; **not part of current workflow**. |
| **Ensemble over folds** | Cheap test-time ensemble; improves robustness vs single fold. |

---

## 10. What we tried, what failed, mitigations

This section mixes **confirmed code facts** with **narrative from project discussions**. Edit if your real history differs.

| Topic | Issue | Mitigation / outcome |
|-------|--------|----------------------|
| **Boundary loss** | Precomputing EDT in the **dataloader** (CPU, SciPy) slows training badly with many workers. | Compute **signed distance maps on GPU** in `train_engine` with **CuPy** when `boundary_weight > 0` and maps not provided; dataset keeps `compute_dist=False`. |
| **CuPy install** | Must match CUDA (e.g. RTX 4060 / CUDA 12 → `cupy-cuda12x`). | Documented in `requirements.txt` and error message in `_compute_dist_maps_gpu`. |
| **8 GB VRAM** | `base=32` + DSD + large batch → OOM or thrashing. | Reduce `MODEL_BASE` (e.g. 24), smaller batch, accumulation, turn off DSD and grad checkpointing by default. |
| **Training instability** | Rare NaN / Inf. | Checks on logits/loss; **crash checkpoint** `nan_crash.pt` + logging GPU memory. |
| **Scheduler API** | `CosineAnnealingLR` uses `.step()` without metric; engine calls `.step(mean_fg)`. | **`CosineSchedulerWrapper`** ignores metric and steps every epoch — **early stopping uses val Dice, not ReduceLROnPlateau**. |

---

## 11. Known issues & technical debt

1. **`base` must match train ↔ infer (operational rule):** `MODEL_BASE` in `v2_5fold.py` and `base` in `predict.py` `V2_CONFIG` are **maintained separately** today. **`base` changes across experiments** — whenever you change training width, **update inference (or save `base` inside checkpoints and read it at load)**. Mismatched shapes = wrong or partially loaded weights.
2. **Hardcoded `E:\data`:** portability hazard.
3. **Fold split is patch-hash, not case-hash:** OK for portfolio use; would need redesign for strict case-level k-fold if requirements ever change.
4. **`parameters.py` `BATCH_SIZE` / `ACCUM_STEPS`:** V2 run overrides with `V2_*` constants — easy to confuse which applies.
5. **Validation loss** uses `compute_combined_segmentation_loss` **without** boundary term (dist_maps None) — intentional simplicity; train/val loss not directly comparable.

---

## 12. Roadmap / not implemented

High-value backlog (from planning sessions; **not** coded unless you add it):

- **Case-level classification** head + multi-task loss + CSV labels + case-aggregated inference. (Full design previously documented in chat; key files: `model.py`, `v2_5fold.py`, `train_engine.py`, `predict.py`, `sliding_window.py`, `evaluate.py`.)
- **Eval:** optional TTA; export metrics to CSV/JSON; HD95 for TC/ET in report; sensitivity/specificity in default report row.
- **Sliding window:** optional TTA; default stride vs doc (`COMMANDS.md` says 24, code default 32).
- **BraTS 2021** (or other) preprocessing parity with 2020 script.
- **Strict case-level k-fold** — only if you later need stricter methodology (not required for resume-style portfolio).
- **Single config file** (YAML/TOML) for train + infer + paths — would also **pin `base` once** and avoid train/infer drift.
- **Deployment:** FastAPI (or similar) **after** one model is finalized; wire `BraTSEnsemble` behind an API.

---

## 13. Reproducibility & operations

- **Seed:** `SEED = 1337` in `parameters.py`; `seed_everything` in run script.
- **CUDA:** `cudnn.benchmark = True`, TF32 enabled in `v2_5fold.py` — small nondeterminism on GPU is possible.
- **Checkpoints:** contain `model`, `optimizer`, `scheduler`, `rng`, `epoch`, `global_step`, `best_mean_fg`, `val_mean_fg_dice`, `run_id`.
- **Logs:** timestamped under `logs/v2/fold_N/`.
- **Aug logs:** `logs/v2/aug/` (PNG + JSON).

---

## 14. Maintainer notes (answered)

| Topic | Answer |
|-------|--------|
| **V1 / `mixed`** | **No longer used.** V2 only for active work. |
| **Deployment / FastAPI** | **Future** — after **one model** is finalized; not built in-repo yet. |
| **`MODEL_BASE` / `base`** | **Changes per experiment**; will continue to tune. Always align infer config with the checkpoints you load (or persist `base` in ckpt). |
| **Publication / regulatory** | **No** — personal project, **resume / portfolio**, not a paper submission. |
| **CV strictness** | Patch-hash 5-fold is **acceptable** for this goal; stricter case-level splits are optional backlog only. |

**Still optional to document (if you want the soul file 100% complete):** where BraTS 2023 patches were produced (script vs download) and modality order — add under [§5](#5-data-sources-format-splits) when convenient.

---

## Document maintenance

- After any **architecture or path** change: update **§3–7** and **§11**.
- After any **failed experiment**: add a row to **§10**.
- After any **major milestone** (new data year, classification shipped): bump **checkpoint date** at top and add a line under **§2**.

---

*End of PROJECT SOUL.*
