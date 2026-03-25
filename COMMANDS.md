# Commands

Run training from `E:\data\training\`. Run inference from `E:\data\`.

---

## Training (V2)

```powershell
cd E:\data\training

# Train each fold
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 1
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 2
# ... fold 3, 4, 5

# Resume
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 1 --resume
```

### Training with case-level classification (multi-task)

Requires a CSV (`case_id`, `class_id`). See `E:\data\labels\README.md`.

```powershell
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 1 --num-cls 2 --cls-csv E:\data\labels\my_labels.csv --lambda-cls 0.5
```

Checkpoints store `num_cls`, `model_base`, `cls_csv_path` for resume and inference.

Or run `start_v2_fold1.bat`.

Models: `models/v2/fold_N_best.pt`
Logs: `logs/v2/fold_N/`

---

## Inference

```powershell
cd E:\data

# Single patch
.venv\Scripts\python.exe inference\predict.py "patches\val\some.npz" --device cuda

# Full volume (NIfTI)
.venv\Scripts\python.exe inference\sliding_window.py volume.nii.gz --output seg.nii.gz --stride 24
```

---

## Evaluation

```powershell
.venv\Scripts\python.exe inference\evaluate.py --max 100 --report
.venv\Scripts\python.exe inference\evaluate.py --max 100 --report --extended
.venv\Scripts\python.exe inference\evaluate.py --cls-csv E:\data\labels\my_labels.csv --device cuda
```

---

## Data

```powershell
.venv\Scripts\python.exe scripts\preprocess_brats2020.py
```
