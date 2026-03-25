# BraTS Brain Tumor Segmentation

3D U-Net with attention gates for BraTS glioma segmentation. V2 model with 5-fold cross-validation on mixed BraTS 2020+2023.

**Project checkpoint (architecture, decisions, failures, roadmap):** see [`PROJECT_SOUL.md`](PROJECT_SOUL.md).

## Structure

```
E:\data\
├── training/           # Model, data loading, augmentation
│   ├── core/           # model.py, augment.py, metrics.py, helpers.py, parameters.py, aug_visualizer.py
│   ├── engine/          # train_engine.py
│   └── runs/            # v2_5fold.py
├── inference/           # Predict, evaluate, sliding window
│   ├── predict.py       # Ensemble inference
│   ├── sliding_window.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── visualize.py
│   ├── attention_viz.py
│   ├── export_onnx.py
│   └── post_process.py
├── scripts/             # Data preprocessing
│   └── preprocess_brats2020.py
├── patches/             # Preprocessed .npz patches (train/val)
├── labels/              # Optional case-level classification CSV (+ README)
├── models/              # Checkpoints (v2/fold_N_best.pt)
├── logs/                # Training logs
└── outputs/             # Predictions, eval reports
```

## Quick Start

```powershell
cd E:\data\training
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 1
```

See `COMMANDS.md` for full commands.
