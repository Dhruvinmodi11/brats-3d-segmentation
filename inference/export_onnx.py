# export_onnx.py — Export UNet3DAttn to ONNX and benchmark inference speed
#
# Usage:
#   python inference/export_onnx.py                    # export + benchmark
#   python inference/export_onnx.py --no-benchmark    # export only

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

INFERENCE_DIR = Path(__file__).resolve().parent
ROOT_DIR = INFERENCE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(INFERENCE_DIR.parent / "training"))

from core.model import UNet3DAttn, UNet3DAttnV2
from project_config import (
    MODEL_IN_CH,
    MODEL_NUM_CLASSES,
    MODEL_V1_BASE,
    MODEL_V2_BASE,
    OUTPUTS_DIR as PROJECT_OUTPUTS_DIR,
    MODELS_DIR as PROJECT_MODELS_DIR,
    V2_MODELS_DIR as PROJECT_V2_MODELS_DIR,
)

V1_CONFIG = dict(in_ch=MODEL_IN_CH, num_classes=MODEL_NUM_CLASSES, base=MODEL_V1_BASE)
V2_CONFIG = dict(in_ch=MODEL_IN_CH, num_classes=MODEL_NUM_CLASSES, base=MODEL_V2_BASE)
OUTPUTS_DIR = PROJECT_OUTPUTS_DIR
MODELS_DIR = PROJECT_MODELS_DIR


def export_onnx(onnx_path: Path, device: str = "cpu", version: str = "v1") -> None:
    if version == "v2":
        ckpt_path = PROJECT_V2_MODELS_DIR / "fold_1_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model"]
        has_dsd = any(k.startswith("dsd_") for k in sd)
        num_cls = int(ckpt.get("num_cls", 0))
        mb = int(ckpt.get("model_base", 0) or 0)
        if mb <= 0:
            w = sd.get("enc1.conv1.weight")
            mb = int(w.shape[0]) if w is not None else V2_CONFIG["base"]
        model = UNet3DAttnV2(
            **{**V2_CONFIG, "base": mb}, use_dsd=has_dsd, num_cls=num_cls,
        )
    else:
        ckpt_path = MODELS_DIR / "run_1_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3DAttn(**V1_CONFIG)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.randn(1, 4, 96, 96, 96, device=device)
    export_model = model
    out_names = ["logits"]
    if version == "v2" and int(getattr(model, "num_cls", 0) or 0) > 0:
        class _SegAndCase(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                seg, case = self.inner(x)
                return seg, case

        export_model = _SegAndCase(model)
        out_names = ["logits", "case_logits"]
    # Opset 18 avoids "No Adapter for Resize" when exporter would downgrade from 18 to 14
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=out_names,
    )
    print(f"Exported: {onnx_path}")


def benchmark_pytorch(device: str, n_warmup: int = 3, n_run: int = 20) -> float:
    from predict import BraTSEnsemble
    ensemble = BraTSEnsemble(model_dir=MODELS_DIR, device=device)
    x = np.random.randn(4, 96, 96, 96).astype(np.float32)
    for _ in range(n_warmup):
        ensemble.predict(x)
    start = time.perf_counter()
    for _ in range(n_run):
        ensemble.predict(x)
    elapsed = time.perf_counter() - start
    return elapsed / n_run


def benchmark_onnx(onnx_path: Path, n_warmup: int = 3, n_run: int = 20) -> float:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; pip install onnxruntime")
        return -1.0
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 4, 96, 96, 96).astype(np.float32)
    for _ in range(n_warmup):
        sess.run(None, {"input": x})
    start = time.perf_counter()
    for _ in range(n_run):
        sess.run(None, {"input": x})
    elapsed = time.perf_counter() - start
    return elapsed / n_run


def main():
    import argparse
    p = argparse.ArgumentParser(description="Export to ONNX and benchmark")
    p.add_argument("--output", default=None, help="ONNX output path")
    p.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")
    args = p.parse_args()

    onnx_path = Path(args.output) if args.output else OUTPUTS_DIR / "unet3d_attn_run1.onnx"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    export_onnx(onnx_path)
    if not args.no_benchmark:
        print("\n--- Benchmark (single patch 4×96³) ---")
        t_pt = benchmark_pytorch("cpu")
        print(f"  PyTorch (3-model ensemble) CPU: {t_pt*1000:.1f} ms per patch")
        t_onnx = benchmark_onnx(onnx_path)
        if t_onnx > 0:
            print(f"  ONNX Runtime (1 model) CPU:  {t_onnx*1000:.1f} ms per patch")


if __name__ == "__main__":
    main()
