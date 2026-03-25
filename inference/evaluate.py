# evaluate.py — Run BraTS metrics and optional per-case evaluation report
#
# Usage:
#   python inference/evaluate.py                    # full val set, print metrics
#   python inference/evaluate.py --max 100          # first 100 patches
#   python inference/evaluate.py --report          # write HTML report to outputs

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

INFERENCE_DIR = Path(__file__).resolve().parent
ROOT_DIR = INFERENCE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(INFERENCE_DIR.parent / "training"))

from metrics import compute_brats_metrics, CLASS_NAMES
from core.case_classification import case_id_from_patch_stem, load_case_class_csv
from project_config import PATCHES_DIR, OUTPUTS_DIR as PROJECT_OUTPUTS_DIR

VAL_DIR = PATCHES_DIR / "val"
OUTPUTS_DIR = PROJECT_OUTPUTS_DIR


def case_id_from_path(path: Path) -> str:
    """e.g. BraTS-GLI-00006-000_patch_0001.npz -> BraTS-GLI-00006-000"""
    return case_id_from_patch_stem(path.stem)


def generate_report(rows: list[dict], per_case: dict, output_path: Path, boxplot_b64: str | None) -> None:
    """Write HTML evaluation report with table and optional box plot."""
    patch_rows = "".join(
        f"<tr><td>{r['path']}</td><td>{r['case_id']}</td>"
        + "".join(f"<td>{r.get(k, ''):.4f}</td>" for k in ["Dice_mean_fg", "Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT"])
        + "</tr>"
        for r in rows[:500]
    )
    case_rows = "".join(
        f"<tr><td>{cid}</td><td>{len(patches)}</td>"
        + "".join(f"<td>{agg.get(k, ''):.4f}</td>" for k in ["Dice_mean_fg", "Dice_WT", "Dice_TC", "Dice_ET"])
        + "</tr>"
        for cid, (patches, agg) in sorted(per_case.items())[:200]
    )
    img_section = f'<img src="data:image/png;base64,{boxplot_b64}" alt="Box plot" style="max-width:600px"/>' if boxplot_b64 else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/><title>BraTS Evaluation Report</title>
<style>body{{font-family:sans-serif;margin:24px;background:#1a1a2e;color:#e0e0e0;}}
table{{border-collapse:collapse;margin:12px 0;}} th,td{{border:1px solid #444;padding:6px 10px;text-align:right;}} th{{background:#2a2a4a;}}
h2{{color:#7c8cf8;margin-top:24px;}}</style></head><body>
<h1>BraTS Evaluation Report</h1>
<p>Total patches: {len(rows)}. Cases: {len(per_case)}.</p>
<h2>Dice distribution (mean_fg)</h2>
{img_section}
<h2>Per-patch (first 500)</h2>
<table><tr><th>Patch</th><th>Case ID</th><th>Dice_mean_fg</th><th>Dice_WT</th><th>Dice_TC</th><th>Dice_ET</th><th>HD95_WT</th></tr>
{patch_rows}</table>
<h2>Per-case (mean over patches, first 200 cases)</h2>
<table><tr><th>Case ID</th><th>N patches</th><th>Dice_mean_fg</th><th>Dice_WT</th><th>Dice_TC</th><th>Dice_ET</th></tr>
{case_rows}</table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main():
    import argparse
    import base64
    import io
    from collections import defaultdict
    from predict import BraTSEnsemble
    from tqdm import tqdm

    p = argparse.ArgumentParser(description="Evaluate ensemble on validation set")
    p.add_argument("--max", type=int, default=None, help="Max number of patches (default: all)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--models", default=None)
    p.add_argument("--report", action="store_true", help="Write HTML report to outputs/eval_report.html")
    p.add_argument("--extended", action="store_true", help="Compute extended metrics (IoU, ASD, ASSD, NSD, VolSim, etc.)")
    p.add_argument(
        "--cls-csv", type=str, default=None,
        help="Optional CSV (case_id,class_id) to report case-level classification accuracy when model has cls head.",
    )
    args = p.parse_args()

    case_to_cls = None
    if args.cls_csv:
        case_to_cls = load_case_class_csv(Path(args.cls_csv))

    files = sorted(VAL_DIR.glob("*.npz"))
    if args.max:
        files = files[: args.max]
    print(f"Evaluating on {len(files)} patches...")
    ensemble = BraTSEnsemble(model_dir=args.models, device=args.device)

    all_metrics = []
    rows = []
    cls_correct = 0
    cls_total = 0
    for path in tqdm(files, desc="Val"):
        with np.load(path) as d:
            images = np.array(d["images"], dtype=np.float32)
            gt = np.array(d["label"])
            if gt.ndim == 4:
                gt = gt[0]
            gt = gt.astype(np.int64)
        pred = ensemble.predict(images)
        m = compute_brats_metrics(pred, gt, per_class=True, composites=True, extended=args.extended)
        all_metrics.append(m)
        case_id = case_id_from_path(path)
        row = {"path": path.name, "case_id": case_id, **m}
        if case_to_cls is not None and getattr(ensemble, "num_cls", 0) > 0:
            gt_c = int(case_to_cls.get(case_id, -1))
            probs = ensemble.predict_case_probs(images)
            if gt_c >= 0 and probs is not None:
                pr = int(np.argmax(probs))
                row["case_gt"] = gt_c
                row["case_pred"] = pr
                cls_total += 1
                if pr == gt_c:
                    cls_correct += 1
        rows.append(row)

    keys = list(all_metrics[0].keys())
    means = {}
    stds = {}
    for k in keys:
        vals = [x[k] for x in all_metrics]
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals))
    print("\n--- BraTS metrics (mean +/- std) ---")
    for k in sorted(means.keys()):
        print(f"  {k}: {means[k]:.4f} +/- {stds[k]:.4f}")
    print(f"  (n={len(all_metrics)} patches)")

    if cls_total > 0:
        print(
            f"\n--- Case classification (patches with CSV label, n={cls_total}) ---\n"
            f"  accuracy: {cls_correct / cls_total:.4f}"
        )
    elif case_to_cls is not None and getattr(ensemble, "num_cls", 0) <= 0:
        print("\n[Note] --cls-csv set but checkpoints have no classification head (num_cls=0).")

    if args.report:
        per_case = defaultdict(lambda: ([], {}))
        for r in rows:
            cid = r["case_id"]
            per_case[cid][0].append(r)
        for cid, (patches, _) in per_case.items():
            agg = {}
            for k in keys:
                agg[k] = float(np.mean([p[k] for p in patches]))
            per_case[cid] = (patches, agg)
        boxplot_b64 = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            dice_vals = [r["Dice_mean_fg"] for r in rows]
            ax.boxplot(dice_vals, labels=["Dice mean_fg"])
            ax.set_ylabel("Dice")
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, facecolor="#1a1a2e", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            boxplot_b64 = base64.b64encode(buf.read()).decode()
        except Exception:
            pass
        generate_report(rows, per_case, OUTPUTS_DIR / "eval_report.html", boxplot_b64)
        print(f"Report written: {OUTPUTS_DIR / 'eval_report.html'}")


if __name__ == "__main__":
    main()
