"""Build a 2x2 PNG grid from existing seg_example_*.png overlays (README asset)."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PATHS = [
    OUT_DIR / "seg_example_00006_0000.png",
    OUT_DIR / "seg_example_00006_0001.png",
    OUT_DIR / "seg_example_00018_0006.png",
    OUT_DIR / "seg_example_00019_0009.png",
]
GRID = OUT_DIR / "seg_comparison_grid.png"


def main() -> None:
    for p in PATHS:
        if not p.exists():
            raise FileNotFoundError(p)
    imgs = [Image.open(p).convert("RGB") for p in PATHS]
    w, h = imgs[0].size
    imgs = [im.resize((w, h)) for im in imgs]
    grid = Image.new("RGB", (w * 2, h * 2))
    for i, im in enumerate(imgs):
        grid.paste(im, ((i % 2) * w, (i // 2) * h))
    grid.save(GRID)
    print(f"Saved {GRID}")


if __name__ == "__main__":
    main()
