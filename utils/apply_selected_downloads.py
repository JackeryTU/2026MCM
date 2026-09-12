"""Apply the figure variants explicitly selected from Downloads.

The PNG files are copied byte-for-byte.  A same-name SVG wrapper embeds the
selected 300 dpi raster so PNG/SVG previews cannot silently diverge.
"""
from base64 import b64encode
from pathlib import Path
from shutil import copy2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\lenovo\Downloads\figures")
TARGET = ROOT / "figures"
SELECTIONS = {
    "中心预测与分位安全轨迹.png": "q2-safe",
    "八组预报信息消融.png": "q3-ablation",
    "更新时间边际收益与信息组合盈亏平衡成本.png": "q3-time-value",
}

def svg_wrapper(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    payload = b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{payload}"/>'
        '</svg>'
    )
    svg_path.write_text(svg, encoding="utf-8")

def main() -> None:
    for source_name, stem in SELECTIONS.items():
        source = SOURCE / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        png = TARGET / f"{stem}.png"
        copy2(source, png)
        svg_wrapper(png, TARGET / f"{stem}.svg")

if __name__ == "__main__":
    main()
