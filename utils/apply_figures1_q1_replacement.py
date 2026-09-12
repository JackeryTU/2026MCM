"""Build the Q1 operation figure from the selected figures_1 panels.

The downloaded schedule figure contains power traces plus a price panel,
whereas the paper's q1-operation figure must retain the SOC evidence chain.
This script therefore combines the power panel from the schedule figure with
the standalone SOC figure and keeps the result as the authoritative raster
used by the LaTeX source.
"""
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\lenovo\Downloads\figures_1")
TARGET = ROOT / "figures"
SCHEDULE = SOURCE / "result_q1_全天计划调度.png"
SOC = SOURCE / "result_q1_SOC轨迹.png"
OUTPUT = TARGET / "q1-operation.png"
SVG_OUTPUT = TARGET / "q1-operation.svg"


def svg_wrapper(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    payload = b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<image width="{width}" height="{height}" '
        f'xlink:href="data:image/png;base64,{payload}"/>'
        "</svg>",
        encoding="utf-8",
    )


def main() -> None:
    if not SCHEDULE.exists():
        raise FileNotFoundError(SCHEDULE)
    if not SOC.exists():
        raise FileNotFoundError(SOC)

    with Image.open(SCHEDULE) as schedule_image, Image.open(SOC) as soc_image:
        schedule = schedule_image.convert("RGB")
        soc = soc_image.convert("RGB")

    # The first panel ends immediately above the price-panel title/axis.
    # Keep its complete title, legend, traces, and x-axis boundary, but omit
    # the downloaded price panel because the paper already presents price in
    # data-overview and q1-operation's lower panel is reserved for SOC.
    power_bottom = 735
    power = schedule.crop((0, 0, schedule.width, power_bottom))

    # Remove only the unused bottom margin below the SOC x-axis label while
    # retaining both endpoint annotations and the boundary labels.
    soc_bottom = min(965, soc.height)
    soc_panel = soc.crop((0, 0, soc.width, soc_bottom))

    canvas_width = max(power.width, soc_panel.width)
    canvas_height = power.height + soc_panel.height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(power, ((canvas_width - power.width) // 2, 0))
    canvas.paste(soc_panel, ((canvas_width - soc_panel.width) // 2, power.height))
    canvas.save(OUTPUT, format="PNG", dpi=(300, 300), optimize=True)
    svg_wrapper(OUTPUT, SVG_OUTPUT)


if __name__ == "__main__":
    main()
