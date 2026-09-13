from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
PATTERN = re.compile(r"(?<![A-Za-z0-9_])(-?\d+)\.(\d{3,})(?!\d)")


def rounded(match: re.Match[str]) -> str:
    value = Decimal(match.group(0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{value:.2f}"


for path in [ROOT / "main.tex", *sorted((ROOT / "sections").glob("*.tex"))]:
    source = path.read_text(encoding="utf-8")
    path.write_text(PATTERN.sub(rounded, source), encoding="utf-8")
