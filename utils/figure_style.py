from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "csv"
FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "附件"
SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
TIMES = Path(r"C:\Windows\Fonts\times.ttf")
CN = FontProperties(fname=str(SIMSUN), size=9)
EN = FontProperties(fname=str(TIMES), size=9)
COLORS = {
    "blue": "#2878B5", "orange": "#F28E2B", "green": "#59A14F",
    "red": "#E15759", "purple": "#8E6C8A", "cyan": "#4EACC5",
    "gray": "#8C8C8C", "dark": "#333333", "light": "#D9EAF4",
}

def setup():
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": ["Times New Roman", "SimSun"],
        "font.size": 9, "axes.unicode_minus": False,
        "mathtext.fontset": "stix", "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
    })

def clean(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

def panel_name(ax, text):
    ax.text(0.5, -0.22, text, transform=ax.transAxes, ha="center", va="top",
            fontproperties=CN, clip_on=False)

def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

def time_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("时刻/h", fontproperties=CN)

def format_wan(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda x, _: f"{x / 10000:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)

