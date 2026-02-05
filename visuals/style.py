from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

PALETTE = {
    "primary": "#753631",
    "secondary": "#245A4D",
    "accent": "#6B5B3B",
    "danger": "#C00000",
    "neutral": "#000000",
    "highlight": "#C8A27A",
}

VIBRANT_COLORS = [
    "#E15759",
    "#F28E2B",
    "#59A14F",
    "#4E79A7",
    "#B07AA1",
    "#EDC948",
]



def custom_theme() -> dict:
    return {
        # --- Axis & Spines ---
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        # Hide top and right lines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # --- Main Titles ---
        "axes.titlesize": 16,       # Size 16
        "axes.titleweight": "bold", # Bold
        "axes.titlecolor": "black", # Black
        "axes.titlelocation": "center", # Centered

        # --- Axis Titles (X/Y Labels) ---
        "axes.labelsize": 13,       # Size 13
        "axes.labelweight": "bold", # Bold
        "axes.labelcolor": "black", # Black

        # --- Axis Labels (Ticks) ---
        "xtick.labelsize": 10,      # Size 10
        "ytick.labelsize": 10,      # Size 10
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.major.pad": 8,
        "ytick.major.pad": 6,

        # --- Colors & Fonts ---
        "axes.prop_cycle": cycler(
            color=[
                PALETTE["primary"],
                PALETTE["secondary"],
                PALETTE["accent"],
                PALETTE["highlight"],
                VIBRANT_COLORS[0],
                VIBRANT_COLORS[1],
                VIBRANT_COLORS[2],
            ]
        ),
        "text.color": "black",
        "font.size": 12,
        "font.family": "Cambria",
    }


def apply_style() -> None:
    plt.rcParams.update(custom_theme())


def annotate_point(ax, text: str, xy: tuple[float, float], xytext: tuple[int, int] = (8, 8), color: str = "#000000") -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="offset points",
        fontsize=10,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
    )


def add_headroom(ax, factor: float = 1.25) -> None:
    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        return
    ax.set_ylim(ymin, ymax * factor)
