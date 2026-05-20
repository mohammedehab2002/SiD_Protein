#!/usr/bin/env python3
"""
Create a 5x4 grid from images named B{nstep}_{nres}.png

Rows: nres = [50, 100, 150, 200, 250]
Cols: nstep = [1, 2, 5, 8]

Caption under each tile:
  $\mathcal{B}^{j\mathrm{-step}}\ \mathrm{at}\ L=i$
(where j=nstep, i=nres)
"""

from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


NSTEPS = [1, 5, 8, 16]
NRES   = [50, 100, 150, 200, 250]

OUTFILE = "grid_5x4.png"

# Caption styling
CAPTION_FONTSIZE = 18       # make larger here
CAPTION_Y = -0.07           # push caption below axes (more negative -> lower)
FIG_DPI = 300

# If you want *full* LaTeX rendering (requires a LaTeX install), set to True.
# Otherwise, Matplotlib's mathtext will render \mathcal and superscripts fine.
USE_TEX = False


def caption(nstep: int, nres: int) -> str:
    # Use \mathrm instead of \text for max compatibility with mathtext.
    return rf"$\mathcal{{B}}^{{{nstep}\mathrm{{-step}}}}, L={nres}$"


def main(folder: str = ".") -> None:
    folder_path = Path(folder)

    plt.rcParams["text.usetex"] = USE_TEX
    # If you enable USE_TEX and want nicer fonts, you can also uncomment:
    # plt.rcParams["font.family"] = "serif"

    n_rows = len(NRES)
    n_cols = len(NSTEPS)

    # Figure size: tune for your image resolution / desired output
    fig_w = 4.0 * n_cols
    fig_h = 3.3 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Spacing: bottom room for captions under each axes
    plt.subplots_adjust(
        left=0.03, right=0.99, top=0.99, bottom=0.03,
        wspace=0.05, hspace=0.35
    )

    missing = []

    for i_row, nres in enumerate(NRES):
        for j_col, nstep in enumerate(NSTEPS):
            ax = axes[i_row, j_col]
            fname = folder_path / f"B{nstep}_{nres}.png"
            if not fname.exists():
                missing.append(str(fname))
                ax.axis("off")
                continue

            img = mpimg.imread(fname)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            # Caption under image
            ax.text(
                0.5, CAPTION_Y,
                caption(nstep, nres),
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=CAPTION_FONTSIZE
            )

    if missing:
        raise FileNotFoundError("Missing expected files:\n" + "\n".join(missing))

    out_path = folder_path / OUTFILE
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    dir = "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/renders/"
    main(dir)
