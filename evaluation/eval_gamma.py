#!/usr/bin/env python3
"""
Designability vs gamma (sc) plot.

Changes vs previous:
- x-axis is gamma (sc), y-axis is designability (%)
- gamma grid: [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0] and x-range [0, 1]
- Data are extracted by counting files in:
    {root}/{model_path}_sc_{gamma}/pdbs/designable/
    {root}/{model_path}_sc_{gamma}/pdbs/undesignable/
  designability = designable / (designable + undesignable)
- Same color mapping by nstep as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

# ---- Same colors as before (by nstep) ----
NSTEP_COLOR_MAP: Dict[int, str] = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    5: "#2ca02c",  # green
    8: "#9467bd",  # purple
}
FALLBACK_COLOR = "#7f7f7f"

GAMMAS: List[float] = [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0]

ROOT = Path("/homes/kasram/broteina/SiD_Protein/design_eval_sc_search")


def count_files(dirpath: Path) -> int:
    """
    Count regular files under a directory (non-recursive).
    If the directory does not exist, return 0.
    """
    if not dirpath.exists() or not dirpath.is_dir():
        return 0
    return sum(1 for p in dirpath.iterdir() if p.is_file())


def load_designability_counts(
    ckpt_names: Sequence[Tuple[str, str]],
    gammas: Sequence[float] = GAMMAS,
    root: Path = ROOT,
) -> pd.DataFrame:
    """
    Build a tidy dataframe with designability computed from filesystem counts.

    Returns columns:
        model_type, model_path, gamma, n_designable, n_undesignable, designability
    """
    rows = []
    for model_type, model_path in ckpt_names:
        for gamma in gammas:
            # Match your requested directory naming
            run_dir = root / (f"{model_path}" + (f"_sc_{gamma}" if gamma != 1 else ""))
            des_dir = run_dir / "pdbs" / "designable"
            undes_dir = run_dir / "pdbs" / "undesignable"

            n_des = count_files(des_dir)
            n_undes = count_files(undes_dir)
            denom = n_des + n_undes
            designability = (n_des / denom) if denom > 0 else float("nan")

            rows.append(
                dict(
                    model_type=model_type,
                    model_path=model_path,
                    gamma=float(gamma),
                    n_designable=int(n_des),
                    n_undesignable=int(n_undes),
                    designability=designability,
                )
            )

    df = pd.DataFrame(rows).sort_values(["model_type", "gamma"]).reset_index(drop=True)
    return df


def nstep_from_model_type(model_type: str) -> int:
    """
    Extract nstep from strings like 'Broteina_1', 'Broteina_2', etc.
    Falls back to 0 if parsing fails.
    """
    try:
        return int(model_type.split("_")[-1])
    except Exception:
        return 0


def main() -> None:
    # Example input list (replace/extend as needed)
    ckpt_names = [
        ("Broteina_1", "network-snapshot-1.000000-002875.pkl"),
        ("Broteina_2", "network-snapshot-1.000000-001892.pkl"),
        ("Broteina_5", "network-snapshot-1.000000-000434.pkl"),
        ("Broteina_8", "network-snapshot-1.000000-000335.pkl"),
    ]

    df = load_designability_counts(ckpt_names)

    # Convert to % for plotting
    df["designability_pct"] = df["designability"] * 100.0

    fig = go.Figure()

    for model_type, _ in ckpt_names:
        dfr = df[df["model_type"] == model_type].copy()
        nstep = nstep_from_model_type(model_type)
        color = NSTEP_COLOR_MAP.get(nstep, FALLBACK_COLOR)

        # Legend label (kept consistent with prior style)
        # You can change the legend text if you want more LaTeX here.
        label = f"{nstep} step{'' if nstep == 1 else 's'}"

        fig.add_trace(
            go.Scatter(
                x=dfr["gamma"],
                y=dfr["designability_pct"],
                mode="lines+markers",
                name=label,
                legendgroup=model_type,
                line=dict(color=color, width=3.0, dash="solid"),
                marker=dict(
                    size=7.5,
                    symbol="circle",
                    color=color,
                    line=dict(width=1.2, color=color),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "γ: %{x}<br>"
                    "Designability: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
                text=[model_type] * len(dfr),
            )
        )

    # ---- Axes styling (matplotlib-like) ----
    axis_common = dict(
        ticks="outside",
        ticklen=7,
        tickwidth=2,
        tickcolor="black",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        griddash="dot",
        gridwidth=1,
        zeroline=False,
    )

    fig.update_xaxes(
        title_text=r"$\gamma$",
        range=[0.0, 1.0],
        tickmode="array",
        tickvals=GAMMAS,
        **axis_common,
    )
    fig.update_yaxes(
        title_text="Designability % ↑",
        range=[0, 100],
        tickmode="array",
        tickvals=list(range(0, 101, 20)),
        **axis_common,
    )

    # ---- Layout ----
    # Legend on the right; increase right margin to avoid clipping
    fig.update_layout(
        template="plotly_white",
        width=760,
        height=360,
        margin=dict(l=85, r=260, t=25, b=75),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Times New Roman, Times, STIXGeneral, serif",
            size=18,
            color="black",
        ),
        legend=dict(
            title=None,
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
            borderwidth=0,
            font=dict(size=18),
            itemsizing="constant",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    fig.show()


if __name__ == "__main__":
    main()
