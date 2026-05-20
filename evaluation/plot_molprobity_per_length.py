#!/usr/bin/env python3
"""
Per-length plot of MolProbity metrics, styled like plot_per_length.py.

Reads the compact summary CSV at:
  /homes/kasram/broteina/SiD_Protein/design_eval_neurips/neurips_selfd/
    molprobity_summary_compact.csv
and renders three subplots side-by-side:
  1. MP-Score        (molprobity_score_mean)   ↓ lower is better
  2. Clash Score     (clashscore_mean)         ↓ lower is better
  3. Ram Outliers %  (rama_outliers_pct_mean)  ↓ lower is better

Same color/dash convention as plot_per_length.py:
  - Color is keyed by nstep (1, 5, 8, 16)
  - sc == 1.0 -> solid, otherwise dashed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NSTEP_COLOR_MAP: Dict[int, str] = {
    1: "#1f77b4",   # blue
    5: "#ff7f0e",   # orange
    8: "#2ca02c",   # green
    16: "#9467bd",  # purple
}
FALLBACK_COLOR = "#7f7f7f"


def main() -> None:
    in_csv = Path(
        "/homes/kasram/broteina/SiD_Protein/design_eval_neurips/"
        "neurips_selfd/molprobity_summary_compact.csv"
    )
    if not in_csv.exists():
        raise FileNotFoundError(f"Could not find input CSV at: {in_csv.resolve()}")

    df = pd.read_csv(in_csv)

    # Keep lengths 50..250, drop rows with no designable structures (count == 0)
    df = df[df["length"].between(50, 250)].copy()
    df = df[df["count"] > 0].copy()
    df = df.sort_values(["steps", "sc", "length"])

    # Standard error of the mean for each metric: sem = std / sqrt(n).
    # With n == 1 the underlying std is 0 (pstdev fallback), so sem == 0 and
    # the error bar collapses — good, since a single sample carries no
    # estimated uncertainty.
    sqrt_n = np.sqrt(df["count"].to_numpy())
    for metric in ("molprobity_score", "clashscore", "rama_outliers_pct"):
        df[f"{metric}_sem"] = df[f"{metric}_std"].to_numpy() / sqrt_n

    # --- Subplots: 1 row x 3 cols ---
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.085)

    # Order legend by (nstep asc, then sc desc so γ=1 appears before γ<1)
    runs = (
        df[["steps", "sc"]]
        .drop_duplicates()
        .sort_values(["steps", "sc"], ascending=[True, False])
        .itertuples(index=False, name=None)
    )

    for nstep, sc in runs:
        color = NSTEP_COLOR_MAP.get(int(nstep), FALLBACK_COLOR)
        dash = "solid" if abs(float(sc) - 1.0) < 1e-9 else "dash"

        dfr = df[(df["steps"] == nstep) & (df["sc"] == sc)].sort_values("length")
        if dfr.empty:
            continue

        # Format sc consistently (e.g. 1.0 -> "1", 0.8 -> "0.8")
        sc_str = f"{sc:g}"
        label = f"{int(nstep)} step{'' if int(nstep) == 1 else 's'} (γ = {sc_str})"

        common = dict(
            mode="lines+markers",
            name=label,
            legendgroup=label,
            line=dict(color=color, width=3.0, dash=dash),
            marker=dict(
                size=7.5,
                symbol="circle",
                color=color,
                line=dict(width=1.2, color=color),
            ),
            text=[label] * len(dfr),
        )

        def err(metric: str) -> dict:
            return dict(
                type="data",
                array=dfr[f"{metric}_sem"].to_numpy(),
                visible=True,
                color=color,
                thickness=1.5,
                width=4,
            )

        fig.add_trace(
            go.Scatter(
                x=dfr["length"],
                y=dfr["molprobity_score_mean"],
                error_y=err("molprobity_score"),
                hovertemplate=(
                    "<b>%{text}</b><br>Residues: %{x}<br>"
                    "MP-Score: %{y:.2f} ± %{error_y.array:.2f}<extra></extra>"
                ),
                **common,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dfr["length"],
                y=dfr["clashscore_mean"],
                error_y=err("clashscore"),
                showlegend=False,
                hovertemplate=(
                    "<b>%{text}</b><br>Residues: %{x}<br>"
                    "Clash Score: %{y:.2f} ± %{error_y.array:.2f}<extra></extra>"
                ),
                **common,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=dfr["length"],
                y=dfr["rama_outliers_pct_mean"],
                error_y=err("rama_outliers_pct"),
                showlegend=False,
                hovertemplate=(
                    "<b>%{text}</b><br>Residues: %{x}<br>"
                    "Ram Outliers: %{y:.2f} ± %{error_y.array:.2f}%<extra></extra>"
                ),
                **common,
            ),
            row=1,
            col=3,
        )

    # ---- Axes styling ----
    x_ticks = list(range(50, 251, 50))

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

    for col in (1, 2, 3):
        fig.update_xaxes(
            title_text="Number of Residues",
            range=[50, 260],
            tickmode="array",
            tickvals=x_ticks,
            **axis_common,
            row=1,
            col=col,
        )

    # MP-Score: values fall roughly in [2.5, 4.5]; pad a bit.
    fig.update_yaxes(
        title_text="MP-Score ↓",
        range=[2.4, 4.5],
        **axis_common,
        row=1,
        col=1,
    )
    # Clash Score: extended to fit SEM bars on the small-n 1-step buckets.
    fig.update_yaxes(
        title_text="Clash Score ↓",
        range=[0, 260],
        **axis_common,
        row=1,
        col=2,
    )
    # Ram Outliers %: extended for the same reason.
    fig.update_yaxes(
        title_text="Ram Outliers % ↓",
        range=[0, 10],
        **axis_common,
        row=1,
        col=3,
    )

    # ---- Layout ----
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=360,
        margin=dict(l=85, r=240, t=25, b=75),
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
            tracegroupgap=5,
        ),
        hovermode="closest",
    )

    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.show()
    # html_path = out_dir / "molprobity_per_length.html"
    # fig.write_html(str(html_path), include_plotlyjs="cdn")
    # png_path = out_dir / "molprobity_per_length.png"
    # fig.write_image(str(png_path), scale=2)


if __name__ == "__main__":
    main()
