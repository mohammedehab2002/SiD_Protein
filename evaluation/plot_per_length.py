#!/usr/bin/env python3
"""
Plotly figure styled to resemble the provided reference (matplotlib-like):
- White background, dotted light grid
- Thick black axes/ticks
- Serif font
- Minimal/no title; legend on the right
- Same nstep => same color; sc==1 solid, otherwise dashed
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Color map by nstep (keep your rule) ----
NSTEP_COLOR_MAP: Dict[int, str] = {
    1: "#1f77b4",  # blue
    5: "#ff7f0e",  # orange
    8: "#2ca02c",  # green
    16: "#9467bd",  # purple
}
FALLBACK_COLOR = "#7f7f7f"

RUN_RE = re.compile(r"^Broteina_(?P<nstep>\d+)_SC_(?P<sc>[\d.]+)$")


def parse_run_name(run_name: str) -> Tuple[int, float]:
    m = RUN_RE.match(run_name.strip())
    if not m:
        raise ValueError(
            f"run_name '{run_name}' does not match expected format Broteina_{{nstep}}_SC_{{sc}}"
        )
    return int(m.group("nstep")), float(m.group("sc"))


def main() -> None:
    in_csv = Path("/homes/kasram/broteina/SiD_Protein/evaluation/csvs/metrics.csv")
    if not in_csv.exists():
        raise FileNotFoundError(f"Could not find input CSV at: {in_csv.resolve()}")

    df = pd.read_csv(in_csv)

    # Keep lengths 50..250 and sort
    df = df[df["length"].between(50, 250)].copy()
    df = df.sort_values(["run_name", "length"])

    # Convert to percentages
    df["designability_pct"] = df["designability"] * 100.0
    df["diversity_pct"] = 100.0 * df["designability"] * df["diversity"]

    # --- Subplots (no subplot titles; reference uses just axis labels) ---
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
    )

    # Sort by (nstep, sc) so the legend reads 1 -> 5 -> 8 -> 16 instead of
    # lexicographic order (which puts "Broteina_16_..." before "Broteina_1_..."
    # because '_' > '6').
    run_names = sorted(df["run_name"].unique(), key=parse_run_name)


    for run in run_names:
        nstep, sc = parse_run_name(run)
        color = NSTEP_COLOR_MAP.get(nstep, FALLBACK_COLOR)
        dash = "solid" if abs(sc - 1.0) < 1e-9 else "dash"

        dfr = df[df["run_name"] == run]

        # Legend label: keep run or simplify; this keeps run_name (as in your current script)
        nsteps = run.split("_")[1]
        sc = run.split("_")[-1]
        label = f"{nsteps} step{'' if nstep == 1 else 's'} (γ = {sc})"

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
            hovertemplate="<b>%{text}</b><br>Residues: %{x}<br>%{y:.2f}<extra></extra>",
            text=[label] * len(dfr),
        )

        fig.add_trace(
            go.Scatter(x=dfr["length"], y=dfr["designability_pct"], **common),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dfr["length"],
                y=dfr["diversity_pct"],
                showlegend=False,  # show legend only once (left subplot)
                **common,
            ),
            row=1,
            col=2,
        )

    # ---- Axes styling to match the reference look ----
    x_ticks = list(range(50, 251, 50))
    y_ticks = list(range(0, 101, 20))

    axis_common = dict(
        ticks="outside",
        ticklen=7,
        tickwidth=2,
        tickcolor="black",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False,  # reference shows left/bottom emphasis; set True if you want full box
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        griddash="dot",
        gridwidth=1,
        zeroline=False,
    )

    fig.update_xaxes(
        title_text="Number of Residues",
        range=[50, 260],
        tickmode="array",
        tickvals=x_ticks,
        **axis_common,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Number of Residues",
        range=[50, 260],
        tickmode="array",
        tickvals=x_ticks,
        **axis_common,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Designability % ↑",
        range=[0, 100],
        tickmode="array",
        tickvals=y_ticks,
        **axis_common,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Diversity (# Designable Clusters) ↑",
        range=[0, 100],
        tickmode="array",
        tickvals=y_ticks,
        **axis_common,
        row=1,
        col=2,
    )

    # ---- Layout: serif font, no big title, legend at right ----
    fig.update_layout(
        template="plotly_white",
        width=1050,
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

    # Slightly bigger axis titles, similar to matplotlib defaults in the reference
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.show()
    # html_path = out_dir / "metrics_plot.html"
    # fig.write_html(str(html_path), include_plotlyjs="cdn")

    # Optional static export (requires kaleido)
    # png_path = out_dir / "metrics_plot.png"
    # fig.write_image(str(png_path), scale=2)


if __name__ == "__main__":
    main()
