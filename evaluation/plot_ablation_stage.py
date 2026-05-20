#!/usr/bin/env python3
"""
Plotly figure styled to resemble the provided reference (matplotlib-like):
- White background, dotted light grid
- Thick black axes/ticks
- Serif font
- Minimal/no title; legend on the right
- Same label => same color; sc==1 solid, otherwise dash-dash
- Only plots designability metric
- Reads metrics_ablation_stage.csv
- Run name format: B_{label}_SC_{sc} (label may be empty, e.g. B__SC_1)
- Legend label: $\mathcal{B}^{\mathrm{1\text{-}step}}_{\mathrm{label}} (\gamma=sc)$
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go

# ---- Color map by label (assigned deterministically in encounter order) ----
# Plotly's default qualitative palette (10 colors), with a fallback cycle.
DEFAULT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
FALLBACK_COLOR = "#7f7f7f"

RUN_RE = re.compile(r"^B_(?P<label>.*)_SC_(?P<sc>[\d.]+)$")


def parse_run_name(run_name: str) -> Tuple[str, float]:
    m = RUN_RE.match(run_name.strip())
    if not m:
        raise ValueError(
            f"run_name '{run_name}' does not match expected format B_{{label}}_SC_{{sc}}"
        )
    label = m.group("label")  # may be empty
    sc = float(m.group("sc"))
    return label, sc


def legend_label(label: str, sc: float) -> str:
    # Keep label visible even if empty
    label_tex = label if label != "" else r"\varnothing"
    # Print sc compactly (1.0 -> 1, 0.5 -> 0.5)
    sc_str = str(int(sc)) if abs(sc - int(sc)) < 1e-12 else str(sc)
    return rf"$\mathcal{{B}}^{{\mathrm{{1\text{{-}}step}}}}_{{\mathrm{{{label_tex}}}}}\ (\gamma={sc_str})$"


def main() -> None:
    in_csv = Path("csvs/metrics_ablation_narrow.csv")
    if not in_csv.exists():
        raise FileNotFoundError(f"Could not find input CSV at: {in_csv.resolve()}")

    df = pd.read_csv(in_csv)

    # Keep lengths 50..250 and sort
    df = df[df["length"].between(50, 250)].copy()
    df = df.sort_values(["run_name", "length"])

    # Convert to percentage (designability only)
    df["designability_pct"] = df["designability"] * 100.0

    run_names = sorted(df["run_name"].unique())

    # Assign colors by label (same label => same color)
    label_to_color: Dict[str, str] = {}
    next_color_idx = 0

    def color_for_label(lbl: str) -> str:
        nonlocal next_color_idx
        if lbl not in label_to_color:
            label_to_color[lbl] = (
                DEFAULT_PALETTE[next_color_idx % len(DEFAULT_PALETTE)]
                if DEFAULT_PALETTE
                else FALLBACK_COLOR
            )
            next_color_idx += 1
        return label_to_color[lbl]

    # ---- Single plot (designability only) ----
    fig = go.Figure()

    for run in run_names:
        lbl, sc = parse_run_name(run)
        color = color_for_label(lbl)
        dash = "solid" if abs(sc - 1.0) < 1e-9 else "dash"

        dfr = df[df["run_name"] == run]

        fig.add_trace(
            go.Scatter(
                x=dfr["length"],
                y=dfr["designability_pct"],
                mode="lines+markers",
                name=legend_label(lbl, sc),
                legendgroup=lbl if lbl != "" else "__EMPTY_LABEL__",
                line=dict(color=color, width=3.0, dash=dash),
                marker=dict(
                    size=7.5,
                    symbol="circle",
                    color=color,
                    line=dict(width=1.2, color=color),
                ),
                hovertemplate="<b>%{text}</b><br>Residues: %{x}<br>%{y:.2f}<extra></extra>",
                text=[run] * len(dfr),
            )
        )

    # ---- Axes styling to match the reference look ----
    x_ticks = list(range(50, 251, 50))
    y_ticks = list(range(0, 65, 20))

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
        title_text="Number of Residues",
        range=[50, 260],
        tickmode="array",
        tickvals=x_ticks,
        **axis_common,
    )

    fig.update_yaxes(
        title_text="Designability % ↑",
        range=[0, 75],
        tickmode="array",
        tickvals=y_ticks,
        **axis_common,
    )

    # ---- Layout: serif font, no big title, legend at right ----
    fig.update_layout(
        template="plotly_white",
        width=750,
        height=220,
        margin=dict(l=85, r=260, t=25, b=75),  # more right margin to fit legend text
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
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
            borderwidth=0,
            font=dict(size=18),
            itemsizing="constant",
            tracegroupgap=10,
        ),
        hovermode="closest",
    )

    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.show()
    # html_path = out_dir / "designability_plot.html"
    # fig.write_html(str(html_path), include_plotlyjs="cdn")

    # Optional static export (requires kaleido)
    # png_path = out_dir / "designability_plot.png"
    # fig.write_image(str(png_path), scale=2)


if __name__ == "__main__":
    main()
