"""
make_cf_map.py

Generates a single interactive HTML choropleth map (Plotly) from country_summary.csv.

Improvements included:
- Metric toggle dropdown: Blocked / Challenged / Mitigated
- “No events” countries muted (0 -> NaN so they appear as neutral land color)
- Cleaner hover tooltip (focused + consistent)
- Optional centroid markers for active countries (helps tiny countries like SG)
- Optional Top-N callout annotation box
- Mobile-friendly sizing + clean geo styling
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = Path("country_summary.csv")
OUTPUT_HTML = Path("cloudflare_map.html")

SHOW_MARKERS = (
    True  # marker dots at approximate country centroids (helps tiny countries)
)
SHOW_TOPN_CALLOUT = True  # adds a small "Top countries" annotation box
TOPN = 5  # number of countries in callout
INCLUDE_PLOTLYJS = (
    "cdn"  # "cdn" (small; needs internet) or True (fully self-contained but big)
)

# If your CSV is ISO-2 country codes (US, SG...), Plotly prefers ISO-3.
# This dictionary covers common ones; script will fall back safely if not found.
ISO2_TO_ISO3_DEFAULT: Dict[str, str] = {
    "US": "USA",
    "SG": "SGP",
    "DE": "DEU",
    "LT": "LTU",
    "AU": "AUS",
    "CA": "CAN",
    "MX": "MEX",
    "GB": "GBR",
    "FR": "FRA",
    "NL": "NLD",
    "SE": "SWE",
    "NO": "NOR",
    "ES": "ESP",
    "IT": "ITA",
    "JP": "JPN",
    "CN": "CHN",
    "RU": "RUS",
    "BR": "BRA",
    "IN": "IND",
}

# Approx centroids (lat, lon) for marker dots.
# Add entries as needed; this is intentionally small/lightweight for “map-only” use.
COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "SG": (1.3521, 103.8198),
    "US": (39.8283, -98.5795),
    "DE": (51.1657, 10.4515),
    "LT": (55.1694, 23.8813),
    "AU": (-25.2744, 133.7751),
    "CA": (56.1304, -106.3468),
    "MX": (23.6345, -102.5528),
    "GB": (55.3781, -3.4360),
    "FR": (46.2276, 2.2137),
    "NL": (52.1326, 5.2913),
}


def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0


def build_topn_text(df_plot: pd.DataFrame, metric_col: str, title: str, n: int) -> str:
    top = (
        df_plot[["country", metric_col]]
        .dropna()
        .sort_values(metric_col, ascending=False)
        .head(n)
    )
    if top.empty:
        return f"{title}\n(no events)"
    lines = [title]
    for _, r in top.iterrows():
        lines.append(f"{r['country']}: {safe_int(r[metric_col])}")
    return "\n".join(lines)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}. Run your aggregation script to generate it first."
        )

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    df["country"] = df["country"].astype(str).str.upper()

    # Normalize expected columns from your exporter
    df["blocked"] = pd.to_numeric(df.get("block", 0), errors="coerce").fillna(0)
    df["challenged"] = pd.to_numeric(
        df.get("managed_challenge", 0), errors="coerce"
    ).fillna(0)
    df["mitigated"] = df["blocked"] + df["challenged"]

    # ISO2 -> ISO3 for Plotly choropleth
    iso2_to_iso3 = ISO2_TO_ISO3_DEFAULT.copy()
    df["iso3"] = df["country"].map(iso2_to_iso3).fillna(df["country"])

    # Prepare “plot” versions of metrics: make zeros NaN so they appear as neutral land color
    df_plot = df.copy()
    for col in ["blocked", "challenged", "mitigated"]:
        df_plot.loc[df_plot[col] <= 0, col] = np.nan

    # Tooltip: tight + meaningful
    # We'll use customdata so hover is consistent even when switching metrics
    customdata = np.stack(
        [
            df["country"].values,
            df["blocked"].values,
            df["challenged"].values,
            df["mitigated"].values,
        ],
        axis=1,
    )

    # Base figure uses graph_objects so we can toggle traces cleanly
    fig = go.Figure()

    # Choropleth traces (one per metric)
    metric_defs = [
        ("blocked", "Blocked"),
        ("challenged", "Managed Challenge"),
        ("mitigated", "Mitigated (Block + Challenge)"),
    ]

    # Choose an initial view
    initial_metric = "mitigated" if df["mitigated"].sum() > 0 else "blocked"

    for metric_key, metric_label in metric_defs:
        visible = metric_key == initial_metric
        fig.add_trace(
            go.Choropleth(
                locations=df_plot["iso3"],
                z=df_plot[metric_key],
                locationmode="ISO-3",
                colorscale="Viridis",
                zmin=0,
                # zmax auto; if you want fixed, set it here
                marker_line_color="rgba(255,255,255,0.35)",
                marker_line_width=0.5,
                colorbar_title=metric_label,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Blocked: %{customdata[1]:.0f}<br>"
                    "Challenged: %{customdata[2]:.0f}<br>"
                    "Total: %{customdata[3]:.0f}"
                    "<extra></extra>"
                ),
                name=metric_label,
                visible=visible,
            )
        )

    # Optional: markers for active countries (helps tiny ones like SG)
    if SHOW_MARKERS:
        active = df[df["mitigated"] > 0].copy()
        active["lat"] = active["country"].map(
            lambda c: COUNTRY_CENTROIDS.get(c, (np.nan, np.nan))[0]
        )
        active["lon"] = active["country"].map(
            lambda c: COUNTRY_CENTROIDS.get(c, (np.nan, np.nan))[1]
        )
        active = active.dropna(subset=["lat", "lon"])

        # One marker trace (always visible; it’s an “attention guide”)
        fig.add_trace(
            go.Scattergeo(
                lat=active["lat"],
                lon=active["lon"],
                mode="markers+text" if len(active) <= 8 else "markers",
                text=active["country"] if len(active) <= 8 else None,
                textposition="top center",
                marker=dict(size=8, opacity=0.85),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Blocked: %{customdata[0]}<br>"
                    "Challenged: %{customdata[1]}<br>"
                    "Total: %{customdata[2]}"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        active["blocked"].astype(int).values,
                        active["challenged"].astype(int).values,
                        active["mitigated"].astype(int).values,
                    ],
                    axis=1,
                ),
                name="Active countries",
                showlegend=False,
            )
        )

    # Dropdown to toggle between metrics (choropleths only; markers stay on)
    # Choropleth traces are indices 0..2; marker trace (if present) is last index
    n_choro = len(metric_defs)
    n_total_traces = len(fig.data)

    def visible_mask_for(metric_index: int) -> list[bool]:
        mask = [False] * n_total_traces
        # show only selected choropleth
        mask[metric_index] = True
        # keep marker trace visible if enabled
        if SHOW_MARKERS and n_total_traces > n_choro:
            mask[-1] = True
        return mask

    buttons = []
    for i, (metric_key, metric_label) in enumerate(metric_defs):
        buttons.append(
            dict(
                label=metric_label,
                method="update",
                args=[
                    {"visible": visible_mask_for(i)},
                    {
                        "title": f"Cloudflare Firewall Events — {metric_label} by Country (last export)"
                    },
                ],
            )
        )

    # Optional top-N callout (updates when switching metric is hard in pure static HTML;
    # we show mitigated by default and keep it simple)
    annotations = []
    if SHOW_TOPN_CALLOUT:
        callout_text = build_topn_text(df, "mitigated", f"Top {TOPN} (Total)", TOPN)
        annotations.append(
            dict(
                text=callout_text.replace("\n", "<br>"),
                x=0.01,
                y=0.02,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                align="left",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Cloudflare Firewall Events — {dict(metric_defs).get(initial_metric, 'Blocked')} by Country (last export)",
            x=0.02,
            xanchor="left",
        ),
        
        margin=dict(l=10, r=10, t=70, b=10),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )
        ],
        annotations=annotations,
    )

    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(255,255,255,0.35)",
        showcoastlines=False,
        showframe=False,
        projection_type="natural earth",
        landcolor="rgb(245,245,245)",  # muted land for “no events”
        bgcolor="white",
    )

    # Add a small subtitle note (keeps interpretation clean)
    fig.add_annotation(
        text="Source: Cloudflare firewall events export (point-in-time; edge mitigations)",
        x=0.02,
        y=1.02,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11, color="rgba(0,0,0,0.65)"),
    )

    fig.write_html(str(OUTPUT_HTML), include_plotlyjs=INCLUDE_PLOTLYJS)
    print(f"Wrote {OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
