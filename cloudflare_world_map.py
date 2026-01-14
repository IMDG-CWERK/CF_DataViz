import json
from pathlib import Path

import numpy as np
import pandas as pd

# Geo packages
import geopandas as gpd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load Cloudflare events
# ----------------------------
# Put your exported JSON in a file, e.g. cloudflare_events.json
INPUT_JSON = Path("./data/cloudflare_events.json")

with INPUT_JSON.open("r", encoding="utf-8") as f:
    events = json.load(f)

df = pd.DataFrame(events)

# Basic cleanup / normalization
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
df["action"] = df["action"].fillna("unknown").str.lower()
df["country"] = df["clientCountryName"].fillna("XX").str.upper()

# Optional: focus on just 24h or just certain actions
# df = df[df["datetime"] >= (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=24))]

# ----------------------------
# 2) Aggregate metrics
# ----------------------------
# All mitigations
counts_all = (
    df.groupby(["country", "action"], as_index=False)
      .size()
      .rename(columns={"size": "events"})
)

# Blocked-only (recommended map)
counts_block = (
    df[df["action"].eq("block")]
    .groupby("country", as_index=False)
    .size()
    .rename(columns={"size": "blocked_events"})
)

# Useful side summaries (top talkers)
top_countries = counts_block.sort_values("blocked_events", ascending=False).head(15)
top_rules = (
    df.groupby(["description", "action"], as_index=False)
      .size()
      .rename(columns={"size": "events"})
      .sort_values("events", ascending=False)
      .head(15)
)
top_asns = (
    df.groupby(["clientASNDescription", "clientAsn"], as_index=False)
      .size()
      .rename(columns={"size": "events"})
      .sort_values("events", ascending=False)
      .head(15)
)

# Export a “council table”
summary_out = (
    counts_all.pivot_table(index="country", columns="action", values="events", fill_value=0)
    .reset_index()
)
summary_out.to_csv("country_summary.csv", index=False)

print("\nTop blocked countries:\n", top_countries.to_string(index=False))
print("\nTop rules:\n", top_rules.to_string(index=False))
print("\nTop ASNs:\n", top_asns.to_string(index=False))

# ----------------------------
# 3) Load country boundaries
# ----------------------------
NE_ADMIN0_SHP = Path(r".\data\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp")

def load_world_boundaries() -> gpd.GeoDataFrame:
    if not NE_ADMIN0_SHP.exists():
        raise FileNotFoundError(
            f"Natural Earth shapefile not found at: {NE_ADMIN0_SHP}\n"
            "Download and extract Natural Earth Admin 0 Countries (110m), then update NE_ADMIN0_SHP."
        )

    world = gpd.read_file(NE_ADMIN0_SHP)

    # Normalize ISO2 column name for joining (Cloudflare uses ISO-2 like US, SG)
    # Natural Earth often has 'ISO_A2' (sometimes '-99' for disputed areas)
    if "ISO_A2" in world.columns:
        world["country"] = world["ISO_A2"].astype(str).str.upper()
    else:
        # Fallback: try ISO3 if needed
        if "ISO_A3" in world.columns:
            world["iso3"] = world["ISO_A3"].astype(str).str.upper()
        else:
            raise RuntimeError("Shapefile missing ISO_A2/ISO_A3 columns; choose a different Natural Earth Admin 0 dataset.")

    return world

world = load_world_boundaries()

# If your world layer has ISO_A2 country codes, great.
# naturalearth_lowres has 'iso_a3' and 'name' commonly; some layers differ.
# We'll attempt to join on ISO_A2 first, then ISO_A3. Your Cloudflare export uses ISO_A2 (e.g., US, SG).
# If your boundary file doesn't include ISO_A2, download Natural Earth Admin 0 Countries which includes it.

# Normalize column options
cols = set(map(str.lower, world.columns))
iso2_col = None
iso3_col = None

for c in world.columns:
    lc = c.lower()
    if lc in {"iso_a2", "iso2", "adm0_a2"}:
        iso2_col = c
    if lc in {"iso_a3", "iso3", "adm0_a3"}:
        iso3_col = c

# Prepare joined geodataframe
world_plot = world.copy()

if iso2_col:
    world_plot["country"] = world_plot[iso2_col].astype(str).str.upper()
    joined = world_plot.merge(counts_block, on="country", how="left")
elif iso3_col:
    # Cloudflare gives ISO2; if only ISO3 exists you can map ISO2->ISO3 via pycountry (optional)
    try:
        import pycountry
        def iso2_to_iso3(code2: str) -> str:
            try:
                return pycountry.countries.get(alpha_2=code2).alpha_3
            except Exception:
                return None

        counts_block["iso3"] = counts_block["country"].map(iso2_to_iso3)
        world_plot["iso3"] = world_plot[iso3_col].astype(str).str.upper()
        joined = world_plot.merge(counts_block, left_on="iso3", right_on="iso3", how="left")
    except ImportError:
        raise RuntimeError(
            "Boundary data doesn't have ISO_A2. Install pycountry OR use Natural Earth Admin 0 Countries shapefile."
        )
else:
    raise RuntimeError(
        "Boundary data missing ISO country code columns. Use Natural Earth Admin 0 Countries shapefile."
    )

# joined["blocked_events"] = joined["blocked_events"].fillna(0)
blocked = joined[joined["blocked_events"].fillna(0) > 0]

# ----------------------------
# 4) Plot choropleth + outline + optional labels
# ----------------------------

# Decide what you want to map:
#   - blocked only (current behavior)
#   - mitigated (block + managed_challenge)
MAP_MODE = "mitigated"   # change to "mitigated" if you want block+challenge totals
SHOW_LABELS = True     # optional labels on outlined countries

# Build the metric column for plotting
joined = joined.copy()

# Ensure blocked_events exists and is numeric
joined["blocked_events"] = pd.to_numeric(joined.get("blocked_events"), errors="coerce")

if MAP_MODE == "blocked":
    joined["metric"] = joined["blocked_events"]
    legend_label = "Blocked events"
elif MAP_MODE == "mitigated":
    # Build mitigated counts by country: block + managed_challenge
    mitig = (
        df[df["action"].isin(["block", "managed_challenge"])]
        .groupby(["country", "action"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
        .pivot_table(index="country", columns="action", values="events", fill_value=0)
        .reset_index()
    )
    mitig["mitigated_events"] = mitig.get("block", 0) + mitig.get("managed_challenge", 0)
    joined = joined.merge(mitig[["country", "mitigated_events"]], on="country", how="left")
    joined["metric"] = pd.to_numeric(joined["mitigated_events"], errors="coerce")
    legend_label = "Mitigated events (block + challenge)"
else:
    raise ValueError("MAP_MODE must be 'blocked' or 'mitigated'")

# Make "no events" show as missing (so missing_kwds paints gray)
joined["metric"] = joined["metric"].fillna(0)
joined.loc[joined["metric"] == 0, "metric"] = np.nan

# Use log scale only for visualization; keep raw metric for labels
joined["metric_log"] = np.log10(joined["metric"].fillna(0) + 1)

# Countries to outline (where metric is present)
hot = joined[joined["metric"].notna()].copy()

fig, ax = plt.subplots(figsize=(14, 8))

# Base choropleth
joined.plot(
    column="metric_log",
    ax=ax,
    legend=True,
    legend_kwds={"label": f"{legend_label} (log10 scale)", "shrink": 0.6},
    missing_kwds={"color": "lightgrey", "label": "No events"},
)

# Outline highlighted countries
hot.boundary.plot(ax=ax, linewidth=2.5, color="red")

# Optional labels
if SHOW_LABELS and not hot.empty:
    for _, row in hot.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # representative_point is safer than centroid for multi-polygons / weird shapes
        pt = geom.representative_point()
        # If you're mapping blocked-only, label with blocked_events; else label with metric
        value = int(row["metric"]) if pd.notna(row["metric"]) else 0
        ax.text(
            pt.x,
            pt.y,
            f"{row['country']} ({value})",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

title_suffix = "Blocked Requests" if MAP_MODE == "blocked" else "Mitigated Requests (Block + Challenge)"
ax.set_title(f"Cloudflare Firewall Events – {title_suffix} by Country (Last 24h export)")
ax.set_axis_off()

plt.tight_layout()
out_name = "cloudflare_world_map_highlighted.png" if MAP_MODE == "blocked" else "cloudflare_world_map_mitigated_highlighted.png"
plt.savefig(out_name, dpi=200)
plt.close()

print(f"\nSaved map: {out_name}")