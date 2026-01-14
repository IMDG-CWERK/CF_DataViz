import pandas as pd
import plotly.express as px

# Input: your existing summary CSV
df = pd.read_csv("country_summary.csv")

# Normalize column names (just in case)
df.columns = [c.strip().lower() for c in df.columns]
df["country"] = df["country"].astype(str).str.upper()

# Choose metric for map (pick one)
df["blocked"] = df.get("block", 0)
df["challenged"] = df.get("managed_challenge", 0)
df["mitigated"] = df["blocked"] + df["challenged"]

METRIC = "blocked"  # change to: "challenged" or "mitigated"

title_map = {
    "blocked": "Cloudflare Firewall Events — Blocked by Country (last export)",
    "challenged": "Cloudflare Firewall Events — Managed Challenges by Country (last export)",
    "mitigated": "Cloudflare Firewall Events — Mitigated (Block + Challenge) by Country (last export)",
}

fig = px.choropleth(
    df,
    locations="country",
    locationmode="ISO-3",  # we'll convert ISO2->ISO3 below if needed
)

# --- Handle ISO2 (US, SG) properly ---
# Plotly prefers ISO-3 for countries (USA, SGP).
# Quick conversion without extra deps:
iso2_to_iso3 = {
    "US": "USA",
    "SG": "SGP",
    "DE": "DEU",
    "LT": "LTU",
    "AU": "AUS",
    # Add more as you see them, or use pycountry for full conversion.
}

df["iso3"] = df["country"].map(iso2_to_iso3).fillna(df["country"])
fig = px.choropleth(
    df,
    locations="iso3",
    color=METRIC,
    hover_name="country",
    hover_data={"blocked": True, "challenged": True, "mitigated": True, "iso3": False},
    color_continuous_scale="Viridis",
    title=title_map.get(METRIC, "Cloudflare by Country"),
)

fig.update_layout(
    margin=dict(l=10, r=10, t=50, b=10),
    geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"),
)

# Write ONE HTML file:
# - include_plotlyjs="cdn" is smaller (needs internet)
# - include_plotlyjs=True makes it fully self-contained (big file)
fig.write_html("cloudflare_map.html", include_plotlyjs="cdn")
# fig.write_html("cloudflare_map.html", include_plotlyjs=True)

print("Wrote cloudflare_map.html")
