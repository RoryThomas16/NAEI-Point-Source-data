import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# Minimal, self-contained Streamlit app built from first principles.
# - Loads an Excel workbook `NAEIPointsSource_2023.xlsx` (or accepts upload)
# - Computes Emissions_GWP robustly
# - Filters by Year, Region, Sector, and threshold
# - Aggregates to site-level and computes pct_of_region and pct_of_uk
# - Renders a Plotly treemap with hover info per point source

# --- Page config (must be the first Streamlit UI call) ---
st.set_page_config(layout="wide", page_title="NAEI Treemap Explorer")
st.title("NAEI Point Source — Treemap Explorer")

DATA_FILE = Path(__file__).with_name("NAEIPointsSource_2023.xlsx")


def load_dataframe(uploaded_file) -> pd.DataFrame | None:
    """Load dataframe from uploaded file or default workbook next to this script.

    Returns None if no file found or it can't be read.
    """
    # prefer uploaded file
    xls = None
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded file: {e}")
            return None

    # otherwise try default workbook next to script
    if xls is None and DATA_FILE.exists():
        try:
            xls = pd.ExcelFile(DATA_FILE, engine="openpyxl")
        except Exception as e:
            st.sidebar.error(f"Failed to read local data file {DATA_FILE.name}: {e}")
            return None

    if xls is None:
        return None

    # choose a sheet similar to the notebook: prefer sheet index 3 if available
    try:
        sheets = xls.sheet_names
        sheet_idx = 3 if len(sheets) > 3 else 0
        df = pd.read_excel(xls, sheet_name=sheets[sheet_idx], engine="openpyxl")
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to load sheet from workbook: {e}")
        return None


def ensure_emissions_gwp(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with numeric Emissions_GWP column.

    Strategy:
    - If `Emissions_GWP` present: coerce numeric.
    - Else look for common emission columns and multiply by `GWP` if available.
    - If `GWP` not available, try to infer from pollutant name (CH4/N2O/CO2), else assume GWP=1.
    """
    df = df.copy()
    if "Emissions_GWP" in df.columns:
        df["Emissions_GWP"] = pd.to_numeric(df["Emissions_GWP"], errors="coerce").fillna(0.0)
        return df

    # detect emission-like columns more flexibly (case-insensitive substring match)
    cols_lower = {c: c.lower() for c in df.columns}
    emis_col = None
    for c, low in cols_lower.items():
        if "emiss" in low or "emission" in low or "tco2" in low:
            emis_col = c
            break
    # also consider some common exact names if not yet found
    if emis_col is None:
        for cand in ["Emissions", "Emission", "Value", "Quantity", "Total_Emissions"]:
            if cand in df.columns:
                emis_col = cand
                break

    # build GWP series
    if "GWP" in df.columns:
        gwp = pd.to_numeric(df["GWP"], errors="coerce").fillna(1.0)
    else:
        pname = next((c for c in ["Pollutant_Name", "Pollutant", "PollutantName"] if c in df.columns), None)
        if pname is None:
            gwp = pd.Series(1.0, index=df.index)
        else:
            names = df[pname].astype(str).str.lower()
            gwp = pd.Series(1.0, index=df.index)
            gwp = gwp.mask(names.str.contains(r"methane|\bch4\b", na=False), 28.0)
            gwp = gwp.mask(names.str.contains(r"nitrous|n2o|n20", na=False), 265.0)
            gwp = gwp.mask(names.str.contains(r"co2|carbon dioxide|\bco2\b|carbon\b", na=False), 1.0)

    if emis_col is None:
        # nothing obvious — set zeros for now
        df["Emissions_GWP"] = 0.0
    else:
        # coerce numerics robustly: remove commas and whitespace then to_numeric
        emis_raw = df[emis_col].astype(str).str.replace(r"[,\s]", "", regex=True).replace(["nan", "None"], np.nan)
        emis = pd.to_numeric(emis_raw, errors="coerce").fillna(0.0)
        try:
            df["Emissions_GWP"] = emis * gwp
        except Exception:
            # gwp might be scalar-like
            scalar = float(gwp.iloc[0]) if hasattr(gwp, "iloc") else float(gwp)
            df["Emissions_GWP"] = emis * scalar

    df["Emissions_GWP"] = pd.to_numeric(df["Emissions_GWP"], errors="coerce").fillna(0.0)
    return df


def normalize_pollutant_names_and_gwp(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pollutant name normalization and add a GWP column mapping as in the notebook."""
    df = df.copy()
    if 'Pollutant_Name' in df.columns:
        rename_map = {
            "Nitrous Oxide": "N20",
            "Methane": "CH3",
            "Carbon Dioxide": "CO2",
            "Carbon Dioxide as Carbon": "CO2",
            "Carbon": "CO2",
        }
        df['Pollutant_Name'] = df['Pollutant_Name'].replace(rename_map)
        gwp_map = {"CO2": 1, "CH3": 28, "N20": 265}
        df['GWP'] = df['Pollutant_Name'].map(gwp_map)
    return df


def convert_easting_northing(df: pd.DataFrame) -> pd.DataFrame:
    """If Easting/Northing present, convert to Latitude/Longitude using geopandas if available."""
    df = df.copy()
    if not ({"Easting", "Northing"}.issubset(set(df.columns))):
        return df

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        e = pd.to_numeric(df["Easting"], errors="coerce")
        n = pd.to_numeric(df["Northing"], errors="coerce")
        valid = e.notna() & n.notna()
        if valid.any():
            valid_idx = df.index[valid]
            geom = [Point(x, y) for x, y in zip(e[valid], n[valid])]
            gdf = gpd.GeoDataFrame(df.loc[valid_idx].copy(), geometry=geom, crs="EPSG:27700")
            gdf = gdf.to_crs("EPSG:4326")
            df.loc[valid_idx, "Longitude"] = gdf.geometry.x.values
            df.loc[valid_idx, "Latitude"] = gdf.geometry.y.values
    except Exception:
        # geopandas not available or conversion failed — skip
        pass
    return df


def aggregate_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to site-level similar to the notebook's df_aggregated creation."""
    ignore = ["PollutantID", "Pollutant_Name", "Emissions_GWP", "GWP", "Emission"]
    group_cols = [c for c in df.columns if c not in ignore]
    agg_dict = {"Emissions_GWP": "sum"}
    for col in ["Pollutant_Name", "PollutantID"]:
        if col in df.columns:
            agg_dict[col] = (lambda s, col=col: ",".join(sorted(set(s.dropna().astype(str)))))

    try:
        df_aggregated = df.groupby(group_cols, dropna=False, as_index=False).agg(agg_dict)
        df_aggregated.drop(columns=["PollutantID", "Pollutant_Name", "GWP"], inplace=True, errors='ignore')
        return df_aggregated
    except Exception:
        # fallback
        return df


def detect_columns(df: pd.DataFrame):
    """Return df (possibly modified) and detected year, region, sector, site column names.

    Provides sane fallbacks when columns are missing.
    """
    # Year
    year_candidates = ["Year", "Reporting Year", "Reporting_Year", "Year_of_Emission"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        df["Year"] = df.get("Year", "All")
        year_col = "Year"

    # Region
    region_candidates = ["Region", "Country", "Nation", "Area"]
    region_col = next((c for c in region_candidates if c in df.columns), None)
    if region_col is None:
        df["Region"] = "United Kingdom"
        region_col = "Region"

    # Sector
    sector_candidates = ["Sector", "Activity", "SIC", "SectorName"]
    sector_col = next((c for c in sector_candidates if c in df.columns), None)
    if sector_col is None:
        df["Sector"] = "All"
        sector_col = "Sector"

    # Site identifier
    site_candidates = ["Site", "Site_Name", "Plant", "Plant_Name", "Name"]
    site_col = next((c for c in site_candidates if c in df.columns), None)
    if site_col is None:
        df = df.reset_index().rename(columns={"index": "row_id"})
        site_col = "row_id"
    else:
        df[site_col] = df[site_col].astype(str)

    return df, year_col, region_col, sector_col, site_col


def build_treemap(df: pd.DataFrame, year_col, region_col, sector_col, site_col,
                  year: str | int, region: str, sector: str, threshold: float,
                  prefer: str, log_color: bool):
    """Filter, aggregate to site and return (fig, agg_df).

    fig may be None if no data or if treemap build failed.
    """
    df_base = df.copy()
    if year != "All":
        df_base = df_base[df_base[year_col].astype(str) == str(year)]
    if region != "All":
        df_base = df_base[df_base[region_col] == region]
    if sector != "All":
        df_base = df_base[df_base[sector_col] == sector]

    uk_total = float(df_base["Emissions_GWP"].sum())
    region_totals = df_base.groupby(region_col)["Emissions_GWP"].sum()

    try:
        thr = float(threshold)
    except Exception:
        thr = 0.0
    df_plot = df_base[df_base["Emissions_GWP"].astype(float) >= thr].copy()
    if df_plot.empty:
        return None, df_plot

    agg = (
        df_plot.groupby([region_col, site_col], as_index=False)
        .agg({"Emissions_GWP": "sum", sector_col: lambda s: ", ".join(sorted(set(s.dropna().astype(str))))})
    )

    agg["region_total"] = agg[region_col].map(region_totals).fillna(0.0)
    agg["pct_of_region"] = np.where(agg["region_total"] > 0, agg["Emissions_GWP"] / agg["region_total"] * 100.0, 0.0)
    agg["pct_of_uk"] = np.where(uk_total > 0, agg["Emissions_GWP"] / uk_total * 100.0, 0.0)

    # color scaling
    if log_color:
        with np.errstate(divide="ignore"):
            agg["_color"] = np.log10(agg["Emissions_GWP"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        agg["_color"] = agg["Emissions_GWP"].astype(float)

    # choose sizing metric
    if prefer == "abs" and float(agg["Emissions_GWP"].sum()) > 0:
        size_col = "Emissions_GWP"
        value_label = "Emissions (tCO2e)"
    elif prefer == "pct" and float(agg["pct_of_region"].sum()) > 0:
        size_col = "pct_of_region"
        value_label = "Pct of region"
    else:
        size_col = "Emissions_GWP" if float(agg["Emissions_GWP"].sum()) > 0 else None
        value_label = "Emissions (tCO2e)"

    # ensure positive-sum values for treemap
    epsilon = 1e-9
    if size_col is None:
        agg["_treemap_value"] = 1.0
    else:
        agg["_treemap_value"] = pd.to_numeric(agg[size_col], errors="coerce").fillna(0.0).astype(float)
        if float(agg["_treemap_value"].sum()) <= 0:
            agg["_treemap_value"] = 1.0
    agg["_treemap_value"] = agg["_treemap_value"].where(agg["_treemap_value"] > epsilon, other=epsilon)

    # build treemap with guarded try/except and a simple fallback
    fig = None
    try:
        fig = px.treemap(
            agg,
            path=[region_col, site_col],
            values="_treemap_value",
            color="_color",
            color_continuous_scale="Plasma",
            custom_data=[sector_col, "region_total", "Emissions_GWP", "pct_of_region", "pct_of_uk"],
        )
        fig.update_traces(root_color="lightgray")
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        fig.update_traces(hovertemplate=(
            "%{label}<br>" + value_label + ": %{customdata[2]:.2f}<br>"
            "Sector(s): %{customdata[0]}<br>Region total: %{customdata[1]:.2f}<br>"
            "Pct of region: %{customdata[3]:.2f}%<br>Pct of UK: %{customdata[4]:.4f}%<extra></extra>"
        ))
    except Exception as e:
        st.sidebar.error(f"Treemap rendering failed: {e}")
        # fallback: show a simple site-count treemap to keep UI responsive
        try:
            agg["_fallback_count"] = 1
            fig = px.treemap(agg, path=[region_col, site_col], values="_fallback_count")
        except Exception:
            fig = None

    return fig, agg


def main():
    st.sidebar.header("Data & filters")
    uploaded = st.sidebar.file_uploader("Upload NAEI workbook (optional)", type=["xlsx", "xls"]) 

    df = load_dataframe(uploaded)
    if df is None:
        st.info("Please upload the NAEI workbook or place 'NAEIPointsSource_2023.xlsx' next to this script.")
        return

    # Try to mirror the notebook preprocessing: normalize pollutant names and map GWP where possible
    df = normalize_pollutant_names_and_gwp(df)
    # Convert Easting/Northing to lat/lon if available (optional, uses geopandas)
    df = convert_easting_northing(df)

    # If raw Emission x GWP columns are present, prefer computing Emissions_GWP directly
    if "Emission" in df.columns and "GWP" in df.columns:
        try:
            emis = pd.to_numeric(df["Emission"].astype(str).str.replace(r"[,\s]", "", regex=True), errors="coerce").fillna(0.0)
            gwp = pd.to_numeric(df["GWP"], errors="coerce").fillna(1.0)
            df["Emissions_GWP"] = emis * gwp
        except Exception:
            df = ensure_emissions_gwp(df)
    else:
        df = ensure_emissions_gwp(df)

    # quick diagnostics: if total emissions are zero, surface helpful info and try simple salvage
    total_em = float(df["Emissions_GWP"].sum())
    if total_em == 0.0:
        st.sidebar.warning("Detected total Emissions_GWP == 0. I will attempt additional heuristics and show diagnostics.")
        # show candidate emission-like columns and their sums
        candidate_cols = [c for c in df.columns if "emiss" in c.lower() or "emission" in c.lower() or "tco2" in c.lower()]
        st.sidebar.write("Emission-like columns found:", candidate_cols)
        # numeric column sums
        num_sums = df.select_dtypes(include=[np.number]).sum().sort_values(ascending=False)
        st.sidebar.write("Top numeric columns (sum):")
        for col, val in num_sums.head(8).items():
            st.sidebar.write(f"- {col}: {val}")

        # salvage heuristic: if 'Emission' and 'GWP' exist, compute product
        if "Emission" in df.columns and "GWP" in df.columns:
            try:
                salv_emis = pd.to_numeric(df["Emission"].astype(str).str.replace(r"[,\s]", "", regex=True), errors="coerce").fillna(0.0)
                salv_gwp = pd.to_numeric(df["GWP"], errors="coerce").fillna(1.0)
                df["Emissions_GWP"] = salv_emis * salv_gwp
                total_em = float(df["Emissions_GWP"].sum())
                st.sidebar.success(f"Salvaged Emissions_GWP from 'Emission' x 'GWP'. New total: {total_em:.2f}")
            except Exception as e:
                st.sidebar.error(f"Salvage attempt failed: {e}")
        else:
            st.sidebar.info("No 'Emission' x 'GWP' pair found for salvage. You may need to check column names in your workbook.")

    # Aggregate to site-level using the notebook-like aggregation; then detect columns from aggregated table
    df_aggregated = aggregate_like_notebook(df)
    df, year_col, region_col, sector_col, site_col = detect_columns(df_aggregated)

    # controls
    years = ["All"] + sorted(df[year_col].dropna().astype(str).unique().tolist())
    regions = ["All"] + sorted(df[region_col].dropna().unique().tolist())
    sectors = ["All"] + sorted(df[sector_col].dropna().astype(str).unique().tolist())

    selected_year = st.sidebar.selectbox("Year", years, index=0)
    selected_region = st.sidebar.selectbox("Region", regions, index=0)
    selected_sector = st.sidebar.selectbox("Sector", sectors, index=0)

    max_val = float(df["Emissions_GWP"].max()) if not df["Emissions_GWP"].empty else 1.0
    max_val = max(1.0, max_val)
    threshold = st.sidebar.number_input("Minimum Emissions to include (Emissions_GWP)", min_value=0.0, max_value=float(max_val), value=0.0, step=max(1e-3, max_val/100.0))

    prefer = st.sidebar.selectbox("Treemap sizing", ["abs", "pct"], index=0, format_func=lambda x: "Absolute emissions (tCO2e)" if x=="abs" else "Pct of region")
    log_color = st.sidebar.checkbox("Log color scale (log10)", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh"):
        st.experimental_rerun()

    fig, agg = build_treemap(df, year_col, region_col, sector_col, site_col,
                             selected_year, selected_region, selected_sector,
                             threshold, prefer, log_color)

    st.markdown(f"### Treemap — Year={selected_year}, Region={selected_region}, Sector={selected_sector}")
    if fig is None:
        st.warning("No data to display for this selection (check filters/threshold).")
    else:
        st.plotly_chart(fig, use_container_width=True)

    if agg is not None and len(agg) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Sites shown", f"{len(agg)}")
        col2.metric("Total Emissions (shown)", f"{agg['Emissions_GWP'].sum():.2f} tCO2e")
        col3.metric("Pct of UK (shown)", f"{agg['pct_of_uk'].sum():.2f}%")

    # optional map if lat/lon present
    lat_candidates = ["Latitude", "Lat", "latitude", "lat"]
    lon_candidates = ["Longitude", "Lon", "longitude", "lon"]
    lat_col = next((c for c in df.columns if c in lat_candidates), None)
    lon_col = next((c for c in df.columns if c in lon_candidates), None)
    if lat_col and lon_col:
        if st.checkbox("Show point map (grouped)"):
            df_map = df.dropna(subset=[lat_col, lon_col]).copy()
            df_map = df_map[df_map["Emissions_GWP"].astype(float) >= threshold]
            if df_map.empty:
                st.info("No points to plot on the map with current selection/threshold.")
            else:
                map_fig = px.scatter_mapbox(df_map, lat=lat_col, lon=lon_col, size="Emissions_GWP", color="Emissions_GWP", zoom=5, height=600, color_continuous_scale="Turbo")
                map_fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(map_fig, use_container_width=True)


if __name__ == "__main__":
    main()
