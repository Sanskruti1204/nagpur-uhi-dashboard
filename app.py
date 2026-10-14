from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nagpur UHI Detection & Climate Resilience",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Nagpur_Deployment_RF_Model_Compressed.pkl"

DATEWISE_DIR = BASE_DIR / "datewise_outputs"

DAILY_DATA_PATH = BASE_DIR / (
    "Nagpur_UHI_Daily_Master_2026-03-01_to_2026-05-31.csv"
)

SATELLITE_DATA_PATH = BASE_DIR / (
    "Nagpur_UHI_Satellite_Observations_2026-03-01_to_2026-05-31.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# LOAD DAILY DATA
# ============================================================

daily_data = pd.read_csv(DAILY_DATA_PATH)

daily_data["Date"] = daily_data["Date"].astype(str)


# ============================================================
# LOAD SATELLITE DATA
# ============================================================

satellite_data = pd.read_csv(SATELLITE_DATA_PATH)

satellite_data["Date"] = satellite_data["Date"].astype(str)


# ============================================================
# LOAD AVAILABLE DATE-WISE PREDICTIONS
# ============================================================

prediction_files = sorted(
    DATEWISE_DIR.glob("Predicted_LST_*.csv")
)


if not prediction_files:

    st.error(
        "No date-wise prediction files were found in "
        "the datewise_outputs folder."
    )

    st.stop()


available_dates = []

for file in prediction_files:

    # Example:
    # Predicted_LST_2026_03_08.csv

    date_text = file.stem.replace(
        "Predicted_LST_",
        ""
    )

    date_text = date_text.replace(
        "_",
        "-"
    )

    available_dates.append(date_text)


available_dates = sorted(available_dates)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌡️ Nagpur Urban Heat Island Detection & Climate Resilience System"
)

st.markdown(
    """
    **Satellite-Based LST Prediction • UHI Heat-Zone Mapping •
    Random Forest Analysis**

    This interactive dashboard presents predicted Land Surface
    Temperature, relative urban heat zones, model validation results,
    feature importance, spatial observations, and model-based
    environmental scenarios for Nagpur.
    """
)


# ============================================================
# DATE SELECTOR
# ============================================================

st.header("📅 Date-wise LST Prediction")

selected_date = st.selectbox(
    "Select Prediction Date",
    available_dates,
    index=(
        available_dates.index("2026-05-03")
        if "2026-05-03" in available_dates
        else 0
    )
)


# ============================================================
# LOAD SELECTED DATE
# ============================================================

selected_file = DATEWISE_DIR / (
    "Predicted_LST_"
    + selected_date.replace("-", "_")
    + ".csv"
)


prediction_data = pd.read_csv(selected_file)


# ============================================================
# BASIC DATA CLEANING
# ============================================================

prediction_data["Latitude"] = pd.to_numeric(
    prediction_data["Latitude"],
    errors="coerce"
)

prediction_data["Longitude"] = pd.to_numeric(
    prediction_data["Longitude"],
    errors="coerce"
)

prediction_data["Predicted_LST"] = pd.to_numeric(
    prediction_data["Predicted_LST"],
    errors="coerce"
)

prediction_data = prediction_data.dropna(
    subset=[
        "Latitude",
        "Longitude",
        "Predicted_LST"
    ]
)


# ============================================================
# DATE-SPECIFIC STATISTICS
# ============================================================

min_lst = prediction_data["Predicted_LST"].min()

max_lst = prediction_data["Predicted_LST"].max()

mean_lst = prediction_data["Predicted_LST"].mean()


# ============================================================
# DATE INFORMATION
# ============================================================

selected_daily = daily_data[
    daily_data["Date"] == selected_date
]


if len(selected_daily) > 0:

    selected_t2m = float(
        selected_daily.iloc[0]["T2M"]
    )

    satellite_available = str(
        selected_daily.iloc[0][
            "Satellite_Observation_Available"
        ]
    )

else:

    selected_t2m = (
        float(prediction_data["T2M"].iloc[0])
        if "T2M" in prediction_data.columns
        else np.nan
    )

    satellite_available = "YES"


# ============================================================
# TOP INFORMATION
# ============================================================

st.info(
    f"📅 Prediction date: {pd.Timestamp(selected_date).strftime('%d %B %Y')}  |  "
    "🛰️ Satellite-based environmental analysis  |  "
    "🤖 Random Forest LST model"
)


# ============================================================
# KEY RESULTS
# ============================================================
st.header("📌 Key Results")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Mean Predicted LST",
        f"{mean_lst:.2f} °C"
    )

with col2:
    st.metric(
        "Minimum Predicted LST",
        f"{min_lst:.2f} °C"
    )

with col3:
    st.metric(
        "Maximum Predicted LST",
        f"{max_lst:.2f} °C"
    )

with col4:
    st.metric(
        "Air Temperature",
        f"{selected_t2m:.2f} °C"
    )

with col5:
    st.metric(
        "Prediction Points",
        f"{len(prediction_data):,}"
    )
# ============================================================
# PREDICTED LST MAP
# ============================================================
st.header("🗺️ Predicted Land Surface Temperature")

st.subheader(
    "Predicted LST — "
    + pd.Timestamp(selected_date).strftime("%d %B %Y")
)

# ------------------------------------------------------------
# MAP DATA
# Full dataset is used for calculations.
# Only 3,000 points are displayed for performance.
# ------------------------------------------------------------

MAX_MAP_POINTS = 3000

if len(prediction_data) > MAX_MAP_POINTS:
    map_data = prediction_data.sample(
        n=MAX_MAP_POINTS,
        random_state=42
    )
else:
    map_data = prediction_data.copy()

# ------------------------------------------------------------
# GEOGRAPHIC MAP — PREDICTED LST WITH COLOR LEGEND
# ------------------------------------------------------------

fig_lst = px.scatter_map(
    map_data,
    lat="Latitude",
    lon="Longitude",
    color="Predicted_LST",

    color_continuous_scale=[
        [0.0, "#30123B"],    # Lower temperature
        [0.25, "#4666D5"],   # Cool
        [0.5, "#F9C846"],    # Warm
        [0.75, "#F66B19"],   # Hot
        [1.0, "#B40426"]     # Highest temperature
    ],

    range_color=[
        map_data["Predicted_LST"].min(),
        map_data["Predicted_LST"].max()
    ],

    zoom=9.5,

    center={
        "lat": 21.145,
        "lon": 79.085
    },

    map_style="open-street-map",

    hover_name=None,

    hover_data={
        "Latitude": ":.5f",
        "Longitude": ":.5f",
        "Predicted_LST": ":.2f"
    },

    labels={
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Predicted_LST": "Land Surface Temperature (°C)"
    }
)

fig_lst.update_coloraxes(
    colorbar_title="Predicted LST (°C)"
)


# ------------------------------------------------------------
# PREDICTION POINT STYLE
# ------------------------------------------------------------

fig_lst.update_traces(
    marker=dict(
        size=6,
        opacity=0.72
    )
)

# ------------------------------------------------------------
# NAGPUR / SURROUNDING LANDMARK LABELS
# Coordinates are based on published geographic data.
# ------------------------------------------------------------

place_names = [
    "Nagpur",
    "Mominpura",
    "Mahal",
    "Seminary Hills",
    "Hingna",
    "Kapsi Khurd",
    "Butibori",
    "Nagpur Airport"
]

place_lat = [
    21.164448,
    21.155722,
    21.142113,
    21.168524,
    21.073549,
    21.144740,
    20.928190,
    21.081190
]

place_lon = [
    79.115337,
    79.094162,
    79.111102,
    79.062541,
    78.961349,
    79.181230,
    79.007080,
    79.056031
]

# Add geographic labels on top of the heat map
fig_lst.add_scattermap(
    lat=place_lat,
    lon=place_lon,
    mode="markers+text",
    text=place_names,
    textposition="top center",
    marker=dict(
        size=8
    ),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Latitude: %{lat:.5f}<br>"
        "Longitude: %{lon:.5f}"
        "<extra></extra>"
    ),
    name="Key locations",
    showlegend=True
)

# ------------------------------------------------------------
# STUDY AREA EXTENT
# Your Google Earth Engine study rectangle:
# Longitude: 78.85–79.33
# Latitude: 20.92–21.37
# ------------------------------------------------------------

study_lat = [
    20.92,
    20.92,
    21.37,
    21.37,
    20.92
]

study_lon = [
    78.85,
    79.33,
    79.33,
    78.85,
    78.85
]

fig_lst.add_scattermap(
    lat=study_lat,
    lon=study_lon,
    mode="lines",
    line=dict(
        width=2
    ),
    name="Study area extent",
    hoverinfo="skip",
    showlegend=True
)

# ------------------------------------------------------------
# MAP LAYOUT
# ------------------------------------------------------------

fig_lst.update_layout(
    height=650,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    coloraxis_colorbar=dict(
        title="LST (°C)"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    )
)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

st.plotly_chart(
    fig_lst,
    width="stretch"
)

st.caption(
    f"Displaying {len(map_data):,} representative prediction points "
    f"from {len(prediction_data):,} total observations. "
    "The map includes geographic context, key Nagpur-area locations, "
    "and the extent of the satellite study region."
)
# ------------------------------------------------------------
# PREDICTED LST COLOR LEGEND
# ------------------------------------------------------------

st.markdown("### 🌡️ Predicted LST — Colour Meaning")

st.markdown("""
- 🟣 **Purple / Dark Blue:** Relatively lower surface temperature
- 🟡 **Yellow:** Moderate surface temperature
- 🟠 **Orange:** High surface temperature
- 🔴 **Red:** Relatively higher surface temperature

*Refer to the colour bar beside the map for actual temperatures in °C.
Colour meanings are relative to the displayed temperature range.*
""")

# ============================================================
# UHI HEAT ZONES
# ============================================================

st.header("🔥 UHI Heat Zones")

st.subheader(
    "Relative Heat-Zone Classification — "
    + pd.Timestamp(selected_date).strftime("%d %B %Y")
)

# ------------------------------------------------------------
# CALCULATE HEAT ZONES USING ALL PREDICTION POINTS
# ------------------------------------------------------------

q80 = prediction_data["Predicted_LST"].quantile(0.80)
q90 = prediction_data["Predicted_LST"].quantile(0.90)


def classify_heat_zone(value):

    if value >= q90:
        return 3
    elif value >= q80:
        return 2
    else:
        return 1


prediction_data["Heat_Zone"] = (
    prediction_data["Predicted_LST"]
    .apply(classify_heat_zone)
)

prediction_data["Heat_Zone_Label"] = (
    prediction_data["Heat_Zone"]
    .map({
        1: "Low",
        2: "Moderate",
        3: "High"
    })
)

# ------------------------------------------------------------
# FAST MAP DISPLAY
# Calculations use ALL points.
# Only representative points are displayed.
# ------------------------------------------------------------

if len(prediction_data) > MAX_MAP_POINTS:

    zone_map_data = prediction_data.sample(
        n=MAX_MAP_POINTS,
        random_state=42
    )

else:

    zone_map_data = prediction_data.copy()


# ------------------------------------------------------------
# GEOGRAPHIC HEAT-ZONE MAP
# ------------------------------------------------------------

fig_zones = px.scatter_map(
    zone_map_data,
    lat="Latitude",
    lon="Longitude",
    color="Heat_Zone_Label",

    category_orders={
        "Heat_Zone_Label": [
            "Low",
            "Moderate",
            "High"
        ]
    },

    color_discrete_map={
        "Low": "green",
        "Moderate": "orange",
        "High": "red"
    },

    zoom=9.5,

    center={
        "lat": 21.145,
        "lon": 79.085
    },

    map_style="open-street-map",

    hover_data={
        "Latitude": ":.5f",
        "Longitude": ":.5f",
        "Predicted_LST": ":.2f",
        "Heat_Zone_Label": True
    },

    labels={
        "Predicted_LST": "Predicted LST (°C)",
        "Heat_Zone_Label": "Heat Zone"
    }
)


# ------------------------------------------------------------
# HEAT-POINT STYLE
# ------------------------------------------------------------

fig_zones.update_traces(
    marker=dict(
        size=7,
        opacity=0.72
    )
)


# ------------------------------------------------------------
# NAGPUR / SURROUNDING LANDMARK LABELS
# ------------------------------------------------------------

place_names = [
    "Nagpur",
    "Mominpura",
    "Mahal",
    "Seminary Hills",
    "Hingna",
    "Kapsi Khurd",
    "Butibori",
    "Nagpur Airport"
]

place_lat = [
    21.164448,
    21.155722,
    21.142113,
    21.168524,
    21.073549,
    21.144740,
    20.928190,
    21.081190
]

place_lon = [
    79.115337,
    79.094162,
    79.111102,
    79.062541,
    78.961349,
    79.181230,
    79.007080,
    79.056031
]


fig_zones.add_scattermap(
    lat=place_lat,
    lon=place_lon,
    mode="markers+text",

    text=place_names,

    textposition="top center",

    marker=dict(
        size=8
    ),

    hovertemplate=(
        "<b>%{text}</b><br>"
        "Latitude: %{lat:.5f}<br>"
        "Longitude: %{lon:.5f}"
        "<extra></extra>"
    ),

    name="Key locations",

    showlegend=True
)


# ------------------------------------------------------------
# STUDY AREA EXTENT
# Google Earth Engine study rectangle
# Longitude: 78.85–79.33
# Latitude: 20.92–21.37
# ------------------------------------------------------------

study_lat = [
    20.92,
    20.92,
    21.37,
    21.37,
    20.92
]

study_lon = [
    78.85,
    79.33,
    79.33,
    78.85,
    78.85
]


fig_zones.add_scattermap(
    lat=study_lat,
    lon=study_lon,

    mode="lines",

    line=dict(
        width=2
    ),

    name="Study area extent",

    hoverinfo="skip",

    showlegend=True
)


# ------------------------------------------------------------
# MAP LAYOUT
# ------------------------------------------------------------

fig_zones.update_layout(

    height=650,

    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    )
)


# ------------------------------------------------------------
# DISPLAY MAP
# ------------------------------------------------------------

st.plotly_chart(
    fig_zones,
    width="stretch"
)


# ------------------------------------------------------------
# MAP INFORMATION
# ------------------------------------------------------------

st.caption(
    f"Heat zones are calculated from all "
    f"{len(prediction_data):,} prediction points. "
    f"The map displays {len(zone_map_data):,} representative points "
    f"for faster interactive performance."
)


st.info(
    f"Relative heat zones are based on the selected-date "
    f"predicted LST distribution. "
    f"Low: below the 80th percentile ({q80:.2f} °C), "
    f"Moderate: 80th–90th percentile, "
    f"High: at or above the 90th percentile ({q90:.2f} °C)."
)
# ============================================================
# ENVIRONMENTAL INTELLIGENCE
# ============================================================

st.header("🌳 Environmental Intelligence")

st.write(
    "Explore how vegetation and built-up characteristics "
    "are spatially associated with predicted land surface temperature."
)

# ------------------------------------------------------------
# USE SAME REPRESENTATIVE SAMPLE FOR FAST MAP DISPLAY
# ------------------------------------------------------------

if len(prediction_data) > MAX_MAP_POINTS:

    environment_map_data = prediction_data.sample(
        n=MAX_MAP_POINTS,
        random_state=42
    )

else:

    environment_map_data = prediction_data.copy()


# ------------------------------------------------------------
# NDVI MAP — VEGETATION
# ------------------------------------------------------------

st.subheader("🌿 Vegetation Index (NDVI)")

fig_ndvi = px.scatter_map(
    environment_map_data,
    lat="Latitude",
    lon="Longitude",
    color="NDVI",

    color_continuous_scale="YlGn",

    zoom=9.5,

    center={
        "lat": 21.145,
        "lon": 79.085
    },

    map_style="open-street-map",

    hover_data={
        "Latitude": ":.5f",
        "Longitude": ":.5f",
        "NDVI": ":.3f",
        "Predicted_LST": ":.2f"
    },

    labels={
        "NDVI": "NDVI",
        "Predicted_LST": "Predicted LST (°C)"
    }
)

fig_ndvi.update_traces(
    marker=dict(
        size=6,
        opacity=0.72
    )
)

fig_ndvi.update_layout(
    height=600,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    coloraxis_colorbar=dict(
        title="NDVI"
    )
)

st.plotly_chart(
    fig_ndvi,
    width="stretch"
)

st.caption(
    "NDVI represents vegetation/greenness. Higher NDVI generally "
    "indicates greater vegetation presence, while lower values "
    "represent less vegetated surfaces."
)


# ------------------------------------------------------------
# NDBI MAP — BUILT-UP INTENSITY
# ------------------------------------------------------------

st.subheader("🏙️ Built-up Index (NDBI)")

fig_ndbi = px.scatter_map(
    environment_map_data,
    lat="Latitude",
    lon="Longitude",
    color="NDBI",

    color_continuous_scale="OrRd",

    zoom=9.5,

    center={
        "lat": 21.145,
        "lon": 79.085
    },

    map_style="open-street-map",

    hover_data={
        "Latitude": ":.5f",
        "Longitude": ":.5f",
        "NDBI": ":.3f",
        "Predicted_LST": ":.2f"
    },

    labels={
        "NDBI": "NDBI",
        "Predicted_LST": "Predicted LST (°C)"
    }
)

fig_ndbi.update_traces(
    marker=dict(
        size=6,
        opacity=0.72
    )
)

fig_ndbi.update_layout(
    height=600,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    coloraxis_colorbar=dict(
        title="NDBI"
    )
)

st.plotly_chart(
    fig_ndbi,
    width="stretch"
)

st.caption(
    "NDBI represents built-up surface characteristics. "
    "Higher values indicate relatively stronger built-up signatures."
)


# ------------------------------------------------------------
# ENVIRONMENTAL SUMMARY
# ------------------------------------------------------------

st.subheader("🔬 Environmental Summary")

mean_lst = prediction_data["Predicted_LST"].mean()
mean_ndvi = prediction_data["NDVI"].mean()
mean_ndbi = prediction_data["NDBI"].mean()

lst_ndvi_corr = prediction_data[
    ["Predicted_LST", "NDVI"]
].corr().iloc[0, 1]

lst_ndbi_corr = prediction_data[
    ["Predicted_LST", "NDBI"]
].corr().iloc[0, 1]


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Mean Predicted LST",
        f"{mean_lst:.2f} °C"
    )

with col2:

    st.metric(
        "Mean NDVI",
        f"{mean_ndvi:.3f}"
    )

with col3:

    st.metric(
        "Mean NDBI",
        f"{mean_ndbi:.3f}"
    )


# ------------------------------------------------------------
# ASSOCIATION WITH TEMPERATURE
# ------------------------------------------------------------

st.subheader("🌡️ Heat–Environment Association")

corr_col1, corr_col2 = st.columns(2)

with corr_col1:

    st.metric(
        "LST ↔ NDVI Correlation",
        f"{lst_ndvi_corr:.3f}"
    )

with corr_col2:

    st.metric(
        "LST ↔ NDBI Correlation",
        f"{lst_ndbi_corr:.3f}"
    )


st.info(
    "Correlation values describe spatial association between "
    "predicted LST and environmental indices for the selected date. "
    "They should not be interpreted as proof of causation."
)
# ------------------------------------------------------------
# ENVIRONMENTAL RELATIONSHIP SCATTER PLOTS
# ------------------------------------------------------------

st.subheader("📊 Heat–Environment Relationships")

scatter_col1, scatter_col2 = st.columns(2)


# ------------------------------------------------------------
# LST vs NDVI
# ------------------------------------------------------------

with scatter_col1:

    fig_lst_ndvi = px.scatter(
        environment_map_data,
        x="NDVI",
        y="Predicted_LST",
        trendline="ols",
        opacity=0.45,
        labels={
            "NDVI": "NDVI",
            "Predicted_LST": "Predicted LST (°C)"
        },
        title="Predicted LST vs NDVI"
    )

    fig_lst_ndvi.update_layout(
        height=450,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    st.plotly_chart(
        fig_lst_ndvi,
        width="stretch"
    )

    st.caption(
        f"Spatial association: LST ↔ NDVI = {lst_ndvi_corr:.3f}"
    )


# ------------------------------------------------------------
# LST vs NDBI
# ------------------------------------------------------------

with scatter_col2:

    fig_lst_ndbi = px.scatter(
        environment_map_data,
        x="NDBI",
        y="Predicted_LST",
        trendline="ols",
        opacity=0.45,
        labels={
            "NDBI": "NDBI",
            "Predicted_LST": "Predicted LST (°C)"
        },
        title="Predicted LST vs NDBI"
    )

    fig_lst_ndbi.update_layout(
        height=450,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    st.plotly_chart(
        fig_lst_ndbi,
        width="stretch"
    )

    st.caption(
        f"Spatial association: LST ↔ NDBI = {lst_ndbi_corr:.3f}"
    )


st.info(
    "The scatter plots show the spatial relationship between "
    "predicted land surface temperature and environmental indices "
    "for the selected date. The trend lines summarize the overall "
    "linear pattern and do not imply causation."
)

st.caption(
    f"Environmental maps display {len(environment_map_data):,} "
    f"representative points from {len(prediction_data):,} "
    "available prediction observations."
)
# ============================================================
# CLIMATE RESILIENCE PRIORITY
# ============================================================

st.header("🌡️ Climate Resilience Priority")

st.write(
    "Identify locations that may deserve greater planning attention "
    "based on the combined spatial patterns of heat, built-up intensity, "
    "and vegetation."
)

st.info(
    "This is a planning-oriented priority indicator derived from "
    "predicted LST, NDBI, and NDVI for the selected date. "
    "It is not a validated public-health vulnerability or risk index."
)

st.caption(
    "Priority classes are determined using the selected-date "
    "priority-score distribution: the highest 20% are Very High, "
    "the next 20% High, the next 20% Moderate, and the lowest 40% "
    "Lower priority."
)


# ------------------------------------------------------------
# CREATE NORMALIZED ENVIRONMENTAL INDICATORS
# ------------------------------------------------------------

resilience_data = prediction_data.copy()


def min_max_normalize(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


# Heat intensity
resilience_data["Heat_Score"] = min_max_normalize(
    resilience_data["Predicted_LST"]
)


# Built-up intensity
resilience_data["Builtup_Score"] = min_max_normalize(
    resilience_data["NDBI"]
)


# Low vegetation score
# Higher value = relatively less vegetation
resilience_data["Low_Vegetation_Score"] = (
    1 - min_max_normalize(
        resilience_data["NDVI"]
    )
)


# ------------------------------------------------------------
# COMBINED PRIORITY SCORE
# ------------------------------------------------------------

resilience_data["Priority_Score"] = (
    0.50 * resilience_data["Heat_Score"]
    + 0.30 * resilience_data["Builtup_Score"]
    + 0.20 * resilience_data["Low_Vegetation_Score"]
)


# Convert to 0–100 scale
resilience_data["Priority_Score_100"] = (
    resilience_data["Priority_Score"] * 100
)


# ------------------------------------------------------------
# PRIORITY CLASSIFICATION
# ------------------------------------------------------------

# ------------------------------------------------------------
# RELATIVE PRIORITY CLASSIFICATION
# ------------------------------------------------------------

priority_q40 = resilience_data["Priority_Score_100"].quantile(0.40)
priority_q60 = resilience_data["Priority_Score_100"].quantile(0.60)
priority_q80 = resilience_data["Priority_Score_100"].quantile(0.80)


def classify_priority(score):

    if score >= priority_q80:
        return "Very High"

    elif score >= priority_q60:
        return "High"

    elif score >= priority_q40:
        return "Moderate"

    else:
        return "Lower"


resilience_data["Priority_Class"] = (
    resilience_data["Priority_Score_100"]
    .apply(classify_priority)
)

# ------------------------------------------------------------
# FAST MAP DISPLAY
# ------------------------------------------------------------

if len(resilience_data) > MAX_MAP_POINTS:

    resilience_map_data = resilience_data.sample(
        n=MAX_MAP_POINTS,
        random_state=42
    )

else:

    resilience_map_data = resilience_data.copy()


# ------------------------------------------------------------
# PRIORITY MAP
# ------------------------------------------------------------

st.subheader("🗺️ Climate Resilience Priority Map")

fig_resilience = px.scatter_map(
    resilience_map_data,
    lat="Latitude",
    lon="Longitude",
    color="Priority_Class",

    category_orders={
        "Priority_Class": [
            "Lower",
            "Moderate",
            "High",
            "Very High"
        ]
    },

    color_discrete_map={
        "Lower": "green",
        "Moderate": "yellow",
        "High": "orange",
        "Very High": "red"
    },

    zoom=9.5,

    center={
        "lat": 21.145,
        "lon": 79.085
    },

    map_style="open-street-map",

    hover_data={
        "Latitude": ":.5f",
        "Longitude": ":.5f",
        "Predicted_LST": ":.2f",
        "NDVI": ":.3f",
        "NDBI": ":.3f",
        "Priority_Score_100": ":.1f",
        "Priority_Class": True
    },

    labels={
        "Predicted_LST": "Predicted LST (°C)",
        "NDVI": "NDVI",
        "NDBI": "NDBI",
        "Priority_Score_100": "Priority Score"
    }
)


fig_resilience.update_traces(
    marker=dict(
        size=7,
        opacity=0.72
    )
)
# ------------------------------------------------------------
# KEY LOCATION LABELS
# ------------------------------------------------------------

fig_resilience.add_scattermap(
    lat=place_lat,
    lon=place_lon,
    mode="markers+text",
    text=place_names,
    textposition="top center",

    marker=dict(
        size=8
    ),

    hovertemplate=(
        "<b>%{text}</b><br>"
        "Latitude: %{lat:.5f}<br>"
        "Longitude: %{lon:.5f}"
        "<extra></extra>"
    ),

    name="Key locations",
    showlegend=True
)

# ------------------------------------------------------------
# STUDY AREA BOUNDARY
# ------------------------------------------------------------

fig_resilience.add_scattermap(
    lat=study_lat,
    lon=study_lon,
    mode="lines",

    line=dict(
        width=2
    ),

    name="Study area boundary",
    showlegend=True
)

fig_resilience.update_layout(
    height=650,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    )
)


st.plotly_chart(
    fig_resilience,
    width="stretch"
)


st.caption(
    f"The map displays {len(resilience_map_data):,} representative "
    f"points from {len(resilience_data):,} available observations."
)


# ------------------------------------------------------------
# PRIORITY SUMMARY
# ------------------------------------------------------------

st.subheader("📊 Priority Summary")

priority_counts = (
    resilience_data["Priority_Class"]
    .value_counts()
    .reindex(
        [
            "Very High",
            "High",
            "Moderate",
            "Lower"
        ],
        fill_value=0
    )
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Very High",
        f"{priority_counts['Very High']:,}",
        f"{priority_counts['Very High'] / len(resilience_data) * 100:.1f}%"
    )

with col2:
    st.metric(
        "High",
        f"{priority_counts['High']:,}",
        f"{priority_counts['High'] / len(resilience_data) * 100:.1f}%"
    )

with col3:
    st.metric(
        "Moderate",
        f"{priority_counts['Moderate']:,}",
        f"{priority_counts['Moderate'] / len(resilience_data) * 100:.1f}%"
    )

with col4:
    st.metric(
        "Lower",
        f"{priority_counts['Lower']:,}",
        f"{priority_counts['Lower'] / len(resilience_data) * 100:.1f}%"
    )
# ------------------------------------------------------------
# PLANNING INTERPRETATION
# ------------------------------------------------------------

st.subheader("🌱 Planning Interpretation")

st.write(
    "Areas classified as Very High or High priority show a stronger "
    "combination of predicted heat, built-up intensity, and relatively "
    "lower vegetation within the selected-date dataset."
)


st.markdown(
    """
**Potential planning responses**

- 🌳 Increase or protect urban vegetation and tree cover
- 🏙️ Prioritize heat-sensitive built-up areas for cooling interventions
- 🏞️ Consider shaded public spaces and green infrastructure
- 🚶 Improve thermal comfort along frequently used pedestrian areas
- 🏠 Consider cool-roof and heat-reduction strategies in suitable locations
- 💧 Explore water-sensitive and heat-resilient urban design where appropriate
"""
)


st.caption(
    "Priority classes are relative to the selected-date dataset and "
    "should support, rather than replace, detailed field surveys, "
    "local planning data, and public-health assessments."
)
# ============================================================
# DATE COMPARISON / HEAT TREND
# ============================================================

st.header("📅 Date Comparison & Heat Trend")

st.write(
    "Explore how predicted land surface temperature changed "
    "across the available observation dates from March to May 2026."
)


# ------------------------------------------------------------
# CALCULATE DATE-WISE TEMPERATURE STATISTICS
# Read every available prediction file, not just the selected date
# ------------------------------------------------------------

trend_rows = []

for trend_file in prediction_files:
    date_text = (
        trend_file.stem
        .replace("Predicted_LST_", "")
        .replace("_", "-")
    )

    trend_df = pd.read_csv(trend_file)

    if "Predicted_LST" not in trend_df.columns:
        continue

    lst_values = pd.to_numeric(
        trend_df["Predicted_LST"],
        errors="coerce"
    ).dropna()

    if not lst_values.empty:
        trend_rows.append({
            "Date": pd.to_datetime(date_text),
            "Mean": lst_values.mean(),
            "Minimum": lst_values.min(),
            "Maximum": lst_values.max()
        })

date_trend = pd.DataFrame(trend_rows)

if not date_trend.empty:
    date_trend = date_trend.sort_values("Date")

    # --------------------------------------------------------
    # LST TREND CHART
    # --------------------------------------------------------

    st.subheader("🌡️ Model-estimated LST Trend")

    fig_trend = px.line(
        date_trend,
        x="Date",
        y=["Mean", "Minimum", "Maximum"],
        markers=True,
        labels={
            "Date": "Date",
            "value": "Model-estimated LST (°C)",
            "variable": "Temperature Statistic"
        },
        title="Land Surface Temperature Across Observation Dates"
    )

    fig_trend.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title="Statistic",
        xaxis_title="Date",
        yaxis_title="Model-estimated LST (°C)"
    )

    fig_trend.update_xaxes(
    tickmode="array",
    tickvals=date_trend["Date"],
    ticktext=date_trend["Date"].dt.strftime("%d %b"),
    tickangle=-45,
    title_text="Date"
)

    st.plotly_chart(
        fig_trend,
        width="stretch"
    )

    st.caption(
        f"Showing {len(date_trend)} available observation dates. "
        "Minimum, mean, and maximum values are calculated from "
        "each date's complete prediction file."
    )

    st.info(
        "This chart compares model-estimated land surface "
        "temperatures across available observation dates. "
        "It does not represent continuous daily satellite observations."
    )
else:
    st.warning("No valid LST values were found in the prediction files.")

# ============================================================
# HOTSPOT PERSISTENCE
# ============================================================

st.header("🔥 Hotspot Persistence")

st.write(
    "Identify locations that repeatedly experience relatively "
    "high predicted land surface temperature across multiple "
    "observation dates."
)

st.info(
    "Hotspot persistence is a relative indicator based on the "
    "top 10% of predicted LST values on each observation date. "
    "It shows locations that repeatedly appear among the hottest "
    "areas within the study dataset."
)

# ------------------------------------------------------------
# LOAD ALL DATE-WISE PREDICTION FILES
# ------------------------------------------------------------

persistence_rows = []

for file in sorted(
    DATEWISE_DIR.glob("Predicted_LST_*.csv")
):

    try:

        date_text = (
            file.stem
            .replace("Predicted_LST_", "")
            .replace("_", "-")
        )

        temp_df = pd.read_csv(file)

        # Round coordinates to create approximate spatial cells
        temp_df["Lat_Grid"] = temp_df["Latitude"].round(3)
        temp_df["Lon_Grid"] = temp_df["Longitude"].round(3)

        threshold = temp_df["Predicted_LST"].quantile(0.90)

        temp_df["Hotspot"] = (
            temp_df["Predicted_LST"] >= threshold
        )

        hot_points = temp_df[
            temp_df["Hotspot"]
        ][
            [
                "Lat_Grid",
                "Lon_Grid"
            ]
        ].copy()

        hot_points["Date"] = pd.to_datetime(date_text)

        persistence_rows.append(hot_points)

    except Exception:
        continue


# ------------------------------------------------------------
# COMBINE ALL HOTSPOT OBSERVATIONS
# ------------------------------------------------------------

if persistence_rows:

    hotspot_history = pd.concat(
        persistence_rows,
        ignore_index=True
    )

    total_dates = len(persistence_rows)

    persistence_data = (
        hotspot_history
        .groupby(
            [
                "Lat_Grid",
                "Lon_Grid"
            ]
        )
        .agg(
            Hot_Count=("Date", "nunique")
        )
        .reset_index()
    )

    persistence_data["Date_Count"] = total_dates

    # --------------------------------------------------------
    # PERSISTENCE CLASSIFICATION
    # --------------------------------------------------------

    def classify_persistence(count):

        if count >= 9:
            return "Very High"

        elif count >= 6:
            return "High"

        elif count >= 3:
            return "Moderate"

        else:
            return "Low"


    persistence_data["Persistence_Class"] = (
        persistence_data["Hot_Count"]
        .apply(classify_persistence)
    )

    # --------------------------------------------------------
    # PERSISTENCE SUMMARY
    # --------------------------------------------------------

    st.subheader("📊 Hotspot Persistence Summary")

    persistence_counts = (
        persistence_data["Persistence_Class"]
        .value_counts()
        .reindex(
            [
                "Very High",
                "High",
                "Moderate",
                "Low"
            ],
            fill_value=0
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Very High",
            f"{persistence_counts['Very High']:,}"
        )

    with col2:
        st.metric(
            "High",
            f"{persistence_counts['High']:,}"
        )

    with col3:
        st.metric(
            "Moderate",
            f"{persistence_counts['Moderate']:,}"
        )

    with col4:
        st.metric(
            "Low",
            f"{persistence_counts['Low']:,}"
        )

    st.caption(
        f"Persistence was evaluated across "
        f"{total_dates} available observation dates. "
        f"A hotspot is defined independently for each date "
        f"using that date's 90th-percentile predicted LST."
    )
# --------------------------------------------------------
# TOP PERSISTENT HOTSPOTS
# --------------------------------------------------------

st.subheader("🔥 Top Persistent Hotspots")

# Create reference table for named key locations
location_reference = pd.DataFrame({
    "Location": place_names,
    "Latitude": place_lat,
    "Longitude": place_lon
})

# Function to calculate approximate distance in km
def calculate_distance_km(lat1, lon1, lat2, lon2):
    lat_distance = (lat2 - lat1) * 111
    lon_distance = (
        (lon2 - lon1)
        * 111
        * np.cos(np.radians((lat1 + lat2) / 2))
    )

    return np.sqrt(
        lat_distance ** 2 +
        lon_distance ** 2
    )


top_hotspots = (
    persistence_data
    .sort_values(
        ["Hot_Count", "Lat_Grid", "Lon_Grid"],
        ascending=[False, True, True]
    )
    .head(10)
    .copy()
)

top_hotspots.insert(
    0,
    "Rank",
    range(1, len(top_hotspots) + 1)
)

# Find nearest named location for every hotspot
nearest_locations = []
nearest_distances = []

for _, hotspot in top_hotspots.iterrows():

    distances = location_reference.apply(
        lambda location: calculate_distance_km(
            hotspot["Lat_Grid"],
            hotspot["Lon_Grid"],
            location["Latitude"],
            location["Longitude"]
        ),
        axis=1
    )

    nearest_index = distances.idxmin()

    nearest_locations.append(
        location_reference.loc[
            nearest_index,
            "Location"
        ]
    )

    nearest_distances.append(
        distances.loc[nearest_index]
    )

top_hotspots["Nearest Key Location"] = nearest_locations

top_hotspots["Distance (km)"] = [
    round(distance, 2)
    for distance in nearest_distances
]

top_hotspots["Hotspot Dates"] = (
    top_hotspots["Hot_Count"].astype(str)
    + " / "
    + top_hotspots["Date_Count"].astype(str)
)

top_hotspots = top_hotspots[
    [
        "Rank",
        "Lat_Grid",
        "Lon_Grid",
        "Nearest Key Location",
        "Distance (km)",
        "Hotspot Dates",
        "Persistence_Class"
    ]
]

top_hotspots.columns = [
    "Rank",
    "Latitude",
    "Longitude",
    "Nearest Key Location",
    "Distance (km)",
    "Hotspot Dates",
    "Persistence"
]

st.dataframe(
    top_hotspots,
    width="stretch",
    hide_index=True
)

st.caption(
    "The table lists the ten spatial cells that most frequently "
    "appeared within the hottest 10% of predicted LST across "
    "the available observation dates. The nearest key location "
    "is provided for geographic interpretation; hotspot cells "
    "do not represent the exact boundaries of named locations."
)
# --------------------------------------------------------
# PREPARE MAP DATA
# --------------------------------------------------------

persistence_map_data = persistence_data.copy()
if len(persistence_map_data) > MAX_MAP_POINTS:

    persistence_map_data = (
        persistence_map_data.sample(
            n=MAX_MAP_POINTS,
            random_state=42
        )
    )

# --------------------------------------------------------
# PREPARE MAP COORDINATES
# --------------------------------------------------------

persistence_map_data["Latitude"] = (
    persistence_map_data["Lat_Grid"]
)

persistence_map_data["Longitude"] = (
    persistence_map_data["Lon_Grid"]
)

# --------------------------------------------------------
# PERSISTENCE MAP
# --------------------------------------------------------

st.subheader("🗺️ Repeated Heat Hotspots")

fig_persistence = px.scatter_map(
    persistence_map_data,
    lat="Latitude",
    lon="Longitude",
    color="Hot_Count",
    size="Hot_Count",
    hover_data={
        "Latitude": ":.3f",
        "Longitude": ":.3f",
        "Hot_Count": True,
        "Persistence_Class": True
    },
    color_continuous_scale="Inferno",
    zoom=9.5,
    center={
        "lat": 21.145,
        "lon": 79.085
    },
    map_style="open-street-map",
    title="Hotspot Persistence Across Observation Dates"
)

fig_persistence.update_traces(
    marker=dict(
        opacity=0.75
    )
)

fig_persistence.add_scattermap(
    lat=place_lat,
    lon=place_lon,
    mode="markers+text",
    text=place_names,
    textposition="top center",
    marker=dict(size=8),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Latitude: %{lat:.5f}<br>"
        "Longitude: %{lon:.5f}"
        "<extra></extra>"
    ),
    name="Key locations",
    showlegend=True
)

fig_persistence.add_scattermap(
    lat=study_lat,
    lon=study_lon,
    mode="lines",
    line=dict(width=2),
    name="Study area boundary",
    showlegend=True
)

fig_persistence.update_layout(
    height=650,
    margin=dict(
        l=10,
        r=10,
        t=60,
        b=10
    ),
    coloraxis_colorbar=dict(
        title="Hotspot<br>Count"
    )
)

st.plotly_chart(
    fig_persistence,
    width="stretch"
)

st.caption(
    f"The map displays {len(persistence_map_data):,} "
    f"representative spatial cells. Higher hotspot counts "
    f"indicate locations that repeatedly appeared within "
    f"the hottest 10% of predicted LST on available dates."
)

st.info(
    "Hotspot persistence is a relative spatial indicator, "
    "not an official heatwave classification or a direct "
    "measure of human thermal exposure. Differences in "
    "satellite observation coverage should also be considered."
)
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🤖 Machine Learning Model")

st.write(
    "Random Forest regression was used to predict land surface "
    "temperature from environmental and spatial features. "
    "Performance was evaluated using spatial-block validation."
)
col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "R² Score",
        "0.8741"
    )


with col2:

    st.metric(
        "RMSE",
        "1.878 °C"
    )


with col3:

    st.metric(
        "MAE",
        "1.391 °C"
    )


st.caption(
    "Validation used spatial blocks to reduce the effect of "
    "spatially nearby training and testing observations."
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

st.subheader("🌳 Feature Importance")

feature_names = [
    "Day of Year",
    "NDBI",
    "Longitude",
    "Latitude",
    "NDVI",
    "T2M"
]

feature_importance = [
    0.530034,
    0.168822,
    0.097664,
    0.084592,
    0.060547,
    0.058341
]


fig_importance = px.bar(
    x=feature_importance,
    y=feature_names,
    orientation="h",
    labels={
        "x": "Feature Importance",
        "y": "Feature"
    },
    title="Random Forest Feature Importance"
)


fig_importance.update_layout(
    height=450,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)


st.plotly_chart(
    fig_importance,
    width="stretch"
)


st.caption(
    "Feature importance values represent the relative contribution "
    "of each input feature according to the Random Forest model."
)
def parse_dates(series):
    s = series.astype(str).str.strip().str[:10]
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    return iso.fillna(dmy)


obs = (
    satellite_data
    .assign(Obs_Date=parse_dates(satellite_data["Date"]))
    .dropna(subset=["Obs_Date"])
    .groupby("Obs_Date", as_index=False)
    .size()
    .rename(columns={"size": "Pixels"})
    .sort_values("Obs_Date")
)

gap_days = obs["Obs_Date"].diff().dt.days.dropna()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Observation dates", len(obs))
c2.metric("First date", obs["Obs_Date"].min().strftime("%d %b %Y"))
c3.metric("Last date", obs["Obs_Date"].max().strftime("%d %b %Y"))
c4.metric("Repeat interval", f"{int(gap_days.mode().iloc[0])} days")

with st.expander("Show all observation dates"):
    obs_table = pd.DataFrame({
        "Observation date": obs["Obs_Date"].dt.strftime("%d %b %Y"),
        "Valid pixels": obs["Pixels"].values
    })
    st.dataframe(obs_table, hide_index=True, width="stretch")

st.caption(
    "Landsat 8–9 observations were available on these dates only. "
    "Each date is one satellite pass over Nagpur."
)
# ============================================================
# DAILY AIR TEMPERATURE
# ============================================================

st.subheader(
    "🌡️ Daily Air Temperature — March to May 2026"
)


trend_dates = daily_data["Date"].tolist()

trend_t2m = [
    float(value)
    for value in daily_data["T2M"]
]


fig_t2m = px.line(
    x=trend_dates,
    y=trend_t2m,
    labels={
        "x": "Date",
        "y": "Air Temperature (°C)"
    },
    title="Daily Air Temperature"
)


fig_t2m.update_layout(
    height=450,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)


st.plotly_chart(
    fig_t2m,
    width="stretch"
)

# ============================================================
# SURFACE TEMPERATURE VS AIR TEMPERATURE
# ============================================================

st.subheader("🛰️🌡️ Surface Temperature vs Air Temperature")

satellite_compare = satellite_data[["Date", "LST"]].copy()
daily_compare = daily_data[["Date", "T2M"]].copy()

# Normalize dates before grouping and merging
satellite_compare["Date"] = pd.to_datetime(
    satellite_compare["Date"], dayfirst=True, errors="coerce"
).dt.normalize()

daily_compare["Date"] = pd.to_datetime(
    daily_compare["Date"], dayfirst=True, errors="coerce"
).dt.normalize()

satellite_compare["LST"] = pd.to_numeric(
    satellite_compare["LST"], errors="coerce"
)
daily_compare["T2M"] = pd.to_numeric(
    daily_compare["T2M"], errors="coerce"
)

satellite_compare = satellite_compare.dropna(
    subset=["Date", "LST"]
)
daily_compare = daily_compare.dropna(
    subset=["Date", "T2M"]
)

# One mean satellite LST per observation date
def parse_dates(series):
    s = series.astype(str).str.strip()
    first_ten = s.str[:10]

    parsed_iso = pd.to_datetime(
        first_ten, format="%Y-%m-%d", errors="coerce"
    )
    parsed_dmy = pd.to_datetime(
        first_ten, format="%d-%m-%Y", errors="coerce"
    )

    return parsed_iso.fillna(parsed_dmy)


satellite_dates = satellite_compare.copy()
daily_dates = daily_compare.copy()

satellite_dates["Date"] = parse_dates(satellite_dates["Date"])
daily_dates["Date"] = parse_dates(daily_dates["Date"])

satellite_dates["LST"] = pd.to_numeric(
    satellite_dates["LST"], errors="coerce"
)
daily_dates["T2M"] = pd.to_numeric(
    daily_dates["T2M"], errors="coerce"
)

sat_mean = (
    satellite_dates.dropna(subset=["Date", "LST"])
    .groupby("Date", as_index=False)["LST"]
    .mean()
)

air_daily = (
    daily_dates.dropna(subset=["Date", "T2M"])
    .groupby("Date", as_index=False)["T2M"]
    .mean()
)

sat_daily = (
    sat_mean
    .merge(air_daily, on="Date", how="left")
    .sort_values("Date")
)

sat_daily["Date_Label"] = sat_daily["Date"].dt.strftime("%d %b")

plot_df = sat_daily.melt(
    id_vars="Date_Label",
    value_vars=["LST", "T2M"],
    var_name="Series",
    value_name="Temperature"
)

plot_df["Series"] = plot_df["Series"].map({
    "LST": "Surface temperature (satellite LST, mean)",
    "T2M": "Air temperature (T2M)"
})

fig_compare = px.line(
    plot_df,
    x="Date_Label",
    y="Temperature",
    color="Series",
    markers=True,
    labels={
        "Date_Label": "Date",
        "Temperature": "Temperature (°C)",
        "Series": ""
    },
    title="Mean Satellite Surface Temperature vs Air Temperature"
)

fig_compare.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=sat_daily["Date_Label"].tolist(),
    tickangle=-45
)

fig_compare.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_compare, width="stretch")

st.caption(
    "Satellite LST is averaged across available pixels for each "
    "observation date. T2M represents air temperature. These are "
    "different measurements and should not be expected to match."
)

# ============================================================
# MODEL-BASED SCENARIO ANALYSIS
# ============================================================
# ============================================================
# MODEL-BASED SCENARIO ANALYSIS
# ============================================================


# ============================================================
# MODEL-BASED SCENARIO ANALYSIS
# ============================================================

st.header(
    "🌱🙏🌡️ Model-Based Scenario Analysis"
)


st.write(
    "Adjust environmental variables to explore how the trained "
    "Random Forest model responds under different input scenarios."
)


# Baseline values

baseline_ndvi = 0.112559

baseline_ndbi = 0.026918

baseline_t2m = 29.06

baseline_latitude = 21.164836

baseline_longitude = 78.946949

baseline_day_of_year = 67


baseline_input = pd.DataFrame(
    [[
        baseline_ndvi,
        baseline_ndbi,
        baseline_t2m,
        baseline_latitude,
        baseline_longitude,
        baseline_day_of_year
    ]],
    columns=[
        "NDVI",
        "NDBI",
        "T2M",
        "Latitude",
        "Longitude",
        "Day_of_Year"
    ]
)


baseline_prediction = model.predict(
    baseline_input
)[0]


# Scenario controls

col1, col2, col3 = st.columns(3)


with col1:

    ndvi_value = st.slider(
        "🌳 Vegetation Index",
        min_value=-0.1323,
        max_value=0.4722,
        value=baseline_ndvi,
        step=0.01
    )


with col2:

    ndbi_value = st.slider(
        "🏙️ Built-up Index",
        min_value=-0.3308,
        max_value=0.2755,
        value=baseline_ndbi,
        step=0.01
    )


with col3:

    t2m_value = st.slider(
        "🌡️ Air Temperature (°C)",
        min_value=29.06,
        max_value=39.96,
        value=baseline_t2m,
        step=0.1
    )


scenario_input = pd.DataFrame(
    [[
        ndvi_value,
        ndbi_value,
        t2m_value,
        baseline_latitude,
        baseline_longitude,
        baseline_day_of_year
    ]],
    columns=[
        "NDVI",
        "NDBI",
        "T2M",
        "Latitude",
        "Longitude",
        "Day_of_Year"
    ]
)


scenario_prediction = model.predict(
    scenario_input
)[0]


temperature_change = (
    scenario_prediction -
    baseline_prediction
)


st.subheader(
    "Scenario Comparison"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Baseline LST",
        f"{baseline_prediction:.2f} °C"
    )


with col2:

    st.metric(
        "Scenario LST",
        f"{scenario_prediction:.2f} °C"
    )


with col3:

    st.metric(
        "Change",
        f"{temperature_change:+.2f} °C"
    )


st.caption(
    "Scenario results represent the Random Forest model's "
    "response to the selected input values and should not "
    "be interpreted as causal evidence."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "Nagpur Urban Heat Island Detection & Climate Resilience System | "
    f"Selected prediction date: "
    f"{pd.Timestamp(selected_date).strftime('%d %B %Y')} | "
    "Satellite-based LST analysis with Random Forest"
)


st.caption(
    "Note: Heat zones are relative to the predicted LST "
    "distribution, and scenario results represent model "
    "responses rather than causal effects."
)
# ============================================================
# METHODOLOGY
# ============================================================

st.header("🔬 Methodology")

st.write(
    "The dashboard combines satellite-derived environmental "
    "indicators, Random Forest prediction, spatial analysis, "
    "hotspot persistence, and climate-resilience prioritization "
    "to analyze urban heat patterns across Nagpur."
)

# ------------------------------------------------------------
# WORKFLOW
# ------------------------------------------------------------

st.subheader("🔄 Overall Workflow")

st.markdown(
    """
**Landsat 8–9**
→ **LST + NDVI + NDBI**
→ **Environmental & Spatial Features**
→ **Random Forest Regression**
→ **Date-wise Predicted LST**
→ **Heat Zone Analysis**
→ **Hotspot Persistence**
→ **Climate Resilience Priority**
"""
)

# ------------------------------------------------------------
# KEY METHODOLOGY STEPS
# ------------------------------------------------------------

st.subheader("📌 Key Steps")

st.markdown(
    """
**1️⃣ Satellite Data**  
Landsat 8–9 observations from March to May 2026 were used
to derive land surface temperature and environmental indicators
for the Nagpur study area.

**2️⃣ Environmental Indicators**  
- 🌡️ **LST:** Land surface temperature derived from satellite
  thermal observations.
- 🌿 **NDVI:** Indicates vegetation or greenness.
- 🏙️ **NDBI:** Represents relative built-up surface characteristics.

**3️⃣ Machine Learning Prediction**  
A Random Forest regression model estimates LST using:

`NDVI + NDBI + T2M + Latitude + Longitude + Day of Year`

**4️⃣ Date-wise Analysis**  
The trained model was applied to available observation dates
to analyze changes in predicted LST across the study period.

**5️⃣ Heat Zone Classification**  
Relative heat zones were defined using the predicted LST
distribution for each date:
- **Low:** Below 80th percentile
- **Moderate:** 80th–90th percentile
- **High:** 90th percentile and above

**6️⃣ Hotspot Persistence**  
Locations appearing within the hottest 10% of predicted LST
across multiple dates were identified as persistent hotspots.

**7️⃣ Climate Resilience Priority**  
A relative planning score combines:
`50% Heat + 30% Built-up + 20% Low Vegetation`

The score is converted to a 0–100 scale and divided into
relative priority classes using the selected-date distribution.
"""
)

st.info(
    "Heat zones and resilience priorities are relative indicators "
    "based on the study dataset. They should support, rather than "
    "replace, detailed field surveys, local planning data, and "
    "public-health assessments."
)
# ------------------------------------------------------------
# 1. DATA SOURCE
# ------------------------------------------------------------

st.subheader("1️⃣ Satellite Data")

st.write(
    "Landsat 8–9 satellite observations for March to May 2026 "
    "were used to derive land surface temperature (LST) and "
    "environmental indicators for the Nagpur study area."
)

# ------------------------------------------------------------
# 2. ENVIRONMENTAL INDICES
# ------------------------------------------------------------

st.subheader("2️⃣ Environmental Indicators")

st.markdown(
    """
- 🌡️ **LST (Land Surface Temperature):** Represents the temperature
  of the land surface derived from satellite thermal observations.

- 🌿 **NDVI (Normalized Difference Vegetation Index):** Represents
  vegetation or greenness.

- 🏙️ **NDBI (Normalized Difference Built-up Index):** Represents
  relative built-up surface characteristics.
"""
)

# ------------------------------------------------------------
# 3. MACHINE LEARNING
# ------------------------------------------------------------

st.subheader("3️⃣ Machine Learning Prediction")

st.write(
    "A Random Forest regression model was used to estimate "
    "land surface temperature using environmental and spatial "
    "features including NDVI, NDBI, T2M, latitude, longitude, "
    "and day of year."
)

st.markdown(
    """
**Model inputs**

`NDVI + NDBI + T2M + Latitude + Longitude + Day of Year`

⬇️

**Random Forest Regression**

⬇️

**Predicted LST**
"""
)

# ------------------------------------------------------------
# 4. DATE-WISE ANALYSIS
# ------------------------------------------------------------

st.subheader("4️⃣ Date-wise Prediction")

st.write(
    "The trained model was applied to multiple observation dates "
    "from March to May 2026 to generate date-wise predicted LST "
    "datasets and examine temporal changes in surface temperature."
)

# ------------------------------------------------------------
# 5. HEAT ZONE CLASSIFICATION
# ------------------------------------------------------------

st.subheader("5️⃣ Heat Zone Classification")

st.write(
    "Predicted LST values are classified into relative heat zones "
    "using the selected-date temperature distribution."
)

st.markdown(
    """
- 🟢 **Low:** below the 80th percentile
- 🟠 **Moderate:** 80th–90th percentile
- 🔴 **High:** at or above the 90th percentile
"""
)

# ------------------------------------------------------------
# 6. ENVIRONMENTAL ANALYSIS
# ------------------------------------------------------------

st.subheader("6️⃣ Environmental Analysis")

st.write(
    "Spatial relationships between predicted LST, NDVI, and NDBI "
    "are examined using maps, summary statistics, correlations, "
    "and scatter plots with trend lines."
)

st.info(
    "Correlation values describe spatial association and should "
    "not be interpreted as proof of causation."
)

# ------------------------------------------------------------
# 7. CLIMATE RESILIENCE PRIORITY
# ------------------------------------------------------------

st.subheader("7️⃣ Climate Resilience Priority")

st.write(
    "A planning-oriented priority indicator combines normalized "
    "heat intensity, built-up intensity, and relatively low "
    "vegetation to identify locations that may deserve greater "
    "planning attention."
)
st.markdown(
    """
**Priority Score**

`50% Heat + 30% Built-up + 20% Low Vegetation`

The resulting score is converted to a 0–100 scale and divided
into relative priority classes using the selected-date
score distribution.
"""
)

st.info(
    "The Climate Resilience Priority indicator is a planning "
    "support tool and is not a validated public-health vulnerability "
    "or risk index."
)

# ------------------------------------------------------------
# METHODOLOGY FLOW
# ------------------------------------------------------------

st.subheader("🔄 Overall Workflow")

st.markdown(
    """
**Landsat 8–9**

↓

**LST + NDVI + NDBI**

↓

**Environmental & Spatial Features**

↓

**Random Forest Regression**

↓

**Date-wise Predicted LST**

↓

**Heat Zone Analysis**

↓

**Environmental Relationship Analysis**

↓

**Climate Resilience Priority**

↓

**Planning Insights**
"""
)

st.caption(
    "The dashboard is designed to support spatial understanding "
    "of urban heat patterns and potential climate-resilience "
    "planning. Results should be interpreted together with "
    "field observations, local planning data, and other relevant "
    "evidence."
)

# ============================================================