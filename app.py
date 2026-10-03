import streamlit as st
import rasterio
import numpy as np
import plotly.express as px
import joblib
@st.cache_resource
def load_model():
    return joblib.load("Nagpur_Deployment_RF_Model_Compressed.pkl")
model = load_model()
st.set_page_config(
    page_title="Nagpur UHI Detection & Climate Resilience",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Nagpur Urban Heat Island Detection & Climate Resilience System")

st.markdown(
    """
    **Satellite-Based LST Prediction • UHI Heat-Zone Mapping • Random Forest Analysis**
    
    This interactive dashboard presents predicted Land Surface Temperature,
    relative urban heat zones, model validation results, feature importance,
    spatial support, and model-based environmental scenarios for Nagpur.
    """
)

st.info(
    "📅 Prediction date: 03 May 2026  |  "
    "🛰️ Satellite-based environmental analysis  |  "
    "🤖 Random Forest LST model"
)
st.header("📌 Key Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean LST", "52.77 °C")

with col2:
    st.metric("Maximum LST", "62.34 °C")

with col3:
    st.metric("Validation R²", "0.8741")

with col4:
    st.metric("Validation RMSE", "1.878 °C")
st.header("🗺️ Predicted Land Surface Temperature")

raster_path = "Nagpur_Predicted_LST_2026_05_03.tif"

with rasterio.open(raster_path) as src:
    lst = src.read(1)
    nodata = src.nodata

if nodata is not None:
    lst = np.where(lst == nodata, np.nan, lst)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Minimum LST", f"{np.nanmin(lst):.2f} °C")

with col2:
    st.metric("Maximum LST", f"{np.nanmax(lst):.2f} °C")

with col3:
    st.metric("Mean LST", f"{np.nanmean(lst):.2f} °C")

st.subheader("Predicted LST — 03 May 2026")

with rasterio.open(raster_path) as src:
    bounds = src.bounds

fig = px.imshow(
    lst,
    x=np.linspace(bounds.left, bounds.right, lst.shape[1]),
    y=np.linspace(bounds.top, bounds.bottom, lst.shape[0]),
    color_continuous_scale="Inferno",
    aspect="auto",
    labels={
        "x": "Longitude",
        "y": "Latitude",
        "color": "LST (°C)"
    }
)

fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, width="stretch")
st.header("🔥 UHI Heat Zones")

heat_zone_path = "Nagpur_Heat_Zones_2026_05_03.tif"

with rasterio.open(heat_zone_path) as src:
    heat_zones = src.read(1)
    heat_bounds = src.bounds

fig_zones = px.imshow(
    heat_zones,
    x=np.linspace(heat_bounds.left, heat_bounds.right, heat_zones.shape[1]),
    y=np.linspace(heat_bounds.top, heat_bounds.bottom, heat_zones.shape[0]),
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    labels={
        "x": "Longitude",
        "y": "Latitude",
        "color": "Heat Zone"
    }
)

fig_zones.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig_zones, width="stretch")

st.info(
    "Heat zones are relative zones derived from the predicted LST distribution "
    "for Nagpur on 03 May 2026. They should not be interpreted as absolute "
    "health-risk temperature thresholds."
)
st.header("📊 Model Performance")

st.write(
    "Model performance based on spatial-block validation."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R² Score", "0.8741")

with col2:
    st.metric("RMSE", "1.878 °C")

with col3:
    st.metric("MAE", "1.391 °C")

st.caption(
    "Validation used spatial blocks to reduce the effect of spatially nearby "
    "training and testing observations."
)
st.header("🌳 Random Forest Feature Importance")

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
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_importance, width="stretch")

st.caption(
    "Feature importance values represent the relative contribution of each "
    "input feature according to the Random Forest model."
)
st.header("🗺️ Model Spatial Support")

support_path = "Nagpur_Prediction_Support_2026_05_03.tif"

with rasterio.open(support_path) as src:
    support = src.read(1)
    support_bounds = src.bounds

fig_support = px.imshow(
    support,
    x=np.linspace(
        support_bounds.left,
        support_bounds.right,
        support.shape[1]
    ),
    y=np.linspace(
        support_bounds.top,
        support_bounds.bottom,
        support.shape[0]
    ),
    color_continuous_scale="Blues",
    aspect="auto",
    labels={
        "x": "Longitude",
        "y": "Latitude",
        "color": "Support"
    }
)

fig_support.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig_support, width="stretch")

st.info(
    "The support map indicates whether each prediction location falls "
    "within the spatial area represented by the model's observed training data. "
    "Locations outside this support represent spatial extrapolation."
)
st.header("📍 Model Spatial Reliability")

reliability_path = "Nagpur_Model_Spatial_Reliability_2026_05_03.tif"

with rasterio.open(reliability_path) as src:
    reliability = src.read(1)
    reliability_bounds = src.bounds
    reliability_nodata = src.nodata

reliability = np.where(
    reliability == reliability_nodata,
    np.nan,
    reliability
)

fig_reliability = px.imshow(
    reliability,
    x=np.linspace(
        reliability_bounds.left,
        reliability_bounds.right,
        reliability.shape[1]
    ),
    y=np.linspace(
        reliability_bounds.top,
        reliability_bounds.bottom,
        reliability.shape[0]
    ),
    color_continuous_scale="Blues",
    aspect="auto",
    labels={
        "x": "Longitude",
        "y": "Latitude",
        "color": "Reliability"
    }
)

fig_reliability.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig_reliability, width="stretch")

st.info(
    "The spatial reliability layer indicates whether prediction locations "
    "are represented by the model's observed spatial training support. "
    "Areas outside observed support should be interpreted as spatial extrapolation."
)
st.header("🌱🏙️🌡️ Model-Based Scenario Analysis")

st.write(
    "Adjust environmental variables to explore how the trained Random Forest "
    "model responds under different input scenarios."
)

# Baseline values
baseline_ndvi = 0.112559
baseline_ndbi = 0.026918
baseline_t2m = 29.06
baseline_latitude = 21.164836
baseline_longitude = 78.946949
baseline_day_of_year = 67

baseline_input = np.array([[
    baseline_ndvi,
    baseline_ndbi,
    baseline_t2m,
    baseline_latitude,
    baseline_longitude,
    baseline_day_of_year
]])

baseline_prediction = model.predict(baseline_input)[0]

# Scenario controls
col1, col2, col3 = st.columns(3)

with col1:
    ndvi_value = st.slider(
        "🌱 NDVI",
        min_value=-0.1323,
        max_value=0.4722,
        value=baseline_ndvi,
        step=0.01
    )

with col2:
    ndbi_value = st.slider(
        "🏙️ NDBI",
        min_value=-0.3308,
        max_value=0.2755,
        value=baseline_ndbi,
        step=0.01
    )

with col3:
    t2m_value = st.slider(
        "🌡️ T2M (°C)",
        min_value=29.06,
        max_value=39.96,
        value=baseline_t2m,
        step=0.1
    )

scenario_input = np.array([[
    ndvi_value,
    ndbi_value,
    t2m_value,
    baseline_latitude,
    baseline_longitude,
    baseline_day_of_year
]])

scenario_prediction = model.predict(scenario_input)[0]

temperature_change = scenario_prediction - baseline_prediction

st.subheader("Scenario Comparison")

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
    "Scenario results represent the Random Forest model's response to the "
    "selected input values and should not be interpreted as causal evidence."
)
st.markdown("---")

st.caption(
    "Nagpur Urban Heat Island Detection & Climate Resilience System | "
    "Prediction date: 03 May 2026 | "
    "Satellite-based LST analysis with Random Forest"
)

st.caption(
    "Note: Heat zones are relative to the predicted LST distribution, "
    "and scenario results represent model responses rather than causal effects."
)