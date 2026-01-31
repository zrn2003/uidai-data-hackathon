import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_processed_data

# Use a minimalist approach first to ensure data visibility
st.set_page_config(page_title="Aadhaar Analytics", layout="wide")

@st.cache_data
def get_data():
    return load_processed_data()

try:
    with st.spinner("Loading Data from scratch..."):
        df = get_data()
except Exception as e:
    st.error(f"Critical Data Error: {e}")
    st.stop()

if df.empty:
    st.warning("Dataframe is empty. Check logs.")
    st.stop()

# --- HEADER ---
st.title("🇮🇳 Aadhaar Dashboard (Live Build)")
st.caption(f"Loaded {len(df):,} records | Date Range: {df['date'].min().date()} - {df['date'].max().date()}")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Filters")
selected_state = st.sidebar.selectbox("State", ["All"] + sorted(df['state'].unique().astype(str)))

if selected_state != "All":
    filtered_df = df[df['state'] == selected_state]
else:
    filtered_df = df

selected_district = "All"
if selected_state != "All":
    districts = ["All"] + sorted(filtered_df['district'].unique().astype(str))
    selected_district = st.sidebar.selectbox("District", districts)
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df['district'] == selected_district]

# --- KPI METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Enrolments", f"{filtered_df['total_enrol'].sum():,.0f}")
col2.metric("Total Updates (Bio)", f"{filtered_df['total_bio'].sum():,.0f}")
col3.metric("Total Updates (Demo)", f"{filtered_df['total_demo'].sum():,.0f}")

# --- RAW DATA CHECK ---
with st.expander("🔍 View Raw Source Data", expanded=False):
    st.dataframe(filtered_df.head(100))

# --- CHARTS ---
st.subheader("Trends Over Time")
# Aggregating by Date to prevent slow rendering of millions of points
daily_agg = filtered_df.groupby('date')[['total_enrol', 'total_bio', 'total_demo']].sum().reset_index()

if not daily_agg.empty:
    melted = daily_agg.melt('date', var_name='Metric', value_name='Count')
    fig = px.line(melted, x='date', y='Count', color='Metric', title="Daily Trends")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trend data available.")

# --- GEOSPATIAL INTELLIGENCE ---
# --- GEOSPATIAL & FORECASTING (ROBUST) ---
st.subheader("🗺️ Unified Intelligence Map & Forecast")

import geopandas as gpd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. LOAD MAP DATA (Cached)
@st.cache_data
def load_india_map():
    geo_url = 'https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson'
    try:
        india = gpd.read_file(geo_url).drop_duplicates().reset_index(drop=True)
        # Normalize state names in Map
        state_col = next((col for col in ['NAME_1', 'ST_NM', 'state'] if col in india.columns), 'NAME_1')
        india[state_col] = india[state_col].astype(str).str.strip().str.upper()
        return india, state_col
    except Exception as e:
        st.error(f"Map Load Error: {e}")
        return None, None

india_map, map_state_col = load_india_map()

# 2. PREPARE DATA FOR MAP
@st.cache_data
def prepare_map_data(df):
    # Aggregate
    state_totals = df.groupby('state')['total_enrol'].sum().reset_index() # Use total_enrol or calculated 'total'
    # Check if 'total' exists, else use total_enrol
    target_col = 'total_enrol'
    
    state_totals['state_raw'] = state_totals['state'].astype(str).str.strip()
    
    # 3. GeoJSON Specific Mapping
    # The data is already clean (Title Case), but we need to match the specific UPPERCASE names in the GeoJSON file.
    # Most will match automatically with .upper(), but we handle exceptions here.
    geojson_fix_map = {
        'NCT Of Delhi': 'NCT OF DELHI', 
        'Andaman And Nicobar Islands': 'ANDAMAN AND NICOBAR', # GeoJSON often shortens this
        'Dadra And Nagar Haveli And Daman And Diu': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU', # Verify this matches your specific GeoJSON
    }
    
    # Map raw states to upper case map keys
    # First try direct upper case, then apply specific fixes for the map file
    state_totals['geo_state'] = state_totals['state_raw'].str.upper()
    state_totals['geo_state'] = state_totals['geo_state'].replace({k.upper(): v for k, v in geojson_fix_map.items()})

    return state_totals, target_col

if india_map is not None:
    state_totals, target_col = prepare_map_data(filtered_df)
    
    # Merge
    india_merged = india_map.merge(state_totals, left_on=map_state_col, right_on='geo_state', how='left')
    india_merged[target_col] = india_merged[target_col].fillna(0)
    
    # Plot Map
    fig_map = px.choropleth_mapbox(
        india_merged,
        geojson=india_merged.geometry.__geo_interface__,
        locations=india_merged.index,
        color=target_col,
        color_continuous_scale="Reds",
        hover_name=map_state_col,
        hover_data={target_col: ':.0f', 'geo_state': False},
        title=f"Aadhaar Enrolments by State",
        mapbox_style="carto-positron",
        center={"lat": 22, "lon": 80},
        zoom=3.5,
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # 3. FORECASTING (Prophet)
    st.markdown("### 🔮 State-wise Demand Forecast (Prophet)")
    
    top_states_list = state_totals.nlargest(5, target_col)['state_raw'].tolist()
    forecast_state = st.selectbox("Select State for Prediction", sorted(filtered_df['state'].unique()), index=0)
    
    if st.button(f"Generate Forecast for {forecast_state}"):
        with st.spinner("Training Prophet Model..."):
            # Prepare Data
            state_daily = filtered_df[filtered_df['state'] == forecast_state].groupby('date')[target_col].sum().reset_index()
            if len(state_daily) > 10:
                state_daily.columns = ['ds', 'y']
                
                m = Prophet(daily_seasonality=True)
                m.fit(state_daily)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                
                # Plot
                st.write(f"**Next 30 Days Forecast: {forecast_state}**")
                
                # Plotly version of forecast
                fig_fc = px.line(forecast, x='ds', y='yhat', title=f"Predicted Demand Trend: {forecast_state}")
                fig_fc.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False)
                fig_fc.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='Confidence')
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.warning("Not enough data history to forecast.")