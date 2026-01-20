import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_processed_data

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
st.subheader("🗺️ Geographic Insights")
import requests

@st.cache_data
def get_india_geojson():
    # Public URL for India States GeoJSON (High quality)
    url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    try:
        r = requests.get(url)
        return r.json()
    except Exception as e:
        st.error(f"Could not load map data: {e}")
        return None

@st.cache_data
def get_district_geojson():
    # Load India District GeoJSON (Cached)
    url = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
    try:
        r = requests.get(url)
        return r.json()
    except Exception as e:
        st.error(f"Could not load district map data: {e}")
        return None

if selected_state == "All":
    # STATE LEVEL CHOROPLETH
    st.markdown("**State-wise Enrolment Saturation**")
    
    geojson = get_india_geojson()
    if geojson:
        # Aggregation
        map_df = filtered_df.groupby('state')[['total_enrol', 'total_bio']].sum().reset_index()
        map_df['Total'] = map_df['total_enrol'] + map_df['total_bio']
        
        # Plotly Choropleth
        fig_map = px.choropleth(
            map_df,
            geojson=geojson,
            featureidkey='properties.ST_NM',
            locations='state',
            color='Total',
            color_continuous_scale='Viridis',
            title="Total Activity by State (Heatmap)",
            hover_data=['total_enrol', 'total_bio']
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        # Fallback to bar if map fails
        geo_agg = filtered_df.groupby('state')[['total_enrol', 'total_bio']].sum().reset_index()
        geo_agg['Total'] = geo_agg['total_enrol'] + geo_agg['total_bio']
        fig_geo = px.bar(geo_agg.sort_values('Total', ascending=False).head(20), x='state', y='Total')
        st.plotly_chart(fig_geo, use_container_width=True)

else:
    # DISTRICT LEVEL CHOROPLETH
    st.markdown(f"**District-wise Saturation for {selected_state}**")
    
    geo_agg = filtered_df.groupby('district')[['total_enrol', 'total_bio']].sum().reset_index()
    geo_agg['Total'] = geo_agg['total_enrol'] + geo_agg['total_bio']
    
    district_geojson = get_district_geojson()
    
    # Try Map First
    if district_geojson:
        # Note: District names often mismatch (e.g. 'Bangalore' vs 'Bengaluru'). 
        # We assume dataset names match GeoJSON 'NAME_2' or 'DISTRICT' properties.
        fig_map = px.choropleth(
            geo_agg,
            geojson=district_geojson,
            featureidkey='properties.NAME_2', # Common property for district names in this file
            locations='district',
            color='Total',
            color_continuous_scale='Magma',
            title=f"District Heatmap: {selected_state}",
            hover_data=['total_enrol', 'total_bio']
        )
        # This focuses the map on the data points available
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Also show Bar Chart for better readability of top districts
    fig_geo = px.bar(geo_agg.sort_values('Total', ascending=False).head(20), x='district', y='Total', title=f"Top Districts in {selected_state} (Ranked)")
    st.plotly_chart(fig_geo, use_container_width=True)
