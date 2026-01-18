import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_and_merge_all
from models import load_or_train_model

st.set_page_config(page_title="Aadhaar Sentinel", layout="wide", page_icon="🛡️")

# Custom CSS for "Command Center" look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Roboto Mono', monospace;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_and_merge_all()
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def get_model(df):
    model, df_scored = load_or_train_model(df)
    return df_scored

def main():
    st.title("🛡️ Aadhaar Sentinel: Integrity Command Center")
    st.markdown("### Operational Intelligence & Anomaly Detection System")
    
    with st.spinner("Initializing Sentinel System... Ingesting Data..."):
        df = get_data()
        
    if df.empty:
        st.error("No data found! Please check the dataset folder.")
        return

    # Sidebar Filters - Hierarchical
    st.sidebar.header("🔍 Filter Scope")
    
    # Level 1: State
    states = ['All'] + sorted(list(df['state'].unique()))
    selected_state = st.sidebar.selectbox("Select State", states)
    
    # Level 2: District
    if selected_state != 'All':
        df_view = df[df['state'] == selected_state]
        districts = ['All'] + sorted(list(df_view['district'].unique()))
        selected_district = st.sidebar.selectbox("Select District", districts)
        
        # Level 3: Pincode / Drill Down
        if selected_district != 'All':
            df_view = df_view[df_view['district'] == selected_district]
            pincodes = ['All'] + sorted(list(df_view['pincode'].unique()))
            selected_pincode = st.sidebar.selectbox("Select Pincode", pincodes)
            if selected_pincode != 'All':
                 df_view = df_view[df_view['pincode'] == selected_pincode]
    else:
        df_view = df

    # Run AI Model
    with st.spinner("The Watchdog is analyzing patterns..."):
        df_view = get_model(df_view)

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    total_recs = len(df_view)
    anomalies = df_view[df_view['is_anomaly'] == -1]
    anomaly_count = len(anomalies)
    
    # Calculate approximations if columns exist
    enrol_cols = [c for c in df_view.columns if 'enrol' in c and pd.api.types.is_numeric_dtype(df_view[c])]
    bio_cols = [c for c in df_view.columns if 'bio_' in c and pd.api.types.is_numeric_dtype(df_view[c])]
    demo_cols = [c for c in df_view.columns if 'demo_' in c and pd.api.types.is_numeric_dtype(df_view[c])]

    total_enrol = df_view[enrol_cols].sum().sum() if enrol_cols else 0
    total_updates = (df_view[bio_cols].sum().sum() if bio_cols else 0) + (df_view[demo_cols].sum().sum() if demo_cols else 0)

    c1.metric("Total Records", f"{total_recs:,}")
    c2.metric("Total Enrolments", f"{int(total_enrol):,}")
    c3.metric("Total Updates", f"{int(total_updates):,}")
    c4.metric("⚠️ Anomalies Detected", f"{anomaly_count}", delta_color="inverse")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🚨 Risk Radar (Anomalies)", "🗺️ Geospatial View"])

    with tab1:
        st.subheader("Activity Trends")
        if 'date' in df_view.columns:
            # Aggregate by date
            daily_stats = df_view.groupby('date')[['total_bio', 'total_demo']].sum().reset_index() if 'total_bio' in df_view.columns else pd.DataFrame()
            if not daily_stats.empty:
                daily_melt = daily_stats.melt(id_vars=['date'], var_name='Metric', value_name='Count')
                fig = px.line(daily_melt, x='date', y='Count', color='Metric', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("⚠️ High Value Anomalies")
        
        if anomaly_count > 0:
            st.warning(f"Found {anomaly_count} suspicious records in this view.")
            
            # Smart Table with Reasons
            display_cols = ['date', 'state', 'district', 'pincode', 'anomaly_reason', 'anomaly_score']
            # Add metric columns if space permits
            display_cols += ['total_bio', 'total_demo'] if 'total_bio' in df_view.columns else []
            
            # Filter columns to only those that exist
            display_cols = [c for c in display_cols if c in df_view.columns]
            
            st.dataframe(
                anomalies.sort_values('anomaly_score')[display_cols],
                use_container_width=True,
                column_config={
                    "anomaly_reason": st.column_config.TextColumn("AI Diagnosis", width="large"),
                    "anomaly_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=-0.5, max_value=0.5),
                }
            )
        else:
            st.success("No significant anomalies detected in this view.")

    with tab3:
        st.subheader("Regional Deep Dive")
        
        # Dynamic Grouping based on Selection Level
        if selected_state == 'All':
            # National View -> Show States
            group_col = 'state'
            title = "Total Activity by State"
        elif selected_district == 'All':
            # State View -> Show Districts
            group_col = 'district'
            title = f"Total Activity by District in {selected_state}"
        else:
            # District View -> Show Pincodes
            group_col = 'pincode'
            title = f"Activity by Pincode in {selected_district}"
            
        if group_col in df_view.columns:
            group_agg = df_view.groupby(group_col)[['total_bio', 'total_demo']].sum().reset_index()
            group_agg['Total'] = group_agg['total_bio'] + group_agg['total_demo']
            
            fig_bar = px.bar(group_agg.sort_values('Total', ascending=False).head(50), 
                             x=group_col, y='Total', color='Total',
                             title=title, template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
