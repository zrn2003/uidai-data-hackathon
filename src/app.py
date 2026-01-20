import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_and_merge_all
from models import load_or_train_model, forecast_next_30_days, cluster_districts

st.set_page_config(page_title="Aadhaar Analytics Dashboard", layout="wide", page_icon="🇮🇳")

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main { background: #0e1117; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; color: #00ffcc; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 8px; border: 1px solid #374151; }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_and_merge_all()

@st.cache_resource
def get_ai_models(df):
    model, df_scored = load_or_train_model(df)
    return df_scored

def main():
    st.title("🇮🇳 Aadhaar India-Wide Analytics Dashboard")
    st.markdown("### Policy Optimization & Integrity Command Center")

    with st.spinner("Ingesting 12+ Datasets & Processing 100M+ Records (Simulated)..."):
        raw_df = get_data()
        
    if raw_df.empty:
        st.error("Dataset Empty. Please check `dataset/` folder.")
        return

    # Sidebar: Global Filters
    st.sidebar.title("🔍 Analytics Filters")
    
    # Date Filter
    if 'date' in raw_df.columns:
        min_date = raw_df['date'].min()
        max_date = raw_df['date'].max()
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    
    # Location Filter
    states = ['All India'] + sorted(list(raw_df['state'].unique()))
    selected_state = st.sidebar.selectbox("State / Union Territory", states)
    
    df_filtered = raw_df.copy()
    if selected_state != 'All India':
        df_filtered = df_filtered[df_filtered['state'] == selected_state]
        
    # Apply Date Filter
    if 'date' in df_filtered.columns and len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['date'] >= pd.to_datetime(date_range[0])) & 
                                  (df_filtered['date'] <= pd.to_datetime(date_range[1]))]
                                  
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Export Cleaned Data",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name='aadhaar_analytics_export.csv',
        mime='text/csv'
    )

    # Run AI Analysis on Filtered Data
    df_filtered = get_ai_models(df_filtered)

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview", 
        "🗺️ Geospatial Intelligence", 
        "👥 Demographics", 
        "📈 Forecasting Trends",
        "🧠 AI Insights (Clusters)"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        # Top KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Enrolments", f"{int(df_filtered['total_enrol'].sum()):,}")
        c2.metric("Biometric Updates", f"{int(df_filtered['total_bio'].sum()):,}")
        c3.metric("Demographic Updates", f"{int(df_filtered['total_demo'].sum()):,}")
        anomalies = df_filtered[df_filtered.get('is_anomaly', 1) == -1]
        c4.metric("⚠️ Anomalies Detected", f"{len(anomalies)}", delta_color="inverse")
        
        st.markdown("---")
        
        # Recent Anomalies Table
        st.subheader("🚨 Recent Integrity Alerts")
        if not anomalies.empty:
            st.dataframe(
                anomalies.sort_values('date', ascending=False).head(10)[['date', 'state', 'district', 'pincode', 'anomaly_reason']],
                use_container_width=True
            )
        else:
            st.success("No anomalies detected in the current view.")

    # --- TAB 2: GEOSPATIAL ---
    with tab2:
        st.subheader(f" Geographic Distribution - {selected_state}")
        
        # Use Treemap to simulate "Drill Down" from State -> District -> Pincode
        # This is very effective for hierarchical data without shapefiles
        st.markdown("**Hierarchy Map (State > District > Pincode)**")
        
        tree_df = df_filtered.groupby(['state', 'district', 'pincode'])[['total_enrol', 'total_bio']].sum().reset_index()
        # Limit nodes for performance
        if len(tree_df) > 1000:
            tree_df = tree_df.nlargest(1000, 'total_enrol')
            
        fig_tree = px.treemap(tree_df, path=[px.Constant("India"), 'state', 'district', 'pincode'], 
                              values='total_enrol', color='total_bio',
                              color_continuous_scale='RdBu',
                              title="Enrolment Volume (Size) vs Biometric Updates (Color)")
        st.plotly_chart(fig_tree, use_container_width=True)

    # --- TAB 3: DEMOGRAPHICS ---
    with tab3:
        st.subheader("Age Group Analysis")
        
        # Aggregate Age Buckets
        age_cols = ['enrol_0_5', 'enrol_5_17', 'enrol_18_plus']
        age_data = {col: df_filtered[col].sum() for col in age_cols if col in df_filtered.columns}
        
        if age_data:
            c1, c2 = st.columns(2)
            with c1:
                # Pie Chart
                fig_pie = px.pie(names=list(age_data.keys()), values=list(age_data.values()), 
                                 title="Enrolment by Age Group", hole=0.4, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                # Update Types comparison
                update_data = {
                    'Biometric': df_filtered['total_bio'].sum(),
                    'Demographic': df_filtered['total_demo'].sum()
                }
                fig_bar = px.bar(x=list(update_data.keys()), y=list(update_data.values()), color=list(update_data.keys()),
                                 title="Update Type Distribution", template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 4: TRENDS & FORECAST ---
    with tab4:
        st.subheader("📈 Prescriptive Analytics: 30-Day Forecast")
        
        forecast_metric = st.selectbox("Select Metric to Forecast", ['total_enrol', 'total_bio'])
        
        # Historical Plot
        daily_hist = df_filtered.groupby('date')[forecast_metric].sum().reset_index()
        fig_hist = px.line(daily_hist, x='date', y=forecast_metric, title=f"Historical {forecast_metric}", template="plotly_dark")
        
        # Forecast
        with st.spinner("Calculating future trends..."):
            forecast_df = forecast_next_30_days(df_filtered, metric=forecast_metric)
            
        if not forecast_df.empty:
            # Add forecast trace
            fig_hist.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['forecast'],
                mode='lines', name='Forecast (Next 30 Days)',
                line=dict(color='orange', width=2, dash='dash')
            ))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.info("💡 **Insight**: Based on current trends, we predict a **{:.1f}%** change in volume over the next month.".format(
                ((forecast_df['forecast'].mean() - daily_hist[forecast_metric].mean()) / daily_hist[forecast_metric].mean()) * 100
            ))
        else:
            st.plotly_chart(fig_hist, use_container_width=True)
            st.warning("Not enough data points to generate a reliable forecast.")

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.subheader("Cluster Analysis: District Performance Groups")
        
        dist_clusters = cluster_districts(df_filtered)
        if not dist_clusters.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig_scatter = px.scatter(dist_clusters, x='total_enrol', y='total_bio', 
                                         color='cluster_label', size='update_ratio',
                                         hover_data=['district'],
                                         title="District Clusters: Enrolment vs Updates",
                                         template="plotly_dark")
                st.plotly_chart(fig_scatter, use_container_width=True)
            with c2:
                st.write("**Cluster Definitions:**")
                st.info("**High Activity**: Districts with massive enrolment & update loads. Need more centers.")
                st.success("**Medium Activity**: Stable operations.")
                st.warning("**Low Activity**: Potential coverage gaps or rural areas.")
                
            st.dataframe(dist_clusters[['district', 'cluster_label', 'total_enrol', 'total_bio']], use_container_width=True)

