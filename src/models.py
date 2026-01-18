import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_anomaly_model(df: pd.DataFrame):
    """
    Trains an Isolation Forest model to detect anomalies in update patterns.
    """
    # Feature Selection:
    # We want to find anomalies in 'BIO updates' and 'DEMO updates' relative to 'Enrolments'
    # or just raw spikes.
    
    feature_cols = [c for c in df.columns if 'bio_' in c or 'demo_' in c or 'enrol_' in c]
    # Filter out non-numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if not feature_cols:
        print("No numeric features found for modeling.")
        return None, None
        
    print(f"Training Isolation Forest on {len(feature_cols)} features...")
    
    # Handle NaNs just in case (though filled in loader)
    X = df[feature_cols].fillna(0)
    
    # Model
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    
    # Predict (-1 is anomaly, 1 is normal)
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly'] = model.predict(X)
    
    return model, df

    return model, df

def generate_anomaly_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-readable explanations for why a record was flagged.
    Strategy: Compare Pincode activity to District Average (Z-Score).
    """
    print("Generating explanations for anomalies...")
    
    # Ensure we have the base columns
    bio_cols = [c for c in df.columns if 'bio_' in c]
    demo_cols = [c for c in df.columns if 'demo_' in c]
    enrol_cols = [c for c in df.columns if 'enrol_' in c]
    
    # Calculate Total Activity per record if not exists
    if 'total_bio' not in df.columns and bio_cols:
        df['total_bio'] = df[bio_cols].sum(axis=1)
    if 'total_demo' not in df.columns and demo_cols:
        df['total_demo'] = df[demo_cols].sum(axis=1)
        
    # Calculate District-Level Statistics (Mean & Std Dev)
    # Group by District and Date? Or just District?
    # Let's do District-level average activity per day
    district_stats = df.groupby('district')[['total_bio', 'total_demo']].agg(['mean', 'std']).reset_index()
    district_stats.columns = ['district', 'mean_bio', 'std_bio', 'mean_demo', 'std_demo']
    
    # Merge stats back to main df
    df = df.merge(district_stats, on='district', how='left')
    
    # Logic for Explanation
    def explain_row(row):
        reasons = []
        
        # Check Biometric Spike
        if row['std_bio'] > 0:
            z_bio = (row['total_bio'] - row['mean_bio']) / row['std_bio']
            if z_bio > 3:
                reasons.append(f"Biometric Updates ({row['total_bio']}) are {z_bio:.1f}x above district avg.")
        
        # Check Demographic Spike
        if row['std_demo'] > 0:
            z_demo = (row['total_demo'] - row['mean_demo']) / row['std_demo']
            if z_demo > 3:
                reasons.append(f"Demographic Updates ({row['total_demo']}) are {z_demo:.1f}x above district avg.")
                
        # Fallback for ML anomaly without clear statistical spike
        if not reasons and row.get('is_anomaly') == -1:
            reasons.append("Unusual combination of update types detected by AI.")
            
        return " | ".join(reasons) if reasons else "Normal"

    # Apply only to anomalies to save time, or all?
    # Applying to all helps see "near misses" too, but let's stick to anomalies for now
    df['anomaly_reason'] = df.apply(explain_row, axis=1)
    
    return df

def load_or_train_model(df):
    model_path = "model_anomaly.pkl"
    # Force retrain if columns changed or to ensure new logic... actually let's just train fresh for now
    model, df = train_anomaly_model(df)
    
    # Generate explanations
    df = generate_anomaly_explanations(df)
    
    return model, df
