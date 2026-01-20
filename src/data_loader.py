import pandas as pd
import os
import glob
from typing import Dict, List, Optional

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def get_file_paths(category: str) -> List[str]:
    search_path = os.path.join(DATASET_DIR, f"api_data_aadhar_{category}", "*.csv")
    files = glob.glob(search_path)
    if not files:
        print(f"Warning: No files found for category '{category}' in {search_path}")
    return files

def load_category_data(category: str) -> pd.DataFrame:
    files = get_file_paths(category)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def load_and_merge_all() -> pd.DataFrame:
    print("Loading data...")
    bio_df = load_category_data("biometric")
    demo_df = load_category_data("demographic")
    enrol_df = load_category_data("enrolment")
    
    # 1. Standardize Column Names
    # We want uniform age buckets: '0_5', '5_17', '18_plus'
    
    if not enrol_df.empty:
        # Enrolment usually has: age_0_5, age_5_17, age_18_greater (or similar)
        # Let's inspect and rename to standard 'enrol_0_5', 'enrol_5_17', 'enrol_18_plus'
        enrol_df.columns = [c.strip().lower().replace(" ", "_") for c in enrol_df.columns]
        
        # Mapping logic based on observed headers
        rename_map = {}
        for c in enrol_df.columns:
            if 'age_0_5' in c: rename_map[c] = 'enrol_0_5'
            elif 'age_5_17' in c: rename_map[c] = 'enrol_5_17'
            elif 'age_18' in c or 'greater' in c: rename_map[c] = 'enrol_18_plus'
        enrol_df = enrol_df.rename(columns=rename_map)
        
        # Ensure date format
        if 'date' in enrol_df.columns:
            enrol_df['date'] = pd.to_datetime(enrol_df['date'], format='%d-%m-%Y', errors='coerce')


    if not bio_df.empty:
        bio_df.columns = [c.strip().lower().replace(" ", "_") for c in bio_df.columns]
        rename_map = {}
        for c in bio_df.columns:
            if '0_5' in c: rename_map[c] = 'bio_0_5'
            elif '5_17' in c: rename_map[c] = 'bio_5_17'
            elif '17_' in c or '18_' in c: rename_map[c] = 'bio_18_plus'
        bio_df = bio_df.rename(columns=rename_map)
        
        if 'date' in bio_df.columns:
            bio_df['date'] = pd.to_datetime(bio_df['date'], format='%d-%m-%Y', errors='coerce')

    if not demo_df.empty:
        demo_df.columns = [c.strip().lower().replace(" ", "_") for c in demo_df.columns]
        rename_map = {}
        for c in demo_df.columns:
            if '0_5' in c: rename_map[c] = 'demo_0_5'
            elif '5_17' in c: rename_map[c] = 'demo_5_17'
            elif '17_' in c or '18_' in c: rename_map[c] = 'demo_18_plus'
        demo_df = demo_df.rename(columns=rename_map)

        if 'date' in demo_df.columns:
            demo_df['date'] = pd.to_datetime(demo_df['date'], format='%d-%m-%Y', errors='coerce')

    print("Merging data streams...")
    merge_keys = ['date', 'state', 'district', 'pincode']
    
    # Outer join logic to keep all data points
    # Start with Bio
    merged = bio_df
    if merged.empty:
        merged = demo_df
    elif not demo_df.empty:
        merged = pd.merge(merged, demo_df, on=merge_keys, how='outer')
        
    if merged.empty:
        merged = enrol_df
    elif not enrol_df.empty:
        merged = pd.merge(merged, enrol_df, on=merge_keys, how='outer')
            
    merged = merged.fillna(0)
    
    # 2. Add Total Columns for easy analysis
    merged['total_enrol'] = merged.get('enrol_0_5', 0) + merged.get('enrol_5_17', 0) + merged.get('enrol_18_plus', 0)
    merged['total_bio'] = merged.get('bio_0_5', 0) + merged.get('bio_5_17', 0) + merged.get('bio_18_plus', 0)
    merged['total_demo'] = merged.get('demo_0_5', 0) + merged.get('demo_5_17', 0) + merged.get('demo_18_plus', 0)
    
    print(f"Data Pipeline Complete. Loaded {len(merged)} records.")
    return merged
