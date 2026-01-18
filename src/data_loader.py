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
    
    # Normalize columns
    for df in [bio_df, demo_df, enrol_df]:
        if not df.empty:
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    print("Merging data...")
    merge_keys = ['date', 'state', 'district', 'pincode']
    
    # Rename value columns to prevent overlap
    if not bio_df.empty:
        bio_df = bio_df.rename(columns={c: f"bio_{c}" for c in bio_df.columns if c not in merge_keys})
    if not demo_df.empty:
        demo_df = demo_df.rename(columns={c: f"demo_{c}" for c in demo_df.columns if c not in merge_keys})
    if not enrol_df.empty:
        enrol_df = enrol_df.rename(columns={c: f"enrol_{c}" for c in enrol_df.columns if c not in merge_keys})
    
    merged = bio_df
    if not demo_df.empty:
        if merged.empty:
            merged = demo_df
        else:
            merged = pd.merge(merged, demo_df, on=merge_keys, how='outer')
            
    if not enrol_df.empty:
        if merged.empty:
            merged = enrol_df
        else:
            merged = pd.merge(merged, enrol_df, on=merge_keys, how='outer')
            
    merged = merged.fillna(0)
    print(f"Loaded {len(merged)} records.")
    return merged

if __name__ == "__main__":
    load_and_merge_all()
