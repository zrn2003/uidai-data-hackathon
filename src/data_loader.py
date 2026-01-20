import pandas as pd
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def load_files_from_folder(folder_name):
    path = os.path.join(DATASET_DIR, folder_name, "*.csv")
    files = glob.glob(path)
    if not files:
        logging.warning(f"No files found in {folder_name}")
        return pd.DataFrame()
    
    logging.info(f"Loading {len(files)} files from {folder_name}...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize headers: lowercase, strip whitespace
            df.columns = [c.strip().lower() for c in df.columns]
            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def normalize_state_names(df):
    if 'state' not in df.columns:
        return df
    
    # 1. Basic cleaning: Strip and Title Case
    df['state'] = df['state'].astype(str).str.strip().str.title()
    
    # 2. Canonical Mapping for all 36 States/UTs
    state_mapping = {
        # West Bengal
        'West Bengal': 'West Bengal', 'West Bangal': 'West Bengal', 'Wb': 'West Bengal', 'Westbengal': 'West Bengal',
        
        # Odisha
        'Odisha': 'Odisha', 'Orissa': 'Odisha', 'Orisa': 'Odisha',
        
        # Jammu & Kashmir
        'Jammu And Kashmir': 'Jammu and Kashmir', 'Jammu & Kashmir': 'Jammu and Kashmir', 'J&K': 'Jammu and Kashmir', 'Jnk': 'Jammu and Kashmir',
        
        # Andaman & Nicobar
        'Andaman And Nicobar Islands': 'Andaman and Nicobar Islands', 'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
        'Andaman And Nicobar': 'Andaman and Nicobar Islands', 'A & N Islands': 'Andaman and Nicobar Islands',
        
        # Dadra & Nagar Haveli and Daman & Diu (Merged UT)
        'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu', 'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu', 'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'DNH': 'Dadra and Nagar Haveli and Daman and Diu', 'D&NH': 'Dadra and Nagar Haveli and Daman and Diu',
        
        # Delhi
        'Delhi': 'NCT of Delhi', 'New Delhi': 'NCT of Delhi', 'Nct Of Delhi': 'NCT of Delhi', 'Delhi Nct': 'NCT of Delhi',
        
        # Puducherry
        'Puducherry': 'Puducherry', 'Pondicherry': 'Puducherry', 'Pondi': 'Puducherry',
        
        # Uttarakhand
        'Uttarakhand': 'Uttarakhand', 'Uttaranchal': 'Uttarakhand', 'Uk': 'Uttarakhand',
        
        # Uttar Pradesh
        'Uttar Pradesh': 'Uttar Pradesh', 'Up': 'Uttar Pradesh', 'Uttarpradesh': 'Uttar Pradesh',
        
        # Telangana
        'Telangana': 'Telangana', 'Telengana': 'Telangana',
        
        # Chhattisgarh
        'Chhattisgarh': 'Chhattisgarh', 'Chhatisgarh': 'Chhattisgarh', 'Cg': 'Chhattisgarh',
        
        # Maharashtra
        'Maharashtra': 'Maharashtra', 'Mh': 'Maharashtra',
        
        # Madhya Pradesh
        'Madhya Pradesh': 'Madhya Pradesh', 'Mp': 'Madhya Pradesh',
        
        # Tamil Nadu
        'Tamil Nadu': 'Tamil Nadu', 'Tamilnadu': 'Tamil Nadu', 'Tn': 'Tamil Nadu',
        
        # Andhra Pradesh
        'Andhra Pradesh': 'Andhra Pradesh', 'Andhrapradesh': 'Andhra Pradesh', 'Ap': 'Andhra Pradesh',
        
        # Arunachal Pradesh
        'Arunachal Pradesh': 'Arunachal Pradesh', 'Arunachalpradesh': 'Arunachal Pradesh',
        
        # Himachal Pradesh
        'Himachal Pradesh': 'Himachal Pradesh', 'Himachalpradesh': 'Himachal Pradesh', 'Hp': 'Himachal Pradesh',
        
        # Karnataka
        'Karnataka': 'Karnataka', 'Ka': 'Karnataka',
        
        # Kerala
        'Kerala': 'Kerala', 'Kl': 'Kerala',
        
        # Ladakh
        'Ladakh': 'Ladakh',
        
        # Lakshadweep
        'Lakshadweep': 'Lakshadweep',
        
        # Goa
        'Goa': 'Goa',
        
        # Gujarat
        'Gujarat': 'Gujarat', 'Gujrat': 'Gujarat',
        
        # Haryana
        'Haryana': 'Haryana',
        
        # Punjab
        'Punjab': 'Punjab',
        
        # Rajasthan
        'Rajasthan': 'Rajasthan', 'Rajsthan': 'Rajasthan',
        
        # Sikkim
        'Sikkim': 'Sikkim',
        
        # Tripura
        'Tripura': 'Tripura',
        
        # Assam
        'Assam': 'Assam', 'Asam': 'Assam',
        
        # Bihar
        'Bihar': 'Bihar',
        
        # Chandigarh
        'Chandigarh': 'Chandigarh',
        
        # Jharkhand
        'Jharkhand': 'Jharkhand',
        
        # Manipur
        'Manipur': 'Manipur',
        
        # Meghalaya
        'Meghalaya': 'Meghalaya',
        
        # Mizoram
        'Mizoram': 'Mizoram',
        
        # Nagaland
        'Nagaland': 'Nagaland'
    }
    
    # Apply mapping (if key exists, replace; else keep original)
    df['state'] = df['state'].replace(state_mapping)
    return df

def clean_dataframe(df, data_type):
    if df.empty:
        return df

    # Parse Dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        # Drop rows with invalid dates if critical, or keep them?
        # Let's drop empty dates as they break time-series
        df = df.dropna(subset=['date'])
        
    # NORMALIZE STATE NAMES
    df = normalize_state_names(df)

    # Numeric conversions
    # Identify numeric columns (usually start with 'age' or 'bio' or 'demo' or just numbers)
    # We will rename important columns to standard names
    
    # Enrolment Mapping
    if data_type == 'enrolment':
        # Look for age columns
        # Observed: age_0_5, age_5_17...
        rename_map = {}
        for c in df.columns:
            if '0' in c and '5' in c and 'age' in c: rename_map[c] = 'enrol_0_5'
            elif '5' in c and '17' in c and 'age' in c: rename_map[c] = 'enrol_5_17'
            elif '18' in c and 'age' in c: rename_map[c] = 'enrol_18_plus'
        df = df.rename(columns=rename_map)
        
    # Biometric Mapping
    if data_type == 'biometric':
        rename_map = {}
        for c in df.columns:
            if '0' in c and '5' in c: rename_map[c] = 'bio_0_5'
            elif '5' in c and '17' in c: rename_map[c] = 'bio_5_17'
            elif '17' in c or '18' in c: rename_map[c] = 'bio_18_plus'
        df = df.rename(columns=rename_map)

    # Demographic Mapping
    if data_type == 'demographic':
        rename_map = {}
        for c in df.columns:
            if '0' in c and '5' in c: rename_map[c] = 'demo_0_5'
            elif '5' in c and '17' in c: rename_map[c] = 'demo_5_17'
            elif '17' in c or '18' in c: rename_map[c] = 'demo_18_plus'
        df = df.rename(columns=rename_map)
        
    return df

def aggregate_data(df, prefix):
    # Aggregating by Pincode + Date to reduce size
    if df.empty: return pd.DataFrame()
    
    group_cols = ['date', 'state', 'district', 'pincode']
    # Filter only available group cols
    group_cols = [c for c in group_cols if c in df.columns]
    
    # Identify value columns (the ones we renamed)
    value_cols = [c for c in df.columns if prefix in c]
    
    if not value_cols:
        return pd.DataFrame()
        
    # GroupBy
    agg_df = df.groupby(group_cols)[value_cols].sum().reset_index()
    return agg_df

def load_processed_data():
    """
    Main entry point. Loads, Cleans, Aggregates, and Joins.
    """
    # 1. Load Raw
    enrol = load_files_from_folder("api_data_aadhar_enrolment")
    bio = load_files_from_folder("api_data_aadhar_biometric")
    demo = load_files_from_folder("api_data_aadhar_demographic")
    
    # 2. Clean & Normalize
    enrol = clean_dataframe(enrol, 'enrolment')
    bio = clean_dataframe(bio, 'biometric')
    demo = clean_dataframe(demo, 'demographic')
    
    # 3. Aggregate (The secret sauce for speed)
    # We reduce 100M rows to ~Total Pincodes * Days
    enrol_agg = aggregate_data(enrol, 'enrol')
    bio_agg = aggregate_data(bio, 'bio')
    demo_agg = aggregate_data(demo, 'demo')
    
    # 4. Merge
    # We use 'date', 'state', 'district', 'pincode' as keys
    join_keys = ['date', 'state', 'district', 'pincode']
    
    # Start with the largest or Enrolment
    merged = enrol_agg
    
    if merged.empty:
        merged = bio_agg
    elif not bio_agg.empty:
        merged = pd.merge(merged, bio_agg, on=join_keys, how='outer')
        
    if merged.empty:
        merged = demo_agg
    elif not demo_agg.empty:
        merged = pd.merge(merged, demo_agg, on=join_keys, how='outer')
        
    # Fill NAs
    merged = merged.fillna(0)
    
    # 5. Grand Totals
    merged['total_enrol'] = merged.filter(like='enrol').sum(axis=1)
    merged['total_bio'] = merged.filter(like='bio').sum(axis=1)
    merged['total_demo'] = merged.filter(like='demo').sum(axis=1)
    
    logging.info(f"Final Merged Data Shape: {merged.shape}")
    return merged

if __name__ == "__main__":
    df = load_processed_data()
    print(df.head())
    print(df.info())
