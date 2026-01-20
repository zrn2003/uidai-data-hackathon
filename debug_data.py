import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_and_merge_all, load_category_data

print("--- DEBUGGING DATA LOADER ---")

# 1. Test Category Loading
print("\n1. Testing Individual Categories:")
for cat in ['biometric', 'demographic', 'enrolment']:
    try:
        df = load_category_data(cat)
        print(f"   [{cat.upper()}] Rows: {len(df)}")
        if not df.empty:
            print(f"     Columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"   [{cat.upper()}] FAILED: {e}")

# 2. Test Merge
print("\n2. Testing Full Merge:")
try:
    # Force reload to bypass potentially broken cache
    merged_df = load_and_merge_all(force_reload=True)
    print(f"   [MERGED] Shape: {merged_df.shape}")
    print(f"   [MERGED] Columns: {list(merged_df.columns)}")
    print(f"   [MERGED] Sample:\n{merged_df.head()}")
    
    if merged_df.empty:
        print("   [CRITICAL] Merged DataFrame is EMPTY!")
except Exception as e:
    print(f"   [MERGED] FAILED: {e}")
