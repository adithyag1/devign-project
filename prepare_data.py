import pandas as pd
from datasets import load_dataset
import os
import shutil

# 1. Load the dataset from HuggingFace
print("Step 1: Downloading DetectVul/devign from HuggingFace...")
dataset = load_dataset("DetectVul/devign", split='train') 
df = pd.DataFrame(dataset)

# 2. Filter by function length (Joern safety)
# Keeps functions between 100 and 5000 characters to prevent Joern hangs
df = df[df['func'].str.len().between(100, 5000)]

# 3. Balance the classes (7,500 Vulnerable, 7,500 Safe = 15,000 Total)
print("Step 2: Balancing classes...")
n_samples = 7500
vuln = df[df['target'] == 1].sample(n=n_samples, random_state=42)
safe = df[df['target'] == 0].sample(n=n_samples, random_state=42)
df_final = pd.concat([vuln, safe]).sample(frac=1, random_state=42)

# 4. Add 'project' column (needed for your recovery code in main.py)
df_final['project'] = 'hf_devign'

# 5. SAVE TO data/raw/
print("Step 3: Saving to data/raw/dataset.json...")
os.makedirs('data/raw', exist_ok=True)
df_final.to_json('data/raw/dataset.json', orient='records')

# 6. CRITICAL CLEANUP: Delete old processed data
# You MUST delete old CPGs and inputs so they don't mix with the new data
print("Step 4: Cleaning up old processed data...")
shutil.rmtree('data/cpg', ignore_errors=True)
shutil.rmtree('data/input', ignore_errors=True)
if os.path.exists('data/split_indices.pkl'):
    os.remove('data/split_indices.pkl')

print(f"\nSUCCESS! {len(df_final)} clean samples ready for 'python main.py -c'")
