import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm   # <-- import tqdm

root = "training"
sets = ["training_setA", "training_setB"]

patients = []
labels = []
for s in sets:
    set_path = os.path.join(root, s)
    patient_files = [f for f in os.listdir(set_path) if f.endswith(".psv")]
    
    for patient_file in tqdm(patient_files, desc=f"Processing {s}", unit="file"):
        file_path = os.path.join(set_path, patient_file)
        df = pd.read_csv(file_path, sep="|")

        # 2. Add missingness indicators
        for col in df.columns:
            if col not in ["SepsisLabel", "ICULOS"]:
                df[col + "_missing"] = df[col].isna().astype(int)

        # 3. Impute missing values (forward-fill, then zeros)
        df = df.ffill().fillna(0)

        patients.append(df.drop("SepsisLabel", axis=1).values)
        labels.append(df["SepsisLabel"].values)

        # Export one patient to CSV for inspection
        out_path = os.path.join("debug_patients", f"{patient_file.replace('.psv','.csv')}")
        os.makedirs("debug_patients", exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"Saved patient {patient_file} to {out_path}")
        break  # stop after first patient


# 1. Load patient PSV files directly with tqdm progress bar
for s in sets:
    set_path = os.path.join(root, s)
    patient_files = [f for f in os.listdir(set_path) if f.endswith(".psv")]
    
    for patient_file in tqdm(patient_files, desc=f"Processing {s}", unit="file"):
        file_path = os.path.join(set_path, patient_file)
        df = pd.read_csv(file_path, sep="|")
        #print("Original columns:", df.columns.tolist())
        #print("Feature count (excluding ICULOS, SepsisLabel):", len(df.columns))
        # 2. Add missingness indicators
        for col in df.columns:
            if col not in ["SepsisLabel", "ICULOS"]:
                df[col + "_missing"] = df[col].isna().astype(int)
        #print("Original columns:", df.columns.tolist())
        #print("Columns after adding indicators:", len(df.columns))  # should be 78


        # 3. Impute missing values (forward-fill within patient, then median fallback)
        medians = df.median(numeric_only=True)
        df = df.ffill().fillna(0)
        
        patients.append(df.drop("SepsisLabel", axis=1).values)
        labels.append(df["SepsisLabel"].values)
        if df.isna().any().any():
            print(f"NaNs still present in {patient_file}")
            print("Columns with NaNs:", df.columns[df.isna().any()].tolist())
            print("NaN counts per column:\n", df.isna().sum())
            print("First 5 rows:\n", df.head())
            break


# 4. Normalize features (global across all patients)
all_data = np.vstack(patients)
means = all_data.mean(axis=0)
stds = all_data.std(axis=0)

stds_safe = np.where((stds == 0) | np.isnan(stds), 1, stds)
means_safe = np.where(np.isnan(means), 0, means)

patients = [(p - means_safe) / stds_safe for p in patients]

# 5. Train/validation split at patient level
train_patients, val_patients, train_labels, val_labels = train_test_split(
    patients, labels, test_size=0.2, random_state=42
)

print(f"Train patients: {len(train_patients)}, Validation patients: {len(val_patients)}")


import pickle

# Bundle everything into a dictionary
clean_data = {
    "train_patients": train_patients,
    "val_patients": val_patients,
    "train_labels": train_labels,
    "val_labels": val_labels,
    "means": means,
    "stds": stds
}

# Save to pickle file
with open("clean_dataset.pkl", "wb") as f:
    pickle.dump(clean_data, f)

print("✅ Clean dataset saved to clean_dataset.pkl")
