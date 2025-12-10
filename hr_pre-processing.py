# ----- hr_pre-processing.py -----
# ----------------------------------------------------------------------------
# 1st Part of the HR data processing pipeline
# ----------------------------------------------------------------------------
# Generates:
#   1. Preprocessed HR data saved to a CSV file:
#      - Cleaned Heart Rate signal
#   2. Visualization of raw and cleaned signals.
#
# NOTE: CHANGE the Subject IDs and input file path before running!
# Also, make sure to have the subjects in correct order in the input file. 
# Who'se HR is first/second in the CSV?
#
# Main steps:
#   1. Reads raw HR data from (iMotions) CSV file.
#   2. Handles missing or zero values and warns about long gaps.
#   3. Cleans the signal (interpolating short gaps).
#   4. Saves the preprocessed data to a new CSV file
#      (filename includes the subject ID, e.g. hr_subject10_preprocessed.csv).
#   5. Plots the raw and cleaned signals for visual inspection.
# ----------------------------------------------------------------------------
# NOTE: CHANGE the input file path (file_path) AND subject IDs before running.
# Supports automatic processing of two subjects if both HR columns
# (e.g., "HEART RATE" and "2HEART RATE") are present in the same recording file.
# (Subsequent scripts handle one subject at a time.)
#
# Author: Tuuli Palomäki
# Created: 15 / 11 / 2025

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ------ Settings ------
subject_ids = [7, 8] # NOTE CHANGE according to the data file used, NOTE the order of subjects!
gap_threshold_seconds = 12  # warn if a missing segment > 12 seconds

# ------ File paths ------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "raw_data", "shimmer_data.csv") # NOTE CHANGE the file name as needed

# ------ Read and prepare data ------
df = pd.read_csv(file_path, low_memory=False)
print(f"Loaded {len(df)} rows")

hr_cols = [c for c in df.columns if "HEART RATE" in c.upper()]
if not hr_cols:
    # Fallback to "Heart rate" (case-insensitive)
    hr_cols = [c for c in df.columns if "heart rate" in c.lower()]
if not hr_cols:
    raise ValueError("No heart rate columns found.")
print(f"Found heart rate columns: {hr_cols}")

# Drop rows where ALL HR columns are NaN (keep only rows with data)
df = df.dropna(subset=hr_cols, how='all')
print(f"After filtering NaN rows: {len(df)} rows with HR data")

# ------ Estimate sampling rate ------
timestamp_cols = [c for c in df.columns if "TIME" in c.upper() or "STAMP" in c.upper()]
if timestamp_cols:
    ts_col = timestamp_cols[0]
    time_series = pd.to_numeric(df[ts_col], errors="coerce").dropna()
    time_diffs = time_series.diff().dropna()
    mean_interval = time_diffs.mean()
    sampling_rate = 1 / mean_interval
    print(f"[INFO] Estimated HR sampling rate ~ {sampling_rate:.2f} Hz (mean interval {mean_interval:.2f} s)")
else:
    sampling_rate = np.nan
    print("[WARN] Could not estimate sampling rate — no timestamp column found.")

# ------ Process each participant ------
for sid, hr_col in zip(subject_ids, hr_cols):
    print(f"\nProcessing {hr_col} -> subject {sid}")
    
    # Select HR column, ignore event markers if present
    df_hr = df[["Timestamp", hr_col]].copy()

    # --- Drop rows without HeartRate values to remove 128 Hz empty rows ---
    df_hr = df_hr.dropna(subset=[hr_col]).reset_index(drop=True)

    # --- Save raw copy before modifications ---
    hr_signal_raw = df_hr[hr_col].copy()
    
    # Replace 0 with NaN
    df_hr[hr_col] = df_hr[hr_col].replace(0, np.nan)

    # Use cleaned signal
    hr_signal = df_hr[hr_col]

    # Replace 0 with NaN
    hr_signal = hr_signal.replace(0, np.nan)

    # Detect missing data
    missing_mask = hr_signal.isna()
    num_missing = missing_mask.sum()
    if num_missing > 0:
        gap_threshold_samples = int(gap_threshold_seconds * sampling_rate)
        gap_lengths = np.diff(np.where(~missing_mask)[0])
        long_gaps = np.sum(gap_lengths > gap_threshold_samples)
        if long_gaps > 0:
            print(f"[WARN] {long_gaps} long gaps (> {gap_threshold_seconds}s) detected in {hr_col}.")
        else:
            print(f"[INFO] Missing values detected: {num_missing} (all short gaps).")
    else:
        print("[OK] No missing values detected.")

    # ------ Clean and preprocess ------
    hr_signal_cleaned = hr_signal.interpolate(limit=1)  # fill short gaps
    hr_signal_cleaned = hr_signal_cleaned.where(~hr_signal_cleaned.isna(), np.nan)  # keep true NaNs
    
    # ------ Save processed file ------
    output_dir = os.path.join(script_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"hr_subject{sid}_preprocessed.csv")

    if "Timestamp" in df.columns:
        out_df = pd.DataFrame({
            "Timestamp": df_hr["Timestamp"].values,
            "HeartRate": hr_signal_cleaned.values
        })
    else:
        out_df = pd.DataFrame({
            "HeartRate": hr_signal_cleaned
        })

    out_df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(out_df)} rows)")


    # ------ Visualize raw + cleaned ------
    print(f"[DEBUG] Plotting {len(hr_signal)} real HR points for subject {sid}")

    plt.figure(figsize=(11, 5))

    plt.subplot(2, 1, 1)
    plt.plot(hr_signal_raw, color="blue", alpha=0.7)
    plt.axhline(hr_signal.mean(), color="red", linestyle="--", label="Mean HR")
    plt.title(f"Raw Heart Rate Signal – Subject {sid}")
    plt.xlabel("")
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(hr_signal_cleaned, color="green", alpha=0.7)
    plt.axhline(hr_signal_cleaned.mean(), color="red", linestyle="--", label="Mean HR (Cleaned)")
    plt.title(f"Cleaned Heart Rate Signal – Subject {sid}")
    plt.xlabel("Samples")
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.show() 

print("\nAll subjects processed successfully!")

