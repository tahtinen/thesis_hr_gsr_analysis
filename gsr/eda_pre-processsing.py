# ----- eda_pre-processing.py -----
# 1st Part of the EDA data processing pipeline
# --------------------------------------------------------------
# This script performs preprocessing of Galvanic Skin Response (GSR)
# or Electrodermal Activity (EDA) data exported from iMotions-style CSV files.
#
## Run this script first in the EDA processing workflow, before running
# eda_event_analysis.py and eda_visualize.py. 
#
# NOTE: CHANGE the input file path (file_path) AND subject IDs before running.
#
# The script generates:
#   1. Preprocessed EDA data saved to a CSV file:
#      - Downsampled, cleaned GSR/EDA signal
#      - Tonic and phasic components extracted
#   2. (Optional) Visualization of raw and cleaned signals.
#
# Main steps:
#   1. Reads raw GSR/EDA data from a CSV file.
#   2. Handles missing or zero values and warns about long gaps.
#   3. Downsamples the raw signal (typically from 128 Hz → 20 Hz).
#   4. Cleans the signal using NeuroKit2 (removing artifacts).
#   5. Extracts tonic and phasic components.
#   6. Saves the preprocessed data to a new CSV file
#      (filename includes the subject ID, e.g. eda_subject10_preprocessed.csv).
#   7. Optionally plots the cleaned signals for visual inspection.
#
# Supports automatic processing of two subjects if both GSR columns
# (e.g., "GSR Coductance CAL" and "2GSR Conductance CAL") are present in the same recording file.
# (Subsequent scripts handle one subject at a time.)
#
#
# Author: Tuuli Palomäki
# Created: 10 / 11 / 2025
# --------------------------------------------------------------



import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import numpy as np

# ------ Settings ------
subject_id = [22,23] # CHANGE according to the data file used
original_sampling_rate = 128
target_sampling_rate = 20
gap_threshold_seconds = 5  # warn if missing segment > 5 seconds
min_valid_samples = 50     # minimum valid samples required to proceed

# ------ File paths ------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# CHANGE the file name as needed
file_path = os.path.join(project_root, "raw_data", "shimmer_data.csv")

# ------ Read and prepare data ------
df = pd.read_csv(file_path, low_memory=False)
# Find all calibrated conductance channels (µS)
found_cols = [c for c in df.columns if "GSR Conductance CAL" in c or "2GSR Conductance CAL" in c]

if not found_cols:
    print("No GSR Conductance CAL columns found in the data. Please check the file.")
    exit(1)

# ------ Process both participants ------
for subject_id, eda_col in zip(subject_id, found_cols):
    print(f"\nProcessing {eda_col} → subject {subject_id}")

    # Remove event marker rows (if present)
    if "Event Marker" in df.columns:
        eda_signal = df.loc[df["Event Marker"].isna(), eda_col]
    else:
        eda_signal = df[eda_col]

    # Convert 0 or negative values to NaN (sensor dropout)
    eda_signal = eda_signal.mask(eda_signal <= 0, np.nan)

    # ------ Check if any valid samples remain ------
    valid_count = eda_signal.notna().sum()
    if valid_count < min_valid_samples:
        print(f"Skipping subject {subject_id}: only {valid_count} valid samples left after cleaning.")
        continue

    # ------ Detect missing data and gaps ------
    missing_mask = eda_signal.isna()
    num_missing = missing_mask.sum()

    # Warn about missing data
    if num_missing > 0:
        gap_threshold_samples = int(gap_threshold_seconds * original_sampling_rate) 
        valid_indices = np.where(~missing_mask)[0] 
        # gap_treshold_seconds (at the beginning of the code) determines the long gap definition
        long_gaps = np.sum(np.diff(valid_indices) > gap_threshold_samples) if len(valid_indices) > 1 else 0 
        if long_gaps > 0: # long gaps detected
            print(f"{long_gaps} long gap(s) (> {gap_threshold_seconds}s) detected in {eda_col}.")
        else: # short gaps only
            print(f"Missing or zero values detected: {num_missing} (short gaps).")
    else:
        print("No missing values.")

    # ------ Fill short gaps before resampling ------
    eda_signal = eda_signal.interpolate(method='linear', limit_direction='both')

    # ------ Resample ------
    # Downsampling the signal
    # Source: https://neuropsychology.github.io/NeuroKit/functions/signal.html
    try:
        eda_downsampled = nk.signal_resample(
            eda_signal,
            sampling_rate=original_sampling_rate,
            desired_sampling_rate=target_sampling_rate,
        )
    except Exception as e:
        print(f"Skipping subject {subject_id}: error during resampling → {e}")
        continue

    # ------ Clean the signal ------
    # Source: https://neuropsychology.github.io/NeuroKit/functions/eda.html#
    eda_clean = nk.eda_clean(eda_downsampled, sampling_rate=target_sampling_rate, method="neurokit")

    # ------ Visualize ------
    # Figure 1: Plot raw and cleaned signals
    nk.signal_plot([eda_downsampled, eda_clean],
                   labels=["Raw GSR", "Cleaned GSR"],
                   title=f"GSR Signal Cleaning – Subject {subject_id}")

    # ------ Extract tonic and phasic components ------

    # eda_process function extracts tonic and phasic components from the cleaned signal
    try:
        # Säilytä pienemmät piikit, esim. 0.01 µS minimi-amplitudi
        signals, info = nk.eda_process(
            eda_clean,
            sampling_rate=target_sampling_rate,
            scr_min_amplitude=0.0001  # oletus 0.05 µS → pienempi arvo tunnistaa useammat piikit
            )
    except Exception as e:
        print(f"Skipping subject {subject_id}: error during EDA processing → {e}")
        continue
    
    # Figure 2: Plot the processed EDA signal with tonic and phasic components
    nk.eda_plot(signals, info)
    plt.text(
        x=0.02, y=0.99, s=f"Subject {subject_id}",
        fontsize=10, transform=plt.gcf().transFigure, ha="left", va="top"
    )
    plt.show()

    # ------ Save processed data ------

    # Create the output directories if they don't exist
    processed_dir = os.path.join(script_dir, "preprocessed")
    analysed_dir = os.path.join(script_dir, "analysed")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(analysed_dir, exist_ok=True)

    # Save per-sample signals to processed (do NOT horizontally concatenate SCR events)
    preproc_filename = os.path.join(processed_dir, f"eda_subject{subject_id}_preprocessed.csv")
    signals.to_csv(preproc_filename, index=False)
    print(f"Saved preprocessed signals: {preproc_filename}")

    # If SCR event table exists, save it separately to analysed_data
    if 'SCR' in info and info['SCR'] is not None:
        scr_df = pd.DataFrame(info['SCR'])
        scr_filename = os.path.join(analysed_dir, f"eda_subject{subject_id}_scr_events_raw.csv")
        scr_df.to_csv(scr_filename, index=False)
        print(f"Saved SCR events (raw): {scr_filename} (rows: {len(scr_df)})")
    else:
        print("No SCR events detected by NeuroKit2 for this subject.")

    print(f"Original length: {len(eda_signal)} | Downsampled: {len(eda_downsampled)}")

