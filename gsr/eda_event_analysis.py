# ----- eda_event_analysis.py -----
# 2nd Part of the EDA data processing pipeline
# --------------------------------------------------------------
# Event-related analysis of preprocessed
# Galvanic Skin Response (GSR) or Electrodermal Activity (EDA) data.
#
# It should be run after eda_pre-processing.py and before
# eda_visualize.py in the analysis workflow.
#
# NOTE: CHANGE the subject ID, AND raw file path (the iMotions file name) before running.
#
# The script generates:
#   1. CSV file with event-related EDA features:
#      - Number of SCR peaks (SCR_Count)
#      - Maximum SCR peak amplitude (SCR_MaxAmplitude)
#   2. Visualization of preprocessed EDA signal with baseline correction and event markers.
#
# Script flows as:
#   1. Reads preprocessed EDA data for one subject.
#   2. Locates event markers from the raw data file.
#   3. Applies baseline correction using BaselineStart and BaselineEnd markers.
#   4. Creates epochs around each event.
#   5. Extracts SCR features for each event epoch.
#   6. Saves the extracted features to a CSV file.
#   7. Visualizes the preprocessed EDA signal with baseline correction and event markers.  
#
# Author: Tuuli Palomäki
# Created: 10 / 11 / 2025
# --------------------------------------------------------------

import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import re

# ----------- Settings -----------

test_subject = 23       # CHANGE subject ID 
sampling_rate = 20
scaling_factor = sampling_rate / 128  # new_rate / old_rate
# Define epoch window: eg. 1 sec before to 15 sec after event
pre_event = int(1 * sampling_rate)   # 1 sec
post_event = int(15 * sampling_rate) # 15 sec


# ----------- Paths -----------

# Input paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Select the correct preprocessed file for the subject
preprocessed_path = os.path.join(script_dir, "preprocessed", f"eda_subject{test_subject}_preprocessed.csv")
# CHANGE the raw file
raw_path = os.path.join(os.path.dirname(script_dir), "raw_data", "shimmer_data.csv") # CHANGE this to your raw data file

# Output directory (create if missing)
analysed_dir = os.path.join(script_dir, "analysed")
os.makedirs(analysed_dir, exist_ok=True)
# Output file path
output_path = os.path.join(analysed_dir, f"eda_subject{test_subject}_event_features.csv")

print(f"[PATHS] Subject {test_subject}")
print("Preprocessed:", preprocessed_path)
print("Raw:", raw_path)
print("Output:", output_path)

# ----------- Read data -----------
signals = pd.read_csv(preprocessed_path)
raw_df = pd.read_csv(raw_path, low_memory=False)

# ----------- Find MarkerEvent rows -----------

events_raw, event_names = [], []

# Find the MarkerEvent lines in the raw CSV
print("Searching for MarkerEvent lines in the raw CSV...")
with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "MarkerEvent" in line:
            events_raw.append(i)
            # Extract the event Name from e.g. "Name:BaselineStart"
            name_match = re.search(r"Name:([^;,\"]+)", line) 
            if name_match:  
                event_names.append(name_match.group(1).strip())
            else:
                event_names.append("Unknown")

if events_raw:
    print(f"[OK] Found {len(events_raw)} MarkerEvent rows.")
else:
    print("[WARN] No MarkerEvent rows found in text search.")

# Scale event indices to match the downsampled (20 Hz) EDA signal
events = [int(e * scaling_factor) for e in events_raw if e * scaling_factor < len(signals)]
print(f"[OK] Created {len(events)} scaled event indices.")

# ----------- Locate baseline markers -----------

baseline_start_line, baseline_end_line = None, None
# Locate BaselineStart and BaselineEnd lines in the raw data as text
with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f): 
        if "BaselineStart" in line and baseline_start_line is None: 
            baseline_start_line = i
        if "BaselineEnd" in line and baseline_end_line is None:
            baseline_end_line = i
        if baseline_start_line is not None and baseline_end_line is not None:
            break

# ----------- Apply baseline correction -----------

# If baseline markers found, compute median tonic during baseline and subtract from EDA_Tonic
if baseline_start_line is not None and baseline_end_line is not None and "EDA_Tonic" in signals.columns:
    b_start = int(baseline_start_line * scaling_factor)
    b_end = int(baseline_end_line * scaling_factor)
    b_start, b_end = min(b_start, len(signals) - 1), min(b_end, len(signals) - 1)
    # Compute median tonic during baseline period
    baseline_tonic = signals.loc[b_start:b_end, "EDA_Tonic"].median()
    # Subtract baseline from tonic component
    signals["EDA_Tonic_BC"] = signals["EDA_Tonic"] - baseline_tonic
    print(f"[OK] Baseline tonic (median): {baseline_tonic:.3f}")
else:
    print("[WARN] No baseline markers or EDA_Tonic column found; skipping baseline correction.")
    b_start, b_end = None, None

# ----------- Create epochs around events -----------

# Source: https://neuropsychology.github.io/NeuroKit/functions/epochs.html

epochs = {}

# Create epochs dictionary
for i, event_idx in enumerate(events):
    start_idx = max(event_idx - pre_event, 0) 
    end_idx = min(event_idx + post_event, len(signals) - 1) 
    epochs[event_names[i]] = signals.loc[start_idx:end_idx, ["EDA_Phasic"]].copy() # Only phasic component for SCR analysis

print(f"[OK] Created {len(epochs)} epochs from events.")

# ----------- Extract event-related features -----------

features_list = []

for i, (epoch_name, epoch) in enumerate(epochs.items()):
    phasic = epoch["EDA_Phasic"].reset_index(drop=True)

    # NeuroKit2 SCR peak detection
    scr_info = nk.eda_findpeaks(phasic, sampling_rate=sampling_rate, amplitude_min=0.00001)

    # SCR count
    scr_onsets = scr_info.get("SCR_Onsets", [])
    scr_count = len([x for x in scr_onsets if not pd.isna(x)])

    # SCR amplitudes
    heights = scr_info.get("SCR_Height", pd.Series(dtype=float))
    max_amplitude = heights.max() if not heights.empty else 0

    # Append features
    features_list.append({
        "EventName": epoch_name,
        "SCR_Count": scr_count,
        "SCR_MaxAmplitude": max_amplitude,
    })

    # Debug print
    print(f"Epoch {i} ({epoch_name}): SCR_Count={scr_count}, MaxAmp={max_amplitude:.6f}")

# Convert to DataFrame and save
features = pd.DataFrame(features_list)
features.to_csv(output_path, index=False)
print(f"[SAVE] Event-related features saved -> {output_path}")
print(features.head())

# ----------- Clean Event Names -----------

def extract_name(marker_str):
    if pd.isna(marker_str):
        return None
    match = re.search(r"Name:([^;,\"]+)", str(marker_str))
    return match.group(1).strip() if match else marker_str

# Clean up EventName column, if it exists
if not features.empty and "EventName" in features.columns:
    features["EventName"] = features["EventName"].apply(extract_name)

# ----------- Save features -----------

# Save features to CSV
if not features.empty:
    features.to_csv(output_path, index=False)
    print(f"[SAVE] Event-related features saved -> {output_path}")
    # Display first few rows
    print(features.head())
else:
    print("[WARN] No features extracted – nothing saved.")

# ----------- Visualization -----------

# Plot one image to combine everything
# Plot EDA_Clean, Baseline-corrected Tonic, Baseline period, and Event lines
plt.figure(figsize=(12, 6))

# 1. Basic signals
plt.plot(signals["EDA_Clean"], label="Cleaned EDA signal (tonic + phasic)", alpha=0.6)
if "EDA_Tonic_BC" in signals.columns:
    plt.plot(signals["EDA_Tonic_BC"], label="Baseline-corrected tonic", color="darkorange", alpha=0.8)

# 2. Baseline period in light blue
if b_start is not None and b_end is not None:
    plt.axvspan(b_start, b_end, color="lightblue", alpha=0.3, label="Baseline period")

# 3. Event lines (red dashed)
for i, (event, name) in enumerate(zip(events, event_names)):
    if i == 0:
        plt.axvline(x=event, color="r", linestyle="--", alpha=0.6, label="Events")
    else:
        plt.axvline(x=event, color="r", linestyle="--", alpha=0.6)

# 4. Settings and labels
plt.title(f"Subject {test_subject} – Preprocessed EDA Signal with Baseline Correction and Event Markers")
plt.xlabel("Samples (20 Hz)")
plt.ylabel("EDA (µS)")
plt.legend()
plt.tight_layout()
plt.show()


print("\nAnalysis complete for subject", test_subject)

