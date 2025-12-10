# hr_event_analysis.py
# ---------------------------------------
# 2nd part of HR analysis pipeline:
# Analyze preprocessed HR data around event markers
#
# NOTE: CHANGE the subject ID and raw file path before running!
#
# Generates:
#   1. CSV file with per-event HR features
#   2. Visualization of HR signal with event markers and baseline period
#
# Main steps:
#   1. Loads preprocessed HR data for a specified subject.
#   2. Loads raw data to extract event markers and baseline period timestamps.
#   3. Maps event markers to HR timeline.
#   4. Computes HR features (mean, max, min, delta) around each event.
#   5. Computes baseline mean HR and changes from baseline.
#   6. Saves event-based HR features to a CSV file.
#   7. Plots HR signal with event markers and baseline period.
#
# Author: Tuuli Palomäki
# Created: 20 / 11 / 2025
# ---------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Disable interactive plotting
plt.ioff()

# ---------------- SETTINGS ----------------
subject_id = 7  # NOTE CHANGE subject ID 
samples_pre = 3 
samples_post = 10

# --------------- FILE PATHS ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# CHANGE the raw path file name
raw_path = os.path.join(project_root, "raw_data", "shimmer_data.csv") # NOTE CHANGE the file name
processed_path = os.path.join(script_dir, "preprocessed", f"hr_subject{subject_id}_preprocessed.csv")

# ------------- LOAD DATA ------------------
raw_df = pd.read_csv(raw_path, low_memory=False)
hr_df = pd.read_csv(processed_path)

print(f"[INFO] HR rows: {len(hr_df)}")

# ------ Automatically estimate HR sampling rate from PREPROCESSED data ------
if len(hr_df) > 1 and "Timestamp" in hr_df.columns:
    hr_time = pd.to_numeric(hr_df["Timestamp"], errors="coerce").dropna()
    time_diffs = hr_time.diff().dropna()
    mean_interval = time_diffs.mean()
    sampling_rate = 1 / mean_interval if mean_interval > 0 else np.nan
    print(f"[INFO] Estimated HR sampling rate ~ {sampling_rate:.2f} Hz (mean interval {mean_interval:.2f} s)")
else:
    sampling_rate = np.nan
    print("[WARN] Could not estimate sampling rate — Timestamp column not found.")

# ---------------- EVENTS ------------------
events_raw, event_names = [], []
print("Searching for MarkerEvent lines in the raw CSV...")

with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "MarkerEvent" in line:
            events_raw.append(i)
            name_match = re.search(r"Name:([^;,\"]+)", line)
            if name_match:
                event_names.append(name_match.group(1).strip())
            else:
                event_names.append("Unknown")

if events_raw:
    print(f"[OK] Found {len(events_raw)} MarkerEvent rows.")
else:
    print("[WARN] No MarkerEvent rows found in text search.")

# ---------------- EVENT TIME MAPPING ----------------
print("Matching events to HR timestamps...")

# Build events table by finding MarkerEvent rows in the loaded raw dataframe
# Restrict search to likely event/marker columns to avoid scanning the whole DF
event_cols = [c for c in raw_df.columns if re.search(r'event|marker', c, re.IGNORECASE)]
if not event_cols:
    event_cols = raw_df.columns.tolist()

# Vectorized detection across only the candidate columns
marker_mask = raw_df[event_cols].astype(str).apply(lambda col: col.str.contains('MarkerEvent', case=False, na=False)).any(axis=1)
marker_rows = raw_df.loc[marker_mask]

events_list = []
# candidate timestamp columns in preferred order
ts_candidates = ["Timestamp", "Timestamp CAL", "Timestamp RAW", "Timestamp RAW.1", "Timestamp CAL.1", "System Timestamp CAL", "System Timestamp CAL.1"]

for idx, row in marker_rows.iterrows():
    # find the cell that contains the MarkerEvent text (only within event_cols)
    marker_cell = None
    for c in event_cols:
        if c in row.index:
            val = str(row[c]) if not pd.isna(row[c]) else ''
            if 'markerevent' in val.lower():
                marker_cell = val
                break

    # extract event name: prefer the name found by the file-based text scan (events_raw/event_names)
    name = 'Unknown'
    # Try to map DataFrame row index to file-line indices (events_raw). There may be an off-by-one
    # because file line numbers include header. Try idx, idx+1, idx-1.
    mapped = False
    for candidate in (int(idx), int(idx) + 1, int(idx) - 1):
        if candidate in events_raw:
            pos = events_raw.index(candidate)
            name = event_names[pos]
            mapped = True
            break
    if not mapped:
        # Search entire row for Name:<eventName>
        row_text = " ".join([str(x) for x in row.values])
        m = re.search(r"Name:([^;,\"]+)", row_text)
        if m:
            name = m.group(1).strip()

    # pick the best timestamp value available on that row
    event_time = np.nan
    for tc in ts_candidates:
        if tc in row.index:
            tval = pd.to_numeric(row[tc], errors='coerce')
            if not pd.isna(tval):
                event_time = float(tval)
                break

    events_list.append({
        'raw_row': int(idx),
        'Event Name': name,
        'Event Time Raw': event_time
    })

events = pd.DataFrame(events_list)

# if we didn't find any via dataframe scan, fallback to earlier file-based indices
if events.empty and events_raw:
    events = pd.DataFrame({
        'raw_row': events_raw,
        'Event Name': event_names,
        'Event Time Raw': [np.nan] * len(events_raw)
    })

# Map each event timestamp to nearest preprocessed HR timestamp using searchsorted (faster)
hr_time = pd.to_numeric(hr_df['Timestamp'], errors='coerce').dropna().values
# ensure hr_time is sorted and keep mapping to original indices
order = np.argsort(hr_time)
hr_time_sorted = hr_time[order]

matched_indices = []
matched_times = []
time_diffs = []

for _, ev in events.iterrows():
    et = ev['Event Time Raw']
    if not pd.isna(et) and len(hr_time_sorted) > 0:
        pos = np.searchsorted(hr_time_sorted, et)
        cand = []
        if pos > 0:
            cand.append(pos - 1)
        if pos < len(hr_time_sorted):
            cand.append(pos)
        cand = np.array(cand, dtype=int)
        # choose the candidate with smallest absolute time difference
        diffs = np.abs(hr_time_sorted[cand] - et)
        best = cand[diffs.argmin()]
        idx = int(order[best])
        matched_indices.append(int(idx))
        matched_times.append(float(hr_time[idx]))
        time_diffs.append(float(hr_time[idx] - et))
    else:
        # fallback: proportional map using raw row index
        if len(raw_df) > 0 and len(hr_df) > 0:
            prop = ev['raw_row'] / max(1, len(raw_df) - 1)
            idx = int(prop * (len(hr_df) - 1))
            matched_indices.append(int(idx))
            matched_times.append(float(hr_df['Timestamp'].iloc[idx]))
            time_diffs.append(float(hr_df['Timestamp'].iloc[idx] - np.nan))
        else:
            matched_indices.append(np.nan)
            matched_times.append(np.nan)
            time_diffs.append(np.nan)

events['Event Index'] = matched_indices
events['Matched HR Time'] = matched_times
events['Time Diff (HR - Raw)'] = time_diffs

print(f"[INFO] Mapped {len(events)} events to HR timeline.")

# ------ Detect baseline period ------
baseline_start_time, baseline_end_time = None, None
ts_candidates = ["Timestamp", "Timestamp CAL", "Timestamp RAW"]

with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "BaselineStart" in line and baseline_start_time is None:
            # Find this line in raw_df and extract timestamp
            for idx, row in raw_df.iterrows():
                if "BaselineStart" in str(row.values):
                    for tc in ts_candidates:
                        if tc in row.index:
                            tval = pd.to_numeric(row[tc], errors='coerce')
                            if not pd.isna(tval):
                                baseline_start_time = float(tval)
                                break
                    break
        if "BaselineEnd" in line and baseline_end_time is None:
            # Find this line in raw_df and extract timestamp
            for idx, row in raw_df.iterrows():
                if "BaselineEnd" in str(row.values):
                    for tc in ts_candidates:
                        if tc in row.index:
                            tval = pd.to_numeric(row[tc], errors='coerce')
                            if not pd.isna(tval):
                                baseline_end_time = float(tval)
                                break
                    break
        if baseline_start_time is not None and baseline_end_time is not None:
            break

# ------ Compute baseline mean ------
baseline_mean = None
b_start_idx, b_end_idx = None, None

if baseline_start_time is not None and baseline_end_time is not None and len(hr_df) > 0:
    # Find HR samples within baseline time window
    mask = (hr_df['Timestamp'] >= baseline_start_time) & (hr_df['Timestamp'] <= baseline_end_time)
    baseline_values = hr_df.loc[mask, 'HeartRate'].dropna()
    
    if len(baseline_values) > 0:
        baseline_mean = baseline_values.mean()
        b_start_idx = hr_df[mask].index.min()
        b_end_idx = hr_df[mask].index.max()
        print(f"[OK] Baseline mean HR = {baseline_mean:.2f} BPM ({len(baseline_values)} samples, time {baseline_start_time:.1f}-{baseline_end_time:.1f}s)")
    else:
        print("[WARN] No HR samples in baseline period.")
else:
    print("[WARN] No baseline period found or missing timestamps.")

# ---------------- ANALYSIS ----------------
# Use nearest-sample selection determined at the beginning of the code
print(f"[INFO] Using nearest-sample windows: {samples_pre} samples before, {samples_post} samples after each event.")

results = []

for _, event in events.iterrows():
    event_name = event["Event Name"]
    event_idx = event.get('Event Index', np.nan)

    if pd.isna(event_idx):
        print(f"[WARN] No matched index for event '{event_name}' — skipping")
        continue

    event_idx = int(event_idx)
    start_idx = max(0, event_idx - samples_pre)
    end_idx = min(len(hr_df), event_idx + samples_post)

    segment = hr_df.loc[start_idx:end_idx, 'HeartRate']

    # per-event diagnostics removed to keep output concise

    if len(segment) == 0:
        print(f"[WARN] No HR samples found for event '{event_name}' — skipping")
        continue

    mean_hr = segment.mean()
    max_hr = segment.max()
    min_hr = segment.min()
    delta_hr = max_hr - min_hr
    baseline_hr = baseline_mean if baseline_mean is not None else np.nan
    max_change_from_baseline = max_hr - baseline_hr if not np.isnan(baseline_hr) else np.nan
    mean_change_from_baseline = mean_hr - baseline_hr if not np.isnan(baseline_hr) else np.nan


    results.append({
        "Subject": subject_id,
        "Event": event_name,
        "Mean_HR": round(mean_hr, 2),
        "Max_HR": round(max_hr, 2),
        "Min_HR": round(min_hr, 2),
        "Delta_HR": round(delta_hr, 2),
        "Mean_change_from_Baseline": round(mean_change_from_baseline, 2),
        "Max_change_from_baseline": round(max_change_from_baseline, 2)
    })
        # ---- Summary print per event ----
    print(f"[SUMMARY] {event_name:<15} | samples: {len(segment):3d} | "
          f"Mean HR: {mean_hr:6.2f} | ΔHR: {delta_hr:5.2f} | ")

# ---------------- SAVE RESULTS ----------------
output_dir = os.path.join(script_dir, "analysed_data")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, f"hr_subject{subject_id}_event_features.csv")

results_df = pd.DataFrame(results)
results_df.to_csv(out_file, index=False)
print(f"\nSaved event-based HR analysis: {out_file}")
print(results_df.head())

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(10, 4))

plt.plot(hr_df["HeartRate"].values, label="Heart Rate signal", color="blue", alpha=0.6)

if baseline_mean is not None:
    plt.axhline(y=baseline_mean, color="dodgerblue", linestyle="--",
                alpha=0.7, label=f"Baseline mean ({baseline_mean:.1f} BPM)")

if b_start_idx is not None and b_end_idx is not None and b_end_idx > b_start_idx:
    plt.axvspan(b_start_idx, b_end_idx, color="lightblue", alpha=0.2, label="Baseline period")

for i, (event_idx, event_name) in enumerate(zip(events["Event Index"], events["Event Name"])):
    if i == 0:
        plt.axvline(x=event_idx, color="r", linestyle="--", alpha=0.6, label="Events")
    else:
        plt.axvline(x=event_idx, color="r", linestyle="--", alpha=0.6)

plt.title(f"Subject {subject_id} – Preprocessed HR Signal with Baseline and Event Markers")
plt.xlabel("Samples")
plt.ylabel("Heart Rate (BPM)")
plt.legend()
plt.tight_layout()
plt.show()  # Commented out for batch processing
plt.close('all')  # Explicitly close all figures

