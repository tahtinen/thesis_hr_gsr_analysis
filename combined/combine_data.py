# combine_data.py
#
# Last part (4th) of the data processing pipeline.
# --------------------------------------------------------------
# Combines analysed Heart Rate (HR) and Electrodermal Activity (EDA/GSR)
# Performed after analysing Heart Rate (HR) and
# Electrodermal Activity (EDA/GSR) data separately.
#
# Generates:
#   1. Combined CSV files for each subject, merging HR and EDA event features.
#   2. A summary CSV file combining all subjects' data.
#
# The script:
#   1. Reads analysed HR and EDA CSV files for each subject.
#   2. Merges them based on event names.
#   3. Saves individual combined files and a summary file.
#   4. Prints out any events missing in either dataset.
#
# Author: Tuuli Palomäki
# Created: 19 / 11 / 2025
# --------------------------------------------------------------

import pandas as pd
import glob
import os
import re

# --------- Settings and Paths ---------
hr_folder = "HR/analysed_data"
eda_folder = "GSR/analysed_data"
output_folder = "combined_data"
os.makedirs(output_folder, exist_ok=True)

output_summary = "combined_summary.csv"

# --------- Combine Data ---------
hr_files = glob.glob(os.path.join(hr_folder, "hr_subject*_event_features.csv"))
eda_files = glob.glob(os.path.join(eda_folder, "eda_subject*_event_features.csv"))

summary_data = []

print("\n=== Starting HR + EDA data combination ===\n")

for hr_file in hr_files:
    subject_id = int(re.findall(r"subject(\d+)", hr_file)[0])
    print(f"Processing Subject {subject_id}...")

    # Find matching EDA file
    matching_eda = [f for f in eda_files if f"subject{subject_id}" in f]
    if not matching_eda:
        print(f"No matching EDA file found for Subject {subject_id}. Skipping.\n")
        continue

    eda_file = matching_eda[0]

    # 
    hr_df = pd.read_csv(hr_file)
    eda_df = pd.read_csv(eda_file)

    # Rename Event column to EventName for merging
    hr_df.rename(columns={"Event": "EventName"}, inplace=True)

    # Check for matching events
    hr_events = set(hr_df["EventName"].dropna())
    eda_events = set(eda_df["EventName"].dropna())
    matched_events = hr_events & eda_events
    missing_in_eda = hr_events - eda_events
    missing_in_hr = eda_events - hr_events

    print(f" Found {len(matched_events)} matching events")
    if missing_in_eda:
        print(f"Events missing in EDA: {', '.join(sorted(missing_in_eda))}")
    if missing_in_hr:
        print(f"Events missing in HR: {', '.join(sorted(missing_in_hr))}")

    # Merge dataframes on EventName
    merged = pd.merge(hr_df, eda_df, on="EventName", how="outer", suffixes=("_HR", "_EDA"))

    # Name the columns more clearly
    if "EDA_SCR" in merged.columns:
        merged.rename(columns={"EDA_SCR": "Number_of_SCR_Peaks"}, inplace=True)

    # --------- Select and order columns for output ---------
    summary_cols = [
        "Subject",
        "EventName",
        "Mean_HR",
        "Delta_HR",
        "Change_from_Baseline",
        "SCR_Peak_Amplitude",
        "EDA_Peak_Amplitude",
        "Number_of_SCR_Peaks"
    ]
    merged = merged[[c for c in summary_cols if c in merged.columns]]

    # Add empty columns for future annotations
    merged["Combined_Physiological_Reaction"] = ""
    merged["Detected_Emotion"] = ""

    # --------- Save individual combined file ---------
    subject_output = os.path.join(output_folder, f"combined_subject{subject_id}.csv")
    merged.to_csv(subject_output, index=False)
    print(f"Saved: {subject_output}\n")

    summary_data.append(merged)

# --------- Create and save summary file ---------
if summary_data:
    combined_df = pd.concat(summary_data, ignore_index=True)
    combined_df.to_csv(output_summary, index=False)
    print(f"Combined summary saved to: {output_summary}")
else:
    print("No data combined – check file names or paths.")
