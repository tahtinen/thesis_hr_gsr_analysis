# ----- eda_visualize.py -----
# 3rd Part of the EDA data processing pipeline
# --------------------------------------------------------------
# This script visualizes event-related EDA (GSR) features extracted
# in the previous analysis stage.
#
# NOTE: Change the subject ID, before running.
#
# The script generates:
#   1. Bar chart of number of GSR peaks (SCR) by event.
#   2. Bar chart of maximum SCR peak amplitude by event.
#
# This script visualizes data for one subject at a time.
#
# Author: Tuuli Palomäki
# Created: 10 / 11 / 2025
# --------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------ Settings ------

subject_id = 22  # CHANGE subject ID 

# ------ Paths ------

script_dir = os.path.dirname(os.path.abspath(__file__))
# Input analysed file path
analysed_path = os.path.join(script_dir, "analysed_data", f"eda_subject{subject_id}_event_features.csv")

print("[PATH] Analysed file:", analysed_path)

# Read analysed EDA CSV file
features = pd.read_csv(analysed_path)
print("Available columns:", list(features.columns))

# --------- Visualization ------

# Figure 1: Number of SCR Peaks by Event
# Plot the SCR_Count column
if "SCR_Count" in features.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(features["EventName"], features["SCR_Count"], color='salmon')
    plt.title("Number of SCR Peaks by Event")
    plt.title(f"Subject {subject_id} – Number of SCR Peaks by Event")
    plt.xlabel("Event Name")
    plt.ylabel("SCR Peak Count")
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()
else:
    print("[WARN] Column 'SCR_Count' not found – skipping peak count plot.")



# Figure 2: EDA Peak Amplitude by Event
# Plot the SCR_MaxAmplitude column
if "SCR_MaxAmplitude" in features.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(features["EventName"], features["SCR_MaxAmplitude"], color='skyblue')
    plt.title(f"Subject {subject_id} – Maximum SCR Amplitude by Event")
    plt.xlabel("Event Name")
    plt.ylabel("Max SCR Amplitude (µS)")
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()
else:
    print("[WARN] Column 'SCR_MaxAmplitude' not found – skipping amplitude plot.")





