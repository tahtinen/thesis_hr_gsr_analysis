# ----- generate_realistic_iMotions_data.py -----
# Creates a *synthetic physiological dataset* for testing and method development.
# The structure imitates an iMotions-style CSV export, including inline "MarkerEvent" rows.
#
# Features:
# - Two simulated subjects, each with GSR (skin conductance) and HR (heart rate) data
# - 10 minutes of recording at 128 Hz sampling rate
# - Baseline and five stimulus events inserted at realistic time points
# - GSR and HR signals include tonic/phasic variation and random noise
# - Output format matches typical iMotions CSV column structure
#
# Columns include:
#   'GSR Conductance CAL', '2GSR Conductance CAL' – conductance in microsiemens (two subjects)
#   'Heart Rate', 'Heart Rate.1' – heart rate in BPM
#   'IBI PPG ALG', 'IBI PPG ALG.1' – inter-beat interval in ms
#   'GSR RAW', 'GSR RAW.1' – raw GSR values
#   'MarkerEvent' rows with event names (BaselineStart, Stimuli1, etc.)
#
# This script is intended for testing EDA (electrodermal activity) preprocessing
# and event-related analysis pipelines, such as eda_pre-processing.py and eda_event_analysis.py.
#
# Note:
#   This is *synthetic data only* — no personal or experimental data is included.
#   The "iMotions-style" structure refers only to the column naming and event format,
#   not to any proprietary content.
#
# Usage:
#   Run this script to generate a test CSV under /synthetic_data/
#   The resulting file can be processed directly with eda_pre-processing.py
#   and eda_event_analysis.py for validation and debugging.

#
# Author: Tuuli Palomäki
# Created: 10 / 11 / 2025
# Source prompt: AI-assisted generation using OpenAI ChatGPT (2025)
# --------------------------------------------------------------


import numpy as np
import pandas as pd
import os
import random

# ---------------- Settings ----------------
fs = 128                   # sampling frequency (Hz)
duration_min = 10
duration_s = duration_min * 60
n_samples = fs * duration_s
timestamps = np.linspace(0, duration_s, n_samples)

# ---------------- Output ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "raw_data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "two_subjects_GSR_HR_data.csv")

# ---------------- Generate synthetic signals ----------------
def generate_person_series():
    tonic = 5 + 0.5 * np.sin(timestamps / 200 * 2 * np.pi)
    noise = np.random.normal(0, 0.05, n_samples)
    gsr = tonic + noise
    for i in np.random.randint(0, n_samples - 100, 35):
        gsr[i:i+25] += np.linspace(0, np.random.uniform(0.2, 1.0), 25)
        gsr[i+25:i+60] -= np.linspace(0, np.random.uniform(0.2, 1.0), 35)
        
    # --- HARVA, “lopullinen” HR (BPM) vain ~5.6 s välein ---
    hr = np.full(n_samples, np.nan, dtype=float)
    step = int(fs * 5.6)  # Polar-tyylinen päivitysväli
    idx = np.arange(0, n_samples, step)
    hr_vals = 75 + np.random.normal(0, 8, size=len(idx))  # esim. 60–120 BPM vaihtelua
    hr_vals = np.clip(hr_vals, 50, 150)                   # rajaa realistiseksi
    hr[idx] = np.round(hr_vals, 2)                        # “lopulliset” arvot, esim. 70.10

    # Luo IBI vain niille kohdille, joissa HR on mitattu
    ibi = np.full(n_samples, np.nan, dtype=float)
    ibi[idx] = np.round(60000 / hr_vals, 2)

    gsr_raw = (gsr * 10000).astype(int)
    return gsr_raw, hr, ibi

gsr1, hr1, ibi1 = generate_person_series()
gsr2, hr2, ibi2 = generate_person_series()

# ---------------- Base dataframe ----------------
# This matches the multi-column structure seen in your real CSV excerpt
data = pd.DataFrame({
    "Row": np.arange(1, n_samples + 1),
    "Timestamp": timestamps,
    "EventSource": ["Screen recording 1"] * n_samples,
    "SlideEvent": [np.nan] * n_samples,
    "StimType": [np.nan] * n_samples,
    "Duration": [np.nan] * n_samples,
    "CollectionPhase": [np.nan] * n_samples,
    "SourceStimuliName": ["Screen recording 1"] * n_samples,
    "EventSource.1": [np.nan] * n_samples,
    "Heart Rate": hr1,
    "R-R interval": ibi1,
    "Energy expended": [np.nan] * n_samples,
    "Contact": [np.nan] * n_samples,
    "EventSource.2": [np.nan] * n_samples,
    "Heart Rate.1": hr2,
    "R-R interval.1": ibi2,
    "Energy expended.1": [np.nan] * n_samples,
    "Contact.1": [np.nan] * n_samples,
    "EventSource.3": [np.nan] * n_samples,
    "InputEventSource": [np.nan] * n_samples,
    "Data": [np.nan] * n_samples,
    "StimType.1": [np.nan] * n_samples,
    "EventSource.4": [np.nan] * n_samples,
    "Event Group": [np.nan] * n_samples,
    "Event Label": [np.nan] * n_samples,
    "Event Text": [np.nan] * n_samples,
    "Event Index": [np.nan] * n_samples,
    "EventSource.5": [np.nan] * n_samples,
    "SampleNumber": np.arange(1, n_samples + 1),
    "Timestamp RAW": np.arange(1000000, 1000000 + n_samples),
    "Timestamp CAL": np.linspace(100, 400, n_samples),
    "System Timestamp CAL": np.linspace(1757412894000, 1757412894000 + n_samples, n_samples),
    "VSenseBatt RAW": np.random.randint(2700, 2900, n_samples),
    "VSenseBatt CAL": np.random.uniform(4000, 4100, n_samples),
    "Internal ADC A13 PPG RAW": np.random.randint(2000, 3000, n_samples),
    "Internal ADC A13 PPG CAL": np.random.uniform(1800, 2000, n_samples),
    
    # ----- GSR signals for two subjects -----
    "GSR RAW": gsr1,
    "GSR Resistance CAL": np.random.uniform(6000, 6500, n_samples),
    "GSR Conductance CAL": gsr1 / 10000,  # subject 1
    
    "2GSR RAW": gsr2,
    "2GSR Resistance CAL": np.random.uniform(6000, 6500, n_samples),
    "2GSR Conductance CAL": gsr2 / 10000,  # subject 2
    
    # ----- Extra signals -----
    "IBI PPG ALG": ibi1,
    "Packet reception rate RAW": [100] * n_samples,
})


# ---------------- Insert inline MarkerEvent rows ----------------
def insert_marker(df, idx, name):
    """Insert a text line that looks exactly like a real iMotions MarkerEvent row."""
    marker_text = f"1,MarkerEvent,Id:L;Name:{name};Key:Q,TestImage"
    # Insert an empty row with only the MarkerEvent info in the correct column position
    insert_data = {
        "Row": np.nan,
        "Timestamp": df.loc[idx, "Timestamp"],
        "EventSource": "Screen recording 1",
        "SlideEvent": np.nan,
        "StimType": np.nan,
        "Duration": np.nan,
        "CollectionPhase": np.nan,
        "SourceStimuliName": "Screen recording 1",
        "EventSource.1": np.nan,
        "Heart rate": np.nan,
        "R-R interval": np.nan,
        "Energy expended": np.nan,
        "Contact": np.nan,
        "EventSource.2": np.nan,
        "Heart rate.1": np.nan,
        "R-R interval.1": np.nan,
        "Energy expended.1": np.nan,
        "Contact.1": np.nan,
        "EventSource.3": np.nan,
        "InputEventSource": 1,
        "Data": marker_text,
        "StimType.1": "TestImage"
    }
    before = df.iloc[:idx]
    after = df.iloc[idx:]
    return pd.concat([before, pd.DataFrame([insert_data]), after], ignore_index=True)

# Event timings (seconds → sample index)
marker_events = [
    ("BaselineStart", int(30 * fs)),
    ("BaselineEnd", int(210 * fs)),
    ("Stimuli1", int(300 * fs)),
    ("Stimuli2", int(360 * fs)),
    ("Stimuli3", int(420 * fs)),
    ("Stimuli4", int(480 * fs)),
    ("Stimuli5", int(540 * fs)),
]

# Insert in reverse order to keep indices valid
for name, idx in sorted(marker_events, key=lambda x: x[1], reverse=True):
    data = insert_marker(data, idx, name)

# ---------------- Save ----------------
data.to_csv(output_path, index=False)
print(f"Realistic iMotions-style CSV saved to:\n{output_path}")
print(f"Total rows (including MarkerEvent lines): {len(data)}")
print("Inserted MarkerEvent names:", [n for n, _ in marker_events])

