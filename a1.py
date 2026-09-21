# ============================================================
# ANOMALY SCORING
# Predictive Maintenance Project
# Senior Data Analyst Level
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Feature_Engineered_Predictive_Maintenance.csv")

# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [
    "Temperature_Deviation",
    "Vibration_Deviation",
    "Power_Deviation",
    "Vibration_Power_Ratio",
    "Error_Escalation",
    "Maintenance_Score_Decay",
    "Temp_Volatility",
    "Vibration_Volatility",
    "Power_Efficiency",
    "Quality_Loss_Index",
    "Network_Instability_Index",
    "Risk_Score",
]

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df[feature_columns] = df[feature_columns].fillna(0)

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[feature_columns])

# ============================================================
# ISOLATION FOREST MODEL
# ============================================================

iso_forest = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)

# ============================================================
# TRAIN MODEL
# ============================================================

iso_forest.fit(X_scaled)

# ============================================================
# PREDICT ANOMALIES
# ============================================================

df["Anomaly_Label"] = iso_forest.predict(X_scaled)

# ============================================================
# ANOMALY STATUS
# ============================================================

df["Anomaly_Status"] = np.where(df["Anomaly_Label"] == -1, "Anomaly", "Normal")

# ============================================================
# ANOMALY SCORE
# ============================================================

# IMPORTANT:
# Lower score = more abnormal

df["Anomaly_Score"] = iso_forest.decision_function(X_scaled)

# ============================================================
# CONVERT TO POSITIVE ANOMALY INDEX
# ============================================================

# Higher value = more abnormal

df["Anomaly_Index"] = -1 * df["Anomaly_Score"]

# ============================================================
# NORMALIZED ANOMALY SCORE  (0–100 scale)
# ============================================================

# Converts Anomaly_Index to a percentile rank
# 100 = most abnormal   |   0 = most normal
# This matches project spec: Higher score = more abnormal

# One line addition — does not affect any existing column
df["Normalized_Anomaly_Score"] = (df["Anomaly_Index"].rank(pct=True) * 100).round(2)

# ============================================================
# ANOMALY SEVERITY CLASSIFICATION
# ============================================================

conditions = [
    (df["Anomaly_Index"] < 0.05),
    (df["Anomaly_Index"] >= 0.05) & (df["Anomaly_Index"] < 0.15),
    (df["Anomaly_Index"] >= 0.15),
]

severity_labels = ["Low", "Medium", "Critical"]

df["Anomaly_Severity"] = np.select(conditions, severity_labels, default="Low")

# ============================================================
# TOP ANOMALY RECORDS
# ============================================================

top_anomalies = df.sort_values(by="Anomaly_Index", ascending=False)

print("\n==============================")
print("TOP ANOMALY RECORDS")
print("==============================")

print(
    top_anomalies[
        [
            "Machine_ID",
            "Temperature_C",
            "Vibration_Hz",
            "Risk_Score",
            "Anomaly_Index",
            "Anomaly_Severity",
        ]
    ].head(10)
)

# ============================================================
# MACHINE-WISE ANOMALY SUMMARY
# ============================================================

machine_anomaly_summary = (
    df.groupby("Machine_ID")["Anomaly_Index"].mean().sort_values(ascending=False)
)

print("\n==============================")
print("TOP ANOMALOUS MACHINES")
print("==============================")

print(machine_anomaly_summary.head(10))

# ============================================================
# VISUALIZATION 1
# ANOMALY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(df["Anomaly_Index"], bins=50, kde=True)

plt.title("Anomaly Score Distribution")

plt.xlabel("Anomaly Index")

plt.show()

# ============================================================
# VISUALIZATION 2
# TEMPERATURE VS VIBRATION
# ============================================================

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_Hz", hue="Anomaly_Severity", alpha=0.7
)

plt.title("Machine Anomaly Severity")

plt.show()

# ============================================================
# VISUALIZATION 3
# MACHINE ANOMALY RANKING
# ============================================================

top_10_machines = machine_anomaly_summary.head(10)

plt.figure(figsize=(12, 6))

top_10_machines.plot(kind="bar")

plt.title("Top 10 Most Abnormal Machines")

plt.xlabel("Machine ID")
plt.ylabel("Average Anomaly Index")

plt.show()

# ============================================================
# VISUALIZATION 4
# ANOMALY SEVERITY COUNTS
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(x="Anomaly_Severity", data=df)

plt.title("Anomaly Severity Distribution")

plt.show()

# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv("Final_Anomaly_Scoring_Output.csv", index=False)

print("\n====================================")
print("ANOMALY SCORING COMPLETED")
print("====================================")
