# ============================================================
# FEATURE ENGINEERING
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

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Thales_Group_Manufacturing.csv")

# =========================
# CREATE DATETIME COLUMN
# =========================

df['Datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Timestamp'],
    dayfirst=True,
    errors='coerce'
)

# =========================
# SORT DATA
# =========================

df = df.sort_values(
    by=['Machine_ID', 'Datetime']
)

# ============================================================
# 1. SENSOR DEVIATION FROM ROLLING BASELINE
# ============================================================

# Rolling Window Size
window_size = 24

# -------------------------
# Rolling Temperature Mean
# -------------------------

df['Rolling_Temp_Mean'] = (
    df.groupby('Machine_ID')['Temperature_C']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()
    )
)

# -------------------------
# Rolling Vibration Mean
# -------------------------

df['Rolling_Vibration_Mean'] = (
    df.groupby('Machine_ID')['Vibration_Hz']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()
    )
)

# -------------------------
# Rolling Power Mean
# -------------------------

df['Rolling_Power_Mean'] = (
    df.groupby('Machine_ID')['Power_Consumption_kW']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()
    )
)

# ============================================================
# SENSOR DEVIATION FEATURES
# ============================================================

# Current Value - Rolling Baseline

df['Temperature_Deviation'] = (
    df['Temperature_C']
    - df['Rolling_Temp_Mean']
)

df['Vibration_Deviation'] = (
    df['Vibration_Hz']
    - df['Rolling_Vibration_Mean']
)

df['Power_Deviation'] = (
    df['Power_Consumption_kW']
    - df['Rolling_Power_Mean']
)

# ============================================================
# 2. VIBRATION-TO-POWER INSTABILITY RATIO
# ============================================================

# Higher ratio may indicate instability

df['Vibration_Power_Ratio'] = (
    df['Vibration_Hz']
    / df['Power_Consumption_kW']
)

# ============================================================
# 3. ERROR ESCALATION TREND INDICATORS
# ============================================================

# Previous Error Rate

df['Previous_Error_Rate'] = (
    df.groupby('Machine_ID')['Error_Rate_%']
    .shift(1)
)

# Error Escalation

df['Error_Escalation'] = (
    df['Error_Rate_%']
    - df['Previous_Error_Rate']
)

# Rolling Error Trend

df['Rolling_Error_Mean'] = (
    df.groupby('Machine_ID')['Error_Rate_%']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()
    )
)

# ============================================================
# 4. MAINTENANCE SCORE DECAY PATTERNS
# ============================================================

# Previous Maintenance Score

df['Previous_Maintenance_Score'] = (
    df.groupby('Machine_ID')
    ['Predictive_Maintenance_Score']
    .shift(1)
)

# Score Decay

df['Maintenance_Score_Decay'] = (
    df['Previous_Maintenance_Score']
    - df['Predictive_Maintenance_Score']
)

# Rolling Maintenance Score

df['Rolling_Maintenance_Score'] = (
    df.groupby('Machine_ID')
    ['Predictive_Maintenance_Score']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()
    )
)

# ============================================================
# ADDITIONAL SENIOR-LEVEL FEATURES
# ============================================================

# ============================================================
# 5. TEMPERATURE VOLATILITY
# ============================================================

df['Temp_Volatility'] = (
    df.groupby('Machine_ID')['Temperature_C']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).std()
    )
)

# ============================================================
# 6. VIBRATION VOLATILITY
# ============================================================

df['Vibration_Volatility'] = (
    df.groupby('Machine_ID')['Vibration_Hz']
    .transform(
        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).std()
    )
)

# ============================================================
# 7. POWER EFFICIENCY INDICATOR
# ============================================================

df['Power_Efficiency'] = (
    df['Production_Speed_units_per_hr']
    / df['Power_Consumption_kW']
)

# ============================================================
# 8. QUALITY LOSS INDICATOR
# ============================================================

df['Quality_Loss_Index'] = (
    df['Quality_Control_Defect_Rate_%']
    * df['Error_Rate_%']
)

# ============================================================
# 9. NETWORK INSTABILITY INDEX
# ============================================================

df['Network_Instability_Index'] = (
    df['Network_Latency_ms']
    * df['Packet_Loss_%']
)

# ============================================================
# 10. COMPOSITE MACHINE RISK SCORE
# ============================================================

df['Risk_Score'] = (
    abs(df['Temperature_Deviation']) * 0.25 +
    abs(df['Vibration_Deviation']) * 0.25 +
    abs(df['Power_Deviation']) * 0.15 +
    abs(df['Error_Escalation'].fillna(0)) * 0.15 +
    abs(df['Temp_Volatility'].fillna(0)) * 0.10 +
    abs(df['Vibration_Volatility'].fillna(0)) * 0.10
)

# ============================================================
# RISK CLASSIFICATION
# ============================================================

conditions = [
    (df['Risk_Score'] < 5),
    (df['Risk_Score'] >= 5) &
    (df['Risk_Score'] < 10),
    (df['Risk_Score'] >= 10)
]

risk_labels = [
    'Low',
    'Medium',
    'High'
]

df['Maintenance_Risk_Level'] = np.select(
    conditions,
    risk_labels,
    default='Low'
)

# ============================================================
# VISUALIZATION 1
# TEMPERATURE VS BASELINE
# ============================================================

sample_machine = 1

machine_data = df[
    df['Machine_ID'] == sample_machine
]

plt.figure(figsize=(15,6))

plt.plot(
    machine_data['Datetime'],
    machine_data['Temperature_C'],
    label='Actual Temperature'
)

plt.plot(
    machine_data['Datetime'],
    machine_data['Rolling_Temp_Mean'],
    label='Rolling Baseline',
    linewidth=3
)

plt.title(
    f"Machine {sample_machine} Temperature Trend"
)

plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()

plt.show()

# ============================================================
# VISUALIZATION 2
# RISK DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['Risk_Score'],
    bins=40,
    kde=True
)

plt.title("Machine Risk Score Distribution")

plt.show()

# ============================================================
# VISUALIZATION 3
# CORRELATION HEATMAP
# ============================================================

feature_columns = [
    'Temperature_Deviation',
    'Vibration_Deviation',
    'Power_Deviation',
    'Vibration_Power_Ratio',
    'Error_Escalation',
    'Maintenance_Score_Decay',
    'Temp_Volatility',
    'Vibration_Volatility',
    'Risk_Score'
]

plt.figure(figsize=(14,10))

correlation_matrix = (
    df[feature_columns].corr()
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    "Feature Engineering Correlation Matrix"
)

plt.show()

# ============================================================
# HIGH RISK MACHINE SUMMARY
# ============================================================

high_risk = df[
    df['Maintenance_Risk_Level'] == 'High'
]

high_risk_summary = (
    high_risk.groupby('Machine_ID')
    .size()
    .sort_values(ascending=False)
)

print("\n========== HIGH RISK MACHINES ==========")

print(high_risk_summary.head(10))

# ============================================================
# SAVE FINAL FEATURE ENGINEERED DATASET
# ============================================================

df.to_csv(
    "Feature_Engineered_Predictive_Maintenance.csv",
    index=False
)

print("\n====================================")
print("FEATURE ENGINEERING COMPLETED")
print("====================================")