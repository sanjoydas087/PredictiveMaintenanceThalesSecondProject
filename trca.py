# ============================================================
# TEMPORAL RISK ESCALATION ANALYSIS
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

df = pd.read_csv(
    "Predictive_Maintenance_Risk_Output.csv"
)

# ============================================================
# CREATE DATETIME
# ============================================================

df['Datetime'] = pd.to_datetime(

    df['Date'] + ' ' + df['Timestamp'],

    dayfirst=True,
    errors='coerce'

)

# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(

    by=['Machine_ID', 'Datetime']

)

# ============================================================
# 1. TRACK ANOMALY SCORE OVER TIME
# ============================================================

# Rolling anomaly trend

window_size = 24

df['Rolling_Anomaly_Index'] = (

    df.groupby('Machine_ID')
    ['Anomaly_Index']

    .transform(

        lambda x: x.rolling(
            window=window_size,
            min_periods=1
        ).mean()

    )

)

# ============================================================
# 2. RISK ESCALATION RATE
# ============================================================

# Previous anomaly index

df['Previous_Anomaly_Index'] = (

    df.groupby('Machine_ID')
    ['Anomaly_Index']

    .shift(1)

)

# Escalation calculation

df['Risk_Escalation_Rate'] = (

    df['Anomaly_Index']
    - df['Previous_Anomaly_Index']

)

# ============================================================
# 3. RAPID RISK INCREASE DETECTION
# ============================================================

conditions = [

    (df['Risk_Escalation_Rate'] < 0.02),

    (df['Risk_Escalation_Rate'] >= 0.02) &
    (df['Risk_Escalation_Rate'] < 0.08),

    (df['Risk_Escalation_Rate'] >= 0.08)

]

escalation_labels = [

    'Stable',
    'Increasing',
    'Rapid Escalation'

]

df['Risk_Trend_Status'] = np.select(

    conditions,
    escalation_labels,
    default='Stable'

)

# ============================================================
# 4. CASCADING FAILURE INDICATOR
# ============================================================

# Multiple sensor instability together

df['Cascading_Failure_Index'] = (

    abs(df['Temperature_Deviation']) * 0.25 +

    abs(df['Vibration_Deviation']) * 0.25 +

    abs(df['Power_Deviation']) * 0.20 +

    abs(df['Error_Escalation'].fillna(0)) * 0.15 +

    abs(df['Network_Instability_Index']) * 0.15

)

# ============================================================
# CASCADING FAILURE CLASSIFICATION
# ============================================================

cascade_conditions = [

    (df['Cascading_Failure_Index'] < 5),

    (df['Cascading_Failure_Index'] >= 5) &
    (df['Cascading_Failure_Index'] < 15),

    (df['Cascading_Failure_Index'] >= 15)

]

cascade_labels = [

    'Low Cascade Risk',

    'Medium Cascade Risk',

    'High Cascade Risk'

]

df['Cascade_Risk_Level'] = np.select(

    cascade_conditions,
    cascade_labels,
    default='Low Cascade Risk'

)



#8. DETECT MULTIPLE ABNORMAL MACHINES
# ============================================================

# Count how many different machines are abnormal
# within the same timestamp.

# ============================================================
# 5. MACHINE RISK TREND SUMMARY
# ============================================================

machine_risk_trend = (

    df.groupby('Machine_ID')
    ['Risk_Escalation_Rate']

    .mean()

    .sort_values(ascending=False)

)

print("\n==============================")
print("TOP ESCALATING MACHINES")
print("==============================")

print(machine_risk_trend.head(10))

# ============================================================
# 6. RAPID ESCALATION MACHINES
# ============================================================

rapid_escalation = (

    df[
        df['Risk_Trend_Status']
        == 'Rapid Escalation'
    ]

    .groupby('Machine_ID')

    .size()

    .sort_values(ascending=False)

)

print("\n==============================")
print("RAPID ESCALATION MACHINES")
print("==============================")

print(rapid_escalation.head(10))

# ============================================================
# VISUALIZATION 1
# ANOMALY TREND OVER TIME
# ============================================================

sample_machine = 1

machine_data = df[
    df['Machine_ID'] == sample_machine
]

plt.figure(figsize=(15,6))

plt.plot(

    machine_data['Datetime'],

    machine_data['Anomaly_Index'],

    label='Actual Anomaly Index'

)

plt.plot(

    machine_data['Datetime'],

    machine_data['Rolling_Anomaly_Index'],

    label='Rolling Risk Trend',

    linewidth=3

)

plt.title(
    f"Machine {sample_machine} Risk Escalation Trend"
)

plt.xlabel("Time")
plt.ylabel("Anomaly Index")

plt.legend()

plt.show()

# ============================================================
# VISUALIZATION 2
# RISK ESCALATION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(

    df['Risk_Escalation_Rate'],

    bins=50,
    kde=True

)

plt.title(
    "Risk Escalation Rate Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 3
# ESCALATION STATUS COUNT
# ============================================================

plt.figure(figsize=(8,6))

sns.countplot(

    x='Risk_Trend_Status',

    data=df

)

plt.title(
    "Risk Trend Status Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 4
# CASCADING FAILURE RISK
# ============================================================

plt.figure(figsize=(8,6))

sns.countplot(

    x='Cascade_Risk_Level',

    data=df

)

plt.title(
    "Cascading Failure Risk Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 5
# TEMPERATURE VS VIBRATION
# ============================================================

plt.figure(figsize=(12,8))

sns.scatterplot(

    data=df,

    x='Temperature_C',

    y='Vibration_Hz',

    hue='Cascade_Risk_Level',

    alpha=0.7

)

plt.title(
    "Cascading Failure Analysis"
)

plt.show()

# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    "Temporal_Risk_Escalation_Output.csv",
    index=False
)

print("\n====================================")
print("TEMPORAL RISK ESCALATION ANALYSIS COMPLETED")
print("====================================")