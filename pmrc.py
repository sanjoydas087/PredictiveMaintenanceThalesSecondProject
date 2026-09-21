# ============================================================
# PREDICTIVE MAINTENANCE RISK CLASSIFICATION
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
    "Final_Anomaly_Scoring_Output.csv"
)

# ============================================================
# RISK CLASSIFICATION
# ============================================================

# ============================================================
# DEFINE CONDITIONS
# ============================================================



conditions = [

    (df['Anomaly_Index'] < 0.02),

    (df['Anomaly_Index'] >= 0.02) &
    (df['Anomaly_Index'] < 0.05),

    (df['Anomaly_Index'] >= 0.05)

]
# ============================================================
# DEFINE LABELS
# ============================================================

risk_labels = [

    'Low Risk',
    'Medium Risk',
    'High Risk'

]

# ============================================================
# APPLY CLASSIFICATION
# ============================================================

df['Predictive_Maintenance_Risk'] = np.select(

    conditions,
    risk_labels,
    default='Low Risk'

)

# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("PREDICTIVE MAINTENANCE RISK")
print("==============================")

print(
    df['Predictive_Maintenance_Risk']
    .value_counts()
)

# ============================================================
# MACHINE-WISE RISK SUMMARY
# ============================================================

machine_risk_summary = (

    df.groupby([
        'Machine_ID',
        'Predictive_Maintenance_Risk'
    ])

    .size()

    .reset_index(name='Count')

)

print("\n==============================")
print("MACHINE RISK SUMMARY")
print("==============================")

print(machine_risk_summary.head(20))

# ============================================================
# HIGH RISK MACHINES
# ============================================================

high_risk_machines = (

    df[
        df['Predictive_Maintenance_Risk']
        == 'High Risk'
    ]

    .groupby('Machine_ID')

    .size()

    .sort_values(ascending=False)

)

print("\n==============================")
print("TOP HIGH RISK MACHINES")
print("==============================")

print(high_risk_machines.head(10))

# ============================================================
# VISUALIZATION 1
# RISK DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,6))

sns.countplot(

    x='Predictive_Maintenance_Risk',
    data=df

)

plt.title(
    "Predictive Maintenance Risk Distribution"
)

plt.xlabel("Risk Level")
plt.ylabel("Count")

plt.show()

# ============================================================
# VISUALIZATION 2
# ANOMALY INDEX DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(

    data=df,

    x='Anomaly_Index',

    hue='Predictive_Maintenance_Risk',

    bins=50,
    kde=True

)

plt.title(
    "Anomaly Index by Risk Level"
)

plt.show()

# ============================================================
# VISUALIZATION 3
# TEMPERATURE VS VIBRATION
# ============================================================

plt.figure(figsize=(12,8))

sns.scatterplot(

    data=df,

    x='Temperature_C',
    y='Vibration_Hz',

    hue='Predictive_Maintenance_Risk',

    alpha=0.7

)

plt.title(
    "Machine Risk Classification"
)

plt.show()

# ============================================================
# VISUALIZATION 4
# TOP HIGH-RISK MACHINES
# ============================================================

top_10_high_risk = (
    high_risk_machines.head(10)
)

plt.figure(figsize=(12,6))

if not top_10_high_risk.empty:

    top_10_high_risk.plot(kind='bar')

    plt.title("Top 10 High Risk Machines")

    plt.xlabel("Machine ID")
    plt.ylabel("High Risk Count")

    plt.show()

else:

    print("No High Risk Machines Found")


plt.title(
    "Top 10 High Risk Machines"
)

plt.xlabel("Machine ID")
plt.ylabel("High Risk Count")

plt.show()

# ============================================================
# MAINTENANCE RECOMMENDATIONS
# ============================================================

df['Maintenance_Action'] = np.select(

    [

        df['Predictive_Maintenance_Risk']
        == 'Low Risk',

        df['Predictive_Maintenance_Risk']
        == 'Medium Risk',

        df['Predictive_Maintenance_Risk']
        == 'High Risk'

    ],

    [

        'Normal Monitoring',

        'Schedule Inspection',

        'Immediate Maintenance Required'

    ],

    default='Normal Monitoring'

)

# ============================================================
# SAMPLE OUTPUT
# ============================================================

print("\n==============================")
print("MAINTENANCE ACTION SAMPLE")
print("==============================")

print(

    df[[
        'Machine_ID',
        'Anomaly_Index',
        'Predictive_Maintenance_Risk',
        'Maintenance_Action'
    ]]

    .head(20)

)

# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    "Predictive_Maintenance_Risk_Output.csv",
    index=False
)

print("\n====================================")
print("PREDICTIVE MAINTENANCE CLASSIFICATION COMPLETED")
print("====================================")