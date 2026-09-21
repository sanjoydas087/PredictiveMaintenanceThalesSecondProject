# ============================================================
# KPI ANALYTICS SYSTEM
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

df = pd.read_csv(
    "Temporal_Risk_Escalation_Output.csv"
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
# 1. ANOMALY SCORE KPI
# ============================================================

# Overall anomaly statistics

anomaly_kpi = {

    'Average_Anomaly_Index':
    df['Anomaly_Index'].mean(),

    'Maximum_Anomaly_Index':
    df['Anomaly_Index'].max(),

    'Minimum_Anomaly_Index':
    df['Anomaly_Index'].min()

}

print("\n==============================")
print("ANOMALY SCORE KPI")
print("==============================")

print(anomaly_kpi)

# ============================================================
# 2. MAINTENANCE RISK LEVEL KPI
# ============================================================

risk_distribution = (

    df['Predictive_Maintenance_Risk']
    .value_counts()

)

print("\n==============================")
print("RISK LEVEL DISTRIBUTION")
print("==============================")

print(risk_distribution)

# ============================================================
# 3. HIGH-RISK MACHINE LIST KPI
# ============================================================

high_risk_machines = (

    df[
        df['Predictive_Maintenance_Risk']
        == 'High Risk'
    ]

    .groupby('Machine_ID')

    .agg({

        'Anomaly_Index':'mean',

        'Risk_Escalation_Rate':'mean',

        'Temperature_C':'mean',

        'Vibration_Hz':'mean'

    })

    .sort_values(
        by='Anomaly_Index',
        ascending=False
    )

)

print("\n==============================")
print("HIGH RISK MACHINES")
print("==============================")

print(high_risk_machines.head(10))

# ============================================================
# 4. EARLY WARNING LEAD TIME KPI
# ============================================================

# Estimated time before failure

# Early warning leaad time calculation by anamoly score



#import numpy as np

# Risk proxy: 0 for all normal machines, positive only for anomalies
df['risk_proxy'] = np.maximum(0, -df['Anomaly_Score'])
df['risk_norm']  = df['risk_proxy'] / df['risk_proxy'].max()

# Lead time
df['Early_Warning_Lead_Time_Hours'] = 100 / (1 + df['risk_norm'] * 4)



# Downtime Prevention Index (0–10 scale)


df['Downtime_Prevention_Index'] = (df['Early_Warning_Lead_Time_Hours'] / 100) * 10



#df['Early_Warning_Lead_Time_Hours'] = (

#    100 / (
#        1 + (df['Anomaly_Index'] * 20)
#    )

#)

print("\n==============================")
print("EARLY WARNING LEAD TIME")
print("==============================")

print(

    df[
        'Early_Warning_Lead_Time_Hours'
    ].describe()

)

# ============================================================
# 5. DOWNTIME PREVENTION INDEX KPI
# ============================================================

# Estimated avoided failures

#df['Downtime_Prevention_Index'] = (

 #   df['Anomaly_Index']
  #  * df['Early_Warning_Lead_Time_Hours']

#)

print("\n==============================")
print("DOWNTIME PREVENTION KPI")
print("==============================")

print(

    df[
        'Downtime_Prevention_Index'
    ].describe()

)

# ============================================================
# MACHINE-LEVEL KPI SUMMARY
# ============================================================

machine_kpi_summary = (

    df.groupby('Machine_ID')

    .agg({

        'Anomaly_Index':'mean',

        'Risk_Escalation_Rate':'mean',

        'Early_Warning_Lead_Time_Hours':'mean',

        'Downtime_Prevention_Index':'mean'

    })

    .sort_values(
        by='Anomaly_Index',
        ascending=False
    )

)

print("\n==============================")
print("MACHINE KPI SUMMARY")
print("==============================")

print(machine_kpi_summary.head(10))

# ============================================================
# VISUALIZATION 1
# RISK LEVEL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,6))

sns.countplot(

    x='Predictive_Maintenance_Risk',

    data=df

)

plt.title(
    "Maintenance Risk Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 2
# ANOMALY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(

    df['Anomaly_Index'],

    bins=50,
    kde=True

)

plt.title(
    "Anomaly Score Distribution"
)

plt.xlabel("Anomaly Index")

plt.show()

# ============================================================
# VISUALIZATION 3
# EARLY WARNING LEAD TIME
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(

    df['Early_Warning_Lead_Time_Hours'],

    bins=50,
    kde=True

)

plt.title(
    "Early Warning Lead Time Distribution"
)

plt.xlabel("Lead Time (Hours)")

plt.show()

# ============================================================
# VISUALIZATION 4
# DOWNTIME PREVENTION INDEX
# ============================================================

plt.figure(figsize=(10,6))

sns.histplot(

    df['Downtime_Prevention_Index'],

    bins=50,
    kde=True

)

plt.title(
    "Downtime Prevention Index Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 5
# TOP HIGH-RISK MACHINES
# ============================================================

top_high_risk = (
    high_risk_machines.head(10)
)

if not top_high_risk.empty:

    plt.figure(figsize=(12,6))

    top_high_risk[
        'Anomaly_Index'
    ].plot(kind='bar')

    plt.title(
        "Top High Risk Machines"
    )

    plt.xlabel("Machine ID")

    plt.ylabel("Average Anomaly Score")

    plt.show()

else:

    print(
        "No High Risk Machines Found"
    )

# ============================================================
# VISUALIZATION 6
# RISK ESCALATION TREND
# ============================================================

sample_machine = 1

machine_data = df[
    df['Machine_ID']
    == sample_machine
]

plt.figure(figsize=(15,6))

plt.plot(

    machine_data['Datetime'],

    machine_data['Anomaly_Index'],

    label='Anomaly Index'

)

plt.plot(

    machine_data['Datetime'],

    machine_data['Rolling_Anomaly_Index'],

    label='Rolling Risk Trend',

    linewidth=3

)

plt.title(
    f"Machine {sample_machine} Risk Trend"
)

plt.xlabel("Time")
plt.ylabel("Risk Level")

plt.legend()

plt.show()

# ============================================================
# SAVE KPI OUTPUT
# ============================================================

df.to_csv(
    "Predictive_Maintenance_KPI_Output.csv",
    index=False
)

print("\n====================================")
print("KPI ANALYTICS COMPLETED")
print("====================================")