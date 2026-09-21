import pandas as pd
import numpy as np

print("\n=== STANDALONE MODE — Adding to existing CSV ===")

# Load
df = pd.read_csv("Predictive_Maintenance_KPI_Output.csv")
print(f"Loaded: {df.shape}")

# Sort
df = df.sort_values(["Machine_ID", "Date", "Timestamp"]).reset_index(drop=True)

# Condition 1
df["Recent_Anomaly_Count"] = df.groupby("Machine_ID")["Anomaly_Label"].transform(
    lambda x: (x == -1).rolling(7, min_periods=1).sum()
)
cond1 = df["Recent_Anomaly_Count"] >= 2

# Condition 2
df["Prev_Rolling_Anomaly"] = (
    df.groupby("Machine_ID")["Rolling_Anomaly_Index"]
    .shift(1)
    .fillna(df["Rolling_Anomaly_Index"])
)
cond2 = (df["Rolling_Anomaly_Index"] > df["Prev_Rolling_Anomaly"]) & (
    df["Risk_Escalation_Rate"] > 0.05
)

# Condition 3
casc_threshold = df["Cascading_Failure_Index"].quantile(0.90)
cond3 = df["Cascading_Failure_Index"] > casc_threshold

# Combine
active_cascade = cond1 & cond2 & cond3
emerging_cascade = (
    (cond1 & cond2) | (cond2 & cond3) | (cond1 & cond3)
) & ~active_cascade

df["Cascade_Failure_Pattern"] = np.select(
    [active_cascade, emerging_cascade],
    ["Active Cascade Risk", "Emerging Cascade"],
    default="No Pattern",
)

# Drop helpers
df.drop(
    columns=["Recent_Anomaly_Count", "Prev_Rolling_Anomaly"],
    inplace=True,
    errors="ignore",
)

# save
output_file = "Predictive_Maintenance_KPI_Output_Cascade.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved with Cascade_Failure_Pattern column")
print(f"Output file: {output_file}")
print(df["Cascade_Failure_Pattern"].value_counts())

# ============================================================
# DASHBOARD USAGE (thales_app.py)
# ============================================================
# After adding this column, use it in Tab 3 and Tab 4:
#
# Tab 3 — Maintenance Alerts:
#
#   casc_pattern = dashboard_df['Cascade_Failure_Pattern'].value_counts()
#   active_count   = casc_pattern.get('Active Cascade Risk', 0)
#   emerging_count = casc_pattern.get('Emerging Cascade', 0)
#
#   # KPI cards
#   st.markdown(kpi_card("Active Cascade Risk",   f"{active_count:,}",   "red"))
#   st.markdown(kpi_card("Emerging Cascade",       f"{emerging_count:,}", "amber"))
#
#   # Table — machines with Active Cascade
#   active_machines = (
#       dashboard_df[dashboard_df['Cascade_Failure_Pattern']=='Active Cascade Risk']
#       .groupby('Machine_ID').agg(
#           Active_Cascade_Readings = ('Cascade_Failure_Pattern','count'),
#           Avg_Cascade_Index       = ('Cascading_Failure_Index','mean'),
#           Avg_Risk_Score          = ('Risk_Score','mean'),
#           Avg_Lead_Time           = ('Early_Warning_Lead_Time_Hours','mean'),
#       ).reset_index()
#       .sort_values('Active_Cascade_Readings', ascending=False)
#   )
#   st.dataframe(active_machines)
#
# Tab 4 — Historical Trends:
#
#   # Chart: Cascade pattern over time for selected machine
#   machine_data['Pattern_Numeric'] = machine_data['Cascade_Failure_Pattern'].map({
#       'No Pattern':          0,
#       'Emerging Cascade':    1,
#       'Active Cascade Risk': 2,
#   })
#   ax.plot(machine_data['Datetime'], machine_data['Pattern_Numeric'])
#
# ============================================================
