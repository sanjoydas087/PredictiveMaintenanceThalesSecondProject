# ============================================================
# AUTOENCODER ANOMALY DETECTION
# Predictive Maintenance — Thales Group Manufacturing IoT
# Deep Learning Variant (Optional — per Project Description)
# ============================================================
#
# What this script does:
#   1. Loads Feature_Engineered_Predictive_Maintenance.csv
#      (or Predictive_Maintenance_KPI_Output.csv)
#   2. Trains an Autoencoder on NORMAL machine data only
#   3. Calculates Reconstruction Error for every reading
#   4. Flags high-error readings as anomalies
#   5. Compares results with Isolation Forest
#   6. Saves final output with Autoencoder anomaly columns
#
# KEY CONCEPT:
#   Autoencoder learns to reconstruct NORMAL data well.
#   When it sees ABNORMAL data → high reconstruction error.
#   Reconstruction Error = Anomaly Score (higher = more abnormal)
#   This directly matches project spec: Higher score = more abnormal
#
# Developed by : Sanjoy Das
# Internship   : Unified Mentor Pvt Ltd
# ============================================================

# =========================
# INSTALL IF NEEDED:
# pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras for Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("=" * 60)
print("AUTOENCODER ANOMALY DETECTION")
print("Thales Group Predictive Maintenance")
print("=" * 60)
print(f"TensorFlow version : {tf.__version__}")

# ============================================================
# STEP 1 : LOAD DATA
# ============================================================

print("\n[1/8] Loading dataset...")

# Try KPI output first, then feature engineered CSV
try:
    df = pd.read_csv("Predictive_Maintenance_KPI_Output_Cascade.csv")
    print("  Loaded: Predictive_Maintenance_KPI_Output_Cascade.csv")
except FileNotFoundError:
    df = pd.read_csv("Feature_Engineered_Predictive_Maintenance.csv")
    print("  Loaded: Feature_Engineered_Predictive_Maintenance.csv")

print(f"  Shape  : {df.shape}")
print(f"  Machines: {df['Machine_ID'].nunique()}")

# ============================================================
# STEP 2 : SELECT FEATURES (same 12 as Isolation Forest)
# ============================================================

print("\n[2/8] Selecting features...")

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

# Handle missing values
df[feature_columns] = df[feature_columns].fillna(0)

print(f"  Features used : {len(feature_columns)}")
for f in feature_columns:
    print(f"    - {f}")

# ============================================================
# STEP 3 : FEATURE SCALING
# ============================================================

print("\n[3/8] Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

print(f"  Scaled shape : {X_scaled.shape}")

# ============================================================
# STEP 4 : SPLIT NORMAL vs ANOMALY DATA
# ============================================================

print("\n[4/8] Splitting normal vs anomaly data...")

# KEY POINT:
# Autoencoders must be trained ONLY on normal data.
# They learn what "normal" looks like.
# When shown anomalous data, they cannot reconstruct it well.
# High reconstruction error = anomaly.

# Use Isolation Forest label (from previous step) if available
# Otherwise use Predictive_Maintenance_Risk to define normal

if "Anomaly_Label" in df.columns:
    # -1 = anomaly, +1 = normal (Isolation Forest convention)
    normal_mask = df["Anomaly_Label"] == 1
    anomaly_mask = df["Anomaly_Label"] == -1
    print("  Using Isolation Forest Anomaly_Label for train/test split")

elif "Predictive_Maintenance_Risk" in df.columns:
    normal_mask = df["Predictive_Maintenance_Risk"] == "Low Risk"
    anomaly_mask = df["Predictive_Maintenance_Risk"] == "High Risk"
    print("  Using Predictive_Maintenance_Risk for train/test split")

else:
    # Fallback: use top 3% as anomaly based on Risk_Score
    threshold = df["Risk_Score"].quantile(0.97)
    normal_mask = df["Risk_Score"] <= threshold
    anomaly_mask = df["Risk_Score"] > threshold
    print("  Using Risk_Score quantile for train/test split")

X_normal = X_scaled[normal_mask]
X_anomaly = X_scaled[anomaly_mask]

print(f"  Normal   readings : {X_normal.shape[0]:,}")
print(f"  Anomaly  readings : {X_anomaly.shape[0]:,}")

# ============================================================
# STEP 5 : BUILD AUTOENCODER MODEL
# ============================================================

print("\n[5/8] Building Autoencoder architecture...")

# ── WHAT IS AN AUTOENCODER? ──────────────────────────────────
# An Autoencoder is a neural network with two parts:
#
#   ENCODER : Compresses input (12 features) into a small
#             "bottleneck" representation (4 neurons)
#             → Forces the network to learn the ESSENTIAL
#               patterns of normal machine behavior
#
#   DECODER : Reconstructs the original 12 features
#             from the compressed 4-neuron representation
#
# TRAINING : Train ONLY on normal data
#   → Network learns to reconstruct normal patterns perfectly
#
# INFERENCE: Run all data through the trained model
#   → Normal data: low reconstruction error (model knows it)
#   → Anomaly data: HIGH reconstruction error (model never
#     saw this pattern during training → cannot reconstruct)
#
# RECONSTRUCTION ERROR = MSE (Mean Squared Error)
# between original input and reconstructed output
# Higher MSE = More anomalous = Higher Anomaly Score ✓
# ────────────────────────────────────────────────────────────

input_dim = len(feature_columns)  # 12 features

# ── INPUT LAYER ──────────────────────────────────────────────
input_layer = Input(shape=(input_dim,), name="Input")

# ── ENCODER ─────────────────────────────────────────────────
# Gradually compresses: 12 → 8 → 4
encoded = Dense(8, activation="relu", name="Encoder_Dense_8")(input_layer)

encoded = Dropout(0.2, name="Encoder_Dropout")(encoded)

# BOTTLENECK — the most compressed representation
bottleneck = Dense(4, activation="relu", name="Bottleneck_4")(encoded)

# ── DECODER ─────────────────────────────────────────────────
# Gradually reconstructs: 4 → 8 → 12
decoded = Dense(8, activation="relu", name="Decoder_Dense_8")(bottleneck)

decoded = Dropout(0.2, name="Decoder_Dropout")(decoded)

# OUTPUT LAYER — reconstructed features (linear activation)
output_layer = Dense(
    input_dim,
    activation="linear",  # Linear because features are continuous
    name="Output",
)(decoded)

# ── COMPILE MODEL ────────────────────────────────────────────
autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Autoencoder")

autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",  # Mean Squared Error — our reconstruction error
)

# Print architecture summary
print("\n  Autoencoder Architecture:")
print("  " + "─" * 50)
autoencoder.summary()
print("  " + "─" * 50)
print(f"\n  Input dimensions  : {input_dim}")
print(f"  Bottleneck neurons: 4  (compresses {input_dim} → 4)")
print(f"  Total parameters  : {autoencoder.count_params():,}")

# ============================================================
# STEP 6 : TRAIN THE AUTOENCODER
# ============================================================

print("\n[6/8] Training Autoencoder on normal data only...")
print("  This may take 1-3 minutes depending on your machine.")

# ── CALLBACKS ────────────────────────────────────────────────
# EarlyStopping: stop training if no improvement for 10 epochs
# ReduceLROnPlateau: reduce learning rate if loss plateaus

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
)

# ── TRAIN ────────────────────────────────────────────────────
# Input = Output = X_normal
# The autoencoder tries to reconstruct its own input
# This is UNSUPERVISED — no labels needed during training

history = autoencoder.fit(
    X_normal,  # Input
    X_normal,  # Target (same as input — self-supervised)
    epochs=100,
    batch_size=256,
    validation_split=0.1,  # 10% of normal data for validation
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=True,
)

print(f"\n  Training stopped at epoch  : {len(history.history['loss'])}")
print(f"  Final training loss (MSE)  : {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss (MSE): {history.history['val_loss'][-1]:.6f}")

# ============================================================
# STEP 7 : CALCULATE RECONSTRUCTION ERROR
# ============================================================

print("\n[7/8] Calculating reconstruction errors...")

# ── RECONSTRUCT ALL 100,000 READINGS ─────────────────────────
X_reconstructed = autoencoder.predict(X_scaled, batch_size=512, verbose=0)

# ── RECONSTRUCTION ERROR per reading ─────────────────────────
# MSE = mean((original - reconstructed)^2) for each row
# Shape: (100000,) — one error value per reading

reconstruction_errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

df["AE_Reconstruction_Error"] = reconstruction_errors.round(6)

print(f"  Reconstruction Error stats:")
print(f"    Min    : {reconstruction_errors.min():.6f}")
print(f"    Mean   : {reconstruction_errors.mean():.6f}")
print(f"    Max    : {reconstruction_errors.max():.6f}")
print(f"    Std    : {reconstruction_errors.std():.6f}")

# ── NORMALIZED AUTOENCODER SCORE (0-100, Higher = More Abnormal)
# Matches project spec: Higher score = more abnormal behavior ✓

df["AE_Anomaly_Score"] = (pd.Series(reconstruction_errors).rank(pct=True) * 100).round(
    2
)

print(
    f"\n  AE_Anomaly_Score range: {df['AE_Anomaly_Score'].min():.1f} – {df['AE_Anomaly_Score'].max():.1f}"
)
print(f"  Convention: Higher = More Abnormal ✓")

# ── SET THRESHOLD FOR ANOMALY DETECTION ──────────────────────
# Threshold = mean + 2×std of reconstruction error on normal data
# Readings above this threshold are flagged as anomalies

normal_errors = reconstruction_errors[normal_mask]
threshold_ae = normal_errors.mean() + 2 * normal_errors.std()

print(f"\n  Anomaly threshold: {threshold_ae:.6f}")
print(f"  (Normal mean + 2× Normal std)")

df["AE_Anomaly_Label"] = np.where(
    reconstruction_errors > threshold_ae,
    -1,  # Anomaly (matches Isolation Forest convention)
    1,  # Normal
)

df["AE_Anomaly_Status"] = np.where(df["AE_Anomaly_Label"] == -1, "Anomaly", "Normal")

# ── ANOMALY SEVERITY from AE Score ───────────────────────────
conditions_ae = [
    (df["AE_Anomaly_Score"] < 33),
    (df["AE_Anomaly_Score"] >= 33) & (df["AE_Anomaly_Score"] < 66),
    (df["AE_Anomaly_Score"] >= 66),
]
df["AE_Anomaly_Severity"] = np.select(
    conditions_ae, ["Low", "Medium", "Critical"], default="Low"
)

ae_anomaly_count = (df["AE_Anomaly_Label"] == -1).sum()
print(f"\n  Anomalies detected by Autoencoder: {ae_anomaly_count:,}")
print(f"  Anomaly rate: {ae_anomaly_count/len(df)*100:.1f}%")

# ============================================================
# STEP 8 : COMPARE ISOLATION FOREST vs AUTOENCODER
# ============================================================

print("\n[8/8] Comparing Isolation Forest vs Autoencoder...")
print("  " + "─" * 50)

if "Anomaly_Label" in df.columns:
    # Agreement between two models
    agreement = (df["Anomaly_Label"] == df["AE_Anomaly_Label"]).mean() * 100
    print(f"  Model agreement rate   : {agreement:.1f}%")

    # Both models flag as anomaly (high confidence)
    both_anomaly = ((df["Anomaly_Label"] == -1) & (df["AE_Anomaly_Label"] == -1)).sum()

    # Only IF flags
    only_if = ((df["Anomaly_Label"] == -1) & (df["AE_Anomaly_Label"] == 1)).sum()

    # Only AE flags
    only_ae = ((df["Anomaly_Label"] == 1) & (df["AE_Anomaly_Label"] == -1)).sum()

    print(
        f"\n  Both models flag      : {both_anomaly:,} readings (HIGH CONFIDENCE anomalies)"
    )
    print(f"  Only Isolation Forest : {only_if:,} readings")
    print(f"  Only Autoencoder      : {only_ae:,} readings")

    # ── ENSEMBLE SCORE ────────────────────────────────────────
    # Combine both models for stronger signal
    # Ensemble = average of normalized scores from both models

    if "Normalized_Anomaly_Score" in df.columns:
        df["Ensemble_Anomaly_Score"] = (
            (df["Normalized_Anomaly_Score"] + df["AE_Anomaly_Score"]) / 2
        ).round(2)

        df["Ensemble_Risk"] = np.select(
            [df["Ensemble_Anomaly_Score"] >= 97, df["Ensemble_Anomaly_Score"] >= 66],
            ["High Risk", "Medium Risk"],
            default="Low Risk",
        )

        print(f"\n  ── ENSEMBLE MODEL (IF + AE combined) ──────────")
        print(f"  Ensemble High Risk  : {(df['Ensemble_Risk']=='High Risk').sum():,}")
        print(f"  Ensemble Medium Risk: {(df['Ensemble_Risk']=='Medium Risk').sum():,}")
        print(f"  Ensemble Low Risk   : {(df['Ensemble_Risk']=='Low Risk').sum():,}")

# ── PRINT RESULTS TABLE ───────────────────────────────────────
print("\n  ── AE Anomaly Score by Original Risk Level ────────")
grp = df.groupby("Predictive_Maintenance_Risk")["AE_Anomaly_Score"].mean().round(2)
for risk, val in grp.items():
    direction = "← HIGH (most abnormal) ✓" if risk == "High Risk" else ""
    print(f"    {risk:15s}: {val:.2f}  {direction}")

# ── VISUALIZATIONS ────────────────────────────────────────────
print("\n  Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Autoencoder Anomaly Detection — Thales Predictive Maintenance",
    fontsize=14,
    fontweight="bold",
)

# ── PLOT 1: Training Loss ─────────────────────────────────────
ax = axes[0, 0]
ax.plot(history.history["loss"], label="Training Loss", linewidth=2)
ax.plot(
    history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--"
)
ax.axvline(
    np.argmin(history.history["val_loss"]),
    color="red",
    linestyle=":",
    alpha=0.7,
    label="Best Epoch",
)
ax.set_title("Autoencoder Training Loss (MSE)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# ── PLOT 2: Reconstruction Error Distribution ─────────────────
ax = axes[0, 1]
ax.hist(
    reconstruction_errors[normal_mask],
    bins=80,
    alpha=0.6,
    color="steelblue",
    label="Normal Readings",
    density=True,
)
ax.hist(
    reconstruction_errors[anomaly_mask],
    bins=80,
    alpha=0.6,
    color="crimson",
    label="Anomalous Readings",
    density=True,
)
ax.axvline(
    threshold_ae,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Threshold ({threshold_ae:.4f})",
)
ax.set_title("Reconstruction Error Distribution")
ax.set_xlabel("Reconstruction Error (MSE)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# ── PLOT 3: AE Score by Risk Level ───────────────────────────
ax = axes[0, 2]
risk_order = ["Low Risk", "Medium Risk", "High Risk"]
risk_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
ae_by_risk = [
    df[df["Predictive_Maintenance_Risk"] == r]["AE_Anomaly_Score"].values
    for r in risk_order
]
bp = ax.boxplot(ae_by_risk, labels=risk_order, patch_artist=True, notch=False)
for patch, color in zip(bp["boxes"], risk_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("AE Anomaly Score by Risk Level\n(Higher = More Abnormal ✓)")
ax.set_ylabel("AE Anomaly Score (0-100)")
ax.grid(True, alpha=0.3, axis="y")

# ── PLOT 4: IF vs AE Score Comparison ────────────────────────
ax = axes[1, 0]
if "Normalized_Anomaly_Score" in df.columns:
    sample = df.sample(5000, random_state=42)
    colors_scatter = np.where(
        (sample["Anomaly_Label"] == -1) & (sample["AE_Anomaly_Label"] == -1),
        "red",  # Both flag
        np.where(
            sample["Anomaly_Label"] == -1,
            "orange",  # Only IF
            np.where(sample["AE_Anomaly_Label"] == -1, "blue", "lightgray"),  # Only AE
        ),  # Both normal
    )
    ax.scatter(
        sample["Normalized_Anomaly_Score"],
        sample["AE_Anomaly_Score"],
        c=colors_scatter,
        alpha=0.4,
        s=8,
    )
    ax.set_xlabel("IF Normalized Score")
    ax.set_ylabel("AE Anomaly Score")
    ax.set_title("Isolation Forest vs Autoencoder Scores")
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="Both flag (high confidence)"),
        Patch(facecolor="orange", label="Only IF flags"),
        Patch(facecolor="blue", label="Only AE flags"),
        Patch(facecolor="lightgray", label="Both normal"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.text(
        0.5,
        0.5,
        "Normalized_Anomaly_Score\nnot available",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

# ── PLOT 5: Top Anomalous Machines ───────────────────────────
ax = axes[1, 1]
top10_ae = (
    df.groupby("Machine_ID")["AE_Anomaly_Score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
colors_bar = [
    "#e74c3c" if v >= top10_ae.quantile(0.7) else "#f39c12" for v in top10_ae.values
]
bars = ax.bar([f"M-{m}" for m in top10_ae.index], top10_ae.values, color=colors_bar)
ax.set_title("Top 10 Most Anomalous Machines\n(Autoencoder AE Score)")
ax.set_xlabel("Machine ID")
ax.set_ylabel("Mean AE Anomaly Score")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")

# ── PLOT 6: Ensemble Score Distribution ──────────────────────
ax = axes[1, 2]
if "Ensemble_Anomaly_Score" in df.columns:
    ax.hist(
        df[df["Ensemble_Risk"] == "Low Risk"]["Ensemble_Anomaly_Score"],
        bins=60,
        alpha=0.6,
        color="green",
        label="Low Risk",
        density=True,
    )
    ax.hist(
        df[df["Ensemble_Risk"] == "Medium Risk"]["Ensemble_Anomaly_Score"],
        bins=60,
        alpha=0.6,
        color="orange",
        label="Medium Risk",
        density=True,
    )
    ax.hist(
        df[df["Ensemble_Risk"] == "High Risk"]["Ensemble_Anomaly_Score"],
        bins=60,
        alpha=0.6,
        color="red",
        label="High Risk",
        density=True,
    )
    ax.set_title("Ensemble Score Distribution\n(IF + AE Combined)")
    ax.set_xlabel("Ensemble Anomaly Score (0-100)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.hist(df["AE_Anomaly_Score"], bins=80, color="steelblue", alpha=0.7)
    ax.set_title("AE Anomaly Score Distribution")
    ax.set_xlabel("AE Anomaly Score (0-100)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Autoencoder_Anomaly_Results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Visualization saved: Autoencoder_Anomaly_Results.png")

# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("SAVING OUTPUT")
print("=" * 60)

# New columns added by Autoencoder
new_cols = [
    "AE_Reconstruction_Error",
    "AE_Anomaly_Score",
    "AE_Anomaly_Label",
    "AE_Anomaly_Status",
    "AE_Anomaly_Severity",
]
if "Ensemble_Anomaly_Score" in df.columns:
    new_cols += ["Ensemble_Anomaly_Score", "Ensemble_Risk"]

df.to_csv("Autoencoder_Anomaly_Output_Cascade.csv", index=False)

print("\n  New columns added:")
for col in new_cols:
    print(f"    + {col}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\n  Total readings          : {len(df):,}")
print(f"  AE Anomalies detected   : {(df['AE_Anomaly_Label']==-1).sum():,}")
print(f"  AE Anomaly rate         : {(df['AE_Anomaly_Label']==-1).mean()*100:.1f}%")

print(f"\n  AE Score by Severity:")
for sev in ["Critical", "Medium", "Low"]:
    cnt = (df["AE_Anomaly_Severity"] == sev).sum()
    print(f"    {sev:10s} : {cnt:,}")

print(f"\n  Top 5 most anomalous machines (AE Score):")
top5 = (
    df.groupby("Machine_ID")["AE_Anomaly_Score"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)
for m, score in top5.items():
    print(f"    Machine M-{m:2d} : {score:.2f}")

print("\n  Output saved : Autoencoder_Anomaly_Output.csv")
print("  Charts saved : Autoencoder_Anomaly_Results.png")
print("\n" + "=" * 60)
print("✓ AUTOENCODER ANOMALY DETECTION COMPLETE")
print("=" * 60)
