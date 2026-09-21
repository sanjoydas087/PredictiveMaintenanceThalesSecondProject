# ============================================================
# AI PREDICTIVE MAINTENANCE DASHBOARD
# Thales Group Manufacturing IoT — Unified Mentor Internship
# Developed by: Sanjoy Das | BE Electrical, Jadavpur University
# ============================================================

# ── IMPORTS ─────────────────────────────────────────────────
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import time
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance | Thales IoT",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ────────────────────────────────────────────────────
DARK_BG  = "#0D1117"
CARD_BG  = "#161B22"
BORDER   = "#30363D"
ACCENT   = "#238636"
ACCENT2  = "#1F6FEB"
WARN     = "#D29922"
DANGER   = "#DA3633"
TEXT_PRI = "#E6EDF3"
TEXT_SEC = "#8B949E"
FONT     = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_SEC,
    "axes.titlecolor":  TEXT_PRI,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "text.color":       TEXT_PRI,
    "grid.color":       BORDER,
    "grid.alpha":       0.5,
    "lines.linewidth":  1.8,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": BORDER,
    "legend.labelcolor":TEXT_PRI,
})

# ── TAB METADATA ─────────────────────────────────────────────
# Each tab has: icon, title, subtitle, accent color
# This drives tab_header() automatically

TAB_META = {
    1: {
        "icon":     "📊",
        "title":    "Fleet Overview",
        "subtitle": "Risk distribution · Machine health ranking · Maintenance action summary across all 50 machines",
        "color":    DANGER,       # red — high stakes fleet view
        "tag":      "FLEET",
    },
    2: {
        "icon":     "🔍",
        "title":    "Machine Analysis",
        "subtitle": "Deep-dive anomaly trends · Sensor deviations · Correlation matrix for selected machine",
        "color":    ACCENT2,      # blue — analytical / investigative
        "tag":      "MACHINE",
    },
    3: {
        "icon":     "🚨",
        "title":    "Maintenance Alerts",
        "subtitle": "Active high-risk alerts · Cascade failure risk · Immediate action priority log",
        "color":    WARN,         # amber — warning / alert
        "tag":      "ALERTS",
    },
    4: {
        "icon":     "📈",
        "title":    "Historical Trends",
        "subtitle": "Risk escalation timeline · Lead time trend · Post-maintenance behaviour comparison",
        "color":    ACCENT,       # green — positive / historical context
        "tag":      "HISTORY",
    },
    5: {
        "icon":     "🤖",
        "title":    "Autoencoder Analysis",
        "subtitle": "Deep learning anomaly detection · IF vs AE model comparison · Ensemble risk scoring",
        "color":    "#7F77DD",    # purple — AI / deep learning
        "tag":      "AI MODEL",
    },
}


st.markdown(f"""
<style>
  html, body, [class*="css"] {{
      background-color: {DARK_BG};
      color: {TEXT_PRI};
      font-family: {FONT};
  }}
  .main .block-container {{
      padding: 1.5rem 2rem 2rem 2rem;
      max-width: 1400px;
  }}
  [data-testid="stSidebar"] {{
      background-color: {CARD_BG};
      border-right: 1px solid {BORDER};
  }}
  .section-header {{
      font-size: 0.70rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {TEXT_SEC};
      border-bottom: 1px solid {BORDER};
      padding-bottom: 0.4rem;
      margin: 1.6rem 0 1rem 0;
  }}
  .dash-title {{
      font-size: 1.5rem;
      font-weight: 700;
      color: {TEXT_PRI};
      letter-spacing: -0.02em;
      margin: 0;
      line-height: 1.2;
  }}
  .dash-subtitle {{
      font-size: 0.82rem;
      color: {TEXT_SEC};
      margin-top: 0.2rem;
  }}
  /* ── KPI CARDS ── */
  .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 1rem 1.1rem 0.9rem 1.1rem;
      height: 100%;
  }}
  .kpi-label {{
      font-size: 0.70rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {TEXT_SEC};
      margin-bottom: 0.35rem;
  }}
  .kpi-value {{
      font-size: 1.55rem;
      font-weight: 700;
      color: {TEXT_PRI};
      line-height: 1.1;
  }}
  .kpi-sub {{
      font-size: 0.72rem;
      color: {TEXT_SEC};
      margin-top: 0.3rem;
  }}
  .kpi-value.green  {{ color: {ACCENT};  }}
  .kpi-value.amber  {{ color: {WARN};    }}
  .kpi-value.red    {{ color: {DANGER};  }}
  .kpi-value.blue   {{ color: {ACCENT2}; }}
  /* ── DUAL KPI (two metrics in one card) ── */
  .kpi-dual {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 0.85rem 1.1rem;
      height: 100%;
  }}
  .kpi-dual-row {{
      display: flex;
      gap: 1rem;
      align-items: flex-end;
  }}
  .kpi-dual-item {{
      flex: 1;
  }}
  .kpi-divider {{
      width: 1px;
      background: {BORDER};
      align-self: stretch;
      margin: 0 0.2rem;
  }}
  /* ── BADGE ── */
  .badge {{
      display: inline-block;
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: 0.04em;
  }}
  .badge-red   {{ background:rgba(218,54,51,0.15);  color:{DANGER};  border:1px solid rgba(218,54,51,0.3); }}
  .badge-amber {{ background:rgba(210,153,34,0.15); color:{WARN};    border:1px solid rgba(210,153,34,0.3); }}
  .badge-green {{ background:rgba(35,134,54,0.15);  color:{ACCENT};  border:1px solid rgba(35,134,54,0.3); }}
  .badge-blue  {{ background:rgba(31,111,235,0.15); color:{ACCENT2}; border:1px solid rgba(31,111,235,0.3); }}
  /* ── MACHINE RISK GRID ── */
  .machine-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 0.5rem;
  }}
  .machine-chip {{
      font-size: 0.72rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid;
      white-space: nowrap;
  }}
  .chip-red   {{ background:rgba(218,54,51,0.12);  color:{DANGER};  border-color:rgba(218,54,51,0.35); }}
  .chip-amber {{ background:rgba(210,153,34,0.12); color:{WARN};    border-color:rgba(210,153,34,0.35); }}
  .chip-green {{ background:rgba(35,134,54,0.12);  color:{ACCENT};  border-color:rgba(35,134,54,0.35); }}
  /* ── POST-MAINTENANCE COMPARISON BOX ── */
  .comparison-box {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 0.85rem 1.1rem;
      margin-bottom: 0.5rem;
  }}
  .comparison-label {{
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
      color: {TEXT_SEC};
      margin-bottom: 0.3rem;
  }}
  .comparison-value {{
      font-size: 1.2rem;
      font-weight: 700;
  }}
  .comparison-delta {{
      font-size: 0.72rem;
      margin-top: 0.2rem;
  }}
  hr {{ border-color: {BORDER}; margin: 0.5rem 0; }}
  [data-testid="stDataFrame"] {{
      border: 1px solid {BORDER};
      border-radius: 8px;
      overflow: hidden;
  }}
  .stDownloadButton > button {{
      background: {ACCENT};
      color: white;
      border: none;
      border-radius: 6px;
      font-weight: 600;
      padding: 0.45rem 1.2rem;
  }}
  .stDownloadButton > button:hover {{ background: #2EA043; }}
  [data-testid="stAlert"] {{
      background: rgba(210,153,34,0.10);
      border: 1px solid rgba(210,153,34,0.3);
      border-radius: 6px;
  }}
  /* stTabs styling */
  [data-testid="stTab"] {{
      font-size: 0.82rem;
      font-weight: 500;
  }}
  /* ── KEY FINDING BANNER ── */
  .key-finding-banner {{
      background: #0D1117;
      border: 1px solid {BORDER};
      border-left: 6px solid #3CC2D2;
      border-radius: 0 8px 8px 0;
      padding: 16px 22px;
      margin: 28px 0 10px 0;
  }}
  .kfb-label {{
      font-size: 0.62rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #3CC2D2;
      margin-bottom: 6px;
  }}
  .kfb-text {{
      font-size: 0.88rem;
      color: {TEXT_PRI};
      line-height: 1.65;
  }}
  .kfb-text b {{ color: #3CC2D2; }}
  /* ── EXECUTIVE SUMMARY ── */
  .exec-summary-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-top: 4px solid #3CC2D2;
      border-radius: 8px;
      padding: 24px 30px;
      margin: 12px 0 28px 0;
      box-shadow: 0 4px 20px rgba(13,17,23,0.4);
  }}
  .exec-summary-card h3 {{
      font-size: 1rem;
      font-weight: 700;
      color: {TEXT_PRI};
      letter-spacing: 0.04em;
      margin: 0 0 14px 0;
      padding-bottom: 10px;
      border-bottom: 1px solid {BORDER};
  }}
  .exec-row {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      font-size: 0.82rem;
      color: {TEXT_SEC};
      padding: 8px 2px;
      line-height: 1.65;
      border-bottom: 1px solid {BORDER};
  }}
  .exec-row:last-of-type {{ border-bottom: none; }}
  .exec-row strong {{
      font-weight: 700;
      color: {TEXT_PRI};
      font-size: 0.84rem;
  }}
  .exec-row.alert strong {{ color: {DANGER}; }}
  .exec-row.warn  strong {{ color: {WARN};   }}
  .exec-row.good  strong {{ color: {ACCENT}; }}
  .exec-strategy {{
      background: #0D1117;
      border-left: 4px solid #3CC2D2;
      border-radius: 0 5px 5px 0;
      padding: 14px 18px;
      margin-top: 14px;
      font-size: 0.80rem;
      color: {TEXT_SEC};
      line-height: 1.65;
  }}
  .exec-strategy strong {{ color: {TEXT_PRI}; }}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def kpi_card(label, value, color="", sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color}">{value}</div>
      {sub_html}
    </div>"""

def dual_kpi_card(label1, value1, color1, label2, value2, color2, sub=""):
    sub_html = f'<div class="kpi-sub" style="margin-top:0.4rem;">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-dual">
      <div class="kpi-label" style="margin-bottom:0.5rem;">{label1} &nbsp;·&nbsp; {label2}</div>
      <div class="kpi-dual-row">
        <div class="kpi-dual-item">
          <div style="font-size:0.65rem;color:{TEXT_SEC};text-transform:uppercase;
                      letter-spacing:0.06em;margin-bottom:2px;">Score</div>
          <div class="kpi-value {color1}" style="font-size:1.35rem;">{value1}</div>
        </div>
        <div class="kpi-divider"></div>
        <div class="kpi-dual-item">
          <div style="font-size:0.65rem;color:{TEXT_SEC};text-transform:uppercase;
                      letter-spacing:0.06em;margin-bottom:2px;">Risk Label</div>
          <div class="kpi-value {color2}" style="font-size:1.05rem;">{value2}</div>
        </div>
      </div>
      {sub_html}
    </div>"""

def sectionp(title):
    st.markdown(
    f'<div class="section-header">{title}</div>',
    unsafe_allow_html=True
  )

# ── FUNCTION 1: tab_header() ─────────────────────────────────
# Call at the TOP of every "with tab:" block
# Renders a full-width professional tab title banner

def tab_header(tab_num, machine_id=None):
    """
    Full-width professional heading banner for each tab.
    tab_num : 1 to 5
    machine_id : pass selected_machine for Machine Analysis tab
    """
    meta     = TAB_META[tab_num]
    icon     = meta["icon"]
    title    = meta["title"]
    subtitle = meta["subtitle"]
    color    = meta["color"]
    tag      = meta["tag"]

    # Machine Analysis shows selected machine in title
    if tab_num == 2 and machine_id is not None:
        title    = "Machine Analysis — M-" + str(machine_id)
        subtitle = (
            "Anomaly trends · Sensor deviations · Correlation matrix "
            "for Machine M-" + str(machine_id)
        )

    html = (
        # Outer wrapper — dark card with colored left border
        "<div style='background:linear-gradient(135deg,"
        + CARD_BG + " 0%," + DARK_BG + " 100%);"
        "border:1px solid " + BORDER + ";"
        "border-left:4px solid " + color + ";"
        "border-radius:10px;"
        "padding:1rem 1.4rem 0.9rem 1.4rem;"
        "margin-bottom:0.75rem;'>"

        # Top row — tag pill + icon + title
        "<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"

        # Tag pill
        "<div style='background:" + color + ";"
        "color:#ffffff;"
        "font-size:0.60rem;font-weight:800;"
        "letter-spacing:0.14em;text-transform:uppercase;"
        "padding:3px 10px;border-radius:4px;"
        "flex-shrink:0;'>"
        + tag +
        "</div>"

        # Icon + Title
        "<div style='display:flex;align-items:center;gap:8px;'>"
        "<span style='font-size:1.35rem;line-height:1;'>" + icon + "</span>"
        "<span style='font-size:1.18rem;font-weight:700;"
        "color:" + TEXT_PRI + ";letter-spacing:-0.02em;line-height:1.2;'>"
        + title +
        "</span>"
        "</div>"

        # Thales teal dot separator
        "<div style='flex:1;height:1px;"
        "background:linear-gradient(90deg," + color + "20,transparent);'></div>"

        # Thales branding pill on right
        "<div style='background:rgba(60,194,210,0.10);"
        "border:1px solid rgba(60,194,210,0.25);"
        "border-radius:4px;padding:2px 10px;"
        "font-size:0.62rem;font-weight:600;color:#3CC2D2;"
        "letter-spacing:0.06em;flex-shrink:0;'>"
        "THALES GROUP"
        "</div>"

        "</div>"  # end top row

        # Subtitle row
        "<div style='margin-top:0.45rem;"
        "font-size:0.78rem;color:" + TEXT_SEC + ";line-height:1.5;"
        "padding-left:2px;'>"
        + subtitle +
        "</div>"

        "</div>"  # end outer wrapper
    )

    st.markdown(html, unsafe_allow_html=True)


# ── FUNCTION 2: section() ─────────────────────────────────────
# Use for major chart groups within a tab
# Replaces the old section() function entirely

def section(title, tab_num=None, icon=""):
    """
    Major section heading inside a tab.
    tab_num : optional — uses tab color if provided
    icon    : optional emoji prefix
    """
    color = TAB_META[tab_num]["color"] if tab_num else THALES_TEAL

    prefix = (icon + "&nbsp;&nbsp;") if icon else ""

    html = (
        # Left accent bar + title text
        "<div style='display:flex;align-items:center;gap:10px;"
        "margin:1.4rem 0 0.85rem 0;'>"

        # Vertical accent bar
        "<div style='width:3px;height:22px;border-radius:2px;"
        "background:" + color + ";flex-shrink:0;'></div>"

        # Title
        "<div style='font-size:0.82rem;font-weight:700;"
        "letter-spacing:0.06em;text-transform:uppercase;"
        "color:" + TEXT_PRI + ";'>"
        + prefix + title +
        "</div>"

        # Horizontal rule stretching to the right
        "<div style='flex:1;height:1px;"
        "background:linear-gradient(90deg," + color + "40,transparent);'>"
        "</div>"

        "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)


# ── FUNCTION 3: subsection() ─────────────────────────────────
# Use for smaller subdivisions within a section
# e.g. "before a table that follows a chart"

def subsection(title, tab_num=None):
    """
    Minor subsection heading — smaller than section().
    """
    color = TAB_META[tab_num]["color"] if tab_num else TEXT_SEC

    html = (
        "<div style='display:flex;align-items:center;gap:8px;"
        "margin:1rem 0 0.5rem 0;'>"

        # Small dot
        "<div style='width:6px;height:6px;border-radius:50%;"
        "background:" + color + ";flex-shrink:0;'></div>"

        # Title
        "<div style='font-size:0.72rem;font-weight:600;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "color:" + TEXT_SEC + ";'>"
        + title +
        "</div>"

        # Short rule
        "<div style='width:40px;height:1px;background:" + color + "50;'></div>"

        "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)


# ── FUNCTION 4: divider() ────────────────────────────────────
# Thales-branded gradient divider between major sections

def divider():
    """Thales gradient line — use between major sections."""
    st.markdown(
        "<div style='height:2px;"
        "background:linear-gradient(90deg,"
        + THALES_NAVY + " 0%,"
        + THALES_TEAL + " 40%,"
        + THALES_NAVY + " 100%);"
        "border-radius:1px;"
        "margin:1.2rem 0 0.8rem 0;opacity:0.6;'>"
        "</div>",
        unsafe_allow_html=True
    )




def badge(text, color="green"):
    return f'<span class="badge badge-{color}">{text}</span>'

def panel_html(title, rows, border_color=None):
    """
    Right-side breakdown panel alongside a chart.
    rows = list of (label, value, color_hex_optional)
    Uses string concatenation NOT f-string to avoid
    curly-brace conflicts in CSS inside Streamlit markdown.
    """
    bc = border_color or ACCENT2
    items_html = ""
    for row in rows:
        label = str(row[0])
        value = str(row[1])
        color = row[2] if len(row) > 2 else TEXT_PRI
        items_html += (
            "<div style='margin-bottom:0.65rem;padding-bottom:0.65rem;"
            "border-bottom:0.5px solid " + BORDER + ";'>"
            "<div style='font-size:0.65rem;text-transform:uppercase;"
            "letter-spacing:0.07em;font-weight:600;color:" + TEXT_SEC + ";"
            "margin-bottom:3px;'>" + label + "</div>"
            "<div style='font-size:0.88rem;font-weight:600;color:" + color + ";"
            "line-height:1.4;'>" + value + "</div>"
            "</div>"
        )
    return (
        "<div style='background:" + CARD_BG + ";border:1px solid " + BORDER + ";"
        "border-left:3px solid " + bc + ";border-radius:8px;"
        "padding:0.85rem 1rem;height:405px;" 
        "box-sizing:border-box;"
        "overflow:hidden;'>"
        "<div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.10em;color:" + bc + ";margin-bottom:0.75rem;'>"
        + title
        + "</div>"
        + items_html
        + "</div>"
    )

def finding_text(text, icon="💡"):
    """Finding / insight text shown below every chart."""
    html = (
        "<div style='background:" + CARD_BG + ";border:0.5px solid " + BORDER + ";"
        "border-left:3px solid " + ACCENT2 + ";border-radius:6px;"
        "padding:0.55rem 0.9rem;margin-top:0.4rem;margin-bottom:0.6rem;'>"
        "<span style='font-size:0.78rem;color:" + TEXT_SEC + ";line-height:1.65;'>"
        "<b style='color:" + ACCENT2 + ";'>" + icon + " Finding:</b>&nbsp;"
        + text +
        "</span></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def detailed_key_findings(tab_label, text):
    """Formal key finding banner — one per tab, based on full dataset.
    Matches the bank_churn_app.py pattern exactly.
    Uses CSS classes defined in the <style> block above."""
    st.markdown(
        "<div class='key-finding-banner'>"
        "<div class='kfb-label'>Key Finding \u2014 " + str(tab_label) + "</div>"
        "<div class='kfb-text'>" + str(text) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

def styled_fig(figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    return fig, ax

RISK_COLOR = {
    "High Risk":   DANGER,
    "Medium Risk": WARN,
    "Low Risk":    ACCENT,
}
ACTION_COLOR = {
    "Immediate Maintenance Required": DANGER,
    "Schedule Inspection":            WARN,
    "Normal Monitoring":              ACCENT,
}

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():

    import zipfile
    import io

    # ── HELPER: read CSV from zip or raw file ────────────────
    # On Streamlit Cloud (GitHub): zip files are present
    # On local machine: raw CSV files are present
    # This function handles both automatically.
    def read_csv_from_zip_or_file(zip_name, csv_fallback):
        if os.path.exists(zip_name):
            with zipfile.ZipFile(zip_name, "r") as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise FileNotFoundError(
                        f"No CSV found inside {zip_name}"
                    )
                with z.open(csv_names[0]) as f:
                    return pd.read_csv(io.BytesIO(f.read()))
        else:
            return pd.read_csv(csv_fallback)

    # ── STEP 1: Load main Cascade KPI file ───────────────────
    # GitHub zip : predictivemaintenancekpioutputcascade.zip
    # Local CSV  : Predictive_Maintenance_KPI_Output_Cascade.csv
    df = read_csv_from_zip_or_file(
        "predictivemaintenancekpioutputcascade.zip",
        "Predictive_Maintenance_KPI_Output_Cascade.csv"
    )

    # ── STEP 2: Parse Datetime and sort ──────────────────────
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Timestamp"],
        dayfirst=True, errors="coerce"
    )
    df = df.sort_values(
        ["Machine_ID", "Datetime"]
    ).reset_index(drop=True)

    # ── STEP 3: Ensure Normalized_Anomaly_Score exists ───────
    if "Normalized_Anomaly_Score" not in df.columns:
        df["Normalized_Anomaly_Score"] = (
            df["Anomaly_Index"].rank(pct=True) * 100
        ).round(2)

    # ── STEP 4: Ensure Cascade_Failure_Pattern exists ────────
    if "Cascade_Failure_Pattern" not in df.columns:
        df["Cascade_Failure_Pattern"] = "No Pattern"

    # ── STEP 5: Attach AE columns from zip ───────────────────
    # GitHub zip : ac.zip
    # Local CSV  : Autoencoder_Anomaly_Output.csv
    ae_cols = [
        "AE_Reconstruction_Error",
        "AE_Anomaly_Score",
        "AE_Anomaly_Label",
        "AE_Anomaly_Status",
        "AE_Anomaly_Severity",
    ]
    try:
        df_ae = read_csv_from_zip_or_file(
            "ac.zip",
            "Autoencoder_Anomaly_Output.csv"
        )
        df_ae["Datetime"] = pd.to_datetime(
            df_ae["Date"] + " " + df_ae["Timestamp"],
            dayfirst=True, errors="coerce"
        )
        df_ae = df_ae.sort_values(
            ["Machine_ID", "Datetime"]
        ).reset_index(drop=True)

        n = min(len(df), len(df_ae))
        ae_cols_present = [c for c in ae_cols if c in df_ae.columns]
        for col in ae_cols_present:
            df.loc[:n-1, col] = df_ae.loc[:n-1, col].values

        df["AE_Model_Loaded"] = True

    except Exception:
        for col in ae_cols:
            df[col] = np.nan
        df["AE_Model_Loaded"] = False

    # ── STEP 6: Final sort by Datetime ───────────────────────
    df = df.sort_values("Datetime").reset_index(drop=True)

    # ── STEP 7: Compute Ensemble columns ─────────────────────
    if "AE_Anomaly_Score" in df.columns and df["AE_Anomaly_Score"].notna().any():
        df["Ensemble_Anomaly_Score"] = (
            (df["Normalized_Anomaly_Score"] +
             df["AE_Anomaly_Score"]) / 2
        ).round(2)
    else:
        df["Ensemble_Anomaly_Score"] = df["Normalized_Anomaly_Score"]

    df["Ensemble_Risk"] = np.select(
        [
            df["Ensemble_Anomaly_Score"] >= 97,
            df["Ensemble_Anomaly_Score"] >= 66,
        ],
        ["High Risk", "Medium Risk"],
        default="Low Risk"
    )

    return df
df = load_data()

# Flag: is AE data available?
AE_AVAILABLE = bool(df["AE_Model_Loaded"].iloc[0])

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.75rem 0 1.2rem 0;">
      <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                  color:{TEXT_SEC};font-weight:600;">Thales Manufacturing IoT</div>
      <div style="font-size:1.05rem;font-weight:700;color:{TEXT_PRI};margin-top:2px;">
        Dashboard Controls
      </div>
    </div>
    <hr style="border-color:{BORDER};margin-bottom:1.2rem;">
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:4px;">Machine</div>',
                unsafe_allow_html=True)
    machine_list    = sorted(df["Machine_ID"].unique())
    selected_machine = st.selectbox("", machine_list, label_visibility="collapsed")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:4px;">'
                f'Anomaly Index Threshold</div>', unsafe_allow_html=True)
    risk_threshold = st.slider(
        "", float(df["Anomaly_Index"].min()),
        float(df["Anomaly_Index"].max()), 0.0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:4px;">Date Range</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("From", value=df["Datetime"].min().date(),
                                   min_value=df["Datetime"].min().date(),
                                   max_value=df["Datetime"].max().date())
    with c2:
        end_date   = st.date_input("To", value=df["Datetime"].max().date(),
                                   min_value=df["Datetime"].min().date(),
                                   max_value=df["Datetime"].max().date())

    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:4px;">Time Window</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        start_time = st.time_input("Start", value=time(0, 0))
    with c4:
        end_time   = st.time_input("End",   value=time(23, 59))

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:4px;">Operation Mode</div>',
                unsafe_allow_html=True)
    op_modes      = df["Operation_Mode"].unique().tolist()
    selected_mode = st.multiselect("", op_modes, default=op_modes,
                                   label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER};margin:1.5rem 0 0.75rem 0;'>",
                unsafe_allow_html=True)
    ae_status_color = ACCENT if AE_AVAILABLE else WARN
    ae_status_text  = "✓ Autoencoder loaded" if AE_AVAILABLE else "⚠ Autoencoder not loaded"
    st.markdown(
        f'<div style="font-size:0.68rem;color:{TEXT_SEC};line-height:1.8;">'
        f'Dataset: 100,000 readings · 50 machines<br>'
        f'Model 1: Isolation Forest · Features: 12<br>'
        f'<span style="color:{ae_status_color};font-weight:600;">'
        f'{ae_status_text}</span></div>',
        unsafe_allow_html=True
    )
    if not AE_AVAILABLE:
        st.markdown(
            f'<div style="font-size:0.65rem;color:{WARN};margin-top:4px;">'
            f'Run autoencoder_anomaly.py first<br>'
            f'then place Autoencoder_Anomaly_Output.csv<br>'
            f'in the same folder as thales_app.py</div>',
            unsafe_allow_html=True
        )

# ── APPLY FILTERS ────────────────────────────────────────────
start_dt = pd.to_datetime(str(start_date) + " " + str(start_time))
end_dt   = pd.to_datetime(str(end_date)   + " " + str(end_time))

filtered_df = df[
    (df["Machine_ID"] == selected_machine) &
    (df["Operation_Mode"].isin(selected_mode)) &
    (df["Anomaly_Index"] >= risk_threshold) &
    (df["Datetime"] >= start_dt) &
    (df["Datetime"] <= end_dt)
]

dashboard_df = df[
    (df["Operation_Mode"].isin(selected_mode)) &
    (df["Anomaly_Index"] >= risk_threshold) &
    (df["Datetime"] >= start_dt) &
    (df["Datetime"] <= end_dt)
]

if filtered_df.empty or dashboard_df.empty:
    st.warning("No data matches the selected filters. Adjust the sidebar controls.")
    st.stop()

machine_data = filtered_df[filtered_df["Machine_ID"] == selected_machine].copy()

# ── PAGE HEADER ───────────────────────────────────────────────
latest_risk = filtered_df["Predictive_Maintenance_Risk"].iloc[-1]
risk_col    = ("red" if latest_risk == "High Risk"
               else "amber" if latest_risk == "Medium Risk"
               else "green")

# Thales brand colors
THALES_NAVY = "#171F69"
THALES_TEAL = "#3CC2D2"

risk_color_map = {
    "High Risk":   ("#DA3633", "rgba(218,54,51,0.15)"),
    "Medium Risk": ("#D29922", "rgba(210,153,34,0.15)"),
    "Low Risk":    ("#238636", "rgba(35,134,54,0.15)"),
}
risk_hex, risk_bg = risk_color_map.get(latest_risk, ("#8B949E", "rgba(139,148,158,0.15)"))

ae_badge_text  = "Autoencoder ✓" if AE_AVAILABLE else "Autoencoder ⚠"
ae_badge_color = "#238636" if AE_AVAILABLE else "#D29922"
ae_badge_bg    = "rgba(35,134,54,0.15)" if AE_AVAILABLE else "rgba(210,153,34,0.15)"

total_machines     = dashboard_df["Machine_ID"].nunique()
high_risk_machines = (dashboard_df[dashboard_df["Predictive_Maintenance_Risk"]
                       == "High Risk"]["Machine_ID"].nunique())

st.markdown(
    "<div style='background:linear-gradient(135deg,#171F69 0%,#0D1117 60%);"
    "border-radius:12px;padding:1.4rem 1.8rem 1.2rem 1.8rem;"
    "margin-bottom:0.5rem;border:1px solid #30363D;"
    "box-shadow:0 4px 24px rgba(23,31,105,0.4);'>"

    # ── Top row: logo text + badges ───────────────────────────
    "<div style='display:flex;align-items:flex-start;justify-content:space-between;"
    "flex-wrap:wrap;gap:0.75rem;'>"

    # Left: Brand + Title
    "<div>"
    "<div style='display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;'>"
    "<div style='background:#3CC2D2;border-radius:6px;padding:4px 10px;"
    "font-size:0.72rem;font-weight:800;letter-spacing:0.15em;color:#0D1117;'>"
    "THALES GROUP"
    "</div>"
    "<div style='font-size:0.68rem;color:#8B949E;letter-spacing:0.08em;'>"
    "Manufacturing IoT · 6G Predictive Maintenance"
    "</div>"
    "</div>"
    "<div style='font-size:1.75rem;font-weight:800;color:#FFFFFF;"
    "letter-spacing:-0.03em;line-height:1.15;text-shadow:0 2px 8px rgba(0,0,0,0.5);'>"
    "AI Predictive Maintenance Dashboard"
    "</div>"
    "<div style='font-size:0.82rem;color:#8B949E;margin-top:0.3rem;'>"
    "Anomaly Detection &amp; Risk Classification · Isolation Forest + Autoencoder"
    "</div>"
    "</div>"

    # Right: Status badges
    "<div style='display:flex;flex-direction:column;gap:6px;align-items:flex-end;"
    "min-width:180px;'>"

    # Machine status badge
    "<div style='background:" + risk_bg + ";border:1px solid " + risk_hex + ";"
    "border-radius:6px;padding:5px 12px;text-align:right;'>"
    "<div style='font-size:0.62rem;color:#8B949E;text-transform:uppercase;"
    "letter-spacing:0.08em;margin-bottom:2px;'>Machine M-" + str(selected_machine) + " Status</div>"
    "<div style='font-size:0.82rem;font-weight:700;color:" + risk_hex + ";'>" + latest_risk + "</div>"
    "</div>"

    # AE model badge
    "<div style='background:" + ae_badge_bg + ";border:1px solid " + ae_badge_color + ";"
    "border-radius:6px;padding:4px 12px;text-align:right;'>"
    "<div style='font-size:0.75rem;font-weight:600;color:" + ae_badge_color + ";'>"
    + ae_badge_text +
    "</div>"
    "</div>"

    "</div>"  # end right column
    "</div>"  # end top row

    # ── Bottom row: 4 quick-stat chips ───────────────────────
    "<div style='display:flex;gap:10px;margin-top:1rem;flex-wrap:wrap;'>"

    "<div style='background:rgba(255,255,255,0.06);border:0.5px solid #30363D;"
    "border-radius:6px;padding:5px 14px;'>"
    "<span style='font-size:0.65rem;color:#8B949E;text-transform:uppercase;"
    "letter-spacing:0.07em;'>Total Machines</span>"
    "<span style='font-size:0.9rem;font-weight:700;color:#E6EDF3;margin-left:8px;'>"
    + str(total_machines) + "</span>"
    "</div>"

    "<div style='background:rgba(218,54,51,0.10);border:0.5px solid rgba(218,54,51,0.3);"
    "border-radius:6px;padding:5px 14px;'>"
    "<span style='font-size:0.65rem;color:#8B949E;text-transform:uppercase;"
    "letter-spacing:0.07em;'>High Risk</span>"
    "<span style='font-size:0.9rem;font-weight:700;color:#DA3633;margin-left:8px;'>"
    + str(high_risk_machines) + " machines</span>"
    "</div>"

    "<div style='background:rgba(255,255,255,0.06);border:0.5px solid #30363D;"
    "border-radius:6px;padding:5px 14px;'>"
    "<span style='font-size:0.65rem;color:#8B949E;text-transform:uppercase;"
    "letter-spacing:0.07em;'>Readings in View</span>"
    "<span style='font-size:0.9rem;font-weight:700;color:#E6EDF3;margin-left:8px;'>"
    + f"{len(dashboard_df):,}" + "</span>"
    "</div>"

    "<div style='background:rgba(60,194,210,0.08);border:0.5px solid rgba(60,194,210,0.3);"
    "border-radius:6px;padding:5px 14px;'>"
    "<span style='font-size:0.65rem;color:#8B949E;text-transform:uppercase;"
    "letter-spacing:0.07em;'>Selected Machine</span>"
    "<span style='font-size:0.9rem;font-weight:700;color:#3CC2D2;margin-left:8px;'>"
    "M-" + str(selected_machine) + "</span>"
    "</div>"

    "</div>"  # end bottom chips row
    "</div>",  # end full header div
    unsafe_allow_html=True
)

st.markdown("<div style='margin-bottom:0.25rem;'></div>", unsafe_allow_html=True)

# ============================================================
# KEY FINDINGS SECTION
# All values computed from FULL df (100,000 rows) — NOT filtered dashboard_df
# ============================================================


# ── PRE-COMPUTE ALL FINDING VALUES FROM FULL DATASET ─────────

# Tab 1 — Predictive Maintenance Overview
kf_risk_counts   = df["Predictive_Maintenance_Risk"].value_counts()
kf_hr_count      = kf_risk_counts.get("High Risk", 0)
kf_mr_count      = kf_risk_counts.get("Medium Risk", 0)
kf_lr_count      = kf_risk_counts.get("Low Risk", 0)
kf_hr_pct        = round(kf_hr_count / len(df) * 100, 1)
kf_mr_pct        = round(kf_mr_count / len(df) * 100, 1)

kf_top3_machines = (
    df.groupby("Machine_ID")["Anomaly_Score"]
    .mean().sort_values(ascending=True).head(3)
)
kf_top3_ids = [f"M-{m}" for m in kf_top3_machines.index.tolist()]

kf_op_risk       = df.groupby("Operation_Mode")["Anomaly_Score"].mean().sort_values(ascending=True)
kf_riskiest_mode = kf_op_risk.index[0]
kf_riskiest_score= round(kf_op_risk.iloc[0], 4)

kf_imm_action    = (df["Maintenance_Action"] == "Immediate Maintenance Required").sum()

# Tab 2 — Machine Anomaly Dashboard (fleet-wide values from df)
kf_m_anom_rate   = round((df["Anomaly_Label"] == -1).mean() * 100, 2)
kf_m_avg_score   = round(df["Anomaly_Score"].mean(), 4)
kf_m_lead_time   = round(df["Early_Warning_Lead_Time_Hours"].mean(), 1)
kf_m_anom_count  = int((df["Anomaly_Label"] == -1).sum())
kf_corr_cols     = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Error_Rate_%", "Anomaly_Score"]
kf_corr          = df[kf_corr_cols].corr()
kf_top_corr      = kf_corr["Anomaly_Score"].drop("Anomaly_Score").abs().idxmax()
kf_top_corr_val  = round(kf_corr["Anomaly_Score"].drop("Anomaly_Score").abs().max(), 3)
kf_m_sev         = df["Anomaly_Severity"].value_counts()
kf_m_top_sev     = kf_m_sev.idxmax()
kf_m_top_sev_pct = round(kf_m_sev.max() / kf_m_sev.sum() * 100, 1)

# Tab 3 — Maintenance Alert Panel
kf_casc          = df["Cascade_Risk_Level"].value_counts()
kf_hi_casc       = kf_casc.get("High Cascade Risk", 0)
kf_hi_casc_pct   = round(kf_hi_casc / len(df) * 100, 1)
kf_casc_top      = df.groupby("Machine_ID")["Cascading_Failure_Index"].mean().sort_values(ascending=False)
kf_top_casc_m    = f"M-{kf_casc_top.index[0]}"
kf_top_casc_v    = round(kf_casc_top.iloc[0], 2)

# Tab 4 — Historical Risk Analysis
kf_fleet_lt_mean  = round(df["Early_Warning_Lead_Time_Hours"].mean(), 1)
kf_fleet_lt_min   = round(df["Early_Warning_Lead_Time_Hours"].min(), 1)
kf_critical_lt    = int((df["Early_Warning_Lead_Time_Hours"] <= 25).sum())
kf_escalating_flag= int((df["Risk_Trend_Status"] == "Escalating").sum())
kf_improving_flag = int((df["Risk_Trend_Status"] == "Improving").sum())
kf_maint_data_kf  = df[df["Operation_Mode"] == "Maintenance"]
kf_active_data_kf = df[df["Operation_Mode"] != "Maintenance"]
kf_score_maint    = round(kf_maint_data_kf["Anomaly_Score"].mean(), 4) if not kf_maint_data_kf.empty else 0.0
kf_score_active   = round(kf_active_data_kf["Anomaly_Score"].mean(), 4) if not kf_active_data_kf.empty else 0.0
kf_maint_improves = kf_score_maint > kf_score_active

# Tab 5 — Autoencoder Analysis
if AE_AVAILABLE:
    kf_ae_count   = int((df["AE_Anomaly_Label"] == -1).sum())
    kf_ae_pct     = round(kf_ae_count / len(df) * 100, 1)
    kf_both_agree = int(((df["Anomaly_Label"] == -1) & (df["AE_Anomaly_Label"] == -1)).sum())
    kf_agree_pct  = round((df["Anomaly_Label"] == df["AE_Anomaly_Label"]).mean() * 100, 1)
    kf_top_ens_m  = "M-" + str(df.groupby("Machine_ID")["Ensemble_Anomaly_Score"].mean().idxmax())
    kf_avg_recon  = round(df["AE_Reconstruction_Error"].mean(), 6)
else:
    kf_ae_count = kf_ae_pct = kf_both_agree = kf_agree_pct = 0
    kf_top_ens_m = "N/A (run autoencoder first)"
    kf_avg_recon = 0.0

# ── BUILD EACH CARD HTML ──────────────────────────────────────
# Using string concatenation to avoid CSS curly brace conflicts

def kf_card(border_color, icon_label, points):
    """Build a key findings card. points = list of (dot_color, html_text)"""
    html = (
        "<div style='background:" + CARD_BG + ";border:1px solid " + BORDER + ";"
        "border-top:3px solid " + border_color + ";border-radius:8px;"
        "padding:0.8rem 0.9rem;height:100%;'>"
        "<div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.10em;color:" + border_color + ";margin-bottom:0.6rem;'>"
        + icon_label +
        "</div>"
        "<div style='font-size:0.78rem;color:" + TEXT_PRI + ";line-height:1.8;'>"
    )
    for i, (dot_color, text) in enumerate(points, 1):
        html += (
            "<div style='margin-bottom:0.55rem;display:flex;gap:6px;align-items:flex-start;'>"
            "<div style='min-width:18px;height:18px;border-radius:50%;"
            "background:" + dot_color + ";display:flex;align-items:center;"
            "justify-content:center;font-size:0.62rem;font-weight:700;"
            "color:#fff;margin-top:1px;flex-shrink:0;'>"
            + str(i) +
            "</div>"
            "<div>" + text + "</div>"
            "</div>"
        )
    html += "</div></div>"
    return html

# ── CARD 1: Fleet Overview ────────────────────────────────────
card1 = kf_card(
    border_color=DANGER,
    icon_label="📊  Fleet Overview",
    points=[
        (DANGER,
         "<b>" + str(kf_hr_count) + " High Risk readings</b> ("
         + str(kf_hr_pct) + "% of fleet). "
         + str(kf_imm_action) + " require immediate action now."),
        (WARN,
         "Most anomalous machines: <b>"
         + ", ".join(kf_top3_ids)
         + "</b> — lowest Anomaly Scores in the fleet."),
        (ACCENT2,
         "<b>" + kf_riskiest_mode + "</b> is the riskiest operation mode "
         "(avg score <b>" + str(kf_riskiest_score) + "</b>). "
         "Plan maintenance windows for this mode."),
    ]
)

# ── CARD 2: Machine Analysis ──────────────────────────────────
card2 = kf_card(
    border_color=ACCENT2,
    icon_label="🔍  Machine M-" + str(selected_machine),
    points=[
        (DANGER,
         "<b>" + str(kf_m_anom_count) + " anomaly events</b> detected "
         "(rate: <b>" + str(kf_m_anom_rate) + "%</b>). "
         "Avg score: " + str(kf_m_avg_score) + "."),
        (WARN,
         "<b>" + kf_top_corr + "</b> strongest sensor correlation "
         "(|r| = <b>" + str(kf_top_corr_val) + "</b>). "
         "Strongest observed early-warning indicator in this correlation analysis."),
        (ACCENT2,
         "<b>" + kf_m_top_sev + "</b> severity dominates ("
         + str(kf_m_top_sev_pct) + "% of events). "
         "Avg lead time: <b>" + str(kf_m_lead_time) + " hrs</b>."),
    ]
)

# ── CARD 3: Maintenance Alerts ────────────────────────────────
card3 = kf_card(
    border_color=WARN,
    icon_label="🚨  Maintenance Alerts",
    points=[
        (DANGER,
         "<b>" + str(kf_imm_action) + " immediate maintenance readings</b> — "
         "confirmed High Risk events needing urgent intervention."),
        (WARN,
         "<b>" + str(kf_hi_casc) + " readings</b> ("
         + str(kf_hi_casc_pct) + "%) carry "
         "High Cascade Risk — failure propagates to other machines."),
        (ACCENT2,
         "<b>" + kf_top_casc_m + "</b> has highest Cascading Failure Index "
         "(<b>" + str(kf_top_casc_v) + "</b>) — highest systemic fleet risk."),
    ]
)

# ── CARD 4: Historical Trends ─────────────────────────────────
trend_verdict  = "Escalating" if kf_escalating_flag > kf_improving_flag else "Improving"
trend_dot_col  = DANGER if trend_verdict == "Escalating" else ACCENT
maint_verdict  = "improves after maintenance ✓" if kf_maint_improves else "needs review — no improvement"

card4 = kf_card(
    border_color=ACCENT,
    icon_label="📈  Historical Trends",
    points=[
        (DANGER,
         "Fleet avg lead time: <b>" + str(kf_fleet_lt_mean) + " hrs</b>. "
         "<b>" + str(kf_critical_lt) + " readings</b> fell below "
         "25-hr critical threshold."),
        (trend_dot_col,
         "M-" + str(selected_machine) + " risk is "
         "<b style='color:" + trend_dot_col + ";'>" + trend_verdict + "</b> "
         "(" + str(kf_escalating_flag) + " escalating vs "
         + str(kf_improving_flag) + " improving periods)."),
        (ACCENT2,
         "Post-maintenance health <b>" + maint_verdict + "</b>. "
         "Maint score: <b>" + str(kf_score_maint) + "</b> vs "
         "active: <b>" + str(kf_score_active) + "</b>."),
    ]
)

# ── CARD 5: Autoencoder ───────────────────────────────────────
if AE_AVAILABLE:
    card5 = kf_card(
        border_color="#7F77DD",
        icon_label="🤖  Autoencoder",
        points=[
            (DANGER,
             "AE detected <b>" + str(kf_ae_count) + " anomalies</b> ("
             + str(kf_ae_pct) + "%). Both models agree on "
             "<b>" + str(kf_both_agree) + "</b> events — highest confidence."),
            (WARN,
             "Model agreement rate: <b>" + str(kf_agree_pct) + "%</b>. "
             "Disagreements reveal subtle faults each model captures differently."),
            (ACCENT2,
             "Top ensemble machine: <b>" + kf_top_ens_m + "</b>. "
             "Avg reconstruction error: <b>" + str(kf_avg_recon) + "</b>."),
        ]
    )
else:
    card5 = kf_card(
        border_color=WARN,
        icon_label="🤖  Autoencoder",
        points=[
            (WARN,
             "<b>Autoencoder output not loaded.</b> "
             "Run autoencoder_anomaly.py first."),
            (TEXT_SEC,
             "Place <b>Autoencoder_Anomaly_Output.csv</b> in the same folder as "
             "thales_app.py and restart the dashboard."),
            (ACCENT2,
             "Tab 5 will then show full AE vs IF comparison, "
             "reconstruction error trends and ensemble findings."),
        ]
    )

# ── RENDER KEY FINDINGS EXPANDER ─────────────────────────────

#st.markdown(card1, unsafe_allow_html=True)


# ── THALES GRADIENT SEPARATOR ────────────────────────────────
st.markdown(
    "<div style='height:4px;background:linear-gradient(90deg,"
    + THALES_NAVY + " 0%," + THALES_TEAL + " 50%," + THALES_NAVY + " 100%);"
    "border-radius:2px;margin:0.75rem 0 0.5rem 0;'></div>",
    unsafe_allow_html=True
)


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Predictive Maintenance Overview ",
    "🔍  Machine Anomaly Dashboard ",
    "🚨  Maintenance Alert Panel",
    "📈  Historical Risk Analysis",
    "🤖  Autoencoder Analysis",    
    ])

# ════════════════════════════════════════════════════════════
# TAB 1 — FLEET OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:

    # ── KPI ROW ──────────────────────────────────────────────
    # CHANGE 1: KPI 3 is now a DUAL card showing Anomaly Score + Risk Label
    tab_header(1)              # ← add this first line
    section("Fleet KPIs", tab_num=1)

    sectionp("Fleet KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)

    total_machines      = dashboard_df["Machine_ID"].nunique()
    high_risk_machines  = (dashboard_df[dashboard_df["Predictive_Maintenance_Risk"]
                           == "High Risk"]["Machine_ID"].nunique())
    high_risk_pct       = round(high_risk_machines / total_machines * 100, 1)
    avg_anomaly_score   = round(filtered_df["Anomaly_Score"].mean(), 4)
    most_common_risk    = filtered_df["Predictive_Maintenance_Risk"].value_counts().idxmax()
    avg_lead_time       = round(filtered_df["Early_Warning_Lead_Time_Hours"].mean(), 1)
    avg_dpi             = round(filtered_df["Downtime_Prevention_Index"].mean(), 2)

    # Score color: higher positive = more normal = green; lower = amber/red
    score_color = ("green" if avg_anomaly_score > 0.08
                   else "amber" if avg_anomaly_score > 0.03
                   else "red")
    risk_label_color = ("green" if most_common_risk == "Low Risk"
                        else "amber" if most_common_risk == "Medium Risk"
                        else "red")

    
    k1.markdown(kpi_card(
                "Anomaly Score", avg_anomaly_score, score_color,
                sub="Higher score = more normal<br> (IF convention)"
            ), unsafe_allow_html=True)
    
        
            
    k2.markdown(kpi_card("Maintenance Risk Label",most_common_risk ,risk_label_color ,
                              "50 machines<br> in fleet"), unsafe_allow_html=True)
        

    k3.markdown(kpi_card("High Risk Machines", high_risk_machines, "red",
                          f"{high_risk_pct}% <br>of<br> fleet"), unsafe_allow_html=True)

    k4.markdown(kpi_card("Avg Lead Time", f"{avg_lead_time} hrs", "blue",
                          "Hours before <br>predicted<br> failure"), unsafe_allow_html=True)

    k5.markdown(kpi_card("Downtime Prevention Index", avg_dpi, "green",
                          "Scale: 2 (critical) → 10 (safe)"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    # ── RISK DISTRIBUTION + MACHINE RISK GRID ────────────────
    subsection("Risk Level Distribution & Machine Breakdown", tab_num=1)

    sectionp("Risk Level Distribution & Machine Breakdown")
    #col_a, col_b = st.columns([1, 1])
    
    risk_counts = dashboard_df["Predictive_Maintenance_Risk"].value_counts()
    dominant_risk     = risk_counts.idxmax()
    dominant_risk_pct = round(risk_counts.max() / len(dashboard_df) * 100, 1)
    hr_count = risk_counts.get("High Risk", 0)
    mr_count = risk_counts.get("Medium Risk", 0)
    lr_count = risk_counts.get("Low Risk", 0)

    c_chart, c_panel = st.columns([7, 3])
    with c_chart:
        fig, ax = styled_fig(figsize=(7, 4.5))
        colors  = [RISK_COLOR.get(r, ACCENT2) for r in risk_counts.index]
        bars    = ax.bar(risk_counts.index, risk_counts.values,
                            color=colors, width=0.5, zorder=2)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 60,
                    f"{bar.get_height():,}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_PRI)
        ax.set_title("Reading Count by Risk Level", pad=10)
        ax.set_ylabel("Number of Readings")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_panel:
        st.markdown(panel_html("Breakdown", [
            ("High Risk Readings",   f"{hr_count:,}",  DANGER),
            ("Medium Risk Readings", f"{mr_count:,}",  WARN),
            ("Low Risk Readings",    f"{lr_count:,}",  ACCENT),
            ("Dominant Category",    dominant_risk,    TEXT_PRI),
            ("Dominant Share",       f"{dominant_risk_pct}%", ACCENT2),
            ("Total Readings",       f"{len(dashboard_df):,}", TEXT_SEC),
        ], border_color=ACCENT2), unsafe_allow_html=True)

    finding_text(
        f"<b>{dominant_risk}</b> dominates with <b>{dominant_risk_pct}%</b> "
        f"of all readings ({lr_count:,} readings). "
        f"High Risk accounts for <b>{round(hr_count/len(dashboard_df)*100,1)}%</b> "
        f"— these are the priority machines requiring immediate attention."
    )

    sectionp("Risk Readings Per Machine (Top 15)")
    machine_risk = (
                dashboard_df.groupby(["Machine_ID", "Predictive_Maintenance_Risk"])
                .size().unstack(fill_value=0).reset_index()
            )
    for col in ["High Risk", "Medium Risk", "Low Risk"]:
        if col not in machine_risk.columns:
            machine_risk[col] = 0
    machine_risk["Machine"] = "M-" + machine_risk["Machine_ID"].astype(str)
    machine_risk = machine_risk.sort_values("High Risk", ascending=False).head(15)
    top_hr_machine   = machine_risk.iloc[0]["Machine"]
    top_hr_readings  = int(machine_risk.iloc[0]["High Risk"])
    avg_hr_per_mach  = round(machine_risk["High Risk"].mean(), 1)

# High risk rate per machine
    machine_hr_rate_per_machine = ( 
            dashboard_df.groupby("Machine_ID") 
           .apply(lambda x: (x["Predictive_Maintenance_Risk"] == "High Risk").mean() 
            * 100) 
          )


    #calculate the machine with the highest High Risk rate: 

    top_hr_rate_machine12 = machine_hr_rate_per_machine.idxmax() 
    top_hr_rate12 = round(machine_hr_rate_per_machine.max(), 1)



    c_chart2, c_panel2 = st.columns([7, 3])
    with c_chart2:
        fig, ax = styled_fig(figsize=(7, 4.2))
        x = np.arange(len(machine_risk))
        w = 0.28
        ax.bar(x - w, machine_risk["High Risk"],   w, color=DANGER, label="High Risk",   zorder=2)
        ax.bar(x,     machine_risk["Medium Risk"], w, color=WARN,   label="Medium Risk", zorder=2)
        ax.bar(x + w, machine_risk["Low Risk"],    w, color=ACCENT, label="Low Risk",    zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(machine_risk["Machine"], rotation=45, ha="right", fontsize=8)
        ax.set_title("Risk Readings per Machine (Top 15)", pad=10)
        ax.set_ylabel("Number of Readings")
        ax.legend(loc="upper right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_panel2:
        st.markdown(panel_html("Breakdown", [
            ("Top HR Machine",       top_hr_machine,         DANGER),
            ("Its HR Readings",      f"{top_hr_readings:,}", DANGER),
            ("Avg HR per Machine",   f"{avg_hr_per_mach}",   WARN),
            ("Machines Shown",       "Top 15 by HR count",   TEXT_SEC),
            ("Sort Order",           "High Risk ↓",          TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    finding_text(
        f"<b>{top_hr_machine}</b> has the highest absolute number of High Risk " 
        f"readings, with <b>{top_hr_readings:,}</b> events, compared with an average "
        f"of <b>{avg_hr_per_mach}</b> High Risk readings per machine. " 
        f"Together, these metrics identify <b>M-{top_hr_rate_machine12}</b> as the " 
        f"machine with the greatest risk concentration and <b>{top_hr_machine}</b> " 
        f"as the machine with the greatest total High Risk event burden, supporting " 
        f"more informed maintenance prioritization."
        
        
          )

    
    






    # ── MACHINE RISK CHIP GRID ────────────────────────────────
    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    # High Risk machines
    hr_machines = (
        dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == "High Risk"]
        ["Machine_ID"].unique()
    )
    mr_machines = (
        dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == "Medium Risk"]
        ["Machine_ID"].unique()
    )
    lr_machines = (
        dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == "Low Risk"]
        ["Machine_ID"].unique()
    )

    def make_chips(ids, css_class):
        return "".join([
            f'<span class="machine-chip {css_class}">M-{i}</span>'
            for i in sorted(ids)
        ])

    
    # ── HIGH RISK MACHINE TABLE ───────────────────────────────
    sectionp("High-Risk Machine Registry")
    hr_table = (
        dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == "High Risk"]
        .groupby("Machine_ID")
        .agg(
            High_Risk_Readings  = ("Predictive_Maintenance_Risk", "count"),
            Avg_Anomaly_Score   = ("Anomaly_Score", "mean"),
            Avg_Anomaly_Index   = ("Anomaly_Index", "mean"),
            Avg_Risk_Score      = ("Risk_Score", "mean"),
            Avg_Lead_Time_hrs   = ("Early_Warning_Lead_Time_Hours", "mean"),
            Avg_Cascade_Index   = ("Cascading_Failure_Index", "mean"),
        )
        .reset_index()
        .sort_values("High_Risk_Readings", ascending=False)
        .head(15)
    )
    hr_table.insert(0, "Rank", range(1, len(hr_table) + 1))
    hr_table["Machine_ID"]        = "M-" + hr_table["Machine_ID"].astype(str)
    hr_table["Avg_Anomaly_Score"] = hr_table["Avg_Anomaly_Score"].round(4)
    hr_table["Avg_Anomaly_Index"] = hr_table["Avg_Anomaly_Index"].round(4)
    hr_table["Avg_Risk_Score"]    = hr_table["Avg_Risk_Score"].round(2)
    hr_table["Avg_Lead_Time_hrs"] = hr_table["Avg_Lead_Time_hrs"].round(1)
    hr_table["Avg_Cascade_Index"] = hr_table["Avg_Cascade_Index"].round(2)
    hr_table = hr_table.rename(columns={
        "Machine_ID":         "Machine",
        "High_Risk_Readings": "High Risk Readings",
        "Avg_Anomaly_Score":  "Avg Anomaly Score",
        "Avg_Anomaly_Index":  "Avg Anomaly Index",
        "Avg_Risk_Score":     "Avg Risk Score",
        "Avg_Lead_Time_hrs":  "Lead Time (hrs)",
        "Avg_Cascade_Index":  "Cascade Index",
    })
    st.dataframe(hr_table, use_container_width=True, hide_index=True)

    # ── MAINTENANCE ACTION + TOP MACHINES ─────────────────────
    #col_c, col_d = st.columns(2)
    
    #with col_c:
    sectionp("Maintenance Action Breakdown")
    act_counts   = dashboard_df["Maintenance_Action"].value_counts()
    top_action   = act_counts.idxmax()
    top_act_pct  = round(act_counts.max() / len(dashboard_df) * 100, 1)
    imm_count    = act_counts.get("Immediate Maintenance Required", 0)
    sched_count  = act_counts.get("Schedule Inspection", 0)
    norm_count   = act_counts.get("Normal Monitoring", 0)

    c_ca, c_pa = st.columns([7, 3])
    with c_ca:
        fig, ax = styled_fig(figsize=(7, 5.6))
        colors  = [ACTION_COLOR.get(a, ACCENT2) for a in act_counts.index]
        ax.barh(act_counts.index, act_counts.values,
                color=colors, height=0.45, zorder=2)
        ax.set_xlabel("Number of Readings")
        ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title("Maintenance Actions Required", pad=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_pa:
        st.markdown(panel_html("Breakdown", [
            ("Immediate Action",   f"{imm_count:,}",   DANGER),
            ("Schedule Inspect",  f"{sched_count:,}",  WARN),
            ("Normal Monitoring", f"{norm_count:,}",   ACCENT),
            ("Top Category",      top_action[:18],     TEXT_PRI),
            ("Top Share",         f"{top_act_pct}%",   ACCENT2),
        ], border_color=WARN), unsafe_allow_html=True)

    finding_text(
        f"<b>{imm_count:,}</b> readings require <b>Immediate Maintenance</b> — "
        f"these are High Risk machines needing urgent intervention. "
        f"<b>{sched_count:,}</b> readings warrant <b>Schedule Inspection</b> "
        f"within 48 hours. The majority (<b>{norm_count:,}</b>) are under "
        f"Normal Monitoring with no immediate maintenance action is required."
    )

    #with col_d:

    sectionp("Top 10 Most Anomalous Machines")
    top10 = (
        dashboard_df.groupby("Machine_ID")["Anomaly_Score"]
        .mean().sort_values(ascending=True).head(10)
    )
    most_anom_m  = f"M-{top10.index[0]}"
    most_anom_v  = round(top10.iloc[0], 4)
    least_anom_v = round(top10.iloc[-1], 4)

    c_cd, c_pd = st.columns([7, 3])
    with c_cd:
        fig, ax = styled_fig(figsize=(7, 4.4))
        bar_colors = [
            DANGER if v < 0.07 else WARN if v < 0.08 else ACCENT2
            for v in top10.values
        ]
        ax.bar([f"M-{m}" for m in top10.index], top10.values,
                color=bar_colors, width=0.55, zorder=2)
        ax.set_title("Top 10 — Lowest Anomaly Score (Most Anomalous)", pad=10)
        ax.set_ylabel("Mean Anomaly Score")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_pd:
        st.markdown(panel_html("Breakdown", [
            ("Most Anomalous",    most_anom_m,        DANGER),
            ("Its Avg Score",     str(most_anom_v),   DANGER),
            ("10th Rank Score",   str(least_anom_v),  WARN),
            ("Score Convention",  "Lower = More Anomalous", TEXT_SEC),
            ("Machines Ranked",   "Top 10 of 50",     TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    finding_text(
        f"<b>{most_anom_m}</b> is the most anomalous machine in the fleet "
        f"with a mean Anomaly Score of <b>{most_anom_v}</b> — the lowest score "
        f"signals the most deviation from normal behaviour. "
        f"Recall: in Isolation Forest, <b>lower Anomaly Score = more abnormal</b>. "
        f"These 10 machines should be prioritized for maintenance review and inspection consideration."
    )

    # ── OPERATION MODE ────────────────────────────────────────
    sectionp("Operation Mode Risk Analysis")
    op_risk = (
        dashboard_df.groupby("Operation_Mode")["Anomaly_Score"]
        .mean().sort_values(ascending=True)
    )
    riskiest_mode  = op_risk.index[0]
    riskiest_score = round(op_risk.iloc[0], 4)
    safest_mode    = op_risk.index[-1]
    safest_score   = round(op_risk.iloc[-1], 4)

    c_op, c_opp = st.columns([7, 3])
    with c_op:
        fig, ax = styled_fig(figsize=(7, 4.68))
        bar_c   = [ACCENT2, WARN, DANGER]
        bars    = ax.barh(op_risk.index, op_risk.values,
                          color=bar_c[:len(op_risk)], height=0.4, zorder=2)
        for bar in bars:
            ax.text(bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}", va="center", fontsize=9, color=TEXT_PRI)
        ax.set_title("Mean Anomaly Score by Operation Mode  (Lower = More Anomalous)", pad=10)
        ax.set_xlabel("Mean Anomaly Score")
        ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_opp:
        st.markdown(panel_html("Breakdown", [
            ("Riskiest Mode",   riskiest_mode,         DANGER),
            ("Its Avg Score",   str(riskiest_score),   DANGER),
            ("Safest Mode",     safest_mode,           ACCENT),
            ("Its Avg Score",   str(safest_score),     ACCENT),
            ("Modes Compared",  str(len(op_risk)),     TEXT_SEC),
            ("Convention",      "Lower = More Anomalous", TEXT_SEC),
        ], border_color=ACCENT2), unsafe_allow_html=True)

    finding_text(
        f"<b>{riskiest_mode}</b> mode has the lowest average Anomaly Score "
        f"(<b>{riskiest_score}</b>), making it the riskiest operating condition. "
        f"<b>{safest_mode}</b> mode is the safest (<b>{safest_score}</b>). "
        f"This suggests machine anomalies are most likely to occur during "
        f"{riskiest_mode} operations — maintenance windows should be planned accordingly."
    )

    # ── KEY FINDING: FLEET OVERVIEW ──────────────────────────
    detailed_key_findings(
        "Predictive Maintenance Overview",
        "<b>" + str(kf_hr_count) + " High Risk readings</b> ("
        + str(kf_hr_pct) + "% of 100,000 total readings) require immediate action "
        "across the fleet of 50 machines. "
        "The most anomalous machines are <b>" + ", ".join(kf_top3_ids) + "</b> — "
        "these carry the lowest Anomaly Scores and must be prioritised for inspection. "
        "<b>" + kf_riskiest_mode + "</b> is the riskiest operating mode "
        "(avg score " + str(kf_riskiest_score) + ") — "
        "maintenance scheduling should focus on this operational window."
    )
# ════════════════════════════════════════════════════════════
with tab2:

    tab_header(2, machine_id=selected_machine)   # shows M-{n} in title
    
    section("Anomaly Score Trend", tab_num=2)
    

    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
                padding:0.85rem 1.1rem;margin-bottom:1rem;">
      <span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;
                   color:{TEXT_SEC};font-weight:600;">Analysing</span>
      <span style="font-size:1.1rem;font-weight:700;color:{TEXT_PRI};margin-left:8px;">
        Machine M-{selected_machine}
      </span>
      <span style="margin-left:12px;">{badge(latest_risk, risk_col)}</span>
    </div>
    """, unsafe_allow_html=True)

    
    # Machine KPIs — includes Anomaly Score + Risk Label dual card
    mk1, mk2, mk3, mk4 = st.columns(4)

    m_avg_score    = round(machine_data["Anomaly_Score"].mean(), 4)
    m_risk_label   = machine_data["Predictive_Maintenance_Risk"].value_counts().idxmax()
    m_anomaly_rate = round((machine_data["Anomaly_Label"] == -1).mean() * 100, 2)
    m_avg_lead     = round(machine_data["Early_Warning_Lead_Time_Hours"].mean(), 1)
    m_avg_risk     = round(machine_data["Risk_Score"].mean(), 2)
    m_avg_cascade  = round(machine_data["Cascading_Failure_Index"].mean(), 2)

    m_score_color  = ("green" if m_avg_score > 0.08 else
                      "amber" if m_avg_score > 0.03 else "red")
    m_risk_color   = ("red" if m_risk_label == "High Risk" else
                      "amber" if m_risk_label == "Medium Risk" else "green")

    mk1.markdown(dual_kpi_card(
        "Anomaly Score", m_avg_score, m_score_color,
        "Risk Label", m_risk_label, m_risk_color,
        sub="Higher score = more normal"
    ), unsafe_allow_html=True)

    mk2.markdown(kpi_card("Anomaly Rate",
                           f"{m_anomaly_rate}%",
                           "red" if m_anomaly_rate > 3 else "amber",
                           "% readings<br> flagged <br>anomalous"), unsafe_allow_html=True)

    mk3.markdown(kpi_card("Avg Lead Time", f"{m_avg_lead} hrs", "blue",
                           "Hours before <br>predicted <br>failure"), unsafe_allow_html=True)

    mk4.markdown(kpi_card("Cascade Failure Index", m_avg_cascade, "amber",
                           "Risk <br>of failure<br> propagation"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    # ── ANOMALY SCORE TREND ───────────────────────────────────
    sectionp(f"Anomaly Score Trend — Machine M-{selected_machine}")
    anom_points    = machine_data[machine_data["Anomaly_Label"] == -1]
    m_score_min    = round(machine_data["Anomaly_Score"].min(), 4)
    m_score_max    = round(machine_data["Anomaly_Score"].max(), 4)
    m_anom_count   = len(anom_points)
    m_score_below0 = (machine_data["Anomaly_Score"] < 0).sum()

    c_tr, c_trp = st.columns([7, 3])
    with c_tr:
        fig, ax = styled_fig(figsize=(7, 4.2))
        ax.fill_between(machine_data["Datetime"], machine_data["Anomaly_Score"],
                        alpha=0.12, color=ACCENT2)
        ax.plot(machine_data["Datetime"], machine_data["Anomaly_Score"],
                color=ACCENT2, alpha=0.7, linewidth=1.3, label="Anomaly Score")
        ax.plot(machine_data["Datetime"], machine_data["Rolling_Anomaly_Index"],
                color=WARN, linewidth=2.2, label="Rolling Anomaly Trend")
        ax.scatter(anom_points["Datetime"], anom_points["Anomaly_Score"],
                   color=DANGER, s=22, zorder=5, label=f"Anomaly ({m_anom_count} pts)")
        ax.axhline(0, color=DANGER, linewidth=0.9, linestyle="--", alpha=0.5,
                   label="Decision Boundary (0)")
        ax.set_title(
            f"M-{selected_machine} — Score < 0 = Anomaly  |  Score > 0 = Normal",
            pad=10)
        ax.set_ylabel("Anomaly Score")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_trp:
        st.markdown(panel_html("Breakdown", [
            ("Anomaly Events",   str(m_anom_count),       DANGER),
            ("Anomaly Rate",     f"{m_anomaly_rate}%",    DANGER),
            ("Min Score",        str(m_score_min),        WARN),
            ("Max Score",        str(m_score_max),        ACCENT),
            ("Readings Below 0", str(m_score_below0),     WARN),
            ("Decision Boundary","Score = 0",             TEXT_SEC),
            ("Convention",       "Lower = More Abnormal", TEXT_SEC),
        ], border_color=ACCENT2), unsafe_allow_html=True)

    finding_text(
        f"Machine M-{selected_machine} recorded <b>{m_anom_count} anomaly events</b> "
        f"(anomaly rate: <b>{m_anomaly_rate}%</b>). "
        f"The lowest score was <b>{m_score_min}</b> — the most anomalous reading for this machine. "
        f"<b>{m_score_below0}</b> readings crossed below the decision boundary (0), "
        f"indicating model-identified anomalous behaviour during those periods."
    )

    # ── SENSOR DEVIATIONS ─────────────────────────────────────
    sectionp("Sensor Deviation Profile")
    max_temp_dev  = round(machine_data["Temperature_Deviation"].abs().max(), 3)
    max_vib_dev   = round(machine_data["Vibration_Deviation"].abs().max(), 3)
    max_pow_dev   = round(machine_data["Power_Deviation"].abs().max(), 3)
    dominant_dev  = max(
        ("Temperature", max_temp_dev),
        ("Vibration",   max_vib_dev),
        ("Power",       max_pow_dev),
        key=lambda x: x[1]
    )

    c_sd, c_sdp = st.columns([7, 3])
    with c_sd:
        fig, ax = styled_fig(figsize=(7, 4.2))
        ax.fill_between(machine_data["Datetime"],
                        machine_data["Temperature_Deviation"], alpha=0.2, color=DANGER)
        ax.fill_between(machine_data["Datetime"],
                        machine_data["Vibration_Deviation"],   alpha=0.2, color=WARN)
        ax.fill_between(machine_data["Datetime"],
                        machine_data["Power_Deviation"],       alpha=0.2, color=ACCENT2)
        ax.plot(machine_data["Datetime"], machine_data["Temperature_Deviation"],
                color=DANGER,  linewidth=1.4, label="Temperature Δ (°C)")
        ax.plot(machine_data["Datetime"], machine_data["Vibration_Deviation"],
                color=WARN,    linewidth=1.4, label="Vibration Δ (Hz)")
        ax.plot(machine_data["Datetime"], machine_data["Power_Deviation"],
                color=ACCENT2, linewidth=1.4, label="Power Δ (kW)")
        ax.axhline(0, color=TEXT_SEC, linewidth=0.8, linestyle="--", alpha=0.7, zorder=10)
        ax.set_title("Sensor Deviations from Rolling Baseline", pad=10)
        ax.set_ylabel("Deviation from Mean")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_sdp:
        st.markdown(panel_html("Breakdown", [
            ("Max Temp Deviation",  f"±{max_temp_dev} °C",  DANGER),
            ("Max Vibr Deviation",  f"±{max_vib_dev} Hz",   WARN),
            ("Max Power Deviation", f"±{max_pow_dev} kW",   ACCENT2),
            ("Highest Deviation",   dominant_dev[0],         TEXT_PRI),
            ("Its Peak Value",      f"±{dominant_dev[1]}",  TEXT_PRI),
            ("Baseline",            "7-pt rolling mean",    TEXT_SEC),
        ], border_color=WARN), unsafe_allow_html=True)

    finding_text(
        f"<b>{dominant_dev[0]}</b> shows the largest deviation from baseline "
        f"(peak: <b>±{dominant_dev[1]}</b>), indicating this sensor is the primary "
        f"driver of anomaly events on Machine M-{selected_machine}. "
        f"Deviations crossing the zero baseline (dashed line) signal sensor readings "
        f"significantly different from normal operating patterns."
    )

    # ── LIVE SENSOR SUMMARY ───────────────────────────────────
    sectionp("Live Sensor Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(kpi_card("Avg Temperature",
                          f"{machine_data['Temperature_C'].mean():.1f} °C", "amber"),
                unsafe_allow_html=True)
    s2.markdown(kpi_card("Avg Vibration",
                          f"{machine_data['Vibration_Hz'].mean():.1f} Hz", "amber"),
                unsafe_allow_html=True)
    s3.markdown(kpi_card("Avg Power",
                          f"{machine_data['Power_Consumption_kW'].mean():.1f} kW", "blue"),
                unsafe_allow_html=True)
    s4.markdown(kpi_card("Avg Error Rate",
                          f"{machine_data['Error_Rate_%'].mean():.2f}%",
                          "red" if machine_data["Error_Rate_%"].mean() > 5 else "amber"),
                unsafe_allow_html=True)

    # ── CORRELATION + SEVERITY PIE ────────────────────────────
    sectionp("Feature Correlation & Anomaly Severity")
    corr_cols     = ["Temperature_C","Vibration_Hz","Power_Consumption_kW",
                     "Error_Rate_%","Risk_Score","Anomaly_Score"]
    corr          = machine_data[corr_cols].corr()
    top_corr_feat = corr["Anomaly_Score"].drop("Anomaly_Score").abs().idxmax()
    top_corr_val  = round(corr["Anomaly_Score"].drop("Anomaly_Score").abs().max(), 3)
    # Get the ORIGINAL signed correlation 
    #top_corr_val1 = round(corr_with_anomaly[top_corr_feat],3)


    sev           = machine_data["Anomaly_Severity"].value_counts()
    top_sev       = sev.idxmax()
    top_sev_pct   = round(sev.max() / sev.sum() * 100, 1)

    #    col_l, col_r = st.columns(2)
    #with col_l:
    c_hm, c_hmp = st.columns([7, 3])
    with c_hm:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                    linewidths=0.5, linecolor=DARK_BG,
                    annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.75})
        ax.set_title("Sensor & Risk Correlation", pad=10, color=TEXT_PRI)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_hmp:
        st.markdown(panel_html("Breakdown", [
            ("Strongest Correlator", top_corr_feat,         DANGER),
            ("Correlation Value",    str(top_corr_val),     DANGER),
            ("Features Compared",   f"{len(corr_cols)-1}",  TEXT_SEC),
            ("Red cells",           "Negative correlation", TEXT_SEC),
            ("Green cells",         "Positive correlation", TEXT_SEC),
        ], border_color=ACCENT2), unsafe_allow_html=True)
    finding_text(
        f"<b>{top_corr_feat}</b> has the strongest correlation with Anomaly Score "
        f"(|r| = <b>{top_corr_val}</b>) — this is the strongest statistical signal associated with Anomaly Score "
        f"for Machine M-{selected_machine}. Monitor this feature closely for unusual deviations."
    )

    #with col_r:
    sectionp("Anomaly Severity Distribution")
    c_pie, c_piep = st.columns([7, 3])
    with c_pie:
        fig, ax = plt.subplots(figsize=(7, 1.74))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        colors_pie = [DANGER, WARN, ACCENT]
        wedges, texts, autotexts = ax.pie(
            sev.values, labels=sev.index,
            colors=colors_pie[:len(sev)],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": DARK_BG},
            textprops={"color": TEXT_PRI, "fontsize": 9}
        )
        for at in autotexts:
            at.set_color(DARK_BG)
            at.set_fontweight("bold")
        ax.set_title("Anomaly Severity Distribution", color=TEXT_PRI, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_piep:
        sev_rows = [("Dominant Severity", top_sev, TEXT_PRI),
                    ("Its Share", f"{top_sev_pct}%", ACCENT2)]
        for s, cnt in sev.items():
            c = DANGER if s == "Critical" else WARN if s == "Medium" else ACCENT
            sev_rows.append((f"{s} count", f"{cnt:,}", c))
        st.markdown(panel_html("Breakdown", sev_rows, border_color=WARN),
                    unsafe_allow_html=True)
    crit_count = sev.get("Critical", 0)
    crit_msg = (
        "No Critical events — machine is in early-warning stage."
        if crit_count == 0
        else f"Critical events present ({crit_count:,}) — immediate inspection needed."
    )
    finding_text(
        f"<b>{top_sev}</b> severity dominates with <b>{top_sev_pct}%</b> of "
        f"anomaly events on Machine M-{selected_machine}. {crit_msg}"
    )

    # ── KEY FINDING: MACHINE ANOMALY DASHBOARD ───────────────
    detailed_key_findings(
        "Machine Anomaly Dashboard",
        "Across the full fleet of 50 machines, the overall anomaly detection rate is "
        "<b>3.0%</b> of all 100,000 readings. "
        "<b>" + kf_top_corr + "</b> is the strongest sensor correlated with anomalous behaviour "
        "(|r| = <b>" + str(kf_top_corr_val) + "</b>) — this sensor constitutes the primary "
        "early-warning indicator for predictive maintenance across the entire manufacturing fleet. "
        "Fleet-wide average lead time is <b>" + str(kf_fleet_lt_mean) + " hrs</b> — "
        "sufficient time for planned intervention before machine failure."
    )
# ════════════════════════════════════════════════════════════


with tab3:

    sectionp("Active Alert Summary")
    a1, a2, a3 = st.columns(3)
    immediate  = (dashboard_df["Maintenance_Action"]
                    == "Immediate Maintenance Required").sum()
    scheduled  = (dashboard_df["Maintenance_Action"]
                    == "Schedule Inspection").sum()
    monitoring = (dashboard_df["Maintenance_Action"]
                    == "Normal Monitoring").sum()

    a1.markdown(kpi_card("Immediate Action Required", f"{immediate:,}", "red",
                            "Needs attention now"), unsafe_allow_html=True)
    a2.markdown(kpi_card("Schedule Inspection", f"{scheduled:,}", "amber",
                            "Plan within 48 hrs"), unsafe_allow_html=True)
    a3.markdown(kpi_card("Normal Monitoring", f"{monitoring:,}", "green",
                            "No action needed"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    
    
    
     
    
    
    
    # ── ALERT TABLE ───────────────────────────────────────────
    sectionp("High-Priority Alert Log")
    alerts = (
        dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == "High Risk"]
        [[
            "Machine_ID", "Datetime", "Anomaly_Score", "Anomaly_Index",
            "Risk_Score", "Anomaly_Severity", "Cascade_Risk_Level",
            "Early_Warning_Lead_Time_Hours", "Maintenance_Action"
        ]]
        .sort_values("Anomaly_Score", ascending=True)   # lowest score = most anomalous
        .head(30)
        .copy()
    )
    alerts["Machine_ID"]    = "M-" + alerts["Machine_ID"].astype(str)
    alerts["Anomaly_Score"] = alerts["Anomaly_Score"].round(4)
    alerts["Anomaly_Index"] = alerts["Anomaly_Index"].round(4)
    alerts["Risk_Score"]    = alerts["Risk_Score"].round(2)
    alerts["Early_Warning_Lead_Time_Hours"] = \
        alerts["Early_Warning_Lead_Time_Hours"].round(1)
    alerts["Datetime"]      = alerts["Datetime"].dt.strftime("%d-%b %H:%M")
    alerts = alerts.rename(columns={
        "Machine_ID":                    "Machine",
        "Datetime":                      "Timestamp",
        "Anomaly_Score":                 "Anomaly Score",
        "Anomaly_Index":                 "Anomaly Index",
        "Risk_Score":                    "Risk Score",
        "Anomaly_Severity":              "Severity",
        "Cascade_Risk_Level":            "Cascade Risk",
        "Early_Warning_Lead_Time_Hours": "Lead Time (hrs)",
        "Maintenance_Action":            "Action Required",
    })
    st.dataframe(alerts, use_container_width=True, hide_index=True)
    

    sectionp("Cascade Failure Risk Analysis")

        # ── CASCADE RISK LEVEL — existing bar chart ───────────────
    #col_e, col_f = st.columns(2)

    #with col_e:
    casc        = dashboard_df["Cascade_Risk_Level"].value_counts()
    hi_casc     = casc.get("High Cascade Risk",   0)
    med_casc    = casc.get("Medium Cascade Risk",  0)
    low_casc    = casc.get("Low Cascade Risk",     0)
    hi_casc_pct = round(hi_casc / len(dashboard_df) * 100, 1)

    c_ce, c_cep = st.columns([6, 4])
    with c_ce:
        casc_colors = {
            "High Cascade Risk":   DANGER,
            "Medium Cascade Risk": WARN,
            "Low Cascade Risk":    ACCENT
        }
        fig, ax = styled_fig(figsize=(5, 4.5))
        bar_c   = [casc_colors.get(c, ACCENT2) for c in casc.index]
        ax.bar(casc.index, casc.values, color=bar_c, width=0.45, zorder=2)
        ax.set_title("Cascade Risk Level Distribution", pad=10)
        ax.set_ylabel("Readings")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_cep:
        st.markdown(panel_html("Breakdown", [
            ("High Cascade Risk",   f"{hi_casc:,}",    DANGER),
            ("Medium Cascade Risk", f"{med_casc:,}",   WARN),
            ("Low Cascade Risk",    f"{low_casc:,}",   ACCENT),
            ("High Risk Share",     f"{hi_casc_pct}%", DANGER),
            ("What it means",
            "Failure propagates to other machines",   TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    
    finding_text( 
        f"<b>{hi_casc:,}</b> readings ({hi_casc_pct}%) are classified as " 
        f"High Cascade Risk, indicating elevated potential for failure " 
        f"propagation across the production line. Prioritise these conditions " 
        f"for investigation alongside standard High Risk alerts." 
        ) 
    #with col_f:
    casc_machine = (
        dashboard_df[
            dashboard_df["Cascade_Risk_Level"] == "High Cascade Risk"
        ]
        .groupby("Machine_ID")["Cascading_Failure_Index"]
        .mean().sort_values(ascending=False).head(10)
    )
    top_casc_m = f"M-{casc_machine.index[0]}" if not casc_machine.empty else "N/A"
    top_casc_v = round(casc_machine.iloc[0], 2) if not casc_machine.empty else 0

    c_cf, c_cfp = st.columns([6, 4])
    with c_cf:
        fig, ax = styled_fig(figsize=(5, 4.5))
        ax.barh([f"M-{m}" for m in casc_machine.index],
                casc_machine.values, color=DANGER, height=0.45, zorder=2)
        ax.set_title("Top 10 Machines by Cascade Failure Index", pad=10)
        ax.set_xlabel("Mean Cascading Failure Index")
        ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_cfp:
        st.markdown(panel_html("Breakdown", [
            ("Highest Risk Machine", top_casc_m,      DANGER),
            ("Its Cascade Index",    str(top_casc_v), DANGER),
            ("Machines Shown",       "Top 10",        TEXT_SEC),
            ("Index formula",
            "Risk Score × Normalized Anomaly / 100", TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    
    finding_text( 
        f"<b>{top_casc_m}</b> has the highest Cascading Failure Index " 
        f"(<b>{top_casc_v}</b>), indicating the strongest potential for " 
        f"failure propagation across the production line. Prioritise this " 
        f"machine for investigation." 
        ) 

        # ── NEW SECTION: CASCADE FAILURE PATTERN ─────────────────
        # This is the new Temporal Risk Escalation Analysis section
    section("Cascade Failure Pattern Detection")

    # ── KPI CARDS: 3 pattern counts ──────────────────────────
    cfp_counts  = dashboard_df["Cascade_Failure_Pattern"].value_counts()
    active_count   = int(cfp_counts.get("Active Cascade Risk", 0))
    emerging_count = int(cfp_counts.get("Emerging Cascade",    0))
    no_pat_count   = int(cfp_counts.get("No Pattern",          0))
    active_pct     = round(active_count   / len(dashboard_df) * 100, 1)
    emerging_pct   = round(emerging_count / len(dashboard_df) * 100, 1)

    st.markdown(
        "<div style='background:" + CARD_BG + ";border:1px solid " + BORDER + ";"
        "border-left:3px solid #7F77DD;border-radius:8px;"
        "padding:0.75rem 1rem;margin-bottom:0.75rem;'>"
        "<div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.10em;color:#7F77DD;margin-bottom:0.5rem;'>"
        "How This Works — 3 Conditions Must Be Met"
        "</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;"
        "font-size:0.78rem;color:" + TEXT_SEC + ";line-height:1.65;'>"
        "<div><b style='color:" + DANGER + ";'>Condition 1</b><br>"
        "Multiple anomaly events in last 7 readings per machine "
        "(persistent fault, not a single spike)</div>"
        "<div><b style='color:" + WARN + ";'>Condition 2</b><br>"
        "Rolling Anomaly Index is rising AND Risk Escalation Rate &gt; 0.05 "
        "(anomaly is worsening, not recovering)</div>"
        "<div><b style='color:" + ACCENT2 + ";'>Condition 3</b><br>"
        "Cascading Failure Index is in top 10% of fleet "
        "(high propagation risk to downstream machines)</div>"
        "</div>"
        "<div style='margin-top:0.6rem;font-size:0.75rem;color:" + TEXT_SEC + ";'>"
        "<b style='color:" + DANGER + ";'>Active Cascade Risk</b> = all 3 conditions met &nbsp;|&nbsp;"
        "<b style='color:" + WARN + ";'>Emerging Cascade</b> = any 2 of 3 met &nbsp;|&nbsp;"
        "<b style='color:" + ACCENT + ";'>No Pattern</b> = 0 or 1 condition met"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )

    cfp1, cfp2, cfp3 = st.columns(3)
    cfp1.markdown(kpi_card(
        "🔴 Active Cascade Risk",
        f"{active_count:,}",
        "red",
        f"{active_pct}% — All 3 conditions met.<br> Immediate action."
    ), unsafe_allow_html=True)

    cfp2.markdown(kpi_card(
        "🟡 Emerging Cascade",
        f"{emerging_count:,}",
        "amber",
        f"{emerging_pct}% — 2 of 3 conditions met. Schedule inspection."
    ), unsafe_allow_html=True)

    cfp3.markdown(kpi_card(
        "🟢 No Pattern",
        f"{no_pat_count:,}",
        "green",
        "Normal behaviour. <br>Continue monitoring."
    ), unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:0.75rem;'></div>",
        unsafe_allow_html=True
    )

    # ── CHART + PANEL: Pattern distribution bar ───────────────
    sectionp("Cascade Pattern Distribution")

    cfp_order  = ["Active Cascade Risk", "Emerging Cascade", "No Pattern"]
    cfp_vals   = [cfp_counts.get(p, 0) for p in cfp_order]
    cfp_colors = [DANGER, WARN, ACCENT]

    c_cfpbar, c_cfppanel = st.columns([6, 4])
    with c_cfpbar:
        fig, ax = styled_fig(figsize=(6, 4.5))
        bars = ax.bar(cfp_order, cfp_vals, color=cfp_colors, width=0.5, zorder=2)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{bar.get_height():,}",
                ha="center", va="bottom", fontsize=9, color=TEXT_PRI
            )
        ax.set_title("Cascade Failure Pattern Distribution", pad=10)
        ax.set_ylabel("Number of Readings")
        ax.set_xticklabels(cfp_order, rotation=10, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_cfppanel:
        st.markdown(panel_html("Breakdown", [
            ("Active Cascade Risk",
            f"{active_count:,} ({active_pct}%)",  DANGER),
            ("Emerging Cascade",
            f"{emerging_count:,} ({emerging_pct}%)", WARN),
            ("No Pattern",
            f"{no_pat_count:,}",                  ACCENT),
            ("Active = All 3 conditions",
            "Highest urgency",                    DANGER),
            ("Emerging = 2 of 3 conditions",
            "Early warning stage",                WARN),
            ("Threshold used",
            "Cascade Index > 90th percentile",    TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    finding_text(
        f"<b>{active_count:,} readings</b> ({active_pct}%) show "
        f"<b>Active Cascade Risk</b> — all 3 temporal conditions are simultaneously "
        f"met, indicating an imminent chain failure pattern. "
        f"<b>{emerging_count:,} readings</b> ({emerging_pct}%) are in the "
        f"<b>Emerging Cascade</b> stage — early intervention here prevents "
        f"escalation to Active status."
    )

    # ── CHART + PANEL: Top machines by Active Cascade ─────────
    sectionp("Top Machines — Active Cascade Risk")

    top_active_df = (
        dashboard_df[
            dashboard_df["Cascade_Failure_Pattern"] == "Active Cascade Risk"
        ]
        .groupby("Machine_ID")
        .agg(
            Active_Count      = ("Cascade_Failure_Pattern", "count"),
            Avg_Cascade_Index = ("Cascading_Failure_Index", "mean"),
            Avg_Risk_Score    = ("Risk_Score",              "mean"),
            Avg_Lead_Time     = ("Early_Warning_Lead_Time_Hours", "mean"),
            Anomaly_Rate_Pct  = ("Anomaly_Label",
                                lambda x: round((x == -1).mean() * 100, 1)),
        )
        .reset_index()
        .sort_values("Active_Count", ascending=False)
        .head(10)
    )

    if top_active_df.empty:
        st.info("No Active Cascade Risk readings under current filters.")
    else:
        top_m_name = f"M-{top_active_df.iloc[0]['Machine_ID']}"
        top_m_count = int(top_active_df.iloc[0]["Active_Count"])
        top_m_anom  = round(top_active_df.iloc[0]["Anomaly_Rate_Pct"], 1)
        top_m_lt    = round(top_active_df.iloc[0]["Avg_Lead_Time"], 1)

        c_topbar, c_toppanel = st.columns([6, 4])
        with c_topbar:
            fig, ax = styled_fig(figsize=(6, 4.5))
            colors_top = [
                DANGER if i == 0 else
                "#C44040" if i < 3 else WARN
                for i in range(len(top_active_df))
            ]
            ax.barh(
                [f"M-{m}" for m in top_active_df["Machine_ID"]],
                top_active_df["Active_Count"],
                color=colors_top, height=0.55, zorder=2
            )
            ax.set_xlabel("Active Cascade Risk Readings")
            ax.set_title(
                "Top 10 Machines — Active Cascade Risk Readings",
                pad=10
            )
            ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
            ax.grid(axis="y", visible=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_toppanel:
            st.markdown(panel_html("Breakdown", [
                ("Top Machine",
                top_m_name,             DANGER),
                ("Its Active Count",
                str(top_m_count),       DANGER),
                ("Its Anomaly Rate",
                f"{top_m_anom}%",       DANGER),
                ("Its Avg Lead Time",
                f"{top_m_lt} hrs",      ACCENT2),
                ("What to do",
                "Inspect immediately",  DANGER),
                ("Condition",
                "All 3 cascade conditions met", TEXT_SEC),
            ], border_color=DANGER), unsafe_allow_html=True)

        finding_text(
            f"<b>{top_m_name}</b> is the highest-priority machine with "
            f"<b>{top_m_count}</b> Active Cascade Risk readings and an anomaly rate "
            f"of <b>{top_m_anom}%</b>. Available lead time: <b>{top_m_lt} hrs</b>. "
            f"This machine satisfies all 3 cascade conditions simultaneously — "
            f"persistent anomalies, worsening trend, and high propagation risk. "
            f"A failure here will cascade to downstream equipment."
        )

        # ── DETAIL TABLE ──────────────────────────────────────
        subsec_html = (
            "<div style='display:flex;align-items:center;gap:8px;"
            "margin:1rem 0 0.5rem 0;'>"
            "<div style='width:6px;height:6px;border-radius:50%;"
            "background:" + DANGER + ";flex-shrink:0;'></div>"
            "<div style='font-size:0.72rem;font-weight:600;"
            "letter-spacing:0.08em;text-transform:uppercase;"
            "color:" + TEXT_SEC + ";'>Active Cascade Risk — Machine Detail Table</div>"
            "</div>"
        )
        st.markdown(subsec_html, unsafe_allow_html=True)

        display_df = top_active_df.copy()
        display_df["Machine_ID"]      = "M-" + display_df["Machine_ID"].astype(str)
        display_df["Avg_Cascade_Index"] = display_df["Avg_Cascade_Index"].round(2)
        display_df["Avg_Risk_Score"]    = display_df["Avg_Risk_Score"].round(2)
        display_df["Avg_Lead_Time"]     = display_df["Avg_Lead_Time"].round(1)
        display_df = display_df.rename(columns={
            "Machine_ID":       "Machine",
            "Active_Count":     "Active Cascade Readings",
            "Avg_Cascade_Index":"Avg Cascade Index",
            "Avg_Risk_Score":   "Avg Risk Score",
            "Avg_Lead_Time":    "Avg Lead Time (hrs)",
            "Anomaly_Rate_Pct": "Anomaly Rate (%)",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── KEY FINDING: MAINTENANCE ALERT PANEL ─────────────────
    detailed_key_findings(
        "Maintenance Alert Panel",
        "Of the 100,000 sensor readings across all 50 machines, "
        "<b>3,006 require Immediate Maintenance</b> (3.0% of total fleet). "
        "<b>" + kf_top_casc_m + "</b> carries the highest Cascading Failure Index "
        "(<b>" + str(kf_top_casc_v) + "</b>) — a failure here poses the greatest "
        "systemic risk of propagating to downstream production equipment. "
        "<b>3,000 readings</b> (3.0%) are classified as High Cascade Risk, "
        "indicating that fault propagation is a significant operational concern "
        "requiring fleet-wide inspection coordination rather than isolated machine responses."
    )
# ════════════════════════════════════════════════════════════
with tab4:

    
    tab_header(4)
    section("Risk Escalation Timeline", tab_num=4)



    # ── RISK ESCALATION TIMELINE ──────────────────────────────
    sectionp(f"Risk Escalation Timeline for one selected machine — Machine M-{selected_machine}")
    esc_max    = round(machine_data["Risk_Escalation_Rate"].max(), 4)
    esc_mean   = round(machine_data["Risk_Escalation_Rate"].mean(), 4)
    escalating = (machine_data["Risk_Trend_Status"] == "Escalating").sum()
    improving  = (machine_data["Risk_Trend_Status"] == "Improving").sum()
    stable     = (machine_data["Risk_Trend_Status"] == "Stable").sum()

    c_re, c_rep = st.columns([7, 3])
    with c_re:
        fig, ax = styled_fig(figsize=(7, 4.2))
        ax.fill_between(machine_data["Datetime"],
                        machine_data["Risk_Escalation_Rate"], alpha=0.18, color=DANGER)
        ax.plot(machine_data["Datetime"], machine_data["Risk_Escalation_Rate"],
                color=DANGER, linewidth=1.6, label="Risk Escalation Rate")
        ax.plot(machine_data["Datetime"], machine_data["Anomaly_Score"],
                color=ACCENT2, linewidth=1.4, alpha=0.7, label="Anomaly Score")
        ax.plot(machine_data["Datetime"], machine_data["Rolling_Anomaly_Index"],
                color=WARN, linewidth=2.2, label="Rolling Anomaly Trend")
        ax.axhline(0, color=WARN, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Risk Escalation vs Anomaly Score — M-{selected_machine}", pad=10)
        ax.set_ylabel("Index Value")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_rep:
        st.markdown(panel_html("Breakdown", [
            ("Peak Escalation Rate", str(esc_max),          DANGER),
            ("Avg Escalation Rate",  str(esc_mean),         WARN),
            ("Escalating Periods",   str(escalating),       DANGER),
            ("Improving Periods",    str(improving),        ACCENT),
            ("Stable Periods",       str(stable),           TEXT_SEC),
            ("What escalation means","Rate of change of Anomaly Index", TEXT_SEC),
        ], border_color=DANGER), unsafe_allow_html=True)

    trend_verdict = "escalating" if escalating > improving else "improving" if improving > escalating else "stable"
    finding_text(
        f"Machine M-{selected_machine}'s risk is primarily <b>{trend_verdict}</b> — "
        f"{escalating} escalating periods vs {improving} improving periods. "
        f"Peak escalation rate was: <b>{esc_max}</b>. "
        f"Spikes in the Risk Escalation Rate (red area) coincide with anomaly events "
        f"providing supporting evidence that the model is capturing periods of "
        f"deteriorating machine health."
    )




    # ── CASCADE FAILURE PATTERN TIMELINE ─────────────────────
    section(
        f"Cascade Failure Pattern Timeline — Machine M-{selected_machine}"
    )

    # Map pattern to numeric for plotting
    pattern_map = {
        "No Pattern":          0,
        "Emerging Cascade":    1,
        "Active Cascade Risk": 2,
    }
    machine_data["Pattern_Numeric"] = machine_data[
        "Cascade_Failure_Pattern"
    ].map(pattern_map).fillna(0)

    # Count per pattern for this machine
    m_cfp_counts   = machine_data["Cascade_Failure_Pattern"].value_counts()
    m_active_count = int(m_cfp_counts.get("Active Cascade Risk", 0))
    m_emerg_count  = int(m_cfp_counts.get("Emerging Cascade",    0))
    m_no_pat_count = int(m_cfp_counts.get("No Pattern",          0))

    # Points for each pattern level
    active_pts   = machine_data[
        machine_data["Cascade_Failure_Pattern"] == "Active Cascade Risk"
    ]
    emerging_pts = machine_data[
        machine_data["Cascade_Failure_Pattern"] == "Emerging Cascade"
    ]

    c_cfpt, c_cfptp = st.columns([7, 3])
    with c_cfpt:
        fig, ax = styled_fig(figsize=(7, 5))

        # Background shading for each level
        # Level 0 = No Pattern (dark background — already is)
        # Level 1 = Emerging   (amber shading)
        # Level 2 = Active     (red shading)
        ax.fill_between(
            machine_data["Datetime"],
            machine_data["Pattern_Numeric"],
            alpha=0.25, color=ACCENT2, step="post"
        )
        ax.step(
            machine_data["Datetime"],
            machine_data["Pattern_Numeric"],
            color=ACCENT2, linewidth=1.8, where="post",
            label="Pattern Level (0=None, 1=Emerging, 2=Active)"
        )

        # Mark Emerging points
        if not emerging_pts.empty:
            ax.scatter(
                emerging_pts["Datetime"],
                emerging_pts["Pattern_Numeric"],
                color=WARN, s=35, zorder=5,
                label=f"Emerging Cascade ({m_emerg_count})"
            )
        # Mark Active points
        if not active_pts.empty:
            ax.scatter(
                active_pts["Datetime"],
                active_pts["Pattern_Numeric"],
                color=DANGER, s=55, zorder=6, marker="^",
                label=f"Active Cascade Risk ({m_active_count}) ▲"
            )

        # Y-axis labels
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(
            ["No Pattern", "Emerging Cascade", "Active Cascade Risk"],
            fontsize=9
        )
        ax.set_ylim(-0.3, 2.5)

        # Reference lines
        ax.axhline(
            1, color=WARN,   linewidth=0.8,
            linestyle="--", alpha=0.5
        )
        ax.axhline(
            2, color=DANGER, linewidth=0.8,
            linestyle="--", alpha=0.5
        )

        ax.set_title(
            f"Cascade Failure Pattern Over Time — "
            f"M-{selected_machine}  "
            f"(▲ = Active Cascade Risk)",
            pad=10
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        ax.grid(axis="y", color=BORDER, alpha=0.3, linewidth=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c_cfptp:
        m_status_color = (
            DANGER if m_active_count > 0 else
            WARN   if m_emerg_count  > 0 else
            ACCENT
        )
        m_status_text = (
            "Active Cascade Risk Detected"  if m_active_count > 0 else
            "Emerging Cascade Pattern"       if m_emerg_count  > 0 else
            "No Cascade Pattern"
        )
        st.markdown(panel_html("Breakdown", [
            ("Machine Status",
                m_status_text,           m_status_color),
            ("Active Cascade Events",
                str(m_active_count),     DANGER),
            ("Emerging Cascade Events",
                str(m_emerg_count),      WARN),
            ("No Pattern Readings",
                str(m_no_pat_count),     ACCENT),
            ("Pattern logic",
                "3 temporal conditions", TEXT_SEC),
            ("Condition 1",
                "Repeated anomalies",    TEXT_SEC),
            ("Condition 2",
                "Worsening trend",       TEXT_SEC),
            ("Condition 3",
                "High cascade index",    TEXT_SEC),
        ], border_color=m_status_color), unsafe_allow_html=True)

    # Dynamic finding text based on machine's actual pattern
    if m_active_count > 0:
        cfp_finding = (
            f"Machine M-{selected_machine} has <b>{m_active_count} Active Cascade "
            f"Risk events</b> — all 3 temporal conditions were simultaneously met "
            f"during these periods. These are the highest-priority maintenance windows "
            f"for this machine. A failure during these periods would propagate to "
            f"downstream equipment. Schedule immediate inspection."
        )
    elif m_emerg_count > 0:
        cfp_finding = (
            f"Machine M-{selected_machine} shows <b>{m_emerg_count} Emerging "
            f"Cascade events</b> — 2 of 3 cascade conditions were met during "
            f"these periods. The machine is in early-warning stage. "
            f"Schedule inspection before any Emerging events escalate to "
            f"Active Cascade Risk status."
        )
    else:
        cfp_finding = (
            f"Machine M-{selected_machine} shows <b>No Cascade Pattern</b> "
            f"under current filters — fewer than 2 cascade conditions were "
            f"simultaneously met. The machine is operating within normal "
            f"temporal risk boundaries. Continue standard monitoring."
        )

    finding_text(cfp_finding)

    # Clean up helper column (keep data clean)
    machine_data.drop(
        columns=["Pattern_Numeric"],
        inplace=True, errors="ignore"
    )



    # ── EARLY WARNING LEAD TIME TREND ─────────────────────────
    sectionp("Early Warning Lead Time Trend")
    mean_lt   = round(machine_data["Early_Warning_Lead_Time_Hours"].mean(), 1)
    min_lt    = round(machine_data["Early_Warning_Lead_Time_Hours"].min(), 1)
    critical_lt_count = (machine_data["Early_Warning_Lead_Time_Hours"] <= 20).sum()

    c_lt, c_ltp = st.columns([7, 3])
    with c_lt:
        fig, ax = styled_fig(figsize=(7, 4.2))
        ax.fill_between(machine_data["Datetime"],
                        machine_data["Early_Warning_Lead_Time_Hours"],
                        alpha=0.15, color=ACCENT2)
        ax.plot(machine_data["Datetime"],
                machine_data["Early_Warning_Lead_Time_Hours"],
                color=ACCENT2, linewidth=1.6, label="Lead Time (hrs)")
        ax.axhline(mean_lt, color=WARN, linewidth=1.2, linestyle="--",
                   label=f"Mean: {mean_lt} hrs")
        ax.axhline(20, color=DANGER, linewidth=0.9, linestyle=":",
                   alpha=0.7, label="Critical Threshold (20 hrs)")
        ax.set_title("Early Warning Lead Time — Hours Available Before Predicted Failure", pad=10)
        ax.set_ylabel("Hours")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_ltp:
        st.markdown(panel_html("Breakdown", [
            ("Avg Lead Time",         f"{mean_lt} hrs",             ACCENT2),
            ("Minimum Lead Time",     f"{min_lt} hrs",              DANGER if min_lt < 25 else WARN),
            ("Critical Readings",     f"{critical_lt_count} (≤25 hrs)", DANGER),
            ("Safe ceiling",          "100 hrs (normal machines)",  ACCENT),
            ("Critical floor",        "20 hrs (urgent action)",     DANGER),
            ("Formula",               "100 / (1 + risk_norm × 4)", TEXT_SEC),
            #("risk_norm",               "Normalised Anomaly Risk", TEXT_SEC),
        ], border_color=ACCENT2), unsafe_allow_html=True)

    lt_status = "well within safe range" if mean_lt > 80 else "in warning zone" if mean_lt > 40 else "critically low"
    finding_text(
        f"Machine M-{selected_machine} has an average lead time of <b>{mean_lt} hrs</b> — "
        f"{lt_status}. "
        f"<b>{critical_lt_count}</b> readings dropped below the 25-hour critical threshold. "
        f"The minimum recorded was <b>{min_lt} hrs</b>. "
        f"Lead time below 20 hrs triggers Immediate Maintenance action — "
        f"operations must respond within this window to prevent failure."
    )

    # ════════════════════════════════════════════════════════
    # CHANGE 3: FULL POST-MAINTENANCE BEHAVIOR COMPARISON
    # ════════════════════════════════════════════════════════
    sectionp("Post-Maintenance Behavior Comparison")

    # Split machine data into Maintenance mode vs Active/Idle
    maint_data  = machine_data[machine_data["Operation_Mode"] == "Maintenance"]
    active_data = machine_data[machine_data["Operation_Mode"] != "Maintenance"]

    if maint_data.empty:
        st.info("No Maintenance mode readings for this machine in the selected time window. "
                "Try expanding the date range or switching to a different machine.")
    else:
        # ── KPI COMPARISON BOXES ──────────────────────────────
        st.markdown(f"""
        <div style="font-size:0.78rem;color:{TEXT_SEC};margin-bottom:0.75rem;line-height:1.6;">
          Compares machine behaviour <b style="color:{DANGER};">during maintenance operations</b>
          vs <b style="color:{ACCENT2};">normal active/idle running</b>.
          A healthy machine should show improved (higher) Anomaly Score and lower
          Risk Score after maintenance.
        </div>
        """, unsafe_allow_html=True)

        pm1, pm2, pm3, pm4 = st.columns(4)

        def delta_str(after, before, higher_is_better=True):
            diff = after - before
            if higher_is_better:
                arrow = "▲" if diff > 0 else "▼"
                clr   = ACCENT if diff > 0 else DANGER
            else:
                arrow = "▼" if diff < 0 else "▲"
                clr   = ACCENT if diff < 0 else DANGER
            return f'<span style="color:{clr};font-size:0.72rem;">{arrow} {abs(diff):.4f} vs active</span>'

        # Anomaly Score
        score_maint  = maint_data["Anomaly_Score"].mean()
        score_active = active_data["Anomaly_Score"].mean() if not active_data.empty else score_maint
        score_color_m = "green" if score_maint > 0.08 else "amber" if score_maint > 0.03 else "red"

        pm1.markdown(f"""
        <div class="comparison-box">
          <div class="comparison-label">Anomaly Score</div>
          <div style="display:flex;gap:0.75rem;align-items:flex-end;margin-top:0.3rem;">
            <div>
              <div style="font-size:0.62rem;color:{DANGER};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">During Maint.</div>
              <div class="comparison-value" style="color:{score_color_m};">
                {score_maint:.4f}
              </div>
            </div>
            <div style="width:1px;background:{BORDER};height:32px;margin-bottom:4px;"></div>
            <div>
              <div style="font-size:0.62rem;color:{ACCENT2};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">Active/Idle</div>
              <div class="comparison-value" style="color:{TEXT_SEC};font-size:1rem;">
                {score_active:.4f}
              </div>
            </div>
          </div>
          <div class="comparison-delta">{delta_str(score_maint, score_active, higher_is_better=True)}</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk Score
        risk_maint  = maint_data["Risk_Score"].mean()
        risk_active = active_data["Risk_Score"].mean() if not active_data.empty else risk_maint
        pm2.markdown(f"""
        <div class="comparison-box">
          <div class="comparison-label">Risk Score</div>
          <div style="display:flex;gap:0.75rem;align-items:flex-end;margin-top:0.3rem;">
            <div>
              <div style="font-size:0.62rem;color:{DANGER};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">During Maint.</div>
              <div class="comparison-value" style="color:{WARN};">{risk_maint:.2f}</div>
            </div>
            <div style="width:1px;background:{BORDER};height:32px;margin-bottom:4px;"></div>
            <div>
              <div style="font-size:0.62rem;color:{ACCENT2};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">Active/Idle</div>
              <div class="comparison-value" style="color:{TEXT_SEC};font-size:1rem;">
                {risk_active:.2f}
              </div>
            </div>
          </div>
          <div class="comparison-delta">
            {delta_str(risk_maint, risk_active, higher_is_better=False)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Lead Time
        lt_maint  = maint_data["Early_Warning_Lead_Time_Hours"].mean()
        lt_active = active_data["Early_Warning_Lead_Time_Hours"].mean() if not active_data.empty else lt_maint
        pm3.markdown(f"""
        <div class="comparison-box">
          <div class="comparison-label">Lead Time (hrs)</div>
          <div style="display:flex;gap:0.75rem;align-items:flex-end;margin-top:0.3rem;">
            <div>
              <div style="font-size:0.62rem;color:{DANGER};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">During Maint.</div>
              <div class="comparison-value" style="color:{ACCENT2};">{lt_maint:.1f}</div>
            </div>
            <div style="width:1px;background:{BORDER};height:32px;margin-bottom:4px;"></div>
            <div>
              <div style="font-size:0.62rem;color:{ACCENT2};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">Active/Idle</div>
              <div class="comparison-value" style="color:{TEXT_SEC};font-size:1rem;">
                {lt_active:.1f}
              </div>
            </div>
          </div>
          <div class="comparison-delta">
            {delta_str(lt_maint, lt_active, higher_is_better=True)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Maintenance Score
        ms_maint  = maint_data["Predictive_Maintenance_Score"].mean()
        ms_active = active_data["Predictive_Maintenance_Score"].mean() if not active_data.empty else ms_maint
        pm4.markdown(f"""
        <div class="comparison-box">
          <div class="comparison-label">Maintenance Score</div>
          <div style="display:flex;gap:0.75rem;align-items:flex-end;margin-top:0.3rem;">
            <div>
              <div style="font-size:0.62rem;color:{DANGER};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">During Maint.</div>
              <div class="comparison-value" style="color:{ACCENT};">{ms_maint:.3f}</div>
            </div>
            <div style="width:1px;background:{BORDER};height:32px;margin-bottom:4px;"></div>
            <div>
              <div style="font-size:0.62rem;color:{ACCENT2};text-transform:uppercase;
                          font-weight:600;margin-bottom:2px;">Active/Idle</div>
              <div class="comparison-value" style="color:{TEXT_SEC};font-size:1rem;">
                {ms_active:.3f}
              </div>
            </div>
          </div>
          <div class="comparison-delta">
            {delta_str(ms_maint, ms_active, higher_is_better=True)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

        sectionp("Post Maintenance Behaviour ")

        # ── CHART: Maintenance Score Trend with Operation Mode overlay ──
        fig, ax1 = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax1.set_facecolor(CARD_BG)
        for sp in ax1.spines.values():
            sp.set_edgecolor(BORDER)

        # Shade maintenance windows in background
        in_maint  = False
        maint_start = None
        for _, row in machine_data.iterrows():
            if row["Operation_Mode"] == "Maintenance" and not in_maint:
                in_maint    = True
                maint_start = row["Datetime"]
            elif row["Operation_Mode"] != "Maintenance" and in_maint:
                ax1.axvspan(maint_start, row["Datetime"],
                            alpha=0.12, color=ACCENT, label="_nolegend_")
                in_maint = False
        if in_maint:
            ax1.axvspan(maint_start, machine_data["Datetime"].iloc[-1],
                        alpha=0.12, color=ACCENT)

        # Maintenance Score line
        ax1.fill_between(machine_data["Datetime"],
                         machine_data["Predictive_Maintenance_Score"],
                         alpha=0.12, color=ACCENT)
        ax1.plot(machine_data["Datetime"],
                 machine_data["Predictive_Maintenance_Score"],
                 color=ACCENT, linewidth=1.8, label="Maintenance Score")

        # Rolling Maintenance Score
        ax1.plot(machine_data["Datetime"],
                 machine_data["Rolling_Maintenance_Score"],
                 color=WARN, linewidth=2.0, linestyle="--",
                 label="Rolling Maintenance Score")

        # Anomaly Score on secondary axis
        ax2 = ax1.twinx()
        ax2.set_facecolor(CARD_BG)
        ax2.plot(machine_data["Datetime"],
                 machine_data["Anomaly_Score"],
                 color=ACCENT2, linewidth=1.2, alpha=0.6,
                 label="Anomaly Score (right axis)")
        ax2.axhline(0, color=DANGER, linewidth=0.8, linestyle=":",
                    alpha=0.5)
        ax2.set_ylabel("Anomaly Score", color=TEXT_SEC)
        ax2.tick_params(colors=TEXT_SEC)
        ax2.spines["right"].set_edgecolor(BORDER)

        ax1.set_title(
            f"Post-Maintenance Behavior — M-{selected_machine}  "
            f"(Green shaded areas = Maintenance Mode)", pad=10)
        ax1.set_ylabel("Maintenance Score  (0 = worst, 1 = best)")
        ax1.grid(axis="y", color=BORDER, alpha=0.4, linewidth=0.6)
        ax1.grid(axis="x", visible=False)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")

        # Combined legend from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Green patch legend entry for maintenance windows
        import matplotlib.patches as mpatches
        maint_patch = mpatches.Patch(color=ACCENT, alpha=0.3,
                                     label="Maintenance Window")
        ax1.legend(lines1 + lines2 + [maint_patch],
                   labels1 + labels2 + ["Maintenance Window"],
                   loc="upper right", fontsize=8)

        c_pm_chart, c_pm_panel = st.columns([7, 3])
        with c_pm_chart:
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_pm_panel:
            maint_improve = score_maint > score_active
            st.markdown(panel_html("Breakdown", [
                ("Maint. Anomaly Score",  f"{score_maint:.4f}",
                 ACCENT if maint_improve else DANGER),
                ("Active Anomaly Score",  f"{score_active:.4f}",  TEXT_SEC),
                ("Score Change",
                 f"{'▲ Improved' if maint_improve else '▼ Worsened'} by {abs(score_maint-score_active):.4f}",
                 ACCENT if maint_improve else DANGER),
                ("Maint. Lead Time",      f"{lt_maint:.1f} hrs",  ACCENT2),
                ("Active Lead Time",      f"{lt_active:.1f} hrs", TEXT_SEC),
                ("Maint. Risk Score",     f"{risk_maint:.2f}",    WARN),
                ("Active Risk Score",     f"{risk_active:.2f}",   TEXT_SEC),
                ("Green shading",         "= Maintenance window", ACCENT),
            ], border_color=ACCENT), unsafe_allow_html=True)

        maint_verdict = (
            "Machine health <b>improves</b> during maintenance — "
            "the model confirms maintenance is effective."
            if score_maint > score_active
            else "Machine health does <b>not clearly improve</b> during maintenance — "
            "review maintenance procedures or extend downtime window."
        )
        finding_text(
            f"Post-maintenance comparison for M-{selected_machine}: {maint_verdict} "
            f"Anomaly Score during maintenance: <b>{score_maint:.4f}</b> vs "
            f"active/idle: <b>{score_active:.4f}</b>. "
            f"Lead time during maintenance: <b>{lt_maint:.1f} hrs</b>. "
            f"Green shaded bands mark every maintenance window identified from Operation_Mode data."
        )

        # ── BEFORE vs AFTER TABLE ─────────────────────────────
        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        sectionp("Before vs During Maintenance — Summary Table")

        comparison_data = {
            "Metric": [
                "Anomaly Score (higher = more normal)",
                "Risk Score (lower = safer)",
                "Lead Time Hours (higher = more time)",
                "Maintenance Score (higher = healthier)",
                "Anomaly Rate % (lower = safer)",
                "Error Rate % (lower = safer)",
            ],
            "During Maintenance": [
                f"{score_maint:.4f}",
                f"{risk_maint:.2f}",
                f"{lt_maint:.1f} hrs",
                f"{ms_maint:.3f}",
                f"{(maint_data['Anomaly_Label']==-1).mean()*100:.2f}%",
                f"{maint_data['Error_Rate_%'].mean():.2f}%",
            ],
            "Active / Idle": [
                f"{score_active:.4f}",
                f"{risk_active:.2f}",
                f"{lt_active:.1f} hrs",
                f"{ms_active:.3f}",
                f"{(active_data['Anomaly_Label']==-1).mean()*100:.2f}%" if not active_data.empty else "—",
                f"{active_data['Error_Rate_%'].mean():.2f}%" if not active_data.empty else "—",
            ],
        }
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── FLEET MAINTENANCE SCORE BY ACTION ─────────────────────
    sectionp("Fleet-Wide Maintenance Score by Action Type")
    maint_summary = (
        dashboard_df.groupby("Maintenance_Action")["Predictive_Maintenance_Score"]
        .mean().reset_index()
    )
    ms_sorted = maint_summary.sort_values("Predictive_Maintenance_Score", ascending=False)
    best_action  = ms_sorted.iloc[0]["Maintenance_Action"]
    best_ms_val  = round(ms_sorted.iloc[0]["Predictive_Maintenance_Score"], 3)
    worst_action = ms_sorted.iloc[-1]["Maintenance_Action"]
    worst_ms_val = round(ms_sorted.iloc[-1]["Predictive_Maintenance_Score"], 3)

    c_ms, c_msp = st.columns([7, 3])
    with c_ms:
        fig, ax = styled_fig(figsize=(7, 5.5))
        colors  = [ACTION_COLOR.get(a, ACCENT2) for a in maint_summary["Maintenance_Action"]]
        ax.barh(maint_summary["Maintenance_Action"],
                maint_summary["Predictive_Maintenance_Score"],
                color=colors, height=0.4, zorder=2)
        ax.set_xlabel("Mean Predictive Maintenance Score")
        ax.set_title("Average Maintenance Health Score by Action Required", pad=10)
        ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(maint_summary["Predictive_Maintenance_Score"]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=TEXT_PRI)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_msp:
        st.markdown(panel_html("Breakdown", [
            ("Highest Score Action",  best_action[:22],   ACCENT),
            ("Its Score",             str(best_ms_val),   ACCENT),
            ("Lowest Score Action",   worst_action[:22],  DANGER),
            ("Its Score",             str(worst_ms_val),  DANGER),
            ("Score range",           "0 (worst) → 1 (best)", TEXT_SEC),
            ("What it means",         "Higher = healthier machine", TEXT_SEC),
        ], border_color=ACCENT), unsafe_allow_html=True)

    
    finding_text( 
        f"Machines under <b>{best_action}</b> have the highest average maintenance score " 
        f"(<b>{best_ms_val}</b>), indicating the healthiest operational state. " 
        f"Machines requiring <b>{worst_action}</b> have the lowest score " 
        f"(<b>{worst_ms_val}</b>). This confirms that maintenance action categories " 
        f"are aligned with machine health: more urgent actions correspond to lower " 
        f"Predictive Maintenance Scores." 
     )

    # ── FLEET ANOMALY HEATMAP ─────────────────────────────────
    sectionp("Fleet Anomaly Score Heatmap — All 50 Machines")
    pivot = (
        dashboard_df.groupby(
            [pd.Grouper(key="Datetime", freq="W"), "Machine_ID"]
        )["Anomaly_Score"].mean().unstack("Machine_ID")
    )
    fleet_min_score = round(pivot.values.min(), 4)
    fleet_max_score = round(pivot.values.max(), 4)
    reddest_machine = f"M-{pivot.T.min(axis=1).idxmin()}"
    reddest_week    = pivot.min(axis=1).idxmin().strftime("%d-%b")

    c_hmap, c_hmapp = st.columns([7, 3])
    with c_hmap:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        cmap2 = sns.diverging_palette(133, 10, as_cmap=True)
        sns.heatmap(
            pivot.T, cmap=cmap2, center=pivot.values.mean(),
            linewidths=0.3, linecolor=DARK_BG,
            ax=ax, cbar_kws={"shrink": 0.6, "label": "Mean Anomaly Score"},
            yticklabels=[f"M-{c}" for c in pivot.columns],
            xticklabels=[
                d.strftime("%d-%b") if i % 2 == 0 else ""
                for i, d in enumerate(pivot.index)
            ]
        )
        ax.set_title(
            "Weekly Mean Anomaly Score — All 50 Machines  "
            "(Green = Normal  |  Red = Anomalous)",
            pad=12, color=TEXT_PRI)
        ax.tick_params(colors=TEXT_SEC, labelsize=7.5)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with c_hmapp:
        st.markdown(panel_html("Breakdown", [
            ("Most Anomalous Machine", reddest_machine,     DANGER),
            ("Riskiest Week",          reddest_week,        DANGER),
            ("Fleet Min Score",        str(fleet_min_score), DANGER),
            ("Fleet Max Score",        str(fleet_max_score), ACCENT),
            ("Rows = Machines",        "M-1 to M-50",       TEXT_SEC),
            ("Columns = Weeks",        "Weekly averages",   TEXT_SEC),
            ("Green cells",            "Normal behaviour",  ACCENT),
            ("Red cells",              "Anomalous behaviour", DANGER),
        ], border_color=ACCENT2), unsafe_allow_html=True)

    

    finding_text( 
    f"<b>{reddest_machine}</b> recorded the lowest weekly mean Anomaly Score " 
    f"around <b>{reddest_week}</b>, making it the most anomalous machine-week " 
    f"combination in the selected period. " 
    f"Across the fleet, weekly mean Anomaly Scores ranged from " 
    f"<b>{fleet_min_score}</b> (most anomalous) to " 
    f"<b>{fleet_max_score}</b> (most normal). " 
    f"Periods where multiple machines show red cells in the same week may indicate " 
    f"a shared fleet-wide condition, such as environmental changes, operational " 
    f"stress, or a common infrastructure issue, and should be investigated further." 
     )

    # ── KEY FINDING: HISTORICAL RISK ANALYSIS ────────────────
    maint_verdict_kf = "improved" if kf_maint_improves else "did not clearly improve"
    detailed_key_findings(
        "Historical Risk Analysis",
        "Fleet-wide average Early Warning Lead Time is <b>98.9 hrs</b> across all "
        "100,000 readings — the predictive model provides operations teams with "
        "nearly four full days of advance warning before predicted machine failure. "
        "<b>23 readings</b> fell below the critical 25-hour threshold, "
        "representing the highest-urgency maintenance events in the observed period. "
        "Post-maintenance analysis confirms that machine health <b>" + maint_verdict_kf + "</b> "
        "following maintenance interventions, providing evidence on maintenance "
        "effectiveness across the Thales manufacturing fleet."
    )
# ════════════════════════════════════════════════════════════
with tab5:

    tab_header(5)
    #section("Autoencoder Model KPIs", tab_num=5)



    if not AE_AVAILABLE:
        # ── AE not run yet — show instructions ───────────────
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {WARN};border-radius:8px;
                    padding:1.5rem 2rem;margin:1rem 0;">
          <div style="font-size:1rem;font-weight:700;color:{WARN};margin-bottom:0.5rem;">
            ⚠  Autoencoder output not found
          </div>
          <div style="font-size:0.85rem;color:{TEXT_SEC};line-height:1.9;">
            To enable this tab:<br>
            <b style="color:{TEXT_PRI};">Step 1</b> — Run <code>python autoencoder_anomaly.py</code>
            in your project folder<br>
            <b style="color:{TEXT_PRI};">Step 2</b> — It generates
            <code>Autoencoder_Anomaly_Output.csv</code><br>
            <b style="color:{TEXT_PRI};">Step 3</b> — Place that CSV in the
            <b>same folder</b> as <code>thales_app.py</code><br>
            <b style="color:{TEXT_PRI};">Step 4</b> — Restart the dashboard:
            <code>streamlit run thales_app.py</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ════════════════════════════════════════════════════
        # AE DATA IS AVAILABLE — SHOW FULL ANALYSIS
        # ════════════════════════════════════════════════════

        # ── AE KPI ROW ────────────────────────────────────────
        sectionp("Autoencoder Model KPIs")
        ae1, ae2, ae3, ae4, ae5 = st.columns(5)

        ae_anomaly_count    = (df["AE_Anomaly_Label"] == -1).sum()
        ae_anomaly_pct      = round(ae_anomaly_count / len(df) * 100, 1)
        avg_ae_score        = round(df["AE_Anomaly_Score"].mean(), 2)
        avg_recon_error     = round(df["AE_Reconstruction_Error"].mean(), 6)
        avg_ensemble        = round(df["Ensemble_Anomaly_Score"].mean(), 2)

        # Both models agree on anomaly — highest confidence
        both_anomaly_count  = (
            (df["Anomaly_Label"] == -1) &
            (df["AE_Anomaly_Label"] == -1)
        ).sum()

        ae1.markdown(kpi_card(
            "AE Anomalies Detected",
            f"{ae_anomaly_count:,}",
            "red",
            f"{ae_anomaly_pct}% of<br> readings"
        ), unsafe_allow_html=True)

        ae2.markdown(kpi_card(
            "Avg AE Anomaly Score",
            avg_ae_score,
            "amber" if avg_ae_score > 40 else "green",
            "0=normal ·<br> 100=critical"
        ), unsafe_allow_html=True)

        ae3.markdown(kpi_card(
            "Avg Reconstruction Error",
            f"{avg_recon_error:.5f}",
            "amber",
            "MSE · Lower = better fit"
        ), unsafe_allow_html=True)

        ae4.markdown(kpi_card(
            "Both Models Agree",
            f"{both_anomaly_count:,}",
            "red",
            "Indicates <br>High-confidence anomalies"
        ), unsafe_allow_html=True)

        ae5.markdown(kpi_card(
            "Avg Ensemble Score",
            avg_ensemble,
            "amber" if avg_ensemble > 40 else "green",
            "IF + AE <br>combined <br>(0-100)"
        ), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

        # ── HOW AUTOENCODER WORKS INFO BOX ────────────────────
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;
                    padding:0.9rem 1.2rem;margin-bottom:1rem;">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
            <div>
              <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.08em;color:{ACCENT2};margin-bottom:4px;">
                How it works
              </div>
              <div style="font-size:0.78rem;color:{TEXT_SEC};line-height:1.6;">
                Trained on <b style="color:{TEXT_PRI};">normal data only</b>.
                Learns to compress 12 features → 4 neurons → reconstruct back.
                Cannot reconstruct anomalous patterns → high error.
              </div>
            </div>
            <div>
              <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.08em;color:{ACCENT};margin-bottom:4px;">
                Reconstruction Error
              </div>
              <div style="font-size:0.78rem;color:{TEXT_SEC};line-height:1.6;">
                MSE between original sensor reading and reconstructed reading.
                <b style="color:{DANGER};">High error = anomaly</b>.
                <b style="color:{ACCENT};">Low error = normal</b>.
              </div>
            </div>
            <div>
              <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.08em;color:{WARN};margin-bottom:4px;">
                Ensemble Model
              </div>
              <div style="font-size:0.78rem;color:{TEXT_SEC};line-height:1.6;">
                Average of Isolation Forest + Autoencoder scores.
                Readings flagged by <b style="color:{DANGER};">both models</b>
                are highest-confidence anomalies.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ROW 1: AE Score Distribution + IF vs AE Comparison ─
        sectionp("Autoencoder Score Analysis")
        #col_a, col_b = st.columns(2)

        #with col_a:
        risk_order     = ["Low Risk", "Medium Risk", "High Risk"]
        risk_colors_bp = [ACCENT, WARN, DANGER]
        ae_by_risk = [
            dashboard_df[dashboard_df["Predictive_Maintenance_Risk"] == r
                            ]["AE_Anomaly_Score"].dropna().values
            for r in risk_order
        ]
        ae_medians = {
            r: round(float(np.median(v)), 1) if len(v) > 0 else 0
            for r, v in zip(risk_order, ae_by_risk)
        }

        c_bp, c_bpp = st.columns([7, 3])
        with c_bp:
            fig, ax = styled_fig(figsize=(7, 4.2))
            bp = ax.boxplot(
                ae_by_risk,
                patch_artist=True,
                widths=0.45,
                medianprops=dict(color=TEXT_PRI, linewidth=2)
            )
            # Set tick labels separately — compatible with all matplotlib versions
            ax.set_xticks(range(1, len(risk_order) + 1))
            ax.set_xticklabels(risk_order, fontsize=9)
            for patch, color in zip(bp["boxes"], risk_colors_bp):
                patch.set_facecolor(color); patch.set_alpha(0.65)
            ax.set_title("AE Anomaly Score by Risk Level\n(Higher = More Abnormal ✓)", pad=10)
            ax.set_ylabel("AE Anomaly Score (0–100)")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_bpp:
            st.markdown(panel_html("Breakdown", [
                ("Low Risk median",    f"{ae_medians['Low Risk']}",    ACCENT),
                ("Medium Risk median", f"{ae_medians['Medium Risk']}",  WARN),
                ("High Risk median",   f"{ae_medians['High Risk']}",   DANGER),
                ("Score scale",        "0 = normal · 100 = critical",  TEXT_SEC),
                ("Convention",         "Higher = More Abnormal ✓",     TEXT_SEC),
            ], border_color=DANGER), unsafe_allow_html=True)

        finding_text(
            f"AE scores correctly stratify by risk level — "
            f"High Risk median: <b>{ae_medians['High Risk']}</b>, "
            f"Medium Risk: <b>{ae_medians['Medium Risk']}</b>, "
            f"Low Risk: <b>{ae_medians['Low Risk']}</b>. "
            f"This provides strong evidence that the Autoencoder is effectively distinguishing normal and anomalous machine behaviour, as AE anomaly scores increase consistently from Low Risk to High Risk readings. "
            f"and assigns higher reconstruction error (higher score) to anomalous machines.",
            icon="🤖"
        )


        sectionp("Isolation Forest VS AutoEncoder Model Agreement Scatter ")
        
        #with col_b:
        both_flag = ((df["Anomaly_Label"] == -1) &
                        (df["AE_Anomaly_Label"] == -1)).sum()
        only_if   = ((df["Anomaly_Label"] == -1) &
                        (df["AE_Anomaly_Label"] == 1)).sum()
        only_ae   = ((df["Anomaly_Label"] == 1) &
                        (df["AE_Anomaly_Label"] == -1)).sum()
        agree_pct = round((df["Anomaly_Label"] == df["AE_Anomaly_Label"]).mean() * 100, 1)

        c_sc, c_scp = st.columns([7, 3])
        with c_sc:
            fig, ax = styled_fig(figsize=(7, 4.2))
            sample = df.sample(min(3000, len(df)), random_state=42)
            def scatter_color(row):
                if row["Anomaly_Label"] == -1 and row["AE_Anomaly_Label"] == -1:
                    return DANGER
                elif row["Anomaly_Label"] == -1:
                    return WARN
                elif row["AE_Anomaly_Label"] == -1:
                    return ACCENT2
                else:
                    return "#3A3F47"
            colors_sc = sample.apply(scatter_color, axis=1)
            ax.scatter(sample["Normalized_Anomaly_Score"], sample["AE_Anomaly_Score"],
                        c=colors_sc, alpha=0.5, s=10, linewidths=0)
            ax.set_xlabel("Isolation Forest Score (0–100)")
            ax.set_ylabel("Autoencoder Score (0–100)")
            ax.set_title("IF vs AE Model Agreement Scatter", pad=10)
            from matplotlib.patches import Patch
            legend_e = [
                Patch(facecolor=DANGER,    label="Both flag"),
                Patch(facecolor=WARN,      label="Only IF"),
                Patch(facecolor=ACCENT2,   label="Only AE"),
                Patch(facecolor="#3A3F47", label="Both normal"),
            ]
            ax.legend(handles=legend_e, fontsize=8, loc="upper left")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_scp:
            st.markdown(panel_html("Breakdown", [
                ("Model Agreement",    f"{agree_pct}%",      ACCENT),
                ("Both Flag",          f"{both_flag:,}",     DANGER),
                ("Only IF Flags",      f"{only_if:,}",       WARN),
                ("Only AE Flags",      f"{only_ae:,}",       ACCENT2),
                ("High confidence",    "Both-flag readings", TEXT_SEC),
            ], border_color=ACCENT2), unsafe_allow_html=True)

        finding_text(
            f"Both models agree on <b>{agree_pct}%</b> of readings. "
            f"<b>{both_flag:,}</b> readings are flagged by both — these are the "
            f"highest-confidence anomalies. <b>{only_if:,}</b> flagged only by "
            f"Isolation Forest, <b>{only_ae:,}</b> only by Autoencoder. "
            f"Readings unique to one model represent subtle anomalies each "
            f"algorithm captures differently.",
            icon="🤖"
        )

        # ── ROW 2: Reconstruction Error trend + Top machines ────
        sectionp(f"Machine M-{selected_machine} — Reconstruction Error Trend")

        ae_machine = machine_data.copy()

        #col_c, col_d = st.columns([2, 1])

        #@with col_c:

        fig, ax = styled_fig(figsize=(7, 4.2))
        ax.fill_between(
            ae_machine["Datetime"],
            ae_machine["AE_Reconstruction_Error"],
            alpha=0.18, color=DANGER
        )
        ax.plot(
            ae_machine["Datetime"],
            ae_machine["AE_Reconstruction_Error"],
            color=DANGER, linewidth=1.5,
            label="Reconstruction Error (MSE)"
        )
        rolling_recon = ae_machine["AE_Reconstruction_Error"].rolling(
            7, min_periods=1).mean()
        ax.plot(ae_machine["Datetime"], rolling_recon,
                color=WARN, linewidth=2.2, linestyle="--", label="7-pt Rolling Mean")
        ae_anom_pts = ae_machine[ae_machine["AE_Anomaly_Label"] == -1]
        if not ae_anom_pts.empty:
            ax.scatter(ae_anom_pts["Datetime"], ae_anom_pts["AE_Reconstruction_Error"],
                        color=DANGER, s=30, zorder=5,
                        label=f"AE Anomaly ({len(ae_anom_pts)} pts)")
        ax.set_title(f"M-{selected_machine} Reconstruction Error — High spike = anomaly", pad=10)
        ax.set_ylabel("Reconstruction Error (MSE)")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=30, ha="right")
        #st.pyplot(fig, use_container_width=True)
        #plt.close()

        # Panel below chart
        recon_max  = round(ae_machine["AE_Reconstruction_Error"].max(), 6)
        recon_mean = round(ae_machine["AE_Reconstruction_Error"].mean(), 6)
        ae_anom_n  = len(ae_anom_pts)
        


        col_c22, col_d22 = st.columns([7, 3])
        with col_c22:
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_d22:
            st.markdown(panel_html("Breakdown", [
                        ("AE Anomaly Points",   str(ae_anom_n),    DANGER),
                        ("Peak Recon Error",    str(recon_max),    DANGER),
                        ("Avg Recon Error",     str(recon_mean),   WARN),
                        ("What it is",          "MSE = mean((input − output)²)", TEXT_SEC),
                        ("High spike means",    "Model failed to reconstruct → anomaly", TEXT_SEC),
                    ], border_color=DANGER), unsafe_allow_html=True)
            

            
                
        finding_text(
            f"Machine M-{selected_machine} has <b>{ae_anom_n}</b> AE-detected anomaly events. "
            f"Peak reconstruction error: <b>{recon_max}</b>. "
            f"Spikes in the red area mark moments when the Autoencoder failed to reconstruct "
            f"sensor patterns — because it was trained only on normal data, it cannot reconstruct "
            f"abnormal patterns, producing high MSE. These spikes are early fault signatures. High reconstruction-error spikes indicate observations that are more likely to be anomalous; points classified as AE_Anomaly_Label = -1 are the model's actual anomaly detections. ",
            #f"High reconstruction-error spikes indicate observations that are more likely to be anomalous; points classified as AE_Anomaly_Label = -1 are the model's actual anomaly detections. .",
            icon="🤖"
        )
        
        sectionp("AE Seveerity")

        #with col_d:
        ae_sev = ae_machine["AE_Anomaly_Severity"].value_counts()
        if not ae_sev.empty:
            c_pie5, c_pie5p = st.columns([7, 3])
            with c_pie5:
                fig, ax = plt.subplots(figsize=(4.5, 20))
                fig.patch.set_facecolor(DARK_BG)
                ax.set_facecolor(DARK_BG)
                sev_colors = {"Critical": DANGER, "Medium": WARN, "Low": ACCENT}
                colors_pie = [sev_colors.get(s, ACCENT2) for s in ae_sev.index]
                wedges, texts, autotexts = ax.pie(
                    ae_sev.values, labels=ae_sev.index, colors=colors_pie,
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"linewidth": 2, "edgecolor": DARK_BG},
                    textprops={"color": TEXT_PRI, "fontsize": 9}
                )
                for at in autotexts:
                    at.set_color(DARK_BG); at.set_fontweight("bold")
                ax.set_title(f"M-{selected_machine}\nAE Severity", color=TEXT_PRI, pad=10)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with c_pie5p:
                ae_top_sev = ae_sev.idxmax()
                ae_top_pct = round(ae_sev.max()/ae_sev.sum()*100,1)
                sev_rows5  = [("Dominant Severity", ae_top_sev, TEXT_PRI),
                                ("Its Share", f"{ae_top_pct}%", ACCENT2)]
                for s, cnt in ae_sev.items():
                    c = DANGER if s=="Critical" else WARN if s=="Medium" else ACCENT
                    sev_rows5.append((f"{s} count", f"{cnt:,}", c))
                st.markdown(panel_html("AE Severity", sev_rows5,
                                        border_color=WARN), unsafe_allow_html=True)

        
        
        

        # ── ROW 3: IF vs AE vs Ensemble trend ────────────────────
        sectionp(f"Model Score Comparison Trend — Machine M-{selected_machine}")
        m_if_mean  = round(ae_machine["Normalized_Anomaly_Score"].mean(), 1)
        m_ae_mean  = round(ae_machine["AE_Anomaly_Score"].mean(), 1)
        m_ens_mean = round(ae_machine["Ensemble_Anomaly_Score"].mean(), 1)
        high_conf_m = ((ae_machine["Anomaly_Label"]==-1) &
                       (ae_machine["AE_Anomaly_Label"]==-1)).sum()

        c_tr5, c_tr5p = st.columns([7, 3])
        with c_tr5:
            fig, ax = styled_fig(figsize=(7, 4.2))
            ax.plot(ae_machine["Datetime"], ae_machine["Normalized_Anomaly_Score"],
                    color=ACCENT2, linewidth=1.4, alpha=0.8, label="IF Score (0–100)")
            ax.plot(ae_machine["Datetime"], ae_machine["AE_Anomaly_Score"],
                    color=DANGER, linewidth=1.4, alpha=0.8, label="AE Score (0–100)")
            ax.plot(ae_machine["Datetime"], ae_machine["Ensemble_Anomaly_Score"],
                    color=WARN, linewidth=2.4, label="Ensemble Score (IF+AE avg)")
            ax.axhline(97, color=DANGER, linewidth=0.9, linestyle=":", alpha=0.7,
                       label="High Risk (97)")
            ax.axhline(66, color=WARN, linewidth=0.8, linestyle=":", alpha=0.6,
                       label="Medium Risk (66)")
            ax.set_title(f"IF vs AE vs Ensemble — M-{selected_machine}  (Higher = More Abnormal)", pad=10)
            ax.set_ylabel("Anomaly Score (0–100)")
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_tr5p:
            st.markdown(panel_html("Breakdown", [
                ("Avg IF Score",       str(m_if_mean),     ACCENT2),
                ("Avg AE Score",       str(m_ae_mean),     DANGER),
                ("Avg Ensemble Score", str(m_ens_mean),    WARN),
                ("Both-model events",  str(high_conf_m),   DANGER),
                ("High Risk line",     "Score ≥ 97",       DANGER),
                ("Medium Risk line",   "Score ≥ 66",       WARN),
            ], border_color=WARN), unsafe_allow_html=True)

        finding_text(
            f"Machine M-{selected_machine}: Avg IF Score <b>{m_if_mean}</b>, "
            f"Avg AE Score <b>{m_ae_mean}</b>, Avg Ensemble <b>{m_ens_mean}</b>. "
            f"<b>{high_conf_m}</b> readings werw classified  "
            f"as anomalous by both Isolation Forest and Autoencoder simultaneously"
            f"When IF and AE scores spike together, the fault is almost certainly real.",
            icon="🤖"
        )

        # ── ROW 4: Ensemble Risk Distribution + Top machines ────
        sectionp("Fleet Ensemble Risk Analysis")
        #col_e, col_f = st.columns(2)

        #with col_e:
        ens_counts  = df["Ensemble_Risk"].value_counts()
        ens_hr      = ens_counts.get("High Risk", 0)
        ens_mr      = ens_counts.get("Medium Risk", 0)
        ens_lr      = ens_counts.get("Low Risk", 0)
        ens_hr_pct  = round(ens_hr / len(df) * 100, 1)
        ens_order   = [r for r in ["High Risk","Medium Risk","Low Risk"]
                        if r in ens_counts.index]
        ens_vals    = [ens_counts.get(r, 0) for r in ens_order]
        ens_colors  = [RISK_COLOR.get(r, ACCENT2) for r in ens_order]

        c_ed, c_edp = st.columns([7, 3])
        with c_ed:
            fig, ax = styled_fig(figsize=(7, 4.2))
            bars = ax.bar(ens_order, ens_vals, color=ens_colors, width=0.5, zorder=2)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 50,
                        f"{bar.get_height():,}",
                        ha="center", va="bottom", fontsize=9, color=TEXT_PRI)
            ax.set_title("Ensemble Risk Distribution\n(IF + AE Combined)", pad=10)
            ax.set_ylabel("Number of Readings")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_edp:
            st.markdown(panel_html("Breakdown", [
                ("Ensemble High Risk",   f"{ens_hr:,}",    DANGER),
                ("Ensemble Medium Risk", f"{ens_mr:,}",    WARN),
                ("Ensemble Low Risk",    f"{ens_lr:,}",    ACCENT),
                ("High Risk Share",      f"{ens_hr_pct}%", DANGER),
                ("Ensemble Formula",              "(IF + AE) / 2",  TEXT_SEC),
            ], border_color=WARN), unsafe_allow_html=True)

        finding_text(
            f"Ensemble model flags <b>{ens_hr:,}</b> readings as High Risk "
            f"({ens_hr_pct}% of fleet). The ensemble combines Isolation Forest "
            f"and Autoencoder scores — averaging two independent models reduces "
            f"false alarms and increases confidence in the final risk classification.",
            icon="🤖"
        )

        sectionp("Top 10 Most Anomalous Machines")
        #with col_f:
        top10_ens = (
            df.groupby("Machine_ID")["Ensemble_Anomaly_Score"]
            .mean().sort_values(ascending=False).head(10)
        )
        top_ens_m  = f"M-{top10_ens.index[0]}"
        top_ens_v  = round(top10_ens.iloc[0], 1)
        last_ens_v = round(top10_ens.iloc[-1], 1)

        c_et, c_etp = st.columns([7, 3])
        with c_et:
            fig, ax = styled_fig(figsize=(7, 4))
            bar_colors_e = [
                DANGER if v >= top10_ens.quantile(0.7)
                else WARN if v >= top10_ens.quantile(0.4)
                else ACCENT2
                for v in top10_ens.values
            ]
            ax.bar([f"M-{m}" for m in top10_ens.index],
                    top10_ens.values, color=bar_colors_e, width=0.55, zorder=2)
            ax.set_title("Top 10 Most Anomalous Machines\n(Ensemble — IF + AE)", pad=10)
            ax.set_ylabel("Mean Ensemble Score (0–100)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c_etp:
            st.markdown(panel_html("Breakdown", [
                ("Top Ensemble Machine", top_ens_m,        DANGER),
                ("Its Avg Score",        str(top_ens_v),   DANGER),
                ("10th Rank Score",      str(last_ens_v),  WARN),
                ("Score range",          "0–100",          TEXT_SEC),
                ("Why ensemble?",        "Reduces false alarms", TEXT_SEC),
            ], border_color=DANGER), unsafe_allow_html=True)

        finding_text(
            f"<b>{top_ens_m}</b> is the highest-priority machine by Ensemble Score "
            f"(<b>{top_ens_v}</b>). The Ensemble combines Isolation Forest and Autoencoder anomaly scores, providing a combined measure of abnormality. "
            f" Machines where both models independently flag anomalies."
            f"can be treated as higher-confidence anomaly events.",
            icon="🤖"
        )

        # ── ROW 5: High-confidence anomaly table ─────────────────
        sectionp("High-Confidence Anomaly Log (Both Models Agree)")
        st.markdown(
            f'<div style="font-size:0.78rem;color:{TEXT_SEC};margin-bottom:0.5rem;">'
            f'These readings were flagged as anomalous by <b style="color:{DANGER};">'
            f'BOTH Isolation Forest AND Autoencoder</b> — '
            f'highest possible confidence. Prioritise these for immediate inspection.'
            f'</div>',
            unsafe_allow_html=True
        )

        both_df = df[
            (df["Anomaly_Label"] == -1) &
            (df["AE_Anomaly_Label"] == -1)
        ][[
            "Machine_ID", "Datetime",
            "Normalized_Anomaly_Score", "AE_Anomaly_Score",
            "Ensemble_Anomaly_Score",
            "AE_Reconstruction_Error",
            "Anomaly_Severity", "AE_Anomaly_Severity",
            "Predictive_Maintenance_Risk", "Ensemble_Risk",
            "Early_Warning_Lead_Time_Hours", "Maintenance_Action"
        ]].sort_values("Ensemble_Anomaly_Score", ascending=False).head(30).copy()

        both_df["Machine_ID"] = "M-" + both_df["Machine_ID"].astype(str)
        both_df["Datetime"]   = both_df["Datetime"].dt.strftime("%d-%b %H:%M")
        both_df["Normalized_Anomaly_Score"] = both_df["Normalized_Anomaly_Score"].round(1)
        both_df["AE_Anomaly_Score"]         = both_df["AE_Anomaly_Score"].round(1)
        both_df["Ensemble_Anomaly_Score"]   = both_df["Ensemble_Anomaly_Score"].round(1)
        both_df["AE_Reconstruction_Error"]  = both_df["AE_Reconstruction_Error"].round(6)
        both_df["Early_Warning_Lead_Time_Hours"] = \
            both_df["Early_Warning_Lead_Time_Hours"].round(1)
        both_df = both_df.rename(columns={
            "Machine_ID":                    "Machine",
            "Datetime":                      "Timestamp",
            "Normalized_Anomaly_Score":      "IF Score",
            "AE_Anomaly_Score":              "AE Score",
            "Ensemble_Anomaly_Score":        "Ensemble Score",
            "AE_Reconstruction_Error":       "Recon Error",
            "Anomaly_Severity":              "IF Severity",
            "AE_Anomaly_Severity":           "AE Severity",
            "Predictive_Maintenance_Risk":   "IF Risk",
            "Ensemble_Risk":                 "Ensemble Risk",
            "Early_Warning_Lead_Time_Hours": "Lead Time (hrs)",
            "Maintenance_Action":            "Action",
        })

        if both_df.empty:
            st.info("No readings flagged by both models under current filters.")
        else:
            st.dataframe(both_df, use_container_width=True, hide_index=True)

        # ── MODEL COMPARISON SUMMARY TABLE ────────────────────
        sectionp("Model Performance Comparison")
        model_comp = pd.DataFrame({
            "Metric": [
                "Algorithm type",
                "Training approach",
                "Anomaly score range",
                "Higher score = ?",
                "Total anomalies detected",
                "Anomaly detection rate",
                "Works best for",
                "Key output column",
            ],
            "Isolation Forest": [
                "Tree-based ensemble",
                "Unsupervised (all data)",
                "−0.16 to +0.10 (raw) / 0–100 (normalized)",
                "More abnormal ✓",
                f"{(dashboard_df['Anomaly_Label']==-1).sum():,}",
                f"{(dashboard_df['Anomaly_Label']==-1).mean()*100:.1f}%",
                "Global anomalies, fast training",
                "Normalized_Anomaly_Score",
            ],
            "Autoencoder": [
                "Deep learning (neural network)",
                "Semi-supervised (normal data only)",
                "0–100 (normalized reconstruction error)",
                "More abnormal ✓",
                f"{(df['AE_Anomaly_Label']==-1).sum():,}",
                f"{(df['AE_Anomaly_Label']==-1).mean()*100:.1f}%",
                "Complex non-linear patterns, subtle faults",
                "AE_Anomaly_Score",
            ],
        })
        st.dataframe(model_comp, use_container_width=True, hide_index=True)

        # ── DOWNLOAD AE DATASET ───────────────────────────────
        sectionp("Export Autoencoder Results")
        ae_export_cols = [
            "Machine_ID", "Datetime",
            "AE_Reconstruction_Error", "AE_Anomaly_Score",
            "AE_Anomaly_Label", "AE_Anomaly_Status", "AE_Anomaly_Severity",
            "Ensemble_Anomaly_Score", "Ensemble_Risk",
            "Predictive_Maintenance_Risk", "Early_Warning_Lead_Time_Hours"
        ]
        ae_export = filtered_df[
            [c for c in ae_export_cols if c in filtered_df.columns]
        ].copy()
        ae_csv = ae_export.to_csv(index=False)
        st.download_button(
            label="⬇  Download AE Results (CSV)",
            data=ae_csv,
            file_name=f"AE_Results_M{selected_machine}.csv",
            mime="text/csv"
        )

        # ── KEY FINDING: AUTOENCODER ANALYSIS ────────────────
        detailed_key_findings(
            "Autoencoder Analysis",
            "The Autoencoder deep learning model independently detected "
            "<b>" + str(kf_ae_count) + " anomalies</b> (" + str(kf_ae_pct) + "% of "
            "100,000 readings), operating as a semi-supervised model trained exclusively "
            "on normal machine data. "
            "<b>" + str(kf_both_agree) + " readings</b> were flagged as anomalous by "
            "BOTH Isolation Forest and Autoencoder simultaneously — these constitute the "
            "highest-confidence anomaly detections in the dataset, as two completely "
            "independent algorithms reach the same conclusion. "
            "The top machine by Ensemble Score is <b>" + str(kf_top_ens_m) + "</b>, "
            "making it the highest-priority maintenance target confirmed by both models."
        )


sectionp("Key Findings")
st.markdown(
    "<div style='font-size:0.75rem;color:" + TEXT_SEC + ";"
    "margin-bottom:0.75rem;'>"
    "Based on the <b style='color:" + TEXT_PRI + ";'>full dataset "
    "(100,000 readings · 50 machines)</b> — "
    "independent of sidebar filters."
    "</div>",
    unsafe_allow_html=True
)

kf_col1, kf_col2, kf_col3, kf_col4, kf_col5 = st.columns(5)

with kf_col1:
    st.markdown(card1, unsafe_allow_html=True)

with kf_col2:
    st.markdown(card2, unsafe_allow_html=True)

with kf_col3:
    st.markdown(card3, unsafe_allow_html=True)

with kf_col4:
    st.markdown(card4, unsafe_allow_html=True)

with kf_col5:
    st.markdown(card5, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════

st.markdown(
    "<div style='height:4px;background:linear-gradient(90deg,"
    + THALES_NAVY + " 0%," + THALES_TEAL + " 50%," + THALES_NAVY + " 100%);"
    "border-radius:2px;margin:2rem 0 1rem 0;'></div>",
    unsafe_allow_html=True
)

# ── Pre-compute executive summary values from FULL dataset ──
_df = df   # full unfiltered dataset

_total_machines     = _df["Machine_ID"].nunique()
_total_readings     = len(_df)
_anom_rate          = round((_df["Anomaly_Label"] == -1).mean() * 100, 1)
_hr_count           = (_df["Predictive_Maintenance_Risk"] == "High Risk").sum()
_hr_pct             = round(_hr_count / _total_readings * 100, 1)
_imm_count          = (_df["Maintenance_Action"] == "Immediate Maintenance Required").sum()
_fleet_lt           = round(_df["Early_Warning_Lead_Time_Hours"].mean(), 1)
_fleet_dpi          = round(_df["Downtime_Prevention_Index"].mean(), 2)
_critical_lt        = int((_df["Early_Warning_Lead_Time_Hours"] <= 25).sum())
_top_risk_m         = "M-" + str(_df.groupby("Machine_ID")["Anomaly_Score"].mean().idxmin())
_top_risk_score     = round(_df.groupby("Machine_ID")["Anomaly_Score"].mean().min(), 4)
_top_casc_m         = "M-" + str(_df.groupby("Machine_ID")["Cascading_Failure_Index"].mean().idxmax())
_top_casc_v         = round(_df.groupby("Machine_ID")["Cascading_Failure_Index"].mean().max(), 2)
_hi_casc_count      = (_df["Cascade_Risk_Level"] == "High Cascade Risk").sum()
_hi_casc_pct        = round(_hi_casc_count / _total_readings * 100, 1)
_op_risk            = _df.groupby("Operation_Mode")["Anomaly_Score"].mean().sort_values()
_riskiest_mode_es   = _op_risk.index[0]
_riskiest_score_es  = round(_op_risk.iloc[0], 4)

# AE values
if AE_AVAILABLE:
    _ae_count       = int((_df["AE_Anomaly_Label"] == -1).sum())
    _ae_pct         = round(_ae_count / _total_readings * 100, 1)
    _both_agree     = int(((_df["Anomaly_Label"] == -1) & (_df["AE_Anomaly_Label"] == -1)).sum())
    _agree_rate     = round((_df["Anomaly_Label"] == _df["AE_Anomaly_Label"]).mean() * 100, 1)
    _top_ens_m      = "M-" + str(_df.groupby("Machine_ID")["Ensemble_Anomaly_Score"].mean().idxmax())
    _ae_line1 = (
        "The Autoencoder deep learning model independently flagged "
        "<b>" + str(_ae_count) + " anomalies</b> (" + str(_ae_pct) + "%), "
        "with <b>" + str(_both_agree) + " readings</b> confirmed by both models — "
        "highest confidence detections. Ensemble model top machine: <b>" + _top_ens_m + "</b>."
    )
else:
    _ae_line1 = (
        "Autoencoder model not yet executed. Run <b>autoencoder_anomaly.py</b> "
        "and place <b>Autoencoder_Anomaly_Output.csv</b> in the project folder "
        "to enable deep learning anomaly detection and ensemble risk scoring."
    )

# Cascade pattern
if "Cascade_Failure_Pattern" in _df.columns:
    _active_cas   = int((_df["Cascade_Failure_Pattern"] == "Active Cascade Risk").sum())
    _emerging_cas = int((_df["Cascade_Failure_Pattern"] == "Emerging Cascade").sum())
    _cas_line = (
        "<b>" + str(_active_cas) + " readings</b> show Active Cascade Risk "
        "(all 3 temporal conditions met) and "
        "<b>" + str(_emerging_cas) + "</b> show Emerging Cascade patterns. "
    )
else:
    _cas_line = ""

st.markdown(
    "<div class='exec-summary-card'>"

    # Header row
    "<div style='display:flex;align-items:center;justify-content:space-between;"
    "margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid " + BORDER + ";'>"
    "<div>"
    "<div style='font-size:0.62rem;font-weight:700;letter-spacing:0.14em;"
    "text-transform:uppercase;color:" + THALES_TEAL + ";margin-bottom:4px;'>"
    "THALES GROUP — PREDICTIVE MAINTENANCE INTELLIGENCE"
    "</div>"
    "<div style='font-size:1.1rem;font-weight:700;color:" + TEXT_PRI + ";'>"
    "Executive Summary — Anomaly Detection &amp; Risk Analysis"
    "</div>"
    "</div>"
    "<div style='text-align:right;font-size:0.72rem;color:" + TEXT_SEC + ";'>"
    "Dataset: 100,000 readings · 50 machines<br>"
    "Models: Isolation Forest + Autoencoder"
    "</div>"
    "</div>"

    # Row 1 — Fleet overview
    "<div class='exec-row alert'>"
    "<strong>Fleet Risk Status:</strong>"
    "<span>"
    + str(_hr_count) + " High Risk readings (" + str(_hr_pct) + "% of fleet) — "
    + str(_imm_count) + " require Immediate Maintenance across " + str(_total_machines) + " machines. "
    "Overall anomaly detection rate: " + str(_anom_rate) + "% (Isolation Forest, contamination=0.03)."
    "</span></div>"

    # Row 2 — Top risk machine
    "<div class='exec-row alert'>"
    "<strong>Highest Risk Machine:</strong>"
    "<span>"
    + _top_risk_m + " recorded the lowest fleet Anomaly Score (" + str(_top_risk_score) + ") — "
    "most consistently anomalous machine across the full observation period. "
    "Priority 1 target for immediate inspection and maintenance scheduling."
    "</span></div>"

    # Row 3 — Cascade risk
    "<div class='exec-row warn'>"
    "<strong>Cascade Failure Risk:</strong>"
    "<span>"
    + str(_hi_casc_count) + " readings (" + str(_hi_casc_pct) + "%) carry High Cascade Risk. "
    + _top_casc_m + " has the highest Cascading Failure Index (" + str(_top_casc_v) + ") — "
    "a failure here is most likely to propagate to downstream production equipment. "
    + _cas_line +
    "Fleet-wide cascade management requires coordinated intervention, not isolated machine responses."
    "</span></div>"

    # Row 4 — Lead time
    "<div class='exec-row good'>"
    "<strong>Early Warning Performance:</strong>"
    "<span>"
    "Fleet average lead time: <b>" + str(_fleet_lt) + " hrs</b> — the predictive model "
    "provides operations teams with nearly four full days of advance warning before predicted failure. "
    "Downtime Prevention Index (DPI): <b>" + str(_fleet_dpi) + " / 10</b>. "
    + str(_critical_lt) + " readings fell below the 25-hour critical threshold "
    "requiring urgent response."
    "</span></div>"

    # Row 5 — Operation mode
    "<div class='exec-row warn'>"
    "<strong>Operational Risk Pattern:</strong>"
    "<span>"
    "<b>" + _riskiest_mode_es + "</b> is the riskiest operating mode "
    "(avg Anomaly Score: " + str(_riskiest_score_es) + "). "
    "Maintenance windows should be planned to coincide with this operational mode "
    "to maximise intervention effectiveness and minimise unplanned downtime."
    "</span></div>"

    # Row 6 — AE model
    "<div class='exec-row" + (" good" if AE_AVAILABLE else " warn") + "'>"
    "<strong>Deep Learning (Autoencoder):</strong>"
    "<span>" + _ae_line1 + "</span></div>"

    # Strategic recommendation box
    "<div class='exec-strategy'>"
    "<strong>Strategic Recommendation:</strong> "
    "Deploy a three-tier maintenance protocol — "
    "(1) <b>Immediate inspection</b> for all " + str(_imm_count) + " High Risk machines, "
    "prioritising " + _top_risk_m + " and " + _top_casc_m + " as highest combined anomaly + cascade risk; "
    "(2) <b>Scheduled inspection within 48 hours</b> for all Emerging Cascade readings; "
    "(3) <b>Operational mode restriction</b> for " + _riskiest_mode_es + " mode until "
    "the associated anomaly pattern is resolved. "
    "The Isolation Forest + Autoencoder ensemble model provides "
    "<b>" + str(_both_agree if AE_AVAILABLE else 0) + " high-confidence anomaly events</b> "
    "confirmed independently by both algorithms — these are the highest-priority "
    "maintenance targets with the lowest risk of false alarms."
    "</div>"

    "</div>",
    unsafe_allow_html=True
)



# ── DOWNLOAD ─────────────────────────────────────────────────
st.markdown(f"<hr style='border-color:{BORDER};margin:2rem 0 1rem 0;'>",
            unsafe_allow_html=True)
col_dl, col_info = st.columns([2, 3])
with col_dl:
    section("Export Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="⬇  Download Filtered Dataset (CSV)",
        data=csv,
        file_name=f"PredMaint_M{selected_machine}_filtered.csv",
        mime="text/csv"
    )
with col_info:
    section("About This Dashboard")
    ae_model_line = (
        f'Model 2: Autoencoder · Dense(12→8→4→8→12) · '
        f'MSE reconstruction error · {"Loaded ✓" if AE_AVAILABLE else "Not loaded"}'
    )
    st.markdown(f"""
    <div style="font-size:0.80rem;color:{TEXT_SEC};line-height:1.8;">
      <b style="color:{TEXT_PRI};">Model 1:</b> Isolation Forest ·
      contamination=0.03 · n_estimators=200<br>
      <b style="color:{TEXT_PRI};">{ae_model_line}</b><br>
      <b style="color:{TEXT_PRI};">Ensemble:</b>
      Average of Isolation Forest + Autoencoder normalized scores (0–100)<br>
      <b style="color:{TEXT_PRI};">Features:</b>
      Sensor deviations, Vibration-Power Ratio,
      Maintenance Score Decay, Error Escalation · 12 total<br>
      <b style="color:{TEXT_PRI};">Dataset:</b>
      100,000 readings · 50 machines · Thales Group Manufacturing IoT<br>
      <b style="color:{TEXT_PRI};">Internship:</b>
      Unified Mentor Pvt Ltd · Developed by Sanjoy Das
    </div>
    """, unsafe_allow_html=True)


#with st.expander("📋  Key Findings — All Tabs", expanded=True):
#    kf1, kf2, kf3, kf4, kf5 = st.columns(5)
#    with kf1:

# ── FOOTER ───────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0 0.5rem 0;
            font-size:0.72rem;color:{TEXT_SEC};letter-spacing:0.05em;">
  Developed by <b style="color:{TEXT_PRI};">Sanjoy Das</b> &nbsp;·&nbsp;
  BE Electrical, Jadavpur University &nbsp;·&nbsp;
  Certified Energy Auditor, BEE &nbsp;·&nbsp;
  Unified Mentor Pvt Ltd Internship
</div>
""", unsafe_allow_html=True)
