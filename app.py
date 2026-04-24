import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnRadar · AI Prediction Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load("rf_model.pkl")

model = load_model()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Root & Body ── */
:root {
  --bg:        #080c14;
  --bg2:       #0d1220;
  --bg3:       #111827;
  --panel:     #0f1825;
  --border:    #1e2d42;
  --cyan:      #00e5ff;
  --cyan-dim:  #00b8cc;
  --purple:    #a855f7;
  --pink:      #f472b6;
  --green:     #22d3a5;
  --red:       #ff4d6d;
  --amber:     #fbbf24;
  --text:      #e2e8f0;
  --muted:     #64748b;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main content background ── */
.main .block-container {
  background: var(--bg);
  padding: 2rem 2.5rem;
  max-width: 1400px;
}

/* ── Headers ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; color: var(--cyan) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }

/* ── Inputs & Selects ── */
.stSelectbox > div > div,
.stSlider, .stNumberInput, .stRadio {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}
.stSelectbox [data-baseweb="popover"] { background: var(--bg3) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--muted) !important; }
[data-testid="stSlider"] div[role="slider"] {
  background: var(--cyan) !important;
  border-color: var(--cyan) !important;
}
.stSlider > div > div > div > div { background: var(--cyan) !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.05em !important;
  background: linear-gradient(135deg, #00e5ff22, #a855f722) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan) !important;
  border-radius: 10px !important;
  padding: 0.6rem 2rem !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #00e5ff44, #a855f744) !important;
  box-shadow: 0 0 20px #00e5ff44 !important;
  transform: translateY(-1px) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--cyan) !important;
}
.streamlit-expanderContent {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Labels ── */
label, .stSelectbox label, .stSlider label {
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── Section label helper ── */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--cyan);
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 6px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, var(--border), transparent);
}

/* ── Result cards ── */
.result-card {
  border-radius: 16px;
  padding: 1.8rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.result-churn {
  background: linear-gradient(135deg, #ff4d6d15, #ff4d6d05);
  border: 1px solid #ff4d6d55;
}
.result-safe {
  background: linear-gradient(135deg, #22d3a515, #22d3a505);
  border: 1px solid #22d3a555;
}
.result-label {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem;
  font-weight: 800;
  margin: 0;
}
.result-sub {
  font-family: 'DM Mono', monospace;
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 0.3rem;
}

/* ── Stat pill ── */
.stat-pill {
  display: inline-block;
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--muted);
  margin: 2px;
}

/* ── Risk badge ── */
.badge-high   { background: #ff4d6d22; color: #ff4d6d; border: 1px solid #ff4d6d55; }
.badge-medium { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
.badge-low    { background: #22d3a522; color: #22d3a5; border: 1px solid #22d3a555; }

.stAlert { border-radius: 10px !important; }

/* ── Tab styling ── */
[data-testid="stTab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 0.5rem 0;'>
      <p style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                background:linear-gradient(90deg,#00e5ff,#a855f7);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>
        ChurnRadar
      </p>
      <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#64748b;margin:0;'>
        AI · PREDICTION ENGINE v2.0
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title">📋 Customer Profile</div>', unsafe_allow_html=True)

    # ─ Demographics ─
    gender     = st.selectbox("Gender", ["Female", "Male"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown("---")
    st.markdown('<div class="section-title">📶 Services</div>', unsafe_allow_html=True)

    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    def inet_opts(label, key):
        opts = ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"]
        default = 0
        return st.selectbox(label, opts, index=default, key=key)

    online_sec  = inet_opts("Online Security", "os")
    online_bak  = inet_opts("Online Backup", "ob")
    device_prot = inet_opts("Device Protection", "dp")
    tech_sup    = inet_opts("Tech Support", "ts")
    stream_tv   = inet_opts("Streaming TV", "stv")
    stream_mov  = inet_opts("Streaming Movies", "smov")

    st.markdown("---")
    st.markdown('<div class="section-title">💳 Account & Billing</div>', unsafe_allow_html=True)

    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.25, 118.75, 65.0, 0.25)
    total_charges   = st.slider("Total Charges ($)", 0.0, 8684.0,
                                float(monthly_charges * tenure) if tenure > 0 else monthly_charges,
                                1.0)

    st.markdown("---")
    predict_btn = st.button("🎯  ANALYZE CUSTOMER", width="stretch")


# ── Helper: build feature vector ───────────────────────────────────────────────
def build_features():
    senior_int = 1 if senior == "Yes" else 0
    row = {
        "SeniorCitizen": senior_int,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        # gender
        "gender_Female": int(gender == "Female"),
        "gender_Male":   int(gender == "Male"),
        # partner
        "Partner_No":  int(partner == "No"),
        "Partner_Yes": int(partner == "Yes"),
        # dependents
        "Dependents_No":  int(dependents == "No"),
        "Dependents_Yes": int(dependents == "Yes"),
        # phone
        "PhoneService_No":  int(phone_service == "No"),
        "PhoneService_Yes": int(phone_service == "Yes"),
        # multi lines
        "MultipleLines_No":               int(multi_lines == "No"),
        "MultipleLines_No phone service": int(multi_lines == "No phone service"),
        "MultipleLines_Yes":              int(multi_lines == "Yes"),
        # internet
        "InternetService_DSL":         int(internet == "DSL"),
        "InternetService_Fiber optic": int(internet == "Fiber optic"),
        "InternetService_No":          int(internet == "No"),
        # online security
        "OnlineSecurity_No":                  int(online_sec == "No"),
        "OnlineSecurity_No internet service": int(online_sec == "No internet service"),
        "OnlineSecurity_Yes":                 int(online_sec == "Yes"),
        # online backup
        "OnlineBackup_No":                  int(online_bak == "No"),
        "OnlineBackup_No internet service": int(online_bak == "No internet service"),
        "OnlineBackup_Yes":                 int(online_bak == "Yes"),
        # device protection
        "DeviceProtection_No":                  int(device_prot == "No"),
        "DeviceProtection_No internet service": int(device_prot == "No internet service"),
        "DeviceProtection_Yes":                 int(device_prot == "Yes"),
        # tech support
        "TechSupport_No":                  int(tech_sup == "No"),
        "TechSupport_No internet service": int(tech_sup == "No internet service"),
        "TechSupport_Yes":                 int(tech_sup == "Yes"),
        # streaming tv
        "StreamingTV_No":                  int(stream_tv == "No"),
        "StreamingTV_No internet service": int(stream_tv == "No internet service"),
        "StreamingTV_Yes":                 int(stream_tv == "Yes"),
        # streaming movies
        "StreamingMovies_No":                  int(stream_mov == "No"),
        "StreamingMovies_No internet service": int(stream_mov == "No internet service"),
        "StreamingMovies_Yes":                 int(stream_mov == "Yes"),
        # contract
        "Contract_Month-to-month": int(contract == "Month-to-month"),
        "Contract_One year":       int(contract == "One year"),
        "Contract_Two year":       int(contract == "Two year"),
        # paperless billing
        "PaperlessBilling_No":  int(paperless == "No"),
        "PaperlessBilling_Yes": int(paperless == "Yes"),
        # payment method
        "PaymentMethod_Bank transfer (automatic)":  int(payment == "Bank transfer (automatic)"),
        "PaymentMethod_Credit card (automatic)":    int(payment == "Credit card (automatic)"),
        "PaymentMethod_Electronic check":           int(payment == "Electronic check"),
        "PaymentMethod_Mailed check":               int(payment == "Mailed check"),
    }
    return pd.DataFrame([row])


# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_tagline = st.columns([2, 3])
with col_logo:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0;'>
      <p style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;
                background:linear-gradient(90deg,#00e5ff,#a855f7,#f472b6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin:0;line-height:1;'>
        ChurnRadar
      </p>
      <p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#64748b;
                letter-spacing:0.15em;margin-top:4px;'>
        CUSTOMER ATTRITION INTELLIGENCE PLATFORM
      </p>
    </div>
    """, unsafe_allow_html=True)
with col_tagline:
    st.markdown("""
    <div style='text-align:right;padding-top:1.2rem;'>
      <span class="stat-pill">🌲 Random Forest Classifier</span>
      <span class="stat-pill">📊 7,043 records trained</span>
      <span class="stat-pill">🎯 45 features</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_pred, tab_insights, tab_about = st.tabs(["🎯  Prediction", "📊  Insights & Analysis", "ℹ️  About Model"])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 · PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:

    if not predict_btn:
        st.markdown("""
        <div style='
          text-align:center;
          padding:5rem 2rem;
          background:linear-gradient(135deg,#0d122088,#0f182588);
          border:1px dashed #1e2d42;
          border-radius:20px;
          margin-top:1rem;
        '>
          <div style='font-size:3.5rem;margin-bottom:1rem;'>🎯</div>
          <p style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;
                    color:#e2e8f0;margin:0;'>
            Ready to Analyze
          </p>
          <p style='font-family:DM Mono,monospace;font-size:0.75rem;color:#64748b;
                    margin-top:0.5rem;'>
            Fill in the customer profile in the sidebar, then hit ANALYZE CUSTOMER
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        df_input   = build_features()
        prob       = model.predict_proba(df_input)[0]
        churn_prob = prob[1]
        stay_prob  = prob[0]
        prediction = model.predict(df_input)[0]
        is_churn   = prediction == "Yes" or prediction == 1

        # risk tier
        if churn_prob >= 0.70:
            risk_tier  = "HIGH RISK"
            risk_color = "#ff4d6d"
            risk_badge = "badge-high"
            risk_icon  = "🔴"
        elif churn_prob >= 0.40:
            risk_tier  = "MEDIUM RISK"
            risk_color = "#fbbf24"
            risk_badge = "badge-medium"
            risk_icon  = "🟡"
        else:
            risk_tier  = "LOW RISK"
            risk_color = "#22d3a5"
            risk_badge = "badge-low"
            risk_icon  = "🟢"

        # ── Top row: verdict + gauge + metrics ──────────────────────────────────
        c1, c2, c3 = st.columns([1.2, 1.6, 1.2])

        with c1:
            card_class = "result-churn" if is_churn else "result-safe"
            label_txt  = "WILL CHURN" if is_churn else "WILL STAY"
            label_col  = "#ff4d6d" if is_churn else "#22d3a5"
            emoji      = "⚠️" if is_churn else "✅"
            st.markdown(f"""
            <div class="result-card {card_class}">
              <div style='font-size:2.8rem;margin-bottom:0.4rem;'>{emoji}</div>
              <p class='result-label' style='color:{label_col};'>{label_txt}</p>
              <p class='result-sub'>MODEL VERDICT</p>
              <div style='margin-top:1rem;'>
                <span class='stat-pill {risk_badge}'>{risk_icon} {risk_tier}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            # Gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(churn_prob * 100, 1),
                number={"suffix": "%", "font": {"family": "DM Mono", "size": 36, "color": risk_color}},
                title={"text": "Churn Probability", "font": {"family": "Syne", "size": 13, "color": "#64748b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#64748b", "size": 9},
                             "tickcolor": "#1e2d42"},
                    "bar": {"color": risk_color, "thickness": 0.25},
                    "bgcolor": "#0d1220",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,   40],  "color": "rgba(34,211,165,0.08)"},
                        {"range": [40,  70],  "color": "rgba(251,191,36,0.08)"},
                        {"range": [70, 100],  "color": "rgba(255,77,109,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": risk_color, "width": 3},
                        "thickness": 0.8,
                        "value": churn_prob * 100,
                    },
                },
            ))
            gauge.update_layout(
                height=240,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="#0d1220",
                plot_bgcolor="#0d1220",
                font_color="#e2e8f0",
            )
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Churn Probability",  f"{churn_prob*100:.1f}%")
            st.metric("Retention Probability", f"{stay_prob*100:.1f}%")
            st.metric("Tenure",  f"{tenure} months")
            st.metric("Monthly Spend", f"${monthly_charges:.2f}")

        st.markdown("---")

        # ── Probability bar ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">⚡ Probability Breakdown</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Churn", "Retain"],
            y=[churn_prob * 100, stay_prob * 100],
            marker_color=["#ff4d6d", "#22d3a5"],
            marker_line_width=0,
            text=[f"{churn_prob*100:.1f}%", f"{stay_prob*100:.1f}%"],
            textposition="outside",
            textfont=dict(family="DM Mono", color="#e2e8f0", size=14),
            width=0.4,
        ))
        fig_bar.update_layout(
            height=220,
            paper_bgcolor="#0d1220", plot_bgcolor="#0d1220",
            margin=dict(t=20, b=10, l=0, r=0),
            yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#1e2d42",
                       ticksuffix="%", tickfont=dict(color="#64748b", family="DM Mono")),
            xaxis=dict(tickfont=dict(family="Syne", size=13, color="#e2e8f0")),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

        # ── Risk factors ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔍 Key Risk Factors Detected</div>', unsafe_allow_html=True)

        factors = []
        if contract == "Month-to-month":
            factors.append(("📅 Month-to-month contract", "high", "Highest churn risk contract type"))
        if internet == "Fiber optic":
            factors.append(("📡 Fiber optic internet", "medium", "Fiber users have elevated churn rates"))
        if tenure < 12:
            factors.append(("⏱️ Short tenure (<12 mo)", "high", "New customers are more likely to leave"))
        if online_sec == "No":
            factors.append(("🔒 No online security", "medium", "Lack of security add-ons linked to churn"))
        if tech_sup == "No":
            factors.append(("🛠️ No tech support", "medium", "Missing tech support raises churn likelihood"))
        if paperless == "Yes" and payment == "Electronic check":
            factors.append(("💳 Electronic check + paperless", "medium", "Combination associated with higher churn"))
        if monthly_charges > 80:
            factors.append(("💰 High monthly charges (>$80)", "medium", "Higher bills correlate with dissatisfaction"))
        if contract in ["One year", "Two year"]:
            factors.append(("📜 Long-term contract", "low", "Multi-year contracts reduce churn risk"))
        if tenure > 36:
            factors.append(("🏅 Long-tenured customer", "low", "Loyal customers less likely to churn"))
        if online_sec == "Yes" and tech_sup == "Yes":
            factors.append(("🛡️ Full security & support", "low", "Engaged customers with add-ons stay longer"))

        if factors:
            for name, level, desc in factors:
                color_map = {"high": "#ff4d6d", "medium": "#fbbf24", "low": "#22d3a5"}
                bg_map    = {"high": "#ff4d6d14", "medium": "#fbbf2414", "low": "#22d3a514"}
                st.markdown(f"""
                <div style='
                  display:flex; align-items:center; gap:12px;
                  background:{bg_map[level]}; border:1px solid {color_map[level]}33;
                  border-radius:10px; padding:0.7rem 1rem; margin-bottom:6px;
                '>
                  <span style='
                    font-family:DM Mono,monospace;font-size:0.65rem;font-weight:500;
                    background:{color_map[level]}22; color:{color_map[level]};
                    border:1px solid {color_map[level]}55;
                    border-radius:999px; padding:2px 9px;
                    text-transform:uppercase;letter-spacing:0.06em;
                    white-space:nowrap;
                  '>{level}</span>
                  <div>
                    <span style='font-family:Syne,sans-serif;font-size:0.85rem;
                                 font-weight:600;color:#e2e8f0;'>{name}</span>
                    <span style='font-family:DM Mono,monospace;font-size:0.7rem;
                                 color:#64748b;margin-left:8px;'>— {desc}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific risk factors flagged for this configuration.")

        # ── Recommendation ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">💡 Recommended Action</div>', unsafe_allow_html=True)
        if is_churn and churn_prob >= 0.70:
            st.error("""
**🚨 URGENT INTERVENTION REQUIRED**  
This customer has a very high churn probability. Initiate an immediate retention campaign:
- Offer a contract upgrade incentive (e.g., 2-month discount on annual plan)
- Assign a dedicated account manager for personal outreach
- Provide complimentary security/tech support trial for 3 months
""")
        elif is_churn:
            st.warning("""
**⚠️ PROACTIVE RETENTION RECOMMENDED**  
Moderate churn risk detected. Consider:
- Send a personalized loyalty offer or bundle discount
- Reach out via preferred contact channel within 7 days
- Evaluate if service plan matches customer's actual usage
""")
        else:
            st.success("""
**✅ CUSTOMER APPEARS STABLE**  
Low churn risk. Maintain relationship with:
- Regular check-ins and satisfaction surveys
- Notify of new features/upgrades relevant to their profile
- Recognize loyalty milestones (e.g., tenure rewards)
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 · INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:

    st.markdown('<div class="section-title">🌲 Feature Importance (Top 20)</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_names  = model.feature_names_in_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

    colors = []
    for f in fi_df["feature"]:
        if "Contract" in f or "tenure" in f:           colors.append("#00e5ff")
        elif "internet" in f.lower() or "Fiber" in f:  colors.append("#a855f7")
        elif "Charges" in f:                           colors.append("#f472b6")
        elif "Security" in f or "TechSupport" in f:    colors.append("#22d3a5")
        else:                                           colors.append("#64748b")

    fig_fi = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{v:.3f}" for v in fi_df["importance"]],
        textposition="outside",
        textfont=dict(family="DM Mono", color="#64748b", size=9),
    ))
    fig_fi.update_layout(
        height=520,
        paper_bgcolor="#0d1220", plot_bgcolor="#0d1220",
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(showgrid=True, gridcolor="#1e2d42",
                   tickfont=dict(color="#64748b", family="DM Mono")),
        yaxis=dict(tickfont=dict(color="#e2e8f0", family="DM Mono", size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig_fi, width="stretch", config={"displayModeBar": False})

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin-top:-0.5rem;margin-bottom:1.5rem;'>
      <span style='font-family:DM Mono,monospace;font-size:0.7rem;'>
        <span style='color:#00e5ff;'>■</span> Contract & Tenure
      </span>
      <span style='font-family:DM Mono,monospace;font-size:0.7rem;'>
        <span style='color:#a855f7;'>■</span> Internet Service
      </span>
      <span style='font-family:DM Mono,monospace;font-size:0.7rem;'>
        <span style='color:#f472b6;'>■</span> Charges
      </span>
      <span style='font-family:DM Mono,monospace;font-size:0.7rem;'>
        <span style='color:#22d3a5;'>■</span> Security & Support
      </span>
      <span style='font-family:DM Mono,monospace;font-size:0.7rem;'>
        <span style='color:#64748b;'>■</span> Other
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", "7,043")
    m2.metric("Features Used", "45")
    m3.metric("Churn Rate",    "~26.5%")
    m4.metric("Retained",      "~73.5%")
    m5.metric("Algorithm",     "RF")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Churn breakdown donut ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">🍩 Churn Distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[73.5, 26.5],
            hole=0.62,
            marker=dict(colors=["#22d3a5", "#ff4d6d"],
                        line=dict(color="#080c14", width=3)),
            textfont=dict(family="DM Mono", size=12, color="#e2e8f0"),
            hovertemplate="%{label}: %{value}%<extra></extra>",
        ))
        fig_pie.add_annotation(
            text="<b>26.5%</b><br>Churn",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Syne", size=18, color="#ff4d6d"),
        )
        fig_pie.update_layout(
            height=300,
            paper_bgcolor="#0d1220", plot_bgcolor="#0d1220",
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(font=dict(family="DM Mono", color="#64748b", size=10),
                        bgcolor="#0d1220"),
        )
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-title">📈 Churn by Contract Type</div>', unsafe_allow_html=True)
        contract_data = {
            "Contract":     ["Month-to-month", "One year", "Two year"],
            "Churn Rate %": [42.7, 11.3, 2.8],
        }
        fig_ct = go.Figure(go.Bar(
            x=contract_data["Contract"],
            y=contract_data["Churn Rate %"],
            marker_color=["#ff4d6d", "#fbbf24", "#22d3a5"],
            marker_line_width=0,
            text=[f"{v}%" for v in contract_data["Churn Rate %"]],
            textposition="outside",
            textfont=dict(family="DM Mono", color="#e2e8f0", size=13),
            width=0.5,
        ))
        fig_ct.update_layout(
            height=300,
            paper_bgcolor="#0d1220", plot_bgcolor="#0d1220",
            margin=dict(t=20, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#1e2d42",
                       ticksuffix="%", tickfont=dict(color="#64748b", family="DM Mono")),
            xaxis=dict(tickfont=dict(family="Syne", size=11, color="#e2e8f0")),
            showlegend=False,
        )
        st.plotly_chart(fig_ct, width="stretch", config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 · ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style='background:#0d1220;border:1px solid #1e2d42;border-radius:14px;padding:1.5rem;'>
          <p style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#00e5ff;margin:0 0 1rem 0;'>
            🌲 Model Architecture
          </p>
          <table style='width:100%;font-family:DM Mono,monospace;font-size:0.75rem;border-collapse:collapse;'>
            <tr><td style='color:#64748b;padding:4px 0;'>Algorithm</td><td style='color:#e2e8f0;text-align:right;'>Random Forest Classifier</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Library</td><td style='color:#e2e8f0;text-align:right;'>scikit-learn 1.6.1</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Input Features</td><td style='color:#e2e8f0;text-align:right;'>45 (one-hot encoded)</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Training Records</td><td style='color:#e2e8f0;text-align:right;'>7,043</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Target Variable</td><td style='color:#e2e8f0;text-align:right;'>Churn (Yes/No)</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Output</td><td style='color:#e2e8f0;text-align:right;'>Binary + Probability</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:#0d1220;border:1px solid #1e2d42;border-radius:14px;padding:1.5rem;'>
          <p style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#a855f7;margin:0 0 1rem 0;'>
            📁 Dataset Details
          </p>
          <table style='width:100%;font-family:DM Mono,monospace;font-size:0.75rem;border-collapse:collapse;'>
            <tr><td style='color:#64748b;padding:4px 0;'>Source</td><td style='color:#e2e8f0;text-align:right;'>Telco Customer Churn</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Total Rows</td><td style='color:#e2e8f0;text-align:right;'>7,043</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Raw Features</td><td style='color:#e2e8f0;text-align:right;'>20 (demographics + services)</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Churn Rate</td><td style='color:#e2e8f0;text-align:right;'>26.54%</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Tenure Range</td><td style='color:#e2e8f0;text-align:right;'>0 – 72 months</td></tr>
            <tr><td style='color:#64748b;padding:4px 0;'>Charges Range</td><td style='color:#e2e8f0;text-align:right;'>$18.25 – $118.75/mo</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1220,#0f1825);
                border:1px solid #1e2d42;border-radius:14px;padding:1.5rem;'>
      <p style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#f472b6;margin:0 0 0.8rem 0;'>
        ⚙️ Feature Engineering Pipeline
      </p>
      <p style='font-family:DM Mono,monospace;font-size:0.75rem;color:#64748b;line-height:1.8;margin:0;'>
        Raw CSV (20 cols) → Drop customerID → Convert TotalCharges to numeric →
        One-Hot Encode all categorical variables (pandas get_dummies) →
        45 binary/numeric features → Random Forest fit → Pickle export
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 All 45 Model Features"):
        feat_list = list(model.feature_names_in_)
        cols = st.columns(3)
        chunk = len(feat_list) // 3 + 1
        for i, col in enumerate(cols):
            with col:
                for f in feat_list[i*chunk:(i+1)*chunk]:
                    st.markdown(
                        f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
                        f"color:#64748b;'>• {f}</span>", unsafe_allow_html=True
                    )













