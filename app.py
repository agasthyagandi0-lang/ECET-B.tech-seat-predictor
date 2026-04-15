"""app.py — Self-contained ECET Rank Predictor. No separate model/pkl needed."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Constants ──
CHANCE_HIGH, CHANCE_MEDIUM, CHANCE_LOW = "High", "Medium", "Low"
CASTE_IDX = {"OC": 3, "BC": 4, "SC": 5, "ST": 6}
PKL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecet_model.pkl")

CUTOFFS = [
    ("University College of Engineering, Osmania University","Hyderabad",   "CSE",   120,  380,   820,  1100),
    ("University College of Engineering, Osmania University","Hyderabad",   "ECE",   200,  510,  1050,  1400),
    ("University College of Engineering, Osmania University","Hyderabad",   "MECH",  350,  760,  1400,  1800),
    ("University College of Engineering, Osmania University","Hyderabad",   "CIVIL", 480,  900,  1650,  2100),
    ("University College of Engineering, Osmania University","Hyderabad",   "EEE",   300,  680,  1250,  1650),
    ("JNTU College of Engineering, Hyderabad",               "Hyderabad",   "CSE",   250,  620,  1200,  1600),
    ("JNTU College of Engineering, Hyderabad",               "Hyderabad",   "ECE",   380,  830,  1500,  1950),
    ("JNTU College of Engineering, Hyderabad",               "Hyderabad",   "MECH",  520, 1100,  2000,  2600),
    ("JNTU College of Engineering, Hyderabad",               "Hyderabad",   "EEE",   430,  950,  1750,  2300),
    ("JNTU College of Engineering, Hyderabad",               "Hyderabad",   "IT",    310,  700,  1320,  1750),
    ("University College of Engineering, Kakatiya University","Warangal",   "CSE",   180,  460,   940,  1250),
    ("University College of Engineering, Kakatiya University","Warangal",   "ECE",   290,  680,  1300,  1700),
    ("University College of Engineering, Kakatiya University","Warangal",   "MECH",  480, 1000,  1900,  2450),
    ("SR Engineering College",                               "Warangal",    "CSE",   620, 1350,  2400,  3100),
    ("SR Engineering College",                               "Warangal",    "ECE",   800, 1700,  3000,  3800),
    ("Sreenidhi Institute of Science & Technology",          "Hyderabad",   "CSE",   700, 1500,  2700,  3500),
    ("Sreenidhi Institute of Science & Technology",          "Hyderabad",   "ECE",   900, 1900,  3300,  4200),
    ("Government Engineering College, Nizamabad",            "Nizamabad",   "MECH",  560, 1200,  2200,  2900),
    ("Government Engineering College, Nizamabad",            "Nizamabad",   "CSE",   480, 1020,  1900,  2500),
    ("Government Engineering College, Karimnagar",           "Karimnagar",  "CSE",   520, 1100,  2050,  2700),
    ("Government Engineering College, Karimnagar",           "Karimnagar",  "MECH",  700, 1480,  2650,  3450),
    ("Government Engineering College, Khammam",              "Khammam",     "CSE",   640, 1380,  2500,  3250),
    ("Government Engineering College, Khammam",              "Khammam",     "ECE",   820, 1750,  3100,  4000),
    ("Government Engineering College, Nalgonda",             "Nalgonda",    "CIVIL", 900, 1950,  3500,  4500),
    ("Government Engineering College, Nalgonda",             "Nalgonda",    "EEE",   750, 1620,  2900,  3750),
    ("Vignana Bharathi Institute of Technology",             "Hyderabad",   "CSE",  1100, 2300,  4000,  5200),
    ("Vignana Bharathi Institute of Technology",             "Hyderabad",   "ECE",  1400, 2900,  5000,  6500),
    ("Government Engineering College, Mahabubnagar",         "Mahabubnagar","CSE",   850, 1800,  3200,  4100),
    ("Government Engineering College, Mahabubnagar",         "Mahabubnagar","MECH", 1100, 2300,  4100,  5300),
    ("Government Engineering College, Medak",                "Medak",       "EEE",   980, 2100,  3700,  4800),
    ("Muffakham Jah College of Engineering",                 "Hyderabad",   "CSE",   430,  930,  1750,  2300),
    ("Muffakham Jah College of Engineering",                 "Hyderabad",   "ECE",   590, 1260,  2300,  3000),
    ("Chaitanya Bharathi Institute of Technology",           "Hyderabad",   "CSE",   380,  820,  1550,  2050),
    ("Chaitanya Bharathi Institute of Technology",           "Hyderabad",   "ECE",   510, 1100,  2000,  2650),
    ("CVR College of Engineering",                           "Rangareddy",  "CSE",   750, 1600,  2850,  3700),
    ("CVR College of Engineering",                           "Rangareddy",  "MECH", 1050, 2250,  4000,  5200),
    ("Gokaraju Rangaraju Institute of Engineering",          "Hyderabad",   "CSE",   680, 1450,  2600,  3400),
    ("Gokaraju Rangaraju Institute of Engineering",          "Hyderabad",   "ECE",   880, 1880,  3350,  4350),
    ("Government Engineering College, Adilabad",             "Adilabad",    "CIVIL",1200, 2500,  4400,  5700),
    ("Government Engineering College, Adilabad",             "Adilabad",    "MECH", 1050, 2200,  3900,  5100),
    ("Vaagdevi College of Engineering",                      "Warangal",    "CSE",  1300, 2750,  4800,  6200),
    ("Vaagdevi College of Engineering",                      "Warangal",    "MECH", 1600, 3400,  6000,  7800),
    ("St. Martin's Engineering College",                     "Hyderabad",   "CSE",  1500, 3100,  5500,  7100),
    ("St. Martin's Engineering College",                     "Hyderabad",   "ECE",  1900, 3900,  6800,  8800),
    ("Anurag Engineering College",                           "Medchal",     "CSE",  1700, 3600,  6400,  8300),
    ("Anurag Engineering College",                           "Medchal",     "MECH", 2100, 4400,  7800, 10100),
]

# ── Train & cache model ──
@st.cache_resource(show_spinner="Training model, please wait...")
def load_model():
    # Build synthetic dataset
    np.random.seed(42)
    records = []
    for row in CUTOFFS:
        college, district, branch = row[0], row[1], row[2]
        for caste, idx in CASTE_IDX.items():
            cutoff = row[idx]
            lo, hi = max(1, int(cutoff * 0.3)), int(cutoff * 2.5)
            for rank in np.random.randint(lo, hi, size=60):
                buf = cutoff * 0.10
                if rank <= cutoff:            label = CHANCE_HIGH
                elif rank <= cutoff + buf:    label = CHANCE_MEDIUM
                else:                         label = CHANCE_LOW
                records.append({
                    "integrated_rank": int(rank),
                    "branch": branch, "caste": caste, "district": district,
                    "cutoff": cutoff,
                    "rank_ratio": round(int(rank) / cutoff, 4),
                    "label": label,
                })

    df = pd.DataFrame(records)

    le_branch   = LabelEncoder().fit(df["branch"])
    le_caste    = LabelEncoder().fit(df["caste"])
    le_district = LabelEncoder().fit(df["district"])
    le_label    = LabelEncoder().fit([CHANCE_HIGH, CHANCE_MEDIUM, CHANCE_LOW])

    df["b"] = le_branch.transform(df["branch"])
    df["c"] = le_caste.transform(df["caste"])
    df["d"] = le_district.transform(df["district"])
    df["y"] = le_label.transform(df["label"])

    FEATS = ["integrated_rank", "b", "c", "d", "cutoff", "rank_ratio"]
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(df[FEATS], df["y"])

    return {"clf": clf, "le_branch": le_branch, "le_caste": le_caste,
            "le_district": le_district, "le_label": le_label, "feats": FEATS}

def predict(m, integrated_rank, branch, caste, districts=None):
    rows = [r for r in CUTOFFS if r[2] == branch]
    if districts:
        rows = [r for r in rows if r[1] in districts]
    if not rows:
        return pd.DataFrame(columns=["College","District","Branch","Cutoff","Your_Rank","Gap","Chance"])

    records = []
    for r in rows:
        cutoff = r[CASTE_IDX[caste]]
        try: b_enc = m["le_branch"].transform([branch])[0]
        except: b_enc = 0
        try: c_enc = m["le_caste"].transform([caste])[0]
        except: c_enc = 0
        try: d_enc = m["le_district"].transform([r[1]])[0]
        except: d_enc = 0
        rr = round(integrated_rank / cutoff, 4)
        X  = pd.DataFrame([[integrated_rank, b_enc, c_enc, d_enc, cutoff, rr]], columns=m["feats"])
        chance = m["le_label"].inverse_transform([m["clf"].predict(X)[0]])[0]
        records.append({"College": r[0], "District": r[1], "Branch": branch,
                        "Cutoff": cutoff, "Your_Rank": integrated_rank,
                        "Gap": cutoff - integrated_rank, "Chance": chance})

    return pd.DataFrame(records).sort_values("Gap", ascending=False).reset_index(drop=True)

# ── Page config ──
st.set_page_config(page_title="ECET Rank Predictor – Telangana", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f0f0f0; }
h1, h2, h3 { color: #a78bfa !important; }
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #a78bfa); color: white;
    border: none; border-radius: 8px; font-weight: 700; font-size: 16px;
    padding: 0.6rem 2rem; transition: 0.3s;
}
.stButton > button:hover { transform: scale(1.04); box-shadow: 0 0 20px #a78bfa88; }
.card { background: rgba(167,139,250,0.10); border: 1px solid #7c3aed55;
    border-radius: 12px; padding: 14px 20px; margin: 6px 0; transition: 0.2s; }
.card:hover { border-color: #a78bfa; background: rgba(167,139,250,0.18); }
.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:13px; font-weight:700; font-family:'JetBrains Mono',monospace; }
.hi { background:#166534; color:#86efac; }
.md { background:#854d0e; color:#fde68a; }
.lo { background:#7f1d1d; color:#fca5a5; }
.mbox { background:rgba(124,58,237,0.15); border:1px solid #7c3aed44; border-radius:10px; padding:12px; text-align:center; }
.mbox .n { font-size:2rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.mbox .l { font-size:0.75rem; color:#c4b5fd88; text-transform:uppercase; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(90deg,#7c3aed22,#a78bfa33,#7c3aed22);border-left:4px solid #a78bfa;padding:14px 20px;border-radius:8px;margin-bottom:20px">
  <h1 style="margin:0;font-size:1.8rem">🎓 ECET Rank Predictor</h1>
  <p style="margin:2px 0 0;color:#c4b5fd99;font-size:0.9rem">Telangana · College Admission Chance Analyser</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
model = load_model()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🔍 Your Details")
    integrated_rank = st.number_input("Integrated Rank", min_value=1, max_value=100_000, value=500, step=1)
    branch_rank     = st.number_input("Branch Rank",     min_value=1, max_value=50_000,  value=200, step=1)
    caste           = st.selectbox("Caste Category", ["OC", "BC", "SC", "ST"])
    branch          = st.selectbox("Branch / Discipline", sorted({r[2] for r in CUTOFFS}))
    districts       = st.multiselect("Preferred Districts (optional)", sorted({r[1] for r in CUTOFFS}))
    st.markdown("---")
    buffer          = st.slider("Rank Buffer %", 0, 40, 10)
    run             = st.button("🚀 Predict Colleges", use_container_width=True)

BADGE = {CHANCE_HIGH: ("hi","✅ High Chance"), CHANCE_MEDIUM: ("md","⚠️ Borderline"), CHANCE_LOW: ("lo","❌ Low Chance")}

def cards(df, empty_msg):
    if df.empty: st.info(empty_msg); return
    for _, r in df.iterrows():
        cls, txt = BADGE[r["Chance"]]
        gap = int(r["Gap"])
        gap_str = f"+{abs(gap)} safe" if gap >= 0 else f"{abs(gap)} short"
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
            <div><b style="color:#e2e8f0">{r['College']}</b><br>
              <span style="color:#a78bfa88;font-size:0.82rem">📍 {r['District']} &nbsp;|&nbsp; 🎓 {r['Branch']}</span>
            </div>
            <div style="text-align:right">
              <span class="badge {cls}">{txt}</span><br>
              <span style="font-size:0.75rem;color:#94a3b8;font-family:'JetBrains Mono',monospace">Cutoff: {int(r['Cutoff'])} · {gap_str}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

if run:
    res    = predict(model, integrated_rank, branch, caste, districts or None)
    counts = res["Chance"].value_counts().to_dict()
    high   = res[res["Chance"] == CHANCE_HIGH]
    med    = res[res["Chance"] == CHANCE_MEDIUM]
    low    = res[res["Chance"] == CHANCE_LOW]

    st.markdown("### 📊 Summary")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, integrated_rank,                  "Integrated Rank", "#60a5fa"),
        (c2, branch_rank,                      "Branch Rank",     "#60a5fa"),
        (c3, counts.get(CHANCE_HIGH,   0),     "High Chance",     "#86efac"),
        (c4, counts.get(CHANCE_MEDIUM, 0),     "Borderline",      "#fde68a"),
    ]:
        col.markdown(f'<div class="mbox"><div class="n" style="color:{color}">{val}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs([
        f"✅ High ({counts.get(CHANCE_HIGH,0)})",
        f"⚠️ Borderline ({counts.get(CHANCE_MEDIUM,0)})",
        f"❌ Low ({counts.get(CHANCE_LOW,0)})",
        "📋 Full Table",
    ])
    with t1: cards(high, "No high-chance colleges found.")
    with t2: cards(med,  "No borderline colleges found.")
    with t3: cards(low,  "No low-chance results!")
    with t4:
        out = res.rename(columns={"Cutoff": f"{caste} Cutoff", "Your_Rank": "Your Rank"})
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(), "ecet_results.csv", "text/csv")

    st.markdown("> ⚠️ **Disclaimer:** Cutoff ranks are illustrative. Verify with the official **TSCHE / ECET counselling portal**.")
else:
    st.info("👈 Fill in your details and click **Predict Colleges**.")
