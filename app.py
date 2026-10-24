import io
import os
import hmac
import hashlib
import secrets
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------- App configuration ---------------------------
st.set_page_config(page_title="CoverIQ | AI Insurance Intelligence", page_icon="🛡️", layout="wide")
APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("COVERIQ_DB_PATH", str(APP_DIR / "insurewise.db")))
DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
TARGET = "charges"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#05091b;--panel:#10132d;--line:#29234f;--text:#f4f5ff;--muted:#9ca7ce;--mint:#a855f7;--blue:#299dff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 85% 5%,#1b1250 0%,#080b20 38%,#030716 100%);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0e1729;border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:var(--text)!important}
p,li,label{color:#d3dced}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px}
.hero{padding:32px 34px;border:1px solid #2d4562;border-radius:25px;background:linear-gradient(115deg,#17133e,#101a43);margin-bottom:22px}
.brand{color:var(--mint);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:2.5px}
.hero h1{font-size:clamp(34px,4vw,50px);line-height:1.1;margin:13px 0 10px;font-weight:800}
.hero p{max-width:800px;font-size:15px;color:#c0cee1;line-height:1.7;margin:0}
.hero-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.pill{border:1px solid #3a536d;background:#0b172c55;padding:7px 11px;border-radius:100px;color:#dbe7f8;font-size:12px}
.kicker{color:var(--mint);text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;margin-bottom:5px}
.section-title{font-size:25px;font-weight:800;margin:0 0 5px}
.sub{color:var(--muted);font-size:13px;margin-bottom:17px}
.metric-card{min-height:125px;border:1px solid var(--line);border-radius:17px;padding:18px;background:linear-gradient(145deg,#171632,#0b1025)}
.metric-label{color:var(--muted);font-size:12px}.metric-value{font-family:'Manrope',sans-serif;font-size:clamp(20px,2vw,27px);font-weight:800;color:#f7faff;margin-top:12px;overflow-wrap:anywhere}.metric-note{color:var(--mint);font-size:11px;margin-top:4px}
.panel{background:linear-gradient(145deg,#171632,#0b1025);border:1px solid var(--line);border-radius:18px;padding:20px}
.result{border:1px solid #6540a5;border-radius:21px;padding:24px;background:radial-gradient(circle at 90% 0%,#a855f71c,transparent 40%),linear-gradient(135deg,#171b43,#111b2d)}
.result-label{color:#c4a7ff;font-size:12px;text-transform:uppercase;letter-spacing:1.7px;font-weight:800}
.result-number{font-family:'Manrope',sans-serif;color:var(--mint);font-size:clamp(32px,4vw,46px);font-weight:800;margin:8px 0}
.result-meta{color:#c4c8ed;font-size:13px}
.notice{background:#191d42;border:1px solid #4b3478;border-left:4px solid var(--mint);border-radius:11px;padding:13px 15px;color:#d7e6f4;font-size:13px}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:700;border:1px solid #35516c}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#a855f7,#299dff);color:#07131c;border:0}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#10192b;border:1px solid var(--line);padding:7px;border-radius:14px}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:9px 12px}
.stTabs [aria-selected="true"]{background:#30205b!important;color:var(--mint)!important}
hr{border-color:var(--line)}

/* CoverIQ motion system: subtle, accessible, responsive */
@keyframes cqFadeUp { from {opacity:0; transform:translateY(16px)} to {opacity:1; transform:translateY(0)} }
@keyframes cqGlow { 0%,100% {filter:drop-shadow(0 0 10px #a855f733)} 50% {filter:drop-shadow(0 0 22px #299dff66)} }
@keyframes cqFloat { 0%,100% {transform:translateY(0)} 50% {transform:translateY(-7px)} }
@keyframes cqPulse { 0%,100% {box-shadow:0 0 0 0 #a855f722} 50% {box-shadow:0 0 0 7px #a855f700} }
.landing-nav,.landing-copy,.landing-visual,.landing-benefits>div,.auth-card-head,.metric-card,.panel,.result {
  animation:cqFadeUp .65s ease both;
}
.landing-benefits>div:nth-child(2){animation-delay:.08s}
.landing-benefits>div:nth-child(3){animation-delay:.16s}
.visual-card {transition:transform .35s ease,border-color .35s ease,box-shadow .35s ease}
.visual-card:hover {transform:rotate(0deg) translateY(-5px);border-color:#8b5cf6;box-shadow:0 28px 85px #5423b344}
.visual-glow {background:radial-gradient(circle,#a855f72b 0%,#299dff1c 42%,transparent 72%);animation:cqPulse 4s ease-in-out infinite}
.orbit-one {animation:cqFloat 7s ease-in-out infinite}
.orbit-two {animation:cqFloat 9s ease-in-out infinite reverse}
.float-chip {transition:transform .25s ease,border-color .25s ease;backdrop-filter:blur(12px)}
.float-chip:hover {transform:translateY(-4px) scale(1.02);border-color:#8b5cf6}
.landing-brand svg,.auth-logo svg {animation:cqGlow 4s ease-in-out infinite}
.stButton>button,.stDownloadButton>button {
  transition:transform .18s ease,box-shadow .25s ease,border-color .25s ease,filter .25s ease!important;
  box-shadow:0 4px 16px #0002;
}
.stButton>button:hover,.stDownloadButton>button:hover {
  transform:translateY(-2px);border-color:#8b5cf6!important;
  box-shadow:0 10px 28px #7c3aed30!important;filter:brightness(1.08);
}
.stButton>button:active,.stDownloadButton>button:active {transform:translateY(0) scale(.99)}
.metric-card,.landing-benefits>div,.panel {transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease}
.metric-card:hover,.landing-benefits>div:hover,.panel:hover {transform:translateY(-3px);border-color:#6540a5;box-shadow:0 12px 35px #160c352c}
.stTabs [data-baseweb="tab"] {transition:background .2s ease,color .2s ease,transform .2s ease}
.stTabs [data-baseweb="tab"]:hover {transform:translateY(-1px)}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#080b1c 0%,#11102c 55%,#0a1125 100%);border-right:1px solid #2a2354}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
  background:linear-gradient(100deg,#a855f7,#299dff)!important;color:white!important;border:0!important;
}
.stButton>button[kind="primary"] {
  background:linear-gradient(105deg,#a855f7 0%,#6545f5 52%,#299dff 100%)!important;color:white!important;
}
.stProgress > div > div {background:linear-gradient(90deg,#a855f7,#299dff)!important}
@media (prefers-reduced-motion: reduce) {
  *,*::before,*::after {animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important}
}



/* Landing page + custom CoverIQ logo */
.landing-nav{display:flex;align-items:center;justify-content:space-between;padding:8px 0 28px;border-bottom:1px solid #22334a;margin-bottom:28px;gap:18px}
.landing-brand{display:flex;align-items:center;gap:12px}.landing-brand svg,.auth-logo svg{width:52px;height:52px;filter:drop-shadow(0 8px 22px #a855f725)}
.landing-name{font-family:'Manrope',sans-serif;font-size:23px;font-weight:800;letter-spacing:-1px;color:#f4f8ff}.landing-name span{font-size:10px;color:#a855f7;vertical-align:top;margin-left:3px}.landing-tag{font-size:9px;letter-spacing:2.5px;color:#8298b7;font-weight:800;margin-top:1px}
.landing-nav-note{border:1px solid #263b55;background:#101a2c;padding:10px 15px;border-radius:99px;font-size:11px;color:#a6b8d0}.landing-nav-note i{display:inline-block;width:7px;height:7px;background:#a855f7;border-radius:50%;margin-right:8px;box-shadow:0 0 10px #a855f7}
.landing-hero{display:grid;grid-template-columns:1.02fr .98fr;gap:25px;align-items:center;min-height:385px;padding:25px 0 28px}.landing-copy{padding:14px 0}.landing-eyebrow{font-size:10px;letter-spacing:2px;font-weight:800;color:#a855f7}.landing-eyebrow span{font-size:15px;margin-right:7px}.landing-copy h1{font-size:clamp(42px,5.4vw,70px)!important;line-height:1.04!important;letter-spacing:-3.7px;margin:23px 0 20px!important}.landing-copy h1 span{color:#a855f7!important}.landing-description{font-size:15px;line-height:1.85;color:#a9bad2;max-width:510px}.landing-actions-hint{display:flex;align-items:center;flex-wrap:wrap;gap:8px 13px;color:#aebed4;font-size:10px;margin-top:25px}.landing-check{color:#a855f7;font-size:14px}
.landing-visual{min-height:335px;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}.orbit{position:absolute;border:1px solid #274b5a;border-radius:50%;transform:rotate(-22deg)}.orbit-one{width:340px;height:340px}.orbit-two{width:280px;height:280px;border-color:#273a58}.visual-glow{position:absolute;width:260px;height:260px;border-radius:50%;background:#7c3aed1a;filter:blur(55px)}.visual-card{position:relative;z-index:2;width:min(100%,390px);padding:23px;border:1px solid #314962;border-radius:22px;background:linear-gradient(145deg,#17273c,#0e192a);box-shadow:0 25px 90px #0005;transform:rotate(-2deg)}.visual-card-top{display:flex;align-items:center;gap:8px;color:#8298b5;font-size:8px;letter-spacing:1.4px;font-weight:800}.mini-shield{width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:#a855f71c;color:#a855f7;font-size:14px}.live-pill{margin-left:auto;border:1px solid #6540a5;color:#a855f7;padding:5px 7px;border-radius:99px;font-size:7px;letter-spacing:.7px}.visual-caption{color:#92a6c2;font-size:9px;letter-spacing:1.8px;margin-top:27px}.visual-value{font-family:'Manrope',sans-serif;font-size:29px;font-weight:800;color:#f3f7ff;margin-top:7px}.visual-value span{color:#a855f7}.visual-chart{height:100px;display:flex;align-items:end;gap:8px;padding:15px 0 8px;border-bottom:1px solid #2a3c52}.visual-chart span{flex:1;border-radius:5px 5px 2px 2px;background:linear-gradient(180deg,#a855f7,#6e9fff);opacity:.82}.visual-card-bottom{display:flex;justify-content:space-between;gap:10px;color:#8398b5;font-size:9px;margin-top:16px}.visual-card-bottom b{color:#a855f7;font-weight:700}.float-chip{position:absolute;z-index:3;background:#14243a;border:1px solid #354b64;border-radius:12px;padding:11px 13px;box-shadow:0 12px 35px #0004;color:#c8d6e9;font-size:10px}.float-chip span{color:#a855f7;margin-right:7px;font-size:14px}.chip-top{top:21px;right:0}.chip-bottom{bottom:19px;left:0}
.landing-benefits{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:20px 0 18px}.landing-benefits>div{border:1px solid #263952;border-radius:17px;background:linear-gradient(145deg,#121e31,#0e1728);padding:18px 19px;min-height:145px}.benefit-icon{width:32px;height:32px;display:grid;place-items:center;border-radius:10px;background:#a855f716;color:#a855f7;font-size:18px;margin-bottom:13px}.landing-benefits b{font-size:13px;color:#edf3ff}.landing-benefits p{font-size:11px;line-height:1.65;color:#91a5c1;margin:7px 0 0}.landing-disclaimer{text-align:center;color:#647c9d;font-size:9px;letter-spacing:1.5px;margin:24px 0 13px}.landing-disclaimer span{color:#a855f7;margin:0 8px}.landing-footnote{text-align:center;color:#7288a8;font-size:10px;margin-top:20px}.landing-footnote span{color:#a855f7}.auth-logo{display:flex;align-items:center}.auth-topbar{align-items:center}.auth-card-head{padding-top:22px}.auth-card-head h2{font-size:29px!important}.auth-footer{margin-top:25px}
@media(max-width:850px){.landing-nav{align-items:flex-start}.landing-nav-note{display:none}.landing-hero{grid-template-columns:1fr;gap:4px;padding-top:5px}.landing-copy h1{font-size:47px!important;letter-spacing:-2.5px}.landing-visual{min-height:300px}.landing-benefits{grid-template-columns:1fr}.landing-benefits>div{min-height:0}.landing-disclaimer{line-height:1.8}.chip-top{right:2px}.chip-bottom{left:2px}}

/* Authentication experience */
.auth-topbar{display:flex;align-items:center;gap:12px;padding:5px 2px 24px;border-bottom:1px solid #24334b;margin-bottom:24px}
.auth-brand-mark{width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#a855f7,#299dff);color:#07131c;font-size:25px;font-weight:900;box-shadow:0 8px 30px #a855f725}
.auth-brand-name{font-family:'Manrope',sans-serif;font-weight:800;font-size:19px;letter-spacing:-.5px;color:#f6f8ff}
.auth-brand-sub{font-size:9px;letter-spacing:1.6px;color:#8fa2bf;font-weight:700;margin-top:1px}
.auth-top-right{margin-left:auto;color:#b6c6da;font-size:12px;border:1px solid #293b55;background:#101a2c;padding:9px 13px;border-radius:99px}
.status-dot{display:inline-block;width:7px;height:7px;background:#a855f7;border-radius:50%;margin-right:7px;box-shadow:0 0 10px #a855f7}
.auth-story{padding:38px 24px 24px 8px;min-height:510px}
.auth-eyebrow,.auth-card-kicker{font-size:10px;letter-spacing:2.1px;font-weight:800;color:#a855f7}
.auth-story h1{font-size:clamp(40px,5vw,62px)!important;line-height:1.06!important;letter-spacing:-2.8px;margin:22px 0 18px!important}
.auth-story h1 span{background:linear-gradient(90deg,#a855f7,#8bb5ff);-webkit-background-clip:text;background-clip:text;color:transparent!important}
.auth-story-copy{font-size:15px;line-height:1.8;color:#aebed5;max-width:520px;margin-bottom:30px}
.auth-feature{display:flex;align-items:center;gap:14px;margin:19px 0}
.auth-feature-icon{height:43px;width:43px;flex:0 0 43px;border-radius:13px;border:1px solid #314663;background:linear-gradient(145deg,#172b42,#111b2d);display:flex;align-items:center;justify-content:center;font-size:21px;color:#a855f7}
.auth-feature b{display:block;color:#f0f5ff;font-size:13px;margin-bottom:4px}
.auth-feature small{display:block;color:#91a4c0;font-size:12px;line-height:1.5}
.auth-trust{margin-top:33px;padding-top:18px;border-top:1px solid #263650;color:#8297b5;font-size:11px;letter-spacing:.2px}
.auth-trust span{color:#a855f7;margin-right:7px}
.auth-card-head{padding:25px 25px 10px}
.auth-card-head h2{font-size:26px!important;letter-spacing:-.7px;margin:9px 0 6px!important}
.auth-card-head p{font-size:13px;color:#94a8c5;margin:0 0 12px}
div[data-testid="stVerticalBlock"]:has(> div .auth-card-head){background:linear-gradient(155deg,#131f34,#0e1728);border:1px solid #2a3b56;border-radius:25px;padding:14px 18px 20px;box-shadow:0 22px 70px #00000025}
div[role="radiogroup"]{gap:7px!important;background:#0a1221;border:1px solid #273953;border-radius:13px;padding:5px!important;margin:0 25px 18px!important}
div[role="radiogroup"] label{flex:1;justify-content:center;background:transparent;border-radius:9px;padding:8px 5px!important;color:#9fb0c9!important;font-size:12px!important}
div[role="radiogroup"] label:has(input:checked){background:#30205b;color:#a855f7!important}
div[data-testid="stForm"]{padding:0 25px!important;border:0!important}
div[data-testid="stForm"] label{font-size:12px!important;color:#bdcbe0!important;font-weight:600!important}
div[data-testid="stForm"] input{background:#0b1424!important;border:1px solid #2b3d58!important;border-radius:11px!important;color:#f3f7ff!important;min-height:44px!important}
div[data-testid="stForm"] input:focus{border-color:#a855f7!important;box-shadow:0 0 0 2px #a855f720!important}
div[data-testid="stForm"] button[kind="primary"]{min-height:48px;border-radius:12px;background:linear-gradient(100deg,#a855f7,#299dff);color:#07131c;border:0;font-weight:800;letter-spacing:.1px;margin-top:9px}
.auth-form-foot{font-size:11px;line-height:1.7;color:#8297b5;padding:15px 25px 5px}
.auth-form-foot b{color:#c4d4e8}
.auth-password-note{font-size:10px;color:#87bda9;padding:0 0 8px}
.auth-footer{text-align:center;border-top:1px solid #22324a;padding:20px 5px 0;margin-top:30px;color:#6f84a3;font-size:9px;letter-spacing:1.6px}
.auth-footer span{color:#a855f7;margin:0 7px}
@media(max-width:850px){.auth-story{padding:8px 4px 25px;min-height:0}.auth-story h1{font-size:43px!important}.auth-top-right{font-size:10px;padding:8px}.auth-card-head{padding:18px 12px 8px}div[data-testid="stForm"]{padding:0 12px!important}div[role="radiogroup"]{margin:0 12px 15px!important}.auth-form-foot{padding-left:12px;padding-right:12px}}

</style>
""", unsafe_allow_html=True)


# --------------------------- Database / accounts ---------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            bmi REAL NOT NULL,
            children INTEGER NOT NULL,
            smoker TEXT NOT NULL,
            region TEXT NOT NULL,
            model_name TEXT NOT NULL,
            annual_estimate REAL NOT NULL,
            monthly_estimate REAL NOT NULL,
            baseline_estimate REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        conn.commit()


def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return salt.hex(), digest.hex()


def create_user(username, password):
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Use a password with at least 8 characters."
    salt, digest = hash_password(password)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users(username,salt,password_hash,created_at) VALUES(?,?,?,?)",
                (username, salt, digest, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
        return True, "Account created. You can sign in now."
    except sqlite3.IntegrityError:
        return False, "That username already exists."


def authenticate(username, password):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if not row:
        return None
    _, digest = hash_password(password, row["salt"])
    if hmac.compare_digest(digest, row["password_hash"]):
        return {"id": row["id"], "username": row["username"]}
    return None


def save_prediction(user_id, profile, model_name, annual, baseline):
    with get_connection() as conn:
        conn.execute("""INSERT INTO predictions(
            user_id,created_at,age,sex,bmi,children,smoker,region,model_name,
            annual_estimate,monthly_estimate,baseline_estimate
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (
            user_id, datetime.now().isoformat(timespec="seconds"),
            int(profile["age"]), str(profile["sex"]), float(profile["bmi"]),
            int(profile["children"]), str(profile["smoker"]), str(profile["region"]),
            model_name, float(annual), float(annual / 12), float(baseline),
        ))
        conn.commit()


def load_history(user_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT created_at,age,sex,bmi,children,smoker,region,model_name,annual_estimate,monthly_estimate,baseline_estimate "
            "FROM predictions WHERE user_id=? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


init_db()


# --------------------------- Data / model pipeline ---------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    for path in [APP_DIR / "data" / "insurance.csv", APP_DIR / "insurance.csv", Path("data/insurance.csv"), Path("insurance.csv")]:
        if path.exists():
            try:
                data = pd.read_csv(path)
                if set(COLUMNS).issubset(data.columns):
                    return data[COLUMNS].copy(), f"Local dataset: {path.name}"
            except Exception:
                pass
    try:
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            data = pd.read_csv(response)
        if set(COLUMNS).issubset(data.columns):
            return data[COLUMNS].copy(), "Public insurance sample dataset"
    except Exception as exc:
        return None, str(exc)
    return None, "Dataset could not be loaded."


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(data_json, test_size, seed):
    data = pd.read_json(io.StringIO(data_json), orient="split")
    X, y = data[NUMERIC + CATEGORICAL], data[TARGET]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    category_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, NUMERIC), ("categorical", category_pipe, CATEGORICAL)])
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=.05, max_depth=2, random_state=seed),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    models, predictions, rows = {}, {}, []
    for name, estimator in estimators.items():
        model = Pipeline([("preprocessor", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name], predictions[name] = model, pred
        try:
            cv = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
        except Exception:
            cv = float("nan")
        rows.append({"Model": name, "R² Score": r2_score(y_test, pred),
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "CV R²": cv})
    results = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    return {"models": models, "predictions": predictions, "results": results,
            "best_name": str(results.iloc[0]["Model"]), "X_test": X_test, "y_test": y_test}


def money(value):
    return f"${float(value):,.2f}"


def styled_chart(fig):
    fig.patch.set_facecolor("#111a2e")
    for ax in fig.axes:
        ax.set_facecolor("#111a2e")
        ax.tick_params(colors="#b9c8dd")
        ax.xaxis.label.set_color("#b9c8dd")
        ax.yaxis.label.set_color("#b9c8dd")
        ax.title.set_color("#f5f8ff")
        for spine in ax.spines.values():
            spine.set_color("#293a54")
    fig.tight_layout()
    return fig


def build_explanation(model, profile, df):
    """One-feature-at-a-time sensitivity against a typical profile; not causal attribution."""
    baseline_profile = {
        "age": int(round(df["age"].median())),
        "sex": str(df["sex"].mode().iloc[0]),
        "bmi": float(df["bmi"].median()),
        "children": int(round(df["children"].median())),
        "smoker": str(df["smoker"].mode().iloc[0]),
        "region": str(df["region"].mode().iloc[0]),
    }
    baseline_df = pd.DataFrame([baseline_profile])
    baseline_pred = max(0.0, float(model.predict(baseline_df)[0]))
    profile_df = pd.DataFrame([profile])
    profile_pred = max(0.0, float(model.predict(profile_df)[0]))
    rows = []
    for feature in ["age", "bmi", "children", "sex", "smoker", "region"]:
        altered = dict(baseline_profile)
        altered[feature] = profile[feature]
        altered_pred = max(0.0, float(model.predict(pd.DataFrame([altered]))[0]))
        rows.append({
            "Feature": feature.replace("_", " ").title(),
            "Your value": str(profile[feature]),
            "Estimate if this were the only changed feature": altered_pred,
            "Difference from typical profile": altered_pred - baseline_pred,
        })
    explanation = pd.DataFrame(rows)
    explanation["Absolute difference"] = explanation["Difference from typical profile"].abs()
    explanation = explanation.sort_values("Absolute difference", ascending=False).drop(columns="Absolute difference")
    return baseline_profile, baseline_pred, profile_pred, explanation


def create_pdf_report(username, profile, model_name, annual, monthly, baseline, explanation, results, created_at):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], textColor=colors.HexColor("#146b5a"),
                              fontName="Helvetica-Bold", fontSize=24, leading=28, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=colors.HexColor("#536174"),
                              fontSize=9, leading=13))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#146b5a"),
                              fontSize=14, leading=18, spaceBefore=14, spaceAfter=7))
    story = [
        Paragraph("CoverIQ", styles["Brand"]),
        Paragraph("Insurance Cost Intelligence · Personal estimate report", styles["Subtle"]),
        Spacer(1, 14),
        Paragraph("Estimated annual charge: " + money(annual), styles["Heading1"]),
        Paragraph("Monthly equivalent: " + money(monthly), styles["Heading2"]),
        Paragraph(f"Generated for: {username} &nbsp;&nbsp; | &nbsp;&nbsp; {created_at}", styles["Subtle"]),
        Spacer(1, 8),
        Paragraph("Profile used", styles["Section"]),
    ]
    profile_rows = [["Input", "Value"], ["Age", str(profile["age"])], ["Sex", str(profile["sex"])],
                    ["BMI", f'{float(profile["bmi"]):.1f}'], ["Children", str(profile["children"])],
                    ["Smoking status", str(profile["smoker"])], ["Region", str(profile["region"])],
                    ["Model", model_name]]
    table = Table(profile_rows, colWidths=[2.2*inch, 4.3*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#146b5a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d7e0e8")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f4f8f8")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story += [table, Paragraph("Beginner-friendly explanation", styles["Section"]),
              Paragraph(f"A typical profile from the dataset produces an estimate of {money(baseline)}. "
                        f"Your entered profile produces {money(annual)}. The difference is "
                        f"{money(annual - baseline)} compared with that typical profile.", styles["BodyText"]),
              Spacer(1, 6)]
    explain_rows = [["Feature", "Your value", "Sensitivity estimate", "Difference vs typical"]]
    for _, row in explanation.iterrows():
        explain_rows.append([row["Feature"], str(row["Your value"]),
                             money(row["Estimate if this were the only changed feature"]),
                             money(row["Difference from typical profile"])])
    ex_table = Table(explain_rows, colWidths=[1.25*inch, 0.85*inch, 2.15*inch, 2.1*inch], repeatRows=1)
    ex_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24364f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d7e0e8")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f8f8")]),
    ]))
    story += [ex_table, Paragraph("Model evaluation", styles["Section"])]
    metric_rows = [["Model", "R²", "MAE", "RMSE", "CV R²"]]
    for _, row in results.iterrows():
        metric_rows.append([row["Model"], f'{row["R² Score"]:.3f}', money(row["MAE"]),
                            money(row["RMSE"]), f'{row["CV R²"]:.3f}'])
    mt = Table(metric_rows, colWidths=[1.55*inch, .65*inch, 1.4*inch, 1.4*inch, .9*inch], repeatRows=1)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#146b5a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d7e0e8")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f8f8")]),
    ]))
    story += [mt, Spacer(1, 12),
              Paragraph("<b>Important limitations:</b> This is an educational estimate based on a public sample dataset, "
                        "not an insurance quote or financial/medical advice. The feature comparison changes one input "
                        "at a time from a typical profile; it is a sensitivity analysis, not proof of causation. "
                        "Real pricing depends on policy details and factors not represented here.", styles["Subtle"])]
    doc.build(story)
    return output.getvalue()


# --------------------------- Welcome / sign-in / sign-up ---------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "Welcome"
if "auth_choice" not in st.session_state:
    st.session_state.auth_choice = (
        "Sign in" if st.session_state.auth_view == "Sign in" else "Create account"
    )

# Apply this switch before the radio widget is created on the next rerun.
# Streamlit does not allow changing a widget's state after that widget has
# already been instantiated during the same run.
if st.session_state.pop("_coveriq_switch_to_signin", False):
    st.session_state.auth_view = "Sign in"
    st.session_state.auth_choice = "Sign in"

if st.session_state.user is None:
    # Inline SVG logo: no external image or extra dependency is needed.
    logo_svg = '''<svg width="58" height="58" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="g" x1="8" y1="4" x2="57" y2="61" gradientUnits="userSpaceOnUse"><stop stop-color="#a855f7"/><stop offset="1" stop-color="#299dff"/></linearGradient></defs><rect x="2" y="2" width="60" height="60" rx="20" fill="url(#g)"/><path d="M32 10L49 17V29C49 40 42 48 32 53C22 48 15 40 15 29V17L32 10Z" fill="#0A1727"/><path d="M22 31.5L28.5 38L42.5 23.5" stroke="#a855f7" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="46" cy="17" r="3" fill="#F4FFF9"/></svg>'''

    if st.session_state.auth_view == "Welcome":
        st.markdown(f'''
        <div class="landing-nav">
          <div class="landing-brand">{logo_svg}<div><div class="landing-name">CoverIQ<span>®</span></div><div class="landing-tag">COST INTELLIGENCE</div></div></div>
          <div class="landing-nav-note"><i></i> AI-powered estimates · Your personal workspace</div>
        </div>
        <div class="landing-hero">
          <div class="landing-copy">
            <div class="landing-eyebrow"><span>✦</span> A SMARTER WAY TO SEE THE BIGGER PICTURE</div>
            <h1>Make sense of your<br>healthcare costs<span>.</span></h1>
            <p class="landing-description">Explore personalized cost estimates, understand the factors behind them, and turn complex numbers into clearer next steps.</p>
            <div class="landing-actions-hint"><span class="landing-check">✓</span> Machine-learning insights <span class="landing-check">✓</span> Personal reports <span class="landing-check">✓</span> Saved history</div>
          </div>
          <div class="landing-visual">
            <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
            <div class="visual-glow"></div>
            <div class="visual-card visual-card-main"><div class="visual-card-top"><span class="mini-shield">✦</span><span>COVERIQ INSIGHTS</span><span class="live-pill">● LIVE DEMO</span></div><div class="visual-caption">YOUR COST OUTLOOK</div><div class="visual-value">Clarity <span>ahead.</span></div><div class="visual-chart"><span style="height:27%"></span><span style="height:42%"></span><span style="height:34%"></span><span style="height:58%"></span><span style="height:49%"></span><span style="height:76%"></span><span style="height:91%"></span><span style="height:66%"></span><span style="height:83%"></span><span style="height:100%"></span><span style="height:75%"></span><span style="height:88%"></span></div><div class="visual-card-bottom"><span>Model-powered estimates</span><b>Explore your profile ↗</b></div></div>
            <div class="float-chip chip-top"><span>✧</span> Understand your estimate</div><div class="float-chip chip-bottom"><span>↗</span> Insights that make sense</div>
          </div>
        </div>
        <div class="landing-benefits">
          <div><div class="benefit-icon">◈</div><b>Predict with context</b><p>Get an estimate based on profile details and a trained ML model.</p></div>
          <div><div class="benefit-icon">⌁</div><b>Explore what matters</b><p>Compare inputs and see how your estimate responds.</p></div>
          <div><div class="benefit-icon">▤</div><b>Keep it close</b><p>Save your history and download a shareable PDF report.</p></div>
        </div>
        <div class="landing-disclaimer">BUILT FOR EXPLORATION <span>•</span> EDUCATIONAL ESTIMATES, NOT INSURANCE QUOTES</div>
        ''', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.1, 1.05, 1.8], gap="small")
        with c1:
            if st.button("Get started  →", type="primary", use_container_width=True, key="landing_start"):
                st.session_state.auth_view = "Create account"
                st.session_state.auth_choice = "Create account"
                st.rerun()
        with c2:
            if st.button("I have an account", use_container_width=True, key="landing_signin"):
                st.session_state.auth_view = "Sign in"
                st.session_state.auth_choice = "Sign in"
                st.rerun()
        st.markdown('<div class="landing-footnote">Your next step starts with a clearer picture. <span>CoverIQ</span></div>', unsafe_allow_html=True)
        st.stop()

    # Auth screen is a separate view, entered only after a landing-page action.
    st.markdown(f'''
    <div class="auth-topbar">
      <div class="auth-logo">{logo_svg}</div>
      <div><div class="auth-brand-name">CoverIQ</div><div class="auth-brand-sub">COST INTELLIGENCE PLATFORM</div></div>
      <div class="auth-top-right"><span class="status-dot"></span> Private workspace</div>
    </div>
    ''', unsafe_allow_html=True)
    back_col, spacer = st.columns([1, 5])
    with back_col:
        if st.button("← Back to welcome", key="back_welcome"):
            st.session_state.auth_view = "Welcome"
            st.session_state.auth_choice = "Sign in"
            st.rerun()
    left, right = st.columns([1.0, 0.92], gap="large")
    with left:
        st.markdown(f'''
        <div class="auth-story">
          <div class="auth-eyebrow">YOUR DATA. A CLEARER PICTURE.</div>
          <h1>Good decisions<br>start with <span>clarity.</span></h1>
          <p class="auth-story-copy">Your own space to estimate healthcare costs, understand model insights, and keep your reports in one place.</p>
          <div class="auth-feature"><div class="auth-feature-icon">↗</div><div><b>See the estimate</b><small>Explore predictions generated by machine learning.</small></div></div>
          <div class="auth-feature"><div class="auth-feature-icon">⌁</div><div><b>Understand the factors</b><small>Compare profile inputs and model sensitivity.</small></div></div>
          <div class="auth-feature"><div class="auth-feature-icon">▤</div><div><b>Keep your insights</b><small>Save your history and download PDF reports.</small></div></div>
          <div class="auth-trust"><span>✦</span> A focused workspace for your cost exploration.</div>
        </div>''', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-card-head"><div class="auth-card-kicker">WELCOME TO YOUR WORKSPACE</div><h2>Let’s get you in.</h2><p>Sign in or create your free account to continue.</p></div>', unsafe_allow_html=True)
        # Keep the selector synchronized with the requested screen. Navigation
        # buttons update auth_choice before rerunning; registration uses a pending
        # flag so this widget key is never changed after the widget is instantiated.
        auth_choice = st.radio(
            "Choose access",
            ["Sign in", "Create account"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_choice",
        )
        st.session_state.auth_view = auth_choice
        if auth_choice == "Sign in":
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username", placeholder="Your username", autocomplete="username")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password", autocomplete="current-password")
                login_submit = st.form_submit_button("Sign in to CoverIQ  →", type="primary", use_container_width=True)
            if login_submit:
                user = authenticate(login_username, login_password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("We couldn't match that username and password. Please try again.")
            st.markdown('<div class="auth-form-foot">First time here? Switch to <b>Create account</b> above to make your workspace.</div>', unsafe_allow_html=True)
        else:
            with st.form("signup_form", clear_on_submit=False):
                new_username = st.text_input("Choose a username", placeholder="At least 3 characters", key="signup_user", autocomplete="username")
                new_password = st.text_input("Create a password", type="password", placeholder="At least 8 characters", key="signup_pw", autocomplete="new-password")
                confirm_password = st.text_input("Confirm password", type="password", placeholder="Enter it again", key="signup_confirm", autocomplete="new-password")
                st.markdown('<div class="auth-password-note">✓ 8+ characters &nbsp; · &nbsp; ✓ Password confirmation</div>', unsafe_allow_html=True)
                signup_submit = st.form_submit_button("Create my account  →", type="primary", use_container_width=True)
            if signup_submit:
                if new_password != confirm_password:
                    st.error("Those passwords don't match yet. Please check them and try again.")
                else:
                    ok, message = create_user(new_username, new_password)
                    if ok:
                        st.session_state.auth_view = "Sign in"
                        st.session_state._coveriq_switch_to_signin = True
                        st.rerun()
                    else:
                        st.error(message)
            st.markdown('<div class="auth-form-foot">Your account keeps your prediction history tied to your workspace.</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-footer">COVERIQ <span>•</span> ESTIMATES ARE EDUCATIONAL, NOT INSURANCE QUOTES</div>', unsafe_allow_html=True)
    st.stop()

user = st.session_state.user

# Prediction controls are hidden until the user clicks "Let's Predict".
if "show_predict_sidebar" not in st.session_state:
    st.session_state.show_predict_sidebar = False

# Defaults keep the dashboard usable before the prediction panel is opened.
test_percent = int(st.session_state.get("test_percent", 20))
seed = int(st.session_state.get("model_seed", 42))
age = int(st.session_state.get("profile_age", 30))
bmi = float(st.session_state.get("profile_bmi", 27.5))
children = int(st.session_state.get("profile_children", 0))
chosen_model = st.session_state.get("chosen_model", "Best-performing model")
predict = False

# --------------------------- Sidebar / profile controls ---------------------------
df, source = load_dataset()
if df is None:
    st.error("Could not load the insurance dataset.")
    st.info("Place `insurance.csv` in this folder or in a `data` subfolder. Required columns: age, sex, bmi, children, smoker, region, charges.")
    st.caption(source)
    st.stop()

df = df[COLUMNS].copy()
for col in ["age", "bmi", "children", "charges"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["age", "bmi", "children", "charges"])
for col in CATEGORICAL:
    df[col] = df[col].astype("string").fillna("unknown").astype(str)
if len(df) < 10:
    st.error("At least 10 valid dataset rows are required.")
    st.stop()

if st.session_state.show_predict_sidebar:
    with st.sidebar:
        st.markdown("## ⚙️ CoverIQ")
        st.caption(f"Signed in as **{user['username']}**")
        if st.button("← Back to dashboard", use_container_width=True, key="hide_predict_panel"):
            st.session_state.show_predict_sidebar = False
            st.rerun()
        if st.button("Sign out", use_container_width=True, key="dashboard_signout"):
            st.session_state.user = None
            st.session_state.show_predict_sidebar = False
            st.rerun()
        st.markdown("---")
        st.markdown("### Model settings")
        test_percent = st.slider("Test set (%)", 10, 40, int(st.session_state.get("test_percent", 20)), 5, key="test_percent")
        seed = st.number_input("Random seed", 1, 9999, int(st.session_state.get("model_seed", 42)), key="model_seed")
        st.markdown("---")
        st.markdown("### 👤 Your profile")
        age = st.slider("Age", 18, 100, int(st.session_state.get("profile_age", 30)), key="profile_age")
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 70.0, float(st.session_state.get("profile_bmi", 27.5)), .1, key="profile_bmi")
        children = st.slider("Number of children", 0, 10, int(st.session_state.get("profile_children", 0)), key="profile_children")
        sex = st.selectbox("Sex", sorted(df["sex"].unique()), key="profile_sex")
        smoker = st.selectbox("Smoking status", sorted(df["smoker"].unique()), key="profile_smoker")
        region = st.selectbox("Region", sorted(df["region"].unique()), key="profile_region")
        chosen_model = st.selectbox("Prediction model", ["Best-performing model", "Linear Regression", "Random Forest", "Gradient Boosting"], key="chosen_model")
        predict = st.button("✦ Let's predict", type="primary", use_container_width=True, key="calculate_estimate")
        st.caption("Educational estimate only; not an actual insurance quote.")

trained = train_models(df.to_json(orient="split"), test_percent / 100, int(seed))
results = trained["results"]
best_name = trained["best_name"]
best_row = results.iloc[0]

st.markdown("""
<div class="hero">
  <div class="brand">🛡️ COVERIQ / COST INTELLIGENCE</div>
  <h1>Know your risk.<br>Understand your costs.</h1>
  <p>Explore insurance patterns, compare predictive models, understand how your profile changes an estimate,
  and save your reports in one workspace.</p>
  <div class="hero-tags"><span class="pill">✦ Machine Learning</span><span class="pill">↗ Predictive Analytics</span><span class="pill">◈ Explainable estimates</span></div>
</div>
""", unsafe_allow_html=True)

# Main call-to-action opens the prediction controls in the sidebar.
cta_col, cta_note = st.columns([1.1, 3.5], gap="medium")
with cta_col:
    if st.button("✦ Let's Predict", type="primary", use_container_width=True, key="main_lets_predict"):
        st.session_state.show_predict_sidebar = True
        st.rerun()
with cta_note:
    st.markdown("<div style='padding:9px 0;color:#aebbd2'>Ready to explore your estimate? Click <b>Let's Predict</b> to open your profile controls.</div>", unsafe_allow_html=True)

st.markdown('<div class="kicker">Analytics overview</div><div class="section-title">The numbers behind the estimate</div><div class="sub">A summary of the cleaned dataset and current model performance.</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
cards = [(c1, "▦", "Records analysed", f"{len(df):,}", "Clean sample records"),
         (c2, "$", "Average annual charge", money(df["charges"].mean()), "Across all records"),
         (c3, "✦", "Top model", best_name, "Highest held-out R²"),
         (c4, "↗", "Test-set R²", f'{best_row["R² Score"]:.3f}', "Closer to 1 is generally better")]
for col, icon, label, value, note in cards:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{icon} &nbsp; {label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

# Prediction action: compute and persist to database.
if predict:
    model_name = best_name if chosen_model == "Best-performing model" else chosen_model
    profile = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
    model = trained["models"][model_name]
    estimate = max(0.0, float(model.predict(pd.DataFrame([profile]))[0]))
    _, baseline, _, explanation = build_explanation(model, profile, df)
    save_prediction(user["id"], profile, model_name, estimate, baseline)
    st.session_state.last_result = {
        "profile": profile, "model_name": model_name, "estimate": estimate,
        "baseline": baseline, "explanation": explanation,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

last = st.session_state.get("last_result")
if last:
    profile, estimate, model_name = last["profile"], last["estimate"], last["model_name"]
    st.markdown('<div class="kicker">Personalized estimate</div><div class="section-title">Your insurance cost snapshot</div><div class="sub">Saved to your account history. Use the PDF button to keep a copy of the result.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(f'<div class="result"><div class="result-label">Estimated annual charge</div><div class="result-number">{money(estimate)}</div><div class="result-meta">Model · <b>{model_name}</b> &nbsp; · &nbsp; {last["created_at"]}</div><div class="result-meta" style="margin-top:8px">Profile: age {profile["age"]}, BMI {profile["bmi"]}, children {profile["children"]}, smoker: {profile["smoker"]}.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Monthly equivalent")
        st.markdown(f'<div class="result-number" style="font-size:34px">{money(estimate/12)}</div>', unsafe_allow_html=True)
        st.caption("Annual estimate divided by 12.")
        st.markdown(f"Typical-profile estimate: **{money(last['baseline'])}**")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="notice">ⓘ Educational estimate from a public sample dataset—not an actual insurance quote. The explanation below is a sensitivity comparison, not a causal explanation.</div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Why did the estimate change?")
    st.write(f"A typical profile based on median numeric values and the most common categories is estimated at **{money(last['baseline'])}**. Your profile is estimated at **{money(estimate)}**. The difference is **{money(estimate-last['baseline'])}**.")
    st.caption("Each row below changes one feature at a time from the typical profile. These comparisons can overlap and should not be added together.")
    explanation_display = last["explanation"].copy()
    explanation_display["Estimate if only this feature changed"] = explanation_display["Estimate if this were the only changed feature"].map(money)
    explanation_display["Difference vs typical profile"] = explanation_display["Difference from typical profile"].map(money)
    st.dataframe(explanation_display[["Feature", "Your value", "Estimate if only this feature changed", "Difference vs typical profile"]], use_container_width=True, hide_index=True)

    pdf_bytes = create_pdf_report(user["username"], profile, model_name, estimate, estimate/12,
                                 last["baseline"], last["explanation"], results, last["created_at"])
    st.download_button("⬇ Download personalized PDF report", data=pdf_bytes,
                       file_name=f"coveriq_report_{user['username']}.pdf", mime="application/pdf",
                       type="primary")

st.markdown(f'<div class="notice">● <b>Workspace:</b> {user["username"]} &nbsp; · &nbsp; <b>Model split:</b> {100-test_percent}% train / {test_percent}% test</div>', unsafe_allow_html=True)

overview, explore, evaluation, features, history_tab = st.tabs([
    "⌂ Overview", "⌕ Explore data", "↗ Model performance", "✧ Explainability", "◷ Saved history"
])

with overview:
    st.markdown("### Insurance patterns")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["charges"], bins=35, kde=True, ax=ax, color="#a855f7")
        ax.set(title="Distribution of annual charges", xlabel="Annual charges ($)", ylabel="Records")
        st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    with b:
        means = df.groupby("smoker")["charges"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        means.plot(kind="bar", ax=ax, color=["#a855f7", "#299dff"][:len(means)])
        ax.set(title="Average charge by smoking status", xlabel="Smoking status", ylabel="Average charges ($)")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    x, y, z = st.columns(3)
    x.metric("Median annual charge", money(df["charges"].median()))
    y.metric("Average age", f'{df["age"].mean():.1f} years')
    z.metric("Average BMI", f'{df["bmi"].mean():.1f}')
    st.info("Observed patterns in this sample do not prove that a feature causes an insurance charge to change.")

with explore:
    st.markdown("### Explore relationships")
    a, b = st.columns(2)
    with a:
        feature = st.selectbox("Numeric feature", ["age", "bmi", "children"])
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x=feature, y="charges", hue="smoker", alpha=.65, ax=ax)
        ax.set_title(f"{feature.title()} vs charges")
        st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    with b:
        category = st.selectbox("Category", ["sex", "smoker", "region", "children"])
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x=category, y="charges", ax=ax, color="#a855f7")
        ax.set_title(f"Charges by {category}")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(df[["age", "bmi", "children", "charges"]].corr(), annot=True, fmt=".2f", cmap="Purples", ax=ax)
    ax.set_title("Numeric correlation matrix")
    st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)

with evaluation:
    st.markdown("### Compare the three models")
    st.caption("R² closer to 1 is generally better; MAE and RMSE closer to 0 indicate smaller errors.")
    st.dataframe(results.style.format({"R² Score": "{:.4f}", "MAE": "${:,.2f}", "RMSE": "${:,.2f}", "CV R²": "{:.4f}"}), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = results.sort_values("R² Score")
    ax.barh(ordered["Model"], ordered["R² Score"], color="#a855f7")
    ax.set(title="Held-out test-set R²", xlabel="R² score")
    st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    model_name = st.selectbox("Actual vs predicted", results["Model"].tolist())
    actual, pred = trained["y_test"], trained["predictions"][model_name]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(actual, pred, alpha=.65, color="#a855f7")
    low, high = min(float(actual.min()), float(np.min(pred))), max(float(actual.max()), float(np.max(pred)))
    ax.plot([low, high], [low, high], "--", color="#ffbf7b")
    ax.set(title=f"Actual vs predicted · {model_name}", xlabel="Actual ($)", ylabel="Predicted ($)")
    st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)

with features:
    st.markdown("### Feature importance")
    st.caption("Model-specific influence, not proof of cause.")
    model_name = st.selectbox("Model for feature analysis", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    pipe = trained["models"][model_name]
    estimator = pipe.named_steps["model"]
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    names = [n.replace("numeric__", "").replace("categorical__", "") for n in names]
    if hasattr(estimator, "feature_importances_"):
        vals, label = estimator.feature_importances_, "Importance"
    else:
        vals, label = estimator.coef_, "Coefficient"
    imp = pd.DataFrame({"Feature": names, label: vals})
    imp["Magnitude"] = imp[label].abs()
    imp = imp.sort_values("Magnitude", ascending=False).head(15)
    ordered = imp.sort_values("Magnitude")
    colors_list = ["#a855f7" if v >= 0 else "#ffbf7b" for v in ordered[label]] if label == "Coefficient" else "#a855f7"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["Feature"], ordered[label], color=colors_list)
    ax.set(title=f"{label} · {model_name}", xlabel=label)
    st.pyplot(styled_chart(fig), use_container_width=True); plt.close(fig)
    st.dataframe(imp.drop(columns="Magnitude"), use_container_width=True, hide_index=True)

with history_tab:
    st.markdown("### Saved prediction history")
    history = load_history(user["id"])
    if not history.empty:
        display = history.rename(columns={
            "created_at": "Created at", "age": "Age", "sex": "Sex", "bmi": "BMI",
            "children": "Children", "smoker": "Smoker", "region": "Region", "model_name": "Model",
            "annual_estimate": "Annual estimate ($)", "monthly_estimate": "Monthly estimate ($)",
            "baseline_estimate": "Typical-profile estimate ($)",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Download my history (CSV)", display.to_csv(index=False).encode("utf-8"),
                           "coveriq_prediction_history.csv", "text/csv")
        if st.button("Delete my prediction history"):
            with get_connection() as conn:
                conn.execute("DELETE FROM predictions WHERE user_id=?", (user["id"],))
                conn.commit()
            st.session_state.pop("last_result", None)
            st.rerun()
    else:
        st.info("No saved predictions yet. Use the profile controls and click Calculate my estimate.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#8292ad;font-size:12px;padding:8px 0">COVERIQ · MACHINE LEARNING PROJECT<br>Educational estimates only. Not an insurance quote.</div>', unsafe_allow_html=True)
