import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

import base64

def img_to_base64(img_path: str) -> str:
    p = Path(img_path)
    return base64.b64encode(p.read_bytes()).decode("utf-8")


# =========================================================
# WAJIB PALING ATAS (set_page_config harus sebelum st.* lain)
# =========================================================
st.set_page_config(page_title="Q Coffee – LSTM + GWO", layout="wide")

# =========================
# PATH DATA DI GOOGLE DRIVE
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SALES_FILE  = DATA_DIR / "data_penjualan_q_coffee.xlsx"
RECIPE_FILE = DATA_DIR / "Resep_Q_Coffee.xlsx"
STOCK_FILE  = DATA_DIR / "Stok_Bahan_Baku_Awal.xlsx"

SALES_SHEET  = "PENJUALAN JUL 22 JUN 25"
RECIPE_SHEET = "Sheet1"
STOCK_SHEET  = "Sheet1"

DEFAULT_PRODUCT = "Q Special Black"

# =========================
# CSS ringan untuk sidebar
# =========================
st.markdown("""
<style>
section[data-testid="stSidebar"] > div { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
.brand { font-size: 20px; font-weight: 900; margin: 0 0 8px 0; }
.card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px; background: #ffffff; margin-bottom: 12px; }
.card h4 { margin: 0 0 8px 0; font-size: 14px; font-weight: 900; opacity: .85; }
.kv { display: grid; grid-template-columns: 90px 1fr; row-gap: 6px; column-gap: 10px; font-size: 13px; }
.k { opacity: .65; }
.v { font-weight: 700; }
.stButton > button { width: 100%; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

LOGIN_TOKEN = "3359ab54dece32"  # token/password untuk login

LOGIN_CSS_TMPL = """
<style>
/* Logo fixed di pojok kiri atas */
.login-logo {
  position: fixed;
  top: 180px;
  left: 16px;
  z-index: 9999;

  display: flex;
  align-items: center;
}

.login-logo img {
  height: 34px;      /* atur ukuran logo */
  width: auto;
  display: block;
}

.login-logo .txt {
  color: rgba(255,255,255,.92);
  font-weight: 900;
  font-size: 13px;
  line-height: 1;
}

/* Hide sidebar only on login */
section[data-testid="stSidebar"] { display: none; }

/* Batasi lebar halaman */
div.block-container {
  max-width: 720px !important;
  padding-top: 2.2rem !important;
  padding-bottom: 2rem !important;
  margin: 0 auto !important;
}

/* Background image + overlay gelap */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  height: 100% !important;
}

[data-testid="stAppViewContainer"], .stApp, [data-testid="stApp"] {
  background:
    linear-gradient(135deg, rgba(0,0,0,.62) 0%, rgba(0,0,0,.55) 50%, rgba(0,0,0,.62) 100%),
    url("data:image/jpeg;base64,__BG__") !important;
  background-size: cover !important;
  background-position: center !important;
  background-attachment: fixed !important;
}

/* Biar main container transparan (nggak nutup background) */
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > section,
section.main {
  background: transparent !important;
}

/* Hero */
.hero {
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 16px;
  padding: 14px;
  background: rgba(255,255,255,.10);
  backdrop-filter: blur(10px);
  margin-bottom: 12px;
}

.badge {
  width: 40px; height: 40px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  font-size: 18px;
  background: linear-gradient(135deg, rgba(245,158,11,.95), rgba(59,130,246,.95));
  color: #fff;
  box-shadow: 0 10px 22px rgba(0,0,0,.35);
}

.brand-title { font-size: 20px; font-weight: 900; color: rgba(255,255,255,.95); margin: 0; }
.brand-sub   { font-size: 12px; color: rgba(255,255,255,.8); margin: 2px 0 0 0; }

/* Form jadi card putih solid */
div[data-testid="stForm"] {
  background: rgba(255,255,255,.92) !important;
  border-radius: 16px !important;
  padding: 16px !important;
  border: 1px solid rgba(0,0,0,.10) !important;
  box-shadow: 0 14px 32px rgba(0,0,0,.25) !important;
}

/* Label hitam */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p,
div[data-testid="stTextInput"] label span {
  color: #111827 !important;
  font-weight: 800 !important;
}

/* Input putih + teks hitam */
div[data-testid="stTextInput"] input {
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid rgba(0,0,0,.15) !important;
  border-radius: 12px !important;
}

div[data-testid="stTextInput"] input::placeholder {
  color: rgba(0,0,0,.45) !important;
}

/* Button */
.stButton > button {
  width: 100%;
  border-radius: 12px;
  font-weight: 900;
  background: linear-gradient(135deg, rgba(245,158,11,.95), rgba(59,130,246,.95)) !important;
  color: white !important;
  border: 0 !important;
  padding: 0.55rem 0.8rem !important;
}
</style>
"""



# =========================================================
# Helper: normalisasi nama kolom supaya lebih tahan typo/spasi
# =========================================================
def _norm(s: str) -> str:
    return "".join(str(s).strip().lower().split())

def ensure_cols(df: pd.DataFrame, mapping: dict):
    """
    mapping: {"canonical_name": ["alt1","alt2",...]}
    """
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    rename = {}
    for canon, alts in mapping.items():
        found = None
        # cek canonical juga
        if _norm(canon) in norm_map:
            found = norm_map[_norm(canon)]
        else:
            for a in alts:
                if _norm(a) in norm_map:
                    found = norm_map[_norm(a)]
                    break
        if found is None:
            raise ValueError(f"Kolom wajib tidak ditemukan: '{canon}'. Kolom terbaca: {cols}")
        rename[found] = canon
    return df.rename(columns=rename)

def assert_files_exist():
    missing = [p for p in [SALES_FILE, RECIPE_FILE, STOCK_FILE] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "File tidak ditemukan di Google Drive:\n" + "\n".join(missing) +
            "\n\nPastikan Drive sudah di-mount dan DATA_DIR sudah benar."
        )

# =========================================================
# LOAD & CLEANING (path dari Drive, bukan uploader)
# =========================================================
def load_sales_data(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SALES_SHEET, header=2)
    df.columns = [str(c).strip() for c in df.columns]

    df = ensure_cols(df, {
        "tanggal": ["Tanggal", "tgl", "date"],
        "nama produk": ["nama_produk", "produk", "product"],
        "jumlah terjual": ["qty", "quantity", "jumlah", "terjual"]
    })

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["nama produk"] = df["nama produk"].astype(str).str.strip()
    df["jumlah terjual"] = pd.to_numeric(df["jumlah terjual"], errors="coerce").fillna(0)

    df = df.dropna(subset=["tanggal"]).reset_index(drop=True)
    return df

def load_recipe_data(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=RECIPE_SHEET, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    df = ensure_cols(df, {
        "NAMA BAHAN": ["nama bahan", "bahan"],
        "gr/ml/pcs": ["satuan", "unit", "grmlpcs"]
    })

    if "NO" not in df.columns:
        df.insert(0, "NO", np.arange(1, len(df) + 1))

    df = df[~df["NAMA BAHAN"].isna()].reset_index(drop=True)
    df["NAMA BAHAN"] = df["NAMA BAHAN"].astype(str).str.strip()
    return df

def recipe_to_long(df_recipe: pd.DataFrame) -> pd.DataFrame:
    id_vars = ["NO", "NAMA BAHAN", "gr/ml/pcs"]
    product_cols = [c for c in df_recipe.columns if c not in id_vars]

    long_df = df_recipe.melt(
        id_vars=["NAMA BAHAN", "gr/ml/pcs"],
        value_vars=product_cols,
        var_name="nama produk",
        value_name="qty_per_saji",
    )

    long_df["qty_per_saji"] = pd.to_numeric(long_df["qty_per_saji"], errors="coerce")
    long_df = long_df.dropna(subset=["qty_per_saji"])
    long_df = long_df[long_df["qty_per_saji"] > 0].copy()

    long_df["nama produk"] = long_df["nama produk"].astype(str).str.strip()
    long_df["NAMA BAHAN"] = long_df["NAMA BAHAN"].astype(str).str.strip()
    return long_df.reset_index(drop=True)

def load_stock_data(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=STOCK_SHEET, header=3)
    df.columns = [str(c).strip() for c in df.columns]

    df = ensure_cols(df, {
        "NAMA BAHAN": ["nama bahan", "bahan"],
        "Total Stok Ml/Gr": ["totalstokmlgr", "stok", "total stok"],
        "Harga": ["harga", "price"],
        "JUMLAH/pack": ["jumlahpack", "jumlah/pack", "qty/pack", "isi pack"]
    })

    df = df[~df["NAMA BAHAN"].isna()].reset_index(drop=True)
    df["NAMA BAHAN"] = df["NAMA BAHAN"].astype(str).str.strip()

    df["Total Stok Ml/Gr"] = pd.to_numeric(df["Total Stok Ml/Gr"], errors="coerce").fillna(0.0)
    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce").fillna(0.0)
    df["JUMLAH/pack"] = pd.to_numeric(df["JUMLAH/pack"], errors="coerce").replace(0, np.nan).fillna(1.0)
    return df

@st.cache_data
def load_all_data():
    assert_files_exist()
    df_sales = load_sales_data(SALES_FILE)
    df_recipe = load_recipe_data(RECIPE_FILE)
    df_recipe_long = recipe_to_long(df_recipe)
    df_stock = load_stock_data(STOCK_FILE)
    return df_sales, df_recipe, df_recipe_long, df_stock

# =========================================================
# SERIES
# =========================================================
def build_daily_series(df_sales: pd.DataFrame, target_product: str) -> pd.Series:
    df = df_sales[df_sales["nama produk"] == target_product].copy()
    if df.empty:
        raise ValueError(f"Produk '{target_product}' tidak ditemukan di data penjualan.")
    daily = df.groupby("tanggal")["jumlah terjual"].sum().sort_index()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_idx, fill_value=0).rename("jumlah_terjual")

# =========================================================
# LSTM
# =========================================================
def make_lstm_dataset(series: pd.Series, window_size: int, train_ratio: float):
    values = series.values.reshape(-1, 1)
    split_idx = int(len(values) * train_ratio)

    scaler = MinMaxScaler((0, 1))
    scaler.fit(values[:split_idx])  # fit hanya train

    scaled_all = scaler.transform(values).ravel()
    dates = series.index

    X, y, target_i = [], [], []
    for i in range(window_size, len(scaled_all)):
        X.append(scaled_all[i - window_size:i])
        y.append(scaled_all[i])
        target_i.append(i)

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    target_i = np.array(target_i)

    train_mask = target_i < split_idx
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    dates_train = dates[target_i[train_mask]]
    dates_test = dates[target_i[~train_mask]]
    return X_train, y_train, X_test, y_test, dates_train, dates_test, scaler

def build_lstm_model(window_size: int, learning_rate: float):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

def train_and_evaluate_lstm(series: pd.Series, window_size, train_ratio, epochs, batch_size, learning_rate):
    X_train, y_train, X_test, y_test, d_train, d_test, scaler = make_lstm_dataset(series, window_size, train_ratio)

    model = build_lstm_model(window_size, learning_rate)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    y_train_pred = model.predict(X_train, verbose=0).ravel()
    y_test_pred  = model.predict(X_test,  verbose=0).ravel()

    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_inv  = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
    y_test_pred_inv  = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    rmse_train = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
    mae_train  = mean_absolute_error(y_train_inv, y_train_pred_inv)
    rmse_test  = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
    mae_test   = mean_absolute_error(y_test_inv, y_test_pred_inv)

    metrics = dict(rmse_train=rmse_train, mae_train=mae_train, rmse_test=rmse_test, mae_test=mae_test)

    fig_loss = plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss (MSE)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()

    fig_pred = plt.figure(figsize=(12, 4))
    plt.plot(d_train, y_train_inv, label="Train Actual")
    plt.plot(d_train, y_train_pred_inv, linestyle="--", label="Train Predicted")
    plt.plot(d_test, y_test_inv, label="Test Actual")
    plt.plot(d_test, y_test_pred_inv, linestyle="--", label="Test Predicted")
    plt.axvline(d_test.min(), linestyle=":", label="Train/Test Split")
    plt.title("Actual vs Predicted (Training & Testing)")
    plt.xlabel("Tanggal"); plt.ylabel("Jumlah Terjual (cup)")
    plt.legend(); plt.grid(True); plt.tight_layout()

    return model, scaler, metrics, fig_loss, fig_pred

def forecast_future(model, scaler, series: pd.Series, window_size: int, n_future: int):
    last_window = series.values[-window_size:].reshape(-1, 1)
    last_scaled = scaler.transform(last_window).ravel()

    current = last_scaled.reshape(1, window_size, 1)
    preds_scaled = []
    for _ in range(n_future):
        nxt = model.predict(current, verbose=0)[0, 0]
        preds_scaled.append(nxt)
        current = np.append(current[:, 1:, :], [[[nxt]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    return np.maximum(preds, 0)

# =========================================================
# MATERIAL + GWO
# =========================================================
def compute_ingredient_needs(recipe_long: pd.DataFrame, product_demand: dict) -> pd.DataFrame:
    demand_df = pd.DataFrame(list(product_demand.items()), columns=["nama produk", "demand"])
    merged = recipe_long.merge(demand_df, on="nama produk", how="inner")
    merged["kebutuhan"] = merged["qty_per_saji"] * merged["demand"]

    result = merged.groupby(["NAMA BAHAN", "gr/ml/pcs"])["kebutuhan"].sum().reset_index()
    return result.rename(columns={"kebutuhan": "total_kebutuhan", "gr/ml/pcs": "satuan"})

def compare_with_stock(ingredient_needs: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    merged = ingredient_needs.merge(stock_df[["NAMA BAHAN", "Total Stok Ml/Gr"]], on="NAMA BAHAN", how="left")
    merged["Total Stok Ml/Gr"] = merged["Total Stok Ml/Gr"].fillna(0.0)
    merged["kekurangan"] = merged["total_kebutuhan"] - merged["Total Stok Ml/Gr"]
    return merged

def prepare_gwo_data(stock_comparison: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    df = stock_comparison.copy()
    stock_df = df_stock.copy()

    cols = ["NAMA BAHAN", "Harga", "JUMLAH/pack", "Total Stok Ml/Gr"]
    df = df.merge(stock_df[cols], on="NAMA BAHAN", how="left", suffixes=("", "_stock"))

    df["total_kebutuhan"] = pd.to_numeric(df["total_kebutuhan"], errors="coerce").fillna(0.0)
    df["Total Stok Ml/Gr"] = pd.to_numeric(df["Total Stok Ml/Gr"], errors="coerce").fillna(0.0)
    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce").fillna(0.0)
    df["JUMLAH/pack"] = pd.to_numeric(df["JUMLAH/pack"], errors="coerce").replace(0, np.nan).fillna(1.0)

    df["unit_price"] = df["Harga"] / df["JUMLAH/pack"]
    mean_unit = df["unit_price"].replace([np.inf, -np.inf], np.nan).mean()
    df["unit_price"] = df["unit_price"].replace([np.inf, -np.inf], np.nan).fillna(mean_unit if mean_unit and mean_unit > 0 else 1.0)
    return df

def objective_purchase_cost(order_vector, data: pd.DataFrame, penalty_shortage: float, penalty_overstock: float) -> float:
    order = np.clip(np.array(order_vector, dtype=float), 0.0, None)

    needs = data["total_kebutuhan"].values.astype(float)
    stock = data["Total Stok Ml/Gr"].values.astype(float)
    unit_price = data["unit_price"].values.astype(float)

    purchase_cost = np.sum(unit_price * order)
    projected = stock + order - needs

    shortage = np.where(projected < 0, -projected, 0.0)
    overstock = np.where(projected > 0, projected, 0.0)

    penalty = penalty_shortage * np.sum(shortage) + penalty_overstock * np.sum(overstock)
    return float(purchase_cost + penalty)

def grey_wolf_optimization(objective_func, dim, lb, ub, num_wolves=25, max_iter=100):
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    wolves = lb + (ub - lb) * np.random.rand(num_wolves, dim)
    fitness = np.array([objective_func(w) for w in wolves])

    idx = np.argsort(fitness)
    alpha, beta, delta = wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy()
    alpha_score, beta_score, delta_score = fitness[idx[0]], fitness[idx[1]], fitness[idx[2]]

    best_hist = [alpha_score]

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)

        for i in range(num_wolves):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2*a*r1 - a; C1 = 2*r2
                D_alpha = abs(C1*alpha[d] - wolves[i, d]); X1 = alpha[d] - A1*D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2*a*r1 - a; C2 = 2*r2
                D_beta = abs(C2*beta[d] - wolves[i, d]); X2 = beta[d] - A2*D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2*a*r1 - a; C3 = 2*r2
                D_delta = abs(C3*delta[d] - wolves[i, d]); X3 = delta[d] - A3*D_delta

                wolves[i, d] = (X1 + X2 + X3) / 3.0

            wolves[i] = np.clip(wolves[i], lb, ub)

        fitness = np.array([objective_func(w) for w in wolves])
        idx = np.argsort(fitness)

        if fitness[idx[0]] < alpha_score:
            alpha_score = fitness[idx[0]]
            alpha = wolves[idx[0]].copy()
        if fitness[idx[1]] < beta_score:
            beta_score = fitness[idx[1]]
            beta = wolves[idx[1]].copy()
        if fitness[idx[2]] < delta_score:
            delta_score = fitness[idx[2]]
            delta = wolves[idx[2]].copy()

        best_hist.append(alpha_score)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(max_iter+1), best_hist)
    plt.title("GWO: Best Objective per Iteration")
    plt.xlabel("Iteration"); plt.ylabel("Objective Value")
    plt.grid(True); plt.tight_layout()

    return alpha, alpha_score, best_hist, fig

def build_solution_table(data: pd.DataFrame, best_order_vector) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)
    order = np.maximum(0.0, np.array(best_order_vector, dtype=float))

    packs = np.ceil(order / df["JUMLAH/pack"].values.astype(float))
    packs = np.maximum(packs, 0)

    order_rounded = packs * df["JUMLAH/pack"].values.astype(float)

    df["order_mlgr_continuous"] = order
    df["packs_to_buy"] = packs.astype(int)
    df["order_mlgr_rounded"] = order_rounded

    df["stok_akhir"] = df["Total Stok Ml/Gr"] + df["order_mlgr_rounded"] - df["total_kebutuhan"]
    df["shortage_setelah_order"] = np.where(df["stok_akhir"] < 0, -df["stok_akhir"], 0.0)
    df["overstock_setelah_order"] = np.where(df["stok_akhir"] > 0, df["stok_akhir"], 0.0)

    df["total_biaya_pembelian"] = df["packs_to_buy"] * df["Harga"]
    return df
import re

def normalize_id_phone(raw: str):
    """
    Terima input: 08xxxx / 62xxxx / +62xxxx / spasi / strip
    Kembalikan: (is_valid, normalized_phone)
    """
    if raw is None:
        return False, None

    s = raw.strip()

    # Buang semua selain angka dan plus (untuk handle +62)
    s = re.sub(r"[^\d+]", "", s)

    # Ubah ke digits-only untuk proses
    digits = re.sub(r"\D", "", s)

    # Konversi awalan jadi format +62
    if digits.startswith("0"):
        digits = "62" + digits[1:]          # 08xxx -> 628xxx
    elif digits.startswith("62"):
        pass                                # sudah 62xxx
    else:
        return False, None                  # bukan 0/62

    # Validasi pola nomor seluler Indonesia: 628 + 8 + digit berikutnya 1-9
    # Panjang total umum: 11-15 digit (62 + nomor)
    if not re.match(r"^628[1-9]\d{7,11}$", digits):
        return False, None

    return True, "+" + digits


# =========================================================
# SESSION
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "last_future_int" not in st.session_state:
    st.session_state.last_future_int = None
if "last_product" not in st.session_state:
    st.session_state.last_product = None

# =========================================================
# UI: Login + Sidebar menu
# =========================================================
LOGIN_TOKEN = "3359ab54dece32"

def login_page():
     # ambil gambar background dari assets
    bg_path = str((BASE_DIR / "assets" / "coffee_bg.jpg"))
    bg64 = img_to_base64(bg_path)

    st.markdown(LOGIN_CSS_TMPL.replace("__BG__", bg64), unsafe_allow_html=True)
        # ===== Logo (pojok kiri atas) =====
    logo_path = str((BASE_DIR / "assets" / "logo.png"))  # ganti sesuai nama file logo kamu
    logo64 = img_to_base64(logo_path)

    st.markdown(f"""
      <div class="login-logo">
        <img src="data:image/png;base64,{logo64}" alt="logo" style="width:460px ;height:380px">
      </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
      <div class="hero">
        <div style="display:flex; align-items:center; gap:12px;">
          <div class="badge">☕</div>
          <div>
            <p class="brand-title">Q Coffee App</p>
            <p class="brand-sub">Sales Forecasting (LSTM) • Material Optimization (GWO)</p>
          </div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-card'><h3 style='margin:0;color:rgba(255,255,255,.92);font-size:15px;font-weight:900;'>Login</h3></div>", unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        nama = st.text_input("Nama", placeholder="Contoh: Ahmad")
        no_telp = st.text_input("No. Telp", placeholder="Contoh: 08xxxxxxxxxx")
        token = st.text_input("Password / Token", type="password", placeholder="Masukkan token")
        submitted = st.form_submit_button("Masuk")

    if submitted:
        ok, phone_norm = normalize_id_phone(no_telp)

        if not ok:
            st.error("No. Telp tidak valid. Contoh format benar: 08xxxxxxxxxx atau +62xxxxxxxxxx")
            return
        if token != LOGIN_TOKEN:
            st.error("Token salah.")
            return
        st.session_state.profile = {
            "nama": nama.strip() if nama else "-",
            "no_telp": phone_norm
        }
        st.session_state.logged_in = True
        st.rerun()


def sidebar_menu():
    p = st.session_state.profile
    with st.sidebar:
        st.markdown("""
          <style>
          /* Sidebar background jadi abu-abu */
          section[data-testid="stSidebar"] > div {
            width: 18rem !important;
            background: #f3f4f6 !important;  /* abu-abu muda */
          }

          /* Opsional: rapikan padding sidebar */
           section[data-testid="stSidebar"] > div {
            width: 18rem !important;/
            padding-top: 1rem;
             padding-left: 1rem;
            padding-right: 1rem;
           }
          </style>
          """, unsafe_allow_html=True)
        st.markdown("<div class='brand'>☕ Q Coffee App</div>", unsafe_allow_html=True)

        st.markdown(f"""
          <div class="card">
            <h4>Profil</h4>
            <div class="kv">
              <div class="k">Nama</div><div class="v">{p.get('nama','-')}</div>
              <div class="k">No. Telp</div><div class="v">{p.get('no_telp','-')}</div>
            </div>
          </div>
          """, unsafe_allow_html=True)


        if option_menu:
            selected = option_menu(
                menu_title=None,
                options=["Prediksi Penjualan", "Optimasi Bahan Baku", "Resep", "Hasil Implementasi"],
                icons=["graph-up-arrow", "boxes", "book", "clipboard-data"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important"},
                    "icon": {"font-size": "16px"},
                    "nav-link": {"font-size": "14px", "border-radius": "10px", "margin": "4px 0"},
                    "nav-link-selected": {"font-weight": "900"},
                }
            )
        else:
            selected = st.radio("Menu", ["Prediksi Penjualan", "Optimasi Bahan Baku", "Resep", "Hasil Implementasi"])

        st.markdown("---")
        if st.button("Reload Data (Drive)"):
            st.cache_data.clear()
            st.rerun()

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.profile = {}
            st.session_state.last_future_int = None
            st.session_state.last_product = None
            st.rerun()

    return selected

# =========================================================
# PAGES
# =========================================================
def page_sales():
    st.title("Prediksi Penjualan (LSTM)")
    df_sales, _, _, _ = load_all_data()

    products = sorted(df_sales["nama produk"].dropna().unique().tolist())
    if not products:
        st.error("Tidak ada produk terbaca dari data penjualan.")
        return

    default_idx = products.index(DEFAULT_PRODUCT) if DEFAULT_PRODUCT in products else 0
    product = st.selectbox("Pilih produk target", options=products, index=default_idx)

    colA, colB, colC = st.columns(3)
    with colA:
        window_size = st.number_input("Window Size", 5, 120, 30)
        train_ratio = st.slider("Train Ratio", 0.5, 0.95, 0.8, 0.05)
    with colB:
        epochs = st.number_input("Epochs", 1, 200, 50)
        batch = st.number_input("Batch Size", 1, 256, 32)
    with colC:
        lr = st.number_input("Learning Rate", value=0.001, format="%.6f")
        n_future = st.number_input("Prediksi N hari", 1, 30, 7)

    topN = 14
    top_products = (df_sales.groupby("nama produk")["jumlah terjual"].sum().astype(int)
                    .sort_values(ascending=False).head(topN))
    st.subheader(f"Top {topN} produk (cup)")
    top_df = top_products.reset_index().rename(columns={
    "nama produk": "Produk",
    "jumlah terjual": "Total (cup)"
})
    top_df.index = np.arange(1, len(top_df) + 1)  # index mulai dari 1

    st.dataframe(top_df, width="stretch")

    if st.button("Run LSTM Forecast"):
        with st.spinner("Training LSTM..."):
            series = build_daily_series(df_sales, product)
            model, scaler, metrics, fig_loss, fig_pred = train_and_evaluate_lstm(
                series,
                window_size=int(window_size),
                train_ratio=float(train_ratio),
                epochs=int(epochs),
                batch_size=int(batch),
                learning_rate=float(lr),
            )
            preds = forecast_future(model, scaler, series, int(window_size), int(n_future))
            preds_int = np.ceil(preds).astype(int)  # cup integer

            st.session_state.last_future_int = preds_int
            st.session_state.last_product = product

        st.success("Selesai.")
        st.subheader("Evaluasi")
        st.json(metrics)

        st.subheader("Loss Curve")
        st.pyplot(fig_loss)

        st.subheader("Actual vs Predicted")
        st.pyplot(fig_pred)

        st.subheader("Prediksi ke depan (cup integer)")
        out_df = pd.DataFrame({"Hari":[f"+{i+1}" for i in range(len(preds_int))],
                               "Prediksi (cup)": preds_int})
        out_df.index= np.arange(1, len(out_df) + 1)  # index mulai dari 1
        st.dataframe(out_df, width="stretch")
        st.line_chart(out_df.set_index("Hari")["Prediksi (cup)"])

def page_material():
    st.title("Optimasi Bahan Baku (Recipe + Stock + GWO)")
    _, _, df_recipe_long, df_stock = load_all_data()

    st.subheader("Demand (cup)")
    if st.session_state.last_future_int is not None:
        st.write(f"Prediksi terakhir untuk produk: **{st.session_state.last_product}**")
        use_pred = st.checkbox("Gunakan rata-rata prediksi LSTM", value=True)
    else:
        use_pred = False
        st.warning("Belum ada prediksi LSTM. Jalankan Sales Prediction dulu atau isi demand manual.")

    if use_pred and st.session_state.last_future_int is not None:
        product = st.session_state.last_product
        demand_units = int(np.ceil(np.mean(st.session_state.last_future_int)))
    else:
        product = st.text_input("Nama produk", value=DEFAULT_PRODUCT)
        demand_units = int(st.number_input("Demand (cup) integer", min_value=0, value=50))

    st.write("Demand dipakai:", demand_units, "cup")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_wolves = st.number_input("Jumlah Wolves", 5, 100, 25)
    with col2:
        max_iter = st.number_input("Max Iter", 10, 500, 100)
    with col3:
        ub_mult = st.number_input("Upper Bound Multiplier", value=2.0)

    if st.button("Run Material + GWO"):
        with st.spinner("Menghitung kebutuhan bahan & menjalankan GWO..."):
            product_demand = {product: demand_units}
            ingredient_needs = compute_ingredient_needs(df_recipe_long, product_demand)
            stock_comparison = compare_with_stock(ingredient_needs, df_stock)

            st.subheader("Kebutuhan vs Stok (Top 20 kekurangan terbesar)")
            cols = ["NAMA BAHAN","satuan","total_kebutuhan","Total Stok Ml/Gr","kekurangan"]
            t = stock_comparison[cols].copy().sort_values("kekurangan", ascending=False)
            st.dataframe(t.head(20), hide_index=True, width="stretch")

            df_gwo = prepare_gwo_data(stock_comparison, df_stock)
            dim = len(df_gwo)

            lb = np.zeros(dim)
            ub = df_gwo["total_kebutuhan"].values * float(ub_mult)

            avg_unit = df_gwo["unit_price"].mean()
            max_unit = df_gwo["unit_price"].max()
            penalty_shortage = 10.0 * max_unit
            penalty_overstock = 0.5 * avg_unit

            def objective_wrapper(order_vec):
                return objective_purchase_cost(order_vec, df_gwo, penalty_shortage, penalty_overstock)

            best_order, best_cost, best_hist, fig_gwo = grey_wolf_optimization(
                objective_wrapper, dim, lb, ub,
                num_wolves=int(num_wolves),
                max_iter=int(max_iter),
            )

            solution = build_solution_table(df_gwo, best_order)

        st.success(f"Selesai. Best Objective: {best_cost:.4f}")

        st.subheader("Grafik Objective Function (GWO)")
        st.pyplot(fig_gwo)

        st.subheader("Tabel Solusi (Top 20 biaya terbesar)")
        cols2 = ["NAMA BAHAN","total_kebutuhan","Total Stok Ml/Gr","packs_to_buy","order_mlgr_rounded",
                 "stok_akhir","shortage_setelah_order","overstock_setelah_order","Harga","total_biaya_pembelian"]
        t2 = solution[cols2].copy().sort_values("total_biaya_pembelian", ascending=False)
        st.dataframe(t2.head(20), hide_index=True, width="stretch")

        st.subheader("Total biaya pembelian")
        st.write(f"Rp {solution['total_biaya_pembelian'].sum():,.0f}".replace(",", "."))

def page_recipe():
    st.title("Resep")
    _, df_recipe, df_recipe_long, _ = load_all_data()

    st.subheader("Resep – Contoh 200 baris")
    view_long: pd.DataFrame = df_recipe_long.rename(columns={
        "gr/ml/pcs": "Satuan",
        "NAMA BAHAN": "Nama Bahan",
        "nama produk": "Nama Produk",
        "qty_per_saji" : "Quantity",
    }).reset_index(drop=True)
    view_long.index = view_long.index + 1
    st.dataframe(view_long.head(200), width="stretch")

def page_results():
    st.title("Hasil Implementasi")

    # --- Tabel Profil ---
    p = st.session_state.profile or {}
    df_profile = pd.DataFrame([
        {"Field": "Nama", "Value": p.get("nama", "-")},
        {"Field": "No. Telp", "Value": p.get("no_telp", "-")},
    ])
    st.subheader("Profil")
    st.dataframe(df_profile, hide_index=True, width="stretch")


    # --- Tabel Status Data (opsional) ---
    df_paths = pd.DataFrame([
    {"File": "DATA_DIR", "Path": str(DATA_DIR)},
    {"File": "SALES_FILE", "Path": str(SALES_FILE)},
    {"File": "RECIPE_FILE", "Path": str(RECIPE_FILE)},
    {"File": "STOCK_FILE", "Path": str(STOCK_FILE)},
    ])

    st.subheader("Status Data")
    st.dataframe(df_paths, hide_index=True, width="stretch")

    # --- Tabel Prediksi Terakhir ---
    if st.session_state.last_future_int is not None and st.session_state.last_product is not None:
        preds = st.session_state.last_future_int
        prod = st.session_state.last_product

        df_pred = pd.DataFrame({
            "Produk": [prod] * len(preds),
            "Hari": [f"+{i+1}" for i in range(len(preds))],
            "Prediksi (cup)": preds.astype(int)
        })

        st.subheader("Prediksi Terakhir (cup integer)")
        st.dataframe(df_pred, hide_index=True, width="stretch")

        # Optional: grafik cepat
        st.line_chart(df_pred.set_index("Hari")["Prediksi (cup)"])
    else:
        st.info("Belum ada hasil prediksi. Jalankan Sales Prediction dulu.")

# =========================================================
# APP FLOW
# =========================================================
try:
    assert_files_exist()
except Exception as e:
    st.error(str(e))
    st.stop()

if not st.session_state.logged_in:
    login_page()
else:
    page = sidebar_menu()
    if page == "Prediksi Penjualan":
        page_sales()
    elif page == "Optimasi Bahan Baku":
        page_material()
    elif page == "Resep":
        page_recipe()
    else:
        page_results()

