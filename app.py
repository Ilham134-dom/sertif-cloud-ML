import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, time

# ================================
# CONFIG (SESUAI EKSPERIMEN)
# ================================
CITY = "Laayoune"
ZONES = ["zone1", "zone2", "zone3", "zone4", "zone5"]

# Lag HOURLY (bukan menit)
LAGS = [1, 2, 3, 6]

st.set_page_config(
    page_title="Electric Load Forecasting — Laayoune",
    layout="wide"
)

st.title(" Electric Load Forecasting — Laayoune")
st.markdown("#### *Hybrid ML (GB, RF, ET, LGBM) + PLCNet | Hourly Load Forecasting*")

with st.sidebar:
    st.header(" Panduan Penggunaan")
    st.markdown("""
    **1. Pilih Waktu Prediksi**  
    Tentukan Tanggal dan Jam yang ingin diprediksi di bagian utama.
    
    **2. Ketersediaan Data**  
    Sistem akan otomatis menarik histori riil (lag 1, 2, 3, 6 jam) dari dataset terpadu sebelum jam target.    
    *(Pastikan tanggal yang dipilih memiliki ketersediaan data di masa lalu untuk di-generate)*.
    
    **3. Prediksi Multi-Model**  
    Klik **Prediksi Beban Listrik** untuk melihat hasil *inference* 25 model (5 Algoritma x 5 Zona) secara bersamaan.
    """)
    st.divider()
    st.info(" **Tips Demo:** Coba gunakan tanggal seperti **15 September 2022** yang berada di pertengahan log untuk uji coba yang lancar.")

# ================================
# LOAD DATA HISTORI (HOURLY)
# ================================
@st.cache_data
def load_history():
    df = pd.read_csv("data_histori.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.sort_values("DateTime")

df_hist = load_history()

# ================================
# LOAD MODELS (SAFE DEPLOY)
# ================================
@st.cache_resource
def load_models():
    models = {}
    for z in ZONES:
        models[z] = {
            "scaler_x": joblib.load(f"scaler_x_{z}.pkl"),
            "scaler_y": joblib.load(f"scaler_y_{z}.pkl"),
            "GB": joblib.load(f"gb_model_{z}.pkl"),
            "RF": joblib.load(f"rf_model_{z}.pkl"),
            "ET": joblib.load(f"et_model_{z}.pkl"),
            "LGBM": joblib.load(f"lgbm_model_{z}.pkl"),
            # compile=False WAJIB agar tidak error keras.metrics.mse
            "PLCNet": load_model(f"plcnet_model_{z}.h5", compile=False)
        }
    return models

models = load_models()

# ================================
# INPUT WAKTU (DIKUNCI PER JAM)
# ================================
st.subheader(" 1. Tentukan Waktu Prediksi")

col_date, col_hour = st.columns(2)

with col_date:
    input_date = st.date_input(
        "Pilih Tanggal",
        value=datetime(2022, 9, 15).date()
    )

with col_hour:
    input_hour = st.selectbox(
        "Pilih Jam (Hourly 0-23)",
        options=[time(h, 0) for h in range(24)],
        index=18
    )

input_dt = datetime.combine(input_date, input_hour)

st.info(
    " Prediksi dikunci per **jam** karena model dilatih dan diuji "
    "menggunakan **hourly load data**."
)

# ================================
# FEATURE ENGINEERING (HOURLY)
# ================================
def get_time_of_day(hour):
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3

def build_features(zone, dt):
    hist = df_hist[df_hist["DateTime"] < dt].tail(max(LAGS))

    if len(hist) < max(LAGS):
        raise ValueError("Data histori tidak cukup untuk membentuk lag.")

    lag_feats = {
        f"{zone}_lag{l}": hist.iloc[-l][zone]
        for l in LAGS
    }

    time_feats = {
        "Hour": dt.hour,
        "Month": dt.month,
        "DayOfWeek": dt.weekday(),
        "DayOfYear": dt.timetuple().tm_yday,
        "WeekOfYear": dt.isocalendar().week,
        "Is_Weekend": 1 if dt.weekday() >= 5 else 0,
        "Time_of_Day": get_time_of_day(dt.hour)
    }

    return pd.DataFrame([{**time_feats, **lag_feats}])


st.markdown("---")
st.subheader(" 2. Eksekusi Model")

if st.button("🔮 Jalankan Prediksi Beban Listrik", type="primary", use_container_width=True):
    with st.spinner("Mengkuantisasi fitur dan menjalankan eksekusi 25 Model AI secara paralel..."):
        results = []
    total_city = {m: 0.0 for m in ["GB", "RF", "ET", "LGBM", "PLCNet"]}

    for zone in ZONES:
        feats = build_features(zone, input_dt)

        scaler_x = models[zone]["scaler_x"]
        scaler_y = models[zone]["scaler_y"]

        X_scaled = scaler_x.transform(feats)

        preds = {}
        preds["GB"] = models[zone]["GB"].predict(X_scaled)[0]
        preds["RF"] = models[zone]["RF"].predict(X_scaled)[0]
        preds["ET"] = models[zone]["ET"].predict(X_scaled)[0]
        preds["LGBM"] = models[zone]["LGBM"].predict(X_scaled)[0]

        X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        plc_scaled = models[zone]["PLCNet"].predict(X_lstm, verbose=0)[0][0]
        preds["PLCNet"] = scaler_y.inverse_transform([[plc_scaled]])[0][0]

        for k in preds:
            total_city[k] += preds[k]

        results.append({
            "Zone": zone,
            **{k: round(v, 2) for k, v in preds.items()}
        })

    # --- KELUAR DARI LOOP ZONA ---
    df_res = pd.DataFrame(results)
    
    st.success(" Prediksi berhasil diselesaikan!")
    
    tab1, tab2 = st.tabs([" Total Beban Kota", " Detail per Zona"])
    
    with tab1:
        st.markdown("#### Total Agregasi Beban Prediksi (MW)")
        cols = st.columns(5)
        for idx, (m_name, m_val) in enumerate(total_city.items()):
            cols[idx].metric(label=f"Model {m_name}", value=f"{round(m_val, 2)} MW")
        
        st.caption("Perbandingan total beban kota hasil dari 5 arsitektur algoritma AI yang berbeda.")
        
    with tab2:
        st.markdown("#### Rincian Prediksi Beban Setiap Zona (MW)")
        st.dataframe(df_res, use_container_width=True)
        st.caption("Breakdown hasil algoritma untuk Zone 1 hingga Zone 5.")
