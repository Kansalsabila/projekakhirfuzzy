# app_singlefile.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="MADM - Web Hosting", layout="wide")

# Perubahan Judul
st.title("MADM — Pemilihan Web Hosting Murah (WP & TOPSIS)")

# ---------- config ----------
PROVIDERS = ["HostA","HostB","HostC","HostD","HostE"]
# Kriteria (nilai crips dari Excel): C1=Cost, C2=Storage, C3=Bandwidth, C4=Uptime, C5=Support
default_df = pd.DataFrame({
    "Cost (Rp)":[100, 80, 80, 80, 100],
    "Storage (GB)":[60, 80, 100, 60, 100],
    "Bandwidth (GB)":[60, 100, 60, 80, 80],
    "Uptime (%)":[80, 80, 100, 60, 60],
    "Support (1-10)":[80, 60, 60, 100, 80]
}, index=PROVIDERS)

if "df" not in st.session_state:
    st.session_state.df = default_df.copy()

st.sidebar.header("Menu")
# Perubahan: Mengganti "Fuzzy SAW" menjadi "WP"
page = st.sidebar.radio("Pilih halaman", ["Home","Input Data","WP","TOPSIS","Perbandingan","Tentang"])

# weights (editable)
st.sidebar.markdown("### Bobot Kriteria")
# Bobot dari data: C1=0.30, C2=0.25, C3=0.20, C4=0.15, C5=0.10
w1 = st.sidebar.slider("Cost (Rp) (w1)", 0.0, 1.0, 0.30, 0.01)
w2 = st.sidebar.slider("Storage (GB) (w2)", 0.0, 1.0, 0.25, 0.01)
w3 = st.sidebar.slider("Bandwidth (GB) (w3)", 0.0, 1.0, 0.20, 0.01)
w4 = st.sidebar.slider("Uptime (%) (w4)", 0.0, 1.0, 0.15, 0.01)
w5 = st.sidebar.slider("Support (1-10) (w5)", 0.0, 1.0, 0.10, 0.01)

# Normalisasi bobot
ws = np.array([w1,w2,w3,w4,w5])
if ws.sum() == 0:
    ws = np.array([0.30,0.25,0.20,0.15,0.10])
else:
    ws = ws / ws.sum()

# Tipe kriteria: C1=Cost, C2-C5=Benefit
TYPES = ["cost","benefit","benefit","benefit","benefit"]

# ----------------------------------------------------------------------
# FUNGSI PERHITUNGAN
# ----------------------------------------------------------------------

def wp_calc(df, weights):
    """Menghitung Weighted Product (WP) klasik."""
    
    # 1. Tentukan Bobot Berpangkat (Cost = negatif, Benefit = positif)
    w_exp = np.array(weights)
    for i, t in enumerate(TYPES):
        if t == "cost":
            w_exp[i] *= -1 
            
    # 2. Hitung Vektor S (S = Pi (R^w))
    df_mat = df.astype(float).values
    S = []
    for row in df_mat:
        # Menghitung Vektor S dengan perkalian berbobot
        s_val = np.prod(np.power(row, w_exp))
        S.append(s_val)
        
    # 3. Hitung Vektor V (Normalisasi Vektor S)
    sum_S = np.sum(S)
    V = [s / sum_S for s in S]
    
    res = pd.DataFrame({"Vektor S":S, "Vektor V (Score)":V}, index=df.index)
    res["Rank"] = res["Vektor V (Score)"].rank(ascending=False).astype(int)
    
    return res, w_exp

def topsis_calc(df, weights):
    """Menghitung TOPSIS klasik."""
    mat = df.astype(float)
    
    # 1. Normalized Matrix (X) - Vector Normalization
    norm_mat = mat.copy()
    for col in mat.columns:
        denom = np.sqrt((mat[col]**2).sum())
        norm_mat[col] = mat[col]/denom if denom!=0 else 0
        
    # 2. Weighted Normalized Matrix (V)
    weighted = norm_mat * weights
    
    # 3. Ideal Solution (A+ dan A-)
    a_plus=[]; a_minus=[]
    for i,col in enumerate(weighted.columns):
        if TYPES[i]=="benefit":
            a_plus.append(weighted[col].max()); a_minus.append(weighted[col].min())
        else: # Cost
            a_plus.append(weighted[col].min()); a_minus.append(weighted[col].max())
            
    a_plus=np.array(a_plus); a_minus=np.array(a_minus)
    
    # 4. Separation Distance (Dp dan Dm)
    Dp=[]; Dm=[]
    for idx in weighted.index:
        row = weighted.loc[idx].values
        # Jarak Euclidean: sqrt(Sigma (row - A_ideal)^2)
        dp = np.sqrt(((row - a_plus)**2).sum())
        dm = np.sqrt(((row - a_minus)**2).sum())
        Dp.append(dp); Dm.append(dm)
        
    # 5. Closeness Coefficient (CC)
    CC = [dm/(dp+dm) if (dp+dm)!=0 else 0 for dp,dm in zip(Dp,Dm)]
    
    # Format A+ dan A- untuk tampilan
    a_plus_df = pd.DataFrame([a_plus], columns=weighted.columns, index=["A+"])
    a_minus_df = pd.DataFrame([a_minus], columns=weighted.columns, index=["A-"])
    
    res = pd.DataFrame({"D_plus":Dp,"D_minus":Dm,"CC":CC}, index=weighted.index)
    res["Rank"] = res["CC"].rank(ascending=False).astype(int)
    
    return res, norm_mat, weighted, a_plus_df, a_minus_df

# ----------------------------------------------------------------------
# ---------- PAGES ----------
# ----------------------------------------------------------------------

if page=="Home":
    st.header("Ringkasan")
    st.write("Aplikasi membandingkan metode *Weighted Product (WP)* & *TOPSIS* untuk pemilihan Web Hosting Murah.")
    st.write("Gunakan menu Input Data, lalu jalankan perhitungan WP & TOPSIS.")
    
elif page=="Input Data":
    st.header("Input / Edit Data (Nilai Crips)")
    st.info("Nilai adalah crips yang didapat dari tabel kriteria (misalnya, Cost ≤ Rp 20.000 = 100).")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic")
    st.session_state.df = edited
    st.download_button("Download data (.csv)", edited.to_csv().encode('utf-8'), file_name="data_input.csv")
    
# Perubahan: Halaman Fuzzy WP menjadi WP
elif page=="WP":
    st.header("Hasil Weighted Product (WP)")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_wp, w_exp = wp_calc(df, ws)
    
    st.subheader("Bobot Kriteria Berpangkat ($W_j$)")
    w_exp_df = pd.DataFrame([w_exp], columns=df.columns, index=["Bobot Berpangkat"])
    st.dataframe(w_exp_df.style.format("{:.6f}"))
    
    st.subheader("Vektor S & Vektor V (Score) Hasil WP")
    st.dataframe(res_wp.style.format("{:.6f}"))
    
    # download
    out = pd.concat([df, w_exp_df, res_wp], axis=0)
    buf = BytesIO()
    out.to_excel(buf, index=True, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download hasil WP (.xlsx)", data=buf, file_name="hasil_wp.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
# Perubahan: Halaman Fuzzy TOPSIS menjadi TOPSIS
elif page=="TOPSIS":
    st.header("Hasil TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_top, norm_mat, weighted, a_plus_df, a_minus_df = topsis_calc(df, ws)
    
    st.subheader("Normalisasi Matrix ($X$) - Vektor")
    st.dataframe(norm_mat.style.format("{:.6f}"))
    
    st.subheader("Normalisasi Berbobot ($V$)")
    st.dataframe(weighted.style.format("{:.6f}"))
    
    st.subheader("Solusi Ideal Positif ($A^+$) dan Negatif ($A^-$)")
    st.dataframe(pd.concat([a_plus_df, a_minus_df]).style.format("{:.6f}"))
    
    st.subheader("Hasil TOPSIS (Jarak & $CC$)")
    st.dataframe(res_top.style.format("{:.6f}"))
    
    # download
    out = pd.concat([df, norm_mat.add_prefix("norm_"), weighted.add_prefix("bobot_"), res_top], axis=1)
    buf = BytesIO()
    out.to_excel(buf, index=True, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download hasil TOPSIS (.xlsx)", data=buf, file_name="hasil_topsis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Perubahan: Halaman Perbandingan (WP vs TOPSIS)
elif page=="Perbandingan":
    st.header("Perbandingan WP vs TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_wp, _ = wp_calc(df, ws)
    res_top, _, _, _, _ = topsis_calc(df, ws)
    
    # Mengambil kolom Vektor V (Score) dari WP dan CC dari TOPSIS
    compare = pd.DataFrame({"WP":res_wp["Vektor V (Score)"], "TOPSIS":res_top["CC"]})
    st.dataframe(compare.style.format("{:.6f}"))
    
    fig,ax = plt.subplots(figsize=(8,4))
    compare.plot(kind='bar', ax=ax)
    ax.set_ylabel("Score / CC")
    st.pyplot(fig)
    
    top_wp = compare["WP"].idxmax(); top_top = compare["TOPSIS"].idxmax()
    if top_wp == top_top:
        st.success(f"Kedua metode memilih: *{top_wp}*")
    else:
        st.info(f"WP -> *{top_wp}, TOPSIS -> **{top_top}*")

elif page=="Tentang":
    st.header("Tentang")
    st.write("Aplikasi untuk Projek MADM — WP & TOPSIS. Dibuat untuk memilih Web Hosting Murah.")
