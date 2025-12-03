import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import time # Untuk simulasi loading

st.set_page_config(page_title="MADM - Web Hosting", layout="wide")

st.title("MADM — Pemilihan Web Hosting Murah (WP & TOPSIS)")

# ---------- config ----------
PROVIDERS = ["HostA","HostB","HostC","HostD","HostE"]

# KRITIS: Data Krips ini disinkronkan agar sesuai dengan kemungkinan data di Excel Anda
# Jika hasil di Excel Anda berbeda, Anda HARUS mengedit data di bagian "Input Data"
# Bobot default dari data: C1=0.30, C2=0.25, C3=0.20, C4=0.15, C5=0.10
synced_df = pd.DataFrame({
    "Cost (Rp)":[80, 100, 100, 100, 80],
    "Storage (GB)":[60, 80, 100, 60, 100],
    "Bandwidth (GB)":[60, 100, 60, 80, 80],
    "Uptime (%)":[80, 80, 100, 60, 60],
    "Support (1-10)":[80, 60, 60, 100, 80]
}, index=PROVIDERS)

if "df" not in st.session_state:
    # Memastikan session_state menggunakan data yang telah disinkronkan
    st.session_state.df = synced_df.copy()

st.sidebar.header("Menu")
page = st.sidebar.radio("Pilih halaman", ["Home","Input Data","WP","TOPSIS","Perbandingan","Tentang"])

# weights (editable)
st.sidebar.markdown("### Bobot Kriteria")
w1 = st.sidebar.slider("Cost (Rp) (w1)", 0.0, 1.0, 0.30, 0.01)
w2 = st.sidebar.slider("Storage (GB) (w2)", 0.0, 1.0, 0.25, 0.01)
w3 = st.sidebar.slider("Bandwidth (GB) (w3)", 0.0, 1.0, 0.20, 0.01)
w4 = st.sidebar.slider("Uptime (%) (w4)", 0.0, 1.0, 0.15, 0.01)
w5 = st.sidebar.slider("Support (1-10) (w5)", 0.0, 1.0, 0.10, 0.01)

# Normalisasi bobot
ws = np.array([w1,w2,w3,w4,w5])
if ws.sum() == 0:
    # Menggunakan bobot default jika total bobot 0
    ws = np.array([0.30,0.25,0.20,0.15,0.10])
else:
    ws = ws / ws.sum()

# Tipe kriteria: C1=Cost, C2-C5=Benefit
TYPES = ["cost","benefit","benefit","benefit","benefit"]

# ----------------------------------------------------------------------
# FUNGSI PERHITUNGAN
# ----------------------------------------------------------------------

@st.cache_data
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
    res["Rank"] = res["Vektor V (Score)"].rank(ascending=False, method='min').astype(int) # Gunakan method='min' untuk menghindari rank yang sama
    
    return res, w_exp

@st.cache_data
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
    res["Rank"] = res["CC"].rank(ascending=False, method='min').astype(int)
    
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
    st.info("Pastikan data ini sama persis dengan yang Anda gunakan di Excel.")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic")
    st.session_state.df = edited
    st.download_button("Download data (.csv)", edited.to_csv().encode('utf-8'), file_name="data_input.csv")
    
elif page=="WP":
    st.header("Hasil Weighted Product (WP)")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    
    with st.spinner('Menghitung WP...'):
        time.sleep(0.5) # Simulasi perhitungan
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
    
elif page=="TOPSIS":
    st.header("Hasil TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    
    with st.spinner('Menghitung TOPSIS...'):
        time.sleep(0.5) # Simulasi perhitungan
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

elif page=="Perbandingan":
    st.header("Perbandingan WP vs TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    
    res_wp, _ = wp_calc(df, ws)
    res_top, _, _, _, _ = topsis_calc(df, ws)
    
    # Mengambil kolom Vektor V (Score) dari WP dan CC dari TOPSIS
    compare = pd.DataFrame({"WP":res_wp["Vektor V (Score)"], "TOPSIS":res_top["CC"]})
    st.dataframe(compare.style.format("{:.6f}"))
    
    # Visualisasi
    st.subheader("Diagram Perbandingan Hasil")
    fig,ax = plt.subplots(figsize=(8,4))
    compare.plot(kind='bar', ax=ax, rot=0, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel("Score / CC")
    ax.set_xlabel("Penyedia Hosting")
    ax.set_title("Perbandingan Skor WP dan TOPSIS")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    
    top_wp = compare["WP"].idxmax(); top_top = compare["TOPSIS"].idxmax()
    st.subheader("Kesimpulan Ranking")
    if top_wp == top_top:
        st.success(f"Kedua metode memilih: **{top_wp}** sebagai alternatif terbaik.")
    else:
        st.info(f"WP memilih: **{top_wp}**, sedangkan TOPSIS memilih: **{top_top}**.")

elif page=="Tentang":
    st.header("Tentang")
    st.write("Aplikasi ini menggunakan Python Streamlit untuk mengimplementasikan metode *Weighted Product (WP)* dan *Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)*.")
    st.write("""
    Metode digunakan untuk mendukung pengambilan keputusan multi-kriteria (Multi-Attribute Decision Making/MADM) dalam pemilihan Web Hosting Murah berdasarkan kriteria:
    1. **Cost (Cost):** Biaya layanan
    2. **Storage (Benefit):** Kapasitas penyimpanan
    3. **Bandwidth (Benefit):** Batas transfer data
    4. **Uptime (Benefit):** Tingkat ketersediaan layanan
    5. **Support (Benefit):** Kualitas layanan dukungan
    """)
