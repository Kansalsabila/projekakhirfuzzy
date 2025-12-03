# app_singlefile.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Fuzzy MADM - Web Hosting", layout="wide")

st.title("Fuzzy MADM — Pemilihan Web Hosting Murah (WP & TOPSIS)")

# ---------- config ----------
PROVIDERS = ["HostA","HostB","HostC","HostD","HostE"]
# Kriteria: C1=Cost (Rp), C2=Storage (GB), C3=Bandwidth (GB), C4=Uptime (%), C5=Support (1-10)
default_df = pd.DataFrame({
    "Cost (Rp)":[100, 80, 80, 80, 100],
    "Storage (GB)":[60, 80, 100, 60, 100],
    "Bandwidth (GB)":[60, 100, 60, 80, 80],
    "Uptime (%)":[80, 80, 100, 60, 60],
    "Support (1-10)":[80, 60, 60, 100, 80]
}, index=PROVIDERS)
# ---------------------------

if "df" not in st.session_state:
    st.session_state.df = default_df.copy()

st.sidebar.header("Menu")
# Perubahan: Mengganti "Fuzzy SAW" menjadi "Fuzzy WP"
page = st.sidebar.radio("Pilih halaman", ["Home","Input Data","Fuzzy WP","Fuzzy TOPSIS","Perbandingan","Tentang"])

# weights (editable)
st.sidebar.markdown("### Bobot Kriteria")
# Bobot dari data: C1=0.30, C2=0.25, C3=0.20, C4=0.15, C5=0.10
w1 = st.sidebar.slider("Cost (Rp) (w1)", 0.0, 1.0, 0.30, 0.01)
w2 = st.sidebar.slider("Storage (GB) (w2)", 0.0, 1.0, 0.25, 0.01)
w3 = st.sidebar.slider("Bandwidth (GB) (w3)", 0.0, 1.0, 0.20, 0.01)
w4 = st.sidebar.slider("Uptime (%) (w4)", 0.0, 1.0, 0.15, 0.01)
w5 = st.sidebar.slider("Support (1-10) (w5)", 0.0, 1.0, 0.10, 0.01)

# normalisasi bobot
ws = np.array([w1,w2,w3,w4,w5])
if ws.sum() == 0:
    ws = np.array([0.30,0.25,0.20,0.15,0.10])
else:
    ws = ws / ws.sum()

# Tipe kriteria: C1=Cost, C2-C5=Benefit
TYPES = ["cost","benefit","benefit","benefit","benefit"]

# Fungsi normalisasi (Digunakan oleh WP dan TOPSIS)
def normalize(df):
    res = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for i,col in enumerate(df.columns):
        if TYPES[i]=="benefit":
            # Normalisasi Benefit: (x - min) / (max - min)
            res[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            # Normalisasi Cost: (max - x) / (max - min)
            res[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    return res

# Fungsi TFN
def tri(v):
    # Fungsi Anggota TFN (a, m, b)
    a = max(0, v-0.1); m = v; b = min(1, v+0.1)
    return np.array([a,m,b])

def wp_calc(df, weights):
    # Fuzzy WP: Menggunakan TFN dan Defuzzifikasi
    normal = normalize(df)
    tfn_total = {}
    scores=[]
    
    # 1. Tentukan Bobot Eksponen
    # Untuk WP, bobot harus disesuaikan untuk tipe kriteria
    w_exp = np.array(weights)
    for i, t in enumerate(TYPES):
        # Dalam WP, bobot untuk kriteria COST biasanya bernilai negatif 
        # (walaupun di Fuzzy WP sering diselesaikan dengan normalisasi sebelumnya).
        # Karena kita sudah menormalisasi agar COST=BENEFIT (nilai tinggi selalu baik),
        # kita hanya menggunakan bobot positif.
        if t == "cost":
            # Jika ingin sesuai dengan WP klasik (sebelum normalisasi):
            # w_exp[i] *= -1 
            pass # Kita biarkan positif karena sudah dinormalisasi (Min-Max)
            
    # 2. Hitung Vector V (Perkalian TFN berbobot)
    for idx in normal.index:
        # Nilai awal perkalian untuk TFN adalah [1, 1, 1]
        total_tfn = np.array([1.0, 1.0, 1.0])
        for j,col in enumerate(normal.columns):
            tfn_r = tri(normal.loc[idx,col])
            
            # Perkalian berbobot: V = Pi (r^w)
            # Karena r dan w adalah float, kita lakukan operasi ini pada TFN
            # (r_a^w, r_m^w, r_b^w)
            
            # Operasi TFN Pangkat Skalar (Power):
            # [a, m, b]^w = [a^w, m^w, b^w]
            tfn_pangkat = np.power(tfn_r, w_exp[j])

            # Operasi TFN Perkalian (Multiplication):
            # [a1, m1, b1] * [a2, m2, b2] = [a1*a2, m1*m2, b1*b2]
            total_tfn = total_tfn * tfn_pangkat
            
        tfn_total[idx] = total_tfn
        # Defuzzifikasi (Mean) untuk mendapatkan Score V (Vektor S di WP klasik)
        scores.append(total_tfn.mean())
        
    res = pd.DataFrame({"Score":scores}, index=normal.index)
    res["Rank"] = res["Score"].rank(ascending=False).astype(int)
    
    return res, normal, tfn_total

def topsis_calc(df, weights):
    # TOPSIS: Menggunakan nilai Crisp (hasil normalisasi)
    normal = normalize(df)
    mat = normal.astype(float)
    
    # 1. Normalized Vector
    normv = mat.copy()
    for col in mat.columns:
        denom = np.sqrt((mat[col]**2).sum())
        normv[col] = mat[col]/denom if denom!=0 else 0
        
    # 2. Weighted Normalized Matrix
    weighted = normv * weights
    
    # 3. Ideal Solution (A+ dan A-)
    a_plus=[]; a_minus=[]
    for i,col in enumerate(weighted.columns):
        # Karena data sudah dinormalisasi (min-max), A+ selalu max dan A- selalu min 
        # dari nilai berbobot. Kita tetap cek TYPES untuk robustsness.
        if TYPES[i]=="benefit":
            a_plus.append(weighted[col].max()); a_minus.append(weighted[col].min())
        else: # Cost (diasumsikan normalisasi telah membalik nilai)
            a_plus.append(weighted[col].max()); a_minus.append(weighted[col].min())
            
    a_plus=np.array(a_plus); a_minus=np.array(a_minus)
    
    # 4. Separation Distance (Dp dan Dm)
    Dp=[]; Dm=[]
    for idx in weighted.index:
        row = weighted.loc[idx].values
        # Jarak Euclidean: sqrt(Sigma (row - A_ideal)^2)
        dp = np.sqrt(((row - a_plus)**2).sum()); 
        dm = np.sqrt(((row - a_minus)**2).sum())
        Dp.append(dp); Dm.append(dm)
        
    # 5. Closeness Coefficient (CC)
    CC = [dm/(dp+dm) if (dp+dm)!=0 else 0 for dp,dm in zip(Dp,Dm)]
    
    res = pd.DataFrame({"D_plus":Dp,"D_minus":Dm,"CC":CC}, index=weighted.index)
    res["Rank"] = res["CC"].rank(ascending=False).astype(int)
    return res, normal, weighted, a_plus, a_minus

# ----------------------------------------------------------------------
# ---------- Pages ----------
# ----------------------------------------------------------------------

if page=="Home":
    st.header("Ringkasan")
    st.write("Aplikasi membandingkan Fuzzy **Weighted Product (WP)** & **TOPSIS** untuk pemilihan Web Hosting Murah.")
    st.write("Gunakan menu Input Data, lalu jalankan perhitungan WP & TOPSIS.")

# ... Halaman Input Data tetap sama ...
elif page=="Input Data":
    st.header("Input / Edit Data")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic")
    st.session_state.df = edited
    st.download_button("Download data (.csv)", edited.to_csv().encode('utf-8'), file_name="data_input.csv")

# Perubahan: Halaman Fuzzy SAW menjadi Fuzzy WP
elif page=="Fuzzy WP":
    st.header("Hasil Fuzzy Weighted Product (WP)")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_wp, normal, tfn_total = wp_calc(df, ws)
    
    st.subheader("Normalisasi (Min-Max)")
    st.dataframe(normal.style.format("{:.6f}"))
    
    st.subheader("TFN Vektor V agregat (a,m,b) per alternatif")
    tfn_df = pd.DataFrame.from_dict(tfn_total, orient='index', columns=["a","m","b"])
    st.dataframe(tfn_df.style.format("{:.6f}"))
    
    st.subheader("Score V (Defuzzified) & Ranking")
    st.dataframe(res_wp.style.format("{:.6f}"))
    
    # download
    out = pd.concat([df, normal.add_prefix("norm_"), tfn_df.add_prefix("TFN_"), res_wp], axis=1)
    buf = BytesIO()
    out.to_excel(buf, index=True, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download hasil WP (.xlsx)", data=buf, file_name="hasil_wp.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
# ... Halaman Fuzzy TOPSIS tetap sama ...
elif page=="Fuzzy TOPSIS":
    st.header("Hasil TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_top, normal, weighted, a_plus, a_minus = topsis_calc(df, ws)
    st.subheader("Weighted normalized")
    st.dataframe(weighted.style.format("{:.6f}"))
    st.subheader("Hasil TOPSIS (CC & ranking)")
    st.dataframe(res_top.style.format("{:.6f}"))
    # download
    buf = BytesIO()
    res_top.to_excel(buf, index=True, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download hasil TOPSIS (.xlsx)", data=buf, file_name="hasil_topsis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Perubahan: Halaman Perbandingan (SAW vs TOPSIS) menjadi (WP vs TOPSIS)
elif page=="Perbandingan":
    st.header("Perbandingan WP vs TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_wp, _, _ = wp_calc(df, ws) # Menggunakan wp_calc
    res_top, _, _, _, _ = topsis_calc(df, ws)
    compare = pd.DataFrame({"WP":res_wp["Score"], "TOPSIS":res_top["CC"]})
    st.dataframe(compare.style.format("{:.6f}"))
    
    fig,ax = plt.subplots(figsize=(8,4))
    compare.plot(kind='bar', ax=ax)
    ax.set_ylabel("Score")
    st.pyplot(fig)
    
    top_wp = compare["WP"].idxmax(); top_top = compare["TOPSIS"].idxmax()
    if top_wp == top_top:
        st.success(f"Kedua metode memilih: **{top_wp}**")
    else:
        st.info(f"WP -> **{top_wp}**, TOPSIS -> **{top_top}**")

# ... Halaman Tentang tetap sama ...
elif page=="Tentang":
    st.header("Tentang")
    st.write("Aplikasi untuk Projek MK Logika Fuzzy — Fuzzy WP & TOPSIS. Dibuat untuk memilih Web Hosting Murah.")
