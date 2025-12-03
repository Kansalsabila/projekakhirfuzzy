# app_singlefile.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Fuzzy WP & TOPSIS - Layanan Hosting ", layout="wide")

st.title("Fuzzy WP & TOPSIS — Layanan Web Hosting Murah")

# ---------- config ----------
PROVIDERS = ["HostA","HostB","HostC","HostD","HostE"]
default_df = pd.DataFrame({
    "Cost":[100,80,80,80,100],
    "Storage":[60,80,100,60,100],
    "Bandwidth":[60,100,60,80,80],
    "Uptime":[80,80,100,60,60],
    "Support":[80,60,60,100,80]
}, index=PROVIDERS)

if "df" not in st.session_state:
    st.session_state.df = default_df.copy()

st.sidebar.header("Menu")
page = st.sidebar.radio("Pilih halaman", ["Home","Input Data","Fuzzy SAW","Fuzzy TOPSIS","Perbandingan","Tentang"])

# weights (editable)
st.sidebar.markdown("### Bobot Kriteria")
w1 = st.sidebar.slider("Cost (w1)", 0.0, 1.0, 0.25, 0.01)
w2 = st.sidebar.slider("Settlement (w2)", 0.0, 1.0, 0.30, 0.01)
w3 = st.sidebar.slider("Security (w3)", 0.0, 1.0, 0.20, 0.01)
w4 = st.sidebar.slider("API ease (w4)", 0.0, 1.0, 0.15, 0.01)
w5 = st.sidebar.slider("Features (w5)", 0.0, 1.0, 0.10, 0.01)
# normalize weights
ws = np.array([w1,w2,w3,w4,w5])
if ws.sum() == 0:
    ws = np.array([0.25,0.30,0.20,0.15,0.10])
else:
    ws = ws / ws.sum()

TYPES = ["cost","benefit","benefit","benefit","benefit"]

def normalize_saw(df):
    res = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for i,col in enumerate(df.columns):
        if TYPES[i]=="benefit":
            res[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            res[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    return res

def tri(v):
    a = max(0, v-0.1); m = v; b = min(1, v+0.1)
    return np.array([a,m,b])

def saw_calc(df, weights):
    normal = normalize_saw(df)
    tfn_total = {}
    scores=[]
    for idx in normal.index:
        total = np.array([0.0,0.0,0.0])
        for j,col in enumerate(normal.columns):
            t = tri(normal.loc[idx,col])
            total += t * weights[j]
        tfn_total[idx] = total
        scores.append(total.mean())
    res = pd.DataFrame({"Score":scores}, index=normal.index)
    res["Rank"] = res["Score"].rank(ascending=False).astype(int)
    return res, normal, tfn_total

def topsis_calc(df, weights):
    # use defuzzified values = normalized values
    normal = normalize_saw(df)
    mat = normal.astype(float)
    normv = mat.copy()
    for col in mat.columns:
        denom = np.sqrt((mat[col]**2).sum())
        normv[col] = mat[col]/denom if denom!=0 else 0
    weighted = normv * weights
    # ideal
    a_plus=[]; a_minus=[]
    for i,col in enumerate(weighted.columns):
        if TYPES[i]=="benefit":
            a_plus.append(weighted[col].max()); a_minus.append(weighted[col].min())
        else:
            a_plus.append(weighted[col].min()); a_minus.append(weighted[col].max())
    a_plus=np.array(a_plus); a_minus=np.array(a_minus)
    Dp=[]; Dm=[]
    for idx in weighted.index:
        row = weighted.loc[idx].values
        dp = np.sqrt(((row - a_plus)*2).sum()); dm = np.sqrt(((row - a_minus)*2).sum())
        Dp.append(dp); Dm.append(dm)
    CC = [dm/(dp+dm) if (dp+dm)!=0 else 0 for dp,dm in zip(Dp,Dm)]
    res = pd.DataFrame({"D_plus":Dp,"D_minus":Dm,"CC":CC}, index=weighted.index)
    res["Rank"] = res["CC"].rank(ascending=False).astype(int)
    return res, normal, weighted, a_plus, a_minus

# ---------- Pages ----------
if page=="Home":
    st.header("Ringkasan")
    st.write("Aplikasi membandingkan Fuzzy SAW & TOPSIS untuk pemilihan Payment Gateway (UMKM).")
    st.write("Gunakan menu Input Data, lalu jalankan perhitungan SAW & TOPSIS.")
elif page=="Input Data":
    st.header("Input / Edit Data")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic")
    st.session_state.df = edited
    st.download_button("Download data (.csv)", edited.to_csv().encode('utf-8'), file_name="data_input.csv")
elif page=="Fuzzy SAW":
    st.header("Hasil Fuzzy SAW")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_saw, normal, tfn_total = saw_calc(df, ws)
    st.subheader("Normalisasi")
    st.dataframe(normal.style.format("{:.6f}"))
    st.subheader("TFN agregat (a,m,b) per provider")
    tfn_df = pd.DataFrame.from_dict(tfn_total, orient='index', columns=["a","m","b"])
    st.dataframe(tfn_df.style.format("{:.6f}"))
    st.subheader("Score & Ranking (defuzzified)")
    st.dataframe(res_saw.style.format("{:.6f}"))
    # download
    out = pd.concat([df, normal.add_prefix("norm_"), tfn_df, res_saw], axis=1)
    buf = BytesIO()
    out.to_excel(buf, index=True, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download hasil SAW (.xlsx)", data=buf, file_name="hasil_saw.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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
elif page=="Perbandingan":
    st.header("Perbandingan SAW vs TOPSIS")
    df = st.session_state.df.copy().apply(pd.to_numeric)
    res_saw, _, _ = saw_calc(df, ws)
    res_top, _, _, _, _ = topsis_calc(df, ws)
    compare = pd.DataFrame({"SAW":res_saw["Score"], "TOPSIS":res_top["CC"]})
    st.dataframe(compare.style.format("{:.6f}"))
    fig,ax = plt.subplots(figsize=(8,4))
    compare.plot(kind='bar', ax=ax)
    ax.set_ylabel("Score")
    st.pyplot(fig)
    top_saw = compare["SAW"].idxmax(); top_top = compare["TOPSIS"].idxmax()
    if top_saw == top_top:
        st.success(f"Kedua metode memilih: {top_saw}")
    else:
        st.info(f"SAW -> {top_saw}, TOPSIS -> {top_top}")
elif page=="Tentang":
    st.header("Tentang")
    st.write("Aplikasi untuk Projek MK Logika Fuzzy — Fuzzy SAW & TOPSIS. Dibuat untuk memilih Payment Gateway (UMKM).")
