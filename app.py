# -*- coding: utf-8 -*-
"""
Dashboard Segmentasi Pola Penjualan Produk Mingguan
Metode : Analisis klaster K-Medoids dengan ukuran jarak Dynamic Time Warping (DTW)
Pembanding : Klaster hierarki (average linkage)
Data : 811 produk x 52 minggu (Sales Transactions Weekly)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

# =====================================================================
# KONFIGURASI HALAMAN & TEMA
# =====================================================================
st.set_page_config(
    page_title="Segmentasi Pola Penjualan Produk",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA = BASE / "data"
WEEKS = [f"W{i}" for i in range(52)]

# Kandidat lokasi file data: folder data/, akar repo, lalu subfolder apa pun.
# Ini membuat app tetap jalan walau folder "data" tidak ikut terunggah dan file CSV
# berada sejajar dengan app.py di akar repositori.
_SEARCH_DIRS = [DATA, BASE, BASE / "dashboard_segmentasi" / "data", BASE / "dashboard_segmentasi"]


def find_file(name: str):
    """Cari file data di beberapa lokasi; kembalikan path pertama yang ada, atau None."""
    for d in _SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    hits = sorted(BASE.rglob(name))
    return hits[0] if hits else None

# Warna tema (biru-putih)
NAVY = "#0F2A4A"
BLUE = "#1E5AA8"
BLUE_2 = "#3B82F6"
BLUE_SOFT = "#EEF5FF"
INK_2 = "#52627A"
GRID = "#E6EDF7"

# Warna klaster: urutan kategori tetap (sudah diuji aman buta warna untuk pasangan berdekatan)
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLUSTER_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "x"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* =====================================================================
   PAKSA MODE TERANG
   Tanpa .streamlit/config.toml, Streamlit Cloud bisa memakai tema gelap
   bawaan, sehingga teks menjadi putih dan tidak terbaca di latar terang.
   Blok ini mengunci warna teks dan latar agar tampilan konsisten terang,
   apa pun tema browser atau akun pengguna.
   ===================================================================== */
html, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { color-scheme: light !important; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"],
.stRadio label p, [data-baseweb="radio"] div, [data-testid="stSidebar"] * {
    color: #0F2A4A;
}
/* kembalikan warna putih untuk elemen yang memang berlatar gelap/berwarna */
.hero, .hero *, .chip, .ccard .tag, .step .n,
[aria-selected="true"] p, [data-testid="stSidebar"] .sbx-good *,
div[style*="background:#1E5AA8"] * { color: inherit; }
/* tombol radio & slider: ikuti warna tema biru, bukan merah bawaan */
input[type="radio"], input[type="checkbox"], input[type="range"] { accent-color: #1E5AA8 !important; }
/* lingkaran radio terpilih (struktur: label > div > div pertama) */
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
    background-color: #1E5AA8 !important; border-color: #1E5AA8 !important;
}
[data-testid="stRadioOption"] > div > div:first-child { border-color: #9EC5F4 !important; background-color: #FFFFFF !important; }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { background-color: #1E5AA8 !important; }
[data-baseweb="radio"] [aria-checked="true"] > div:first-child { background-color: #1E5AA8 !important; border-color: #1E5AA8 !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #1E5AA8 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] ~ div { background: #1E5AA8 !important; }

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp { background: linear-gradient(180deg, #F2F7FF 0%, #FFFFFF 38%, #F7FAFF 100%); }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1350px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8F1FF 0%, #F6FAFF 100%);
    border-right: none;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
[data-testid="stSidebar"] .stRadio > div, [data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSlider > div { border-radius: 14px; }
div[data-baseweb="select"] > div { border-radius: 14px !important; }

/* ---------- Hero ---------- */
.hero {
    background: radial-gradient(circle at 85% 20%, rgba(255,255,255,.18) 0, rgba(255,255,255,0) 40%),
                linear-gradient(120deg, #123C78 0%, #1E5AA8 45%, #3B82F6 100%);
    border-radius: 28px; padding: 30px 36px; color: #fff;
    box-shadow: 0 14px 34px rgba(30,90,168,.25); margin-bottom: 18px;
}
.hero h1 { color: #fff; font-size: 2.0rem; font-weight: 800; margin: 0 0 6px 0; line-height: 1.2; }
.hero p { color: #DCE9FF; font-size: 1.0rem; margin: 0; max-width: 900px; }
.hero .chips { margin-top: 14px; }
.hero .chip {
    display: inline-block; background: rgba(255,255,255,.16); color: #fff;
    border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
    padding: 5px 14px; margin: 4px 6px 0 0; font-size: .82rem; font-weight: 600;
}

/* ---------- Kartu ---------- */
.card {
    background: #FFFFFF; border: 1px solid #E3EDFA; border-radius: 22px;
    padding: 18px 22px; box-shadow: 0 6px 20px rgba(30,90,168,.07); height: 100%;
}
.kpi { text-align: left; min-height: 138px; }
.kpi .lab { color: #5B6B82; font-size: .80rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.kpi .val { color: #0F2A4A; font-size: 1.75rem; font-weight: 800; margin-top: 4px; line-height: 1.1; }
.kpi .sub { color: #6B7A90; font-size: .80rem; margin-top: 4px; }
.kpi .ico {
    width: 38px; height: 38px; border-radius: 12px; background: #EEF5FF;
    display: flex; align-items: center; justify-content: center; font-size: 1.15rem; float: right;
}

.ccard { background: #fff; border: 1px solid #E3EDFA; border-radius: 24px; padding: 20px 22px;
         box-shadow: 0 8px 24px rgba(30,90,168,.08); height: 100%; }
.ccard .tag { display: inline-flex; align-items: center; gap: 8px; font-weight: 800; color: #0F2A4A; font-size: 1.12rem; }
.ccard .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
.ccard .seg { color: #1E5AA8; font-weight: 700; margin: 2px 0 12px 0; font-size: .98rem; }
.ccard .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
.ccard .mini { background: #F4F8FF; border-radius: 14px; padding: 9px 12px; }
.ccard .mini .l { font-size: .72rem; color: #5B6B82; font-weight: 600; }
.ccard .mini .v { font-size: 1.05rem; color: #0F2A4A; font-weight: 800; }
.ccard h5 { color: #0F2A4A; font-size: .9rem; margin: 10px 0 4px 0; font-weight: 700; }
.ccard ul { margin: 0 0 0 1.05rem; padding: 0; color: #33445C; font-size: .88rem; }
.ccard li { margin-bottom: 3px; }

/* ---------- Kotak keterangan ---------- */
.box { border-radius: 18px; padding: 14px 18px; margin: 6px 0 14px 0; font-size: .92rem; line-height: 1.55; }
.box b.t { display: block; margin-bottom: 4px; font-size: .95rem; }
.box-read { background: #F1F6FF; color: #1D3557; border: 1px solid #DCE8FB; }
.box-ins  { background: #EAF4FF; color: #0F2A4A; border: 1px solid #C9DEFA; }
.box-good { background: #EAF8F1; color: #0E5132; border: 1px solid #C8EBD8; }
.box-warn { background: #FFF6E5; color: #6B4500; border: 1px solid #F6DFB0; }
.box-bad  { background: #FDEEEE; color: #7A1F1F; border: 1px solid #F5CFCF; }
.box ul { margin: 4px 0 0 1.1rem; padding: 0; }

.sec-title { color: #0F2A4A; font-weight: 800; font-size: 1.35rem; margin: 18px 0 2px 0; }
.sec-sub { color: #5B6B82; font-size: .92rem; margin-bottom: 10px; }

.step { background: #fff; border: 1px solid #E3EDFA; border-radius: 20px; padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(30,90,168,.06); height: 100%; }
.step .n { width: 30px; height: 30px; border-radius: 50%; background: #1E5AA8; color: #fff;
           display: flex; align-items: center; justify-content: center; font-weight: 800; margin-bottom: 8px; }
.step .h { font-weight: 700; color: #0F2A4A; font-size: .95rem; }
.step .d { color: #52627A; font-size: .84rem; margin-top: 4px; }

.sbx { border-radius: 16px; padding: 10px 13px; margin: 2px 0 10px 0; font-size: .84rem; line-height: 1.45; }
.sbx span { font-size: .78rem; }
.sbx-good { background: #EAF8F1; color: #0E5132; border: 1px solid #C8EBD8; }
.sbx-warn { background: #FFF6E5; color: #6B4500; border: 1px solid #F6DFB0; }
.sbx-info { background: #FFFFFF; color: #1D3557; border: 1px solid #DCE8FB; }
.pill { display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: .78rem; font-weight: 700; }

/* ---------- Tab berbentuk pil ---------- */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #EAF2FF; padding: 6px; border-radius: 22px; flex-wrap: wrap;
    border: none !important; box-shadow: inset 0 0 0 1px #DCE8FB;
}
.stTabs [role="tab"], .stTabs [data-baseweb="tab"] {
    height: 40px; border-radius: 999px !important; padding: 0 12px !important; background: transparent;
    color: #1E5AA8; font-weight: 700; border: none !important; display: flex; align-items: center;
    transition: background .2s ease;
}
.stTabs [role="tab"]:hover { background: #DCE9FF; }
.stTabs [role="tab"] p { font-weight: 700; color: #1E5AA8; font-size: .88rem; }
.stTabs [role="tab"][aria-selected="true"] { background: #1E5AA8 !important; box-shadow: 0 4px 12px rgba(30,90,168,.3); }
.stTabs [role="tab"][aria-selected="true"] p { color: #fff !important; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
[role="tabpanel"] { padding-top: 10px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: .6; }

/* ---------- Grafik, tabel, expander ---------- */
[data-testid="stPlotlyChart"] {
    background: #fff; border: 1px solid #E3EDFA; border-radius: 22px; padding: 8px 8px 2px 8px;
    box-shadow: 0 6px 20px rgba(30,90,168,.06); overflow: hidden;
}
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 18px; overflow: hidden; border: 1px solid #E3EDFA; }
[data-testid="stExpander"] details { border-radius: 18px !important; border: 1px solid #DCE8FB !important; background: #fff; }
[data-testid="stExpander"] summary { border-radius: 18px !important; }
.stButton > button, .stDownloadButton > button {
    border-radius: 999px !important; background: #1E5AA8 !important; color: #fff !important;
    border: none !important; padding: .5rem 1.4rem !important; font-weight: 700 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #123C78 !important; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #E3EDFA; border-radius: 20px; padding: 14px 18px; }
hr { border-color: #E3EDFA; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# FUNGSI BANTU TAMPILAN
# =====================================================================
def box(kind: str, title: str, body: str):
    """Kotak keterangan berujung lengkung. kind: read | ins | good | warn | bad"""
    st.markdown(f'<div class="box box-{kind}"><b class="t">{title}</b>{body}</div>', unsafe_allow_html=True)


def read_box(body: str):
    box("read", "📖 Cara membaca grafik", body)


def ins_box(body: str):
    box("ins", "💡 Interpretasi", body)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, icon, label, value, sub=""):
    col.markdown(
        f'<div class="card kpi"><div class="ico">{icon}</div><div class="lab">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def fmt(x, d=0):
    """Format angka gaya Indonesia: titik ribuan, koma desimal."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, d=1):
    return f"{fmt(x, d)}%"


def style_fig(fig, height=380, title=None, legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0.01, font=dict(size=15, color=NAVY)) if title else None,
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#33445C"),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#C9DEFA", font=dict(color=NAVY)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    if title and legend:
        fig.update_layout(margin=dict(t=80), legend=dict(y=1.0, yanchor="bottom"), title=dict(y=0.97, yanchor="top"))
    fig.update_xaxes(showgrid=False, linecolor="#C9D6E8", ticks="outside", tickcolor="#C9D6E8")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#C9D6E8")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# =====================================================================
# FUNGSI STATISTIK
# =====================================================================
def minmax_rows(X):
    """Normalisasi min-maks per produk: (x - min) / (maks - min)."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / rng


def dtw_matrix(X, block=60000):
    """Matriks jarak DTW (Dynamic Time Warping) antarproduk, dihitung berpasangan."""
    n, T = X.shape
    iu, ju = np.triu_indices(n, 1)
    out = np.empty(len(iu))
    for s in range(0, len(iu), block):
        a = X[iu[s:s + block]].astype(np.float32)
        b = X[ju[s:s + block]].astype(np.float32)
        m = len(a)
        prev = np.full((m, T + 1), np.inf, np.float32)
        prev[:, 0] = 0
        for i in range(1, T + 1):
            cur = np.full((m, T + 1), np.inf, np.float32)
            for j in range(1, T + 1):
                c = (a[:, i - 1] - b[:, j - 1]) ** 2
                cur[:, j] = c + np.minimum(np.minimum(prev[:, j], prev[:, j - 1]), cur[:, j - 1])
            prev = cur
        out[s:s + block] = np.sqrt(prev[:, T])
    return squareform(out)


def k_medoids(D, k, max_iter=100):
    """K-Medoids metode PAM (tahap BUILD lalu SWAP) pada matriks jarak D."""
    n = len(D)
    med = [int(np.argmin(D.sum(1)))]
    for _ in range(1, k):
        dmin = D[:, med].min(1)
        gain = np.maximum(dmin[:, None] - D, 0).sum(0)
        gain[med] = -1
        med.append(int(np.argmax(gain)))
    med = np.array(med)
    cost = D[:, med].min(1).sum()
    for _ in range(max_iter):
        best = (cost, None, None)
        for mi in range(k):
            for h in range(n):
                if h in med:
                    continue
                trial = med.copy()
                trial[mi] = h
                c = D[:, trial].min(1).sum()
                if c < best[0] - 1e-9:
                    best = (c, mi, h)
        if best[1] is None:
            break
        med[best[1]] = best[2]
        cost = best[0]
    labels = np.argmin(D[:, med], 1)
    return labels, med, cost


def silhouette_label(s):
    """Kategori koefisien Silhouette menurut Kaufman & Rousseeuw (1990)."""
    if s > 0.70:
        return "Kuat", "good", "Struktur klaster kuat: produk dalam satu klaster sangat mirip dan klaster terpisah jelas."
    if s > 0.50:
        return "Wajar", "good", "Struktur klaster wajar: pemisahan antarklaster cukup jelas."
    if s > 0.25:
        return "Lemah", "warn", ("Struktur klaster lemah: pengelompokan ada tetapi batasnya tidak tegas. "
                                 "Sebagian produk berada di perbatasan dua klaster, sehingga hasil dipakai sebagai "
                                 "panduan umum, bukan batas mutlak.")
    return "Tidak substansial", "bad", ("Tidak ditemukan struktur klaster yang berarti. Coba jumlah klaster atau ukuran "
                                        "jarak lain sebelum hasil dipakai untuk keputusan.")


# =====================================================================
# MEMUAT DATA & PERHITUNGAN (DISIMPAN DALAM CACHE)
# =====================================================================
@st.cache_data(show_spinner=False)
def load_data():
    pa = find_file("EDA_Features_Sales_All.csv")
    pf = find_file("EDA_Features_Sales_Filtered.csv")
    missing = [n for n, p in [("EDA_Features_Sales_All.csv", pa),
                              ("EDA_Features_Sales_Filtered.csv", pf)] if p is None]
    if missing:
        listing = "\n".join(
            f"- `{d.relative_to(BASE) if d == BASE else d.name}/`: "
            + (", ".join(sorted(x.name for x in d.iterdir())) if d.exists() else "(folder tidak ada)")
            for d in [BASE, DATA])
        st.error(
            "### ❌ File data tidak ditemukan\n"
            f"Dashboard membutuhkan file berikut, tetapi tidak menemukannya di repositori: "
            f"**{', '.join(missing)}**.\n\n"
            "**Penyebab umum:** folder `data/` tidak ikut terunggah ke GitHub, atau file CSV terunggah "
            "ke lokasi yang salah.\n\n"
            "**Cara memperbaiki:** pastikan struktur repositori seperti ini — `app.py` di akar, dan file "
            "data di dalam folder `data/`:\n"
            "```\n"
            "app.py\n"
            "requirements.txt\n"
            "data/\n"
            "  ├─ EDA_Features_Sales_All.csv\n"
            "  ├─ EDA_Features_Sales_Filtered.csv\n"
            "  ├─ jarak_dtw.npy\n"
            "  └─ jarak_euklidean.npy\n"
            "```\n"
            "Saat mengunggah di GitHub (*Add file → Upload files*), seret **folder `data`** secara utuh, "
            "bukan hanya file-file di dalamnya.\n\n"
            "Isi repositori yang terbaca sekarang:\n" + listing)
        st.stop()
    a = pd.read_csv(pa)
    f = pd.read_csv(pf)
    for d in (a, f):
        d.columns = d.columns.str.strip()
    return a.copy(), f.copy()


@st.cache_data(show_spinner="Menghitung matriks jarak (hanya sekali)...")
def full_distance(kind: str):
    """Matriks jarak untuk seluruh 811 produk (urutan sesuai file All)."""
    a, _ = load_data()
    path = find_file("jarak_dtw.npy" if kind == "DTW" else "jarak_euklidean.npy")
    if path is not None:
        D = np.load(path).astype(float)
        if D.shape == (len(a), len(a)):
            return D
    Xn = minmax_rows(a[WEEKS].to_numpy(float))
    return dtw_matrix(Xn) if kind == "DTW" else squareform(pdist(Xn, "euclidean"))


@st.cache_data(show_spinner=False)
def subset(dataset: str, kind: str):
    a, f = load_data()
    df = a if dataset == "all" else f
    pos = pd.Series(np.arange(len(a)), index=a["Product_Code"])
    idx = pos.loc[df["Product_Code"]].to_numpy()
    D = full_distance(kind)[np.ix_(idx, idx)]
    return df.reset_index(drop=True), D


def order_by_sales(labels, df):
    """Penomoran ulang klaster: Klaster 1 = median total penjualan tertinggi."""
    med = pd.Series(df["Total_Sales"].to_numpy()).groupby(labels).median().sort_values(ascending=False)
    mapping = {old: new + 1 for new, old in enumerate(med.index)}
    return np.array([mapping[x] for x in labels])


@st.cache_data(show_spinner="Menjalankan analisis klaster...")
def run_kmedoids(dataset, kind, k):
    df, D = subset(dataset, kind)
    lab, med, cost = k_medoids(D, k)
    new = order_by_sales(lab, df)
    medoids = {int(new[m]): int(m) for m in med}
    sil = silhouette_samples(D, new, metric="precomputed")
    return new, medoids, float(cost), sil


@st.cache_data(show_spinner=False)
def run_hier(dataset, kind, k):
    df, D = subset(dataset, kind)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=k, criterion="maxclust")
    return order_by_sales(lab, df), Z


@st.cache_data(show_spinner="Membandingkan jumlah klaster 2 sampai 6...")
def k_table(dataset, kind):
    _, D = subset(dataset, kind)
    rows = []
    for k in range(2, 7):
        lab, _, cost, _ = run_kmedoids(dataset, kind, k)
        hl, _ = run_hier(dataset, kind, k)
        rows.append({
            "k": k,
            "Silhouette K-Medoids": silhouette_score(D, lab, metric="precomputed"),
            "Silhouette Hierarki": silhouette_score(D, hl, metric="precomputed"),
            "Total jarak ke medoid": cost,
            "Ukuran klaster terkecil (Hierarki)": int(np.bincount(hl)[1:].min()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def classical_mds(dataset, kind):
    """Penskalaan multidimensi klasik (Torgerson) ke 2 dimensi."""
    _, D = subset(dataset, kind)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    coords = v[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    prop = w[:2] / w[w > 0].sum()
    return coords, prop


DB_MS = [3, 4, 5, 6, 8, 10]      # nilai min_samples yang dicoba
DB_MAX_NOISE = 30                 # batas proporsi pencilan (%) agar kombinasi dianggap layak
NOISE_COLOR = "#9AA9BF"


def kdist_sorted(D, ms):
    """Jarak tiap produk ke tetangga terdekat ke-(min_samples - 1), diurutkan naik.
    Produk menjadi titik inti DBSCAN jika nilai ini <= eps."""
    return np.sort(np.sort(D, axis=1)[:, ms - 1])


def knee_point(y):
    """Titik siku: titik dengan jarak terjauh dari garis lurus yang menghubungkan ujung kurva."""
    x = np.linspace(0, 1, len(y))
    yn = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    i = int(np.argmax(x - yn))
    return i, float(y[i])


@st.cache_data(show_spinner="Mencari parameter DBSCAN terbaik...")
def dbscan_grid(dataset, kind):
    """Coba kombinasi min_samples x eps; pilih Silhouette tertinggi di antara kombinasi yang layak."""
    _, D = subset(dataset, kind)
    S = np.sort(D, axis=1)
    lo = np.quantile(S[:, DB_MS[0] - 1], 0.30)
    hi = np.quantile(S[:, DB_MS[-1] - 1], 0.97)
    eps_grid = np.round(np.linspace(lo, hi, 16), 3)
    rows = []
    for ms in DB_MS:
        for eps in eps_grid:
            lab = DBSCAN(eps=float(eps), min_samples=ms, metric="precomputed").fit_predict(D)
            m = lab != -1
            nc = len(set(lab[m]))
            s_ = silhouette_score(D[np.ix_(m, m)], lab[m], metric="precomputed") if nc >= 2 else np.nan
            sizes = np.bincount(lab[m]) if nc else np.array([0])
            rows.append({"min_samples": ms, "eps": float(eps), "Jumlah klaster": nc,
                         "Pencilan (%)": (~m).mean() * 100, "Silhouette": s_,
                         "Klaster terbesar (%)": sizes.max() / len(lab) * 100,
                         "Klaster terkecil": int(sizes[sizes > 0].min()) if nc else 0})
    g = pd.DataFrame(rows)
    g["Layak"] = (g["Jumlah klaster"] >= 2) & (g["Pencilan (%)"] <= DB_MAX_NOISE) & (g["Silhouette"] > 0)
    ok = g[g["Layak"]]
    if len(ok):
        b = ok.sort_values(["Silhouette", "Pencilan (%)"], ascending=[False, True]).iloc[0]
        best = {"eps": float(b["eps"]), "ms": int(b["min_samples"]), "found": True}
    else:
        _, e = knee_point(kdist_sorted(D, 5))
        best = {"eps": round(e, 3), "ms": 5, "found": False}
    return g, best


@st.cache_data(show_spinner=False)
def run_dbscan(dataset, kind, eps, ms):
    """DBSCAN pada matriks jarak. Klaster dinomori ulang (1 = median total penjualan tertinggi); -1 = pencilan."""
    df, D = subset(dataset, kind)
    lab = DBSCAN(eps=max(float(eps), 1e-6), min_samples=int(ms), metric="precomputed").fit_predict(D)
    m = lab != -1
    out = np.full(len(lab), -1)
    if m.any():
        out[m] = order_by_sales(lab[m], df[m].reset_index(drop=True))
    nc = len(set(out[m]))
    s_ = float(silhouette_score(D[np.ix_(m, m)], out[m], metric="precomputed")) if nc >= 2 else float("nan")
    return out, nc, s_


def db_name(c):
    if c == -1:
        return "Pencilan"
    return f"Klaster D{c}" if c <= 6 else "Klaster D7+ (lainnya)"


def db_color(name):
    if name == "Pencilan":
        return NOISE_COLOR
    if name.startswith("Klaster D7+"):
        return "#5B6B82"
    return CLUSTER_COLORS[int(name.split("D")[1]) - 1]


def ari_text(a):
    return "kesepakatan rendah" if a < 0.3 else "kesepakatan sedang" if a < 0.65 else "kesepakatan tinggi"


def name_segment(row, q_hi):
    """Nama segmen berdasarkan statistik median klaster (tanpa istilah teknis)."""
    if row["zero"] >= 75:
        base = "Sangat Jarang Terjual"
    elif row["zero"] >= 50:
        base = "Jarang Terjual"
    elif row["cv"] < 0.5:
        base = "Laris & Stabil" if row["mean"] >= q_hi else "Rutin & Stabil"
    elif row["cv"] <= 1.0:
        base = "Cukup Berfluktuasi"
    else:
        base = "Sangat Berfluktuasi"
    return base


TRAITS = {
    "Laris & Stabil": [
        "Terjual hampir setiap minggu dengan volume tinggi.",
        "Koefisien variasi rendah: naik-turun penjualan kecil dibanding rata-ratanya.",
    ],
    "Rutin & Stabil": [
        "Terjual hampir setiap minggu dengan volume sedang.",
        "Koefisien variasi rendah: penjualan mudah diperkirakan.",
    ],
    "Cukup Berfluktuasi": [
        "Penjualan rutin tetapi naik-turunnya cukup besar.",
        "Sesekali ada minggu tanpa penjualan.",
    ],
    "Sangat Berfluktuasi": [
        "Penjualan tidak menentu, simpangan baku lebih besar dari rata-rata.",
        "Banyak minggu dengan penjualan rendah atau nol.",
    ],
    "Sangat Jarang Terjual": [
        "Sekitar tiga perempat minggu atau lebih tidak ada penjualan.",
        "Penjualan sangat kecil dan tidak beraturan; sering hanya beberapa unit setahun.",
    ],
    "Jarang Terjual": [
        "Lebih dari separuh minggu tidak ada penjualan sama sekali.",
        "Penjualan muncul sesekali dalam jumlah kecil (pola terputus-putus).",
    ],
}
ACTIONS = {
    "Sangat Jarang Terjual": [
        "Jangan menyimpan stok; sediakan hanya berdasarkan pesanan.",
        "Tinjau kelayakan produk: pertahankan, gabungkan dalam paket, atau hentikan.",
        "Bila tetap dijual, tawarkan bersama produk laris.",
    ],
    "Laris & Stabil": [
        "Prioritaskan ketersediaan stok dan pengadaan rutin.",
        "Tetapkan stok pengaman agar tidak kehabisan barang.",
        "Jadikan acuan perencanaan karena polanya paling dapat diandalkan.",
    ],
    "Rutin & Stabil": [
        "Pengadaan berkala dengan jumlah tetap.",
        "Uji promosi ringan untuk menaikkan volume.",
    ],
    "Cukup Berfluktuasi": [
        "Pantau penjualan mingguan dan sesuaikan stok secara fleksibel.",
        "Gunakan promosi saat penjualan menurun.",
    ],
    "Sangat Berfluktuasi": [
        "Hindari stok besar; pesan dalam jumlah kecil tetapi sering.",
        "Telusuri penyebab lonjakan (musim, promosi, pesanan khusus).",
    ],
    "Jarang Terjual": [
        "Kurangi stok; sediakan berdasarkan pesanan.",
        "Pertimbangkan paket bundel dengan produk laris.",
        "Evaluasi apakah produk tetap layak dipertahankan.",
    ],
}


def profile_table(df, labels, medoids):
    d = df.assign(K=labels)
    tot_units = d["Total_Sales"].sum()
    rows = []
    for k, g in d.groupby("K"):
        h1 = g[WEEKS[:26]].to_numpy().sum()
        h2 = g[WEEKS[26:]].to_numpy().sum()
        rows.append({
            "K": k,
            "n": len(g),
            "share_prod": len(g) / len(d) * 100,
            "share_units": g["Total_Sales"].sum() / tot_units * 100,
            "mean": g["Mean_Sales"].median(),
            "total_med": g["Total_Sales"].median(),
            "cv": g["CV_Sales"].median(),
            "zero": g["Zero_Sales_Pct"].median(),
            "trend": (h2 - h1) / h1 * 100 if h1 > 0 else np.nan,
            "medoid": df.loc[medoids[k], "Product_Code"],
        })
    p = pd.DataFrame(rows).set_index("K")
    q_hi = df["Mean_Sales"].quantile(2 / 3)
    p["segmen"] = p.apply(lambda r: name_segment(r, q_hi), axis=1)
    return p


def trend_text(t):
    if np.isnan(t):
        return "–"
    if t >= 10:
        return f"naik {fmt(t, 1)}%"
    if t <= -10:
        return f"turun {fmt(abs(t), 1)}%"
    return f"relatif datar ({'+' if t >= 0 else '−'}{fmt(abs(t), 1)}%)"


# =====================================================================
# SIDEBAR
# =====================================================================
all_df, filt_df = load_data()

with st.sidebar:
    st.markdown(
        '<div style="background:#1E5AA8;color:#fff;border-radius:20px;padding:14px 16px;margin-bottom:14px;">'
        '<div style="font-weight:800;font-size:1.05rem;">⚙️ Pengaturan Analisis</div>'
        '<div style="font-size:.8rem;color:#DCE9FF;">Ubah pilihan di bawah; seluruh grafik ikut diperbarui.</div></div>',
        unsafe_allow_html=True,
    )
    ds_label = st.radio(
        "Data produk",
        [f"Semua produk ({len(all_df)})", f"Produk terfilter ({len(filt_df)})"],
        help="Data terfilter membuang produk yang tidak terjual lebih dari 45 dari 52 minggu.",
    )
    dataset = "all" if ds_label.startswith("Semua") else "filt"

    kind_label = st.radio(
        "Ukuran jarak antarpola",
        ["DTW (disarankan)", "Euklidean"],
        help=("DTW membandingkan bentuk pola penjualan dan memperbolehkan pergeseran waktu. "
              "Euklidean membandingkan minggu demi minggu secara kaku."),
    )
    kind = "DTW" if kind_label.startswith("DTW") else "Euklidean"

    ktab = k_table(dataset, kind)
    best_k = int(ktab.loc[ktab["Silhouette K-Medoids"].idxmax(), "k"])
    sil_best = float(ktab["Silhouette K-Medoids"].max())
    k = st.slider("Jumlah klaster K-Medoids (k)", 2, 6, best_k,
                  help=f"Nilai awal k = {best_k} adalah k dengan koefisien Silhouette K-Medoids tertinggi.")
    if k == best_k:
        st.markdown(
            f'<div class="sbx sbx-good">✅ <b>k optimal = {best_k}</b><br>'
            f'<span>Silhouette K-Medoids tertinggi ({fmt(sil_best, 3)}) untuk data dan ukuran jarak yang dipilih. '
            f'Anda sedang memakai k optimal.</span></div>', unsafe_allow_html=True)
    else:
        sil_k = float(ktab.set_index("k").loc[k, "Silhouette K-Medoids"])
        st.markdown(
            f'<div class="sbx sbx-warn">⭐ <b>k optimal = {best_k}</b> (Silhouette {fmt(sil_best, 3)})<br>'
            f'<span>Anda memilih k = {k} dengan Silhouette {fmt(sil_k, 3)}. Geser kembali ke {best_k} '
            f'untuk pengelompokan terbaik.</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-weight:800;color:#0F2A4A;margin-bottom:4px;">🌀 Parameter DBSCAN</div>',
                unsafe_allow_html=True)
    db_grid, db_best = dbscan_grid(dataset, kind)
    db_mode = st.radio("Pengaturan parameter DBSCAN", ["Optimal (disarankan)", "Atur sendiri"], horizontal=True,
                       help="Optimal = kombinasi eps dan min_samples dengan Silhouette tertinggi di antara kombinasi "
                            "yang menghasilkan ≥ 2 klaster dan pencilan ≤ 30%.")
    db_mode_opt = db_mode.startswith("Optimal")
    if db_mode_opt:
        db_eps, db_ms = db_best["eps"], db_best["ms"]
    else:
        _, D_side = subset(dataset, kind)
        S_side = np.sort(D_side, axis=1)
        e_lo = max(0.01, float(np.floor(np.quantile(S_side[:, 2], 0.05) * 100) / 100))
        e_hi = float(np.ceil(np.quantile(S_side[:, 14], 0.99) * 100) / 100)
        db_eps = st.slider("eps (radius ketetanggaan)", e_lo, e_hi,
                           float(min(max(db_best["eps"], e_lo), e_hi)), step=0.005, format="%.3f")
        db_ms = st.slider("min_samples", 2, 15, int(db_best["ms"]))
    if db_best["found"]:
        st.markdown(
            f'<div class="sbx sbx-info">⭐ <b>Parameter optimal</b>: eps = {fmt(db_best["eps"], 3)}, '
            f'min_samples = {db_best["ms"]}<br><span>Jumlah klaster DBSCAN tidak diatur dengan k; '
            f'jumlahnya ditentukan oleh kepadatan data.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="sbx sbx-warn">⚠️ <b>Tidak ada kombinasi yang layak</b> (≥ 2 klaster, pencilan ≤ 30%).<br>'
            f'<span>Dipakai titik siku grafik jarak: eps = {fmt(db_best["eps"], 3)}, min_samples = 5.</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🧭 Panduan singkat", expanded=False):
        st.markdown(
            "1. **Ringkasan** – temuan utama dan rekomendasi tiap klaster.\n"
            "2. **Eksplorasi Data** – kondisi data sebelum dikelompokkan.\n"
            "3. **Jumlah Klaster** – dasar pemilihan k.\n"
            "4. **Profil** – pola mingguan dan statistik tiap klaster K-Medoids.\n"
            "5. **DBSCAN** – optimasi parameter dan hasil pengelompokan berbasis kepadatan.\n"
            "6. **Perbandingan** – K-Medoids vs hierarki vs DBSCAN.\n"
            "7. **Produk** – cari produk dan unduh hasil.\n"
            "8. **Metodologi** – langkah analisis dan glosarium."
        )
    st.caption("Sumber data: Sales Transactions Weekly (811 produk × 52 minggu).")

# =====================================================================
# PERHITUNGAN UTAMA
# =====================================================================
df, D = subset(dataset, kind)
labels, medoids, cost, sil_each = run_kmedoids(dataset, kind, k)
hl, Z = run_hier(dataset, kind, k)
sil = float(sil_each.mean())
sil_hier = silhouette_score(D, hl, metric="precomputed")
ari = adjusted_rand_score(labels, hl)
prof = profile_table(df, labels, medoids)
Xraw = df[WEEKS].to_numpy(float)
Xn = minmax_rows(Xraw)
sil_cat, sil_kind, sil_msg = silhouette_label(sil)
clusters = list(prof.index)
cname = {c: f"Klaster {c}" for c in clusters}
seg = prof["segmen"].to_dict()

db_lab, db_nc, db_sil = run_dbscan(dataset, kind, db_eps, db_ms)
db_noise = float((db_lab == -1).mean() * 100)
db_disp = np.array([db_name(c) for c in db_lab])
db_groups = [g for g in [db_name(c) for c in sorted(set(db_lab[db_lab != -1]))] if g]
db_groups = list(dict.fromkeys(db_groups)) + (["Pencilan"] if (db_lab == -1).any() else [])
ari_kd = adjusted_rand_score(labels, db_lab)
ari_hd = adjusted_rand_score(hl, db_lab)
db_sizes = pd.Series(db_lab[db_lab != -1]).value_counts()
db_largest = float(db_sizes.max() / len(df) * 100) if len(db_sizes) else 0.0
hsize = np.bincount(hl)[1:]


def ccolor(c):
    return CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)]


# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    f"""
<div class="hero">
  <h1>📦 Segmentasi Pola Penjualan Produk</h1>
  <p>Dashboard ini mengelompokkan produk berdasarkan <b>kemiripan pola penjualan mingguan</b> selama 52 minggu.
  Produk dengan pola serupa masuk ke klaster yang sama, sehingga strategi stok dan pemasaran dapat disusun per kelompok,
  bukan per produk satu per satu.</p>
  <div class="chips">
    <span class="chip">Metode: K-Medoids</span>
    <span class="chip">Ukuran jarak: {kind}</span>
    <span class="chip">Data: {len(df)} produk</span>
    <span class="chip">Jumlah klaster: {k}</span>
    <span class="chip">⭐ k optimal: {best_k}</span>
    <span class="chip">Pembanding: Hierarki & DBSCAN</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "📦", "Jumlah produk", fmt(len(df)), "dianalisis")
kpi(c2, "🗓️", "Periode", "52", "minggu (W0 – W51)")
kpi(c3, "🛒", "Total unit terjual", fmt(df["Total_Sales"].sum()), f"rata-rata {fmt(df['Mean_Sales'].mean(), 1)} unit/produk/minggu")
kpi(c4, "⭕", "Minggu tanpa penjualan", pct((Xraw == 0).mean() * 100), "dari seluruh sel produk × minggu")
kpi(c5, "🎯", "Koefisien Silhouette", fmt(sil, 3), f"{sil_cat} · k = {k} (optimal {best_k})")
st.write("")

tabs = st.tabs([
    "🏠 Ringkasan", "🔍 Eksplorasi", "📐 Jumlah Klaster", "📊 Profil", "🌀 DBSCAN",
    "⚖️ Perbandingan", "📋 Produk", "📖 Metodologi",
])
T_RING, T_EKS, T_JML, T_PROF, T_DB, T_PERB, T_DAFTAR, T_METOD = tabs

# =====================================================================
# TAB 1 — RINGKASAN
# =====================================================================
with T_RING:
    box("read", "🧭 Cara membaca dashboard ini",
        "<ul><li>Pilih <b>data</b>, <b>ukuran jarak</b>, dan <b>jumlah klaster</b> di panel kiri.</li>"
        "<li>Mulai dari tab ini untuk melihat <b>temuan utama</b> dan <b>rekomendasi</b>.</li>"
        "<li>Setiap grafik dilengkapi kotak <b>Cara membaca</b> dan <b>Interpretasi</b>.</li>"
        "<li><b>Klaster 1</b> selalu klaster dengan median total penjualan tertinggi.</li>"
        f"<li>Jumlah klaster optimal (<b>k = {best_k}</b>) ditandai di panel kiri dan di tab <b>Jumlah Klaster</b>.</li></ul>")

    section("🔑 Temuan Utama")
    top = prof.iloc[0]
    low = prof.sort_values("mean").iloc[0]
    weekly = df[WEEKS].sum()
    chg = weekly.pct_change() * 100
    wk_drop = chg.idxmin()
    wk_prev = WEEKS[WEEKS.index(wk_drop) - 1]
    h1, h2 = weekly[:26].mean(), weekly[26:].mean()

    f1, f2 = st.columns(2)
    with f1:
        box("ins", f"1. {cname[1]} ({seg[1]}) menjadi tulang punggung penjualan",
            f"Klaster ini berisi <b>{fmt(top['n'])} produk ({pct(top['share_prod'])})</b> tetapi menyumbang "
            f"<b>{pct(top['share_units'])}</b> dari seluruh unit terjual. Median penjualannya "
            f"{fmt(top['mean'], 1)} unit per minggu dengan koefisien variasi {fmt(top['cv'], 2)}.")
        box("ins", "3. Penjualan semester kedua lebih rendah",
            f"Rata-rata total penjualan mingguan turun dari <b>{fmt(h1)}</b> unit (W0–W25) menjadi "
            f"<b>{fmt(h2)}</b> unit (W26–W51), atau {trend_text((h2 - h1) / h1 * 100)}. "
            f"Penurunan terbesar terjadi dari {wk_prev} ke {wk_drop} (<b>{fmt(chg.min(), 1)}%</b>).")
    with f2:
        if low.name != 1:
            box("ins", f"2. {cname[low.name]} ({seg[low.name]}) perlu perhatian khusus",
                f"Berisi <b>{fmt(low['n'])} produk ({pct(low['share_prod'])})</b> namun hanya menyumbang "
                f"<b>{pct(low['share_units'])}</b> unit terjual. Median {fmt(low['zero'], 1)}% minggu tanpa penjualan.")
        else:
            box("ins", "2. Hanya satu kelompok volume", "Semua klaster memiliki volume serupa.")
        box(sil_kind, f"4. Kualitas pengelompokan: {sil_cat} (Silhouette = {fmt(sil, 3)})", sil_msg)

    box("ins", "5. Metode pembanding: klaster hierarki dan DBSCAN",
        f"Klaster hierarki menempatkan <b>{pct(hsize.max() / len(df) * 100)}</b> produk dalam satu klaster "
        f"(ukuran: {', '.join(map(str, hsize))}). DBSCAN membentuk <b>{db_nc} klaster</b> dengan "
        f"<b>{pct(db_noise)} pencilan</b>; klaster terbesarnya memuat {pct(db_largest)} produk. "
        + ("Kedua metode pembanding tidak menemukan kelompok berukuran seimbang, sehingga segmentasi K-Medoids "
           "dipakai sebagai acuan utama. Lihat tab <b>Perbandingan</b>."
           if hsize.max() / len(df) >= 0.7 and db_largest >= 70 else
           "Lihat tab <b>Perbandingan</b> untuk menilai kesesuaian ketiga metode."))

    section("🧩 Profil Singkat Tiap Klaster", "Ringkasan statistik median dan rekomendasi tindakan.")
    per_row = 2 if k in (2, 4) else 3
    for start in range(0, k, per_row):
        cols = st.columns(per_row)
        for col, c in zip(cols, clusters[start:start + per_row]):
            r = prof.loc[c]
            traits = "".join(f"<li>{t}</li>" for t in TRAITS[r["segmen"]])
            acts = "".join(f"<li>{t}</li>" for t in ACTIONS[r["segmen"]])
            col.markdown(
                f"""
<div class="ccard">
  <div class="tag"><span class="dot" style="background:{ccolor(c)}"></span>{cname[c]}</div>
  <div class="seg">{r['segmen']}</div>
  <div class="grid">
    <div class="mini"><div class="l">Jumlah produk</div><div class="v">{fmt(r['n'])} ({pct(r['share_prod'])})</div></div>
    <div class="mini"><div class="l">Porsi unit terjual</div><div class="v">{pct(r['share_units'])}</div></div>
    <div class="mini"><div class="l">Median unit/minggu</div><div class="v">{fmt(r['mean'], 2)}</div></div>
    <div class="mini"><div class="l">Median koef. variasi</div><div class="v">{fmt(r['cv'], 2)}</div></div>
    <div class="mini"><div class="l">Median minggu tanpa jual</div><div class="v">{pct(r['zero'])}</div></div>
    <div class="mini"><div class="l">Semester II vs I</div><div class="v">{trend_text(r['trend'])}</div></div>
  </div>
  <h5>Ciri-ciri</h5><ul>{traits}</ul>
  <h5>Rekomendasi</h5><ul>{acts}</ul>
  <div style="margin-top:10px;font-size:.8rem;color:#5B6B82;">Produk representatif (medoid): <b>{r['medoid']}</b></div>
</div>""",
                unsafe_allow_html=True,
            )
        st.write("")

    section("📦 Porsi Produk vs Porsi Penjualan")
    fig = go.Figure()
    ylab = [f"{cname[c]} · {seg[c]}" for c in clusters][::-1]
    fig.add_bar(y=ylab, x=prof["share_prod"].values[::-1], orientation="h", name="% jumlah produk",
                marker=dict(color="#9EC5F4", cornerradius=6), text=[pct(v) for v in prof["share_prod"].values[::-1]],
                textposition="outside", hovertemplate="%{y}<br>Porsi produk: %{x:.1f}%<extra></extra>")
    fig.add_bar(y=ylab, x=prof["share_units"].values[::-1], orientation="h", name="% unit terjual",
                marker=dict(color=BLUE, cornerradius=6), text=[pct(v) for v in prof["share_units"].values[::-1]],
                textposition="outside", hovertemplate="%{y}<br>Porsi unit terjual: %{x:.1f}%<extra></extra>")
    style_fig(fig, height=140 + 70 * k)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(title="Persentase (%)", range=[0, 115], showgrid=True, gridcolor=GRID)
    show(fig)
    read_box("Batang biru muda = porsi klaster terhadap <b>jumlah produk</b>; batang biru tua = porsi terhadap "
             "<b>total unit terjual</b>. Jika batang biru tua jauh lebih panjang, klaster itu menghasilkan penjualan "
             "jauh di atas proporsi jumlah produknya.")
    ins_box(f"{cname[1]} mencakup {pct(top['share_prod'])} produk tetapi {pct(top['share_units'])} unit terjual. "
            "Ketersediaan stok produk di klaster ini paling berpengaruh terhadap total penjualan.")

# =====================================================================
# TAB 2 — EKSPLORASI DATA
# =====================================================================
with T_EKS:
    section("🧹 Pemeriksaan Kualitas Data", "Data harus bersih sebelum dikelompokkan.")
    a1, a2, a3, a4 = st.columns(4)
    miss = int(df[WEEKS].isna().sum().sum())
    dup = int(df["Product_Code"].duplicated().sum())
    neg = int((df[WEEKS] < 0).sum().sum())
    never = int((df["Total_Sales"] == 0).sum())
    kpi(a1, "✅" if miss == 0 else "⚠️", "Data kosong", fmt(miss), "sel tanpa nilai")
    kpi(a2, "✅" if dup == 0 else "⚠️", "Kode produk ganda", fmt(dup), "duplikasi")
    kpi(a3, "✅" if neg == 0 else "⚠️", "Nilai negatif", fmt(neg), "penjualan < 0")
    kpi(a4, "✅" if never == 0 else "⚠️", "Produk tidak pernah terjual", fmt(never), "total = 0")
    st.write("")
    if miss + dup + neg + never == 0:
        box("good", "✅ Data siap dianalisis",
            "Tidak ada data kosong, kode ganda, nilai negatif, atau produk yang tidak pernah terjual.")

    section("📈 Tren Total Penjualan Mingguan")
    fig = go.Figure()
    fig.add_scatter(x=WEEKS, y=weekly.values, mode="lines+markers", name="Total unit",
                    line=dict(color=BLUE, width=2.5), marker=dict(size=7, color=BLUE, line=dict(color="white", width=1.5)),
                    hovertemplate="%{x}<br>Total: %{y:,.0f} unit<extra></extra>")
    fig.add_scatter(x=WEEKS[:26], y=[h1] * 26, mode="lines", name=f"Rata-rata W0–W25 ({fmt(h1)})",
                    line=dict(color="#86B6EF", dash="dash", width=2), hoverinfo="skip")
    fig.add_scatter(x=WEEKS[26:], y=[h2] * 26, mode="lines", name=f"Rata-rata W26–W51 ({fmt(h2)})",
                    line=dict(color="#1baf7a", dash="dash", width=2), hoverinfo="skip")
    fig.add_annotation(x=wk_drop, y=weekly[wk_drop], text=f"Penurunan terbesar<br>{wk_prev}→{wk_drop}: {fmt(chg.min(), 1)}%",
                       showarrow=True, arrowhead=2, ax=110, ay=8, bgcolor="white", bordercolor="#C9DEFA",
                       borderpad=6, font=dict(size=11, color=NAVY), xanchor="left")
    style_fig(fig, height=400)
    fig.update_yaxes(title="Total unit terjual")
    fig.update_xaxes(title="Minggu", tickvals=WEEKS[::4])
    show(fig)
    read_box("Garis biru menunjukkan jumlah unit terjual seluruh produk setiap minggu. Garis putus-putus adalah "
             "rata-rata pada paruh pertama dan paruh kedua tahun.")
    ins_box(f"Penjualan naik perlahan hingga {weekly.idxmax()} (puncak {fmt(weekly.max())} unit), lalu turun tajam pada "
            f"{wk_prev}–{wk_drop}. Setelah itu penjualan pulih bertahap tetapi tidak kembali ke tingkat semester pertama. "
            "Perubahan tingkat yang mendadak seperti ini sebaiknya dikonfirmasi ke pihak bisnis (misalnya perubahan "
            "harga, stok habis, atau pergantian katalog).")

    section("⭕ Sebaran Minggu Tanpa Penjualan")
    e1, e2 = st.columns([3, 2])
    with e1:
        zc = df["Zero_Sales_Count"]
        fig = go.Figure()
        fig.add_histogram(x=zc, xbins=dict(start=-0.5, end=52.5, size=1), marker=dict(color=BLUE, cornerradius=4,
                          line=dict(color="white", width=1)),
                          hovertemplate="%{x} minggu tanpa penjualan<br>%{y} produk<extra></extra>")
        fig.add_vline(x=45.5, line=dict(color="#E34948", dash="dash", width=2))
        fig.add_annotation(x=45.5, y=1, yref="paper", text="Batas filter: > 45 minggu", showarrow=False,
                           xanchor="right", font=dict(color="#B42318", size=11), xshift=-6)
        style_fig(fig, height=340, legend=False)
        fig.update_xaxes(title="Jumlah minggu tanpa penjualan (dari 52)")
        fig.update_yaxes(title="Jumlah produk")
        show(fig)
    with e2:
        n0 = int((zc == 0).sum())
        nhalf = int((zc > 26).sum())
        read_box("Setiap batang menunjukkan berapa banyak produk yang memiliki jumlah minggu tanpa penjualan tertentu. "
                 "Garis merah adalah batas yang dipakai untuk membuat <b>data terfilter</b>.")
        ins_box(f"Sebarannya <b>bimodal</b> (dua puncak): {fmt(n0)} produk ({pct(n0 / len(df) * 100)}) selalu terjual "
                f"setiap minggu, sedangkan {fmt(nhalf)} produk ({pct(nhalf / len(df) * 100)}) tidak terjual pada lebih "
                "dari separuh minggu. Dua kelompok alami ini sudah terlihat bahkan sebelum analisis klaster.")

    section("📉 Variasi Penjualan Tiap Produk", "Koefisien variasi (KV) = simpangan baku ÷ rata-rata.")
    v1, v2 = st.columns(2)
    with v1:
        cat_map = {"Low": "Rendah (KV < 0,5)", "Medium": "Sedang (0,5–1)", "High": "Tinggi (KV > 1)"}
        vc = df["Variability"].map(cat_map).value_counts().reindex(cat_map.values()).fillna(0)
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker=dict(color=["#86B6EF", "#3987E5", "#184F95"], cornerradius=6),
                               text=[f"{fmt(v)} ({pct(v / len(df) * 100)})" for v in vc.values], textposition="outside",
                               hovertemplate="%{x}<br>%{y} produk<extra></extra>"))
        style_fig(fig, height=340, title="Kategori variasi penjualan", legend=False)
        fig.update_yaxes(title="Jumlah produk", range=[0, vc.max() * 1.2])
        show(fig)
    with v2:
        fig = go.Figure(go.Scatter(
            x=df["Mean_Sales"], y=df["CV_Sales"], mode="markers", text=df["Product_Code"],
            marker=dict(size=8, color=hex_rgba(BLUE, 0.55), line=dict(color="white", width=1)),
            hovertemplate="%{text}<br>Rata-rata: %{x:.2f} unit/minggu<br>KV: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=1, line=dict(color="#9AA9BF", dash="dot"))
        fig.add_hline(y=0.5, line=dict(color="#9AA9BF", dash="dot"))
        style_fig(fig, height=340, title="Rata-rata penjualan vs koefisien variasi", legend=False)
        fig.update_xaxes(title="Rata-rata unit per minggu (skala log)", type="log", dtick=1, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title="Koefisien variasi")
        show(fig)
    read_box("Kiri: jumlah produk pada tiap kategori variasi. Kanan: setiap titik adalah satu produk; sumbu mendatar "
             "memakai skala logaritma agar produk bervolume kecil tetap terlihat.")
    ins_box("Terdapat hubungan terbalik yang jelas: makin kecil rata-rata penjualan, makin besar koefisien variasinya. "
            "Produk bervolume kecil cenderung tidak menentu karena satu-dua penjualan saja sudah mengubah pola secara "
            "besar. Karena itu, variasi tinggi pada produk kecil tidak selalu berarti ada lonjakan permintaan.")

    section("🏆 Produk Terlaris dan Paling Jarang Terjual")
    t1, t2 = st.columns(2)
    cols_show = {"Product_Code": "Kode", "Total_Sales": "Total unit", "Mean_Sales": "Rata-rata/minggu",
                 "CV_Sales": "Koef. variasi", "Zero_Sales_Count": "Minggu tanpa jual"}
    with t1:
        st.markdown("**10 produk terlaris**")
        st.dataframe(df.nlargest(10, "Total_Sales")[list(cols_show)].rename(columns=cols_show).round(2),
                     hide_index=True, width="stretch")
    with t2:
        st.markdown("**10 produk paling jarang terjual**")
        st.dataframe(df.nsmallest(10, "Total_Sales")[list(cols_show)].rename(columns=cols_show).round(2),
                     hide_index=True, width="stretch")

    section("🔁 Mengapa Data Dinormalisasi?", "Normalisasi min–maks per produk: (nilai − minimum) ÷ (maksimum − minimum).")
    p_sel = st.selectbox("Pilih produk contoh", df["Product_Code"].tolist(), index=0, key="norm_prod")
    i = int(df.index[df["Product_Code"] == p_sel][0])
    n1, n2 = st.columns(2)
    with n1:
        fig = go.Figure(go.Scatter(x=WEEKS, y=Xraw[i], mode="lines+markers", line=dict(color=BLUE, width=2),
                                   marker=dict(size=6), hovertemplate="%{x}: %{y} unit<extra></extra>"))
        style_fig(fig, height=300, title=f"{p_sel} – unit terjual (asli)", legend=False)
        fig.update_xaxes(tickvals=WEEKS[::6])
        show(fig)
    with n2:
        fig = go.Figure(go.Scatter(x=WEEKS, y=Xn[i], mode="lines+markers", line=dict(color="#1baf7a", width=2),
                                   marker=dict(size=6), hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        style_fig(fig, height=300, title=f"{p_sel} – setelah normalisasi (0–1)", legend=False)
        fig.update_xaxes(tickvals=WEEKS[::6])
        show(fig)
    ins_box("Bentuk pola kedua grafik sama persis; yang berubah hanya skalanya. Normalisasi membuat produk bervolume "
            "besar dan kecil dapat dibandingkan <b>berdasarkan bentuk polanya</b>, bukan besar volumenya. "
            "Konsekuensinya, produk yang hanya terjual 1 unit setahun akan terlihat sebagai satu lonjakan setinggi 1.")

# =====================================================================
# TAB 3 — JUMLAH KLASTER
# =====================================================================
with T_JML:
    section("📐 Menentukan Jumlah Klaster",
            "Jumlah klaster dipilih dengan membandingkan koefisien Silhouette untuk k = 2 sampai 6.")
    box("good", f"⭐ Jumlah klaster optimal: k = {best_k}",
        f"Untuk data <b>{len(df)} produk</b> dengan jarak <b>{kind}</b>, koefisien Silhouette K-Medoids tertinggi "
        f"({fmt(sil_best, 3)}) dicapai pada <b>k = {best_k}</b>. "
        + ("Anda sedang memakai k optimal." if k == best_k else
           f"Saat ini Anda memakai k = {k}; geser slider di panel kiri ke {best_k} untuk hasil terbaik.")
        + " DBSCAN tidak memakai k; jumlah klasternya ditentukan oleh parameter eps dan min_samples (tab <b>DBSCAN</b>).")
    j1, j2 = st.columns(2)
    with j1:
        fig = go.Figure()
        for col_name, color, nm in [("Silhouette K-Medoids", BLUE, "K-Medoids"), ("Silhouette Hierarki", "#eb6834", "Hierarki")]:
            fig.add_scatter(x=ktab["k"], y=ktab[col_name], mode="lines+markers+text", name=nm,
                            line=dict(color=color, width=2.5), marker=dict(size=10, line=dict(color="white", width=2)),
                            text=[fmt(v, 3) for v in ktab[col_name]],
                            textposition=[("bottom right" if (nm == "K-Medoids" and kk == best_k) else
                                           "bottom center" if nm == "K-Medoids" else
                                           "bottom left" if kk == best_k else "top center") for kk in ktab["k"]],
                            textfont=dict(size=10, color=INK_2),
                            hovertemplate=f"{nm}<br>k = %{{x}}<br>Silhouette = %{{y:.3f}}<extra></extra>")
        fig.add_vline(x=k, line=dict(color="#9AA9BF", dash="dot"))
        fig.add_scatter(x=[best_k], y=[sil_best], mode="markers", name=f"k optimal = {best_k}",
                        marker=dict(size=18, symbol="star", color="#FFD43B", line=dict(color=NAVY, width=1.5)),
                        hovertemplate="k optimal = %{x}<br>Silhouette = %{y:.3f}<extra></extra>")
        style_fig(fig, height=360, title="Koefisien Silhouette menurut jumlah klaster")
        fig.update_xaxes(title="Jumlah klaster (k)", dtick=1)
        fig.update_yaxes(title="Koefisien Silhouette", range=[min(0, ktab.iloc[:, 1:3].min().min()) - 0.02,
                                                              ktab.iloc[:, 1:3].max().max() + 0.08])
        show(fig)
    with j2:
        fig = go.Figure(go.Scatter(x=ktab["k"], y=ktab["Total jarak ke medoid"], mode="lines+markers",
                                   line=dict(color=BLUE, width=2.5), marker=dict(size=10, line=dict(color="white", width=2)),
                                   hovertemplate="k = %{x}<br>Total jarak = %{y:.1f}<extra></extra>"))
        fig.add_vline(x=k, line=dict(color="#9AA9BF", dash="dot"))
        style_fig(fig, height=360, title="Total jarak produk ke medoid (metode siku)", legend=False)
        fig.update_xaxes(title="Jumlah klaster (k)", dtick=1)
        fig.update_yaxes(title="Total jarak dalam klaster")
        show(fig)
    read_box("<b>Kiri</b>: koefisien Silhouette mengukur seberapa mirip produk dengan klasternya sendiri dibanding "
             "klaster terdekat lain (rentang −1 sampai 1; makin tinggi makin baik). <b>Kanan</b>: total jarak "
             "seluruh produk ke medoid klasternya; titik yang membentuk ‘siku’ menandai tambahan klaster yang tidak lagi "
             "banyak mengurangi jarak. Garis titik-titik menandai k yang sedang dipilih.")
    best_row = ktab.loc[ktab["Silhouette K-Medoids"].idxmax()]
    ins_box(f"Koefisien Silhouette K-Medoids tertinggi ada pada <b>k = {int(best_row['k'])}</b> "
            f"({fmt(best_row['Silhouette K-Medoids'], 3)}), sehingga k ini menjadi pilihan utama. Nilai klaster "
            "hierarki kadang lebih tinggi, tetapi lihat tab <b>Perbandingan</b>: nilai tinggi itu sering berasal "
            "dari klaster berisi hanya beberapa produk ekstrem, sehingga tidak berguna untuk segmentasi.")

    st.dataframe(
        ktab.rename(columns={"k": "Jumlah klaster (k)"}).style.format({
            "Silhouette K-Medoids": "{:.3f}", "Silhouette Hierarki": "{:.3f}", "Total jarak ke medoid": "{:.1f}"}),
        hide_index=True, width="stretch")

    section("📏 Panduan Kategori Koefisien Silhouette", "Mengacu pada Kaufman & Rousseeuw (1990).")
    g1, g2, g3, g4 = st.columns(4)
    for col, (rng, nm, kd, ds) in zip([g1, g2, g3, g4], [
        ("0,71 – 1,00", "Kuat", "good", "Klaster terpisah sangat jelas."),
        ("0,51 – 0,70", "Wajar", "good", "Pemisahan cukup jelas."),
        ("0,26 – 0,50", "Lemah", "warn", "Ada struktur, batas antarklaster kabur."),
        ("≤ 0,25", "Tidak substansial", "bad", "Tidak ada struktur klaster berarti."),
    ]):
        with col:
            box(kd, f"{nm} · {rng}", ds)

    section("🧪 Kecocokan Setiap Produk dengan Klasternya", f"Nilai Silhouette per produk (k = {k}).")
    order = np.lexsort((-sil_each, labels))
    fig = go.Figure()
    pos = 0
    for c in clusters:
        idx = order[labels[order] == c]
        fig.add_bar(x=sil_each[idx], y=np.arange(pos, pos + len(idx)), orientation="h", name=f"{cname[c]} · {seg[c]}",
                    marker=dict(color=ccolor(c), line=dict(width=0)), width=1.0,
                    customdata=df["Product_Code"].to_numpy()[idx],
                    hovertemplate="%{customdata}<br>Silhouette: %{x:.3f}<extra></extra>")
        pos += len(idx) + 12
    fig.add_vline(x=sil, line=dict(color=NAVY, dash="dash", width=1.5))
    fig.add_annotation(x=sil, y=1, yref="paper", text=f"Rata-rata = {fmt(sil, 3)}", showarrow=False, xanchor="left",
                       xshift=6, font=dict(size=11, color=NAVY))
    style_fig(fig, height=430)
    fig.update_yaxes(showticklabels=False, showgrid=False, title="Produk (diurutkan per klaster)")
    fig.update_xaxes(title="Nilai Silhouette produk", showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#9AA9BF")
    fig.update_layout(bargap=0)
    show(fig)
    neg_share = (sil_each < 0).mean() * 100
    per_c = pd.Series(sil_each).groupby(labels).mean()
    read_box("Setiap garis tipis mewakili satu produk. Nilai mendekati 1 = produk sangat cocok dengan klasternya; "
             "mendekati 0 = berada di perbatasan; negatif = lebih mirip dengan klaster lain.")
    ins_box(f"{pct(neg_share)} produk memiliki nilai Silhouette negatif (kemungkinan lebih cocok di klaster lain). "
            + " ".join(f"Rata-rata {cname[c]}: {fmt(per_c[c], 3)}." for c in clusters)
            + " Klaster dengan rata-rata lebih tinggi memiliki anggota yang lebih seragam.")

# =====================================================================
# TAB 4 — PROFIL KLASTER
# =====================================================================
with T_PROF:
    section("〰️ Pola Penjualan Mingguan Tiap Klaster",
            "Skala normalisasi 0–1. Garis tebal = median; pita = rentang antarkuartil (25%–75%); garis putus = medoid.")
    ncol = min(k, 3)
    nrow = int(np.ceil(k / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.14,
                        subplot_titles=[f"{cname[c]} · {seg[c]} (n = {prof.loc[c, 'n']})" for c in clusters])
    xw = list(range(52))
    for n_, c in enumerate(clusters):
        r_, c_ = n_ // ncol + 1, n_ % ncol + 1
        M = Xn[labels == c]
        q1, q2, q3 = np.percentile(M, [25, 50, 75], axis=0)
        col = ccolor(c)
        fig.add_scatter(x=xw + xw[::-1], y=list(q3) + list(q1[::-1]), fill="toself", fillcolor=hex_rgba(col, 0.18),
                        line=dict(width=0), hoverinfo="skip", showlegend=False, row=r_, col=c_)
        fig.add_scatter(x=xw, y=q2, mode="lines", line=dict(color=col, width=3), name="Median",
                        showlegend=False, hovertemplate="W%{x}<br>Median: %{y:.2f}<extra></extra>", row=r_, col=c_)
        fig.add_scatter(x=xw, y=Xn[medoids[c]], mode="lines", line=dict(color=NAVY, width=1.5, dash="dash"),
                        name="Medoid", showlegend=False,
                        hovertemplate=f"Medoid {prof.loc[c, 'medoid']}<br>W%{{x}}: %{{y:.2f}}<extra></extra>", row=r_, col=c_)
    style_fig(fig, height=330 * nrow, legend=False)
    fig.update_xaxes(title="Minggu", tickvals=list(range(0, 52, 8)), ticktext=[f"W{i}" for i in range(0, 52, 8)])
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.update_annotations(font=dict(size=13, color=NAVY))
    show(fig)
    read_box("Setiap panel adalah satu klaster. Garis tebal berwarna adalah <b>median</b> nilai normalisasi seluruh "
             "produk di klaster itu per minggu; pita berwarna menunjukkan 50% produk di bagian tengah; garis putus-putus "
             "biru tua adalah <b>medoid</b>, yaitu produk nyata yang paling mewakili klaster.")
    txt = []
    for c in clusters:
        M = Xn[labels == c]
        med = np.median(M, axis=0)
        iqr = (np.percentile(M, 75, axis=0) - np.percentile(M, 25, axis=0)).mean()
        txt.append(f"<li><b>{cname[c]}</b>: median normalisasi rata-rata {fmt(med.mean(), 2)}, lebar pita rata-rata "
                   f"{fmt(iqr, 2)} → " + ("pola padat dan rutin." if med.mean() > 0.3 else
                                          "sebagian besar minggu berada di titik nol; penjualan muncul sesekali.") + "</li>")
    ins_box("<ul>" + "".join(txt) + "</ul>")

    section("🛒 Rata-rata Unit Terjual per Produk per Minggu", "Skala asli (unit). Sumbu tegak tiap panel berbeda.")
    fig = make_subplots(rows=nrow, cols=ncol, horizontal_spacing=0.06, vertical_spacing=0.14,
                        subplot_titles=[f"{cname[c]} · {seg[c]}" for c in clusters])
    for n_, c in enumerate(clusters):
        r_, c_ = n_ // ncol + 1, n_ % ncol + 1
        m = Xraw[labels == c].mean(axis=0)
        fig.add_scatter(x=xw, y=m, mode="lines", line=dict(color=ccolor(c), width=2.5), showlegend=False,
                        hovertemplate="W%{x}<br>%{y:.2f} unit/produk<extra></extra>", row=r_, col=c_)
        fig.add_scatter(x=[0, 25], y=[m[:26].mean()] * 2, mode="lines", line=dict(color="#9AA9BF", dash="dash", width=1.5),
                        showlegend=False, hoverinfo="skip", row=r_, col=c_)
        fig.add_scatter(x=[26, 51], y=[m[26:].mean()] * 2, mode="lines", line=dict(color="#9AA9BF", dash="dash", width=1.5),
                        showlegend=False, hoverinfo="skip", row=r_, col=c_)
    style_fig(fig, height=300 * nrow, legend=False)
    fig.update_xaxes(tickvals=list(range(0, 52, 8)), ticktext=[f"W{i}" for i in range(0, 52, 8)])
    fig.update_annotations(font=dict(size=13, color=NAVY))
    show(fig)
    ins_box(" ".join(f"{cname[c]}: semester II dibanding semester I {trend_text(prof.loc[c, 'trend'])}." for c in clusters)
            + " Garis putus-putus abu-abu menunjukkan rata-rata tiap semester.")

    section("📦 Sebaran Statistik Tiap Klaster", "Diagram kotak: garis tengah = median, kotak = kuartil 1–3.")
    b1, b2, b3 = st.columns(3)
    d_ = df.assign(Klaster=[f"{cname[c]}" for c in labels])
    for col, var, ttl, logy in [(b1, "Total_Sales", "Total unit terjual (skala log)", True),
                                (b2, "CV_Sales", "Koefisien variasi", False),
                                (b3, "Zero_Sales_Pct", "% minggu tanpa penjualan", False)]:
        with col:
            fig = go.Figure()
            for c in clusters:
                fig.add_box(y=d_.loc[labels == c, var], name=cname[c], marker_color=ccolor(c),
                            line=dict(width=1.5), fillcolor=hex_rgba(ccolor(c), 0.25), boxpoints="outliers",
                            marker=dict(size=4))
            style_fig(fig, height=340, title=ttl, legend=False)
            if logy:
                fig.update_yaxes(type="log", dtick=1)
            show(fig)
    read_box("Kotak menunjukkan 50% produk di tengah; garis di dalam kotak adalah median; titik di luar adalah pencilan.")
    ins_box("Perbedaan antarklaster paling tegas terlihat pada <b>% minggu tanpa penjualan</b> dan <b>total unit</b>. "
            "Artinya pengelompokan berbasis bentuk pola pada dasarnya memisahkan produk yang <b>rutin terjual</b> "
            "dari yang <b>terjual sesekali</b>.")

    section("🧮 Tabel Statistik Deskriptif per Klaster")
    stat = df.assign(Klaster=labels).groupby("Klaster").agg(
        Jumlah_produk=("Product_Code", "count"),
        Total_unit=("Total_Sales", "sum"),
        Rata_rata_mingguan=("Mean_Sales", "mean"),
        Median_mingguan=("Median_Sales", "median"),
        Simpangan_baku=("SD_Sales", "mean"),
        Koef_variasi_median=("CV_Sales", "median"),
        Minggu_tanpa_jual_median=("Zero_Sales_Count", "median"),
    )
    stat.index = [f"{cname[c]} · {seg[c]}" for c in stat.index]
    decimals = [0, 0, 2, 2, 2, 2, 0]
    stat_t = pd.DataFrame(
        {col: [fmt(v, d) for v, d in zip(stat.loc[col].values, decimals)] for col in stat.index},
        index=["Jumlah produk", "Total unit terjual", "Rata-rata unit per minggu", "Median unit per minggu",
               "Rata-rata simpangan baku", "Median koefisien variasi", "Median minggu tanpa penjualan"],
    )
    stat_t.index.name = "Statistik"
    st.dataframe(stat_t, width="stretch")
    read_box("Kolom = klaster, baris = statistik. Rata-rata peka terhadap nilai ekstrem, sedangkan median "
             "menunjukkan nilai tengah yang lebih tahan pencilan; bila keduanya jauh berbeda, sebaran data menceng.")

    section("🗺️ Peta Kemiripan Produk",
            "Penskalaan multidimensi klasik: matriks jarak diringkas menjadi 2 dimensi.")
    coords, prop = classical_mds(dataset, kind)
    fig = go.Figure()
    for c in clusters:
        m = labels == c
        fig.add_scatter(x=coords[m, 0], y=coords[m, 1], mode="markers", name=f"{cname[c]} · {seg[c]}",
                        marker=dict(size=8, color=hex_rgba(ccolor(c), 0.7), symbol=CLUSTER_SYMBOLS[(c - 1) % 6],
                                    line=dict(color="white", width=1)),
                        text=df["Product_Code"][m],
                        hovertemplate="%{text}<extra>" + cname[c] + "</extra>")
    mx = [medoids[c] for c in clusters]
    fig.add_scatter(x=coords[mx, 0], y=coords[mx, 1], mode="markers+text", name="Medoid",
                    marker=dict(size=18, symbol="star", color="#FFD43B", line=dict(color=NAVY, width=1.5)),
                    text=[prof.loc[c, "medoid"] for c in clusters], textposition="middle right",
                    textfont=dict(color=NAVY, size=13), hovertemplate="Medoid %{text}<extra></extra>")
    style_fig(fig, height=500)
    fig.update_xaxes(title=f"Dimensi 1 ({pct(prop[0] * 100)} variasi)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=f"Dimensi 2 ({pct(prop[1] * 100)} variasi)")
    show(fig)
    read_box("Setiap titik adalah satu produk. Produk yang berdekatan memiliki pola penjualan yang mirip. "
             "Bintang kuning menandai medoid tiap klaster. Warna dan bentuk titik menunjukkan klaster.")
    ins_box(f"Dua dimensi ini mewakili {pct((prop[0] + prop[1]) * 100)} dari total variasi jarak antarproduk, "
            "sehingga peta ini merupakan ringkasan, bukan gambaran lengkap. Daerah tempat warna saling tumpang tindih "
            "adalah produk perbatasan yang nilai Silhouette-nya rendah.")

# =====================================================================
# TAB 5 — DBSCAN
# =====================================================================
with T_DB:
    section(f"🌀 DBSCAN dengan Jarak {kind}",
            "Pengelompokan berbasis kepadatan: produk di daerah padat membentuk klaster, produk di daerah renggang "
            "menjadi pencilan.")
    box("read", "📖 Apa itu DBSCAN?",
        f"<ul><li><b>eps</b> = radius ketetanggaan: dua produk dianggap bertetangga bila jarak {kind}-nya ≤ eps.</li>"
        "<li><b>min_samples</b> = jumlah minimum produk (termasuk dirinya sendiri) dalam radius eps agar sebuah produk "
        "menjadi <b>titik inti</b>, yaitu pusat daerah padat.</li>"
        "<li>Titik inti yang saling bertetangga membentuk satu klaster. Produk yang tidak terjangkau titik inti mana pun "
        "menjadi <b>pencilan</b>.</li>"
        "<li>Berbeda dengan K-Medoids, <b>jumlah klaster tidak ditentukan di awal</b>; jumlahnya muncul dari kepadatan "
        "data.</li></ul>")
    d1, d2, d3, d4, d5 = st.columns(5)
    kpi(d1, "📏", "eps", fmt(db_eps, 3), "radius ketetanggaan")
    kpi(d2, "👥", "min_samples", str(db_ms), "minimum produk per daerah padat")
    kpi(d3, "🧩", "Klaster terbentuk", str(db_nc), "di luar pencilan")
    kpi(d4, "⚪", "Pencilan", pct(db_noise), f"{fmt(int((db_lab == -1).sum()))} produk")
    kpi(d5, "🎯", "Silhouette", "–" if np.isnan(db_sil) else fmt(db_sil, 3), "dihitung tanpa pencilan")
    st.write("")
    if db_mode_opt and db_best["found"]:
        box("good", "✅ Parameter optimal sedang dipakai",
            f"eps = <b>{fmt(db_eps, 3)}</b> dan min_samples = <b>{db_ms}</b>, hasil pencarian kombinasi pada Langkah 2.")
    elif db_mode_opt:
        box("warn", "⚠️ Tidak ditemukan kombinasi parameter yang layak",
            f"Tidak ada kombinasi yang menghasilkan ≥ 2 klaster dengan pencilan ≤ {DB_MAX_NOISE}% dan Silhouette > 0. "
            f"Dipakai titik siku grafik jarak (eps = {fmt(db_eps, 3)}, min_samples = {db_ms}). Artinya, dengan jarak "
            f"{kind} DBSCAN tidak menemukan lebih dari satu daerah padat.")
    else:
        box("warn", "✋ Parameter diatur sendiri",
            f"Parameter optimal: eps = {fmt(db_best['eps'], 3)}, min_samples = {db_best['ms']}. "
            "Pilih <b>Optimal</b> di panel kiri untuk kembali.")

    # ---------------- Langkah 1: grafik jarak tetangga ke-k ----------------
    section("Langkah 1 · Grafik jarak tetangga ke-k",
            f"Untuk min_samples = {db_ms}: jarak tiap produk ke tetangga terdekat ke-{db_ms - 1}, diurutkan dari kecil ke besar.")
    kd = kdist_sorted(D, db_ms)
    ki, kv = knee_point(kd)
    core_share = (kd <= db_eps).mean() * 100
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, len(kd) + 1), y=kd, mode="lines", name=f"Jarak ke tetangga ke-{db_ms - 1}",
                    line=dict(color=BLUE, width=2.5), hovertemplate="Urutan %{x}<br>Jarak: %{y:.3f}<extra></extra>")
    fig.add_scatter(x=[1, len(kd)], y=[db_eps, db_eps], mode="lines", name=f"eps terpakai ({fmt(db_eps, 3)})",
                    line=dict(color="#E34948", dash="dash", width=2), hoverinfo="skip")
    fig.add_scatter(x=[ki + 1], y=[kv], mode="markers", name=f"Titik siku ({fmt(kv, 3)})",
                    marker=dict(size=14, color="#eda100", symbol="diamond", line=dict(color=NAVY, width=1.5)),
                    hovertemplate="Titik siku<br>Jarak: %{y:.3f}<extra></extra>")
    style_fig(fig, height=380)
    fig.update_xaxes(title="Produk (diurutkan menurut jarak)")
    fig.update_yaxes(title=f"Jarak {kind}")
    show(fig)
    read_box("Setiap titik pada garis biru adalah satu produk. Nilai di sumbu tegak = seberapa jauh produk itu dari "
             f"tetangga terdekatnya yang ke-{db_ms - 1}. Produk yang berada <b>di bawah garis merah (eps)</b> menjadi "
             "titik inti. <b>Titik siku</b> (belah ketupat kuning) adalah tempat kurva mulai naik tajam; nilai ini lazim "
             "dipakai sebagai acuan awal eps.")
    zero_nb = np.sort(D, axis=1)[:, db_ms - 1] == 0
    zmsg = ""
    if zero_nb.sum() >= 5 and kind == "DTW":
        zmsg = (f" <b>Catatan:</b> {fmt(int(zero_nb.sum()))} produk berjarak <b>0</b> ke tetangganya (bagian datar di kiri "
                f"grafik). Ini produk yang sangat jarang terjual (median total {fmt(df.loc[zero_nb, 'Total_Sales'].median())} "
                "unit setahun). Setelah normalisasi polanya hanya berisi lonjakan sesekali, dan DTW dapat menyejajarkan "
                "lonjakan itu dengan minggu mana pun, sehingga produk-produk ini dianggap identik walaupun minggu "
                "penjualannya berbeda. Ini keterbatasan DTW pada deret yang sangat jarang.")
    ins_box(f"Pada eps = {fmt(db_eps, 3)}, <b>{pct(core_share)}</b> produk memenuhi syarat titik inti. Titik siku berada "
            f"di jarak {fmt(kv, 3)}; produk di sebelah kanannya terletak di daerah renggang. Bila kurva naik landai tanpa "
            "patahan tegas, kepadatan antarproduk relatif merata, sehingga DBSCAN sulit memisahkan kelompok." + zmsg)

    # ---------------- Langkah 2: pencarian kombinasi ----------------
    g = db_grid.copy()
    n_comb, n_ok = len(g), int(g["Layak"].sum())
    section("Langkah 2 · Pencarian kombinasi parameter",
            f"{len(DB_MS)} nilai min_samples × {g['eps'].nunique()} nilai eps = {n_comb} kombinasi. "
            f"Syarat layak: ≥ 2 klaster, pencilan ≤ {DB_MAX_NOISE}%, Silhouette > 0. Dipilih Silhouette tertinggi.")
    piv_s = g.pivot(index="min_samples", columns="eps", values="Silhouette")
    piv_n = g.pivot(index="min_samples", columns="eps", values="Pencilan (%)")
    piv_c = g.pivot(index="min_samples", columns="eps", values="Jumlah klaster")
    piv_v = g.pivot(index="min_samples", columns="eps", values="Layak")
    xlab = [f"{e:.3f}" for e in piv_s.columns]
    ylab = [str(m_) for m_ in piv_s.index]
    zmax = float(np.nanmax(piv_s.where(piv_v).values)) if n_ok else 1.0
    zmin = -0.05 * max(zmax, 0.1)
    zplot = np.where(piv_v.values, piv_s.values, zmin)
    b0 = (0 - zmin) / (zmax - zmin)
    cscale = [[0, "#EEF1F5"], [b0, "#EEF1F5"], [min(b0 + 1e-3, 1), "#DCE9FF"], [1, "#104281"]]
    txt = [[(f"{s_:.2f}" if ok_ else "·") for s_, ok_ in zip(rs, rv)] for rs, rv in zip(piv_s.values, piv_v.values)]
    cd = np.dstack([piv_c.values, piv_n.values, np.nan_to_num(piv_s.values, nan=-9)])
    fig = go.Figure(go.Heatmap(
        z=zplot, x=xlab, y=ylab, zmin=zmin, zmax=zmax, colorscale=cscale, showscale=False,
        text=txt, texttemplate="%{text}", textfont=dict(size=10), xgap=3, ygap=3, customdata=cd,
        hovertemplate=("min_samples = %{y}<br>eps = %{x}<br>Klaster: %{customdata[0]}<br>"
                       "Pencilan: %{customdata[1]:.1f}%<br>Silhouette: %{customdata[2]:.3f}<extra></extra>")))
    if db_best["found"]:
        xi = list(piv_s.columns).index(db_best["eps"]) if db_best["eps"] in list(piv_s.columns) else None
        yi = list(piv_s.index).index(db_best["ms"]) if db_best["ms"] in list(piv_s.index) else None
        if xi is not None and yi is not None:
            fig.add_shape(type="rect", x0=xi - 0.5, x1=xi + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                          line=dict(color="#E34948", width=3))
    style_fig(fig, height=330, title="Koefisien Silhouette tiap kombinasi (sel abu-abu = tidak layak; kotak merah = optimal)",
              legend=False)
    fig.update_xaxes(title="eps", tickangle=-45, showgrid=False)
    fig.update_yaxes(title="min_samples", showgrid=False)
    show(fig)

    txt2 = [[f"{int(c_)} kl<br>{n_:.0f}%" for c_, n_ in zip(rc, rn)] for rc, rn in zip(piv_c.values, piv_n.values)]
    fig = go.Figure(go.Heatmap(
        z=piv_n.values, x=xlab, y=ylab, zmin=0, zmax=100,
        colorscale=[[0, "#F4F8FF"], [0.3, "#B7D3F6"], [0.6, "#5598E7"], [1, "#184F95"]], showscale=False,
        text=txt2, texttemplate="%{text}", textfont=dict(size=9), xgap=3, ygap=3,
        hovertemplate="min_samples = %{y}<br>eps = %{x}<br>Pencilan: %{z:.1f}%<extra></extra>"))
    style_fig(fig, height=330, title="Jumlah klaster dan % pencilan tiap kombinasi", legend=False)
    fig.update_xaxes(title="eps", tickangle=-45, showgrid=False)
    fig.update_yaxes(title="min_samples", showgrid=False)
    show(fig)
    read_box("Baris = min_samples, kolom = eps. <b>Atas</b>: angka di sel adalah koefisien Silhouette (tanpa pencilan); "
             "sel abu-abu bertanda titik tidak memenuhi syarat. <b>Bawah</b>: “kl” = jumlah klaster yang terbentuk, "
             "persen = proporsi pencilan; warna makin gelap berarti pencilan makin banyak.")
    big_eps = g[g["eps"] >= g["eps"].median()]
    merge_share = float((big_eps["Jumlah klaster"] <= 1).mean())
    msg = f"Dari {n_comb} kombinasi, <b>{n_ok}</b> memenuhi syarat. "
    if merge_share > 0.5:
        msg += ("Pada eps yang lebih besar, hampir semua produk melebur menjadi <b>satu klaster</b>; pada eps yang lebih "
                "kecil, pencilan melonjak. Pola ini menandakan tidak ada celah kepadatan yang jelas di antara kelompok "
                "produk: produk tersebar sebagai satu kesatuan yang padat secara merata. ")
    if db_best["found"]:
        b_ = g[(g["eps"] == db_best["eps"]) & (g["min_samples"] == db_best["ms"])].iloc[0]
        msg += (f"Kombinasi optimal (eps = {fmt(db_best['eps'], 3)}, min_samples = {db_best['ms']}) menghasilkan "
                f"{int(b_['Jumlah klaster'])} klaster, {pct(b_['Pencilan (%)'])} pencilan, dan Silhouette "
                f"{fmt(b_['Silhouette'], 3)}; klaster terbesarnya memuat {pct(b_['Klaster terbesar (%)'])} produk.")
    ins_box(msg)
    if n_ok:
        top = g[g["Layak"]].sort_values(["Silhouette", "Pencilan (%)"], ascending=[False, True]).head(10)
        with st.expander("📋 10 kombinasi layak terbaik"):
            st.dataframe(top.drop(columns="Layak").style.format({"eps": "{:.3f}", "Pencilan (%)": "{:.1f}",
                                                                 "Silhouette": "{:.3f}", "Klaster terbesar (%)": "{:.1f}"}),
                         hide_index=True, width="stretch")

    # ---------------- Langkah 3: hasil ----------------
    section("Langkah 3 · Hasil pengelompokan DBSCAN", f"eps = {fmt(db_eps, 3)}, min_samples = {db_ms}.")
    r1, r2 = st.columns([1, 1])
    counts = pd.Series(db_disp).value_counts().reindex(db_groups).fillna(0).astype(int)
    with r1:
        fig = go.Figure(go.Bar(
            y=counts.index[::-1], x=counts.values[::-1], orientation="h",
            marker=dict(color=[db_color(n_) for n_ in counts.index[::-1]], cornerradius=6),
            text=[f"{v} ({pct(v / len(df) * 100)})" for v in counts.values[::-1]], textposition="outside",
            hovertemplate="%{y}<br>%{x} produk<extra></extra>"))
        style_fig(fig, height=120 + 46 * len(counts), title="Jumlah produk per kelompok DBSCAN", legend=False)
        fig.update_xaxes(title="Jumlah produk", range=[0, counts.max() * 1.3], showgrid=True, gridcolor=GRID)
        show(fig)
    with r2:
        ct = pd.crosstab(pd.Series([cname[c] for c in labels], name="K-Medoids"),
                         pd.Series(db_disp, name="DBSCAN")).reindex(columns=db_groups, fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=ct.values, x=list(ct.columns), y=list(ct.index),
            colorscale=[[0, "#F4F8FF"], [0.25, "#B7D3F6"], [0.6, "#3987E5"], [1, "#104281"]],
            text=ct.values, texttemplate="%{text}", textfont=dict(size=13), xgap=4, ygap=4, showscale=False,
            hovertemplate="%{y} & %{x}<br>%{z} produk<extra></extra>"))
        style_fig(fig, height=max(260, 120 + 46 * len(counts)), title="Tabulasi silang K-Medoids × DBSCAN", legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig)

    ng = len(db_groups)
    ncol_d = min(ng, 3)
    nrow_d = int(np.ceil(ng / ncol_d))
    fig = make_subplots(rows=nrow_d, cols=ncol_d, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.16,
                        subplot_titles=[f"{g_} (n = {int(counts[g_])})" for g_ in db_groups])
    for n_, g_ in enumerate(db_groups):
        r_, c_ = n_ // ncol_d + 1, n_ % ncol_d + 1
        M = Xn[db_disp == g_]
        q1, q2, q3 = np.percentile(M, [25, 50, 75], axis=0)
        col = db_color(g_)
        fig.add_scatter(x=xw + xw[::-1], y=list(q3) + list(q1[::-1]), fill="toself", fillcolor=hex_rgba(col, 0.2),
                        line=dict(width=0), hoverinfo="skip", showlegend=False, row=r_, col=c_)
        fig.add_scatter(x=xw, y=q2, mode="lines", line=dict(color=col, width=3), showlegend=False,
                        hovertemplate="W%{x}<br>Median: %{y:.2f}<extra></extra>", row=r_, col=c_)
        fig.add_scatter(x=xw, y=M.mean(axis=0), mode="lines", line=dict(color=NAVY, width=1.3, dash="dot"),
                        showlegend=False, hovertemplate="W%{x}<br>Rata-rata: %{y:.2f}<extra></extra>", row=r_, col=c_)
    style_fig(fig, height=300 * nrow_d, legend=False)
    fig.update_xaxes(tickvals=list(range(0, 52, 8)), ticktext=[f"W{i}" for i in range(0, 52, 8)])
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.update_annotations(font=dict(size=13, color=NAVY))
    show(fig)
    read_box("Pola mingguan tiap kelompok DBSCAN (skala normalisasi 0–1). Garis tebal = median, pita = rentang "
             "antarkuartil, garis titik-titik = rata-rata. Kelompok <b>Pencilan</b> berwarna abu-abu.")

    d_tot = df["Total_Sales"].sum()
    stat_db = {}
    for g_ in db_groups:
        sub = df[db_disp == g_]
        stat_db[g_] = [fmt(len(sub)), pct(len(sub) / len(df) * 100), pct(sub["Total_Sales"].sum() / d_tot * 100),
                       fmt(sub["Mean_Sales"].median(), 2), fmt(sub["CV_Sales"].median(), 2),
                       pct(sub["Zero_Sales_Pct"].median())]
    st.dataframe(pd.DataFrame(stat_db, index=["Jumlah produk", "Porsi produk", "Porsi unit terjual",
                                               "Median unit per minggu", "Median koefisien variasi",
                                               "Median minggu tanpa penjualan"]).rename_axis("Statistik"),
                 width="stretch")

    coords, prop = classical_mds(dataset, kind)
    fig = go.Figure()
    for i_, g_ in enumerate(db_groups):
        m_ = db_disp == g_
        fig.add_scatter(x=coords[m_, 0], y=coords[m_, 1], mode="markers", name=g_,
                        marker=dict(size=7 if g_ == "Pencilan" else 8, color=hex_rgba(db_color(g_), 0.75),
                                    symbol="x" if g_ == "Pencilan" else CLUSTER_SYMBOLS[i_ % 6],
                                    line=dict(color="white", width=0 if g_ == "Pencilan" else 1)),
                        text=df["Product_Code"][m_], hovertemplate="%{text}<extra>" + g_ + "</extra>")
    style_fig(fig, height=460, title="Peta kemiripan produk menurut kelompok DBSCAN")
    fig.update_xaxes(title=f"Dimensi 1 ({pct(prop[0] * 100)} variasi)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=f"Dimensi 2 ({pct(prop[1] * 100)} variasi)")
    show(fig)

    # interpretasi hasil
    parts = []
    if db_nc == 0:
        parts.append("Semua produk menjadi pencilan; eps terlalu kecil atau min_samples terlalu besar.")
    else:
        main = "Klaster D1"
        main_mix = ct[main] / ct[main].sum() if main in ct.columns and ct[main].sum() else None
        parts.append(f"Klaster terbesar (<b>{main}</b>) memuat <b>{pct(db_largest)}</b> produk.")
        if main_mix is not None and (main_mix >= 0.2).sum() >= 2:
            mix_txt = ", ".join(f"{pct(v * 100)} dari {i}" for i, v in main_mix.items() if v > 0)
            parts.append(f"Isinya campuran klaster K-Medoids ({mix_txt}), sehingga DBSCAN <b>tidak memisahkan produk "
                         "laris dan jarang terjual</b>: dalam jarak " + kind + " kedua jenis pola berada di satu daerah "
                         "padat yang sama.")
        small = [g_ for g_ in db_groups if g_ not in ("Pencilan", main) and counts[g_] < 0.05 * len(df)]
        if small:
            zs = ", ".join(f"{g_} ({int(counts[g_])} produk, median {pct(df[db_disp == g_]['Zero_Sales_Pct'].median())} "
                           "minggu tanpa penjualan)" for g_ in small)
            parts.append(f"Klaster kecil yang terbentuk: {zs}. Ini adalah produk dengan pola ekstrem yang saling mirip.")
    if (db_lab == -1).any():
        pz = df[db_disp == "Pencilan"]["Zero_Sales_Pct"].median()
        parts.append(f"Sebanyak {pct(db_noise)} produk menjadi pencilan (median {pct(pz)} minggu tanpa penjualan): "
                     "polanya tidak cukup mirip dengan produk lain untuk masuk daerah padat mana pun, sehingga layak "
                     "ditinjau satu per satu.")
    ins_box(" ".join(parts))

# =====================================================================
# TAB 6 — PERBANDINGAN METODE
# =====================================================================
with T_PERB:
    section("⚖️ Perbandingan Tiga Metode",
            "K-Medoids (utama), klaster hierarki (average linkage), dan DBSCAN pada matriks jarak yang sama.")
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "🎯", "Silhouette K-Medoids", fmt(sil, 3), f"k = {k} · {silhouette_label(sil)[0]}")
    kpi(m2, "🌳", "Silhouette Hierarki", fmt(sil_hier, 3), f"k = {k} · {silhouette_label(sil_hier)[0]}")
    kpi(m3, "🌀", "Silhouette DBSCAN", "–" if np.isnan(db_sil) else fmt(db_sil, 3),
        f"{db_nc} klaster · tanpa pencilan")
    kpi(m4, "⚪", "Pencilan DBSCAN", pct(db_noise), "tidak masuk klaster mana pun")
    st.write("")
    a1, a2, a3 = st.columns(3)
    kpi(a1, "🤝", "ARI K-Medoids – Hierarki", fmt(ari, 3), ari_text(ari))
    kpi(a2, "🤝", "ARI K-Medoids – DBSCAN", fmt(ari_kd, 3), ari_text(ari_kd))
    kpi(a3, "🤝", "ARI Hierarki – DBSCAN", fmt(ari_hd, 3), ari_text(ari_hd))
    st.write("")
    read_box("<b>ARI (Indeks Rand terkoreksi)</b> mengukur kesepakatan dua pengelompokan: 0 = setara kebetulan, "
             "1 = identik. Pada DBSCAN, pencilan dihitung sebagai satu kelompok tersendiri. "
             "<b>Catatan</b>: Silhouette DBSCAN dihitung tanpa pencilan, sehingga tidak sepenuhnya setara dengan "
             "Silhouette K-Medoids dan hierarki yang mencakup semua produk.")

    p1, p2 = st.columns([1, 1])
    with p1:
        ct = pd.crosstab(pd.Series(labels, name="K-Medoids"), pd.Series(hl, name="Hierarki"))
        fig = go.Figure(go.Heatmap(
            z=ct.values, x=[f"Hierarki {c}" for c in ct.columns], y=[f"K-Medoids {c}" for c in ct.index],
            colorscale=[[0, "#F4F8FF"], [0.25, "#B7D3F6"], [0.6, "#3987E5"], [1, "#104281"]],
            text=ct.values, texttemplate="%{text}", textfont=dict(size=14), xgap=4, ygap=4,
            hovertemplate="%{y} & %{x}<br>%{z} produk<extra></extra>", showscale=False))
        style_fig(fig, height=360, title="Tabulasi silang K-Medoids × Hierarki", legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig)
    with p2:
        # ukuran klaster tiap metode (batang bertumpuk)
        seg_cols = ["#184f95", "#2a78d6", "#6da7ec", "#9ec5f4", "#b7d3f6", "#cde2fb"]
        methods = {
            "K-Medoids": [(f"Klaster {c}", int(prof.loc[c, "n"])) for c in clusters],
            "Hierarki": [(f"Klaster {i + 1}", int(v)) for i, v in enumerate(hsize)],
            "DBSCAN": [(g_, int((db_disp == g_).sum())) for g_ in db_groups],
        }
        fig = go.Figure()
        for mname, parts_ in methods.items():
            ordered = sorted([p_ for p_ in parts_ if p_[0] != "Pencilan"], key=lambda t: -t[1]) + \
                      [p_ for p_ in parts_ if p_[0] == "Pencilan"]
            for j, (lbl, n_) in enumerate(ordered):
                col = NOISE_COLOR if lbl == "Pencilan" else seg_cols[min(j, len(seg_cols) - 1)]
                fig.add_bar(y=[mname], x=[n_], orientation="h", showlegend=False,
                            marker=dict(color=col, line=dict(color="white", width=2)),
                            text=[f"{n_}" if n_ / len(df) > 0.04 else ""], textposition="inside",
                            insidetextanchor="middle", textfont=dict(color="white" if j < 2 and lbl != "Pencilan" else NAVY),
                            hovertemplate=f"{mname} · {lbl}<br>{n_} produk<extra></extra>")
        fig.add_bar(y=[None], x=[None], name="Klaster (urut dari terbesar)", marker=dict(color=seg_cols[0]))
        fig.add_bar(y=[None], x=[None], name="Pencilan (DBSCAN)", marker=dict(color=NOISE_COLOR))
        style_fig(fig, height=360, title="Ukuran klaster tiap metode")
        fig.update_layout(barmode="stack", bargap=0.45)
        fig.update_xaxes(title="Jumlah produk", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(categoryorder="array", categoryarray=["DBSCAN", "Hierarki", "K-Medoids"])
        show(fig)
    read_box("<b>Kiri</b>: angka di tiap sel = jumlah produk yang masuk klaster K-Medoids (baris) dan klaster hierarki "
             "(kolom). Jika kedua metode sepakat, angka besar hanya muncul di satu sel per baris. <b>Kanan</b>: "
             "setiap batang adalah satu metode; potongan batang = klaster, diurutkan dari yang terbesar. "
             "Batang yang didominasi satu potongan berarti metode itu memasukkan hampir semua produk ke satu klaster.")

    with st.expander("🌳 Dendrogram klaster hierarki"):
        dn = dendrogram(Z, truncate_mode="lastp", p=15, no_plot=True)
        fig = go.Figure()
        for xs, ys in zip(dn["icoord"], dn["dcoord"]):
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=2), hoverinfo="skip", showlegend=False)
        cut = Z[-(k - 1), 2] if k > 1 else Z[-1, 2]
        prev_cut = Z[-k, 2] if k < len(Z) else 0
        fig.add_hline(y=(cut + prev_cut) / 2, line=dict(color="#E34948", dash="dash"))
        fig.add_annotation(x=1, xref="paper", y=(cut + prev_cut) / 2, text=f"Potongan k = {k}", showarrow=False,
                           xanchor="right", yshift=10, font=dict(color="#B42318", size=11))
        style_fig(fig, height=380, title="Dendrogram (15 gabungan teratas)", legend=False)
        fig.update_xaxes(tickvals=[5 + 10 * i for i in range(len(dn["ivl"]))], ticktext=dn["ivl"], tickangle=-45,
                         tickfont=dict(size=9), title="Kelompok produk (angka dalam kurung = jumlah produk)")
        fig.update_yaxes(title=f"Jarak {kind} (rata-rata)")
        show(fig)
        st.caption("Garis yang bergabung makin tinggi berarti kelompok makin berbeda; garis merah = tempat pohon "
                   "dipotong menjadi k klaster.")

    lk = prof["n"].max() / len(df) * 100
    lh = hsize.max() / len(df) * 100
    small_h = hsize.min() / len(df) * 100
    concl = (f"<b>K-Medoids</b> membagi produk menjadi klaster berukuran {', '.join(str(int(x)) for x in prof['n'])} "
             f"(terbesar {pct(lk)}). <b>Klaster hierarki</b> menempatkan {pct(lh)} produk dalam satu klaster "
             f"(ukuran {', '.join(map(str, hsize))}). <b>DBSCAN</b> membentuk {db_nc} klaster dengan klaster terbesar "
             f"{pct(db_largest)} dan {pct(db_noise)} pencilan. ")
    if lh >= 70 and db_largest >= 70:
        concl += (f"Kedua metode pembanding sepakat bahwa produk <b>tidak terbagi menjadi kelompok yang terpisah "
                  f"secara alami</b>: pola penjualan membentuk satu sebaran yang bersambung, dan hanya sedikit produk "
                  f"ekstrem yang memisah. Temuan ini sejalan dengan Silhouette K-Medoids yang tergolong "
                  f"<b>{sil_cat.lower()}</b>. Karena itu, segmentasi K-Medoids sebaiknya dibaca sebagai "
                  "<b>pembagian praktis</b> untuk menyusun strategi, bukan sebagai kelompok alami. K-Medoids tetap "
                  "dipakai sebagai acuan karena menghasilkan kelompok berukuran seimbang yang dapat diberi strategi "
                  "berbeda; klaster kecil hierarki dan pencilan DBSCAN berguna untuk menandai produk ekstrem.")
    else:
        concl += (f"Kesepakatan K-Medoids dengan hierarki (ARI {fmt(ari, 3)}) dan dengan DBSCAN (ARI {fmt(ari_kd, 3)}) "
                  "menunjukkan seberapa konsisten struktur klaster antarmetode; makin tinggi, makin meyakinkan.")
    box("good", "🏁 Kesimpulan perbandingan", concl)

    comp = pd.DataFrame({
        "Aspek": ["Prinsip", "Parameter", "Pusat klaster", "Kelebihan", "Keterbatasan", "Hasil pada data ini"],
        "K-Medoids": ["Membagi produk ke k kelompok dengan meminimalkan jarak ke medoid", "k",
                      "Produk nyata (medoid)", "Tahan terhadap pencilan; medoid mudah ditafsirkan",
                      "k harus ditentukan di awal; selalu memaksa semua produk masuk klaster",
                      f"Silhouette {fmt(sil, 3)}; ukuran {', '.join(str(int(x)) for x in prof['n'])}"],
        "Klaster Hierarki": ["Menggabungkan produk paling mirip secara bertahap (average linkage)",
                             "k (tinggi potongan pohon)", "Tidak ada; berupa pohon gabungan",
                             "Struktur bertingkat terlihat di dendrogram",
                             "Peka terhadap pencilan; klaster bisa sangat timpang",
                             f"Silhouette {fmt(sil_hier, 3)}; ukuran {', '.join(map(str, hsize))}"],
        "DBSCAN": ["Mengelompokkan produk di daerah padat; produk di daerah renggang menjadi pencilan",
                   "eps dan min_samples", "Tidak ada; klaster dibentuk dari rangkaian titik inti",
                   "Tidak perlu menentukan k; mengenali pencilan secara otomatis",
                   "Peka terhadap eps dan min_samples; sulit bila kepadatan data merata",
                   ("Silhouette –" if np.isnan(db_sil) else f"Silhouette {fmt(db_sil, 3)}")
                   + f"; {db_nc} klaster; pencilan {pct(db_noise)}"],
    })
    st.table(comp.set_index("Aspek"))

    ext_rows = []
    for i_, c_ in enumerate(hsize, start=1):
        if c_ / len(df) < 0.05:
            for code in df["Product_Code"][hl == i_]:
                ext_rows.append({"Kode produk": code, "Dipisahkan oleh": f"Hierarki (Klaster {i_}, {c_} produk)"})
    for g_ in db_groups:
        if g_ != "Pencilan" and (db_disp == g_).sum() / len(df) < 0.05:
            for code in df["Product_Code"][db_disp == g_]:
                ext_rows.append({"Kode produk": code, "Dipisahkan oleh": f"DBSCAN ({g_})"})
    if ext_rows:
        ext = (pd.DataFrame(ext_rows).groupby("Kode produk")["Dipisahkan oleh"].agg("; ".join).reset_index()
               .merge(df[["Product_Code", "Total_Sales", "Zero_Sales_Count", "CV_Sales"]],
                      left_on="Kode produk", right_on="Product_Code").drop(columns="Product_Code")
               .rename(columns={"Total_Sales": "Total unit", "Zero_Sales_Count": "Minggu tanpa jual",
                                "CV_Sales": "Koef. variasi"}).round(2))
        ext.insert(1, "Klaster K-Medoids", [cname[labels[int(df.index[df["Product_Code"] == c][0])]]
                                            for c in ext["Kode produk"]])
        with st.expander(f"🔎 Produk ekstrem yang dipisahkan metode pembanding ({len(ext)} produk)"):
            st.dataframe(ext, hide_index=True, width="stretch")
            st.caption("Klaster berisi < 5% produk pada klaster hierarki atau DBSCAN. Produk ini memiliki pola yang "
                       "sangat berbeda dari mayoritas dan layak ditinjau secara khusus.")


# =====================================================================
# TAB 7 — DAFTAR PRODUK
# =====================================================================
with T_DAFTAR:
    section("📋 Daftar Produk dan Klasternya")
    out = pd.DataFrame({
        "Kode produk": df["Product_Code"],
        "Klaster": [cname[c] for c in labels],
        "Segmen": [seg[c] for c in labels],
        "Medoid": ["Ya" if i in medoids.values() else "" for i in range(len(df))],
        "Silhouette": sil_each.round(3),
        "Total unit": df["Total_Sales"],
        "Rata-rata/minggu": df["Mean_Sales"].round(2),
        "Median/minggu": df["Median_Sales"].round(2),
        "Simpangan baku": df["SD_Sales"].round(2),
        "Koef. variasi": df["CV_Sales"].round(2),
        "Kategori variasi": df["Variability"].map({"Low": "Rendah", "Medium": "Sedang", "High": "Tinggi"}),
        "Minggu tanpa jual": df["Zero_Sales_Count"],
        "Klaster hierarki": hl,
        "Kelompok DBSCAN": db_disp,
    })
    s1, s2 = st.columns([2, 1])
    with s1:
        pick = st.multiselect("Filter klaster", [cname[c] for c in clusters], default=[cname[c] for c in clusters])
    with s2:
        q = st.text_input("Cari kode produk", placeholder="contoh: P409")
    view = out[out["Klaster"].isin(pick)]
    if q:
        view = view[view["Kode produk"].str.contains(q.strip(), case=False)]
    st.caption(f"Menampilkan {fmt(len(view))} dari {fmt(len(out))} produk.")
    st.dataframe(view, hide_index=True, width="stretch", height=380,
                 column_config={"Silhouette": st.column_config.ProgressColumn(
                     "Silhouette", min_value=-1.0, max_value=1.0, format="%.3f")})
    st.download_button("⬇️ Unduh hasil klaster (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"hasil_klaster_{kind.lower()}_k{k}.csv", mime="text/csv")

    section("🔎 Lihat Pola Satu Produk")
    prod = st.selectbox("Pilih produk", df["Product_Code"].tolist(),
                        index=int(np.argmax(df["Total_Sales"].to_numpy())), key="prod_detail")
    i = int(df.index[df["Product_Code"] == prod][0])
    c = int(labels[i])
    mi = medoids[c]
    d1, d2 = st.columns([2, 1])
    with d1:
        fig = go.Figure()
        fig.add_scatter(x=WEEKS, y=Xn[i], mode="lines+markers", name=prod, line=dict(color=ccolor(c), width=2.5),
                        marker=dict(size=6), hovertemplate="%{x}: %{y:.2f}<extra>" + prod + "</extra>")
        fig.add_scatter(x=WEEKS, y=Xn[mi], mode="lines", name=f"Medoid {df.loc[mi, 'Product_Code']}",
                        line=dict(color=NAVY, width=1.5, dash="dash"),
                        hovertemplate="%{x}: %{y:.2f}<extra>medoid</extra>")
        style_fig(fig, height=340, title=f"{prod} dibanding medoid {cname[c]} (skala normalisasi)")
        fig.update_xaxes(tickvals=WEEKS[::4])
        show(fig)
    with d2:
        r = out.iloc[i]
        sv = float(r["Silhouette"])
        fit = "sangat cocok" if sv > 0.5 else "cukup cocok" if sv > 0.25 else "berada di perbatasan" if sv >= 0 else "lebih mirip klaster lain"
        st.markdown(
            f"""<div class="ccard"><div class="tag"><span class="dot" style="background:{ccolor(c)}"></span>{prod}</div>
<div class="seg">{cname[c]} · {seg[c]}</div>
<div class="grid">
<div class="mini"><div class="l">Total unit</div><div class="v">{fmt(r['Total unit'])}</div></div>
<div class="mini"><div class="l">Rata-rata/minggu</div><div class="v">{fmt(r['Rata-rata/minggu'], 2)}</div></div>
<div class="mini"><div class="l">Koef. variasi</div><div class="v">{fmt(r['Koef. variasi'], 2)}</div></div>
<div class="mini"><div class="l">Minggu tanpa jual</div><div class="v">{r['Minggu tanpa jual']}</div></div>
<div class="mini"><div class="l">Silhouette</div><div class="v">{fmt(sv, 3)}</div></div>
<div class="mini"><div class="l">Jarak ke medoid</div><div class="v">{fmt(D[i, mi], 2)}</div></div>
</div>
<div style="font-size:.86rem;color:#33445C;">Produk ini <b>{fit}</b> dengan klasternya.</div></div>""",
            unsafe_allow_html=True)

# =====================================================================
# TAB 8 — METODOLOGI
# =====================================================================
with T_METOD:
    section("🛠️ Alur Analisis")
    steps = [
        ("Pemeriksaan data", "Cek data kosong, kode ganda, nilai negatif, dan konsistensi kolom normalisasi."),
        ("Statistik deskriptif", "Hitung total, rata-rata, median, simpangan baku, koefisien variasi, dan minggu tanpa penjualan."),
        ("Normalisasi min–maks", "Skala tiap produk diubah ke 0–1 agar yang dibandingkan adalah bentuk pola."),
        ("Matriks jarak DTW", "Hitung jarak antarsetiap pasang produk; DTW memperbolehkan pergeseran waktu."),
        ("K-Medoids & hierarki", "Kelompokkan produk untuk k = 2–6; k optimal = Silhouette K-Medoids tertinggi."),
        ("DBSCAN", "Cari kombinasi eps × min_samples; pilih Silhouette tertinggi dengan ≥ 2 klaster dan pencilan ≤ 30%."),
        ("Evaluasi & profil", "Bandingkan tiga metode (Silhouette, ARI, ukuran klaster), tentukan medoid, susun rekomendasi."),
    ]
    for row_steps, start_n in [(steps[:4], 1), (steps[4:], 5)]:
        cols = st.columns(4)
        for n_, (col, (h, d)) in enumerate(zip(cols, row_steps), start=start_n):
            col.markdown(f'<div class="step"><div class="n">{n_}</div><div class="h">{h}</div><div class="d">{d}</div></div>',
                         unsafe_allow_html=True)
        st.write("")

    section("📚 Glosarium Istilah")
    gl = pd.DataFrame([
        ("Analisis klaster", "Teknik statistik untuk mengelompokkan objek (di sini: produk) sehingga anggota satu kelompok saling mirip."),
        ("Klaster", "Kelompok produk dengan pola penjualan serupa."),
        ("K-Medoids (PAM)", "Metode pengelompokan yang memilih k produk nyata sebagai pusat (medoid) dan memasukkan produk lain ke medoid terdekat."),
        ("Medoid", "Anggota klaster yang total jaraknya ke anggota lain paling kecil; produk paling representatif."),
        ("Klaster hierarki (average linkage)", "Menggabungkan produk/kelompok paling mirip secara bertahap; jarak antarkelompok = rata-rata jarak semua pasangan anggotanya."),
        ("Dendrogram", "Diagram pohon hasil klaster hierarki; tinggi gabungan menunjukkan tingkat perbedaan."),
        ("Dynamic Time Warping (DTW)", "Ukuran jarak dua deret waktu yang menyejajarkan titik-titik secara lentur, sehingga pola yang sama tetapi bergeser beberapa minggu tetap dianggap mirip."),
        ("Jarak Euklidean", "Akar jumlah kuadrat selisih nilai pada minggu yang sama; tidak memperhitungkan pergeseran waktu."),
        ("Normalisasi min–maks", "(nilai − minimum) ÷ (maksimum − minimum); mengubah skala menjadi 0–1."),
        ("Koefisien variasi (KV)", "Simpangan baku ÷ rata-rata; ukuran variasi relatif yang dapat dibandingkan antarproduk."),
        ("Koefisien Silhouette", "Ukuran kecocokan objek dengan klasternya dibanding klaster terdekat lain; rentang −1 sampai 1."),
        ("Indeks Rand terkoreksi (ARI)", "Ukuran kesepakatan dua pengelompokan, dikoreksi untuk kebetulan; 0 = setara acak, 1 = identik."),
        ("DBSCAN", "Metode pengelompokan berbasis kepadatan: produk di daerah padat membentuk klaster, produk di daerah renggang menjadi pencilan; jumlah klaster tidak ditentukan di awal."),
        ("eps", "Radius ketetanggaan DBSCAN: dua produk bertetangga bila jaraknya ≤ eps."),
        ("min_samples", "Jumlah minimum produk (termasuk dirinya) dalam radius eps agar suatu produk menjadi titik inti."),
        ("Titik inti", "Produk yang memiliki sedikitnya min_samples produk dalam radius eps; menjadi pusat daerah padat."),
        ("Pencilan (noise)", "Produk yang tidak terjangkau titik inti mana pun sehingga tidak masuk klaster DBSCAN."),
        ("Grafik jarak tetangga ke-k", "Jarak tiap produk ke tetangga terdekat ke-k, diurutkan; titik sikunya dipakai sebagai acuan eps."),
        ("Penskalaan multidimensi klasik", "Teknik meringkas matriks jarak menjadi koordinat 2 dimensi agar dapat digambar."),
        ("Rentang antarkuartil", "Selisih kuartil ketiga (75%) dan kuartil pertama (25%); memuat 50% data di tengah."),
    ], columns=["Istilah", "Arti"])
    st.dataframe(gl, hide_index=True, width="stretch", height=(len(gl) + 1) * 35 + 4)

    with st.expander("🔧 Catatan teknis"):
        st.markdown(
            f"""
- **Data**: {len(all_df)} produk (semua) atau {len(filt_df)} produk (terfilter: minggu tanpa penjualan ≤ 45).
- **Normalisasi** dihitung ulang dari kolom W0–W51 agar presisi penuh (kolom *Normalized* di data asli dibulatkan 2 desimal).
- **Jarak DTW** dihitung penuh tanpa batas jendela; hasil diverifikasi terhadap perhitungan manual.
- **K-Medoids** memakai prosedur PAM (BUILD + SWAP) pada matriks jarak.
- **Klaster hierarki** memakai *average linkage* pada matriks jarak dalam bentuk ringkas (condensed).
- **Silhouette** dihitung dari matriks jarak yang sama dengan yang dipakai untuk pengelompokan.
- **k optimal** = nilai k (2–6) dengan Silhouette K-Medoids tertinggi untuk data dan ukuran jarak yang dipilih.
- **DBSCAN**: dicoba min_samples ∈ {{3, 4, 5, 6, 8, 10}} × 16 nilai eps (persentil ke-30 jarak tetangga ke-2 sampai
  persentil ke-97 jarak tetangga ke-9). Kombinasi layak: ≥ 2 klaster, pencilan ≤ 30%, Silhouette > 0; dipilih
  Silhouette tertinggi (seri → pencilan lebih sedikit). Jika tidak ada yang layak, dipakai titik siku grafik jarak
  tetangga untuk min_samples = 5.
- **Silhouette DBSCAN** dihitung tanpa pencilan; **ARI** dengan DBSCAN memperlakukan pencilan sebagai satu kelompok.
- **Penomoran klaster** diurutkan dari median total penjualan tertinggi (Klaster 1) ke terendah.
- **Nama segmen** ditentukan dari median klaster: minggu tanpa penjualan ≥ 75% → *Sangat Jarang Terjual*;
  50–75% → *Jarang Terjual*;
  KV < 0,5 → *Stabil*; 0,5–1 → *Cukup Berfluktuasi*; > 1 → *Sangat Berfluktuasi*.
"""
        )
    box("read", "📌 Rujukan",
        "Kaufman, L. & Rousseeuw, P. J. (1990). <i>Finding Groups in Data: An Introduction to Cluster Analysis</i>. Wiley.<br>"
        "Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. "
        "<i>IEEE Trans. ASSP</i>, 26(1), 43–49.<br>"
        "Hubert, L. & Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2, 193–218.<br>"
        "Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based algorithm for discovering clusters in "
        "large spatial databases with noise. <i>Proceedings of KDD-96</i>, 226–231.")
