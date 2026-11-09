# Dashboard Segmentasi Pola Penjualan Produk

Dashboard Streamlit untuk mengelompokkan 811 produk berdasarkan kemiripan pola penjualan mingguan (52 minggu)
memakai **K-Medoids** dengan ukuran jarak **Dynamic Time Warping (DTW)**. Dua metode pembanding:
**klaster hierarki (average linkage)** dan **DBSCAN** (dengan optimasi eps × min_samples).
Jumlah klaster optimal (k dengan Silhouette K-Medoids tertinggi) ditampilkan di panel kiri.

## Isi folder

```
dashboard_segmentasi/
├── app.py                         # kode dashboard
├── requirements.txt               # paket Python yang dibutuhkan
├── README.md
├── .streamlit/config.toml         # tema warna biru-putih
└── data/
    ├── EDA_Features_Sales_All.csv       # 811 produk
    ├── EDA_Features_Sales_Filtered.csv  # 717 produk (minggu tanpa penjualan ≤ 45)
    ├── jarak_dtw.npy                    # matriks jarak DTW 811 × 811 (dihitung di notebook)
    └── jarak_euklidean.npy              # matriks jarak Euklidean 811 × 811
```

File `.npy` membuat dashboard langsung terbuka. Jika file ini tidak ada, dashboard menghitung ulang
jarak DTW sendiri (sekitar 25–60 detik pada pemuatan pertama).

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Mendapatkan link permanen (Streamlit Community Cloud, gratis)

1. Buat repositori **publik** baru di GitHub, misalnya `dashboard-segmentasi-produk`.
2. Ekstrak ZIP, lalu unggah **isi** folder `dashboard_segmentasi` (bukan foldernya) ke repositori:
   *Add file → Upload files* → seret `app.py`, `requirements.txt`, `README.md`, folder `data`, dan folder `.streamlit`
   → *Commit changes*.
   - Pastikan `app.py` berada di halaman utama repositori, sejajar dengan folder `data`.
   - Folder `.streamlit` tersembunyi di macOS (tekan `Cmd + Shift + .` untuk menampilkannya). Jika terlewat,
     dashboard tetap berjalan dengan tema biru-putih karena gaya tampilan juga ditulis di `app.py`.
3. Buka **https://share.streamlit.io**, masuk dengan akun GitHub.
4. Pilih **Create app → Deploy a public app from GitHub**.
5. Isi: *Repository* = repositori tadi, *Branch* = `main`, *Main file path* = `app.py`.
   Kolom *App URL* dapat diganti dengan nama yang mudah diingat.
6. Klik **Deploy**. Setelah 2–5 menit, link `https://<nama-anda>.streamlit.app` siap dibagikan.

Setiap perubahan yang di-*commit* ke GitHub otomatis memperbarui dashboard.
