# NLP Text Classification Benchmark
Natural Language Processing Project — Institut Teknologi Sumatera (ITERA)
# Project Description 
Proyek ini bertujuan untuk membandingkan performa pendekatan Machine Learning (ML) dan Deep Learning (DL) dalam tugas Pemrosesan Bahasa Alami (NLP), khusunya pada analisis sentimen ulasan aplikasi.Model Machine Learning akan diimplementasikan menggunakan AutoML PyCaret, sementara model Deep Learning akan diimplementasikan menggunakan PyTorch. Kedua pendekatan ini akan dievaluasi dan dibandingkan pada dataset yang sama untuk menentukan metode mana yang memberikan hasil terbaik untuk klasifikasi teks ulasan Mobile Legends.
# Team Members: Kelompok 16 
| Name | NIM | GitHub Username |
|------|-----|----------------|
| Daris Samudra | 122450102 | @darissamudra2924 |
| Kharisa Harvany | 122450061 | @kharisaharvanny |
| Vira Putri Maharani | 122450129 | @Viramhrani |
# Dataset
Dataset yang digunakan dalam proyek ini bersumber dari ulasan publik aplikasi Mobile Legends di platform Kaggle.
- Dataaset Name : Mobile Legends App Reviews
- Dataset Link  : Kaggle - [Kaggle - Sentiment Analysis Mobile Legends App](https://www.kaggle.com/code/auliyyaaini/sentiment-analysis-mobile-legends-app?select=mobile_legends_reviews_cleaned1.csv)
- Task          : Klasifikasi sentimen (Positif, Negatif, Netral) dari teks ulasan pengguna.
## Project Objectives
Tujuan utama dari proyek ini meliputi:
1.  Melakukan *Exploratory Data Analysis (EDA)* dan preprocessing pada dataset ulasan.
2.  Mengimplementasikan model Machine Learning menggunakan *PyCaret AutoML* dengan membandingkan setidaknya 3 algoritma.
3.  Mengimplementasikan model Deep Learning menggunakan *PyTorch* dengan batasan maksimal *10 juta parameter*.
4.  Membandingkan performa antara model ML dan DL berdasarkan metrik evaluasi NLP standar.
5.  Melakukan deployment demo interaktif menggunakan *Hugging Face Spaces*.
6.  Mempublikasikan laporan ilmiah dalam format *ArXiv*.
## Repository Structure
Format struktur repositori disesuaikan dengan standar pengerjaan tugas:
```
text
pba2026-kelompok16
│
├── data/
│   ├── raw/            # Dataset asli dari Kaggle
│   └── processed/      # Dataset setelah tahap cleaning & preprocessing
│
├── notebooks/
│   ├── 01_eda.ipynb            # Eksplorasi data & visualisasi
│   ├── 02_preprocessing.ipynb  # Pembersihan teks & tokenisasi
│   ├── 03_pycaret_model.ipynb  # Benchmark AutoML PyCaret
│   └── 04_deep_learning.ipynb  # Training model PyTorch
│
├── src/
│   ├── preprocessing.py        # Skrip utilitas pengolahan teks
│   ├── train_ml.py             # Skrip pelatihan ML
│   ├── train_dl.py             # Skrip pelatihan DL
│   └── utils.py                # Fungsi pembantu lainnya
│
├── models/                     # Penyimpanan model yang telah dilatih
│
├── app/                        # Kode sumber untuk demo interaktif
│   ├── ml_demo/                # Gradio/Streamlit untuk model ML
│   └── dl_demo/                # Gradio/Streamlit untuk model DL
│
├── paper/                      # Source code LaTeX (main.tex & .bib)
│
└── README.md                   # Dokumentasi utama proyek
```
## Machine Learning Approach
Model Machine Learning dikembangkan menggunakan modul pycaret.classification. Kami akan melakukan benchmark otomatis pada beberapa algoritma, seperti:
*   Logistic Regression
*   Random Forest
*   Support Vector Machine (SVM)
*   Gradient Boosting Machines (GBM)

Model terbaik akan dipilih berdasarkan hasil evaluasi terbaik pada fase benchmarking.
