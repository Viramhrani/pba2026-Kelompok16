import os
import re
import string
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==============================
# 1. Setup
# ==============================
tqdm.pandas()

INPUT_PATH = '../data/mobile_legends_reviews_cleaned1.csv'
OUTPUT_PATH = '../data/ml_reviews_ready.csv'


# ==============================
# 2. Load Data
# ==============================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    
    df = pd.read_csv(path)
    print("Data berhasil dimuat")
    print("Shape:", df.shape)
    print(df.head())
    
    return df


# ==============================
# 3. Label Sentimen
# ==============================
def convert_sentiment(score):
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    else:
        return 'positif'


def add_sentiment_label(df):
    if 'score' not in df.columns:
        raise ValueError("Kolom 'score' tidak ditemukan di dataset")

    df['sentiment'] = df['score'].apply(convert_sentiment)

    print("\nDistribusi label:")
    print(df['sentiment'].value_counts())

    return df


# ==============================
# 4. Seleksi & Cleaning Awal
# ==============================
def basic_cleaning(df):
    df = df[['content', 'sentiment']].copy()

    # hapus null
    df = df.dropna()

    # hapus string kosong
    df = df[df['content'].str.strip() != '']

    print("\nSetelah cleaning awal:", df.shape)

    return df


# ==============================
# 5. Preprocessing Teks
# ==============================
def init_nlp_tools():
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        print("Download stopwords NLTK...")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))

    stemmer = StemmerFactory().create_stemmer()

    return stop_words, stemmer


def clean_text(text, stop_words, stemmer):
    text = str(text).lower()

    # hapus URL, mention, hashtag
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    # hapus angka
    text = re.sub(r'\d+', '', text)

    # hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # hapus stopwords
    words = [w for w in text.split() if w not in stop_words]

    # batasi panjang
    text = ' '.join(words[:100])

    # stemming
    text = stemmer.stem(text)

    return text


def apply_preprocessing(df, stop_words, stemmer):
    print("\nMelakukan preprocessing teks...")
    df['clean_text'] = df['content'].progress_apply(
        lambda x: clean_text(x, stop_words, stemmer)
    )
    return df


# ==============================
# 6. Visualisasi
# ==============================
def plot_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='sentiment')
    plt.title('Distribusi Sentimen')
    plt.tight_layout()
    plt.show()


# ==============================
# 7. Final Cleaning & Save
# ==============================
def final_clean_and_save(df, output_path):
    # hapus kosong
    df = df[df['clean_text'].str.strip() != '']

    # hapus duplikat
    df = df.drop_duplicates(subset=['clean_text', 'sentiment'])

    df_final = df[['clean_text', 'sentiment']]

    df_final.to_csv(output_path, index=False)

    print("\nData berhasil disimpan ke:", output_path)
    print("Final shape:", df_final.shape)
    print(df_final.head())

    return df_final


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = load_data(INPUT_PATH)
    df = add_sentiment_label(df)
    df = basic_cleaning(df)

    stop_words, stemmer = init_nlp_tools()
    df = apply_preprocessing(df, stop_words, stemmer)

    plot_distribution(df)

    final_clean_and_save(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()