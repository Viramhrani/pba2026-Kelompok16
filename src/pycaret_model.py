import os
import pandas as pd

# PyCaret (import spesifik, bukan *)
from pycaret.classification import (
    setup,
    create_model,
    compare_models,
    evaluate_model,
    predict_model,
    save_model
)

# ==============================
# CONFIG
# ==============================
DATA_PATH = '../data/ml_reviews_ready.csv'
MODEL_PATH = '../app/best_model'


# ==============================
# 1. Load Data
# ==============================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    
    df = pd.read_csv(path)
    print("Data berhasil dimuat")
    print("Shape:", df.shape)
    print(df.head())

    return df


# ==============================
# 2. Cleaning
# ==============================
def clean_data(df):
    print("\nCek missing value:")
    print(df.isnull().sum())

    df = df.dropna()

    # validasi kolom
    required_cols = ['clean_text', 'sentiment']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan")

    print("Data setelah cleaning:", df.shape)
    return df


# ==============================
# 3. Setup PyCaret
# ==============================
def setup_model(df):
    print("\nSetup PyCaret...")

    clf = setup(
        data=df,
        target='sentiment',
        text_features=['clean_text'],
        session_id=42,
        train_size=0.8,
        fold=5,
        verbose=False
    )

    return clf


# ==============================
# 4. Training Model
# ==============================
def train_models():
    print("\nTraining model...")

    lr = create_model('lr')
    nb = create_model('nb')
    rf = create_model('rf')

    return lr, nb, rf


# ==============================
# 5. Compare Model
# ==============================
def select_best_model():
    print("\nMembandingkan model...")
    
    best_model = compare_models(include=['lr', 'nb', 'rf'])

    print("Model terbaik:", best_model)
    return best_model


# ==============================
# 6. Evaluasi
# ==============================
def evaluate(best_model):
    print("\nEvaluasi model...")
    evaluate_model(best_model)


# ==============================
# 7. Prediksi
# ==============================
def test_prediction(best_model):
    print("\nTesting prediksi...")

    new_data = pd.DataFrame({
        'clean_text': [
            'game nya seru banget dan skin nya keren',
            'server sering lag dan banyak bug setelah update'
        ]
    })

    hasil = predict_model(best_model, data=new_data)

    print("\nHasil prediksi:")
    print(hasil)

    return hasil


# ==============================
# 8. Save Model
# ==============================
def save(best_model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_model(best_model, path)
    print(f"\nModel berhasil disimpan di: {path}")


# ==============================
# MAIN
# ==============================
def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    setup_model(df)

    train_models()
    best_model = select_best_model()

    evaluate(best_model)
    test_prediction(best_model)

    save(best_model, MODEL_PATH)


if __name__ == "__main__":
    main()