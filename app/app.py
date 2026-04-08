"""
app.py — Gradio App untuk Sentiment Analysis Review Mobile Legends
Deploy di Hugging Face Spaces
Model: PyCaret Classification Pipeline (.pkl)
"""

import re
import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# =========================================================
# LOAD MODEL
# File hasil save_model(best_model, '../app/best_model')
# =========================================================

model = load_model("best_model")

# =========================================================
# PREPROCESSING
# Samakan dengan preprocessing di notebook training
# =========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # hapus url
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # hapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# FUNGSI PREDIKSI
# =========================================================

def predict_sentiment(text):
    if not text or not text.strip():
        return {"Teks kosong": 1.0}

    cleaned = clean_text(text)

    df_input = pd.DataFrame({
        "clean_text": [cleaned]
    })

    result = predict_model(model, data=df_input)

    # PyCaret biasanya menghasilkan prediction_label dan prediction_score
    label = result["prediction_label"].iloc[0]

    if "prediction_score" in result.columns:
        score = float(result["prediction_score"].iloc[0])
    else:
        score = 1.0

    return {str(label): score}

# =========================================================
# CONTOH REVIEW
# =========================================================

examples = [
    ["game nya seru banget dan skin nya keren"],
    ["server sering lag dan banyak bug setelah update"],
    ["bagus sih tapi kadang matchmaking nya aneh"],
    ["saya suka event terbaru dan hadiahnya menarik"],
    ["aplikasi sering keluar sendiri, sangat mengecewakan"],
    ["lumayan, tapi masih perlu perbaikan"],
]

# =========================================================
# GRADIO INTERFACE
# =========================================================

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Tulis Review Mobile Legends",
        placeholder="Contoh: game nya seru banget dan skin nya keren",
        lines=4,
    ),
    outputs=gr.Label(
        label="Hasil Sentimen",
        num_top_classes=3,
    ),
    title="🎮 Sentiment Analysis Review Mobile Legends",
    description=(
        "Model machine learning untuk mengklasifikasikan ulasan Mobile Legends "
        "menjadi sentimen Positif, Negatif, atau Netral. "
        "Dataset berasal dari Kaggle: Mobile Legends App Reviews."
    ),
    examples=examples,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()