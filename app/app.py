"""
app.py — Gradio App untuk Sentiment Analysis Review Mobile Legends
Cocok untuk Hugging Face Spaces
"""

import re
import pandas as pd
import gradio as gr
from pycaret.classification import load_model, predict_model

# =========================================================
# LOAD MODEL
# Jika file di folder bernama best_model.pkl,
# maka cukup tulis load_model("best_model")
# =========================================================

model = load_model("best_model")

# =========================================================
# PREPROCESSING
# Disamakan dengan preprocessing training sederhana
# =========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # hapus mention, hashtag
    text = re.sub(r"@\w+|#\w+", "", text)

    # hapus angka
    text = re.sub(r"\d+", "", text)

    # sisakan huruf dan spasi
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# PREDIKSI
# =========================================================

def predict_sentiment(text):
    if not text or not text.strip():
        return "Silakan masukkan review terlebih dahulu."

    cleaned_text = clean_text(text)

    input_df = pd.DataFrame({
        "clean_text": [cleaned_text]
    })

    result = predict_model(model, data=input_df)

    # Ambil label prediksi
    label = str(result.loc[0, "prediction_label"])

    # Ambil confidence score jika tersedia
    if "prediction_score" in result.columns:
        confidence = float(result.loc[0, "prediction_score"]) * 100
        return f"Sentimen: {label}\nConfidence: {confidence:.2f}%"

    return f"Sentimen: {label}"

# =========================================================
# CONTOH INPUT
# =========================================================

examples = [
    ["produk ini sangat bagus dan pengirimannya cepat"],
    ["saya kecewa karena barang rusak"],
    ["game nya seru banget dan skin nya keren"],
    ["server sering lag dan banyak bug setelah update"],
    ["bagus sih tapi kadang matchmaking nya aneh"],
]

# =========================================================
# TAMPILAN GRADIO
# =========================================================

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        label="Masukkan Review",
        placeholder="Contoh: game ini seru banget dan tidak lag"
    ),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    examples=examples,
    title="🎮 Analisis Sentimen Review Mobile Legends",
    description="Masukkan review Mobile Legends, lalu model akan memprediksi apakah review tersebut positif, netral, atau negatif.",
    theme=gr.themes.Soft()
)

# =========================================================
# JALANKAN APP
# =========================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)