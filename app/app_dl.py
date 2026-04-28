"""
app.py — Gradio App untuk Sentiment Analysis (Deep Learning - LSTM)
Cocok untuk Hugging Face Spaces
"""

import re
import torch
import pickle
import gradio as gr
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =========================================================
# LOAD TOKENIZER & LABEL ENCODER
# =========================================================

with open("tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# =========================================================
# PARAMETER (HARUS SAMA DENGAN TRAINING)
# =========================================================

max_len = 100
max_words = 5000
num_classes = len(le.classes_)

# =========================================================
# MODEL LSTM
# =========================================================

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x)

# =========================================================
# LOAD MODEL
# =========================================================

model = LSTMModel(max_words, 128, 64, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# =========================================================
# PREPROCESSING (SAMAKAN DENGAN TRAINING)
# =========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# PREDIKSI
# =========================================================

def predict_sentiment(text):
    if not text or not text.strip():
        return {}, "⚠️ Masukkan teks terlebih dahulu", ""

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)

    input_tensor = torch.tensor(padded).long()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    labels = le.classes_

    result_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    pred_index = int(np.argmax(probs))
    pred_label = labels[pred_index]
    confidence = probs[pred_index] * 100

    # Insight
    insight = f"""
    ### 🤖 Analisis Model
    - Prediksi: **{pred_label}**
    - Confidence: **{confidence:.2f}%**
    """

    # Tampilkan teks setelah preprocessing
    cleaned_view = f"🔎 Cleaned Text:\n{cleaned}"

    return result_dict, insight, cleaned_view


# =========================================================
# UI MODERN
# =========================================================

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:

    # HEADER (ringkas biar tidak makan tempat)
    gr.Markdown("""
      # 🎮 Mobile Legends Sentiment Analyzer  
    ### 🤖 Deep Learning (LSTM)
    """)

    with gr.Row():

        # KIRI: INPUT
        with gr.Column(scale=2):

            input_text = gr.Textbox(
                lines=4,
                placeholder="Tulis review...",
                label="📝 Input"
            )

            with gr.Row():
                btn = gr.Button("⚡ Analyze", variant="primary")
                clear = gr.Button("Reset")

            # contoh kecil (tidak panjang)
            gr.Examples(
                examples=[
                        ["game ini seru banget dan tidak lag"],
                        ["saya kecewa karena banyak bug"],
                        ["skin nya keren dan gameplay nya enak"],
                        ["server sering lag dan tidak stabil"],
                        ["lumayan bagus tapi matchmaking aneh"]
                ],
                inputs=input_text
            )

        # KANAN: OUTPUT
        with gr.Column(scale=3):

            output_label = gr.Label(label="📊 Sentiment")

            insight_box = gr.Markdown("💡 Insight")

            cleaned_text_box = gr.Textbox(
                label="🧹 Clean Text",
                interactive=False,
                lines=2
            )

    # ACTION
    btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[output_label, insight_box, cleaned_text_box],
        show_progress=True
    )

    clear.click(
        fn=lambda: ("", {}, "💡 Insight", ""),
        inputs=[],
        outputs=[input_text, output_label, insight_box, cleaned_text_box]
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)