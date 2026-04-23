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
# PREDIKSI (UBAH RETURN JADI DICT BIAR BISA LABEL VISUAL)
# =========================================================

def predict_sentiment(text):
    if not text or not text.strip():
        return {"Silakan masukkan teks": 1.0}

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)

    input_tensor = torch.tensor(padded).long()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    labels = le.classes_

    # return dictionary untuk gr.Label
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# =========================================================
# CONTOH INPUT
# =========================================================

examples = [
    ["game ini seru banget dan tidak lag"],
    ["saya kecewa karena banyak bug"],
    ["skin nya keren dan gameplay nya enak"],
    ["server sering lag dan tidak stabil"],
    ["lumayan bagus tapi matchmaking aneh"],
]


# =========================================================
# UI MODERN (BLOCKS)
# =========================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # HEADER
    gr.Markdown("""
    # 🎮 Mobile Legends Sentiment Analyzer  
    ### 🤖 Deep Learning (LSTM)

    💡 Masukkan review game, dan model akan menganalisis sentimen secara otomatis.
    """)

    with gr.Row():

        # INPUT
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=5,
                placeholder="Contoh: game ini seru banget dan tidak lag...",
                label="📝 Masukkan Review"
            )

            btn = gr.Button("🔍 Analisis Sekarang", variant="primary")

        # OUTPUT (LEBIH KEREN)
        with gr.Column(scale=1):
            output = gr.Label(label="📊 Hasil Sentimen")

    # EXAMPLES
    gr.Markdown("### ✨ Contoh Review")
    gr.Examples(
        examples=examples,
        inputs=input_text
    )

    # FOOTER
    gr.Markdown("""
    ---
    👩‍💻 Dibuat untuk Analisis Sentimen Mobile Legends  
    🚀 Powered by Deep Learning
    """)

    # ACTION
    btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=output
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)