"""
🔥 Mobile Legends Sentiment Analyzer (ML - PyCaret)
"""

import re
import pandas as pd
import gradio as gr
from pycaret.classification import load_model, predict_model

# =========================================================
# LOAD MODEL
# =========================================================
model = load_model("best_model")

# =========================================================
# PREPROCESSING
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
# PREDICTION
# =========================================================
def predict_sentiment(text):
    if not text or not text.strip():
        return {"Input kosong": 1.0}, "⚠️ Silakan masukkan teks"

    cleaned = clean_text(text)

    input_df = pd.DataFrame({
        "clean_text": [cleaned]
    })

    result = predict_model(model, data=input_df)

    label = str(result.loc[0, "prediction_label"])

    if "prediction_score" in result.columns:
        confidence = float(result.loc[0, "prediction_score"])
    else:
        confidence = 1.0

    # Insight tambahan
    if label.lower() == "positif":
        insight = "😊 Review menunjukkan pengalaman yang baik."
    elif label.lower() == "negatif":
        insight = "😡 Review menunjukkan ketidakpuasan pengguna."
    else:
        insight = "😐 Review bersifat netral atau campuran."

    return {label: confidence}, insight

# =========================================================
# EXAMPLES
# =========================================================
examples = [
    ["game ini seru banget dan tidak lag"],
    ["server sering error dan bikin kesal"],
    ["lumayan bagus tapi matchmaking aneh"],
    ["skin nya keren dan gameplay enak"],
]

# =========================================================
# UI PREMIUM
# =========================================================

with gr.Blocks(theme=gr.themes.Glass()) as demo:

    # HEADER
    gr.Markdown("""
    # 🎮 Mobile Legends Sentiment Analyzer  
    ### 🤖 Machine Learning (PyCaret)

    ✨ Analisis sentimen review game secara otomatis  
    🚀 Cepat dan interaktif
    """)

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=2):

            input_text = gr.Textbox(
                lines=6,
                placeholder="Ketik review kamu di sini...",
                label="📝 Masukkan Review"
            )

            with gr.Row():
                btn = gr.Button("🔍 Analisis", variant="primary")
                clear = gr.Button("🧹 Clear")

        # RIGHT PANEL
        with gr.Column(scale=1):

            output_label = gr.Label(label="📊 Hasil Sentimen")
            insight_box = gr.Markdown("💡 Insight akan muncul di sini")

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
    ⚡ Machine Learning dengan PyCaret  
    🎯 Siap untuk demo & presentasi
    """)

    # ACTION
    btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[output_label, insight_box],
        show_progress=True
    )

    clear.click(
        fn=lambda: ("", {}, "💡 Insight akan muncul di sini"),
        inputs=[],
        outputs=[input_text, output_label, insight_box]
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)