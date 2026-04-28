import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader


# ==============================
# CONFIG
# ==============================
DATA_PATH = '../data/ml_reviews_ready.csv'
OUTPUT_DIR = '../app'

MAX_WORDS = 5000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
SEED = 42


# ==============================
# SET SEED (biar reproducible)
# ==============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = ['clean_text', 'sentiment']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan")

    df = df[['clean_text', 'sentiment']].dropna()
    df.rename(columns={'clean_text': 'content'}, inplace=True)

    print("Data loaded:", df.shape)
    return df


# ==============================
# PREPROCESSING
# ==============================
def preprocess(df):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['content'])

    X = tokenizer.texts_to_sequences(df['content'])
    X = pad_sequences(X, maxlen=MAX_LEN)

    y = df['label'].values

    return X, y, tokenizer, le


# ==============================
# SPLIT DATA
# ==============================
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


# ==============================
# TO TENSOR
# ==============================
def to_tensor(X_train, X_test, y_train, y_test):
    return (
        torch.tensor(X_train).long(),
        torch.tensor(X_test).long(),
        torch.tensor(y_train).long(),
        torch.tensor(y_test).long()
    )


# ==============================
# MODEL LSTM
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
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


# ==============================
# TRAINING
# ==============================
def train_model(model, train_loader, criterion, optimizer):
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    return loss_history


# ==============================
# EVALUASI
# ==============================
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)

    print("\nClassification Report:")
    print(classification_report(y_test, preds))


# ==============================
# PLOT LOSS
# ==============================
def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# ==============================
# SAVE MODEL
# ==============================
def save_artifacts(model, tokenizer, le):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))

    with open(os.path.join(OUTPUT_DIR, "tokenizer.json"), "w") as f:
        f.write(tokenizer.to_json())

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("\nModel & artifacts berhasil disimpan!")


# ==============================
# MAIN
# ==============================
def main():
    set_seed(SEED)

    df = load_data(DATA_PATH)

    X, y, tokenizer, le = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test = to_tensor(
        X_train, X_test, y_train, y_test
    )

    num_classes = len(set(y))
    print("Jumlah kelas:", num_classes)

    model = LSTMModel(
        vocab_size=MAX_WORDS,
        embed_dim=128,
        hidden_dim=64,
        output_dim=num_classes
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = train_model(model, train_loader, criterion, optimizer)

    evaluate(model, X_test, y_test)
    plot_loss(loss_history)

    save_artifacts(model, tokenizer, le)


if __name__ == "__main__":
    main()