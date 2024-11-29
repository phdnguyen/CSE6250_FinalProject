import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import logging
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

log_path = root_dir / "logs" / "lstm_bidirectional_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

# Load data
with open(input_tensor_path, "rb") as f:
    input_tensor = pickle.load(f)

with open(embedding_matrix_path, "rb") as f:
    embedding_matrix = pickle.load(f)

cleaned_data = pd.read_csv(cleaned_data_path)

# Parameters
MAX_SEQ_LENGTH = input_tensor.shape[1]
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = embedding_matrix.shape[1]
BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES = 5  # Number of classes for multi-class classification

# Log basic information
logger.info("Starting LSTM training for binary and multi-class classification")
logger.info(f"Maximum sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"Embedding dimensions: {EMBEDDING_DIM}")
logger.info(f"Max number of words: {MAX_NUM_WORDS}")

# Define PyTorch BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.global_max_pool(x.permute(0, 2, 1)).squeeze(2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, is_binary):
    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    logger.info("Training completed.")

    # Evaluate the model
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if is_binary:
                preds.extend(torch.round(torch.sigmoid(outputs)).cpu().numpy().flatten())
            else:
                preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # Log classification report
    logger.info("Generating classification report")
    report = classification_report(actuals, preds, digits=3)
    logger.info("Classification Report:\n" + report)
    logger.info("------------------------------------------")


# Prepare data for binary classification
x_train_bin, x_test_bin, y_train_bin, y_test_bin = train_test_split(
    input_tensor,
    cleaned_data["behavior_tobacco_binary"].values,
    test_size=0.3,
    random_state=42,
)
y_train_bin = torch.tensor(y_train_bin, dtype=torch.float32).unsqueeze(1)
y_test_bin = torch.tensor(y_test_bin, dtype=torch.float32).unsqueeze(1)

train_data_bin = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_train_bin, y_train_bin)]
test_data_bin = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_test_bin, y_test_bin)]

train_loader_bin = DataLoader(train_data_bin, batch_size=BATCH_SIZE, shuffle=True)
test_loader_bin = DataLoader(test_data_bin, batch_size=BATCH_SIZE)

# Binary classification model
model_bin = BiLSTMModel(MAX_NUM_WORDS, EMBEDDING_DIM, 100, 1, embedding_matrix)
criterion_bin = nn.BCEWithLogitsLoss()
optimizer_bin = optim.Adam(model_bin.parameters(), lr=0.001)

# Train and evaluate for binary classification
logger.info("Starting binary classification")
train_and_evaluate(model_bin, train_loader_bin, test_loader_bin, criterion_bin, optimizer_bin, torch.device("cuda" if torch.cuda.is_available() else "cpu"), is_binary=True)

# Prepare data for multi-class classification
x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
    input_tensor,
    cleaned_data["behavior_tobacco"].values,
    test_size=0.3,
    random_state=42,
)
y_train_multi = torch.tensor(y_train_multi, dtype=torch.long)
y_test_multi = torch.tensor(y_test_multi, dtype=torch.long)

train_data_multi = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_train_multi, y_train_multi)]
test_data_multi = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_test_multi, y_test_multi)]

train_loader_multi = DataLoader(train_data_multi, batch_size=BATCH_SIZE, shuffle=True)
test_loader_multi = DataLoader(test_data_multi, batch_size=BATCH_SIZE)

# Multi-class classification model
model_multi = BiLSTMModel(MAX_NUM_WORDS, EMBEDDING_DIM, 100, NUM_CLASSES, embedding_matrix)
criterion_multi = nn.CrossEntropyLoss()
optimizer_multi = optim.Adam(model_multi.parameters(), lr=0.001)

# Train and evaluate for multi-class classification
logger.info("Starting multi-class classification")
train_and_evaluate(model_multi, train_loader_multi, test_loader_multi, criterion_multi, optimizer_multi, torch.device("cuda" if torch.cuda.is_available() else "cpu"), is_binary=False)

print("Training and evaluation completed. Check the log file for detailed results.")
