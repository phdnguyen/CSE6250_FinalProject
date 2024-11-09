import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
import logging
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

log_path = root_dir / "logs" / "lstm_bi_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,  # Use log_path as the log file
    level=logging.INFO,  # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s", 
    filemode='a'  
)
logger = logging.getLogger(__name__)

embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

with open(input_tensor_path, "rb") as f:
    input_tensor = pickle.load(f)

with open(embedding_matrix_path, "rb") as f:
    embedding_matrix = pickle.load(f)

cleaned_data = pd.read_csv(cleaned_data_path)

MAX_SEQ_LENGTH = input_tensor.shape[1]  # max length of a note
MAX_NUM_WORDS = embedding_matrix.shape[0]  # number of words in the embedding matrix
EMBEDDING_DIM = embedding_matrix.shape[1]  # dim of GoogleNews
BATCH_SIZE = 128
EPOCHS = 2

# Log basic information
logger.info("Starting LSTM model training")
logger.info(f"Maximum sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"Embedding dimensions: {EMBEDDING_DIM}")
logger.info(f"Max number of words: {MAX_NUM_WORDS}")

# Define PyTorch bidirectional LSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.global_max_pool(x.permute(0, 2, 1)).squeeze(2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
hidden_dim = 100
model = BiLSTMModel(MAX_NUM_WORDS, EMBEDDING_DIM, hidden_dim, embedding_matrix)

# Log the model summary
logger.info("Model Summary:")
logger.info(model)

# Prepare data for training
x_train, x_test, y_train, y_test = train_test_split(
    input_tensor, cleaned_data["behavior_tobacco_binary"].values, test_size=0.3
)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_data = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_train, y_train)]
test_data = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_test, y_test)]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
logger.info("Training model with the following parameters:")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
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
        preds.extend(outputs.cpu().numpy().flatten())
        actuals.extend(labels.cpu().numpy().flatten())

preds = np.round(preds)

# Log classification report
logger.info("Generating classification report")
report = classification_report(actuals, preds, digits=3)
logger.info("Classification Report:\n" + report)
logger.info("Evaluation completed.")
logger.info("------------------------------------------")
