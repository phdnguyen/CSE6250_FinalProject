import os
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
output_dir = root_dir / "output" / "lstm_bidirectional"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

log_path = root_dir / "logs" / "lstm_bidirectional_avg_results.log"

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

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device):
    model.to(device)
    epoch_losses = []
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
        epoch_losses.append(avg_loss)
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
            if criterion.__class__.__name__ == "BCEWithLogitsLoss":
                preds.extend(torch.round(torch.sigmoid(outputs)).cpu().numpy().flatten())
            else:
                preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    return preds, actuals, epoch_losses

def save_confusion_matrix(actuals, preds, is_binary, output_dir, run):
    try:
        cm = confusion_matrix(actuals, preds)
        display_labels = ['Class 0', 'Class 1'] if is_binary else [f"Class {i}" for i in range(NUM_CLASSES)]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title(f"Confusion Matrix - Run {run}")
        file_path = output_dir / f"confusion_matrix_run{run}.png"
        plt.savefig(file_path)
        print(f"Confusion matrix saved to {file_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix for run {run}: {e}")

def save_loss_plot(losses, output_dir, run):
    try:
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title(f"Loss Function Over Epochs - Run {run}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        file_path = output_dir / f"loss_plot_run{run}.png"
        plt.savefig(file_path)
        print(f"Loss plot saved to {file_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving loss plot for run {run}: {e}")

# Helper function to calculate average and std for metrics
def calculate_avg_std(metrics_list):
    avg_metrics = {key: np.mean([metrics[key] for metrics in metrics_list]) for key in metrics_list[0]}
    std_metrics = {key: np.std([metrics[key] for metrics in metrics_list]) for key in metrics_list[0]}
    return avg_metrics, std_metrics

# Main loop for binary and multi-class classification
for classification_type, y, description in [('binary', cleaned_data["behavior_tobacco_binary"].values, "Binary"),
                                             ('multiclass', cleaned_data["behavior_tobacco"].values, "Multi-Class")]:
    logger.info(f"Starting {description} classification")
    results = []

    # Create subdirectories for binary and multiclass
    classification_output_dir = output_dir / description.lower()
    classification_output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(3):  # Repeat 3 times
        logger.info(f"Run {run + 1} for {description} classification")

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(input_tensor, y, test_size=0.3, random_state=run)
        
        if classification_type == 'binary':
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            output_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        else:  # Multi-class
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            output_dim = NUM_CLASSES
            criterion = nn.CrossEntropyLoss()

        # Prepare DataLoaders
        train_data = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_train, y_train)]
        test_data = [(x.clone().detach().to(torch.long), y) for x, y in zip(x_test, y_test)]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        # Initialize and train the model
        model = BiLSTMModel(MAX_NUM_WORDS, EMBEDDING_DIM, 100, output_dim, embedding_matrix)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        preds, actuals, losses = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Save confusion matrix and loss plot for this run
        save_confusion_matrix(actuals, preds, is_binary=(classification_type == 'binary'), output_dir=classification_output_dir, run=run + 1)
        save_loss_plot(losses, output_dir=classification_output_dir, run=run + 1)

        # Generate classification report
        report = classification_report(actuals, preds, output_dict=True, digits=3)
        accuracy = accuracy_score(actuals, preds)
        results.append({
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1-score": report["macro avg"]["f1-score"],
            "accuracy": accuracy
        })

    # Calculate averages and standard deviations
    avg_metrics, std_metrics = calculate_avg_std(results)

    # Log the results
    logger.info(f"Averages for {description} classification: {avg_metrics}")
    logger.info(f"Standard deviations for {description} classification: {std_metrics}")

print("Training and evaluation completed. Check the log file for detailed results.")