import torch
import torch.nn as nn
import torch.optim as optim
import re
import logging
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

##### THIS IS THE CODE TO RUN ON KAGGLE############################################
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load data
cleaned_data = pd.read_csv('/kaggle/input/processed2/1_cleaned_data.csv')

with open('/kaggle/input/processed2/embedding_matrix.pkl', "rb") as f:
    embedding_matrix = pickle.load(f)
with open('/kaggle/input/processed2/input_tensor.pkl', "rb") as f:
    input_tensor = pickle.load(f)
with open('/kaggle/input/processed2/word_index.pkl', "rb") as f:
    word_index = pickle.load(f)


# Constants
MAX_SEQ_LENGTH = input_tensor.shape[1]
MAX_NUM_WORDS = embedding_matrix.shape[0]
EMBEDDING_DIM = embedding_matrix.shape[1]
BATCH_SIZE = 64
EPOCHS = 10  # Changed to 10 epochs
NUM_RUNS = 10

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=False
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, 100)
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

def train_and_evaluate(run_number):
    """Train and evaluate the model for one run"""
    # Set different random seed for each run
    torch.manual_seed(42 + run_number)
    
    # Initialize model and move to device
    hidden_dim = 100
    model = LSTMModel(MAX_NUM_WORDS, EMBEDDING_DIM, hidden_dim, embedding_matrix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert input_tensor to torch tensor if it isn't already
    if not isinstance(input_tensor, torch.Tensor):
        X = torch.tensor(input_tensor, dtype=torch.long)
    else:
        X = input_tensor.clone()
    
    # Convert labels to torch tensor
    y = torch.tensor(cleaned_data["behavior_tobacco_binary"].values, dtype=torch.float32).unsqueeze(1)
    
    # Create train/test split indices
    dataset_size = len(X)
    indices = torch.randperm(dataset_size)
    train_size = int(0.7 * dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Split the data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    # Training loop
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
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Run {run_number + 1}, Epoch [{epoch + 1}/{EPOCHS}], '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            all_preds.append(predictions.cpu())
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Get classification report
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    
    return report_dict, train_losses, val_losses, all_labels, all_preds

def run_multiple_experiments():
    """Run multiple experiments and calculate average metrics"""
    all_metrics = []
    all_runs = []
    
    print(f"Starting {NUM_RUNS} experiments...")
    
    for run in range(NUM_RUNS):
        report_dict, train_losses, val_losses, true_labels, predictions = train_and_evaluate(run)
        all_metrics.append(report_dict)
        
        # Store run information
        all_runs.append({
            'report': report_dict,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'labels': true_labels,
            'predictions': predictions,
            'run_number': run
        })
        
        # Print current run results
        print(f"\nRun {run + 1} Results:")
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                print(f"\n{class_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name}: {value:.3f}")
        print("-" * 50)
    
    # Find best run based on F1 score
    best_run = max(all_runs, 
                   key=lambda x: x['report']['weighted avg']['f1-score'])
    
    # Print Best Run Results
    print("\nBest Run Results:")
    print("=" * 50)
    for class_name, metrics in best_run['report'].items():
        if isinstance(metrics, dict):
            print(f"\n{class_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.3f}")
        elif class_name == 'accuracy':
            print(f"\nAccuracy: {metrics:.3f}")
    
    # Plot results for best run
    print("\nPlotting results for best run...")
    
    # Plot training curves
    plt.figure(figsize=(12, 6))  # Made figure wider for better visibility
    epochs = range(1, len(best_run['train_losses']) + 1)
    plt.plot(epochs, best_run['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, best_run['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss - Best Run')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)  # This will show all epoch numbers
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(best_run['labels'], best_run['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Best Run')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate and print average metrics
    avg_metrics = {}
    std_metrics = {}
    
    # Calculate accuracies
    accuracies = [run['report']['accuracy'] for run in all_runs]
    avg_metrics['accuracy'] = np.mean(accuracies)
    std_metrics['accuracy'] = np.std(accuracies)
    
    # Calculate other metrics
    for class_name in all_metrics[0].keys():
        if isinstance(all_metrics[0][class_name], dict):
            avg_metrics[class_name] = {}
            std_metrics[class_name] = {}
            
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in all_metrics[0][class_name]:
                    values = [run[class_name][metric] for run in all_metrics]
                    avg_metrics[class_name][metric] = np.mean(values)
                    std_metrics[class_name][metric] = np.std(values)
    
    # Print average metrics
    print("\nAverage Metrics Over All Runs:")
    print("=" * 50)
    print(f"\nAccuracy: {avg_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}")
    
    for class_name in avg_metrics.keys():
        if class_name != 'accuracy':
            print(f"\n{class_name}:")
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in avg_metrics[class_name]:
                    mean_value = avg_metrics[class_name][metric]
                    std_value = std_metrics[class_name][metric]
                    print(f"{metric:10s}: {mean_value:.3f} ± {std_value:.3f}")
    
    return avg_metrics, std_metrics #, best_run['report']

if __name__ == "__main__":
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    
    # Run multiple experiments
    avg_metrics, std_metrics = run_multiple_experiments()