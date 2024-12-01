import torch
import torch.nn as nn
import torch.optim as optim
import re
import logging
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
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
NUM_RUNS = 10
MAX_SEQ_LENGTH = input_tensor.shape[1]
MAX_NUM_WORDS = embedding_matrix.shape[0]
EMBEDDING_DIM = embedding_matrix.shape[1]
NUM_CLASSES = len(cleaned_data["behavior_tobacco"].unique())
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.long() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, max_len):
        super(LSTMModel, self).__init__()
        # Embedding layer with dropout
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=False
        )
        self.dropout1 = nn.Dropout(0.5)
        
        # Unidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            128, 
            batch_first=True, 
            bidirectional=False, 
            num_layers=2, 
            dropout=0.3
        )
        
        # Output layers
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        # Embedding Layer
        x = self.embedding(x)
        x = self.dropout1(x)
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Global max pooling
        x = torch.max(x, dim=1)[0]
        
        # Dense layers
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def calculate_class_weights(labels):
    """Calculate balanced class weights"""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    smoothing = 0.1
    smoothed_counts = class_counts + smoothing * total_samples
    weights = total_samples / (len(class_counts) * smoothed_counts)
    weights = weights / weights.sum() * len(class_counts)
    min_weight = weights.min()
    weights = weights / min_weight
    return weights

def train_and_evaluate_single_run(run_number):
    """Train and evaluate the model for a single run"""
    # Set random seeds for reproducibility
    torch.manual_seed(42 + run_number)
    np.random.seed(42 + run_number)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRun {run_number + 1}/{NUM_RUNS}")
    print(f"Using device: {device}")
    
    # Prepare data split with stratification
    y = torch.tensor(cleaned_data["behavior_tobacco"].values)
    all_indices = torch.arange(len(input_tensor))
    train_indices = []
    test_indices = []
    
    # Stratified split
    for cls in range(NUM_CLASSES):
        cls_indices = all_indices[y == cls]
        n_train = int(0.7 * len(cls_indices))
        perm = torch.randperm(len(cls_indices))
        train_indices.extend(cls_indices[perm[:n_train]].tolist())
        test_indices.extend(cls_indices[perm[n_train:]].tolist())
    
    # Shuffle indices
    train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices))]
    test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))]
    
    # Split data
    x_train = input_tensor[train_indices]
    x_test = input_tensor[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train).to(device)
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    
    # Calculate sample weights for balanced sampling
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LSTMModel(
        vocab_size=len(word_index)+1,
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        max_len=MAX_SEQ_LENGTH
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_preds.append(predictions)
                val_labels.append(labels)
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"New best validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Final evaluation
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            test_preds.append(predictions.cpu())
            test_labels.append(labels)
    
    final_preds = torch.cat(test_preds)
    final_labels = torch.cat(test_labels)
    
    # Calculate metrics
    report_dict = classification_report(final_labels, final_preds, zero_division=0, output_dict=True)
    
    return report_dict, train_losses, val_losses, final_labels, final_preds

def run_multiple_experiments():
    """Run multiple experiments and calculate average metrics"""
    all_metrics = []
    all_runs = []
    
    print(f"Starting {NUM_RUNS} experiments...")
    
    for run in range(NUM_RUNS):
        report_dict, train_losses, val_losses, true_labels, predictions = train_and_evaluate_single_run(run)
        
        # Store run information
        all_runs.append({
            'report': report_dict,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'labels': true_labels,
            'predictions': predictions,
            'run_number': run
        })
        all_metrics.append(report_dict)
        
        # Print current run results
        print(f"\nRun {run + 1} Results:")
        print(f"Accuracy: {report_dict['accuracy']:.3f}")
        
    # Calculate average metrics and standard deviations
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
    
    # Find best run based on F1 score
    best_run = max(all_runs, 
                   key=lambda x: x['report']['weighted avg']['f1-score'])
    
    # Plot results for best run
    print("\nPlotting results for best run...")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(best_run['train_losses'], label='Training Loss')
    plt.plot(best_run['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss - Best Run')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(best_run['labels'], best_run['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Best Run')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print final results
    print("\nFinal Results Over", NUM_RUNS, "Runs:")
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
    
    return avg_metrics, std_metrics

if __name__ == "__main__":
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    
    avg_metrics, std_metrics = run_multiple_experiments()