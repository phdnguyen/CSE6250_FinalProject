import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import logging
import pickle
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
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
NUM_RUNS = 10
MAX_SEQ_LENGTH = input_tensor.shape[1]
MAX_NUM_WORDS = embedding_matrix.shape[0]
EMBEDDING_DIM = embedding_matrix.shape[1]
NUM_CLASSES = len(cleaned_data["behavior_tobacco"].unique())
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.long() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleCNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, num_filters=128):
        super(SimpleCNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False
        )
        
        # Single CNN layer for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding='same')
            for k in [3, 4, 5]  # Multiple kernel sizes
        ])
        
        # Output layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 3, NUM_CLASSES)  # *3 because we have 3 filter sizes
    
    def forward(self, x):
        # Embedding Layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # Apply CNN layers with different kernel sizes and max pooling
        x = [torch.relu(conv(x)) for conv in self.convs]  # List of [batch_size, num_filters, seq_len]
        x = [torch.max(i, dim=2)[0] for i in x]  # List of [batch_size, num_filters]
        
        # Concatenate all features
        x = torch.cat(x, dim=1)  # [batch_size, num_filters * 3]
        
        # Dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def calculate_class_weights(labels):
    """Calculate balanced class weights"""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights

def set_seed(seed):
    """Set random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def train_and_evaluate_single_run(run_number):
    """Train and evaluate the model for a single run"""
    set_seed(42 + run_number)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRun {run_number + 1}/{NUM_RUNS}")
    
    # Prepare data
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
    
    train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices))]
    test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))]
    
    x_train = input_tensor[train_indices]
    x_test = input_tensor[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    
    class_weights = calculate_class_weights(y_train).to(device)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = SimpleCNNModel(
        vocab_size=MAX_NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
    
    # Calculate final statistics
    avg_metrics = {}
    std_metrics = {}
    
    # Overall accuracy
    accuracies = [m['accuracy'] for m in all_metrics]
    avg_metrics['accuracy'] = np.mean(accuracies)
    std_metrics['accuracy'] = np.std(accuracies)
    
    # Other metrics
    for class_name in all_metrics[0].keys():
        if isinstance(all_metrics[0][class_name], dict):
            avg_metrics[class_name] = {}
            std_metrics[class_name] = {}
            
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in all_metrics[0][class_name]:
                    values = [run[class_name][metric] for run in all_metrics]
                    avg_metrics[class_name][metric] = np.mean(values)
                    std_metrics[class_name][metric] = np.std(values)
    
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
    
    return avg_metrics, std_metrics, best_run['report']

if __name__ == "__main__":
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    
    avg_metrics, std_metrics, best_run['report'] = run_multiple_experiments()