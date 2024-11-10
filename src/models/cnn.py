from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "pytorch_cnn_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)
logger = logging.getLogger(__name__)

# Constants
MAX_SEQ_LENGTH = None  # Will be set after loading data
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.from_numpy(texts).long()
        self.labels = torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class CNNClassifier(nn.Module):
    def __init__(self, max_words, embedding_dim, 
                 num_filters=256, 
                 filter_sizes=[2, 3, 4, 5], 
                 dropout=0.3):
        super(CNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(max_words, embedding_dim)
        
        # Convolutional layers with batch normalization
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                         out_channels=num_filters,
                         kernel_size=fs),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for fs in filter_sizes
        ])
        
        # Dense layers with batch normalization
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conved = [conv(embedded) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        
        x = self.fc1(cat)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model = None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

def calculate_pos_weight(labels):
    """Calculate positive class weight for imbalanced dataset"""
    neg_count = np.sum(labels == 0)
    pos_count = np.sum(labels == 1)
    return torch.tensor([neg_count / pos_count])

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    labels_list = []
    
    for texts, labels in tqdm(dataloader, desc='Training'):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), predictions, labels_list

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    labels_list = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc='Evaluation'):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), predictions, labels_list

def main():
    try:
        # Load data
        embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
        input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
        cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

        with open(input_tensor_path, "rb") as f:
            input_tensor = pickle.load(f)
        with open(embedding_matrix_path, "rb") as f:
            embedding_matrix = pickle.load(f)
        cleaned_data = pd.read_csv(cleaned_data_path)

        global MAX_SEQ_LENGTH
        MAX_SEQ_LENGTH = len(max(cleaned_data["TEXT"]))
        
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor)

        # Use subset of data
        subset_size = len(input_tensor) // 5
        indices = np.random.choice(len(input_tensor), subset_size, replace=False)
        input_tensor_subset = input_tensor[indices]
        labels_subset = cleaned_data["behavior_tobacco_binary"].values[indices]

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            input_tensor_subset, 
            labels_subset,
            test_size=0.3,
            random_state=42,
            stratify=labels_subset
        )

        # Calculate positive class weight
        pos_weight = calculate_pos_weight(y_train)
        logger.info(f"Positive class weight: {pos_weight.item()}")

        # Create datasets and dataloaders
        train_dataset = TextDataset(x_train, y_train)
        test_dataset = TextDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Initialize model
        model = CNNClassifier(
            max_words=MAX_NUM_WORDS,
            embedding_dim=EMBEDDING_DIM
        ).to(DEVICE)
        
        # Load pre-trained embeddings
        model.embedding.weight.data.copy_(embedding_matrix)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        early_stopping = EarlyStopping(patience=3)

        # Training loop
        logger.info("Starting training")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Training samples: {len(x_train)}")
        logger.info(f"Testing samples: {len(x_test)}")

        history = []
        
        for epoch in range(EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            # Training
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            
            # Validation
            val_loss, val_preds, val_labels = evaluate(
                model, test_loader, criterion, DEVICE
            )
            
            # Calculate metrics
            train_report = classification_report(train_labels, train_preds, digits=3)
            val_report = classification_report(val_labels, val_preds, digits=3)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info("Train Report:\n" + train_report)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info("Validation Report:\n" + val_report)
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping
            early_stopping(model, val_loss)
            if early_stopping.should_stop:
                logger.info("Early stopping triggered")
                break
            
            # Save history
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_report': train_report,
                'val_report': val_report
            })

        # Load best model
        if early_stopping.best_model is not None:
            model.load_state_dict(early_stopping.best_model)

        # Final evaluation
        final_loss, final_preds, final_labels = evaluate(
            model, test_loader, criterion, DEVICE
        )
        final_report = classification_report(final_labels, final_preds, digits=3)
        logger.info("\nFinal Test Report:\n" + final_report)

        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = root_dir / "models" / f"cnn_model_{timestamp}.pt"
        results_path = root_dir / "models" / f"cnn_results_{timestamp}.json"
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'max_words': MAX_NUM_WORDS,
                'embedding_dim': EMBEDDING_DIM,
                'num_filters': 256,
                'filter_sizes': [2, 3, 4, 5],
                'dropout': 0.3
            }
        }, model_path)

        with open(results_path, 'w') as f:
            json.dump({
                'history': history,
                'final_report': final_report,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'early_stopping_patience': 3,
                    'pos_weight': pos_weight.item()
                }
            }, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()