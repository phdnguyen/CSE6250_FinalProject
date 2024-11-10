from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

# Set the root directory and logging
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "pytorch_cnn_improved_results.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)
logger = logging.getLogger(__name__)

# Constants
MAX_SEQ_LENGTH = None
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300
BATCH_SIZE = 16
EPOCHS = 10
N_FOLDS = 5
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
                 num_filters=128,
                 filter_sizes=[2, 3, 4],
                 dropout=0.5):
        super(CNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(max_words, embedding_dim)
        
        # Convolutional layers with batch normalization
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                         out_channels=num_filters,
                         kernel_size=fs,
                         padding='same'),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for fs in filter_sizes
        ])
        
        # Simplified dense layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conved = [conv(embedded) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        
        return self.fc(cat)

def create_weighted_sampler(labels):
    """Create a weighted sampler to handle class imbalance"""
    class_counts = np.bincount(labels.astype(int))
    total_samples = len(labels)
    
    # Compute weights for each sample
    class_weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = torch.tensor([class_weights[label] for label in labels])
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, fold):
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for texts, labels in tqdm(train_loader, desc=f'Fold {fold+1}, Epoch {epoch+1}'):
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            
            # Add L2 regularization
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_report = classification_report(train_labels, train_preds, digits=3)
        val_report = classification_report(val_labels, val_preds, digits=3)
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        logger.info(f'\nFold {fold+1}, Epoch {epoch+1}:')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info('Train Report:\n' + train_report)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        logger.info('Validation Report:\n' + val_report)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return best_model_state, best_val_loss

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

        # Use full dataset
        X = input_tensor
        y = cleaned_data["behavior_tobacco_binary"].values
        
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Prepare K-fold cross-validation
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"\nTraining Fold {fold+1}/{N_FOLDS}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create datasets
            train_dataset = TextDataset(X_train, y_train)
            val_dataset = TextDataset(X_val, y_val)
            
            # Create weighted sampler for training data
            train_sampler = create_weighted_sampler(y_train)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                sampler=train_sampler
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE
            )

            # Initialize model
            model = CNNClassifier(
                max_words=MAX_NUM_WORDS,
                embedding_dim=EMBEDDING_DIM
            ).to(DEVICE)
            
            model.embedding.weight.data.copy_(embedding_matrix)
            
            # Initialize loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=0.001, 
                weight_decay=0.01
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

            # Train fold
            best_model_state, best_val_loss = train_fold(
                model, train_loader, val_loader, criterion, optimizer, scheduler, fold
            )
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': best_val_loss,
                'model_state': best_model_state
            })

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find best fold
        best_fold = min(fold_results, key=lambda x: x['best_val_loss'])
        logger.info(f"\nBest fold: {best_fold['fold']} with validation loss: {best_fold['best_val_loss']:.4f}")
        
        # Save best model
        model_path = root_dir / "models" / f"cnn_model_improved_{timestamp}.pt"
        results_path = root_dir / "models" / f"cnn_results_improved_{timestamp}.json"
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': best_fold['model_state'],
            'config': {
                'max_words': MAX_NUM_WORDS,
                'embedding_dim': EMBEDDING_DIM,
                'num_filters': 128,
                'filter_sizes': [2, 3, 4],
                'dropout': 0.5
            }
        }, model_path)
        
        # Save cross-validation results
        with open(results_path, 'w') as f:
            json.dump({
                'fold_results': [
                    {
                        'fold': res['fold'],
                        'best_val_loss': res['best_val_loss']
                    } for res in fold_results
                ],
                'best_fold': best_fold['fold'],
                'best_val_loss': best_fold['best_val_loss'],
                'config': {
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'n_folds': N_FOLDS
                }
            }, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()