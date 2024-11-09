from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "pytorch_lstm_results.log"

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
BATCH_SIZE = 128
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.FloatTensor(texts)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, max_words, embedding_dim, hidden_dim=100, dropout=0.25):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(max_words, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if self.training else 0
        )
        
        # Dense layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dense1 = nn.Linear(hidden_dim, 100)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(100, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding
        x = self.embedding(x.long())
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Global Max Pooling
        # Transpose to get the correct dimension for max pooling
        lstm_out = lstm_out.transpose(1, 2)
        pooled = self.global_max_pool(lstm_out).squeeze(-1)
        
        # Dense layers
        x = self.dense1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        
        return x

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    logger.info(f"Training on {DEVICE}")
    model = model.to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (texts, labels) in enumerate(tqdm(train_loader)):
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in valid_loader:
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Log results
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Training Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')
        logger.info(f'Validation Loss: {val_loss/len(valid_loader):.4f}, Accuracy: {val_acc:.2f}%')

def main():
    # Load data
    try:
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

        # Convert embedding matrix to torch tensor
        embedding_matrix = torch.FloatTensor(embedding_matrix)

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            input_tensor, 
            cleaned_data["behavior_tobacco_binary"].values, 
            test_size=0.3,
            random_state=42
        )

        # Create datasets and dataloaders
        train_dataset = TextDataset(x_train, y_train)
        test_dataset = TextDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Initialize model
        model = LSTMClassifier(MAX_NUM_WORDS, EMBEDDING_DIM)
        
        # Load pre-trained embeddings
        model.embedding.weight.data.copy_(embedding_matrix)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        logger.info("Starting model training")
        train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS)

        # Evaluate on test set
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(DEVICE)
                outputs = model(texts).squeeze()
                predicted = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(predicted)
                all_labels.extend(labels.numpy())

        # Generate and log classification report
        report = classification_report(all_labels, all_preds, digits=3)
        logger.info("Classification Report:\n" + report)

        # Save model
        model_path = root_dir / "models" / "pytorch_lstm.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()