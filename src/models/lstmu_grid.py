# from pathlib import Path
# import pandas as pd
# import numpy as np
# import pickle
# import logging
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from tqdm import tqdm
# import itertools
# import json
# from datetime import datetime

# # Set the root directory to the project root
# root_dir = Path(__file__).resolve().parents[2]
# log_path = root_dir / "logs" / "pytorch_lstm_gridsearch_results.log"

# # Ensure the log directory exists
# log_path.parent.mkdir(parents=True, exist_ok=True)
# logging.basicConfig(
#     filename=log_path,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     filemode='a'
# )
# logger = logging.getLogger(__name__)

# # Constants
# MAX_SEQ_LENGTH = None  # Will be set after loading data
# MAX_NUM_WORDS = 10000
# EMBEDDING_DIM = 300
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Grid Search Parameters
# param_grid = {
#     'hidden_dim': [50, 100, 150],
#     'dropout': [0.2, 0.25, 0.3],
#     'learning_rate': [0.0001, 0.001, 0.01],
#     'batch_size': [64, 128],
#     'epochs': [2, 3],
#     'num_layers': [1, 2]
# }

# class TextDataset(Dataset):
#     def __init__(self, texts, labels):
#         self.texts = torch.from_numpy(texts).long()
#         self.labels = torch.from_numpy(labels).float()
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         return self.texts[idx], self.labels[idx]

# class LSTMClassifier(nn.Module):
#     def __init__(self, max_words, embedding_dim, hidden_dim=100, dropout=0.25, num_layers=1):
#         super(LSTMClassifier, self).__init__()
        
#         self.embedding = nn.Embedding(max_words, embedding_dim)
#         self.lstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         self.global_max_pool = nn.AdaptiveMaxPool1d(1)
#         self.dense1 = nn.Linear(hidden_dim, 100)
#         self.dropout = nn.Dropout(dropout)
#         self.dense2 = nn.Linear(100, 1)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out.transpose(1, 2)
#         pooled = self.global_max_pool(lstm_out).squeeze(-1)
        
#         x = self.dense1(pooled)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.dense2(x)
#         x = self.sigmoid(x)
        
#         return x

# def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
#     model = model.to(DEVICE)
#     best_val_acc = 0
#     best_epoch = 0
#     training_history = []
    
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         for texts, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
#             texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
#             optimizer.zero_grad()
#             outputs = model(texts).squeeze()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             predicted = (outputs > 0.5).float()
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         train_acc = 100. * correct / total
#         train_loss = total_loss / len(train_loader)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for texts, labels in valid_loader:
#                 texts, labels = texts.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(texts).squeeze()
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 predicted = (outputs > 0.5).float()
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
        
#         val_acc = 100. * val_correct / val_total
#         val_loss = val_loss / len(valid_loader)
        
#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_epoch = epoch
        
#         # Save history
#         epoch_history = {
#             'epoch': epoch + 1,
#             'train_loss': train_loss,
#             'train_acc': train_acc,
#             'val_loss': val_loss,
#             'val_acc': val_acc
#         }
#         training_history.append(epoch_history)
    
#     return best_val_acc, best_epoch, training_history

# def grid_search():
#     try:
#         # Load data
#         embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
#         input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
#         cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

#         with open(input_tensor_path, "rb") as f:
#             input_tensor = pickle.load(f)
#         with open(embedding_matrix_path, "rb") as f:
#             embedding_matrix = pickle.load(f)
#         cleaned_data = pd.read_csv(cleaned_data_path)

#         global MAX_SEQ_LENGTH
#         MAX_SEQ_LENGTH = len(max(cleaned_data["TEXT"]))
        
#         # Convert embedding matrix to torch tensor
#         embedding_matrix = torch.FloatTensor(embedding_matrix)
        
#         # Convert input tensor to numpy if needed
#         if not isinstance(input_tensor, np.ndarray):
#             input_tensor = np.array(input_tensor)

#         # Split data
#         x_train, x_test, y_train, y_test = train_test_split(
#             input_tensor, 
#             cleaned_data["behavior_tobacco_binary"].values, 
#             test_size=0.3,
#             random_state=42
#         )

#         # Generate all parameter combinations
#         param_combinations = [dict(zip(param_grid.keys(), v)) 
#                             for v in itertools.product(*param_grid.values())]
        
#         # Store results
#         results = []
#         best_accuracy = 0
#         best_params = None
#         best_model_state = None
        
#         # Log start of grid search
#         logger.info("Starting grid search with parameters:")
#         logger.info(json.dumps(param_grid, indent=2))
        
#         # Iterate through all parameter combinations
#         for params in tqdm(param_combinations, desc="Grid Search Progress"):
#             logger.info(f"\nTrying parameters: {params}")
            
#             # Create datasets with current batch size
#             train_dataset = TextDataset(x_train, y_train)
#             test_dataset = TextDataset(x_test, y_test)
            
#             train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
#             test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
            
#             # Initialize model with current parameters
#             model = LSTMClassifier(
#                 max_words=MAX_NUM_WORDS,
#                 embedding_dim=EMBEDDING_DIM,
#                 hidden_dim=params['hidden_dim'],
#                 dropout=params['dropout'],
#                 num_layers=params['num_layers']
#             )
            
#             # Load pre-trained embeddings
#             model.embedding.weight.data.copy_(embedding_matrix)
            
#             # Initialize optimizer and criterion
#             criterion = nn.BCELoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
#             # Train model
#             val_acc, best_epoch, history = train_model(
#                 model, train_loader, test_loader, criterion, optimizer, params['epochs']
#             )
            
#             # Save results
#             result = {
#                 'params': params,
#                 'val_accuracy': val_acc,
#                 'best_epoch': best_epoch,
#                 'history': history
#             }
#             results.append(result)
            
#             # Update best model if necessary
#             if val_acc > best_accuracy:
#                 best_accuracy = val_acc
#                 best_params = params
#                 best_model_state = model.state_dict()
            
#             logger.info(f"Validation accuracy: {val_acc:.2f}%")
        
#         # Log final results
#         logger.info("\nGrid Search completed.")
#         logger.info(f"Best parameters: {best_params}")
#         logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")
        
#         # Save best model and results
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         results_path = root_dir / "models" / f"grid_search_results_{timestamp}.json"
#         model_path = root_dir / "models" / f"best_model_{timestamp}.pt"
        
#         with open(results_path, 'w') as f:
#             json.dump({
#                 'best_params': best_params,
#                 'best_accuracy': float(best_accuracy),
#                 'all_results': results
#             }, f, indent=2, default=str)
        
#         torch.save(best_model_state, model_path)
        
#         logger.info(f"Results saved to {results_path}")
#         logger.info(f"Best model saved to {model_path}")
        
#     except Exception as e:
#         logger.error(f"Error during grid search: {str(e)}")
#         raise

# if __name__ == "__main__":
#     grid_search()

#################FASTER VERSION
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
import itertools
import json
from datetime import datetime

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "pytorch_lstm_gridsearch_results.log"

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduced parameter grid for faster search
param_grid = {
    'hidden_dim': [100],
    'dropout': [0.25],
    'learning_rate': [0.001],
    'batch_size': [128],
    'epochs': [2],
    'num_layers': [1]
}

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.from_numpy(texts).long()
        self.labels = torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, max_words, embedding_dim, hidden_dim=100, dropout=0.25, num_layers=1):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dense1 = nn.Linear(hidden_dim, 100)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(100, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(1, 2)
        pooled = self.global_max_pool(lstm_out).squeeze(-1)
        
        x = self.dense1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    model = model.to(DEVICE)
    best_val_acc = 0
    best_epoch = 0
    training_history = []
    early_stopping = EarlyStopping(patience=2)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
        
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
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
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(valid_loader)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        # Save history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        training_history.append(epoch_history)
        
        # Log epoch results
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    return best_val_acc, best_epoch, training_history

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
        
        # Convert embedding matrix to torch tensor
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        
        # Convert input tensor to numpy if needed
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor)

        # Use subset of data for faster training
        subset_size = len(input_tensor) // 5  # Use 20% of data
        indices = np.random.choice(len(input_tensor), subset_size, replace=False)
        input_tensor_subset = input_tensor[indices]
        labels_subset = cleaned_data["behavior_tobacco_binary"].values[indices]

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            input_tensor_subset, 
            labels_subset,
            test_size=0.3,
            random_state=42
        )

        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        # Store results
        results = []
        best_accuracy = 0
        best_params = None
        best_model_state = None
        
        # Log start of grid search
        logger.info("Starting grid search with parameters:")
        logger.info(json.dumps(param_grid, indent=2))
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"Dataset size: {len(x_train)} training samples")
        
        # Iterate through all parameter combinations
        for params in tqdm(param_combinations, desc="Grid Search Progress"):
            logger.info(f"\nTrying parameters: {params}")
            
            # Create datasets with current batch size
            train_dataset = TextDataset(x_train, y_train)
            test_dataset = TextDataset(x_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
            
            # Initialize model with current parameters
            model = LSTMClassifier(
                max_words=MAX_NUM_WORDS,
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=params['hidden_dim'],
                dropout=params['dropout'],
                num_layers=params['num_layers']
            )
            
            # Load pre-trained embeddings
            model.embedding.weight.data.copy_(embedding_matrix)
            
            # Initialize optimizer and criterion
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Train model
            val_acc, best_epoch, history = train_model(
                model, train_loader, test_loader, criterion, optimizer, params['epochs']
            )
            
            # Save results
            result = {
                'params': params,
                'val_accuracy': val_acc,
                'best_epoch': best_epoch,
                'history': history
            }
            results.append(result)
            
            # Update best model if necessary
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = params
                best_model_state = model.state_dict()
            
            logger.info(f"Validation accuracy: {val_acc:.2f}%")
        
        # Log final results
        logger.info("\nGrid Search completed.")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")
        
        # Save best model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = root_dir / "models" / f"grid_search_results_{timestamp}.json"
        model_path = root_dir / "models" / f"best_model_{timestamp}.pt"
        
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_accuracy': float(best_accuracy),
                'all_results': results
            }, f, indent=2, default=str)
        
        # Save model
        torch.save(best_model_state, model_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Best model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during grid search: {str(e)}")
        raise

if __name__ == "__main__":
    main()