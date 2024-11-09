from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "lstm_grid_search_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load data
embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

with open(input_tensor_path, "rb") as f:
    input_tensor = pickle.load(f)
with open(embedding_matrix_path, "rb") as f:
    embedding_matrix = pickle.load(f)
cleaned_data = pd.read_csv(cleaned_data_path)

# Constants
MAX_SEQ_LENGTH = len(max(cleaned_data["TEXT"]))
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300

def create_model(lstm_units=100, 
                dense_units=100, 
                dropout_rate=0.25, 
                recurrent_dropout=0.1,
                learning_rate=0.001):
    """
    Create and return compiled model with given parameters
    """
    inp = Input(shape=(MAX_SEQ_LENGTH,))
    x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
    x = LSTM(lstm_units, 
            return_sequences=True, 
            dropout=dropout_rate, 
            recurrent_dropout=recurrent_dropout)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", 
                 optimizer=optimizer, 
                 metrics=["accuracy"])
    return model

# Split data
logger.info("Splitting data into training and testing sets")
x_train, x_test, y_train, y_test = train_test_split(
    input_tensor, 
    cleaned_data["behavior_tobacco_binary"].values, 
    test_size=0.3,
    random_state=42
)

# Create KerasClassifier
model = KerasClassifier(
    build_fn=create_model,
    verbose=0
)

# Define parameter grid
param_grid = {
    'lstm_units': [50, 100, 150],
    'dense_units': [50, 100, 150],
    'dropout_rate': [0.2, 0.25, 0.3],
    'recurrent_dropout': [0.1, 0.15, 0.2],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [64, 128],
    'epochs': [2, 3]
}

# Create GridSearchCV
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Log grid search parameters
logger.info("Starting Grid Search with parameters:")
logger.info(f"Parameter grid: {param_grid}")

# Fit GridSearchCV
try:
    grid_result = grid.fit(x_train, y_train)
    
    # Log results
    logger.info("Grid Search completed.")
    logger.info(f"Best score: {grid_result.best_score_}")
    logger.info(f"Best parameters: {grid_result.best_params_}")
    
    # Get best model and evaluate on test set
    best_model = grid_result.best_estimator_
    y_pred = best_model.predict(x_test)
    
    # Log classification report
    report = classification_report(y_test, y_pred, digits=3)
    logger.info("Classification Report for Best Model:\n" + report)
    
    # Save best model parameters
    best_params_path = root_dir / "models" / "best_lstm_params.pkl"
    with open(best_params_path, 'wb') as f:
        pickle.dump(grid_result.best_params_, f)
    logger.info(f"Best parameters saved to {best_params_path}")
    
except Exception as e:
    logger.error(f"Error during grid search: {str(e)}")
    raise

logger.info("Grid search process completed.")
logger.info("------------------------------------------")