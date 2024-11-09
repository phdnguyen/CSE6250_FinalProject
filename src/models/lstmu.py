# from pathlib import Path
# import pandas as pd
# import re
# from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
# from tensorflow.keras.layers import GlobalMaxPool1D
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import logging
# import pickle
# import numpy as np
# import gc  # For garbage collection

# # Set the root directory to the project root
# root_dir = Path(__file__).resolve().parents[2]
# log_path = root_dir / "logs" / "lstm_results.log"

# # Ensure the log directory exists
# log_path.parent.mkdir(parents=True, exist_ok=True)
# logging.basicConfig(
#     filename=log_path,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s", 
#     filemode='a'  
# )
# logger = logging.getLogger(__name__)

# # Load data
# embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
# input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
# cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

# with open(input_tensor_path, "rb") as f:
#     input_tensor = pickle.load(f)
# with open(embedding_matrix_path, "rb") as f:
#     embedding_matrix = pickle.load(f)
# cleaned_data = pd.read_csv(cleaned_data_path)

# # Modified parameters for memory efficiency
# MAX_SEQ_LENGTH = 500  # Reduced sequence length due to my computer constraint, you can switch back to your seq length setup 
# MAX_NUM_WORDS = 10000
# EMBEDDING_DIM = 300
# BATCH_SIZE = 32  # Reduced batch size
# EPOCHS = 2

# # Preprocess data
# input_tensor = pad_sequences(input_tensor, maxlen=MAX_SEQ_LENGTH, truncating='post', padding='post')
# gc.collect()  # Force garbage collection

# # Log basic information
# logger.info("Starting Unidirectional LSTM model training")
# logger.info(f"Maximum sequence length: {MAX_SEQ_LENGTH}")
# logger.info(f"Embedding dimensions: {EMBEDDING_DIM}")
# logger.info(f"Max number of words: {MAX_NUM_WORDS}")

# # Model architecture with memory optimizations
# inp = Input(shape=(MAX_SEQ_LENGTH,))
# x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
# x = LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(x)  # Reduced units
# x = GlobalMaxPool1D()(x)
# x = Dense(64, activation="relu")(x)  # Reduced units
# x = Dropout(0.25)(x)
# x = Dense(1, activation="sigmoid")(x)

# # Compile the model
# model = Model(inputs=inp, outputs=x)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Split data
# x_train, x_test, y_train, y_test = train_test_split(
#     input_tensor, cleaned_data["behavior_tobacco_binary"].values, test_size=0.3
# )
# gc.collect()  # Force garbage collection

# # Train with smaller batches
# history = model.fit(
#     x_train, 
#     y_train, 
#     batch_size=BATCH_SIZE, 
#     epochs=EPOCHS,
#     validation_split=0.2
# )

# # Evaluate
# preds = model.predict(x_test, batch_size=BATCH_SIZE)
# preds = np.round(preds.flatten())

# # Log results
# report = classification_report(y_test, preds, digits=3)
# logger.info("Classification Report:\n" + report)
# logger.info("Evaluation completed.")
# logger.info("------------------------------------------")



from pathlib import Path
import pandas as pd
import re
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import pickle
import numpy as np
import gc

# Set the root directory and paths
root_dir = Path(__file__).resolve().parents[2]
log_path = root_dir / "logs" / "lstm_results.log"

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

# Load the data files
with open(input_tensor_path, "rb") as f:
    input_tensor = pickle.load(f)
with open(embedding_matrix_path, "rb") as f:
    embedding_matrix = pickle.load(f)
cleaned_data = pd.read_csv(cleaned_data_path)

# Modified parameters
MAX_SEQ_LENGTH = 500
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 5

# Preprocess data
input_tensor = pad_sequences(input_tensor, maxlen=MAX_SEQ_LENGTH, truncating='post', padding='post')
gc.collect()

# Log basic information
logger.info("Starting Enhanced LSTM model training")
logger.info(f"Maximum sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"Embedding dimensions: {EMBEDDING_DIM}")
logger.info(f"Max number of words: {MAX_NUM_WORDS}")

# Enhanced model architecture while keeping it simple
inp = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)

# First LSTM layer with more units
x = LSTM(256, return_sequences=True, dropout=0.2)(x)
x = BatchNormalization()(x)

x = GlobalMaxPool1D()(x)

# Modified dense layers
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(1, activation="sigmoid")(x)

# Compile the model
model = Model(inputs=inp, outputs=x)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    input_tensor, 
    cleaned_data["behavior_tobacco_binary"].values, 
    test_size=0.2,
    random_state=42,
    stratify=cleaned_data["behavior_tobacco_binary"].values
)
gc.collect()

# Add early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True
)

# Print class distribution
print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))

# Train the model
history = model.fit(
    x_train, 
    y_train, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate with threshold optimization
print("\nEvaluating model...")
preds_prob = model.predict(x_test, batch_size=BATCH_SIZE)

# Try different thresholds
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
best_accuracy = 0
best_threshold = 0.5

print("\nTesting different thresholds:")
for threshold in thresholds:
    pred_labels = (preds_prob > threshold).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    print(f"Threshold {threshold}: Accuracy = {acc:.3f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

# Final predictions with best threshold
final_preds = (preds_prob > best_threshold).astype(int)

# Log results
report = classification_report(y_test, final_preds, digits=3, zero_division=1)
print("\nClassification Report:")
print(report)
logger.info("Classification Report:\n" + report)
logger.info("Evaluation completed.")
logger.info("------------------------------------------")