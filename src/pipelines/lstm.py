from pathlib import Path
import pandas as pd
import re
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import logging
import pickle

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

log_path = root_dir / "logs" / "lstm_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

# Configure logging to use log_path
logging.basicConfig(
    filename=log_path,  # Use log_path as the log file
    level=logging.INFO,  # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)
logger = logging.getLogger(__name__)

embedding_matrix_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
input_tensor_path = root_dir / "data" / "processed" / "input_tensor.pkl"
cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

with open(input_tensor_path, "rb") as f:
    input_tensor = pickle.load(f)

with open(embedding_matrix_path, "rb") as f:
    embedding_matrix = pickle.load(f)

cleaned_data = pd.read_csv(cleaned_data_path)

MAX_SEQ_LENGTH = len(max(cleaned_data["TEXT"]))  # max length of a note
MAX_NUM_WORDS = 10000  # keep only x top words in the corpus
EMBEDDING_DIM = 300  # dim of GoogleNews


inp = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(
    LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
)(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation="sigmoid")(x)


# Compile the model
model = Model(inputs=inp, outputs=x)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    input_tensor, cleaned_data["behavior_tobacco_binary"].values, test_size=0.3
)

model.fit(x_train, y_train, batch_size=16, epochs=10)
preds = model.predict(x_test)

preds = np.round(preds.flatten())


logger.info("------------------------------------------")
logger.info("LSTM Results")
logger.info("------------------------------------------")
report = classification_report(y_test, predictions, digits=3)
logger.info("\n" + report)
logger.info("------------------------------------------")
