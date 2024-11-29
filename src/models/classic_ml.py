import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

log_path = root_dir / "logs" / "classic_ml_results.log"

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a',
)
logger = logging.getLogger(__name__)

# Paths to data files
cleaned_data_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

# Load the cleaned data
cleaned_data = pd.read_csv(cleaned_data_path)
X = cleaned_data['TEXT']
y_binary = cleaned_data['behavior_tobacco_binary'].values
y_multiclass = cleaned_data['behavior_tobacco'].values

# Log basic information
logger.info("Starting classic ML model training")

# Split the data for binary classification
x_train_binary, x_test_binary, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Split the data for multiclass classification
x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multiclass, test_size=0.3, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000)
x_train_binary_tfidf = vectorizer.fit_transform(x_train_binary)
x_test_binary_tfidf = vectorizer.transform(x_test_binary)

x_train_multi_tfidf = vectorizer.fit_transform(x_train_multi)
x_test_multi_tfidf = vectorizer.transform(x_test_multi)

# Scale the data
scaler = StandardScaler(with_mean=False)
x_train_binary_scaled = scaler.fit_transform(x_train_binary_tfidf)
x_test_binary_scaled = scaler.transform(x_test_binary_tfidf)

x_train_multi_scaled = scaler.fit_transform(x_train_multi_tfidf)
x_test_multi_scaled = scaler.transform(x_test_multi_tfidf)

# Define classical ML models
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'SVM': SVC(kernel='linear', probability=True),
    'NaiveBayes': MultinomialNB(),
}

# Train and evaluate each model for binary classification
logger.info("Starting binary classification")
for model_name, model in models.items():
    logger.info(f"Training {model_name} (Binary)")
    model.fit(x_train_binary_scaled, y_train_binary)

    # Make predictions
    preds = model.predict(x_test_binary_scaled)

    # Log classification report
    logger.info(f"Generating classification report for {model_name} (Binary)")
    report = classification_report(y_test_binary, preds, digits=3)
    logger.info(f"Classification Report for {model_name} (Binary):\n{report}")
    logger.info("------------------------------------------")

# Train and evaluate each model for multiclass classification
logger.info("Starting multiclass classification")
for model_name, model in models.items():
    logger.info(f"Training {model_name} (Multiclass)")
    model.fit(x_train_multi_scaled, y_train_multi)

    # Make predictions
    preds = model.predict(x_test_multi_scaled)

    # Log classification report
    logger.info(f"Generating classification report for {model_name} (Multiclass)")
    report = classification_report(y_test_multi, preds, digits=3)
    logger.info(f"Classification Report for {model_name} (Multiclass):\n{report}")
    logger.info("------------------------------------------")

logger.info("All classic ML model training completed.")
print("Training and evaluation completed. Check the log file for detailed results.")
