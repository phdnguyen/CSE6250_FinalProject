import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]
output_dir = root_dir / "output" / "classic_ml"
output_dir.mkdir(parents=True, exist_ok=True)

log_path = root_dir / "logs" / "classic_ml_avg_results.log"

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
logger.info("Starting classic ML model training with averages")

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000)
scaler = StandardScaler(with_mean=False)

# Define classical ML models
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'SVM': SVC(kernel='linear', probability=True),
    'NaiveBayes': MultinomialNB(),
}

# Helper function to calculate average and std for metrics
def calculate_avg_std(reports):
    avg_metrics = {}
    std_metrics = {}

    for metric in reports[0].keys():
        values = [report[metric] for report in reports]
        avg_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)

    return avg_metrics, std_metrics

# Function to save the confusion matrix
def save_confusion_matrix(y_true, y_pred, model_name, description, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    display_labels = ['Class 0', 'Class 1'] if description == 'Binary' else [f'Class {i}' for i in np.unique(y_true)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - {model_name} ({description})")
    output_path = output_dir / f"confusion_matrix_{model_name}_{description}.png"
    plt.savefig(output_path)
    plt.close()

# Run and evaluate models for binary and multiclass classification
for classification_type, y, description in [('binary', y_binary, 'Binary'), ('multiclass', y_multiclass, 'Multiclass')]:
    logger.info(f"Starting {description} classification")
    all_results = {model_name: [] for model_name in models.keys()}  # Store results for each model

    for model_name, model in models.items():
        logger.info(f"Running {model_name} ({description}) 10 times")

        reports = []
        for run in range(10):
            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run)

            # Vectorize the data
            x_train_tfidf = vectorizer.fit_transform(x_train)
            x_test_tfidf = vectorizer.transform(x_test)

            # Scale the data
            x_train_scaled = scaler.fit_transform(x_train_tfidf)
            x_test_scaled = scaler.transform(x_test_tfidf)

            # Train the model
            model.fit(x_train_scaled, y_train)

            # Make predictions
            preds = model.predict(x_test_scaled)

            # Save the confusion matrix for the last run
            if run == 9:  # Last run
                save_confusion_matrix(y_test, preds, model_name, description, output_dir)

            # Generate classification report
            report = classification_report(y_test, preds, output_dict=True, digits=3)
            accuracy = accuracy_score(y_test, preds)
            reports.append({
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1-score": report["macro avg"]["f1-score"],
                "accuracy": accuracy,
            })

        # Calculate averages and standard deviations
        avg_metrics, std_metrics = calculate_avg_std(reports)

        logger.info(f"Results for {model_name} ({description}):")
        logger.info(f"Average metrics: {avg_metrics}")
        logger.info(f"Standard deviations: {std_metrics}")

        all_results[model_name].append((avg_metrics, std_metrics))

    logger.info(f"Completed {description} classification")
    logger.info("------------------------------------------")

logger.info("All classic ML model training with averages completed.")
print("Training and evaluation completed. Check the log file for detailed results.")