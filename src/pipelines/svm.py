from pathlib import Path
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import logging



# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

in_path = root_dir / 'data' / 'processed' / '1_cleaned_data.csv'

log_path = root_dir / 'logs' / 'svm_results.log'

# Ensure the log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

# Configure logging to use log_path
logging.basicConfig(
    filename=log_path,               # Use log_path as the log file
    level=logging.INFO,               # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)
logger = logging.getLogger(__name__)


def main():

    df = pd.read_csv(in_path)

    MAX_SEQ_LENGTH = max(len(text.split()) for text in df['TEXT']) #maximum length of a note event
    MAX_FEATURES = 10000 #only consider the top 10000 most frequent words in the corpus
    
    #tokenize and vectorize texts
    corpus = df['TEXT'].values.astype('U') 
    tfidf = TfidfVectorizer(max_features = MAX_FEATURES, ngram_range = (1, 5))   
    tdidf_tensor = tfidf.fit_transform(corpus)
    
    # binary classification
    X_train, X_test, Y_train, Y_test = train_test_split(tdidf_tensor,
                                                        df['behavior_tobacco_binary'].values,
                                                        test_size=0.3,
                                                        random_state=42
                                                       )
    
    binary_svm = SVC(kernel = 'linear')
    
    binary_svm.fit(X_train, Y_train)
    
    predictions = binary_svm.predict(X_test)
    
    logger.info('------------------------------------------')
    logger.info('Binary Results')
    logger.info('------------------------------------------')
    binary_report = classification_report(Y_test, predictions, digits=3)
    logger.info("\n" + binary_report)
    logger.info('------------------------------------------')
    
    
    #multi-class classification
    
    X_train, X_test, Y_train, Y_test = train_test_split(tdidf_tensor,
                                                        df['behavior_tobacco'].values,
                                                        test_size=0.3,
                                                        random_state=42
                                                       )
    
    multi_svm = SVC(kernel = 'linear', decision_function_shape = 'ovo')
    
    multi_svm.fit(X_train, Y_train)
    
    predictions = multi_svm.predict(X_test)

    logger.info('------------------------------------------')
    logger.info('Multi-class Results')
    logger.info('------------------------------------------')
    multi_class_report = classification_report(Y_test, predictions, digits=3)
    logger.info("\n" + multi_class_report)
    logger.info('------------------------------------------')

if __name__ == "__main__":
    main()

