import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model

from gensim.models import KeyedVectors

import pickle
import pandas as pd
from pathlib import Path
import numpy as np


# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

in_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"

embedding_file_path = (
    root_dir / "data" / "external" / "GoogleNews-vectors-negative300.bin.gz"
)

# out_path = root_dir / 'data' / 'processed' / '2_embedded_data.csv'

cleaned_data = pd.read_csv(in_path)
X = list(cleaned_data["TEXT"])


MAX_NUM_WORDS = 10000  # keep only x top words in the corpus

# TOKENIZATION

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X)
word_vector = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index

print("--------------------------")
print("----Saving Word Index-----")

out_path = root_dir / "data" / "processed" / "word_index.pkl"

with open(out_path, "wb") as f:
    pickle.dump(word_index, f)
print("--------------------------")


MAX_SEQ_LENGTH = len(max(cleaned_data["TEXT"]))  # max length of a note

input_tensor = pad_sequences(word_vector, maxlen=MAX_SEQ_LENGTH)

print("----Saving Input Tensor-----")

out_path = root_dir / "data" / "processed" / "input_tensor.pkl"

with open(out_path, "wb") as f:
    pickle.dump(input_tensor, f)

print("--------------------------")
print("Input Tensor Shape")
print(input_tensor.shape)


# WORD EMBEDDING

word2vec = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)

EMBEDDING_DIM = 300  # dim of GoogleNews

embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

for word, idx in word_index.items():
    if word in word2vec.key_to_index and idx < MAX_NUM_WORDS:
        embedding_matrix[idx] = word2vec[word]

out_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"

print("----Saving embedding matrix-----")
with open(out_path, "wb") as f:
    pickle.dump(embedding_matrix, f)

print("--------------------------")
