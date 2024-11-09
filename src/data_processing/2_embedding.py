import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

in_path = root_dir / "data" / "processed" / "1_cleaned_data.csv"
embedding_file_path = (
    root_dir / "data" / "external" / "GoogleNews-vectors-negative300.bin.gz"
)

cleaned_data = pd.read_csv(in_path)
X = list(cleaned_data["TEXT"])

MAX_NUM_WORDS = 10000  # keep only x top words in the corpus
EMBEDDING_DIM = 300  # dim of GoogleNews

# TOKENIZATION
# Create vocabulary from the dataset
counter = Counter()
for text in X:
    counter.update(text.split())

# Create a vocab object without the 'max_size' argument
vocab = build_vocab_from_iterator(
    [text.split() for text in X],
    specials=['<unk>', '<pad>'],
    special_first=True
)

# Set unknown index
vocab.set_default_index(vocab['<unk>'])

# Create input tensors
word_vector = [[vocab[token] for token in text.split()] for text in X]
input_tensor = [torch.tensor(seq, dtype=torch.long) for seq in word_vector]

# Save word index
print("--------------------------")
print("----Saving Word Index-----")

out_path = root_dir / "data" / "processed" / "word_index.pkl"
with open(out_path, "wb") as f:
    pickle.dump(vocab.get_stoi(), f)

print("--------------------------")

# Pad sequences to the maximum length
MAX_SEQ_LENGTH = max(len(seq) for seq in word_vector)
input_tensor = pad_sequence(input_tensor, batch_first=True, padding_value=vocab['<pad>'])

# Save input tensor
print("----Saving Input Tensor-----")
out_path = root_dir / "data" / "processed" / "input_tensor.pkl"
with open(out_path, "wb") as f:
    pickle.dump(input_tensor, f)

print("--------------------------")
print("Input Tensor Shape")
print(input_tensor.shape)

# WORD EMBEDDING
word2vec = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)

embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
for word, idx in vocab.get_stoi().items():
    if word in word2vec.key_to_index:
        embedding_matrix[idx] = word2vec[word]

# Save embedding matrix
out_path = root_dir / "data" / "processed" / "embedding_matrix.pkl"
print("----Saving embedding matrix-----")
with open(out_path, "wb") as f:
    pickle.dump(embedding_matrix, f)

print("--------------------------")
