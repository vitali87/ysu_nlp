import numpy as np
import torch
from torch import nn
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec

batch_size = 10
seq_len = 200
n_epochs = 10

data = pd.read_csv("amazon_reviews.csv")

# drop row when Sentiment is 3 as it is ambigous
data = data[data.Sentiment != 3]

# Sentiments in the range (1,2) are converted to 0 (negative) and (4,5) are converted to 1 (positive)
data["Sentiment"] = data["Sentiment"].apply(lambda rating: 1 if rating > 3 else 0)

# define the vocabulary size (number of unique words) and the maximum length of a review (in words)
vocab_size = 50000

# Dumb tokenization
reviews = data["Review"].tolist()
reviews = [str(review).lower().split() for review in reviews]

all_words = [word for review in reviews for word in review]
word_freq = Counter(all_words)
sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
limited_vocab = {
    word: index for index, (word, _) in enumerate(sorted_vocab[:vocab_size])
}

# Train the word2vec model
model = Word2Vec(reviews, vector_size=100, window=5, min_count=1, workers=4)


def create_batch(reviews, model):
    batch = []
    for review in reviews:
        if vecs := [model.wv[word] for word in review if word in model.wv.index_to_key]:
            vecs = np.array(vecs)
            avg_vec = np.mean(vecs, axis=0)
            batch.append(avg_vec)
    return np.array(batch)


class FFNNStatic(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(data_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, data):
        hidden = self.fc1(data)
        return self.fc2(self.activation(hidden))


# Instanciate the model
import itertools

ffnn_static = FFNNStatic(100, 100, 1)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(ffnn_static.parameters(), lr=0.0001)

total_reviews = len(reviews)


# Training loop
for _, i in itertools.product(range(n_epochs), range(0, total_reviews, batch_size)):
    batch = reviews[i : i + batch_size]
    target = data["Sentiment"].to_list()[i : i + batch_size]
    batch = create_batch(batch, model)
    output = ffnn_static(torch.tensor(batch))
    loss = loss_fn(output, torch.tensor(target, dtype=torch.float32))
    print(f"Current loss: {loss}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Add special tokens to the vocabulary
limited_vocab["<UNK>"] = vocab_size
limited_vocab["<PAD>"] = vocab_size + 1


# Convert reviews to indices
reviews_with_indices = []

for review in reviews:
    review_indices = [
        limited_vocab.get(word, limited_vocab["<UNK>"]) for word in review
    ]
    reviews_with_indices.append(review_indices)

# Convert words to indices
reviews_ids = [limited_vocab.get(word, limited_vocab["<UNK>"]) for word in all_words]


# Batched vectorized mapping function of input sentences to indices
def vectorized_seqs(seq, word_to_idx, seq_len):
    """
    Args:
        seq: List of sentences (reviews)
        word_to_idx: Dictionary mapping words to indices
        seq_len: Maximum length of a sentence (review)
    Returns:
        Vectorized sequence
    """
    seq = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in seq]
    seq += [word_to_idx["<PAD>"]] * (seq_len - len(seq))
    seq = seq[:seq_len]
    return seq


# Vectorize the reviews
vectorized_sequences = [vectorized_seqs(seq, limited_vocab, seq_len) for seq in reviews]

# Create a tensor for the reviews
X = torch.tensor(vectorized_sequences, dtype=torch.int64)
y = torch.tensor(data["Sentiment"].tolist(), dtype=torch.float32)

# create a dataloader with batch size of batch_size that randomly shuffles the data
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# embedding = nn.Embedding(
#     vocab_size + 2, 50, padding_idx=50001
# )  # 50 is the embedding size

# X_emb = embedding(X)

# X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
# H, W_hh = torch.randn(3, 4), torch.randn(4, 4)

# print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))  # Or just X @ W_xh + H @ W_hh
# print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))


# First let's train this with simple Feed Forward Neural Network
# Define the model
class FFNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(data_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.embed = nn.Embedding(vocab_size + 2, data_size, padding_idx=50001)

    def forward(self, data):
        data = self.embed(data)
        hidden = self.fc1(data)
        return self.fc2(hidden)


# Instanciate the model
ffnn = FFNN(200, 200, 1)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.0001)

# Training loop
for _ in range(n_epochs):
    for batch, target in dataloader:
        output = ffnn(batch)
        loss = loss_fn(output, target)
        print(f"Current loss: {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input_ = torch.cat((data, last_hidden), 1)
        hidden_ = self.i2h(input_)
        output_ = self.h2o(hidden_)
        return hidden_, output_


# Instanciate the RNN
rnn = RNN(200, 20, 1)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001)

# Training loop
for batch, target in dataloader:
    hidden = torch.zeros(batch_size, 20)
    batch_loss = 0
    for i in range(seq_len):
        hidden, output = rnn(batch, hidden)
        current_loss = loss_fn(output, target)
        batch_loss += current_loss
        print(
            f"TimeStep {i}, Current loss: {current_loss}, Cumulative loss {batch_loss.item()}"
        )
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

rnn_pytorch = torch.nn.RNN(200, 20, batch_first=True)

# Training loop (needs work)
for batch, target in dataloader:
    batch = batch.unsqueeze(1)  # Add a sequence length dimension

    # Initialize hidden state
    hidden = torch.zeros(1, batch.shape[0], 20)

    batch_loss = 0
    batch = torch.tensor(batch, dtype=torch.float32)
    for i in range(seq_len):
        output, hidden = rnn_pytorch(batch, hidden)
        output_reduced = output.mean(dim=2).squeeze(
            1
        )  # Reducing dimension 2 by taking mean and then removing singleton dimension 1.

        # current_loss = loss_fn(output, target)
        current_loss = loss_fn(output_reduced, target)

        batch_loss += current_loss
        print(
            f"TimeStep {i}, Current loss: {current_loss}, Cumulative loss {batch_loss.item()}"
        )
