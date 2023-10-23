import numpy as np
import torch
from torch import nn
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

batch_size = 10
seq_len = 200
n_epochs = 10
hidden_size = 100

data = pd.read_csv("amazon_reviews.csv")

# drop row when Sentiment is 3 as it is ambigous
data = data[data.Sentiment != 3]


# Sentiments in the range (1,2) are converted to 0 (negative) and (4,5) are converted to 1 (positive)
data["Sentiment"] = data["Sentiment"].apply(lambda rating: 1 if rating > 3 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    data["Review"], data["Sentiment"], test_size=0.2, random_state=1
)

# define the vocabulary size (number of unique words) and the maximum length of a review (in words)
vocab_size = 50_000

# Dumb tokenization
reviews = X_train.tolist()
reviews = [str(review).lower().split() for review in reviews]

all_words = [word for review in reviews for word in review]
word_freq = Counter(all_words)
sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
limited_vocab = {
    word: index for index, (word, _) in enumerate(sorted_vocab[:vocab_size])
}

# Train the word2vec model
model = Word2Vec(reviews, vector_size=hidden_size, window=5, min_count=1, workers=4)

model.wv["good"]


def create_batch(reviews, model):
    batch = []
    index_keys = set(model.wv.index_to_key)

    for review in reviews:
        if vecs := [model.wv[word] for word in review if word in index_keys]:
            vecs = torch.tensor(vecs)

            # Compute the mean vector
            avg_vec = torch.mean(vecs, axis=0)

            # Append the mean vector to the batch list
            batch.append(avg_vec)

    # Convert list of tensors to a single 2D tensor
    return torch.stack(batch)


class FFNNStatic(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, data):
        hidden = self.fc1(data)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
        return self.activation(output)


ffnn_static = FFNNStatic(hidden_size, 1)

# Define the loss function
loss_fn = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(ffnn_static.parameters(), lr=0.0001)

total_reviews = len(reviews)


# Training loop
for epoch in range(n_epochs):
    for i in range(0, total_reviews, batch_size):
        batch = reviews[i : i + batch_size]
        target = y_train.to_list()[i : i + batch_size]

        if len(batch) != len(target):
            continue

        batch = create_batch(batch, model)
        output = ffnn_static(batch)
        loss = loss_fn(output.flatten(), torch.tensor(target, dtype=torch.float32))
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
y = torch.tensor(y_train.tolist(), dtype=torch.float32)

# create a dataloader with batch size of batch_size that randomly shuffles the data
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class FFNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.embed = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=50001)
        self.activation = nn.Sigmoid()

    def forward(self, data):
        data = self.embed(data)
        data = torch.mean(data, dim=1)  # average along the sequence dimension
        hidden = self.fc1(data)
        hidden = self.relu(hidden)
        output = self.fc2(hidden).squeeze()
        return self.activation(output)


# Instanciate the model
ffnn = FFNN(200, 1)

# Define the loss function
loss_fn = nn.BCELoss()

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

# RNN
# X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
# H, W_hh = torch.randn(3, 4), torch.randn(4, 4)

# print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))  # Or just X @ W_xh + H @ W_hh
# print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Be cautious here: embed_dim + hidden_size should match the last dimension
        # of the concatenated tensor.
        self.i2h = nn.Linear(embed_dim + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, data, last_hidden):
        embedded = self.embedding(data)
        last_hidden_expanded = last_hidden.unsqueeze(1).expand(
            embedded.size(0), embedded.size(1), self.hidden_size
        )
        input_ = torch.cat((embedded, last_hidden_expanded), 2)
        hidden_ = self.i2h(input_.view(-1, embedded.size(2) + last_hidden.size(1)))
        hidden_ = hidden_.view(embedded.size(0), embedded.size(1), self.hidden_size)
        output_ = self.h2o(hidden_)
        output_ = self.activation(output_)
        return hidden_[:, -1, :], output_  # Only return the last hidden state


# Instantiate the RNN
rnn = RNN(vocab_size=vocab_size + 2, embed_dim=100, hidden_size=100, output_size=1)

# Define the loss function
loss_fn = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001)

for batch, target in dataloader:
    optimizer.zero_grad()
    # Initialize hidden to zero at the beginning of processing each batch
    hidden = torch.zeros(batch_size, hidden_size)
    for _ in range(seq_len):
        hidden, output = rnn(batch, hidden)
    current_loss = loss_fn(output[:, -1].squeeze(), target)
    print(f"Final time step loss: {current_loss.item()}")
    current_loss.backward()
    optimizer.step()


# We will look at native pytorch rnn too.
rnn_pytorch = torch.nn.RNN(200, hidden_size, batch_first=True)
