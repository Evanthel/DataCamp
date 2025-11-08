from collections import Counter
import nltk, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
nltk.download('punkt')
# Import data and labels
with open("words.json", 'r') as f1:
    words = json.load(f1)
with open("text.json", 'r') as f2:
    text = json.load(f2)
labels = np.load('labels.npy')
# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

# Looking up the mapping dictionary and assigning the index to the respective words
for i, sentence in enumerate(text):
    text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    
# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

text = pad_input(text, 50)
# Splitting dataset
train_text, test_text, train_label, test_label = train_test_split(text, labels, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.from_numpy(train_text), torch.from_numpy(train_label).long())
test_data = TensorDataset(torch.from_numpy(test_text), torch.from_numpy(test_label).long())
batch_size = 400
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, target_size):
        super(TicketClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, target_size)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2)
        return self.fc(conved)

# Model setup
vocab_size = len(words)
embed_size = 64
num_classes = len(set(labels))

model = TicketClassifier(vocab_size, embed_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for epoch in range(3):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

accuracy_metric = Accuracy(task='multiclass', num_classes=len(np.unique(labels)))
precision_metric = Precision(task='multiclass', num_classes=len(np.unique(labels)), average=None)
recall_metric = Recall(task='multiclass', num_classes=len(np.unique(labels)), average=None)

model.eval()
predicted = []

for i, (inputs, labels) in enumerate(test_loader):
    output = model(inputs)
    cat = torch.argmax(output, dim=-1)
    predicted.extend(cat.tolist())
    accuracy_metric(cat, labels)
    precision_metric(cat, labels)
    recall_metric(cat, labels)

accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()
print('Accuracy:', accuracy)
print('Precision (per class):', precision)
print('Recall (per class):', recall)