import numpy as np
import random
import json

import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

stemmer = PorterStemmer()

with open('dataset.json', 'r') as f:
    datasets = json.load(f)

all_words = []
tags = []
xy = []

# split sentence into array of words/tokens
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# find the root form of the word
def stem(word):
    return stemmer.stem(word.lower())

# return bag of words array:
# 1 for each known word that exists in the sentence, otherwise 0
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


# loop through each sentence in our intents patterns
for dataset in datasets['intents']:
    tag = dataset['tag']
    # add to tag list
    tags.append(tag)
    for pattern in dataset['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to the words list
        all_words.extend(w)
        # add to the xy pair => word and its tag
        xy.append((w, tag))

# ignoring ?, ., !
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort the arrays all_words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=8,
                          shuffle=True,
                          num_workers=0)
#  you can change num_workers=0 to 2 , if you have pytorch gpu(cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
