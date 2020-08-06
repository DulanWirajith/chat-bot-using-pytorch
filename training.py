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

