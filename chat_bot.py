import random
import json
import torch
import nltk
import numpy as np

from chat_bot_model import NeuralNetwork
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dataset.json', 'r') as f:
    datasets = json.load(f)

# load chat_data.pth file
FILE = "chat_data.pth"
chat_data = torch.load(FILE)

input_size = chat_data["input_size"]
hidden_size = chat_data["hidden_size"]
output_size = chat_data["output_size"]
model_state = chat_data["model_state"]

#  load model
chat_bot_model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
chat_bot_model.load_state_dict(model_state)
chat_bot_model.eval()

bot_name = "ZeeLot"
print("Hey Buddy Welcome to the ZeeLot Shop.. Let's Chat! (type 'quit' to exit)")

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

all_words = chat_data['all_words']
tags = chat_data['tags']

while True:
    # getting users input
    sentence = input("You: ")
    if sentence == "quit":
        break

    # tokenize the sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = chat_bot_model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    # get te probability of predicted sentence
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for dataset in datasets['intents']:
            if tag == dataset["tag"]:
                print(f"{bot_name}: {random.choice(dataset['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")