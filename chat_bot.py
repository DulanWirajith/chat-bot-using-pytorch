import random
import json
import torch

from chat_bot_model import NeuralNetwork

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
