# chat-bot-using-pytorch

This is a chatbot can use for Online Ecommerce Platform.
Firstly you have to give datasets about the ecommerce site. Then chatbot will learn about the shop and work finely...

If you are running this first time please uncomment 
"nltk.download('punkt')" 
in training.py file. Because it wants the "punkt" for run the model

#Training data in dataset.json file. 
In dataset have tag, pattern and response.
	Tag is the main category of the sentence (y)
	Patterns are the things which customer says (X)
	Responses are the things which chatbot have to respond to the user. 
E.g.:

![Alt text](dataset_explain.jpg?raw=true "dataset explaining...")


#NLP Preprocessing Pipeline

![Alt text](nlp.jpg?raw=true "NLP...")

Firstly Tokenize the sentence.
And then convert all words in to lowercase.
Then stemming the words.
After that, exclude all punctuation characters like !, ?, ’, ”.
And then put all words to the bag of words.


#Feed-Forward Neural Network

![Alt text](model.jpg?raw=true "Neural Network...")

For the chatbot, I used simple feed-forward neural network. 
	It has one input layer with 54 neurons (all number of words = 54) and 
	two hidden layers, and 
	an output layer with 7 neurons (all number of tags= 7)
The neural network predicts the probabilities of tags. And if probability greater than 0.75 I get that tag as output. And then, I give a response to the user using that tag’s response in the chat_data.pth file.

Chat with chatbot

Run chat_bot.py file in the console.
![Alt text](chat.jpg?raw=true "Chat with ChatBot...")
