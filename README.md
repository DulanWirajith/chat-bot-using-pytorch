# chat-bot-using-pytorch

This is a chatbot can use for Online Ecommerce Platform.
Firstly you have to give datasets about the ecommerce site. Then chatbot will learn about the shop and work finely...

If you are running this first time please uncomment 
"nltk.download('punkt')" 
in training.py file. Because it wants the "punkt" for run the model

Training data in dataset.json file. In dataset have tag, pattern and response.
	Tag is the main category of the sentence (y)
	Patterns are the things which customer says (X)
	Responses are the things which chatbot have to respond to the user. 
E.g.:

![Alt text](dataset_explain.jpg?raw=true "dataset explaining...")
