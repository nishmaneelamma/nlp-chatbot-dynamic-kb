# Task 1 - Dynamic Knowledge Base Chatbot

## What I Did

For this task I extended the chatbot I built during training to have a dynamic knowledge base. The problem with the original chatbot was that if I wanted to add new information I had to retrain the entire model from scratch which takes a lot of time. So I looked into ways to fix this and found ChromaDB which is a vector database that lets you store and search text by meaning.

## How It Works

The original Keras model still runs the same way as before. But now when the model is not confident enough about its answer, the chatbot automatically checks ChromaDB for a better response. ChromaDB converts text into vectors and finds the most similar match even if the exact words are different.

I also added a Knowledge Base tab to the GUI where you can add new topics and responses directly from the app without writing any code. There is also a button to automatically load all the existing intents from intents.json into ChromaDB.

One more thing I added is an auto-refresh feature. A background thread checks if intents.json has changed every 60 seconds and automatically updates the database if it has.

## Files

- gui1.py - the original chatbot from training
- gui2_dynamic.py - the new chatbot with dynamic knowledge base
- intents.json - training data
- requirements.txt

## Libraries Used

- TensorFlow and Keras
- NLTK
- ChromaDB
- Tkinter
