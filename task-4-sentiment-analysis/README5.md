# Task 5 - Sentiment Analysis Chatbot

## What I Did

For this task I added sentiment analysis to the existing fitness chatbot. The idea is that the chatbot should detect whether the user is happy, sad or neutral from their message and then respond in an appropriate way.

I used NLTK VADER for sentiment detection. We already used NLTK during training for text preprocessing so I was already familiar with it. VADER is a built in tool in NLTK that is specifically designed for short conversational text and gives a compound score between -1 and +1.

## How It Works

Every time the user sends a message it first goes through VADER before the chatbot generates a reply. If the compound score is above 0.05 it is positive, below -0.05 is negative, and everything in between is neutral.

Based on the sentiment the chatbot changes how it responds. For positive messages it adds an enthusiastic tone. For negative messages it adds an empathetic statement like "I understand that can be tough" before the actual answer. For the Gemini fallback I also change the prompt to match the mood.

The sentiment indicator at the top of the chat window shows the detected emotion and score in real time. The Sentiment Stats tab keeps a count of positive, negative and neutral messages throughout the conversation.

## Files

- gui1.py - original chatbot from training
- gui5_sentiment.py - new sentiment aware chatbot
- intents.json
- requirements.txt

## Libraries Used

- TensorFlow and Keras
- NLTK and VADER
- Google Generativeai
- Tkinter
