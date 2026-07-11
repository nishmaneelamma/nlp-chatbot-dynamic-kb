# Task 2 - Multimodal Chatbot

## What I Did

For this task I extended the chatbot to handle both text and image inputs using Google Gemini AI. During training we learned how to use Gemini for text chat in QA.py and for image analysis in img_model.py. For this task I combined both of those into one single Tkinter application along with the original fitness chatbot.

## How It Works

The app has three tabs. The Chat tab works the same as the original chatbot using the Keras model for fitness and first aid questions. If the model is not confident it falls back to Gemini for text answers.

The Image Analysis tab is the new part. The user can upload any image and type an optional question about it. When they click Analyse, the image gets sent to Gemini Vision which reads the image and returns a description or answer. The response also shows up in the Chat tab so the full conversation is in one place.

The API Key tab is where you enter the Gemini API key. I did not hardcode it in the file because that is a security risk.

## Training Files Included

I included QA.py and img_model.py from the training phase to show where I learned the Gemini concepts from. gui3_multimodal.py builds directly on both of them.

## Files

- QA.py - Gemini text chat from training
- img_model.py - Gemini image analysis from training
- intro_palm_api.ipynb - PaLM API notebook from training
- python.ipynb - Gemini Python quickstart from training
- gui3_multimodal.py - new multimodal chatbot
- intents.json
- requirements.txt

## Libraries Used

- TensorFlow and Keras
- Google Generativeai
- Pillow
- NLTK
- Tkinter
