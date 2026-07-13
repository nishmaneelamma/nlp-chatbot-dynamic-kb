# Multilingual Fitness Chatbot

This project extends the original fitness chatbot to support multiple languages. The chatbot can automatically detect the language the user is typing in and reply in the same language.

---

## Languages Supported

The chatbot supports five languages in total. English is the original language from the training phase. I added Hindi, Spanish, French and German as the three additional languages required for this task.

---

## How It Works

When a user sends a message, the chatbot first uses the googletrans library to detect what language it is written in. It then translates the message to English so the Keras model can process it, since the model was only trained on English data. After getting the response in English, it translates the response back to the user's detected language before displaying it.

If the user wants to manually choose a language instead of using auto-detection, they can go to the Language Settings tab and select from the dropdown.

Gemini AI is used as a fallback for questions outside the trained intents, and it is also prompted to respond in the appropriate language.

---

## Features

- Automatic language detection using googletrans
- Supports English, Hindi, Spanish, French and German
- Auto-detect toggle in Language Settings tab
- Manual language selection via dropdown
- Cultural greetings shown for each language
- Gemini AI fallback with multilingual support
- All responses translated back to user's language

---

## Files

- gui1.py — original English chatbot from training
- gui6_multilingual.py — new multilingual chatbot
- intents.json — dataset used for the Keras model
- requirements.txt — dependencies

---

## Libraries Used

- TensorFlow and Keras for the text classification model
- googletrans for language detection and translation
- NLTK for text preprocessing
- Google Generativeai for Gemini AI fallback
- Tkinter for the user interface
