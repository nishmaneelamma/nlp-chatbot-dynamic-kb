# Task 3 - Medical Q&A Chatbot

## What I Did

For this task I built a medical question answering chatbot using the MedQuAD dataset. The dataset was downloaded from Kaggle at kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research which is the CSV version of the original MedQuAD dataset from github.com/abachaa/MedQuAD. It has 16412 medical question and answer pairs from sources like NIH and CDC.

I used Streamlit for the interface because we learned it during training in QA.py and img_model.py.

## How It Works

When the user types a question the chatbot searches the dataset for the most relevant answer. I wrote a retrieval function that gives scores based on three things - how many words match between the question and dataset questions, whether the focus area matches, and a similarity score using Python's difflib library.

The chatbot also does basic medical entity recognition. It checks the response text for known disease names, symptoms, treatments and drugs and shows them as coloured tags below the answer so the user can quickly see what type of information is being discussed.

If no good match is found in the dataset it falls back to Gemini AI.

## Files

- medical_chatbot.py - main Streamlit app
- medquad.csv - the dataset from Kaggle
- requirements.txt

## Libraries Used

- Streamlit
- Pandas
- Google Generativeai
- Python difflib
