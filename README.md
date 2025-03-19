# emotion_detector

This repository provides an emotion recognition system that can identify the emotion behind a given text or transcribed speech. The system is built using machine learning techniques and trained on the Emotion dataset. It uses Natural Language Processing (NLP) to analyze text and the Whisper model to transcribe audio, followed by emotion prediction using a Naive Bayes classifier.

**Requirements**
You’ll need Python 3.x and the following libraries:
numpy
pandas
seaborn
neattext
joblib
whisper
sklearn
datasets

**How It Works**
Dataset Loading: The dataset, which contains text labeled with emotions, is loaded using Hugging Face's datasets library.

Text Preprocessing: The text undergoes cleaning by removing special characters and stopwords using the neattext library.

Vectorization: The processed text is converted into numerical data using the TfidfVectorizer.

Model Training: A Naive Bayes classifier (MultinomialNB) is trained to predict emotions based on the vectorized text.

Saving Models: After training, the model and vectorizer are saved using joblib for future use.

Emotion Prediction: You can predict emotions for text using the predict_emotion function. Additionally, you can predict the emotion of audio files after they are transcribed using the Whisper model.
