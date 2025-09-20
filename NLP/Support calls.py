# Analyzing Customer Support Calls
!pip install SpeechRecognition
!pip install pydub
!pip install spacy
!python3 -m spacy download en_core_web_sm

# Import required libraries
import pandas as pd

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import speech_recognition as sr
from pydub import AudioSegment

import spacy

# Is the audio compatible for future speech recognition modeling?
recognizer = sr.Recognizer()
with sr.AudioFile('sample_customer_call.wav') as source:
    audio_data = recognizer.record(source)
transcribed_text = recognizer.recognize_google(audio_data)
audio = AudioSegment.from_file("sample_customer_call.wav")
number_channels = audio.channels
frame_rate = audio.frame_rate

# How many calls have a true positive sentiment?

cct = pd.read_csv('customer_call_transcriptions.csv')
sia = SentimentIntensityAnalyzer()
def get_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
        
cct["sentiment_predicted"] = cct["text"].apply(get_sentiment)
true_positive = cct.loc[ (cct["sentiment_label"] == cct["sentiment_predicted"]) & (cct["sentiment_predicted"] == "positive")]
true_positive = len(true_positive)

# What is the most frequently named entity across all of the transcriptions?

nlp = spacy.load("en_core_web_sm")
texts = cct["text"].fillna("").astype(str)
entity_freq = {}
for text in cct["text"].astype(str):
    doc = nlp(text)
    for ent in doc.ents:
        entity_freq[ent_text] = entity_freq.get(ent_text, 0) + 1

most_freq_ent = max(entity_freq, key=entity_freq.get)

#Which call is the most similar to "wrong package delivery"?

query = "wrong package delivery"
query_doc = nlp(query)
highest_score = -1.0
most_similar_text = ""

for text in texts:
    doc = nlp(text)
    score = doc.similarity(query_doc)
    if score > highest_score:
        highest_score = score
        most_similar_text = text
