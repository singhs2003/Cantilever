
from textblob import TextBlob

text = "I love programming in Python. It's so much fun and exciting!"
blob = TextBlob(text)
sentiment_blob = blob.sentiment
print("TextBlob - Polarity:", sentiment_blob.polarity)
print("TextBlob - Subjectivity:", sentiment_blob.subjectivity)

# NLTK VADER Example
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
sentiment_vader = sid.polarity_scores(text)
print("VADER - Sentiment:", sentiment_vader)

# Hugging Face's Transformers Example
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier(text)
print("Transformers - Sentiment:", result)
