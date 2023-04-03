from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
import spacy
from lime.lime_text import LimeTextExplainer

app = FastAPI()


class NewsArticle(BaseModel):
    text: str


class Prediction(BaseModel):
    prediction: str
    confidence: float
    explanation: dict


# Load pre-trained models and resources
nlp = spacy.load("en_core_web_sm")
logistic_regression_model = joblib.load("logistic_regression_model.joblib")

# Define a list of news sources to scrape
news_sources = [
    {"name": "CNN", "url": "https://www.cnn.com/", "selector": ".cd__headline-text"},
    {"name": "Fox News", "url": "https://www.foxnews.com/", "selector": ".title"},
    {"name": "BBC News", "url": "https://www.bbc.com/news",
        "selector": ".gs-c-promo-heading"},
]


def scrape_news_articles():
    news_articles = []
    for source in news_sources:
        response = requests.get(source["url"])
        soup = BeautifulSoup(response.content, "html.parser")
        headlines = soup.select(source["selector"])
        for headline in headlines:
            text = headline.get_text()
            if text:
                news_articles.append({"source": source["name"], "text": text})
    return news_articles


def explain_prediction(text):
    # Preprocess text using spaCy
    doc = nlp(text)
    # Extract features from the news article
    word_freq = {token.text: token.count for token in doc}
    sentiment = doc.sentiment.polarity
    named_entities = [ent.text for ent in doc.ents]
    # ...
    # Combine features into a feature vector and scale them to the same range as the training data
    feature_vector = [word_freq["fake"],
                      word_freq["real"], sentiment, len(named_entities)]
    # ...
    # Feed preprocessed news article into the trained model to obtain prediction and confidence score
    prediction = logistic_regression_model.predict(
        np.array(feature_vector).reshape(1, -1))[0]
    confidence = logistic_regression_model.predict_proba(
        np.array(feature_vector).reshape(1, -1))[0][prediction]
    # Generate an explanation for the prediction using LIME
    explainer = LimeTextExplainer(class_names=["fake", "real"])
    exp = explainer.explain_instance(
        text, logistic_regression_model.predict_proba, num_features=len(feature_vector))
    explanation = {}
    for i in range(len(exp.as_list())):
        explanation[exp.as_list()[i][0]] = exp.as_list()[i][1]
    return prediction, confidence, explanation


@app.get("/scrape")
def scrape_and_predict():
    # Scrape news articles from various sources
    news_articles = scrape_news_articles()

    # Predict whether each news article is real or fake
    predictions = []
    for article in news_articles:
        prediction, confidence, explanation = explain_prediction(
            article["text"])
        predictions.append({"source": article["source"], "text": article["text"], "prediction": "real" if prediction ==
                           1 else "fake", "confidence": confidence, "explanation": explanation})

    # Return predictions as a JSON response
    return predictions
