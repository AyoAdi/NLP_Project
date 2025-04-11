# Importing the libraries
import numpy as np
import joblib
import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub("<.*?>", " ", text)
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("\n", " ", text)
    return " ".join(text.split())

# Lemmatization function
def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# Load vectorizers
word_tfidf = joblib.load("models/word_tfidf_vectorizer.pkl")
char_tfidf = joblib.load("models/char_tfidf_vectorizer.pkl")

# Load models
lr_toxic = joblib.load("models/logistic_regression_toxic.pkl")
lr_severe = joblib.load("models/logistic_regression_severe_toxic.pkl")
lr_obscene = joblib.load("models/logistic_regression_obscene.pkl")
lr_threat = joblib.load("models/logistic_regression_threat.pkl")
lr_insult = joblib.load("models/logistic_regression_insult.pkl")
lr_identity = joblib.load("models/logistic_regression_identity_hate.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    text_input = request.form["text"]
    text_cleaned = clean_text(text_input)
    lemmatized_text = word_lemmatizer(text_cleaned)
    input_data = [lemmatized_text]

    # Vectorize input
    word_features = word_tfidf.transform(input_data)
    char_features = char_tfidf.transform(input_data)
    combined_features = hstack([word_features, char_features])

    # Predict probabilities
    predictions = {
        "Toxic": lr_toxic.predict_proba(combined_features)[0][1],
        "Severe Toxic": lr_severe.predict_proba(combined_features)[0][1],
        "Obscene": lr_obscene.predict_proba(combined_features)[0][1],
        "Threat": lr_threat.predict_proba(combined_features)[0][1],
        "Insult": lr_insult.predict_proba(combined_features)[0][1],
        "Identity Hate": lr_identity.predict_proba(combined_features)[0][1]
    }

    # Format predictions as percentages
    predictions = {k: f"{round(v * 100, 2)}%" for k, v in predictions.items()}

    return render_template("index.html",
                           comment_text=f"Your input comment: {text_input}",
                           pred_toxic=f"Toxic: {predictions['Toxic']}",
                           pred_severe=f"Severe Toxic: {predictions['Severe Toxic']}",
                           pred_obscene=f"Obscene: {predictions['Obscene']}",
                           pred_threat=f"Threat: {predictions['Threat']}",
                           pred_insult=f"Insult: {predictions['Insult']}",
                           pred_identity=f"Identity Hate: {predictions['Identity Hate']}"
                          )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
