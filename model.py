# Importing the Libraries
import pandas as pd
import numpy as np
import re, string
import swifter
from nltk.corpus import stopwords
stop_words = stopwords.words("english")  # ✅ FIXED: changed from set() to list
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load Data
train_df = pd.read_csv("/Users/adithya/Desktop/NLp_Project/data/raw-data/train.csv")
test_data = pd.read_csv("/Users/adithya/Desktop/NLp_Project/data/raw-data/test.csv")
test_labels = pd.read_csv("/Users/adithya/Desktop/NLp_Project/data/raw-data/test_labels.csv")

# Merge test data with labels
test_df = pd.merge(test_data, test_labels, on="id")
test_df = test_df[(test_df['toxic'] != -1)]
test_df.reset_index(drop=True, inplace=True)

# Cleaning Functions
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

def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# Apply cleaning
train_df["comment_text"] = train_df["comment_text"].swifter.apply(clean_text).swifter.apply(word_lemmatizer)
test_df["comment_text"] = test_df["comment_text"].swifter.apply(clean_text).swifter.apply(word_lemmatizer)

# Split train/val
train, val = train_test_split(train_df, test_size=0.2, random_state=42)
X_train = train.comment_text
X_val = val.comment_text
X_test = test_df.comment_text
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# TF-IDF Vectorizers
word_tfidf = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 1), sublinear_tf=True,
    strip_accents="unicode", analyzer="word", stop_words=stop_words,
    token_pattern=r"\w{1,}"
)
char_tfidf = TfidfVectorizer(
    max_features=30000, ngram_range=(2, 6), sublinear_tf=True,
    strip_accents="unicode", analyzer="char"  # ❗ No stop_words here
)

# Fit and transform
word_tfidf.fit(train_df.comment_text)
char_tfidf.fit(train_df.comment_text)

# Save vectorizers
joblib.dump(word_tfidf, "word_tfidf_vectorizer.pkl")
joblib.dump(char_tfidf, "char_tfidf_vectorizer.pkl")

# Transform
train_features = hstack([
    word_tfidf.transform(X_train),
    char_tfidf.transform(X_train)
])
val_features = hstack([
    word_tfidf.transform(X_val),
    char_tfidf.transform(X_val)
])
test_features = hstack([
    word_tfidf.transform(X_test),
    char_tfidf.transform(X_test)
])

# Train and Save Models
lr_model = OneVsRestClassifier(LogisticRegression(solver="saga"))

for label in labels:
    print(f"Training for {label}...")
    lr_model.fit(train_features, train[label])
    joblib.dump(lr_model, f"logistic_regression_{label}.pkl")

print("✅ All models and vectorizers saved!")
