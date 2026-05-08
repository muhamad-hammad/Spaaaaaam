import os
import pickle
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_NLTK_DATA = os.path.join(BASE_DIR, "nltk_data")

if os.path.isdir(LOCAL_NLTK_DATA):
    nltk.data.path.insert(0, LOCAL_NLTK_DATA)

for resource_path, package in (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)

ps = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))
_PUNCT = set(string.punctuation)


def transformText(text):
    if text is None:
        return ""
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in _STOPWORDS and t not in _PUNCT]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df["label"] = LabelEncoder().fit_transform(df["label"])
    df["Text"] = df["message"].apply(transformText)
    return df


def main():
    csv_path = os.path.join(BASE_DIR, "spam.csv")
    df = load_dataset(csv_path)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["Text"]).toarray()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")

    with open(os.path.join(BASE_DIR, "Vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(BASE_DIR, "MultinomialNB.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("Wrote Vectorizer.pkl and MultinomialNB.pkl")


if __name__ == "__main__":
    main()
