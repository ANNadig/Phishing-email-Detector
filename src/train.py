"""
train.py
Script to train models on data/combined.csv and save vectorizer + best model to models/
Generates models/vectorizer.pkl and models/model.pkl
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.preprocess import clean_text
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "combined.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['text_clean'] = df['text'].astype(str).apply(clean_text)

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# vectorizer
vect = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_tfidf = vect.fit_transform(X_train)
X_test_tfidf = vect.transform(X_test)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_preds = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_preds)

# Train SVM (linear)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, svm_preds)

print("Naive Bayes accuracy:", nb_acc)
print("SVM accuracy:", svm_acc)

# choose best model
best_model = nb if nb_acc >= svm_acc else svm
best_name = "naive_bayes" if nb_acc >= svm_acc else "svm_linear"
print("Selected best model:", best_name)

# save vectorizer and model
joblib.dump(vect, MODELS_DIR / "vectorizer.pkl")
joblib.dump(best_model, MODELS_DIR / "model.pkl")

print("Saved vectorizer and model to models/")
print("Classification report for best model:")
best_preds = best_model.predict(X_test_tfidf)
print(classification_report(y_test, best_preds))
