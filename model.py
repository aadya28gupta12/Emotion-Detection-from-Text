import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("data/emotions.csv")

# Build pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(df['text'], df['emotion'])
joblib.dump(model, "model/emotion_model.pkl")
