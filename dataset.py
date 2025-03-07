import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("resume_dataset.csv")  

# Rename columns
df = df.rename(columns={"Category": "job_category", "Resume": "resume_text"})

# Encode job categories as numbers
job_categories = df["job_category"].unique()
category_mapping = {category: idx for idx, category in enumerate(job_categories)}
df["job_category_encoded"] = df["job_category"].map(category_mapping)

# Save category mapping
with open("category_mapping.json", "w") as f:
    json.dump(category_mapping, f)

print("✅ Saved category_mapping.json")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["resume_text"], df["job_category_encoded"], test_size=0.2, random_state=42)

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Evaluate the Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to a file
with open("model_accuracy.txt", "w") as f:
    f.write(str(accuracy))

print("🎯 Model Accuracy:", accuracy)
print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=job_categories))
