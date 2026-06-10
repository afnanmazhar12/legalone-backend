import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_json("cleaned.json")

# Build training text
data["text"] = data["section_title"] + " " + data["section_desc"]
data["label"] = data["section"]

# Load embedding model (brain)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text → vectors
X = embedder.encode(data["text"].tolist())

# Labels
y = data["label"]

# Train model (REAL ML TRAINING)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model + embedder
joblib.dump(model, "legal_model.pkl")
joblib.dump(embedder, "embedder.pkl")

print("Training complete! Model saved.")