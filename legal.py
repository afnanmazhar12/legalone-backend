import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_json("cleaned.json")
data["text"] = data["section_title"] + " " + data["section_desc"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
embeddings = model.encode(data["text"].tolist())

while True:
    query = input("\nAsk legal question: ")

    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]

    # TOP 3 results instead of 1 (IMPORTANT FIX)
    top_indices = np.argsort(scores)[-3:][::-1]

    print("\n📌 Top Relevant Laws:\n")

    for i in top_indices:
        print("Section:", data.iloc[i]["section"])
        print("Title:", data.iloc[i]["section_title"])
        print("----")