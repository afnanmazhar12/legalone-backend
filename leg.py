from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_json("cleaned.json")
data["text"] = data["section_title"] + " " + data["section_desc"]

# -----------------------------
# MODELS
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# PRECOMPUTE EMBEDDINGS (RAG MEMORY)
embeddings = embedder.encode(data["text"].tolist(), show_progress_bar=True)

# -----------------------------
# GROQ API KEY
# -----------------------------
GROQ_API_KEY ="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"

# -----------------------------
# REQUEST FORMAT
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# LLM FUNCTION
# -----------------------------
def ask_llm(context, query):

    prompt = f"""
You are a legal expert assistant.

Use ONLY the provided law context.

LAW:
{context}

QUESTION:
{query}

Explain:
- Meaning in simple English
- Section/Article number
- Punishment
- Imprisonment
- Key points
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    result = response.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"]

    return f"API ERROR: {result}"

# -----------------------------
# RAG PIPELINE (FULL OLD + NEW DEBUG)
# -----------------------------
def run_rag(query):

    print("\n==============================")
    print("🔍 NEW QUERY:", query)
    print("==============================\n")

    # STEP 1: RETRIEVAL
    query_vec = embedder.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]

    top_candidates = np.argsort(scores)[-30:][::-1]

    # STEP 2: RERANKING
    pairs = [
        (query + " legal punishment section offence", data.iloc[i]["text"])
        for i in top_candidates
    ]

    rerank_scores = reranker.predict(pairs)
    final_order = np.argsort(rerank_scores)[-3:][::-1]

    # STEP 3: DEBUG OUTPUT (OLD FEATURE RESTORED)
    print("\n📌 TOP 3 RELEVANT LAWS (RAG RESULT)\n")

    retrieved_laws = []
    context_list = []

    for idx in final_order:
        i = top_candidates[idx]

        section = data.iloc[i]["section"]
        title = data.iloc[i]["section_title"]

        print(f"Section: {section}")
        print(f"Title: {title}")
        print("----")

        retrieved_laws.append({
            "section": section,
            "title": title
        })

        context_list.append(
            f"Section {section}: {data.iloc[i]['text']}"
        )

    context = "\n\n".join(context_list)

    # STEP 4: LLM CALL
    print("\n🧠 Generating Explanation...\n")

    answer = ask_llm(context, query)

    print("\n📖 FINAL ANSWER:\n")
    print(answer)

    print("\n==============================\n")

    # RETURN EVERYTHING TO FRONTEND
    return {
        "answer": answer,
        "retrieved": retrieved_laws,
        "context": context
    }

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/ask")
def ask(query: Query):

    return run_rag(query.question)

# -----------------------------
# OPTIONAL: LOCAL TEST MODE (OLD STYLE FEEL)
# -----------------------------
if __name__ == "__main__":
    print("✅ LEGAL AI SYSTEM STARTED (FASTAPI + RAG + GROQ)\n")

    while True:
        q = input("Ask legal question: ")
        run_rag(q)