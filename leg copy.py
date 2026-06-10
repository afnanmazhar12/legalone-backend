import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

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

# -----------------------------
# PRECOMPUTE EMBEDDINGS
# -----------------------------
embeddings = embedder.encode(data["text"].tolist(), show_progress_bar=True)

# -----------------------------
# GROQ API KEY
# -----------------------------
GROQ_API_KEY = "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"

# -----------------------------
# LLM CALL (SAFE + FIXED)
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

- Meaning in semi detailed english 
- article no. 
- Punishment
- Imprisonment
- if no refrence is found in law discription then say there is no refrence found in law discription
- Clearly format each section for better readability

- Key points
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",   # ✅ FIXED MODEL
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    data = response.json()

    # -----------------------------
    # DEBUG (REMOVE LATER IF YOU WANT)
    # -----------------------------
    # print("\nDEBUG:", data)

    # -----------------------------
    # SAFE HANDLING
    # -----------------------------
    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    if "error" in data:
        return f"API ERROR: {data['error']['message']}"

    return f"UNKNOWN RESPONSE: {data}"

# -----------------------------
# MAIN LOOP (RAG PIPELINE)
# -----------------------------
print("✅ Legal AI System Running (RAG + Groq)\n")

while True:
    query = input("\nAsk legal question: ")

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

    # STEP 3: BUILD CONTEXT
    context_list = []

    print("\n📌 Top Relevant Laws:\n")

    for idx in final_order:
        i = top_candidates[idx]

        print("Section:", data.iloc[i]["section"])
        print("Title:", data.iloc[i]["section_title"])
        print("----")

        context_list.append(
            f"Section {data.iloc[i]['section']}: {data.iloc[i]['text']}"
        )

    context = "\n\n".join(context_list)

    # STEP 4: LLM RESPONSE
    print("\n🧠 Generating Explanation...\n")

    answer = ask_llm(context, query)

    print("\n📖 ANSWER:\n")
    print(answer)




print("data:",data.head());
print("data.isnull().sum():",data.isnull().sum());