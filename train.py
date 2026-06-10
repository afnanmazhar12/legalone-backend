import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_json("cleaned.json")

print("Loaded:", type(data))

# -----------------------------
# CONVERT SAFE FORMAT
# -----------------------------
records = data.to_dict(orient="records")

train_data = []

for row in records:   # ✅ SAFE FIX
    text = str(row["section_title"]) + " " + str(row["section_desc"])
    section = str(row["section"])

    train_data.append(
        InputExample(texts=[
            f"pakistan law cheating fraud theft section {section}",
            text
        ])
    )

print("Samples:", len(train_data))

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)

train_loss = losses.MultipleNegativesRankingLoss(model)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=10,
    show_progress_bar=True
)

# -----------------------------
# SAVE
# -----------------------------
model.save("legal-model")

print("DONE ✅")