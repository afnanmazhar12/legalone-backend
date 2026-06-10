import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_json("cleaned.json")

# -----------------------------
# BUILD TRAINING PAIRS
# -----------------------------
train_data = []

for row in data:
    text = row["section_title"] + " " + row["section_desc"]
    section = row["section"]

    # simple but effective training format
    train_data.append(
        InputExample(texts=[
            f"pakistan law crime cheating fraud theft {section}",
            text
        ])
    )

print(f"Training samples: {len(train_data)}")

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# DATA LOADER
# -----------------------------
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)

# -----------------------------
# LOSS FUNCTION
# -----------------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

# -----------------------------
# TRAINING
# -----------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=10,
    show_progress_bar=True
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("legal-model")

print("Model saved as 'legal-model'")