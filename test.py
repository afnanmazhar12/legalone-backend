import joblib

model = joblib.load("legal_model.pkl")
embedder = joblib.load("embedder.pkl")

text = input("Enter legal question: ")

vec = embedder.encode([text])

pred = model.predict(vec)

print("Predicted Section:", pred[0])