from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
print("Model labels:", model.config.id2label)

pairs = [
    ("The shared embedding model is loaded entirely on the CPU to save massive VRAM.", "The shared embedding model is loaded entirely on the CPU to save massive VRAM."),
    ("The system triggers a cascade fallback.", "This document doesn't mention anything about fallback.")
]

scores = model.predict(pairs)
print("Scores type:", type(scores))
print("Scores shape:", scores.shape)
print("Raw scores:")
for pair, score in zip(pairs, scores):
    print(f"Pair: {pair}")
    print(f"  Score: {score}")
