import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "models/trained"


def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # CPU inference
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

    pred_id = int(torch.argmax(logits, dim=-1).item())
    sentiment = "positive" if pred_id == 1 else "negative"
    confidence = probs[pred_id]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probs": {"negative": float(probs[0]), "positive": float(probs[1])},
    }


if __name__ == "__main__":
    tests = [
        "This movie was amazing! I loved it.",
        "Worst movie ever. Total waste of time.",
        "It was okay, not great but not terrible.",
    ]

    for t in tests:
        print(predict(t))
