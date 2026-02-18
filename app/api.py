from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import sklearn

app = Flask(__name__)

# ----------------------------
# Load model ONCE at startup
# ----------------------------
MODEL_DIR = "models"

vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
model_path = os.path.join(MODEL_DIR, "logreg_baseline.joblib")

# Fail fast if files missing (better logs in CI)
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Missing vectorizer at: {vectorizer_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing model at: {model_path}")

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

LABELS = {0: "negative", 1: "positive"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_type": "TF-IDF + LogisticRegression",
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Lightweight Sentiment API running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict  { text: ... }"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field"}), 400

    text = data["text"]

    vectorized_text = vectorizer.transform([text])

    prediction = int(model.predict(vectorized_text)[0])
    probabilities = model.predict_proba(vectorized_text)[0]

    confidence = float(np.max(probabilities))
    sentiment = LABELS[prediction]

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
