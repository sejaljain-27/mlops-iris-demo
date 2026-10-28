from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("iris_model.pkl")
model = joblib.load(MODEL_PATH)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = [float(payload[name]) for name in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Send numeric sepal_length, sepal_width, petal_length, and petal_width"
        }), 400

    prediction = model.predict(pd.DataFrame([values], columns=FEATURES))[0]
    # PyCaret's Iris dataset predicts species labels; the starter sklearn model
    # predicts numeric class IDs. Support both artifacts.
    species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    try:
        label = species.get(int(prediction), str(prediction))
    except (TypeError, ValueError):
        label = str(prediction)
    return jsonify({"prediction": label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
