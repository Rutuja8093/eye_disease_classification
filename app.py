from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("eye_disease_model.h5")

classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

@app.route("/")
def home():
    return "Eye Disease Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = Image.open(file).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "disease": classes[index],
        "confidence": round(confidence, 4)
    })
