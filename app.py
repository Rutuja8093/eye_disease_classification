from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("eye_disease_model.h5")

CLASS_NAMES = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            prediction = CLASS_NAMES[np.argmax(preds)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
