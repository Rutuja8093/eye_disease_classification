import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model("eye_disease_model.h5")

CLASS_NAMES = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = image.load_img(file, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            prediction = CLASS_NAMES[np.argmax(preds)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
