import os
import numpy as np
from pathlib import Path

import keras
from flask import Flask, request
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)


THRESH = 0.51
IMG_SIZE = (224, 224)
DIRECTORY_ROOT = os.path.abspath(Path(os.getcwd()))


def preprocess_image(img):
    """Helper function to process the image so our model can make a prediction

    Args:
        img (np array): Array of the image data

    Returns:
        np array: The processed image
    """
    img = load_img(img, grayscale=True, target_size=IMG_SIZE)
    img = img_to_array(img)
    proc_img = img.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 1))
    proc_img = np.repeat(proc_img, 3, axis=3)
    return proc_img


def load_model():
    """Helper function to load the model

    Returns: ML Model
    """
    model_path = DIRECTORY_ROOT + "/server/my_model.json"
    weight_path = DIRECTORY_ROOT + "/server/best.model.hdf5"
    with open(model_path, 'r') as json_file:
        model_file = json_file.read()
        model = keras.models.model_from_json(model_file)
        model.load_weights(weight_path)
    return model


def predict_image(model, img):
    """Helper function to make a prediction

    Args:
        model (ML Model): A ML Model
        img (np array): np array of the image data

    Returns:
        str: A Prediction
    """
    if model.predict(img) > THRESH:
        return 'Pneumonia'
    else:
        return 'No Pneumonia'


def createPrediction(data):
    """Helper function to create a prediction

    Args:
        data (form-data): Image data sent via POST-Request

    Returns:
        object: Prediction
    """
    if data is None:
        return {
            "msg": "Invalid image"
        }
    else:
        img_path = DIRECTORY_ROOT + "/server/" + data.filename
        data.save(img_path)
        img_proc = preprocess_image(img_path)
        model = load_model()
        pred = predict_image(model, img_proc)
        os.remove(img_path)
        return {
            "prediction": pred
        }


@ app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_image = request.files['file']
        prediction = createPrediction(uploaded_image)
        return prediction
    return {
        "msg": "Invalid Request"
    }


# Run app if directly called
if __name__ == "__main__":
    app.run(host='0.0.0.0')
