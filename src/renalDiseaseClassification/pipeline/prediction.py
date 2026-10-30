import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "training" / "model.h5"

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run the training pipeline first."
            )

        model = load_model(MODEL_PATH)

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = tf.keras.applications.vgg16.preprocess_input(
            np.expand_dims(test_image, axis=0)
        )
        result = np.argmax(model.predict(test_image, verbose=0), axis=1)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]