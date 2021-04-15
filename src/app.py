
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from flask import Flask, request
import numpy as np
import os

app = Flask(__name__)

@app.route('/inference', methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)
    
    # img_arr should be in format (x, x, 3)- rgb
    img_arr = tf.convert_to_tensor(img_arr, dtype=np.uint8)
    img_arr = img_arr[tf.newaxis, ...] # prepend so is (1, x, x, 3)
    unet = tf.keras.models.load_model('unet.h5')
    mask = unet.predict(img_arr)
    return {"prediction": mask.tolist()}


if __name__ == '__main__':
    app.run(debug=True)

