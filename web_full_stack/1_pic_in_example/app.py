from flask import Flask, request, jsonify, render_template
import io
import cv2
import numpy as np
import base64
from gevent import pywsgi
from cv2box import CVImage


def your_func(*args):
    return args[0]


app = Flask(__name__)
# 上传大小限制在10M
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = file.read()
    image = CVImage(img_bytes, image_format='bytes').bgr

    success_code, latent, crop_face = your_func(image)

    if success_code == 1:
        return jsonify({'latent': '', 'crop_face': '', 'status': 'face detect fail !'})
    elif success_code == 2:
        return jsonify({'latent': '', 'crop_face': '', 'status': 'detect multi face !'})
    elif success_code == 3:
        return jsonify({'latent': '', 'crop_face': '', 'status': 'file is not image !'})
    else:
        # cv2.imshow('2', crop_face)
        # cv2.waitKey(999)
        return jsonify({'latent': latent.tolist(), 'crop_face': CVImage(crop_face).base64, 'status': 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)

    # server = pywsgi.WSGIServer(('0.0.0.0', 5005), app)
    # server.serve_forever()
