from flask import Flask, request, jsonify, render_template
import io
import cv2
import numpy as np
import base64
from gevent import pywsgi
from cv2box import CVImage


def your_func(*args):
    return 0, args[0]


app = Flask(__name__)
# 上传大小限制在10M
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.getlist('image')

    image0 = CVImage(file[0].read(), image_format='bytes').bgr
    image1 = CVImage(file[1].read(), image_format='bytes').bgr
    # print(image0)
    # print(image1)

    success_code, output_face = your_func(image0, image1)

    if success_code == 1:
        return jsonify({'output_face': '', 'status': 'face detect fail !'})
    elif success_code == 2:
        return jsonify({'output_face': '', 'status': 'detect multi face !'})
    elif success_code == 3:
        return jsonify({'output_face': '', 'status': 'file is not image !'})
    else:
        # cv2.imshow('2', crop_face)
        # cv2.waitKey(999)
        return jsonify({'output_face': CVImage(output_face).base64, 'status': 'success'})


if __name__ == '__main__':
    # for test
    app.run(host='0.0.0.0', port=5005)

    # # for product
    # server = pywsgi.WSGIServer(('0.0.0.0', 5005), app)
    # server.serve_forever()
