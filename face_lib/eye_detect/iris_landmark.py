#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ref https://github.com/Kazuhito00/iris-detection-using-py-mediapipe
"""
import cv2 as cv
import numpy as np
import tensorflow as tf


class IrisLandmark(object):
    def __init__(
        self,
        model_path='iris_landmark/iris_landmark.tflite',
        num_threads=1,
    ):
        self._interpreter = tf.lite.Interpreter(model_path=model_path,
                                                num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def __call__(
        self,
        image,
    ):
        input_shape = self._input_details[0]['shape']

        # 正規化・リサイズ
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = img / 255.0
        img_resized = tf.image.resize(img, [input_shape[1], input_shape[2]],
                                      method='bicubic',
                                      preserve_aspect_ratio=False)
        img_input = img_resized.numpy()
        img_input = (img_input - 0.5) / 0.5

        reshape_img = img_input.reshape(1, input_shape[1], input_shape[2],
                                        input_shape[3])
        tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

        # 推論実行
        input_details_tensor_index = self._input_details[0]['index']
        self._interpreter.set_tensor(input_details_tensor_index, tensor)
        self._interpreter.invoke()

        # 推論結果取得
        output_details_tensor_index0 = self._output_details[0]['index']
        output_details_tensor_index1 = self._output_details[1]['index']
        eye_contour = self._interpreter.get_tensor(
            output_details_tensor_index0)
        iris = self._interpreter.get_tensor(output_details_tensor_index1)

        return np.squeeze(eye_contour), np.squeeze(iris)

    def get_input_shape(self):
        input_shape = self._input_details[0]['shape']
        return [input_shape[1], input_shape[2]]