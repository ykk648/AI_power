# -- coding: utf-8 --
# @Time : 2022/3/16
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import tensorflow as tf
import tensorflow_hub as hub
import cv2
from cv2box import CVImage, CVVideoLoader
import numpy as np
from tqdm import tqdm
from estimation_3d.body_detector_movenet.movenet_utils import crop_and_resize_tf, init_crop_region, determine_crop_region, \
    draw_prediction_on_image
from model_convert.onnx_model import ONNXModel


# MOVE_NET_MODEL_PATH = 'pretrain_models/digital_human/body_detector_movenet/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
# MOVE_NET_MODEL_PATH = 'pretrain_models/digital_human/body_detector_movenet/lite-model_movenet_singlepose_lightning_3.tflite'
# MOVE_NET_MODEL_PATH = 'pretrain_models/digital_human/body_detector_movenet/lite-model_movenet_singlepose_thunder_3.tflite'


def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inferece on the cropped region.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize_tf(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    # Run model inference.
    keypoints_with_scores = movenet(input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
                                                      crop_region['y_min'] * image_height +
                                                      crop_region['height'] * image_height *
                                                      keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
                                                      crop_region['x_min'] * image_width +
                                                      crop_region['width'] * image_width *
                                                      keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores


class MoveNet:
    def __init__(self, image_height, image_width):
        self.crop_region = init_crop_region(image_height, image_width)

        # TFLite
        # self.interpreter = tf.lite.Interpreter(model_path=MOVE_NET_MODEL_PATH)
        # self.interpreter.allocate_tensors()

        # TF
        # model = hub.load("pretrain_models/digital_human/body_detector_movenet/movenet_singlepose_thunder_4")
        # self.movenet = model.signatures['serving_default']

        # ONNX
        self.movenet = ONNXModel(
            'pretrain_models/digital_human/body_detector_movenet/movenet_singlepose_thunder_4.onnx')

    def forward(self, image):
        image_height, image_width, _ = image.shape

        image = crop_and_resize_tf(
            tf.expand_dims(image, axis=0), self.crop_region, crop_size=(256, 256))

        # TF Lite format expects tensor type of float32.
        input_image = np.array(tf.cast(image, dtype=tf.int32))

        # input_details = self.interpreter.get_input_details()
        # output_details = self.interpreter.get_output_details()
        # self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # self.interpreter.invoke()
        # # Output is a [1, 1, 17, 3] numpy array.
        # keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])

        outputs = self.movenet.forward(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        # keypoints_with_scores = np.array(outputs['output_0'])
        keypoints_with_scores = np.array(outputs[0])

        # print(keypoints_with_scores)

        for idx in range(17):
            keypoints_with_scores[0, 0, idx, 0] = (
                                                          self.crop_region['y_min'] * image_height +
                                                          self.crop_region['height'] * image_height *
                                                          keypoints_with_scores[0, 0, idx, 0]) / image_height
            keypoints_with_scores[0, 0, idx, 1] = (
                                                          self.crop_region['x_min'] * image_width +
                                                          self.crop_region['width'] * image_width *
                                                          keypoints_with_scores[0, 0, idx, 1]) / image_width
        self.crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width)
        return keypoints_with_scores, self.crop_region


if __name__ == '__main__':
    # image_path = '/workspace/84_cluster/ljt/local_test_data/3D/t_pose.jpeg'
    #
    # image_bgr = CVImage(image_path).bgr
    mn = MoveNet(image_height=1920, image_width=1080)

    with CVVideoLoader('/workspace/84_cluster/ljt/local_test_data/3D/video/test_1_1080p.mp4') as cvvl:
        for i in tqdm(range(len(cvvl))):
            _, image_bgr = cvvl.get()

            keypoints_with_scores, crop_region = mn.forward(image_bgr)

            # Visualize the predictions with image.
            display_image = tf.expand_dims(image_bgr, axis=0)
            # display_image = tf.cast(tf.image.resize_with_pad(
            #     display_image, 1080, 1080), dtype=tf.int32)
            output_overlay = draw_prediction_on_image(
                np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores, crop_region=crop_region)

            # CVImage(output_overlay).save(
            #     '/workspace/84_cluster/ljt/local_test_data/3D/video/test_1_1080p_output/{}.jpg'.format(i))

            # kp_list = []
            # for kp in kps:
            #     kp_list.append([kp[1] * 1080, kp[0] * 1920])
            #
            # image_bgr = cv2.drawKeypoints(image_bgr, cv2.KeyPoint_convert(kp_list), None, color=(0, 0, 255), flags=0)
            #
            CVImage(output_overlay).show(1)
