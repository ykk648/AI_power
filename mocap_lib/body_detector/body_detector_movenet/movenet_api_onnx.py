# -- coding: utf-8 --
# @Time : 2022/3/16
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, CVVideoLoader
import numpy as np
from tqdm import tqdm
from estimation_3d.body_detector.body_detector_movenet.movenet_utils import crop_and_resize, init_crop_region, determine_crop_region, \
    draw_prediction_on_image
from model_lib.onnx_model import ONNXModel


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

        image = crop_and_resize(
            np.expand_dims(image, axis=0), self.crop_region, crop_size=(256, 256))

        input_image = np.array(image, dtype=np.int32)

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

    with CVVideoLoader('/workspace/84_cluster/ljt/local_test_data/3D/video/0317/2022-03-17-170847.webm') as cvvl:
        for i in tqdm(range(len(cvvl))):
            _, image_bgr = cvvl.get()

            keypoints_with_scores, crop_region = mn.forward(image_bgr)

            # Visualize the predictions with image.
            display_image = np.expand_dims(image_bgr, axis=0)
            # display_image = tf.cast(tf.image.resize_with_pad(
            #     display_image, 1080, 1080), dtype=tf.int32)
            output_overlay = draw_prediction_on_image(
                np.squeeze(display_image, axis=0), keypoints_with_scores, crop_region=crop_region)

            # CVImage(output_overlay).save(
            #     '/workspace/84_cluster/ljt/local_test_data/3D/video/test_1_1080p_output/{}.jpg'.format(i))

            # kp_list = []
            # for kp in kps:
            #     kp_list.append([kp[1] * 1080, kp[0] * 1920])
            #
            # image_bgr = cv2.drawKeypoints(image_bgr, cv2.KeyPoint_convert(kp_list), None, color=(0, 0, 255), flags=0)
            #
            CVImage(output_overlay).show(1)
