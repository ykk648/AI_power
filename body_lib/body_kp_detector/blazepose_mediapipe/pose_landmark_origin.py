# -- coding: utf-8 --
# @Time : 2022/7/21
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from cv2box import CVImage, MyFpsCounter

from apstone import ONNXModel
from body_lib.body_kp_detector.blazepose_mediapipe.blaze_utils import postprocess, denormalize_landmarks, detection2roi, \
    extract_roi
from body_lib.body_kp_detector.blazepose_mediapipe.body_bbox_detector import BodyDetector

# input 1*256*256*3 output , 1*1 , , ,
LITE_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_lite.onnx'
FULL_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_full.onnx'
HEAVY_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_heavy.onnx'


class LandmarkDetectorOrigin:
    def __init__(self, model_complexity=0, provider='gpu'):
        self.bd = BodyDetector(provider=provider)

        model_path_list = [LITE_MODEL, FULL_MODEL, HEAVY_MODEL]
        self.model = ONNXModel(model_path_list[model_complexity], provider=provider)

    def forward(self, image_in_, show=False):
        """

        Args:
            image_in_:

        Returns:
            landmarks: 33*4

        """
        filtered_detections = self.bd.forward(image_in_)
        if show:
            print(filtered_detections)
        if filtered_detections.shape == (0, 13):
            return np.zeros((33, 4))
        elif len(filtered_detections) > 1:
            filtered_detections = filtered_detections[0].reshape(1, 13)

        xc, yc, scale, theta = detection2roi(filtered_detections, detection2roi_method='alignment')
        img, affine, box = extract_roi(CVImage(image_in_).bgr, xc, yc, theta, scale)
        if show:
            CVImage(img[0]).show(0, 'img_in')
        normalized_landmarks, f, _, _, _ = self.model.forward(img.astype(np.float32))
        normalized_landmarks = postprocess(normalized_landmarks)
        landmarks_ = denormalize_landmarks(normalized_landmarks, affine)[0]

        # CVImage(img[0].cpu().numpy().transpose(2, 1, 0)).show()
        # print(normalized_landmarks)
        if show:
            show_img = CVImage(image_in_).draw_landmarks(landmarks_)
            CVImage(show_img).show(0, 'results')

        return landmarks_


if __name__ == '__main__':
    # image_path = 'resource/for_pose/t_pose_1500x.jpeg'
    image_path = 'resource/for_pose/girl_640x480.jpg'
    # image_path = 'resource/yoga1.webp'
    # image_path = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/multi_view_human/0707_women_dance/0/00000035.jpg'
    # image_path = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/multi_view_human/0707_women_dance/0/resize.jpg'
    image_in = CVImage(image_path).bgr

    """
    model 0
    gpu 70fps trt 133-196fps trt16 235-278fps t_pose_1500x
    gpu  trt 221fps trt16 269fps t_pose_1080p
    model 1 282fps
    """
    ld = LandmarkDetectorOrigin(model_complexity=2, provider='gpu')

    # landmarks = ld.forward(image_in, show=True)
    # print(landmarks)

    # with MyFpsCounter('model forward 10 times fps: ') as mfc:
    #     for i in range(10):
    #         filtered_detections = ld.forward(image_in)

    # video tracking test
    from cv2box import CVVideoLoader

    with CVVideoLoader('/workspace/84_cluster/mnt/cv_data_ljt/dataset/multi_view_human/0802/hand1/videos/268.mp4') as cvvl:
        for _ in range(len(cvvl)):
            _, frame = cvvl.get()
            landmarks = ld.forward(frame, show=True)
