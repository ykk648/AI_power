# -- coding: utf-8 --
# @Time : 2022/8/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
ref https://github.com/zju3dv/EasyMocap/blob/master/easymocap/estimator/mediapipe_wrapper.py
"""

import numpy as np
import cv2
import mediapipe as mp
from cv2box import CVBbox

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.05, MIN_PIXEL=5):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:, -1] > detection_thresh
    if valid.sum() < 3:
        return [0, 0, 100, 100, 0]
    valid_keypoints = keypoints[valid][:, :-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    if bbox_size[0] < MIN_PIXEL or bbox_size[1] < MIN_PIXEL:
        return [0, 0, 100, 100, 0]
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0] / 2,
        center[1] - bbox_size[1] / 2,
        center[0] + bbox_size[0] / 2,
        center[1] + bbox_size[1] / 2,
        keypoints[valid, 2].mean()
    ]
    return bbox


def get_hand_bbox(pose_keypoints, image_shape):
    """

    Args:
        pose_keypoints: 33*3

    Returns:

    """
    left_hand_index = [15, 21, 17, 19]
    right_hand_index = [16, 18, 20, 22]
    left_hand_points = pose_keypoints[left_hand_index]
    right_hand_points = pose_keypoints[right_hand_index]
    # print(left_hand_points)
    left_bbox = CVBbox(None).get_bbox_from_points(left_hand_points[:, :2], image_shape)
    right_bbox = CVBbox(None).get_bbox_from_points(right_hand_points[:, :2], image_shape)

    return left_bbox, right_bbox


NUM_BODY = 33
NUM_HAND = 21
NUM_FACE = 468


class MediapipeDetector:
    def __init__(self, model_type, show, cfg) -> None:

        self.model_type = model_type
        cfg = cfg
        self.show = show

        if model_type == 'holistic':
            model_name = mp_holistic.Holistic
        elif model_type == 'pose':
            model_name = mp.solutions.pose.Pose
        elif model_type == 'face':
            model_name = mp.solutions.face_mesh.FaceMesh
            cfg.pop('model_complexity')
        elif model_type in ['hand', 'handl', 'handr']:
            model_name = mp.solutions.hands.Hands
        else:
            raise NotImplementedError
        self.model = model_name(**cfg)

    @staticmethod
    def to_array(pose, W, H, start=0):
        N = len(pose.landmark) - start
        res = np.zeros((N, 3))
        for i in range(start, len(pose.landmark)):
            res[i - start, 0] = pose.landmark[i].x * W
            res[i - start, 1] = pose.landmark[i].y * H
            res[i - start, 2] = pose.landmark[i].visibility
        return res

    @staticmethod
    def to_array_world(pose, start=0):
        N = len(pose.landmark) - start
        res = np.zeros((N, 4))
        for i in range(start, len(pose.landmark)):
            res[i - start, 0] = pose.landmark[i].x
            res[i - start, 1] = pose.landmark[i].y
            res[i - start, 2] = pose.landmark[i].z
            res[i - start, 3] = pose.landmark[i].visibility
        return res

    def get_body(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((NUM_BODY, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        world_poses = self.to_array_world(pose)
        return poses, bbox_from_keypoints(poses), world_poses

    def get_hand(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((NUM_HAND, 3))
            return bodies, [0, 0, 100, 100, 0.]
        poses = self.to_array(pose, W, H)
        poses[:, 2] = 1.
        return poses, bbox_from_keypoints(poses)

    def get_face(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((NUM_FACE, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        poses[:, 2] = 1.
        return poses, bbox_from_keypoints(poses)

    def process_body(self, data, results, image_width, image_height):
        if self.model_type in ['pose', 'holistic']:
            keypoints, bbox, world_poses = self.get_body(results.pose_landmarks, image_width, image_height)
            data['keypoints'] = keypoints
            data['bbox'] = bbox
            data['keypoints_world'] = world_poses

    def process_hand(self, data, results, image_width, image_height):
        lm = {'Left': None, 'Right': None}
        if self.model_type in ['hand', 'handl', 'handr']:
            if results.multi_hand_landmarks:
                for i in range(len(results.multi_hand_landmarks)):
                    label = results.multi_handedness[i].classification[0].label
                    if lm[label] is not None:
                        continue
                    lm[label] = results.multi_hand_landmarks[i]
            if self.model_type == 'handl':
                lm['Right'] = None
            elif self.model_type == 'handr':
                lm['Left'] = None
        elif self.model_type == 'holistic':
            lm = {'Left': results.left_hand_landmarks, 'Right': results.right_hand_landmarks}
        if self.model_type in ['holistic', 'hand', 'handl', 'handr']:
            handl, bbox_handl = self.get_hand(lm['Left'], image_width, image_height)
            handr, bbox_handr = self.get_hand(lm['Right'], image_width, image_height)

            # flip
            if self.model_type != 'holistic':
                handl[:, 0] = image_width - handl[:, 0] - 1
                handr[:, 0] = image_width - handr[:, 0] - 1
                bbox_handl[0] = image_width - bbox_handl[0] - 1
                bbox_handl[2] = image_width - bbox_handl[2] - 1
                bbox_handr[0] = image_width - bbox_handr[0] - 1
                bbox_handr[2] = image_width - bbox_handr[2] - 1
                bbox_handl[0], bbox_handl[2] = bbox_handl[2], bbox_handl[0]
                bbox_handr[0], bbox_handr[2] = bbox_handr[2], bbox_handl[0]
            if self.model_type in ['hand', 'handl', 'holistic']:
                data['handl2d'] = handl.tolist()
                data['bbox_handl2d'] = bbox_handl
            if self.model_type in ['hand', 'handr', 'holistic']:
                data['handr2d'] = handr.tolist()
                data['bbox_handr2d'] = bbox_handr

    def process_face(self, data, results, image_width, image_height):
        if self.model_type == 'holistic':
            face2d, bbox_face2d = self.get_face(results.face_landmarks, image_width, image_height)
            data['face2d'] = face2d
            data['bbox_face2d'] = bbox_face2d
        elif self.model_type == 'face':
            if results.multi_face_landmarks:
                # only select the first
                face_landmarks = results.multi_face_landmarks[0]
            else:
                face_landmarks = None
            face2d, bbox_face2d = self.get_face(face_landmarks, image_width, image_height)
            data['face2d'] = face2d
            data['bbox_face2d'] = bbox_face2d

    def forward(self, image_, show=False):
        image_height, image_width, _ = image_.shape
        image_ = CVImage(image_).rgb()
        if self.model_type in ['hand', 'handl', 'handr']:
            image_ = cv2.flip(image_, 1)
        image_.flags.writeable = False

        results_ = self.model.process(image_)
        data = {
            'personID': 0,
        }
        self.process_body(data, results_, image_width, image_height)
        self.process_hand(data, results_, image_width, image_height)
        self.process_face(data, results_, image_width, image_height)

        if show:
            show_img = CVImage(CVImage(image_).rgb()).draw_landmarks(data['keypoints'])
            CVImage(show_img).show(0, 'results')

        return data


if __name__ == '__main__':
    from cv2box import CVImage, MyFpsCounter

    image_in = CVImage(
        'resources/for_pose/t_pose_1080p.jpeg').bgr

    md = MediapipeDetector(model_type='pose', show=True, cfg={
        'model_complexity': 2,
        'static_image_mode': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'smooth_landmarks': False,  # only for holistic & pose
        # 'refine_face_landmarks': True,  # only for holistic
        # 'max_num_faces' :1 # only for face
    })
    # with MyFpsCounter() as mfc:
    #     for i in range(100):
    # results = md.forward(image_in, show=True)
    # annots = detector([image_in,image_in,image_in,image_in,])
    # print(results)
    # print(results.keys())
    # print(len(results['keypoints']))
    # print(get_hand_bbox(results['keypoints'], image_in.shape))

    # video tracking test
    from cv2box import CVVideoLoader

    with CVVideoLoader('') as cvvl:
        for _ in range(len(cvvl)):
            _, frame = cvvl.get()
            landmarks = md.forward(frame, show=True)
