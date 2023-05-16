# -- coding: utf-8 --
# @Time : 2022/2/11
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import time

import cv2
import mediapipe as mp
from cv2box.utils import MyFpsCounter
import numpy as np
from estimation_3d.hand_mesh.minimal_hands.kinematics import xyz_to_delta, MPIIHandJoints, mano_to_mpii
from estimation_3d.hand_mesh.minimal_hands.ik_model import IKModel
from mocap_lib.middleware.VMCApi import VMCApi, LEFT_MPII_HAND_LABELS
from utils.ai_utils import load_pkl

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

IK_UNIT_LENGTH = 0.09473151311686484
mano_ref_xyz = load_pkl('pretrain_models/digital_human/minimal_hands/hand_mesh_model.pkl')['joints']
# convert the kinematic definition to MPII style, and normalize it
mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
mpii_ref_xyz -= mpii_ref_xyz[9:10]
# get bone orientations in the reference pose
mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
mpii_ref_delta = mpii_ref_delta * mpii_ref_length

ikm = IKModel()
# For webcam input:
# cap = cv2.VideoCapture(0)


ip = '192.168.6.5'  # ip address
port = 39542  # port number
vmca = VMCApi(ip, port)

with mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while True:
        cap = cv2.VideoCapture('')
        while cap.isOpened():
            with MyFpsCounter() as mfc:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                try:
                    left_hand = results.left_hand_landmarks.landmark
                    # right_hand = results.right_hand_landmarks.landmark
                except:
                    continue
                left_hand_np = []
                right_hand_np = []
                # print(left_hand[0].x)
                for i in range(21):
                    left_hand_np.append([left_hand[i].x, left_hand[i].y, left_hand[i].z])
                    # right_hand_np.append([right_hand[i].x, right_hand[i].y, right_hand[i].z])
                # print(np.array(left_hand_np).shape)
                labels_list = [LEFT_MPII_HAND_LABELS, ]  # RIGHT_UNI_HAND_LABELS
                for index, hand_np in enumerate([left_hand_np, ]):  # right_hand_np
                    xyz = np.array(hand_np)
                    delta, length = xyz_to_delta(xyz, MPIIHandJoints)
                    delta *= length
                    pack = np.concatenate(
                        [xyz, delta, mpii_ref_xyz, mpii_ref_delta], 0
                    )
                    theta = ikm.forward(pack)
                    # theta_mano = mpii_to_mano(theta)
                    for dummy in range(21):
                        bone = labels_list[index][dummy]
                        vmca.sendosc(bone, theta[index][0], theta[index][1], theta[index][2], theta[index][3])
                    time.sleep(0.01)

                # print(theta_mano)
                # print(len(theta_mano))

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # mp_drawing.draw_landmarks(
                #     image,
                #     results.face_landmarks,
                #     mp_holistic.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_contours_style())

                # mp_drawing.draw_landmarks(
                #     image,
                #     results.pose_landmarks,
                #     mp_holistic.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles
                #         .get_default_pose_landmarks_style())

                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

                # Flip the image horizontally for a selfie-view display.
                # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

                # if cv2.waitKey(1) & 0xFF == 27:
                #     break
        cap.release()
