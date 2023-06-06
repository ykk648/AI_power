# -- coding: utf-8 --
# @Time : 2022/9/5
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
using human keypoints instead of charuco boards to do calibration
"""
import numpy as np
from cv2box import CVFile, CVCamera
import cv2
import aniposelib

human_body_height = 1.82 - 0.25
side_bias = 1 / 15 * human_body_height

used_kps_3d = np.array(
    [[0, -human_body_height, 0], [-side_bias, -1 / 2 * human_body_height, 0],
     [side_bias, -1 / 2 * human_body_height, 0], [-side_bias, -1 / 4 * human_body_height, 0],
     [side_bias, -1 / 4 * human_body_height, 0], [-side_bias, 0, 0], [side_bias, 0, 0], ])

frame = 120
all_kps = []
used_kps_index = [1, 9, 12, 10, 13, 11, 14]
cameras = []
cvc = CVCamera(
    multical_pkl_path='./0809cal/front_4_0809_window_1080.pkl')
for camera_name in ['268', '617', '728', '886']:
    kp_p = './0906_pm/stand1/{}_2dkp.pkl'.format(camera_name)
    kps = CVFile(kp_p).data[frame][used_kps_index, :2]

    kp_p_2 = './0906_pm/walk2/{}_2dkp.pkl'.format(
        camera_name)
    all_kps.append(CVFile(kp_p_2).data)

    results = cv2.solvePnP(used_kps_3d, kps, cvc.intri_matrix()[camera_name], cvc.dist()[camera_name])

    camera = aniposelib.cameras.Camera(name=camera_name,
                                       size=cvc.image_size()[camera_name],
                                       matrix=cvc.intri_matrix()[camera_name],
                                       rvec=results[1],
                                       tvec=results[2],
                                       dist=cvc.dist()[camera_name])
    cameras.append(camera)

cvc_empty = CVCamera()
cvc_empty.camera_group = aniposelib.cameras.CameraGroup(cameras)
final_camera_group = cvc_empty.bundle_adjust_iter(np.array(all_kps))
CVFile('./0906_pm/cgroup_from_human.pkl').pickle_write(final_camera_group)
