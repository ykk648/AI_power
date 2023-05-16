# -- coding: utf-8 --
# @Time : 2022/6/16
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
insert body coco 17-kp to openpose 25-kp
"""
import numpy as np

from mocap_lib.skeleton_transfer.keypoints_map import POSE_BODY_25_BODY_PARTS, COCO_JOINTS, COCO_WHOLE_BODY_JOINTS
from mocap_lib.skeleton_transfer.utils import geJointIndexFromName, computeNeck, computeMidHip, json2Keypoints, \
    keypoints2json, load_openpose, create_annot_file
from cv2box.utils import CalDistance, get_path_by_ext
from tqdm import tqdm
from pathlib import Path
from cv2box import CVFile


def insert_coco2openpose(coco_kps, openpose_kps, skeleton_dict):
    """
    Args:
        coco_kps: list[list]
        openpose_kps: list[tuple]
    Returns: new openpose kps
    """
    new_openpose_kps = []
    for i, openposeKeypointName in enumerate(POSE_BODY_25_BODY_PARTS):
        try:
            cocoIndex = geJointIndexFromName(openposeKeypointName, skeleton_dict)
            cocoKeypoint = coco_kps[cocoIndex]
            # if cocoKeypoint[2] == 1 or cocoKeypoint[2] == 2:
            new_keypoint = [cocoKeypoint[0], cocoKeypoint[1], 1.0]
            # else:
            #     new_keypoint = (0.0, 0.0, 0.0)
            new_openpose_kps.append(new_keypoint)
        except ValueError:
            if openposeKeypointName == "Neck":
                new_keypoint = computeNeck(coco_kps)
                new_openpose_kps.append(new_keypoint)
            elif openposeKeypointName == "MidHip":
                new_keypoint = computeMidHip(coco_kps)
                new_openpose_kps.append(new_keypoint)
            elif openposeKeypointName == "Background":
                pass
            else:
                new_keypoint = openpose_kps[i]
                new_openpose_kps.append(new_keypoint)
                # sys.exit()
        # openpose_keypoints.append(new_keypoint)

    return new_openpose_kps


def insert_coco2mp(coco_kp_, left_hand_kps_, right_hand_kps_, json_in_p, json_out_p, skeleton_dict):
    # print(len(coco_kp_))
    # print(left_hand_kps_)
    # print(right_hand_kps_)

    mp_json_data = CVFile(json_in_p).data

    mp_json_data['annots'][0]['keypoints'] = np.array(
        insert_coco2openpose(coco_kp_, mp_json_data['annots'][0]['keypoints'], skeleton_dict)).tolist()
    mp_json_data['annots'][0]['handl2d'] = np.array(left_hand_kps_).tolist()
    mp_json_data['annots'][0]['handr2d'] = np.array(right_hand_kps_).tolist()

    # print(mp_json_data)

    CVFile(json_out_p).json_write(mp_json_data)


def separate_hand_from_body(body_kps, hand_kps_dict_list):
    if len(hand_kps_dict_list) < 1:
        return np.zeros((21, 3)), np.zeros((21, 3))

    left_hand_kps_result = None
    right_hand_kps_result = None
    left_wrist_kp = body_kps[9]
    right_wrist_kp = body_kps[10]
    # print(left_wrist_kp, right_wrist_kp)

    dist_from_left = []
    dist_from_right = []

    if left_wrist_kp[2] < 0.3:
        left_hand_kps_result = np.zeros((21, 3))
    if right_wrist_kp[2] < 0.3:
        right_hand_kps_result = np.zeros((21, 3))

    for i in range(len(hand_kps_dict_list)):
        # hand_kps_dict_list[i]['bbox'][:1]
        dist_from_left.append((CalDistance(left_wrist_kp[:2], hand_kps_dict_list[i]['bbox'][:2]).euc() + CalDistance(
            left_wrist_kp[:2], hand_kps_dict_list[i]['bbox'][2:4]).euc()) / 2)
        dist_from_right.append((CalDistance(right_wrist_kp[:2], hand_kps_dict_list[i]['bbox'][:2]).euc() + CalDistance(
            right_wrist_kp[:2], hand_kps_dict_list[i]['bbox'][2:4]).euc()) / 2)

    if np.argmin(np.array(dist_from_left)) == np.argmin(np.array(dist_from_right)):
        if len(dist_from_left) == len(dist_from_right) == 1:
            if min(dist_from_left) <= min(dist_from_right):
                right_hand_kps_result = np.zeros((21, 3))
            else:
                left_hand_kps_result = np.zeros((21, 3))
        if min(dist_from_left) <= min(dist_from_right):
            dist_from_right[np.argmin(np.array(dist_from_right))] = 9999
    # print(dist_from_left, dist_from_right)
    if left_hand_kps_result is None:
        left_hand_kps_result = hand_kps_dict_list[np.argmin(np.array(dist_from_left))]['keypoints']
    if right_hand_kps_result is None:
        right_hand_kps_result = hand_kps_dict_list[np.argmin(np.array(dist_from_right))]['keypoints']

    return left_hand_kps_result, right_hand_kps_result


if __name__ == '__main__':
    # openpose_dir = ''
    base_dir = ''
    cam_name_list = []

    for cam_name in cam_name_list:
        mp_dir = '{}/annots_mp/{}'.format(base_dir,cam_name)
        body_coco_pkl = '{}/{}_body_coco_kp.pkl'.format(base_dir,cam_name)
        whole_body_coco_pkl = '{}/{}_whole_body_kp.pkl'.format(base_dir,cam_name)
        hand_pkl = '{}/{}_hand_kp.pkl'.format(base_dir,cam_name)
        output_mp_dir = '{}/mp_new_whole_body/{}'.format(base_dir,cam_name)

        Path(output_mp_dir).mkdir(parents=True, exist_ok=True)

        coco_kps = CVFile(body_coco_pkl).data
        hand_kps_dict = CVFile(hand_pkl).data
        whole_body_dict = CVFile(whole_body_coco_pkl).data

        for index, json_p in enumerate(tqdm(get_path_by_ext(mp_dir, ext_list=['.json']))):
            json_p = str(json_p)

            # whole body
            left_hand_kps = whole_body_dict[index][0]['keypoints'][91:112]
            right_hand_kps = whole_body_dict[index][0]['keypoints'][112:133]
            insert_coco2mp(whole_body_dict[index][0]['keypoints'][:23], left_hand_kps, right_hand_kps, json_p,
                           json_p.replace(mp_dir, output_mp_dir),
                           COCO_WHOLE_BODY_JOINTS)

            # body hand separate
            # left_hand_kps, right_hand_kps = separate_hand_from_body(coco_kps[index], hand_kps_dict[index])
            # insert_coco2mp(coco_kps[index], left_hand_kps, right_hand_kps, json_p, json_p.replace(mp_dir, output_mp_dir), COCO_JOINTS)
