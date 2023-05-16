# -- coding: utf-8 --
# @Time : 2022/6/15
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import json
from mocap_lib.skeleton_transfer.keypoints_map import POSE_BODY_25_BODY_PARTS, COCO_JOINTS
import os
from cv2box import CVFile
import cv2


def dummyKeypoints(howMany):
    dummy = []
    for i in range(howMany):
        new_keypoint = (0, 0, 0.0)
        dummy.append(new_keypoint)
    dummy = np.array(dummy).flatten()
    return dummy


def create_annot_file(annotname, imgname):
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    height, width = img.shape[0], img.shape[1]
    imgnamesep = imgname.split(os.sep)
    filename = os.sep.join(imgnamesep[imgnamesep.index('images'):])
    annot = {
        'filename': filename,
        'height': height,
        'width': width,
        'annots': [],
        'isKeyframe': False
    }
    CVFile(annotname).json_write(annot)
    return annot

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.01):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0] / 2,
        center[1] - bbox_size[1] / 2,
        center[0] + bbox_size[0] / 2,
        center[1] + bbox_size[1] / 2,
        keypoints[valid, 2].mean()
    ]
    return bbox


def load_openpose(opname):
    mapname = {'face_keypoints_2d': 'face2d', 'hand_left_keypoints_2d': 'handl2d', 'hand_right_keypoints_2d': 'handr2d'}
    assert os.path.exists(opname), opname
    data = CVFile(opname).data
    out = []
    pid = 0
    for i, d in enumerate(data['people']):
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)
        annot = {
            'bbox': bbox_from_openpose(keypoints),
            'personID': pid + i,
            'keypoints': keypoints.tolist(),
            'isKeyframe': False
        }
        for key in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
        out.append(annot)
    return out


def json2Keypoints(path):
    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    if len(data['people']) > 0:
        person = data['people'][0]
    else:
        raise ValueError("No people in " + path)

    keypointsFlat = person['pose_keypoints_2d']

    # keypointsFlat = list(map(int, keypointsFlat))

    keypoints = list(zip(
        list(map(int, keypointsFlat[0::3])),
        list(map(int, keypointsFlat[1::3])),
        list(map(float, keypointsFlat[2::3]))
    ))

    f.close()

    return keypoints


def keypoints2json(body_keypoints, path):
    '''
    output:
    {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": [560.046, 254.438, 0.956364, 579.055, 287.017, 0.945341, 554.58, 289.731, 0.933184, 527.202, 300.732, 0.802688, 510.957, 311.574, 0.878938, 601.005, 284.288, 0.959404, 633.701, 322.576, 0.925591, 655.588, 366.202, 0.923742, 581.996, 382.658, 0.924048, 568.311, 382.697, 0.878411, 579.271, 459.06, 0.944777, 590.088, 530.022, 0.935949, 600.911, 382.618, 0.933565, 598.347, 461.771, 0.951028, 603.761, 538.2, 0.90781, 557.229, 251.571, 0.97114, 568.165, 248.972, 0.964149, 554.653, 254.308, 0.0591213, 581.807, 251.6, 0.963746, 587.377, 551.834, 0.660232, 598.261, 554.523, 0.730152, 606.492, 546.402, 0.813981, 565.52, 543.672, 0.917552, 562.853, 538.168, 0.901656, 598.199, 538.176, 0.625304], "face_keypoints_2d": [567.755, 285.861, 0.0264805, 567.755, 285.861, 0.0011107, 560.488, 263.614, 0.0405039, 562.713, 255.754, 0.0273063, 567.31, 287.493, 0.00312189, 554.852, 256.94, 0.0260331, 582.883, 270.14, 0.0289196, 554.555, 256.792, 0.00992974, 555.89, 256.347, 0.0199999, 591.486, 269.399, 0.00610582, 560.784, 250.118, 0.0144156, 560.784, 267.915, 0.0280541, 577.544, 270.585, 0.0334872, 579.917, 256.94, 0.0469211, 577.989, 262.279, 0.0713802, 579.027, 251.304, 0.0441189, 576.951, 245.075, 0.0497181, 552.627, 256.644, 0.00119348, 552.479, 256.495, 0.0259865, 552.924, 255.754, 0.0452728, 553.22, 255.605, 0.0587203, 553.072, 255.457, 0.00173667, 558.56, 258.275, 0.00131636, 556.78, 257.088, 0.00675738, 558.263, 258.423, 0.0748387, 558.56, 257.237, 0.0774293, 559.153, 256.347, 0.0351459, 557.225, 255.16, 0.00075239, 558.263, 253.826, 0.00742681, 555.89, 255.605, 0.0161634, 556.038, 258.127, 0.021296, 555.89, 259.313, 0.0286869, 556.038, 258.868, 0.0281499, 558.411, 259.61, 0.072706, 559.45, 259.313, 0.101319, 560.784, 258.275, 0.138944, 570.128, 259.61, 0.0291072, 568.793, 259.906, 0.0109359, 553.517, 255.754, 0.00075579, 553.517, 255.902, 0.000373998, 565.679, 260.945, 0.00962272, 570.128, 259.758, 0.00728712, 559.598, 257.682, 0.0167088, 559.598, 257.682, 0.0724896, 559.895, 258.275, 0.0390386, 561.823, 257.533, 0.0216627, 560.191, 258.127, 0.0437787, 560.043, 257.83, 0.0655467, 557.225, 249.524, 0.013698, 556.632, 251.304, 0.0161038, 559.598, 263.021, 0.0183623, 560.191, 262.428, 0.0156363, 560.933, 260.351, 0.0302365, 561.674, 260.203, 0.0472756, 562.268, 262.428, 0.0276113, 562.713, 262.428, 0.0259921, 562.119, 262.724, 0.0173563, 555.445, 252.342, 0.0152972, 556.038, 250.859, 0.0168393, 556.928, 250.414, 0.0085928, 556.78, 249.673, 0.0170601, 558.708, 262.279, 0.031341, 560.043, 262.724, 0.0200892, 562.119, 261.835, 0.00535929, 560.784, 257.533, 0.0407231, 562.564, 262.279, 0.0078524, 560.784, 262.131, 0.0336926, 560.339, 262.279, 0.0247457, 553.665, 255.605, 0.00161107, 560.488, 257.83, 0.058714], "hand_left_keypoints_2d": [654.653, 370.453, 0.388057, 651.072, 376.222, 0.512436, 650.276, 384.578, 0.760358, 653.062, 390.347, 0.801366, 654.852, 395.321, 0.877627, 658.234, 384.976, 0.649731, 659.627, 393.133, 0.78041, 659.826, 397.111, 0.87892, 659.229, 400.096, 0.853599, 661.815, 384.379, 0.580774, 663.606, 391.939, 0.69692, 662.014, 395.52, 0.559785, 660.423, 397.907, 0.428586, 664.004, 383.185, 0.620956, 665.595, 388.756, 0.60064, 664.402, 392.337, 0.57832, 661.616, 394.326, 0.431797, 665.595, 381.992, 0.617871, 666.59, 386.567, 0.63573, 665.595, 389.154, 0.473403, 664.004, 390.546, 0.434321], "hand_right_keypoints_2d": [512.415, 331.496, 0.153315, 516.853, 305.302, 0.0561528, 516.42, 303.786, 0.0254495, 503.864, 307.467, 0.0352869, 502.89, 306.384, 0.152461, 503.864, 308.332, 0.0299997, 513.389, 317.208, 0.012805, 503.648, 307.791, 0.0160843, 502.782, 306.817, 0.0238978, 504.297, 310.93, 0.026109, 506.462, 316.667, 0.0143305, 503.215, 311.471, 0.0233719, 502.89, 306.925, 0.0155019, 508.194, 310.606, 0.0353555, 513.065, 312.446, 0.0364679, 507.328, 319.373, 0.0212762, 503.215, 322.079, 0.018511, 503.864, 323.053, 0.0286689, 513.065, 316.775, 0.0485211, 514.147, 317.316, 0.0551642, 503.107, 322.079, 0.0277985], "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []}]}
    '''

    data = {"version": 1.3, "people": [
        {"person_id": [-1], "pose_keypoints_2d": [], "pose_keypoints_3d": [], "face_keypoints_3d": [],
         "face_keypoints_2d": [], "hand_left_keypoints_3d": [], "hand_left_keypoints_2d": [],
         "hand_right_keypoints_3d": [], "hand_right_keypoints_2d": []}]}

    data['people'][0]['pose_keypoints_2d'] = np.array(body_keypoints).flatten().tolist()

    data['people'][0]['face_keypoints_2d'] = dummyKeypoints(70).tolist()

    data['people'][0]['hand_right_keypoints_2d'] = dummyKeypoints(21).tolist()

    data['people'][0]['hand_left_keypoints_2d'] = dummyKeypoints(21).tolist()

    outfile = open(path, 'w')

    json.dump(data, outfile)

    outfile.close()


def geJointIndexFromName(jointName, jointNamesDict):
    idx = list(jointNamesDict.keys())[list(jointNamesDict.values()).index(jointName)]
    return idx


def computeNeck(keypoints):
    # Middle point between shoulders
    return computeMiddleJoint(keypoints, "LShoulder", "RShoulder")


def computeMiddleJoint(keypoints, joint1name, joint2name):
    # Middle point between shoulders
    keypoint1 = keypoints[geJointIndexFromName(joint1name, COCO_JOINTS)]
    keypoint2 = keypoints[geJointIndexFromName(joint2name, COCO_JOINTS)]
    if keypoint1[2] > 0 and keypoint2[2] > 0:
        middle_keypoint_x = int((keypoint1[0] + keypoint2[0]) / 2.)
        middle_keypoint_y = int((keypoint1[1] + keypoint2[1]) / 2.)
        threshold = (keypoint1[2] + keypoint2[2]) / 2.
        new_keypoint = [int(middle_keypoint_x), int(middle_keypoint_y), threshold]
    else:
        new_keypoint = [0, 0, 0]
    return new_keypoint


def computeMidHip(keypoints):
    # Middle point between shoulders
    return computeMiddleJoint(keypoints, "LHip", "RHip")


def coco2openpose(coco_keypoints):
    openpose_keypoints = []
    for i, openposeKeypointName in enumerate(POSE_BODY_25_BODY_PARTS):
        try:
            cocoIndex = geJointIndexFromName(openposeKeypointName, COCO_JOINTS)
            cocoKeypoint = coco_keypoints[cocoIndex]
            if cocoKeypoint[2] == 1 or cocoKeypoint[2] == 2:
                new_keypoint = (cocoKeypoint[0], cocoKeypoint[1], 1.0)
            else:
                new_keypoint = (0.0, 0.0, 0.0)
            openpose_keypoints.append(new_keypoint)
        except ValueError:
            if openposeKeypointName == "Neck":
                new_keypoint = computeNeck(coco_keypoints)
                openpose_keypoints.append(new_keypoint)
            elif openposeKeypointName == "MidHip":
                new_keypoint = computeMidHip(coco_keypoints)
                openpose_keypoints.append(new_keypoint)
            elif openposeKeypointName == "Background":
                pass
            else:
                new_keypoint = (0, 0, 0)
                openpose_keypoints.append(new_keypoint)
                # sys.exit()
        # openpose_keypoints.append(new_keypoint)

        # COCO_JOINTS.keys()[COCO_JOINTS.values().index(openposeKeypointName)]
    print("**********")
    print("Total num. of keypoints:", len(openpose_keypoints))
    print("**********")
    return openpose_keypoints
