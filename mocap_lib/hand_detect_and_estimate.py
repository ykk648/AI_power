# -- coding: utf-8 --
# @Time : 2022/3/3
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from estimation_3d import HandDetectorYolox, MediapipeHand, IKModel, ThirdViewDetector
from cv2box import CVVideoLoader, CVImage, CVFile, MyFpsCounter
from tqdm import tqdm
import cv2
import numpy as np

if __name__ == '__main__':
    video_p = ''

    hdy = HandDetectorYolox(thres=0.3)
    # hdd2 = ThirdViewDetector()
    mph = MediapipeHand()
    ikm = IKModel()

    with CVVideoLoader(video_p) as cvvl:
        for i in tqdm(range(len(cvvl))):
            _, frame = cvvl.get()
            # CVImage(frame).show()
            bboxs, show_image = hdy.forward(frame, show=True)
            # hdd2_results, show_image = hdd2.forward(frame, show=True)
            # CVImage(frame).show()
            # bboxs = np.array(hdd2_results["instances"].pred_boxes.tensor.to('cpu'))

            if len(bboxs) != 2:
                print(i)

            # frame_info = {'left': None, 'right': None}
            # for bbox_ in bboxs:
            #     # print(bbox_)
            #     frame_crop = crop_padding_and_resize(frame, bbox_)
            #     # CVImage(frame_crop).show()
            #     hand_xyz, side_label, side_label_score = mph.forward(frame_crop)
            #     # print(hand_xyz, side_label)
            #     if hand_xyz is None:
            #         continue
            #     theta = ikm.forward_np(np.array(hand_xyz))
            #     # print(theta)
            #
            #     if side_label == 'Left':
            #         frame_info['left'] = theta
            #     elif side_label == 'Right':
            #         frame_info['right'] = theta
            # CVFile('./yolox_mp_thres0_world_xyz/frame{}.pkl'.format(i)).pickle_write(frame_info)
