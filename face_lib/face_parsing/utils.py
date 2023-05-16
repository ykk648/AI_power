# -- coding: utf-8 --
# @Time : 2023/1/11
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import cv2


def generate_mask(mask_shape):
    mask = np.zeros((512, 512)).astype(np.float32)
    # cv2.circle(mask, (285, 285), 110, (255, 255, 255), -1)  # -1 表示实心
    cv2.ellipse(mask, (256, 256), (220, 160), 90, 0, 360, (255, 255, 255), -1)
    thres = 20
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0

    mask = cv2.stackBlur(mask, (201, 201))

    mask = mask / 255.
    mask = cv2.resize(mask, mask_shape)
    return mask[..., np.newaxis]
