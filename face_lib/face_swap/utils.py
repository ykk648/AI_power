# -- coding: utf-8 --
# @Time : 2022/8/26
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numexpr as ne
# ne.set_num_threads(1)
import cv2
import numpy as np


def mat2mask(frame, mat, mask_size):
    """

    Args:
        frame:
        mat:
        mask_size: (w,h)

    Returns:

    """
    kernel_size = int(0.05 * min((frame.shape[1], frame.shape[0])))

    img_mask = np.full(mask_size, 255, dtype=float)
    img_mask = cv2.warpAffine(img_mask, mat, (frame.shape[1], frame.shape[0]), borderValue=0.0)
    # print(img_mask.shape)
    img_mask[img_mask > 20] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)

    blur_kernel_size = (20, 20)
    blur_size = tuple(2 * i + 1 for i in blur_kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

    # img_mask[img_mask > 0] = 255
    img_mask /= 255
    # if angle != -1:
    #     img_mask = np.reshape(img_mask, [img_mask.shape[1], img_mask.shape[1], 1]).astype(np.float32)
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1]).astype(np.float32)
    return img_mask

