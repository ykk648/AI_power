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


def reverse2wholeimage_hifi(swaped_img, mat_rev, frame_wait_merge, orisize, img_mask=None, crop_size=None, crop_coord=None):
    """
    swaped_img: [1,512,512,3]
    """
    if crop_coord:
        x = crop_coord[0]
        y = crop_coord[1]
        w = crop_coord[2]
        h = crop_coord[3]
        orisize = (w, h)

    swaped_img = ((swaped_img[0] + 1) / 2)[0].transpose((1, 2, 0))
    target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)

    if not img_mask:
        img_mask = mat2mask(frame_wait_merge, mat_rev, crop_size)

    img = ne.evaluate('img_mask * (target_image * 255) ')[..., ::-1]

    if not crop_coord:
        crop = frame_wait_merge[y:y + h, x:x + w]
        frame_wait_merge[y:y + h, x:x + w] = ne.evaluate('img + (1 - img_mask) * crop').astype(np.uint8)
        img = frame_wait_merge
    else:
        img = ne.evaluate('img + frame_wait_merge')

    final_img = img.astype(np.uint8)
    return final_img
