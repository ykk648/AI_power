# -- coding: utf-8 --
# @Time : 2022/8/15
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVBbox
import numpy as np


class BboxTracking:
    def __init__(self, image_shape_, batch=1):
        """
        Args:
            image_shape_: (W,H)
        """
        self.image_shape_ = image_shape_
        self.batch_ = batch
        self.last_bbox_array = None

    def reset_condition(self, area_limit):
        """
        judge bbox area by px^2
        """
        if self.last_bbox_array is None:
            return self.last_bbox_array
        else:
            bbox_area = CVBbox(self.last_bbox_array).area()
            reset_index = np.where(bbox_area < area_limit)
            self.last_bbox_array[reset_index] = np.array([0, 0, self.image_shape_[0], self.image_shape_[1]])
            return self.last_bbox_array

    def forward(self, keypoints_batch=None, margin=0.1, area_limit=1000):
        """
        Args:
            keypoints_batch: N_view*N_kp*N_axis [N*3, N*3, ...]
            margin:
            area_limit:
        Returns:
        """

        if not keypoints_batch:
            # part_w = int(self.image_shape_[0] * 1 / 3)
            # return [part_w, 0, 2 * part_w, self.image_shape_[1]]
            return np.array([[0, 0, self.image_shape_[0], self.image_shape_[1]]]).repeat(self.batch_, axis=0)
        else:
            self.last_bbox_array = np.array(
                [CVBbox(None).get_bbox_from_points(keypoints[:, :2], self.image_shape_, margin_ratio=margin) for
                 keypoints in keypoints_batch])
            return self.reset_condition(area_limit)
