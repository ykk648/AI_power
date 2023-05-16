# -- coding: utf-8 --
# @Time : 2022/7/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

mediapipe33_to_openpose25 = [0, 0, 12, 14, 16, 11, 13, 15, 0, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 31, 31, 29,
                           32, 32, 30]

alphapose17_to_openpose25 = [9, 8, 14, 15, 16, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1,
                             -1, -1, -1, -1, -1, -1, -1, ]


class Openpose25:
    def __init__(self, poses=None):
        self.poses = poses

    def from_mediapipe_33(self, poses):
        """

        Args:
            poses: 33 * 3

        Returns: 25 * 3

        """
        poses = poses[mediapipe33_to_openpose25]
        poses[8, :2] = poses[[9, 12], :2].mean(axis=0)
        poses[8, 2] = poses[[9, 12], 2].min(axis=0)
        poses[1, :2] = poses[[2, 5], :2].mean(axis=0)
        poses[1, 2] = poses[[2, 5], 2].min(axis=0)
        return poses
