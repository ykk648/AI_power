# -- coding: utf-8 --
# @Time : 2022/7/6
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from .smoothnet_api import SmoothNetFilter
from .one_euro_api import OneEuroFilter
import numpy as np

SMOOTH_NET_8 = 'pretrain_models/smooth_filter/smoothnet_ws8_h36m.pth'
SMOOTH_NET_16 = 'pretrain_models/smooth_filter/smoothnet_ws16_h36m.pth'
SMOOTH_NET_32 = 'pretrain_models/smooth_filter/smoothnet_ws32_h36m.pth'
SMOOTH_NET_64 = 'pretrain_models/smooth_filter/smoothnet_ws64_h36m.pth'


class SmoothFilter:
    def __init__(self, filter_type, **kwargs):

        if filter_type == 'one_euro':
            self.filter = OneEuroFilter()
        # can not use !
        # if filter_type == 'smooth_net_8':
        #     self.window = 8
        #     self.filter = SmoothNetFilter(8, SMOOTH_NET_8, root_index=kwargs['root_index'])
        # elif filter_type == 'smooth_net_16':
        #     self.window = 16
        #     self.filter = SmoothNetFilter(16, SMOOTH_NET_16, root_index=kwargs['root_index'])
        # elif filter_type == 'smooth_net_32':
        #     self.window = 32
        #     self.filter = SmoothNetFilter(32, SMOOTH_NET_32, root_index=kwargs['root_index'])
        # elif filter_type == 'smooth_net_64':
        #     self.window = 64
        #     self.filter = SmoothNetFilter(64, SMOOTH_NET_64, root_index=kwargs['root_index'])

        # self.history_list = [[], [], []]
        # self.thres_list = [[], [], []]

    def forward(self, x):
        """

        Args:
            x: [N, 2] or [N, 3]

        Returns:

        """
        # history_now = self.history_list[id]
        # thres_list_now = self.thres_list[id]

        # if x.shape[1] == 3:
        #     x_new = x[:, :2]
        #     thres = x[:, 2]
        # else:
        #     x_new = x
        #     thres = 0

        results = self.filter.forward(np.array([x.copy()]))
        return results[0]
        # return np.concatenate((results[0], thres.reshape(-1, 1)), 1)

        # if len(self.history_list[id]) < self.window:
        #     self.history_list[id].append(x_new)
        #     self.thres_list[id].append(thres)
        #     return x
        # else:
        #     self.history_list[id].append(x_new)
        #     self.thres_list[id].append(thres)
        #     self.history_list[id] = self.history_list[id][-self.window:]
        #     self.thres_list[id] = self.thres_list[id][-self.window:]
        #     results = self.filter.forward(np.array(self.history_list[id].copy()))
        #
        #     return np.concatenate((results[-1], self.thres_list[id][-1].reshape(-1,1)), 1)
