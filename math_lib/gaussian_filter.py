# -- coding: utf-8 --
# @Time : 2022/8/12
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np


def make_gaussian_kernel(sigma, kernel_size):
    """
    Args:
        sigma:
        kernel_size:
    Returns: torch tensor 1*kernel_size
    """
    ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()
    return kernel


def init_model(num_keypoints, kernel):
    seq = nn.Sequential(
        nn.ReflectionPad2d(kernel // 2),
        nn.Conv2d(num_keypoints, num_keypoints, kernel, stride=1, padding=0, bias=None,
                  groups=num_keypoints))
    return seq


class GaussianLayer(nn.Module):
    def __init__(self, num_keypoints, kernel):
        """
        batch gaussian layer
        Args:
            num_keypoints: cocowholebody 133
            kernel:
        """
        super(GaussianLayer, self).__init__()
        self.kernel = kernel
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel // 2),
            nn.Conv2d(num_keypoints, num_keypoints, kernel, stride=1, padding=0, bias=None,
                      groups=num_keypoints))
        self.weights_init()

    def forward(self, x):
        """
        Args:
            x: N keypoints number B*N*H*W
        Returns:
        """
        return self.seq(x)

    def weights_init(self):
        # check mmpose /mmpose/mmpose/core/evaluation/top_down_eval.py
        sigma = 0.3 * ((self.kernel - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel, self.kernel))
        n[self.kernel // 2, self.kernel // 2] = 1
        k = ndimage.gaussian_filter(n, sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


class GaussianLayerPicklable(nn.Module):
    def __init__(self, num_keypoints, kernel):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kernel = kernel
        # self.seq = init_model(self.num_keypoints, self.kernel).cuda()
        # self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        # check mmpose /mmpose/mmpose/core/evaluation/top_down_eval.py
        sigma = 0.3 * ((self.kernel - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel, self.kernel))
        n[self.kernel // 2, self.kernel // 2] = 1
        k = ndimage.gaussian_filter(n, sigma=sigma)
        for name, f in self.named_parameters():
            # f.data.copy_(k)
            f.data.copy_(torch.from_numpy(k).cuda())

    def __getstate__(self):
        return {
            'num_keypoints': self.num_keypoints,
            'kernel': self.kernel,
        }

    def __setstate__(self, values):
        super().__init__()
        self.num_keypoints = values['num_keypoints']
        self.kernel = values['kernel']
        self.seq = init_model(self.num_keypoints, self.kernel).cuda()
        self.weights_init()


if __name__ == '__main__':
    sigma = 2.9
    kernel = 17
    n = np.zeros((kernel, kernel))
    n[kernel // 2, kernel // 2] = 1
    k = ndimage.gaussian_filter(n, sigma=sigma)
    print(k)
