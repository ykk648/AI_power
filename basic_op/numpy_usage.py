# -- coding: utf-8 --
# @Time : 2021/11/12
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np

# == process mask to cv show ==
mask = []
mask = np.repeat(mask, 3, axis=0).transpose((1, 2, 0)).astype(np.uint8)
mask = mask * 255
mask = mask[np.newaxis, :, :]
mask = mask[None, :, :]
