# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import numpy as np


def convert98to68(list_info):
    points = list_info[0, 0:196]
    info_68 = []
    for j in range(17):
        x = points[j * 2 * 2 + 0]
        y = points[j * 2 * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(33, 38):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(42, 47):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(51, 61):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    point_38_x = (float(points[60 * 2 + 0]) + float(points[62 * 2 + 0])) / 2.0
    point_38_y = (float(points[60 * 2 + 1]) + float(points[62 * 2 + 1])) / 2.0
    point_39_x = (float(points[62 * 2 + 0]) + float(points[64 * 2 + 0])) / 2.0
    point_39_y = (float(points[62 * 2 + 1]) + float(points[64 * 2 + 1])) / 2.0
    point_41_x = (float(points[64 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
    point_41_y = (float(points[64 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
    point_42_x = (float(points[60 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
    point_42_y = (float(points[60 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
    point_44_x = (float(points[68 * 2 + 0]) + float(points[70 * 2 + 0])) / 2.0
    point_44_y = (float(points[68 * 2 + 1]) + float(points[70 * 2 + 1])) / 2.0
    point_45_x = (float(points[70 * 2 + 0]) + float(points[72 * 2 + 0])) / 2.0
    point_45_y = (float(points[70 * 2 + 1]) + float(points[72 * 2 + 1])) / 2.0
    point_47_x = (float(points[72 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
    point_47_y = (float(points[72 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
    point_48_x = (float(points[68 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
    point_48_y = (float(points[68 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
    info_68.append((point_38_x))
    info_68.append((point_38_y))
    info_68.append((point_39_x))
    info_68.append((point_39_y))
    info_68.append(points[64 * 2 + 0])
    info_68.append(points[64 * 2 + 1])
    info_68.append((point_41_x))
    info_68.append((point_41_y))
    info_68.append((point_42_x))
    info_68.append((point_42_y))
    info_68.append(points[68 * 2 + 0])
    info_68.append(points[68 * 2 + 1])
    info_68.append((point_44_x))
    info_68.append((point_44_y))
    info_68.append((point_45_x))
    info_68.append((point_45_y))
    info_68.append(points[72 * 2 + 0])
    info_68.append(points[72 * 2 + 1])
    info_68.append((point_47_x))
    info_68.append((point_47_y))
    info_68.append((point_48_x))
    info_68.append((point_48_y))
    for j in range(76, 96):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(len(list_info[196:])):
        info_68.append(list_info[196 + j])
    return np.array(info_68)
