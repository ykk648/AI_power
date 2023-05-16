# -- coding: utf-8 --
# @Time : 2023/5/13
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import pandas as pd


def cal_dis(points, centroids, k):
    cal_dis_res = []
    for point in points:
        dis = np.linalg.norm(np.tile(point, (k, 1)) - centroids)
        cal_dis_res.append(dis)
    return cal_dis_res


def update_centroids(points, centroids, k):
    cal_dis_list = cal_dis(points, centroids, k)
    min_cal_dis_list = np.argmin(cal_dis_list, axis=1)
    new_centroids = pd.DataFrame(points).groupby(min_cal_dis_list).mean()
    diff = new_centroids - centroids
    return new_centroids, diff


def k_means(points, k):
    centroids = points.sample(k)
    # use min diff or optim fix rounds
    for i in range(100):
        centroids, _ = update_centroids(points, centroids, k)
    return centroids
