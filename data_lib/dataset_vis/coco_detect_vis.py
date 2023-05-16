# -- coding: utf-8 --
# @Time : 2022/9/21
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://github.com/wukaishuns/Coco-datasets-Visualization-and-change-tools/blob/main/viscoco.py
"""
import os
import sys

# if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
#   sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

from pycocotools.coco import COCO

matplotlib.use('TkAgg')
annfile = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/mmlab/mmdetection_data/local_multi_view_all/0915_multi_view_4person_test_5k/annotations/annotations.json'
imgroot = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/mmlab/mmdetection_data/local_multi_view_all/0915_multi_view_4person_test_5k/images'


def showAnns(anns):
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    captions = []
    polygons = []
    rectangles = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    # print(132131,ann['category_id'])
                    # print(cat_names[0])
                    captions.append(cat_names[ann['category_id'] - 1])
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    l_corner, w, h = (ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3]
                    rectangles.append(Rectangle(l_corner, w, h))
                    polygons.append(Polygon(poly))
                    color.append(c)

    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, alpha=1, linestyle='--', linewidths=2)
    ax.add_collection(p)

    for i in range(len(captions)):
        x = rectangles[i].xy[0]
        y = rectangles[i].xy[1]
        ax.text(x, y, captions[i], size=10, verticalalignment='top', color='w', backgroundcolor="none")

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.6)
    ax.add_collection(p)
    # p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    p = PatchCollection(polygons, facecolor='none', edgecolors='b', linewidths=0.5)
    ax.add_collection(p)
    print('Ok!')


import random

coco = COCO(annfile)
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print(cat_names)
catids = coco.getCatIds(catNms=random.randint(0, len(cat_names) - 1))
imgids = coco.getImgIds(catIds=catids)


def draw(m, n, i):
    img = coco.loadImgs(imgids[np.random.randint(0, len(imgids))])[0]
    I = io.imread(os.path.join(imgroot, img['file_name']))
    plt.subplot(m, n, i)
    plt.axis('off')
    plt.title(img['file_name'], fontsize=8, color='blue')
    plt.imshow(I, aspect='equal')
    annids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annids)
    showAnns(anns)


if 1:
    m = 4
    n = 4
    plt.figure(figsize=(m * 6, n * 4))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # fig = plt.figure(figsize=(18*m,12*n))
    for i in range(1, m * n + 1):
        draw(m, n, i)
    plt.savefig('detect_example.png')
    plt.show()
