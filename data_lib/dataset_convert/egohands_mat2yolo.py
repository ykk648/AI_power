"""
http://vision.soic.indiana.edu/projects/egohands/
https://blog.csdn.net/wukong168/article/details/122783179
"""
import scipy.io as sio
import os
import gc
from PIL import Image
import six.moves.urllib as urllib
import xml.etree.cElementTree as ET
import shutil as sh
import numpy as np
import cv2
import random
from shutil import copyfile

def save_txt(outpath, ans):
    with open(outpath, 'w') as outfile:
        outfile.write(ans)

def get_bbox_txt(base_path, dir):
    image_path_array = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if (f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)

    # sort image_path_array to ensure its in the low to high order expected in polygon.mat
    image_path_array.sort()
    pointindex = 0
    boxes = sio.loadmat(base_path + dir + "/polygons.mat")
    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    # classLabel
    
    for first in polygons:
        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)

        width = int(np.size(img, 1))
        height = int(np.size(img, 0))
        
        pointindex += 1
        ans = ''
        for pointlist in first:
            # print(np.shape(pst)) #(0,2)
            max_x = max_y = min_x = min_y = 0

            findex = 0

            for point in pointlist:
                if (len(point) == 2):

                    x = int(point[0])
                    y = int(point[1])
                    # print('x', x)
                    # print('y', y)

                    if (findex == 0):
                        min_x = x
                        min_y = y

                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                norm_width = (max_x - min_x) / width
                norm_height = (max_y - min_y) / height
                center_x, center_y = (max_x + min_x) / 2, (max_y + min_y) / 2
                norm_center_x = center_x / width
                norm_center_y = center_y / height
                ans = ans + '2' + ' ' + str(norm_center_x) +' ' +str(norm_center_y) +' ' +str(norm_width) +' ' +str(norm_height) + '\n'
        
        txt_path = img_id.split(".")[0]
        if not os.path.exists(txt_path + ".txt"):
            save_txt(txt_path + ".txt", ans)

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_txt_files(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_bbox_txt(image_dir, dir)
    
def split_data_test_eval_train(image_dir):
    create_directory("images")
    create_directory("images/train")
    create_directory("images/test")

    data_size = 4000
    loop_index = 0
    data_sampsize = int(0.1 * data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (f.split(".")[1] == "jpg"):
                    loop_index += 1
                    print(loop_index, f)

                    if loop_index in test_samp_array:
                        os.rename(image_dir + dir +
                                  "/" + f, "images/test/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".txt", "images/test/" + f.split(".")[0] + ".txt")
                    else:
                        os.rename(image_dir + dir +
                                  "/" + f, "images/train/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".txt", "images/train/" + f.split(".")[0] + ".txt")
                    #print(loop_index, image_dir + f)
            #print(">   done scanning director ", dir)
            os.remove(image_dir + dir + "/polygons.mat")
            os.rmdir(image_dir + dir)

def rename_files(image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (dir not in f):
                    if (f.split(".")[1] == "jpg"):
                        loop_index += 1
                        os.rename(image_dir + dir +
                                  "/" + f, image_dir + dir +
                                  "/" + dir + "_" + f)
                else:
                    break

    generate_txt_files("_LABELLED_SAMPLES/")


rename_files("_LABELLED_SAMPLES/")
split_data_test_eval_train("_LABELLED_SAMPLES/")

