import numpy as np
import cv2
import os
from tqdm import tqdm


def count_mean_std(img_path):
    path = img_path
    means = [0, 0, 0]
    stdevs = [0, 0, 0]

    # index = 1
    num_imgs = 0
    img_names = os.listdir(path)
    for img_name in tqdm(img_names):
        num_imgs += 1
        # print(img_name)
        img = cv2.imread(os.path.join(path, img_name))
        img = np.asarray(img)
        img = img.astype(np.float32)  # / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
    # print(num_imgs)
    means.reverse()
    stdevs.reverse()

    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean={},normStd = {})'.format(means, stdevs).replace(' ', ','))
