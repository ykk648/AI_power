import os
import glob
import sys
import random
from pathlib import Path
from utils import get_path_by_ext
from cv2box import CVFile


def gen_txt_from_path(base_path, img_format='jpg', train_ratio=0.8):
    train_data_path = os.path.join(base_path, 'dataset')

    labels = os.listdir(train_data_path)

    for index, label in enumerate(labels):
        print('label: {}\t index: {}'.format(label, index))
        # img_list = glob.glob(os.path.join(train_data_path, label, '*.{}'.format(img_format)))
        img_list = list(Path(os.path.join(train_data_path, label)).glob('*/*.{}'.format(img_format)))
        random.shuffle(img_list)
        print(len(img_list))
        train_list = img_list[:int(train_ratio * len(img_list))]
        val_list = img_list[(int(train_ratio * len(img_list)) + 1):]
        with open(os.path.join(base_path, 'train.txt'), 'a') as f:
            for img in train_list:
                img = str(img).replace(base_path, '')
                # print(img)
                f.write(img + ' ' + str(index))
                f.write('\n')

        with open(os.path.join(base_path, 'val.txt'), 'a') as f:
            for img in val_list:
                img = str(img).replace(base_path, '')
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

    # imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    # with open(txtpath + 'test.txt', 'a') as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')


"""
'female','male', 
'front', 'side',
'clean','occlusion', 
'super_hq', 'hq', 'blur',
'nonhuman'
"""


def gen_txt_from_json(base_path, train_ratio=0.8):
    # multi label , labelme
    train_list = {}
    test_list = {}

    for img_path in get_path_by_ext(base_path):
        label = ''
        img_path_str = str(img_path)[54:]
        json_path = str(img_path.parent / (str(img_path.stem) + '.json'))
        # print(json_path)
        json_data = CVFile(json_path).data
        try:
            label += '01' if json_data['flags']['男'] else '10'
            label += '01' if json_data['flags']['侧脸'] else '10'
            label += '01' if json_data['flags']['遮挡'] else '10'
            if json_data['flags']['非常清晰']:
                label += '100'
            elif json_data['flags']['清晰']:
                label += '010'
            else:
                label += '001'
            label += '1' if json_data['flags']['非人脸'] else '0'
        except TypeError:
            print(json_path, json_data)
            continue

        if random.random() > train_ratio:
            test_list[img_path_str] = label
        else:
            train_list[img_path_str] = label

    with open(os.path.join(base_path, 'train.txt'), 'a') as f:
        for k, v in train_list.items():
            f.write(k + ' ' + v)
            f.write('\n')

    with open(os.path.join(base_path, 'val.txt'), 'a') as f:
        for k, v in test_list.items():
            f.write(k + ' ' + v)
            f.write('\n')


if __name__ == '__main__':
    gen_txt_from_json('')
