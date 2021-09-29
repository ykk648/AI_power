import os
import glob
import sys
import random


def gen_txt_from_path(base_path, img_format='jpg', train_ratio=0.8):
    train_data_path = os.path.join(base_path, 'dataset')

    labels = os.listdir(train_data_path)

    for index, label in enumerate(labels):
        print('label: {}\t index: {}'.format(label, index))
        img_list = glob.glob(os.path.join(train_data_path, label, '*.{}'.format(img_format)))
        random.shuffle(img_list)
        print(len(img_list))
        train_list = img_list[:int(train_ratio * len(img_list))]
        val_list = img_list[(int(train_ratio * len(img_list)) + 1):]
        with open(os.path.join(base_path, 'train.txt'), 'a') as f:
            for img in train_list:
                img = img.replace(base_path, '')
                # print(img)
                f.write(img + ' ' + str(index))
                f.write('\n')

        with open(os.path.join(base_path, 'val.txt'), 'a') as f:
            for img in val_list:
                img = img.replace(base_path, '')
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

    # imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    # with open(txtpath + 'test.txt', 'a') as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')
