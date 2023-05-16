# -- coding: utf-8 --
# @Time : 2022/8/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
for hand detect
"""
from cv2box import CVFile
from tqdm import tqdm


def del_other_category(json_in_, json_out_):
    json_data = CVFile(json_in_).data
    out_data = json_data.copy()
    out_data['annotations'] = []
    out_data['categories'] = json_data['categories'][:1]
    out_data['categories'][0]['name'] = 'hand'
    out_data['categories'][0]['id'] = 1
    print(len(json_data['annotations']))
    for i in tqdm(range(len(json_data['annotations']))):
        dummy = json_data['annotations'][i]
        if dummy['category_id'] != 1 and 'bbox' in dummy.keys():
            dummy['category_id'] = 1
            out_data['annotations'].append(dummy)

    print(len(out_data['annotations']))
    CVFile(json_out_).json_write(out_data)


def del_some_name(json_in_, json_out_):
    json_data = CVFile(json_in_).data
    out_data = json_data.copy()
    out_data['annotations'] = []
    out_data['images'] = []
    print(len(json_data['images']))
    print(len(json_data['annotations']))
    del_image_id_list = []

    for i in tqdm(range(len(json_data['images']))):
        dummy = json_data['images'][i]
        if '0707_4_' in dummy['file_name']:
            del_image_id_list.append(dummy['id'])
        else:
            out_data['images'].append(dummy)

    for i in tqdm(range(len(json_data['annotations']))):
        dummy = json_data['annotations'][i]
        if dummy['image_id'] not in del_image_id_list:
            out_data['annotations'].append(dummy)

    print(len(out_data['images']))
    print(len(out_data['annotations']))
    CVFile(json_out_).json_write(out_data)


def concat_2_json(json_in_1, json_in_2, json_out_):
    json_data_1 = CVFile(json_in_1).data
    json_data_2 = CVFile(json_in_2).data
    out_data = json_data_1.copy()
    out_data['annotations'] += json_data_2['annotations']
    out_data['images'] += json_data_2['images']
    print(len(out_data['annotations']))
    print(len(out_data['images']))
    CVFile(json_out_).json_write(out_data)


if __name__ == '__main__':
    json_in = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/mmlab/mmdetection_data/datasets_TVCOCO_hand_train/annotations/train.json'
    json_out = '/workspace/84_cluster/mnt/cv_data_ljt/dataset/mmlab/mmdetection_data/datasets_TVCOCO_hand_train/annotations/train_out.json'
    del_other_category(json_in, json_out)

    # json_in = ''
    # json_out = ''
    # del_some_name(json_in, json_out)

    # json_in_1 = ''
    # json_in_2 = ''
    # json_out = ''
    # concat_2_json(json_in_1, json_in_2, json_out)
