# -- coding: utf-8 --
# @Time : 2022/8/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVFile
from tqdm import tqdm


def get_coco_bbox_gt(json_in_, json_out_):
    json_data = CVFile(json_in_).data

    out_list = []
    for i in tqdm(range(len(json_data['annotations']))):
        dummy = json_data['annotations'][i]
        if dummy['category_id'] == 1:
            out_list.append({
                'bbox': dummy['bbox'],
                'category_id': dummy['category_id'],
                'image_id': dummy['image_id'],
                'score': 1.0,
            })
    print(len(out_list))
    CVFile(json_out_).json_write(out_list)


def del_other_category(json_in_, json_out_):
    json_data = CVFile(json_in_).data
    out_data = json_data.copy()
    out_data['annotations'] = []
    print(len(json_data['annotations']))
    for i in tqdm(range(len(json_data['annotations']))):
        dummy = json_data['annotations'][i]
        if dummy['category_id'] == 1 and 'keypoints' in dummy.keys():
            out_data['annotations'].append(dummy)

    print(len(out_data['annotations']))
    CVFile(json_out_).json_write(out_data)


if __name__ == '__main__':
    json_in = ''
    json_out = ''
    # get_coco_bbox_gt(json_in, json_out)
    del_other_category(json_in, json_out)
