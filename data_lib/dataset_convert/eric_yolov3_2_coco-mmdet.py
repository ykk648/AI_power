# -- coding: utf-8 --
# @Time : 2022/9/1
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://gitcode.net/EricLee/yolo_v3
coco hand
some error may exists, use 'yolo2coco.py' instead
"""
from cv2box import get_path_by_ext, CVFile, CVImage
from tqdm import tqdm

image_p = ''
labels_p = ''
coco_temp = ''

coco_out = CVFile(coco_temp).data

coco_out['images'] = []
coco_out['annotations'] = []
count = 10010
count2 = 100101
for image_p in tqdm(get_path_by_ext(image_p)):
    file_name = str(image_p.stem + image_p.suffix)
    image_p = str(image_p)
    height, width = CVImage(image_p).bgr.shape[0:2]
    image_label_p = image_p.replace('/images', '/labels').replace('.jpg', '.txt')
    # print(image_label_p)
    labels = CVFile(image_label_p).data
    # print(labels)
    coco_out['images'].append({
        'id': count,
        'path': image_p[68:],
        'width': width,
        'height': height,
        'file_name': file_name,
    })
    for i in range(len(labels)):
        coco_out['annotations'].append({
            'image_id': count,
            'id': count2,
            'category_id': 0,
            'bbox': [
                int(float(str(labels[i]).split(' ')[1]) * width - float(str(labels[i]).split(' ')[3]) * width * 0.5),
                int(float(str(labels[i]).split(' ')[2]) * height - float(str(labels[i]).split(' ')[4][:-5]) * height * 0.5),
                int(float(str(labels[i]).split(' ')[3]) * width),
                int(float(str(labels[i]).split(' ')[4][:-5]) * height)],
            'iscrowd': False,
            'isbbox': True,
            'area': int(float(str(labels[i]).split(' ')[3]) * width) * int(float(str(labels[i]).split(' ')[4][:-5]) * height)
        })
        count2 += 1

    count += 1

CVFile(coco_temp.replace('.json', '_out.json')).json_write(coco_out)
