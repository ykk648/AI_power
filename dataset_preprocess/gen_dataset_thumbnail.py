# -- coding: utf-8 --
# @Time : 2022/1/7
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import os

import PIL.Image as Image
from cv2box.utils import get_path_by_ext


def resize_by_width(infile, image_size):
    """按照宽度进行所需比例缩放"""
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = int(x // lv)
    y_s = int(y // lv)
    print("x_s", x_s, y_s)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return out


def get_new_img_xy(infile, image_size):
    """返回一个图片的宽、高像素"""
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = x // lv
    y_s = y // lv
    # print("x_s", x_s, y_s)
    # out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return x_s, y_s


# 定义图像拼接函数
def image_compose(image_colnum, image_size, image_rownum, image_names, image_save_path, x_new, y_new):
    to_image = Image.new('RGB', (image_colnum * x_new, image_rownum * y_new))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    total_num = 0
    for y in range(1, image_rownum + 1):
        for x in range(1, image_colnum + 1):
            from_image = resize_by_width(image_names[image_colnum * (y - 1) + x - 1], image_size)
            # from_image = Image.open(image_names[image_colnum * (y - 1) + x - 1]).resize((image_size,image_size ), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * x_new, (y - 1) * y_new))
            total_num += 1
            if total_num == len(image_names):
                break
    return to_image.save(image_save_path)  # 保存新图


def merge_images(image_dir_path,image_size,image_colnum):
    # 获取图片集地址下的所有图片名称
    image_fullpath_list = get_path_by_ext(image_dir_path)[:100]
    print("image_fullpath_list", len(image_fullpath_list), image_fullpath_list)

    image_save_path = r'{}_thumbnail.jpg'.format(image_dir_path)  # 图片转换后的地址
    # image_rownum = 4  # 图片间隔，也就是合并成一张图后，一共有几行
    image_rownum_yu = len(image_fullpath_list) % image_colnum
    if image_rownum_yu == 0:
        image_rownum = len(image_fullpath_list) // image_colnum
    else:
        image_rownum = len(image_fullpath_list) // image_colnum + 1

    x_list = []
    y_list = []
    for img_file in image_fullpath_list:
        img_x, img_y = get_new_img_xy(str(img_file), image_size)
        x_list.append(img_x)
        y_list.append(img_y)

    print("x_list", sorted(x_list))
    print("y_list", sorted(y_list))
    x_new = int(x_list[len(x_list) // 5 * 4])
    y_new = int(x_list[len(y_list) // 5 * 4])
    image_compose(image_colnum, image_size, image_rownum, image_fullpath_list, image_save_path, x_new, y_new)  # 调用函数


if __name__ == '__main__':
    image_dir_path = ''  # 图片集地址
    image_size = 128  # 每张小图片的大小
    image_colnum = 10  # 合并成一张图后，一行有几个小图
    merge_images(image_dir_path, image_size, image_colnum)



