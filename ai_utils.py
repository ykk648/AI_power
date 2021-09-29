import cv2
import PIL
import torch.nn.functional as F


def down_sample(target_, size):
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


def img_show(img, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is PIL.Image.Image:
        img.show()
    else:
        cv2.imshow('', img)
        cv2.waitKey(0)


def img_save(img, img_save_p, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is PIL.Image.Image:
        img.save(img_save_p, quality=95)
    else:
        cv2.imwrite(img_save_p, img)
