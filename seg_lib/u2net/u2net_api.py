# -- coding: utf-8 --
# @Time : 2023/1/13
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, MyFpsCounter
from apstone import ModelBase
import cv2
import numpy as np

MODEL_ZOO = {
    # https://github.com/danielgatis/rembg
    # input_name:['input.1'], shape:[[1, 3, 320, 320]]
    # output_name:['1959', '1960', '1961', '1962', '1963', '1964', '1965'], shape:[[1, 1, 320, 320], [1, 1, 320, 320], [1, 1, 320, 320], [1, 1, 320, 320], [1, 1, 320, 320], [1, 1, 320, 320], [1, 1, 320, 320]]
    'u2net': {
        'model_path': 'pretrain_models/seg_lib/u2net/u2net.onnx'
    },
    'u2net_human_seg': {
        'model_path': 'pretrain_models/seg_lib/u2net/u2net_human_seg.onnx'
    },
    # same as u2net, smaller
    'u2netp': {
        'model_path': 'pretrain_models/seg_lib/u2net/u2netp.onnx'
    },
    # quantization from onnx-runtime
    # https://github.com/xuebinqin/U-2-Net/issues/295#issuecomment-1083041216
    'silueta': {
        'model_path': 'pretrain_models/seg_lib/u2net/silueta.onnx'
    },
}


class U2netSeg(ModelBase):
    def __init__(self, model_type='u2net', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        self.input_size = (320, 320)

    def forward(self, image_in, post_process=False):
        """
        Args:
            image_in: CVImage access type
            post_process: Post Process the mask for a smooth boundary by applying Morphological Operations
                Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
        Returns: mask 0-1
        """
        image_in_size = CVImage(image_in).bgr.shape
        image_in_pre = CVImage(image_in).blob_innormal(self.input_size, self.input_mean, self.input_std, rgb=True,
                                                   interpolation=cv2.INTER_LANCZOS4)
        pred = self.model.forward(image_in_pre)[0][:, 0, :, :].transpose(1, 2, 0)
        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred = (pred*255).astype(np.uint8)
        if post_process:
            # 开操作 平滑mask边缘
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            pred = cv2.GaussianBlur(pred, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
            pred = np.where(pred < 0.5, 0, 1)[..., np.newaxis].astype(np.float32)
        pred = CVImage(pred).resize(image_in_size[:-1][::-1], interpolation=cv2.INTER_LANCZOS4).bgr

        # First create the image with alpha channel
        rgba = cv2.cvtColor(CVImage(image_in).bgr, cv2.COLOR_RGB2RGBA)
        # Then assign the mask to the last channel of the image
        rgba[:, :, 3] = pred
        # CVImage(rgba).show()

        # rgb = cv2.bitwise_and(rgba, rgba, mask=mask)

        return pred, rgba


if __name__ == '__main__':
    fb_cur = U2netSeg(model_type='u2net', provider='gpu')
    mask, rgba = fb_cur.forward('resource/test1.jpg', post_process=False)
    CVImage(mask).show()
    CVImage(rgba).save('output.png')
