# -- coding: utf-8 --
# @Time : 2023/1/13
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, MyFpsCounter
from apstone import ModelBase
import cv2
import numpy as np

MODEL_ZOO = {
    # https://github.com/danielgatis/rembg/blob/main/rembg/session_cloth.py
    # input_name:['input'], shape:[['batch_size', 3, 768, 768]]
    # output_name:['output', 'd1', 'onnx::Concat_1876', 'onnx::Concat_1896', 'onnx::Concat_1916', 'onnx::Concat_1936', 'onnx::Concat_1956'], shape:[['batch_size', 4, 768, 768], ['Convd1_dim_0', 4, 768, 768], ['Resizeonnx::Concat_1876_dim_0', 'Resizeonnx::Concat_1876_dim_1', 'Resizeonnx::Concat_1876_dim_2', 'Resizeonnx::Concat_1876_dim_3'], ['Resizeonnx::Concat_1896_dim_0', 'Resizeonnx::Concat_1896_dim_1', 'Resizeonnx::Concat_1896_dim_2', 'Resizeonnx::Concat_1896_dim_3'], ['Resizeonnx::Concat_1916_dim_0', 'Resizeonnx::Concat_1916_dim_1', 'Resizeonnx::Concat_1916_dim_2', 'Resizeonnx::Concat_1916_dim_3'], ['Resizeonnx::Concat_1936_dim_0', 'Resizeonnx::Concat_1936_dim_1', 'Resizeonnx::Concat_1936_dim_2', 'Resizeonnx::Concat_1936_dim_3'], ['Resizeonnx::Concat_1956_dim_0', 'Resizeonnx::Concat_1956_dim_1', 'Resizeonnx::Concat_1956_dim_2', 'Resizeonnx::Concat_1956_dim_3']]
    'u2net_cloth_seg': {
        'model_path': 'pretrain_models/seg_lib/u2net/u2net_cloth_seg.onnx',
        'input_dynamic_shape': (1, 3, 768, 768),
    },
}


class U2netClothSeg(ModelBase):
    def __init__(self, model_type='u2net_cloth_seg', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        self.input_size = (768, 768)

    def forward(self, image_in, **kwargs):
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
        pred_mask = self.model.forward(image_in_pre)
        from scipy.special import log_softmax
        pred_mask = log_softmax(pred_mask[0], 1)
        pred_mask = np.argmax(pred_mask, axis=1, keepdims=True)
        pred_mask = np.squeeze(pred_mask, 0)
        pred_mask = np.squeeze(pred_mask, 0)
        pred_mask = pred_mask.astype(np.uint8)

        pred_mask = CVImage(pred_mask).resize(image_in_size[:-1][::-1], interpolation=cv2.INTER_LANCZOS4).bgr

        # First create the image with alpha channel
        rgba = CVImage(image_in).bgr
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGBA)
        # Then assign the mask to the last channel of the image
        rgba[:, :, 3] = pred_mask

        upper_body_mask = pred_mask.copy()
        upper_body_mask[np.where(upper_body_mask != 1)] = 0
        upper_body_mask[np.where(upper_body_mask == 1)] = 255

        lower_body_mask = pred_mask.copy()
        lower_body_mask[np.where(lower_body_mask != 3)] = 0
        lower_body_mask[np.where(lower_body_mask == 3)] = 255

        # full_body_mask = pred_mask.copy()
        # full_body_mask[np.where(full_body_mask != 2)] = 0
        # full_body_mask[np.where(full_body_mask == 2)] = 255

        return [upper_body_mask, lower_body_mask]


if __name__ == '__main__':
    fb_cur = U2netClothSeg(model_type='u2net_cloth_seg', provider='gpu')
    mask = fb_cur.forward('resource/for_pose/yoga2.jpg', post_process=False)
    CVImage(mask[0]).show()
    CVImage(mask[1]).show()

