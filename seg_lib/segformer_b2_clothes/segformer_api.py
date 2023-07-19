# -- coding: utf-8 --
# @Time : 2023/7/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from apstone import ModelBase
import numpy as np
from cv2box import CVImage
from PIL import Image

"""
0-1 RGB
input_name:['pixel_values'], shape:[['batch', 'num_channels', 'height', 'width']]
output_name:['last_hidden_state'], shape:[['batch', 'sequence', 'Transposelast_hidden_state_dim_2', 'Transposelast_hidden_state_dim_3']]
"""
MODEL_ZOO = {
    # https://huggingface.co/mattmdjaga/segformer_b2_clothes
    'segformer_b2_clothes': {
        'model_path': 'pretrain_models/seg_lib/segformer_clothes/segformer_b2_clothes.onnx'
    },
}


class SegFormer(ModelBase):
    def __init__(self, model_name='segformer_b2_clothes', provider='gpu'):
        super(SegFormer, self).__init__(MODEL_ZOO[model_name], provider)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (512, 512)

    def forward(self, image_in):
        input_size_ = CVImage(image_in).bgr.shape[:2]
        input_image = CVImage(image_in).blob_innormal(self.input_size, input_mean=self.mean, input_std=self.std,
                                                      rgb=True)
        pred_mask = self.model.forward(input_image)[0]
        pred_mask = np.transpose(pred_mask[0], (1, 2, 0))
        pred_mask = CVImage(pred_mask).resize(input_size_).bgr
        pred_mask = pred_mask.argmax(axis=2)[..., np.newaxis]
        return pred_mask.astype(np.int8)


if __name__ == '__main__':
    img_p = ''

    sf = SegFormer(model_name='segformer_b2_clothes')
    mask_img = sf.forward(img_p)
    CVImage(mask_img).show()
    print(mask_img.shape)
