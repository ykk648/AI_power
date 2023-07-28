# -- coding: utf-8 --
# @Time : 2023/7/28
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter
from apstone import ModelBase
import numpy as np

MODEL_ZOO = {
    # ref https://github.com/Sanster/lama-cleaner/blob/main/lama_cleaner/model/lama.py
    # pytorch do not support ifft op: https://github.com/pytorch/pytorch/issues/81075
    # input : RGB 0-1 (1,3,H,W)  (1,1,H,W)
    'big_lama': {
        'model_path': 'pretrain_models/art_lib/inpainting/big-lama.tjm'
    },
}


class LAMA(ModelBase):
    def __init__(self, model_type='big_lama', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, image_, mask_):
        """
        Args:
            image_: CVImage acceptable class (path BGR tensor byte PIL etc.)
            mask_: [H, W]
        Returns: [H, W, C] BGR
        """

        image_in = CVImage(CVImage(image_).rgb()).tensor()
        mask_ = CVImage(CVImage(mask_).mask(rgb=True)).tensor()
        mask_in = (mask_ > 0) * 1
        inpainted_image_ = self.model.forward([image_in, mask_in])
        inpainted_image_ = inpainted_image_[0].permute(1, 2, 0).detach().cpu().numpy()
        inpainted_image_ = np.clip(inpainted_image_ * 255, 0, 255).astype("uint8")
        inpainted_image_ = CVImage(inpainted_image_).rgb()  # rgb2bgr
        return inpainted_image_


if __name__ == '__main__':
    image_p = 'resources/inpainting/dog_chair.png'
    mask_p = 'resources/inpainting/dog_chair_mask.png'

    fb_cur = LAMA(model_type='big_lama', provider='gpu')
    inpaint_result = fb_cur.forward(image_p, mask_p)
    print(inpaint_result.shape)
    CVImage(inpaint_result).show()
