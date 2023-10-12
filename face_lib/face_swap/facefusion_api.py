# -- coding: utf-8 --
# @Time : 2023/9/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
from https://modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/
onnx convert script: https://github.com/ykk648/modelscope/commit/bae919e7ac4c46d502360e0bb1ac38c116f9d912
"""

from apstone import ModelBase
from cv2box import CVImage


MODEL_ZOO = {
    # input_name: ['input_1', 'input_2', 'input_3', 'input_4'], shape: [[1, 3, 256, 256], [1, 512], [1, 17, 2], [1, 17, 2]]
    # output_name: ['output_1'], shape: [[1, 3, 256, 256]]
    'face_fusion_damo': {
        'model_path': 'pretrain_models/face_lib/face_swap/face_fusion_damo.onnx',
    },
}


class FaceFusionDamo(ModelBase):
    def __init__(self, model_name='face_fusion_damo', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)
        self.input_std = self.input_mean = 127.5
        self.input_size = (256, 256)

    def forward(self, src_face_image, dst_face_latent, kp_fuse, kp_target):
        """
        Args:
            src_face_image: BGR，aligned face, CVImage acceptable class
            dst_face_latent: [1, 512] from CurricularFace
            kp_fuse: [1, 17, 2] from Deep3DFaceRecon
            kp_target: [1, 17, 2] from Deep3DFaceRecon
        Returns: [1, 3, 256, 256] 0-1
        """
        src_face_image = CVImage(src_face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=False)
        swap_face = self.model.forward([src_face_image, dst_face_latent, kp_fuse, kp_target])
        return [swap_face_ * 0.5 + 0.5 for swap_face_ in swap_face]
