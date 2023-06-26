# -- coding: utf-8 --
# @Time : 2022/6/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage
from apstone import ModelBase

"""
'identity':    [80,]   # 身份系数
'expression':  [64,]   # 表情系数
'texture':     [80,]   # 纹理系数
'rotation':    [3,]    # 旋转系数
'gamma':       [27,]   # 亮度系数
'translation': [3,]    # 平移系数
"""

MODEL_ZOO = {
    # https://github.com/sicxu/Deep3DFaceRecon_pytorch
    # input_name: ['input_1'], shape: [[1, 3, 224, 224]]
    # output_name: ['output_1'], shape: [[1, 257]]
    'facerecon_230425': {
        'model_path': 'pretrain_models/face_lib/face_3d/epoch_20_facerecon_230425.onnx'
    },
}


class Face3dCoeffs(ModelBase):
    def __init__(self, model_type='facerecon_230425', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.input_size = (224, 224)
        self.input_std = self.input_mean = 127.5

    def forward(self, face_image):
        """
        Args:
            face_image: CVImage acceptable class, 5 landmark align with 'arcface_224'
        Returns: coeffs of face reconstruction
        """
        face_image = CVImage(face_image).rgb()
        face = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=True)
        return self.model.forward(face)[0]


if __name__ == '__main__':
    f3c = Face3dCoeffs(model_type='facerecon_230425', provider='gpu')
    coeffs = f3c.forward('resource/cropped_face/112.png')
    print(coeffs.shape)
