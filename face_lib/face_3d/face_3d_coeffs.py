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

BFM_DIR = 'pretrain_models/face_lib/face_3d/BFM'

MODEL_ZOO = {
    # https://github.com/sicxu/Deep3DFaceRecon_pytorch
    # input_name: ['input_1'], shape: [[1, 3, 224, 224]] 0-1 RGB
    # output_name: ['output_1'], shape: [[1, 257]]
    'facerecon_230425': {
        'model_path': 'pretrain_models/face_lib/face_3d/epoch_20_facerecon_230425.onnx'
    },
    'facerecon_modelscope': {
        'model_path': 'pretrain_models/face_lib/face_3d/facerecon_modelscope.onnx'
    },
}


class Face3dCoeffs(ModelBase):
    def __init__(self, model_type='facerecon_230425', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.input_size = (224, 224)
        self.face_model = None
        self.input_mean = 0
        self.input_std = 255

    def forward(self, face_image):
        """
        Args:
            face_image: BGR, 0-255, 5 landmark align with 'arcface_224'
        Returns: [(1, 257)], 0-1, coeffs of face reconstruction
        """
        face = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=True)
        return self.model.forward(face)

    def get_3d_params(self, coeffs):
        """
        https://github.com/modelscope/modelscope/blob/c4a6e843a9b1b34126463944313ad62e5ebc43a0/modelscope/models/cv/image_face_fusion/image_face_fusion.py#L194
        Returns: numpy
        """
        from face_lib.face_3d.utils.bfm import ParametricFaceModel
        import torch
        if not self.face_model:
            self.face_model = ParametricFaceModel(BFM_DIR)
        if not isinstance(coeffs, torch.Tensor):
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
        face_vertex, face_texture, face_color, landmark = self.face_model.compute_for_render(coeffs)
        return face_vertex.numpy(), face_texture.numpy(), face_color.numpy(), landmark.numpy()


if __name__ == '__main__':
    f3c = Face3dCoeffs(model_type='facerecon_modelscope', provider='gpu')
    coeffs_ = f3c.forward('resources/cropped_face/112.png')
    print(coeffs_.shape)
    print(coeffs_)

    face_vertex_, face_texture_, face_color_, landmark_ = f3c.get_3d_params(coeffs_)
    # torch.Size([1, 35709, 3]) torch.Size([1, 35709, 3]) torch.Size([1, 35709, 3]) torch.Size([1, 68, 2])
    print(face_vertex_.shape, face_texture_.shape, face_color_.shape, landmark_.shape)
