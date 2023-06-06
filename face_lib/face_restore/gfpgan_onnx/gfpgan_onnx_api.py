# -- coding: utf-8 --
# @Time : 2022/11/8
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter
from apstone import ModelBase

MODEL_ZOO = {
    # https://github.com/xuanandsix/GFPGAN-onnxruntime-demo
    # input_name:['input'], shape:[[1, 3, 512, 512]]
    # output_name:['1392'], shape:[[1, 3, 512, 512]]
    'GFPGANv1.4': {
        'model_path': 'pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx'
    },
}

# import torch
# import ctypes
#
# def create_tensor_from_ptr(ptr, shape, dtype):
#     n_elems = int(torch.tensor(shape).prod())
#     buffer = ctypes.c_void_p(ptr)
#     storage = torch.Storage.from_buffer(buffer)
#     tensor = torch.tensor([], dtype=dtype)  # 先创建一个空 Tensor 对象
#     tensor.set_(source=storage,
#                 storage_offset=0,
#                 size=shape,
#                 stride=None)
#     return tensor

class GFPGAN(ModelBase):
    def __init__(self, model_type='GFPGANv1.4', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (512, 512)

    # def cuda_forward(self, face_image_tensor):
    #     # image_out = self.model.cuda_binding_forward([face_image_tensor])[0].numpy()
    #     image_out = self.model.cuda_binding_forward([face_image_tensor])[0]
    #     output_face = create_tensor_from_ptr(image_out.data_ptr(), image_out.shape(), torch.float)
    #
    #     # output_face = ((image_out[0] + 1) / 2)[::-1].transpose(1, 2, 0).clip(0, 1)
    #     return output_face

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 image 0-255 BGR
        Returns:
            BGR 512x512x3 0-1
        """
        image_in = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=True)
        image_out = self.model.forward(image_in)
        output_face = ((image_out[0][0] + 1) / 2)[::-1].transpose(1, 2, 0).clip(0, 1)
        return output_face


if __name__ == '__main__':
    face_img_p = 'resource/cropped_face/512.jpg'
    fa = GFPGAN(model_type='GFPGANv1.4', provider='gpu')

    with MyFpsCounter() as mfc:
        for i in range(10):
            face = fa.forward(face_img_p)
    # CVImage(face, image_format='cv2').save('./gfpgan.jpg')
    CVImage(face, image_format='cv2').show()

    # # cuda forward test
    # import onnxruntime
    # import torch
    # face_image_tensor_ = CVImage(face_img_p).blob(fa.input_size, fa.input_mean, fa.input_std, rgb=True)
    # # face_image_tensor_ = onnxruntime.OrtValue.ortvalue_from_numpy(face_image_tensor_, 'cuda', 0)
    # face_image_tensor_ = torch.tensor(face_image_tensor_).cuda()
    # output_face = fa.cuda_forward(face_image_tensor_)
    # CVImage(output_face, image_format='cv2').show()
