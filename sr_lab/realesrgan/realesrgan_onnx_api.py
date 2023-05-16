# -- coding: utf-8 --
# @Time : 2022/11/11
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
from cv2box import CVImage, MyFpsCounter
from apstone import ModelBase

MODEL_ZOO = {
    # https://github.com/xinntao/Real-ESRGAN
    # input_name: ['input_1'], shape: [[1, 3, w, h]]
    # output_name: ['output_1'], shape: [[1, 3, w*4, h*4]]
    'realesr-general-x4v3': {
        'model_path': 'pretrain_models/sr_lib/realesr-general-x4v3-dynamic.onnx'
    },
    # onnx will raise alloc memory error when big image input
    'RealESRGAN_x4plus-dynamic': {
        'model_path': 'pretrain_models/sr_lib/RealESRGAN_x4plus-dynamic.onnx'
    },
    'RealESRGAN_x2plus-dynamic': {
        'model_path': 'pretrain_models/sr_lib/RealESRGAN_x2plus-dynamic.onnx'
    },
}


class GFPGAN(ModelBase):
    def __init__(self, model_type='realesr-general-x4v3', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = 255
        self.input_mean = 0
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        self.scale = 4

    def pad_for_scale_2(self, image_in_):
        self.scale = 2
        h, w, _ = image_in_.shape
        if h % self.scale != 0:
            self.mod_pad_h = (self.scale - h % self.scale)
        if w % self.scale != 0:
            self.mod_pad_w = (self.scale - w % self.scale)
        image_out_ = cv2.copyMakeBorder(image_in_, 0, self.mod_pad_h, 0, self.mod_pad_w, cv2.BORDER_REPLICATE)
        return image_out_

    def forward(self, input_image):
        """
        Args:
            input_image: cv2 image 0-255 BGR
        Returns:
            BGR 512x512x3 0-1
        """
        if self.model_type == 'RealESRGAN_x2plus-dynamic':
            input_image = self.pad_for_scale_2(CVImage(input_image).bgr)
        ori_size = CVImage(input_image).bgr.shape[:2][::-1]
        # print(ori_size)
        image_in = CVImage(input_image).blob(ori_size, self.input_mean, self.input_std, rgb=True)
        image_out = self.model.forward(image_in)
        output_image = (image_out[0][0])[::-1].transpose(1, 2, 0).clip(0, 1)
        if self.model_type == 'RealESRGAN_x2plus-dynamic':
            output_h, output_w, _ = output_image.shape
            output_image = output_image[0:output_h - self.mod_pad_h * self.scale, 0:output_w - self.mod_pad_w * self.scale, :]
        # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        output_image = CVImage(output_image).resize(ori_size, interpolation=cv2.INTER_LANCZOS4).bgr
        return output_image


if __name__ == '__main__':
    face_img_p = 'resource/test1.jpg'
    fa = GFPGAN(model_type='RealESRGAN_x2plus-dynamic', provider='gpu')
    face = fa.forward(face_img_p)
    # CVImage(face, image_format='cv2').save('./gfpgan.jpg')
    CVImage(face, image_format='cv2').show()
