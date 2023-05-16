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


class GFPGAN(ModelBase):
    def __init__(self, model_type='GFPGANv1.4', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (512, 512)

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
