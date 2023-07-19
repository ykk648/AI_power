# -- coding: utf-8 --
# @Time : 2021/11/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://github.com/zllrunning/face-parsing.PyTorch
"""
import numpy as np
import cv2
from cv2box import CVImage
from apstone import ModelBase

MODEL_ZOO = {
    # input_name: ['x'], shape: [[1, 3, 512, 512]]
    # output_name: ['feat_out'], shape: [[1, 19, 512, 512]]
    'face_parse_onnx': {
        'model_path': 'pretrain_models/face_lib/face_parsing/79999_iter.onnx'
    },
    # no more support for tjm model
    # 'face_parse_tjm': {
    #     'model_path': 'pretrain_models/face_lib/face_parsing/79999_iter.tjm'
    # },
}

# Colors for all 20 parts
PART_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
               [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
# Face Mask
MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]


class FaceParsing(ModelBase):
    def __init__(self, model_name='face_parse_onnx', provider='gpu'):
        super().__init__(model_info=MODEL_ZOO[model_name], provider=provider)
        self.parsing_results = None
        self.face_image = None
        self.input_size = 512
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 0-255 (3,h,w)
        Returns: (512,512)
        """
        self.face_image = face_image
        face_image_in = CVImage(self.face_image).blob_innormal(self.input_size, self.input_mean, self.input_std,
                                                               rgb=False)
        self.parsing_results = self.model.forward(face_image_in)[0].squeeze(0).argmax(0)
        return self.parsing_results

    def get_face_mask(self, mask_shape):
        mask = np.zeros((512, 512)).astype(np.float32)
        for idx, color in enumerate(MASK_COLORMAP):
            mask[self.parsing_results == idx] = color
        # blur the mask
        # mask = cv2.GaussianBlur(mask, (101, 101), 11)
        # mask = cv2.GaussianBlur(mask, (101, 101), 11)
        # mask = cv2.stackBlur(mask, (101, 101))
        # mask = cv2.stackBlur(mask, (101, 101))
        mask = cv2.stackBlur(mask, (201, 201))

        # remove the black borders
        thres = 10
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = mask / 255.
        mask = cv2.resize(mask, mask_shape)
        return mask[..., np.newaxis]

    def show(self):
        vis_im = CVImage(self.face_image).bgr.copy().astype(np.uint8)
        vis_parsing_anno = self.parsing_results.copy().astype(np.uint8)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = PART_COLORS[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        CVImage(vis_im).show()


if __name__ == "__main__":
    test_img = 'resource/cropped_face/512.jpg'
    fp = FaceParsing(model_name='face_parse_onnx', provider='gpu')

    parsing = fp.forward(test_img)
    # mask = fp.get_face_mask((512, 512))
    fp.show()
