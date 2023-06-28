# -- coding: utf-8 --
# @Time : 2023/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from apstone import ModelBase
import numpy as np
from cv2box import CVImage
from PIL import Image

"""
input_name:['create_inputs/sub:0'], shape:[['unk__40886', 'unk__40887', 3]]
output_name:['ExpandDims_1:0', 'Max:0', 'Sigmoid:0'], shape:[[1, 'unk__40888', 'unk__40889', 1], [1, 'unk__40890', 'unk__40891'], [1, 'unk__40892', 'unk__40893', 1]]
"""
MODEL_ZOO = {
    # https://github.com/Engineering-Course/CIHP_PGN
    'cihp_pgn': {
        'model_path': 'pretrain_models/seg_lib/cihp_pgn/cihp_pgn.onnx'
    },
}

label_colours = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221),
                 (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221),
                 (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]
N_CLASSES = 20


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


class CIHPPGN(ModelBase):
    def __init__(self, model_name='cihp_pgn', provider='gpu'):
        super(CIHPPGN, self).__init__(MODEL_ZOO[model_name], provider)
        self.mean = [125.0, 114.4, 107.9]
        self.std = [1, 1, 1]

    def forward(self, image_in):
        input_size_ = CVImage(image_in).bgr.shape[:2]
        input_image = CVImage(image_in).blob_innormal(input_size_, input_mean=self.mean, input_std=self.std)
        # h,w,3
        input_image = input_image[0].transpose(1, 2, 0)
        parsing_, scores, edge_ = self.model.forward(input_image)
        mask_ = decode_labels(parsing_, num_classes=N_CLASSES)
        return mask_[0], parsing_[0].astype(np.uint8), (edge_[0] * 255).astype(np.uint8)


if __name__ == '__main__':
    cihp = CIHPPGN()

    img_p = ''
    # decrease size to reduce GPU mem
    img_p = CVImage(img_p).resize((320, 180)).bgr
    mask, parsing, edge = cihp.forward(img_p)
    print(mask.shape)
    print(parsing.shape)
    print(edge.shape)
    CVImage(mask).show()
    CVImage(parsing).show()
    CVImage(edge).show()
