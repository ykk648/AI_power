# -- coding: utf-8 --
# @Time : 2023/10/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from apstone import ModelBase
from cv2box import CVImage, CVFile
import numpy as np

MODEL_ZOO = {
    # input_name:['input_1:0'], shape:[[1, 448, 448, 3]]
    # output_name:['predictions_sigmoid'], shape:[[1, 9083]]
    'moat': {
        'model_path': 'sd_models/tagger/SmilingWolf_wd-v1-4-moat-tagger-v2.onnx',
        'tag_path': 'sd_models/tagger/selected_tags.csv',
    },
}

del_list = ['no_humans', 'english_text', 'monochrome', 'greyscale', 'blurry', 'solo', 'horse']


class Tagger(ModelBase):
    def __init__(self, model_name='moat', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)
        self.input_size = (448, 448)
        self.tags = CVFile(MODEL_ZOO[model_name]['tag_path']).data
        # print(self.tag_data)

    def forward(self, image_in_):
        image_in_ = CVImage(image_in_).resize(self.input_size).bgr
        image_in_ = image_in_[None, :].astype(np.float32)
        outputs = self.model.forward(image_in_)[0]
        outputs = 1 / (1 + np.exp(-outputs))
        tags = {tag: float(conf) for tag, conf in zip(self.tags['name'][4:], outputs.flatten()[4:]) if
                float(conf) > 0.6}
        tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        tags = [tag[0] for tag in tags]

        # for tag in tags:
        #     if tag in del_list or tag.find('background') > 0:
        #         tags.remove(tag)

        return ','.join(tags)


if __name__ == '__main__':
    image_p = 'resources/for_sd/An_astronaut_is_riding_a_horse_on_Mars_seed-444264997.png'
    image_in = CVImage(image_p).bgr

    tagger = Tagger(model_name='moat')

    output = tagger.forward(image_in)
    print(output)
