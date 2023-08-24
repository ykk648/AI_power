# -- coding: utf-8 --
# @Time : 2023/8/21
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import numpy as np
from art_lib.optical_flow_estimate.raft.utils import flow_to_image
from cv2box import CVImage
from apstone import ModelBase

MODEL_ZOO = {
    # https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation
    # 0-255 RGB
    # input_name:['0', '1'], shape:[[1, 3, 480, 640], [1, 3, 480, 640]]
    # output_name:['23437', '23436'], shape:[[1, 2, 60, 80], [1, 2, 480, 640]]
    'raft_kitti_iter20_480x640': {
        'model_path': 'pretrain_models/art_lib/optical_flow_estimate/raft/iter20/raft_kitti_iter20_480x640.onnx'
    },
}


class Raft(ModelBase):

    def __init__(self, model_name='raft_kitti_iter20_480x640', provider='gpu'):
        super(Raft, self).__init__(MODEL_ZOO[model_name], provider)
        self.input_width = 640
        self.input_height = 480
        self.mean = 0
        self.std = 1

    def forward(self, img1_, img2_):
        img_width, img_height = CVImage(img1_).bgr.shape[:-1][::-1]
        img1_input = CVImage(img1_).blob((self.input_width, self.input_height), self.mean, self.std, rgb=True)
        img2_input = CVImage(img2_).blob((self.input_width, self.input_height), self.mean, self.std, rgb=True)
        outputs = self.model.forward([img1_input, img2_input])
        outputs = outputs[1][0].transpose(1, 2, 0)
        # draw
        flow_img_ = flow_to_image(outputs)
        flow_img_ = CVImage(flow_img_).resize((img_width, img_height)).rgb()
        return flow_img_


if __name__ == '__main__':
    # Initialize model
    model_name_ = 'raft_kitti_iter20_480x640'
    raft = Raft(model_name_)

    # Read inference image
    img1 = CVImage("resources/for_optical_flow/frame_0016.png").rgb()
    img2 = CVImage("resources/for_optical_flow/frame_0025.png").rgb()

    # Estimate flow and colorize it
    flow_map = raft.forward(img1, img2)
    combined_img = np.hstack((img1, img2, flow_map))

    CVImage(combined_img).show(0, "Estimated flow")

