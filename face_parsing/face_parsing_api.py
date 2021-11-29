# -- coding: utf-8 --
# @Time : 2021/11/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
# !/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from utils.image_io import load_img_rgb, img_show
from utils.ai_utils import MyTimer
from model import BiSeNet
import onnxruntime
from model_convert.onnx_model import ONNXModel

# MODEL_PATH = 'pretrain_models/face_parsing/79999_iter.onnx'
# MODEL_PATH = 'pretrain_models/face_parsing/79999_iter.pth'
MODEL_PATH = 'pretrain_models/face_parsing/79999_iter.tjm'


class FaceParsing:
    def __init__(self):
        self.parsing = None
        self.image = None

        # # onnx gpu 1.6s/100iter cpu 3.05s/100iter
        # self.net = ONNXModel(MODEL_PATH)

        # torch.jit  gpu 1.2s/100iter cpu 7.8s/100iter
        self.net = torch.jit.load(MODEL_PATH)
        # self.net = torch.jit.load(MODEL_PATH, map_location=torch.device('cpu'))

        # torch
        # n_classes = 19
        # self.net = BiSeNet(n_classes=n_classes)
        # self.net.cuda()
        # self.net.load_state_dict(torch.load(MODEL_PATH))

        self.net.eval()

    def forward(self, face_img_):

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = load_img_rgb(face_img_)
            self.image = cv2.resize(img, (512, 512))
            img = to_tensor(self.image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            # input_names = ["x"]
            # output_names = ["feat_out"]
            # torch.onnx.export(self.net, img, "face_parsing.onnx", verbose=True, input_names=input_names,
            #                   output_names=output_names, opset_version=13)
            #
            # out = self.net(img)

            out = self.net.forward(img)[0]

            self.parsing = out.cpu().numpy().squeeze(0).argmax(0)
            # print(self.parsing)
            # print(np.unique(self.parsing))
        return self.parsing

    def show(self, stride=1):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(self.image)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = self.parsing.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        img_show(vis_im)


if __name__ == "__main__":
    test_img = 'test_img/croped_face/512.jpg'
    fp = FaceParsing()
    parsing = fp.forward(test_img)
    fp.show()
