# -- coding: utf-8 --
# @Time : 2021/11/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
# -*-coding: utf-8 -*-

import onnxruntime
import numpy as np
from cv2box import MyFpsCounter


class ONNXModel:
    def __init__(self, onnx_path, debug=False):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, None,
                                                         providers=["CUDAExecutionProvider"])  # CPUExecutionProvider
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

        if debug:
            input_cfg = self.onnx_session.get_inputs()[0]
            input_shape = input_cfg.shape
            self.input_size = tuple(input_shape[2:4][::-1])
            print(self.input_size)
            print("input_name:{}".format(self.input_name))
            print("output_name:{}".format(self.output_name))

    def speed_test(self):
        ort_inputs = {}
        for node in self.onnx_session.get_inputs():
            print(node.name, node.shape)
            try:
                # ort_inputs[node.name] = np.random.randn(1，3, 256, 256).astype(np.float32)
                ort_inputs[node.name] = np.random.randn(*([1] + node.shape[1:])).astype(np.float32)
            except TypeError:
                raise 'Find dynamic input, name the input shape by yourself !'

        with MyFpsCounter('onnx 1000 times') as mfc:
            for i in range(1000):
                _ = self.onnx_session.run(None, ort_inputs)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor, trans=False):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        if trans:
            image_tensor = image_tensor.transpose(2, 0, 1)
            image_tensor = image_tensor[np.newaxis, :]
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        return self.onnx_session.run(self.output_name, input_feed=input_feed)


if __name__ == '__main__':
    ox = ONNXModel('').speed_test()
