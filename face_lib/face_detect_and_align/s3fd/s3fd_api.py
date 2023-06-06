# -- coding: utf-8 --
# @Time : 2023/6/1
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, MyFpsCounter
from cv2box.utils.math import Normalize
from apstone import ModelBase
import numpy as np

"""
input_name:['in'], shape:[['batch_size', 3, 'height', 'width']]
output_name:['cls1', 'reg1', 'cls2', 'reg2', 'cls3', 'reg3', 'cls4', 'reg4', 'cls5', 'reg5', 'cls6', 'reg6'], shape:[[1, 2, 'height', 'width'], ['batch_size', 4, 'height', 'width'], ['batch_size', 2, 'height', 'width'], ['batch_size', 4, 'height', 'width'], ['batch_size', 2, 'height', 'width'], ['batch_size', 4, 'height', 'width'], ['batch_size', 2, 'height', 'width'], ['batch_size', 4, 'height', 'width'], ['batch_size', 2, 'height', 'width'], ['batch_size', 4, 'height', 'width'], ['batch_size', 2, 'height', 'width'], ['batch_size', 4, 'height', 'width']]
"""
MODEL_ZOO = {
    # https://github.com/iperov/DeepFaceLive/blob/master/modelhub/onnx/S3FD/S3FD.py
    's3fd': {
        'model_path': 'pretrain_models/face_lib/face_detect/S3FD/S3FD.onnx',
        'input_dynamic_shape': (1, 3, 640, 640),
    },
}


def nms(x1, y1, x2, y2, scores, thresh):
    """
    Non-Maximum Suppression

        x1,y1,x2,y2,scores  np.ndarray of box coords with the same length

    returns indexes of boxes
    """
    keep = []
    if len(x1) == 0:
        return keep

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx_1, yy_1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx_2, yy_2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
        ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class S3FD(ModelBase):
    def __init__(self, model_type='s3fd', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

        self.input_mean = [104, 117, 123]
        self.input_std = [1, 1, 1]
        self.input_size = (640, 640)

    def forward(self, image, threshold=0.95):
        """
        Args:
            image: CVImage acceptable type
            threshold:
        Returns:
        """
        resize_image, ratio_, pad_w_, pad_h_ = CVImage(image).resize_keep_ratio(self.input_size)
        face = CVImage(resize_image).blob_innormal(self.input_size, self.input_mean, self.input_std, rgb=False)
        results = self.model.forward(face)
        results = [result[0] for result in results]

        bbox_list = self.refine(results, threshold)

        bbox_list_new = []
        for bbox in bbox_list:
            # left top right bottom
            locs = bbox[:-1].reshape(-1, 2)
            locs = CVImage(None).recover_from_resize(locs, ratio_, pad_w_, pad_h_)
            bbox_list_new.append(np.concatenate((locs.reshape(1, -1)[0], [bbox[-1]]), axis=0))
        return bbox_list_new

    def refine(self, olist, threshold):
        bboxlist = []
        variances = [0.1, 0.2]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]

            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            for hindex, windex in [*zip(*np.where(ocls[1, :, :] > threshold))]:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[1, hindex, windex]
                loc = np.ascontiguousarray(oreg[:, hindex, windex]).reshape((1, 4))
                priors = np.array([[axc, ayc, stride * 4, stride * 4]])
                bbox = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                       priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
                bbox[:, :2] -= bbox[:, 2:] / 2
                bbox[:, 2:] += bbox[:, :2]
                x1, y1, x2, y2 = bbox[0]
                bboxlist.append([x1, y1, x2, y2, score])

        if len(bboxlist) != 0:
            bboxlist = np.array(bboxlist)
            bboxlist = bboxlist[
                       nms(bboxlist[:, 0], bboxlist[:, 1], bboxlist[:, 2], bboxlist[:, 3], bboxlist[:, 4], 0.3), :]
            bboxlist = [x for x in bboxlist if x[-1] >= 0.5]

        return bboxlist


if __name__ == '__main__':
    s3 = S3FD(model_type='s3fd', provider='gpu')
    detect_results = s3.forward('resource/test3.jpg')
    print(detect_results)
