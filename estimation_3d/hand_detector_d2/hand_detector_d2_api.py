# -- coding: utf-8 --
# @Time : 2022/2/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from cv2box import CVImage
import numpy as np

class ThirdViewDetector:
    """
    Hand Detector for third-view input.(https://github.com/ddshan/hand_detector.d2)
    """

    def __init__(self):
        print("Loading Third View Hand Detector")
        self.__load_hand_detector()
        self.cfg = None

    def __load_hand_detector(self):
        # load cfg and model
        self.cfg = get_cfg()
        self.cfg.merge_from_file("pretrain_models/digital_human/hand_detector_d2/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
        self.cfg.MODEL.WEIGHTS = 'pretrain_models/digital_human/hand_detector_d2/model_0529999.pth'  # add model weight here
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 0.5 , use low thresh to increase recall
        self.hand_detector = DefaultPredictor(self.cfg)

    def get_cfg(self):
        return self.cfg

    def forward(self, img, show=False):
        results = self.hand_detector(img)
        final_image = None
        if show:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("100DOH_hand_trainval"), scale=1.2)
            v = v.draw_instance_predictions(results["instances"].to("cpu"))
            final_image = v.get_image()[:, :, ::-1]
            CVImage(final_image).show(1)
        return results, final_image

    def get_hand_bbox(self, img):
        bbox_tensor = self.hand_detector(img)['instances'].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        return bboxes


if __name__ == '__main__':
    # data path
    test_img = 'test_img/test1.jpg'
    im = CVImage(test_img).bgr

    tvd = ThirdViewDetector()
    outputs = tvd.forward(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("100DOH_hand_trainval"), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    CVImage(v.get_image()[:, :, ::-1]).show()

    # print
    print(outputs["instances"].pred_classes)
    bboxs = np.array(outputs["instances"].pred_boxes.tensor.to('cpu'))
    print(bboxs)
