# -- coding: utf-8 --
# @Time : 2022/6/27
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, MyFpsCounter
from apstone.wrappers.mmlab_wrapper import BboxDetectorBase

MODEL_ZOO = {
    # gpu 55fps
    'yolox_tiny': {
        'model_path': 'pretrain_models/body_lib/body_bbox_detector/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906_dynamic.onnx',
        'model_input_size': (416, 416),
    },
    # 207fps
    'yolox_tiny_trt16': {
        'model_path': 'pretrain_models/body_lib/body_bbox_detector/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906_static.engine',
        'model_input_size': (416, 416),
    },
    # gpu 49fps
    'yolox_s': {
        'model_path': 'pretrain_models/body_lib/body_bbox_detector/yolox_s_8x8_300e_coco_20211121_095711-4592a793_dynamic.onnx',
        'model_input_size': (640, 640),
    },
}


class BodyBboxDetector(BboxDetectorBase):
    def __init__(self, model='yolox_tiny', threshold=0.5, provider='gpu'):
        self.threshold = threshold
        super().__init__(MODEL_ZOO[model], provider)

    def forward(self, image_in_, show=False, max_bbox_num=1):
        """
        Args:
            image_in_:
            show:
            max_bbox_num:
        Returns: N*4
        """
        model_results = self.model.forward(self.preprocess(image_in_))
        results_after = self.postprocess(model_results, self.threshold, max_bbox_num=max_bbox_num)
        if show:
            self.show(image_in_, results_after)
        return results_after


if __name__ == '__main__':

    image_path = 'resource/for_pose/t_pose_1080p.jpeg'
    image_in = CVImage(image_path).bgr

    # yolox_tiny yolox_s yolox_tiny_static_trt
    bbd = BodyBboxDetector(model='yolox_s', provider='gpu')

    bboxes = bbd.forward(image_in, show=True, max_bbox_num=3)

    from cv2box import CVBbox

    bboxes = CVBbox(bboxes).area_center_filter(image_in.shape)
    print(bboxes)

    with MyFpsCounter('model forward 10 times fps:') as mfc:
        for i in range(10):
            bboxes = bbd.forward(image_in, max_bbox_num=3)
